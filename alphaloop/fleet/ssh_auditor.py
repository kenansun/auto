"""Hardened Read-Only SSH Fleet Auditor for server clusters (Red-Team #8 & #15 hardened).

Security invariants:
- Strictly enforces read-only SSH username (`agent-ro` by default, rejects root & `-` flags).
- Validates IPv4 addresses via `ipaddress.IPv4Address` and passes `--` before `user@ip`.
- ONLY executes a fixed dictionary of pre-approved, strictly read-only shell probes.
- Scopes file reads strictly to allowed `/opt/` data prefixes (blocks `/etc/passwd`, `.env`, keys).
- Writes fleet snapshots atomically via temp file + `os.replace()`.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alphaloop.core.provenance import utc_now_ms

logger = logging.getLogger(__name__)

READONLY_PROBES: dict[str, str] = {
    "health_snapshot": (
        "echo '==HOST==' && hostname && "
        "echo '==UPTIME==' && uptime && "
        "echo '==MEM==' && free -m | awk 'NR==2 {print $2,$3,$7}' && "
        "echo '==DISK==' && df -P / /mnt/backup_disk 2>/dev/null | awk 'NR>1 {print $6,$2,$3,$4,$5}' && "
        "echo '==PUBLIC_PORTS==' && ss -tln 2>/dev/null | awk 'NR>1 && ($4 ~ /^0\\.0\\.0\\.0:/ || $4 ~ /^\\[::\\]:/ || $4 ~ /^\\*:/) {print $4}' | sort -u"
    ),
    "watchdog_local": "curl -s --max-time 3 http://127.0.0.1:8794/api/v1/local_status 2>/dev/null || echo '{\"ok\":false}'",
    "watchdog_hub": "curl -s --max-time 4 http://127.0.0.1:8793/api/v1/status 2>/dev/null || echo '{\"ok\":false}'",
}

FORBIDDEN_TOKENS = (
    ";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r",
    "rm ", "mv ", "cp ", "chmod ", "chown ", "sudo ", "systemctl restart", "systemctl stop", "kill ",
)
ALLOWED_REMOTE_PREFIXES = ("/opt/",)
FORBIDDEN_FILENAMES = (".env", "security.json", "id_ed25519", "id_rsa", ".pem", ".key")
SAFE_SSH_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def validate_ssh_user(user: str) -> str:
    if not user or not SAFE_SSH_USER_RE.fullmatch(user):
        raise ValueError(f"Invalid SSH username: {user!r}")
    if user in ("root", "admin", "ubuntu", "ec2-user"):
        raise PermissionError(f"Refusing privileged SSH user {user!r}; strictly read-only account required")
    return user


def validate_ipv4(ip: str) -> str:
    try:
        return str(ipaddress.IPv4Address(ip))
    except ipaddress.AddressValueError as exc:
        raise ValueError(f"Invalid IPv4 address: {ip!r}") from exc


def validate_safe_path(remote_path: str) -> str:
    """Validate that a remote path for read-only inspection is strictly under /opt/ and non-secret."""
    if not remote_path or not remote_path.startswith("/"):
        raise ValueError(f"Remote path must be absolute: {remote_path!r}")
    if ".." in remote_path:
        raise ValueError(f"Path traversal forbidden: {remote_path!r}")
    for tok in FORBIDDEN_TOKENS:
        if tok in remote_path:
            raise ValueError(f"Forbidden token {tok!r} in path {remote_path!r}")
    if not re.fullmatch(r"/[A-Za-z0-9._/\-]+", remote_path):
        raise ValueError(f"Invalid characters in remote path: {remote_path!r}")
    if not any(remote_path.startswith(pfx) for pfx in ALLOWED_REMOTE_PREFIXES):
        raise PermissionError(f"Remote path outside allowed read-only prefixes {ALLOWED_REMOTE_PREFIXES}: {remote_path!r}")
    lower = remote_path.lower()
    if any(sec in lower for sec in FORBIDDEN_FILENAMES):
        raise PermissionError(f"Refusing read of sensitive secret file pattern: {remote_path!r}")
    return remote_path


@dataclass
class NodeAuditResult:
    node_id: str
    label: str
    ip: str
    primary_role: str
    reachable: bool
    hostname: str = ""
    uptime_raw: str = ""
    mem_total_mb: int = 0
    mem_used_mb: int = 0
    mem_avail_mb: int = 0
    mem_used_pct: float = 0.0
    root_disk_pct: float = 0.0
    root_disk_free_gb: float = 0.0
    extra_mounts: list[dict[str, Any]] = field(default_factory=list)
    public_ports: list[str] = field(default_factory=list)
    alerts: list[dict[str, str]] = field(default_factory=list)
    error: str = ""
    audited_ts_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReadOnlyFleetAuditor:
    """Audits configured servers strictly via allowlisted read-only SSH probes."""

    def __init__(self, roles_path: str | Path, ssh_key_path: str | Path) -> None:
        self.roles_path = Path(roles_path)
        self.ssh_key_path = Path(ssh_key_path)
        self.config = json.loads(self.roles_path.read_text(encoding="utf-8"))
        self.ssh_user = validate_ssh_user(str(self.config.get("ssh_readonly_user", "agent-ro")))

    def _base_ssh_cmd(self, ip: str) -> list[str]:
        clean_ip = validate_ipv4(ip)
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new",
            "-i", str(self.ssh_key_path),
            "--",
            f"{self.ssh_user}@{clean_ip}",
        ]

    def run_probe(self, ip: str, probe_name: str, timeout_sec: int = 12) -> tuple[bool, str, str]:
        """Run ONLY a named allowlisted read-only probe via ssh BatchMode."""
        if probe_name not in READONLY_PROBES:
            raise PermissionError(f"Probe '{probe_name}' is not in READONLY_PROBES allowlist")
        cmd = self._base_ssh_cmd(ip) + [READONLY_PROBES[probe_name]]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return proc.returncode == 0, proc.stdout, proc.stderr.strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, "", str(exc)

    def read_remote_tail(self, ip: str, remote_path: str, lines: int = 100, timeout_sec: int = 15) -> tuple[bool, str, str]:
        """Safely read the last N lines of a validated /opt/ file path (strictly read-only)."""
        safe_path = validate_safe_path(remote_path)
        n = max(1, min(int(lines), 5000))
        cmd = self._base_ssh_cmd(ip) + ["tail", "-n", str(n), safe_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return proc.returncode == 0, proc.stdout, proc.stderr.strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, "", str(exc)

    def read_remote_head_line(self, ip: str, remote_path: str, timeout_sec: int = 12) -> tuple[bool, str, str]:
        """Safely read line 1 (CSV header) of a validated /opt/ file path."""
        safe_path = validate_safe_path(remote_path)
        cmd = self._base_ssh_cmd(ip) + ["head", "-n", "1", safe_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, "", str(exc)

    def audit_node(self, node_cfg: dict[str, Any]) -> NodeAuditResult:
        now = utc_now_ms()
        res = NodeAuditResult(
            node_id=node_cfg["node_id"],
            label=node_cfg["label"],
            ip=node_cfg["ip"],
            primary_role=node_cfg["primary_role"],
            reachable=False,
            audited_ts_ms=now,
        )
        ok, out, err = self.run_probe(node_cfg["ip"], "health_snapshot")
        if not ok:
            res.error = err or "SSH probe failed"
            res.alerts.append({"severity": "P1", "code": "SSH_UNREACHABLE", "detail": res.error})
            return res

        res.reachable = True
        sections: dict[str, list[str]] = {}
        cur = ""
        for ln in out.splitlines():
            s = ln.strip()
            if s.startswith("==") and s.endswith("=="):
                cur = s.strip("=")
                sections[cur] = []
            elif cur and s:
                sections[cur].append(s)

        if sections.get("HOST"):
            res.hostname = sections["HOST"][0]
        if sections.get("UPTIME"):
            res.uptime_raw = sections["UPTIME"][0]
        if sections.get("MEM"):
            parts = sections["MEM"][0].split()
            if len(parts) >= 3:
                try:
                    res.mem_total_mb = int(parts[0])
                    res.mem_used_mb = int(parts[1])
                    res.mem_avail_mb = int(parts[2])
                    if res.mem_total_mb > 0:
                        res.mem_used_pct = round((res.mem_used_mb / res.mem_total_mb) * 100.0, 1)
                except ValueError:
                    res.alerts.append({"severity": "P1", "code": "PARSE_MEM_ERROR", "detail": sections["MEM"][0]})

        seen_mounts: set[str] = set()
        for row in sections.get("DISK", []):
            parts = row.split()
            if len(parts) < 5:
                continue
            mnt = parts[0]
            if mnt in seen_mounts:
                continue
            seen_mounts.add(mnt)
            try:
                total_kb = int(parts[1])
                avail_kb = int(parts[3])
                pct = float(parts[4].rstrip("%"))
            except ValueError:
                continue
            free_gb = round(avail_kb / (1024.0 * 1024.0), 2)
            if mnt == "/":
                res.root_disk_pct = pct
                res.root_disk_free_gb = free_gb
            else:
                res.extra_mounts.append({
                    "mount": mnt,
                    "used_pct": pct,
                    "free_gb": free_gb,
                    "total_gb": round(total_kb / (1024.0 * 1024.0), 2),
                })

        res.public_ports = sections.get("PUBLIC_PORTS", [])

        disk_warn = float(node_cfg.get("disk_warn_pct", 80))
        if res.root_disk_pct >= disk_warn:
            res.alerts.append({
                "severity": "P0" if res.root_disk_pct >= 85 else "P1",
                "code": "DISK_HIGH",
                "detail": f"Root disk {res.root_disk_pct:.0f}% used ({res.root_disk_free_gb:.2f} GB free)",
            })
        if res.mem_used_pct >= float(node_cfg.get("mem_warn_pct", 85)):
            res.alerts.append({
                "severity": "P1",
                "code": "MEM_HIGH",
                "detail": f"RAM {res.mem_used_pct:.1f}% used ({res.mem_avail_mb} MB avail)",
            })
        sensitive_ports = [str(p) for p in node_cfg.get("sensitive_ports", ["8788", "8794", "8793", "8789", "8791", "8000"])]
        for p in res.public_ports:
            if any(p.endswith(f":{port}") for port in sensitive_ports):
                res.alerts.append({
                    "severity": "P0" if p.endswith(":8788") else "P1",
                    "code": "SENSITIVE_PORT_PUBLIC_BIND",
                    "detail": f"Port {p} bound to all interfaces (prefer 127.0.0.1 or overlay network)",
                })
        return res

    def audit_all(self, save_path: str | Path | None = None) -> dict[str, Any]:
        results = [self.audit_node(n) for n in self.config.get("nodes", [])]
        payload = {
            "generated_ts_ms": utc_now_ms(),
            "policy": self.config.get("policy", ""),
            "nodes_total": len(results),
            "nodes_reachable": sum(1 for r in results if r.reachable),
            "p0_alerts": sum(1 for r in results for a in r.alerts if a["severity"] == "P0"),
            "p1_alerts": sum(1 for r in results for a in r.alerts if a["severity"] == "P1"),
            "nodes": [r.to_dict() for r in results],
        }
        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            fd, tmp_name = tempfile.mkstemp(prefix=".fleet.", suffix=".tmp", dir=str(p.parent))
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, p)
            except Exception:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise
        return payload
