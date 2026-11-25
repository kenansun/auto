# AlphaLoop Strategy R&D Platform v3 (`alphaloop/platform/`)

> 权威架构蓝图：`Strategic-Command-Headquarters/docs/architecture/UNIFIED_MULTI_ASSET_STRATEGY_RD_PLATFORM_V3_20261123.md`（工单 `CC-20261123-002`）
> 纯 Python 3.12 标准库零外依赖 · STD-34 严格审计 0 违规 · 100% 研究与只读数据采集（零实盘写单）

## 模块结构

| 模块 | 职责 | 借鉴来源（均自研重写，零外仓绑定） |
|---|---|---|
| `assets.py` | 多资产规格 `AssetSpec` / `TradingCalendar` / `CostModel`（Web3 24/7、黄金 PAXG 24/7 + COMEX 23/5、宏观日线、A 股 T+1/禁空/印花税模板） | QuantConnect LEAN + Rob Carver 成本先行 |
| `data_lake.py` | 双时间戳 Point-in-Time SQLite WAL 数据底座（`ts_event_ms` / `ts_avail_ms`）、`DQ-MA-01..06` 质检、PAXG↔GC=F 双源背离仲裁、JSONL 导出、复用 HL/Gate + 新增 Yahoo 黄金/宏观真实采集器 | NautilusTrader 双时间戳 + STD-32 复用 + DuckDB/Parquet 兼容边界 |
| `factors.py` | 单核流批同构 20+ 跨资产技术/多因子引擎 `StreamingFactorEngine`、Point-in-Time `MacroRegimeTimeline`（Gold×IEF/DXY/TIP/VIX）、G2 方向/Q1-Q5 单调/t-stat 审计与相关性冗余剪枝 | NautilusTrader + Carver EWMAC + QuantPedia (2026) + BAISYS_QUANT + RD-Agent(Q) |
| `regime_inference.py` | 因果 6 态政权识别（带迟滞防抖）+ 物理参数自动推理 `AutoParameterInferrer`（成本-持仓期法则、Carver 波动率定杠杆、持久度自适应周期、噪音/肥尾止损与惯性缓冲带） | Carver Vol-Targeting + 因果政权过滤 |
| `validation.py` | 精确 PSR/DSR（含偏度/超额峰度/试参折价）、带清洗与隔离的 CPCV、CSCV 过拟合概率 PBO、4-连通高原几何中心选点 | AFML (López de Prado) / `algomaschine-kit` + BAISYS_QUANT |
| `strategy_customizer.py` | 单核流批同构策略内核 `AdaptiveMultiFactorKernel` + 7 道机构级门禁（G1–G7，含 G7 逐 bar 比特一致检验）自动定制编译器 | NautilusTrader Parity + LEAN 五段解耦 |

## 一键运行

```bash
# 运行全部 15 项平台 v3 单测与红队负向测试
python3 -m unittest discover -s tests -p "test_platform_v3_*.py" -v

# 拉取真实 Web3（Hyperliquid+Gate）与黄金/宏观（Yahoo 5 年）数据并运行 4 赛道端到端实证
python3 scripts/run_strategy_platform_v3.py --data-dir data --crypto-days 200 --gold-range 5y --capital 3000

# 复用本地已采 SQLite 数据湖快速重跑（离线，~15 秒）
python3 scripts/run_strategy_platform_v3.py --data-dir data --no-fetch
```
