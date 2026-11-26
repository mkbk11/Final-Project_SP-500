# S&P 500 数据科学课程项目

本项目使用 1996-11-13 至 2026-11-12 的标普 500 日线数据，完成数据清洗、特征构造、探索分析、分类与回归、数据泄漏检查，并扩展到模拟交易和市场状态分析。课程展示以根目录的 **01–09 Notebook** 为主。每份 Notebook 在文件内定义所需的辅助函数，不依赖 `python/` 下的脚本；不同阶段之间会读取先前保存的数据和预测结果。

## 研究任务

在交易日 `t` 收盘后生成信号，仅使用当时已知的数据；`t+1` 开盘入场，`t+2` 开盘退出。

- **回归**：预测 `Open[t+2] / Open[t+1] - 1`。
- **分类**：上述收益率严格大于 `0.002`（0.20%）时标为 1，否则为 0。
- **评估**：按时间先后划分训练、验证和测试集；模型与阈值在训练、验证阶段选择，最后在测试集上报告结果。分界处移除持有期跨越边界的样本。

分类 A/B 比较 **同一批样本上的特征**：A 使用 6 项原始价量特征，B 在此基础上加入 5 项跨市场特征（VIX、美债收益率、DXY、WTI），共 11 项。由于外部数据的覆盖范围，分类实验从 2009 年开始。回归比较 XGBoost 与岭回归，两组使用相同样本和 6 项原始特征。02 阶段另构造 14 项价量特征用于数据分析，它们并非分类 B 的输入。

## 文件与执行顺序

| 文件 | 内容 | 运行前提 |
| --- | --- | --- |
| [01_data_preprocessing.ipynb](01_data_preprocessing.ipynb) | 读取原始 JSON，检查交易日和异常值，清洗并按时间划分数据 | `data/raw/` |
| [02_feature_engineering.ipynb](02_feature_engineering.ipynb) | 构造特征、标签和跨市场输入，检查日期对齐与边界 | 01、`data/market/` |
| [03_feature_analysis.ipynb](03_feature_analysis.ipynb) | 缺失值、分布、相关性及年度统计图表 | 01–02 |
| [04_classification_training.ipynb](04_classification_training.ipynb) | 6 项与 11 项特征的 XGBoost 分类对比 | 02 |
| [05_regression_training.ipynb](05_regression_training.ipynb) | XGBoost、岭回归与扩展的 MLP 预测 | 02 |
| [06_compare_results.ipynb](06_compare_results.ipynb) | 汇总分类、回归及 MLP 结果 | 04–05 |
| [07_validation_experiments.ipynb](07_validation_experiments.ipynb) | 时间划分与数据泄漏对照实验 | 01；可单独运行 |
| [08_trading_evaluation.ipynb](08_trading_evaluation.ipynb) | 已保存预测的模拟交易、成本敏感性和 Double DQN | 04–05 的预测及 MLP |
| [09_market_regime_analysis.ipynb](09_market_regime_analysis.ipynb) | 按市场状态分析预测和交易表现 | 04–05、08 |

建议按 **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09** 展示与运行。07 在 01 之后即可独立执行。Notebook 中的结果和图表是保存的实验快照；重新运行可能产生新的输出目录，不能把历史快照当成当前运行结果。

## 环境与运行

使用 Python 3.12，在 SP500 根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

在 VS Code 或 Jupyter 中选择该 `.venv` 的 Python 内核，按上表顺序运行 Notebook。首次运行会重建 `data/processed/`、`data/features/` 等中间文件；模型训练和 07–09 阶段会花费更长时间。

另有合并后的独立 Python 入口，可按顺序执行相应阶段：

```powershell
.\.venv\Scripts\python.exe python\run_pipeline.py
.\.venv\Scripts\python.exe python\run_pipeline.py --include-validation --include-fusion
```

默认运行 01–06；`--include-validation` 增加 07；`--include-fusion` 启用扩展部分并运行 08–09；`--include-literature` 生成文献对照表，需与 `--include-fusion` 同时使用。脚本位于 `python/prepare.py`（01–03）、`python/models.py`（04–07）、`python/analysis.py`（08–09），公共工具在 `python/experiment_utils.py`。脚本与 Notebook 是两套入口，课堂展示建议打开 Notebook。

## 数据、结果与仓库范围

- `data/raw/` 保存标普 500 原始 JSON；`data/market/` 保存跨市场输入。中间数据、划分结果和逐样本预测由运行过程生成。
- `output/` 保存图表、指标与实验记录。仓库保留用于展示的精选 CSV 和图表；重复运行目录、模型权重、虚拟环境等由 [.gitignore](.gitignore) 排除。
- [RESULTS.md](RESULTS.md) 汇总分类与回归实验；[FUSION_RESULTS.md](FUSION_RESULTS.md) 说明 MLP、交易和状态分析的历史结果及限制；[code_explanation.md](code_explanation.md) 记录部分课堂可能问到的代码细节。
- `report_drafts/` 为双语报告草稿和生成脚本。结果属于指定历史测试区间，不能据此推断未来交易收益。

课程展示可重点讲数据预处理、按时间划分与防止未来信息泄漏、同样本条件下的模型比较，以及预测指标与交易结果之间的差别。
