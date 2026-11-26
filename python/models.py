"""S&P 500 stages: 04, 05, 06, 07. Run with a stage number or all."""

import argparse


def classification():
    """Run stage 04 (04_classification_training)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 04 分类训练：基线与最终方案
    # 
    # **一次准备数据 → 单次时序划分 → 分别训练调参 → 单次测试 → 同表比较。** 从上至下运行全部即可。
    # 
    # | 基线 | 最终方案 |
    # |---|---|
    # | A 组 · 6 特征 + XGBoost | B 组 · 11 特征 + XGBoost |
    # 
    # - 在 t 日收盘预测 `open[t+2] / open[t+1] - 1`。分类判断收益是否**严格大于 +0.20%**；回归预测收益率。
    # - 两方案共用 train : val : test = 8 : 1 : 1 区间，按时间序列划分且不打乱。
    # - 每次运行独立保存到 `output/classification_comparison/[timestamp]/`，两方案训练、测试分别存放。
    # 整理说明：读取第 02 步的特征表；模型、训练函数、验证选参和单独测试流程沿用原 Notebook。
    # 分类参数以原 Notebook 为准：max_depth=1，min_child_weight=20。

    # %% [markdown]
    # ## 1. 导入与配置

    # %%
    # 脚本集中在 python/；Notebook 保留在项目根目录。
    from pathlib import Path
    import sys
    PYTHON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "python"
    if not (PYTHON_DIR / "experiment_utils.py").is_file():
        raise FileNotFoundError("请在 SP500_organized 项目根目录打开 Notebook。")
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_curve
    import experiment_utils

    from experiment_utils import (
        RAW_FILE, CLEAN_FILE, FEATURE_DIR, MARKET_FILES, ORIGINAL, MARKET, Run,
        stage_log, LoggedEarlyStopping, load_prepared_samples, chronological_split,
        save_csv, save_json, new_training_dir, save_training, evaluate_saved,
    )

    # %%
    # 固定的树参数沿用前面的实验；本方案调节轮次，分类还调节概率阈值。
    # 不扩大搜索范围，避免把这次整理变成另一套实验。

    XGB_PARAMS = {
        "tree_method": "hist",       # 直方图分裂，适合这份日频小数据
        "max_depth": 1,              # 限制树深，降低过拟合风险
        "eta": 0.03,                 # 每轮学习步长
        "min_child_weight": 20,    # 沿用原 Notebook，区别于旧脚本的 5
        "subsample": 0.8,            # 每轮抽取部分训练行
        "colsample_bytree": 0.9,     # 每棵树抽取部分特征
        "lambda": 5.0,               # L2 正则化
        "seed": 42, "nthread": 4,
    }
    MAX_ROUNDS = 400
    EARLY_STOPPING = 30              # 验证指标连续 30 轮未改善时停止

    SCENARIO = "classification_comparison"
    ARM_LABELS = {'baseline': 'A 组 · 6 特征 + XGBoost', 'final': 'B 组 · 11 特征 + XGBoost'}
    FEATURE_SETS = {"baseline": ORIGINAL, "final": ORIGINAL + MARKET}
    CONFIG = {
        "task": "classification", "evaluation": "single_chronological_holdout",
        "split_ratios": [0.8, 0.1, 0.1], "horizon_trading_days": 1,
        "positive_threshold": 0.002, "label": "open[t+2] / open[t+1] - 1",
        "last_data_date": "2026-11-12", "seed": 42,
        "tree_params": XGB_PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING,
    }
    RUN = experiment_utils.start_run(SCENARIO, CONFIG, [RAW_FILE, CLEAN_FILE, FEATURE_DIR / "classification_samples.csv", FEATURE_DIR / "feature_build.json"] + MARKET_FILES)
    arm_runs = {}
    for arm in ARM_LABELS:
        folder = RUN.path / arm
        folder.mkdir()
        arm_config = dict(CONFIG, arm=ARM_LABELS[arm], features=FEATURE_SETS[arm])
        save_json(folder / "config.json", arm_config)
        arm_runs[arm] = Run(folder, arm_config, RUN.logger)

    # %% [markdown]
    # ## 2. 特征与标签
    # 
    # **A = 原 6 特征**
    # | 特征 | 计算方式 |
    # |---|---|
    # | `return_1d` / `return_5d` / `return_20d` | 收盘价 / 1、5、20 个交易日前收盘价 − 1 |
    # | `volatility_20d` | 最近 20 个日收益率的样本标准差，ddof=1 |
    # | `intraday_return` | 当日收盘价 / 开盘价 − 1 |
    # | `intraday_range` |（最高价 − 最低价）/ 开盘价 |
    # 
    # 原 6 特征均由 OHLC 价格计算，不含成交量。收益率使用小数形式。
    # 
    # **B = 原 6 特征 + 以下 5 项，共 11 项。** CPI、非农和 FOMC 不在 B 中。
    # 
    # | 新增特征 | 含义与单位 |
    # |---|---|
    # | `vix_close` | 最近可用 VIX 收盘水平，指数点 |
    # | `vix_change` | 最近两次可用 VIX 收盘值之差，指数点 |
    # | `treasury_2y_change_bp` | 2 年期美债收益率变化，基点 |
    # | `dxy_change_pct` | 美元指数收盘价百分比变化，% |
    # | `wti_change_usd` | WTI 近月期货收盘价变化，美元/桶 |
    # 
    # 日线至少滞后 1 个纽约日历日；2016-10-11 以前的美债收益率滞后 7 日。
    # 在可用日期之后向前匹配，最多容许 7 日的陈旧值。来源日和可用日会保存为审计表。
    # 这是复用历史实验的保守时间约定。美元指数和 WTI 使用已保存的 Yahoo 日线。

    # %%
    # 特征公式集中在 02_feature_engineering，避免维护多份实现。

    # %%
    samples = load_prepared_samples("classification")
    save_csv(RUN.path / "features_and_targets.csv", samples)
    experiment_utils.show_table(samples[ORIGINAL + ["target_return", "target_class"]].tail(3))

    # %% [markdown]
    # ## 3. 单次按时间划分
    # 在本任务有效样本池上确定约 80% / 10% / 10% 的日期边界。入场、结算均须落在同一分区，边界处剔除越界标签，因此最终数量略有变化。两方案使用同一训练集、验证集和测试集。

    # %%
    # 与原代码相同的默认边界；配置中的比例实际传入划分函数。
    splits, split_summary = chronological_split(samples, CONFIG["split_ratios"])
    save_csv(RUN.path / "split_summary.csv", split_summary, index=False)
    experiment_utils.show_table(split_summary)
    RUN.logger.info("单次时序划分：训练 %s / 验证 %s / 测试 %s",
                    *(len(splits[k]) for k in ("train", "val", "test")))

    # %% [markdown]
    # ## 4. 定义 XGBoost 训练与调参函数
    # A/B 共用此函数，仅特征不同。验证 AUC 选最佳轮次，验证平衡准确率选分类阈值。函数只接收训练和验证数据。

    # %%
    def train_xgboost(run, stage, train, val, columns):
        """只接收训练和验证数据；函数内不接触测试集。"""
        folder = new_training_dir(run, stage)
        stage_log(run, folder, f"{stage} 开始 XGBoost 训练 | train={len(train)} / val={len(val)}")
        label = "target_class"
        dtrain = xgb.DMatrix(train[columns], label=train[label], nthread=4)
        dval = xgb.DMatrix(val[columns], label=val[label], nthread=4)
        params = dict(XGB_PARAMS, objective="binary:logistic", eval_metric="auc")
        history = {}

        # 每次训练创建新的回调对象；save_best=True 只保留验证集最佳轮次的树。
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=MAX_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=history,
            callbacks=[
                # 每一轮先打印训练/验证指标，再检查早停，最后一轮也完整记录。
                LoggedEarlyStopping(run, folder, log_every=1,
                    rounds=EARLY_STOPPING, metric_name="auc",
                    data_name="val", maximize=True, save_best=True,
                ),
            ],
            verbose_eval=False,
        )
        selected = {
            "params": params, "best_rounds": int(model.best_iteration + 1),
            "trained_rounds": len(history["val"]["auc"]),
            "validation_metric": "auc", "validation_score": float(model.best_score),
            "prediction_unit": "probability",
            "training_window": "同一固定训练集",
        }

        # 阈值只看验证集。最大化 TPR-FPR 等价于最大化平衡准确率。
        val_probability = model.predict(dval)
        fpr, tpr, thresholds = roc_curve(val[label], val_probability)
        finite = np.isfinite(thresholds)
        threshold = float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
        selected["threshold"] = threshold
        stage_log(run, folder, f"验证集最佳轮次 {selected['best_rounds']}；分类阈值 {threshold:.6f}")

        stage_log(run, folder, f"实际执行 {selected['trained_rounds']} 轮；保留最佳 {selected['best_rounds']} 轮模型")
        return save_training(run, stage, folder, model, {"train": train, "val": val},
                             columns, selected, history)

    # %% [markdown]
    # ## 实际训练
    # 只循环比较两种方案，每种方案仅训练、调参一次。XGBoost 每轮打印训练与验证指标，结束时分别打印实际执行轮次与最佳轮次。连续 30 轮没有改善才会早停。模型和日志分别保存。

    # %%
    artifacts = {}
    for arm in ARM_LABELS:
        RUN.logger.info("训练方案：%s", ARM_LABELS[arm])
        artifacts[arm] = train_xgboost(
            arm_runs[arm], "holdout", splits["train"], splits["val"], FEATURE_SETS[arm])

    # %% [markdown]
    # ## 测试
    # 读取已保存模型，在同一个测试集上分别评价。测试数据不参与轮次、分类阈值或 alpha 的选择。

    # %%
    rows = []
    for arm in ARM_LABELS:
        RUN.logger.info("测试方案：%s", ARM_LABELS[arm])
        metrics, predictions = evaluate_saved(arm_runs[arm], "holdout", artifacts[arm], splits["test"])
        metrics.pop("period", None)
        rows.append(dict(scenario=ARM_LABELS[arm], **metrics))
        save_csv(arm_runs[arm].path / "test_predictions.csv", predictions)
        save_json(arm_runs[arm].path / "summary.json", metrics)
    comparison = pd.DataFrame(rows)
    save_csv(RUN.path / "comparison_summary.csv", comparison, index=False)
    save_json(RUN.path / "status.json", {"status": "completed", "evaluation": "single_chronological_holdout"})

    # %% [markdown]
    # ## 同一测试集的结果对比实验
    # ROC-AUC、AP 越高越好。

    # %%
    experiment_utils.show_table(comparison[["scenario", "n_test", "roc_auc", "ap", "balanced_accuracy", "f1"]])
    print("本次输出目录：", experiment_utils.display_path(RUN.path))

    # Optional reference-paper comparison, calculated from saved local predictions.
    import os
    if os.environ.get("SP500_INCLUDE_LITERATURE") == "1":
        SECTION='04_classification'
        import json, hashlib
        from pathlib import Path
        from datetime import datetime
        import numpy as np
        import pandas as pd
        ROOT = Path(__file__).resolve().parents[1]
        assert (ROOT/'01_data_preprocessing.ipynb').is_file()
        LIT_OUT = ROOT/'output/literature'/SECTION/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        LIT_OUT.mkdir(parents=True)
        runs=sorted(p for p in (ROOT/'output/classification_comparison').iterdir() if (p/'status.json').exists() and json.loads((p/'status.json').read_text())['status']=='completed')
        run=runs[-1];rows=[];INPUTS=[]
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
        for name,arm in [('Local Class A','baseline'),('Local Class B','final')]:
            path=run/arm/'test_predictions.csv';INPUTS.append(path)
            f=pd.read_csv(path);y=f.target_class;p=f.prediction;d=f.predicted_class
            rows.append({'origin':'local','model':name,'n':len(f),'first_signal':f.signal_date.min(),'last_signal':f.signal_date.max(),'ROC_AUC':roc_auc_score(y,p),'accuracy':accuracy_score(y,d),'F1':f1_score(y,d,zero_division=0),'AP':average_precision_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,d),'test_majority_accuracy':max(y.mean(),1-y.mean()),'label':'open-to-open >0.002'})
        rows += [{'origin':'paper reported','model':'Qin XGBoost','n':None,'first_signal':'NR','last_signal':'NR','ROC_AUC':.5393,'accuracy':.5360,'F1':.5550,'AP':None,'balanced_accuracy':None,'test_majority_accuracy':.5690,'label':'next adjusted close > current adjusted close'}]
        PAPER={'authors':'Mingyu Qin','year':2026,'doi':'10.70267/aitia.2026412420','url':'https://journals.zeuspress.org/index.php/conference/article/download/1343/1306','location':'Table 1, PDF p5 / printed p416','publication':'conference paper','unit':'fractions, not percentages','not_reported':['AP','balanced accuracy','exact test dates','test n'],'not_reproduced':True}
        literature_table=pd.DataFrame(rows)

        literature_table.to_csv(LIT_OUT/'literature_comparison.csv',index=False,encoding='utf-8-sig')
        (LIT_OUT/'extraction.json').write_text(json.dumps({'checked_on':'2026-11-24','paper':PAPER,'table':literature_table.replace({np.nan:None}).to_dict('records'),'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}},ensure_ascii=False,indent=2),encoding='utf-8')
        print(literature_table.fillna('N/A').to_string())
        print('Saved:',LIT_OUT.relative_to(ROOT))

def regression():
    """Run stage 05 (05_regression_training)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 05 回归训练：基线与最终方案
    # 
    # | 基线 | 最终方案 |
    # |---|---|
    # | 基线 · 6 特征 + XGBoost | 最终 · 6 特征 + 岭回归 |
    # 
    # - 在 t 日收盘预测 `open[t+2] / open[t+1] - 1`。回归直接预测该收益率。
    # - 两方案共用 train : val : test = 8 : 1 : 1 区间，按时间序列划分且不打乱。
    # - 每次运行独立保存到 `output/regression_comparison/[timestamp]/`，两方案训练、测试分别存放。
    # 整理说明：读取第 02 步的特征表；模型、训练函数、验证选参和单独测试流程沿用原 Notebook。

    # %% [markdown]
    # ## 1. 导入与配置

    # %%
    # 脚本集中在 python/；Notebook 保留在项目根目录。
    from pathlib import Path
    import sys
    PYTHON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "python"
    if not (PYTHON_DIR / "experiment_utils.py").is_file():
        raise FileNotFoundError("请在 SP500_organized 项目根目录打开 Notebook。")
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import experiment_utils

    from experiment_utils import (
        RAW_FILE, CLEAN_FILE, FEATURE_DIR, ORIGINAL, BP, Run,
        stage_log, LoggedEarlyStopping, load_prepared_samples, chronological_split,
        save_csv, save_json, new_training_dir, save_training, evaluate_saved,
    )
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # %%
    # 树参数沿用原回归方案；验证 MAE 选 XGBoost 轮次，验证 RMSE 选岭回归 alpha。
    # 不扩大搜索范围，避免把这次整理变成另一套实验。
    XGB_PARAMS = {
        "tree_method": "hist",       # 直方图分裂，适合这份日频小数据
        "max_depth": 3,              # 限制树深，降低过拟合风险
        "eta": 0.03,                 # 每轮学习步长
        "min_child_weight": 5,
        "subsample": 0.8,            # 每轮抽取部分训练行
        "colsample_bytree": 0.9,     # 每棵树抽取部分特征
        "lambda": 5.0,               # L2 正则化
        "seed": 42, "nthread": 4,
    }
    MAX_ROUNDS = 400
    EARLY_STOPPING = 30              # 验证指标连续 30 轮未改善时停止

    ALPHAS = [0.1, 10, 100, 1_000, 10_000]

    SCENARIO = "regression_comparison"
    ARM_LABELS = {'baseline': '基线 · 6 特征 + XGBoost', 'final': '最终 · 6 特征 + 岭回归'}
    FEATURE_SETS = {"baseline": ORIGINAL, "final": ORIGINAL}
    CONFIG = {
        "task": "regression", "evaluation": "single_chronological_holdout",
        "split_ratios": [0.8, 0.1, 0.1], "horizon_trading_days": 1,
        "positive_threshold": 0.002, "label": "open[t+2] / open[t+1] - 1",
        "last_data_date": "2026-11-12", "seed": 42,
        "tree_params": XGB_PARAMS, "max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING,
    }
    CONFIG["ridge_alpha_candidates"] = ALPHAS
    RUN = experiment_utils.start_run(SCENARIO, CONFIG, [RAW_FILE, CLEAN_FILE, FEATURE_DIR / "regression_samples.csv", FEATURE_DIR / "feature_build.json"])
    arm_runs = {}
    for arm in ARM_LABELS:
        folder = RUN.path / arm
        folder.mkdir()
        arm_config = dict(CONFIG, arm=ARM_LABELS[arm], features=FEATURE_SETS[arm])
        save_json(folder / "config.json", arm_config)
        arm_runs[arm] = Run(folder, arm_config, RUN.logger)

    # %% [markdown]
    # ## 2. 特征与标签
    # 
    # | 特征 | 计算方式 |
    # |---|---|
    # | `return_1d` / `return_5d` / `return_20d` | 收盘价 / 1、5、20 个交易日前收盘价 − 1 |
    # | `volatility_20d` | 最近 20 个日收益率的样本标准差，ddof=1 |
    # | `intraday_return` | 当日收盘价 / 开盘价 − 1 |
    # | `intraday_range` |（最高价 − 最低价）/ 开盘价 |
    # 
    # 原 6 特征均由 OHLC 价格计算，不含成交量。收益率使用小数形式。

    # %%
    # 特征公式集中在 02_feature_engineering，避免维护多份实现。

    # %%
    samples = load_prepared_samples("regression")
    save_csv(RUN.path / "features_and_targets.csv", samples)
    experiment_utils.show_table(samples[ORIGINAL + ["target_return", "target_class"]].tail(3))

    # %% [markdown]
    # ## 3. 单次按时间划分
    # 在本任务有效样本池上确定约 80% / 10% / 10% 的日期边界。入场、结算均须落在同一分区，边界处剔除越界标签，因此最终数量略有变化。两方案使用同一训练集、验证集和测试集。

    # %%
    # 与原代码相同的默认边界；配置中的比例实际传入划分函数。
    splits, split_summary = chronological_split(samples, CONFIG["split_ratios"])
    save_csv(RUN.path / "split_summary.csv", split_summary, index=False)
    experiment_utils.show_table(split_summary)
    RUN.logger.info("单次时序划分：训练 %s / 验证 %s / 测试 %s",
                    *(len(splits[k]) for k in ("train", "val", "test")))

    # %% [markdown]
    # ## 4. 定义 XGBoost 训练与调参函数
    # 保留历史基线的平方误差训练、验证 MAE 选轮次设置。函数只接收训练和验证数据。

    # %%
    def train_xgboost(run, stage, train, val, columns):
        """只接收训练和验证数据；函数内不接触测试集。"""
        folder = new_training_dir(run, stage)
        stage_log(run, folder, f"{stage} 开始 XGBoost 训练 | train={len(train)} / val={len(val)}")
        label = "target_return"
        dtrain = xgb.DMatrix(train[columns], label=train[label], nthread=4)
        dval = xgb.DMatrix(val[columns], label=val[label], nthread=4)
        params = dict(XGB_PARAMS, objective="reg:squarederror", eval_metric="mae")
        history = {}

        # 每次训练创建新的回调对象；save_best=True 只保留验证集最佳轮次的树。
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=MAX_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=history,
            callbacks=[
                # 每一轮先打印训练/验证指标，再检查早停，最后一轮也完整记录。
                LoggedEarlyStopping(run, folder, log_every=1,
                    rounds=EARLY_STOPPING, metric_name="mae",
                    data_name="val", maximize=False, save_best=True,
                ),
            ],
            verbose_eval=False,
        )
        selected = {
            "params": params, "best_rounds": int(model.best_iteration + 1),
            "trained_rounds": len(history["val"]["mae"]),
            "validation_metric": "mae", "validation_score": float(model.best_score),
            "prediction_unit": "return",
            "training_window": "同一固定训练集",
        }

        # 保留历史基线的“平方误差训练 + MAE 选轮次”，用于呈现改进前的状态。
        # 这是旧设定的复现；最终回归方案统一按 RMSE 选参和评价。
        stage_log(run, folder, f"验证 MAE 最佳轮次 {selected['best_rounds']}")

        stage_log(run, folder, f"实际执行 {selected['trained_rounds']} 轮；保留最佳 {selected['best_rounds']} 轮模型")
        return save_training(run, stage, folder, model, {"train": train, "val": val},
                             columns, selected, history)

    # %% [markdown]
    # ## 5. 定义岭回归调参函数
    # 标准化只拟合训练集；在同一验证集上比较 5 个 alpha，按 RMSE 选择最佳者。

    # %%
    def train_ridge(run, stage, train, val, columns):
        """在同一个固定训练集中，用验证 RMSE 选择 alpha。"""
        folder = new_training_dir(run, stage)
        stage_log(run, folder, f"{stage} 开始岭回归调参 | train={len(train)} / val={len(val)}")
        candidates, fitted = [], []
        for alpha in ALPHAS:
            candidate_dir = folder / f"alpha_{alpha:g}"
            stage_log(run, candidate_dir, f"拟合 alpha={alpha:g}")
            # Pipeline 保证标准化只在训练数据上拟合；验证与测试使用同一变换。
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[columns], train["target_return"] * BP)
            validation_prediction_bp = model.predict(val[columns])
            error_bp = validation_prediction_bp - val["target_return"].to_numpy() * BP
            rmse_bp = float(np.sqrt(np.mean(error_bp ** 2)))
            candidates.append({"alpha": alpha, "validation_rmse_bp": rmse_bp})
            fitted.append(model)
            # 每次候选训练分别保存模型、验证预测和日志，可追溯选择过程。
            joblib.dump(model, candidate_dir / "model.joblib")
            save_json(candidate_dir / "metrics.json", candidates[-1])
            save_csv(candidate_dir / "validation_predictions.csv", val[
                ["trade_date", "settle_date", "target_return"]
            ].assign(prediction=validation_prediction_bp / BP))
            stage_log(run, candidate_dir, f"alpha={alpha:g} | 验证 RMSE={rmse_bp:.6f} 基点")

        search = pd.DataFrame(candidates)
        best_index = int(search["validation_rmse_bp"].idxmin())
        selected = dict(candidates[best_index], prediction_unit="basis_points",
                        training_window="同一固定训练集")
        save_csv(folder / "alpha_search.csv", search, index=False)
        model = fitted[best_index]
        # 系数对应标准化后的特征；预测输出为基点，保存收益率时统一除以 10,000。
        save_csv(folder / "coefficients.csv", pd.DataFrame({
            "feature": columns, "coefficient_bp": model.named_steps["ridge"].coef_,
        }), index=False)
        stage_log(run, folder, f"选定 alpha={selected['alpha']:g}")
        return save_training(run, stage, folder, model, {"train": train, "val": val}, columns, selected)

    # %% [markdown]
    # ## 实际运行训练
    # 只循环比较两种方案，每种方案仅训练、调参一次。XGBoost 每轮打印训练与验证指标，结束时分别打印实际执行轮次与最佳轮次。最佳为第 1 轮不代表只训练了 1 轮：连续 30 轮没有改善才会早停。岭回归逐个打印 alpha 候选的验证 RMSE。模型和日志分别保存。

    # %%
    TRAINERS = {"baseline": train_xgboost, "final": train_ridge}
    artifacts = {}
    for arm in ARM_LABELS:
        RUN.logger.info("训练方案：%s", ARM_LABELS[arm])
        artifacts[arm] = TRAINERS[arm](
            arm_runs[arm], "holdout", splits["train"], splits["val"], FEATURE_SETS[arm])

    # %% [markdown]
    # ## 单独运行测试
    # 读取已保存模型，在同一个测试集上分别评价。测试数据不参与轮次、分类阈值或 alpha 的选择。

    # %%
    rows = []
    for arm in ARM_LABELS:
        RUN.logger.info("测试方案：%s", ARM_LABELS[arm])
        metrics, predictions = evaluate_saved(arm_runs[arm], "holdout", artifacts[arm], splits["test"])
        metrics.pop("period", None)
        rows.append(dict(scenario=ARM_LABELS[arm], **metrics))
        save_csv(arm_runs[arm].path / "test_predictions.csv", predictions)
        save_json(arm_runs[arm].path / "summary.json", metrics)
    comparison = pd.DataFrame(rows)
    save_csv(RUN.path / "comparison_summary.csv", comparison, index=False)
    save_json(RUN.path / "status.json", {"status": "completed", "evaluation": "single_chronological_holdout"})

    # %% [markdown]
    # ## 同一测试集的结果对比
    # 误差单位为基点，越低越好。MAE 对训练中位数，RMSE 对训练均值；ΔRMSE 为模型减常数。这里只报告这一次测试的指标。

    # %%
    experiment_utils.show_table(comparison[["scenario", "n_test", "mae_bp", "rmse_bp", "constant_rmse_bp", "delta_rmse_bp"]])
    print("本次输出目录：", experiment_utils.display_path(RUN.path))

    # %% [markdown]
    # 每个方案的 `train_时间戳/` 保存模型、选参记录和训练/验证预测；`test_时间戳/` 保存测试日志与预测。辅助读取、保存逻辑位于 `experiment_utils.py`。

    # Optional fusion stage. Run after the original experiment, using its saved dates.
    import os
    from experiment_utils import (ROOT, read_fusion_frame as read_frame,
        new_fusion_output as new_output, latest_completed_run as complete_run,
        latest_training_folder as training_folder, regression_scores, save_json as write_json)
    FEATURES = experiment_utils.ORIGINAL
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value)

    def run_fusion_extension():
        from sklearn.neural_network import MLPRegressor
        from copy import deepcopy
        OUT = new_output('05_mlp')
        SOURCE_RUN = complete_run('regression_comparison')
        source_train = training_folder(SOURCE_RUN,'baseline')
        samples = read_frame(ROOT/'data/features/regression_samples.csv')
        parts = {s: samples.loc[read_frame(source_train/f'{s}_predictions.csv').index].copy() for s in ['train','val']}
        parts['test'] = samples.loc[read_frame(SOURCE_RUN/'baseline/test_predictions.csv').index].copy()
        assert parts['train'].settle_date.max() <= parts['val'].index.min()
        assert parts['val'].settle_date.max() <= parts['test'].index.min()
        xs = StandardScaler().fit(parts['train'][FEATURES])
        ys = StandardScaler().fit(parts['train'][['target_return']])
        X = {s:xs.transform(f[FEATURES]) for s,f in parts.items()}
        ytrain = ys.transform(parts['train'][['target_return']]).ravel()
        candidates = [((32,),.001,.001),((64,32),.001,.001),((64,32),.01,.0005),((16,),.01,.001)]
        history=[]; choices=[]
        for i,(hidden,alpha,lr) in enumerate(candidates):
            m = MLPRegressor(hidden_layer_sizes=hidden,alpha=alpha,learning_rate_init=lr,random_state=42,early_stopping=False,max_iter=1)
            best=np.inf; stale=0
            for epoch in range(1,301):
                m.partial_fit(X['train'],ytrain)
                p=ys.inverse_transform(m.predict(X['val']).reshape(-1,1)).ravel()
                loss=np.mean((p-parts['val'].target_return.to_numpy())**2)
                history.append({'candidate':i,'epoch':epoch,'val_mse':loss,'train_loss_standardized':m.loss_})
                if loss < best:
                    best=loss; saved=deepcopy(m); best_epoch=epoch; stale=0
                else: stale+=1
                if stale>=20: break
            choices.append((best,i,best_epoch,saved))
            print('Candidate',i,'best epoch',best_epoch,'validation MSE',best)
        best_loss,best_id,best_epoch,model=min(choices,key=lambda x:(x[0],x[1]))
        bundle={'model':model,'x_scaler':xs,'y_scaler':ys,'features':FEATURES}
        joblib.dump(bundle,OUT/'model.joblib')
        loaded=joblib.load(OUT/'model.joblib')
        rows=[]
        for s,f in parts.items():
            p=ys.inverse_transform(model.predict(X[s]).reshape(-1,1)).ravel()
            q=loaded['y_scaler'].inverse_transform(loaded['model'].predict(loaded['x_scaler'].transform(f[FEATURES])).reshape(-1,1)).ravel()
            np.testing.assert_array_equal(p,q)
            f[['trade_date','settle_date','target_return','target_class']].assign(prediction=q).to_csv(OUT/f'{s}_predictions.csv')
            rows.append({'split':s,**regression_scores(f.target_return,q)})
        pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        write_json(OUT/'config.json',{'source':'material1/src/train_mlp.py','seed':42,'candidates':candidates,'selected':best_id,'epoch':best_epoch,'max_epochs':300,'patience':20,'features':FEATURES,'source_run':str(SOURCE_RUN.relative_to(ROOT)),'reload_exact':True})
        write_json(ROOT/'output/fusion/mlp_latest.json',{'path':str(OUT.relative_to(ROOT))})
        display(pd.DataFrame(rows))
        print('Saved:',OUT)


    if os.environ.get("SP500_INCLUDE_FUSION") == "1":
        run_fusion_extension()

    if os.environ.get("SP500_INCLUDE_LITERATURE") == "1":
        SECTION='05_regression'
        import json, hashlib
        from pathlib import Path
        from datetime import datetime
        import numpy as np
        import pandas as pd
        ROOT = ROOT
        assert (ROOT/'01_data_preprocessing.ipynb').is_file()
        LIT_OUT = ROOT/'output/literature'/SECTION/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        LIT_OUT.mkdir(parents=True)
        mlp=ROOT/json.loads((ROOT/'output/fusion/mlp_latest.json').read_text())['path']
        run=ROOT/json.loads((mlp/'config.json').read_text())['source_run'];rows=[];INPUTS=[mlp/'config.json']
        for name,path in [('Local Regression XGB',run/'baseline/test_predictions.csv'),('Local Ridge',run/'final/test_predictions.csv')]:
            INPUTS.append(path);f=pd.read_csv(path);y=f.target_return.to_numpy();p=f.prediction.to_numpy();mse=np.mean((y-p)**2)
            rows.append({'origin':'local','model':name,'n':len(f),'MSE_percent_squared':mse*10000,'RMSE_bps':np.sqrt(mse)*10000,'MAE_bps':abs(y-p).mean()*10000,'R2_test_mean':1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),'R2_oos_20day_mean':None,'direction_accuracy':np.mean((y>0)==(p>0)),'RMSE_source':'computed from saved predictions'})
        for name,mse,r2pct in [('Ridge',1.152,-17.832),('XGB',1.072,-2.093),('MLP1',1.298,-49.676),('MLP2',1.189,-25.527),('MLP3',1.123,-12.049),('MLP4',1.114,-10.307),('MLP5',1.064,-.650)]:
            rows.append({'origin':'paper reported; metric ambiguity','model':'Chun '+name,'n':2348,'paper_MSE_as_labeled':mse,'MSE_percent_squared':None,'RMSE_bps':None,'MAE_bps':None,'R2_test_mean':None,'R2_oos_20day_mean':r2pct/100,'direction_accuracy':None,'RMSE_source':'N/A: not safely derivable from ambiguous MSE label'})
        PAPER={'authors':'Dohyun Chun; Jongho Kang; Jihun Kim','year':2024,'doi':'10.1186/s40854-024-00648-w','url':'https://link.springer.com/content/pdf/10.1186/s40854-024-00648-w.pdf','location':'Table 3 panel b PDF p11; Eq5; Table2 note','test':'2012-01-02 to 2020-08-31','unit':'reported R2_oos percent divided by 100; MSE kept as labeled, no RMSE conversion due internal consistency ambiguity','not_reproduced':True}
        literature_table=pd.DataFrame(rows)

        literature_table.to_csv(LIT_OUT/'literature_comparison.csv',index=False,encoding='utf-8-sig')
        (LIT_OUT/'extraction.json').write_text(json.dumps({'checked_on':'2026-11-24','paper':PAPER,'table':literature_table.replace({np.nan:None}).to_dict('records'),'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}},ensure_ascii=False,indent=2),encoding='utf-8')
        display(literature_table.fillna('N/A'))
        print('Saved:',LIT_OUT.relative_to(ROOT))

def comparison():
    """Run stage 06 (06_compare_results)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 06 最终结果对比
    # 读取第 04、05 步各自最新的完整运行，只汇总已保存的测试结果，不重新调参。
    # 分类比较 A/B，回归比较 XGBoost/岭回归；两个任务的样本日期不同，分别列出，不把指标混成一个总分。

    # %% [markdown]
    # ## 1. 选择已完成的运行
    # 默认选择最新完整结果；复核旧运行时，可明确填写下面两个相对路径。

    # %%
    # 脚本集中在 python/；Notebook 保留在项目根目录。
    from pathlib import Path
    import sys
    PYTHON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "python"
    if not (PYTHON_DIR / "experiment_utils.py").is_file():
        raise FileNotFoundError("请在 SP500_organized 项目根目录打开 Notebook。")
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))

    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from experiment_utils import ROOT, start_run, save_csv, save_json, show_table, show_figure

    CLASSIFICATION_RUN = None  # 示例："output/classification_comparison/时间戳"
    REGRESSION_RUN = None      # 示例："output/regression_comparison/时间戳"

    def completed_run(scenario, explicit=None):
        """不读取未完成的训练，也不依赖原项目的历史输出。"""
        if explicit is not None:
            path = Path(explicit)
            candidates = [path if path.is_absolute() else ROOT / path]
        else:
            candidates = sorted((ROOT / "output" / scenario).glob("*"), reverse=True)
        for folder in candidates:
            status_file = folder / "status.json"
            if not status_file.is_file():
                continue
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if (status.get("status") == "completed"
                    and (folder / "comparison_summary.csv").is_file()
                    and all((folder / arm / "test_predictions.csv").is_file()
                            for arm in ("baseline", "final"))):
                return folder
        raise FileNotFoundError(f"没有完整的 {scenario} 结果，请先运行对应训练文件。")

    from pathlib import Path
    classification_run = completed_run("classification_comparison", CLASSIFICATION_RUN)
    regression_run = completed_run("regression_comparison", REGRESSION_RUN)
    print("分类来源：", classification_run.relative_to(ROOT))
    print("回归来源：", regression_run.relative_to(ROOT))

    # %% [markdown]
    # ## 2. 核查两方案使用相同测试样本
    # 对每个任务核对信号日、入场日、结算日与真实标签；分类和回归之间不要求日期相同。

    # %%
    def read_comparison(folder, task):
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config["task"] != task:
            raise ValueError("任务与结果目录不匹配。")
        label = "target_class" if task == "classification" else "target_return"
        predictions = {}
        for arm in ("baseline", "final"):
            predictions[arm] = pd.read_csv(folder / arm / "test_predictions.csv",
                                          index_col="signal_date",
                                          parse_dates=["signal_date", "trade_date", "settle_date"])
        columns = ["trade_date", "settle_date", label]
        pd.testing.assert_frame_equal(predictions["baseline"][columns], predictions["final"][columns])
        table = pd.read_csv(folder / "comparison_summary.csv")
        assert len(table) == 2 and table["n_test"].eq(len(predictions["baseline"])).all()
        split_summary = pd.read_csv(folder / "split_summary.csv")
        return table, split_summary

    classification, classification_split = read_comparison(classification_run, "classification")
    regression, regression_split = read_comparison(regression_run, "regression")
    classification_metrics = ["scenario", "n_test", "roc_auc", "ap", "accuracy", "balanced_accuracy", "f1"]
    regression_metrics = ["scenario", "n_test", "mae_bp", "rmse_bp", "r2", "constant_rmse_bp", "delta_rmse_bp"]
    show_table(classification[classification_metrics])
    show_table(regression[regression_metrics])

    # %% [markdown]
    # ## 3. 保存对比表与差值
    # 差值统一为“final − baseline”。分类 AUC、AP 等越高越好；回归 MAE、RMSE 越低越好。

    # %%
    sources = [folder / file for folder in (classification_run, regression_run)
               for file in ("comparison_summary.csv", "split_summary.csv", "config.json")]
    RUN = start_run("final_comparison", {
        "classification_run": classification_run.relative_to(ROOT).as_posix(),
        "regression_run": regression_run.relative_to(ROOT).as_posix(),
        "delta": "final minus baseline", "retraining": False,
    }, sources)
    save_csv(RUN.path / "classification_comparison.csv", classification, index=False)
    save_csv(RUN.path / "regression_comparison.csv", regression, index=False)
    delta_rows = []
    for task, table, metrics in (
        ("classification", classification, ["roc_auc", "ap", "accuracy", "balanced_accuracy", "f1"]),
        ("regression", regression, ["mae_bp", "rmse_bp", "r2"]),
    ):
        for metric in metrics:
            base, final = float(table.iloc[0][metric]), float(table.iloc[1][metric])
            delta_rows.append({"task": task, "metric": metric, "baseline": base, "final": final,
                               "delta_final_minus_baseline": final - base,
                               "better_direction": "lower" if metric in ("mae_bp", "rmse_bp") else "higher"})
    deltas = pd.DataFrame(delta_rows)
    save_csv(RUN.path / "metric_differences.csv", deltas, index=False)
    show_table(deltas)

    # %% [markdown]
    # ## 4. 分类与回归分图展示

    # %%
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ["#7189a6", "#2977c9"]
    for ax, table, metric, title, labels in (
        (axes[0, 0], classification, "roc_auc", "Classification ROC-AUC (higher is better)", ["A: 6 features", "B: 11 features"]),
        (axes[0, 1], classification, "ap", "Classification AP (higher is better)", ["A: 6 features", "B: 11 features"]),
        (axes[1, 0], regression, "mae_bp", "Regression MAE (bp, lower is better)", ["XGBoost", "Ridge"]),
        (axes[1, 1], regression, "rmse_bp", "Regression RMSE (bp, lower is better)", ["XGBoost", "Ridge"]),
    ):
        bars = ax.bar(labels, table[metric], color=colors, width=0.55)
        if hasattr(ax, "bar_label"):
            ax.bar_label(bars, fmt="%.4f", padding=3)
        else:
            for bar in bars:
                ax.annotate(f"{bar.get_height():.4f}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(table[metric]) * 1.2)
        ax.grid(axis="y", alpha=0.2)
        if metric in ("roc_auc", "ap"):
            reference = 0.5 if metric == "roc_auc" else classification.iloc[0]["positive_rate"]
            ax.axhline(reference, color="#999999", linestyle="--", label="Reference")
            ax.legend(fontsize=8)
    show_figure(fig, RUN.path / "comparison.png")

    # %% [markdown]
    # ## 5. 输出可用于报告的汇总说明

    # %%
    def markdown_table(table):
        """避免额外依赖，直接生成便于阅读的 Markdown 表格。"""
        shown = table.copy()
        for column in shown.select_dtypes(include="number").columns:
            shown[column] = shown[column].map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
        lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
        for row in shown.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(lines)

    report = "# 最终结果对比\n\n"
    report += f"分类来源：`{classification_run.relative_to(ROOT).as_posix()}`\n\n"
    report += f"回归来源：`{regression_run.relative_to(ROOT).as_posix()}`\n\n"
    report += "## 分类：A 组与 B 组\n\n" + markdown_table(classification[classification_metrics]) + "\n\n"
    report += "## 回归：XGBoost 与岭回归\n\n" + markdown_table(regression[regression_metrics]) + "\n\n"
    report += "## 实际样本划分\n\n分类：\n\n" + markdown_table(classification_split) + "\n\n"
    report += "回归：\n\n" + markdown_table(regression_split) + "\n\n"
    report += "## 指标差值（final − baseline）\n\n" + markdown_table(deltas) + "\n\n"
    report += "回归收益率以基点展示，1% = 100 基点。delta_rmse_bp 为模型减训练均值常数基准，负值表示改善。\n\n"
    report += "分类新增跨市场特征只改变输入；回归比较改变模型。14 特征分析表没有自动成为分类 B 组输入。\n\n"
    report += "这些是单次历史时序留出结果，数据此前已参与方案探索；小幅差值不构成统计显著性证据，也不保证未来有效。\n"
    (RUN.path / "final_comparison.md").write_text(report, encoding="utf-8")
    save_json(RUN.path / "status.json", {"status": "completed"})
    print("最终报告：", (RUN.path / "final_comparison.md").relative_to(ROOT))

    # Optional fusion stage. Run after the original experiment, using its saved dates.
    import os
    from experiment_utils import (read_fusion_frame as read_frame,
                                  new_fusion_output as new_output, regression_scores)
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value)

    def run_fusion_extension():
        OUT = new_output('06_mlp_comparison')
        mlp = ROOT/json.loads((ROOT/'output/fusion/mlp_latest.json').read_text())['path']
        run = ROOT/json.loads((mlp/'config.json').read_text())['source_run']
        a=read_frame(run/'baseline/test_predictions.csv'); b=read_frame(run/'final/test_predictions.csv'); c=read_frame(mlp/'test_predictions.csv')
        rows=[]
        for name,f in [('Regression XGB',a),('Ridge',b),('MLP',c)]:
            pd.testing.assert_index_equal(a.index,f.index)
            np.testing.assert_allclose(a.target_return,f.target_return,atol=1e-15)
            rows.append({'model':name,**regression_scores(f.target_return,f.prediction)})
        for col in ['constant_mean','constant_median']:
            rows.append({'model':col,**regression_scores(a.target_return,a[col])})
        table=pd.DataFrame(rows);table.to_csv(OUT/'mlp_comparison.csv',index=False);display(table)


    if os.environ.get("SP500_INCLUDE_FUSION") == "1":
        run_fusion_extension()

def validation():
    """Run stage 07 (07_validation_experiments)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 07 补充实验：数据泄漏与划分方式
    # 
    # 本文件合并已有实验，公共准备只做一次，并复用相同设置下的正常时序基线。
    # 
    # 1. **公共准备**：数据检查、原 6 特征、分类/回归标签、统一 XGBoost 参数和训练函数。
    # 2. **实验① 数据泄漏影响**：固定时序划分，正常输入与故意加入未来信息的输入对照。
    # 3. **实验② 随机与时序划分**：只用原 6 特征，一次时序与 10 次随机划分对照。
    # 4. **汇总与分析**：分类、回归指标；随机结果报告均值和样本标准差。
    # 
    # 两任务均用 **XGBoost**。在 t 日收盘预测 `open[t+2] / open[t+1] - 1`，分类为收益严格大于 +0.20%。
    # 从上到下运行全部即可。总共训练和测试 **24 个模型**：正常时序 2 个、时序泄漏 2 个、随机正常 20 个。正常时序的两个模型只运行一次。
    # 
    # 这是原有补充实验，主流程 01–06 不自动运行；可单独从上至下执行。

    # %% [markdown]
    # ## 一、公共准备
    # 
    # > 数据检查包括日期排序与唯一性、OHLC 有限且大于零；价格按原项目保留三位小数。删除滚动窗口不足或没有完整未来标签的样本，不用未来值填补缺失。

    # %%
    # 脚本集中在 python/；Notebook 保留在项目根目录。
    from pathlib import Path
    import sys
    PYTHON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "python"
    if not (PYTHON_DIR / "experiment_utils.py").is_file():
        raise FileNotFoundError("请在 SP500_organized 项目根目录打开 Notebook。")
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))

    import importlib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from experiment_utils import CLEAN_FILE, show_figure
    from fractions import Fraction
    from datetime import datetime
    from sklearn.metrics import (
        roc_curve, roc_auc_score, average_precision_score, accuracy_score,
        balanced_accuracy_score, f1_score, confusion_matrix, r2_score,
    )
    import experiment_utils as utils
    utils = importlib.reload(utils)
    from experiment_utils import (
        RAW_FILE, ORIGINAL, BP, Run, load_daily, save_csv, save_json,
        stage_log, LoggedEarlyStopping, new_training_dir, save_training,
    )

    # 在这里修改模型设置；CONFIG 会把本次设置保存到文件中。
    XGB_PARAMS = {
        "tree_method": "hist",     # 用直方图寻找树的分裂点
        "max_depth": 3,            # 每棵树的最大深度
        "eta": 0.03,               # 学习率：每棵新树的贡献大小
        "min_child_weight": 5,     # 限制过小的叶节点
        "subsample": 0.8,          # 每轮使用 80% 的训练行
        "colsample_bytree": 0.9,   # 每棵树使用 90% 的特征
        "lambda": 5.0,             # L2 正则化，约束叶节点权重
        "seed": 42,                # 固定模型随机种子
        "nthread": 4,              # 使用 4 个计算线程
    }
    MAX_ROUNDS = 400                # 最多训练 400 轮
    EARLY_STOPPING = 30             # 验证指标连续 30 轮不改善就停止
    RANDOM_SEEDS = list(range(42, 52))
    LEAK = "LEAK_future_intraday_return"
    CONFIG = {
        "models": {"classification": "XGBoost", "regression": "XGBoost"},
        "features": ORIGINAL, "leaked_feature": LEAK, "random_seeds": RANDOM_SEEDS,
        # 顺序为训练、验证、测试；划分函数会实际读取这三个比例。
        "split_ratios": [0.8, 0.1, 0.1], "tree_params": XGB_PARAMS,
        "max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING,
        "regression_objective": "reg:squarederror", "regression_selection_metric": "rmse",
        "label": "open[t+2] / open[t+1] - 1",
        "classification_threshold": 0.002,  # 0.002 = 0.20%，标签函数实际读取此值
        "warning": "故意泄漏组只作反例，历史结果属于开发数据上的教学对照",
    }
    RUN = utils.start_run('validation_experiments', CONFIG, [RAW_FILE, CLEAN_FILE])

    # %% [markdown]
    # ### 原 6 特征
    # 
    # | 特征 | 计算方式 |
    # |---|---|
    # | `return_1d` / `return_5d` / `return_20d` | 收盘价 / 1、5、20 个交易日前收盘价 − 1 |
    # | `volatility_20d` | 最近 20 个日收益率的样本标准差，ddof=1 |
    # | `intraday_return` | 当日收盘价 / 开盘价 − 1 |
    # | `intraday_range` |（最高价 − 最低价）/ 开盘价 |
    # 
    # 原 6 特征均由 OHLC 价格计算，不含成交量。收益率使用小数形式。
    # 

    # %%
    def build_original_features(bars):
        """只使用 t 日收盘时已经知道的价格；滚动计算不向未来取值。"""
        close = bars["close"]
        features = pd.DataFrame(index=bars.index)
        for days in (1, 5, 20):
            features[f"return_{days}d"] = close.pct_change(days, fill_method=None)
        features["volatility_20d"] = features["return_1d"].rolling(20).std(ddof=1)
        features["intraday_return"] = close / bars["open"] - 1
        features["intraday_range"] = (bars["high"] - bars["low"]) / bars["open"]
        return features

    # %%
    def build_targets(bars, features, positive_threshold):
        """输入价格和特征，返回加入入场日、结算日、回归与分类标签的数据。"""
        dates = bars.index.to_series()
        entry_price = bars["open"].shift(-1)   # 当前行 t 对应的 t+1 开盘价
        exit_price = bars["open"].shift(-2)    # 当前行 t 对应的 t+2 开盘价

        samples = features.copy()
        samples["trade_date"] = dates.shift(-1)
        samples["settle_date"] = dates.shift(-2)
        samples["target_return"] = exit_price / entry_price - 1

        # 价格已保留三位小数，转为整数可避免浮点误差影响严格大于的边界。
        # Fraction 把 0.002 精确表示为 1/500；修改阈值时会重新计算分子和分母。
        threshold = Fraction(str(positive_threshold))
        entry_integer = (entry_price * 1000).round().astype("Int64")
        exit_integer = (exit_price * 1000).round().astype("Int64")
        price_change = exit_integer - entry_integer
        is_positive = price_change * threshold.denominator > entry_integer * threshold.numerator
        samples["target_class"] = is_positive.astype("Int8")

        # 前 20 日缺少滚动特征，最后两日缺少完整标签，因此删除这些行。
        samples = samples.replace([np.inf, -np.inf], np.nan)
        required_columns = ORIGINAL + ["target_return", "target_class"]
        return samples.dropna(subset=required_columns)

    # %%
    bars = load_daily()
    features = build_original_features(bars)
    samples = build_targets(bars, features, CONFIG["classification_threshold"])
    save_csv(RUN.path / "features_and_targets.csv", samples)
    utils.show_table(samples[ORIGINAL + ["target_return", "target_class"]].tail(3))

    # %% [markdown]
    # ### 定义划分与时间审计函数
    # 
    # 先按约 80/10/10 确定时序边界，剔除跨分区收益标签。随机划分沿用同一保留样本池及相同数量。特征在划分前按日期顺序计算，且只使用当时可知历史。

    # %%
    def temporal_split(samples, ratios):
        """按日期顺序划分；比例在 CONFIG 中设置，边界越界标签会被剔除。"""
        if len(ratios) != 3 or not np.isclose(sum(ratios), 1):
            raise ValueError("训练、验证、测试三个比例之和必须为 1。")
        if min(ratios) <= 0:
            raise ValueError("每个划分比例都必须大于 0。")
        validation_size = round(len(samples) * ratios[1])
        test_size = round(len(samples) * ratios[2])
        training_size = len(samples) - validation_size - test_size
        if min(training_size, validation_size, test_size) < 2:
            raise ValueError("分区太小，无法完成训练与评价。")

        validation_start = samples.iloc[training_size]["trade_date"]
        test_start = samples.iloc[training_size + validation_size]["trade_date"]
        # 训练标签必须在验证期之前结算；验证标签不能跨入测试期。
        train_mask = samples["settle_date"] < validation_start
        val_mask = (samples["trade_date"] >= validation_start) & (samples["settle_date"] < test_start)
        test_mask = samples["trade_date"] >= test_start
        splits = {}
        splits["train"] = samples.loc[train_mask].copy()
        splits["val"] = samples.loc[val_mask].copy()
        splits["test"] = samples.loc[test_mask].copy()
        assert splits["train"]["settle_date"].max() <= splits["val"].index.min()
        assert splits["val"]["settle_date"].max() <= splits["test"].index.min()
        return splits

    # %%
    def random_split(pool, counts, seed):
        """打乱行号，再按固定数量取三组；最后按日期排序，便于查看。"""
        random_numbers = np.random.default_rng(seed)
        shuffled_positions = random_numbers.permutation(len(pool))
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits = {}
        splits["train"] = pool.iloc[shuffled_positions[:train_end]].sort_index().copy()
        splits["val"] = pool.iloc[shuffled_positions[train_end:val_end]].sort_index().copy()
        splits["test"] = pool.iloc[shuffled_positions[val_end:]].sort_index().copy()
        return splits

    # %%
    def audit_split(name, splits):
        """检查拟合用到的训练标签，在每个预测时点是否已经发生。"""
        train_label_dates = splits["train"]["settle_date"].to_numpy()
        rows = []
        for part, frame in splits.items():
            future_fractions = []
            has_future_labels = []
            for signal_date in frame.index.to_numpy():
                # True 表示该训练标签在这个预测时点还未知。
                future_mask = train_label_dates > signal_date
                future_fractions.append(future_mask.mean())
                has_future_labels.append(future_mask.any())
            rows.append({
                "plan": name, "split": part, "n": len(frame),
                "first_signal": frame.index.min(), "last_signal": frame.index.max(),
                "positive_rate": float(frame["target_class"].mean()),
                "return_std_bp": float(frame["target_return"].std() * BP),
                "mean_future_train_label_fraction": float(np.mean(future_fractions)),
                "any_future_train_label_fraction": float(np.mean(has_future_labels)),
            })
        return rows

    # %% [markdown]
    # ### 定义共同的 XGBoost 训练与调参函数
    # 
    # 树参数在各组固定。分类使用验证 AUC 早停，再按验证平衡准确率选阈值；回归使用平方误差训练、验证 RMSE 早停。最多 400 轮、耐心值 30；逐轮打印训练/验证指标，包括早停最后一轮。测试不参与选择。

    # %%
    def train_xgboost(run, train, val, columns):
        task = run.config["task"]
        # 分类和回归共用训练流程，但标签、损失函数和选轮次指标不同。
        classifier = task == "classification"
        if classifier:
            label = "target_class"
            metric = "auc"
            objective = "binary:logistic"
        else:
            label = "target_return"
            metric = CONFIG["regression_selection_metric"]
            objective = CONFIG["regression_objective"]
        folder = new_training_dir(run, "holdout")
        stage_log(run, folder, f"开始训练 {run.config['case']} | {task} | 特征数={len(columns)}")
        # DMatrix 是 XGBoost 接收的特征和标签容器。
        dtrain = xgb.DMatrix(train[columns], label=train[label], nthread=4)
        dval = xgb.DMatrix(val[columns], label=val[label], nthread=4)
        params = XGB_PARAMS.copy()
        params["objective"] = objective
        params["eval_metric"] = metric
        history = {}
        # save_best=True 保存验证集最佳轮次，测试时不会使用后续较差的树。
        model = xgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")], evals_result=history,
            callbacks=[LoggedEarlyStopping(
                run, folder, log_every=1, rounds=EARLY_STOPPING,
                metric_name=metric, data_name="val", maximize=classifier, save_best=True,
            )], verbose_eval=False,
        )
        selected = {
            "params": params, "best_rounds": int(model.best_iteration + 1),
            "trained_rounds": len(history["val"][metric]), "validation_metric": metric,
            "validation_score": float(model.best_score),
            "prediction_unit": "probability" if classifier else "return",
            "intentional_future_feature": LEAK in columns,
        }
        if classifier:
            validation_probability = model.predict(dval)
            fpr, tpr, thresholds = roc_curve(val[label], validation_probability)
            # ROC 的第一个阈值可能为无穷，先排除；剩余候选与原方案一致。
            finite = np.isfinite(thresholds)
            valid_thresholds = thresholds[finite]
            # 平衡准确率 = (TPR + 1 - FPR) / 2，最大化它等价于最大化 TPR-FPR。
            threshold_scores = tpr[finite] - fpr[finite]
            best_position = np.argmax(threshold_scores)
            selected["threshold"] = float(valid_thresholds[best_position])
        stage_log(run, folder, f"实际 {selected['trained_rounds']} 轮，最佳 {selected['best_rounds']} 轮")
        return save_training(run, "holdout", folder, model, {"train": train, "val": val}, columns, selected, history)

    # %% [markdown]
    # ### 定义保存与测试流程
    # 
    # 训练和测试分开调用。测试从磁盘重新加载模型；每次训练和测试均创建独立目录。以下函数用于组织记录，核心训练逻辑在上一块。

    # %%
    def fit_case(plan_name, method, seed, mode, splits):
        """输入一个实验设置和三组数据，返回分类、回归两个已保存模型的记录。"""
        trained = []
        for task in ("classification", "regression"):
            columns = ORIGINAL.copy()
            if mode == "leak":
                columns.append(LEAK)
            folder = RUN.path / plan_name / task / mode
            folder.mkdir(parents=True, exist_ok=True)
            settings = CONFIG.copy()
            settings["task"] = task
            settings["case"] = f"{plan_name}/{task}/{mode}"
            settings["method"] = method
            settings["split_seed"] = seed
            settings["feature_mode"] = mode
            settings["features"] = columns
            save_json(folder / "config.json", settings)
            # Run 只保存输出目录、设置和日志对象，不包含模型算法。
            subrun = Run(folder, settings, RUN.logger)
            model_record = train_xgboost(subrun, splits["train"], splits["val"], columns)
            trained.append({
                "run": subrun, "artifact": model_record, "splits": splits,
                "plan_name": plan_name, "task": task, "method": method,
                "seed": seed, "feature_mode": mode,
            })
        return trained

    # %%
    def test_one_model(run, model_record, test):
        """重新加载模型，计算测试指标，分别保存预测、指标和日志。"""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = run.path / f"test_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        selected = model_record["selected"]
        model = xgb.Booster(params={"nthread": 4})
        model.load_model(model_record["folder"] / "model.ubj")
        test_matrix = xgb.DMatrix(test[selected["features"]], nthread=4)
        prediction = model.predict(test_matrix)
        assert np.isfinite(prediction).all()
        stage_log(run, folder, f"开始测试：{run.config['case']}，样本数={len(test)}")

        metrics = {"n_test": len(test)}
        if selected["task"] == "classification":
            label = "target_class"
            actual = test[label].to_numpy(dtype=float)
            threshold = selected["threshold"]  # 只使用验证集选好的概率阈值。
            predicted_class = (prediction >= threshold).astype(int)
            metrics["roc_auc"] = roc_auc_score(actual, prediction)
            metrics["ap"] = average_precision_score(actual, prediction)
            metrics["accuracy"] = accuracy_score(actual, predicted_class)
            metrics["balanced_accuracy"] = balanced_accuracy_score(actual, predicted_class)
            metrics["f1"] = f1_score(actual, predicted_class, zero_division=0)
            metrics["threshold"] = threshold
            metrics["positive_rate"] = float(actual.mean())
            matrix = confusion_matrix(actual, predicted_class, labels=[0, 1])
            save_csv(folder / "confusion_matrix.csv", pd.DataFrame(matrix, columns=["predicted_0", "predicted_1"]), index=False)
        else:
            label = "target_return"
            actual = test[label].to_numpy(dtype=float)
            # 收益率乘 10,000 转为基点；MAE 取绝对误差平均，RMSE 取平方误差平均的平方根。
            error_bp = (prediction - actual) * BP
            metrics["mae_bp"] = np.abs(error_bp).mean()
            metrics["rmse_bp"] = np.sqrt(np.mean(error_bp ** 2))
            metrics["r2"] = r2_score(actual, prediction)
            # 两种常数基准都只从训练集计算，不使用测试收益来拟合。
            metrics["constant_mae_bp"] = np.abs(actual - selected["train_median"]).mean() * BP
            metrics["constant_rmse_bp"] = np.sqrt(np.mean((actual - selected["train_mean"]) ** 2)) * BP
            metrics["delta_mae_bp"] = metrics["mae_bp"] - metrics["constant_mae_bp"]
            metrics["delta_rmse_bp"] = metrics["rmse_bp"] - metrics["constant_rmse_bp"]

        predictions = test[["trade_date", "settle_date", label]].copy()
        predictions["prediction"] = prediction
        if selected["task"] == "classification":
            predictions["predicted_class"] = predicted_class
        else:
            predictions["constant_mean"] = selected["train_mean"]
            predictions["constant_median"] = selected["train_median"]
        save_csv(folder / "predictions.csv", predictions)
        save_json(folder / "metrics.json", metrics)
        save_json(folder / "model_source.json", {"training_directory": str(model_record["folder"]), "selected": selected})
        stage_log(run, folder, f"测试完成：{metrics}")
        return metrics

    # %%
    def test_cases(trained):
        """逐个测试已训练模型，把各自结果整理为一张表。"""
        rows = []
        for item in trained:
            metrics = test_one_model(item["run"], item["artifact"], item["splits"]["test"])
            row = {}
            for key in ("plan_name", "task", "method", "seed", "feature_mode"):
                row[key] = item[key]
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    # %%
    def save_split(name, splits):
        """保存每一行被分到哪一组，便于老师核对样本没有重复使用。"""
        folder = RUN.path / "splits" / name
        folder.mkdir(parents=True, exist_ok=True)
        parts = []
        for part, frame in splits.items():
            current = frame.copy()
            current["split"] = part
            parts.append(current)
        assignments = pd.concat(parts).sort_index()
        assert assignments.index.is_unique
        save_csv(folder / "assignments.csv", assignments[["split", "trade_date", "settle_date"]])
        return audit_split(name, splits)

    # %% [markdown]
    # ## 二、实验① 数据泄漏的影响
    # 
    # 固定同一次时序划分，原 6 特征作为正常组。泄漏组增加 `intraday_return.shift(-1)`，即 `close[t+1] / open[t+1] - 1`，这个特征在 t+1 收盘才可获得。
    # 
    # 这是**故意引入未来信息**的反例，没有直接复制标签；两组全部样本日期和模型参数一致。此处不引入随机划分。

    # %%
    time_splits = temporal_split(samples, CONFIG["split_ratios"])
    pool = pd.concat(time_splits.values()).sort_index()
    counts = {}
    for part, frame in time_splits.items():
        counts[part] = len(frame)
    audit_rows = save_split("time", time_splits)
    future_intraday = features["intraday_return"].shift(-1)
    # 复制正常数据后只加一列，避免意外修改正常组。
    leak_splits = {}
    for part, frame in time_splits.items():
        leaked_frame = frame.copy()
        leaked_frame[LEAK] = future_intraday.reindex(frame.index)
        leak_splits[part] = leaked_frame
    for part in time_splits:
        pd.testing.assert_frame_equal(leak_splits[part].drop(columns=LEAK), time_splits[part])
        assert leak_splits[part][LEAK].notna().all()
        assert (leak_splits[part].trade_date > leak_splits[part].index).all()
    save_csv(RUN.path / "intentional_leak_feature.csv", pd.concat(leak_splits.values())[[LEAK, "trade_date"]].sort_index())
    utils.show_table(pd.DataFrame(audit_rows)[["plan", "split", "n", "first_signal", "last_signal"]])

    # %% [markdown]
    # ### 实际训练：正常与泄漏组
    # 
    # 正常时序基线在这里拟合一次，实验②直接复用。

    # %%
    # 每次调用只处理一种设置；返回分类和回归两个模型。
    normal_models = fit_case("time", "时序", -1, "clean", time_splits)
    leaked_models = fit_case("time", "时序", -1, "leak", leak_splits)
    experiment1_models = normal_models + leaked_models

    # %% [markdown]
    # ### 单独测试与基线缓存
    # 
    # 保存四个模型的测试结果，并记录供下一章复用的正常基线模型目录。

    # %%
    experiment1_results = test_cases(experiment1_models)
    baseline_results = experiment1_results.loc[experiment1_results.feature_mode == "clean"].copy()
    baseline_sources = {}
    for item in normal_models:
        baseline_sources[item["task"]] = str(item["artifact"]["folder"])
    save_json(RUN.path / "reused_baseline.json", {"trained_in": "experiment_1", "reused_in": "experiment_2",
                                                "training_directories": baseline_sources})
    save_csv(RUN.path / "experiment1_metrics.csv", experiment1_results, index=False)
    utils.show_table(experiment1_results[["task", "feature_mode", "n_test", "roc_auc", "ap", "mae_bp", "rmse_bp"]])

    # %% [markdown]
    # ## 三、实验② 随机划分与时序划分
    # 
    # 只使用原 6 个正常特征。时序参考直接取实验①的正常基线，不重新训练或测试；随机划分使用相同样本池与数量，种子 42–51 重复 10 次。
    # 
    # 两类测试样本覆盖时段和行情分布不同，指标差异同时反映这些因素，不能全部解释为泄漏效应。随机结果不保证更高，不挑选最佳种子。

    # %%
    random_cases = []
    for seed in RANDOM_SEEDS:
        plan_name = f"random_{seed}"
        splits = random_split(pool, counts, seed)
        for part in splits:
            assert len(splits[part]) == counts[part]
        pd.testing.assert_index_equal(pd.concat(splits.values()).sort_index().index, pool.index)
        assert LEAK not in splits["train"].columns
        audit_rows.extend(save_split(plan_name, splits))
        random_cases.append({"name": plan_name, "seed": seed, "splits": splits})
    audit = pd.DataFrame(audit_rows)
    assert audit.loc[(audit.plan == "time") & audit.split.isin(["val", "test"]), "mean_future_train_label_fraction"].eq(0).all()
    save_csv(RUN.path / "split_audit.csv", audit, index=False)
    utils.show_table(audit.loc[audit.split == "test"].drop(columns="split"))

    # %% [markdown]
    # ### 实际训练：仅训练随机组
    # 
    # 10 个随机划分 × 两个任务，共 20 个模型。XGBoost 自身的随机种子固定为 42，变化的仅是数据划分种子。

    # %%
    experiment2_random_models = []
    for case in random_cases:
        models = fit_case(case["name"], "随机", case["seed"], "clean", case["splits"])
        experiment2_random_models.extend(models)

    # %% [markdown]
    # ### 单独测试并复用正常时序基线

    # %%
    random_results = test_cases(experiment2_random_models)
    experiment2_results = pd.concat([baseline_results, random_results], ignore_index=True)
    pd.testing.assert_frame_equal(experiment2_results.loc[experiment2_results.method == "时序"].reset_index(drop=True),
                                  baseline_results.reset_index(drop=True))
    save_csv(RUN.path / "experiment2_metrics.csv", experiment2_results, index=False)
    # 全部唯一运行仅 24 次；实验章节的视图可以复用基线，但不重复计入唯一运行。
    results = pd.concat([experiment1_results, random_results], ignore_index=True)
    # 正常时序 2 个 + 泄漏时序 2 个 + 每个随机划分 2 个。
    assert len(results) == 4 + 2 * len(RANDOM_SEEDS)
    assert len(experiment2_results) == 2 + 2 * len(RANDOM_SEEDS)
    save_csv(RUN.path / "all_test_metrics.csv", results, index=False)

    # %% [markdown]
    # ## 四、汇总与分析
    # 
    # 时序结果报告单次指标；随机组报告 10 次的均值 ± 样本标准差。标准差反映划分敏感性，不是置信区间，随机测试集之间存在重叠。
    # 
    # 分类同时列出正类比例，回归列出训练均值预测的 RMSE，辅助辨别测试样本难度。不同任务不适用的指标在明细中留空。

    # %%
    def result_table(task, columns):
        rows = []
        task_results = results.loc[results["task"] == task]
        groups = task_results.groupby(["method", "feature_mode"], sort=False)
        for (method, mode), group in groups:
            row = {"划分": method, "特征": "原6" if mode == "clean" else "原6+未来信息（故意泄漏）", "次数": len(group),
                   "每次测试样本": int(group.n_test.iloc[0])}
            for column, label in columns.items():
                values = group[column]
                if len(values) == 1:
                    row[label] = f"{values.iloc[0]:.4f}"
                else:
                    mean_value = values.mean()
                    standard_deviation = values.std(ddof=1)  # 样本标准差，不是置信区间。
                    row[label] = f"{mean_value:.4f} ± {standard_deviation:.4f}"
            rows.append(row)
        table = pd.DataFrame(rows)
        save_csv(RUN.path / f"{task}_summary.csv", table, index=False)
        return table

    classification_table = result_table("classification", {
        "roc_auc": "ROC-AUC ↑", "ap": "AP ↑", "balanced_accuracy": "平衡准确率 ↑", "positive_rate": "正类比例"})
    regression_table = result_table("regression", {
        "mae_bp": "MAE 基点 ↓", "rmse_bp": "RMSE 基点 ↓", "constant_rmse_bp": "均值基准 RMSE", "delta_rmse_bp": "ΔRMSE 对常数"})
    utils.show_table(classification_table)
    utils.show_table(regression_table)

    # %% [markdown]
    # ### 实验①：同一测试集上的配对差值
    # 
    # Δ 为泄漏组减正常组。分类 ΔAUC 为正、回归 ΔRMSE 为负，表示未来信息带来的表面改善。

    # %%
    delta_rows = []
    for task, metric in (("classification", "roc_auc"), ("regression", "rmse_bp")):
        scores = experiment1_results.loc[experiment1_results.task == task].set_index("feature_mode")[metric]
        delta_rows.append({"任务": task, "指标": metric, "正常组": scores["clean"], "泄漏组": scores["leak"],
                           "泄漏减正常": scores["leak"] - scores["clean"]})
    delta_table = pd.DataFrame(delta_rows)
    utils.show_table(delta_table)
    save_csv(RUN.path / "experiment1_paired_effect.csv", delta_table, index=False)

    # %% [markdown]
    # ### 两个实验的图示
    # 
    # 每一行对应一个实验；随机组误差线为样本标准差，单次时序不绘制误差线。

    # %%
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    def draw_comparison(ax, table, task, metric, conditions, names, title):
        """每次只画一个任务的一张图，两个柱子表示两种设置。"""
        means = []
        standard_deviations = []
        sizes = []
        for method, mode in conditions:
            mask = (table["task"] == task) & (table["method"] == method) & (table["feature_mode"] == mode)
            values = table.loc[mask, metric]
            means.append(values.mean())
            sizes.append(len(values))
            standard_deviations.append(values.std(ddof=1))
        ax.bar([0, 1], means, color=["#3977ab", "#d77d3e"])
        if sizes[1] > 1:
            ax.errorbar([1], [means[1]], yerr=[standard_deviations[1]], fmt="none", color="black", capsize=4)
        ax.set_xticks([0, 1], names)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    # 四张图逐一调用，便于看清每张图对应的实验、任务和指标。
    leak_conditions = [("时序", "clean"), ("时序", "leak")]
    split_conditions = [("时序", "clean"), ("随机", "clean")]
    draw_comparison(axes[0, 0], experiment1_results, "classification", "roc_auc", leak_conditions,
                    ["正常输入", "故意泄漏"], "实验① · 分类 AUC ↑")
    draw_comparison(axes[0, 1], experiment1_results, "regression", "rmse_bp", leak_conditions,
                    ["正常输入", "故意泄漏"], "实验① · 回归 RMSE（基点）↓")
    draw_comparison(axes[1, 0], experiment2_results, "classification", "roc_auc", split_conditions,
                    ["时序基线（复用）", "随机划分（10次）"], "实验② · 分类 AUC ↑")
    draw_comparison(axes[1, 1], experiment2_results, "regression", "rmse_bp", split_conditions,
                    ["时序基线（复用）", "随机划分（10次）"], "实验② · 回归 RMSE（基点）↓")
    fig.savefig(RUN.path / "two_experiments.png", dpi=160)
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    save_json(RUN.path / "status.json", {"status": "completed", "unique_models_trained_and_tested": len(results),
                                        "normal_time_baseline_reused": True, "random_repeats": len(RANDOM_SEEDS)})
    print("本次输出目录：", utils.display_path(RUN.path))

    # %% [markdown]
    # ### 结果解释的边界
    # 
    # - **实验①**展示故意引入未来信息的影响。泄漏特征在预测时不可获得，其改善不是可实现的预测能力，也不代表原正常方案存在此错误。
    # - **实验②**的差异还受到测试样本覆盖时段和行情分布影响。随机划分使拟合可能使用测试时点之后的标签；但随机与时序分数之差不是纯粹的数据泄漏效应，也不保证随机分数更高。
    # - 所有正常特征只用当时已知历史；使用较早价格计算后期特征是正常做法。历史数据已参与探索，本次整理与重跑不构成新的独立验证证据。
    # 
    # [数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html) · [时序划分说明](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
    # 
    # 每次运行独立保存到 `output/validation_experiments/时间戳/`。公共输入、分区审计和汇总位于根目录，各模型的训练与测试目录分别保存。`reused_baseline.json` 记录两个章节共用的基线模型路径。

    # %%
    save_json(RUN.path / "status.json", {"status": "completed", "unique_models": len(results)})


STAGES = {
    "04": classification,
    "05": regression,
    "06": comparison,
    "07": validation,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="Notebook stage number")
    args = parser.parse_args()
    for number in (STAGES if args.stage == "all" else [args.stage]):
        STAGES[number]()


if __name__ == "__main__":
    main()
