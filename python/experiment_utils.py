"""整理版各阶段共用的路径、数据读取、时间划分、日志和模型保存工具。

关键模型训练和调参逻辑写在各方案的 .py / Notebook 中。
这里不访问网络，复用项目里已保存的行情和跨市场数据。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, r2_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]  # python/ 的上一级是项目根目录
DATA_DIR = ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
CLEAN_FILE = DATA_DIR / "processed" / "sp500_clean.csv"
FEATURE_DIR = DATA_DIR / "features"
RAW_FILE = DATA_DIR / "raw" / "sp500_daily_1996-11-13_to_2026-11-12_raw.json"
ORIGINAL = [
    "return_1d", "return_5d", "return_20d", "volatility_20d",
    "intraday_return", "intraday_range",
]
MARKET = [
    "vix_close", "vix_change", "treasury_2y_change_bp",
    "dxy_change_pct", "wti_change_usd",
]
BP = 10_000  # 收益率 0.01 = 1% = 100 基点。
LAST_DATE = "2026-11-12"
MARKET_FILES = [
    MARKET_DIR / "vix_history.csv",
    *[MARKET_DIR / name for name in (
        "treasury_archive.csv", "treasury_2025.csv", "treasury_example.csv",
        "dxy.json", "wti.json",
    )],
]


def display_path(path):
    """以项目目录为基准显示路径；跨盘外部文件显示文件名。"""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return Path(os.path.relpath(path, ROOT)).as_posix()
    except ValueError:
        return path.name


def _relative_display_text(text):
    for prefix in (str(ROOT), ROOT.as_posix(), str(ROOT).replace("\\", "\\\\")):
        text = re.sub(re.escape(prefix) + r"[\\/]+", "", text, flags=re.IGNORECASE)
    return text


def _display_value(value):
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, str):
        if Path(value).is_absolute():
            return display_path(value)
        return _relative_display_text(value)
    return value


def _relative_warning(message, category, filename, lineno, line=None):
    text = _relative_warning.original(message, category, filename, lineno, line)
    return _relative_display_text(text.replace(str(filename), display_path(filename)))


# 重载模块时复用最初的格式化函数，避免警告格式化器递归包装。
_relative_warning.original = getattr(warnings.formatwarning, "_experiment_original", warnings.formatwarning)
_relative_warning._experiment_original = _relative_warning.original


class RelativePathFormatter(logging.Formatter):
    def format(self, record):
        return _relative_display_text(super().format(record))


def save_json(path, value):
    """统一使用 UTF-8；禁止输出非法的 NaN / Infinity JSON 值。"""
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (Path, pd.Timestamp, datetime)):
            return str(item)
        raise TypeError(f"不能序列化 {type(item)}")
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False),
        encoding="utf-8",
    )


def save_csv(path, frame, index=True):
    frame.to_csv(path, index=index, encoding="utf-8-sig", index_label="signal_date" if index else None)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Run:
    path: Path
    config: dict
    logger: logging.Logger


def start_run(name, config, sources):
    """每次执行新建目录；同一 Notebook 重跑也不会覆盖上次模型或测试。"""
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    path = ROOT / "output" / name / stamp
    path.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"{name}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    warnings.formatwarning = _relative_warning
    fmt = RelativePathFormatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.FileHandler(path / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    config = dict(config, scenario=name, created_at=datetime.now(ZoneInfo("Asia/Singapore")).isoformat())
    save_json(path / "config.json", config)
    save_json(path / "environment.json", {
        "python": platform.python_version(), "python_executable": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__, "timezone": "Asia/Singapore",
    })
    save_json(path / "data_sources.json", [
        {"path": str(p), "sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sources
    ])
    snapshot = path / "code"
    snapshot.mkdir()
    for source in sorted([*(ROOT / "python").glob("*.py"), *ROOT.glob("*.ipynb")]):
        if source.exists():
            shutil.copy2(source, snapshot / source.name)
    logger.info("开始 %s；输出目录 %s", name, display_path(path))
    return Run(path, config, logger)


def stage_log(run, directory, message):
    directory.mkdir(parents=True, exist_ok=True)
    message = _relative_display_text(message)
    run.logger.info(message)
    with (directory / "stage.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now().isoformat(timespec='seconds')} | {message}\n")


class TrainingLog(xgb.callback.TrainingCallback):
    """将 XGBoost 的迭代进度同时写入屏幕、总日志和本次训练日志。"""
    def __init__(self, run, directory, every=1):
        self.run, self.directory, self.every = run, directory, every

    def after_iteration(self, model, epoch, evals_log):
        if epoch == 0 or (epoch + 1) % self.every == 0:
            values = [f"{part}.{metric}={scores[-1]:.8f}"
                      for part, metrics in evals_log.items() for metric, scores in metrics.items()]
            stage_log(self.run, self.directory, f"第 {epoch + 1} 轮 | " + " | ".join(values))
        return False


class LoggedEarlyStopping(xgb.callback.EarlyStopping):
    """先记录本轮指标，再判断早停，确保触发早停的最后一轮也会打印。"""
    def __init__(self, run, directory, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.progress = TrainingLog(run, directory, every=log_every)

    def after_iteration(self, model, epoch, evals_log):
        self.progress.after_iteration(model, epoch, evals_log)
        return super().after_iteration(model, epoch, evals_log)


def load_daily():
    """读取第 01 步清洗的日线；价格精度与原训练入口保持一致。"""
    if not CLEAN_FILE.is_file():
        raise FileNotFoundError("请先运行 python/prepare.py 01（或 01_data_preprocessing.ipynb）。")
    bars = pd.read_csv(CLEAN_FILE, parse_dates=["date"], float_precision="round_trip")
    bars = bars.set_index("date").sort_index().loc[:LAST_DATE]
    prices = ["open", "high", "low", "close"]
    bars[prices] = bars[prices].astype(float).round(3)
    if not bars.index.is_unique or bars.index.hasnans:
        raise ValueError("日期缺失或重复。")
    if not (np.isfinite(bars[prices]) & bars[prices].gt(0)).all().all():
        raise ValueError("存在无效价格。")
    bars.index.name = "signal_date"
    return bars


def add_targets(bars, features):
    """t 日收盘预测；t+1 开盘入场、t+2 开盘结算。

    原项目价格保留三位小数。整数比较保证恰好 +0.20% 时标签为 0。
    """
    dates = bars.index.to_series()
    mills = (bars["open"] * 1000).round().astype("Int64")
    entry, exit_ = mills.shift(-1), mills.shift(-2)
    samples = features.assign(
        trade_date=dates.shift(-1), settle_date=dates.shift(-2),
        target_return=bars["open"].shift(-2) / bars["open"].shift(-1) - 1,
        target_class=(500 * (exit_ - entry) > entry).astype("Int8"),
    )
    return samples.replace([np.inf, -np.inf], np.nan).dropna(subset=[*ORIGINAL, "target_return", "target_class"])


def _market_close(filename, symbol):
    chart = json.loads((MARKET_DIR / filename).read_text(encoding="utf-8"))["chart"]["result"][0]
    if chart["meta"]["symbol"] != symbol:
        raise ValueError(f"{filename} 的指数代码错误。")
    dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
    dates = dates.tz_convert("America/New_York").tz_localize(None).normalize()
    values = pd.Series(chart["indicators"]["quote"][0]["close"], index=dates, dtype=float).dropna().sort_index()
    if not values.index.is_unique:
        raise ValueError(f"{filename} 存在重复日期。")
    return values


def load_market_features(dates):
    """构造 B 组新增 5 项；同时返回来源日、可用日，供防泄漏审计。

    日线至少滞后 1 个日历日。早于 2016-10-11 的美债收益率滞后 7 日。
    这是沿用旧实验的保守可用时间约定，并非精确发布时刻的估计。
    """
    raw = pd.read_csv(MARKET_FILES[0], usecols=["DATE", "CLOSE"])
    vix = raw.set_index(pd.to_datetime(raw["DATE"], format="%m/%d/%Y"))["CLOSE"].sort_index()
    yield_parts = []
    for filename, fmt in (("treasury_archive.csv", "%m/%d/%y"),
                          ("treasury_2025.csv", "%m/%d/%Y"), ("treasury_example.csv", "%m/%d/%Y")):
        raw = pd.read_csv(MARKET_DIR / filename, usecols=["Date", "2 Yr"])
        yield_parts.append(raw.set_index(pd.to_datetime(raw["Date"], format=fmt))["2 Yr"])
    treasury = pd.concat(yield_parts, verify_integrity=True).dropna().sort_index()
    dxy, wti = _market_close("dxy.json", "DX-Y.NYB"), _market_close("wti.json", "CL=F")
    series = [vix, vix.diff(), treasury.diff() * 100, dxy.pct_change(fill_method=None) * 100, wti.diff()]
    features, audit = pd.DataFrame(index=dates), []
    for name, values in zip(MARKET, series, strict=True):
        delay = np.ones(len(values), dtype=int)
        if name == "treasury_2y_change_bp":
            delay[values.index < "2016-10-11"] = 7
        available = values.index + pd.to_timedelta(delay, unit="D")
        aligned = pd.DataFrame({"value": values.to_numpy(), "source_date": values.index,
                                "available_date": available}, index=available)
        aligned = aligned.loc[~aligned.index.duplicated(keep="last")].sort_index().dropna(subset=["value"])
        aligned = aligned.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=7))
        valid = aligned["value"].notna()
        if not (aligned.loc[valid, "available_date"] <= dates[valid]).all():
            raise AssertionError("跨市场特征使用了信号日之后的数据。")
        features[name] = aligned["value"]
        audit.append(aligned.assign(feature=name, signal_date=dates))
    return features, pd.concat(audit, ignore_index=True)


def show_table(frame, decimals=4):
    """Notebook 显示简洁 HTML 表格；命令行打印普通表格。"""
    shown = frame.copy()
    numeric = shown.select_dtypes(include="number", exclude="timedelta").columns
    shown[numeric] = shown[numeric].round(decimals)
    for column in shown.select_dtypes(include=["object", "string"]).columns:
        shown[column] = shown[column].map(_display_value)
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(shown)
            return
    except ImportError:
        pass
    print(shown.to_string(index=False))


def predict_model(model, frame, columns, unit):
    if isinstance(model, xgb.Booster):
        result = model.predict(xgb.DMatrix(frame[columns], nthread=4))
    else:
        result = model.predict(frame[columns])
    return result / BP if unit == "basis_points" else result


def measure(task, actual, prediction, threshold=None):
    if task == "classification":
        decisions = prediction >= threshold
        return {
            "roc_auc": roc_auc_score(actual, prediction),
            "ap": average_precision_score(actual, prediction),
            "accuracy": accuracy_score(actual, decisions),
            "balanced_accuracy": balanced_accuracy_score(actual, decisions),
            "f1": f1_score(actual, decisions, zero_division=0),
        }
    error = (prediction - actual) * BP
    return {"mae_bp": np.abs(error).mean(), "rmse_bp": np.sqrt(np.mean(error ** 2)), "r2": r2_score(actual, prediction)}


def new_training_dir(run, period):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parent = run.path if period == "holdout" else run.path / "folds" / period
    folder = parent / f"train_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_training(run, period, folder, model, splits, columns, selected, history=None):
    """保存实际拟合模型、训练与验证预测；测试在另一个代码块中执行。"""
    folder.mkdir(parents=True, exist_ok=True)
    task = run.config["task"]
    selected = dict(selected, task=task, features=list(columns),
                    train_mean=float(splits["train"].target_return.mean()),
                    train_median=float(splits["train"].target_return.median()))
    save_json(folder / "selected.json", selected)
    if isinstance(model, xgb.Booster):
        model.save_model(folder / "model.ubj")
    else:
        joblib.dump(model, folder / "model.joblib")
    if history:
        save_json(folder / "history.json", history)
        history_frame = pd.DataFrame({f"{part}_{metric}": values
                                      for part, metrics in history.items() for metric, values in metrics.items()})
        history_frame.index += 1
        history_frame.index.name = "round"
        history_frame.to_csv(folder / "history.csv", encoding="utf-8-sig")
    label = "target_class" if task == "classification" else "target_return"
    metrics = {}
    for split in ("train", "val"):
        frame = splits[split]
        prediction = predict_model(model, frame, columns, selected["prediction_unit"])
        output = frame[["trade_date", "settle_date", label]].assign(prediction=prediction)
        save_csv(folder / f"{split}_predictions.csv", output)
        metrics[split] = measure(task, frame[label].to_numpy(dtype=float), prediction, selected.get("threshold"))
    save_json(folder / "metrics.json", metrics)
    stage_log(run, folder, f"{period} 训练完成 | 选择 {selected.get('alpha', selected.get('best_rounds'))}")
    return {"folder": folder, "selected": selected}


def evaluate_saved(run, period, artifact, test):
    """重新读取已保存模型后测试；反复调用也创建新的 test 目录。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parent = run.path if period == "holdout" else run.path / "folds" / period
    folder = parent / f"test_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    selected = artifact["selected"]
    source = artifact["folder"]
    if (source / "model.ubj").exists():
        model = xgb.Booster(params={"nthread": 4})
        model.load_model(source / "model.ubj")
    else:
        model = joblib.load(source / "model.joblib")
    columns, task = selected["features"], selected["task"]
    stage_log(run, folder, f"{period} 开始测试 | 样本 {len(test)} | 从磁盘加载模型")
    prediction = predict_model(model, test, columns, selected["prediction_unit"])
    if not np.isfinite(prediction).all():
        raise ValueError("模型产生了无效预测。")
    label = "target_class" if task == "classification" else "target_return"
    actual = test[label].to_numpy(dtype=float)
    metrics = {"period": period, "n_test": len(test), **measure(task, actual, prediction, selected.get("threshold"))}
    predictions = test[["trade_date", "settle_date", label]].assign(prediction=prediction)
    if task == "classification":
        predictions["predicted_class"] = (prediction >= selected["threshold"]).astype(int)
        metrics.update(threshold=selected["threshold"], positive_rate=float(actual.mean()))
        save_csv(folder / "confusion_matrix.csv", pd.DataFrame(
            confusion_matrix(actual, predictions.predicted_class, labels=[0, 1]),
            columns=["predicted_0", "predicted_1"]), index=False)
    else:
        # MAE 对训练中位数；RMSE 对训练均值。两者都不使用测试集拟合。
        predictions["constant_mean"] = selected["train_mean"]
        predictions["constant_median"] = selected["train_median"]
        metrics["constant_mae_bp"] = np.abs(actual - selected["train_median"]).mean() * BP
        metrics["constant_rmse_bp"] = np.sqrt(np.mean((actual - selected["train_mean"]) ** 2)) * BP
        metrics["delta_mae_bp"] = metrics["mae_bp"] - metrics["constant_mae_bp"]
        metrics["delta_rmse_bp"] = metrics["rmse_bp"] - metrics["constant_rmse_bp"]
    save_csv(folder / "predictions.csv", predictions)
    save_json(folder / "metrics.json", metrics)
    save_json(folder / "model_source.json", {"training_directory": str(source), "selected": selected})
    stage_log(run, folder, f"{period} 测试完成 | " + " | ".join(
        f"{key}={metrics[key]:.4f}" for key in (("roc_auc", "ap") if task == "classification" else ("mae_bp", "rmse_bp"))))
    return metrics, predictions




def show_figure(fig, path):
    """图表始终保存；Notebook 内显示，脚本运行时不弹出阻塞窗口。"""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(fig)
    finally:
        plt.close(fig)


def chronological_split(samples, ratios=(0.8, 0.1, 0.1)):
    """沿用原训练入口：按有效样本定边界，剔除跨分区的收益标签。"""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("请提供三个正比例，且比例之和为 1。")
    val_size, test_size = (round(len(samples) * x) for x in ratios[1:])
    train_size = len(samples) - val_size - test_size
    if min(train_size, val_size, test_size) < 2:
        raise ValueError("分区样本过少。")
    validation_start = samples.iloc[train_size]["trade_date"]
    test_start = samples.iloc[train_size + val_size]["trade_date"]
    splits = {
        "train": samples.loc[samples.settle_date < validation_start].copy(),
        "val": samples.loc[(samples.trade_date >= validation_start) & (samples.settle_date < test_start)].copy(),
        "test": samples.loc[samples.trade_date >= test_start].copy(),
    }
    assert all(len(frame) > 0 for frame in splits.values())
    assert splits["train"].settle_date.max() <= splits["val"].index.min()
    assert splits["val"].settle_date.max() <= splits["test"].index.min()
    all_dates = pd.Index(np.concatenate([f.index.to_numpy() for f in splits.values()]))
    assert all_dates.is_unique, "训练、验证、测试有重复信号日。"
    summary = pd.DataFrame([
        {"split": name, "samples": len(frame), "first_signal": frame.index.min(),
         "last_signal": frame.index.max(), "first_entry": frame.trade_date.min(),
         "last_settlement": frame.settle_date.max()}
        for name, frame in splits.items()
    ])
    return splits, summary


def load_prepared_samples(task):
    """读取第 02 步产物，并确认上游数据没有在构造特征后被替换。"""
    if task not in ("classification", "regression"):
        raise ValueError("未知任务。")
    path = FEATURE_DIR / f"{task}_samples.csv"
    manifest_file = FEATURE_DIR / "feature_build.json"
    if not path.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("请先运行 01_data_preprocessing 和 02_feature_engineering。")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        if file_hash(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("上游数据已变更，请重新运行第 02 步。")
    if file_hash(path) != manifest["outputs"][path.name]:
        raise ValueError("特征文件已被修改，请重新运行第 02 步。")
    samples = pd.read_csv(path, index_col="signal_date",
                          parse_dates=["signal_date", "trade_date", "settle_date"],
                          float_precision="round_trip")
    samples["target_class"] = samples["target_class"].astype("Int8")
    assert samples.index.is_monotonic_increasing and samples.index.is_unique
    return samples


# Shared operations for the optional MLP, trading, and market-state stages.
def read_fusion_frame(path):
    frame = pd.read_csv(path, index_col="signal_date", float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index)
    for column in ("trade_date", "settle_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be unique and ordered: {path}")
    return frame


def latest_completed_run(scenario):
    directory = ROOT / "output" / scenario
    if not directory.is_dir():
        raise FileNotFoundError(f"Run the prerequisite stage first: {scenario}")
    def has_predictions(path):
        return all((path / arm / "test_predictions.csv").is_file()
                   and any((folder / "selected.json").is_file()
                           and (folder / "train_predictions.csv").is_file()
                           and (folder / "val_predictions.csv").is_file()
                           for folder in (path / arm).glob("train_*"))
                   for arm in ("baseline", "final"))
    runs = sorted(path for path in directory.iterdir()
                  if (path / "status.json").is_file()
                  and json.loads((path / "status.json").read_text(encoding="utf-8")).get("status") == "completed"
                  and has_predictions(path))
    if not runs:
        raise FileNotFoundError(f"No complete predictions in {directory}; rerun the prerequisite training stage")
    return runs[-1]


def latest_training_folder(run, arm):
    runs = sorted((run / arm).glob("train_*"))
    if not runs:
        raise FileNotFoundError(f"No training artifacts: {run / arm}")
    return runs[-1]


def new_fusion_output(name):
    folder = ROOT / "output" / "fusion" / name / datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "environment.json", {"python": sys.version, "executable": sys.executable,
                                              "numpy": np.__version__, "pandas": pd.__version__})
    inputs = sorted([*(FEATURE_DIR.glob("*.csv")), *((DATA_DIR / "processed").glob("*.csv"))])
    save_json(folder / "fingerprints.json", {display_path(p): file_hash(p) for p in inputs})
    return folder


def regression_scores(y, prediction):
    from sklearn.metrics import r2_score
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {"n": len(y), "mae_bp": np.mean(abs(y-prediction))*BP,
            "rmse_bp": np.sqrt(np.mean((y-prediction)**2))*BP,
            "r2": r2_score(y, prediction) if len(y)>1 and np.var(y)>0 else np.nan,
            "direction_accuracy": np.mean((y>0)==(prediction>0))}


def load_fusion_inputs():
    """Read saved predictions and audit dates and labels against the original prices."""
    pointer = ROOT / "output/fusion/mlp_latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("Run the MLP extension in stage 05 first")
    mlp_dir = ROOT / json.loads(pointer.read_text(encoding="utf-8"))["path"]
    reg_run = ROOT / json.loads((mlp_dir / "config.json").read_text(encoding="utf-8"))["source_run"]
    cls_run = latest_completed_run("classification_comparison")
    samples = {task: read_fusion_frame(FEATURE_DIR / f"{task}_samples.csv")
               for task in ("classification", "regression")}
    predictions, thresholds, meta = {}, {}, {}
    specifications = (("Class A", cls_run, "baseline", "classification"),
                      ("Class B", cls_run, "final", "classification"),
                      ("Regression XGB", reg_run, "baseline", "regression"),
                      ("Ridge", reg_run, "final", "regression"))
    for name, run, arm, task in specifications:
        train = latest_training_folder(run, arm)
        selected = json.loads((train / "selected.json").read_text(encoding="utf-8"))
        predictions[name] = {split: read_fusion_frame(train / f"{split}_predictions.csv") for split in ("train", "val")}
        predictions[name]["test"] = read_fusion_frame(run / arm / "test_predictions.csv")
        thresholds[name] = selected.get("threshold", .002)
        meta[name] = {"task": task, "train_mean": selected["train_mean"],
                      "train_median": selected["train_median"], "source_run": display_path(run)}
        for frame in predictions[name].values():
            source = samples[task].loc[frame.index]
            for date_column in ("trade_date", "settle_date"):
                pd.testing.assert_series_equal(frame[date_column], source[date_column])
            frame["target_return"] = source.target_return
            frame["target_class"] = source.target_class.astype(int)
            assert (frame.index < frame.trade_date).all() and (frame.trade_date < frame.settle_date).all()
    predictions["MLP"] = {split: read_fusion_frame(mlp_dir / f"{split}_predictions.csv")
                          for split in ("train", "val", "test")}
    thresholds["MLP"] = .002
    meta["MLP"] = {**meta["Ridge"], "source_run": display_path(mlp_dir)}
    prices = pd.read_csv(CLEAN_FILE, parse_dates=["date"]).set_index("date")
    for parts in predictions.values():
        for frame in parts.values():
            expected = prices.open.reindex(frame.settle_date).to_numpy() / prices.open.reindex(frame.trade_date).to_numpy()-1
            np.testing.assert_allclose(frame.target_return, expected, atol=1e-14)
    return mlp_dir, reg_run, cls_run, predictions, thresholds, meta, prices


def backtest(frame, positions, fee=.0005):
    """Match the Notebook's open-to-open accounting, including final liquidation."""
    f = frame[["trade_date", "settle_date", "target_return"]].copy()
    p = np.asarray(positions, dtype=float)
    assert len(f) == len(p) and len(f) > 0 and np.isin(p, [0, 1]).all()
    previous = np.r_[0, p[:-1]]
    gaps = np.r_[False, f.trade_date.iloc[1:].to_numpy() != f.settle_date.iloc[:-1].to_numpy()]
    exit_before = np.zeros(len(f))
    exit_before[np.flatnonzero(gaps)-1] = p[np.flatnonzero(gaps)-1]
    previous[gaps] = 0
    turnover = abs(p-previous)
    exit_after = exit_before.copy()
    exit_after[-1] = p[-1]
    net = (1-fee*turnover)*(1+p*f.target_return.to_numpy())*(1-fee*exit_after)-1
    f["position"] = p
    f["entry_turnover"] = turnover
    f["exit_turnover"] = exit_after
    f["turnover"] = turnover + exit_after
    f["add_day"] = (p > previous).astype(int)
    f["net_return"] = net
    f["equity"] = np.cumprod(1+net)
    f["drawdown"] = f.equity/np.maximum.accumulate(np.r_[1, f.equity])[1:]-1
    return f


def trading_scores(history):
    returns = history.net_return
    std = returns.std(ddof=1)
    return {"n": len(history), "total_return": np.prod(1+returns)-1,
            "max_drawdown": -history.drawdown.min(),
            "sharpe": np.sqrt(252)*returns.mean()/std if len(history)>1 and std>0 else np.nan,
            "trades": int((history.entry_turnover>0).sum()+(history.exit_turnover>0).sum()),
            "turnover": history.turnover.sum(), "add_day_fraction": history.add_day.mean(),
            "holding_time_fraction": (history.position>0).mean(),
            "average_position": history.position.mean()}
