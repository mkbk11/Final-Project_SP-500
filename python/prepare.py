"""S&P 500 stages: 01, 02, 03. Run with a stage number or all."""

import argparse


def preprocess():
    """Run stage 01 (01_data_preprocessing)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 01 数据预处理
    # 沿用原预处理流程：解析 JSON → 基础 EDA → 日期与类型 → 交易日检查 → 清洗并导出 → 原始日线 8:1:1 划分。
    # 只将已核查的 2023-05-24 零成交量置为缺失；OHLC 保留 3 位小数。下一步统一构造收益标签。

    # %% [markdown]
    # ### 1. 解析 JSON
    # 
    # 读取 Yahoo Finance 的原始 JSON，将时间戳、开盘价、最高价、最低价、收盘价和成交量按位置对应成表，每行代表一个交易日。此步骤只解析数据，保留原始时间戳和数值。

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
    from pathlib import Path
    import pandas as pd
    from experiment_utils import ROOT, RAW_FILE, CLEAN_FILE, save_json, show_table, show_figure

    display = show_table
    FIGURE_DIR = ROOT / "output" / "preprocessing"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


    # 从 datasets 文件夹读取原始文件
    path = RAW_FILE

    with path.open(encoding="utf-8") as file:
        data = json.load(file)


    # 提取时间戳和日线行情，按位置对齐成表
    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "timestamp": result["timestamp"],
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    })

    assert result["meta"]["symbol"] == "^GSPC"
    display(df.head())

    # %% [markdown]
    # ### 2. 初步探索数据
    # 
    # - **基本信息**：查看样本量、字段数量、字段类型和缺失值数量。
    # - **描述统计**：统计开盘价、最高价、最低价、收盘价和成交量的均值、样本方差、分位数及极值。
    # - **可视化**：用直方图查看分布，用箱线图定位值得核查的点，并用相关性热力图查看字段间的线性关系。原始时间戳仅用于标识交易时间，不参与统计和绘图。
    # 
    # 箱线图标出的点需要结合原始行情核查。四个价格字段高度相关也可能来自它们共同的长期趋势，仅凭相关系数无法判断预测能力。

    # %%
    import matplotlib.pyplot as plt

    market_cols = ["open", "high", "low", "close", "volume"]
    market = df[market_cols]

    # 查看基本信息
    print(f"样本量 {len(df):,} | 总字段数 {df.shape[1]} | 行情字段数 {len(market_cols)}")
    display(pd.DataFrame({"字段类型": df.dtypes, "缺失数量": df.isna().sum()}))

    # 描述统计，方差使用样本方差
    stats = market.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
    stats["variance"] = market.var()
    display(stats)

    # 每列使用独立坐标轴，便于查看价格和成交量各自的分布
    fig, axes = plt.subplots(2, len(market_cols), figsize=(15, 5), layout="constrained")
    for i, col in enumerate(market_cols):
        values = market[col].dropna()
        axes[0, i].hist(values, bins=30)
        axes[0, i].set_title(f"{col} histogram")
        axes[1, i].boxplot(values)
        axes[1, i].set_title(f"{col} boxplot")
        axes[1, i].set_xticks([])
    show_figure(fig, FIGURE_DIR / "raw_distributions.png")

    # 查看 Pearson 相关系数
    corr = market.corr()
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    heatmap = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(market_cols)), market_cols, rotation=30)
    ax.set_yticks(range(len(market_cols)), market_cols)
    ax.set_title("Pearson correlation")
    fig.colorbar(heatmap, ax=ax)
    show_figure(fig, FIGURE_DIR / "raw_correlations.png")

    # 单独列出非正成交量，便于后续核查
    display(df.loc[df["volume"].le(0), ["timestamp", "close", "volume"]])

    # %% [markdown]
    # ### 3. 转换日期和类型
    # 
    # 将 Unix 时间戳从 UTC 转为纽约时间，再提取交易日期，并按日期升序排列。价格列转换为浮点数，成交量转换为允许缺失的整数。保留原始 `timestamp` 列和成交量中的 `0`，方便后续检查。

    # %%
    # 将秒级 Unix 时间戳转换为美东交易日期
    df["date"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # 设置价格和成交量的数据类型
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].astype("float64")
    df["volume"] = df["volume"].astype("Int64")

    # 按交易日期升序排列，查看前五行
    df = df.sort_values("date", ignore_index=True)
    df.head()

    # %% [markdown]
    # ### 4. 检查缺失和异常值，检查交易日
    # 
    # - 统计各列缺值，并筛查非正或无穷的价格、超出当日高低区间的开盘价或收盘价，以及小于等于 0 的成交量。这里只列出异常供核对，保留 `df` 中的原始值，包括 2023-05-24 的成交量 `0`。
    # - 将 `date` 与纽约证券交易所（NYSE）的交易日历逐日比对，找出缺失的交易日、休市日记录和重复日期。使用数据集的预定起止日期，也能发现首尾是否漏掉交易日。

    # %%
    import exchange_calendars as xcals

    # 3. 检查交易日
    start, end = "1996-11-13", "2026-11-12"
    nyse = xcals.get_calendar("XNYS", start=start, end=end)
    expected = nyse.sessions.tz_localize(None)
    actual = pd.DatetimeIndex(df["date"].dropna().unique())

    print("应有交易日数", len(expected))
    print("缺失交易日", expected.difference(actual).strftime("%Y-%m-%d").tolist())
    print("额外日期", actual.difference(expected).strftime("%Y-%m-%d").tolist())

    duplicates = df["date"].notna() & df["date"].duplicated(keep=False)
    print("涉及重复日期的记录数", duplicates.sum())
    assert not duplicates.any(), "重复交易日需要核查。"
    display(df.loc[duplicates])

    # 4. 检查缺失和异常
    display(df.isna().sum().rename("缺失值数量").to_frame())

    prices = df[["open", "high", "low", "close"]]
    bad_price = (prices.le(0) | prices.isin([float("inf"), -float("inf")])).any(axis=1)

    bad_range = (
        df["low"].gt(df["high"])
        | df["open"].lt(df["low"]) | df["open"].gt(df["high"])
        | df["close"].lt(df["low"]) | df["close"].gt(df["high"])
    )

    bad_volume = df["volume"].le(0).fillna(False)

    print("价格非正或无穷的记录数", bad_price.sum())
    print("价格区间异常的记录数", bad_range.sum())
    print("成交量小于等于 0 的记录数", bad_volume.sum())
    display(df.loc[bad_price | bad_range | bad_volume])
    # 不删除真实市场极值；无效价格需要人工核查后才能进入训练。
    assert not (bad_price | bad_range).any()
    assert not prices.isna().any().any()
    save_json(FIGURE_DIR / "data_quality.json", {
        "rows": len(df), "missing_sessions": expected.difference(actual).strftime("%Y-%m-%d").tolist(),
        "extra_dates": actual.difference(expected).strftime("%Y-%m-%d").tolist(),
        "duplicate_rows": int(duplicates.sum()), "bad_price_rows": int(bad_price.sum()),
        "bad_range_rows": int(bad_range.sum()), "nonpositive_volume_rows": int(bad_volume.sum()),
    })

    # %% [markdown]
    # ## 5. 清洗、导出和回读核对
    # 清洗 CSV 仅输出一次，统一使用 `.csv` 扩展名；不再生成重复的 `.csv1`。

    # %%
    df_clean = df.copy()

    # 将已核查的异常成交量标记为缺失
    mask = df_clean["date"].eq(pd.Timestamp("2023-05-24")) & df_clean["volume"].eq(0)
    df_clean.loc[mask, "volume"] = pd.NA

    # 对全部交易日的四个价格字段保留 3 位小数
    price_cols = ["open", "high", "low", "close"]
    df_clean[price_cols] = df_clean[price_cols].round(3)

    # 统一输出位置，便于后续特征与训练代码复用。
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    df_clean.to_csv(CLEAN_FILE, index=False, float_format="%.3f", date_format="%Y-%m-%d")
    saved = pd.read_csv(CLEAN_FILE, parse_dates=["date"], dtype={"volume": "Int64"})
    pd.testing.assert_frame_equal(saved, df_clean, check_dtype=False)
    assert len(saved) == 7547
    assert saved.loc[saved["volume"].isna(), "date"].tolist() == [pd.Timestamp("2023-05-24")]
    display(saved.head())
    print(f"复核通过：{len(saved):,} 条记录，{saved['volume'].isna().sum()} 个成交量缺值。")

    # %% [markdown]
    # ## 6. 原始日线按 8:1:1 导出
    # 保留原来的顺序划分，用于 14 特征的 EDA。模型训练按各自的有效样本池重新确定边界，并排除跨界标签；分类和回归的实际边界以各自 `split_summary.csv` 为准。
    # 原文件中的“预测次日收盘价、按年份切分”属于另一任务示例，此处不再作为主流程执行。

    # %%
    split_dir = ROOT / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    # 按交易日期排序，再按时间顺序切分
    data = df_clean.sort_values("date").reset_index(drop=True)
    holdout_size = round(len(data) * 0.1)
    train_end = len(data) - 2 * holdout_size
    val_end = train_end + holdout_size

    splits = {
        "train": data.iloc[:train_end],
        "val": data.iloc[train_end:val_end],
        "test": data.iloc[val_end:],
    }

    # 确认所有记录均已分配，且各集合的日期区间没有重叠
    assert sum(len(part) for part in splits.values()) == len(data)
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()

    # 分别导出到原始 JSON 所在的 datasets 文件夹，价格写出 3 位小数
    for name, part in splits.items():
        file = split_dir / f"sp500_{name}.csv"
        part.to_csv(file, index=False, float_format="%.3f", date_format="%Y-%m-%d")

        start, end = part["date"].iloc[[0, -1]].dt.strftime("%Y-%m-%d")
        print(f"{file.name}  {len(part):,} 条  {start} 至 {end}")

def features():
    """Run stage 02 (02_feature_engineering)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 02 特征构造与标签
    # 保留原 6 + 8 共 14 项特征的公式、严格大于 +0.20% 的标签和截断重算检查。
    # 另沿用分类方案的 5 项跨市场特征，生成分类 11 特征与回归 6 特征的数据表。14 特征分析表与分类 11 特征表用途不同。
    # 所有历史特征在完整时间序列上计算，允许后期样本使用早期历史；未来价格仅用于构造标签。

    # %% [markdown]
    # ## 1. 读取已清洗的原始划分

    # %%
    # 脚本集中在 python/；Notebook 保留在项目根目录。
    from pathlib import Path
    import sys
    PYTHON_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "python"
    if not (PYTHON_DIR / "experiment_utils.py").is_file():
        raise FileNotFoundError("请在 SP500_organized 项目根目录打开 Notebook。")
    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))

    # 1. 读取数据与基础检查
    from pathlib import Path
    from decimal import Decimal
    import numpy as np
    import pandas as pd
    from experiment_utils import (ROOT, RAW_FILE, CLEAN_FILE, FEATURE_DIR, ORIGINAL, MARKET,
        MARKET_FILES, add_targets, load_market_features, chronological_split,
        save_csv, save_json, file_hash, show_table)
    display = show_table

    DATA_DIR = ROOT / "data" / "splits"
    OUTPUT_DIR = FEATURE_DIR

    FILES = {split: DATA_DIR / f"sp500_{split}.csv" for split in ("train", "val", "test")}
    required = {"date", "open", "high", "low", "close", "volume"}
    parts = []

    print("=" * 65)
    print("[1/5] 读取 S&P 500 数据")
    for split, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"未找到文件：{path}")
        df = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")
        if df.empty:
            raise ValueError(f"{path.name} 是空文件")
        df = df.sort_values("date")
        parts.append(df)
        print(f"  {split:5s}: {len(df):5,d} 行 | {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")

    split_starts = [part["date"].min() for part in parts[1:]]
    daily = pd.concat(parts, ignore_index=True).set_index("date").sort_index()
    daily.index.name = "signal_date"

    assert daily.index.notna().all() and daily.index.is_unique, "日期包含空值或重复值"
    assert daily.index.is_monotonic_increasing, "日期未排序"
    assert len(daily) >= 3, "数据量不足"
    prices = daily[["open", "high", "low", "close"]].to_numpy(dtype=float)
    assert (np.isfinite(prices) & (prices > 0)).all(), "OHLC 存在无效或非正数价格"
    assert (daily["high"] >= daily[["open", "close", "low"]].max(axis=1)).all(), "最高价字段异常"
    assert (daily["low"] <= daily[["open", "close", "high"]].min(axis=1)).all(), "最低价字段异常"

    print(f"\n合并完成：{len(daily):,} 个交易日 | {daily.index.min():%Y-%m-%d} ~ {daily.index.max():%Y-%m-%d}")
    print(f"成交量缺失/非正数：{(daily['volume'].isna() | (daily['volume'] <= 0)).sum():,} 行")
    display(daily.head(3))

    # %% [markdown]
    # ## 2. 原有 14 项特征

    # %%
    # 2. 构造 14 项特征（仅使用 t 日及此前的数据）
    def build_features(bars: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = (bars[col] for col in ("open", "high", "low", "close"))
        v = bars["volume"].where(bars["volume"] > 0)
        price_range = h - l
        f = pd.DataFrame(index=bars.index)

        # 原有 6 项：保持名称与公式不变
        for n in (1, 5, 20):
            f[f"return_{n}d"] = c.pct_change(n, fill_method=None)
        f["volatility_20d"] = f["return_1d"].rolling(20).std(ddof=1)
        f["intraday_return"] = c / o - 1
        f["intraday_range"] = price_range / o

        # 新增 8 项：趋势（2）
        sma5, sma20 = c.rolling(5).mean(), c.rolling(20).mean()
        f["close_vs_sma20"] = c / sma20 - 1
        f["sma5_vs_sma20"] = sma5 / sma20 - 1

        # 隔夜与日内（2）
        f["overnight_return"] = o / c.shift(1) - 1
        f["close_location"] = (c - l) / price_range.where(price_range > 0)

        # 波动率（2）
        f["volatility_5d"] = f["return_1d"].rolling(5).std(ddof=1)
        f["avg_intraday_range_20d"] = f["intraday_range"].rolling(20).mean()

        # 成交量（2）：与前 20 天比较，不含 t 日
        f["volume_change_1d"] = v.pct_change(fill_method=None)
        f["volume_vs_prev20"] = v / v.shift(1).rolling(20).mean() - 1
        return f

    all_features = build_features(daily)
    assert all_features.shape[1] == 14
    assert not np.isinf(all_features.to_numpy(dtype=float)).any(), "特征中存在无穷大"

    print("=" * 65)
    print("[2/5] 特征构造完成")
    print(f"特征数：{all_features.shape[1]}（原有 6 + 新增 8）")
    print("特征名称：", ", ".join(all_features.columns))
    missing = all_features.isna().sum()
    print("\n特征缺失统计（只展示有缺失的列）：")
    print(missing[missing > 0].to_string() if (missing > 0).any() else "无缺失")
    print(f"完整特征行数：{all_features.notna().all(axis=1).sum():,}/{len(all_features):,}")
    display(all_features.tail(3))

    # %% [markdown]
    # ## 3. 分类和回归标签

    # %%
    # 3. 标签构造：t 收盘发信号，t+1 开盘入场，t+2 开盘退出
    open_ = daily["open"]
    entry_open, exit_open = open_.shift(-1), open_.shift(-2)

    # 原始 CSV 开盘价有 3 位小数：整数比较确保严格 > 0.002 的边界准确
    open_mills = (open_ * 1000).round().astype("Int64")
    entry_mills, exit_mills = open_mills.shift(-1), open_mills.shift(-2)

    labels = pd.DataFrame({
        "trade_date": daily.index.to_series().shift(-1),
        "settle_date": daily.index.to_series().shift(-2),
        "target_return": (exit_open - entry_open) / entry_open,
        "target_class": (500 * (exit_mills - entry_mills) > entry_mills).astype("Int8"),
    }).dropna()
    features = all_features.loc[labels.index].copy()

    assert features.index.equals(labels.index), "特征与标签未对齐"
    assert labels.index.equals(daily.index[:-2]), "末尾两日未正确排除"
    assert labels["target_class"].isin([0, 1]).all()

    count = labels["target_class"].value_counts().reindex([0, 1], fill_value=0)
    print("=" * 65)
    print("[3/5] 标签生成完成")
    print(f"有效样本：{len(labels):,} 行 | 信号日 {labels.index.min():%Y-%m-%d} ~ {labels.index.max():%Y-%m-%d}")
    print(f"分类 0（收益率 <= +0.20%）：{count[0]:,} ({count[0]/len(labels):.2%})")
    print(f"分类 1（收益率 >  +0.20%）：{count[1]:,} ({count[1]/len(labels):.2%})")
    print(f"回归目标：均值 {labels['target_return'].mean():.4%} | 中位数 {labels['target_return'].median():.4%} | 标准差 {labels['target_return'].std():.4%}")
    print("\n标签示例：")
    display(labels.head(5))

    # %% [markdown]
    # ## 4. 特征历史依赖与标签核对

    # %%
    # 4. 验证：随机日期与原切分边界处的特征没有使用未来信息
    random_dates = np.random.default_rng(42).choice(
        features.index.to_numpy(), size=min(6, len(features)), replace=False
    )
    boundary_dates = [daily.index[daily.index.get_loc(d) - 1] for d in split_starts]
    check_dates = pd.DatetimeIndex(list(random_dates) + boundary_dates).unique().sort_values()

    print("=" * 65)
    print("[4/5] 时间对齐与防泄露检查")
    for date in check_dates:
        truncated_features = build_features(daily.loc[:date])
        pd.testing.assert_series_equal(
            all_features.loc[date], truncated_features.loc[date], check_exact=True
        )
        pos = daily.index.get_loc(date)
        future_dates = daily.index[pos + 1:pos + 3]
        start, end = (Decimal(str(x)) for x in daily.loc[future_dates, "open"])
        expected_return = (end - start) / start
        row = labels.loc[date]
        assert [row["trade_date"], row["settle_date"]] == list(future_dates)
        assert np.isclose(row["target_return"], float(expected_return), rtol=0, atol=1e-14)
        assert row["target_class"] == int(expected_return > Decimal("0.002"))
        print(f"  PASS | {date:%Y-%m-%d} -> {future_dates[0]:%Y-%m-%d} -> {future_dates[1]:%Y-%m-%d} | class={row['target_class']}")

    print(f"\n检查通过：{len(check_dates)} 个信号日，特征截断重算一致，标签独立核对一致。")
    print("注意：该检查验证特征的历史依赖；训练/验证/测试仍须按时间切分，且考虑标签跨边界问题。")

    # %% [markdown]
    # ## 5. 导出分析特征表

    # %%
    # 5. 导出三个 CSV + 回读检查
    feature_notes = pd.DataFrame([
        ("return_1d", "原有｜趋势", "C[t] / C[t-1] - 1"),
        ("return_5d", "原有｜趋势", "C[t] / C[t-5] - 1"),
        ("return_20d", "原有｜趋势", "C[t] / C[t-20] - 1"),
        ("volatility_20d", "原有｜波动率", "最近 20 个 return_1d 的样本标准差"),
        ("intraday_return", "原有｜隔夜与日内", "C[t] / O[t] - 1"),
        ("intraday_range", "原有｜隔夜与日内", "(H[t] - L[t]) / O[t]"),
        ("close_vs_sma20", "趋势", "C[t] / SMA20[t] - 1"),
        ("sma5_vs_sma20", "趋势", "SMA5[t] / SMA20[t] - 1"),
        ("overnight_return", "隔夜与日内", "O[t] / C[t-1] - 1"),
        ("close_location", "隔夜与日内", "(C[t] - L[t]) / (H[t] - L[t])，价差为零留空"),
        ("volatility_5d", "波动率", "最近 5 个 return_1d 的样本标准差"),
        ("avg_intraday_range_20d", "波动率", "最近 20 个 intraday_range 的均值"),
        ("volume_change_1d", "成交量", "V[t] / V[t-1] - 1"),
        ("volume_vs_prev20", "成交量", "V[t] / mean(V[t-20], …, V[t-1]) - 1"),
    ], columns=["feature", "group", "formula"])
    assert feature_notes["feature"].tolist() == features.columns.tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sp500_features.csv": features,
        "sp500_labels.csv": labels,
        "sp500_feature_notes.csv": feature_notes,
    }
    print("=" * 65)
    print("[5/5] 导出结果")
    for filename, table in outputs.items():
        path = OUTPUT_DIR / filename
        table.to_csv(path, index=(filename != "sp500_feature_notes.csv"),
                     encoding="utf-8-sig", date_format="%Y-%m-%d")
        assert len(pd.read_csv(path)) == len(table), f"回读行数异常：{filename}"
        print(f"  OK | {filename:26s} | {len(table):5,d} 行 | {path.stat().st_size / 1024:.1f} KB")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")
    print("提示：输出 CSV 保留了历史不足/成交量异常带来的 NaN；建模时避免用整份数据拟合填补器或标准化器。")

    # %% [markdown]
    # ## 6. 生成现有分类与回归训练输入
    # 分类：2009 年起，A/B 使用跨市场特征完整的共同样本，A 输入 6 项，B 输入 11 项。
    # 回归：完整历史上原 6 特征有效的样本，XGBoost 与岭回归共用。
    # 日线可用时间、滞后约定、缺失处理沿用原 `load_market_features()`，并导出来源日期供核查。

    # %%
    # 不因为 14 特征表中的成交量缺失而删掉只需要 6 特征的训练样本。
    original = all_features[ORIGINAL]
    market, audit = load_market_features(daily.index)
    regression_samples = add_targets(daily, original)
    classification_samples = add_targets(daily, original.join(market))
    classification_samples = classification_samples.loc["2009-01-01":].dropna(subset=ORIGINAL + MARKET)
    save_csv(OUTPUT_DIR / "market_availability.csv", audit, index=False)

    for task, samples in (("classification", classification_samples), ("regression", regression_samples)):
        save_csv(OUTPUT_DIR / f"{task}_samples.csv", samples)
        splits, summary = chronological_split(samples)
        split_dir = ROOT / "data" / "model_splits" / task
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            save_csv(split_dir / f"{name}.csv", frame)
        save_csv(split_dir / "split_summary.csv", summary, index=False)
        print(f"\n{task} 有效样本：{len(samples):,}")
        display(summary)

    # 记录输入与输出指纹，防止改过数据后误用旧特征；不记录机器专属路径。
    sources = [RAW_FILE, CLEAN_FILE, *FILES.values(), *MARKET_FILES]
    save_json(OUTPUT_DIR / "feature_build.json", {
        "classification_label_threshold": 0.002,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": file_hash(p)} for p in sources],
        "outputs": {f"{task}_samples.csv": file_hash(OUTPUT_DIR / f"{task}_samples.csv")
                    for task in ("classification", "regression")},
    })

def eda():
    """Run stage 03 (03_feature_analysis)."""
    # -*- coding: utf-8 -*-
    # 与同名 Notebook 使用完全相同的代码；按单元格顺序执行。

    # %% [markdown]
    # # 03 特征分析（EDA）
    # 沿用原来的六个步骤：读取、时间边界审计、缺失与标签分布、训练集相关性、年度变化。
    # 分析的是原 6 + 新增 8 共 14 项特征，采用原始日线的 8:1:1 划分。后续训练仍沿用各自有效样本池的划分。
    # 相关性只用训练集；包含测试集的分布图仅作描述。

    # %% [markdown]
    # ## 1. 导入与路径检查

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from experiment_utils import ROOT, FEATURE_DIR, save_csv, show_table, show_figure

    DATA_DIR = ROOT / "data" / "splits"
    OUTPUT_DIR = ROOT / "output" / "feature_analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS = ("train", "val", "test")
    LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    for path in [FEATURE_DIR / "sp500_features.csv", FEATURE_DIR / "sp500_labels.csv",
                 *[DATA_DIR / f"sp500_{name}.csv" for name in SPLITS]]:
        if not path.is_file():
            raise FileNotFoundError(f"请先运行 01 和 02，缺少：{path.name}")

    # %% [markdown]
    # ## 2. 读取特征与标签
    # 仍按日期对齐；检查三份原始日线不重叠，且特征、入场日和退出日都能在行情中找到。

    # %%
    features = pd.read_csv(FEATURE_DIR / "sp500_features.csv", index_col="signal_date",
                           parse_dates=["signal_date"]).sort_index()
    labels = pd.read_csv(FEATURE_DIR / "sp500_labels.csv", index_col="signal_date",
                         parse_dates=["signal_date", "trade_date", "settle_date"]).sort_index()
    original_splits = {}
    for name in SPLITS:
        frame = pd.read_csv(DATA_DIR / f"sp500_{name}.csv", parse_dates=["date"]).sort_values("date")
        assert not frame.empty and frame.date.notna().all() and frame.date.is_unique
        original_splits[name] = frame
    assert features.index.is_unique and features.index.equals(labels.index)
    assert features.shape[1] == 14 and not np.isinf(features.to_numpy()).any()
    val_start = original_splits["val"].date.min()
    test_start = original_splits["test"].date.min()
    assert original_splits["train"].date.max() < val_start
    assert original_splits["val"].date.max() < test_start
    all_dates = pd.DatetimeIndex(pd.concat([original_splits[s].date for s in SPLITS]))
    assert all_dates.is_unique
    assert labels.index.isin(all_dates).all()
    assert labels.trade_date.isin(all_dates).all() and labels.settle_date.isin(all_dates).all()
    print(f"特征 {features.shape}；标签 {labels.shape}")
    show_table(features.head(3))

    # %% [markdown]
    # ## 3. 时间边界检查
    # 沿用原 EDA 的规则：信号日、入场日、退出日必须全部属于同一个原始分区。跨界样本剔除，滚动历史不足造成的缺失保留供检查。

    # %%
    split_data, summary_rows = {}, []
    for name in SPLITS:
        dates = pd.DatetimeIndex(original_splits[name].date)
        candidates = labels.loc[labels.index.isin(dates)]
        # 不能用训练阶段的信号去拟合已经跨进验证或测试阶段的收益。
        safe_labels = candidates.loc[candidates.trade_date.isin(dates) & candidates.settle_date.isin(dates)]
        split_data[name] = {"X": features.loc[safe_labels.index], "labels": safe_labels,
                            "y_reg": safe_labels.target_return.astype(float),
                            "y_cls": safe_labels.target_class.astype(int)}
        assert len(safe_labels) > 0
        summary_rows.append({"split": name, "candidates": len(candidates), "samples": len(safe_labels),
                             "removed_boundary": len(candidates) - len(safe_labels),
                             "first_signal": safe_labels.index.min(), "last_signal": safe_labels.index.max(),
                             "last_settlement": safe_labels.settle_date.max()})
    assert split_data["train"]["labels"].settle_date.max() < split_data["val"]["X"].index.min()
    assert split_data["val"]["labels"].settle_date.max() < split_data["test"]["X"].index.min()
    summary = pd.DataFrame(summary_rows)
    save_csv(OUTPUT_DIR / "split_summary.csv", summary, index=False)
    show_table(summary)

    # %% [markdown]
    # ## 4. 缺失与标签分布
    # 收益率统计乘 100 转为百分数，样本数量 count 保持原值。所有缺失仅报告，不使用完整数据拟合填补器。

    # %%
    missing_report = pd.DataFrame({LABELS[s]: split_data[s]["X"].isna().sum() for s in SPLITS})
    missing_report["Total"] = missing_report.sum(axis=1)
    class_report = pd.DataFrame({LABELS[s]: split_data[s]["y_cls"].value_counts().reindex([0, 1], fill_value=0)
                                 for s in SPLITS})
    return_report = pd.DataFrame({LABELS[s]: split_data[s]["y_reg"].describe() * 100 for s in SPLITS})
    return_report.loc["count"] /= 100
    for name, table in (("missing_values", missing_report), ("class_distribution", class_report),
                        ("return_statistics", return_report)):
        print(name)
        show_table(table.reset_index())
        table.to_csv(OUTPUT_DIR / f"{name}.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    class_report.T.plot.bar(ax=axes[0], rot=0, title="Class Distribution")
    ratios = class_report.div(class_report.sum(axis=0), axis=1).T * 100
    ratios.plot.bar(ax=axes[1], stacked=True, rot=0, title="Class Ratio (%)")
    show_figure(fig, OUTPUT_DIR / "class_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for name in SPLITS:
        sns.kdeplot(split_data[name]["y_reg"] * 100, label=LABELS[name], ax=ax)
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(0.2, color="red", linestyle="--", label="Threshold: +0.2%")
    ax.set(xlabel="Return (%)", title="Target Return Distribution")
    ax.legend()
    show_figure(fig, OUTPUT_DIR / "return_distribution.png")

    # %% [markdown]
    # ## 5. 训练集特征相关性
    # 保留 Pearson 相关、特征与两种标签的相关、绝对相关系数至少 0.85 的特征对。高相关并不自动意味着删除特征。

    # %%
    X_train = split_data["train"]["X"]
    corr_matrix = X_train.corr(method="pearson")
    target_corr = X_train.join(split_data["train"]["labels"][["target_return", "target_class"]]).corr()
    target_corr = target_corr.loc[features.columns, ["target_return", "target_class"]]
    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    high_corr = upper.stack().reset_index()
    high_corr.columns = ["Feature A", "Feature B", "Abs Correlation"]
    high_corr = high_corr.loc[high_corr["Abs Correlation"] >= 0.85].sort_values("Abs Correlation", ascending=False)
    show_table(target_corr.reset_index())
    show_table(high_corr)
    corr_matrix.to_csv(OUTPUT_DIR / "feature_correlations.csv", encoding="utf-8-sig")
    target_corr.to_csv(OUTPUT_DIR / "target_correlations.csv", encoding="utf-8-sig")
    save_csv(OUTPUT_DIR / "high_correlations.csv", high_corr, index=False)

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.4, ax=ax)
    ax.set_title("Feature Correlations - Training Set")
    show_figure(fig, OUTPUT_DIR / "feature_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, target in zip(axes, ["target_return", "target_class"]):
        target_corr[target].sort_values().plot.barh(ax=ax)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set(title=f"Correlation with {target}", xlabel="Pearson Correlation")
    show_figure(fig, OUTPUT_DIR / "target_correlations.png")

    # %% [markdown]
    # ## 6. 年度分布变化
    # 仍按年计算样本数、平均收益、正类比例与收益标准差。该图包含验证和测试数据，仅描述分布，不能用来挑选特征或参数。

    # %%
    plot_data = pd.concat([split_data[s]["labels"] for s in SPLITS]).sort_index()
    yearly_stats = plot_data.groupby(plot_data.index.year).agg(
        sample_count=("target_return", "size"), mean_return=("target_return", "mean"),
        positive_ratio=("target_class", "mean"), return_std=("target_return", "std"))
    yearly_stats.index.name = "year"
    yearly_stats.to_csv(OUTPUT_DIR / "yearly_statistics.csv", encoding="utf-8-sig")
    show_table(yearly_stats.tail(10).reset_index())
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, metric, title in ((axes[0], "mean_return", "Annual Mean Target Return (%)"),
                              (axes[1], "positive_ratio", "Annual Positive Class Ratio (%)")):
        ax.plot(yearly_stats.index, yearly_stats[metric] * 100, marker="o", markersize=3)
        ax.axvline(val_start.year, color="orange", linestyle="--", label="Validation Start")
        ax.axvline(test_start.year, color="red", linestyle="--", label="Test Start")
        ax.set_title(title)
        ax.legend()
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[1].axhline(50, color="gray", linestyle="--")
    axes[1].set_xlabel("Year")
    show_figure(fig, OUTPUT_DIR / "yearly_statistics.png")
    print("特征分析完成，图表与统计已保存到 output/feature_analysis。")

    # Optional fusion stage. Run after the original experiment, using its saved dates.
    import os
    from experiment_utils import (ROOT, read_fusion_frame as read_frame,
                                  new_fusion_output as new_output)
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value)

    def run_fusion_extension():
        from scipy import stats
        OUT = new_output('03_tail_analysis')
        r = read_frame(ROOT/'data/features/regression_samples.csv')
        nval = ntest = round(len(r)*0.1)
        cut = r.iloc[len(r)-nval-ntest].trade_date
        r = r.loc[r.settle_date < cut, 'target_return']
        z = (r-r.mean())/r.std()
        summary = pd.DataFrame([{'n':len(r),'skew':r.skew(),'excess_kurtosis':r.kurt(),
                                'abs_z_over_3':(abs(z)>3).mean(),'normal_reference':2*stats.norm.sf(3)}])
        summary.to_csv(OUT/'tail_summary.csv',index=False)
        display(summary)
        lags = range(1,41)
        series = {'return':r,'abs_return':r.abs(),'squared_return':r**2,'rolling_vol20':r.rolling(20).std().dropna()}
        acf = pd.DataFrame({k:[v.autocorr(i) for i in lags] for k,v in series.items()},index=lags)
        acf.to_csv(OUT/'acf.csv')
        block_vol = r.groupby(np.arange(len(r))//20).std().dropna()
        print('Non-overlapping 20-session volatility lag-1 correlation:',block_vol.autocorr())
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        axes[0].hist(z,bins=80,density=True,alpha=.6); x=np.linspace(-5,5,300)
        axes[0].plot(x,stats.norm.pdf(x)); axes[0].set(title='Standardized returns',yscale='log')
        stats.probplot(z,dist='norm',plot=axes[1])
        acf.plot(ax=axes[2]); axes[2].set(title='Autocorrelation',xlabel='Lag (sessions)')
        fig.tight_layout(); fig.savefig(OUT/'tail_acf.png'); display(fig); plt.close(fig)


    if os.environ.get("SP500_INCLUDE_FUSION") == "1":
        run_fusion_extension()


STAGES = {
    "01": preprocess,
    "02": features,
    "03": eda,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="Notebook stage number")
    args = parser.parse_args()
    for number in (STAGES if args.stage == "all" else [args.stage]):
        STAGES[number]()


if __name__ == "__main__":
    main()
