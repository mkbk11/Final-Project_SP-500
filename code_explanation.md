## 01_data_preprocessing.ipynb

```python
# Extract timestamps and daily market trends, align them into a table by position
result = data["chart"]["result"][0]
quote = result["indicators"]["quote"][0]
```
通过字典的多层索引，获取行情数据。其中 `result` 包含时间戳和基本信息，`quote` 包含每日价格和成交量，`[0]` 表示取列表中的第一个元素。


```python
df = pd.DataFrame({
    "timestamp": result["timestamp"],
    "open": quote["open"],
    "high": quote["high"],
    "low": quote["low"],
    "close": quote["close"],
    "volume": quote["volume"],
})
```
使用 Pandas 将数据按位置对应关系转换为二维表格，每行对应一个交易日。

```python
assert result["meta"]["symbol"] == "^GSPC"      # verify data
display(df.head())
```
检查数据是否属于标普500指数。如果不匹配，则抛出 AssertionError，停止执行。
打印，默认为head(5)


```python
# 计算描述性统计指标，并设置指定的分位数
stats = market.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T

# 计算各字段的样本方差
stats["variance"] = market.var()

# 展示统计结果
display(stats)
```
- `describe()`：计算数据数量、均值、标准差、最小值、最大值等统计指标。
- `percentiles=[]`：指定需要计算的分位数，即5%、25%、50%、75%、95%。
- `.T`：转置表格，使每行对应一个行情字段，方便阅读。
- `market.var()`：计算样本方差，衡量数据的离散程度，默认使用 \(n-1\) 作为分母。
- `stats["variance"]`：在统计表中新增方差这一列。
- `display(stats)`：展示最终的统计表格。
注意： `describe()` 默认只计算25%、50%、75%分位数，且不包含方差，因此这里额外指定了5%、95%分位数，并单独计算方差。





```python

```



```python

```




```python

```
