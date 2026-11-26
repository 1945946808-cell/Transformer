# Transformer
参考论文：Transformer-based forecasting for intraday trading in the Shanghai crude oil market_ Analyzing open-high-low-close prices
论文比较 5 分钟、30 分钟、日、周、月五种时间粒度，以及两种价格表征。标准 OHLC 使用开盘、最高、最低、收盘价。分位点表征保留开盘和收盘，将周期内分钟 OHLC 合并后计算上25%分位点和下上25%分位点。预测实验比较 Transformer、LSTM、GRU、随机森林、支持向量回归（SVR）和朴素法，使用 MAPE、MAE、RMSE、AR 和 UD 评价价格误差与方向判断。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [data/original data/](data/original%20data/) | 三个期货品种的原始 Excel 文件 |
| [data/](data/) | 各品种的多粒度 OHLC CSV 和分位点计算脚本 |
| [predict/](predict/) | Transformer、LSTM、GRU、随机森林、SVR 和朴素法的预测实验脚本 |
| [strategy/predict1.py](strategy/predict1.py) | 训练甲醇日线 Transformer，或加载权重并导出验证区间的预测值和真实值 |
| [strategy/strategy.py](strategy/strategy.py) | 甲醇 OHLC 策略回测与高低价修正参数搜索 |

## 环境配置

从仓库根目录运行以下命令。Windows 上可使用 PowerShell。

```powershell
git clone https://github.com/1945946808-cell/Transformer.git
cd Transformer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas openpyxl torch scikit-learn joblib matplotlib
```

## 运行甲醇预测与回测

### 1. 准备日线数据

仓库中的 [甲醇日线 CSV](data/MA.CZ_OHLC%20data/daily_ohlc_result.csv) 使用中文列名。将 `日期、开盘、最高、最低、收盘` 对应改为 `Data,open,high,low,close`，按日期升序排列，将转换后的 CSV 放到 `MA.CZ_data/daily_ohlc_result.csv`，或修改脚本中的 `csv_file_name` 指向转换后的文件。

如果需要重新生成分位点数据，可运行 [data/data_25%.py](data/data_25%25.py)。先将脚本中的输入路径指向包含时间和分钟 OHLC 的工作簿，并核对表头和交易时段。脚本将各周期的 CSV 和汇总 Excel 写入 `multi_period_output/`。

### 2. 训练并导出预测

将 `strategy/predict1.py` 中的 `is_train` 设为 `True`，运行以下命令训练并保存 `best.pth`。再将 `is_train` 改回 `False`，在同一目录下重复运行以导出结果。

```powershell
.\.venv\Scripts\python.exe strategy/predict1.py
```

每次使用紧邻目标日前的 5 条真实日线预测目标日的 OHLC。导出的 `transformer_prediction_results.csv` 每行包含目标日期、4 列预测值和 4 列真实值。当前入口默认按行位置选取验证窗口。复现论文的全年实验时，应按预测目标日期筛选 2024 年全部 242 个交易日，并核对模型训练和验证切分。

### 3. 运行交易回测

在仓库根目录准备以下两个文件：

- `2024daily_ohlc.csv`。以日期为第一列，接着是预测开、高、低、收，再接实际开、高、低、收。可从上一步导出的结果整理。
- `2024_MA.CZC.xlsx`。准备与日线预测同一批交易日的分钟数据，表头为 `Data,open,high,low,close`。当前回测按行位置配对日线和分钟数据，运行前需核对两边的交易日期和顺序完全一致。

```powershell
.\.venv\Scripts\python.exe strategy/strategy.py
```

回测比较收盘到收盘、开盘到收盘做多、开盘到收盘做空、高低价和开盘到收盘多空组合五种策略。脚本输出 `backtest_results_Methanol.csv`、`backtest_summary_Methanol.csv`，以及启用参数搜索时的 `alpha_beta_optimization_results.csv`。

## 论文复现口径

- **数据切分**。当前预测脚本用全量数据计算标准化参数，窗口按起点划分。若评估 2024 年样本外表现，应按目标日期划分窗口，并用训练集估计标准化参数。
- **交易成本与年化**。论文手续费率为 1.000‰、1.200‰ 和 1.500‰，保证金比例为 14%、17% 和 21%。当前代码费率小一位，且年化使用成交笔数。论文式（3-14）按测试期交易日数年化，全年取 242。
- **高低价规则**。当前 OHLC 回测保留 a、b、d 三种场景。
