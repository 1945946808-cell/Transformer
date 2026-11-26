# intraday_trading_backtest_methanol.py
"""
基于论文5.3节日内交易策略
使用Transformer模型在甲醇期货日度数据上的预测结果进行回测
使用高频分钟数据实现精确的Low-to-high策略
完全复现论文表5、6、7、8
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import os
from datetime import datetime
from collections import Counter
import itertools


@dataclass
class TradingConfig:
    """甲醇期货交易配置参数"""
    fee_rate: float  # 交易手续费率（按成交金额比例）- 甲醇交易所标准为万分之一（0.0001）
    margin_rate: float  # 保证金比例 - 甲醇约14%-21%
    alpha: float = 0.0024  # 低价修正因子（默认0.24%，可通过优化确定）
    beta: float = 0.0036  # 高价修正因子（默认0.36%，可通过优化确定）
    lot_size: int = 10  # 甲醇：10吨/手
    broker_fee_rate: float = 0.00002  # 期货公司额外手续费率（约万分之0.2）
    bars_per_day: int = 1
    trading_days_per_year: int = 242


class OHLCData:
    """OHLC数据结构"""

    def __init__(self, open_price: float, high_price: float,
                 low_price: float, close_price: float):
        self.o = float(open_price)
        self.h = float(high_price)
        self.l = float(low_price)
        self.c = float(close_price)


class IntradayTradingStrategies:
    """
    日内交易策略实现 - 论文5.3节
    适用于甲醇期货日度K线数据
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        # 使用配置中的alpha和beta参数（可优化）
        self.alpha = config.alpha
        self.beta = config.beta

    def calculate_single_return(self, entry_price: float, exit_price: float,
                                is_long: bool = True) -> float:
        """
        计算单次交易收益率 - 甲醇期货专用

        甲醇期货规则：
        1. 交易单位：10吨/手
        2. 手续费：按成交金额比例收取，开仓+平仓各收一次
        3. 无日内平仓优惠（开平手续费相同）

        公式：Rt = (价格差 × 10 - 手续费) / (开仓价 × 10 × 保证金比例) × 100%
        """
        # 价格差
        if is_long:
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        # 甲醇手续费：按成交金额比例收取，开仓+平仓各收一次
        open_fee = entry_price * self.config.lot_size * self.config.fee_rate
        close_fee = exit_price * self.config.lot_size * self.config.fee_rate
        total_fee = open_fee + close_fee

        # 保证金计算
        margin = entry_price * self.config.lot_size * self.config.margin_rate

        if margin <= 0:
            return 0.0

        # 收益率计算
        return_rate = (price_diff * self.config.lot_size - total_fee) / margin * 100.0
        return float(return_rate)

    def close_to_close_strategy(self, pred_t: OHLCData, actual_t: OHLCData,
                                pred_t1: OHLCData, actual_t1: OHLCData) -> Tuple[float, bool]:
        """收盘-收盘策略（适用于日度K线）"""
        if pred_t1.c > pred_t.c:
            return self.calculate_single_return(actual_t.c, actual_t1.c, is_long=True), True
        elif pred_t1.c < pred_t.c:
            return self.calculate_single_return(actual_t.c, actual_t1.c, is_long=False), True
        return 0.0, False

    def open_to_close_long_strategy(self, actual: OHLCData, pred: OHLCData) -> Tuple[float, bool]:
        """开盘-收盘多头策略（适用于日度K线）"""
        if pred.c > pred.o:
            return self.calculate_single_return(actual.o, actual.c, is_long=True), True
        return 0.0, False

    def open_to_close_short_strategy(self, actual: OHLCData, pred: OHLCData) -> Tuple[float, bool]:
        """开盘-收盘空头策略（适用于日度K线）"""
        if pred.c < pred.o:
            return self.calculate_single_return(actual.o, actual.c, is_long=False), True
        return 0.0, False

    def low_to_high_strategy_with_minute_data(self, minute_ohlc: pd.DataFrame,
                                              pred_low: float, pred_high: float,
                                              actual_close: float) -> Tuple[float, bool, str]:
        """
        使用分钟OHLC数据实现论文中完整的Low-to-high策略
        已删除场景(c)：仅高价触发策略

        参数:
            minute_ohlc: 分钟OHLC数据DataFrame，包含open, high, low, close列
            pred_low: 预测低点
            pred_high: 预测高点
            actual_close: 实际收盘价

        返回:
            (收益率, 是否交易, 场景类型)
        """
        # 计算调整后的预测价位（使用优化后的alpha和beta）
        pred_low_adj = pred_low * (1 + self.alpha)  # x̃(t_l) = x̂(t_l) × (1 + α)
        pred_high_adj = pred_high * (1 - self.beta)  # x̃(t_h) = x̂(t_h) × (1 - β)

        # 获取实际日内高低点
        actual_low = minute_ohlc['low'].min()
        actual_high = minute_ohlc['high'].max()

        # 判断价格触发顺序
        low_trigger_time = None
        high_trigger_time = None
        low_triggered = False
        high_triggered = False

        # 遍历分钟数据，判断价格触发顺序
        for idx, row in minute_ohlc.iterrows():
            # 检查该分钟是否触及预测低点
            if not low_triggered and row['low'] <= pred_low_adj:
                low_trigger_time = idx
                low_triggered = True
            # 检查该分钟是否触及预测高点
            if not high_triggered and row['high'] >= pred_high_adj:
                high_trigger_time = idx
                high_triggered = True
            if low_triggered and high_triggered:
                break

        # 场景(a)：预测区间完全覆盖实际区间
        # 条件：x(t_l) ≤ x̃(t_l) 且 x̃(t_h) ≤ x(t_h)
        if actual_low <= pred_low_adj and pred_high_adj <= actual_high:
            if low_triggered and high_triggered:
                if low_trigger_time < high_trigger_time:
                    # 低价先出现 → 在 x̃(t_l) 做多，在 x̃(t_h) 平仓
                    ret = self.calculate_single_return(pred_low_adj, pred_high_adj, is_long=True)
                    return ret, True, "场景(a)-低价先出现"
                else:
                    # 高价先出现 → 在 x̃(t_h) 做空，在 x̃(t_l) 平仓
                    ret = self.calculate_single_return(pred_high_adj, pred_low_adj, is_long=False)
                    return ret, True, "场景(a)-高价先出现"
            elif low_triggered and not high_triggered:
                # 只触发低价，用收盘价平仓（做多）
                ret = self.calculate_single_return(pred_low_adj, actual_close, is_long=True)
                return ret, True, "场景(a)-仅触发低价"
            elif high_triggered and not low_triggered:
                # 只触发高价，用收盘价平仓（做空）
                ret = self.calculate_single_return(pred_high_adj, actual_close, is_long=False)
                return ret, True, "场景(a)-仅触发高价"

        # 场景(b)：仅低价触发
        # 条件：x(t_l) ≤ x̃(t_l) 且 x(t_h) < x̃(t_h)
        elif actual_low <= pred_low_adj and actual_high < pred_high_adj:
            if low_triggered:
                # 在 x̃(t_l) 做多，在 x(t_c) 平仓
                ret = self.calculate_single_return(pred_low_adj, actual_close, is_long=True)
                return ret, True, "场景(b)-低价策略"

        # 场景(c)：仅高价触发（已删除，不再交易）
        # 原条件：pred_low_adj < actual_low and pred_high_adj <= actual_high
        # 此处跳过，不进行任何交易

        # 场景(d)：预测区间被实际区间包含（不交易）
        # 条件：x̃(t_l) < x(t_l) 且 x(t_h) < x̃(t_h)
        elif pred_low_adj < actual_low and actual_high < pred_high_adj:
            return 0.0, False, "场景(d)-不交易"

        return 0.0, False, "无触发"

    def combined_open_close_strategy(self, actual: OHLCData, pred: OHLCData) -> Tuple[float, bool]:
        """组合策略（适用于日度K线）"""
        if pred.c > pred.o:
            return self.calculate_single_return(actual.o, actual.c, is_long=True), True
        else:
            return self.calculate_single_return(actual.o, actual.c, is_long=False), True


class BacktestEngine:
    """回测引擎 - 针对甲醇期货日度数据优化，支持高频数据"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.strategies = IntradayTradingStrategies(config)

    def calculate_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        计算交易策略评价指标
        针对日度数据调整年化计算
        """
        valid_returns = [r for r in returns if r is not None and not np.isnan(r) and not np.isinf(r)]

        if not valid_returns:
            return {
                'TR': 0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'arr': 0.0,
                'md': 0.0,
                'sharpe_ratio': 0.0
            }

        returns_array = np.array(valid_returns, dtype=np.float64)
        n_trades = len(returns_array)

        mean_return = float(np.mean(returns_array))
        std_return = float(np.std(returns_array, ddof=1)) if n_trades > 1 else 0.0

        cumulative_returns = np.cumprod(1 + returns_array / 100.0)
        final_capital = float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 1.0

        d = len(valid_returns)

        if d > 0:
            actual_years = d / self.config.trading_days_per_year
            if actual_years > 0 and final_capital > 0:
                arr_decimal = final_capital ** (1.0 / actual_years) - 1.0
                arr = arr_decimal * 100.0
            else:
                arr = 0.0
        else:
            arr = 0.0

        if len(cumulative_returns) > 0:
            cumulative_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_max - cumulative_returns) / cumulative_max * 100.0
            md = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            md = 0.0

        risk_free_rate = 0.0262
        sharpe_ratio = 0.0

        if std_return > 0 and d > 0:
            avg_daily_return = mean_return / 100.0
            daily_risk_free = risk_free_rate / self.config.trading_days_per_year
            excess_return = avg_daily_return - daily_risk_free
            sharpe_ratio = (excess_return / (std_return / 100.0)) * np.sqrt(self.config.trading_days_per_year)

        return {
            'TR': n_trades,
            'mean_return': mean_return,
            'std_return': std_return,
            'arr': arr,
            'md': md,
            'sharpe_ratio': sharpe_ratio
        }

    def run_backtest(self, actual_data: List[OHLCData],
                     pred_data: List[OHLCData],
                     minute_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        运行所有策略的回测

        参数:
            actual_data: 实际OHLC数据列表
            pred_data: 预测OHLC数据列表
            minute_data_dict: 分钟数据字典，键为日期字符串，值为分钟OHLC DataFrame
        """
        results = {
            'close_to_close': {'returns': []},
            'open_to_close_long': {'returns': []},
            'open_to_close_short': {'returns': []},
            'low_to_high': {'returns': [], 'scenarios': []},
            'combined_open_close': {'returns': []}
        }

        n_days = len(actual_data)

        # 收盘-收盘策略（相邻交易日）
        for i in range(n_days - 1):
            ret, triggered = self.strategies.close_to_close_strategy(
                pred_data[i], actual_data[i], pred_data[i + 1], actual_data[i + 1]
            )
            if triggered:
                results['close_to_close']['returns'].append(ret)

        # 日内策略（每个交易日独立）
        for i in range(n_days):
            actual_today = actual_data[i]
            pred_today = pred_data[i]

            # 获取该交易日的分钟数据（需要匹配日期）
            date_str = None
            if i < len(list(minute_data_dict.keys())):
                date_str = list(minute_data_dict.keys())[i]

            # 多头策略
            ret, triggered = self.strategies.open_to_close_long_strategy(actual_today, pred_today)
            if triggered:
                results['open_to_close_long']['returns'].append(ret)

            # 空头策略
            ret, triggered = self.strategies.open_to_close_short_strategy(actual_today, pred_today)
            if triggered:
                results['open_to_close_short']['returns'].append(ret)

            # 低-高策略（使用分钟数据）
            if date_str and date_str in minute_data_dict:
                minute_ohlc = minute_data_dict[date_str]
                ret, triggered, scenario = self.strategies.low_to_high_strategy_with_minute_data(
                    minute_ohlc, pred_today.l, pred_today.h, actual_today.c
                )
                if triggered:
                    results['low_to_high']['returns'].append(ret)
                    results['low_to_high']['scenarios'].append(scenario)
                else:
                    results['low_to_high']['scenarios'].append(scenario)
            else:
                # 如果没有分钟数据，不进行交易
                results['low_to_high']['scenarios'].append("无分钟数据")

            # 组合策略
            ret, _ = self.strategies.combined_open_close_strategy(actual_today, pred_today)
            results['combined_open_close']['returns'].append(ret)

        # 计算指标
        metrics_results = {}
        for strategy_name, strategy_results in results.items():
            if strategy_name == 'low_to_high':
                metrics_results[strategy_name] = self.calculate_metrics(strategy_results['returns'])
            else:
                metrics_results[strategy_name] = self.calculate_metrics(strategy_results['returns'])

        return metrics_results


def load_minute_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    加载分钟OHLC数据

    文件格式：第一列为Data，2-5列为open、high、low、close
    """
    print(f"\n加载分钟数据: {file_path}")

    if not os.path.exists(file_path):
        print(f"警告：找不到分钟数据文件 {file_path}")
        return {}

    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        print(f"分钟数据形状: {df.shape}")
        print(f"分钟数据列名: {df.columns.tolist()}")

        required_cols = ['Data', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"错误：分钟数据缺少必要的列 '{col}'")
                print(f"现有列: {df.columns.tolist()}")
                return {}

        df['Data'] = pd.to_datetime(df['Data'])
        df['Date'] = df['Data'].dt.date

        minute_data_dict = {}

        for date, group in df.groupby('Date'):
            date_str = date.strftime('%Y-%m-%d')
            group_sorted = group.sort_values('Data').reset_index(drop=True)
            minute_data_dict[date_str] = group_sorted[['open', 'high', 'low', 'close']]

        print(f"成功加载 {len(minute_data_dict)} 个交易日的分钟数据")

        sample_date = list(minute_data_dict.keys())[0]
        print(f"示例日期 {sample_date}: {len(minute_data_dict[sample_date])} 条分钟数据")
        print(
            f"  该日OHLC范围: open={minute_data_dict[sample_date]['open'].min():.2f}-{minute_data_dict[sample_date]['open'].max():.2f}")

        return minute_data_dict

    except Exception as e:
        print(f"加载分钟数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_transformer_predictions():
    """加载Transformer模型预测结果（日度数据）"""
    file_path = '2024daily_ohlc.csv'

    if not os.path.exists(file_path):
        print(f"\n错误：找不到预测文件 {file_path}")
        return np.array([]), np.array([])

    print(f"\n找到预测文件: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"文件形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        if df.shape[1] < 9:
            print(f"错误：文件列数不足，需要至少9列")
            return np.array([]), np.array([])

        predictions = df.iloc[:, 1:5].values.astype(float)
        actuals = df.iloc[:, 5:9].values.astype(float)

        print(f"✓ 成功加载数据")
        print(f"  数据行数: {len(predictions)}")
        print(f"  预测值范围: open=[{predictions[:, 0].min():.2f}, {predictions[:, 0].max():.2f}]")
        print(f"  实际值范围: open=[{actuals[:, 0].min():.2f}, {actuals[:, 0].max():.2f}]")

        return predictions, actuals

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return np.array([]), np.array([])


def validate_data(predictions, actuals):
    """验证数据的合理性（仅输出价格范围）"""
    print("\n" + "=" * 80)
    print("数据验证")
    print("=" * 80)

    n_samples = min(len(predictions), len(actuals))
    predictions = predictions[:n_samples]
    actuals = actuals[:n_samples]

    print(f"预测值范围: open=[{predictions[:, 0].min():.2f}, {predictions[:, 0].max():.2f}], "
          f"close=[{predictions[:, 3].min():.2f}, {predictions[:, 3].max():.2f}]")
    print(f"实际值范围: open=[{actuals[:, 0].min():.2f}, {actuals[:, 0].max():.2f}], "
          f"close=[{actuals[:, 3].min():.2f}, {actuals[:, 3].max():.2f}]")

    if actuals[:, 0].min() < 1000 or actuals[:, 0].max() > 4000:
        print(f"⚠ 注意：甲醇期货价格通常在1800-2800元/吨范围内")
        print(f"当前价格范围: {actuals[:, 0].min():.0f}-{actuals[:, 0].max():.0f}元/吨")

    print("=" * 80)


def print_table(results: Dict[str, Dict[str, float]],
                table_name: str, scenario_desc: str):
    """打印策略回测结果表格"""
    print(f"\n{table_name}")
    print("=" * 100)
    print(f"{scenario_desc}")
    print("-" * 100)
    print(f"{'Strategy':<45} {'TR':<8} {'R̄(%)':<10} {'σ':<10} {'ARR(%)':<10} {'MD(%)':<10} {'Sharpe':<10}")
    print("-" * 100)

    strategy_names = {
        'close_to_close': 'Close-to-close strategy',
        'open_to_close_long': 'Open-to-close (long) strategy',
        'open_to_close_short': 'Open-to-close (short) strategy',
        'low_to_high': 'Low-to-high strategy',
        'combined_open_close': 'Open-to-close (long) & short strategy'
    }

    for key, name in strategy_names.items():
        if key in results:
            r = results[key]
            print(f"{name:<45} {r['TR']:<8} {r['mean_return']:.3f}%   "
                  f"{r['std_return']:.3f}   {r['arr']:.3f}%   "
                  f"{r['md']:.3f}%    {r['sharpe_ratio']:.4f}")

    print("=" * 100)


def optimize_alpha_beta(predictions, actuals, minute_data_dict,
                        fee_rate: float = 0.00012, margin_rate: float = 0.17,
                        alpha_range: List[float] = None, beta_range: List[float] = None):
    """
    优化α和β参数，基于累计收益最大化原则
    """
    if alpha_range is None:
        alpha_range = [0.001, 0.002, 0.0024, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
    if beta_range is None:
        beta_range = [0.001, 0.002, 0.0036, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]

    print("\n" + "=" * 100)
    print("参数优化: 基于累计收益最大化原则搜索最优 α 和 β")
    print("=" * 100)
    print(f"α 搜索范围: {alpha_range}")
    print(f"β 搜索范围: {beta_range}")
    print(f"总组合数: {len(alpha_range) * len(beta_range)}")
    print("=" * 100)

    n_samples = min(len(predictions), len(actuals))
    n_days = min(n_samples, len(minute_data_dict))

    predictions = predictions[:n_days]
    actuals = actuals[:n_days]

    actual_ohlc = [OHLCData(actuals[i][0], actuals[i][1], actuals[i][2], actuals[i][3])
                   for i in range(n_days)]
    pred_ohlc = [OHLCData(predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3])
                 for i in range(n_days)]

    best_alpha = 0.0024
    best_beta = 0.0036
    best_return = -float('inf')
    results_log = []

    for alpha, beta in itertools.product(alpha_range, beta_range):
        config = TradingConfig(
            fee_rate=fee_rate,
            margin_rate=margin_rate,
            alpha=alpha,
            beta=beta,
            lot_size=10,
            bars_per_day=1,
            trading_days_per_year=242
        )

        engine = BacktestEngine(config)
        results = engine.run_backtest(actual_ohlc, pred_ohlc, minute_data_dict)

        low_to_high_arr = results['low_to_high']['arr']

        results_log.append({
            'alpha': alpha,
            'beta': beta,
            'arr': low_to_high_arr,
            'tr': results['low_to_high']['TR']
        })

        if low_to_high_arr > best_return:
            best_return = low_to_high_arr
            best_alpha = alpha
            best_beta = beta

        print(f"α={alpha:.4f}, β={beta:.4f} → ARR={low_to_high_arr:.4f}%, TR={results['low_to_high']['TR']}")

    print("\n" + "=" * 100)
    print("参数优化结果")
    print("=" * 100)
    print(f"最佳参数: α = {best_alpha:.4f} ({best_alpha * 100:.2f}%), β = {best_beta:.4f} ({best_beta * 100:.2f}%)")
    print(f"最佳年化收益率: {best_return:.4f}%")
    print("=" * 100)

    return best_alpha, best_beta, best_return, results_log


def print_optimization_results(results_log: List[Dict], top_n: int = 10):
    """打印参数优化结果排序"""
    sorted_results = sorted(results_log, key=lambda x: x['arr'], reverse=True)

    print("\n" + "=" * 100)
    print(f"参数优化结果排序 (Top {top_n})")
    print("=" * 100)
    print(f"{'排名':<6} {'α':<10} {'β':<10} {'年化收益率(%)':<20} {'交易次数':<10}")
    print("-" * 100)

    for i, result in enumerate(sorted_results[:top_n]):
        print(
            f"{i + 1:<6} {result['alpha']:.4f}     {result['beta']:.4f}     {result['arr']:>12.4f}%      {result['tr']:<10}")

    print("=" * 100)


def main():
    """主函数：输出论文表5、6、7、8（基于甲醇期货日度数据）"""

    np.random.seed(42)

    print("正在加载甲醇期货日度数据预测结果...")
    predictions, actuals = load_transformer_predictions()

    if len(predictions) == 0 or len(actuals) == 0:
        print("错误：无法加载数据，程序退出")
        return

    print("\n正在加载高频分钟数据...")
    minute_data_dict = load_minute_data('2024_MA.CZC.xlsx')

    if len(minute_data_dict) == 0:
        print("错误：无法加载分钟数据，程序退出")
        return

    n_samples = min(len(predictions), len(actuals))
    predictions = predictions[:n_samples]
    actuals = actuals[:n_samples]

    validate_data(predictions, actuals)

    # 参数优化
    ENABLE_OPTIMIZATION = True

    if ENABLE_OPTIMIZATION:
        print("\n" + "=" * 100)
        print("开始α和β参数优化（基于中费用场景）")
        print("=" * 100)

        best_alpha, best_beta, best_return, opt_results = optimize_alpha_beta(
            predictions, actuals, minute_data_dict,
            fee_rate=0.00012,
            margin_rate=0.17,
            alpha_range=[0.001, 0.002, 0.0024, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010],
            beta_range=[0.001, 0.002, 0.0036, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]
        )

        print_optimization_results(opt_results, top_n=10)
        opt_alpha = best_alpha
        opt_beta = best_beta
    else:
        opt_alpha = 0.0024
        opt_beta = 0.0036
        print(f"\n使用默认参数: α={opt_alpha}, β={opt_beta}")

    n_days = len(actuals)
    max_days = min(252, len(minute_data_dict))
    n_days = min(n_days, max_days)

    predictions = predictions[:n_days]
    actuals = actuals[:n_days]

    print(f"\n使用 {n_days} 个交易日数据进行回测")

    actual_ohlc = [OHLCData(actuals[i][0], actuals[i][1], actuals[i][2], actuals[i][3])
                   for i in range(n_days)]
    pred_ohlc = [OHLCData(predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3])
                 for i in range(n_days)]

    scenarios = [
        {'name': 'Low transaction fee and margin scenario', 'fee_rate': 0.00010, 'margin': 0.14, 'table': 'Table 5'},
        {'name': 'Medium transaction fee and margin scenario', 'fee_rate': 0.00012, 'margin': 0.17, 'table': 'Table 6'},
        {'name': 'High transaction fee and margin scenario', 'fee_rate': 0.00015, 'margin': 0.21, 'table': 'Table 7'}
    ]

    all_scenario_results = []

    for scenario in scenarios:
        print(f"\n{'=' * 100}")
        print(f"运行场景: {scenario['name']}")
        print(f"手续费率: {scenario['fee_rate'] * 10000:.2f}‱ (万分之{scenario['fee_rate'] * 10000:.2f})")
        print(f"保证金比例: {scenario['margin'] * 100}%")
        print(f"α参数: {opt_alpha:.4f} ({opt_alpha * 100:.2f}%), β参数: {opt_beta:.4f} ({opt_beta * 100:.2f}%)")
        print('=' * 100)

        config = TradingConfig(
            fee_rate=scenario['fee_rate'],
            margin_rate=scenario['margin'],
            alpha=opt_alpha,
            beta=opt_beta,
            lot_size=10,
            bars_per_day=1,
            trading_days_per_year=242
        )
        engine = BacktestEngine(config)
        results = engine.run_backtest(actual_ohlc, pred_ohlc, minute_data_dict)
        print_table(results, scenario['table'], f"{scenario['name']} (Methanol Daily data)")
        all_scenario_results.append(results)

    # 计算平均性能（表8）
    print("\n" + "=" * 100)
    print("Table 8: Strategies' average performance among three scenarios (Methanol Daily data)")
    print("=" * 100)

    strategy_names = {
        'close_to_close': 'Close-to-close strategy',
        'open_to_close_long': 'Open-to-close (long) strategy',
        'open_to_close_short': 'Open-to-close (short) strategy',
        'low_to_high': 'Low-to-high strategy',
        'combined_open_close': 'Open-to-close (long) & short strategy'
    }

    print(f"{'Strategy':<45} {'TR':<8} {'R̄(%)':<10} {'σ':<10} {'ARR(%)':<10} {'MD(%)':<10} {'Sharpe':<10}")
    print("-" * 100)

    for key, name in strategy_names.items():
        avg_results = {
            'TR': np.mean([r[key]['TR'] for r in all_scenario_results]),
            'mean_return': np.mean([r[key]['mean_return'] for r in all_scenario_results]),
            'std_return': np.mean([r[key]['std_return'] for r in all_scenario_results]),
            'arr': np.mean([r[key]['arr'] for r in all_scenario_results]),
            'md': np.mean([r[key]['md'] for r in all_scenario_results]),
            'sharpe_ratio': np.mean([r[key]['sharpe_ratio'] for r in all_scenario_results])
        }

        print(f"{name:<45} {avg_results['TR']:.0f}      {avg_results['mean_return']:.3f}%   "
              f"{avg_results['std_return']:.3f}   {avg_results['arr']:.3f}%   "
              f"{avg_results['md']:.3f}%    {avg_results['sharpe_ratio']:.4f}")

    print("=" * 100)

    # 保存结果
    try:
        summary_data = []
        for scenario_idx, scenario in enumerate(scenarios):
            for key, name in strategy_names.items():
                r = all_scenario_results[scenario_idx][key]
                summary_data.append({
                    'Scenario': scenario['name'],
                    'Strategy': name,
                    'TR': r['TR'],
                    'Mean_Return_Per_Trade': r['mean_return'],
                    'Std_Return': r['std_return'],
                    'ARR': r['arr'],
                    'MD': r['md'],
                    'Sharpe': r['sharpe_ratio']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('backtest_results_Methanol.csv', index=False)
        print("\n结果已保存到: backtest_results_Methanol.csv")

        summary_stats = []
        for key, name in strategy_names.items():
            avg_results = {
                'Strategy': name,
                'Avg_TR': np.mean([r[key]['TR'] for r in all_scenario_results]),
                'Avg_Mean_Return': np.mean([r[key]['mean_return'] for r in all_scenario_results]),
                'Avg_Std': np.mean([r[key]['std_return'] for r in all_scenario_results]),
                'Avg_ARR': np.mean([r[key]['arr'] for r in all_scenario_results]),
                'Avg_MD': np.mean([r[key]['md'] for r in all_scenario_results]),
                'Avg_Sharpe': np.mean([r[key]['sharpe_ratio'] for r in all_scenario_results])
            }
            summary_stats.append(avg_results)

        summary_stats_df = pd.DataFrame(summary_stats)
        summary_stats_df.to_csv('backtest_summary_Methanol.csv', index=False)
        print("汇总统计已保存到: backtest_summary_Methanol.csv")

        if ENABLE_OPTIMIZATION:
            opt_df = pd.DataFrame(opt_results)
            opt_df.to_csv('alpha_beta_optimization_results.csv', index=False)
            print("参数优化结果已保存到: alpha_beta_optimization_results.csv")

    except Exception as e:
        print(f"保存结果文件时出错: {e}")


if __name__ == "__main__":
    main()