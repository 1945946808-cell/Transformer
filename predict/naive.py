import pandas as pd
import numpy as np
import os


def naive_method(file_path):
    """
    Naive方法: 使用前一天的价格作为当天的预测
    - 前80%的数据用于训练（但Naive方法不需要训练）
    - 后20%的数据用于测试
    - 各项指标混合所有价格类型计算
    """
    # 数据加载
    df = pd.read_csv(file_path, header=0)

    # 查找OHLC列（支持中英文列名）
    ohlc_cols = []
    for col in ['open', 'Open', '开盘', '开盘价', 'OPEN']:
        if col in df.columns:
            ohlc_cols.append(col)
            break
    for col in ['high', 'High', '最高', '最高价', 'HIGH']:
        if col in df.columns:
            ohlc_cols.append(col)
            break
    for col in ['low', 'Low', '最低', '最低价', 'LOW']:
        if col in df.columns:
            ohlc_cols.append(col)
            break
    for col in ['close', 'Close', '收盘', '收盘价', 'CLOSE']:
        if col in df.columns:
            ohlc_cols.append(col)
            break

    if len(ohlc_cols) != 4:
        # 尝试按位置取列（假设第3-6列是OHLC）
        print(f"警告: 未找到标准OHLC列名，尝试按位置提取（第3-6列）")
        ohlc_data = df.iloc[:, 2:6].values
    else:
        ohlc_data = df[ohlc_cols].values

    # 转换为浮点数（处理可能的字符串格式）
    try:
        ohlc_data = ohlc_data.astype(float)
    except ValueError as e:
        # 如果转换失败，尝试清理数据
        print(f"数据转换错误: {e}")
        # 将数据框转换为数值，非数值变为NaN
        if len(ohlc_cols) == 4:
            ohlc_data = pd.to_numeric(df[ohlc_cols].stack(), errors='coerce').unstack().values
        else:
            ohlc_data = pd.to_numeric(df.iloc[:, 2:6].stack(), errors='coerce').unstack().values
        # 删除包含NaN的行
        ohlc_data = ohlc_data[~np.isnan(ohlc_data).any(axis=1)]

    # 取最后20%的数据作为测试样本
    split_idx = int(round(len(ohlc_data) * 0.8))

    if split_idx == 0:
        print(f"错误: 数据量不足，无法分割训练集和测试集")
        return None

    # 获取训练数据（前80%）和测试数据（后20%）
    train_ohlc = ohlc_data[:split_idx, :]
    test_ohlc = ohlc_data[split_idx:, :]

    if len(train_ohlc) == 0 or len(test_ohlc) == 0:
        print(f"错误: 训练集或测试集为空")
        return None

    # 使用训练数据的最后一天作为测试集第一天的预测值
    last_train_value = train_ohlc[-1]  # 训练集最后一天的值 [open, high, low, close]

    # 构建预测值序列：前一天的值作为当天的预测
    predictions = np.vstack([last_train_value.reshape(1, -1), test_ohlc[:-1]])
    actuals = test_ohlc

    # 计算指标
    metrics = calculate_metrics(predictions, actuals)

    return metrics


def calculate_metrics(predictions, actuals):
    """
    计算五项指标（与论文保持一致）

    Parameters:
    -----------
    predictions : np.array
        预测的OHLC值，形状为 (n_samples, 4) [open, high, low, close]
    actuals : np.array
        实际的OHLC值，形状为 (n_samples, 4) [open, high, low, close]

    Returns:
    --------
    dict : 包含五项指标的字典
    """
    eps = 1e-10
    n = len(predictions)

    if n == 0:
        return {'MAPE': 0, 'MAE': 0, 'RMSE': 0, 'AR': 0, 'UDRATE': 0}

    # 初始化累加器
    MAPE_total = 0
    MAE_total = 0
    RMSE_total = 0
    AR_total = 0
    UD_correct = 0
    UD_total = 0
    ar_count = 0

    for i in range(n):
        open_pred, high_pred, low_pred, close_pred = predictions[i]
        open_act, high_act, low_act, close_act = actuals[i]

        # MAPE, MAE, RMSE (混合所有4个价格)
        for pred_val, act_val in zip([open_pred, high_pred, low_pred, close_pred],
                                     [open_act, high_act, low_act, close_act]):
            if abs(act_val) > eps:
                MAPE_total += abs((pred_val - act_val) / act_val)
            MAE_total += abs(pred_val - act_val)
            RMSE_total += (pred_val - act_val) ** 2

        # AR (区间重叠度) - 使用low和high
        intersection = max(min(high_pred, high_act) - max(low_pred, low_act), 0)
        union = max(high_pred, high_act) - min(low_pred, low_act)

        if union > eps:
            AR_total += intersection / union
            ar_count += 1

        # UDRATE (方向预测准确率) - 使用open和close
        pred_direction = close_pred - open_pred  # 注意：收盘-开盘
        actual_direction = close_act - open_act

        if abs(actual_direction) > eps:
            UD_total += 1
            if pred_direction * actual_direction > 0:
                UD_correct += 1

    # 计算结果（保留更多小数位）
    total_samples = n * 4

    MAPE = MAPE_total / total_samples if total_samples > 0 else 0
    MAE = MAE_total / total_samples if total_samples > 0 else 0
    RMSE = np.sqrt(RMSE_total / total_samples) if total_samples > 0 else 0
    AR = AR_total / max(ar_count, 1)
    UDRATE = UD_correct / max(UD_total, 1)

    return {
        'MAPE': MAPE,
        'MAE': MAE,
        'RMSE': RMSE,
        'AR': AR,
        'UDRATE': UDRATE
    }


if __name__ == "__main__":
    # 测试不同时间粒度的数据
    data_files = {
        '5min': 'MA.CZ_data/5min_ohlc_result.csv',
        '30min': 'MA.CZ_data/30minutes_ohlc_result.csv',
        'daily': 'MA.CZ_data/daily_ohlc_result.csv',
        'weekly': 'MA.CZ_data/weekly_ohlc_result.csv',
        'monthly': 'MA.CZ_data/month_ohlc_result.csv'
    }

    results = {}

    print("=" * 80)
    print("Naive方法预测结果（前一天价格作为当天预测）")
    print("=" * 80)

    for granularity, file_path in data_files.items():
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue

        try:
            # 计算Naive方法结果
            metrics = naive_method(file_path)
            if metrics is not None:
                results[granularity] = metrics
                print(f"\n✓ {granularity} 数据处理成功")
            else:
                print(f"\n✗ {granularity} 数据处理失败")

        except Exception as e:
            print(f"\n✗ 处理 {granularity} 数据时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 输出汇总结果表格（保留小数点后6位）
    if results:
        print("\n" + "=" * 80)
        print("汇总结果")
        print("=" * 80)

        # 创建DataFrame显示结果
        summary_df = pd.DataFrame({
            granularity: {
                'MAPE': f"{res['MAPE']:.6f}",
                'MAE': f"{res['MAE']:.6f}",
                'RMSE': f"{res['RMSE']:.6f}",
                'AR': f"{res['AR']:.6f}",
                'UDRATE': f"{res['UDRATE']:.6f}"
            }
            for granularity, res in results.items()
        }).T

        # 设置列顺序
        summary_df = summary_df[['MAPE', 'MAE', 'RMSE', 'AR', 'UDRATE']]

        # 打印表格
        print(summary_df.to_string())

        # 同时打印数值格式（便于查看）
        print("\n" + "-" * 80)
        print("数值格式（保留6位小数）:")
        print("-" * 80)
        for granularity, res in results.items():
            print(
                f"{granularity:8s} | MAPE={res['MAPE']:.6f} | MAE={res['MAE']:.6f} | RMSE={res['RMSE']:.6f} | AR={res['AR']:.6f} | UDRATE={res['UDRATE']:.6f}")

        # 保存汇总结果
        summary_df.to_csv('naive_results_summary.csv')

        # 同时保存数值格式（用于后续计算）
        numeric_df = pd.DataFrame({
            granularity: {
                'MAPE': res['MAPE'],
                'MAE': res['MAE'],
                'RMSE': res['RMSE'],
                'AR': res['AR'],
                'UDRATE': res['UDRATE']
            }
            for granularity, res in results.items()
        }).T
        numeric_df.to_csv('naive_results_numeric.csv', float_format='%.6f')

        print("\n" + "=" * 80)
        print("✓ 汇总结果已保存到:")
        print("  - naive_results_summary.csv (文本格式)")
        print("  - naive_results_numeric.csv (数值格式)")
        print("=" * 80)
    else:
        print("\n✗ 没有成功处理任何数据文件")