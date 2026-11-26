import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os


def random_forest_method(file_path, is_train=True, model_save_path='rf_models'):
    """
    随机森林方法: 使用随机森林回归预测价格

    Parameters:
    -----------
    file_path : str
        CSV文件路径
    is_train : bool
        True: 训练模式，False: 测试模式
    model_save_path : str
        模型保存路径
    """
    # 数据加载
    df = pd.read_csv(file_path, header=0)

    # 查找OHLC列（支持中英文列名）
    ohlc_cols = []
    open_col, high_col, low_col, close_col = None, None, None, None

    for col in ['open', 'Open', '开盘', '开盘价']:
        if col in df.columns:
            open_col = col
            break
    for col in ['high', 'High', '最高', '最高价']:
        if col in df.columns:
            high_col = col
            break
    for col in ['low', 'Low', '最低', '最低价']:
        if col in df.columns:
            low_col = col
            break
    for col in ['close', 'Close', '收盘', '收盘价']:
        if col in df.columns:
            close_col = col
            break

    if all([open_col, high_col, low_col, close_col]):
        price_data = df[[open_col, high_col, low_col, close_col]].copy()
    else:
        # 按位置取列（假设第3-6列是OHLC）
        print(f"警告: 未找到标准OHLC列名，按位置提取（第3-6列）")
        price_data = df.iloc[:, 2:6].copy()
        # 重命名列
        price_data.columns = ['open', 'high', 'low', 'close']

    # 转换为数值
    for col in price_data.columns:
        price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
    price_data = price_data.dropna()

    # 按论文标准顺序获取数据: [open, high, low, close]
    ohlc_data = price_data[['open', 'high', 'low', 'close']].values

    # 取最后20%的数据作为测试样本
    split_idx = int(round(len(ohlc_data) * 0.8))

    if split_idx < 6:  # 至少需要6个样本（5个lag + 1个目标）
        print(f"错误: 数据量不足，需要至少6个样本，当前只有{len(ohlc_data)}个")
        return None

    train_data = ohlc_data[:split_idx, :]  # [open, high, low, close]
    test_data = ohlc_data[split_idx:, :]

    # 特征工程：使用前n_lags天的数据作为特征
    n_lags = 5

    # 价格类型（按论文顺序）
    price_types = ['open', 'high', 'low', 'close']

    if is_train:
        # 训练模式
        os.makedirs(model_save_path, exist_ok=True)

        # 准备训练数据
        X_train = []
        y_train = {pt: [] for pt in price_types}

        for i in range(n_lags, len(train_data)):
            # 特征：前n_lags天的所有4个价格
            features = train_data[i - n_lags:i].flatten()
            X_train.append(features)

            # 目标：当天的价格
            y_train['open'].append(train_data[i, 0])  # open
            y_train['high'].append(train_data[i, 1])  # high
            y_train['low'].append(train_data[i, 2])  # low
            y_train['close'].append(train_data[i, 3])  # close

        X_train = np.array(X_train)

        if len(X_train) == 0:
            print("错误: 训练数据不足")
            return None

        models = {}
        scalers = {}

        # 训练模型
        for pt in price_types:
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            scalers[pt] = scaler

            # 训练随机森林
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train[pt])
            models[pt] = model

            # 保存模型和标准化器
            joblib.dump(model, os.path.join(model_save_path, f'{pt}_model.pkl'))
            joblib.dump(scaler, os.path.join(model_save_path, f'{pt}_scaler.pkl'))

        # 对训练集进行预测（用于评估训练效果）
        predictions_train = []
        actuals_train = []

        for i in range(n_lags, len(train_data)):
            features = train_data[i - n_lags:i].flatten().reshape(1, -1)

            pred = []
            for pt in price_types:
                features_scaled = scalers[pt].transform(features)
                pred.append(models[pt].predict(features_scaled)[0])

            predictions_train.append(pred)
            actuals_train.append(train_data[i])

        if len(predictions_train) == 0:
            print("错误: 训练预测数据不足")
            return None

        predictions_train = np.array(predictions_train)
        actuals_train = np.array(actuals_train)

        # 计算训练集指标
        train_metrics = calculate_metrics(predictions_train, actuals_train)
        return train_metrics

    else:
        # 测试模式
        models = {}
        scalers = {}

        # 加载模型和标准化器
        for pt in price_types:
            model_path = os.path.join(model_save_path, f'{pt}_model.pkl')
            scaler_path = os.path.join(model_save_path, f'{pt}_scaler.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models[pt] = joblib.load(model_path)
                scalers[pt] = joblib.load(scaler_path)
            else:
                print(f"错误: 模型文件 {model_path} 不存在！")
                print("请先设置 is_train = True 运行训练。")
                return None

        # 使用测试数据进行预测
        predictions = []
        actuals = []

        for i in range(n_lags, len(test_data)):
            features = test_data[i - n_lags:i].flatten().reshape(1, -1)

            pred = []
            for pt in price_types:
                features_scaled = scalers[pt].transform(features)
                pred.append(models[pt].predict(features_scaled)[0])

            predictions.append(pred)
            actuals.append(test_data[i])

        if len(predictions) == 0:
            print("错误: 测试数据不足")
            return None

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 计算测试集指标
        test_metrics = calculate_metrics(predictions, actuals)

        return {
            'MAPE': test_metrics['MAPE'],
            'MAE': test_metrics['MAE'],
            'RMSE': test_metrics['RMSE'],
            'AR': test_metrics['AR'],
            'UDRATE': test_metrics['UDRATE'],
            'predictions': predictions,
            'actuals': actuals
        }


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
        pred_direction = close_pred - open_pred
        actual_direction = close_act - open_act

        if abs(actual_direction) > eps:
            UD_total += 1
            if pred_direction * actual_direction > 0:
                UD_correct += 1

    # 计算结果
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


def run_all_granularities(data_files, model_dir_prefix='rf_models'):
    """
    运行所有时间粒度的随机森林预测

    Parameters:
    -----------
    data_files : dict
        数据文件路径字典
    model_dir_prefix : str
        模型保存目录前缀
    """
    results = {}

    print("=" * 80)
    print("随机森林方法预测结果")
    print("=" * 80)

    for granularity, file_path in data_files.items():
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue

        model_save_path = f"{model_dir_prefix}_{granularity}"

        try:
            # 训练模式
            print(f"\n训练 {granularity} 数据...")
            train_metrics = random_forest_method(file_path, is_train=True, model_save_path=model_save_path)

            if train_metrics is None:
                print(f"✗ {granularity} 训练失败")
                continue

            print(f"训练集 - MAPE: {train_metrics['MAPE']:.6f}")

            # 测试模式
            print(f"测试 {granularity} 数据...")
            test_results = random_forest_method(file_path, is_train=False, model_save_path=model_save_path)

            if test_results is None:
                print(f"✗ {granularity} 测试失败")
                continue

            results[granularity] = {
                'MAPE': test_results['MAPE'],
                'MAE': test_results['MAE'],
                'RMSE': test_results['RMSE'],
                'AR': test_results['AR'],
                'UDRATE': test_results['UDRATE']
            }

            print(f"✓ {granularity} 完成 - MAPE: {test_results['MAPE']:.6f}")

        except Exception as e:
            print(f"✗ 处理 {granularity} 数据时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 输出汇总结果
    if results:
        print("\n" + "=" * 80)
        print("随机森林方法汇总结果")
        print("=" * 80)

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

        summary_df = summary_df[['MAPE', 'MAE', 'RMSE', 'AR', 'UDRATE']]
        print(summary_df.to_string())

        # 保存结果
        summary_df.to_csv('random_forest_results_summary.csv')

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
        numeric_df.to_csv('random_forest_results_numeric.csv', float_format='%.6f')

        print("\n✓ 结果已保存到:")
        print("  - random_forest_results_summary.csv")
        print("  - random_forest_results_numeric.csv")
    else:
        print("\n✗ 没有成功处理任何数据文件")

    return results


if __name__ == "__main__":
    # 配置所有时间粒度的数据文件
    data_files = {
        '5min': 'MA.CZ_data/5min_ohlc_result.csv',
        '30min': 'MA.CZ_data/30minutes_ohlc_result.csv',
        'daily': 'MA.CZ_data/daily_ohlc_result.csv',
        'weekly': 'MA.CZ_data/weekly_ohlc_result.csv',
        'monthly': 'MA.CZ_data/month_ohlc_result.csv'
    }

# 运行所有粒度
    results = run_all_granularities(data_files)