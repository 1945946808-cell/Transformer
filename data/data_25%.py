import pandas as pd
import numpy as np
from datetime import time, datetime
import os
import time as time_module


def calculate_percentiles(group):
    """
    计算该周期内所有价格（O、H、L、C）的分位数

    参数:
    group: 包含open, high, low, close列的数据组

    返回:
    上25%分位点(75%分位数)和下25%分位点(25%分位数)
    """
    # 将所有价格展开成一维数组
    all_prices = np.concatenate([
        group['open'].values,
        group['high'].values,
        group['low'].values,
        group['close'].values
    ])

    # 计算分位数
    upper_25 = np.percentile(all_prices, 75)  # 75%分位数（上四分位）
    lower_25 = np.percentile(all_prices, 25)  # 25%分位数（下四分位）

    return pd.Series({
        '开盘': group.iloc[0]['open'],
        '收盘': group.iloc[-1]['close'],
        '上25%分位点': upper_25,
        '下25%分位点': lower_25
    })


def process_5min_ohlc(df):
    """
    处理5分钟K线数据

    5分钟K线计算规则：
    - 每5分钟一个时间段（9:30-9:34, 9:35-9:39, ...）
    - 10:15、11:30、15:00不形成单独的K线
    """
    print("\n处理5分钟K线...")

    # 创建5分钟时间段标识
    def get_5min_slot(dt):
        hour = dt.hour
        minute = dt.minute

        # 排除边界时间点
        if hour == 10 and minute == 15:
            return None
        if hour == 11 and minute == 30:
            return None
        if hour == 15 and minute == 0:
            return None

        slot_minute = (minute // 5) * 5
        slot_id = hour * 12 + (slot_minute // 5)
        return int(slot_id)

    def get_5min_label(slot_id):
        if slot_id is None or pd.isna(slot_id):
            return None
        slot_id = int(slot_id)
        hour = slot_id // 12
        minute = (slot_id % 12) * 5
        return f"{hour:02d}:{minute:02d}"

    # 应用函数
    df_copy = df.copy()
    df_copy['slot_id'] = df_copy['datetime'].apply(get_5min_slot)
    df_copy = df_copy[df_copy['slot_id'].notna()]
    df_copy['slot_id'] = df_copy['slot_id'].astype(int)
    df_copy['时间段'] = df_copy['slot_id'].apply(get_5min_label)

    # 按日期和slot_id分组聚合
    result = df_copy.groupby(['date', 'slot_id']).apply(calculate_percentiles).reset_index()

    # 整理结果
    result = result.rename(columns={'date': '日期'})
    result = result.drop(['slot_id', 'level_0', 'level_1'], axis=1, errors='ignore')

    # 添加时间段列（从原始数据中获取）
    # 创建一个字典映射slot_id到时间段标签
    slot_label_dict = dict(zip(df_copy['slot_id'], df_copy['时间段']))
    result['时间段'] = result['日期'].apply(lambda x: '')  # 临时列

    # 但实际上时间段应该从分组信息中获取，需要重新处理

    # 重新处理，确保时间段列存在
    # 更简单的方法：重新分组时保留时间段信息
    def aggregate_with_label(group):
        """聚合时保留时间段标签"""
        result_series = calculate_percentiles(group)
        result_series['时间段'] = group['时间段'].iloc[0]
        return result_series

    result = df_copy.groupby(['date', 'slot_id']).apply(aggregate_with_label).reset_index()
    result = result.rename(columns={'date': '日期'})
    result = result.drop(['slot_id', 'level_0', 'level_1'], axis=1, errors='ignore')

    # 排序
    result['sort_key'] = pd.to_datetime(result['日期'].astype(str) + ' ' + result['时间段'])
    result = result.sort_values('sort_key').reset_index(drop=True)
    result = result.drop('sort_key', axis=1)

    # 选择输出列
    result = result[['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘']]

    print(f"  生成5分钟K线: {len(result)} 条")
    return result


def process_30min_ohlc(df):
    """
    处理30分钟K线数据

    30分钟K线计算规则：
    - 每30分钟一个时间段（9:30-9:59, 10:30-10:59, 11:00-11:29, 13:30-13:59, 14:00-14:29, 14:30-14:59）
    - 10:00-10:30不满30分钟，省略
    - 11:30 休市
    """
    print("\n处理30分钟K线...")

    def get_30min_slot(dt):
        hour = dt.hour
        minute = dt.minute

        # 排除不满30分钟的时段
        # 10:00-10:30 不满30分钟（实际只有15分钟交易时间到10:15，中间有休息）
        if hour == 10 and minute < 30:
            return None
        # 11:30 是休市时间
        if hour == 11 and minute >= 30:
            return None
        # 15:00 收盘
        if hour == 15 and minute == 0:
            return None

        # 计算30分钟段：0-29分钟为一段，30-59分钟为一段
        if minute < 30:
            slot_minute = 0
        else:
            slot_minute = 30

        slot_id = hour * 2 + (slot_minute // 30)
        return int(slot_id)

    def get_30min_label(slot_id):
        if slot_id is None or pd.isna(slot_id):
            return None
        slot_id = int(slot_id)
        hour = slot_id // 2
        minute = (slot_id % 2) * 30
        return f"{hour:02d}:{minute:02d}"

    # 应用函数
    df_copy = df.copy()
    df_copy['slot_id'] = df_copy['datetime'].apply(get_30min_slot)
    df_copy = df_copy[df_copy['slot_id'].notna()]

    if len(df_copy) == 0:
        print("  警告：没有生成30分钟K线数据")
        return pd.DataFrame(columns=['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘'])

    df_copy['slot_id'] = df_copy['slot_id'].astype(int)
    df_copy['时间段'] = df_copy['slot_id'].apply(get_30min_label)

    # 按日期和slot_id分组聚合（保留时间段标签）
    def aggregate_with_label(group):
        result_series = calculate_percentiles(group)
        result_series['时间段'] = group['时间段'].iloc[0]
        return result_series

    result = df_copy.groupby(['date', 'slot_id']).apply(aggregate_with_label).reset_index()

    # 整理结果
    result = result.rename(columns={'date': '日期'})
    result = result.drop(['slot_id', 'level_0', 'level_1'], axis=1, errors='ignore')

    # 排序
    result['sort_key'] = pd.to_datetime(result['日期'].astype(str) + ' ' + result['时间段'])
    result = result.sort_values('sort_key').reset_index(drop=True)
    result = result.drop('sort_key', axis=1)

    # 选择输出列
    result = result[['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘']]

    print(f"  生成30分钟K线: {len(result)} 条")
    return result


def process_daily_ohlc(df):
    """
    处理日K线数据
    """
    print("\n处理日K线...")

    # 按日期分组聚合
    result = df.groupby('date').apply(calculate_percentiles).reset_index()

    # 整理结果
    result = result.rename(columns={'date': '日期'})
    result = result.drop(['level_0', 'level_1'], axis=1, errors='ignore')
    result['时间段'] = '日线'

    # 选择输出列
    result = result[['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘']]

    print(f"  生成日K线: {len(result)} 条")
    return result


def process_weekly_ohlc(df):
    """
    处理周K线数据
    """
    print("\n处理周K线...")

    # 添加周数标识
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    df_copy['week'] = df_copy['datetime'].dt.isocalendar().week

    # 按年周分组聚合
    result = df_copy.groupby(['year', 'week']).apply(calculate_percentiles).reset_index()

    # 生成周标签
    result['日期'] = result.apply(
        lambda row: f"{int(row['year'])}-W{int(row['week']):02d}",
        axis=1
    )

    # 整理结果
    result = result.drop(['year', 'week', 'level_0', 'level_1'], axis=1, errors='ignore')
    result['时间段'] = '周线'

    # 选择输出列
    result = result[['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘']]

    print(f"  生成周K线: {len(result)} 条")
    return result


def process_monthly_ohlc(df):
    """
    处理月K线数据
    """
    print("\n处理月K线...")

    # 添加月份标识
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    df_copy['month'] = df_copy['datetime'].dt.month

    # 按年月分组聚合
    result = df_copy.groupby(['year', 'month']).apply(calculate_percentiles).reset_index()

    # 生成月份标签
    result['日期'] = result.apply(
        lambda row: f"{int(row['year'])}-{int(row['month']):02d}",
        axis=1
    )

    # 整理结果
    result = result.drop(['year', 'month', 'level_0', 'level_1'], axis=1, errors='ignore')
    result['时间段'] = '月线'

    # 选择输出列
    result = result[['日期', '时间段', '开盘', '上25%分位点', '下25%分位点', '收盘']]

    print(f"  生成月K线: {len(result)} 条")
    return result


def generate_statistics_report(results):
    """生成多周期K线统计报告"""
    print("\n" + "=" * 80)
    print("多周期K线统计报告（含分位数）")
    print("=" * 80)

    for period_name, result_df in results.items():
        if len(result_df) == 0:
            continue

        print(f"\n{period_name}K线统计:")
        print(f"  总K线数: {len(result_df):,}")
        print(f"  日期范围: {result_df['日期'].iloc[0]} 到 {result_df['日期'].iloc[-1]}")
        print(f"  开盘价范围: {result_df['开盘'].min():.2f} - {result_df['开盘'].max():.2f}")
        print(f"  上25%分位点范围: {result_df['上25%分位点'].min():.2f} - {result_df['上25%分位点'].max():.2f}")
        print(f"  下25%分位点范围: {result_df['下25%分位点'].min():.2f} - {result_df['下25%分位点'].max():.2f}")
        print(f"  收盘价范围: {result_df['收盘'].min():.2f} - {result_df['收盘'].max():.2f}")


def save_all_periods(results, output_dir="multi_period_output"):
    """保存所有周期的K线数据"""
    print("\n" + "=" * 80)
    print("保存多周期K线数据:")
    print("=" * 80)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")

    saved_files = []

    for period_name, result_df in results.items():
        if len(result_df) == 0:
            continue

        # 保存为CSV
        csv_file = os.path.join(output_dir, f"{period_name}_ohlc_with_percentiles.csv")
        result_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"  已保存: {csv_file} ({len(result_df)} 条)")

        saved_files.append((period_name, result_df))

    # 保存所有周期到同一个Excel文件的不同sheet
    if saved_files:
        excel_file = os.path.join(output_dir, "all_periods_ohlc.xlsx")
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for period_name, result_df in saved_files:
                    sheet_name = f"{period_name}K线"
                    # 限制sheet名称长度（Excel限制31个字符）
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  已保存合并Excel: {excel_file}")
        except Exception as e:
            print(f"  保存Excel失败: {e}，但CSV文件已成功保存")

    return saved_files


def main():
    """主函数"""
    # 输入文件路径
    file_path = r"Data\MA.CZC.xlsx"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 尝试其他可能的文件名和路径
        possible_files = [
            r"Data\MA.CZC.xlsx",
            r"Data\MA.CZC_py.xlsx",
            r"MA.CZC.xlsx",
            r"MA.CZC_py.xlsx",
            r"Data\SA.CZC.xlsx",
            r"SA.CZC.xlsx"
        ]

        found = False
        for pf in possible_files:
            if os.path.exists(pf):
                file_path = pf
                found = True
                print(f"找到文件: {file_path}")
                break

        if not found:
            print(f"错误：找不到数据文件")
            print(f"当前工作目录: {os.getcwd()}")

            # 列出所有Excel文件
            print("\n当前目录下的Excel文件:")
            for f in os.listdir('.'):
                if f.endswith(('.xlsx', '.xls')):
                    print(f"  - {f}")

            if os.path.exists("Data"):
                print("\nData文件夹中的文件:")
                for f in os.listdir("Data"):
                    print(f"  - {f}")
            return None

    print(f"使用数据文件: {file_path}")
    total_start = time_module.time()

    try:
        # 步骤1: 读取数据
        print("\n" + "=" * 80)
        print("步骤1: 读取原始分钟数据")
        print("=" * 80)
        read_start = time_module.time()

        # 读取Excel文件
        df = pd.read_excel(file_path)

        print(f"读取成功！数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前3行预览:")
        print(df.head(3))

        # 根据实际列名处理数据
        # 您的列名是：time, open, high, low, close
        if len(df.columns) >= 5:
            # 如果列名是英文
            if 'time' in df.columns or 'Time' in df.columns:
                # 找到时间列
                time_col = None
                for col in ['time', 'Time', 'datetime', 'DateTime', '日期', '时间']:
                    if col in df.columns:
                        time_col = col
                        break

                if time_col:
                    df['datetime'] = pd.to_datetime(df[time_col])
                else:
                    # 假设第一列是时间
                    df['datetime'] = pd.to_datetime(df.iloc[:, 0])

                # 找到OHLC列
                open_col = 'open' if 'open' in df.columns else ('Open' if 'Open' in df.columns else df.columns[1])
                high_col = 'high' if 'high' in df.columns else ('High' if 'High' in df.columns else df.columns[2])
                low_col = 'low' if 'low' in df.columns else ('Low' if 'Low' in df.columns else df.columns[3])
                close_col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else df.columns[4])

                df['open'] = pd.to_numeric(df[open_col], errors='coerce')
                df['high'] = pd.to_numeric(df[high_col], errors='coerce')
                df['low'] = pd.to_numeric(df[low_col], errors='coerce')
                df['close'] = pd.to_numeric(df[close_col], errors='coerce')
            else:
                # 直接使用前5列
                df = df.iloc[:, :5]
                df.columns = ['datetime', 'open', 'high', 'low', 'close']
        else:
            print(f"错误：文件只有{len(df.columns)}列，需要至少5列")
            return None

        read_time = time_module.time() - read_start
        print(f"\n  读取完成，用时: {read_time:.2f}秒")
        print(f"  数据形状: {df.shape}")

        # 步骤2: 数据清洗
        print("\n步骤2: 数据清洗")
        print("=" * 80)
        clean_start = time_module.time()

        # 删除无效数据
        before = len(df)
        df = df.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
        after = len(df)
        if after < before:
            print(f"  删除了 {before - after} 行无效数据")

        # 去重
        df = df.drop_duplicates(subset=['datetime'])
        print(f"  去重后数据: {len(df)} 行")

        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)

        clean_time = time_module.time() - clean_start
        print(f"  清洗完成，用时: {clean_time:.2f}秒")
        print(f"  日期范围: {df['datetime'].min()} 到 {df['datetime'].max()}")

        # 步骤3: 筛选交易时间（9:30-15:00）
        print("\n步骤3: 筛选交易时间")
        print("=" * 80)

        # 提取时间部分
        df['time_only'] = df['datetime'].dt.time

        # 定义交易时间范围
        market_start = time(9, 30)
        market_end = time(15, 0)

        # 筛选交易时间内的数据
        before = len(df)
        df = df[(df['time_only'] >= market_start) & (df['time_only'] <= market_end)]
        after = len(df)
        print(f"  筛选后: {after} 行 (过滤掉 {before - after} 行)")

        # 添加日期列
        df['date'] = df['datetime'].dt.date

        if len(df) == 0:
            print("错误：没有符合条件的数据")
            return None

        # 步骤4: 生成各周期K线
        print("\n步骤4: 生成多周期K线数据")
        print("=" * 80)

        results = {}

        # 生成5分钟K线
        results['5min'] = process_5min_ohlc(df)

        # 生成30分钟K线
        results['30min'] = process_30min_ohlc(df)

        # 生成日K线
        results['daily'] = process_daily_ohlc(df)

        # 生成周K线
        results['weekly'] = process_weekly_ohlc(df)

        # 生成月K线
        results['monthly'] = process_monthly_ohlc(df)

        total_time = time_module.time() - total_start

        # 步骤5: 生成统计报告
        generate_statistics_report(results)

        # 步骤6: 显示示例数据
        print("\n" + "=" * 80)
        print("各周期K线示例数据（前5行）:")
        print("=" * 80)

        for period_name, result_df in results.items():
            if len(result_df) > 0:
                print(f"\n{period_name}K线:")
                print(result_df.head(5).to_string(index=False))

        # 步骤7: 保存结果
        save_all_periods(results, output_dir="multi_period_output")

        # 总用时
        print("\n" + "=" * 80)
        print(f"处理完成！总用时: {total_time:.2f}秒")
        print("=" * 80)

        # 输出结果摘要
        print(f"\n结果摘要:")
        for period_name, result_df in results.items():
            if len(result_df) > 0:
                print(f"  - {period_name}K线: {len(result_df):,} 条")
        print(f"  - 输出目录: multi_period_output/")
        print(f"  - 输出文件:")
        print(f"    - 各周期独立CSV文件")
        print(f"    - all_periods_ohlc.xlsx（合并Excel）")
        print(f"  - 核心列: 日期、时间段、开盘、上25%分位点、下25%分位点、收盘")

        return results

    except Exception as e:
        print(f"\n处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()