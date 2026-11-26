import copy
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import random
import math


class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=16, num_layers=2, batch_first=True)
        self.out_1 = nn.Linear(16, 1)
        self.out_2 = nn.Linear(16, 1)
        self.out_3 = nn.Linear(16, 1)
        self.out_4 = nn.Linear(16, 1)

    def forward(self, x):
        # GRU的输出与LSTM略有不同，只返回输出和隐藏状态
        r_out, h_n = self.rnn(x, None)  # GRU只有hidden state，没有cell state
        y1 = self.out_1(r_out[:, -1])
        y2 = self.out_2(r_out[:, -1])
        y3 = self.out_3(r_out[:, -1])
        y4 = self.out_4(r_out[:, -1])
        return y1, y2, y3, y4


def preprocess_normohlc_ohlc2ohlc(csv_file, val_percent=0.8):
    data = pd.read_csv(csv_file)
    gt_labels = ['open', 'high', 'low', 'close']
    input_labels = ['open', 'high', 'low', 'close']

    input_data = []
    gt_data = []
    close_data = getattr(data, 'close')
    total_length = len(close_data)
    train_length = round(total_length * val_percent)

    for i in range(total_length):
        input_i = []
        gt_i = []
        for label in input_labels:
            values = getattr(data, label)
            input_i.append(values[i])
        for label in gt_labels:
            values = getattr(data, label)
            gt_i.append(values[i])
        input_data.append(input_i)
        gt_data.append(gt_i)

    total_index = list(range(total_length))
    train_index = total_index[:train_length]
    val_index = total_index[train_length:]

    input_data = np.array(input_data)
    # 标准化输入数据
    for i in range(4):
        input_data[:, i] = (input_data[:, i] - np.mean(input_data[:, i])) / np.std(input_data[:, i])

    gt_data = np.array(gt_data)
    gt_mean = np.mean(gt_data, axis=0)
    gt_std = np.std(gt_data, axis=0)
    gt_data = (gt_data - gt_mean) / gt_std

    return input_data, gt_data, train_index, val_index, gt_mean, gt_std


def calculate_all_metrics(pred_values, gt_values):
    """
    计算所有指标：MAPE, MAE, RMSE, AR, UDRATE
    混合所有价格类型计算
    """
    eps = 1e-10

    # 将4列数据展平为一维数组，混合所有价格类型
    pred_flat = pred_values.flatten()
    gt_flat = gt_values.flatten()

    n_samples = len(pred_values)
    n_total = len(pred_flat)

    # MAPE - 混合所有价格类型计算
    mape_values = np.abs(pred_flat - gt_flat) / (np.abs(gt_flat) + eps)
    MAPE = np.mean(mape_values)

    # MAE - 混合所有价格类型计算
    MAE = np.mean(np.abs(pred_flat - gt_flat))

    # RMSE - 混合所有价格类型计算
    RMSE = np.sqrt(np.mean(np.power(pred_flat - gt_flat, 2)))

    # AR (区间重叠度) 和 UDRATE (方向预测准确率)
    AR = 0
    UD = 0
    total_ar = 0
    total_ud = 0

    for i in range(n_samples):
        # AR - 使用HIGH作为MAX，LOW作为MIN
        Y_u_fit = pred_values[i, 1]  # 预测HIGH
        Y_l_fit = pred_values[i, 2]  # 预测LOW
        Y_upper_test = gt_values[i, 1]  # 实际HIGH
        Y_lower_test = gt_values[i, 2]  # 实际LOW

        intersection = max(min(Y_u_fit, Y_upper_test) - max(Y_l_fit, Y_lower_test), 0)
        union = max(Y_u_fit, Y_upper_test) - min(Y_l_fit, Y_lower_test)

        if union > eps:
            AR += intersection / union
            total_ar += 1

        # UDRATE - 方向预测准确率
        pred_direction = pred_values[i, 3] - pred_values[i, 0]  # 预测涨跌 (OPEN - CLOSE)
        actual_direction = gt_values[i, 3] - gt_values[i, 0]  # 实际涨跌

        if abs(pred_direction) > eps and abs(actual_direction) > eps:
            total_ud += 1
            if pred_direction * actual_direction > 0:
                UD += 1

    finalAR = AR / total_ar if total_ar > 0 else 0
    finalUDRATE = UD / total_ud if total_ud > 0 else 0

    return {
        'MAPE': MAPE,
        'MAE': MAE,
        'RMSE': RMSE,
        'AR': finalAR,
        'UDRATE': finalUDRATE
    }


def train(input_data, flip_input_data, gt_data, train_index, val_index, gt_mean, gt_std, net, loss, optimizer,
          scheduler, device, epoch,
          split=10,
          save_path=None,
          save_epoch=100):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    net = net.to(device)
    net.train()
    loss = loss.to(device)
    total_length = len(input_data)
    ori_std = gt_std.copy()
    ori_mean = gt_mean.copy()

    # 使用 .to(device) 而不是 .cuda()
    gt_std_tensor = torch.from_numpy(gt_std.astype(np.float32)).to(device)
    gt_mean_tensor = torch.from_numpy(gt_mean.astype(np.float32)).to(device)

    optimizer.param_groups[0]['lr'] /= 10
    best_mae = 9999

    for e in range(epoch):
        total_loss = 0
        net.train()
        for index in train_index:
            if index < total_length - split:
                inputs = input_data[index:index + split]
                inputs = inputs[np.newaxis, :, :]
                gt_label = gt_data[index + split]
                tgt_label = gt_data[index:index + split]
                tgt_label = tgt_label[np.newaxis, :, :]

                tgt_label = torch.Tensor(tgt_label)
                inputs = torch.Tensor(inputs)
                gt_label = torch.Tensor(gt_label)

                inputs = inputs.to(device)
                tgt_label = tgt_label.to(device)
                gt_label = gt_label.to(device)

                # forward
                open_pred, high_pred, low_pred, close_pred = net(inputs)
                pred = torch.cat([open_pred, high_pred, low_pred, close_pred]).squeeze()

                train_loss = loss(pred, gt_label)
                total_loss += train_loss.item()

                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        # 评估
        net.eval()
        pred_values, gt_values = eval(input_data, gt_data, val_index, ori_mean, ori_std, net, device, split)

        # 计算所有指标
        metrics = calculate_all_metrics(pred_values, gt_values)
        current_mae = metrics['MAE']

        if current_mae < best_mae:
            best_mae = current_mae
            if save_path is not None:
                torch.save(net.state_dict(), os.path.join(save_path, 'best.pth'))
                save_data = {'open': [], 'high': [], 'low': [], 'close': []}
                for j in range(len(pred_values)):
                    save_data['open'].append(pred_values[j][0])
                    save_data['high'].append(pred_values[j][1])
                    save_data['low'].append(pred_values[j][2])
                    save_data['close'].append(pred_values[j][3])
                df = pd.DataFrame(save_data)
                df.to_csv('gru_pred_train_Daily.csv', index=False)
                print(f'epoch:{e}, best MAE: {best_mae:.6f}')

        # 学习率调整
        if e == 10:
            optimizer.param_groups[0]['lr'] *= 10
        if e == 100:
            optimizer.param_groups[0]['lr'] /= 10
        if e == 200:
            optimizer.param_groups[0]['lr'] /= 2
        if e == 300:
            optimizer.param_groups[0]['lr'] /= 2
        if e == 400:
            optimizer.param_groups[0]['lr'] /= 2

        print(f'epoch:{e}, loss: {total_loss / len(train_index):.6f}')

        if (e + 1) % save_epoch == 0 and save_path is not None:
            torch.save(net.state_dict(), os.path.join(save_path, f'{e + 1}.pth'))


def eval(input_data, gt_data, val_index, gt_mean, gt_std, net, device, split=10):
    """评估函数，只返回预测值和真实值"""
    net = net.to(device)
    net.eval()

    pred_values = []
    gt_values = []
    total_length = len(input_data)

    # 使用 .to(device) 而不是 .cuda()
    gt_std_tensor = torch.from_numpy(gt_std.astype(np.float32)).to(device)
    gt_mean_tensor = torch.from_numpy(gt_mean.astype(np.float32)).to(device)

    for index in val_index:
        if index < total_length - split:
            inputs = input_data[index:index + split]
            inputs = inputs[np.newaxis, :, :]
            gt_label = gt_data[index + split]

            inputs = torch.Tensor(inputs)
            gt_label = torch.Tensor(gt_label)

            inputs = inputs.to(device)
            gt_label = gt_label.to(device)

            # forward
            y1, y2, y3, y4 = net(inputs)
            pred = torch.cat([y1, y2, y3, y4]).squeeze()

            # 反标准化
            pred = pred * gt_std_tensor + gt_mean_tensor
            gt_label = gt_label * gt_std_tensor + gt_mean_tensor

            pred_values.append(pred.squeeze().detach().cpu().numpy())
            gt_values.append(gt_label.squeeze().detach().cpu().numpy())

    pred_values = np.array(pred_values)
    gt_values = np.array(gt_values)

    return pred_values, gt_values


if __name__ == '__main__':
    input_size = 4
    csv_file_name = 'MA.CZ_data/daily_ohlc_result.csv'
    loss = MSELoss()
    net = GRUModel(input_size)  # 使用GRU模型
    lr = 0.001
    optimizer = Adam(net.parameters(), lr=lr)

    # 自动选择设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    split = 5
    epoch = 500
    val_percent = 0.8

    # 创建调度器（兼容不同版本的 PyTorch）
    try:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=30, verbose=True, min_lr=1e-6)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=30, min_lr=1e-6)
        print("注意: 当前 PyTorch 版本不支持 verbose 参数")

    save_epoch = 50
    save_path = f'gru_epoch{epoch}_lr{lr}_Daily'
    is_train = False # 第一次运行设为True进行训练

    # 数据预处理
    input_data, gt_data, train_index, val_index, gt_mean, gt_std = preprocess_normohlc_ohlc2ohlc(
        csv_file_name, val_percent)

    print(f"数据形状: input={input_data.shape}, target={gt_data.shape}")
    print(f"训练样本数: {len(train_index)}, 验证样本数: {len(val_index)}")

    if is_train:
        print("\n====== 开始训练 GRU 模型 ======")
        train(input_data, None, gt_data, train_index, val_index, gt_mean, gt_std,
              net, loss, optimizer, scheduler, device, epoch, split, save_path, save_epoch)
        print("====== 训练完成 ======")
    else:
        # 测试模式 - 加载训练好的模型进行测试
        model_path = os.path.join(save_path, 'best.pth')
        if os.path.exists(model_path):
            print(f"\n====== 加载模型: {model_path} ======")
            net.load_state_dict(torch.load(model_path, map_location=device))

            # 进行预测
            pred_values, gt_values = eval(input_data, gt_data, val_index, gt_mean, gt_std, net, device, split)

            # 计算所有指标
            metrics = calculate_all_metrics(pred_values, gt_values)

            # 输出结果
            print("\n====== GRU Method Test Results ======")
            print(f"MAPE: {metrics['MAPE']:.6f}")
            print(f"MAE: {metrics['MAE']:.6f}")
            print(f"RMSE: {metrics['RMSE']:.6f}")
            print(f"AR: {metrics['AR']:.6f}")
            print(f"UDRATE: {metrics['UDRATE']:.6f}")

            # 保存预测结果
            save_data = {'open': [], 'high': [], 'low': [], 'close': []}
            for j in range(len(pred_values)):
                save_data['open'].append(pred_values[j][0])
                save_data['high'].append(pred_values[j][1])
                save_data['low'].append(pred_values[j][2])
                save_data['close'].append(pred_values[j][3])
            df = pd.DataFrame(save_data)
            df.to_csv('gru_pred_test_Daily.csv', index=False)

            # 绘制结果
            plt.figure(figsize=(12, 8))
            titles = ['Open', 'High', 'Low', 'Close']
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.plot(pred_values[:, i], label='Predicted', marker='o', markersize=3)
                plt.plot(gt_values[:, i], label='Actual', linestyle='--', marker='s', markersize=3)
                plt.title(titles[i])
                plt.xlabel('Sample Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('gru_prediction_results.png', dpi=300)
            plt.show()
        else:
            print(f"\n错误: 模型文件 {model_path} 不存在！")
            print("请先设置 is_train = True 运行训练，然后再设置 is_train = False 进行测试。")