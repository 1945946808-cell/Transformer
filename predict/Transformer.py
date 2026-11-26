import copy
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# sin/cos position
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # size = [batch, L, d_model]


# transformer - 编码器和解码器层数均为1，注意力头维度32
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.input_embedding = nn.Linear(input_size, 32)
        self.input_position = PositionalEncoding(32)
        self.output_embedding = nn.Linear(4, 32)
        self.output_position = PositionalEncoding(32)
        self.input_dropout = nn.Dropout(0.1)
        self.output_dropout = nn.Dropout(0.1)
        # 编码器和解码器层数均设置为1，注意力头维度32
        self.transformer = nn.Transformer(d_model=32, nhead=1,
                                          num_encoder_layers=1, num_decoder_layers=1,
                                          dim_feedforward=32, batch_first=True)
        self.predictor = nn.Linear(32, 4)

    def forward(self, x, tgt_label):
        src = self.input_embedding(x) + self.input_position(x)
        tgt = self.output_embedding(tgt_label) + self.output_position(tgt_label)
        r_out = self.transformer(src, tgt)
        r_out = r_out[:, -1, :]
        r_out = self.predictor(r_out)
        return r_out[:, 0], r_out[:, 1], r_out[:, 2], r_out[:, 3]


def preprocess_normohlc_ohlc2ohlc(csv_file, val_percent=0.8):
    # 自动查找文件的多个可能位置
    possible_paths = [
        csv_file,  # 原始路径
        os.path.join('data', csv_file),  # 在data文件夹中
        os.path.join('原始所有', csv_file),  # 在原始所有文件夹中
        os.path.join('原油期货', csv_file),  # 在原油期货文件夹中
        os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file),  # 相对于当前脚本
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', csv_file),
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path is None:
        # 列出当前目录结构帮助调试
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        print("目录内容:", os.listdir(current_dir))
        raise FileNotFoundError(f"无法找到数据文件。尝试的路径: {possible_paths}")

    print(f"使用数据文件: {found_path}")
    data = pd.read_csv(found_path)

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
    # 每个特征使用自己的均值和标准差
    for i in range(input_data.shape[1]):
        input_data[:, i] = (input_data[:, i] - np.mean(input_data[:, i])) / np.std(input_data[:, i])

    gt_data = np.array(gt_data)
    gt_mean = np.mean(gt_data, axis=0)
    gt_std = np.std(gt_data, axis=0)
    gt_data = (gt_data - gt_mean) / gt_std

    return input_data, gt_data, train_index, val_index, gt_mean, gt_std


def calculate_all_metrics(pred_values, gt_values):
    """
    计算所有指标：MAPE, MAE, RMSE, AR, UDRATE
    与LSTM完全相同的计算方式
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
    ori_std = gt_std
    ori_mean = gt_mean
    gt_std_tensor = torch.from_numpy(gt_std.astype(np.float32)).to(device)
    gt_mean_tensor = torch.from_numpy(gt_mean.astype(np.float32)).to(device)

    # 初始学习率已经是1e-3，不需要再除以10
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
                open_pred, high_pred, low_pred, close_pred = net(inputs, tgt_label)
                pred = torch.cat([open_pred, high_pred, low_pred, close_pred]).squeeze()
                train_loss = loss(pred, gt_label)
                total_loss += train_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        if flip_input_data is not None:
            for index in train_index:
                if index < total_length - split:
                    inputs = flip_input_data[index:index + split]
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
                    open_pred, high_pred, low_pred, close_pred = net(inputs, tgt_label)
                    pred = torch.cat([open_pred, high_pred, low_pred, close_pred]).squeeze()

                    train_loss = loss(pred, gt_label)
                    total_loss += train_loss.item()
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

        # 验证
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
                df.to_csv('transformer_pred_train_Daily.csv', index=False)
                print(f'epoch:{e}, best MAE: {best_mae:.6f}')

        random.shuffle(train_index)

        # 学习率调整 - 使用scheduler进行恒定衰减
        scheduler.step()

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'epoch:{e}, loss: {total_loss / len(train_index):.6f}, lr: {current_lr:.6f}')

        if (e + 1) % save_epoch == 0 and save_path is not None:
            torch.save(net.state_dict(), os.path.join(save_path, f'{e + 1}.pth'))

    return pred_values, gt_values


def eval(input_data, gt_data, val_index, gt_mean, gt_std, net, device, split=10):
    net = net.to(device)
    net.eval()

    pred_values = []
    gt_values = []
    total_length = len(input_data)
    gt_std_tensor = torch.from_numpy(gt_std.astype(np.float32)).to(device)
    gt_mean_tensor = torch.from_numpy(gt_mean.astype(np.float32)).to(device)

    for index in val_index:
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
            y1, y2, y3, y4 = net(inputs, tgt_label)
            pred = torch.cat([y1, y2, y3, y4]).squeeze()
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
    net = Model(input_size)

    # 使用Adam优化器，初始学习率1e-3，动量0.9，权重衰减1e-5
    lr = 1e-3
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    split = 5
    epoch = 500
    val_percent = 0.8

    # 学习率调度器：每100个epoch下降0.1倍
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    save_epoch = 50
    save_path = 'Transformer_epoch{}_lr{}_Daily'.format(epoch, lr)

    is_train = False  # 第一次运行设为True进行训练，第二次设为False进行测试

    try:
        input_data, gt_data, train_index, val_index, gt_mean, gt_std = preprocess_normohlc_ohlc2ohlc(
            csv_file_name, val_percent)

        print(f"数据形状: input={input_data.shape}, target={gt_data.shape}")
        print(f"训练样本数: {len(train_index)}, 验证样本数: {len(val_index)}")

        if is_train:
            print("\n====== 开始训练 Transformer 模型 ======")
            print(f"优化器: Adam, 初始lr={lr}, 动量=0.9, 权重衰减=1e-5")
            print(f"学习率调度: 每100epoch下降0.1倍")
            print(f"网络预热: 10个epoch")

            # 先进行10个epoch的预热训练（使用较低学习率？这里保持原学习率）
            # 根据要求，预热10个epoch，然后进行500个epoch训练（总共510？还是500包含预热？）
            # 这里理解为总共500个epoch，包含前10个预热

            pred_values, gt_values = train(input_data, None, gt_data, copy.deepcopy(train_index), val_index, gt_mean,
                                           gt_std, net, loss, optimizer, scheduler, device, epoch, split, save_path,
                                           save_epoch)
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
                print("\n====== Transformer Method Test Results ======")
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
                df.to_csv('transformer_pred_test_Daily.csv', index=False)
            else:
                print(f"\n错误: 模型文件 {model_path} 不存在！")
                print("请先设置 is_train = True 运行训练，然后再设置 is_train = False 进行测试。")
                exit()

        # 绘图
        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.plot(range(len(pred_values[:, 0])), pred_values[:, 0], label='pred')
        plt.plot(range(len(gt_values[:, 0])), gt_values[:, 0], label='gt', linestyle='dashed')
        plt.title('open')
        plt.legend()

        plt.subplot(222)
        plt.plot(range(len(pred_values[:, 1])), pred_values[:, 1], label='pred')
        plt.plot(range(len(gt_values[:, 1])), gt_values[:, 1], label='gt', linestyle='dashed')
        plt.title('high')
        plt.legend()

        plt.subplot(223)
        plt.plot(range(len(pred_values[:, 2])), pred_values[:, 2], label='pred')
        plt.plot(range(len(gt_values[:, 2])), gt_values[:, 2], label='gt', linestyle='dashed')
        plt.title('low')
        plt.legend()

        plt.subplot( 224)
        plt.plot(range(len(pred_values[:, 3])), pred_values[:, 3], label='pred')
        plt.plot(range(len(gt_values[:, 3])), gt_values[:, 3], label='gt', linestyle='dashed')
        plt.title('close')
        plt.legend()

        plt.tight_layout()
        plt.savefig('transformer_prediction_results.png', dpi=300)
        plt.show()

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据文件存在，或修改 csv_file_name 变量为正确的文件路径")
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback

        traceback.print_exc()