import copy
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import numpy as np
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
        return self.pe[:, :x.size(1)]


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
    possible_paths = [
        csv_file,
        os.path.join('data', csv_file),
        os.path.join('原始所有', csv_file),
        os.path.join('原油期货', csv_file),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_file),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', csv_file),
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path is None:
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
    for i in range(input_data.shape[1]):
        input_data[:, i] = (input_data[:, i] - np.mean(input_data[:, i])) / np.std(input_data[:, i])

    gt_data = np.array(gt_data)
    gt_mean = np.mean(gt_data, axis=0)
    gt_std = np.std(gt_data, axis=0)
    gt_data = (gt_data - gt_mean) / gt_std

    return input_data, gt_data, train_index, val_index, gt_mean, gt_std, data


def train(input_data, flip_input_data, gt_data, train_index, val_index, gt_mean, gt_std, net, loss, optimizer,
          scheduler, device, epoch, split=10, save_path=None, save_epoch=100):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    net = net.to(device)
    net.train()
    loss = loss.to(device)
    total_length = len(input_data)
    ori_std = gt_std
    ori_mean = gt_mean

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

                    open_pred, high_pred, low_pred, close_pred = net(inputs, tgt_label)
                    pred = torch.cat([open_pred, high_pred, low_pred, close_pred]).squeeze()

                    train_loss = loss(pred, gt_label)
                    total_loss += train_loss.item()
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

        net.eval()
        pred_values, gt_values = eval(input_data, gt_data, val_index, ori_mean, ori_std, net, device, split)

        pred_flat = pred_values.flatten()
        gt_flat = gt_values.flatten()
        current_mae = np.mean(np.abs(pred_flat - gt_flat))

        if current_mae < best_mae:
            best_mae = current_mae
            if save_path is not None:
                torch.save(net.state_dict(), os.path.join(save_path, 'best.pth'))

        random.shuffle(train_index)
        scheduler.step()
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

            y1, y2, y3, y4 = net(inputs, tgt_label)
            pred = torch.cat([y1, y2, y3, y4]).squeeze()
            pred = pred * gt_std_tensor + gt_mean_tensor
            gt_label = gt_label * gt_std_tensor + gt_mean_tensor
            pred_values.append(pred.squeeze().detach().cpu().numpy())
            gt_values.append(gt_label.squeeze().detach().cpu().numpy())

    pred_values = np.array(pred_values)
    gt_values = np.array(gt_values)

    return pred_values, gt_values


def get_val_dates(original_data, val_index, split=5):
    """获取验证集对应的日期"""
    # 验证集的预测点对应 index + split 的位置
    val_dates = []
    total_length = len(original_data)

    for index in val_index:
        if index < total_length - split:
            pred_index = index + split
            if pred_index < len(original_data):
                val_dates.append(original_data.iloc[pred_index]['Data'])

    return val_dates


if __name__ == '__main__':
    input_size = 4
    csv_file_name = 'MA.CZ_data/daily_ohlc_result.csv'
    loss = MSELoss()
    net = Model(input_size)

    lr = 1e-3
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    split = 5
    epoch = 500
    val_percent = 0.8

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    save_epoch = 50
    save_path = 'Transformer_epoch{}_lr{}_Daily'.format(epoch, lr)

    is_train = False  # 第一次运行设为True进行训练，第二次设为False进行测试

    try:
        # 数据预处理，同时获取原始数据（包含日期）
        input_data, gt_data, train_index, val_index, gt_mean, gt_std, original_data = preprocess_normohlc_ohlc2ohlc(
            csv_file_name, val_percent)

        print(f"数据形状: input={input_data.shape}, target={gt_data.shape}")
        print(f"训练样本数: {len(train_index)}, 验证样本数: {len(val_index)}")

        if is_train:
            print("\n====== 开始训练 Transformer 模型 ======")
            print(f"优化器: Adam, 初始lr={lr}, 动量=0.9, 权重衰减=1e-5")
            print(f"学习率调度: 每100epoch下降0.1倍")

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

                # 获取验证集对应的日期
                val_dates = get_val_dates(original_data, val_index, split)

                # 创建结果DataFrame
                result_data = []
                for i in range(len(pred_values)):
                    row = {
                        '日期': val_dates[i] if i < len(val_dates) else f'样本_{i + 1}',
                        '预测_open': pred_values[i][0],
                        '预测_high': pred_values[i][1],
                        '预测_low': pred_values[i][2],
                        '预测_close': pred_values[i][3],
                        '真实_open': gt_values[i][0],
                        '真实_high': gt_values[i][1],
                        '真实_low': gt_values[i][2],
                        '真实_close': gt_values[i][3]
                    }
                    result_data.append(row)

                df_result = pd.DataFrame(result_data)

                # 输出到控制台
                print("\n====== Transformer 预测结果 ======")
                print(df_result.to_string(index=False))

                # 保存到CSV文件
                df_result.to_csv('transformer_prediction_results.csv', index=False, encoding='utf-8-sig')
                print(f"\n结果已保存到: transformer_prediction_results.csv")
                print(f"共 {len(pred_values)} 条预测结果")

            else:
                print(f"\n错误: 模型文件 {model_path} 不存在！")
                print("请先设置 is_train = True 运行训练，然后再设置 is_train = False 进行测试。")
                exit()

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据文件存在，或修改 csv_file_name 变量为正确的文件路径")
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback

        traceback.print_exc()