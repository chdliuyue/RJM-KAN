import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, r2_score


# 将离散数据依次映射成数字
def cats2ints(Q_df):
    UNIQUE_CATS = sorted(list(set(Q_df.values.reshape(-1))))
    cat2index = {}
    for i in range(len(UNIQUE_CATS)):
        cat2index[UNIQUE_CATS[i]] = i

    return cat2index


# 将string转成int
def cats2ints_transform(Q_df, cat2index):
    Q_int = []
    for obs in Q_df.values:
        input_i = [cat2index[cat] for cat in obs]
        Q_int.append(input_i)

    return np.array(Q_int)


# 将X, Q, Y放在一起实现dataloader可以迭代
def create_dataset(x_data, q_data, y_data):
    class MyDataset(Dataset):
        def __init__(self):
            self.x_data = x_data
            self.q_data = q_data
            self.y_data = y_data

        def __len__(self):
            # 返回数据数量
            return len(self.x_data)

        def __getitem__(self, idx):
            # 返回一个数据样本和对应的索引
            return self.x_data[idx], self.q_data[idx], self.y_data[idx]

    return MyDataset()


class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: 形状为 (n, 3), 表示模型对每个样本的3类得分
        # target: 形状为 (n,), 表示每个样本的真实类别标签 (0, 1 或 2)
        # 使用 log_softmax 计算每个类别的对数概率
        log_probs = F.log_softmax(input, dim=1)
        # 计算每个样本对应的真实标签的对数概率
        target_log_probs = log_probs[torch.arange(target.size(0)), target]
        # 损失为负的平均对数概率
        loss = -target_log_probs.mean()

        return loss


class JeffriesMatusitaLoss(nn.Module):
    def __init__(self):
        super(JeffriesMatusitaLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: 模型输出的 logits（未经过 softmax 处理），形状为 (n, C)
        target: 真实标签，大小为 (n, C)，通常为 one-hot 编码的形式
        """
        # 使用 softmax 将输入转化为概率分布
        probs = torch.softmax(input, dim=1)
        # 计算平方根的预测概率
        sqrt_probs = torch.sqrt(probs + 1e-8)
        # 选择每个样本的真实类别概率
        selected_probs = sqrt_probs[torch.arange(target.size(0)), target]
        # 计算 Jeffries-Matusita 损失
        loss = 1 - selected_probs
        return loss.mean()  # 对每个样本求和，然后取平均