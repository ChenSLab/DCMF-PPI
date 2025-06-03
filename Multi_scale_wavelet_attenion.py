import numpy as np
import pywt
from torch import nn
import torch.nn.functional as F
import torch


def initialize_weight(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)


'''
自己定义一个Conv1的层,具有batch—normalization和dropout
'''


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), dropout_rate=0.3,
                 if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        self.dropout_rate = dropout_rate
        # 这个是进行填充，保证输入的序列长度和输入的序列长度是一样，但是前提必须kernel_size为奇数
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.SELU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        return x


'''
first Parallel line
'''


class Parallel_line1(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Parallel_line1, self).__init__()
        self.conv0 = Conv1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(1,), )

    def forward(self, x):
        x = self.conv0(x)
        return x


'''
second Parallel line
'''


class Parallel_line2(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Parallel_line2, self).__init__()
        self.conv0_and_conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,)),
            Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,)),
        )

    def forward(self, x):
        x = self.conv0_and_conv1(x)
        return x


'''

third Parallel line
'''


class Parallel_line3(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Parallel_line3, self).__init__()
        self.conv0_and_conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,)),
            Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,)),
            Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=(5,)),
        )

    def forward(self, x):
        x = self.conv0_and_conv1(x)
        return x


'''
将多线程的进行拼接然后残差
'''


class mutil_Parallel(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(mutil_Parallel, self).__init__()
        self.line1 = Parallel_line1(in_channels, out_channels)
        self.line2 = Parallel_line2(in_channels, out_channels)
        self.line3 = Parallel_line3(in_channels, out_channels)
        self.linear = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x):
        x1 = self.line1(x)
        x2 = self.line2(x)
        x3 = self.line3(x)
        # 这里就是由于1维度卷积需要将序列长度放入前面去
        x4 = torch.cat([x1, x2, x3], dim=1).transpose(1, 2)
        return self.linear(x4) + x.transpose(1, 2)


class MultiHeadAttentionWithWavelet(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=2):
        super(MultiHeadAttentionWithWavelet, self).__init__()

        # 初始化 attn_size 和 reduced_attn_size
        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.wavelet_attn_size = (attn_size // 2) + 3 if attn_size % 2 == 0 else ((attn_size + 1) // 2) + 3
        self.scale = attn_size ** -0.5

        # 线性变换，用于生成 Q, K, V
        self.linear_q = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * attn_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        # Dropout 防止过拟合
        self.attn_dropout = nn.Dropout(dropout_rate)

    def wavelet_transform(self, v):
        # 对 V 进行小波变换，返回低频分量 cA 和高频分量 cD
        cA, cD = [], []
        for head in v:  # 针对每个注意力头分别进行小波变换
            batch_cA, batch_cD = [], []
            for seq in head:  # 针对每个序列特征进行变换
                coeffs = pywt.dwt(seq.detach().cpu().numpy(), 'db4', )  # 使用 Daubechies 小波
                batch_cA.append(torch.tensor(coeffs[0], device=v.device))
                batch_cD.append(torch.tensor(coeffs[1], device=v.device))
            cA.append(torch.stack(batch_cA))
            cD.append(torch.stack(batch_cD))
        return torch.stack(cA), torch.stack(cD)

    def forward(self, x):
        # 假设输入 x 为二维张量 (seq_len, hidden_size)
        origin_x = x
        seq_len, hidden_size = x.size(1), x.size(2)

        d_k = self.attn_size
        d_v = self.attn_size

        # 增加 batch 维度
        x = x.unsqueeze(0)  # (1, seq_len, hidden_size)
        # 生成 Q, K, V 并调整形状
        q = self.linear_q(x).view(-1, seq_len, self.head_size, d_k).transpose(1, 2)
        k = self.linear_k(x).view(-1, seq_len, self.head_size, d_k).transpose(1, 2)
        v = self.linear_v(x).view(-1, seq_len, self.head_size, d_k).transpose(1, 2)

        # Scaled Dot-Product Attention 计算
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # (1, head_size, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 对 V 进行小波变换
        # cA表示低频全局
        # cD表示高频局部
        # 这里的特征如果是奇数就要遭，应为小波变换要除以二的维度
        cA, cD = self.wavelet_transform(v)
        # 使用高频分量 cD进行注意力系数的计算
        weighted_cD = torch.matmul(attn, cD)  # (1, head_size, seq_len, reduced_attn_size)

        # 调整回三维形状并移除 batch 维度
        x = weighted_cD.transpose(1, 2).contiguous().view(-1, seq_len, self.head_size * self.wavelet_attn_size)

        assert x.size() == (origin_x.size(0), seq_len, self.head_size * self.wavelet_attn_size)  # 验证输出形状是否正确

        return x, cA.view(-1, seq_len, self.head_size * self.wavelet_attn_size)


class MPSWA(nn.Module):  # Multiple Parallel Scale Wavelet Attention
    def __init__(self, in_channels, out_channels) -> None:
        super(MPSWA, self).__init__()
        self.mutil_Parallel = mutil_Parallel(in_channels, out_channels)
        self.linear = nn.Linear(112, out_channels)
        self.AttentionWithWavelet = MultiHeadAttentionWithWavelet(in_channels, dropout_rate=0.2, head_size=4)

    def forward(self, x):
        x = self.mutil_Parallel(x.transpose(1, 2))
        identity = x
        x, cA = self.AttentionWithWavelet(x)
        x = F.relu(x)
        x = torch.cat((x, cA), dim=-1)
        x = self.linear(x)
        return identity + x


