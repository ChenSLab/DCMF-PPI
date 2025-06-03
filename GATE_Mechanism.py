import torch
from torch import nn
import torch.nn.functional as F

'''

这个是实现两个分支的lamda的λ计算，控制两个路的分支，然后后续可以进行拼接

'''


class GATE_Mechanism_Branch(nn.Module):
    def __init__(self, hidden_dim_Wavelet, hidden_dim_gat, output_dim):
        super(GATE_Mechanism_Branch, self).__init__()

        # 直接定义 Tensor（不使用 nn.Parameter）
        self.W_1 = nn.Linear(hidden_dim_Wavelet + hidden_dim_gat, output_dim)
        self.W_2 = nn.Linear(hidden_dim_gat, output_dim)

    def forward(self, lstm_features, gat_features):
        # 拼接 LSTM 和 GAT 特征
        concatenated_features = torch.cat([lstm_features, gat_features],
                                          dim=1)  # (batch_size, hidden_dim_lstm + hidden_dim_gat)

        # 通过 W_1 矩阵进行线性变换
        transformed_concatenated_feature = self.W_1(concatenated_features)  # (batch_size, output_dim)

        # 填充 LSTM 输出，使其列数变为 256，后续可以进行Hadamard
        padding_size = gat_features.size(1) - lstm_features.size(1)  # 256 - 68 = 188
        lstm_padded = torch.cat(
            [lstm_features, torch.zeros(gat_features.size(0), padding_size, device=gat_features.device)], dim=1)

        # LSTM 和 GAT 特征 Hadamard 乘积
        hadamard_feature = gat_features * lstm_padded
        # 通过 W_2 矩阵进行线性变换
        transformed_hadamard_feature = self.W_2(hadamard_feature)

        # 求出gate参数λ
        lamda = F.sigmoid((transformed_concatenated_feature + transformed_hadamard_feature) / 2)

        return lamda


'''
对于VGAE的gate机制,利用统计量来得到

'''


class GATE_Mechanism_VGAE(nn.Module):
    def __init__(self, separate_number, hidden_dim=1):
        super(GATE_Mechanism_VGAE, self).__init__()
        self.separate_number = separate_number
        # 添加线性层，将统计向量映射到一个标量
        self.linear = nn.Linear(separate_number, hidden_dim, bias=False)

    # histogram直方图的思想实现，beat的参数
    def get_histogram(self, result_matrix):
        # 将tensor转到CPU并转为numpy进行统计计算
        if result_matrix.is_cuda:
            result_matrix = result_matrix.cpu()
        result_matrix = result_matrix.detach()

        # 找出最大值和最小值
        min_val = torch.min(result_matrix)
        max_val = torch.max(result_matrix)

        # 计算每个区间的范围
        interval = (max_val - min_val) / self.separate_number

        # 创建区间边界
        bins = [min_val.item() + i * interval.item() for i in range(self.separate_number + 1)]

        # 统计每个区间的数量
        hist_counts = torch.zeros(self.separate_number, device=result_matrix.device)

        # 遍历result_matrix中的每个元素，统计落在各个区间的数量
        for i in range(len(bins) - 1):
            # 对于最后一个区间，包含右端点
            if i == len(bins) - 2:
                count = torch.sum((result_matrix >= bins[i]) & (result_matrix <= bins[i + 1]))
            else:
                count = torch.sum((result_matrix >= bins[i]) & (result_matrix < bins[i + 1]))
            hist_counts[i] = count

        # 应用sigmoid函数,这里由于统计两肯定很大，进行降低到0-1之间
        hist_counts_sigmoid = torch.sigmoid(hist_counts)

        return hist_counts_sigmoid

    def forward(self, current_feature, previous_feature):
        # 检查维度是否相同
        if current_feature.shape != previous_feature.shape:
            min_cols = min(current_feature.size(1), previous_feature.size(1))

            # 截断较长的矩阵
            current_feature = current_feature[:, :min_cols]  # 截断matrix1
            previous_feature = previous_feature[:, :min_cols]

        # 左乘
        result_matrix_left = torch.matmul(current_feature, previous_feature.transpose(0, 1))
        hist_left = self.get_histogram(result_matrix_left)

        # 右乘
        result_matrix_right = torch.matmul(previous_feature, current_feature.transpose(0, 1))
        hist_right = self.get_histogram(result_matrix_right)

        # 求平均
        beta = ((hist_left + hist_right) / 2).to(current_feature.device)
        # 通过线性层得到最终输出
        beta = self.linear(beta)

        beta = F.sigmoid(beta)

        return beta


'''
处理 两个分支维度不匹配的分支,
'''


class change_dim_wavelet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(change_dim_wavelet, self).__init__()
        self.linear_transform = nn.Linear(input_dim, output_dim)  # 添加到 model 中
        # 其他层的初始化

    def forward(self, wavelet_feature):
        wavelet_feature = self.linear_transform(wavelet_feature)  # 在 forward 中调用
        # 继续你的 forward 操作
        return wavelet_feature


