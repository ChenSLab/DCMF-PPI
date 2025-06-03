from abc import ABC
import torch
from torch import nn
from torch_geometric.nn import VGAE
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import glorot, zeros

'''
这个是结合节点特征和边特征，进行融合卷积的卷积层
'''


class GCNWithEdgeFeatures(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, edge_in_channels=1, edge_out_channels=64):
        super(GCNWithEdgeFeatures, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        # 节点特征更新权重
        self.node_weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        # 边特征更新权重
        self.edge_weight = torch.nn.Parameter(torch.Tensor(edge_in_channels, edge_out_channels))
        # 节点-边联合权重
        self.node_edge_weight = torch.nn.Parameter(torch.Tensor(out_channels + edge_out_channels, out_channels))

        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.node_weight)
        glorot(self.edge_weight)
        glorot(self.node_edge_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, edge_in_channels]

        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 为自环添加边特征 (假设自环边特征为零向量)
        self_loop_attr = torch.zeros(x.size(0), self.edge_in_channels, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # 归一化系数,这个就是度矩阵的归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        # 避免除零错误
        deg = deg + 1e-6  # 给度数一个小的正数，避免出现零度的节点
        deg_inv_sqrt = deg.pow(-0.5)  # 计算倒数平方根
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        # x_j: [E, in_channels]
        # edge_attr: [E, edge_in_channels]
        # norm: [E]

        # 节点特征变换
        node_message = torch.matmul(x_j, self.node_weight)  # [E, out_channels]
        # 边特征变换
        edge_message = torch.matmul(edge_attr, self.edge_weight)  # [E, edge_out_channels]

        # 融合节点特征和边特征
        combined_message = torch.cat([node_message, edge_message], dim=1)  # [E, out_channels + edge_out_channels]
        combined_message = torch.matmul(combined_message, self.node_edge_weight)  # [E, out_channels]

        # 应用归一化
        return norm.view(-1, 1) * combined_message

    def update(self, aggr_out):
        return aggr_out + self.bias


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super(Encoder, self).__init__()
        # 图卷积层
        self.conv1 = GCNWithEdgeFeatures(in_channels, in_channels)
        self.conv2 = GCNWithEdgeFeatures(hidden_channels, hidden_channels)
        self.conv3_mean = GCNWithEdgeFeatures(out_channels, out_channels)
        self.conv3_logstd = GCNWithEdgeFeatures(out_channels, out_channels)

        # 线性层
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

        # 残差转换层
        self.proj_identity_to_hidden = nn.Linear(in_channels, hidden_channels)
        self.proj_hidden_to_out = nn.Linear(hidden_channels, out_channels)
        # 激活函数
        self.relu = torch.nn.ReLU()

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # BatchNorm
        self.batch_norm1 = torch.nn.BatchNorm1d(in_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_mean = torch.nn.BatchNorm1d(out_channels)
        self.batch_norm_logstd = torch.nn.BatchNorm1d(out_channels)

        # Gate
        # 目前不需要了
        # self.Gate = GATE_Mechanism_VGAE(separate_number=20)

    def forward(self, x, edge_index, edge_attr):
        identity = x

        # First block
        x1 = self.conv1(x, edge_index, edge_attr)  # 256 256
        x1 = self.batch_norm1(x1)  # 添加 BatchNorm
        x1 = self.relu(x1)
        x1 = self.dropout(x1)  # 添加 Dropout
        x1 = x1 + identity  # 这里利用initial和identity残差

        # Second block
        x2 = self.linear1(x1)  # 256 125
        x2 = self.conv2(x2, edge_index, edge_attr)  # 125 125
        x2 = self.batch_norm2(x2)  # 添加 BatchNorm
        x2 = self.relu(x2)
        x2 = self.dropout(x2)  # 添加 Dropout
        x2 = x2 + self.proj_identity_to_hidden(x1)

        # Third block
        x3 = self.linear2(x2)  # 125 64

        logstd = self.conv3_logstd(x3, edge_index, edge_attr)
        logstd = self.batch_norm_logstd(logstd)

        mean = self.conv3_mean(x3, edge_index, edge_attr)
        mean = self.batch_norm_mean(mean)

        mean = mean + self.proj_hidden_to_out(x2)
        logstd = logstd + self.batch_norm_logstd(logstd)
        return mean, logstd


'''
解码器，其中内嵌全连接层进行ppi的预测

这里Decoder为前反馈层进行PPI作用的预测，需要用到edge_index索引进行拼接两个具有相互作用的蛋白

'''


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # 输入是两个蛋白质特征的拼接，所以维度是2*gat_output_dim
            nn.Linear(2 * input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.transform_edge = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
        )

    def forward(self, z, edge_index):
        # 获取边连接的节点潜在表示
        src_z = z[edge_index[0]]
        dst_z = z[edge_index[1]]

        # 拼接特征
        edge_features = torch.cat([src_z, dst_z], dim=1)

        # 预测相互作用概率
        return self.decoder(edge_features).squeeze()

    def decode_all(self, z):
        # 用于推理时预测所有可能的边
        z = self.transform_edge(z)
        prob_adj = torch.matmul(z, z.t())
        return torch.sigmoid(prob_adj)


# 创建VGAE模型
class PPI_VGAE(VGAE):
    def __init__(self, encoder, decoder):
        super(PPI_VGAE, self).__init__(encoder, decoder)

    def reparameterize(self, z_mean, z_logstd):
        # 基于均值和对数标准差采样潜在表示
        std = torch.exp(0.5 * z_logstd)  # 标准差是logstd的指数
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        z = z_mean + eps * std  # 重参数化公式
        return z

    def encode(self, x, edge_index, edge_attr):
        z_mean, z_logstd = self.encoder(x, edge_index, edge_attr)
        z = self.reparameterize(z_mean, z_logstd)
        return z, z_mean, z_logstd

    def decode_all(self, z):
        return self.decoder.decode_all(z)


# 这个是消融实验不需要VGAE模型进行。

class MLP_Ablation_experiment_PPINet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(MLP_Ablation_experiment_PPINet, self).__init__()
        self.MLP = nn.Sequential(
            # 输入是两个蛋白质特征的拼接，所以维度是2*gat_output_dim
            nn.Linear(2 * input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.encoder = Encoder(256, 125, 64)

    def forward(self, x, edge_index, edge_attr):
        z, _ = self.encoder(x, edge_index, edge_attr)

        # 获取边连接的节点潜在表示
        src_z = z[edge_index[0]]
        dst_z = z[edge_index[1]]

        # 拼接特征
        edge_features = torch.cat([src_z, dst_z], dim=1)
        return self.MLP(edge_features).squeeze()

# def train():
#     model.train()
#     optimizer.zero_grad()
#
#     '''
#     这里关于Z是怎么来的解释下,
#     首先对于encoder而言你最后的输出就是潜在的特征的维度，也就是说对于每个节点而言
#     都会输出均值和方差都是你设置的潜在的特征的维度，然后通过重构，这样每个节点的特征就为你潜在的维度
#     然后有多个节点，这个矩阵就表示为 matrix ： L * N（设置的潜在维度，其实就是你encoder层的输出）
#     '''
#     z = model.encode(train_data.x, train_data.edge_index)  # 使用训练边
#     # 这里这个函数就会调用decode的方法，利用生成的边于真实的边进行比较，也就是VAE的一个loss
#     loss = model.recon_loss(z, train_data.edge_index)
#     # 这个KLloss,就是保证conv2_mean，conv2_logstd这两个层的输出的概率要于正态分布接近，这也是loss的第二个损失
#     loss += (1 / data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#     return loss
