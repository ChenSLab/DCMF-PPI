import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

'''

这个是自定义的GAT层加入了边的特征

'''


class GATWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, heads, concat=True, dropout=0.6):
        super(GATWithEdgeFeatures, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout  # Dropout 比例

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)  # 节点特征线性变换
        self.lin_edge = nn.Linear(1, heads * out_channels, bias=False)  # 边特征线性变换
        self.att = nn.Parameter(torch.Tensor(1, heads, 3 * out_channels))  # 注意力参数

        self.reset_parameters()

        # 定义 Dropout 层
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x).view(-1, self.heads, self.out_channels)  # 节点特征变换 (N, heads, out_channels)
        edge_attr = self.lin_edge(edge_attr.view(-1, 1)).view(-1, self.heads, self.out_channels)  # 边特征变换

        row, col = edge_index
        x_i = x[row]  # 源节点特征 (E, heads, out_channels)
        x_j = x[col]  # 目标节点特征 (E, heads, out_channels)

        # 拼接节点和边特征
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)  # (E, heads, 3 * out_channels)

        # 注意力权重计算
        alpha = (self.att * z).sum(dim=-1)  # 计算注意力分数 (E, heads)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # 对注意力权重进行 dropout
        alpha = self.dropout_layer(alpha)
        alpha = softmax(alpha, row)  # 对每个节点的出边进行归一化

        # 加权聚合节点特征
        alpha = self.dropout_layer(alpha)  # 对 alpha 再次应用 dropout
        out = alpha.unsqueeze(-1) * x_j  # (E, heads, out_channels)
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))  # 聚合邻居信息 (N, heads, out_channels)

        return self.update(out)

    def update(self, aggr_out):
        # 聚合多头结果
        if self.concat:
            # 拼接所有头 (N, heads * out_channels)
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            # 平均所有头 (N, out_channels)
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out


def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if len(m.shape) >= 2:  # 只对维度大于等于2的参数使用xavier初始化
            nn.init.xavier_uniform_(m)
        else:
            nn.init.uniform_(m)


'''
这种设计的优势(使用LayernNor和BatchNor)：

更高效的特征学习：
初始阶段使用BN有助于特征的标准化和学习
最终阶段使用LN确保输出的稳定性
计算效率：
避免了重复归一化带来的额外计算开销
每个阶段只使用最适合的归一化方法
更好的训练稳定性：
正则化效果：
BN在特征提取阶段帮助减少内部协变量偏移
LN在输出阶段提供样本级的归一化，使得预测更稳定


'''


class GATForPortT5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads,
                 dropout_rate=0.1):  # 这里input：1024，hidden：512，output：256
        super().__init__()

        # GAT layers
        self.gat1 = GATWithEdgeFeatures(input_dim, input_dim // heads, heads=heads, )
        self.gat2 = GATWithEdgeFeatures(hidden_dim, hidden_dim // heads, heads=heads, )
        self.gat3 = GATWithEdgeFeatures(output_dim, output_dim, heads=1, )

        # Normalization layers
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.ln_final = nn.LayerNorm(output_dim)

        # Linear layers
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # 残差连接的投影层
        self.skip1 = nn.Linear(input_dim, input_dim)
        self.skip2 = nn.Linear(input_dim, hidden_dim)
        self.skip3 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weight(m)

    def forward(self, x, edge_index, edge_attr):
        # First block
        identity = x
        x = self.linear1(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.skip1(identity)

        # Second block
        identity = x
        x = self.linear2(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + self.skip2(identity)

        # Final block
        identity = x
        x = self.linear3(x)
        x = self.gat3(x, edge_index, edge_attr)
        x = self.ln_final(x)
        x = self.relu(x)
        x = x + self.skip3(identity)

        return x


