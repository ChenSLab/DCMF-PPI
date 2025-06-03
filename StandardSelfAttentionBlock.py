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
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.line1(x)
        x2 = self.line2(x)
        x3 = self.line3(x)
        # 这里就是由于1维度卷积需要将序列长度放入前面去
        x4 = torch.cat([x1, x2, x3], dim=1).transpose(1, 2)
        x4 = self.relu(self.linear(x4))  # 后激活函数
        return x4 + x.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, head_size=2):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.scale = attn_size ** -0.5

        # Linear projections for Q, K, V
        self.linear_q = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_out = nn.Linear(head_size * attn_size, hidden_size)

        # Initialize weights
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)
        initialize_weight(self.linear_out)

        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.size()

        # For single-sample case, add batch dimension
        if batch_size == 1:
            x = x.unsqueeze(0)
            batch_size = 1

        d_k = self.attn_size

        # Project inputs to queries, keys and values
        # Shape: (batch_size, head_size, seq_len, attn_size)
        q = self.linear_q(x).view(batch_size, seq_len, self.head_size, d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.head_size, d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.head_size, d_k).transpose(1, 2)

        # Scaled dot-product attention
        q = q * self.scale
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, head_size, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, v)  # (batch_size, head_size, seq_len, attn_size)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Final linear projection
        output = self.linear_out(output)
        output = self.output_dropout(output)

        # Remove the batch dimension if it was added
        if batch_size == 1:
            output = output.squeeze(0)

        return output


class StandardSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, head_size=2):
        super(StandardSelfAttentionBlock, self).__init__()

        # Self-attention layer
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            head_size=head_size
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Apply self-attention with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Apply feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class MPSWA(nn.Module):  # Multiple Parallel Scale Wavelet Attention
    def __init__(self, in_channels, out_channels) -> None:
        super(MPSWA, self).__init__()
        self.mutil_Parallel = mutil_Parallel(in_channels, out_channels)
        self.StandardAttention = StandardSelfAttentionBlock(in_channels, dropout_rate=0.3, head_size=4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.mutil_Parallel(x.transpose(1, 2))
        identity = x
        x = self.StandardAttention(x)
        x = self.dropout(x)
        x = F.relu(x)  # 最后加 ReLU
        return identity + x  # 残差连接
