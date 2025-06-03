import ExtractedFeature_68_LSTM as Lstm68
import PortT5GCN as PortT5GCN
import GATE_Mechanism as Gate_Mechanism
import Concat83Feature as Feature83
import torch
import Multi_scale_wavelet_attenion as Multi_scale_wavelet_attenion
import generate_subgraphs as generate_subgraphs
from utils.FilePATH import load_and_get_subset_paths, Return_pdbFile_Path
from utils.Tools import create_data_batch

PortT5_input_dim = 1024
PortT5_hidden_dim = 512
PortT5_output_dim = 256
PortT5_num_layers = 8
LSTM68_input_dim = 68
LSTM68_hidden_dim = 68
LSTM68_num_layers = 2
Wavelet_input_dim = 84
Wavelet_output_dim = 84
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PortT5 module branch
Port5_Gat_model = PortT5GCN.GATForPortT5(input_dim=PortT5_input_dim,
                                         hidden_dim=PortT5_hidden_dim,
                                         output_dim=PortT5_output_dim,
                                         heads=PortT5_num_layers).to(device)
# 68feature of LSTM branch
BiLSTMAtten = Lstm68.BiLSTMAtten(input_size=LSTM68_input_dim,
                                 hidden_size=LSTM68_hidden_dim,
                                 num_layers=LSTM68_num_layers, ).to(device)

# GATE Mechanism for two branch
GATE_Mechanism_Branch = Gate_Mechanism.GATE_Mechanism_Branch(hidden_dim_Wavelet=Wavelet_input_dim,
                                                             hidden_dim_gat=PortT5_output_dim,
                                                             output_dim=1).to(device)
# change wavelet dim  as same as PortT5feature
change_dim_wavelet = Gate_Mechanism.change_dim_wavelet(input_dim=Wavelet_input_dim, output_dim=PortT5_output_dim).to(
    device=device)

# Multiple Parallel Scale Wavelet Attention
MPSWA = Multi_scale_wavelet_attenion.MPSWA(in_channels=Wavelet_input_dim,
                                           out_channels=Wavelet_output_dim,
                                           ).to(device)

'''

这个是创建一个batch的dataloader，这个dataloader里面是包含了，对于两个分支路线的
蛋白质id，和分别在两条路线的特征数据
值得注意的是，对于GAT图数据的Data是存储在一个list列表中
对于LSTM数据已经进行了张量堆叠（三维度），data数据在提取过程中就还需要进行一次for循环的取值


'''


class BuildDataloader:
    def __init__(self, dsspPath, pssmPath, sequencePath, atomPath, PortT5FeaturePath, ppiFile, coordinationPath,
                 mode='random'):
        self.feature_68 = Feature83.ReturnFeature83(
            dsspPath=dsspPath,
            pssmPath=pssmPath,
            sequencePath=sequencePath,
            atomPath=atomPath,
            coordinationPath=coordinationPath
        )
        self.PortT6Gat_dataset = torch.load(PortT5FeaturePath, weights_only=False)
        self.generate_subgraph = generate_subgraphs.build_ppi_subgraph(ppiFile, mode=mode)

    # 这个是生成整个数据的子图和每个子图的数据id
    def build_subgraph(self):
        subgraphs_data_list, subgraphs_protein_ids = self.generate_subgraph
        return subgraphs_data_list, subgraphs_protein_ids

    # 这个是将每个子图的数据节点，需要进过两个分支的数据都提取出来放入列表中
    def generate_subgraph_data_for_two_branch(self):
        subgraphs_data_list_portT5 = []
        subgraphs_data_list_wavelet = []
        _, subgraphs_protein_ids = self.build_subgraph()
        for protein_id in subgraphs_protein_ids:
            data_list, wavelet_features = generate_subgraphs.process_protein_branch_feature(protein_id,
                                                                                            self.PortT6Gat_dataset,
                                                                                            self.feature_68)
            subgraphs_data_list_portT5.append(data_list)
            subgraphs_data_list_wavelet.append(wavelet_features)
        return subgraphs_data_list_portT5, subgraphs_data_list_wavelet


'''
这个函数是输入到GAT中进行一个批次的处理~
'''


def process_gat_batch(gat_batch, model):
    """
    处理 GAT 分支，将每个图的数据单独提取特征并汇总为向量。

    参数：
    - gat_batch: batch['gat_batch']，包含多个 data 数据。
    - gat_model: 已定义的 GAT 模型，用于提取图特征。

    返回：
    - gat_features: (batch_size, hidden_size)，每个图的特征向量。
    """

    gat_features = []  # 用来存储每个图的特征向量
    for data in gat_batch:
        data = data.to(device)
        x, edge_index, edge_arr = data.x, data.edge_index, data.edge_attr
        # 将单个图的数据送入 GAT 模型
        # 假设 GAT 模型返回节点特征矩阵 (num_nodes, hidden_dim)
        node_features = model(x, edge_index, edge_arr)
        # 对节点特征进行全局池化，得到图特征向量 (hidden_dim)
        graph_feature = node_features.mean(dim=0)  # 对节点特征取平均 (1, hidden_dim)

        gat_features.append(graph_feature.unsqueeze(0))  # 将每个图的特征向量添加到列表

        # 将所有图的特征堆叠起来 (batch_size, hidden_dim)
    gat_features = torch.cat(gat_features, dim=0)
    return gat_features


'''
这个函数是输入到LSTM中进行一个batch处理
'''


def process_lstm_batch(lstm_batch, model):
    lstm_batch = lstm_batch.to(device)
    lstm_feature = model(lstm_batch)
    return lstm_feature


'''
这个函数是输入到wavelet_attention中进行一个batch处理
'''


def process_wavelet_attention_batch(feature68, model):
    feature68 = feature68.to(device)
    wavelet_feature = model(feature68)
    wavelet_feature = wavelet_feature.mean(dim=1)  # 对节点特征取平均
    return wavelet_feature


'''

这个是构建提取两个分支的合并的矩阵利用gate机制，然后利用蛋白质id最为key，放入
字典中，用于后续PPI网络节点的构建

'''


def Build_Joint_FeatureMatrix(gat_feature, wavelet_feature, model):
    lamda = model(wavelet_feature, gat_feature)
    # 使用线性层将 lstm_feature 的列数调整到 gat_feature 的列数
    wavelet_feature = change_dim_wavelet(wavelet_feature)
    # gate控制
    combined_features = lamda * gat_feature + (1 - lamda) * wavelet_feature
    return combined_features


def Temporal_protein_Gat_batch(gat_batch, model,dataset_name):
    """
       处理 GAT 分支，将每个图的数据单独提取特征并汇总为向量。

       参数：
       - gat_batch: batch['gat_batch']，包含多个 data 数据。
       - gat_model: 已定义的 GAT 模型，用于提取图特征。

       返回：
       - gat_features: (batch_size, hidden_size)，每个图的特征向量。
       """

    gat_features = []  # 用来存储每个图的特征向量
    for data in gat_batch:
        # 获取pdb文件路径
        pdbFilePath = Return_pdbFile_Path(dataset_name, 'dynamic_adj_File_'+dataset_name, data.name)
        # 当前蛋白质的不同时序的edge_index,edge_attr
        dynamicDic = torch.load(pdbFilePath, weights_only=False)
        Temporal_batch = create_data_batch(data, dynamicDic["adj_matrices"], dynamicDic["edge_attr_matrices"])
        Temporal_batch.to(device)
        x, edge_index, edge_arr = Temporal_batch.x, Temporal_batch.edge_index, Temporal_batch.edge_attr
        # 将单个图的数据送入 GAT 模型
        # 假设 GAT 模型返回节点特征矩阵 (num_nodes, hidden_dim)
        node_features = model(x, edge_index, edge_arr)

        # 对节点特征进行全局池化，得到图特征向量 (hidden_dim)
        graph_feature = node_features.mean(dim=0)  # 对节点特征取平均 (1, hidden_dim)

        gat_features.append(graph_feature.unsqueeze(0))  # 将每个图的特征向量添加到列表
    # 将所有图的特征堆叠起来 (batch_size, hidden_dim)
    gat_features = torch.cat(gat_features, dim=0)
    return gat_features

# ppi_file = r'C:\Apps\soft\ProteinFeatureExtrat\PPIData\humanData\ppiNet\human_ppi_network_val.txt'
# sequencePath = r'C:\Apps\soft\ProteinFeatureExtrat\PPIData\humanData\sample_human_sequence.fasta'
# atomFilePath = load_and_get_subset_paths('atomFile', 'val')
# dsspFilePath = load_and_get_subset_paths('dssp_output_14', 'val')
# pssmFilePath = load_and_get_subset_paths('pssmFile', 'val')
# coordination_trans = load_and_get_subset_paths('Mol_coordinate_trans_File', 'val')
# PortT5FeaturePath = load_and_get_subset_paths('PortT5FeatureValGeodata')
#
# dataloader = BuildDataloader(dsspPath=dsspFilePath, pssmPath=pssmFilePath,
#                              sequencePath=sequencePath,
#                              atomPath=atomFilePath, PortT5FeaturePath=PortT5FeaturePath,
#                              ppiFile=ppi_file, coordinationPath=coordination_trans, mode='dfs')
#
# subgraphs_data_list, _ = dataloader.build_subgraph()
# subgraphs_data_list_portT5, subgraphs_data_list_wavelet = dataloader.generate_subgraph_data_for_two_branch()
#
# output1 = Temporal_protein_Gat_batch(subgraphs_data_list_portT5[0], Port5_Gat_model)
#
# output2 = process_wavelet_attention_batch(subgraphs_data_list_wavelet[0], MPSWA)
#
# out3 = Build_Joint_FeatureMatrix(output1, output2, GATE_Mechanism_Branch)
#
# subgraphs_data_list[0].x = out3
# data0 = subgraphs_data_list[0]
# print(data0.edge_index, data0.y)
