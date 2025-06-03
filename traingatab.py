import os
import torch
import torch.nn as nn
from VGAE import Encoder, Decoder, PPI_VGAE
from tqdm import tqdm
import torch.nn.functional as F
import IntermediateConcatenate as ConcatDic
from utils.FilePATH import load_and_get_subset_paths
from torch.optim import Adam
from IntermediateConcatenate import Port5_Gat_model, MPSWA, GATE_Mechanism_Branch, change_dim_wavelet
from Concat83Feature import ReturnFeature83
import generate_subgraphs
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from datetime import datetime


# ---------------------- 绘图函数 ----------------------
def plot_auc_curves(true_labels, pred_probs, epoch, mode='val'):
    # 确保保存目录存在
    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）

    # ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc_score(true_labels, pred_probs):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve @ Epoch {epoch}')
    plt.legend(loc="lower right")

    # PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR (AP = {average_precision_score(true_labels, pred_probs):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve @ Epoch {epoch}')
    plt.legend(loc="upper right")

    plt.tight_layout()

    # 修改保存路径到 ./plots/
    save_path = os.path.join(save_dir, f'epoch{epoch}_{mode}_curves.png')  # 跨平台路径拼接
    plt.savefig(save_path, dpi=300)  # 使用新路径
    plt.close()


# ---------------------- logs cofig ----------------------
def setup_logger():
    """配置同时输出到文件和终端的日志系统"""
    try:
        # 确保logs目录存在
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)  # 自动创建目录，exist_ok避免重复创建报错

        # 生成带时间戳的日志文件名
        log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(log_dir, log_filename)  # 兼容不同操作系统的路径

        # 清除之前的日志处理器（避免重复）
        logging.getLogger().handlers = []

        # 文件日志配置
        logging.basicConfig(
            filename=log_filepath,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'  # 覆盖模式（每次运行生成新文件）
        )

        # 控制台日志配置（与文件日志级别一致）
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        logging.info(f"日志文件已创建：{os.path.abspath(log_filepath)}")

    except PermissionError:
        logging.error("无权限创建日志目录，回退到当前目录")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    except Exception as e:
        logging.error(f"日志初始化失败: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    setup_logger()
    logging.info("这是一条测试日志")

'''
划分数据集
'''


def get_dataloader(dataset, mode):
    atomFilePath = load_and_get_subset_paths(datasets=dataset, folder_name='atomFile')
    dsspFilePath = load_and_get_subset_paths(datasets=dataset, folder_name='dssp_output_14', )
    pssmFilePath = load_and_get_subset_paths(datasets=dataset, folder_name='pssmFile', )
    PortT5FeaturePath = load_and_get_subset_paths(datasets=dataset, folder_name='PortT5FeatureGeodata')
    sequenceFile = load_and_get_subset_paths(datasets=dataset, folder_name='sequenceFile')
    ppiFile = load_and_get_subset_paths(datasets=dataset, folder_name='ppiFile')
    coordinationFile = load_and_get_subset_paths(datasets=dataset, folder_name='Mol_coordinate_trans_File')

    Feature84 = ReturnFeature83(dsspPath=dsspFilePath, pssmPath=pssmFilePath, sequencePath=sequenceFile,
                                atomPath=atomFilePath,
                                coordinationPath=coordinationFile)
    PortT5Feature = torch.load(PortT5FeaturePath, weights_only=False)

    subgraphs_data, subgraphs_protein_ids = generate_subgraphs.build_ppi_subgraph(ppi_file=ppiFile, mode=mode)
    fold_data_splits = generate_subgraphs.k_fold_cross_validation(subgraphs_data, subgraphs_protein_ids)
    return fold_data_splits[0], Feature84, PortT5Feature


def create_optimizer_for_models(models, learning_rate=0.001, weight_decay=1e-5):
    """
    为多个模型创建一个统一的优化器。

    参数：
    - models: list, 包含多个PyTorch模型的列表。
    - learning_rate: float, 优化器的学习率，默认为0.001。
    - weight_decay: float, 优化器的L2正则化项，默认为1e-5。

    返回：
    - optimizer: PyTorch优化器对象，能够同时优化多个模型的参数。
    """

    # 合并所有模型的参数
    all_params = []
    for model in models:
        all_params.extend(list(model.parameters()))

    # 创建优化器
    optimizer = Adam(all_params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer, all_params


'''
这个是重构出邻接矩阵，利用真实的标签进行计算损失
'''


def edge_loss(reconstructed_adj, edge_index, edge_label):
    # 重构的邻接矩阵是一个概率矩阵，真实的边是一个稀疏矩阵
    # edge_label 是真实的边标签（0或1）
    # reconstructed_adj 是重构的边概率矩阵
    # 获取边的源节点和目标节点的索引
    row, col = edge_index
    # 预测的边概率
    pred = reconstructed_adj[row, col]

    predictions = (pred > 0.5).float()
    accuracy = (predictions == edge_label).float().mean()

    # 计算二元交叉熵损失
    return F.binary_cross_entropy(pred, edge_label), accuracy


'''
把模型一键设置为train or eval
'''


def set_models_mode(mode, *models):
    """
    设置所有传入的模型为 train 或 eval 模式
    :param mode: 'train' 或 'eval'
    :param models: 需要设置模式的模型，可以传入多个
    """
    assert mode in ['train', 'eval'], "mode 参数必须是 'train' 或 'eval'"

    for model in models:
        if model is not None:  # 确保模型不是 None
            if mode == 'train':
                model.train()
            else:
                model.eval()


def train(model, optimizer, device, subgraphs_data_list_train, subgraphs_data_list_portT5_train,
          subgraphs_data_list_wavelet_train):
    # 首先构建dataloader
    all_loss = 0.0
    batch_count = 0
    all_acc_node = 0.0
    all_acc_edge = 0.0
    set_models_mode('train', Port5_Gat_model, MPSWA, GATE_Mechanism_Branch, VGAE_model, change_dim_wavelet)
    for GeoData, PorT5Date, waveletFeature in tqdm(
            list(zip(subgraphs_data_list_train, subgraphs_data_list_portT5_train, subgraphs_data_list_wavelet_train)),
            desc="Processing subgraph_train batch ", ncols=100):
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # 清理缓存，避免内存碎片化
        # 进入两个分支模型，得到当前分支的数据
        gat_features = ConcatDic.process_gat_batch(PorT5Date, Port5_Gat_model)
        features_84 = ConcatDic.process_wavelet_attention_batch(waveletFeature, MPSWA)
        # 利用GATE机制，将两个分支的特征进行与lamda相乘后拼接特征，并放入字典
        combined_features = ConcatDic.Build_Joint_FeatureMatrix(gat_features, features_84.to(device),
                                                                GATE_Mechanism_Branch)
        # 利用蛋白质字典和ppi文件构建data数据
        GeoData.x = combined_features

        data_train = GeoData

        data_train.to(device)

        # 使用完整的节点特征，但只使用训练集的边
        z, z_mean, z_logstd = model.encode(
            data_train.x.to(device),
            data_train.edge_index.to(device),
            data_train.edge_attr.to(device)
        )

        # 计算 KL 散度
        kl_loss = -0.5 * torch.mean(1 + z_logstd - z_mean.pow(2) - z_logstd.exp())

        # 进入全连接层，输出关于ppi作用的概率
        pred = model.decoder(z, data_train.edge_index)
        # 交叉熵损失（边的预测）
        recon_node_loss = nn.BCELoss()(pred, data_train.y.float())
        # # 重构出边，然后于真实的边进行loss
        recon_edge_loss, accuracy_edge = edge_loss(model.decode_all(z),
                                                   data_train.edge_index,
                                                   data_train.y.float())

        loss = recon_node_loss + kl_loss + recon_edge_loss
        # 总损失
        tqdm.write(
            f'recon_node_loss: {recon_node_loss.item():.4f}, recon_edge_loss: {recon_edge_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}')

        # 反向传播
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # 清理缓存，避免内存碎片化
        predictions = (pred > 0.5).float()
        accuracy_node = (predictions == data_train.y).float().mean()
        all_loss += loss.item()
        all_acc_node += accuracy_node.item()
        all_acc_edge += accuracy_edge.item()
        batch_count += 1
    return all_loss / batch_count, all_acc_node / batch_count, all_acc_edge / batch_count


def validate(model, device, subgraphs_data_list_val, subgraphs_data_list_portT5_val, subgraphs_data_list_wavelet_val):
    all_loss = 0.0
    batch_count = 0
    all_acc_node = 0.0
    all_acc_edge = 0.0
    all_preds = []
    all_labels = []
    set_models_mode('eval', Port5_Gat_model, MPSWA, GATE_Mechanism_Branch, change_dim_wavelet)
    torch.cuda.empty_cache()  # 清理缓存，避免内存碎片化
    with torch.no_grad():
        for GeoData, PorT5Date, waveletFeature in tqdm(
                list(zip(subgraphs_data_list_val, subgraphs_data_list_portT5_val,
                         subgraphs_data_list_wavelet_val)),
                desc="Processing subgraph_val batch", ncols=100):
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # 清理缓存，避免内存碎片化
            # 进入两个分支模型，得到当前分支的数据
            gat_features = ConcatDic.process_gat_batch(PorT5Date, Port5_Gat_model)
            features_68 = ConcatDic.process_wavelet_attention_batch(waveletFeature, MPSWA)
            # 利用GATE机制，将两个分支的特征进行与lamda相乘后拼接特征，并放入字典
            combined_features = ConcatDic.Build_Joint_FeatureMatrix(gat_features, features_68,
                                                                    GATE_Mechanism_Branch)
            # 利用蛋白质字典和ppi文件构建data数据
            GeoData.x = combined_features

            data_val = GeoData

            data_val.to(device)

            z, z_mean, z_logstd = model.encode(
                data_val.x.to(device),
                data_val.edge_index.to(device),
                data_val.edge_attr.to(device)
            )

            # 计算 KL 散度
            kl_loss = -0.5 * torch.mean(1 + z_logstd - z_mean.pow(2) - z_logstd.exp())
            # 预测是否相互作用的概率

            pred = model.decoder(z, data_val.edge_index)

            # 交叉熵损失（边的预测）
            recon_node_loss = nn.BCELoss()(pred, data_val.y.float())
            #  重构出边，然后于真实的边进行loss
            recon_edge_loss, accuracy_edge = edge_loss(model.decode_all(z),
                                                       data_val.edge_index,
                                                       data_val.y.float())
            # 总损失
            loss = recon_node_loss + kl_loss + recon_edge_loss
            batch_count += 1
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(data_val.y.cpu().numpy())

            # 计算其他指标
            predictions = (pred > 0.5).float()
            accuracy_node = (predictions == data_val.y).float().mean()
            all_loss += loss.item()
            all_acc_node += accuracy_node.item()
            all_acc_edge += accuracy_edge.item()
        # 计算高级指标
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        advanced_metrics = {
            'precision': float(precision_score(all_labels, all_preds > 0.5)),
            'recall': float(recall_score(all_labels, all_preds > 0.5)),
            'f1': float(f1_score(all_labels, all_preds > 0.5)),
            'auc_roc': float(roc_auc_score(all_labels, all_preds)),
            'aupr': float(average_precision_score(all_labels, all_preds))
        }

        return all_loss / batch_count, all_acc_node / batch_count, all_acc_edge / batch_count, advanced_metrics, all_labels, all_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输入，输出维度
encode_input = 256
encode_hidden = 125
encode_output = 64

decode_input = 64
decode_hidden = 32

VGAE_model = PPI_VGAE(Encoder(encode_input, encode_hidden, encode_output), Decoder(decode_input, decode_hidden))

optimizer, all_params = create_optimizer_for_models(
    [Port5_Gat_model, MPSWA, GATE_Mechanism_Branch, VGAE_model, change_dim_wavelet],
    learning_rate=1e-4, weight_decay=1e-4)

# 2. 创建学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
VGAE_model.to(device)

torch.cuda.empty_cache()


def main():
    setup_logger()  # initial log
    data_splits, Feature84, PortT5Feature = get_dataloader('27k', mode='bfs')

    all_branch_PortT5_data_list_train = []
    all_branch_wavelet_features_list_train = []
    all_branch_PortT5_data_list_val = []
    all_branch_wavelet_features_list_val = []

    subgraphs_data_list_train, subgraphs_data_list_val, train_protein_ids, val_protein_ids = data_splits

    # 训练集特征处理
    for branch_protein_ids_train in train_protein_ids:
        branch_PortT5_data_list_train, branch_wavelet_padded_features_train = generate_subgraphs.process_protein_branch_feature(
            branch_protein_ids_train, PortT5Feature, Feature84)
        all_branch_PortT5_data_list_train.append(branch_PortT5_data_list_train)
        all_branch_wavelet_features_list_train.append(branch_wavelet_padded_features_train)

    # 验证集特征处理
    for branch_protein_ids_val in val_protein_ids:
        branch_PortT5_data_list_val, branch_wavelet_padded_features_val = generate_subgraphs.process_protein_branch_feature(
            branch_protein_ids_val, PortT5Feature, Feature84)
        all_branch_PortT5_data_list_val.append(branch_PortT5_data_list_val)
        all_branch_wavelet_features_list_val.append(branch_wavelet_padded_features_val)

    num_epochs = 60
    lr_decay_epoch = 20
    logging.info(
        "Epoch\tTrainLoss\tTrainAccN\tTrainAccE\tValLoss\tValAccN\tValAccE\tPrecision\tRecall\tF1"
    )
    for epoch in tqdm(range(num_epochs), desc='Training'):
        # ================== train ==================
        train_loss, train_acc_node, train_acc_edge = train(
            VGAE_model, optimizer, device,
            subgraphs_data_list_train,
            all_branch_PortT5_data_list_train,
            all_branch_wavelet_features_list_train
        )
        # ================== val  ==================
        val_loss, val_acc_node, val_acc_edge, val_advanced, val_labels, val_preds = validate(
            VGAE_model, device,
            subgraphs_data_list_val,
            all_branch_PortT5_data_list_val,
            all_branch_wavelet_features_list_val
        )

        # ================== data process ==================
        precision = float(val_advanced.get('precision', 0.0))
        recall = float(val_advanced.get('recall', 0.0))
        f1 = float(val_advanced.get('f1', 0.0))

        # ================== logs file ==================
        logging.info(
            f"{epoch + 1}\t"
            f"{train_loss:.4f}\t{train_acc_node:.4f}\t{train_acc_edge:.4f}\t"
            f"{val_loss:.4f}\t{val_acc_node:.4f}\t{val_acc_edge:.4f}\t"
            f"{val_advanced.get('precision', 0):.4f}\t"
            f"{val_advanced.get('recall', 0):.4f}\t"
            f"{val_advanced.get('f1', 0):.4f}\n"
        )

        # ================== print ==================
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"[Train] Loss: {train_loss:.4f}  Node Acc: {train_acc_node:.2%}  Edge Acc: {train_acc_edge:.2%}")
        print(f"[Val]   Loss: {val_loss:.4f}  Node Acc: {val_acc_node:.2%}  Edge Acc: {val_acc_edge:.2%}")
        print(
            f"Metrics >> P: {val_advanced.get('precision', 0):.4f}  R: {val_advanced.get('recall', 0):.4f}  F1: {val_advanced.get('f1', 0):.4f}")
        print(f"AUC-ROC: {val_advanced.get('auc_roc', 0):.4f}  AUPR: {val_advanced.get('aupr', 0):.4f}")
        print("-" * 60)

        # ================== learning modify ==================
        if epoch + 1 == lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"\n学习率在 epoch {epoch + 1} 调整为 {optimizer.param_groups[0]['lr']:.2e}")

        # ================== gradient ==================
        torch.nn.utils.clip_grad_norm_(VGAE_model.parameters(), max_norm=1.0)
        scheduler.step(val_loss)
        # ================== plot curve ==================
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
            try:
                plot_auc_curves(val_labels, val_preds, epoch + 1, 'val')
                print(f"已保存评估曲线图：epoch{epoch + 1}_val_curves.png")
            except Exception as e:
                logging.error(f"绘图失败: {str(e)}")


if __name__ == '__main__':
    main()
