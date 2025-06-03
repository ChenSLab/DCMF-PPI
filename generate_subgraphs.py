import random
from collections import deque
from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
from typing import List, Set, Tuple
from tqdm import tqdm
'''
这个是生成蛋白质的子图，返回每个子图的list的集合，和对应的蛋白质id
'''


def build_ppi_subgraph(ppi_file: str, negative_sample_ratio: float =0.8, mode: str = 'random', min_nodes: int = 6):
    """
    构建PPI图数据（包括正负样本），并为每个边生成标签。支持添加对称边。
    支持三种子图构建模式：random（随机）、BFS（广度优先）、DFS（深度优先）。

    Args:
        ppi_file (str): PPI文件路径
        negative_sample_ratio (float): 负样本比例
        mode (str): 子图构建模式 ('random', 'bfs', 或 'dfs')
        min_nodes (int): 每个子图的最小节点数
    """

    def build_graph_from_edges(edges_list):
        print("构建邻接表...")
        graph = {}
        for p1, p2, _ in tqdm(edges_list, desc="构建邻接表"):
            if p1 not in graph:
                graph[p1] = set()
            if p2 not in graph:
                graph[p2] = set()
            graph[p1].add(p2)
            graph[p2].add(p1)
        return graph

    # 改进的随机节点选择方式
    def get_subgraph_nodes_random(proteins, graph, size):
        """选择一部分连接较多的节点，放宽连接要求"""
        proteins_list = list(proteins)
        max_attempts = 1000  # 最大尝试次数

        for _ in range(max_attempts):
            # 随机选择起始节点
            start_node = random.choice(proteins_list)
            selected_nodes = {start_node}

            # 获取所有可能的候选节点（邻居）
            candidate_nodes = set(graph[start_node])

            # 从候选节点中选择剩余节点
            candidate_nodes &= proteins  # 只保留未处理的节点
            if len(candidate_nodes) >= size - 1:
                # 优先选择度数较高的节点
                candidates_with_degree = [(node, len(graph[node])) for node in candidate_nodes]
                candidates_with_degree.sort(key=lambda x: x[1], reverse=True)

                # 选择前 size-1 个节点
                selected_nodes.update(node for node, _ in candidates_with_degree[:size - 1])

                if len(selected_nodes) >= size:
                    return selected_nodes

        # 如果上述方法失败，回退到完全随机选择
        return set(random.sample(proteins_list, min(size, len(proteins))))

    def get_subgraph_nodes_bfs(graph, start_node, size):
        visited = set()
        queue = deque([start_node])
        subgraph_nodes = []

        while queue and len(subgraph_nodes) < size:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                subgraph_nodes.append(node)
                queue.extend(n for n in graph[node] if n not in visited)
                if len(queue) > size * 2:
                    queue = deque(list(queue)[:size])

        return set(subgraph_nodes)

    def get_subgraph_nodes_dfs(graph, start_node, size):
        visited = set()
        subgraph_nodes = []

        def dfs(node, depth=0):
            if len(subgraph_nodes) >= size or depth > size:
                return
            if node not in visited:
                visited.add(node)
                subgraph_nodes.append(node)
                neighbors = list(graph[node])
                random.shuffle(neighbors)
                for neighbor in neighbors[:size]:
                    if len(subgraph_nodes) < size:
                        dfs(neighbor, depth + 1)

        dfs(start_node)
        return set(subgraph_nodes)

    def check_connectivity(nodes, graph):
        """检查节点集合是否形成连通子图"""
        if not nodes:
            return False

        visited = set()
        stack = [next(iter(nodes))]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(n for n in graph[node] if n in nodes and n not in visited)

        return len(visited) == len(nodes)

    # 读取边并构建图
    print("读取PPI文件...")
    all_edges = []
    with open(ppi_file, 'r') as f:
        for line in tqdm(f, desc="读取PPI文件"):
            p1, p2, score = line.strip().split()
            all_edges.append((p1, p2, float(score)))

    if not all_edges:
        raise ValueError("No edges found in the input file")

    graph = build_graph_from_edges(all_edges)
    all_proteins = set(graph.keys())
    processed_proteins = set()

    subgraphs_data = []
    subgraphs_protein_ids = []

    max_iterations = len(all_proteins) // min_nodes * 2
    iteration_count = 0
    failed_attempts = 0
    max_failed_attempts = 50  # 最大失败尝试次数

    with tqdm(total=len(all_proteins), desc="构建子图") as pbar:
        while all_proteins and iteration_count < max_iterations and failed_attempts < max_failed_attempts:
            iteration_count += 1

            if len(all_proteins) < min_nodes:
                pbar.update(len(all_proteins))
                break

            try:
                if mode == 'random':
                    protein_ids_in_graph = get_subgraph_nodes_random(all_proteins, graph, min_nodes)
                else:
                    start_node = random.choice(list(all_proteins))
                    if mode == 'bfs':
                        protein_ids_in_graph = get_subgraph_nodes_bfs(graph, start_node, min_nodes)
                    else:  # dfs
                        protein_ids_in_graph = get_subgraph_nodes_dfs(graph, start_node, min_nodes)

                # 检查子图的连通性
                if not check_connectivity(protein_ids_in_graph, graph):
                    failed_attempts += 1
                    continue

            except Exception as e:
                print(f"Error in subgraph generation: {e}")
                failed_attempts += 1
                continue

            if len(protein_ids_in_graph) < min_nodes:
                failed_attempts += 1
                continue

            # 构建子图边
            edges = []
            edge_attrs = []
            labels = []
            edge_set = set()
            protein_id_map = {pid: idx for idx, pid in enumerate(protein_ids_in_graph)}

            # 添加正样本边
            for p1, p2, score in all_edges:
                if p1 in protein_id_map and p2 in protein_id_map:
                    idx1 = protein_id_map[p1]
                    idx2 = protein_id_map[p2]
                    if (idx1, idx2) not in edge_set:
                        edges.append([idx1, idx2])
                        edge_attrs.append([score])
                        labels.append(1)
                        edge_set.add((idx1, idx2))

            if len(edges) < min_nodes // 2:  # 确保有足够的边
                failed_attempts += 1
                continue

            # 生成负样本
            num_positive = len(edges)
            num_negative = int(num_positive * negative_sample_ratio)
            negative_attempts = 0
            max_attempts = min(num_negative * 10, 1000)

            while len(edges) < num_positive + num_negative and negative_attempts < max_attempts:
                idx1 = random.randint(0, len(protein_ids_in_graph) - 1)
                idx2 = random.randint(0, len(protein_ids_in_graph) - 1)
                if idx1 != idx2 and (idx1, idx2) not in edge_set and (idx2, idx1) not in edge_set:
                    edges.append([idx1, idx2])
                    edge_attrs.append([0.0])
                    labels.append(0)
                    edge_set.add((idx1, idx2))
                negative_attempts += 1

            # 构建PyG的Data对象
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            x = torch.zeros(len(protein_ids_in_graph), 1, dtype=torch.float32)

            subgraphs_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            subgraphs_protein_ids.append(list(protein_ids_in_graph))

            # 更新进度
            processed_proteins.update(protein_ids_in_graph)
            all_proteins -= protein_ids_in_graph
            failed_attempts = 0  # 重置失败计数
            pbar.update(len(protein_ids_in_graph))

            # 打印调试信息
            if iteration_count % 10 == 0:
                print(f"\n当前进度：")
                print(f"- 剩余蛋白质数量: {len(all_proteins)}")
                print(f"- 已处理蛋白质数量: {len(processed_proteins)}")
                print(f"- 已生成子图数量: {len(subgraphs_data)}")

    if not subgraphs_data:
        raise ValueError("Unable to create any valid subgraphs with the given parameters")

    print(f"\n构建完成！共生成 {len(subgraphs_data)} 个子图")
    return subgraphs_data, subgraphs_protein_ids


'''

这个函数是生成两个分支的数据,根据上个文件生成的子图蛋白质的id进行提取的
'''


def process_protein_branch_feature(protein_id_list, data_dict, feature_dict):
    """
    处理蛋白质ID列表，从两个字典中查找数据，填充特征矩阵，返回结果。

    参数：
    - protein_id_list: 一个包含蛋白质ID的列表，按顺序排列。
    - data_dict: 第一个字典，存储蛋白质ID到`torch_geometric.data.Data`对象的映射。
    - feature_dict: 第二个字典，存储蛋白质ID到特征矩阵的映射。

    返回：
    - data_list: 从`data_dict`中获取的`Data`对象的列表。
    - padded_features: 填充后的特征矩阵组成的Tensor。
    """
    # 用于存储找到的Data对象和特征矩阵
    data_list = []
    features_list = []

    # 找到最大序列长度
    max_seq_len = max(len(feature_dict[protein_id]) for protein_id in protein_id_list)

    # 遍历protein_id_list，提取每个蛋白质的Data和特征矩阵
    for protein_id in protein_id_list:
        # 在data_dict中查找对应的Data对象
        data = next((data for data in data_dict if data.name == protein_id), None)

        if data:
            data_list.append(data)

            # 获取蛋白质特征矩阵并填充到最大长度
            feature_matrix = feature_dict.get(protein_id)
            if feature_matrix is not None:
                # 确保将numpy数组转换为torch.Tensor，并设置为需要梯度计算
                feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, requires_grad=True)

                # 创建一个新的张量，避免原地修改
                padded_feature_matrix = torch.zeros((max_seq_len, feature_matrix.shape[1]), dtype=torch.float32,
                                                    requires_grad=True)

                # 通过创建一个新的tensor来填充，避免原地操作
                padded_feature_matrix.data[:feature_matrix.shape[0], :] = feature_matrix.data
                features_list.append(padded_feature_matrix)

    # 将所有特征矩阵堆叠成一个大的tensor
    padded_features = torch.stack(features_list)

    return data_list, padded_features


'''

进行5-fold cross-validation
'''


def k_fold_cross_validation(data_list, protein_id_list, num_folds=5):
    """
    Perform k-fold cross-validation on a dataset of PPI subgraphs and protein IDs.

    Parameters:
    - data_list: List of torch_geometric.data.Data objects representing PPI subgraphs.
    - protein_id_list: List of protein IDs corresponding to each graph in data_list.
    - num_folds: The number of folds for cross-validation (default is 5).

    Returns:
    - A list of tuples where each tuple is (train_data, val_data, train_protein_ids, val_protein_ids)
    """
    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # List to store the train and validation data splits for each fold
    fold_data_splits = []

    # Iterate through the folds
    for fold, (train_index, val_index) in enumerate(kf.split(data_list)):
        print(f"Fold {fold + 1}")

        # Split data into training and validation sets
        train_data = [data_list[i] for i in train_index]
        val_data = [data_list[i] for i in val_index]

        train_protein_ids = [protein_id_list[i] for i in train_index]
        val_protein_ids = [protein_id_list[i] for i in val_index]

        # Optionally print the size of the train and validation sets
        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")

        # Append the split data to the list
        fold_data_splits.append((train_data, val_data, train_protein_ids, val_protein_ids))

    return fold_data_splits


# subgraphs_data, subgraphs_protein_ids = build_ppi_subgraph(r"C:\SHS_data\27k\ppi_interaction_27k.txt",mode='bfs')
# fold_data_splits = k_fold_cross_validation(subgraphs_data, subgraphs_protein_ids, num_folds=5)
# print(fold_data_splits)
# 访问每一折的训练和验证集数据
# for fold, (train_data, val_data, train_protein_ids, val_protein_ids) in enumerate(fold_data_splits):
