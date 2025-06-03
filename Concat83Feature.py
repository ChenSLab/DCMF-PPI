import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import MinMaxScaler
from Bio import PDB
from pathlib import Path
from tqdm import tqdm


def merge_features_dsspAndpssm(dssp_folder_path, pssm_folder_path):
    """
    合并dssp和pssm特征文件

    Args:
        dssp_folder_path: 包含dssp(.dssp)文件的文件夹路径
        pssm_folder_path: 包含pssm文件的文件夹路径

    Returns:
        merged_dict: 字典，key为蛋白质ID，value为拼接后的特征
    """
    merged_dict = {}

    # 获取所有dssp文件
    dssp_files = [f for f in os.listdir(dssp_folder_path) if f.endswith('.dssp')]

    # 遍历每个dssp文件
    for dssp_file in tqdm(dssp_files, desc='Merge dssp and pssm files feature'):
        # 获取protein_id (去掉.dssp后缀)
        protein_id = dssp_file[:-5]  # 移除'.dssp'后缀
        pssm_file = protein_id + '.pssm'  # 对应的pssm文件名

        # 检查pssm文件是否存在
        if os.path.exists(os.path.join(pssm_folder_path, pssm_file)):
            # 读取dssp特征
            dssp_features = np.loadtxt(os.path.join(dssp_folder_path, dssp_file))

            # 读取pssm特征
            pssm_features = np.loadtxt(os.path.join(pssm_folder_path, pssm_file))

            # 确保两个特征的行数相同
            if len(dssp_features) == len(pssm_features):
                # 水平拼接特征
                merged_features = np.hstack((dssp_features, pssm_features))
                merged_dict[protein_id] = merged_features
            else:
                print(f"Warning: {protein_id} 的dssp和pssm特征行数不匹配，已跳过")
        else:
            print(f"Warning: 未找到对应的PSSM文件 {pssm_file}")

    return merged_dict


def generate_oneHot_feature(fasta_file_path, amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    """
    从FASTA文件读取蛋白质序列，并将其转换为20维one-hot编码。

    :param fasta_file_path: FASTA文件路径
    :param amino_acids: 默认的20种标准氨基酸字符 (str)
    :return: 一个字典，键为蛋白质ID，值为对应的one-hot编码 (形状为 [len(sequence), 20])
    """
    # 创建氨基酸到索引的映射
    amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

    # 存储one-hot编码的字典
    one_hot_dict = {}

    # 解析FASTA文件并进行编码
    for record in tqdm(SeqIO.parse(fasta_file_path, "fasta"), desc='Processing for oneHotFeature'):
        protein_id = record.id
        sequence = str(record.seq)

        # 初始化编码矩阵
        encoding = np.zeros((len(sequence), 20), dtype=np.float32)

        # 进行one-hot编码
        for i, aa in enumerate(sequence):
            if aa in amino_acid_dict:  # 检查氨基酸是否为标准氨基酸
                encoding[i, amino_acid_dict[aa]] = 1.0

        # 将编码结果存入字典
        one_hot_dict[protein_id] = encoding

    return one_hot_dict


'''
这个函数用于提取，蛋白质的理化性质文件的特征
'''


# 函数：归一化理化性质并生成特征矩阵
def generate_chemical_features(fasta_path):
    """
    根据蛋白质序列生成理化性质特征矩阵。

    参数：
    - fasta_path (str): 输入的FASTA文件路径。
    - properties_dict (dict): 氨基酸的理化性质表格。

    返回：
    - dict: 蛋白质ID为键，特征矩阵为值。
    """
    # 理化性质表格
    physicochemical_properties = {
        'A': [0.52, 0.05, 0.48, 1.8, 6.01, 1.45, 0.97],
        'R': [0.84, 0.25, 0.98, -4.5, 10.76, 0.79, 0.93],
        'N': [0.76, 0.06, 0.76, -3.5, 5.41, 0.73, 0.92],
        'D': [0.76, 0.07, 0.78, -3.5, 2.77, 0.85, 0.91],
        'C': [0.62, 0.13, 0.61, 2.5, 5.02, 0.77, 1.19],
        'Q': [0.68, 0.09, 0.85, -3.5, 5.65, 1.10, 1.10],
        'E': [0.68, 0.11, 0.92, -3.5, 3.22, 1.29, 0.87],
        'G': [0.00, 0.00, 0.00, -0.4, 5.97, 0.57, 0.75],
        'H': [0.70, 0.23, 0.70, -3.2, 7.59, 1.00, 1.08],
        'I': [1.00, 0.19, 1.00, 4.5, 6.02, 1.25, 1.60],
        'L': [1.00, 0.13, 0.98, 3.8, 6.02, 1.34, 1.22],
        'K': [0.80, 0.22, 1.08, -3.9, 9.74, 1.23, 0.75],
        'M': [0.78, 0.22, 0.92, 1.9, 5.74, 1.20, 1.05],
        'F': [0.85, 0.29, 1.13, 2.8, 5.48, 1.29, 1.17],
        'P': [0.64, 0.00, 0.72, -1.6, 6.30, 0.34, 0.55],
        'S': [0.52, 0.06, 0.54, -0.8, 5.68, 0.79, 0.76],
        'T': [0.65, 0.12, 0.73, -0.7, 5.60, 1.09, 0.84],
        'W': [0.85, 0.41, 1.37, -0.9, 5.89, 1.27, 1.37],
        'Y': [0.76, 0.32, 1.24, -1.3, 5.66, 1.14, 1.25],
        'V': [0.86, 0.13, 0.85, 4.2, 6.00, 1.06, 1.49]
    }

    # 将理化性质表格转为DataFrame
    df = pd.DataFrame(physicochemical_properties).T
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(df)  # 归一化
    normalized_dict = {aa: list(values) for aa, values in zip(physicochemical_properties.keys(), normalized_array)}

    # 解析FASTA文件并生成特征矩阵
    protein_features = {}
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc='Processing for chemicalFeature'):
        protein_id = record.id
        sequence = record.seq
        feature_matrix = []
        for aa in sequence:
            if aa in normalized_dict:
                feature_matrix.append(normalized_dict[aa])
            else:
                feature_matrix.append([0] * len(normalized_dict['A']))  # 未知氨基酸填充0
        protein_features[protein_id] = np.array(feature_matrix)

    # 这里是蛋白质ID为索引对应的字典
    return protein_features


'''
蛋白质原子特征，提取存入txt文件，以及提取txt文件特征存字典中
'''


def extract_and_save_Atomfeatures(pdb_folder, output_folder):
    """
    Extract atomic features from PDB files and save to text files.

    Args:
        pdb_folder (str): Path to folder containing PDB files
        output_folder (str): Path to folder where feature files will be saved
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize parser
    parser = PDB.PDBParser(QUIET=True)

    # Atomic properties
    mass_dict = {'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'P': 30.97}
    charge_dict = {'C': 0, 'N': -0.5, 'O': -1, 'S': -0.5, 'P': -0.5}
    vdw_radius = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80}

    # Get list of PDB files
    pdb_files = list(Path(pdb_folder).glob('*.pdb'))

    # Process each PDB file with progress bar
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files for atomFeature", unit="file"):
        try:
            # Get protein ID from filename
            protein_id = pdb_file.stem

            # Parse structure
            structure = parser.get_structure(protein_id, pdb_file)

            # List to store features for each residue
            residue_features = []

            # Process each residue
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Lists to store atomic features
                        masses = []
                        bfactors = []
                        is_sidechain = []
                        charges = []
                        h_bonds = []
                        in_ring = []
                        vdw_radii = []

                        # Process each atom in residue
                        for atom in residue:
                            if atom.element != 'H':  # Exclude hydrogen
                                masses.append(mass_dict.get(atom.element, 0))
                                bfactors.append(atom.get_bfactor())
                                is_sc = int(atom.get_name() not in ['N', 'CA', 'C', 'O'])
                                is_sidechain.append(is_sc)
                                charges.append(charge_dict.get(atom.element, 0))
                                h_count = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3}
                                h_bonds.append(h_count.get(atom.element, 0))
                                ring_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
                                in_ring.append(int(residue.get_resname() in ring_residues))
                                vdw_radii.append(vdw_radius.get(atom.element, 0))

                        # Calculate average features for the residue
                        if masses:  # Check if residue has non-hydrogen atoms
                            avg_features = [
                                np.mean(masses),
                                np.mean(bfactors),
                                np.mean(is_sidechain),
                                np.mean(charges),
                                np.mean(h_bonds),
                                np.mean(in_ring),
                                np.mean(vdw_radii)
                            ]
                            residue_features.append(avg_features)

            # Save features to file if we have any
            if residue_features:
                feature_matrix = np.array(residue_features)
                output_file = os.path.join(output_folder, f"{protein_id}.txt")
                np.savetxt(output_file, feature_matrix,
                           fmt='%.6f',
                           header='mass,bfactor,is_sidechain,charge,h_bonds,in_ring,vdw_radius',
                           delimiter=',')

        except Exception as e:
            print(f"Error processing {protein_id}: {str(e)}")
            continue

    # Print summary
    successful = len([f for f in os.listdir(output_folder) if f.endswith('_features.txt')])
    print(f"\nProcessed {successful} proteins successfully")
    failed = len(pdb_files) - successful
    if failed > 0:
        print(f"Failed to process {failed} proteins")


'''
原子特征文件提取
'''


def generate_atom_features(features_folder):
    """
    Load protein features from text files into a dictionary.
~
    Args:
        features_folder (str): Path to folder containing feature files

    Returns:
        dict: Dictionary with protein IDs as keys and feature matrices as values
    """

    protein_features = {}
    feature_files = list(Path(features_folder).glob('*.txt'))

    for feature_file in tqdm(feature_files, desc="Loading features for atomDic", unit="file"):
        try:
            # Get protein ID from filename (remove '_features.txt')
            protein_id = feature_file.stem.replace('_features', '')

            # Load features
            feature_matrix = np.loadtxt(feature_file, delimiter=',', skiprows=1)
            protein_features[protein_id] = feature_matrix

        except Exception as e:
            print(f"Error loading {protein_id}: {str(e)}")
            continue

    print(f"\nLoaded features for {len(protein_features)} proteins")
    return protein_features


'''
分子运动坐标变换差值
'''


def generate_coordination_features(folder_path):
    # 创建一个空字典来存储文件名和对应的NumPy数组
    file_contents = {}

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(folder_path), desc='process coordination features', unit="file"):
        if filename.endswith(".txt"):  # 确保是txt文件
            file_path = os.path.join(folder_path, filename)  # 获取文件的完整路径
            # 使用numpy的loadtxt函数读取文件内容为NumPy数组
            content_array = np.loadtxt(file_path, delimiter=' ')  # 假设特征值之间以空格分隔
            file_contents[filename[:-4]] = content_array  # 将文件名和NumPy数组添加到字典中

    return file_contents


'''
这个是进行特征归一化处理的函数
'''


def normalize_features_per_protein(input_dict):
    """
    对字典中的蛋白质特征矩阵按列归一化处理，范围为 [0, 1]。

    参数:
    - input_dict (dict): 输入字典，其中 key 是蛋白质的 ID，value 是 L * N 的特征矩阵。

    返回:
    - normalized_dict (dict): 返回归一化后的字典，结构与输入字典相同。
    """
    normalized_dict = {}

    for protein_id, feature_matrix in tqdm(input_dict.items(), desc='normalize_features_per_protein'):
        # 确保输入是 NumPy 数组
        feature_matrix = np.array(feature_matrix, dtype=float)

        # 初始化归一化矩阵
        normalized_matrix = np.zeros_like(feature_matrix)

        # 对每一列单独归一化
        for col_idx in range(feature_matrix.shape[1]):
            col = feature_matrix[:, col_idx]
            col_min = np.min(col)
            col_max = np.max(col)

            if col_max != col_min:
                # 归一化公式
                normalized_matrix[:, col_idx] = (col - col_min) / (col_max - col_min)
            else:
                # 如果最大值等于最小值，则该列全为 0
                normalized_matrix[:, col_idx] = 0

        # 存入结果字典
        normalized_dict[protein_id] = normalized_matrix

    return normalized_dict


'''
这个函数是进行特征合并的函数，传入字典，进行拼接
'''


def concatenate_common_protein_features(*feature_dicts):
    """
    Concatenate protein feature matrices for proteins that exist in all input dictionaries.

    Args:
        *feature_dicts: Variable number of dictionaries containing protein features
                       Each dict has protein IDs as keys and feature matrices (L×N) as values

    Returns:
        dict: A dictionary with common protein IDs as keys and concatenated feature matrices as values

    Raises:
        ValueError: If no dictionaries provided or no common proteins found or if sequence lengths are inconsistent
    """
    # Check if at least one dictionary is provided
    if len(feature_dicts) == 0:
        raise ValueError("At least one feature dictionary must be provided")

    # Find common protein IDs across all dictionaries using set intersection
    common_proteins = set(feature_dicts[0].keys())
    for feature_dict in feature_dicts[1:]:
        common_proteins = common_proteins.intersection(set(feature_dict.keys()))

    if not common_proteins:
        raise ValueError("No common proteins found across all feature dictionaries")

    result = {}
    # Process each common protein
    for protein_id in tqdm(common_proteins, desc='concatenate common_protein features '):
        # Get feature matrices for current protein from all dictionaries
        matrices = [feature_dict[protein_id] for feature_dict in feature_dicts]

        # Check if sequence lengths (L) are consistent
        seq_length = matrices[0].shape[0]
        if not all(matrix.shape[0] == seq_length for matrix in matrices):
            raise ValueError(f"Inconsistent sequence lengths for protein {protein_id}")

        # Concatenate matrices along feature dimension (axis=1)
        result[protein_id] = np.concatenate(matrices, axis=1)
        result[protein_id] = np.pad(result[protein_id], ((0, 0), (0, 1)), mode='constant')

    return result

    # 这个函数是统一返回68维度的值，方便调用


def ReturnFeature83(dsspPath, pssmPath, sequencePath, atomPath, coordinationPath):
    # 进行提取特征，并分别进行归一化处理
    merged_features = merge_features_dsspAndpssm(dsspPath, pssmPath)
    merged_features = normalize_features_per_protein(merged_features)

    one_hot_feature = generate_oneHot_feature(sequencePath)
    one_hot_feature = normalize_features_per_protein(one_hot_feature)

    chemical_feature = generate_chemical_features(sequencePath)
    chemical_feature = normalize_features_per_protein(chemical_feature)

    atom_feature = generate_atom_features(atomPath)
    atom_feature = normalize_features_per_protein(atom_feature)

    coordination_feature = generate_coordination_features(coordinationPath)
    coordination_feature = normalize_features_per_protein(coordination_feature)

    return concatenate_common_protein_features(
        merged_features,
        one_hot_feature,
        chemical_feature,
        atom_feature,
        coordination_feature
    )

