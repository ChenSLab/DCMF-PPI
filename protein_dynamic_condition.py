import os
import torch
import Bio.PDB as PDB
import numpy as np
from scipy.linalg import eigh


class ElasticNetworkModel:
    # [保持原有的ElasticNetworkModel类不变]
    def __init__(self, pdb_file, cutoff=12.0, force_constant=1.0):
        self.cutoff = cutoff
        self.force_constant = force_constant
        self.coords = None
        self.n_atoms = None
        self.eigenvals = None
        self.eigenvecs = None
        self.kirchhoff = None
        self.load_structure(pdb_file)

    def load_structure(self, pdb_file):
        parser = PDB.PDBParser()
        structure = parser.get_structure('protein', pdb_file)

        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'].get_coord())

        self.coords = np.array(ca_atoms)
        self.n_atoms = len(self.coords)
        print(f"Loaded {self.n_atoms} CA atoms from structure")

    def build_kirchhoff(self):
        self.kirchhoff = np.zeros((3 * self.n_atoms, 3 * self.n_atoms))

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                diff = self.coords[i] - self.coords[j]
                dist = np.sqrt(np.sum(diff ** 2))

                if dist < self.cutoff:
                    cos_terms = np.outer(diff, diff) / (dist ** 2)

                    for di in range(3):
                        for dj in range(3):
                            i_idx = 3 * i + di
                            j_idx = 3 * j + dj
                            self.kirchhoff[i_idx, j_idx] = -cos_terms[di, dj]
                            self.kirchhoff[j_idx, i_idx] = -cos_terms[di, dj]

        for i in range(3 * self.n_atoms):
            self.kirchhoff[i, i] = -np.sum(self.kirchhoff[i, :])

    def compute_modes(self):
        if self.kirchhoff is None:
            self.build_kirchhoff()

        self.eigenvals, self.eigenvecs = eigh(self.kirchhoff)
        self.eigenvals = self.eigenvals[6:]
        self.eigenvecs = self.eigenvecs[:, 6:]

    def build_dynamic_adjacency(self, mode_index, amplitude=10.0):
        if self.eigenvals is None:
            self.compute_modes()

        mode = self.eigenvecs[:, mode_index].reshape(self.n_atoms, 3)
        deformed_coords = self.coords + amplitude * mode

        edge_index = []
        edge_attr = []

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                diff = deformed_coords[i] - deformed_coords[j]
                dist = np.sqrt(np.sum(diff ** 2))

                if dist < self.cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(dist)
                    edge_attr.append(dist)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        return edge_index, edge_attr


def process_pdb_folder(pdb_folder_path, output_folder_path, num_modes=3):
    """
    处理文件夹中的所有PDB文件并保存结果

    Args:
        pdb_folder_path (str): PDB文件所在文件夹路径
        output_folder_path (str): 输出文件夹路径
        num_modes (int): 要分析的模式数量
    """
    # 创建输出文件夹
    os.makedirs(output_folder_path, exist_ok=True)

    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(pdb_folder_path) if f.endswith('.pdb')]

    for pdb_file in pdb_files:
        try:
            # 获取蛋白质ID
            protein_id = pdb_file.split('.pdb')[0]
            print(f"Processing protein: {protein_id}")

            # 构建完整的文件路径
            pdb_path = os.path.join(pdb_folder_path, pdb_file)

            # 创建ENM模型
            enm = ElasticNetworkModel(pdb_path)

            # 计算模式
            enm.compute_modes()

            # 存储结果的字典
            result_dict = {
                'adj_matrices': [],
                'edge_attr_matrices': []
            }

            # 获取不同运动模式下的邻接矩阵
            for mode in range(num_modes):
                adj, attr = enm.build_dynamic_adjacency(mode)
                result_dict['adj_matrices'].append(adj)
                result_dict['edge_attr_matrices'].append(attr)

            # 保存结果为pt文件
            output_path = os.path.join(output_folder_path, f"{protein_id}.pt")
            torch.save(result_dict, output_path)
            print(f"Saved results for {protein_id}")

        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            continue


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出路径
    pdb_folder = r"C:\SHS_data\27k\pdbFile_27k"
    output_folder = r"C:\SHS_data\27k\dynamic_adj_File_27k"

    # 处理所有PDB文件
    process_pdb_folder(pdb_folder, output_folder)
