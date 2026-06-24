# graph_loader.py
from config import *
import torch
import scipy.sparse as spp
import os
import numpy as np
import sys
from torch.utils.data import Dataset
from torch_geometric.data import Data
import zipfile

# 移除DGL相关导入，添加PyG导入
from torch_geometric.data import Batch

if len(sys.argv) > 1:
    datasetname, rst_file, pkl_path, batchsize = sys.argv[1:]
    batchsize = int(batchsize)
else:
    datasetname = 'receptor-peptide'
    # 修改默认路径，使用 config.py 中定义的路径
    rst_file = f'{RESULTS_PATH}/equi_results.tsv'
    pkl_path = f'{MODEL_PKL_PATH}/Equi'
    batchsize = 32

# 修复设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 氨基酸映射表
aa_map = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}


def sequence_to_tensor(sequence, max_length=1200):
    """将氨基酸序列转换为张量"""
    seq_indices = [aa_map.get(aa, 0) for aa in sequence[:max_length]]
    if len(seq_indices) < max_length:
        seq_indices.extend([0] * (max_length - len(seq_indices)))
    return torch.tensor(seq_indices, dtype=torch.long)


def collate(samples):
    p1, p2, graphs1, dmaps1, graphs2, dmaps2, seqs1, seqs2, labels = map(list, zip(*samples))
    # 确保标签张量在正确的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 确保序列张量也在正确的设备上
    seqs1 = [seq.to(device) for seq in seqs1]
    seqs2 = [seq.to(device) for seq in seqs2]
    return p1, p2, graphs1, dmaps1, graphs2, dmaps2, torch.stack(seqs1), torch.stack(seqs2), torch.tensor(labels).to(device)


class EmbeddingLoader:
    def __init__(self, pos_embed_path, neg_embed_path):
        self.pos_embeddings = {}
        self.neg_embeddings = {}
        self.load_embeddings(pos_embed_path, 'pos')
        self.load_embeddings(neg_embed_path, 'neg')

    def load_embeddings(self, npz_path, type):
        if not os.path.exists(npz_path):
            print(f"Warning: Embedding file not found: {npz_path}")
            return

        try:
            with zipfile.ZipFile(npz_path, 'r') as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        embedding = np.load(f)
                        seq_id = name.replace('.npy', '')
                        if type == 'pos':
                            self.pos_embeddings[seq_id] = embedding
                        else:
                            self.neg_embeddings[seq_id] = embedding
        except Exception as e:
            print(f"Error loading embeddings from {npz_path}: {e}")

    def get_embedding(self, seq_id):
        if seq_id in self.pos_embeddings:
            return self.pos_embeddings[seq_id]
        elif seq_id in self.neg_embeddings:
            return self.neg_embeddings[seq_id]
        else:
            return None


# 初始化嵌入加载器
embed_loader = EmbeddingLoader(POS_EMBEDDINGS, NEG_EMBEDDINGS)


def default_loader(cpath, pid):
    cmap_data = np.load(cpath)
    seq = str(cmap_data['seq'])
    nodenum = len(seq)
    cmap = cmap_data['contact']

    # 获取序列嵌入
    embedding = embed_loader.get_embedding(pid)
    if embedding is None:
        # 如果找不到对应ID的嵌入，创建随机嵌入作为占位符
        embedding = np.random.randn(nodenum, 1280).astype(np.float32)

    # 确保嵌入长度与序列长度一致
    if len(embedding) > nodenum:
        embedding = embedding[:nodenum]
    elif len(embedding) < nodenum:
        # 如果嵌入长度不足，用零填充
        padding = np.zeros((nodenum - len(embedding), embedding.shape[1]))
        embedding = np.concatenate([embedding, padding], axis=0)

    # 创建PyG图数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_embed = torch.tensor(embedding).float().to(device)

    # 从接触图创建边索引
    adj = spp.coo_matrix(cmap)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long).to(device)

    # 创建PyG Data对象
    G = Data(x=g_embed, edge_index=edge_index)

    if nodenum > 1200:
        global textembed
        textembed = embedding[:1200]
    elif nodenum < 1200:
        textembed = np.concatenate((embedding, np.zeros((1200 - nodenum, 1280))))

    textembed = torch.tensor(textembed).float().to(device)

    # 添加序列张量
    seq_tensor = sequence_to_tensor(seq)

    return G, textembed, seq_tensor


class MyDataset(Dataset):
    def __init__(self, list1, list2, list3, transform=None, target_transform=None, loader=default_loader):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # 确定接触图路径
        self.pos_cmap_files = set()
        self.neg_cmap_files = set()

        if os.path.exists(POS_CMAP_OUTPUT):
            self.pos_cmap_files = {f.replace('.npz', '') for f in os.listdir(POS_CMAP_OUTPUT) if f.endswith('.npz')}

        if os.path.exists(NEG_CMAP_OUTPUT):
            self.neg_cmap_files = {f.replace('.npz', '') for f in os.listdir(NEG_CMAP_OUTPUT) if f.endswith('.npz')}

    def get_cmap_path(self, pid):
        """根据蛋白质ID确定接触图文件路径"""
        if pid in self.pos_cmap_files:
            return os.path.join(POS_CMAP_OUTPUT, pid + '.npz')
        elif pid in self.neg_cmap_files:
            return os.path.join(NEG_CMAP_OUTPUT, pid + '.npz')
        else:
            # 尝试在任一路径中查找
            pos_path = os.path.join(POS_CMAP_OUTPUT, pid + '.npz')
            neg_path = os.path.join(NEG_CMAP_OUTPUT, pid + '.npz')
            if os.path.exists(pos_path):
                return pos_path
            elif os.path.exists(neg_path):
                return neg_path
            else:
                raise FileNotFoundError(f"Contact map not found for {pid}")

    def __getitem__(self, index):
        p1 = self.list1[index]
        p2 = self.list2[index]
        label = self.list3[index]

        try:
            p1_cmap_path = self.get_cmap_path(p1)
            p2_cmap_path = self.get_cmap_path(p2)

            G1, embed1, seq1 = self.loader(p1_cmap_path, p1)
            G2, embed2, seq2 = self.loader(p2_cmap_path, p2)

            return p1, p2, G1, embed1, G2, embed2, seq1, seq2, label
        except Exception as e:
            print(f"Error loading data for {p1} and {p2}: {e}")
            # 返回默认值以避免程序崩溃
            dummy_graph = Data(x=torch.zeros(1, 1280), edge_index=torch.empty(2, 0, dtype=torch.long)).to(device)
            dummy_embed = torch.zeros(1200, 1280).to(device)
            dummy_seq = torch.zeros(1200, dtype=torch.long).to(device)
            return p1, p2, dummy_graph, dummy_embed, dummy_graph, dummy_embed, dummy_seq, dummy_seq, label

    def __len__(self):
        return len(self.list1)


def pad_sequences(vectorized_seqs, seq_lengths, contactMaps, contact_sizes, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    contactMaps_tensor = torch.zeros((len(contactMaps), contact_sizes.max(), contact_sizes.max())).float()

    for idx, (con, con_size) in enumerate(zip(contactMaps, contact_sizes)):
        contactMaps_tensor[idx, :con_size, :con_size] = torch.FloatTensor(con)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    contactMaps_tensor = contactMaps_tensor[perm_idx]
    contact_sizes = contact_sizes[perm_idx]

    target = properties.double()
    if len(properties):
        target = target[perm_idx]

    contactMaps_tensor = contactMaps_tensor.unsqueeze(1)
    return seq_tensor, seq_lengths, contactMaps_tensor, contact_sizes, target


def pad_dmap(dmaplist):
    # 确保所有张量在相同的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1200, 1280)).float().to(device)
    for idx, d in enumerate(dmaplist):
        d = d.float().to(device)
        # 确保维度正确
        if d.dim() == 2 and d.size(1) == 1280:
            # 如果序列长度小于1200，进行填充
            if d.size(0) <= 1200:
                pad_dmap_tensors[idx, :d.size(0), :] = d
            else:
                # 如果序列长度超过1200，截断
                pad_dmap_tensors[idx] = d[:1200, :]
        else:
            print(f"Warning: Unexpected dmap shape at index {idx}: {d.shape}")
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1)  # 添加通道维度
    return pad_dmap_tensors
