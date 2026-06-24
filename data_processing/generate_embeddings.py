# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, EsmModel
import re
import numpy as np
from tqdm import tqdm
import argparse
import io
from Bio import SeqIO
import gc
import zipfile
import os
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--fasta')
args = parser.parse_args()

# 处理正样本数据
print("Processing positive samples...")

# 处理负样本数据
print("Processing negative samples...")


def process_fasta_file(directory_path):
    """处理目录中的所有FASTA文件，返回序列ID和序列的列表"""
    if not os.path.exists(directory_path):
        print(f"警告: 目录不存在 {directory_path}")
        return [], []

    ids = []
    sequences = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 检查是否为FASTA文件
        if filename.endswith(('.fasta', '.fa', '.txt', '.seq')) or '.' not in filename:
            file_path = os.path.join(directory_path, filename)
            try:
                # 解析FASTA文件
                for record in SeqIO.parse(file_path, "fasta"):
                    ids.append(record.id)
                    sequences.append(str(record.seq))
            except Exception as e:
                print(f"解析文件 {filename} 时出错: {e}")
                continue

    return ids, sequences


def preprocess_sequences(sequences):
    """预处理序列"""
    processed_sequences = sequences.copy()
    # 将特殊氨基酸替换为X
    processed_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    return processed_sequences


def chunks(lst, n):
    """将列表分块"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def generate_embeddings(model, tokenizer, sequences, sequence_ids, output_path, device):
    """生成嵌入并向量并保存"""
    if not sequences:
        return

    print(f"Generating embeddings for {len(sequences)} sequences...")

    # 预处理序列
    processed_sequences = preprocess_sequences(sequences)

    # 分批处理
    seq_chunks = chunks(processed_sequences, 10)
    id_chunks = chunks(sequence_ids, 10)

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 打开输出文件
    with open(output_path, 'wb') as fh:
        with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for chunk_idx, (seq_chunk, id_chunk) in enumerate(zip(seq_chunks, id_chunks)):
                with torch.no_grad():
                    # 批量编码
                    inputs = tokenizer(seq_chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    print(f"Processing chunk {chunk_idx + 1}/{len(seq_chunks)}")

                    # 生成嵌入
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state.cpu().numpy()

                    # 处理每个序列的嵌入
                    for seq_idx, (embedding, seq_id) in enumerate(zip(embeddings, id_chunk)):
                        # 根据注意力掩码确定实际序列长度
                        seq_len = (attention_mask[seq_idx] == 1).sum().item()
                        # 去除开始和结束标记，只保留实际氨基酸的嵌入
                        seq_embedding = embedding[1:seq_len - 1]

                        # 保存嵌入 - 修复版本
                        # 使用numpy.save或直接写入数据
                        embedding_data = np.asanyarray(seq_embedding)
                        with zf.open(f"{seq_id}.npy", 'w') as buf:  # 移除forceZip64参数
                            # 直接使用numpy.save
                            np.save(buf, embedding_data)

                print(f"Done with chunk {chunk_idx + 1} of {len(seq_chunks)}")


def process_dataset(peptide_fasta, protein_fasta, output_path, model, tokenizer, device):
    """处理整个数据集"""
    # 处理肽序列
    peptide_ids, peptide_sequences = process_fasta_file(peptide_fasta)
    print(f"Found {len(peptide_sequences)} peptide sequences")

    # 处理蛋白质序列
    protein_ids, protein_sequences = process_fasta_file(protein_fasta)
    print(f"Found {len(protein_sequences)} protein sequences")

    # 合并所有序列
    all_ids = peptide_ids + protein_ids
    all_sequences = peptide_sequences + protein_sequences

    if all_sequences:
        print(f"Total sequences to process: {len(all_sequences)}")
        generate_embeddings(model, tokenizer, all_sequences, all_ids, output_path, device)
    else:
        print("No sequences found to process")


# 加载ESM2模型
print("Step 1/2 | Loading ESM2 transformer model...")

# 检查模型路径是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = EsmModel.from_pretrained(MODEL_PATH)

# 清理内存
gc.collect()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
model = model.eval()

# 处理正样本
process_dataset(
    POS_PEPTIDE_SEQ,
    POS_PROTEIN_SEQ,
    POS_EMBEDDINGS,
    model,
    tokenizer,
    device
)

# 处理负样本
process_dataset(
    NEG_PEPTIDE_SEQ,
    NEG_PROTEIN_SEQ,
    NEG_EMBEDDINGS,
    model,
    tokenizer,
    device
)

print("Embedding generation completed.")
