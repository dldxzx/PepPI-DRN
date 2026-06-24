# -*- coding: utf-8 -*-
import Bio.PDB
import numpy as np
import os
from Bio import SeqIO
from config import *


def get_center_atom(residue):
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom


def calc_residue_dist(residue_one, residue_two):
    """Returns the distance between two residues based on their center atoms"""
    c_atom1 = get_center_atom(residue_one)
    c_atom2 = get_center_atom(residue_two)
    diff_vector = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two):
    """Returns a matrix of distances between two chains"""
    residue_len = 0
    for row, residue_one in enumerate(chain_one):
        hetfield = residue_one.get_id()[0]
        hetname = residue_one.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            residue_len = residue_len + 1
    answer = np.zeros((residue_len, residue_len), np.float64)
    x = -1
    for residue_one in chain_one:
        y = -1
        hetfield1 = residue_one.get_id()[0]
        hetname1 = residue_one.get_resname()
        if hetfield1 == ' ' and hetname1 in aa_codes.keys():
            x = x + 1
            for residue_two in chain_two:
                hetfield2 = residue_two.get_id()[0]
                hetname2 = residue_two.get_resname()
                if hetfield2 == ' ' and hetname2 in aa_codes.keys():
                    y = y + 1
                    answer[x, y] = calc_residue_dist(residue_one, residue_two)
    for i in range(residue_len):
        answer[i, i] = 100
    return answer


def extract_sequence_from_pdb(structure, chain_id):
    """从PDB结构中提取氨基酸序列"""
    sequence = ""
    chain = structure[0][chain_id]
    for residue in chain:
        hetfield = residue.get_id()[0]
        hetname = residue.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            sequence += aa_codes[hetname]
    return sequence


def find_sequence_for_pdb(pdb_id, chain_id, seq_directory):
    """根据PDB ID和链ID在对应目录中查找完全匹配的序列文件"""
    # 构造精确匹配的文件名
    possible_filenames = [
        f"{pdb_id}_{chain_id}.fasta",
        f"{pdb_id}_{chain_id}.fa",
        f"{pdb_id}_{chain_id}.txt",
        f"{pdb_id}_{chain_id}.seq"
    ]

    # 如果目录不存在，返回None
    if not os.path.exists(seq_directory):
        return None

    # 遍历目录中的所有文件，寻找精确匹配
    for filename in os.listdir(seq_directory):
        if filename in possible_filenames:
            seq_file_path = os.path.join(seq_directory, filename)
            try:
                # 解析FASTA文件
                records = list(SeqIO.parse(seq_file_path, "fasta"))
                if records:
                    # 返回第一个记录的序列
                    return str(records[0].seq)
            except Exception as e:
                print(f"Error parsing {seq_file_path}: {e}")
                continue

    return None


def find_best_chain_match(model, expected_chain_id):
    """
    在PDB模型中查找最匹配的链ID（支持大小写不敏感匹配）
    """
    available_chains = [chain.id for chain in model]

    # 精确匹配
    if expected_chain_id in available_chains:
        return expected_chain_id

    # 大小写不敏感匹配
    for chain_id in available_chains:
        if chain_id.lower() == expected_chain_id.lower():
            return chain_id

    # 如果没有找到匹配，返回None
    return None


def calc_contact_map(pdb_id, chain_id, data_root_path):
    # 首先尝试精确匹配
    pdb_path = os.path.join(data_root_path, f"{pdb_id}_{chain_id}.pdb")

    # 如果文件不存在，尝试其他可能的大小写变体
    if not os.path.exists(pdb_path):
        # 尝试不同的大小写组合
        variants = [
            f"{pdb_id}_{chain_id.lower()}.pdb",
            f"{pdb_id}_{chain_id.upper()}.pdb",
            f"{pdb_id}_{chain_id.capitalize()}.pdb",
            f"{pdb_id}_{chain_id}.pdb"
        ]

        pdb_path = None
        for variant in variants:
            path = os.path.join(data_root_path, variant)
            if os.path.exists(path):
                pdb_path = path
                # 更新实际的链ID
                actual_chain_id = variant.replace(f"{pdb_id}_", "").replace(".pdb", "")
                chain_id = actual_chain_id
                break

        if pdb_path is None:
            raise FileNotFoundError(f"PDB file not found for {pdb_id}_{chain_id} (tried: {variants})")

    structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
    model = structure[0]

    # 查找最佳匹配的链
    final_chain_id = find_best_chain_match(model, chain_id)
    if final_chain_id is None:
        available_chains = [chain.id for chain in model]
        raise ValueError(f"Chain {chain_id} not found in {pdb_id}. Available chains: {available_chains}")

    dist_matrix = calc_dist_matrix(model[final_chain_id], model[final_chain_id])
    contact_map = (dist_matrix < 8.0).astype(np.int_)

    # 从PDB中提取序列用于验证
    pdb_sequence = extract_sequence_from_pdb(structure, final_chain_id)
    return pdb_sequence, contact_map


aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
}

# 创建输出目录
os.makedirs(POS_CMAP_OUTPUT, exist_ok=True)
os.makedirs(NEG_CMAP_OUTPUT, exist_ok=True)

# 处理所有PDB文件
print("Processing all PDB files...")

# 处理正样本肽
contact_map_count = 0
if os.path.exists(POS_PEPTIDE_PDB):
    print("Processing positive peptide samples...")
    for filename in os.listdir(POS_PEPTIDE_PDB):
        if filename.endswith('.pdb'):
            pdb_file = filename.replace('.pdb', '')
            pdb_id, chain_id = pdb_file.split('_', 1)

            try:
                print(f"Processing: {pdb_id}_{chain_id}")
                pdb_sequence, contact_map = calc_contact_map(pdb_id, chain_id, POS_PEPTIDE_PDB)
                print(f"PDB sequence length: {len(pdb_sequence)}")
                print(f"Contact map shape: {contact_map.shape}")

                # 只在对应的肽序列目录中查找完全匹配的文件
                sequence = find_sequence_for_pdb(pdb_id, chain_id, POS_PEPTIDE_SEQ)

                if sequence:
                    contact_file = os.path.join(POS_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=sequence, contact=contact_map)
                    print(f"Saved contact map to: {contact_file}")
                    contact_map_count += 1
                else:
                    print(f"Warning: No sequence file found for {pdb_id}_{chain_id}")
                    # 使用PDB中提取的序列作为备选
                    contact_file = os.path.join(POS_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=pdb_sequence, contact=contact_map)
                    print(f"Saved contact map with PDB sequence to: {contact_file}")
                    contact_map_count += 1

            except Exception as e:
                print(f"Error processing {pdb_id}_{chain_id}: {e}")

print(f"Positive peptide contact maps generated: {contact_map_count}")

# 处理正样本蛋白质
contact_map_count = 0
if os.path.exists(POS_PROTEIN_PDB):
    print("Processing positive protein samples...")
    for filename in os.listdir(POS_PROTEIN_PDB):
        if filename.endswith('.pdb'):
            pdb_file = filename.replace('.pdb', '')
            pdb_id, chain_id = pdb_file.split('_', 1)

            try:
                print(f"Processing: {pdb_id}_{chain_id}")
                pdb_sequence, contact_map = calc_contact_map(pdb_id, chain_id, POS_PROTEIN_PDB)
                print(f"PDB sequence length: {len(pdb_sequence)}")
                print(f"Contact map shape: {contact_map.shape}")

                # 只在对应的蛋白质序列目录中查找完全匹配的文件
                sequence = find_sequence_for_pdb(pdb_id, chain_id, POS_PROTEIN_SEQ)

                if sequence:
                    contact_file = os.path.join(POS_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=sequence, contact=contact_map)
                    print(f"Saved contact map to: {contact_file}")
                    contact_map_count += 1
                else:
                    print(f"Warning: No sequence file found for {pdb_id}_{chain_id}")
                    # 使用PDB中提取的序列作为备选
                    contact_file = os.path.join(POS_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=pdb_sequence, contact=contact_map)
                    print(f"Saved contact map with PDB sequence to: {contact_file}")
                    contact_map_count += 1

            except Exception as e:
                print(f"Error processing {pdb_id}_{chain_id}: {e}")

print(f"Positive protein contact maps generated: {contact_map_count}")

# 处理负样本肽
contact_map_count = 0
if os.path.exists(NEG_PEPTIDE_PDB):
    print("Processing negative peptide samples...")
    for filename in os.listdir(NEG_PEPTIDE_PDB):
        if filename.endswith('.pdb'):
            pdb_file = filename.replace('.pdb', '')
            pdb_id, chain_id = pdb_file.split('_', 1)

            try:
                print(f"Processing: {pdb_id}_{chain_id}")
                pdb_sequence, contact_map = calc_contact_map(pdb_id, chain_id, NEG_PEPTIDE_PDB)
                print(f"PDB sequence length: {len(pdb_sequence)}")
                print(f"Contact map shape: {contact_map.shape}")

                # 只在对应的肽序列目录中查找完全匹配的文件
                sequence = find_sequence_for_pdb(pdb_id, chain_id, NEG_PEPTIDE_SEQ)

                if sequence:
                    contact_file = os.path.join(NEG_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=sequence, contact=contact_map)
                    print(f"Saved contact map to: {contact_file}")
                    contact_map_count += 1
                else:
                    print(f"Warning: No sequence file found for {pdb_id}_{chain_id}")
                    # 使用PDB中提取的序列作为备选
                    contact_file = os.path.join(NEG_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=pdb_sequence, contact=contact_map)
                    print(f"Saved contact map with PDB sequence to: {contact_file}")
                    contact_map_count += 1

            except Exception as e:
                print(f"Error processing {pdb_id}_{chain_id}: {e}")

print(f"Negative peptide contact maps generated: {contact_map_count}")

# 处理负样本蛋白质
contact_map_count = 0
if os.path.exists(NEG_PROTEIN_PDB):
    print("Processing negative protein samples...")
    for filename in os.listdir(NEG_PROTEIN_PDB):
        if filename.endswith('.pdb'):
            pdb_file = filename.replace('.pdb', '')
            pdb_id, chain_id = pdb_file.split('_', 1)

            try:
                print(f"Processing: {pdb_id}_{chain_id}")
                pdb_sequence, contact_map = calc_contact_map(pdb_id, chain_id, NEG_PROTEIN_PDB)
                print(f"PDB sequence length: {len(pdb_sequence)}")
                print(f"Contact map shape: {contact_map.shape}")

                # 只在对应的蛋白质序列目录中查找完全匹配的文件
                sequence = find_sequence_for_pdb(pdb_id, chain_id, NEG_PROTEIN_SEQ)

                if sequence:
                    contact_file = os.path.join(NEG_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=sequence, contact=contact_map)
                    print(f"Saved contact map to: {contact_file}")
                    contact_map_count += 1
                else:
                    print(f"Warning: No sequence file found for {pdb_id}_{chain_id}")
                    # 使用PDB中提取的序列作为备选
                    contact_file = os.path.join(NEG_CMAP_OUTPUT, f"{pdb_id}_{chain_id}.npz")
                    np.savez(contact_file, seq=pdb_sequence, contact=contact_map)
                    print(f"Saved contact map with PDB sequence to: {contact_file}")
                    contact_map_count += 1

            except Exception as e:
                print(f"Error processing {pdb_id}_{chain_id}: {e}")

print(f"Negative protein contact maps generated: {contact_map_count}")

print("Contact map generation completed.")
