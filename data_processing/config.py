# config.py
import os

# 数据路径配置
DATA_ROOT = "C:/Users/15438/Desktop/data"

# 正样本路径
POSITIVE_DATA_PATH = "C:/Users/15438/Desktop/data/positive"
POS_PEPTIDE_SEQ = "C:/Users/15438/Desktop/data/positive/peptideseq"
POS_PROTEIN_SEQ = "C:/Users/15438/Desktop/data/positive/proteinseq"
POS_PEPTIDE_PDB = "C:/Users/15438/Desktop/data/positive/peptidestruc"
POS_PROTEIN_PDB = "C:/Users/15438/Desktop/data/positive/proteinstruc"
POS_CMAP_OUTPUT = "C:/Users/15438/Desktop/data/positive/cmap"

# 负样本路径 - 修复了NEG_PEPTIDE_SEQ路径错误
NEGATIVE_DATA_PATH = "C:/Users/15438/Desktop/data/negative"
NEG_PEPTIDE_SEQ = "C:/Users/15438/Desktop/data/negative/peptideseq"  # 修复：原错误指向positive路径
NEG_PROTEIN_SEQ = "C:/Users/15438/Desktop/data/negative/proteinseq"
NEG_PEPTIDE_PDB = "C:/Users/15438/Desktop/data/negative/peptidestruc"
NEG_PROTEIN_PDB = "C:/Users/15438/Desktop/data/negative/proteinstruc"
NEG_CMAP_OUTPUT = "C:/Users/15438/Desktop/data/negative/cmap"

# 嵌入文件路径
POS_EMBEDDINGS = "C:/Users/15438/Desktop/data/positive/embeddings.npz"
NEG_EMBEDDINGS = "C:/Users/15438/Desktop/data/negative/embeddings.npz"

# 模型路径
MODEL_PATH = "D:/PycharmProjects/PepPI/models/esm2_t33_650M_UR50D"

# 结果输出路径
RESULTS_PATH = "D:/PycharmProjects/PepPI/results"
MODEL_PKL_PATH = "D:/PycharmProjects/PepPI/model_pkl"

# 数据文件路径
ACTIONS_FILE = "D:/PycharmProjects/PepPI/data/receptor-peptide.actions.tsv"

modelArgs = {}
modelArgs['emb_dim'] = 1280  # ESM-2 650M模型的嵌入维度
modelArgs['hidden_dim'] = 256
modelArgs['output_dim'] = 128
modelArgs['num_layers'] = 3
modelArgs['task_type'] = 'binary_classification'
modelArgs['n_classes'] = 1

trainArgs = {}
trainArgs['epochs'] = 50
trainArgs['lr'] = 0.001
trainArgs['doTest'] = True
trainArgs['doSave'] = True
