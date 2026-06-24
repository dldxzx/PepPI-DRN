# model/config.py
import os

# 数据路径配置
DATA_ROOT = "/home/dldx/dwf/data"

# 正样本路径
POSITIVE_DATA_PATH = "/home/dldx/dwf/data/positive"
POS_PEPTIDE_SEQ = "/home/dldx/dwf/data/positive/peptideseq"
POS_PROTEIN_SEQ = "/home/dldx/dwf/data/positive/proteinseq"
POS_PEPTIDE_PDB = "/home/dldx/dwf/data/positive/peptidestruc"
POS_PROTEIN_PDB = "/home/dldx/dwf/data/positive/proteinstruc"
POS_CMAP_OUTPUT = "/home/dldx/dwf/data/positive/cmap"

# 负样本路径
NEGATIVE_DATA_PATH = "/home/dldx/dwf/data/negative"
NEG_PEPTIDE_SEQ = "/home/dldx/dwf/data/negative/peptideseq"
NEG_PROTEIN_SEQ = "/home/dldx/dwf/data/negative/proteinseq"
NEG_PEPTIDE_PDB = "/home/dldx/dwf/data/negative/peptidestruc"
NEG_PROTEIN_PDB = "/home/dldx/dwf/data/negative/proteinstruc"
NEG_CMAP_OUTPUT = "/home/dldx/dwf/data/negative/cmap"

# 嵌入文件路径
POS_EMBEDDINGS = "/home/dldx/dwf/data/positive/embeddings.npz"
NEG_EMBEDDINGS = "/home/dldx/dwf/data/negative/embeddings.npz"

# 模型路径
MODEL_PATH = "/home/dldx/dwf/PepPI/models/esm2_t33_650M_UR50D"

# 结果输出路径
RESULTS_PATH = "/home/dldx/dwf/PepPI/results"
MODEL_PKL_PATH = "/home/dldx/dwf/PepPI/model_pkl"

# 数据文件路径
ACTIONS_FILE = "/home/dldx/dwf/PepPI/data/receptor-peptide.actions.tsv"

modelArgs = {}
modelArgs['emb_dim'] = 1280  # ESM-2 650M模型的嵌入维度
modelArgs['hidden_dim'] = 256
modelArgs['output_dim'] = 128
modelArgs['num_layers'] = 3
modelArgs['task_type'] = 'binary_classification'
modelArgs['n_classes'] = 1
modelArgs['seq_processor'] = 'transformer'  # 或 'cnn'

trainArgs = {}
trainArgs['epochs'] = 100  # 增加到100轮
trainArgs['lr'] = 0.001
trainArgs['weight_decay'] = 0.01  # 添加权重衰减
trainArgs['warmup_epochs'] = 10   # 预热轮数
trainArgs['doTest'] = True
trainArgs['doSave'] = True
trainArgs['gradient_clip'] = 1.0  # 梯度裁剪
