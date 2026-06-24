PepPI-DRN  蛋白质-肽相互作用预测模型
本项目实现了一个基于 等变图神经网络 (EGNN) 和 序列 CNN 的深度学习模型PepPI-DRN，用于预测蛋白质与肽之间的相互作用。
项目简介
本项目实现了一个深度学习模型PepPI-DRN，用于预测蛋白质与肽段之间的相互作用。模型结合了：
- **序列特征**：通过多尺度 CNN 提取氨基酸序列的局部模式
- **结构特征**：通过等变图神经网络（EGNN）捕获三维空间结构信息
- **多模态融合**：整合序列和结构特征进行高精度预测

环境配置
Python	3.9
Numpy	2.0.2
Biopython	1.85
torch   2.8.0 
torch-geometric  2.6.1 
torchaudio 	2.8.0 
torchvision 	0.23.0 
transformers	4.56.1 
scikit-learn	1.6.1
scipy	1.13.1

安装依赖库
pip install torch torch-geometric transformers scipy numpy matplotlib biopython pandas openpyxl xlwt

数据准备
1. **序列数据**：准备蛋白质和肽段的 FASTA 格式序列文件
2. **结构数据**：准备蛋白质和肽段的PDB 文件提取
3. **标签文件**：创建tsv文件，每行格式为：protein_id  peptide_id  label
生成接触图
python generate_contact_map.py
生成序列嵌入
python generate_embeddings.py

模型训练
编辑 config.py后
python main.py

模型预测
创建一个 TSV 文件，每行格式：protein_id  peptide_id  label（label可随意，仅用于记录）
python predict.py

