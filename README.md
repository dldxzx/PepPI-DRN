# PepPI-DRN: Protein–Peptide Interaction Prediction Model
This project implements a deep learning model, PepPI-DRN, based on an Equivariant Graph Neural Network (EGNN) and a sequence CNN, for predicting interactions between proteins and peptides.
## Project Overview
PepPI-DRN is a deep learning model designed to predict protein–peptide interactions. It integrates:
- **Sequence features**: extracted via a multi‑scale CNN that captures local patterns in amino acid sequences.
- **Structural features**: captured by an Equivariant Graph Neural Network (EGNN) that processes 3D spatial information.
- **Multimodal fusion**: combines sequence and structure features to achieve high‑accuracy predictions.

### Environment Setup

| Package | Version |
|------|------|
| Python | 3.9 |
| NumPy | 2.0.2 |
| torch | 2.8.0 |
| torch‑geometric | 2.6.1 |
| Biopython | 1.85 |
| torchaudio | 2.8.0 |
| torchvision | 0.23.0 |
| transformers | 4.56.1 |
| scikit‑learn | 1.6.1 |
| scipy | 1.13.1 |
## Installing Dependencies


```sh
pip install torch torch-geometric transformers scipy numpy matplotlib biopython pandas openpyxl xlwt
```

### Data Preparation

1. Sequence data: Prepare FASTA‑format sequence files for proteins and peptides.
2. Structure data: Extract PDB files for proteins and peptides.
3. Label file: Create a TSV file with each line in the：
- protein_id
- peptide_id
- label

### Generating Contact Maps
```sh
python generate_contact_map.py
```

### Generating Sequence Embeddings
```sh
python generate_embeddings.py
```
### Model Training
Edit config.py, then run:
```sh
python main.py
```
### Model Prediction
Create a TSV file where each line follows: protein_id peptide_id label (the label can be arbitrary, used only for record keeping), then run:
```sh
python predict.py
```






