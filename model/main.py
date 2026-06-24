# main.py
import warnings
import torch
from config import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train import *


def main():
    if len(sys.argv) > 1:
        datasetname, rst_file, pkl_path, batchsize = sys.argv[1:]
        batchsize = int(batchsize)
    else:
        datasetname = 'receptor-peptide'
        rst_file = f'{RESULTS_PATH}/equi_results.tsv'
        pkl_path = f'{MODEL_PKL_PATH}/Equi'
        batchsize = 8


    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(MODEL_PKL_PATH, exist_ok=True)

    train(trainArgs)


if __name__ == "__main__":
    main()
