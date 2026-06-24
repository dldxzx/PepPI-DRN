import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from PepPI_RDN import EquivariantPPI
from graph_loader import MyDataset, collate
from config import *
import os


def predict(model, device, loader, output_file):
    """进行预测并保存结果"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, seq1, seq2, _) in enumerate(loader):
            try:
                # 确保数据在正确的设备上
                batch_G1 = Batch.from_data_list(G1)
                batch_G2 = Batch.from_data_list(G2)

                # 进行预测（修复参数数量）
                predict_score_tensor = model(batch_G1, batch_G2, seq1, seq2)
                predict_scores = predict_score_tensor.cpu().numpy()

                # 保存结果
                for i in range(len(p1)):
                    predictions.append({
                        'protein_id': p1[i],
                        'peptide_id': p2[i],
                        'interaction_score': float(predict_scores[i][0]),
                        'predicted_interaction': int(predict_scores[i][0] > 0.5)
                    })

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches")

                # 分批保存结果以防内存溢出
                if len(predictions) >= 10000:
                    df_predictions = pd.DataFrame(predictions)
                    df_predictions.to_csv(f"{output_file}_{batch_idx}.csv", index=False)
                    predictions = []

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue  # 跳过出错的批次

    # 保存剩余结果
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(f"{output_file}_final.csv", index=False)

    return predictions


def main():
    # 确保结果目录存在
    os.makedirs(PREDICTION_RESULTS_PATH, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    try:
        checkpoint = torch.load(f'{MODEL_PKL_PATH}/final_model_complete.pth')
        model = EquivariantPPI(modelArgs).to(device)
        model.load_state_dict(torch.load(f'{MODEL_PKL_PATH}/final_model_complete.pth'))
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 准备预测数据
    all_protein_ids = []
    all_peptide_ids = []
    all_labels = []

    try:
        with open(PREDICTION_PAIRS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Skipping invalid line: {line}")
                    continue
                protein_id, peptide_id, label = parts
                all_protein_ids.append(protein_id)
                all_peptide_ids.append(peptide_id)
                all_labels.append(int(label))
    except Exception as e:
        print(f"Failed to read prediction pairs file: {e}")
        return

    # 创建数据集
    try:
        predict_dataset = MyDataset(
            list1=all_protein_ids,
            list2=all_peptide_ids,
            list3=all_labels
        )
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    # 创建数据加载器
    try:
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate
        )
    except Exception as e:
        print(f"Failed to create data loader: {e}")
        return

    # 进行预测
    print("Starting prediction...")
    try:
        predict(model, device, predict_loader, f'{PREDICTION_RESULTS_PATH}/predictions')
        print(f"Prediction completed. Results saved to {PREDICTION_RESULTS_PATH}")
    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
