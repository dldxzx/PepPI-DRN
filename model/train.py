# train.py
from graph_loader import *
from PepPI_RDN import EquivariantPPI
from config import *
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xlwt
import os
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor.cuda())
    else:
        return torch.autograd.Variable(tensor)


def validation(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_score = torch.Tensor()
    print('Make validation for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, seq1, seq2, y) in enumerate(loader):
            # 确保标签在正确的设备上
            y = y.float().to(device)

            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            # 使用PyG的批处理
            batch_G1 = Batch.from_data_list(G1)
            batch_G2 = Batch.from_data_list(G2)
            output_score = model(batch_G1, batch_G2, seq1, seq2)
            output = torch.round(output_score.squeeze(1))  # 阈值为0.5

            total_labels = torch.cat((total_labels, y.cpu()), 0)
            total_preds_score = torch.cat((total_preds_score, output_score.cpu()), 0)
            total_preds = torch.cat((total_preds, output.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_score.numpy().flatten()


def test(model, device, loader, k):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_score = torch.Tensor()
    print('Make test for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, seq1, seq2, y) in enumerate(loader):
            print('p1:', p1)
            print('p2:', p2)
            print(y)

            y = y.float().to(device)

            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            batch_G1 = Batch.from_data_list(G1)
            batch_G2 = Batch.from_data_list(G2)
            predict_score_tensor = model(batch_G1, batch_G2, seq1, seq2)


            predict_score_numpy = predict_score_tensor.cpu().numpy()

            predict_score_list = predict_score_numpy.tolist()

            predict_label_tensor = torch.round(predict_score_tensor.squeeze(1))  # 阈值为0.5
            predict_label_numpy = predict_label_tensor.cpu().numpy()

            predict_label_list = predict_label_numpy.tolist()

            result_dir = f'{RESULTS_PATH}/equi_train_5fold_cross_validation/fold{k}'
            os.makedirs(result_dir, exist_ok=True)

            file = xlwt.Workbook(encoding='utf-8')
            sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)

            sheet1.write(0, 0, "index")
            sheet1.write(0, 1, "receptor")
            sheet1.write(0, 2, "peptide")
            sheet1.write(0, 3, "label")
            sheet1.write(0, 4, "predict_score")
            sheet1.write(0, 5, "predict_label")

            # 循环填入数据
            for i in range(len(predict_score_list)):
                sheet1.write(i + 1, 0, i)  # 第1列index
                sheet1.write(i + 1, 1, str(p1[i]))  # 第2列receptor
                sheet1.write(i + 1, 2, str(p2[i]))  # 第3列peptide
                sheet1.write(i + 1, 3, str(y[i].cpu().item()))  # 第4列真实标签
                sheet1.write(i + 1, 4, str(predict_score_list[i]))  # 第5列预测分数
                sheet1.write(i + 1, 5, str(predict_label_list[i]))  # 第6列预测标签

            # 保存Excel到.py源文件同级目录
            file.save(f'{result_dir}/{batch_idx}.xls')

            total_labels = torch.cat((total_labels, y.cpu()), 0)
            total_preds_score = torch.cat((total_preds_score, predict_score_tensor.cpu()), 0)
            total_preds = torch.cat((total_preds, predict_label_tensor.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_score.numpy().flatten()


# 在 train 函数中修改优化器和学习率调度器
def train(trainArgs):
    all_protein1 = []
    all_protein2 = []
    all_Y = []

    # 存储每折的评估指标
    fold_metrics = {
        'test_acc': [],
        'test_prec': [],
        'test_recall': [],
        'test_f1': [],
        'test_auc': [],
        'test_spec': [],
        'test_mcc': [],
        'test_auprc': [],
        'test_sen': []
    }

    # 确保结果目录存在
    os.makedirs(f'{RESULTS_PATH}/equi_train_5fold_cross_validation', exist_ok=True)
    os.makedirs(MODEL_PKL_PATH, exist_ok=True)

    with open(ACTIONS_FILE, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            row = line.rstrip().split('\t')
            all_protein1.append(row[0])
            all_protein2.append(row[1])
            all_Y.append(float(row[2]))

    k = 0
    Skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for split, (train_index, test_index) in enumerate(Skf.split(all_Y, all_Y)):
        k = k + 1
        print('第', k, '折:')

        # 创建fold目录
        os.makedirs(f'{RESULTS_PATH}/equi_train_5fold_cross_validation/fold{k}', exist_ok=True)

        # train_valid_dataset
        train_valid_protein1_cv = np.array(all_protein1)[train_index]
        train_valid_protein2_cv = np.array(all_protein2)[train_index]
        train_valid_Y_cv = np.array(all_Y)[train_index]

        # test_dataset
        test_protein1_cv = np.array(all_protein1)[test_index]
        test_protein2_cv = np.array(all_protein2)[test_index]
        test_Y_cv = np.array(all_Y)[test_index]

        train_size = train_valid_protein2_cv.shape[0]
        print("训练集和验证集的蛋白质对为:", train_size)
        valid_size = int(train_size * 0.05)
        print("验证集的蛋白质对为:", valid_size)

        train_protein2_cv = np.concatenate((train_valid_protein2_cv[:int(train_size / 2 - valid_size / 2)],
                                            train_valid_protein2_cv[int(train_size / 2 + valid_size / 2):]), axis=0)
        train_protein1_cv = np.concatenate((train_valid_protein1_cv[:int(train_size / 2 - valid_size / 2)],
                                            train_valid_protein1_cv[int(train_size / 2 + valid_size / 2):]), axis=0)
        train_Y_cv = np.concatenate((train_valid_Y_cv[:int(train_size / 2 - valid_size / 2)],
                                     train_valid_Y_cv[int(train_size / 2 + valid_size / 2):]), axis=0)
        print("训练集的蛋白质对为：", train_protein2_cv.shape[0])

        valid_protein2_cv = train_valid_protein2_cv[
                            int(train_size / 2 - valid_size / 2):int(train_size / 2 + valid_size / 2)]
        valid_protein1_cv = train_valid_protein1_cv[
                            int(train_size / 2 - valid_size / 2):int(train_size / 2 + valid_size / 2)]
        valid_Y_cv = train_valid_Y_cv[int(train_size / 2 - valid_size / 2):int(train_size / 2 + valid_size / 2)]
        print("验证集的蛋白质对为：", valid_protein2_cv.shape[0])

        train_ds = MyDataset(list1=train_protein1_cv, list2=train_protein2_cv, list3=train_Y_cv)
        valid_ds = MyDataset(list1=valid_protein1_cv, list2=valid_protein2_cv, list3=valid_Y_cv)
        test_ds = MyDataset(list1=test_protein1_cv, list2=test_protein2_cv, list3=test_Y_cv)

        train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, drop_last=False, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=batchsize, shuffle=True, drop_last=False, collate_fn=collate)
        validation_loader = DataLoader(valid_ds, batch_size=batchsize, shuffle=True, drop_last=False,
                                       collate_fn=collate)

        train_losses = []
        train_accs = []
        max_acc = 0

        # 使用新的等变模型
        trainArgs['model'] = EquivariantPPI(modelArgs).to(device)

        # 使用 AdamW 优化器
        trainArgs['optimizer'] = AdamW(
            trainArgs['model'].parameters(),
            lr=trainArgs['lr'],
            weight_decay=trainArgs['weight_decay'],
            betas=(0.9, 0.999)
        )

        optimizer = trainArgs['optimizer']

        # 使用 CosineAnnealingWarmRestarts 学习率调度器
        trainArgs['lr_scheduler'] = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # 初始周期长度
            T_mult=2,  # 周期倍增因子
            eta_min=1e-6  # 最小学习率
        )

        criterion = torch.nn.BCELoss()
        attention_model = trainArgs['model']

        # 学习率预热函数
        def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
            def f(x):
                if x >= warmup_iters:
                    return 1
                alpha = float(x) / warmup_iters
                return warmup_factor * (1 - alpha) + alpha

            return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

        # 创建预热调度器
        warmup_scheduler = warmup_lr_scheduler(
            optimizer,
            warmup_iters=len(train_loader) * trainArgs['warmup_epochs'],
            warmup_factor=1e-3
        )

        # 在 train.py 的训练循环中修改以下部分
        # 修改 train.py 中的训练循环部分
        for i in range(trainArgs['epochs']):
            print("Running EPOCH", i + 1)
            total_loss = 0
            n_batches = 0
            correct = 0

            # 梯度累积参数
            accumulation_steps = 4  # 每4个批次更新一次梯度

            attention_model.train()
            for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, seq1, seq2, y) in enumerate(train_loader):
                # 确保标签在正确的设备上
                y = y.float().to(device)

                # 梯度累积：只有在积累完成时才清零梯度
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()

                # 使用PyG的批处理
                batch_G1 = Batch.from_data_list(G1)
                batch_G2 = Batch.from_data_list(G2)

                # 传入图特征和序列特征（现在同时传入肽和蛋白质序列）
                # 假设 seq1 是肽序列，seq2 是蛋白质序列
                y_pred = attention_model(batch_G1, batch_G2, seq1, seq2)


                loss = criterion(y_pred.squeeze(1), y)

                # 梯度累积：除以累积步数
                loss = loss / accumulation_steps
                loss.backward()

                # 梯度裁剪
                if trainArgs.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        attention_model.parameters(),
                        trainArgs['gradient_clip']
                    )

                # 梯度累积：只有在积累完成时才更新参数
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()

                    # 学习率调度
                    if i < trainArgs['warmup_epochs']:
                        # 预热阶段使用预热调度器
                        warmup_scheduler.step()
                    else:
                        # 预热后使用余弦退火调度器
                        trainArgs['lr_scheduler'].step(i - trainArgs['warmup_epochs'] + batch_idx / len(train_loader))

                correct += torch.eq(torch.round(y_pred.squeeze(1)), y).data.sum()
                total_loss += loss.data * accumulation_steps  # 恢复原始损失值
                n_batches += 1

            avg_loss = total_loss / n_batches
            acc = correct.cpu().numpy() / (len(train_loader.dataset))

            train_losses.append(avg_loss)
            train_accs.append(acc)

            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {i + 1}: train avg_loss = {avg_loss:.4f}, train ACC = {acc:.4f}, LR = {current_lr:.6f}")

            # validation
            attention_model.eval()
            total_labels, total_preds, total_preds_score = validation(attention_model, device, validation_loader)
            validation_acc = accuracy_score(total_labels, total_preds)
            validation_prec = precision_score(total_labels, total_preds)
            validation_recall = recall_score(total_labels, total_preds)
            validation_f1 = f1_score(total_labels, total_preds)
            validation_auc = roc_auc_score(total_labels, total_preds_score)
            validation_auprc = average_precision_score(total_labels, total_preds_score)

            con_matrix = confusion_matrix(total_labels, total_preds)
            validation_sen = con_matrix[1, 1] / (con_matrix[1, 1] + con_matrix[1, 0])
            validation_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            validation_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
                    ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (con_matrix[0][
                                                                                                          0] +
                                                                                                      con_matrix[
                                                                                                          0][
                                                                                                          1]) * (
                             con_matrix[
                                 0][
                                 0] +
                             con_matrix[
                                 1][
                                 0])) ** 0.5)
            print("acc: ", validation_acc, " ; prec: ", validation_prec, " ; recall: ", validation_recall, " ; f1: ",
                  validation_f1, " ; auc: ", validation_auc, " ; spec:", validation_spec, " ; mcc: ", validation_mcc,
                  " ; auprc: ", validation_auprc, " ; sensitivity: ", validation_sen)

            with open(rst_file.replace('GAT', 'Equi'), 'a+') as fp:
                fp.write('epoch:' + str(i + 1) + '\ttrainacc=' + str(acc) + '\ttrainloss=' + str(
                    avg_loss.item()) + '\tacc=' + str(validation_acc) + '\tprec=' + str(
                    validation_prec) + '\tf1=' + str(validation_f1) + '\tauc=' + str(validation_auc) + '\tspec=' + str(
                    validation_spec) + '\tmcc=' + str(validation_mcc) + '\tauprc=' + str(
                    validation_auprc) + '\tsensitivity=' + str(validation_sen) + '\n')

            if validation_acc > max_acc:
                max_acc = validation_acc
                print("save model")
                torch.save(attention_model.state_dict(), pkl_path.replace('GAT', 'Equi') + '.pkl')

        # test
        attention_model = trainArgs['model']
        checkpoint = torch.load(f'{MODEL_PKL_PATH}/Equi.pkl')
        attention_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)

        total_labels, total_preds, total_preds_score = test(attention_model, device, test_loader, k)
        test_acc = accuracy_score(total_labels, total_preds)
        test_prec = precision_score(total_labels, total_preds)
        test_recall = recall_score(total_labels, total_preds)
        test_f1 = f1_score(total_labels, total_preds)
        test_auc = roc_auc_score(total_labels, total_preds_score)
        con_matrix = confusion_matrix(total_labels, total_preds)
        test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
        test_auprc = average_precision_score(total_labels, total_preds_score)
        test_sen = con_matrix[1, 1] / (con_matrix[1, 1] + con_matrix[1, 0])
        test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
                ((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[
                    1][
                    0]) * (
                         con_matrix[
                             0][
                             0] +
                         con_matrix[
                             0][
                             1]) * (
                         con_matrix[
                             0][
                             0] +
                         con_matrix[
                             1][
                             0])) ** 0.5)
        print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ",
              test_auc, " ; spec:", test_spec, " ; mcc: ", test_mcc, " ; auprc: ", test_auprc, " ; sensitivity: ",
              test_sen)

        # 保存当前折的指标
        fold_metrics['test_acc'].append(test_acc)
        fold_metrics['test_prec'].append(test_prec)
        fold_metrics['test_recall'].append(test_recall)
        fold_metrics['test_f1'].append(test_f1)
        fold_metrics['test_auc'].append(test_auc)
        fold_metrics['test_spec'].append(test_spec)
        fold_metrics['test_mcc'].append(test_mcc)
        fold_metrics['test_auprc'].append(test_auprc)
        fold_metrics['test_sen'].append(test_sen)

        with open(rst_file.replace('GAT', 'Equi'), 'a+') as fp:
            fp.write(
                'acc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) + '\tf1=' + str(
                    test_f1) + '\tauc=' + str(test_auc) + '\tspec=' + str(test_spec) + '\tmcc=' + str(
                    test_mcc) + '\tauprc=' + str(test_auprc) + '\tsensitivity=' + str(test_sen) + '\n')

    # 计算并打印五折交叉验证的平均指标
    print("\n" + "=" * 50)
    print("五折交叉验证结果汇总:")
    print("=" * 50)

    avg_acc = sum(fold_metrics['test_acc']) / len(fold_metrics['test_acc'])
    avg_prec = sum(fold_metrics['test_prec']) / len(fold_metrics['test_prec'])
    avg_recall = sum(fold_metrics['test_recall']) / len(fold_metrics['test_recall'])
    avg_f1 = sum(fold_metrics['test_f1']) / len(fold_metrics['test_f1'])
    avg_auc = sum(fold_metrics['test_auc']) / len(fold_metrics['test_auc'])
    avg_spec = sum(fold_metrics['test_spec']) / len(fold_metrics['test_spec'])
    avg_mcc = sum(fold_metrics['test_mcc']) / len(fold_metrics['test_mcc'])
    avg_auprc = sum(fold_metrics['test_auprc']) / len(fold_metrics['test_auprc'])
    avg_sen = sum(fold_metrics['test_sen']) / len(fold_metrics['test_sen'])

    print(f"平均准确率 (ACC): {avg_acc:.4f}")
    print(f"平均精确率 (Precision): {avg_prec:.4f}")
    print(f"平均召回率 (Recall): {avg_recall:.4f}")
    print(f"平均F1分数: {avg_f1:.4f}")
    print(f"平均AUC: {avg_auc:.4f}")
    print(f"平均特异性 (Specificity): {avg_spec:.4f}")
    print(f"平均MCC: {avg_mcc:.4f}")
    print(f"平均AUPRC: {avg_auprc:.4f}")
    print(f"平均敏感性 (Sensitivity): {avg_sen:.4f}")

    print("\n各折详细结果:")
    for fold in range(1, 6):
        print(f"第{fold}折 - ACC: {fold_metrics['test_acc'][fold - 1]:.4f}, "
              f"Prec: {fold_metrics['test_prec'][fold - 1]:.4f}, "
              f"Recall: {fold_metrics['test_recall'][fold - 1]:.4f}, "
              f"F1: {fold_metrics['test_f1'][fold - 1]:.4f}, "
              f"AUC: {fold_metrics['test_auc'][fold - 1]:.4f}")

    print("=" * 50)
