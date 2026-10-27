import torch
import numpy as np
import torch.utils.data as Data
from scipy.io import loadmat
from parameters import *

def load_data(file_path):
    data = loadmat(file_path)

    ground_truth = data['all_beam_numbers'].reshape(-1, NUM_SAMPLES, 1)
    observed = data['all_coords_measurement'].reshape(-1, NUM_SAMPLES, 3)
    real_value = data['all_trajectories'].reshape(-1, NUM_SAMPLES, 3)
    signal = data['all_received'].reshape(-1, NUM_SAMPLES, 2)

    return ground_truth, observed, real_value, signal

def prepare_datasets(ground_truth, observed, real_value):
    samples_enc_coor, samples_enc_grid, samples_dec, labels_true, offset_true = [], [], [], [], []

    num_obj, num_samples, input_dim = ground_truth.shape[0], ground_truth.shape[1], observed.shape[2]

    for i in range(num_obj):
        for j in range(num_samples - WINDOW_SIZE - PREDICT_SIZE):
            samples_enc_coor.append(np.squeeze(observed[i,j:j + WINDOW_SIZE,:]))
            samples_enc_grid.append(np.squeeze(ground_truth[i,j:j + WINDOW_SIZE,:]))
            dec_in = np.zeros_like(ground_truth[i,j + WINDOW_SIZE:j + WINDOW_SIZE + PREDICT_SIZE,:])
            samples_dec.append(np.squeeze(np.concatenate([ground_truth[i,j + WINDOW_SIZE-1:j + WINDOW_SIZE,:], dec_in], axis=0)))
            labels_true.append(np.squeeze(ground_truth[i,j + WINDOW_SIZE:j+WINDOW_SIZE+PREDICT_SIZE,:]))
            offset_true.append(np.squeeze(real_value[i,j + WINDOW_SIZE:j+WINDOW_SIZE+1,:]))

    # 转换为 PyTorch Tensor
    samples_enc_coor = torch.tensor(np.array(samples_enc_coor), dtype=torch.float32)
    samples_enc_grid = torch.tensor(np.array(samples_enc_grid), dtype=torch.long)
    samples_dec = torch.tensor(np.array(samples_dec), dtype=torch.long)
    labels_true = torch.tensor(np.array(labels_true), dtype=torch.long)
    if PREDICT_SIZE ==1:
        labels_true = labels_true.unsqueeze(1)
    offset_true = torch.tensor(np.array(offset_true), dtype=torch.float32)

    # 划分数据集
    # 划分训练集，测试集 按9:1划分
    train_rate, vali_rate, test_rate=0.9,0.05,0.05
    train_vali_boundary = int(samples_enc_coor.shape[0] * train_rate)
    vali_test_boundary = int(samples_enc_coor.shape[0] * (train_rate + vali_rate))
    train_samples_enc_coor = samples_enc_coor[:train_vali_boundary, :].to(DEVICE)
    train_samples_enc_grid = samples_enc_grid[:train_vali_boundary, :].to(DEVICE)
    train_samples_dec = samples_dec[:train_vali_boundary, :].to(DEVICE)
    train_labels = labels_true[:train_vali_boundary, :].to(DEVICE)
    train_offset = offset_true[:train_vali_boundary, :].to(DEVICE)

    vali_samples_enc_coor = samples_enc_coor[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_samples_enc_grid = samples_enc_grid[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_samples_dec = samples_dec[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_labels = labels_true[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_offset = offset_true[train_vali_boundary:vali_test_boundary, :].to(DEVICE)

    test_samples_enc_coor = samples_enc_coor[vali_test_boundary:, :].to(DEVICE)
    test_samples_enc_grid = samples_enc_grid[vali_test_boundary:, :].to(DEVICE)
    test_samples_dec = samples_dec[vali_test_boundary:, :].to(DEVICE)
    test_labels = labels_true[vali_test_boundary:, :].to(DEVICE)
    print("Train dataset number: ",train_samples_enc_coor.shape)

    train_dataset = Data.TensorDataset(train_samples_enc_coor, train_samples_enc_grid, train_samples_dec, train_labels, train_offset)
    vali_dataset = Data.TensorDataset(vali_samples_enc_coor, vali_samples_enc_grid, vali_samples_dec, vali_labels, vali_offset)
    test_dataset = Data.TensorDataset(test_samples_enc_coor, test_samples_enc_grid, test_samples_dec, test_labels)

    return train_dataset, vali_dataset, test_dataset

def get_dataloaders(file_path):
    ground_truth, observed, real_value, _ = load_data(file_path)
    train_dataset, val_dataset, test_dataset = prepare_datasets(ground_truth, observed, real_value)

    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader