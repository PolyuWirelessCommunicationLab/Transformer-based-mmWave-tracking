import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import pandas as pd
import math
import time
from bert_model import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import preprocessing
import torch.utils.data
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from utils.tools import *
import os
from torch.optim import lr_scheduler
# from parameters import *
from parameters import *
from scipy.io import loadmat, savemat
# from aoa_estimation_new import multi_beam_aoa_estimation

if torch.cuda.is_available():
    print("GPU is available")
    DEVICE = torch.device("cuda")
else:
    print("GPU is not available")  
    DEVICE = torch.device("cpu")

samples_enc_coor = []
samples_enc_grid = []
samples_fil_coor = []
samples_fil_grid = []
label_true = []
offset_true = []

# 准备进行数据的处理（划分样本和标签，训练集与测试集的划分）
# 读取csv表格数据
data = loadmat(RE_DATA_PATH)
ground_truth = data['all_beam_numbers'].reshape(-1, NUM_SAMPLES, 1)
observed = data['all_coords_measurement'].reshape(-1, NUM_SAMPLES, 3)
# observed = data['all_received'].reshape(-1, NUM_SAMPLES, 2)
signal = data['all_received'].reshape(-1, NUM_SAMPLES, 2)
real_value = data['all_trajectories'].reshape(-1, NUM_SAMPLES, 3)

tgt_vocab_size=NUM_CLASSES
# tgt_vocab_size = 1601
print("target vocab size:", NUM_CLASSES)
print("velocity:", VELOCITY)
src_len=WINDOW_SIZE
tgt_len=WINDOW_SIZE

# num_obj, num_samples, input_dim = ground_truth.shape[0], ground_truth.shape[1], observed.shape[2]

num_obj, num_samples, input_dim = ground_truth.shape[0], ground_truth.shape[1], observed.shape[2]
print("num_obj:", num_obj, "num_samples:", num_samples)

for i in range(num_obj):
    for j in range(num_samples - WINDOW_SIZE - PREDICT_SIZE):
        # samples_enc_coor.append(np.squeeze(observed[i,j:j + WINDOW_SIZE,:]))
        # samples_enc_grid.append(np.squeeze(ground_truth[i,j:j + WINDOW_SIZE,:]))
        coor_in = np.zeros_like(observed[i,j + WINDOW_SIZE:j + WINDOW_SIZE + PREDICT_SIZE,:])
        grid_in = np.zeros_like(ground_truth[i,j + WINDOW_SIZE:j + WINDOW_SIZE + PREDICT_SIZE,:])
        samples_enc_coor.append(np.squeeze(np.concatenate([observed[i,j:j + WINDOW_SIZE,:], coor_in], axis=0)))
        samples_enc_grid.append(np.squeeze(np.concatenate([ground_truth[i,j:j + WINDOW_SIZE,:], grid_in], axis=0)))
            
        label_true.append(np.squeeze(ground_truth[i,j + WINDOW_SIZE:j+WINDOW_SIZE + PREDICT_SIZE,:]))
        offset_true.append(np.squeeze(real_value[i,j:j+WINDOW_SIZE+PREDICT_SIZE,:]))

# 转换为 PyTorch Tensor
samples_enc_coor = torch.tensor(np.array(samples_enc_coor), dtype=torch.float32)
samples_enc_grid = torch.tensor(np.array(samples_enc_grid), dtype=torch.long)
label_true = torch.tensor(np.array(label_true), dtype=torch.long)
offset_true = torch.tensor(np.array(offset_true), dtype=torch.float32)

# 划分数据集
# 划分训练集，测试集 按9:1划分
train_rate, vali_rate, test_rate=0.98,0.01,0.01
train_vali_boundary = int(samples_enc_coor.shape[0] * train_rate)
vali_test_boundary = int(samples_enc_coor.shape[0] * (train_rate + vali_rate))
train_samples_enc_coor = samples_enc_coor[:train_vali_boundary, :].to(DEVICE)
train_samples_enc_grid = samples_enc_grid[:train_vali_boundary, :].to(DEVICE)
train_label = label_true[:train_vali_boundary, :].to(DEVICE)
train_offset = offset_true[:train_vali_boundary, :].to(DEVICE)

vali_samples_enc_coor = samples_enc_coor[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
vali_samples_enc_grid = samples_enc_grid[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
vali_label = label_true[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
vali_offset = offset_true[train_vali_boundary:vali_test_boundary, :].to(DEVICE)

test_samples_enc_coor = samples_enc_coor[vali_test_boundary:, :].to(DEVICE)
test_samples_enc_grid = samples_enc_grid[vali_test_boundary:, :].to(DEVICE)
test_labels = offset_true[vali_test_boundary:, :].to(DEVICE)

train_dataset = Data.TensorDataset(train_samples_enc_coor, train_samples_enc_grid, train_label, train_offset)
vali_dataset = Data.TensorDataset(vali_samples_enc_coor, vali_samples_enc_grid, vali_label, vali_offset)
test_dataset = Data.TensorDataset(test_samples_enc_coor, test_samples_enc_grid, test_labels)

train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
vali_loader = Data.DataLoader(vali_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
def test(model):
    """Test the model."""
    model.eval()
    lost_data = []
    k_lost_stats = {3: 0, 4: 0, 5: 0}  # Track lost data for each K value
    with torch.no_grad():
        for src_enc_coor, src_enc_grid, real_label, real_value in tqdm(train_loader):
            src_enc_coor, src_enc_grid, real_label, real_value = (
                src_enc_coor.to(DEVICE),
                src_enc_grid.to(DEVICE),
                real_label.to(DEVICE),
                real_value.to(DEVICE)
            )
            rand_val = np.random.random()
            if rand_val < 0.75:
                K = 3
            elif rand_val < 0.9:
                K = 4
            else:
                K = 5

            beam_info = torch.zeros((PREDICT_SIZE, NUM_CLASSES),device=DEVICE, dtype=torch.float32)
            true_values_lost = torch.zeros((PREDICT_SIZE, 1), device=DEVICE, dtype=torch.int32)
            consecutive_lost_steps = 0

            logits, _ = model(src_enc_coor, src_enc_grid)

            pred_label = logits[:,-PREDICT_SIZE:,:]
            for i in range(PREDICT_SIZE):
                next_prob = F.softmax(pred_label[:,i,:], dim=-1)
                sorted_probs, sorted_indices = torch.sort(next_prob, descending=True)
                filtered_indices = sorted_indices[:,:K].squeeze(0)  # Take only top K indices
                topk_list = filtered_indices.tolist()
                true_value = real_label[:,i].item()
                if true_value in topk_list:
                    if i >= 1:
                        lost_info = {
                                'beam_info_2': beam_info.clone().cpu(),
                                'logits_info_2': logits.squeeze(0).cpu(),
                                'true_values_lost': true_values_lost.clone().cpu(),
                                'lost_slots': i,
                            }
                        lost_data.append(lost_info)
                        k_lost_stats[K] += 1
                    break
                else:
                    consecutive_lost_steps += 1 
                    if consecutive_lost_steps >= PREDICT_SIZE:
                        lost_info = {
                                'beam_info_2': beam_info.clone().cpu(),
                                'logits_info_2': logits.squeeze(0).cpu(),
                                'true_values_lost': true_values_lost.clone().cpu(),
                                'lost_slots': consecutive_lost_steps,    
                            }
                        lost_data.append(lost_info)
                        k_lost_stats[K] += 1 
                        break

                    true_values_lost = real_label.squeeze(0)
                    # current_position = real_value[:,i+WINDOW_SIZE,:].squeeze(0)
                    # result = multi_beam_aoa_estimation(topk_list, current_position, true_value)
                    # received_signal =  torch.as_tensor(result['y_received']).to(DEVICE)
                    # received_power = received_signal.abs() ** 2
                    # received_power = received_power.ravel()

                    # Update beam info, record current prediction distribution
                    # for value, idx in zip(received_power, filtered_indices.cpu().numpy()):
                    #     beam_info[i, idx] = -torch.log(value)
                    for idx in filtered_indices.cpu().numpy():
                        beam_info[i, idx] = 1
        
        # Print lost data statistics per K
        print(f"Lost data per K value: K=3: {k_lost_stats[3]} items, K=4: {k_lost_stats[4]} items, K=5: {k_lost_stats[5]} items")
        
        # 保存target lost数据为.mat文件
        if lost_data:
            # 转换为numpy数组，便于保存
            beam_info_2_list = [item['beam_info_2'].numpy() for item in lost_data]
            logits_info_2_list = [item['logits_info_2'].numpy() for item in lost_data]
            true_value_list = [item['true_values_lost'].numpy() for item in lost_data]
            lost_slots_list = [item['lost_slots'] for item in lost_data]

            lost_data_mat = {
                'beam_info_2': np.array(beam_info_2_list),
                'logits_info_2': np.array(logits_info_2_list),
                'true_values_lost': np.array(true_value_list),
                'lost_slots': np.array(lost_slots_list)
            }
            
            savemat(DATA_PATH_R, lost_data_mat)
            print(f"Saved {len(lost_data)} target lost records to file.")

def testtest(model):
    """Test the model."""
    model.eval()
    lost_data = []
    k_lost_stats = {3: 0, 4: 0, 5: 0}  # Track lost data for each K value
    with torch.no_grad():
        for src_enc_coor, src_enc_grid, real_label, real_value in tqdm(train_loader):
            batch_size = src_enc_coor.size(0)
            src_enc_coor, src_enc_grid, real_label, real_value = (
                src_enc_coor.to(DEVICE),
                src_enc_grid.to(DEVICE),
                real_label.to(DEVICE),
                real_value.to(DEVICE)
            )
            
            # Set K value for this batch
            rand_val = np.random.random()
            if rand_val < 0.7:
                K = 3
            elif rand_val < 0.9:
                K = 4
            else:
                K = 5

            # Initialize batch-sized tensors to track data
            beam_info = torch.zeros((batch_size, PREDICT_SIZE, NUM_CLASSES), device=DEVICE, dtype=torch.float32)
            true_values_lost = torch.zeros((batch_size, PREDICT_SIZE, 1), device=DEVICE, dtype=torch.int32)
            consecutive_lost_steps = torch.zeros(batch_size, device=DEVICE, dtype=torch.int32)
            true_values_lost = real_label.unsqueeze(-1)
            # Track whether each item in batch is finished
            is_finished = torch.zeros(batch_size, device=DEVICE, dtype=torch.bool)

            logits, _ = model(src_enc_coor, src_enc_grid)
            pred_label = logits[:,-PREDICT_SIZE:,:]
            
            for i in range(PREDICT_SIZE):
                # Skip processing for already finished items
                if is_finished.all():
                    break
                    
                next_prob = F.softmax(pred_label[:,i,:], dim=-1)
                sorted_probs, sorted_indices = torch.sort(next_prob, descending=True)
                filtered_indices = sorted_indices[:,:K]  # Shape: [batch_size, K]
                
                # Process each example in batch
                for b in range(batch_size):
                    if is_finished[b]:
                        continue
                        
                    topk_list = filtered_indices[b].tolist()
                    true_value = real_label[b,i].item()
                    # true_values_lost[b,i,0] = true_value
                    
                    if true_value in topk_list:
                        is_finished[b] = True
                        if i >= 1:
                            lost_info = {
                                'beam_info_2': beam_info[b].clone().cpu(),
                                'logits_info_2': logits[b].cpu(),
                                'true_values_lost': true_values_lost[b].clone().cpu(),
                                'lost_slots': i,
                            }
                            lost_data.append(lost_info)
                            k_lost_stats[K] += 1
                    else:
                        consecutive_lost_steps[b] += 1
                        if consecutive_lost_steps[b] >= PREDICT_SIZE:
                            is_finished[b] = True
                            lost_info = {
                                'beam_info_2': beam_info[b].clone().cpu(),
                                'logits_info_2': logits[b].cpu(),
                                'true_values_lost': true_values_lost[b].clone().cpu(),
                                'lost_slots': consecutive_lost_steps[b].item(),
                            }
                            lost_data.append(lost_info)
                            k_lost_stats[K] += 1
                        else:
                            # Update beam info for this example
                            for idx in filtered_indices[b].cpu().numpy():
                                beam_info[b, i, idx] = 1
        
        # Print lost data statistics per K
        print(f"Lost data per K value: K=3: {k_lost_stats[3]} items, K=4: {k_lost_stats[4]} items, K=5: {k_lost_stats[5]} items")
        
        # 保存target lost数据为.mat文件
        if lost_data:
            # 转换为numpy数组，便于保存
            beam_info_2_list = [item['beam_info_2'].numpy() for item in lost_data]
            logits_info_2_list = [item['logits_info_2'].numpy() for item in lost_data]
            true_value_list = [item['true_values_lost'].numpy() for item in lost_data]
            lost_slots_list = [item['lost_slots'] for item in lost_data]

            lost_data_mat = {
                'beam_info_2': np.array(beam_info_2_list),
                'logits_info_2': np.array(logits_info_2_list),
                'true_values_lost': np.array(true_value_list),
                'lost_slots': np.array(lost_slots_list)
            }
            
            savemat(DATA_PATH_R, lost_data_mat)
            print(f"Saved {len(lost_data)} target lost records to file.")

if __name__ == "__main__":
    torch.manual_seed(42)
    model = MyBert(
        input_dim=input_dim, 
        d_model=FEATURE_SIZE, 
        n_head=N_HEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS, 
        num_decoder_layers=NUM_DECODER_LAYERS, 
        tgt_vocab_size=tgt_vocab_size
        ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    testtest(model=model)
