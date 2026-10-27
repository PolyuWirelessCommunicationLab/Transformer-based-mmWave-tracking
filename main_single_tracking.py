import torch
import torch.optim as optim
import torch.nn as nn
from bert_model import MyBert, get_attn_pad_mask
from refine_Model import ReacquisitionModel_new
from process_data import *
from parameters import *
from utils.tools import *
import math
import time
import torch.utils.data as Data
from tqdm import tqdm
from torch.optim import lr_scheduler
from datetime import datetime
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.io import savemat
# from aoa_estimation_new import multi_beam_aoa_estimation

def process_single_trajectory(ground_truth_data, observed_data, real_data, signal, trajectory_idx):
    single_ground_truth = torch.tensor(ground_truth_data[trajectory_idx]).long()
    single_observed = torch.tensor(observed_data[trajectory_idx]).float()
    single_real = torch.tensor(real_data[trajectory_idx]).float()
    single_signal = torch.tensor(signal[trajectory_idx]).float()
    
    return single_ground_truth, single_observed, single_real, single_signal


def normal_tracking_mode(model, filtering_model, ground_truth, observed, estimated, start_idx, current_idx, flags, WINDOW_SIZE, PREDICT_SIZE, DEVICE):
    """Handle normal tracking mode operations."""

    # predict next probability
    # Run transformer encoder - now with just window size input (not window+predict)
    coor_in = torch.zeros_like(observed[start_idx:start_idx+1,:])
    grid_in = torch.zeros_like(ground_truth[start_idx:start_idx+1,:])
    src_enc_coor = torch.cat([observed[start_idx:start_idx+WINDOW_SIZE,:], coor_in], dim=0).unsqueeze(0)
    src_enc_grid = torch.cat([ground_truth[start_idx:start_idx+WINDOW_SIZE], grid_in], dim=0).unsqueeze(0).squeeze(2)
    
    attn_pad_mask = get_attn_pad_mask(src_enc_grid, src_enc_grid).to(DEVICE)
    enc_mask = torch.gt(attn_pad_mask, 0).to(DEVICE)
    # Apply mask for any missing values in window
    for i in range(WINDOW_SIZE):
        if flags[start_idx+i] == 0:
            src_enc_coor[:, i, :] = 0
            src_enc_grid[:, i] = 0
            enc_mask[:, :, i] = True
            enc_mask[:, i, :] = True
    
    # Run transformer encoder
    enc_input_coor = model.encoder_embedding_coor(src_enc_coor)
    enc_input_grid = model.encoder_embedding_grid(src_enc_grid)
    enc_input = enc_input_coor + enc_input_grid
  
    enc_input = model.pos_encoder(enc_input)
    enc_output = model.encoder(enc_input, enc_mask)
    
    offsets = model.mlp(enc_output[:,0:WINDOW_SIZE,:])
    estimated_loc = src_enc_coor[:,WINDOW_SIZE-1,:] + offsets[:,-1,:]
    estimated[current_idx,:]=estimated_loc

    logits = model.projection(enc_output)
    logits = logits[:,-1:,:]
    projected = F.softmax(logits, dim=-1)
    next_prob = projected[:,0,:]
    
    return next_prob, estimated

def reacquisition_mode(model, refine_model, beam_info, ground_truth, observed, estimated, start_idx, current_idx, flags, consecutive_lost_steps, PREDICT_SIZE, DEVICE):
    """Handle target reacquisition mode operations."""
    """Handle target reacquisition mode operations with full prediction window."""
    # Create encoder input with full window + prediction size
    coor_in = torch.zeros_like(estimated[start_idx:start_idx+PREDICT_SIZE,:])
    grid_in = torch.zeros_like(ground_truth[start_idx:start_idx+PREDICT_SIZE,:])
    src_enc_coor = torch.cat([estimated[start_idx:start_idx+WINDOW_SIZE,:], coor_in], dim=0).unsqueeze(0)
    src_enc_grid = torch.cat([ground_truth[start_idx:start_idx+WINDOW_SIZE], grid_in], dim=0).unsqueeze(0).squeeze(2)
    
    # Create attention mask
    attn_pad_mask = get_attn_pad_mask(src_enc_grid, src_enc_grid).to(DEVICE)
    enc_mask = torch.gt(attn_pad_mask, 0).to(DEVICE)

    # Apply mask for any missing values in window
    for i in range(WINDOW_SIZE):
        if flags[start_idx+i] == 0:
            src_enc_coor[:, i, :] = 0
            src_enc_grid[:, i] = 0
            enc_mask[:, :, i] = True
            enc_mask[:, i, :] = True
    
    # Run through encoder
    enc_input_coor = model.encoder_embedding_coor(src_enc_coor)
    enc_input_grid = model.encoder_embedding_grid(src_enc_grid)
    enc_input = enc_input_coor + enc_input_grid
    
    enc_input = model.pos_encoder(enc_input)
    enc_output = model.encoder(enc_input, enc_mask)
    
    # Generate full prediction window logits
    logits = model.projection(enc_output)
    # logits = logits[:,-PREDICT_SIZE:,:]  # Take all future steps
    
    # Prepare input for refinement model
    beam_refine_input = torch.tensor(beam_info[0:consecutive_lost_steps,:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # prob_info = logits[:,0:PREDICT_SIZE,:]
    prob_info = logits[:,0:WINDOW_SIZE,:]
    # Apply refinement model
    delta_logits = refine_model(beam_refine_input, prob_info)
    delta_logits = delta_logits[:,-1:,:]
    
    # Refine logits
    logits = logits[:,-PREDICT_SIZE:,:]
    origin_logits = logits[:,consecutive_lost_steps:consecutive_lost_steps+1,:]
    refined_logits = origin_logits + delta_logits
    refined_projected = F.softmax(refined_logits, dim=-1)
    refine_next_prob = refined_projected[:,0,:]

    # next_logits = refined_logits[:,0,:]
    # next_logits = origin_logits[:,0,:]
    
    return refine_next_prob, logits

def predict_trajectory_iter(model, refine_model, filtering_model, ground_truth_data, observed_data, real_data, signal_data, trajectory_idx, K):
    """Main trajectory prediction function with mode separation."""
    model.eval()
    
    # Get processed single trajectory data
    ground_truth, observed, real, signal = process_single_trajectory(ground_truth_data, observed_data, real_data, signal_data, trajectory_idx)
    
    predictions = ground_truth[:WINDOW_SIZE].numpy().tolist()
    flags = [1] * WINDOW_SIZE  # Flag=1 for initial window points
    probabilities = [1.0] * WINDOW_SIZE  # Probability=1.0 for initial window points
    adaptive_k_sizes = [1] * WINDOW_SIZE  # Record adaptive K value for each timestep
    
    recovery_count = 0  # Recovery counter
    missing_count = 0  # Missing value counter
    consecutive_lost_steps = 0  # Consecutive lost steps counter
    MAX_LOST_STEPS = 4 # Maximum allowed consecutive lost steps
    
    estimated_trajectory = torch.zeros_like(observed).to(DEVICE)
    estimated_trajectory[:WINDOW_SIZE] = observed[:WINDOW_SIZE]
    
    # Move data to device
    ground_truth = ground_truth.to(DEVICE)
    observed = observed.to(DEVICE)
    real = real.to(DEVICE)
    signal = signal.to(DEVICE)

    # Tracking success flag
    tracking_success = True
    
    # Target lost information list
    lost_data = []
    current_mode = "normal"  # Initial mode is normal tracking
    
    with torch.no_grad():
        current_idx = WINDOW_SIZE  # Start prediction from after WINDOW_SIZE
        
        # Initialize encoder input
        start_idx = 0  # Starting index for encoder input
        predict_idx = 0
        
        # Initial transformer run
        next_prob, _ = normal_tracking_mode(
            model, filtering_model, ground_truth, observed, estimated_trajectory, start_idx, current_idx, 
            flags, WINDOW_SIZE, PREDICT_SIZE, DEVICE
        )
        
        # Create beam info vector, 1=not in top K, 0=in top K
        beam_info = np.zeros((PREDICT_SIZE, NUM_CLASSES), dtype=np.int32)
        true_values_lost = np.zeros((PREDICT_SIZE, 1), dtype=np.int32)

        while current_idx < len(ground_truth) and tracking_success:
            # Check if current point is out of range (value=0)
            if ground_truth[current_idx].item() == 0:
                break  # Normal trajectory end
            
            # Check if exceeded maximum allowed consecutive lost steps
            if consecutive_lost_steps >= MAX_LOST_STEPS:
                tracking_success = False
                missing_count += 1
                break
            
            # Get next prediction distribution
            predict_idx += 1
            
            # Get TopK predictions
            sorted_probs, sorted_indices = torch.sort(next_prob, descending=True)
            filtered_probs = sorted_probs[:,:K].squeeze(0)  # Take only top K probabilities
            filtered_indices = sorted_indices[:,:K].squeeze(0)  # Take only top K indices
            
            # Record adaptive K value for current timestep
            current_k = len(filtered_indices)
            adaptive_k_sizes.append(current_k)
            
            # Get true value
            true_value = ground_truth[current_idx].item()
            topk_list = filtered_indices.tolist()
            topk_prob_list = filtered_probs.tolist()
            
            # Check if true value is in filtered predictions
            if true_value in topk_list:
                # Correct prediction, normal tracking mode
                current_mode = "normal"
                # If successfully recovered target, record recovery count
                if consecutive_lost_steps >= 1:
                    recovery_count += 1
                    missing_count += 1

                    # lost_info = {
                    #         'beam_info_2': beam_info,
                    #         'logits_info_2': logits.squeeze(0).cpu().numpy(),
                    #         'true_values_lost': true_values_lost,
                    #         'lost_slots': consecutive_lost_steps,
                    #     }
                    # lost_data.append(lost_info)

                # Record prediction result
                idx_in_topk = topk_list.index(true_value)
                flag = idx_in_topk + 1
                flags.append(flag)
                prob = topk_prob_list[idx_in_topk]
                probabilities.append(prob)
                predictions.append(true_value)
                
                # Slide window, update encoder input
                start_idx += 1 + consecutive_lost_steps
                
                # Run normal tracking mode
                next_prob, estimated_trajectory = normal_tracking_mode(
            model, filtering_model, ground_truth, observed, estimated_trajectory, start_idx, current_idx, 
            flags, WINDOW_SIZE, PREDICT_SIZE, DEVICE)
                
                # Reset beam info
                beam_info = np.zeros((PREDICT_SIZE, NUM_CLASSES), dtype=np.int32)
                true_values_lost = np.zeros((PREDICT_SIZE, 1), dtype=np.int32)
                predict_idx = 0
                consecutive_lost_steps = 0  # Reset consecutive lost counter
                
            else:
                # Wrong prediction, enter reacquisition mode
                consecutive_lost_steps += 1 
                if consecutive_lost_steps > MAX_LOST_STEPS:
                    tracking_success = False
                    missing_count += 1
                    # lost_info = {
                    #         'beam_info_2': beam_info,
                    #         'logits_info_2': logits.squeeze(0).cpu().numpy(),
                    #         'true_values_lost': true_values_lost,
                    #         'lost_slots': consecutive_lost_steps,    
                    #     }
                    # lost_data.append(lost_info)
                    break

                # predict_idx += 1
                current_mode = "reacquisition"
                
                # Record prediction result
                flag = 0
                flags.append(flag)
                probabilities.append(0)
                predictions.append(0)
                estimated_trajectory[current_idx,:] = torch.tensor([0, 0, 0]).to(DEVICE)
                # best_beam_number = ground_truth[current_idx].cpu()
                # current_position = real[current_idx,:].cpu()
                # result = multi_beam_aoa_estimation(topk_list, current_position, best_beam_number)
                # received_signal = result['y_received']
                # received_power = np.abs(received_signal) ** 2

                ## collect lost data
                lost_duration = consecutive_lost_steps
                temp = ground_truth[start_idx+WINDOW_SIZE:start_idx+WINDOW_SIZE+PREDICT_SIZE].cpu().numpy()
                n = temp.shape[0]
                true_values_lost[:n,:] = temp
                for idx in filtered_indices.cpu().numpy():
                    beam_info[consecutive_lost_steps - 1, idx] = 1
                # # Update beam info, record current prediction distribution
                # for value, idx in zip(received_power, filtered_indices.cpu().numpy()):
                #     beam_info[consecutive_lost_steps - 1, idx] = -np.log(value.item())
                
                # Run reacquisition mode
                next_prob, logits = reacquisition_mode(
                    model, refine_model, beam_info, ground_truth, observed, estimated_trajectory,
                    start_idx, current_idx, flags, consecutive_lost_steps, 
                    PREDICT_SIZE, DEVICE
                )

            # Move to next timestep
            current_idx += 1
    
    # Calculate metrics
    mask = ~torch.all(estimated_trajectory[WINDOW_SIZE:] == 0, dim=1)
    if torch.any(mask):
        obs_rmse = torch.sqrt(torch.mean(torch.sum((observed[WINDOW_SIZE:][mask] - real[WINDOW_SIZE:][mask])**2, dim=1)))
        pred_rmse = torch.sqrt(torch.mean(torch.sum((estimated_trajectory[WINDOW_SIZE:][mask] - real[WINDOW_SIZE:][mask])**2, dim=1)))
    else:
        obs_rmse = torch.tensor(0.0).to(DEVICE)
        pred_rmse = torch.tensor(0.0).to(DEVICE)
    
    metrics = {'observation_rmse': obs_rmse, 'prediction_rmse': pred_rmse}
    
    return (
        predictions,
        flags,
        probabilities,
        adaptive_k_sizes,
        tracking_success,
        metrics,
        estimated_trajectory,
        recovery_count,
        missing_count,
        lost_data
    )

def predict_longseq(model, refine_model, filtering_model, K):
    trajectory_idx = 100
    successful_tracks = 0
    failure_positions = []
    adaptive_k_stats = []
    all_obs_rmse = []
    all_pred_rmse = []
    total_recoveries = 0
    total_missing = 0
    all_lost_data = []  
    running_time_total = []
    
    # 定义要统计的步长
    if NUM_SAMPLES== 205:
        step_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    else:
        step_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # step_sizes = [x + WINDOW_SIZE for x in step]

    
    print(f"TRANSFORMER model, window size={WINDOW_SIZE}, prediction size={PREDICT_SIZE}, K={K}")

    for idx in range(-trajectory_idx, 0):
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        predictions, flags, probabilities, adaptive_k_sizes, success, metrics, _, recovery_count, missing_count, lost_data = (
            predict_trajectory_iter(
                model, 
                refine_model,
                filtering_model,
                ground_truth, 
                observed, 
                real_value, 
                signal,
                idx, 
                K
            )
        )
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        all_lost_data.extend(lost_data)
        
        avg_adaptive_k = sum(adaptive_k_sizes[WINDOW_SIZE:]) / len(adaptive_k_sizes[WINDOW_SIZE:])
        adaptive_k_stats.append(avg_adaptive_k)
        all_obs_rmse.append(metrics['observation_rmse'].cpu().numpy())
        all_pred_rmse.append(metrics['prediction_rmse'].cpu().numpy())
        total_recoveries += recovery_count
        total_missing += missing_count

        if success:
            successful_tracks += 1
            failure_pos = len(flags)  
        else:
            failure_pos = len(flags) - PREDICT_SIZE
        
        failure_positions.append(failure_pos)

        
        running_time = (end-start)/failure_pos
        running_time_total.append(running_time)
        
        # print("index:", idx)
        # print("Failure position:", failure_pos)
        # print("successful tracking:", success)
        # print("reacquisition count:", str(recovery_count) + "/" + str(missing_count))
        # print(flags)

    success_rates = []
    for step in step_sizes:
        success_count = sum(1 for pos in failure_positions if pos >= step)
        success_rate = success_count / trajectory_idx
        success_rates.append(success_rate)
        print(f"Success rate for {step} steps: {success_rate:.2%}")

    success_rate = successful_tracks / trajectory_idx
    failure_rate = 1 - success_rate
    overall_avg_adaptive_k = sum(adaptive_k_stats) / len(adaptive_k_stats)
    overall_recovery_rate = total_recoveries / total_missing if total_missing > 0 else 1.0
    
    valid_tracking_time_95 = None
    valid_tracking_time_90 = None
    
    fine_step_size = 20
    max_tracking_length = max(failure_positions)
    fine_steps = list(range(WINDOW_SIZE, int(max_tracking_length) + fine_step_size, fine_step_size))
    
    fine_success_rates = []
    for step in fine_steps:
        success_count1 = sum(1 for pos in failure_positions if pos >= step)
        success_rate1 = success_count1 / trajectory_idx
        fine_success_rates.append(success_rate1)
        
     
        if success_rate1 < 0.95 and valid_tracking_time_95 is None:
            valid_tracking_time_95 = (step - WINDOW_SIZE)*dt
        if success_rate1 < 0.90 and valid_tracking_time_90 is None:
            valid_tracking_time_90 = (step - WINDOW_SIZE)*dt
    
   
    if valid_tracking_time_95 is None:
        valid_tracking_time_95 = (NUM_SAMPLES - WINDOW_SIZE)*dt
    if valid_tracking_time_90 is None:
        valid_tracking_time_90 = (NUM_SAMPLES - WINDOW_SIZE)*dt

# Filter out zero values and convert to numpy arrays
    obs_rmse_nonzero = np.array([mse for mse in all_obs_rmse if mse > 0])
    pred_rmse_nonzero = np.array([mse for mse in all_pred_rmse if mse > 0])

    obs_rmse_sorted, obs_rmse_cdf = calculate_cdf(obs_rmse_nonzero)
    pred_rmse_sorted, pred_rmse_cdf = calculate_cdf(pred_rmse_nonzero)

    
  
    avg_tracking_length = sum(failure_positions) / len(failure_positions) -WINDOW_SIZE

    avg_running_time = sum(running_time_total) / len(running_time_total)



  
    save_data = {
        'success_rates': success_rates,
        'step_sizes': step_sizes,
        'failure_positions': failure_positions,
        'overall_success_rate': success_rate,
        'overall_recovery_rate': overall_recovery_rate,
        'avg_adaptive_k': overall_avg_adaptive_k,
        'avg_tracking_time': avg_tracking_length * dt,
        'obs_rmse_mean': np.mean(all_obs_rmse),
        'obs_rmse_std': np.std(all_obs_rmse),
        'pred_rmse_mean': np.mean(all_pred_rmse),
        'pred_rmse_std': np.std(all_pred_rmse),
        'pred_rmse':all_pred_rmse,
        'obs_rmse_cdf_x': obs_rmse_sorted,
        'obs_rmse_cdf_y': obs_rmse_cdf,
        'pred_rmse_cdf_x': pred_rmse_sorted,
        'pred_rmse_cdf_y': pred_rmse_cdf,
        'valid_tracking_time_95': valid_tracking_time_95,
        'valid_tracking_time_90': valid_tracking_time_90,
        'avg_running_time': avg_running_time
    }


    # savemat(f'./TWC_results/m3trackformer_K{K}_vel{v}_mt{MT}.mat', save_data)
    
    print('\nOverall Success rate:', success_rate)
    print('Overall Failure rate:', failure_rate)
    print(f'Average adaptive K: {overall_avg_adaptive_k:.2f}')
    print(f'Total recovery count: {total_recoveries}')
    print(f'Total missing count: {total_missing}')
    print(f'Overall recovery rate: {overall_recovery_rate:.2%}')
    print(f"Average tracking time: {(avg_tracking_length*dt):.2f}")
    print(f"Valid tracking time (95% accuracy): {valid_tracking_time_95} (seconds)")
    print(f"Valid tracking time (90% accuracy): {valid_tracking_time_90} (seconds)")
    print("\nPerformance Metrics:")
    print(f"Average Observation RMSE: {np.mean(all_obs_rmse):.4f} ± {np.std(all_obs_rmse):.4f}")
    print(f"Average Prediction RMSE: {np.mean(all_pred_rmse):.4f} ± {np.std(all_pred_rmse):.4f}")
    print(f"Average Running Time: {avg_running_time:.4f}")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        print("GPU is available")
        DEVICE = torch.device("cuda")
    else:
        print("GPU is not available")  
        DEVICE = torch.device("cpu")
    for v in [35]:
        data_path = f"./data/testing_data_vel{v}_mt{MT}_dft.mat"
        ground_truth, observed, real_value, signal = load_data(data_path)
        # tgt_vocab_size=ground_truth.max()+1
        tgt_vocab_size = NUM_CLASSES
        num_obj, num_samples, input_dim = ground_truth.shape[0], ground_truth.shape[1], observed.shape[2]

        model = MyBert(
            input_dim=input_dim, 
            d_model=FEATURE_SIZE, 
            n_head=N_HEAD, 
            num_encoder_layers=NUM_ENCODER_LAYERS, 
            num_decoder_layers=NUM_DECODER_LAYERS, 
            tgt_vocab_size=tgt_vocab_size
            ).to(DEVICE)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()

        filtering_model = None
            
        for K in [2,3,4,5,6]:
            print(f"\n===== Transformer Model w refine, velocity={v}, MT={MT}, K={K}  =====\n")
            refine_model = ReacquisitionModel_new(
            input_dim=tgt_vocab_size,
            d_model=FEATURE_SIZE_R,
            n_head=N_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=1,
            tgt_vocab_size=tgt_vocab_size,
            dropout=DROPOUT
            ).to(DEVICE)
            # MODEL_PATH_R = f"./model state/reacquisition_model_K{K}_mt{MT}_new.pth"
            refine_model.load_state_dict(torch.load(MODEL_PATH_R, map_location=DEVICE, weights_only=True))
            refine_model.eval()
            predict_longseq(
                model=model, 
                refine_model= refine_model, 
                filtering_model=filtering_model, 
                K=K)

