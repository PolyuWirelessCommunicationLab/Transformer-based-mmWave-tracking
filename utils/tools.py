import numpy as np
import torch
import torch.nn as nn
from parameters import *
import torch.nn.functional as F

def calculate_best_beam(target_pos):
    # Calculate the best beam number for a given target position
    x, y, z = target_pos
    az_true = torch.atan2(y, x)
    el_true = torch.atan2(torch.sqrt(x**2 + y**2), z)
    u_true = torch.sin(el_true) * torch.cos(az_true)
    v_true = torch.sin(el_true) * torch.sin(az_true)
    
    # Build DFT codebook
    u_list = torch.linspace(-(NY-1)/2, (NY-1)/2, NY, device=target_pos.device) / (NY/2)
    v_list = torch.linspace(-(NZ-1)/2, (NZ-1)/2, NZ, device=target_pos.device) / (NZ/2)
    
    # Find the beam closest to the true direction
    min_angle_diff = float('inf')
    best_beam_number = 1
    u_idx_best = 0
    v_idx_best = 0
    
    for v_idx in range(NZ):
        for u_idx in range(NY):
            beam_u = u_list[u_idx]
            beam_v = v_list[v_idx]
            angle_diff = torch.sqrt((beam_u/2 - u_true/2)**2 + (beam_v/2 - v_true/2)**2)
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_beam_number = v_idx * NY + u_idx + 1
                u_idx_best = u_idx
                v_idx_best = v_idx
    
    beam_u = u_list[u_idx_best]
    beam_v = v_list[v_idx_best]
    
    return best_beam_number, u_idx_best, v_idx_best, beam_u, beam_v, u_list, v_list

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    

def generate_grid_centers(n_radius=6, n_theta=36):
    """
    生成极坐标网格中心点坐标
    注意：coverage_radius和n_radius参数不再影响实际的网格划分，
    仅保留参数以保持函数接口兼容性
    """
    grid_centers = []
    dtheta = 2 * np.pi / n_theta
    n_radius = 6  # 固定为6层
    
    # 计算每层的半径边界
    radius_bounds = np.array([10 * np.tan(n * 10 * np.pi/180) for n in range(n_radius + 1)])
    
    # 对每一层半径
    for r_idx in range(n_radius):
        # 取每个网格的中心半径（相邻半径的平均值）
        r_mid = (radius_bounds[r_idx] + radius_bounds[r_idx + 1]) / 2
        
        # 对每个角度分区
        for theta_idx in range(n_theta):
            # 取每个网格的中心角度
            theta = theta_idx * dtheta + dtheta/2
            
            # 转换为笛卡尔坐标
            x = r_mid * np.cos(theta)
            y = r_mid * np.sin(theta)
            grid_centers.append([x, y])
    
    return np.array(grid_centers)

def get_grid_center(grid_index, grid_centers):
    """
    获取特定网格编号的中心点
    grid_index: 网格编号（从1开始）
    grid_centers: 网格中心点坐标数组
    """
    if grid_index <= 0 or grid_index > len(grid_centers):
        return None  # 对于无效的网格编号返回None
    return grid_centers[grid_index - 1]  # 网格编号从1开始

# 综合损失函数类
class CombinedLoss(nn.Module):
    def __init__(self, grid_centers, weight_ce=1.0, weight_mse=1.0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.grid_centers = torch.tensor(grid_centers, dtype=torch.float32)
        self.weight_ce = weight_ce
        self.weight_mse = weight_mse

    def forward(self, predictions, targets):
        # 计算交叉熵损失
        ce_loss = self.cross_entropy_loss(predictions, targets)
        device = predictions.device
        self.grid_centers = self.grid_centers.to(device)
        targets = targets.to(device)

        # 计算预测网格中心点
        pred_indices = torch.argmax(predictions, dim=1)
        pred_centers = torch.stack([self.grid_centers[idx-1] for idx in pred_indices])
        
        # 计算真实网格中心点
        true_centers = torch.stack([self.grid_centers[idx-1] for idx in targets])
        
        # 创建掩码，忽略目标值为0的点
        mask = targets != 0  # 目标值不为0的点为True
        
        # 计算MSE损失，仅对掩码为True的点计算
        mse_loss = nn.functional.mse_loss(pred_centers[mask], true_centers[mask])
        
        # 综合损失
        total_loss = self.weight_ce * ce_loss + self.weight_mse * mse_loss
        return total_loss, ce_loss, mse_loss


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=-1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


def update_prob(model, src_enc_coor, src_enc_grid, indices, probs, prediction_idx):
    pre_idx = prediction_idx -1
    new_enc_coor = torch.zeros_like(src_enc_coor)  # 创建相同shape的全0tensor
    new_enc_coor[:,:-pre_idx,:] = src_enc_coor[:,pre_idx:,:]
    new_enc_grid = torch.zeros_like(src_enc_grid)
    new_enc_grid[:,:-pre_idx] = src_enc_grid[:,pre_idx:]

    sum_weighted_pred = None

    for i in range(len(indices)):
        idx=indices[i]
        prob = probs[i]

        new_enc_grid[:,-1]=idx
        dec_grid_start = new_enc_grid[:,-1].unsqueeze(0)
        dec_in = torch.zeros(dec_grid_start.size(0), PREDICT_SIZE).to(DEVICE)
        dec_grid = torch.cat((dec_grid_start, dec_in), dim=-1)
        dec_grid = dec_grid.long()

        model.eval()
    
        with torch.no_grad():
            grid_output, _ = model(new_enc_coor, new_enc_grid, dec_grid)
            projected = grid_output[:,-PREDICT_SIZE:,:]
            projected = F.softmax(projected, dim=-1)
                    
            # 获取TopK预测
            pred = projected[:,0,:]
            weighted_pred = prob * pred

            # 第一次循环时创建累加器
            if sum_weighted_pred is None:
                sum_weighted_pred = weighted_pred
            else:
                sum_weighted_pred += weighted_pred

    sum_weighted_pred = sum_weighted_pred/sum_weighted_pred.sum(dim=1, keepdim=True)
    sum_weighted_pred = 1 - DELAY_FACTOR * sum_weighted_pred
    return sum_weighted_pred


class ComplementaryCELoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, eps=1e-8):
        """
        alpha: 控制补集项权重
        num_classes: 分类总数 C
        eps: 防止 log(0)
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # 1) 标准交叉熵
        ce_loss = self.ce(logits, target)

        # 2) 计算补集项
        #  2.1 得到 softmax 概率
        probs = F.softmax(logits, dim=1)
        #  2.2 one‑hot 真实标签
        target_onehot = F.one_hot(target, self.num_classes).float()

        #  2.3 对每个“错的类别”计算 -log(1 - p_i)
        #      mask = 1 对错类生效，target 类被置 0 不参与
        mask = 1.0 - target_onehot
        comp = -torch.log(1.0 - probs + self.eps) * mask

        #  2.4 对每条样本的所有错类求平均
        comp_loss = comp.sum(dim=1) / (self.num_classes - 1)
        comp_loss = comp_loss.mean()

        # 3) 合并
        return ce_loss + self.alpha * comp_loss
    

   # Calculate CDF points
def calculate_cdf(data):
    if len(data) == 0:
        return [], []
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def generate_spatial_smoothing_matrix(num_classes, Ny, Nz, sigma=0.2):
    """
    适配 1-based 索引的软标签矩阵。
    索引 0 仅作占位符，不参与任何物理空间计算。
    """
    Ny_2 = Ny * 2
    Nz_2 = Nz // 2
    
    # 确保传入的 num_classes 包含了那个无效的 0 类
    assert num_classes == Ny_2 * Nz_2 + 1, "类别总数必须等于 物理波束总数 + 1"
    
    Q = np.zeros((num_classes, num_classes))
    
    # 防御性设置：理论上 ground truth 不为 0，即使出现异常数据，保证它不引发 NaN
    Q[0, 0] = 1.0  
    
    # 计算物理波束（索引 1 到末尾）的角度
    az_min, az_max = 0, 2 * np.pi
    el_min, el_max = 0, np.pi / 2
    
    az_step = (az_max - az_min) / Ny_2
    el_step = (el_max - el_min) / Nz_2
    
    angles = np.zeros((num_classes, 2))
    for idx in range(1, num_classes):
        grid_idx = idx - 1  # 还原回 0-based 用于网格计算
        el_idx = grid_idx // Ny_2
        az_idx = grid_idx % Ny_2
        
        az = az_min + az_idx * az_step + az_step / 2
        el = el_min + el_idx * el_step + el_step / 2
        angles[idx] = [az, el]
        
    # 计算物理波束间的空间高斯概率 (注意：i 和 j 都从 1 开始)
    for i in range(1, num_classes):
        for j in range(1, num_classes):
            az_i, el_i = angles[i]
            az_j, el_j = angles[j]
            
            d_az = abs(az_i - az_j)
            d_az = min(d_az, 2 * np.pi - d_az)  # 环形距离
            d_el = abs(el_i - el_j)             # 线性距离
            
            d_sq = d_az**2 + d_el**2
            Q[i, j] = np.exp(-d_sq / (2 * sigma**2))
            
    # 按行归一化
    # Q = Q / Q.sum(axis=1, keepdims=True)
    # 引入目标概率参数（例如 0.85）
    target_prob = 0.95
    
    for i in range(1, num_classes):
        # 1. 提取当前行的目标波束自己 (对角线元素必然是 1.0)
        # 为了不干扰剩余概率分配，先将其设为 0
        Q[i, i] = 0.0 
        
        # 2. 计算剩余波束的权重和
        row_sum = Q[i].sum()
        
        # 3. 如果还有剩余权重，按比例分配剩下的 (1 - target_prob) 的概率
        if row_sum > 1e-6:
            Q[i] = Q[i] / row_sum * (1.0 - target_prob)
        
        # 4. 强制赋予目标波束设定的极高概率
        Q[i, i] = target_prob
        
    # 验证无效类 0 是否保持了纯粹的 One-hot
    Q[0, :] = 0.0
    Q[0, 0] = 1.0
    
    return torch.tensor(Q, dtype=torch.float32)

def spatial_cross_entropy_loss(logits, targets, smooth_matrix):
    """
    logits: [batch_size, num_classes] (未经过 softmax 的网络输出)
    targets: [batch_size] (真实的类别索引)
    smooth_matrix: [num_classes, num_classes] (预计算的软标签矩阵)
    """
    # 直接通过查表(Look-up)获取 batch 对应的软标签分布
    soft_targets = smooth_matrix[targets] 
    
    # 计算对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 计算软交叉熵 Loss: - sum(p * log(q))
    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    return loss.mean()


def generate_angular_distance_matrix(num_classes, Ny, Nz):
    """
    生成物理角距离平方矩阵，用于 RMSE Penalty。
    矩阵元素 D[i, j] 表示波束 i 和波束 j 之间的角距离平方。
    """
    Ny_2 = Ny * 2
    Nz_2 = Nz // 2
    assert num_classes == Ny_2 * Nz_2 + 1, "类别总数必须等于 物理波束总数 + 1"
    
    # 距离矩阵
    D = np.zeros((num_classes, num_classes))
    
    az_min, az_max = 0, 2 * np.pi
    el_min, el_max = 0, np.pi / 2
    az_step = (az_max - az_min) / Ny_2
    el_step = (el_max - el_min) / Nz_2
    
    # 计算最大可能的平方距离，用于惩罚预测到无效类 0 的行为
    max_dist_sq = (np.pi)**2 + (np.pi/2)**2 
    
    angles = np.zeros((num_classes, 2))
    for idx in range(1, num_classes):
        grid_idx = idx - 1
        el_idx = grid_idx // Ny_2
        az_idx = grid_idx % Ny_2
        
        az = az_min + az_idx * az_step + az_step / 2
        el = el_min + el_idx * el_step + el_step / 2
        angles[idx] = [az, el]
        
    for i in range(1, num_classes):
        for j in range(1, num_classes):
            az_i, el_i = angles[i]
            az_j, el_j = angles[j]
            
            d_az = abs(az_i - az_j)
            d_az = min(d_az, 2 * np.pi - d_az)  # 方位角环形距离
            d_el = abs(el_i - el_j)             # 俯仰角线性距离
            
            # 记录距离的平方
            D[i, j] = d_az**2 + d_el**2
            
    # 处理类别 0 (无效类)
    # 如果目标是物理波束，预测为 0 会受到最大惩罚
    for i in range(1, num_classes):
        D[i, 0] = max_dist_sq
        D[0, i] = max_dist_sq
    D[0, 0] = 0.0
    
    return torch.tensor(D, dtype=torch.float32)


def distance_penalized_ce_loss(logits, targets, dist_matrix, lambda_reg=0.1):
    """
    结合了 Hard CE 和 Angular RMSE 惩罚的联合 Loss
    logits: [batch_size, num_classes] (未经过 softmax)
    targets: [batch_size]
    dist_matrix: [num_classes, num_classes]
    lambda_reg: 平衡系数，控制 RMSE 惩罚的力度
    """
    # 1. 计算标准的 Hard Cross Entropy Loss (保 Top-K 命中率)
    ce_loss = F.cross_entropy(logits, targets)
    
    # 2. 计算预测的概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 3. 查表获取当前 Batch 中每个 target 对应的距离惩罚向量
    # target_distances 形状为 [batch_size, num_classes]
    target_distances = dist_matrix[targets]
    
    # 4. 计算 Expected Squared Error (只对 ground truth 之外的 p_i 有非零惩罚)
    # sum(p_i * D_i)
    expected_sq_error = torch.sum(probs * target_distances, dim=-1)
    
    # 5. 开根号得到 RMSE 惩罚 (加 1e-8 防止全预测对时 sqrt(0) 导致梯度变 NaN)
    rmse_penalty = torch.mean(torch.sqrt(expected_sq_error + 1e-8))
    # print("rmse_penalty:", rmse_penalty.item())
    # print("ce_loss:", ce_loss.item())
    # 6. 总 Loss
    total_loss = ce_loss + lambda_reg * rmse_penalty
    
    return total_loss