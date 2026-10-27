import torch
import torch.nn as nn
from parameters import *
from transformers import *
import torch.utils.data as Data
from scipy.io import loadmat


class ReacquisitionModel_new(nn.Module):
    def __init__(self, input_dim, d_model, n_head, num_encoder_layers, num_decoder_layers, tgt_vocab_size,  dropout=0.1):
        super(ReacquisitionModel_new, self).__init__()
        
        self.embedding = nn.Linear(NUM_CLASSES, d_model)
        self.prob_embedding = nn.Linear(NUM_CLASSES, d_model)
    

        self.pos_encoder = PositionalEncoding1(d_model, dropout)

        self.refine_decoder =RevisedTransformerDecoder(d_model, n_head, num_layers=num_decoder_layers, dropout=dropout)
        self.projection = nn.Linear(d_model, NUM_CLASSES)

        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        self.eps = 1e-8
        self.sigmoid = nn.Sigmoid()

    def forward(self, beam_info, prob_info):
        # prob_info = prob_info[:,-PREDICT_SIZE+1:,:]
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dec_self_attn_subsequence_mask = self.get_attn_subsequence_mask(beam_info).to(DEVICE)
        # 将两个mask叠加，布尔值可以视为0和1，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
        dec_self_attn_mask = torch.gt(dec_self_attn_subsequence_mask, 0).to(DEVICE) 
        
        # prob_old = prob_info[:,1:PREDICT_SIZE,:]
        # logits_old = torch.log(prob_old + self.eps)
        # logits_old = prob_old

        beam_emb = self.embedding(beam_info)
        prob_emb = self.prob_embedding(prob_info)
        prob_emb = self.pos_encoder(prob_emb)

        # prob_in = prob_emb[:,0:PREDICT_SIZE-1,:]
        # x= self.pos_encoder(beam_emb)
        
        temp = torch.zeros_like(prob_emb[:,0:WINDOW_SIZE,:])
        beam_emb_pad = torch.cat([temp, beam_emb], dim=1)
        beam_emb_pe = self.pos_encoder(beam_emb_pad)
        x = beam_emb_pe[:,WINDOW_SIZE:,:]

        hidden = self.refine_decoder(x,prob_emb,dec_self_attn_mask=dec_self_attn_mask, dec_enc_attn_mask=None)
        
        hidden = hidden[:,-PREDICT_SIZE+1:,:]
        # Δ
        delta_logits = self.projection(hidden)

        return delta_logits
    
    def get_attn_subsequence_mask(self, seq):
        """seq: [batch_size, tgt_len]"""
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte() # 因为只有0、1所以用byte节省内存
        return subsequence_mask
    

def load_data(file_path):
    data = loadmat(file_path)
    
    beam_info = data['beam_info_2']
    prob_info = data['logits_info_2']
    target_data = data['true_values_lost']
    
    return beam_info, prob_info, target_data

def prepare_datasets(beam_info, prob_info, target_data):
    samples_beam, samples_prob, output_prob, labels = [], [], [], []
    
    num_obj, num_samples, feature_dim = beam_info.shape[0], beam_info.shape[1], beam_info.shape[2]
    
    # 滑动窗口处理数据
    for i in range(num_obj):
        # 提取输入序列
        samples_beam.append(np.squeeze(beam_info[i, 0:PREDICT_SIZE-1, :]))
        samples_prob.append(np.squeeze(prob_info[i, 0:WINDOW_SIZE, :]))
        output_prob.append(np.squeeze(prob_info[i, -PREDICT_SIZE+1:, :]))
        # 提取目标值
        labels.append(np.squeeze(target_data[i, 1:PREDICT_SIZE]))

    # 转换为PyTorch Tensor
    samples_beam = torch.tensor(np.array(samples_beam), dtype=torch.float32)
    samples_prob = torch.tensor(np.array(samples_prob), dtype=torch.float32)
    output_prob = torch.tensor(np.array(output_prob), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)

  
    train_rate, vali_rate, test_rate = 0.9, 0.05, 0.05
    train_vali_boundary = int(samples_beam.shape[0] * train_rate)
    vali_test_boundary = int(samples_beam.shape[0] * (train_rate + vali_rate))

    train_samples_beam = samples_beam[:train_vali_boundary, :].to(DEVICE)
    train_samples_prob = samples_prob[:train_vali_boundary, :].to(DEVICE)
    train_output_prob = output_prob[:train_vali_boundary, :].to(DEVICE)
    train_labels = labels[:train_vali_boundary, :].to(DEVICE)
    

    vali_samples_beam = samples_beam[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_samples_prob = samples_prob[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_output_prob = output_prob[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    vali_labels = labels[train_vali_boundary:vali_test_boundary, :].to(DEVICE)
    
  
    test_samples_beam = samples_beam[vali_test_boundary:, :].to(DEVICE)
    test_samples_prob = samples_prob[vali_test_boundary:, :].to(DEVICE)
    test_output_prob = output_prob[vali_test_boundary:, :].to(DEVICE)
    test_labels = labels[vali_test_boundary:, :].to(DEVICE)
    
    print("Train dataset number: ", train_samples_beam.shape)

    train_dataset = Data.TensorDataset(train_samples_beam, train_samples_prob, train_output_prob, train_labels)
    vali_dataset = Data.TensorDataset(vali_samples_beam, vali_samples_prob, vali_output_prob, vali_labels)
    test_dataset = Data.TensorDataset(test_samples_beam, test_samples_prob, test_output_prob, test_labels)
    
    return train_dataset, vali_dataset, test_dataset

def get_dataloaders(file_path):
    beam_info, prob_info, target_data = load_data(file_path)
    train_dataset, val_dataset, test_dataset = prepare_datasets(beam_info, prob_info, target_data)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader