import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import pandas as pd
import math
from sklearn import preprocessing
import torch.utils.data
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from torch.autograd import Variable
import scipy.io as sio
# from visualizer import get_local
# from visualizer.visualizer_functions import *
# get_local.activate()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        # 通过sin和cos定义positional encoding
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        # repeat的作用:对于repeat(x,y,z),把参数通道数复制x遍,行复制y遍,列复制z遍
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)

class PositionalEncoding1(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding1, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, n_ffn=d_model*4, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        _x = x
        x= self.self_attn(x, x, x, src_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, d_model, n_ffn, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, n_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_ffn, d_model)

    def forward(self, x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, save_path="./attn_map.mat"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.save_path = save_path
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
    
    # @get_local('attn')
    def forward(self, q, k, v, attn_mask=None):
        batch, seq, dim = q.shape
        batch_k, seq_k,  dim_k = k.shape
        n_d = self.d_model//self.n_head
        q = self.q_linear(q).view(batch, seq, self.n_head, n_d).transpose(1,2)
        k = self.k_linear(k).view(batch, seq_k, self.n_head, n_d).transpose(1,2)
        v = self.v_linear(v).view(batch, seq_k, self.n_head, n_d).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n_d)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            attn_mask = attn_mask.to(scores.device)
            scores = scores.masked_fill(attn_mask, -1e9)
            # print(scores.cpu().detach().numpy())
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        # print(attn.shape)
        # print(attn.cpu().detach().numpy())
        # attn_np = attn[0].detach().cpu().numpy()  # 取 batch=0: [n_head, seq, seq]
        # sio.savemat(self.save_path, {"attn": attn_np})

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq, dim)
        output = self.out_proj(output)
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(self.d_model, self.n_head, self.dropout) for _ in range(self.num_layers)
                ]
            )
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
        
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, n_ffn=d_model*4, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        _x = tgt
        x = self.self_attn(tgt, tgt, tgt, dec_self_attn_mask)
        x = self.dropout1(x)
        x = self.norm1(_x+x)

        _x=x
        x= self.multihead_attn(x, memory, memory, dec_enc_attn_mask)
        x = self.dropout2(x)
        x = self.norm2(_x+x)

        _x=x
        x = self.ffn(x)
        x = self.dropout3(x)    
        x = self.norm3(_x+x)    
        return x
            
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_head, dropout) for _ in range(num_layers)])
    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):  
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt

    

class RevisedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(RevisedTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, n_ffn=d_model*4, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, beam, prob, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        # _x = beam
        # beam_out = self.self_attn(beam, beam, beam, dec_self_attn_mask)
        # beam_out = self.dropout1(beam_out)
        # beam_self = self.norm1(beam+beam_out)

        # _x = beam_self
        # x = self.multihead_attn(beam_self, prob, prob, dec_enc_attn_mask)
        # x = self.dropout2(x)
        # x = self.norm2(_x+x)

        _x = beam
        x = self.multihead_attn(beam, prob, prob, dec_enc_attn_mask)
        x = self.dropout2(x)
        x = self.norm2(_x+x)

        _x=x
        x = self.ffn(x)
        x = self.dropout3(x)    
        x = self.norm3(_x+x)    
        return x
            
class RevisedTransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.1):
        super(RevisedTransformerDecoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.layers = nn.ModuleList([RevisedTransformerDecoderLayer(d_model, n_head, dropout) for _ in range(num_layers)])
    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):  
        _tgt = tgt
        tgt = self.self_attn(tgt, tgt, tgt, dec_self_attn_mask)
        tgt = self.dropout1(tgt)
        tgt = self.norm1(tgt+_tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt

