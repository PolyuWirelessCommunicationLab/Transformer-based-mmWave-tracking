import torch
import torch.nn as nn
from parameters import *
from transformers import *

class MyBert(nn.Module):
    def __init__(self, input_dim, d_model, n_head, num_encoder_layers, num_decoder_layers, tgt_vocab_size,  dropout=0.1):
        super(MyBert, self).__init__()
        
        self.encoder_embedding_coor = nn.Linear(input_dim, d_model)
        self.encoder_embedding_grid = nn.Embedding(tgt_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # self.fusion_projection = nn.Linear(2 * d_model, d_model)

        # 调用先前定义的PositionalEncoding类，定义位置编码层
        self.pos_encoder = PositionalEncoding1(d_model, dropout)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.encoder =TransformerEncoder(d_model, n_head, num_layers=num_encoder_layers, dropout=dropout)
        # self.cls_decoder =TransformerDecoder(d_model, n_head, num_layers=num_decoder_layers, dropout=dropout)
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 3),
        ) 
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, src_coor, src_grid):
        dec_self_attn_pad_mask = get_attn_pad_mask(src_grid, src_grid).to(DEVICE)
        
        # 将两个mask叠加，布尔值可以视为0和1，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
        enc_mask = torch.gt(dec_self_attn_pad_mask, 0).to(DEVICE)
        
        # Encoder部分，与原模型相同
        src_coor = self.encoder_embedding_coor(src_coor)
        src_grid = self.encoder_embedding_grid(src_grid)
        src = src_coor + src_grid

        enc_input = self.pos_encoder(src)
        enc_output = self.encoder(enc_input, enc_mask)

        # tgt = self.encoder_embedding_grid(tgt)
        # dec_input = self.pos_encoder(tgt)
        # dec_output = self.cls_decoder(dec_input, enc_output, dec_self_attn_mask=None, dec_enc_attn_mask=None)
        cls_output = self.projection(enc_output)
        offsets = self.mlp(enc_output[:,0:WINDOW_SIZE,:])
        
        # # filtering部分！！
        # fil_coor = self.encoder_embedding_coor(fil_coor)
        # fil_grid = self.encoder_embedding_grid(fil_grid)
        # enc_flat = enc_output.reshape(enc_output.size(0), -1)
        # fil_in = torch.cat([fil_coor+fil_grid, enc_flat], dim=-1)
        # fil_out = self.reg_decoder(fil_in)
        
        grid = cls_output[:, -PREDICT_SIZE:, :]  # 取最后一个时间步的输出作为offset
        # offset = fil_out  # 取最后一个时间步的输出作为offset
        return cls_output, offsets

def get_attn_pad_mask(seq_q, seq_k):
    # len_q、len_k其实是q的length和k的length，q和k都是一个序列即一个句子，长度即句子中包含的词的数量
    batch_size, len_q = seq_q.size() # 获取作为q的序列（句子）长度
    batch_size, len_k = seq_k.size() # 获取作为k的序列长度
    # seq_k.data.eq(0)返回一个和seq_k等大的布尔张量，seq_k元素等于0的位置为True,否则为False
    # 然后扩维以保证后续操作的兼容(广播)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # pad_attn_mask: [batch_size,1,len_k]
    # 要为每一个q提供一份k，所以把第二维度扩展了q次，这样只有最后一列是True，正常来说最后一行也需要是True，但是由于作为padding的token对其他词的注意力不重要，所以可以这样写
    res = pad_attn_mask.expand(batch_size, len_q, len_k)
    return res # return: [batch_size, len_q, len_k]
    # 返回的是batch_size个 len_q * len_k的矩阵，内容是True和False，第i行第j列表示的是query的第i个词对key的第j个词的注意力是否无意义，若无意义则为True，有意义的为False（即被padding的位置是True）

