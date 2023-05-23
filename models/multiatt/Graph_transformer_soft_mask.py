import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.layer2 = nn.Linear(ffn_size, out_size)
        self.dropout =nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size, layer_name):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.layer_name = layer_name

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        if "VA1" in layer_name:
            self.v_embedding = torch.nn.Embedding(50,1)
        if "A1" in layer_name:
            self.embedding = torch.nn.Embedding(200,1)
        #self.att_embedding = torch.nn.Embedding(100,1)
        self.att_embedding = torch.nn.Embedding(100,1)

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, clue_len, mask, box_seq = None, att_bias = None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        seq_len = q.size(1)

        if "R" not in self.layer_name:
            if att_bias is None:
                full_path_matrix = torch.ones(seq_len,seq_len).cuda()
                if 'VA' in self.layer_name:
                    A_path_matirx = torch.arange(seq_len-clue_len).unsqueeze(-1).repeat(1,seq_len-clue_len).cuda()
                    A_path_matirx_trans = A_path_matirx.transpose(1,0)
                    A_path_matirx = abs(A_path_matirx-A_path_matirx_trans)

                    full_path_matrix[clue_len:,clue_len:] = A_path_matirx
                
                if 'QA' in self.layer_name:
                    A_path_matirx = torch.arange(seq_len-clue_len).unsqueeze(-1).repeat(1,seq_len-clue_len).cuda()
                    A_path_matirx_trans = A_path_matirx.transpose(1,0)
                    A_path_matirx = abs(A_path_matirx-A_path_matirx_trans)

                    Q_path_matirx = torch.arange(clue_len).unsqueeze(-1).repeat(1,clue_len).cuda()
                    Q_path_matirx_trans = Q_path_matirx.transpose(1,0)
                    Q_path_matirx = abs(Q_path_matirx-Q_path_matirx_trans)

                    full_path_matrix[clue_len:,clue_len:] = A_path_matirx
                    full_path_matrix[:clue_len:,:clue_len] = Q_path_matirx
                
                full_path_matrix = full_path_matrix.reshape(seq_len*seq_len)
                #print(layer_name+"full_path_matrix_0:",full_path_matrix.reshape(seq_len,seq_len))
                full_path_matrix = self.embedding(full_path_matrix.long()).squeeze(-1).reshape(seq_len, seq_len)
                full_path_matrix = full_path_matrix.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.head_size, 1, 1)
                #print(layer_name+"full_path_matrix_1:",full_path_matrix[0,0,:,:])
                if 'VA' in self.layer_name:
                    box_seq = box_seq.unsqueeze(2).repeat(1,1,clue_len,1)
                    box_seq_trans = box_seq.transpose(2,1)
                    path_matirx = abs(box_seq-box_seq_trans)
                    V_path_matirx = torch.ceil(torch.sqrt(path_matirx[:,:,:,0]*path_matirx[:,:,:,0]+path_matirx[:,:,:,1]*path_matirx[:,:,:,1])*32)
                    #V_path_matirx = torch.exp(-V_path_matirx)-1/2.718281828459
                    V_path_matirx = self.v_embedding(V_path_matirx.reshape(int(batch_size/4), clue_len*clue_len).long()).reshape(int(batch_size/4), clue_len,clue_len)
                    V_path_matirx = V_path_matirx.unsqueeze(1).repeat(1,4,1,1).reshape(batch_size, clue_len, clue_len)
                    
                    V_path_matirx = V_path_matirx.unsqueeze(1).repeat(1, self.head_size, 1, 1)
                    full_path_matrix[:,:,:clue_len:,:clue_len] = V_path_matirx
                    #print(layer_name+"full_path_matrix_2:",full_path_matrix[0,0,:,:])
            else:
                att_bias = torch.ceil(att_bias*64)
                att_bias = self.att_embedding(att_bias.reshape(batch_size, self.head_size*seq_len*seq_len).long()).reshape(batch_size, self.head_size, seq_len, seq_len)

        # else:
        #     path_matirx = torch.arange(seq_len).unsqueeze(-1).repeat(1,seq_len).cuda()
        #     path_matirx_trans = path_matirx.transpose(1,0)
        #     path_matirx = abs(path_matirx-path_matirx_trans)
        #     full_path_matrix = path_matirx

        
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        ## input_mask: [b,h,len,1]
        ## input_att_bias: [b,h,len,len]
        mask = mask.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        mask_trans = mask.transpose(3,2)
        mask = mask_trans*mask


        x = torch.matmul(q, k)* self.scale 

        ## choose att_score or before softmax
        if "R" not in self.layer_name:
            if att_bias is None:
                x = x + full_path_matrix
                embedding = full_path_matrix
            else:
                x = x + att_bias
                embedding = att_bias
        else:
            embedding = None

        att_score = x.masked_fill((mask.int()).to(torch.bool)==False, -1e9)

        att_map = masked_softmax(att_score, mask, dim=3)

        att_map_dp = self.att_dropout(att_map)
        
        x = att_map_dp.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        x = self.output_layer(x)  
        
        assert x.size() == orig_q_size
        return x, embedding, att_map


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, head_size, layer_name):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size, layer_name)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, out_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, clue_len, mask, box_seq = None, att_bias = None):

        y = self.self_attention_norm(x)
        y, embedding, att_map = self.self_attention(y, y, y, clue_len, mask, box_seq, att_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)     
        y = self.ffn_dropout(y)
        x = x + y
        return x, embedding, att_map