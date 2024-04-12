import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.containers import Module

class ScaledDotProductdilateAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None , dilate_p = 1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductdilateAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model//8)
        # self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.ones(h, 1, 1))
        self.V = nn.Parameter(torch.zeros(h, 1, 1))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

        self.dilate_p = dilate_p

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values,attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        
        i, j = torch.meshgrid(torch.arange(7), torch.arange(7))
        indexes = torch.stack([i.flatten(), j.flatten()]).cuda()
        # print(indexes)


        distances = torch.abs(indexes.transpose(0, 1).unsqueeze(-1) - indexes.unsqueeze(0))
        # 切比雪夫距离
        distances=torch.max(distances, dim=1, keepdim=False).values

        # 曼哈顿距离
        distances1 = torch.abs(indexes.transpose(0, 1).unsqueeze(-1) - indexes.unsqueeze(0)).sum(1)
        

        # print(distances1)
        # print(distances)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)


        R = torch.eq(distances, self.dilate_p) & torch.logical_or(torch.eq(distances1, self.dilate_p), torch.eq(distances1, 2*self.dilate_p))


        mask = R.to(torch.int)
        # print(mask)

        att = torch.matmul(q, k)       # (b_s, h, nq, nk)

        att = att * mask

        att = att / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        #注意力加入位置编码
        # w_g = box_relation_embed_matrix
        # w_a = att
        # w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        att = torch.softmax(att, -1)  # bs * 8 * r * r
        
        #att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



class MultiHeaddilateAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None,dilation=[0,2,2,3,1,2,2,3]):
        super(MultiHeaddilateAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.num_dilation = len(dilation)
        self.attention = nn.ModuleList(
            [ScaledDotProductdilateAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=1, comment=comment, dilate_p=dilation[i])
              for i in range(self.num_dilation)])
        # self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)


        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values,attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            outs=[]
            for i in range(self.num_dilation):
             outs.append(self.attention[i](q_norm, k_norm, v_norm,attention_mask, attention_weights))
            out = torch.cat(outs, dim=-1)
            # out = self.attention(q_norm, k_norm, v_norm,relative_geometry_weights,attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            outs=[]
            for i in range(self.num_dilation):
             outs.append(self.attention[i](queries, keys, values,attention_mask, attention_weights))
            out = torch.cat(outs, dim=-1)
            # out = self.attention(queries, keys, values,relative_geometry_weights,attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out
