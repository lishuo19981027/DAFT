from torch.nn import functional as F
from .utils import PositionWiseFeedForward, save_freq
import torch
from torch import nn
from .dilate_attention import MultiHeaddilateAttention
# from .grid_aug import BoxRelationalEmbedding
from .attention import MultiHeadAttention as MultiHeadAttention2
import math

class SR(nn.Module):
    def __init__(self, N, d_model=512):
        super(SR, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv1d((N) * d_model, d_model, kernel_size=3, padding=1),  # 一维卷积层
        #     nn.LeakyReLU()
        # )
        
        # self.MLP=nn.Sequential(
        #     nn.Linear(d_model,d_model),
        #     nn.LeakyReLU(),
        #     nn.Linear(d_model,d_model),
        #     nn.LeakyReLU()
        # )
        self.MLP=nn.Sequential(
            nn.Linear((N)*d_model,(N)*d_model),
            nn.LeakyReLU(),
            nn.Linear((N)*d_model,d_model),
            nn.LeakyReLU()
        )

    def forward(self,outs):
        outs = torch.cat(outs, dim=-1)  # 沿最后一个维度进行拼接
        # outs = outs.permute(0, 2, 1)  # 重新排列维度，以适应卷积层的输入
        # outs = self.conv(outs)  # 应用卷积
        # outs = outs.permute(0, 2, 1)  # 重新排列维度，以适应线性层的输入
        outs = self.MLP(outs)  # 应用MLP
        
        return outs
    
# 仿FPN，conv逐层聚合
# class SR(nn.Module):
#     def __init__(self, N, d_model=512):
#         super(SR, self).__init__()
#         self.d=d_model
#         self.CNNs = nn.ModuleList()  # 添加MLP层
        
#         for i in range(N):
            
#             # 添加conv层，确保输出维度与d_model一致
#            CNN = nn.Sequential(
#             nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
#             nn.BatchNorm2d(d_model),
#             nn.ReLU(),
#             nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
#             nn.BatchNorm2d(d_model)
#         )
#            self.CNNs.append(CNN)

#     def forward(self, outs):
#         # 初始化上一层的输出
#         prev_out = 0
        
#         # 遍历每一层
#         for i, out in enumerate(outs):
#             bs, n, c = out.size()
#             h, w = int(math.sqrt(n)), int(math.sqrt(n))
#             out = out.view(bs, h, w, c)
            
#             # 将上一层的输出的2倍与卷积结果相加
#             out = out + prev_out
            
#             # 应用MLP层，确保输出维度与d_model一致
#             out = self.CNNs[i](out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) 

#             # out = out.view(bs, -1, self.d)
            
#             # 更新上一层的输出
#             prev_out = out

#         return out.view(bs, -1, self.d)
# 仿FPN，MLP逐层聚合
# class SR(nn.Module):
#     def __init__(self, N, d_model=512):
#         super(SR, self).__init__()
#         self.mlps = nn.ModuleList()  # 添加MLP层
        
#         for i in range(N):
            
#             # 添加MLP层，确保输出维度与d_model一致
#             mlp = nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.LeakyReLU(),
#                 nn.Linear(d_model, d_model),
#                 nn.LeakyReLU()
#             )
#             self.mlps.append(mlp)

#     def forward(self, outs):
#         # 初始化上一层的输出
#         prev_out = 0
        
#         # 遍历每一层
#         for i, out in enumerate(outs):
#             # 将上一层的输出的2倍与卷积结果相加
#             out = out + prev_out
            
#             # 应用MLP层，确保输出维度与d_model一致
#             out = self.mlps[i](out)
            
#             # 更新上一层的输出
#             prev_out = out

#         return out

# 双特征预交叉模块
class CrossLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(CrossLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention2(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values,attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values,attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeaddilateAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        # self.mhatt = MultiHeadAttention2(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
        #                                 attention_module=attention_module,
        #                                 attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values,attention_mask=None, attention_weights=None):

        q = queries
        k = keys
        att = self.mhatt(q, k, values,attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self,N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.N = N
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        
        self.layers2 =CrossLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                identity_map_reordering=identity_map_reordering,
                                attention_module=attention_module,
                                attention_module_kwargs=attention_module_kwargs)
        #加入SR
        self.SR = SR(N, d_model)
        self.padding_idx = padding_idx

        # self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, pixel, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pixel_attention_mask = (torch.sum(pixel, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        
        # grid geometry embedding
        # relative_geometry_embeddings = BoxRelationalEmbedding(input)
        # flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        # box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        # box_size_per_head.insert(1, 1)
        # relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        # relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        # relative_geometry_weights = F.relu(relative_geometry_weights)
        
        outs=[]
        out = input
        out1 = pixel

        # 先交叉注意
        out2=self.layers2(out, out, out1,attention_mask, attention_weights)
        out3=self.layers2(out1, out1, out,attention_mask, attention_weights)
        out4=self.layers2(out1, out, out,attention_mask, attention_weights)
        out5=self.layers2(out, out1, out1,attention_mask, attention_weights)
        # outs.append(out2 + out3)
        out6 = out2 + out3 + out4 + out5
        
        # outs.append(out6)

        # 再单网格进行自定义注意
        for l in self.layers:
            out = l(out, out, out,attention_mask, attention_weights)
            outs.append(out)
        
        stack_out = self.SR(outs)
        out = out + 0.2 * stack_out + out6

        return out, attention_mask


class DifnetEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(DifnetEncoder, self).__init__(N, padding_idx, **kwargs)

    def forward(self, input, pixel, attention_weights=None):

        return super(DifnetEncoder, self).forward(input, pixel, attention_weights=attention_weights)

