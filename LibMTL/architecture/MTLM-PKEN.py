     # 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from torch.nn import MultiheadAttention

import numpy as np

import copy

# 从LibMTL.architecture.MMoE导入MMoE类
from LibMTL.architecture.MMoE import MMoE




    
class MTLM_PKEN(MMoE):
    
    # 类文档字符串，描述MTLM_PKEN类的功能和来源
    r"""Customized Gate Control (MTLM_PKEN).
    
    This method is proposed in `Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (ACM RecSys 2020 Best Paper) <https://dl.acm.org/doi/10.1145/3383313.3412236>`_ \
    and implemented by us. 

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (list): The numbers of experts shared by all the tasks and specific to each task, respectively. Each expert is an encoder network.

    """
    
    # 初始化函数，接受参数并设置一些初始值
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        # 调用父类的初始化函数
        super(MTLM_PKEN, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        # 根据传入的参数设置每个任务使用的专家数量
        self.num_experts = {task: self.kwargs['num_experts'][tn+1] for tn, task in enumerate(self.task_name)}
        # 设置共享专家的数量
        self.num_experts['share'] = self.kwargs['num_experts'][0]
        # 初始化特定任务的专家列表
        self.experts_specific = nn.ModuleDict({task: nn.ModuleList([encoder_class() for _ in range(self.num_experts[task])]) for task in self.task_name})
        # 初始化特定任务的门控网络
        # self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts['share']+self.num_experts[task]),
        #                                                         nn.Softmax(dim=-1)) for task in self.task_name})
        # 全连接
        # self.fc = nn.Sequential(nn.Linear(768 * 2, 768), nn.ReLU())
 
        # 多头注意力
        self.multihead_attn = MultiheadAttention(embed_dim=768, 
                                                    num_heads=self.kwargs['nhead'], 
                                                    dropout=self.kwargs['dropout_transformer'], 
                                                    batch_first=True)


        # transformer
        self.encoder_layer = TransformerEncoderLayer(d_model=768, 
                                                        nhead=self.kwargs['nhead'], 
                                                        dim_feedforward=self.kwargs['dim_feedforward'],
                                                        dropout=self.kwargs['dropout_transformer'],
                                                        batch_first=True)
        # self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.kwargs['num_layers'])

    # 前向传播函数，定义了模型如何处理输入数据并产生输出
    def forward(self, inputs, task_name=None):
        # 通过共享的专家网络获取共享表示
        experts_shared_rep = torch.stack([e(inputs[0]) for e in self.experts_shared])
        # 初始化输出字典
        out = {}
        # 对每个任务进行处理
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue  # 如果指定了任务名称并且当前任务不是该任务，则跳过当前任务
            # 通过特定任务的专家网络获取特定任务的表示
            experts_specific_rep = torch.stack([e(inputs[0]) for e in self.experts_specific[task]])

            # Transformer
            gate_rep = self.transformer_encoder(torch.cat([experts_shared_rep.permute(1, 0, 2), experts_specific_rep.permute(1, 0, 2), inputs[1].unsqueeze(1)], dim=1))
            
            # 多头注意力
            # rep_all = torch.cat([experts_shared_rep.permute(1, 0, 2), experts_specific_rep.permute(1, 0, 2), inputs[1].unsqueeze(1)], dim=1)
            # gate_rep = self.multihead_attn(rep_all, rep_all, rep_all)[0]

            # 直接拼接
            # gate_rep = torch.cat([experts_shared_rep.permute(1, 0, 2), experts_specific_rep.permute(1, 0, 2), inputs[1].unsqueeze(1)], dim=1)
            
            # 平均池化
            # gate_rep = torch.mean(torch.cat([experts_shared_rep.permute(1, 0, 2), experts_specific_rep.permute(1, 0, 2), inputs[1].unsqueeze(1)], dim=1))

            # 展开成一维的矩阵
            gate_rep = gate_rep.flatten(start_dim=1, end_dim=2)
            

            # 对选择的表示进行预处理
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            # 通过解码器得到任务的输出结果并存入输出字典
            out[task] = self.decoders[task](gate_rep)

            
        return out  # 返回输出字典，包含所有任务的结果
    