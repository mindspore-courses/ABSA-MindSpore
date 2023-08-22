# -*- coding: utf-8 -*-
# file: asgcn.py
# author:  <gene_zhangchen@163.com>
# Copyright (C) 2020. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from mindspore import Tensor
import mindspore

class GraphConvolution(mindspore.nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mindspore.Parameter(Tensor(np.array(in_features, out_features), mindspore.float32))
        if bias:
            self.bias = mindspore.Parameter(Tensor(np.array(out_features), mindspore.float32))
        else:
            self.register_parameter('bias', None)

    def construct(self, text, adj):
        hidden = mindspore.ops.matmul(text, self.weight)
        denom = mindspore.ops.sum(adj, dim=2, keepdim=True) + 1
        output = mindspore.ops.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ASGCN(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = mindspore.nn.Embedding(mindspore.tensor(embedding_matrix, dtype=ms.float32))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = mindspore.nn.Dense(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = mindspore.nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = mindspore.tensor(weight, dtype=ms.float32).unsqueeze(2)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = mindspore.tensor(mask, dtype=ms.float32).unsqueeze(2)
        return mask*x

    def construct(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = mindspore.ops.sum(text_indices != 0, dim=-1)
        aspect_len = mindspore.ops.sum(aspect_indices != 0, dim=-1)
        left_len = mindspore.ops.sum(left_indices != 0, dim=-1)
        aspect_double_idx = mindspore.ops.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = mindspore.ops.matmul(x, text_out.transpose(1, 2))
        alpha = mindspore.ops.softmax(alpha_mat.sum(1, keepdim=True), axis=2)
        x = mindspore.ops.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output