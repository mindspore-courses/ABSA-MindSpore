# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
from layers.squeeze_embedding import SqueezeEmbedding
import numpy as np
import mindspore


class MemNet(mindspore.nn.Cell):
    
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1-float(idx+1)/memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = mindspore.tensor(weight)
        memory = weight.unsqueeze(2)*memory
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MemNet, self).__init__()
        self.opt = opt
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = mindspore.nn.Dense(opt.embed_dim, opt.embed_dim)
        self.dense = mindspore.nn.Dense(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        t_1 = mindspore.tensor(np.array(text_raw_without_aspect_indices) != 0, mindspore.int32)
        memory_len = mindspore.ops.sum(t_1, dim=-1)
        t_2 = mindspore.tensor(np.array(aspect_indices) != 0, mindspore.int32)
        aspect_len = mindspore.ops.sum(t_2, dim=-1)
        nonzeros_aspect = mindspore.tensor(aspect_len, dtype=mindspore.float32)

        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = mindspore.ops.sum(aspect, dim=1)
        aspect = mindspore.ops.div(aspect, nonzeros_aspect.view(nonzeros_aspect.shape[0], 1))
        x = aspect.unsqueeze(1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.shape[0], -1)
        out = self.dense(x)
        return out
