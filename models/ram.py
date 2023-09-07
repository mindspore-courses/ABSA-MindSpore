# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import mindspore
import numpy as np

class RAM(mindspore.nn.Cell):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.numpy()
        left_len = left_len.numpy()
        aspect_len = aspect_len.numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1-(left_len[i]-idx)/memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i]+aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i]+aspect_len[i], memory_len[i]):
                weight[i].append(1-(idx-left_len[i]-aspect_len[i]+1)/memory_len[i])
                u[i].append(idx-left_len[i]-aspect_len[i]+1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = mindspore.tensor(u, dtype=memory.dtype).unsqueeze(2)
        weight = mindspore.tensor(weight).unsqueeze(2)
        v = memory*weight
        memory = mindspore.ops.cat([v, u], axis=2)
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.att_linear = mindspore.nn.Dense(opt.hidden_dim*2 + 1 + opt.embed_dim*2, 1)
        self.gru_cell = mindspore.nn.GRUCell(opt.hidden_dim*2 + 1, opt.embed_dim)
        self.dense = mindspore.nn.Dense(opt.embed_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs[0], inputs[1], inputs[2]
        t_1 = mindspore.tensor(np.array(text_left_indices) != 0, mindspore.int32)
        left_len = mindspore.ops.sum(t_1, dim=-1)
        t_2 = mindspore.tensor(np.array(text_raw_indices) != 0, mindspore.int32)
        memory_len = mindspore.ops.sum(t_2, dim=-1)
        t_3 = mindspore.tensor(np.array(aspect_indices) != 0, mindspore.int32)
        aspect_len = mindspore.ops.sum(t_3, dim=-1)
        nonzeros_aspect = aspect_len.float()

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        
        aspect = self.embed(aspect_indices)
        aspect = mindspore.ops.sum(aspect, dim=1)
        aspect = mindspore.ops.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = mindspore.ops.zeros_like(aspect)

        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        for _ in range(self.opt.hops):
            g = self.att_linear(mindspore.ops.cat([memory, 
                mindspore.ops.zeros((batch_size, seq_len, self.opt.embed_dim)) + et.unsqueeze(1), 
                mindspore.ops.zeros((batch_size, seq_len, self.opt.embed_dim)) + aspect.unsqueeze(1)], 
                axis=-1))
            alpha = mindspore.ops.softmax(g, axis=1)
            i = mindspore.ops.bmm(mindspore.ops.swapaxes(alpha, 1, 2), memory).squeeze(1)  
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out
