# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import math
import mindspore
import numpy as np
from layers.dynamic_rnn import DynamicLSTM

class AOA(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        assert mindspore.tensor(embedding_matrix).dim() == 2
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.embed.embedding_table.requires_grad = False
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = mindspore.nn.Dense(2 * opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] # batch_size x seq_len
        # it goes wrong when we use mindspore.ops.sum with the type of bool, so we transform it into numpy array first.
        t_1 = mindspore.tensor(np.array(inputs[0]) != 0, mindspore.int32)
        ctx_len = mindspore.ops.sum(t_1, dim=1)
        t_2 = mindspore.tensor(np.array(inputs[1]) != 0, mindspore.int32)
        asp_len = mindspore.ops.sum(t_2, dim=1)
        ctx = self.embed(text_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = mindspore.ops.matmul(ctx_out, mindspore.ops.swapaxes(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = mindspore.ops.softmax(interaction_mat, axis=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = mindspore.ops.softmax(interaction_mat, axis=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(axis=1, keep_dims=True) # batch_size x 1 x (asp) seq_len
        gamma = mindspore.ops.matmul(alpha, beta_avg.transpose(0, 2, 1)) # batch_size x (ctx) seq_len x 1
        weighted_sum = mindspore.ops.matmul(ctx_out.transpose(0, 2, 1), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        weighted_sum = weighted_sum.astype(mindspore.dtype.float32)
        out = self.dense(weighted_sum) # batch_size x polarity_dim
        return out