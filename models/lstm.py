# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import mindspore
from layers.dynamic_rnn import DynamicLSTM

class LSTM(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = mindspore.nn.Dense(opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        t_1 = mindspore.tensor(text_raw_indices != 0, mindspore.int32)
        x_len = mindspore.ops.sum(t_1, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out
