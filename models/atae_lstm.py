# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import numpy as np
import mindspore
from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = mindspore.nn.Dense(opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        t_1 = mindspore.tensor(np.array(text_indices) != 0, mindspore.int32)
        x_len = mindspore.ops.sum(t_1, dim=-1)
        aspect_len = mindspore.ops.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = mindspore.ops.div(mindspore.ops.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).broadcast_to((-1, 85, -1))
        x = mindspore.ops.cat((aspect, x), axis=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = mindspore.ops.cat((h, aspect), axis=-1)
        _, score = self.attention(ha)
        output = mindspore.ops.squeeze(mindspore.ops.bmm(score, h), axis=1)

        out = self.dense(output)
        return out
