# -*- coding: utf-8 -*-
# file: atae-lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import mindspore
from layers.squeeze_embedding import SqueezeEmbedding


class ATAE_LSTM(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(mindspore.tensor(embedding_matrix, dtype=ms.float32))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = mindspore.nn.Dense(opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x_len = mindspore.ops.sum(text_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = mindspore.ops.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = mindspore.ops.div(mindspore.ops.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        x = mindspore.ops.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = mindspore.ops.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        return out
