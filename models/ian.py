# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import mindspore

class IAN(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(mindspore.tensor(embedding_matrix, dtype=ms.float32))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = mindspore.nn.Dense(opt.hidden_dim*2, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = mindspore.ops.sum(text_raw_indices != 0, dim=-1)
        aspect_len = mindspore.ops.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = mindspore.tensor(aspect_len, dtype=ms.float32)
        aspect_pool = mindspore.ops.sum(aspect, dim=1)
        aspect_pool = mindspore.ops.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = mindspore.tensor(text_raw_len, dtype=ms.float32)
        context_pool = mindspore.ops.sum(context, dim=1)
        context_pool = mindspore.ops.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = mindspore.ops.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out
