# -*- coding: utf-8 -*-
# file: tc_lstm.py
# author: songyouwei <youwei0314@gmail.com>, ZhangYikai <zykhelloha@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import mindspore

class TC_LSTM(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(TC_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(mindspore.tensor(embedding_matrix, dtype=ms.float32))
        self.lstm_l = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = mindspore.nn.Dense(opt.hidden_dim*2, opt.polarities_dim)

    def construct(self, inputs):
        # Get the target and its length(target_len)
        x_l, x_r, target = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = mindspore.ops.sum(x_l != 0, dim=-1), mindspore.ops.sum(x_r != 0, dim=-1)
        target_len = mindspore.ops.sum(target != 0, dim=-1, dtype=ms.float32)[:, None, None]
        x_l, x_r, target = self.embed(x_l), self.embed(x_r), self.embed(target)
        v_target = mindspore.ops.div(target.sum(dim=1, keepdim=True),
                             target_len)  # v_{target} in paper: average the target words

        # the concatenation of word embedding and target vector v_{target}:
        x_l = mindspore.ops.cat(
            (x_l, mindspore.ops.cat(([v_target] * x_l.shape[1]), 1)),
            2
        )
        x_r = mindspore.ops.cat(
            (x_r, mindspore.ops.cat(([v_target] * x_r.shape[1]), 1)),
            2
        )

        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = mindspore.ops.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out