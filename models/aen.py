# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import mindspore

# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(mindspore.nn.Cell):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = mindspore.nn.LogSoftMax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = mindspore.ops.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def construct(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = mindspore.ops.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return mindspore.ops.mean(loss)
        else:
            return mindspore.ops.sum(loss)



class AEN_BERT(mindspore.nn.Cell):
    def __init__(self, bert, opt):
        super(AEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = mindspore.nn.Dropout(p=opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = mindspore.nn.Dense(opt.hidden_dim*3, opt.polarities_dim)

    def construct(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = mindspore.ops.sum(context != 0, dim=-1)
        target_len = mindspore.ops.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        hc_mean = mindspore.ops.div(mindspore.ops.sum(hc, dim=1), context_len.unsqueeze(1).float())
        ht_mean = mindspore.ops.div(mindspore.ops.sum(ht, dim=1), target_len.unsqueeze(1).float())
        s1_mean = mindspore.ops.div(mindspore.ops.sum(s1, dim=1), context_len.unsqueeze(1).float())

        x = mindspore.ops.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out
