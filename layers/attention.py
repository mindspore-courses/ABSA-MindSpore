# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import mindspore
import mindspore.nn as nn


class Attention(nn.Cell):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = int(hidden_dim)
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = mindspore.nn.Dense(embed_dim, n_head * hidden_dim)
        self.w_q = mindspore.nn.Dense(embed_dim, n_head * hidden_dim)
        self.proj = mindspore.nn.Dense(n_head * hidden_dim, out_dim)
        self.dropout = mindspore.nn.Dropout(p=dropout)
        if score_function == 'mlp':
            self.weight = mindspore.Parameter(mindspore.numpy.randn((hidden_dim*2), dtype=mindspore.float32))
        elif self.score_function == 'bi_linear':
            self.weight = mindspore.Parameter(mindspore.numpy.randn((hidden_dim, hidden_dim), dtype=mindspore.float32))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            mindspore.ops.uniform(self.weight.data, mindspore.tensor(-stdv, mindspore.float32), mindspore.tensor(stdv, mindspore.float32))

    def construct(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = mindspore.ops.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = mindspore.ops.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = mindspore.ops.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = mindspore.ops.bmm(qx, kt)
            score = mindspore.ops.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = mindspore.ops.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = mindspore.ops.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = mindspore.ops.cat((kxx, qxx), axis=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = mindspore.ops.tanh(mindspore.ops.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = mindspore.ops.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = mindspore.ops.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = mindspore.ops.softmax(score, axis=-1)
        output = mindspore.ops.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = mindspore.ops.cat(mindspore.ops.split(output, mb_size, axis=0), axis=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = mindspore.Parameter(mindspore.numpy.randn((q_len, embed_dim), dtype=mindspore.float32))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        mindspore.ops.uniform(self.q.data, mindspore.tensor(-stdv, mindspore.float32), mindspore.tensor(stdv, mindspore.float32))

    def construct(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).construct(k, q)
