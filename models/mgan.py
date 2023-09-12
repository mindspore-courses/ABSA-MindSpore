# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import mindspore
import numpy as np
from layers.dynamic_rnn import DynamicLSTM

class LocationEncoding(mindspore.nn.Cell):
    def __init__(self, opt):
        super(LocationEncoding, self).__init__()
        self.opt = opt

    def construct(self, x, pos_inx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][0]):
                relative_pos = pos_inx[i][0] - j
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            for j in range(pos_inx[i][0], pos_inx[i][1] + 1):
                weight[i].append(0)
            for j in range(pos_inx[i][1] + 1, seq_len):
                relative_pos = j - pos_inx[i][1]
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
        weight = mindspore.tensor(weight)
        return weight

class AlignmentMatrix(mindspore.nn.Cell):
    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = mindspore.Parameter(mindspore.numpy.randn((6*opt.hidden_dim, 1), dtype=mindspore.float32))

    def construct(self, batch_size, ctx, asp):
        ctx_len = ctx.shape[1]
        asp_len = asp.shape[1]
        alignment_mat = mindspore.ops.zeros((batch_size, ctx_len, asp_len), mindspore.float32)
        ctx_chunks = ctx.chunk(ctx_len, axis=1)
        asp_chunks = asp.chunk(asp_len, axis=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = mindspore.ops.cat([ctx_chunk, asp_chunk, ctx_chunk*asp_chunk], axis=2) # batch_size x 1 x 6*hidden_dim 
                alignment_mat[:, i, j] = feat.matmul(self.w_u.expand_dims(0)).squeeze(-1).squeeze(-1) 
        return alignment_mat

class MGAN(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(MGAN, self).__init__()
        self.opt = opt
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.location = LocationEncoding(opt)
        self.w_a2c = mindspore.Parameter(mindspore.numpy.randn((2*opt.hidden_dim, 2*opt.hidden_dim), dtype=mindspore.float32))
        self.w_c2a = mindspore.Parameter(mindspore.numpy.randn((2*opt.hidden_dim, 2*opt.hidden_dim), dtype=mindspore.float32))
        self.alignment = AlignmentMatrix(opt)
        self.dense = mindspore.nn.Dense(8*opt.hidden_dim, opt.polarities_dim)

    def construct(self, inputs):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] 
        text_left_indices= inputs[2]
        batch_size = text_raw_indices.shape[0]
        t_1 = mindspore.tensor(np.array(text_raw_indices) != 0, mindspore.int32)
        ctx_len = mindspore.ops.sum(t_1, dim=1)
        t_2 = mindspore.tensor(np.array(aspect_indices) != 0, mindspore.int32)
        asp_len = mindspore.ops.sum(t_2, dim=1)
        t_3 = mindspore.tensor(np.array(text_left_indices) != 0, mindspore.int32)
        left_len = mindspore.ops.sum(t_3, dim=-1)
        aspect_in_text = mindspore.ops.cat([left_len.unsqueeze(-1), (left_len+asp_len-1).unsqueeze(-1)], axis=-1)

        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim

        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) 
        ctx_out = self.location(ctx_out, aspect_in_text) # batch_size x (ctx)seq_len x 2*hidden_dim
        ctx_pool = mindspore.ops.sum(ctx_out, dim=1)
        ctx_pool = mindspore.ops.div(ctx_pool, ctx_len.float().unsqueeze(-1)).unsqueeze(-1) # batch_size x 2*hidden_dim x 1

        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp)seq_len x 2*hidden_dim
        asp_pool = mindspore.ops.sum(asp_out, dim=1)
        asp_pool = mindspore.ops.div(asp_pool, asp_len.float().unsqueeze(-1)).unsqueeze(-1) # batch_size x 2*hidden_dim x 1

        alignment_mat = self.alignment(batch_size, ctx_out, asp_out) # batch_size x (ctx)seq_len x (asp)seq_len
        # batch_size x 2*hidden_dim
        f_asp2ctx = mindspore.ops.matmul(mindspore.ops.swapaxes(ctx_out, 1, 2), mindspore.ops.softmax(alignment_mat.max(2, keepdims=True)[0], axis=1)).squeeze(-1)
        f_ctx2asp = mindspore.ops.matmul(mindspore.ops.softmax(mindspore.ops.max(alignment_mat, 1, keepdims=True)[0], axis=2), asp_out)
        f_ctx2asp = mindspore.ops.swapaxes(f_ctx2asp, 1, 2).squeeze(-1) 
        
        c_asp2ctx_alpha = mindspore.ops.softmax(ctx_out.matmul(self.w_a2c.expand_dims(0)).matmul(asp_pool), axis=1)
        c_asp2ctx = mindspore.ops.matmul(mindspore.ops.swapaxes(ctx_out, 1, 2), c_asp2ctx_alpha).squeeze(-1)
        c_ctx2asp_alpha = mindspore.ops.softmax(asp_out.matmul(self.w_c2a.expand_dims(0)).matmul(ctx_pool), axis=1)
        c_ctx2asp = mindspore.ops.matmul(mindspore.ops.swapaxes(asp_out, 1, 2), c_ctx2asp_alpha).squeeze(-1)

        feat = mindspore.ops.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], axis=1)
        out = self.dense(feat) # bathc_size x polarity_dim

        return out