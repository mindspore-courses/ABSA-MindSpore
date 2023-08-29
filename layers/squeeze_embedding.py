# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import mindspore
import numpy as np
from layers.p_sequence import pack_padded_sequence, pad_packed_sequence

class SqueezeEmbedding(mindspore.nn.Cell):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def construct(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = mindspore.ops.sort(-x_len)[1].long()
        x_unsort_idx = mindspore.ops.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """unpack: out"""
        out = pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]  #
        """unsort"""
        out = out[x_unsort_idx]
        return out
