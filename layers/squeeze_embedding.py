# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import mindspore
import numpy
import mindspore.numpy as np
from layers.p_sequence import pack_padded_sequence, pad_packed_sequence

class SqueezeEmbedding(mindspore.nn.Cell):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def construct(self, x, x_len):
        x_sort_idx = mindspore.ops.sort(mindspore.tensor(x_len, dtype=mindspore.int32), descending=True)[1].long()
        x_unsort_idx = mindspore.ops.sort(mindspore.tensor(x_sort_idx, dtype=mindspore.int32), descending=True)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        for idx in range(x.shape[0]):
            np.pad(x[idx], x_len[0]-x_len[idx])
        out = x
        out = out[0] 
        out = out[x_unsort_idx]
        return out
