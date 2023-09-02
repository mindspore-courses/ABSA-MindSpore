# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import mindspore
import numpy
import mindspore.numpy as np

class SqueezeEmbedding(mindspore.nn.Cell):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def construct(self, x, x_len):
        x_sort_idx = mindspore.ops.sort(mindspore.tensor(x_len, dtype=mindspore.int32), descending=True)[1].long()
        x = x[x_sort_idx]
        return x
