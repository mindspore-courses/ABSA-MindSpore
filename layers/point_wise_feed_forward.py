# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import mindspore

class PositionwiseFeedForward(mindspore.nn.Cell):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = mindspore.nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = mindspore.nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.dropout = mindspore.nn.Dropout(p=dropout)
        self.relu = mindspore.nn.ReLU()

    def construct(self, x):
        output = self.relu(self.w_1(mindspore.ops.swapaxes(x, 1, 2)))
        output = self.w_2(output).swapaxes(2, 1)
        output = self.dropout(output)
        return output
