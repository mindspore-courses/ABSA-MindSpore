# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import mindspore
import numpy as np

class DynamicLSTM(mindspore.nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = mindspore.nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = mindspore.nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = mindspore.nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                has_bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def construct(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = mindspore.ops.sort(mindspore.tensor(x_len, dtype=mindspore.int32), descending=True)[1].long()
        x_unsort_idx = mindspore.ops.sort(mindspore.tensor(x_sort_idx, dtype=mindspore.int32), descending=True)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        # process using the selected RNN
        if self.rnn_type == 'LSTM': 
            out_pack, (ht, ct) = self.RNN(x, None, x_len)
        else: 
            out_pack, ht = self.RNN(x, None, x_len)
            ct = None
        """unsort: h"""
        ht = mindspore.ops.swapaxes(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = mindspore.ops.swapaxes(ht, 0, 1)
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = out_pack[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = mindspore.ops.swapaxes(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = mindspore.ops.swapaxes(ct, 0, 1)
            return out, (ht, ct)
