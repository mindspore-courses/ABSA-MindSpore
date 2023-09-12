# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>
import numpy as np
import numpy
from layers.dynamic_rnn import DynamicLSTM

import mindspore

class Absolute_Position_Embedding(mindspore.nn.Cell):
    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def construct(self, x, pos_inx):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = x.shape[0], x.shape[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = mindspore.tensor(weight, dtype=mindspore.float32)
        return weight

class TNet_LF(mindspore.nn.Cell):
    def __init__(self, embedding_matrix, opt):
        super(TNet_LF, self).__init__()
        rows, cols = embedding_matrix.shape
        self.embed = mindspore.nn.Embedding(rows, cols, embedding_table=mindspore.tensor(embedding_matrix, dtype=mindspore.float32))
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        D = opt.embed_dim  # 模型词向量维度
        C = opt.polarities_dim  # 分类数目
        L = opt.max_seq_len
        HD = opt.hidden_dim
        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = mindspore.nn.Conv1d(2 * HD, 50, 3, padding=1, pad_mode='pad', has_bias=True)
        self.fc1 = mindspore.nn.Dense(4 * HD, 2 * HD)
        self.fc = mindspore.nn.Dense(50, C)

    def construct(self, inputs):
        text_raw_indices, aspect_indices, aspect_in_text = inputs[0], inputs[1], inputs[2]
        t_1 = mindspore.tensor(np.array(text_raw_indices) != 0, mindspore.int32)
        feature_len = mindspore.ops.sum(t_1, dim=-1)
        t_2 = mindspore.tensor(np.array(aspect_indices) != 0, mindspore.int32)
        aspect_len = mindspore.ops.sum(t_2, dim=-1)
        feature = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = mindspore.ops.swapaxes(v, 1, 2)
        e = mindspore.ops.swapaxes(e, 1, 2)
        for i in range(2):
            a = mindspore.ops.bmm(mindspore.ops.swapaxes(e, 1, 2), v)
            a = mindspore.ops.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = mindspore.ops.bmm(e, a)
            aspect_mid = mindspore.ops.cat((aspect_mid, v), axis=1)
            aspect_mid = mindspore.ops.swapaxes(aspect_mid, 1, 2)
            aspect_mid = mindspore.ops.relu(mindspore.ops.swapaxes(self.fc1(aspect_mid), 1, 2))
            v = aspect_mid + v
            v = self.position(mindspore.ops.swapaxes(v, 1, 2), aspect_in_text)
            v = mindspore.ops.swapaxes(v, 1, 2)
        z = mindspore.ops.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = mindspore.nn.MaxPool1d(z.shape[2])(z).squeeze(2)
        out = self.fc(z)
        return out
