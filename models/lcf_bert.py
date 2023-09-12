# -*- coding: utf-8 -*-
# The code is based on repository: https://github.com/songyouwei/ABSA-PyTorch
# author: Runjia Zeng <rain1709@foxmail.com>

import copy
import numpy as np
import mindspore
from mindnlp.models.bert.bert import BertPooler, BertSelfAttention

class SelfAttention(mindspore.nn.Cell):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = mindspore.nn.Tanh()

    def construct(self, inputs):
        zero_tensor = mindspore.tensor(np.zeros((inputs.shape[0], 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=mindspore.float32)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class LCF_BERT(mindspore.nn.Cell):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert   # Default to use single Bert and reduce memory requirements
        self.dropout = mindspore.nn.Dropout(p=opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = mindspore.nn.Dense(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = mindspore.nn.Dense(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = mindspore.nn.Dense(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.numpy()
        asps = aspect_indices.numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.shape[0], self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = mindspore.Tensor.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.numpy()
        asps = aspect_indices.numpy()
        masked_text_raw_indices = np.ones((text_local_indices.shape[0], self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.opt.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = mindspore.Tensor.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices

    def construct(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, _ = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out, _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = mindspore.ops.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            bert_local_out = mindspore.ops.mul(bert_local_out, weighted_text_local_features)

        out_cat = mindspore.ops.cat((bert_local_out, bert_spc_out), axis=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out
