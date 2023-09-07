# -*- encoding: utf-8 -*-
'''
@File    :   train_ms.py
@Time    :   2023/09/01 
@Author  :   rain 
@Mail    :   work@rainz.tech
'''
import os
import sys
import math
import random
import numpy
import logging
import argparse

from sklearn import metrics
from time import strftime, localtime
from transformers import BertModel

import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore import context

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
            self.model = opt.model_class(bert, opt)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)

        self.trainset = GeneratorDataset(ABSADataset(opt.dataset_file['train'], tokenizer), column_names=['data'], shuffle=True).batch(batch_size=opt.batch_size, drop_remainder=True)
        self.testset = GeneratorDataset(ABSADataset(opt.dataset_file['test'], tokenizer), column_names=['data'], shuffle=True).batch(batch_size=opt.batch_size)
        self.valset = self.testset

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        net_with_loss = nn.WithLossCell(self.model, criterion)
        train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
        train_network.set_train()
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1

                inputs = [batch[0][col] for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch[0]['polarity'].astype(mindspore.int32)
                loss = train_network(inputs, targets)

                n_correct += (mindspore.ops.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = float(n_correct) / float(n_total)
                    train_loss = loss_total / n_total
                    print(train_acc, train_loss)
                    logger.info('loss: {}, acc: {}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {}, val_f1: {}'.format(val_acc, val_f1))
            if val_acc >= max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}.ckpt'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                mindspore.save_checkpoint(self.model, path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 >= max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.set_train(False)
        for i_batch, t_batch in enumerate(data_loader):
            t_inputs = [t_batch[0][col] for col in self.opt.inputs_cols]
            t_targets = t_batch[0]['polarity']
            t_outputs = self.model(t_inputs)

            n_correct += (mindspore.ops.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = mindspore.ops.cat((t_targets_all, t_targets), axis=0)
                t_outputs_all = mindspore.ops.cat((t_outputs_all, t_outputs), axis=0)

        print(n_correct, n_total)
        acc = float(n_correct) / float(n_total)
        f1 = metrics.f1_score(t_targets_all.asnumpy(), mindspore.ops.argmax(t_outputs_all, -1).asnumpy(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        criterion = mindspore.nn.CrossEntropyLoss()
        optimizer = self.opt.optimizer(self.model.trainable_params(), learning_rate=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = self.trainset.create_tuple_iterator(num_epochs=self.opt.num_epoch)
        test_data_loader = self.testset.create_tuple_iterator(num_epochs=self.opt.num_epoch)
        val_data_loader = self.valset.create_tuple_iterator(num_epochs=self.opt.num_epoch)

        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        mindspore.load_param_into_net(self.model, mindspore.load_checkpoint(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='tnet_lf', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default='GPU', type=str, help='e.g. GPU')
    parser.add_argument('--device_id', default=0, type=str, help='e.g. 5')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': mindspore.common.initializer.XavierUniform,
        'xavier_normal_': mindspore.common.initializer.XavierNormal,
        'orthogonal_': mindspore.common.initializer.Orthogonal,
    }
    optimizers = {
        'adadelta': mindspore.nn.Adadelta,  # default lr=1.0
        'adagrad': mindspore.nn.Adagrad,  # default lr=0.01
        'adam': mindspore.nn.Adam,  # default lr=0.001
        'adamax': mindspore.nn.AdaMax,  # default lr=0.002
        'asgd': mindspore.nn.ASGD,  # default lr=0.01
        'rmsprop': mindspore.nn.RMSProp,  # default lr=0.01
        'sgd': mindspore.nn.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    context.set_context(device_target=opt.device , device_id=opt.device_id)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
