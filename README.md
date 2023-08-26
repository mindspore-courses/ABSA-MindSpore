# ABSA-PyTorch

> Aspect Based Sentiment Analysis, MindSpore Implementations.
>
> 基于方面的情感分析，使用MindSpore实现。
>
> 原实现仓库：https://github.com/songyouwei/ABSA-PyTorch/tree/master

## Requirement

* mindspore >= 2.0.0
* numpy >= 1.13.3
* sklearn
* python 3.6 / 3.7
* transformers

* For non-BERT-based models,
[GloVe pre-trained word vectors](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) are required, please refer to [data_utils.py](./data_utils.py) for more detail.

## Usage

### Training

```sh
python train_ms.py --model_name bert_spc --dataset restaurant
```

* All implemented models are listed in [models directory](./models/).
* See [train_ms.py](./train_ms.py) for more training arguments.

### Tips

* For non-BERT-based models, training procedure is not very stable.
* BERT-based models are more sensitive to hyperparameters (especially learning rate) on small data sets, see [this issue](https://github.com/songyouwei/ABSA-PyTorch/issues/27).
* Fine-tuning on the specific task is necessary for releasing the true power of BERT.

### Framework
For flexible training/inference and aspect term extraction, try [PyABSA](https://github.com/yangheng95/PyABSA), which includes all the models in this repository.

## Reviews / Surveys

Qiu, Xipeng, et al. "Pre-trained Models for Natural Language Processing: A Survey." arXiv preprint arXiv:2003.08271 (2020). [[pdf]](https://arxiv.org/pdf/2003.08271)

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)


## BERT-based models

### BERT-ADA ([official](https://github.com/deepopinion/domain-adapted-atsc))

Rietzler, Alexander, et al. "Adapt or get left behind: Domain adaptation through bert language model finetuning for aspect-target sentiment classification." arXiv preprint arXiv:1908.11860 (2019). [[pdf](https://arxiv.org/pdf/1908.11860)]

### BERR-PT ([official](https://github.com/howardhsu/BERT-for-RRC-ABSA))

Xu, Hu, et al. "Bert post-training for review reading comprehension and aspect-based sentiment analysis." arXiv preprint arXiv:1904.02232 (2019). [[pdf](https://arxiv.org/pdf/1904.02232)]

### ABSA-BERT-pair ([official](https://github.com/HSLCY/ABSA-BERT-pair))

Sun, Chi, Luyao Huang, and Xipeng Qiu. "Utilizing bert for aspect-based sentiment analysis via constructing auxiliary sentence." arXiv preprint arXiv:1903.09588 (2019). [[pdf](https://arxiv.org/pdf/1903.09588.pdf)]

### LCF-BERT ([lcf_bert.py](./models/lcf_bert.py)) ([official](https://github.com/yangheng95/LCF-ABSA))

Zeng Biqing, Yang Heng, et al. "LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification." Applied Sciences. 2019, 9, 3389. [[pdf]](https://www.mdpi.com/2076-3417/9/16/3389/pdf)

### AEN-BERT ([aen.py](./models/aen.py))

Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1902.09314 (2019). [[pdf]](https://arxiv.org/pdf/1902.09314.pdf)

### BERT for Sentence Pair Classification ([bert_spc.py](./models/bert_spc.py))

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). [[pdf]](https://arxiv.org/pdf/1810.04805.pdf)


## Non-BERT-based models

### ASGCN ([asgcn.py](./models/asgcn.py)) ([official](https://github.com/GeneZC/ASGCN))

Zhang, Chen, et al. "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. 2019. [[pdf]](https://www.aclweb.org/anthology/D19-1464)

### MGAN ([mgan.py](./models/mgan.py))

Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)

### AOA ([aoa.py](./models/aoa.py))

Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)

### TNet ([tnet_lf.py](./models/tnet_lf.py)) ([official](https://github.com/lixin4ever/TNet))

Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)

### Cabasc ([cabasc.py](./models/cabasc.py))

Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

### RAM ([ram.py](./models/ram.py))

Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)

### MemNet ([memnet.py](./models/memnet.py)) ([official](https://drive.google.com/open?id=1Hc886aivHmIzwlawapzbpRdTfPoTyi1U))

Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)

### IAN ([ian.py](./models/ian.py))

Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)

### ATAE-LSTM ([atae_lstm.py](./models/atae_lstm.py))

Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.

### TD-LSTM ([td_lstm.py](./models/td_lstm.py), [tc_lstm.py](./models/tc_lstm.py)) ([official](https://drive.google.com/open?id=17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6))

Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)

### LSTM ([lstm.py](./models/lstm.py))

Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780. [[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)]


## Licence

MIT
