#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/6/25'
# 
"""
import random

import numpy as np
from tensorflow.contrib import learn

from tools.utility import Utility
from works.train.base.trainer_tf_base import TrainerTfBase
from works.train.work_fine_grained_sentiment.model.aspect_cnn import AspectCnn
from works.train.work_fine_grained_sentiment.model.aspect_multi_head import AspectMultiHead
from works.train.work_fine_grained_sentiment.model.aspect_rnn import AspectRnn
from works.train.work_fine_grained_sentiment.model.aspect_cnn_rnn import AspectCnnRnn
from works.train.work_fine_grained_sentiment.model.aspect_transformer import AspectTransformer


class Trainer(TrainerTfBase):
    """
    训练器
    """

    def __init__(self):
        super().__init__()
        self.log = Utility.get_logger('aspect_dev')
        self.load_config('aspect_dev')
        # 由于使用了自定义的分词器，所以要在main模块中声明url_tokenizer函数，并且名字必须为url_tokenizer
        self.content_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.config['vocab_pkl'])

    def batch_iter_train(self, batch_size, num_epochs, iterator=False):
        """
        训练数据batch
        :param iterator: 
        :param batch_size: 
        :param num_epochs: 
        :return: 
        """
        return self.batch_data(batch_size, num_epochs, self.train_data_path, ['content_token'], 'y',
                               class_num=self.train_args['num_classes'], fun=self.eval_x, onehot_y=False)

    def batch_iter_test(self, batch_size, iterator=False):
        """
        测试数据batch
        :param batch_size: 
        :param iterator: 
        :return: 
        """
        return self.batch_data(batch_size, 1, self.test_data_path, ['content_token'], 'y',
                               class_num=self.train_args['num_classes'], fun=self.eval_x_test, onehot_y=False)

    def eval_x_test(self, x, step, epoch):
        """

        :param x:
        :param step:
        :param epoch:
        :return:
        """
        contents = x[:, 0].tolist()
        content = np.array(list(self.content_vocab_processor.transform(contents)))
        return content

    def eval_x(self, x, step, epoch):
        """
        csv 文件存x为列表的字符形式，需要转换为list
        :param x: 
        :return: 
        """
        contents = x[:, 0].tolist()
        if self.train_args['enhance']:
            temps = []
            for content in contents:
                content_token = content.split(' ')
                if step % random.choice([7, 11, 30]) == 0:
                    content = self.enhance_drop(content_token)
                    self.log.debug('step {} enhance_drop'.format(step))
                # elif step % random.choice([4, 6, 9, 13, 17]) == 0: # epoch % 2 == 0 and
                #     content = self.enhance_shuffle(content_token)
                #     self.log.debug('epoch {} step {} enhance_drop'.format(epoch, step))
                else:
                    content = content_token

                temps.append(' '.join(content))
            contents = temps

        content = np.array(list(self.content_vocab_processor.transform(contents)))
        return content

    def enhance_drop(self, x):
        """
        随机drop单词 5%
        :param x:
        :return:
        """
        count = len(x)
        r = random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        e = int(r * count)
        for _ in range(e):
            choice_w = random.choice(x)
            x.remove(choice_w)

        return x

    def enhance_shuffle(self, x):
        """
        随机打乱句子
        :param x:
        :return:
        """
        x = x.copy()
        random.shuffle(x)

        return x

    def build_model(self):
        """
        搭建神经网络
        :return: 
        """
        if self.train_args['model'] == 'cnn':
            self.model = AspectCnn(
                sequence_length=self.train_args['content_len'],
                num_classes=self.train_args['num_classes'],
                vocab_size=len(self.content_vocab_processor.vocabulary_),
                embedding_size=self.train_args['embedding_size'],
                filter_sizes=self.train_args['filter_sizes'],
                num_filters=self.train_args['num_filters'],
                l2_reg_lambda=self.train_args['l2_reg_lambda'],
            )
        elif self.train_args['model'] == 'rnn':
            self.model = AspectRnn(
                sequence_length=self.train_args['content_len'],
                vocab_size=len(self.content_vocab_processor.vocabulary_),
                embedding_size=self.train_args['embedding_size'],
                num_hidden=self.train_args['num_hidden'],
                num_layer=self.train_args['num_layer'],
                num_classes=self.train_args['num_classes'],
                l2_reg_lambda=self.train_args['l2_reg_lambda'],
                bidirectional=self.train_args['bidirectional'],
                cell_type=self.train_args['cell_type'],
                atten=self.train_args['atten'],
                atten_size=self.train_args['atten_size'],
                pre_word=self.train_args['pre_word'],
                special_class=self.train_args['special_class'],
            )
        elif self.train_args['model'] == 'cnnrnn':
            self.model = AspectCnnRnn(
                sequence_length=self.train_args['content_len'],
                vocab_size=len(self.content_vocab_processor.vocabulary_),
                embedding_size=self.train_args['embedding_size'],
                num_hidden=self.train_args['num_hidden'],
                filter_sizes=self.train_args['filter_sizes'],
                num_filters=self.train_args['num_filters'],
                num_layer=self.train_args['num_layer'],
                num_classes=self.train_args['num_classes'],
                l2_reg_lambda=self.train_args['l2_reg_lambda'],
                bidirectional=self.train_args['bidirectional'],
                cell_type=self.train_args['cell_type'],
                atten=self.train_args['atten'],
                atten_size=self.train_args['atten_size'],
            )
        elif self.train_args['model'] == 'tran':
            self.model = AspectTransformer(
                sequence_length=self.train_args['content_len'],
                num_classes=self.train_args['num_classes'],
                batch_size=self.train_args['batch_size'],
                vocab_size=len(self.content_vocab_processor.vocabulary_),
                embedding_size=self.train_args['embedding_size'],
                d_model=self.train_args['embedding_size'],  # 同词嵌入大小
                d_k=self.train_args['d_k'],
                d_v=self.train_args['d_v'],
                h=self.train_args['h'],
                num_layer=self.train_args['t_num_layer'],
                l2_reg_lambda=self.train_args['l2_reg_lambda'],
                use_residual_conn=self.train_args['use_residual_conn']
            )
        elif self.train_args['model'] == 'mhead':
            self.model = AspectMultiHead(
                sequence_length=self.train_args['content_len'],
                num_classes=self.train_args['num_classes'],
                vocab_size=len(self.content_vocab_processor.vocabulary_),
                embedding_size=self.train_args['embedding_size'],
                hidden_size=self.train_args['m_hidden'],
                num_layer=self.train_args['m_num_layer'],
                l2_reg_lambda=self.train_args['l2_reg_lambda'],
            )
        else:
            raise ValueError('non support')

        self.log.info('build model {}'.format(self.model.__class__.__name__))

    def run(self):
        """
        训练
        :return: 
        """
        super().run()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
