#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = ''
# 每个worker都应该有一个run函数，不管是类还是模块
"""
import argparse

# 由于使用了自定义的分词器，所以这里要reload前声明，并且名字必须为bulid时使用的函数
from tools.vocab_build import content_tokenizer


def runner_wrap(run_arg):
    """
    
    :param run_arg: 
    :return: 
    """
    print('running:', run_arg)
    # 僵尸域名
    if run_arg.fs_train:
        from works.train.work_fine_grained_sentiment.train import Trainer
        runner = Trainer()
    elif run_arg.fs_infer:
        from works.test.work_fine_grained_sentiment.work_fine_grained_sentiment import FineGrainedSentimentInfer
        runner = FineGrainedSentimentInfer()
    else:
        raise TypeError('run arg error')

    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI APP')
    group = parser.add_mutually_exclusive_group()

    group.add_argument('-fst', '--fs_train', action='store_true', help='train')
    group.add_argument('-fsi', '--fs_infer', action='store_true', help='inference')

    args = parser.parse_args()
    runner_wrap(args)
