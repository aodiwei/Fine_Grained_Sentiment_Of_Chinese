#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/5/16'
# 
"""
import re
import json
from tensorflow.contrib import learn

# 自定义解析正则表达式
URL_TOKENIZER_RE = re.compile(r"[/A-Za-z0-9_\-:=\?~&;%+@#\*\(\)\|\!\$,\}\{^\]\[`.'<>]{1}", re.UNICODE)
DOMAIN_TOKENIZER_RE = re.compile(r"[/a-z0-9_\-.]{1}", re.UNICODE)


def url_tokenizer(iterator):
    """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
    for value in iterator:
        yield URL_TOKENIZER_RE.findall(value)


def domain_tokenizer(iterator):
    """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
    for value in iterator:
        yield DOMAIN_TOKENIZER_RE.findall(value)


def title_tokenizer_(iterator):
    """

    :param iterator: 
    :return: 
    """
    import jieba
    from hanziconv import HanziConv
    for title in iterator:
        title = title.strip().lower()
        title = re.sub(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007F\u2000-\u206f\u2500-\u257F'
                       r'\u3000-\u303f\uff00-\uffef\u00A0-\u00BF\uFFF0-\uFFFF\u30fb\u2605♪™丨]', '', title)
        if re.findall(r'[\u4E00-\u9FA5]', title):
            title = HanziConv.toSimplified(title)
        if re.findall(r'[\u2E80-\u9FFF]', title):
            title_cut = jieba.cut(title)
            title = [x for x in title_cut if x != '' and x != ' ' and x != '\t' and x != '\n']
        else:
            title = title.split()
            title = [x for x in title if x != '' and x != ' ' and x != '\t' and x != '\n']
        yield title


def content_tokenizer(iterator):
    i = 0
    try:
        for value in iterator:
            i += 1
            yield value.split(' ')
    except Exception as e:
        print(e)
        pass


def build_url_vocab():
    vocab_processor = learn.preprocessing.VocabularyProcessor(128, tokenizer_fn=url_tokenizer)
    le = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:;=?@[]^_`{|}~!#$%&'()*+,-./<>"
    r = vocab_processor.fit_transform([' '.join(list(le))])
    vocab_processor.save('../src/url_vocab.pickle')
    print(list(r), '\nvocab size: ', len(vocab_processor.vocabulary_))


def build_domain_vocab():
    vocab_processor = learn.preprocessing.VocabularyProcessor(128, tokenizer_fn=domain_tokenizer)
    le = "0123456789abcdefghijklmnopqrstuvwxyz-._"
    r = vocab_processor.fit_transform([' '.join(list(le))])
    vocab_processor.save('../src/domain_vocab.pickle')
    print(list(r), '\nvocab size: ', len(vocab_processor.vocabulary_))


def build_content_vocab():
    with open(r'F:\Gdrive\aspect_ai\aspect_ai_pro\20181023\words_1.txt', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_processor = learn.preprocessing.VocabularyProcessor(800, tokenizer_fn=content_tokenizer)
    # le = ''.join(vocab)
    r = vocab_processor.fit_transform(vocab)
    vocab_processor.save('content_vocab_1_1023_800.pickle')
    print(list(r)[0], '\nvocab size: ', len(vocab_processor.vocabulary_))


def reload_vocab(path):
    """
    reload 前必须保证相应的分词函数x_tokenizer函数在main 入口处import
    :param path: 
    :return: 
    """
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(path)
    return vocab_processor


if __name__ == '__main__':
    # build_url_vocab()
    # build_domain_vocab()
    build_content_vocab()
    v = reload_vocab('content_vocab_1022.pickle')
    r = v.transform(['4 人 同行 点了 10 个 小吃'])
    print(list(r))
