#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = '常量'
__author__ = 'David Ao'
__mtime__ = '2017/6/13'
# 
"""
import numpy as np


class _Const:
    """
    some const val
    """
    LOCAL_FORMAT_DATETIME = "%Y-%m-%d %H:%M:%S"
    LOCAL_FORMAT_DATETIME_PRX = "%Y%m%d%H%M%S"
    LOCAL_FORMAT_DATETIME_FILE = "%m%d_%H_%M_%S"
    LOCAL_FORMAT_DATE = "%Y-%m-%d"
    LOCAL_FORMAT_DATE_YMD = "%Y%m%d"
    LOCAL_FORMAT_DATE_M = "%Y%m"
    LOCAL_FORMAT_HM = "%H%M"

    UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    @classmethod
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise cls.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise cls.ConstCaseError('const name "%s" is not all uppercase' % name)
        cls.__dict__[name] = value


# 对外调用
Const = _Const()
