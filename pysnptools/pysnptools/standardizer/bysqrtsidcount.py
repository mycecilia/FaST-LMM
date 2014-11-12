import numpy as np
import scipy as sp
import logging

class BySqrtSidCount(object):  #IStandardizer  #!!LATER need to make abstract class and to document
    """The specificiation for BySqrtSidCount standardization"""
    def __init__(self, sid_count=None):
        self._sid_count = sid_count

    def standardize(self, snps, blocksize=None, force_python_only=False):
        sid_count = snps.shape[1] if self._sid_count is None else self._sid_count
        snps /= sp.sqrt(sid_count)
        return snps

    def __repr__(self): 
            return "{0}()".format(self.__class__.__name__)
        