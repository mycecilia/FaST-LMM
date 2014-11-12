import numpy as np
import scipy as sp
import logging

class Identity(object):  #IStandardizer #!!LATER make an abstract object
    """The specificiation for unit standardization"""
    def __init__(self):
        pass

    def standardize(self, snps, blocksize=None, force_python_only=False):
        return snps

    def __repr__(self): 
        return "{0}()".format(self.__class__.__name__)

    def lambdaFactory(self, snps, blocksize=None, force_python_only=False):
        import pysnptools.pysnptools.standardizer as stdizer
        return lambda s : s



