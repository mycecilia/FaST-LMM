import numpy as np
import scipy as sp
import logging

class Beta(object): #IStandardizer
    """The specificiation for beta standardization"""
    def __init__(self,a=1,b=25):
        import warnings
        #!!warnings.warn("This Beta is deprecated. Pysnptools includes newer versions of Beta", DeprecationWarning)
        self.a = a
        self.b = b

    def standardize(self, snps, blocksize=None, force_python_only=False):
        l = self.lambdaFactory(snps, blocksize=blocksize, force_python_only=force_python_only)
        import fastlmm.util.standardizer as stdizer
        return stdizer.standardize_with_lambda(snps, l, blocksize)


    def lambdaFactory(self, snps, blocksize=None, force_python_only=False):
        from pysnptools.snpreader import wrap_plink_parser
        if not force_python_only:
            if snps.dtype == np.float64:
                if snps.flags['F_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes): #!!create a method called is_single_segment
                    return lambda s, a=self.a, b=self.b : wrap_plink_parser.standardizedoubleFAAA(s,True,a,b)
                elif snps.flags['C_CONTIGUOUS']  and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes) and blocksize is None:
                    return lambda s, a=self.a, b=self.b  : wrap_plink_parser.standardizedoubleCAAA(s,True,a,b)
                else:
                    logging.info("Array is not contiguous, so will standarize with python only instead of C++")
            elif snps.dtype == np.float32:
                if snps.flags['F_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes):
                    return lambda s, a=self.a, b=self.b : wrap_plink_parser.standardizefloatFAAA(s,True,a,b)
                elif snps.flags['C_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes) and blocksize is None:
                    return lambda s, a=self.a, b=self.b : wrap_plink_parser.standardizefloatCAAA(s,True,a,b)
                else:
                    logging.info("Array is not contiguous, so will standarize with python only instead of C++")
            else:
                logging.info("Array type is not float64 or float32, so will standarize with python only instead of C++")

        import fastlmm.util.standardizer as stdizer
        return lambda s, a=self.a, b=self.b, stdizer=stdizer: stdizer.standardize_beta_python(s, a, b)
