import numpy as np
import scipy as sp
import logging

class Unit(object):  #IStandardizer
    """The specification for unit standardization"""
    def __init__(self):
        import warnings
        #!!warnings.warn("This Unit is deprecated. Pysnptools includes newer versions of Unit", DeprecationWarning)
        pass

    def standardize(self, snps, blocksize=None, force_python_only=False):
        l = self.lambdaFactory(snps, blocksize=blocksize, force_python_only=force_python_only)
        import fastlmm.util.standardizer as stdizer
        return stdizer.standardize_with_lambda(snps, l, blocksize)


    def lambdaFactory(self, snps, blocksize=None, force_python_only=False):
        from pysnptools.snpreader import wrap_plink_parser
        if not force_python_only:
            if snps.dtype == np.float64:
                if snps.flags['F_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes):
                    return lambda s : wrap_plink_parser.standardizedoubleFAAA(s,False,float("NaN"),float("NaN"))
                elif snps.flags['C_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes) and blocksize is None:
                    return lambda s : wrap_plink_parser.standardizedoubleCAAA(s,False,float("NaN"),float("NaN"))
                else:
                    logging.info("Array is not contiguous, so will standarize with python only instead of C++")
            elif snps.dtype == np.float32:
                if snps.flags['F_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes):
                    return lambda s: wrap_plink_parser.standardizefloatFAAA(s,False,float("NaN"),float("NaN"))
                elif snps.flags['C_CONTIGUOUS'] and (snps.flags["OWNDATA"] or snps.base.nbytes == snps.nbytes) and blocksize is None:
                    return lambda s: wrap_plink_parser.standardizefloatCAAA(s,False,float("NaN"),float("NaN"))
                else:
                    logging.info("Array is not contiguous, so will standarize with python only instead of C++")
            else:
                logging.info("Array type is not float64 or float32, so will standarize with python only instead of C++")

        import fastlmm.util.standardizer as stdizer
        return lambda s, stdizer=stdizer: stdizer.standardize_unit_python(s)



    def __str__(self):
        return "{0}()".format(self.__class__.__name__)
