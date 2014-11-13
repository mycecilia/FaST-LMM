import numpy as np

class DiagKtoN(object):
    """The specificiation for diag(K)=N standardization"""
    def __init__(self, N):
        self._N = N

    def standardize(self, snps):
        flag = snps.flags.writeable
        vec = snps.reshape(-1, order="A")
        
        # make sure no copy was made
        assert vec.base is snps
        squared_sum = vec.dot(vec)
        factor = 1./np.sqrt(squared_sum / float(self._N))
        
        snps.flags.writeable = flag
        
        snps *= factor
        
        return snps

    def __repr__(self): 
            return "{0}()".format(self.__class__.__name__)
