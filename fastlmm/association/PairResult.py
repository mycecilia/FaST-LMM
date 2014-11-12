class PairResult:
    '''
    Similar to class Result, but used when testing pairs of sets
    'lik0'          : null likelihood
                        'nLL'       : negative log-likelihood
                        'sigma2'    : the model variance sigma^2
                        'beta'      : [D*1] array of fixed effects weights beta
                        'h2'        : mixture weight between Covariance and noise
                        'REML'      : True: REML was computed, False: ML was computed
                        'a2'        : mixture weight between K0 and K1
    'lik1'          : alternative likelihood
                        'nLL'       : negative log-likelihood
                        'sigma2'    : the model variance sigma^2
                        'beta'      : [D*1] array of fixed effects weights beta
                        'h2'        : mixture weight between Covariance and noise
                        'REML'      : True: REML was computed, False: ML was computed
                        'a2'        : mixture weight between K0 and K1
    'nexclude'      : array of the number of excluded snps from null
    'test'          : "lrt", "sc_davies", sc_..."
    '''

    def __init__(self,setname,iset,setname2,iset2,iperm):
        self.setname = setname
        self.iset = iset
        self.setname2 = setname2
        self.iset2 = iset2
        self.iperm = iperm

    # computing observed lrt statistics and a2 parameters
    @property
    def lrt(self):
        return 2 * (self.lik0['nLL'] - self.lik1['nLL'])

    @property
    def a2(self):
        return self.lik1['a2']


