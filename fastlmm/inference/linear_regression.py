"""
Created on 2013-08-02
@author: Christian Widmer <chris@shogun-toolbox.org>
@summary: Module for univariate feature selection in the presence of covariates


Motivated by sklearn's linear regression method for feature
selection, we've come up with an extended version that takes
care of covariates

based on sklearn code (f_regression):
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/univariate_selection.py

"""

import numpy as np
from sklearn.utils import safe_sqr, check_arrays
from scipy import stats

 

#def TESTBEFOREUSING_get_example_data():
#    """
#    load plink files
#    """
    
#    import fastlmm.pyplink.plink as plink
#    import pysnptools.snpreader.bed as Bed
#    import fastlmm.util.util as util
    
    
#    ipheno = 0
#    foldIter = 0


#    """
#    import dataset
#    dat = dataset.importDataset("pheno4")

#    fn_bed = dat["bedFile"]
#    fn_pheno = dat["phenoFile"]
#    """

#    fn_bed = "../../featureSelection/examples/toydata"
#    fn_pheno = "../../featureSelection/examples/toydata.phe"

    
#    pheno = pstpheno.loadPhen(fn_pheno)
  
#    # load data
#    bed = plink.Bed(fn_bed)
    
#    indarr = util.intersect_ids([pheno['iid'],bed.iid])
    
#    pheno['iid'] = pheno['iid'][indarr[:,0]]
#    pheno['vals'] = pheno['vals'][indarr[:,0]]
#    bed = bed[indarr[:,1],:]

#    N = pheno['vals'].shape[0]
#    y = pheno['vals'][:,ipheno]
#    iid = pheno['iid']

#    snps = bed.read().standardize()

#    return snps, y


def f_regression_block(fun,X,y,blocksize=None,**args):
   """
   runs f_regression for each block seperately (saves memory).

   -------------------------
   fun  : method that returns statistics,pval
   X    : {array-like, sparse matrix}  shape = (n_samples, n_features)
          The set of regressors that will tested sequentially.
   y    : array of shape(n_samples).
          The data matrix
   blocksize    : number of SNPs per block
   """
   if blocksize==None:
       return fun(X,y,**args)

   idx_start = 0
   idx_stop = int(blocksize)
    
   pval = np.zeros(X.shape[1])
   stats = np.zeros(X.shape[1])

   while idx_start<X.shape[1]:
        stats[idx_start:idx_stop], pval[idx_start:idx_stop] = fun(X[:,idx_start:idx_stop],y,**args)

        idx_start = idx_stop
        idx_stop += blocksize
        if idx_stop>X.shape[1]:
            idx_stop = X.shape[1]

   return stats,pval


def f_regression_cov_alt(X, y, C):
    """
    Implementation as derived in tex document

    See pg 12 of following document for definition of F-statistic
    http://www-stat.stanford.edu/~jtaylo/courses/stats191/notes/simple_diagnostics.pdf

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    c : {array-like, sparse matrix}  shape = (n_samples, n_covariates)
        The set of covariates.


    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.
    """
    # make sure we don't overwrite input data
    old_flag_X = X.flags.writeable
    old_flag_C = C.flags.writeable
    old_flag_y = y.flags.writeable
    X.flags.writeable = False
    C.flags.writeable = False
    y.flags.writeable = False


    #X, C, y = check_arrays(X, C, y, dtype=np.float)
    y = y.ravel()

    # make copy of input data
    X = X.copy(order="F")
    y = y.copy()

    assert C.shape[1] < C.shape[0]
    cpinv = np.linalg.pinv(C)
    X -= np.dot(C,(np.dot(cpinv, X))) #most expensive line (runtime)
    y -= np.dot(C,(np.dot(cpinv, y)))

    yS = safe_sqr(y.T.dot(X)) # will create a copy

    # Note: (X*X).sum(0) = X.T.dot(X).diagonal(), computed efficiently
    # see e.g.: http://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
    # TODO: make this smarter using either stride tricks or cython
    X *= X
    denom = X.sum(0) * y.T.dot(y) - yS
    F = yS / denom

    # degrees of freedom
    dof = (X.shape[0] - 1 - C.shape[1]) / (1) #(df_fm / (df_rm - df_fm))
    F *= dof

    # convert to p-values
    pv = stats.f.sf(F, 1, dof)

    # restore old state
    X.flags.writeable = old_flag_X
    C.flags.writeable = old_flag_C
    y.flags.writeable = old_flag_y

    return F, pv


def f_regression_cov(X, y, C):
    """Univariate linear regression tests

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 3 steps:
    1. the regressor of interest and the data are orthogonalized
    wrt constant regressors
    2. the cross correlation between data and regressors is computed
    3. it is converted to an F score then to a p-value

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    c : {array-like, sparse matrix}  shape = (n_samples, n_covariates)
        The set of covariates.


    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.
    """

    X, C, y = check_arrays(X, C, y, dtype=np.float)
    y = y.ravel()

    assert C.shape[1] < C.shape[0]
    cpinv = np.linalg.pinv(C)
    X -= np.dot(C,(np.dot(cpinv, X)))
    y -= np.dot(C,(np.dot(cpinv, y)))

    # compute the correlation
    corr = np.dot(y, X)
    corr /= np.asarray(np.sqrt(safe_sqr(X).sum(axis=0))).ravel()
    corr /= np.asarray(np.sqrt(safe_sqr(y).sum())).ravel()

    # convert to p-value
    dof = (X.shape[0] - 1 - C.shape[1]) / (1) #(df_fm / (df_rm - df_fm))
    F = corr ** 2 / (1 - corr ** 2) * dof
    pv = stats.f.sf(F, 1, dof)
    return F, pv


def test_bias():
    """
    make sure we get the same result for setting C=unitvec
    """

    S, y = get_example_data()
    C = np.ones((len(y),1))

    from sklearn.feature_selection import f_regression

    F1, pval1 = f_regression(S, y, center=True)
    F2, pval2 = f_regression_cov(S, C, y)
    F3, pval3 = f_regression_cov_alt(S, C, y)

    # make sure values are the same
    np.testing.assert_array_almost_equal(F1, F2)
    np.testing.assert_array_almost_equal(F2, F3)
    np.testing.assert_array_almost_equal(pval1, pval2)
    np.testing.assert_array_almost_equal(pval2, pval3)


def test_cov():
    """
    compare different implementations, make sure results are the same
    """

    S, y = get_example_data()
    C = S[:,0:10]
    S = S[:,10:]

    F1, pval1 = f_regression_cov(S, C, y)
    F2, pval2 = f_regression_cov_alt(S, C, y)

    np.testing.assert_array_almost_equal(F1, F2)
    np.testing.assert_array_almost_equal(pval1, pval2)


def main():
 
    test_cov()
    test_bias()


if __name__ == "__main__":
    main()

