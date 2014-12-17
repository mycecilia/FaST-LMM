from fastlmm.util.runner import *
import logging
import fastlmm.pyplink.plink as plink
import pysnptools.util.pheno as pstpheno
import pysnptools.util as pstutil
import fastlmm.util.util as flutil
import numpy as np
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd
from fastlmm.inference.lmm_cov import LMM as fastLMM

def single_snp(test_snps,pheno,
                 G0=None, G1=None, mixing=0.0, #!!test mixing and G1
                 covar=None, output_file_name=None, log_delta=None, min_log_delta=-5, max_log_delta=10,
                 cache_file = None):
    """
    Function performing single SNP GWAS with REML

    :param test_snps: SNPs to test. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type pheno: a 'pheno dictionary' or a string

    :param G0: SNPs from which to construct a similarity matrix.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G0: a :class:`.SnpReader` or a string

    :param G1: SNPs from which to construct a second similarity kernel, optional. Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G1: a :class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to G1 relative to G0.
            If you give no mixing number, G0 will get all the weight and G1 will be ignored.
    :type mixing: number

    :param covar: covariate information, optional: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type covar: a 'pheno dictionary' or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param log_delta: A parameter to LMM learning, optional
            If not given will search for best value.
    :type log_delta: number

    :param min_log_delta: (default:-5)
            When searching for log_delta, the lower bounds of the search.
    :type min_log_delta: number

    :param max_log_delta: (default:10)
            When searching for log_delta, the upper bounds of the search.
    :type max_log_delta: number

    :param cache_file: Name of  file to read or write cached precomputation values to, optional.
                If not given, no cache file will be used.
                If given and file does not exists, will write precomputation values to file.
                If given and file does exists, will read precomputation values from file.
                The file contains the U and S matrix from the decomposition of the training matrix. It is in Python's np.savez (*.npz) format.
                Calls using the same cache file should have the same 'G0' and 'G1'
                If given and the file does exist then G0 and G1 need not be given.
    :type cache_file: file name





    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> snpreader = Bed("../feature_selection/examples/toydata")
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp(test_snps=snpreader[:,5000:10000],pheno=pheno_fn,G0=snpreader[:,0:5000],log_delta=np.log(4.0))
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_7487 3.4e-06 5000

    """
    t0 = time.time()
    test_snps = _snp_fixup(test_snps)
    pheno = _pheno_fixup(pheno)
    covar = _pheno_fixup(covar, iid_source_if_none=pheno)

    if G0 is not None or G1 is not None:
        G0 = _snp_fixup(G0)
        G1 = _snp_fixup(G1, iid_source_if_none=G0)
        G0, G1, test_snps, pheno, covar,  = pstutil.intersect_apply([G0, G1, test_snps, pheno, covar])
        G0_standardized = G0.read().standardize()
        G1_standardized = G1.read().standardize()
    else:
        test_snps, pheno, covar,  = pstutil.intersect_apply([test_snps, pheno, covar])
        G0_standardized, G1_standardized = None, None


    frame =  _internal_single(G0_standardized=G0_standardized, test_snps=test_snps, pheno=pheno,
                                covar=covar, G1_standardized=G1_standardized, mixing=mixing, 
                                external_log_delta=log_delta, min_log_delta=min_log_delta, max_log_delta=max_log_delta,
                                cache_file = cache_file)

    frame.sort("PValue", inplace=True)
    frame.index = np.arange(len(frame))


    if output_file_name is not None:
        frame.to_csv(output_file_name, sep="\t", index=False)

    logging.info("PhenotypeName\t{0}".format(pheno['header']))
    if G0 is not None:
        logging.info("SampleSize\t{0}".format(G0.iid_count))
        logging.info("SNPCount\t{0}".format(G0.sid_count))
    logging.info("Runtime\t{0}".format(time.time()-t0))


    return frame

    

#!!might one need to pre-compute log_delta for each chrom?
#!!clusterize????
def single_snp_leave_out_one_chrom(test_snps, pheno,
                 G1=None, mixing=0.0, #!!test mixing and G1
                 covar=None,covar_by_chrom=None,
                 log_delta=None, min_log_delta=-5, max_log_delta=10, output_file_name=None):
    """
    Function performing single SNP GWAS via cross validation over the chromosomes with REML

    :param test_snps: SNPs to test and to construct similarity matrix.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type pheno: a 'pheno dictionary' or a string


    :param G1: SNPs from which to construct a second simalirty matrix, optional. Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G1: a :class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to G1 relative to G0.
            If you give no mixing number, G0 will get all the weight and G1 will be ignored.
    :type mixing: number

    :param covar: covariate information, optional: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type covar: a 'pheno dictionary' or a string

    :param covar_by_chrom: covariate information, optional: A way to give different covariate information for each chromosome.
            It is a dictionary from chromosome number to a 'pheno dictionary' or a string
    :type covar_by_chrom: A dictionary from chromosome number to a 'pheno dictionary' or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param log_delta: A parameter to LMM learning, optional
            If not given will search for best value.
    :type log_delta: number

    :param min_log_delta: (default:-5)
            When searching for log_delta, the lower bounds of the search.
    :type min_log_delta: number

    :param max_log_delta: (default:10)
            When searching for log_delta, the upper bounds of the search.
    :type max_log_delta: number


    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp_leave_out_one_chrom
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp_leave_out_one_chrom(test_snps="../feature_selection/examples/toydata.5chrom", pheno=pheno_fn, log_delta=np.log(4.0))
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_576 1e-07 10000

    """
    t0 = time.time()
    test_snps = _snp_fixup(test_snps)
    G1 = _snp_fixup(G1, iid_source_if_none=test_snps)
    pheno = _pheno_fixup(pheno)
    covar = _pheno_fixup(covar, iid_source_if_none=pheno)
    test_snps, G1, pheno, covar,  = pstutil.intersect_apply([test_snps, G1, pheno, covar])
    G0_standardized = test_snps.read().standardize()
    G1_standardized = G1.read().standardize()

    chrom_set = set(G0_standardized.pos[:,0]) # find the set of all chroms mentioned in G0_standardized, the main training data
    assert len(chrom_set) > 1, "single_leave_out_one_chrom requires more than one chromosome"
    frame_list = []
    for chrom in chrom_set:
        #!!is it OK to read (and standardize) G0_standardized and G1 over and over again, once for each chrom?
        G0_standardized_chrom = G0_standardized[:,G0_standardized.pos[:,0] != chrom].read() # train on snps that don't match this chrom
        test_snps_chrom = G0_standardized[:,G0_standardized.pos[:,0] == chrom].read() # test on snps that do match this chrom
        G1_standardized_chrom = G1_standardized[:,G1_standardized.pos[:,0] != chrom].read() # train on snps that don't match the chrom
        covar_chrom = _create_covar_chrom(covar, covar_by_chrom, chrom)

        frame_chrom = _internal_single(G0_standardized=G0_standardized_chrom, test_snps=test_snps_chrom, pheno=pheno,
                                covar=covar_chrom, G1_standardized=G1_standardized_chrom, mixing=mixing,
                                external_log_delta=log_delta, min_log_delta=min_log_delta, max_log_delta=max_log_delta, cache_file=None)

        frame_list.append(frame_chrom)

    frame = pd.concat(frame_list)
    frame.sort("PValue", inplace=True)
    frame.index = np.arange(len(frame))

    if output_file_name is not None:
        frame.to_csv(output_file_name, sep="\t", index=False)

    logging.info("PhenotypeName\t{0}".format(pheno['header']))
    logging.info("SampleSize\t{0}".format(test_snps.iid_count))
    logging.info("SNPCount\t{0}".format(test_snps.sid_count))
    logging.info("Runtime\t{0}".format(time.time()-t0))

    return frame


def _internal_single(G0_standardized, test_snps, pheno,covar, G1_standardized,
                 mixing, #!!test mixing and G1
                 external_log_delta, min_log_delta, max_log_delta,
                 cache_file):


    covar = np.hstack((covar['vals'],np.ones((test_snps.iid_count, 1))))  #We always add 1's to the end.
    y =  pheno['vals']

    from pysnptools.standardizer import DiagKtoN

    assert 0.0 <= mixing <= 1.0

    if cache_file is not None and os.path.exists(cache_file):
        lmm = fastLMM(X=covar, Y=y, G=None, K=None)
        with np.load(cache_file) as data: #!! similar code in epistasis
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
    else:
        # combine two kernels (normalize kernels to diag(K)=N
        if mixing == 0.0:
            #G0_standardized_val = 1./np.sqrt((G0_standardized.val**2).sum() / float(G0_standardized.val.shape[0])) * G0_standardized.val
            G = DiagKtoN(G0_standardized.val.shape[0]).standardize(G0_standardized.val)
        elif mixing == 1.0:
            #G1_standardized_val = 1./np.sqrt((G1_standardized.val**2).sum() / float(G1_standardized.val.shape[0])) * G1_standardized.val
            G = DiagKtoN(G1_standardized.val.shape[0]).standardize(G1_standardized.val)
        else:
            assert G1_standardized.sid_count > 0, "If a nonzero mixing weight is given, G1 is required"
            logging.info("concat G1, mixing {0}".format(mixing))
        
            G0_standardized_val = DiagKtoN(G0_standardized.val.shape[0]).standardize(G0_standardized.val)
            G1_standardized_val = DiagKtoN(G1_standardized.val.shape[0]).standardize(G1_standardized.val)
         
            G0_standardized_val *= (np.sqrt(1.0-mixing))
            G1_standardized_val *= np.sqrt(mixing) 
            G = np.concatenate((G0_standardized_val, G1_standardized_val),1)
        
        #TODO: make sure low-rank case is handled correctly
        lmm = fastLMM(X=covar, Y=y, G=G, K=None)


    if external_log_delta is None:
        result = lmm.find_log_delta(sid_count=1, min_log_delta=min_log_delta, max_log_delta=max_log_delta)
        external_log_delta = result['log_delta']
    internal_delta = np.exp(external_log_delta)
    logging.info("internal_delta={0}".format(internal_delta))
    logging.info("external_log_delta={0}".format(external_log_delta))

    snps_read = test_snps.read().standardize()
    res = lmm.nLLeval(delta=internal_delta, dof=None, scale=1.0, penalty=0.0, snps=snps_read.val)

    if cache_file is not None and not os.path.exists(cache_file):
        pstutil.create_directory_if_necessary(cache_file)
        np.savez(cache_file, lmm.U,lmm.S) #using np.savez instead of pickle because it seems to be faster to read and write


    beta = res['beta']
        
    chi2stats = beta*beta/res['variance_beta']
    #p_values = stats.chi2.sf(chi2stats,1)[:,0]
    if G0_standardized is not None:
        assert G.shape[0] == lmm.U.shape[0]
    p_values = stats.f.sf(chi2stats,1,lmm.U.shape[0]-3)[:,0]#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+SNP)


    items = [
                ('SNP', snps_read.sid),
                ('Chr', snps_read.pos[:,0]), 
                ('GenDist', snps_read.pos[:,1]),
                ('ChrPos', snps_read.pos[:,2]), 
                ('PValue', p_values),
                ('SnpWeight', beta[:,0]),
                ('SnpWeightSE', np.sqrt(res['variance_beta'][:,0])),
                ('NullLogDelta', np.zeros((snps_read.sid_count)) + external_log_delta)
            ]
    frame = pd.DataFrame.from_items(items)

    return frame

def _create_covar_chrom(covar, covar_by_chrom, chrom):
    if covar_by_chrom is not None:
        covar_by_chrom_chrom = covar_by_chrom[chrom]
        covar_by_chrom_chrom = _pheno_fixup(covar_by_chrom_chrom, iid_source_if_none=covar)
        covar_after,  covar_by_chrom_chrom = pstutil.intersect_apply([covar,  covar_by_chrom_chrom])
        assert np.all(covar_after['iid'] == covar['iid']), "covar_by_chrom must contain all iids found in the intersection of the other datasets"

        ret = {
        'header':covar['header']+covar_by_chrom_chrom['header'],
        'vals': np.hstack([covar['vals'],covar_by_chrom_chrom['vals']]),
        'iid':covar['iid']
        }
        return ret
    else:
        return covar


def _snp_fixup(snp_input, iid_source_if_none=None):
    if isinstance(snp_input, str):
        return Bed(snp_input)
    elif snp_input is None:
        return iid_source_if_none[:,0:0] #return snpreader with no snps
    else:
        return snp_input

def _pheno_fixup(pheno_input, iid_source_if_none=None):
    if isinstance(pheno_input, str):
        return pstpheno.loadPhen(pheno_input) #!!what about missing=-9?

    if pheno_input is None:
        ret = {
        'header':[],
        'vals': np.empty((iid_source_if_none['vals'].shape[0], 0)),
        'iid':iid_source_if_none['iid']
        }
        return ret

    if len(pheno_input['vals'].shape) == 1:
        pheno_input['vals'] = np.reshape(pheno_input['vals'],(-1,1))

    return pheno_input


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    import doctest
    doctest.testmod()

    ##import logging
    #from pysnptools.snpreader import Bed
    #logging.basicConfig(level=logging.INFO)
    #snpreader = Bed("../feature_selection/examples/toydata")
    #pheno_fn = "../feature_selection/examples/toydata.phe"

    ##null_9930 = snpreader[:,snpreader.sid_to_index(['null_9930'])].read()
    ##from pysnptools.snpreader import Dat
    ##Dat.write(null_9930,r"c:\deldir\9930.dat")




    #log_delta = np.log(4.0)
    #frame = single_snp(test_snps=snpreader[:,5000:10000], pheno=pheno_fn, G0=snpreader[:,0:5000], log_delta=log_delta,output_file_name=r"c:\deldir\toydata.out.txt")
    #print frame.iloc[0].SNP,round(frame.iloc[0].PValue,7),len(frame)
    ##null_7487 2.6e-06 5000

    #snpreader2 = "../feature_selection/examples/toydata.5chrom"
    #frame2 = single_snp_leave_out_one_chrom(test_snps=snpreader2, pheno=pheno_fn, log_delta=log_delta,output_file_name=r"c:\deldir\toydata.2.out.txt")
    #print frame2.iloc[0].SNP,round(frame2.iloc[0].PValue,7),len(frame2)
    ##null_576 1e-07 10000


    print "done"

