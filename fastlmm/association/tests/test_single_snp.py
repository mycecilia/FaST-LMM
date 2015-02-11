import numpy as np
import logging
import unittest
import os.path
import doctest
import pandas as pd

from fastlmm.association import single_snp
from fastlmm.association import single_snp_leave_out_one_chrom
import pysnptools.util.pheno as pstpheno
from fastlmm.feature_selection.test import TestFeatureSelection
from fastlmm.util.runner import Local, HPC, LocalMultiProc

class TestSingleSnp(unittest.TestCase):

    #!!created a Expect Durbin, too

    @classmethod
    def setUpClass(self):
        from fastlmm.util.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'tests/datasets/all_chr.maf0.001.N300')
        self.phen_fn = os.path.join(self.pythonpath, 'tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
        self.cov_fn = os.path.join(self.pythonpath,  'tests/datasets/all_chr.maf0.001.covariates.N300.txt')

    tempout_dir = "tempout/single_snp"

    def test_match_cpp(self):
        '''
        match
            FaSTLMM.207\Data\DemoData>..\.cd.\bin\windows\cpp_mkl\fastlmmc -bfile snps -extract topsnps.txt -bfileSim snps -extractSim ASout.snps.txt -pheno pheno.txt -covar covariate.txt -out topsnps.singlesnp.txt -logDelta 0 -verbose 100

        '''
        logging.info("TestSingleSnp test_match_cpp")
        from pysnptools.snpreader import Bed
        snps = Bed(os.path.join(self.pythonpath, "tests/datasets/selecttest/snps"))
        pheno = os.path.join(self.pythonpath, "tests/datasets/selecttest/pheno.txt")
        covar = os.path.join(self.pythonpath, "tests/datasets/selecttest/covariate.txt")
        sim_sid = ["snp26250_m0_.19m1_.19","snp82500_m0_.28m1_.28","snp63751_m0_.23m1_.23","snp48753_m0_.4m1_.4","snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp11253_m0_.2m1_.2","snp86250_m0_.33m1_.33","snp3753_m0_.23m1_.23","snp75003_m0_.32m1_.32","snp30002_m0_.25m1_.25","snp26252_m0_.19m1_.19","snp67501_m0_.15m1_.15","snp63750_m0_.28m1_.28","snp30001_m0_.28m1_.28","snp52502_m0_.35m1_.35","snp33752_m0_.31m1_.31","snp37503_m0_.37m1_.37","snp15002_m0_.11m1_.11","snp3751_m0_.34m1_.34","snp7502_m0_.18m1_.18","snp52503_m0_.3m1_.3","snp30000_m0_.39m1_.39","isnp4457_m0_.11m1_.11","isnp23145_m0_.2m1_.2","snp60001_m0_.39m1_.39","snp33753_m0_.16m1_.16","isnp60813_m0_.2m1_.2","snp82502_m0_.34m1_.34","snp11252_m0_.13m1_.13"]
        sim_idx = snps.sid_to_index(sim_sid)
        test_sid = ["snp26250_m0_.19m1_.19","snp63751_m0_.23m1_.23","snp82500_m0_.28m1_.28","snp48753_m0_.4m1_.4","snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp86250_m0_.33m1_.33","snp15002_m0_.11m1_.11","snp33752_m0_.31m1_.31","snp26252_m0_.19m1_.19","snp30001_m0_.28m1_.28","snp11253_m0_.2m1_.2","snp67501_m0_.15m1_.15","snp3753_m0_.23m1_.23","snp52502_m0_.35m1_.35","snp30000_m0_.39m1_.39","snp30002_m0_.25m1_.25"]
        test_idx = snps.sid_to_index(test_sid)

        frame = single_snp(test_snps=snps[:,test_idx], pheno=pheno, G0=snps[:,sim_idx], covar=covar,log_delta=0)

        referenceOutfile = TestFeatureSelection.reference_file("single_snp/topsnps.single.txt")

        reference = pd.read_table(referenceOutfile,sep="\t") # We've manually remove all comments and blank lines from this file
        assert len(frame) == len(reference)



        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            reldiff = abs(row.Pvalue - pvalue)/row.Pvalue
            assert reldiff < .035, "'{0}' pvalue_list differ too much {4} -- {2} vs {3}".format(sid,None,row.PValue,pvalue,reldiff)
 
    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one(self):
        logging.info("TestSingleSnp test_one")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,
                                  G0=test_snps, covar=covar, 
                                  output_file_name=output_file
                                  )

        #Check the output file
        self.compare_files(frame,"one")

    def test_preload_files(self):
        logging.info("TestSingleSnp test_preload_files")
        from pysnptools.snpreader import Bed
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps)

        output_file_name = self.file_name("preload_files")

        frame = single_snp(test_snps=bed[:,:10], pheno=pheno, G0=test_snps, 
                                  covar=covar, output_file_name=output_file_name
                                  )
        self.compare_files(frame,"one")
        
    def test_SNC(self):
        logging.info("TestSNC")
        from pysnptools.snpreader import Bed
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps)
        snc = bed.read()
        snc.val[:,2] = [0] * snc.iid_count # make SNP #2 have constant values (aka a SNC)

        output_file_name = self.file_name("snc")

        frame = single_snp(test_snps=snc[:,:10], pheno=pheno, G0=snc, 
                                  covar=covar, output_file_name=output_file_name
                                  )
        self.compare_files(frame,"snc")

    def test_G0_has_reader(self):
        logging.info("TestSingleSnp test_G0_has_reader")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G0_has_reader")

        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, 
                                  covar=covar, 
                                  output_file_name=output_file_name
                                  )
        self.compare_files(frame,"one")

    def test_no_cov(self):
        logging.info("TestSingleSnp test_no_cov")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, 
                                          output_file_name=output_file_name
                                          )

        self.compare_files(frame,"no_cov")

    def test_no_cov_b(self):
        logging.info("TestSingleSnp test_no_cov_b")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov_b")
        covar = pstpheno.loadPhen(self.cov_fn)
        covar['vals'] = np.delete(covar['vals'], np.s_[:],1) #Remove all the columns

        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, 
                                  covar=covar,
                                  output_file_name=output_file_name
                                  )

        self.compare_files(frame,"no_cov")

    def test_G1(self):
        logging.info("TestSingleSnp test_G1")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], 
                                      covar=covar, G1=test_snps[:,100:200],
                                      mixing=.5,
                                      output_file_name=output_file_name
                                      )

        self.compare_files(frame,"G1")


    def test_file_cache(self):
        logging.info("TestSingleSnp test_file_cache")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        cache_file = self.file_name("cache_file")+".npz"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], 
                                      covar=covar, G1=test_snps[:,100:200],
                                      mixing=.5,
                                      output_file_name=output_file_name,
                                      cache_file = cache_file
                                      )
        self.compare_files(frame,"G1")

        frame2 = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=None, 
                                      covar=covar, G1=None,
                                      mixing=.5,
                                      output_file_name=output_file_name,
                                      cache_file = cache_file
                                      )
        self.compare_files(frame2,"G1")


    def test_G1_mixing(self):
        logging.info("TestSingleSnp test_G1_mixing")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1_mixing")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, 
                                      covar=covar, 
                                      G1=test_snps[:,100:200],
                                      mixing=0,
                                      output_file_name=output_file_name
                                      )

        self.compare_files(frame,"one")

    def test_unknown_sid(self):
        logging.info("TestSingleSnp test_unknown_sid")

        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        try:
            frame = single_snp(test_snps=test_snps,G0=test_snps,pheno=pheno,covar=covar,sid_list=['1_4','bogus sid','1_9'])
            failed = False
        except:
            failed = True

        assert(failed)

    def test_cid_intersect(self):
        logging.info("TestSingleSnp test_cid_intersect")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
        pheno['iid'] = np.vstack([pheno['iid'][::-1],[['Bogus','Bogus']]])
        pheno['vals'] = np.hstack([pheno['vals'][::-1],[-34343]])

        
        covar = self.cov_fn
        output_file_name = self.file_name("cid_intersect")
        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, 
                                  covar=covar, 
                                  output_file_name=output_file_name
                                  )

        self.compare_files(frame,"one")

    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

        #sid_list,pvalue_list = frame['SNP'].as_matrix(),frame['Pvalue'].as_matrix()

        #sid_to_pvalue = {}
        #for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference=pd.read_csv(reffile,delimiter='\s',comment=None)
        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            assert abs(row.PValue - pvalue) < 1e-5, "pair {0} differs too much from file '{1}'".format(sid,reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testfile("../single_snp.py")
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

class TestSingleSnpLeaveOutOneChrom(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from fastlmm.util.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.bedbase = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.5chrom')
        self.phen_fn = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.phe')
        self.cov_fn = os.path.join(self.pythonpath,  'fastlmm/feature_selection/examples/toydata.cov')

    tempout_dir = "tempout/single_snp"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_one_looc")
        from pysnptools.snpreader import Bed
        test_snps = Bed(self.bedbase)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one_looc")
        frame = single_snp_leave_out_one_chrom(test_snps, pheno,
                                  covar=covar, 
                                  output_file_name=output_file
                                  )

        self.compare_files(frame,"one_looc")

    def test_covar_by_chrom(self):
            logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom")
            from pysnptools.snpreader import Bed
            test_snps = Bed(self.bedbase)
            pheno = self.phen_fn
            covar = self.cov_fn
            covar_by_chrom = {}
            for chrom in xrange(1,6):
                covar_by_chrom[chrom] = covar
            output_file = self.file_name("covar_by_chrom")
            frame = single_snp_leave_out_one_chrom(test_snps, pheno,
                                      covar=covar,
                                      covar_by_chrom=covar_by_chrom,
                                      output_file_name=output_file
                                      )

            self.compare_files(frame,"covar_by_chrom")

    def compare_files(self,frame,ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

        #sid_list,pvalue_list = frame['SNP'].as_matrix(),frame['Pvalue'].as_matrix()

        #sid_to_pvalue = {}
        #for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference=pd.read_csv(reffile,delimiter='\s',comment=None)
        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
            assert abs(row.PValue - pvalue) < 1e-5, "snp {0} differs too much from file '{1}'".format(sid,reffile)


        


def getTestSuite():
    

    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnp)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpLeaveOutOneChrom)
    return unittest.TestSuite([suite1,suite2])



if __name__ == '__main__':
    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp import TestSingleSnp
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run 
        r = unittest.TextTestRunner(failfast=True)
        r.run(suites)
    else: #Cluster test run
        from fastlmm.util.distributabletest import DistributableTest


        runner = HPC(10, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond',
                     remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
                     update_remote_python_parent=True,
                     priority="AboveNormal",mkl_num_threads=1)
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done with testing")
