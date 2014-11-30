import numpy as np
import scipy as sp
import logging
import doctest

from pysnptools.pysnptools.snpreader.bed import *
from pysnptools.pysnptools.snpreader.hdf5 import Hdf5
from pysnptools.pysnptools.snpreader.dat import Dat
from pysnptools.pysnptools.snpreader.ped import Ped
from pysnptools.pysnptools.standardizer.unit import Unit
from pysnptools.pysnptools.standardizer.beta import Beta


import unittest
import os.path
import time


from fastlmm.util.runner import Local, Hadoop, Hadoop2, HPC, LocalMultiProc, LocalInParts
from fastlmm.util.distributabletest import DistributableTest

   
class TestLoader(unittest.TestCase):     

    def xtest_aaa_hdf5_speed(self): #!!to0 slow to use all the time

        #currentFolder + "/examples/toydata"
        #currentFolder + "/examples/delme.hdf5"
        bedFileName = r"d:\data\carlk\cachebio\genetics\synthetic\wtccclikeH\snps" #!! local paths
        hdf5Pattern = r"d:\data\carlk\cachebio\genetics\synthetic\wtccclikeH\del.{0}.hdf5"#!!
        tt0 = time.time()
        snpreader_bed = Bed(bedFileName)

        S0 = snpreader_bed.sid_count
        snp_index_list0 = range(min(S0,15000)) 

        hdf5FileName = hdf5Pattern.format(len(snp_index_list0))

        #!!        snpDataBed = snpreader_bed.read(SnpIndexList(snp_index_list0))
        tt1 = time.time()
        logging.info("Read bed %.2f seconds" % (tt1 - tt0))
        #!!        Hdf5.write(snpDataBed, hdf5FileName)
        tt2 = time.time()
        logging.info("write Hdf5 bed %.2f seconds" % (tt2 - tt1))

        snpreader_hdf5 = Hdf5(hdf5FileName)
        assert(snpreader_hdf5.iid_count == snpreader_bed.iid_count)
        S = snpreader_hdf5.sid_count
        N_original = snpreader_hdf5.iid_count
        iid_index_list = sorted(range(N_original - 1,0,-2))

        snp_index_list = sorted(range(S - 1,0,-2))#!!
        #snp_index_list = range(S/2)

        snpreader_hdf5 = snpreader_hdf5[iid_index_list,:]
        snpDataHdf5 = snpreader_hdf5[:,snp_index_list].read()
        tt3 = time.time()
        logging.info("read Hdf5 with reversed indexes bed %.2f seconds" % (tt3 - tt2))

        snpDataHdf5C = snpreader_hdf5[:,snp_index_list].read(order = "C")
        tt4 = time.time()
        logging.info("read Hdf5 C with reversed indexes bed %.2f seconds" % (tt4 - tt3))

        print "the end"
    

    @classmethod
    def setUpClass(self):
        self.currentFolder = os.path.dirname(os.path.realpath(__file__))
        #TODO: get data set with NANs!
        snpreader = Bed(self.currentFolder + "/examples/toydata")
        self.pheno_fn = self.currentFolder + "/examples/toydata.phe"
        self.snps = snpreader.read(order='F',force_python_only=True).val

    def test_c_reader_bed(self):
        snpreader = Bed(self.currentFolder + "/examples/toydata")
        self.c_reader(snpreader)

    def test_c_reader_hdf5(self):
        snpreader = Hdf5(self.currentFolder + "/examples/toydata.snpmajor.hdf5")
        self.c_reader(snpreader)

    def test_c_reader_dat(self):
        snpreader = Dat(self.currentFolder + "/examples/toydata.dat")
        self.c_reader(snpreader)

    def c_reader(self,snpreader):
        """
        make sure c-reader yields same result
        """
        snp_c = snpreader.read(order='F',force_python_only=False).val
        
        self.assertEqual(np.float64, snp_c.dtype)
        self.assertTrue(np.allclose(self.snps, snp_c, rtol=1e-05, atol=1e-05))

    def test_standardize_bed(self):
        snpreader = Bed(self.currentFolder + "/examples/toydata")
        self.standardize(snpreader)

    def test_standardize_hdf5(self):
        snpreader = Hdf5(self.currentFolder + "/examples/toydata.iidmajor.hdf5")
        self.standardize(snpreader)

    def test_standardize_dat(self):
        snpreader = Dat(self.currentFolder + "/examples/toydata.dat")
        self.standardize(snpreader)

    def test_standardize_ped(self):
        snpreader = Ped(self.currentFolder + "/examples/toydata")
        self.standardize(snpreader)


    def standardize(self,snpreader):
        """
        make sure blocked standardize yields same result as regular standardize
        """

        for dtype in [sp.float64,sp.float32]:

            snps = snpreader.read(order='F',force_python_only=True,dtype=dtype).val
            self.assertEqual(dtype, snps.dtype)

            snp_s1 = Unit().standardize(snps.copy(), force_python_only=True)
            snp_s2 = Unit().standardize(snps.copy(), blocksize=100, force_python_only=True)
            snps_F = np.array(snps, dtype=dtype, order="F")
            snp_s3 = Unit().standardize(snps_F)
            snps_C = np.array(snps, dtype=dtype, order="C")
            snp_s4 = Unit().standardize(snps_C)

            snp_beta1 = Beta(1, 25).standardize(snps.copy(), force_python_only=True)
            snps_F = np.array(snps, dtype=dtype, order="F")
            snp_beta2 = Beta(1, 25).standardize(snps_F)
            snps_C = np.array(snps, dtype=dtype, order="C")
            snp_beta3 = Beta(1, 25).standardize(snps_C)

            self.assertEqual(snp_s1.shape[0], snp_s2.shape[0])
            self.assertEqual(snp_s1.shape[1], snp_s2.shape[1])

            self.assertEqual(snp_s1.shape[0], snp_s3.shape[0])
            self.assertEqual(snp_s1.shape[1], snp_s3.shape[1])
        
            self.assertEqual(snp_s1.shape[0], snp_s4.shape[0])
            self.assertEqual(snp_s1.shape[1], snp_s4.shape[1])

            self.assertTrue(np.allclose(snp_s1, snp_s2, rtol=1e-05, atol=1e-05))
            self.assertTrue(np.allclose(snp_s1, snp_s3, rtol=1e-05, atol=1e-05))
            self.assertTrue(np.allclose(snp_s1, snp_s4, rtol=1e-05, atol=1e-05))

            self.assertEqual(snp_beta1.shape[0], snp_beta2.shape[0])
            self.assertEqual(snp_beta1.shape[1], snp_beta2.shape[1])
            self.assertEqual(snp_beta1.shape[0], snp_beta3.shape[0])
            self.assertEqual(snp_beta1.shape[1], snp_beta3.shape[1])
        
            self.assertTrue(np.allclose(snp_beta1, snp_beta2, rtol=1e-05, atol=1e-05))
            self.assertTrue(np.allclose(snp_beta1, snp_beta3, rtol=1e-05, atol=1e-05))

    def test_load_and_standardize_bed(self):
        snpreader2 = Bed(self.currentFolder + "/examples/toydata")
        self.load_and_standardize(snpreader2, snpreader2)

    def test_subset_view(self):
        snpreader2 = Bed(self.currentFolder + "/examples/toydata")[:,:]
        result = snpreader2.read(view_ok=True)
        self.assertFalse(snpreader2 is result)
        result2 = result[:,:].read()
        self.assertFalse(SP.may_share_memory(result2.val,result.val))
        result3 = result[:,:].read(view_ok=True)
        self.assertTrue(SP.may_share_memory(result3.val,result.val))
        result4 = result3.read()
        self.assertFalse(SP.may_share_memory(result4.val,result3.val))
        result5 = result4.read(view_ok=True)
        self.assertTrue(SP.may_share_memory(result4.val,result5.val))


    def test_load_and_standardize_hdf5(self):
        snpreader2 = Hdf5(self.currentFolder + "/examples/toydata.snpmajor.hdf5")
        snpreader3 = Hdf5(self.currentFolder + "/examples/toydata.iidmajor.hdf5")
        self.load_and_standardize(snpreader2, snpreader3)
        snpreaderref = Bed(self.currentFolder + "/examples/toydata")
        self.load_and_standardize(snpreader2, snpreaderref)



    def test_load_and_standardize_dat(self):
        snpreader2 = Dat(self.currentFolder + "/examples/toydata.dat")
        self.load_and_standardize(snpreader2, snpreader2)
        #snpreaderref = Bed(self.currentFolder + "/examples/toydata")
        #self.load_and_standardize(snpreader2, snpreaderref)

    def test_load_and_standardize_ped(self):

        #!!Ped columns can be ambiguous
        ###Creating Ped data ...
        #currentFolder = os.path.dirname(os.path.realpath(__file__))
        #snpData = Bed(currentFolder + "/examples/toydata").read()
        ##Ped.write(snpData, currentFolder + "/examples/toydata.ped")
        #fromPed = Ped(currentFolder + "/examples/toydata").read()
        #self.assertTrue(np.allclose(snpData.val, fromPed.val, rtol=1e-05, atol=1e-05))


        snpreader2 = Ped(self.currentFolder + "/examples/toydata")
        self.load_and_standardize(snpreader2, snpreader2)
        #snpreaderref = Bed(self.currentFolder + "/examples/toydata")
        #self.load_and_standardize(snpreader2, snpreaderref)

    def load_and_standardize(self, snpreader2, snpreader3):
        """
        test c-version of load and standardize
        """

        S = snpreader2.sid_count
        N_original = snpreader2.iid_count

        iid_index_list = range(N_original - 1,0,-2)
        snpreader3 = snpreader3[iid_index_list,:]

        for dtype in [sp.float64,sp.float32]:

            G2 = snpreader2.read(order='F',force_python_only=True).val
            G2 = Unit().standardize(G2, blocksize=10000, force_python_only=True)

            SNPs_floatF = snpreader2.read(order="F", dtype=dtype, force_python_only=False).val
            GF = Unit().standardize(SNPs_floatF)

            SNPs_floatC = snpreader2.read(order="C", dtype=dtype, force_python_only=False).val
            GC = Unit().standardize(SNPs_floatC)

            self.assertTrue(np.allclose(GF, G2, rtol=1e-05, atol=1e-05))
            self.assertTrue(np.allclose(GF, GC, rtol=1e-05, atol=1e-05))

            #testing selecting a subset of snps and iids
            snp_index_list = range(S - 1,0,-2)

            G2x = snpreader2.read(order='F',force_python_only=True).val
            G2x = G2x[iid_index_list,:][:,snp_index_list]
            G2x = Unit().standardize(G2x, blocksize=10000, force_python_only=True)


            SNPs_floatFx = snpreader3[:,snp_index_list].read(order="F", dtype=dtype, force_python_only=False).val
            GFx = Unit().standardize(SNPs_floatFx)
            self.assertTrue(np.allclose(GFx, G2x, rtol=1e-05, atol=1e-05))

            SNPs_floatCx = snpreader3[:,snp_index_list].read(order="C", dtype=dtype, force_python_only=False).val
            GCx = Unit().standardize(SNPs_floatCx)
            self.assertTrue(np.allclose(GFx, G2x, rtol=1e-05, atol=1e-05))

class NaNCNCTestCases(unittest.TestCase):
    def __init__(self, iid_index_list, snp_index_list, standardizer, snpreader, dtype, order, force_python_only, reference_snps, reference_dtype):
        self.iid_index_list = iid_index_list
        self.snp_index_list = snp_index_list
        self.standardizer = standardizer
        self.snpreader = snpreader
        self.dtype = dtype
        self.order = order
        self.force_python_only = force_python_only
        self.reference_snps = reference_snps
        self.reference_dtype = reference_dtype

    _testMethodName = "runTest"
    _testMethodDoc = None

    @staticmethod
    def factory_iterator():
        snp_reader_factory_bed = lambda : Bed("examples/toydata")
        snp_reader_factory_snpmajor_hdf5 = lambda : Hdf5("examples/toydata.snpmajor.hdf5")
        snp_reader_factory_iidmajor_hdf5 = lambda : Hdf5("examples/toydata.iidmajor.hdf5",blocksize=6000)
        snp_reader_factory_dat = lambda : Dat("examples/toydata.dat")

        previous_wd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        snpreader0 = snp_reader_factory_bed()
        S_original = snpreader0.sid_count
        N_original = snpreader0.iid_count

        snps_to_read_count = min(S_original, 100)

        for iid_index_list in [range(N_original), range(N_original/2), range(N_original - 1,0,-2)]:
            for snp_index_list in [range(snps_to_read_count), range(snps_to_read_count/2), range(snps_to_read_count - 1,0,-2)]:
                for standardizer in [Unit(),Beta(1,25)]:
                    reference_snps, reference_dtype = NaNCNCTestCases(iid_index_list, snp_index_list, standardizer, snp_reader_factory_bed(), sp.float64, "C", "False", None, None).read_and_standardize()
                    for snpreader_factory in [snp_reader_factory_bed, 
                                             snp_reader_factory_snpmajor_hdf5, snp_reader_factory_iidmajor_hdf5,
                                              snp_reader_factory_dat]:
                        for dtype in [sp.float64,sp.float32]:
                            for order in ["C", "F"]:
                                for force_python_only in [False, True]:
                                    snpreader = snpreader_factory()
                                    test_case = NaNCNCTestCases(iid_index_list, snp_index_list, standardizer, snpreader, dtype, order, force_python_only, reference_snps, reference_dtype)
                                    yield test_case
        os.chdir(previous_wd)

    def __str__(self):
        iid_index_list = self.iid_index_list
        snp_index_list = self.snp_index_list
        standardizer = self.standardizer
        snpreader = self.snpreader
        dtype = self.dtype
        order = self.order
        force_python_only = self.force_python_only
        return "{0}(iid_index_list=[{1}], snp_index_list=[{2}], standardizer={3}, snpreader={4}, dtype={5}, order='{6}', force_python_only=={7})".format(
            self.__class__.__name__,
            ",".join([str(i) for i in iid_index_list]) if len(iid_index_list) < 10 else ",".join([str(i) for i in iid_index_list[0:10]])+",...",
            ",".join([str(i) for i in snp_index_list]) if len(snp_index_list) < 10 else ",".join([str(i) for i in snp_index_list[0:10]])+",...",
            standardizer, snpreader, dtype, order, force_python_only)

    def read_and_standardize(self):
        previous_wd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        iid_index_list = self.iid_index_list
        snp_index_list = self.snp_index_list
        standardizer = self.standardizer
        snpreader = self.snpreader
        dtype = self.dtype
        order = self.order
        force_python_only = self.force_python_only
        
        snps = snpreader[iid_index_list,snp_index_list].read(order=order, dtype=dtype, force_python_only=force_python_only).val
        snps[0,0] = np.nan # add a NaN
        snps[:,1] = 2 # make a SNC
        snps = standardizer.standardize(snps,force_python_only=force_python_only)
        os.chdir(previous_wd)
        return snps, dtype

    def doCleanups(self):
        pass
        #return super(NaNCNCTestCases, self).doCleanups()

    def runTest(self, result = None):
        snps, dtype = self.read_and_standardize()
        self.assertTrue(snps[0,0] == 0)
        self.assertTrue(np.all(snps[:,1] == 0))
        if self.reference_snps is not None:
            self.assertTrue(np.allclose(self.reference_snps, snps, rtol=1e-04 if dtype == sp.float32 or self.reference_dtype == sp.float32 else 1e-12))

# We do it this way instead of using doctest.DocTestSuite because doctest.DocTestSuite requires modules to be pickled, which python doesn't allow.
# We need tests to be pickleable so that they can be run on a cluster.
class TestDocStrings(unittest.TestCase):
    def test_snpreader(self):
        import pysnptools.pysnptools.snpreader.snpreader
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(pysnptools.pysnptools.snpreader.snpreader)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    def test_bed(self):
        import pysnptools.pysnptools.snpreader.bed
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(pysnptools.pysnptools.snpreader.bed)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    def test_snpdata(self):
        import pysnptools.pysnptools.snpreader.snpdata
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(pysnptools.pysnptools.snpreader.snpdata)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


    def test_util(self):
        import pysnptools.pysnptools.util.util
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/util")
        result = doctest.testmod(pysnptools.pysnptools.util.util)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoader))
    test_suite.addTests(NaNCNCTestCases.factory_iterator())
    from pysnptools.pysnptools.util.intrangeset import TestLoader as IntRangeSetTestLoader
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(IntRangeSetTestLoader))

    return test_suite

if __name__ == '__main__':

    suites = getTestSuite()
    if True:
        # TestFeatureSelection().test_aaa_hdf5_speed()
        r = unittest.TextTestRunner(failfast=False)
        r.run(suites)
    else: #Cluster test run
        task_count = 500
        runner = HPC(task_count, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage6\Redmond',
                     remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage6\REDMOND\carlk\Source\carlk\july_7_14\pythonpath",
                     update_remote_python_parent=True,
                     min=150,
                     priority="AboveNormal",mkl_num_threads=1)
        #runner = Local()
        runner = LocalMultiProc(taskcount=4,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop2(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


