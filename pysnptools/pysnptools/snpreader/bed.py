import numpy as SP
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from snpreader import SnpReader
from snpdata import SnpData
import math



WRAPPED_PLINK_PARSER_PRESENT = None

def _decide_once_on_plink_reader():
    #This is now done in a method, instead of at the top of the file, so that messages can be re-directed to the appropriate stream.
    #(Usually messages go to stdout, but when the code is run on Hadoop, they are sent to stderr)

    global WRAPPED_PLINK_PARSER_PRESENT
    if WRAPPED_PLINK_PARSER_PRESENT == None:
        # attempt to import wrapped plink parser
        try:
            import pysnptools.pysnptools.snpreader.wrap_plink_parser
            WRAPPED_PLINK_PARSER_PRESENT = True #!! does the standardizer work without c++
            logging.info("using c-based plink parser")
        except Exception, detail:
            logging.warn(detail)
            WRAPPED_PLINK_PARSER_PRESENT = False


#!!LATER fix bug in Hadoop whereas it won't use data two levels down

class Bed(SnpReader):
    '''
    This is a class that does random-access reads of a Bed/Bim/Fam files from disk.

    See :class:`.SnpReader` for details and examples.

    Constructor:
        basefilename    : string of the basename of [basename].bed, [basename].bim,
                            and [basename].fam
    '''
    _ran_once = False
    _filepointer = None

    def __init__(self, basefilename):
        self.basefilename = basefilename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.basefilename)

    @property
    def iid(self):
        """list of iids
        """
        self._run_once()
        return self._original_iid

    @property
    def sid(self):
        """list of sids
        """
        self._run_once()
        return self._sid

    @property
    def pos(self):
        """list of position information
        """
        self._run_once()
        return self._pos

    def _run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        famfile = self.basefilename+ '.fam'
        bimfile = self.basefilename+'.bim'

        logging.info("Loading fam file {0}".format(famfile))
        self._original_iid = SP.loadtxt(famfile,delimiter = ' ',dtype = 'str',usecols=(0,1),comments=None)
        logging.info("Loading bim file {0}".format(bimfile))

        bimfields = pd.read_csv(bimfile,delimiter = '\t',usecols = (0,1,2,3),header=None,index_col=False)
        self._sid = SP.array(bimfields[1].tolist(),dtype='str')
        self._pos = bimfields.as_matrix([0,2,3])

        bedfile = self.basefilename+ '.bed'
        self._filepointer = open(bedfile, "rb")
        mode = self._filepointer.read(2)
        if mode != 'l\x1b': raise Exception('No valid binary BED file')
        mode = self._filepointer.read(1) #\x01 = SNP major \x00 = individual major
        if mode != '\x01': raise Exception('only SNP-major is implemented')
        logging.info("bed file is open {0}".format(bedfile))

    def __del__(self):
        if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because only uses original inputs
        copier.input(self.basefilename + ".bed")
        copier.input(self.basefilename + ".bim")
        copier.input(self.basefilename + ".fam")


    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self._run_once()
        assert not hasattr(self, 'ind_used'), "A SnpReader should not have a 'ind_used' attribute"


        if order is None:
            order = "F"
        if dtype is None:
            dtype = SP.float64
        if force_python_only is None:
            force_python_only = False


        _decide_once_on_plink_reader()
        global WRAPPED_PLINK_PARSER_PRESENT

        #!! this could be re-factored to not use so many names
        iid_count_in = self.iid_count
        sid_count_in = self.sid_count

        if iid_index_or_none is not None:
            iid_count_out = len(iid_index_or_none)
            iid_index_out = iid_index_or_none
        else:
            iid_count_out = iid_count_in
            iid_index_out = range(iid_count_in)

        if sid_index_or_none is not None:
            sid_count_out = len(sid_index_or_none)
            sid_index_out = sid_index_or_none
        else:
            sid_count_out = sid_count_in
            sid_index_out = range(sid_count_in)

        if WRAPPED_PLINK_PARSER_PRESENT and not force_python_only:
            import pysnptools.pysnptools.snpreader.wrap_plink_parser as wrap_plink_parser
            val = SP.zeros((iid_count_out, sid_count_out), order=order, dtype=dtype)
            bed_fn = self.basefilename + ".bed"

            if dtype == SP.float64:
                if order=="F":
                    wrap_plink_parser.readPlinkBedFiledoubleFAAA(bed_fn, iid_count_in, sid_count_in, iid_index_out, sid_index_out, val)
                elif order=="C":
                    wrap_plink_parser.readPlinkBedFiledoubleCAAA(bed_fn, iid_count_in, sid_count_in, iid_index_out, sid_index_out, val)
                else:
                    raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
            elif dtype == SP.float32:
                if order=="F":
                    wrap_plink_parser.readPlinkBedFilefloatFAAA(bed_fn, iid_count_in, sid_count_in, iid_index_out, sid_index_out, val)
                elif order=="C":
                    wrap_plink_parser.readPlinkBedFilefloatCAAA(bed_fn, iid_count_in, sid_count_in, iid_index_out, sid_index_out, val)
                else:
                    raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
            else:
                raise Exception("dtype '{0}' not known, only float64 and float32".format(dtype))
            
        else:
            # An earlier version of this code had a way to read consecutive SNPs of code in one read. May want
            # to add that ability back to the code. 
            # Also, note that reading with python will often result in non-contigious memory, so the python standardizers will automatically be used, too.       
            logging.warn("using pure python plink parser (might be much slower!!)")
            val = SP.zeros(((int(SP.ceil(0.25*iid_count_in))*4),sid_count_out),order=order, dtype=dtype) #allocate it a little big
            for SNPsIndex, bimIndex in enumerate(sid_index_out):

                startbit = int(SP.ceil(0.25*iid_count_in)*bimIndex+3)
                self._filepointer.seek(startbit)
                nbyte = int(SP.ceil(0.25*iid_count_in))
                bytes = SP.array(bytearray(self._filepointer.read(nbyte))).reshape((int(SP.ceil(0.25*iid_count_in)),1),order='F')

                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=64]=SP.nan
                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=128]=1
                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=192]=2
                bytes=SP.mod(bytes,64)
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=16]=SP.nan
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=32]=1
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=48]=2
                bytes=SP.mod(bytes,16)
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=4]=SP.nan
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=8]=1
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=12]=2
                bytes=SP.mod(bytes,4)
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=1]=SP.nan
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=2]=1
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=3]=2
            val = val[iid_index_out,:] #reorder or trim any extra allocation


            #!!LATER this can fail because the trim statement above messes up the order
            #assert(SnpReader._array_properties_are_ok(val, order, dtype)) #!!

        return val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ##What if you want to some level cached?  e.g.
    #from standardizer.BySqrtSidCount import BySqrtSidCount
    #l0 = Bed('../../tests/datasets/all_chr.maf0.001.N300').read(order='F')
    #l0.standardize()
    #for test_start_index in range(10):
    #    test = l0[test_start_index::10,:].read(order='C') #1/10th of the cids by starting at 0 to 9 and then incrementing by 10

    #    for snp_start_index in range(0, test.sid_count, 5):
    #        gene_on_test = test[:,snp_start_index:snp_start_index+5].read().standardize(BySqrtSidCount())
    #        #print gene_on_test
    #        #print gene_on_test.val
    #print "done"


    #snpreader0 = Bed('../../tests/datasets/all_chr.maf0.001.N300')


    ##from hdf5 import Hdf5
    ##Hdf5.write(snpreader, r'../../tests/datasets/all_chr.maf0.001.N300.hdf5')

    ##from dat import Dat
    ##Dat.write(snpreader, r'../../tests/datasets/all_chr.maf0.001.N300.dat')

    #G0 = snpreader0.read()
    #assert(G0 is not snpreader0.read())

    #G1 = G0.read()
    #assert(G0 is not G1)
    #assert(G0.val is not G1.val)

    #assert(G0.val[0,0] == 2)
    #G0.val[0,0] = 3
    #assert(G0.val[0,0] == 3)
    #assert(G1.val[0,0] == 2)

    #snpreader0b = snpreader0[:,:]
    #assert(snpreader0b is not snpreader0)

    #G = snpreader0[:,:].read()
    #assert snpreader0.iid_count == G.val.shape[0]
    #assert snpreader0.sid_count == G.val.shape[1]

    #sid_index_list = snpreader0.sid_to_index(['23_9','23_2'])
    #snpreader = snpreader0[:,sid_index_list]
    #assert((snpreader.sid == ['23_9','23_2'])).all()
    #snpreader = snpreader0[:,0:10]
    #assert((snpreader.sid == snpreader0.sid[0:10]).all())

    #snpreader2 = snpreader0[::-1,4]
    #print snpreader2
    #assert(snpreader2.iid_count == snpreader0.iid_count)
    #assert(snpreader2.sid_count == 1)
    #assert(len(snpreader2.pos) == 1)
    #assert(snpreader2.read().val.shape == (snpreader2.iid_count, snpreader2.sid_count))

    #assert(snpreader2[5,:].read().val.shape == (1L,1L))

    #iid_index_list = snpreader2.iid_to_index(snpreader2.iid[::2])
    #G = snpreader2[iid_index_list,::3].read()
    #assert(G.val.shape == (math.ceil(snpreader2.iid_count/2.0),math.ceil(snpreader2.sid_count/3.0) ))

    #assert(snpreader0[[4,5],:].read().val.shape == (2, snpreader0.sid_count))

    #snpreader_half = snpreader0[::2,::2]
    #assert(snpreader_half.read().val.shape == (math.ceil(snpreader0.iid_count/2.0), math.ceil(snpreader0.sid_count/2.0)))
    #snpreader_quarter = snpreader_half[::2,::2]
    #assert(snpreader_quarter.read().val.shape == (math.ceil(snpreader_half.iid_count/2.0), math.ceil(snpreader_half.sid_count/2.0)))

    #print snpreader2
    #print snpreader[::-1,4]
    #print snpreader2[iid_index_list,::3]
    #print snpreader[:,sid_index_list]
    #print snpreader2[5,:]
    #print snpreader[[4,5],:]

    #near_front = snpreader.pos[:,1] < .1
    #boolex = snpreader[:,near_front]
    #assert(boolex.read().val.shape == (snpreader.iid_count, 4))
    #boolex = snpreader[:,~near_front]
    #assert(boolex.read().val.shape == (snpreader.iid_count, 6))
    
    #print snpreader.read()
    #print snpreader.read().standardize()
    #print snpreader.read()



    import doctest
    doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
