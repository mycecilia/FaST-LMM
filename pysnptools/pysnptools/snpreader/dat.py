import numpy as SP
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from snpreader import SnpReader

class Dat(SnpReader):
    '''
    This is a class that reads into memory from DAT/FAM/MAP files.
    '''

    _ran_once = False

    def __init__(self, dat_filename):
        '''
        filename    : string of the name of the Dat file.
        '''
        self.dat_filename = dat_filename

    #!! similar code in fastlmm
    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.dat_filename)


    @property
    def iid(self):
        self.run_once()
        return self._original_iid

    @property
    def sid(self):
        self.run_once()
        return self._sid

    @property
    def pos(self):
        self.run_once()
        return self._pos



    def names_of_other_files(self):
        return Dat.names_of_other_files_static(self.dat_filename)

    @staticmethod
    def names_of_other_files_static(dat_filename):
        base_filename = (".").join(dat_filename.split(".")[:-1])
        famfile = base_filename + '.fam'
        mapfile = base_filename + '.map'
        return famfile, mapfile

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        famfile, mapfile = self.names_of_other_files()

        #!!similar code in BED reader
        logging.info("Loading fam file {0}".format(famfile))
        self._original_iid = SP.loadtxt(famfile,delimiter = ' ',dtype = 'str',usecols=(0,1),comments=None)

        #!!similar code in BED reader
        logging.info("Loading map file {0}".format(mapfile))
        bimfields = pd.read_csv(mapfile,delimiter = '\t',usecols = (0,1,2,3),header=None,index_col=False)
        self._sid = SP.array(bimfields[1].tolist(),dtype='str')
        self._pos = bimfields.as_matrix([0,2,3])

        return self

    #def __del__(self):
    #    if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
    #        self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because creates name of all files itself
        famfile, mapfile = self.names_of_other_files()
        copier.input(self.dat_filename)
        copier.input(famfile)
        copier.input(mapfile)

    @property
    def datfields(self):
        if not hasattr(self,"_datfields"):
            #!!could change to just create/find an index to the file position of each row. Instead, reading all into memory
            datfields = pd.read_csv(self.dat_filename,delimiter = '\t',header=None,index_col=False)
            if not SP.array_equal(SP.array(datfields[0],dtype="string"), self.sid) : raise Exception("Expect snp list in map file to exactly match snp list in dat file")
            self.start_column = 3
            if len(self._original_iid) != datfields.shape[1]-self.start_column : raise Exception("Expect # iids in fam file to match dat file")
            self._datfields = datfields.T
        return self._datfields

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        '''
        Output dictionary:
        'iid' : [N*2] array of family IDs and individual IDs
        'sid' : [S] array rs-numbers or snp identifiers
        'pos' : [S*3] array of positions [chromosome, genetic dist, basepair dist]
        'val' : [N*S] matrix of per iid snp values
        '''
        assert not hasattr(self, 'ind_used'), "A SnpReader should not have a 'ind_used' attribute"
        if order is None:
            order = "F"
        if dtype is None:
            dtype = SP.float64
        if force_python_only is None:
            force_python_only = False

        #This could be re-factored to not use so many names
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


        val = SP.zeros((iid_count_out,sid_count_out),order=order, dtype=dtype)
        datfields = self.datfields
        for SNPsIndex, sid_index in enumerate(sid_index_out):
            row = SP.array(datfields[sid_index])[self.start_column:,]
            val[:,SNPsIndex] = row[iid_index_out]
        return val


    @staticmethod
    def write(snpdata, basefilename):

        iid_list = snpdata.iid
        sid_list = snpdata.sid
        pos_list = snpdata.pos
        snpsarray = snpdata.val

        famfile, mapfile = Dat.names_of_other_files_static(basefilename)

        with open(famfile,"w") as fam_filepointer:
            for iid_row in iid_list:
                fam_filepointer.write("{0} {1} 0 0 0 0\n".format(iid_row[0],iid_row[1]))

        with open(mapfile,"w") as map_filepointer:
            for sid_index, sid in enumerate(sid_list):
                posrow = pos_list[sid_index]
                map_filepointer.write("{0}\t{1}\t{2}\t{3}\n".format(posrow[0], sid, posrow[1], posrow[2]))

        with open(basefilename,"w") as dat_filepointer:
            for sid_index, sid in enumerate(sid_list):
                if sid_index % 1000 == 0:
                    logging.info("Writing snp # {0} to file '{1}'".format(sid_index, basefilename))
                dat_filepointer.write("{0}\tj\tn\t".format(sid)) #use "j" and "n" as the major and minor allele
                row = snpsarray[:,sid_index]
                dat_filepointer.write("\t".join((str(i) for i in row)) + "\n")
        logging.info("Done writing " + basefilename)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    snpreader = Dat(r'../../tests/datasets/all_chr.maf0.001.N300.dat')
    snp_matrix = snpreader.read()
    print len(snp_matrix['sid'])
    snp_matrix = snpreader[:,:].read()
    print len(snp_matrix['sid'])
    sid_index_list = snpreader.sid_to_index(['23_9','23_2'])
    snp_matrix = snpreader[:,sid_index_list].read()
    print ",".join(snp_matrix['sid'])
    snp_matrix = snpreader[:,0:10].read()
    print ",".join(snp_matrix['sid'])

    print snpreader.iid_count
    print snpreader.sid_count
    print len(snpreader.pos)

    snpreader2 = snpreader[::-1,4]
    print snpreader.iid_count
    print snpreader2.sid_count
    print len(snpreader2.pos)

    snp_matrix = snpreader2.read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    snp_matrix = snpreader2[5,:].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    iid_index_list = snpreader2.iid_to_index(snpreader2.iid[::2])
    snp_matrix = snpreader2[iid_index_list,::3].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    snp_matrix = snpreader[[4,5],:].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    print snpreader2
    print snpreader[::-1,4]
    print snpreader2[iid_index_list,::3]
    print snpreader[:,sid_index_list]
    print snpreader2[5,:]
    print snpreader[[4,5],:]
