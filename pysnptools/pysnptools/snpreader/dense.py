import numpy as SP
import logging
from snpreader import SnpReader

class Dense(SnpReader):
    '''
    This is a class that reads into memory from DAT/FAM/MAP files.
    '''

    _ran_once = False

    def __init__(self, filename, extract_iid_function=lambda s:("0",s), extract_bim_function=lambda s:("0",s,0,0)):
        '''
        filename    : string of the name of the Dat file.
        '''
        self.filename = filename
        self.extract_iid_function = extract_iid_function
        self.extract_bim_function = extract_bim_function

    #!! similar code in fastlmm
    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.filename)


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

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        bim_list = []
        with open(self.filename,"r") as fp:
            header = fp.readline()
            iid_string_list = header.strip().split()[1:]
            self._original_iid = SP.array([self.extract_iid_function(iid_string) for iid_string in iid_string_list],dtype="string")
            val_list = []
            for line_index,line in enumerate(fp):
                if line_index % 1000 == 0:
                    logging.info("reading sid and iid info from line {0} of file '{1}'".format(line_index, self.filename))
                sid_string, rest = line.strip().split()
                bim_list.append(self.extract_bim_function(sid_string))
                assert len(rest) == len(self._original_iid)

        self._sid = SP.array([bim[1] for bim in bim_list],dtype='str')
        self._pos = SP.array([[bim[0],bim[2],bim[3]] for bim in bim_list],dtype='int')

        return self

    #def __del__(self):
    #    if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
    #        self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because creates name of all files itself
        copier.input(self.filename)

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
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

        if not hasattr(self,"val_list_list"):
            val_list_list = []
            with open(self.filename,"r") as fp:
                header = fp.readline()
                for line_index,line in enumerate(fp):
                    if line_index % 1000 == 0:
                        logging.info("reading values from line {0} of file '{1}'".format(line_index, self.filename))
                    sid_string, rest = line.strip().split()
                    assert len(rest) == len(self._original_iid)
                    val_list = [int(val) for val in rest]
                    val_list_list.append(val_list)
            self.val_list_list = val_list_list


        val = SP.zeros((iid_count_out,sid_count_out),order=order, dtype=dtype)
        for SNPsIndex, sid_index in enumerate(sid_index_out):
            row = SP.array(self.val_list_list[sid_index])
            val[:,SNPsIndex] = row[iid_index_out]
        return val


    #@staticmethod
    #def write(snpdata, basefilename):

    #    iid = snpdata.iid
    #    sid = snpdata.sid
    #    pos = snpdata.pos
    #    snpsarray = snpdata.val

    #    famfile, mapfile = Dat.names_of_other_files_static(basefilename)

    #    with open(famfile,"w") as fam_filepointer:
    #        for iid_row in iid:
    #            fam_filepointer.write("{0} {1} 0 0 0 0\n".format(iid_row[0],iid_row[1]))

    #    with open(mapfile,"w") as map_filepointer:
    #        for sid_index, sid in enumerate(sid):
    #            posrow = pos[sid_index]
    #            map_filepointer.write("{0}\t{1}\t{2}\t{3}\n".format(posrow[0], sid, posrow[1], posrow[2]))

    #    with open(datfile,"w") as dat:
    #        for sid_index, sid in enumerate(sid):
    #            dat.write("{0}\tj\tn\t".format(sid)) #use "j" and "n" as the major and minor allele
    #            row = snpsarray[:,sid_index]
    #            dat.write("\t".join([str(i) for i in row]) + "\n")

#if __name__ == "__main__":
#    logging.basicConfig(level=logging.INFO)

#    snpreader = Dat(r'../../tests/datasets/all_chr.maf0.001.N300.dat')
#    snp_matrix = snpreader.read()
#    print len(snp_matrix['sid'])
#    snp_matrix = snpreader[:,:].read()
#    print len(snp_matrix['sid'])
#    sid_index_list = snpreader.sid_to_index(['23_9','23_2'])
#    snp_matrix = snpreader[:,sid_index_list].read()
#    print ",".join(snp_matrix['sid'])
#    snp_matrix = snpreader[:,0:10].read()
#    print ",".join(snp_matrix['sid'])

#    print snpreader.iid_count
#    print snpreader.sid_count
#    print len(snpreader.pos)

#    snpreader2 = snpreader[::-1,4]
#    print snpreader.iid_count
#    print snpreader2.sid_count
#    print len(snpreader2.pos)

#    snp_matrix = snpreader2.read()
#    print len(snp_matrix['iid'])
#    print len(snp_matrix['sid'])

#    snp_matrix = snpreader2[5,:].read()
#    print len(snp_matrix['iid'])
#    print len(snp_matrix['sid'])

#    iid_index_list = snpreader2.iid_to_index(snpreader2.iid[::2])
#    snp_matrix = snpreader2[iid_index_list,::3].read()
#    print len(snp_matrix['iid'])
#    print len(snp_matrix['sid'])

#    snp_matrix = snpreader[[4,5],:].read()
#    print len(snp_matrix['iid'])
#    print len(snp_matrix['sid'])

#    print snpreader2
#    print snpreader[::-1,4]
#    print snpreader2[iid_index_list,::3]
#    print snpreader[:,sid_index_list]
#    print snpreader2[5,:]
#    print snpreader[[4,5],:]
