try:
    import h5py
except:
    pass

import logging
import scipy as sp
from snpreader import SnpReader

#!! document the format

class Hdf5(SnpReader):

    _ran_once = False
    h5 = None

    def __init__(self, filename, blocksize=5000):
        ##!! copy relevent comments from Bed reader
        self.filename=filename
        self.blocksize = blocksize

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.filename) #!!LATER print non-default values, too

    def copyinputs(self, copier):
        copier.input(self.filename)


    @property
    def iid(self):
        self.run_once()
        return self._iid

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
        try:
            self.h5 = h5py.File(self.filename, "r")
        except IOError, e:
            raise IOError("Missing or unopenable file '{0}' -- Native error message: {1}".format(self.filename,e))

        self._iid = sp.empty(self.h5['iid'].shape,dtype=self.h5['iid'].dtype) #make a 2D deepcopy from h5 (more direct methods, don't seem to work)
        for iRow, row in enumerate(self.h5['iid']):
            for iCol, value in enumerate(row):
                self._iid[iRow,iCol] = value

        self._pos = sp.array(self.h5['pos'])
        try: #new format
            self._sid = sp.array(self.h5['sid'])
            self.val_in_file = self.h5['val']
        except: # try old format
            self._sid = sp.array(self.h5['rs'])
            self.val_in_file = self.h5['snps']

        if "SNP-major" not in self.val_in_file.attrs: raise Exception("In Hdf5 the 'val' matrix must have a Boolean 'SNP-major' attribute")
        self.is_snp_major = self.val_in_file.attrs["SNP-major"]

        S_original = len(self._sid)
        N_original = len(self._iid)
        if self.is_snp_major:
            if not self.val_in_file.shape == (S_original, N_original, ) : raise Exception("In Hdf5, snps matrix dimensions don't match those of 'sid' and 'iid'")
        else:
            if not self.val_in_file.shape == (N_original, S_original) : raise Exception("In Hdf5, snps matrix dimensions don't match those of 'sid' and 'iid'")

        self._ran_once = True


    @staticmethod
    #should move into utils
    def is_sorted_without_repeats(list):
        if len(list) < 2:
            return True
        for i in xrange(1,len(list)):
            if not list[i-1] < list[i]:
                return False
        return True
    

    def __del__(self):
        if self.h5 != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self.h5.close()

    def read_direct(self, snps, selection=sp.s_[:,:]):

        if self.is_snp_major:
            selection = tuple(reversed(selection))

        if snps.flags["F_CONTIGUOUS"]:
            self.val_in_file.read_direct(snps.T,selection)
        else:
            self.val_in_file.read_direct(snps,selection)

    #!! much code the same as for Bed
    def create_block(self, blocksize, order, dtype):

        N_original = len(self._iid) #similar code else where -- make a method
        matches_order = self.is_snp_major == (order =="F") #similar code else where -- make a method
        opposite_order = "C" if order == "F" else "F"#similar code else where -- make a method
        if matches_order:
            return sp.empty([N_original,blocksize], dtype=dtype, order=order)
        else:
            return sp.empty([N_original,blocksize], dtype=dtype, order=opposite_order)

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self.run_once()
        assert not hasattr(self, 'ind_used'), "A SnpReader should not have a 'ind_used' attribute"

        if order is None:
            order = "F"
        if dtype is None:
            dtype = SP.float64
        if force_python_only is None:
            force_python_only = False


        opposite_order = "C" if order == "F" else "F"

        N_original = self.iid_count
        S_original = self.sid_count

        if iid_index_or_none is not None:
            N = len(iid_index_or_none)
            iid_index_list = iid_index_or_none
            iid_is_sorted = Hdf5.is_sorted_without_repeats(iid_index_list)
        else:
            N = N_original
            iid_index_list = range(N_original)
            iid_is_sorted = True

        if sid_index_or_none is not None:
            S = len(sid_index_or_none)
            sid_index_list = sid_index_or_none
        else:
            S = S_original
            sid_index_list = range(S_original)
        # Check if snps and iids indexes are in order and in range
        snps_are_sorted = Hdf5.is_sorted_without_repeats(sid_index_list)

        val = sp.empty([N, S], dtype=dtype, order=order)

        matches_order = self.is_snp_major == (order=="F")
        is_simple = not force_python_only and iid_is_sorted and snps_are_sorted and matches_order #If 'is_simple' may be able to use a faster reader

        # case 1 - all snps & all ids requested
        if is_simple and S == S_original and N == N_original:
            self.read_direct(val)

        # case 2 - some snps and all ids
        elif is_simple and N == N_original:
            self.read_direct(val, sp.s_[:,sid_index_list])

        # case 3 all snps and some ids
        elif is_simple and S == S_original:
            self.read_direct(val, sp.s_[iid_index_list,:])

        # case 4 some snps and some ids -- use blocks
        else:
            blocksize = min(self.blocksize, S)
            block = self.create_block(blocksize, order, dtype)

            if not snps_are_sorted:
                sid_index_index_list = sp.argsort(sid_index_list)
                sid_index_list_sorted = sid_index_list[sid_index_index_list]
            else:
                sid_index_index_list = sp.arange(S)
                sid_index_list_sorted = sid_index_list

            for start in xrange(0, S, blocksize):
                #print start
                end = min(start+blocksize,S)
                if end-start < blocksize:  #On the last loop, the buffer might be too big, so make it smaller
                    block = self.create_block(end-start, order, dtype)
                sid_index_list_forblock = sid_index_list_sorted[start:end]
                sid_index_index_list_forblock = sid_index_index_list[start:end]
                self.read_direct(block, sp.s_[:,sid_index_list_forblock])
                val[:,sid_index_index_list_forblock] = block[iid_index_list,:]

        #sid = self._sid[sid_index_list]
        #pos = self._pos[sid_index_list,:]
        #iid = sp.array(self._original_iid[iid_index_list],dtype="S") #Need to make another copy of to stop it from being converted to a list of 1-d string arrays


        #!!LATER does this test work when the size is 1 x 1 and order if F? iid_index_or_none=[0], sid_index_or_none=[1000] (based on test_blocking_hdf5)
        has_right_order = (order=="C" and val.flags["C_CONTIGUOUS"]) or (order=="F" and val.flags["F_CONTIGUOUS"])
        assert val.shape == (N, S) and val.dtype == dtype and has_right_order

        #ret = {
        #        'iid'    :iid,
        #        'sid'    :sid,
        #        'pos'    :pos,
        #        'val'    :val
        #        }
        return val




    @staticmethod
    def write(snpdata, hdf5file, dtype='f8',snp_major=True):
        if not isinstance(dtype, str) or len(dtype) != 2 or dtype[0] != 'f' : raise Exception("Expect dtype to start with 'f', e.g. 'f4' for single, 'f8' for double")
        val = (snpdata.val.T) if snp_major else snpdata.val
        with h5py.File(hdf5file, "w") as h5:
            h5.create_dataset('iid', data=snpdata.iid)
            h5.create_dataset('sid', data=snpdata.sid)
            h5.create_dataset('pos', data=snpdata.pos)
            h5.create_dataset('val', data=val,dtype=dtype,shuffle=True)#compression="gzip", doesn't seem to work with Anaconda
            h5['val'].attrs["SNP-major"] = snp_major

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    #snpreader = Hdf5(r'../../tests/datasets/all_chr.maf0.001.N300.hdf5')
    #snp_matrix = snpreader.read()
    #print len(snp_matrix['sid'])
    #snp_matrix = snpreader[:,:].read()
    #print len(snp_matrix['sid'])
    #sid_index_list = snpreader.sid_to_index(['23_9','23_2'])
    #snp_matrix = snpreader[:,sid_index_list].read()
    #print ",".join(snp_matrix['sid'])
    #snp_matrix = snpreader[:,0:10].read()
    #print ",".join(snp_matrix['sid'])

    #print snpreader.iid_count
    #print snpreader.sid_count
    #print len(snpreader.pos)

    #snpreader2 = snpreader[::-1,4]
    #print snpreader.iid_count
    #print snpreader2.sid_count
    #print len(snpreader2.pos)

    #snp_matrix = snpreader2.read()
    #print len(snp_matrix['iid'])
    #print len(snp_matrix['sid'])

    #snp_matrix = snpreader2[5,:].read()
    #print len(snp_matrix['iid'])
    #print len(snp_matrix['sid'])

    #iid_index_list = snpreader2.iid_to_index(snpreader2.iid[::2])
    #snp_matrix = snpreader2[iid_index_list,::3].read()
    #print len(snp_matrix['iid'])
    #print len(snp_matrix['sid'])

    #snp_matrix = snpreader[[4,5],:].read()
    #print len(snp_matrix['iid'])
    #print len(snp_matrix['sid'])

    #print snpreader2
    #print snpreader[::-1,4]
    #print snpreader2[iid_index_list,::3]
    #print snpreader[:,sid_index_list]
    #print snpreader2[5,:]
    #print snpreader[[4,5],:]


    import doctest
    doctest.testmod()

