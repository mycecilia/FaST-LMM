import numpy as SP
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from snpreader import SnpReader
from pysnptools.pysnptools.standardizer.unit import Unit
from pysnptools.pysnptools.standardizer.identity import Identity

class SnpData(SnpReader):
    """  This is a class hold SNP values in-memory along with related iid and sid information.
    It is created by calling the :meth:`.SnpReader.read` method on another :class:`.SnpReader`, for example, :class:`.Bed`.

    See :class:`.SnpReader` for details and examples.
    """
    def __init__(self, iid, sid, pos, val, parent_string="",copyinputs_function=None):
        self._iid = iid
        self._sid = sid
        self._pos = pos
        assert type(val) == SP.ndarray, "expect SnpData's val to be a ndarray"
        self.val = val
        self._parent_string = parent_string
        self._std_string_list = []
        if copyinputs_function is not None:
            self.copyinputs = copyinputs_function

    val = None
    """The in-memory SNP data. A numpy.ndarray with dimensions :attr:`.iid_count` x :attr:`.sid_count`.

    See :class:`.SnpReader` for details and examples.
    """

    def __repr__(self):
        if self._parent_string == "":
            if len(self._std_string_list) > 0:
                s = "SnpData({0})".format(",".join(self._std_string_list))
            else:
                s = "SnpData()"
        else:
            if len(self._std_string_list) > 0:
                s = "SnpData({0},{1})".format(self._parent_string,",".join(self._std_string_list))
            else:
                s = "SnpData({0})".format(self._parent_string)
        return s

    def copyinputs(self, copier):
        pass

    @property
    def iid(self):
        """A ndarray of the iids.

        See :attr:`.SnpReader.iid` for details and examples.
        """
        return self._iid

    @property
    def sid(self):
        """A ndarray of the sids.

        See :attr:`.SnpReader.sid` for details and examples.
        """
        return self._sid

    @property
    def pos(self):
        """A ndarray of the position information for each sid.

        See :attr:`.SnpReader.pos` for details and examples.
        """
        return self._pos

    #!!Seems like we can't short cut the view_OK this because the .val wouldn't necessarily have the right order and dtype
    #def read(self, order='F', dtype=SP.float64, force_python_only=False, view_ok=False):
    #    """creates an in-memory :class:`.SnpData` with a copy of the genotype data
    #    """
    #    if view_ok:
    #        return self
    #    else:
    #        return SnpReader.read(self, order, dtype, force_python_only, view_ok)


    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = None
    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        val, shares_memory = self._apply_sparray_or_slice_to_val(self.val, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only)
        if shares_memory and not view_ok:
            val = val.copy(order='K')
        return val

    def standardize(self, standardizer=Unit(), blocksize=None, force_python_only=False):
        """Does in-place standardization of the in-memory
        SNP data. By default, it applies 'Unit' standardization, that is: the values for each SNP will have mean zero and standard deviation 1.0.
        Note that, for efficiently, this method works in-place, actually changing values in the ndarray. Although it works in place, for convenience
        it also returns itself.

        :param standardizer: optional -- Specify standardization to be applied before the matrix multiply. 
             Any :class:`.Standardizer` may be used. Some choices include :class:`.Unit` (default, makes values for each SNP have mean zero and
             standard deviation 1.0), :class:`.Beta`, :class:`.BySidCount`, :class:`.BySqrtSidCount`.
        :type order: :class:`.Standardizer`

        :param blocksize: optional -- Default of None. None means to load all. Suggested number of sids to read into memory at a time.
        :type blocksize: int or None

        :rtype: :class:`.SnpData` (standardizes in place, but for convenience, returns 'self')

        >>> from pysnptools.pysnptools.snpreader.bed import Bed
        >>> snp_on_disk = Bed('../../tests/datasets/all_chr.maf0.001.N300') # Specify some data on disk in Bed format
        >>> snpdata1 = snp_on_disk.read() # read all SNP values into memory
        >>> print snpdata1 # Prints the specification for this SnpData
        SnpData(Bed('../../tests/datasets/all_chr.maf0.001.N300'))
        >>> print snpdata1.val[0,0]
        2.0
        >>> snpdata1.standardize() # standardize changes the values in snpdata1.val and changes the specification.
        SnpData(Bed('../../tests/datasets/all_chr.maf0.001.N300'),Unit())
        >>> print snpdata1.val[0,0]
        0.229415733871
        >>> snpdata2 = snp_on_disk.read().standardize() # Read and standardize in one expression with only one ndarray allocated.
        >>> print snpdata2.val[0,0]
        0.229415733871
        """
        self.val = standardizer.standardize(self.val, blocksize=blocksize, force_python_only=force_python_only)
        self._std_string_list.append(str(standardizer))
        return self

    def kernel(self, standardizer, blocksize=10000, allowlowrank=False):
        """
            See :meth:`.SnpReader.kernel` for details and examples.
        """
        if type(standardizer) is Identity:
            K = self.val.dot(self.val.T)
            return K
        else:
            K = SnpReader.kernel(self, standardizer, blocksize=blocksize, allowlowrank=allowlowrank)
            return K

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
