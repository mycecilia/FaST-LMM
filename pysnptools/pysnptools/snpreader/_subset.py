import numpy as SP
import subprocess, sys, os.path
from itertools import *
from pysnptools.pysnptools.altset_list import *
import pandas as pd
import logging
from snpreader import SnpReader
from snpdata import SnpData

class _Subset(SnpReader):

    def __init__(self, internal, iid_indexer, sid_indexer):
        '''
        an indexer can be:
             an integer i (same as [i])
             a slice
             a list of integers
             a list of booleans
        '''
        self._internal = internal
        self._iid_indexer = SnpReader._make_sparray_or_slice(iid_indexer)
        self._sid_indexer = SnpReader._make_sparray_or_slice(sid_indexer)

    _ran_once = False

    def __repr__(self): #!!Should this be __str__ (and elsewhere) because it uses "nice_string" which uses "..." so may be ambiguous?
        s = "{0}[{1},{2}]".format(self._internal,self._nice_string(self._iid_indexer),self._nice_string(self._sid_indexer))
        return s

    _slice_format = {(False,False,False):":",
                     (False,False,True):"::{2}",
                     (False,True,False):":{1}",
                     (False,True,True):":{1}:{2}",
                     (True,False,False):"{0}:",
                     (True,False,True):"{0}::{2}",
                     (True,True,False):"{0}:{1}",
                     (True,True,True):"{0}:{1}:{2}"}

    def _nice_string(self, some_slice):
        if isinstance(some_slice,slice):
            return self._slice_format[(some_slice.start is not None, some_slice.stop is not None, some_slice.step is not None)].format(some_slice.start, some_slice.stop, some_slice.step)
        elif len(some_slice) == 1:
            return str(some_slice[0])
        elif len(some_slice) < 10:
            return "[{0}]".format(",".join([str(i) for i in some_slice]))
        else:
            return "[{0},...]".format(",".join([str(i) for i in some_slice[:10]]))

    def copyinputs(self, copier):
        self._internal.copyinputs(copier)

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

    #!!commented out because doesn't guarantee that the shortcut will return with the dtype and order requested.
    #                  Also, didn't handle stacked do-nothing subsets
    #def read(self, order='F', dtype=SP.float64, force_python_only=False, view_ok=False):
    #    if view_ok and hasattr(self._internal,"val") and _Subset._is_all_slice(self._iid_indexer) and _Subset._is_all_slice(self._sid_indexer):
    #        return self._internal
    #    else:
    #        return SnpReader.read(self, order, dtype, force_python_only, view_ok)


    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = None
    def _read(self, iid_indexer, sid_indexer, order, dtype, force_python_only, view_ok):
        self.run_once()

        if hasattr(self._internal,'_read_accepts_slices'):
            composed_iid_index_or_none = self.compose_indexer_with_indexer(self._internal.iid_count, self._iid_indexer, self.iid_count, iid_indexer)
            composed_sid_index_or_none = self.compose_indexer_with_indexer(self._internal.sid_count, self._sid_indexer, self.sid_count, sid_indexer)
            val = self._internal._read(composed_iid_index_or_none, composed_sid_index_or_none, order, dtype, force_python_only, view_ok)
            return val
        else:
            iid_index_or_none = self._make_sparray_from_sparray_or_slice(self.iid_count, iid_indexer)
            composed_iid_index_or_none = self.compose_indexer_with_index_or_none(self._internal.iid_count, self._iid_indexer, self.iid_count, iid_index_or_none)
            sid_index_or_none = self._make_sparray_from_sparray_or_slice(self.sid_count, sid_indexer)
            composed_sid_index_or_none = self.compose_indexer_with_index_or_none(self._internal.sid_count, self._sid_indexer, self.sid_count, sid_index_or_none)
            val = self._internal._read(composed_iid_index_or_none, composed_sid_index_or_none, order, dtype, force_python_only, view_ok)
            return val

    def run_once(self):
        if self._ran_once:
            return

        self._ran_once = True
        self._iid = self._internal.iid[self._iid_indexer]
        self._sid = self._internal.sid[self._sid_indexer]
        self._pos = self._internal.pos[self._sid_indexer]

    @staticmethod
    def compose_indexer_with_index_or_none(countA, indexerA, countB, index_or_noneB):
        if _Subset._is_all_slice(indexerA):
            return index_or_noneB

        indexA = SnpReader._make_sparray_from_sparray_or_slice(countA, indexerA)

        if _Subset._is_all_slice(index_or_noneB):
            return indexA

        indexAB = indexA[index_or_noneB]

        return indexAB


    @staticmethod
    def compose_indexer_with_indexer(countA, indexerA, countB, indexerB):
        if _Subset._is_all_slice(indexerA):
            return indexerB

        if _Subset._is_all_slice(indexerB):
            return indexerA

        indexA = SnpReader._make_sparray_from_sparray_or_slice(countA, indexerA)
        indexB = SnpReader._make_sparray_from_sparray_or_slice(countB, indexerB)

        indexAB = indexA[indexB]

        return indexAB
