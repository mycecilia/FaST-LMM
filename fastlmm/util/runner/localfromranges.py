'''
Runs one part of a distributable job locally. The last part will return the jobs value. The other parts return 'None'

See SamplePi.py for examples.
'''
from fastlmm.util.runner import *
import os, sys
import logging
import fastlmm.util.util as util
import cPickle as pickle
import itertools

class LocalFromRanges: # implements IRunner
    '''
    Created mostly for testing. This runner divides the work_sequence into a series of ranges.
    '''
    def __init__(self, end_list, mkl_num_threads=None):
        self.end_list = end_list
        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


    def work_sequence_range(self,distributable,start,end):
        if hasattr(distributable,"work_sequence_range"):
            return distributable.work_sequence_range(start,end)
        else:
            return isplice(distributable.work_sequence,start,end)

    def distributable_to_result_sequence_via_range(self, distributable):
        start = 0
        for end in self.end_list:
            assert start <= end, "The end_list must be a list of sorted non-negative numbers."
            work_sequence = self.work_sequence_range(distributable,start,end)
            result_sequence = work_sequence_to_result_sequence(work_sequence)
            for result in result_sequence:
                yield result
            start = end

    def run(self, distributable):
        JustCheckExists().input(distributable)

        if callable(distributable):
            result = distributable()
        else:
            count = self.end_list[-1]
            assert 0 <= count, "The end_list must be a list of sorted non-negative numbers."
            shaped_distributable = shape_to_desired_workcount(distributable, count)
            result_sequence = self.distributable_to_result_sequence_via_range(shaped_distributable)
            result = shaped_distributable.reduce(result_sequence)

        JustCheckExists().output(distributable)
        return result

