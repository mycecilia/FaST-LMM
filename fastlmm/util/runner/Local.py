'''
Runs a distributable job locally in one process. Returns the value of the job.

See SamplePi.py for examples.
'''

from fastlmm.util.runner import *
import os, sys
import logging

class Local: # implements IRunner
    def __init__(self, mkl_num_threads = None, logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(logging_handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        
        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

    def run(self, distributable):
        JustCheckExists().input(distributable)
        result = run_all_in_memory(distributable)
        JustCheckExists().output(distributable)
        return result

class JustCheckExists(object): #Implements ICopier

    def __init__(self,doPrintOutputNames=False):
        self.doPrintOutputNames = doPrintOutputNames
    
    def input(self,item):
        if isinstance(item, str):
            if not os.path.exists(item): raise Exception("Missing input file '{0}'".format(item))
        elif hasattr(item,"copyinputs"):
            item.copyinputs(self)
        # else -- do nothing

    def output(self,item):
        if isinstance(item, str):
            if not os.path.exists(item): raise Exception("Missing output file '{0}'".format(item))
            if self.doPrintOutputNames:
                print item
        elif hasattr(item,"copyoutputs"):
            item.copyoutputs(self)
        # else -- do nothing
