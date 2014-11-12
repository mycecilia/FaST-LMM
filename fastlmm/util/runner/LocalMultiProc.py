'''
Runs a distributable job on multiple processors.  Returns the value of the job.

See SamplePi.py for examples.
'''

from fastlmm.util.runner import *
import os
import cPickle as pickle
import subprocess, sys, os.path
import multiprocessing
import logging
import fastlmm.util.util as util

class LocalMultiProc: # implements IRunner

    def __init__(self, taskcount, mkl_num_threads = None,logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(logging_handler)

        self.taskcount = taskcount
        self.mkl_num_threads = mkl_num_threads

    def run(self, distributable):
        JustCheckExists().input(distributable)

        localpath = os.environ["PATH"]
        localwd = os.getcwd()

        import datetime
        now = datetime.datetime.now()
        run_dir_rel = os.path.join("runs",util.datestamp(appendrandom=True))
        run_dir_abs = os.path.join(localwd,run_dir_rel)
        util.create_directory_if_necessary(run_dir_rel, isfile=False)

        distributablep_filename = os.path.join(run_dir_rel, "distributable.p")
        with open(distributablep_filename, mode='wb') as f:
            pickle.dump(distributable, f, pickle.HIGHEST_PROTOCOL)

        distributable_py_file = os.path.join(os.path.dirname(__file__),"..","distributable.py")
        if not os.path.exists(distributable_py_file): raise Exception("Expect file at " + distributable_py_file + ", but it doesn't exist.")
        command_format_string = sys.executable + " " + distributable_py_file + " " + distributablep_filename +" LocalInParts({0},{1},mkl_num_threads={2})".format("{0}", self.taskcount, self.mkl_num_threads)

        proc_list = []
        for taskindex in xrange(self.taskcount):
            command_string = command_format_string.format(taskindex)
            proc = subprocess.Popen(command_string.split(" "), cwd=os.getcwd())
            proc_list.append(proc)

        for taskindex, proc in enumerate(proc_list):            
            rc = proc.wait()
            #for line in proc.stdout.readlines():
            #    sys.stdout.write(line)
            if not 0 == rc : raise Exception("Running python in python results in non-zero return code in task#{0}".format(taskindex))

        result = run_one_task(distributable, self.taskcount, self.taskcount, distributable.tempdirectory)
        JustCheckExists().output(distributable)
        return result
