from fastlmm.util.runner import *
import os
import base64
import numpy as SP
import sys
import logging
import zlib

class LocalMapper: # implements IRunner

    def __init__(self, taskcount, output_file_ignored, mkl_num_threads,logging_handler=logging.StreamHandler(sys.stdout),instream=sys.stdin,outstream=sys.stdout):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(logging_handler)

        self.taskcount = taskcount

        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

        if isinstance(instream, str):
            self.instream = open(instream,"r")
        else:
            self.instream = instream

        self.outstream = outstream



    def stream_result(self, work, zgoal, workindex):
        result = run_all_in_memory(work) #!! need to capture all stdout #!!test this and make a new method for the code that is the same on the two branches of the "if"
        s = pickle.dumps(result, pickle.HIGHEST_PROTOCOL)
        c = zlib.compress(s)
        self.outstream.write(str(workindex).zfill(zgoal))
        self.outstream.write("\tuu\t")
        self.outstream.write(base64.b64encode(c))
        self.outstream.write("\n")

    def run(self, original_distributable):
        #try:
        #    logging.info("hello")
        #    is_there = os.path.join(os.environ.get("userprofile"),".continuum","license_cluster_20140618175101.txt")
        #    logging.info('is there? {0}, {1}'.format(os.path.exists(is_there),is_there))
        #    #is_there = os.path.join(os.environ.get("userprofile"),"continuum","license_mkl_optimizations_20140602111401.txt")
        #    #logging.info('is there? {0}, {1}'.format(os.path.exists(is_there),is_there))
        #    logging.info("files {0}".format(os.listdir(".")))
        #    logging.info(".continuum files {0}".format(os.listdir(".continuum")))
        #    with open(r".continuum\license_cluster_20140618175101.txt") as f:
        #        logging.info(f.readlines())
        #    #logging.info("pythonpath.0.june_17_14 {0}".format(os.listdir("pythonpath.0.june_17_14")))
        #    logging.info((sys.version))
        #    logging.info("HOME={0}".format(os.environ.get("HOME")))
        #    for key,val in os.environ.iteritems():
        #        logging.info("set {0}={1}".format(key, val))
        #    logging.info("good night")
        #    #import mkl
        #    #import time
        #    #time.sleep(5 * 60 * 60)
        #except Exception, detail:
        #    logging.warn(detail)

        #import mkl
        #import numpy as SP
        zgoal = int(SP.ceil(SP.log(self.taskcount)/SP.log(10)))
        JustCheckExists().input(original_distributable)
        for line in self.instream:

            lineparts = line.split('\t')
            if len(lineparts) == 2 :
                taskindex = int(lineparts[1])
            else:
                taskindex = int(line)

            logging.info("reporter:counter:LocalMapper,sumTaskIndex,{0}".format(taskindex))
            if taskindex < 0 or taskindex >= self.taskcount : raise Exception("taskindex {0} should be at least 0 and stictly less than {1}.".format(taskindex,self.taskcount))
            shaped_distributable = shape_to_desired_workcount(original_distributable, self.taskcount)
            assert shaped_distributable.work_count == self.taskcount, "expect workcount == taskcount"


            if hasattr(shaped_distributable,"work_sequence_range"):
                is_first_and_only = True
                for work in shaped_distributable.work_sequence_range(taskindex, taskindex+1):
                    assert is_first_and_only, "real assert"
                    is_first_and_only = False
                    self.stream_result(work, zgoal, taskindex)
            else:
                workDone = False
                for workindex, work in enumerate(shaped_distributable.work_sequence()):
                    if workindex == self.taskcount : raise Exception("Expect len(work_sequence) to match work_count, but work_sequence was too long")
                    if workindex == taskindex :
                        self.stream_result(work, zgoal, workindex)
                        workDone = True
                        if workindex != self.taskcount-1  : #the work is done, so quit enumerating work (but don't quit early if you're the last workIndex because we want to double check that the work_sequence and work_count match up)
                                break
                if not workDone : raise Exception("Expect len(work_sequence) to match work_count, but work_sequence was too short")
        return None
