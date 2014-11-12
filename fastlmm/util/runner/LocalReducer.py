import os, sys
from fastlmm.util.runner import *
import base64
import logging
import fastlmm.util.util as util
import cPickle as pickle

class LocalReducer: # implements IRunner

    def __init__(self, taskcount, result_file, mkl_num_threads, logging_handler=logging.StreamHandler(sys.stdout), instream=sys.stdin):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(logging_handler)

        self.taskcount = taskcount
        self.result_file = result_file

        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

        if isinstance(instream, str):
            self.instream = open(instream,"r")
        else:
            self.instream = instream



    def work_sequence_from_stdin(self):
        import re
        import zlib
        uuencodePattern = re.compile("[0-9]+\tuu\t")

        for line in self.instream:
            #e.g. 000	gAJ
            if None != uuencodePattern.match(line): # reminder: python's "match" looks for match at the start of the string
                # hack to get around info messages in stdout
                taskindex, uu, encoded = line.split('\t')
                c = base64.b64decode(encoded)
                s = zlib.decompress(c)
                logging.info("taskindex={0}, len(encoded)={1}, len(zipped)={2}, len(pickle)={3}".format(taskindex,len(encoded),len(c),len(s)))
                result = pickle.loads(s)
                yield result

    def run(self, original_distributable):
        result_sequence = self.work_sequence_from_stdin()
        shaped_distributable = shape_to_desired_workcount(original_distributable, self.taskcount)
        if shaped_distributable.work_count != self.taskcount : raise Exception("Assert: expect workcount == taskcount")
        result = shaped_distributable.reduce(result_sequence)
        #close the instream if it is a file?

        #Check that all expected output files are there
        JustCheckExists(doPrintOutputNames=True).output(original_distributable)

        #Pickle the result to a file
        #logging.info("AAA\n\n\n\nABCwd='{0}'\n\nfile='{1}'DEF\n\n\nZZZ".format(os.getcwd(),self.output_file))
        if self.result_file is not None:
            util.create_directory_if_necessary(self.result_file)
            with open(self.result_file, mode='wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        return result

