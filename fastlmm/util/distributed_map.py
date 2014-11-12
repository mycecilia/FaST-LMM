#!/usr/bin/env python2.7
#
# Copyright (C) 2014 Microsoft Research

"""
Created on 2014-04-19
@summary: This module implements a minimally intrusive 
          parallel map on top of the IDistributable interface
"""

import logging
import distributed_map


class DistributedMap(object): #implements IDistributable
    """
    class to run distributed map using the idistributable back-end
    """


    def __init__(self, function, input_args, input_files=None, output_files=None):

        self.function = function
        self.input_args = input_args

        if input_files is None:
            self.input_files = []
        else:
            self.input_files = input_files

        if output_files is None:
            self.output_files = []
        else:
            self.output_files = output_files


#start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return len(self.input_args)

    def work_sequence(self):

        for i, input_arg in enumerate(self.input_args):
            logging.debug("executing %i" % i)
            yield lambda i=i, input_arg=input_arg: self.dowork(i, input_arg)
            # the 'i=i',etc is need to get around a strangeness in Python


    def reduce(self, result_sequence):
        '''
        '''

        return [r for r in result_sequence]


    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        return "{0}".format(self.function.__name__)
 #end of IDistributable interface---------------------------------------

    def dowork(self, i, input_args):
        #logging.info("{0}, {1}".format(len(train_snp_idx), len(test_snp_idx)))
        logging.debug("executing %s" % str(input_args))
        result = apply(self.function, [input_args])

        return result
   
    # required by IDistributable
    @property
    def tempdirectory(self):
        return ".work_directory.None"
        

    def copyinputs(self, copier):
        for fn in self.input_files:
            copier.input(fn)

    #Note that the files created are not automatically copied. Instead,
    # whenever we want another file to be created, a second change must be made here so that it will be copied.
    def copyoutputs(self,copier):
        for fn in self.output_files:
            copier.output(fn)



def d_map(f, args, runner, input_files=None, output_files=None):
    """interface for parallelizing embarrasibly parallel code
    
    Parameters
    ----------
    f : function 
        Function to be parallelized

    args : list of tuples
        List of arguments, one tuple for each call

    input_files : list of strings
        List of file names of input files

    output_files : list of strings
        List of file names of output files

    Returns
    -------
    list of return values (one for each input argument)
    """

    if output_files is not None:
        raise NotImplementedError("output files are not implemented yet")

    dist = distributed_map.DistributedMap(f, args, input_files, output_files)
    result = runner.run(dist)

    assert len(result) == len(args)

    return result


# example function to be "mapped"
def dummy(input_tuple):
    a, fn = input_tuple
    lines = ""
    for line in open(fn):
        lines += line.strip()
    return lines[0:a]
