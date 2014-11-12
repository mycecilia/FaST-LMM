import os
import cPickle as pickle
import subprocess, sys, os.path
from fastlmm.util.runner import *
import logging

class IndirectDistributable(object): #implements IDistributable

    def __init__(self, indirect_runner, distributable_list=None):
        self.indirect_runner = indirect_runner
        if distributable_list is None:
            distributable_list = []

        self._distributable_list = distributable_list

    def append(self, distributable):
        self._distributable_list.append(distributable)

    def copyinputs(self, copier):
        for distributable in self._distributable_list:
            copier.input(distributable)

    def copyoutputs(self, copier):
        for distributable in self._distributable_list:
            copier.output(distributable)

     #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return len(self._distributable_list)

    def work_sequence(self):
        for distributable in self._distributable_list:
                yield lambda distributable=distributable : self.do_work(distributable)

    def do_work(self, distributable):
        result = self.indirect_runner.run(distributable)
        return result

    def reduce(self, result_list):
        return list(result_list)

    @property
    def tempdirectory(self):
        if len(self._distributable_list) == 0 : raise Exception("Expect at least one work item when tempdirectory is requested")
        return self._distributable_list[0].tempdirectory

    def __str__(self):
        str_list = []
        for distributable in self._distributable_list:
            str_list.append(str(distributable))
        return "IndirectDistributable({0})".format(",".join(str_list))

     #end of IDistributable interface---------------------------------------

    def __repr__(self):
        import cStringIO
        fp = cStringIO.StringIO()
        fp.write("{0}(\n".format(self.__class__.__name__))
        varlist = []
        for f in dir(self):
            if f.startswith("_"): # remove items that start with '_'
                continue
            if type(self.__class__.__dict__.get(f,None)) is property: # remove @properties
                continue
            if callable(getattr(self, f)): # remove methods
                continue
            varlist.append(f)
        for var in varlist[:-1]: #all but last
            fp.write("\t{0}={1},\n".format(var, getattr(self, var).__repr__()))
        var = varlist[-1] # last
        fp.write("\t{0}={1})\n".format(var, getattr(self, var).__repr__()))
        result = fp.getvalue()
        fp.close()
        return result

