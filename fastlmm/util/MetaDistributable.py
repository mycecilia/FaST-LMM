import os
import cPickle as pickle
import subprocess, sys, os.path
from fastlmm.util.runner import *
import logging

class MetaDistributable(object): #implements IDistributable

    def __init__(self, hadoop_mode, distributable_list=None):
        self.hadoop_mode = hadoop_mode
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
            if self.hadoop_mode:
                yield _NoOutputDistributable(distributable)
            else:
                yield distributable

    def reduce(self, result_and_file_and_contents_list):
        if self.hadoop_mode:
            result_list = []
            for original_result, file_and_contents_list in result_and_file_and_contents_list:
                result_list.append(original_result)
                for file_name, contents in file_and_contents_list:
                    with open(file_name, 'wb') as content_file:
                        content_file.write(contents)
            return result_list
        else:
            return list(result_and_file_and_contents_list)

    @property
    def tempdirectory(self):
        if len(self._distributable_list) == 0 : raise Exception("Expect at least one work item when tempdirectory is requested")
        return self._distributable_list[0].tempdirectory

    def __str__(self):
        str_list = []
        for distributable in self._distributable_list:
            str_list.append(str(distributable))
        return ",".join(str_list)

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

class _NoOutputDistributable(object): #implements IDistributable

    def __init__(self, distributable):
        self._distributable = distributable

    def copyinputs(self, copier):
        copier.input(self._distributable)

    def copyoutputs(self, copier):
        pass

     #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return self._distributable.work_count

    def work_sequence(self):
        return self._distributable.work_sequence()

    def reduce(self, result_sequence):
        original_result = self._distributable.reduce(result_sequence)
        file_name_list = []
        ListCopier([],file_name_list).output(self._distributable)
        file_and_contents_list = []
        for file_name in file_name_list:
            with open(file_name, 'rb') as content_file:
                contents = content_file.read()
                file_and_contents_list.append((file_name,contents))

        return original_result, file_and_contents_list
