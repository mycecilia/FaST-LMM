'''

Some examples of IDistributable and IRunner. Classes that implement IDistributable specify work to be done.
The class defined in this file, SamplePi, implements IDistributable to approximate  PI by simulating dart throws.

Classes that implement IRunner tell how to do that work. Examples of IRunner classes are Local, LocalMultiProc,
LocalRunInParts, and HPC.

Here are examples of each IRunner running a SamplePi job (which is defined below):


Run local in a single process

    >>> from fastlmm.util.SamplePi import *
    >>> round(Local().run(SamplePi(dartboard_count=100,dart_count=100)),2)
    pi ~ 3.162
    3.16


Run local on 12 processors (also, increase the # of dartboards and darts)

    >>> from fastlmm.util.SamplePi import *          #LocalMultiProc and HPC won't work without this 'from'
    >>> runner = LocalMultiProc(12,mkl_num_threads=1)
    >>> distributable = SamplePi(dartboard_count=1000,dart_count=1000,tempdirectory='pi_work')
    >>> round(runner.run(distributable),2)
    pi ~ 3.138856
    3.14


Behind the scenes LocalMultiProc and HPC call LocalInParts, but it can be called directly, too.

>>> from fastlmm.util.SamplePi import *
>>> distributable = SamplePi(dartboard_count=100,dart_count=100,tempdirectory='pi_work')
>>> LocalInParts(taskindex=0, taskcount=2, mkl_num_threads=4).run(distributable)               # do first half of work
>>> LocalInParts(taskindex=1, taskcount=2, mkl_num_threads=4).run(distributable)               # do second half of work
>>> round(LocalInParts(taskindex=2, taskcount=2, mkl_num_threads=4).run(distributable),2)      # tabulate the results
pi ~ 3.162
3.16


Here is an example of a cluster run.

#>>> from fastlmm.util.SamplePi import *
#>>> runner = HPC(10,'RR1-N13-16-H44',r'\\msr-arrays\scratch\msr-pool\eScience3')
#>>> distributable = SamplePi(dartboard_count=1000,dart_count=1000,tempdirectory='pi_work')
#>>> runner.run(distributable)

## no return value, but the last line of the reduce task's stdout says: pi ~ 3.138856



'''

from fastlmm.util.runner import *
import logging

class SamplePi(object) : #implements IDistributable
    '''
    Finds an approximation of pi by throwing  darts in a 2 x 2 square and seeing how many land within 1 of the center.
    '''
    #  dartboard_count is the number of work items
    def __init__(self,dartboard_count,dart_count,tempdirectory=None):
        self.dartboard_count = dartboard_count
        self.dart_count = dart_count
        self.__tempdirectory = tempdirectory

 #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return self.dartboard_count

    def work_sequence(self):
        for work_index in xrange(self.dartboard_count):
            yield lambda work_index=work_index : self.dowork(work_index)  # the 'work_index=work_index' is need to get around a strangeness in Python

    def reduce(self, result_sequence):
        '''
        result_sequence contains the sequence (that you can do a foreach on) of all results,
        where each item was create with the call inside the loop of work_sequence. Here, these
        results came from calls to dowork(). The order of results in the sequence is arbitary.
        '''
        average = float(sum(result_sequence)) / self.dartboard_count
        # the circle has area pi * r ** 2 = pi. the square has area 2**2=4, so the fraction_in_circle ~ pi /4
        pi = average * 4
        print("pi ~ {0}".format(pi))
        return pi

    @property
    def tempdirectory(self):
        return self.__tempdirectory

    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        return "{0}({1},{2})".format(self.__class__.__name__, self.dartboard_count, self.dart_count)
 #end of IDistributable interface---------------------------------------

    def dowork(self, work_index):
        '''
        This can return anything, but note that it will be binary serialized (pickleable), and you don't want to have more than is required there for reduce
        '''
        import scipy as sp
        from numpy.random import RandomState
        # seed the global random number generator with work_index xor'd with an arbitrary constant
        randomstate = RandomState(work_index ^ 284882)
        sum = 0.0
        for i in xrange(self.dart_count):
            x = randomstate.uniform(2)
            y = randomstate.uniform(2)
            is_in_circle = sp.sqrt((x-1)**2+(y-1)**2) < 1
            if is_in_circle:
                sum += 1
        fraction_in_circle = sum / self.dart_count
        return fraction_in_circle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
