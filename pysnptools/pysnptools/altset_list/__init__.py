'''
a altset_list is a list of snpsets


A altset_list is defined with two classes that implement these two interfaces: ISnpSetList and ISnpSetListPlusBed.
Note: Python doesn't know enforce interfaces.

interface ISnpSetList
    def addbed(self, bed):
        return # ISnpSetListPlusBed

interface ISnpSetListPlusBed:
    def __len__(self):
        return # number of snpsets in this list

    def __iter__(self):
        return # a sequence of ISnpSetPlusBed's

'''
from .snpandsetnamecollection import *
from .subset import *
from .minmaxsetsize import *
from .consecutive import *
#!! ISnpSetList Later replace all interfaces with abstract classes
#!!remove all these __init__.py imports?