import re
from bisect import bisect_left
import logging
import numpy as np
import unittest
import doctest

class IntRangeSet(object):
    '''
    IntRangeSet is a class for manipulating ranges of integers (including longs) as sets. For example,
    here we take the union of two IntRangeSets:

    >>> print IntRangeSet("100-499,500-1000") | IntRangeSet("-20,400-600")
    IntRangeSet('-20,100-1000')

    '''

    _rangeExpression = re.compile(r"^(?P<start>-?\d+)(-(?P<last>-?\d+))?$")

    def __init__(self, *ranges_inputs):
        #!!! confirm that this appears in docs
        '''
        Create a IntRangeSet from zero or more ranges input.
       
        :Example:

        >>> print IntRangeSet() #create from zero ranges input
        IntRangeSet('')

        >>> print IntRangeSet('-10-100,200-500') #create from one ranges input
        IntRangeSet('-10-100,200-500')

        >>> print IntRangeSet('-10-100,200-500', '150') #create from two ranges inputs
        IntRangeSet('-10-100,150,200-500')

        A ranges input can be an integer. Here we have four integer inputs:

        >>> print IntRangeSet(5,6,3+4,-10)
        IntRangeSet('-10,5-7')

        A ranges input can be a string.

        >>> print IntRangeSet('-10--3,5,-30--9')
        IntRangeSet('-30--3,5')

        A ranges input can be an iteration of ranges inputs.

        >>> print IntRangeSet([3,4,5,7,'-10--3'])
        IntRangeSet('-10--3,3-5,7')

        >>> print IntRangeSet(xrange(0,100))
        IntRangeSet('0-99')

        A ranges input can be a tuple of exactly two items: a start integer and a last integer.
        Note that the LAST integer is INCLUSIVE. This differs from the STOP value common in other
        Python libraries that are exclusive.

        >>> print IntRangeSet((-10,-3)) #Go from -10 (inclusive) to -3 (inclusive)
        IntRangeSet('-10--3')
        >>> print IntRangeSet([(-10,-3),(2,5)]) # A list of tuples.
        IntRangeSet('-10--3,2-5')

        A ranges input can be a IntRangeSet (or anything with a .ranges() iterator):

        >>> print IntRangeSet(IntRangeSet("3-10"))
        IntRangeSet('3-10')

        A ranges input can be a slice:

        >>> import numpy as np
        >>> print IntRangeSet(np.s_[0:100],np.s_[75:500:100]) #Two ranges ranges, each one a slice
        IntRangeSet('0-99,175,275,375,475')

        The parts of a ranges input can be in any order and may overlap

        >>> print IntRangeSet('12-15,11-12,5')
        IntRangeSet('5,11-15')

        Negatives integers and large integers are fine

        >>> print IntRangeSet('-16000000000-20000000000,-17000000001')
        IntRangeSet('-17000000001,-16000000000-20000000000')
        '''
        if len(ranges_inputs) > 0 and isinstance(ranges_inputs[0],IntRangeSet): #Because we know self is empty, optimize for the case in which the first item is a IntRangeSet
            self._start_items = list(ranges_inputs[0]._start_items)
            self._start_to_length = dict(ranges_inputs[0]._start_to_length)
            ranges_inputs = ranges_inputs[1:]
        else:
            self._start_items = []
            self._start_to_length = {}
        self.add(*ranges_inputs) 

    def add(self, *ranges_inputs):
        '''
        Union zero or more ranges inputs into the current IntRangeSet.

        These are the same:
        
        * ``a |= b``
        * ``a += b``
        * ``a.add(b)``
        * ``a.update(b)``

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a |= 5
        >>> print a
        IntRangeSet('0-10')


        The 'add' and 'update' methods also support unioning multiple ranges inputs,

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a.add('5','100-200')
        >>> print a
        IntRangeSet('0-10,100-200')
        '''

        #!!consider special casing the add of a single int. Anything else?
        for start,last in IntRangeSet._static_ranges(*ranges_inputs):
            self._internal_add(start, last-start+1)
    #update(other, ...)set |= other | ...
    #Update the set, adding elements from all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __iadd__(self, *ranges_inputs):
        self.add(*ranges_inputs)
        return self
    def __ior__(self, *ranges_inputs):
        self.add(*ranges_inputs)
        return self
    def update(self, *ranges_inputs):
       self.__iadd__(*ranges_inputs)

    def copy(self):
        '''
        Create a deep copy of a IntRangeSet.
        '''
        return IntRangeSet(self)

    def ranges(self):
        '''
        Iterate, in order, the ranges of a IntRangeSet as (start,last) tuples.

        :Example:

        >>> for start,last in IntRangeSet('0-10,100-200').ranges():
        ...       print "start is {0}, last is {1}".format(start,last)
        start is 0, last is 10
        start is 100, last is 200

        '''
        for item in self._start_items:
            last = item + self._start_to_length[item] - 1
            yield item, last

    def __iter__(self):
        #!!! be sure this appears in the documentation
        '''
        Iterate, in order from smallest to largest, the integer elements of the IntRangeSet

        :Example:

        >>> for i in IntRangeSet('1-3,10'):
        ...    print i
        1
        2
        3
        10

        '''
        for (first, last) in self.ranges():
            for i in xrange(first,last+1):
                yield i

    def clear(self):
        '''
        Remove all ranges from this IntRangeSet.

        >>> a = IntRangeSet('0-9,12')
        >>> a.clear()
        >>> print a
        IntRangeSet('')

        '''
        del self._start_items[:]
        self._start_to_length.clear()

    def __len__(self):
        #!!! be sure this appears in the documentation
        '''
        The number of integer elements in the IntRangeSet

        >>> print len(IntRangeSet('0-9,12'))
        11

        Note: This is computed in time linear in the number of ranges, rather than integer elements.

        '''
        return sum(self._start_to_length.values())


    @property
    def ranges_len(self):
        '''
        The number of contiguous ranges in the IntRangeSet

        >>> print IntRangeSet('0-9,12').ranges_len
        2

        '''
        return len(self._start_items)


    def sum(self):
        '''
        The sum of the integer elements in the IntRangeSet

        >>> print IntRangeSet('0-9,12').sum()
        57

        Note: This is more efficient than ``sum(IntRangeSet('0-9,12'))`` because is computed
        in time linear in the number of ranges, rather than integer elements.
        '''
        result = 0
        for start in self._start_items:
            length = self._start_to_length[start]
            result += (start + start + length - 1)*length//2
        return result

    def __eq__(self, other):
        #!!! be sure this appears in the documentation
        '''
        True exacty when the IntRangeSet on the left is set equivalent to the ranges input on the right.

        >>> print IntRangeSet('0-9,12') == IntRangeSet('0-9,12')
        True
        >>> print IntRangeSet('0-9,12') == IntRangeSet('0-9')
        False
        >>> print IntRangeSet('0-9,12') == IntRangeSet('12,0-4,5-9')
        True
        >>> print IntRangeSet('0-9,12') == '0-9,12' # The right-hand can be any ranges input
        True
        '''
        self, other = IntRangeSet._make_args_range_set(self, other)
        if other is None or len(self._start_items)!=len(other._start_items):
            return False
        for i, start in enumerate(self._start_items):
            if start != other._start_items[i] or self._start_to_length[start] != other._start_to_length[start]:
                return False
        return True

    def __ne__(self, other):
        #!!! be sure this appears in the documentation
        '''
        a != b
        is the same as
        not a == b
        '''
        #Don't need to call _make_args_range_set because __eq__ will call it
        return not self==other

    #Same: a >= b, a.issuperset(b,...), b in a
    #Returns True iff item is within the ranges of this IntRangeSet.
    def __contains__(self, *ranges_inputs):
        #!!! be sure this appears in the documentation
        '''
        True exactly when all the ranges input is a subset of the IntRangeSet.

        These are the same:

        * ``b in a``
        * ``a >= b``
        * ``a.issuperset(b)``

        :Example:

        >>> print 3 in IntRangeSet('0-4,6-10')
        True
        >>> print IntRangeSet('4-6') in IntRangeSet('0-4,6-10')
        False
        >>> '6-8' in IntRangeSet('0-4,6-10') # The left-hand of 'in' can be any ranges input
        True
        >>> print IntRangeSet('0-4,6-10') >= '6-8' # The right-hand of can be any ranges input
        True

        The 'issuperset' method also supports unioning multiple ranges inputs.

        :Example:

        >>> print IntRangeSet('0-4,6-10').issuperset(4,7,8)
        True
        >>> print IntRangeSet('0-4,6-10').issuperset(4,7,8,100)
        False

        Note: By definition, any set is a superset of itself.
        '''
        for start_in,last_in in IntRangeSet._static_ranges(*ranges_inputs):
            start_self,length_self,index,contains = self._best_start_length_index_contains(start_in)
            if not contains or last_in > start_self+length_self-1:
                return False
        return True
    #issuperset(other)set >= other
    #Test whether every element in other is in the set.
    def __ge__(self,other):
        return other in self
    #!!! check that the documentation for all of these is OK
    issuperset = __contains__

    
    @property
    def isempty(self):
        '''
        True exactly when the IntRangeSet is empty.

        >>> print IntRangeSet().isempty
        True
        >>> print IntRangeSet(4).isempty
        False
        '''
        return len(self._start_items) == 0

    def __str__(self):
        #!!! be sure this appears in the documentation
        '''
        Use the standard str(a) function to create a string representation of a, an IntRangeSet.

        >>> print "Hello " + str(IntRangeSet(2,3,4,10))
        Hello IntRangeSet('2-4,10')
        '''
        return repr(self)


    def __repr__(self):
        #!!! be sure this appears in the documentation
        '''
        Use the standard repr(a) function to create a string representation of a, an IntRangeSet.

        >>> print "Hello " + repr(IntRangeSet(2,3,4,10))
        Hello IntRangeSet('2-4,10')
        '''
        return "IntRangeSet('{0}')".format(self._repr_internal("-", ","))

    def _repr_internal(self, seperator1, separator2):
        if self.isempty:
            return ""

        from cStringIO import StringIO
        fp = StringIO()

        for index, (start, last) in enumerate(self.ranges()):
            if index > 0:
                fp.write(separator2)

            if start == last:
                fp.write(str(start))
            else:
                fp.write("{0}{1}{2}".format(start, seperator1, last))
        return fp.getvalue()

    @staticmethod
    def _test():
        int_range_set = IntRangeSet()
        int_range_set.add(0)
        assert "IntRangeSet('0')" == str(int_range_set)
        int_range_set.add(1)
        assert "IntRangeSet('0-1')" == str(int_range_set)
        int_range_set.add(4)
        assert "IntRangeSet('0-1,4')" == str(int_range_set)
        int_range_set.add(5)
        assert "IntRangeSet('0-1,4-5')" == str(int_range_set)
        int_range_set.add(7)
        assert "IntRangeSet('0-1,4-5,7')" == str(int_range_set)
        int_range_set.add(2)
        assert "IntRangeSet('0-2,4-5,7')" == str(int_range_set)
        int_range_set.add(3)
        assert "IntRangeSet('0-5,7')" == str(int_range_set)
        int_range_set.add(6)
        assert "IntRangeSet('0-7')" == str(int_range_set)
        int_range_set.add(-10)
        assert "IntRangeSet('-10,0-7')" == str(int_range_set)
        int_range_set.add(-5)
        assert "IntRangeSet('-10,-5,0-7')" == str(int_range_set)

        assert IntRangeSet("-10--5") == "-10--5"
        assert IntRangeSet("-10--5,-3") == "-10--5,-3"
        assert IntRangeSet("-10--5,-3,-2-1") == "-10--5,-3-1"
        assert IntRangeSet("-10--5,-3,-2-1,1-5") == "-10--5,-3-5"
        assert IntRangeSet("-10--5,-3,-2-1,1-5,7-12") == "-10--5,-3-5,7-12"
        assert IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15") == "-10--5,-3-5,7-15"
        assert IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16") == "-10--5,-3-5,7-16"
        assert IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25") == "-10--5,-3-5,7-16,20-25"
        assert IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25,22-23") == "-10--5,-3-5,7-16,20-25"

        range = "-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25,22-23"
        int_range_set = IntRangeSet(range)
        assert int_range_set == "-10--5,-3-5,7-16,20-25"

        range = "1-5,0,4-10,-10--5,-12--3,15-20,12-21,-13"
        int_range_set = IntRangeSet(range)
        assert int_range_set == "-13--3,0-10,12-21"

        assert len(int_range_set) == 32

        int_range_set1 = IntRangeSet("-10--5")
        int_range_set2 = int_range_set1.copy()
        assert int_range_set1 is not int_range_set2
        assert int_range_set1 == int_range_set2
        int_range_set2.add(7)
        assert int_range_set1 != int_range_set2

        assert str(IntRangeSet(7)) == "IntRangeSet('7')"
        assert str(IntRangeSet((7,7))) == "IntRangeSet('7')"
        assert str(IntRangeSet((7,10))) == "IntRangeSet('7-10')"
        assert str(IntRangeSet(xrange(7,11))) == "IntRangeSet('7-10')"
        assert str(IntRangeSet(np.s_[7:11])) == "IntRangeSet('7-10')"
        assert str(IntRangeSet(np.s_[7:11:2])) == "IntRangeSet('7,9')"
        assert str(IntRangeSet(xrange(7,11,2))) == "IntRangeSet('7,9')"
        assert str(IntRangeSet(None)) == "IntRangeSet('')"
        assert str(IntRangeSet()) == "IntRangeSet('')"
        assert [e for e in IntRangeSet("-10--5,-3")] == [-10,-9,-8,-7,-6,-5,-3]
        int_range_set3 = IntRangeSet(7,10)
        int_range_set3.clear()
        assert str(int_range_set3) == "IntRangeSet('')" 
        assert len(IntRangeSet("-10--5,-3")) == 7

        int_range_set4 = IntRangeSet("-10--5,-3")
        int_range_set4.add(-10,-7)
        assert int_range_set4 == "-10--5,-3"
        int_range_set4.add(-10,-4)
        assert int_range_set4 == "-10--3"

        int_range_set5 = IntRangeSet("-10--5,-3")
        assert -11 not in int_range_set5
        assert -10 in int_range_set5
        assert -5 in int_range_set5
        assert -4 not in int_range_set5
        assert -3 in int_range_set5
        assert -2 not in int_range_set5
        assert 19999 not in int_range_set5
        assert "-11" not in int_range_set5
        assert "-10" in int_range_set5
        assert "-10--7" in int_range_set5
        assert "-10--4" not in int_range_set5
        assert "-10--7,-3" in int_range_set5
        assert "-10--7,-3,100" not in int_range_set5
        assert IntRangeSet("-11") not in int_range_set5
        assert IntRangeSet("-10") in int_range_set5
        assert IntRangeSet("-10--7") in int_range_set5
        assert IntRangeSet("-10--4") not in int_range_set5
        assert IntRangeSet("-10--7,-3") in int_range_set5
        assert IntRangeSet("-10--7,-3,100") not in int_range_set5
        assert [-11] not in int_range_set5
        assert [-10] in int_range_set5
        assert xrange(-10,-6) in int_range_set5
        assert xrange(-10,-3) not in int_range_set5
        assert [-10,-9,-8,-7,-3] in int_range_set5
        assert [-10,-9,-8,-7,-3,-100] not in int_range_set5

        assert IntRangeSet("-11,-10,-9") == IntRangeSet("-11--9")
        a = IntRangeSet("-11,-10")
        b = IntRangeSet("-11--9")
        assert a!=b
        assert IntRangeSet("-11,1") != IntRangeSet("-11-1")

        assert IntRangeSet("1-3") | IntRangeSet("2,4,6") == IntRangeSet("1-4,6")
        assert IntRangeSet("1-3").union("2,4,6","1,6,-100") == IntRangeSet("-100,1-4,6")


        assert IntRangeSet() & IntRangeSet() == IntRangeSet()
        assert IntRangeSet().intersection("","") == IntRangeSet()
        assert IntRangeSet() & IntRangeSet() & IntRangeSet() == IntRangeSet()

        assert IntRangeSet("1") & IntRangeSet() == IntRangeSet()
        assert IntRangeSet() & IntRangeSet("1") == IntRangeSet()
        assert IntRangeSet("1-5") & IntRangeSet() == IntRangeSet()
        assert IntRangeSet() & IntRangeSet("1-5") == IntRangeSet()
        assert IntRangeSet("1-5,7") & IntRangeSet() == IntRangeSet()
        assert IntRangeSet() & IntRangeSet("1-5,7") == IntRangeSet()

        assert IntRangeSet("1") & IntRangeSet("1") == IntRangeSet("1")
        assert IntRangeSet("1-5") & IntRangeSet("1") == IntRangeSet("1")
        assert IntRangeSet("1") & IntRangeSet("1-5") == IntRangeSet("1")
        assert IntRangeSet("1-5,7") & IntRangeSet("1") == IntRangeSet("1")

        assert IntRangeSet("2") & IntRangeSet("1-5,7") == IntRangeSet("2")
        assert IntRangeSet("1-5") & IntRangeSet("2") == IntRangeSet("2")
        assert IntRangeSet("2") & IntRangeSet("1-5") == IntRangeSet("2")
        assert IntRangeSet("1-5,7") & IntRangeSet("2") == IntRangeSet("2")


        assert IntRangeSet("-2") & IntRangeSet("1-5,7") == IntRangeSet()
        assert IntRangeSet("1-5") & IntRangeSet("-2") == IntRangeSet()
        assert IntRangeSet("-2") & IntRangeSet("1-5") == IntRangeSet()
        assert IntRangeSet("1-5,7") & IntRangeSet("-2") == IntRangeSet()

        assert IntRangeSet("22") & IntRangeSet("1-5,7") == IntRangeSet()
        assert IntRangeSet("1-5") & IntRangeSet("22") == IntRangeSet()
        assert IntRangeSet("22") & IntRangeSet("1-5") == IntRangeSet()
        assert IntRangeSet("1-5,7") & IntRangeSet("22") == IntRangeSet()


        assert IntRangeSet("-2,1-3,20,25-99,101") & IntRangeSet("1-100") == IntRangeSet("1-3,20,25-99")
        assert IntRangeSet("2-3,90-110") & IntRangeSet("1-100") == IntRangeSet("2-3,90-100")
        assert IntRangeSet("1-100") & IntRangeSet("-2,1-3,20,25-99,101") == IntRangeSet("1-3,20,25-99")
        assert IntRangeSet("1-100") & IntRangeSet("2-3,90-110") == IntRangeSet("2-3,90-100")

        assert IntRangeSet("0-66") - "30-100" == IntRangeSet("0-29")
        assert IntRangeSet("0-29,51-66").difference("40-100") == IntRangeSet("0-29")
        assert IntRangeSet("0-66").difference("30-50","40-100") == IntRangeSet("0-29")
        assert IntRangeSet("0-66,100,200") - "30-100,300" == "0-29,200"
        assert IntRangeSet("30-100") - "0-66" == "67-100"
        assert IntRangeSet("30-100,300")-IntRangeSet("0-66,100,200") == IntRangeSet("67-99,300")

        assert IntRangeSet("30-100,300")^IntRangeSet("0-66,100,200") == IntRangeSet("0-29,67-99,200,300")
        assert IntRangeSet([1,2]).symmetric_difference([2,3]) == IntRangeSet([1,3])

        assert IntRangeSet("10-14,55-59")[0:5] == IntRangeSet("10-14")
        assert IntRangeSet("10-14,55-59")[1:5] == IntRangeSet("11-14")
        assert IntRangeSet("10-14,55-59")[9:10] == IntRangeSet("59")
        assert IntRangeSet("10-14,55-59")[-2:] == IntRangeSet("58-59")
        assert IntRangeSet("10-14,55-59")[0:5:2] == IntRangeSet("10,12,14")

        try:
            IntRangeSet("10-14,55-59")[100]
        except KeyError:
            pass
        try:
            IntRangeSet("10-14,55-59")[-100]
        except KeyError:
            pass

        assert IntRangeSet("10-14,55-59").max() == 59
        assert IntRangeSet("10-14,55-59").min() == 10

        assert IntRangeSet("10-14,55-59") + IntRangeSet("58-60,100") == IntRangeSet("10-14,55-60,100")
        assert IntRangeSet("10-14,55-59") + "58-60,100" == IntRangeSet("10-14,55-60,100")

        mult_test0 = IntRangeSet("10-14,55-59")
        mult_test1 = mult_test0 * 3
        assert mult_test0 == mult_test1
        assert mult_test0 is not mult_test1
        assert (mult_test0 * 0).isempty
        assert (mult_test0 * -1000).isempty

        assert IntRangeSet("10-14,55-59").index(10) == 0
        assert IntRangeSet("10-14,55-59").index(11) == 1
        assert IntRangeSet("10-14,55-59").index(55) == 5
        assert IntRangeSet("10-14,55-59").index(56) == 6
        try:
            IntRangeSet("10-14,55-59").index(9)
        except IndexError:
            pass
        try:
            IntRangeSet("10-14,55-59").index(15)
        except IndexError:
            pass

        assert IntRangeSet("10-14,55-59").index("57,56-56") == '6-7' #returns the index of the start of the contiguous place where 57,"56-56" occurs
        assert IntRangeSet("10-14,55-59").index([10,55]) == '0,5'
        try:
            IntRangeSet("10-14,55-59").index("100-110")
        except IndexError:
            pass
        try:
            IntRangeSet("10-14,55-59").index("14-16")
        except IndexError:
            pass



        try:
            IntRangeSet(3.34)
        except Exception:
            pass
        try:
            3.34 in IntRangeSet()
        except Exception:
            pass

        assert IntRangeSet("10-14,55-59").count(10) == 1
        assert IntRangeSet("10-14,55-59").count(100) == 0

        assert IntRangeSet("10-14,55-59").isdisjoint(100)
        assert not IntRangeSet("10-14,55-59").isdisjoint("57-100")

        assert IntRangeSet("10-14,55-59") <= "10-14,55-59"
        assert IntRangeSet("10-14,55-59").issubset("10-14,55-59")
        assert not IntRangeSet("10-14,55-59") < "10-14,55-59"
        assert IntRangeSet("10-14,55-59") < "9-14,55-59"

        assert IntRangeSet("10-14,55-59") >= "10-14,55-59"
        assert IntRangeSet("10-14,55-59").issuperset("10-14,55-59")
        assert not IntRangeSet("10-14,55-59") > "10-14,55-59"
        assert IntRangeSet("9-14,55-59") > "10-14,55-59"

        update0 = IntRangeSet("9-14,55-59")
        update0.update("10,30,100","30")
        assert update0 == IntRangeSet("9-14,30,55-59,100")

        assert IntRangeSet() != None

        update0 = IntRangeSet()
        update0 |= []
        assert update0 == IntRangeSet()

        update0 = IntRangeSet("9-14,55-59")
        update0.intersection_update("10,100","0-100")
        assert update0 == IntRangeSet("10")

        update0 = IntRangeSet("9-14,55-59")
        update0 |= IntRangeSet("10,30,100") | "30"
        assert update0 == IntRangeSet("9-14,30,55-59,100")

        update0 = IntRangeSet("9-14,55-59")
        update0 += IntRangeSet("10,30,100") + "30"
        assert update0 == IntRangeSet("9-14,30,55-59,100")


        update0 = IntRangeSet("9-14,55-59")
        update0 &= IntRangeSet("10,100") & "0-100"
        assert update0 == IntRangeSet("10")

        update0 = IntRangeSet("9-14,55-59")
        update0 -= IntRangeSet("10,100") - "30"
        assert update0 == IntRangeSet("9,11-14,55-59")

        update0 = IntRangeSet("9-14,55-59")
        update0.difference_update("10,100","30")
        assert update0 == IntRangeSet("9,11-14,55-59")


        update0 = IntRangeSet("9-14,55-59")
        update0.symmetric_difference_update("10,100,30")
        assert update0 == IntRangeSet("9,11-14,30,55-59,100")


        update0 = IntRangeSet("9-14,55-59")
        update0 ^= IntRangeSet("10,100,30")
        assert update0 == IntRangeSet("9,11-14,30,55-59,100")

        remove0 = IntRangeSet("9-14,55-59")
        remove0.remove(9)
        assert remove0 == IntRangeSet("10-14,55-59")

        remove0 = IntRangeSet("9-14,55-59")
        remove0.remove(10,13)
        assert remove0 == IntRangeSet("9,11-12,14,55-59")

        remove0 = IntRangeSet("9-14,55-59")
        try:
            remove0.remove("100")
        except KeyError:
            pass

        remove0 = IntRangeSet("9-14,55-59")
        try:
            remove0.remove(IntRangeSet("100"),101)
        except Exception:
            pass

        discard0 = IntRangeSet("9-14,55-59")
        discard0.discard(9)
        assert discard0 == IntRangeSet("10-14,55-59")

        discard0 = IntRangeSet("9-14,55-59")
        discard0.discard(10,13)
        assert discard0 == IntRangeSet("9,11-12,14,55-59")

        discard0 = IntRangeSet("9-14,55-59")
        discard0.discard("100")
        assert discard0 == "9-14,55-59"

        discard0 = IntRangeSet("9-14,55-59")
        discard0.discard(IntRangeSet("100"),101)
        assert discard0 == "9-14,55-59"

        pop0 = IntRangeSet("9-14,55-59")
        assert pop0.pop() == 59
        assert pop0 == "9-14,55-58"

        pop0 = IntRangeSet([1,3])
        assert pop0.pop() == 3
        assert pop0.pop() == 1
        try:
            pop0.pop()
        except KeyError:
            pass

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[0:5]
        delitem0 == "55-59"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[1:5]
        delitem0 == "10,55-59"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[9:10]
        delitem0 == "10-14,55-58"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[-2:]
        delitem0 == "10-14,55-57"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[0:5:2]
        delitem0 == "11,33,55-59"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[-2]
        delitem0 == "10-14,55-57,59"

        delitem0 = IntRangeSet("10-14,55-59")
        del delitem0[3]
        delitem0 == "10-12,14,55-59"

        delitem0 = IntRangeSet("10-14,55-59")
        try:
            del delitem0[100]
        except KeyError:
            pass
        try:
            del delitem0[-100]
        except KeyError:
            pass

        assert list(reversed(IntRangeSet("10-14,55-59"))) == list(reversed(list(IntRangeSet("10-14,55-59"))))

        IntRangeSet("10-14,55-59").sum() == sum(IntRangeSet("10-14,55-59"))

        try:
            IntRangeSet("10:14")
        except Exception:
            pass

        add0 = IntRangeSet("1,12-14,55-60,71,102")
        add0.add("12-100")
        assert add0 == "1,12-100,102"

        assert IntRangeSet("1,12-14,55-60,71,102") - "5-71" == "1,102"
        assert IntRangeSet("1,12-14,55-60,71,102") - "12-65" == "1,71,102"
        assert IntRangeSet("1,12-14,55-60,71,102") - "13-56" == "1,12,57-60,71,102"

        a = IntRangeSet('100-200,1000')
        del a['2-10']
        assert a == '100-101,111-200,1000'

        assert IntRangeSet('0-4,6-10') - '3-100' == '0-2'


    #s[i] ith item of s, origin 0 (3) 
    #s[i:j] slice of s from i to j (3)(4) 
    #s[i:j:k] slice of s from i to j with step k (3)(5) 
    def __getitem__(self, key):
        #!!! be sure this appears in the documentation
        '''
        ``a[i]`` returns the ith integer in sorted order (origin 0) from a, an IntRangeSet

        >>> print IntRangeSet('100-200,1000')[0]
        100
        >>> print IntRangeSet('100-200,1000')[10]
        110

        If i is negative, the indexing goes from the end

        >>> print IntRangeSet('100-200,1000')[-1]
        1000

        Python's standard slice notation may be used and returns IntRangeSets.
        (Remember that the Stop number in slice notation is exclusive.)

        >>> print IntRangeSet('100-200,1000')[0:11] # Integers 0 (inclusive) to 11 (exclusive)
        IntRangeSet('100-110')

        >>> print IntRangeSet('100-200,1000')[0:11:2] # Integers 0 (inclusive) to 11 (exclusive) with step 2
        IntRangeSet('100,102,104,106,108,110')

        >>> print IntRangeSet('100-200,1000')[-3:] # The last three integers in the IntRangeSet.
        IntRangeSet('199-200,1000')

        An IntRangeSet can also be accessed with any ranges input.

        >>> IntRangeSet('100-200,1000')['0-10,20']
        IntRangeSet('100-110,120')
        '''
        if isinstance(key,(int,long)):
            if key >= 0:
                for start in self._start_items:
                    length = self._start_to_length[start]
                    if key < length:
                        return start+key
                    key -= length
                raise KeyError()
            else:
                assert key < 0
                key = -key-1
                for start_index in xrange(len(self._start_items)):
                    start = self._start_items[-1-start_index]
                    length = self._start_to_length[start]
                    if key < length:
                        return start+length-1-key
                    key -= length
                raise KeyError()
        elif isinstance(key, slice):
            lenx = len(self)
            start_index,stop_index,step_index = key.start,key.stop,key.step
            start_index = start_index or 0
            stop_index = stop_index or lenx
            step_index = step_index or 1

            if step_index == 1:
                return self & (self[start_index],self[stop_index-1])
            else:
                return IntRangeSet(self[index] for index in xrange(*key.indices(lenx)))
        else:
            start_and_last_generator = (self._two_index(start_index,last_index) for start_index,last_index in IntRangeSet._static_ranges(key))
            return self.intersection(start_and_last_generator)
            

    #max(s) largest item of s   
    def max(self):
        '''
        The largest integer element in the IntRangeSet

        >>> print IntRangeSet('0-9,12').max()
        12

        Note: This is more efficient than max(IntRangeSet('0-9,12')) because is computed
        in constant time rather than in time linear to the number of integer elements.
        '''
        start = self._start_items[-1]
        return start + self._start_to_length[start] - 1

    #min(s) smallest item of s   
    def min(self):
        '''
        The smallest integer element in the IntRangeSet

        :Example:

        >>> print IntRangeSet('0-9,12').min()
        0

        Note: This is more efficient than ``min(IntRangeSet('0-9,12'))`` because is computed
        in constant time rather than in time linear to the number of integer elements.
        '''
        return self._start_items[0]


    def _make_args_range_set(*args):
        for arg in args:
            if arg is None:
                yield None
            elif isinstance(arg,IntRangeSet):
                yield arg
            else:
                yield IntRangeSet(arg)

    #same: a.union(b,...), a+b, a|b
    def __concat__(*ranges_inputs):
        '''
        Return the union of a IntRangeSet with zero or more ranges inputs. The original IntRangeSet is not changed.

        These are the same:
        
        * ``a | b``
        * ``a + b``
        * ``a.union(b)``

        :Example:

        >>> print IntRangeSet('0-4,6-10') | 5
        IntRangeSet('0-10')

        The 'union' method also support unioning multiple ranges inputs,

        :Example:

        >>> print IntRangeSet('0-4,6-10').union(5,'100-200')
        IntRangeSet('0-10,100-200')
        '''
        result = IntRangeSet()
        result.add(*ranges_inputs)
        return result
    #s + t the concatenation of s and t (6) 
    __add__ = __concat__ #!!!expand all these out
        #union(other, ...)set | other | ...
    #Return a new set with elements from the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __or__(self, other):
        return self+other
    union = __concat__


    #s * n, n shallow copies of s concatenated (2) 
    def __mul__(self, n):
        #!!! be sure this appears in the documentation
        '''
        ``a * n``, produces n shallow copies of a unioned, where a is an IntRangeSet.
        Because a is a set, the result will either be an empty IntRangeSet (n is 0 or less) or a copy of
        the original IntRangeSet.
        '''
        if n<=0:
            return IntRangeSet()
        else:
            return IntRangeSet(self)

    #s.index(x) index of the x in s
    def index(self, other):
        '''
        ``a.index(x)``, index of the integer element x in a, an IntRangeSet. Raises an IndexError is x not in a.

        >>> print IntRangeSet('100-110,1000').index(110)
        10

        x also can be any ranges input, in which case, an IntRangeSet is returned containing the indexes of all integers in x.

        >>> print IntRangeSet('100-110,1000').index('110,100-102')
        IntRangeSet('0-2,10')
        '''
        if isinstance(other,(int,long)):
            return self._index_element(other)
        else:
            #If start and last are the same, only call _index_element once
            start_and_last_index_generator = ((self._index_element(start),self._index_element(last)) for start,last in IntRangeSet._static_ranges(other))
            return IntRangeSet(start_and_last_index_generator)

    def _index_element(self, element):
        index = bisect_left(self._start_items, element)

        # Find the position_in_its_range of this element
        if index != len(self._start_items) and self._start_items[index] == element: #element is start value
            position_in_its_range = 0
        elif index == 0:   # item is before any of the ranges
            raise IndexError()
        else:
            index -= 1
            start = self._start_items[index]
            last = start+self._start_to_length[start]-1
            if element > last: # we already know it's greater than start
                raise IndexError()
            else:
                position_in_its_range = element - start

        # Sum up the length of all preceding ranges
        preceeding_starts = self._start_items[0:index]
        preceeding_lengths = (self._start_to_length[start] for start in preceeding_starts)
        result = position_in_its_range + sum(preceeding_lengths)
        return result

    #s.count(x) total number of occurrences of x in s   
    def count(self, ranges):
        '''
        The number of times that the elements of ranges appears in the IntRangeSet. Because IntRangeSet is 
        a set, the number will be either 0 or 1.

        >>> print IntRangeSet('100-110,1000').count('105-107,1000')
        1
        '''
        if ranges in self:
            return 1
        else:
            return 0

    #Return True if the set has no elements in common with other. Sets are disjoint if and only if their intersection is the empty set.
    def isdisjoint(self, ranges):
        '''
        True exactly when the two sets have no integer elements in common.

        :Example:

        >>> print IntRangeSet('100-110,1000').isdisjoint('900-2000')
        False
        >>> print IntRangeSet('100-110,1000').isdisjoint('1900-2000')
        True
        '''
        isempty_generator = (IntRangeSet(tuple)._binary_intersection(self).isempty for tuple in IntRangeSet._static_ranges(ranges))
        return all(isempty_generator)

    #Same: a <= b, a.issubset(b)
    #issubset(other)set <= other
    #Test whether every element in the set is in other.
    def __le__(self, ranges):
        #!!! be sure this appears in the documentation
        '''
        True exactly when the IntRangeSet is a subset of the ranges.

        These are the same:
        
        * ``a <= b``
        * ``a.issubset(b)``

        :Example:

        >>> print IntRangeSet('0-4,6-10') <= '-1-100' # The right-hand can be any ranges input
        True

        Note: By definition, any set is a subset of itself.
        '''
        self, ranges = IntRangeSet._make_args_range_set(self, ranges)
        return self in ranges
    issubset = __le__

    #set < other
    #Test whether the set is a proper subset of other, that is, set <= other and set != other.
    def __lt__(self, ranges):
        #!!! be sure this appears in the documentation
        '''
        True exactly when the IntRangeSet is a proper subset of the ranges.

        :Example:

        >>> print IntRangeSet('0-4,6-10') < '-1-100' # The right-hand can be any ranges input
        True

        Note: By definition, no set is a proper subset of itself.
        '''
        self, ranges = IntRangeSet._make_args_range_set(self, ranges)
        return self != ranges and self in ranges

    #set > other
    #Test whether the set is a proper superset of other, that is, set >= other and set != other.
    def __gt__(self, other):
        #!!! be sure this appears in the documentation
        '''
        True exactly when the IntRangeSet is a proper superset of the ranges.

        :Example:

        >>> print IntRangeSet('0-4,6-10') > '7-9' # The right-hand can be any ranges input
        True

        Note: By definition, no set is a proper superset of itself.
        '''
        self, other = IntRangeSet._make_args_range_set(self, other)
        return self != other and other in self


    #Same: a & b, a.intersection(b,...)
    #intersection(other, ...)set & other & ...
    #Return a new set with elements common to the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __and__(*ranges_inputs):
        #!!! be sure this appears in the documentation
        '''
        Return the intersection of a IntRangeSet and zero or more ranges inputs. The original IntRangeSet is not changed.

        These are the same:
        
        * ``a & b``
        * ``a.intersection(b)``

        :Example:

        >>> print IntRangeSet('0-4,6-10') & '3-7'
        IntRangeSet('3-4,6-7')

        The 'intersection' method also support intersecting multiple ranges inputs,

        :Example:

        >>> print IntRangeSet('0-4,6-10').intersection('3-7','4-6')
        IntRangeSet('4,6')
        '''
        ranges_inputs = IntRangeSet._make_args_range_set(*ranges_inputs) #generator to made every ranges a IntRangeSet
        ranges_inputs = sorted(ranges_inputs,key=lambda int_range_set:len(int_range_set._start_items)) #sort so that IntRangeSet with smaller range_count is first
        result = ranges_inputs[0] #!!!what if no args, emtpy? The universe?
        for ranges in ranges_inputs[1:]:
            result = result._binary_intersection(ranges)
        return result
    intersection = __and__

    def _binary_intersection(self,other):
        result = IntRangeSet()

        if self.isempty:
            return result

        index = 0
        start0 = self._start_items[index]
        length0 = self._start_to_length[start0]
        last0 = start0+length0-1
        while True:
            start1,length1,index1,contains = other._best_start_length_index_contains(start0)
            last1=start1+length1-1
            if contains:
                if last0 <= last1: #All of range0 fits inside some range1, so add it the intersection and next look at the next range0
                    result._internal_add(start0,length0)
                    index+=1
                    if index >= len(self._start_items):
                        break # leave the while loop
                    start0 = self._start_items[index]
                    length0 = self._start_to_length[start0]
                    last0 = start0+length0-1
                else: #Only part of range0 fits inside some range0, so add that part and then next look at the rest of range0
                    result._internal_add(start0,last1-start0+1)
                    start0 = last1+1
                    length0=last0-start0+1
            else: #start0 is not contained in any range1, so swap self and other and then next look at the next range0
                temp = other
                other = self
                self = temp
                index = index1+1
                if index >= len(self._start_items):
                    break # leave the while loop
                start0 = self._start_items[index]
                length0 = self._start_to_length[start0]
                last0 = start0+length0-1
        return result


    #Same a-b, a.difference(b,...)
    #difference(other, ...)set - other - ...
    #Return a new set with elements in the set that are not in the others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __sub__(self, *ranges_inputs): #!!could be made faster by being more direct instead of using complements
        #!!! be sure this appears in the documentation
        '''
        Return the set difference of a IntRangeSet with zero or more ranges inputs. The original IntRangeSet is not changed.

        These are the same:

        * ``a - b``
        * ``a.difference(b)``

        :Example:

        >>> print IntRangeSet('0-4,6-10') - 1
        IntRangeSet('0,2-4,6-10')
        >>> print IntRangeSet('0-4,6-10') - '3-100'
        IntRangeSet('0-2')

        The 'difference' method also supports subtracting multiple input ranges

        :Example:

        >>> print IntRangeSet('0-4,6-10').difference('3-100',1)
        IntRangeSet('0,2')
        '''
        result = self.copy()
        result.difference_update(*ranges_inputs)
        return result
    difference = __sub__

    #same a^b, a.symmetric_difference(b)
    #symmetric_difference(other)set ^ other
    #Return a new set with elements in either the set or other but not both.
    def __xor__(self, ranges):
        #!!! be sure this appears in the documentation
        '''
        Returns a new IntRangeSet set with elements in either the input IntRangeSet or the input range but not both.

        These are the same:

        * ``a ^ b``
        * ``a.symmetric_difference(b)``

        :Example:

        >>> print IntRangeSet('0-4,6-10') ^ '3-8'
        IntRangeSet('0-2,5,9-10')
        '''
        result = self - ranges
        diff_generator = (IntRangeSet(tuple)-self for tuple in IntRangeSet._static_ranges(ranges))
        result += diff_generator
        return result
    symmetric_difference = __xor__

    def _clone_state(self, result):
        self._start_items = result._start_items
        self._start_to_length = result._start_to_length
        return self



    #intersection_update(other, ...)set &= other & ...
    #Update the set, keeping only elements found in it and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __iand__(*ranges_inputs):
        #!!! be sure this appears in the documentation
        '''
        Set the IntRangeSet to itself intersected with a input range

        These are the same:

        * ``a &= b``
        * ``a.intsersection_update(b)``

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a &= '3-7'
        >>> print a
        IntRangeSet('3-4,6-7')
        '''
        return ranges_inputs[0]._clone_state(IntRangeSet.intersection(*ranges_inputs))
    def intersection_update(*ranges_inputs):
       IntRangeSet.__iand__(*ranges_inputs)

    #same a-=b, a.difference_update(b,...), a.discard(b,...), a.remove(b,...). Note that "remove" is the only one that raises an error if the b,... aren't in a.
    #difference_update(other, ...)set -= other | ...
    #Update the set, removing elements found in others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __isub__(self, *ranges_inputs):
        #!!! be sure this appears in the documentation
        '''
        Remove the elements of the range inputs from the IntRangeSet

        These are the same:

        * ``a -= b``
        * ``a.difference_update(b)``
        * ``a.discard(b)``

        remove' is almost the same except that it raises a KeyError if any element of b is not in a.

        * ``a.remove(b)``

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a -= '3-6'
        >>> print a
        IntRangeSet('0-2,7-10')


        The 'difference_update', 'discard' and 'remove' methods also support subtracting multiple ranges inputs.

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a.difference_update('3-6','8-100')
        >>> print a
        IntRangeSet('0-2,7')
        '''
        for start,last in IntRangeSet._static_ranges(*ranges_inputs):
            self._internal_isub(start, last-start+1)
        return self
    def difference_update(self, *ranges_inputs):
        self.__isub__(*ranges_inputs)

    #remove(elem)
    #Remove element elem from the set. Raises KeyError if elem is not contained in the set.
    def remove(self, *ranges_inputs):
        for start_last_tuple in IntRangeSet._static_ranges(*ranges_inputs):
            if not start_last_tuple in self:
                raise KeyError()
            self -= start_last_tuple

    #discard(elem)
    #Remove element elem from the set if it is present.
    def discard(self, *ranges_inputs):
        self.difference_update(*ranges_inputs)



    #symmetric_difference_update(other)set ^= other
    #Update the set, keeping only elements found in either set, but not in both.
    def __ixor__(self, ranges):
        #!!! be sure this appears in the documentation
        '''
        Set the IntRangeSet to contains exactly those elements that appear in either itself or the input ranges but not both

        These are the same:

        * ``a ^= b``
        * ``a.symmetric_difference_update(b)``

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> a ^= '3-7'
        >>> print a
        IntRangeSet('0-2,5,8-10')
        '''
        return self._clone_state(self ^ ranges)
    def symmetric_difference_update(self, ranges):
        self.__ixor__(ranges)

    #pop()
    def pop(self):
        '''
        Remove and return the largest integer element from the IntRangeSet. Raises KeyError if the IntRangeSet is empty.

        :Example:

        >>> a = IntRangeSet('0-4,6-10')
        >>> print a.pop()
        10
        >>> print a
        IntRangeSet('0-4,6-9')
        '''
        if self.isempty:
            raise KeyError()
        #Get the last range
        start = self._start_items[-1]
        length = self._start_to_length[start]
        if length == 1:
            del self._start_to_length[start]
            self._start_items.pop()
        else:
            self._start_to_length[start] = length - 1
        return start+length-1


    def __delitem__(self,key):
        #!!! be sure this appears in the documentation
        '''
        Remove elements from the IntRangeSet by position index. Position index can be specified by an integer with
        negative integers counting from the end. Position indexes can also be specified with slices and a ranges input.

        :Example:

        Removing with an integer position index:

        >>> a = IntRangeSet('100-200,1000')
        >>> del a[2]
        >>> print a
        IntRangeSet('100-101,103-200,1000')
        >>> del a[-1]
        >>> print a
        IntRangeSet('100-101,103-200')

        :Example:
       
        Removing with a slice:

        >>> a = IntRangeSet('100-200,1000')
        >>> del a[2:11]
        >>> print a
        IntRangeSet('100-101,111-200,1000')

        :Example:
        Removing with a ranges input:

        >>> a = IntRangeSet('100-200,1000')
        >>> del a['2-10']
        >>> print a
        IntRangeSet('100-101,111-200,1000')
        '''
        if isinstance(key,(int,long)):
            if key >= 0:
                for start in self._start_items:
                    length = self._start_to_length[start]
                    if key < length:
                        self -= start+key 
                        return 
                    key -= length
                raise KeyError()
            else:
                assert key < 0
                key = -key-1
                for start_index in xrange(len(self._start_items)):
                    start = self._start_items[-1-start_index]
                    length = self._start_to_length[start]
                    if key < length:
                        self -= start+length-1-key
                        return 
                    key -= length
                raise KeyError()
        elif isinstance(key, slice):
            lenx = len(self)
            start,stop,step = key.start,key.stop,key.step
            start = start or 0
            stop = stop or lenx
            step = step or 1

            if step == 1:
                self -= (self[start],self[stop-1])
            else:
                self -= (self[index] for index in xrange(*key.indices(lenx)))
        else:
            start_and_last_generator = (self._two_index(start_index,last_index) for start_index,last_index in IntRangeSet._static_ranges(key))
            self -= (start_and_last_generator)

    def _two_index(self,start_index,last_index):
        start = self[start_index]
        if last_index == start_index:
            last = start
        else:
            last = self[last_index]
        return start,last

    def __reversed__(self):
        #!!! be sure this appears in the documentation
        '''
        reversed(a) is a generator that produces the integer elements of a in order from largest to smallest.

        :Example:
        
        >>> for i in reversed(IntRangeSet('1-3,10')):
        ...     print i
        10
        3
        2
        1
        '''
        for start in reversed(self._start_items):
            length = self._start_to_length[start]
            for item in xrange(start+length-1, start-1, -1):
                yield item

    @staticmethod
    #This will gather adjacent ranges together into one range, e.g.  1-3,4,5-6 -> 1-6
    def _static_ranges(*iterables):
        iter = IntRangeSet._inner_static_ranges(*iterables)
        try:
            start0,last0 = iter.next()
            assert start0 <= last0, "Invalid range. Start " + str(start0) + " must be no greater than last " + str(last0) + "."
        except StopIteration:
            return
        while True:
            try:
                start1,last1 = iter.next()
                assert start1 <= last1, "Invalid range. Start " + str(start1) + " must be no greater than last " + str(last1) + "."
            except StopIteration:
                yield start0,last0
                return
            if last0+1==start1: #We don't try to merge all cases, just the most common
                last0=last1
            elif last1+1==start0:
                start0=start1
            else:
                yield start0,last0
                start0,last0=start1,last1

    @staticmethod
    def _inner_static_ranges(*iterables):
        for iterable in iterables:
            if isinstance(iterable,(int,long)):
                yield iterable,iterable
            elif isinstance(iterable,tuple):
                assert len(iterable)==2 and isinstance(iterable[0],(int,long)) and isinstance(iterable[1],(int,long)), "Tuples must contain exactly two int elements that represent the start (inclusive) and last (inclusive) elements of a range."
                yield iterable[0],iterable[1]
            elif isinstance(iterable,slice):
                start = iterable.start
                stop = iterable.stop
                step = iterable.step or 1
                assert start is not None and start >=0 and stop is not None and start < stop and step > 0, "With slice, start and stop must be nonnegative numbers, stop must be more than start, and step, if given, must be at least 1"
                if step == 1:
                    yield start,stop-1
                else:
                    for start in xrange(start,stop,step):
                        yield start, start
            elif iterable is None:
                pass
            elif isinstance(iterable,str):
            # Parses strings of the form -10--5,-2-10,12-12. Spaces are allowed, no other characters are.
            #  will return an empty range
                if iterable == "":
                    pass
                else:
                    for range_string in iterable.split(","):
                        match = IntRangeSet._rangeExpression.match(range_string) #!!! is there a good error message if it is not well formed?
                        if match is None:
                            raise Exception("The string is not well-formed. '{0}'".format(range_string))
                        start = int(match.group("start"))
                        last = int(match.group("last") or start)
                        yield start,last
            elif hasattr(iterable, 'ranges'):
                for start, last in iterable.ranges():
                    yield start,last
            elif hasattr(iterable, '__iter__'):
                for start, last in IntRangeSet._static_ranges(*iterable):
                    yield start,last
            else:
                raise Exception("Don't know how to construct a InRangeSet from '{0}'".format(iterable))

    def _internal_add(self, start, length=1): #!! should it be "length" or "last"
        assert length > 0, "Length must be greater than zero"
        assert len(self._start_items) == len(self._start_to_length)
        index = bisect_left(self._start_items, start)
        if index != len(self._start_items) and self._start_items[index] == start:
            if length <= self._start_to_length[start]:
                return
            else:
                self._start_to_length[start] = length
                index += 1	  # index should point to the following range for the remainder of this method
                previous = start
                last = start + length - 1
        elif index == 0:
            self._start_items.insert(index, start)
            self._start_to_length[start] = length
            previous = start
            last = start + length - 1
            index += 1  # index_of_miss should point to the following range for the remainder of this method
        else:
            previous = self._start_items[index - 1]
            last = previous + self._start_to_length[previous] - 1

            if start <= last + 1:
                new_length = start - previous + length
                assert new_length > 0 # real assert
                if new_length < self._start_to_length[previous]:
                    return
                else:
                    self._start_to_length[previous] = new_length
                    last = previous + new_length - 1
            else: # after previous range, not contiguous with previous range
                self._start_items.insert(index, start)
                self._start_to_length[start] = length
                previous = start
                last = start + length - 1
                index += 1

        if index == len(self._start_items):
            return

        # collapse next range into this one
        next = self._start_items[index]
        while last >= next - 1:
            new_last = max(last, next + self._start_to_length[next] - 1)
            self._start_to_length[previous] = new_last - previous + 1 #ItemToLength[previous] + ItemToLength[next]
            del self._start_to_length[next]
            del self._start_items[index]
            last = new_last
            if index >= len(self._start_items):
                break
            next = self._start_items[index]
        return

    # return the range that has the largest start and for which start<=element
    def _best_start_length_index_contains(self, element):
        index = bisect_left(self._start_items, element)
        if index != len(self._start_items) and self._start_items[index] == element: #element is the start element of some range
            return element, self._start_to_length[element], index, True
        elif index == 0: # element is before any of the ranges
            return element, 0, -1, False
        else:
            index -= 1
            start = self._start_items[index]
            length = self._start_to_length[start]
            return start, length, index, element <= start+length-1 # we already know element is greater than start

    def _delete_ranges(self,start_range_index,stop_range_index):
        for range_index in xrange(start_range_index,stop_range_index):
            del self._start_to_length[self._start_items[range_index]]
        del self._start_items[start_range_index:stop_range_index]

    def _shorten_last_range(self,last_in,start1,length1,index1):
        delta_start = last_in-start1+1
        self._start_items[index1] += delta_start
        del self._start_to_length[start1]
        self._start_to_length[start1+delta_start]=length1-delta_start
        assert len(self._start_items) == len(self._start_to_length)
           
    def _shorten_first_range(self,start_in,start0,length0):
        assert len(self._start_items) == len(self._start_to_length)
        self._start_to_length[start0] = start_in-start0
        assert len(self._start_items) == len(self._start_to_length)

    def _internal_isub(self, start_in, length_in=1): #!! should it be "length" or "last"?
        assert length_in > 0, "Length must be greater than zero"
        assert len(self._start_items) == len(self._start_to_length)
        last_in = start_in+length_in-1

        # return the range that has the largest start and for which start<=element
        start0,length0,index0,contains0 = self._best_start_length_index_contains(start_in)
        if length_in > 1:
            start1,length1,index1,contains1 = self._best_start_length_index_contains(last_in)
        else:
            start1,length1,index1,contains1 = start0,length0,index0,contains0

        #Is the front of first range unchanged, changed, or deleted?
        if not contains0:#unchanged
            #Is the end of last range unchanged, changed, or deleted?
            if not contains1:#unchanged
                self._delete_ranges(index0+1,index1+1) #delete any middle range
            elif start1+length1-1 == last_in: # deleted
                self._delete_ranges(index0+1,index1+1) #delete any middle and the last ranges
            else: #changed
                assert index0 < index1, "real assert"
                self._shorten_last_range(last_in,start1,length1,index1)                #shorten last range
                self._delete_ranges(index0+1,index1)                    #delete any middle ranges
        elif start0 == start_in: # deleted
            #Is the end of last range unchanged, changed, or deleted?
            if not contains1:#unchanged
                self._delete_ranges(index0,index1+1) #delete start range and any middle ranges
            elif start1+length1-1 == last_in: # deleted
                self._delete_ranges(index0,index1+1) #delete start range and any middle ranges and last range
            else: #changed
                assert index0 <= index1, "real assert"
                self._shorten_last_range(last_in,start1,length1,index1)              #shorten last range
                self._delete_ranges(index0,index1)                    #delete start range and any middle ranges
        else: #changed
            #Is the end of last range unchanged, changed, or deleted?
            if not contains1:#unchanged
                self._shorten_first_range(start_in,start0,length0)              #shorten first range
                self._delete_ranges(index0+1,index1+1)                    #delete any middle ranges
            elif start1+length1-1 == last_in: # deleted
                self._shorten_first_range(start_in,start0,length0)              #shorten first range
                self._delete_ranges(index0+1,index1+1)                    #delete any middle ranges and last range
            else: #changed
                if index0 == index1: #need to split into two ranges
                    self._start_items.insert(index0+1,last_in+1)
                    self._start_to_length[last_in+1] = (start1+length1-1)-(last_in+1)+1
                    self._shorten_first_range(start_in,start0,length0)              #shorten first range
                else:
                    self._shorten_last_range(last_in,start1,length1,index1)              #shorten last range
                    self._shorten_first_range(start_in,start0,length0)              #shorten first range
                    self._delete_ranges(index0+1,index1)                    #delete any middle ranges
        assert len(self._start_items) == len(self._start_to_length)


class TestLoader(unittest.TestCase):     

    def test_int_range_set(self):
        IntRangeSet._test()

    def test_doc(self):
        doctest.testmod()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    IntRangeSet._test()
    doctest.testmod()
