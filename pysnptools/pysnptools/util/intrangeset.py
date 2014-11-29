#!!!cmk see email todo

import re
from bisect import bisect_left
import logging
import numpy as np

class IntRangeSet(object):
#This is an IntRangeSet it represents a set of integers as a collection of ranges (aka intervals, regions)

    _rangeExpression = re.compile(r"^(?P<start>-?\d+)(-(?P<last>-?\d+))?$")

    def __init__(self, *ranges_args):
        if len(ranges_args) > 0 and isinstance(ranges_args[0],IntRangeSet): #Because we know self is empty, optimize for the case in which the first item is a IntRangeSet
            self._start_items = list(ranges_args[0]._start_items)
            self._start_to_length = dict(ranges_args[0]._start_to_length)
            ranges_args = ranges_args[1:]
        else:
            self._start_items = []
            self._start_to_length = {}
        self.add(*ranges_args) 

    #same: a.add(b,...), a+=b, a|=b, a.update(b,...), !!!cmk extend? append?
    def add(self, *ranges_args):
        #!!consider special casing the add of a single int. Anything else?
        for start,last in IntRangeSet._static_ranges(*ranges_args):
            self._internal_add(start, last-start+1)
    #update(other, ...)set |= other | ...
    #Update the set, adding elements from all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __iadd__(self, *ranges_args):
        self.add(*ranges_args)
        return self
    def __ior__(self, *ranges_args):
        self.add(*ranges_args)
        return self
    update = __ior__

    def copy(self):
            return IntRangeSet(self)

    def ranges(self):
        for item in self._start_items:
            last = item + self._start_to_length[item] - 1
            yield item, last

    def __iter__(self):
        for (first, last) in self.ranges():
            for i in xrange(first,last+1):
                yield i

    def clear(self):
        del self._start_items[:]
        self._start_to_length.clear()

    def __len__(self):
        return sum(self._start_to_length.values())

    def sum(self):
        result = 0
        for start in self._start_items:
            length = self._start_to_length[start]
            result += (start + start + length - 1)*length//2
        return result

    def __eq__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        if other is None or len(self._start_items)!=len(other._start_items):
            return False
        for i, start in enumerate(self._start_items):
            if start != other._start_items[i] or self._start_to_length[start] != other._start_to_length[start]:
                return False
        return True

    def __ne__(self, other):
        #Don't need to call _make_args_range_set because __eq__ will call it
        return not self==other

    #Same: a >= b, b.issuperset(a,...), b in a
    #Returns True iff item is within the ranges of this IntRangeSet.
    def __contains__(self, *ranges_args):
        for start_in,last_in in IntRangeSet._static_ranges(*ranges_args):
            start_self,length_self,index,contains = self._best_start_length_index_contains(start_in)
            if not contains or last_in > start_self+length_self-1:
                return False
        return True
    #issuperset(other)set >= other
    #Test whether every element in other is in the set.
    def __ge__(self,other):
        return other in self
    issuperset = __ge__



    @property
    def isempty(self):
        return len(self._start_items)==0

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self._repr_internal("-", ",")

    def _repr_internal(self, seperator1, separator2):
        if self.isempty:
            return "empty" #!!const

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
        assert "0" == str(int_range_set)
        int_range_set.add(1)
        assert "0-1" == str(int_range_set)
        int_range_set.add(4)
        assert "0-1,4" == str(int_range_set)
        int_range_set.add(5)
        assert "0-1,4-5" == str(int_range_set)
        int_range_set.add(7)
        assert "0-1,4-5,7" == str(int_range_set)
        int_range_set.add(2)
        assert "0-2,4-5,7" == str(int_range_set)
        int_range_set.add(3)
        assert "0-5,7" == str(int_range_set)
        int_range_set.add(6)
        assert "0-7" == str(int_range_set)
        int_range_set.add(-10)
        assert "-10,0-7" == str(int_range_set)
        int_range_set.add(-5)
        assert "-10,-5,0-7" == str(int_range_set)

        assert str(IntRangeSet("-10--5")) == "-10--5"
        assert str(IntRangeSet("-10--5,-3")) == "-10--5,-3"
        assert str(IntRangeSet("-10--5,-3,-2-1")) == "-10--5,-3-1"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5")) == "-10--5,-3-5"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5,7-12")) == "-10--5,-3-5,7-12"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15")) == "-10--5,-3-5,7-15"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16")) == "-10--5,-3-5,7-16"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25")) == "-10--5,-3-5,7-16,20-25"
        assert str(IntRangeSet("-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25,22-23")) == "-10--5,-3-5,7-16,20-25"

        range = "-10--5,-3,-2-1,1-5,7-12,13-15,14-16,20-25,22-23"
        int_range_set = IntRangeSet(range)
        assert str(int_range_set) == "-10--5,-3-5,7-16,20-25"

        range = "1-5,0,4-10,-10--5,-12--3,15-20,12-21,-13"
        int_range_set = IntRangeSet(range)
        assert str(int_range_set) == "-13--3,0-10,12-21"

        assert len(int_range_set) == 32

        int_range_set1 = IntRangeSet("-10--5")
        int_range_set2 = int_range_set1.copy()
        assert int_range_set1 is not int_range_set2
        assert int_range_set1 == int_range_set2
        int_range_set2.add(7)
        assert int_range_set1 != int_range_set2

        assert str(IntRangeSet(7)) == "7"
        assert str(IntRangeSet((7,7))) == "7"
        assert str(IntRangeSet((7,10))) == "7-10"
        assert str(IntRangeSet(xrange(7,11))) == "7-10"
        assert str(IntRangeSet(np.s_[7:11])) == "7-10"
        assert str(IntRangeSet(np.s_[7:11:2])) == "7,9"
        assert str(IntRangeSet(xrange(7,11,2))) == "7,9"
        assert str(IntRangeSet(None)) == "empty"
        assert str(IntRangeSet("empty")) == "empty"
        assert [e for e in IntRangeSet("-10--5,-3")] == [-10,-9,-8,-7,-6,-5,-3]
        int_range_set3 = IntRangeSet(7,10)
        int_range_set3.clear()
        assert str(int_range_set3) == "empty"
        assert len(IntRangeSet("-10--5,-3")) == 7

        int_range_set4 = IntRangeSet("-10--5,-3")
        int_range_set4.add(-10,-7)
        assert str(int_range_set4) == "-10--5,-3"
        int_range_set4.add(-10,-4)
        assert str(int_range_set4) == "-10--3"

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


        assert IntRangeSet("empty") & IntRangeSet("empty") == IntRangeSet("empty")
        assert IntRangeSet("empty").intersection("empty","empty") == IntRangeSet("empty")
        assert IntRangeSet("empty") & IntRangeSet("empty") & IntRangeSet("empty") == IntRangeSet("empty")

        assert IntRangeSet("1") & IntRangeSet("empty") == IntRangeSet("empty")
        assert IntRangeSet("empty") & IntRangeSet("1") == IntRangeSet("empty")
        assert IntRangeSet("1-5") & IntRangeSet("empty") == IntRangeSet("empty")
        assert IntRangeSet("empty") & IntRangeSet("1-5") == IntRangeSet("empty")
        assert IntRangeSet("1-5,7") & IntRangeSet("empty") == IntRangeSet("empty")
        assert IntRangeSet("empty") & IntRangeSet("1-5,7") == IntRangeSet("empty")

        assert IntRangeSet("1") & IntRangeSet("1") == IntRangeSet("1")
        assert IntRangeSet("1-5") & IntRangeSet("1") == IntRangeSet("1")
        assert IntRangeSet("1") & IntRangeSet("1-5") == IntRangeSet("1")
        assert IntRangeSet("1-5,7") & IntRangeSet("1") == IntRangeSet("1")

        assert IntRangeSet("2") & IntRangeSet("1-5,7") == IntRangeSet("2")
        assert IntRangeSet("1-5") & IntRangeSet("2") == IntRangeSet("2")
        assert IntRangeSet("2") & IntRangeSet("1-5") == IntRangeSet("2")
        assert IntRangeSet("1-5,7") & IntRangeSet("2") == IntRangeSet("2")


        assert IntRangeSet("-2") & IntRangeSet("1-5,7") == IntRangeSet("empty")
        assert IntRangeSet("1-5") & IntRangeSet("-2") == IntRangeSet("empty")
        assert IntRangeSet("-2") & IntRangeSet("1-5") == IntRangeSet("empty")
        assert IntRangeSet("1-5,7") & IntRangeSet("-2") == IntRangeSet("empty")

        assert IntRangeSet("22") & IntRangeSet("1-5,7") == IntRangeSet("empty")
        assert IntRangeSet("1-5") & IntRangeSet("22") == IntRangeSet("empty")
        assert IntRangeSet("22") & IntRangeSet("1-5") == IntRangeSet("empty")
        assert IntRangeSet("1-5,7") & IntRangeSet("22") == IntRangeSet("empty")


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

        assert IntRangeSet("10-14,55-59").index("57,56-56") == 6 #returns the index of the start of the contiguous place where 57,"56-56" occurs
        try:
            IntRangeSet("10-14,55-59").index([10,55]) #Doesn't pass because 10,55 doesn't occur contiguously
        except IndexError:
            pass
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




    #s[i] ith item of s, origin 0 (3) 
    #s[i:j] slice of s from i to j (3)(4) 
    #s[i:j:k] slice of s from i to j with step k (3)(5) 
    def __getitem__(self, key):
        if isinstance(key, slice):
            lenx = len(self)
            start,stop,step = key.start,key.stop,key.step
            start = start or 0
            stop = stop or lenx
            step = step or 1

            if step == 1:
                return self & (self[start],self[stop-1])
            else:
                logging.info("Slicing with a step other than 1 is implemented slowly") #!!
                return IntRangeSet(self[index] for index in xrange(*key.indices(lenx)))
        elif key >= 0:
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

    #max(s) largest item of s   
    def max(self, *ranges_args):
        lo,hi = self._min_and_max(self,*ranges_args)
        return hi

    #min(s) smallest item of s   
    def min(self, *ranges_args):
        lo,hi = self._min_and_max(self,*ranges_args)
        return lo

    @staticmethod
    def _min_and_max(*ranges_args):
        lo = float("+inf")
        hi = float("-inf")
        for ranges in ranges_args:
            if isinstance(ranges,IntRangeSet):
                lo = min(lo, ranges._start_items[0])
                start = ranges._start_items[-1]
                hi = max(hi, start+ranges._start_to_length[start]-1)
            else:
                for start,last in IntRangeSet._static_ranges(ranges):
                    lo = min(lo,start)
                    hi = max(hi,last)
        return lo,hi

    def _make_args_range_set(*args):
        for arg in args:
            if arg is None:
                yield None
            elif isinstance(arg,IntRangeSet):
                yield arg
            else:
                yield IntRangeSet(arg)

    #same: a.union(b,...), a+b, a|b, a.concat(b,...), a.or(b,...)
    def __concat__(*ranges_args):
        result = IntRangeSet()
        result.add(*ranges_args)
        return result
    #s + t the concatenation of s and t (6) 
    __add__ = __concat__ #!!!expand all these out
        #union(other, ...)set | other | ...
    #Return a new set with elements from the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __or__(self, other):
        return self+other
    union = __concat__


    #s * n, n * s n shallow copies of s concatenated (2) 
    def __mul__(self, n):
        return IntRangeSet(self)

    #s.index(x) index of the x in s
    def index(self, other):
        if isinstance(other,(int,long)):
            return self._index_element(other)
        else:
            piece = self & IntRangeSet._min_and_max(other)
            if piece == other:
                return self._index_element(piece[0])
            else:
                raise IndexError()

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
        if ranges in self:
            return 1
        else:
            return 0

    #Return True if the set has no elements in common with other. Sets are disjoint if and only if their intersection is the empty set.
    def isdisjoint(self, ranges):
        intersection = self & ranges #!! this could be faster by not materializing the full intersection
        return intersection.isempty

    #Same: a <= b, a.issubset(b)
    #issubset(other)set <= other
    #Test whether every element in the set is in other.
    def __le__(self, ranges):
        self, ranges = IntRangeSet._make_args_range_set(self, ranges)
        return self in ranges
    issubset = __le__

    #set < other
    #Test whether the set is a proper subset of other, that is, set <= other and set != other.
    def __lt__(self, ranges):
        self, ranges = IntRangeSet._make_args_range_set(self, ranges)
        return self != ranges and self in ranges

    #set > other
    #Test whether the set is a proper superset of other, that is, set >= other and set != other.
    def __gt__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return self != other and other in self


    #Same: a & b, a.intersection(b,...)
    #intersection(other, ...)set & other & ...
    #Return a new set with elements common to the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __and__(*ranges_args):
        ranges_args = IntRangeSet._make_args_range_set(*ranges_args) #generator to made every ranges a IntRangeSet
        ranges_args = sorted(ranges_args,key=lambda int_range_set:len(int_range_set._start_items)) #sort so that IntRangeSet with smaller range_count is first
        result = ranges_args[0] #!!!what if no args, emtpy? The universe?
        for ranges in ranges_args[1:]: #!!!cmk rename ranges to ranges
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
    def __sub__(self, *ranges_args): #!!could be made faster by being more direct instead of using complements
        result = self.copy()
        result.difference_update(*ranges_args)
        return result
    difference = __sub__

    #same a^b, a.symmetric_difference(b)
    #symmetric_difference(other)set ^ other
    #Return a new set with elements in either the set or other but not both.
    def __xor__(self, other):
        result = self | other
        result -= self & other
        return result #!!could be made faster by being more direct
    symmetric_difference = __xor__

    def _clone_state(self, result):
        self._start_items = result._start_items
        self._start_to_length = result._start_to_length
        return self



    #intersection_update(other, ...)set &= other & ...
    #Update the set, keeping only elements found in it and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __iand__(*ranges_args):
        return ranges_args[0]._clone_state(IntRangeSet.intersection(*ranges_args))
    intersection_update = __iand__

    #same a-=b, a.difference_update(b,...), a.discard(b,...), a.remove(b,...). Note that "remove" is the only one that raises an error if the b,... aren't in a.
    #difference_update(other, ...)set -= other | ...
    #Update the set, removing elements found in others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __isub__(self, *ranges_args):
        #!!consider special casing the add of a single int. Anything else?
        for start,last in IntRangeSet._static_ranges(*ranges_args):
            self._internal_isub(start, last-start+1)
        return self
    difference_update = __isub__

    #remove(elem)
    #Remove element elem from the set. Raises KeyError if elem is not contained in the set.
    #!!could implement more efficiently like add/_internal_add
    def remove(self, *ranges_args):
        if not self.__contains__(*ranges_args):
            raise KeyError()
        self.difference_update(*ranges_args)


    #discard(elem)
    #Remove element elem from the set if it is present.
    #!!could implement more efficiently like add/_internal_add
    def discard(self, *ranges_args):
        self.difference_update(*ranges_args)



    #symmetric_difference_update(other)set ^= other
    #Update the set, keeping only elements found in either set, but not in both.
    def __ixor__(self, other):
        return self._clone_state(self ^ other)
    symmetric_difference_update = __ixor__

    #pop()
    #Remove and return an arbitrary element from the set. Raises KeyError if the set is empty.
    def pop(self):
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
        if isinstance(key, slice):
            lenx = len(self)
            start,stop,step = key.start,key.stop,key.step
            start = start or 0
            stop = stop or lenx
            step = step or 1

            if step == 1:
                self -= (self[start],self[stop-1])
            else:
                logging.info("Slicing with a step other than 1 is implemented slowly") #!!
                self -= (self[index] for index in xrange(*key.indices(lenx)))
        elif key >= 0:
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

    def __reversed__(self):
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
            # "empty" will return an empty range
                if iterable == "empty": #!!const #!! is "empty" the best way to print an empty set?
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
                self._delete_ranges(index0+1,index1)                    #delete any middle ranges
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IntRangeSet._test()
