import re
from bisect import bisect_left
import logging

class IntRangeSet(object):
#This is an IntRangeSet it represents a set of integers as a collection of ranges (aka intervals)

    _rangeExpression = re.compile(r"^(?P<start>-?\d+)(-(?P<last>-?\d+))?$")

    def __init__(self, input=None, input2=None):
        if isinstance(input,IntRangeSet):
            assert input2 is None
            self._start_items = list(input._start_items)
            self._item_to_length = dict(input._item_to_length)
        else:
            self._start_items = []
            self._item_to_length = {}
            self.add(input, input2)


    def copy(self):
            return IntRangeSet(self)

    def ranges(self):
        for item in self._start_items:
            last = item + self._item_to_length[item] - 1
            yield (item, last)

    def __iter__(self):
        for (first, last) in self.ranges():
            for i in xrange(first,last+1):
                yield i

    def clear(self):  #!!!cmk is add_range the best name? Should we support slice notation?
        del self._start_items[:]
        self._item_to_length.clear()

    def __len__(self):
        return sum(self._item_to_length.values())

    def __eq__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        if other is None or len(self._start_items)!=len(other._start_items):
            return False
        for i, start in enumerate(self._start_items):
            if start != other._start_items[i] or self._item_to_length[start] != other._item_to_length[start]:
                return False
        return True

    def __ne__(self, other):
        #Don't need to call _make_args_range_set because __eq__ will call it
        return not self==other

    def add(self, input, input2=None):
        if isinstance(input,(int,long)) and isinstance(input2,(int,long)):
            assert input <= input2, "Invalid range. Start " + str(input) + " must be no greater than last " + str(input2) + "."
            length = input2 - input + 1
            self._try_add(input, length)
            return

        assert input2 is None, "If input2 is not None, then both input and input2 should be integers"

        if input is None:
            pass
        elif isinstance(input,(int,long)) and input2 is None:
            self._try_add(input)
        elif isinstance(input,str):
            # Parses strings of the form -10--5,-2-10,12-12. Spaces are allowed, no other characters are.
            # "empty" will return an empty range
            if input == "empty": #!!const #!!!cmk is "empty" the best way to print an empty set?
                pass
            else:
                for range_string in input.split(","):
                    match = self._rangeExpression.match(range_string) #!!! is there a good error message if it is not well formed?
                    start = int(match.group("start"))
                    last = match.group("last")
                    if last is None:
                        last = start
                    else:
                        last = int(last)
                    assert start <= last, "Invalid range. Start " + str(start) + " must be no greater than last " + str(last) + "."
                    length = last - start + 1
                    self._try_add(start, length)
        elif hasattr(input, 'ranges'):
            for start, last in input.ranges():
                self._try_add(start, last-start+1)
        elif hasattr(input, '__iter__'):
            for element in input:
                self.add(element)
        else:
            raise Exception("Don't know how to construct a InRangeSet from '{0}' (and '{1}')".format(input, input2))



    #True if any item was added. False if everything was already in the set
    #!!!cmk should there be any trys
    def _try_add(self, start, length=1): #!!!cmk should it be "length" or "last"
        assert length > 0, "Length must be greater than zero"
        assert len(self._start_items) == len(self._item_to_length)
        index = bisect_left(self._start_items, start)
        if index != len(self._start_items) and self._start_items[index] == start:
            if length <= self._item_to_length[start]:
                return False
            else:
                self._item_to_length[start] = length
                index += 1	  # index should point to the following range for the remainder of this method
                previous = start
                last = start + length - 1
        elif index == 0:
            self._start_items.insert(index, start)
            self._item_to_length[start] = length
            previous = start
            last = start + length - 1
            index += 1  # index_of_miss should point to the following range for the remainder of this method
        else:
            previous = self._start_items[index - 1]
            last = previous + self._item_to_length[previous] - 1

            if start <= last + 1:
                new_length = start - previous + length
                assert new_length > 0 # real assert
                if new_length < self._item_to_length[previous]:
                    return False
                else:
                    self._item_to_length[previous] = new_length
                    last = previous + new_length - 1
            else: # after previous range, not contiguous with previous range
                self._start_items.insert(index, start)
                self._item_to_length[start] = length
                previous = start
                last = start + length - 1
                index += 1

        if index == len(self._start_items):
            return True

        # collapse next range into this one
        next = self._start_items[index]
        while last >= next - 1:
            new_last = max(last, next + self._item_to_length[next] - 1)
            self._item_to_length[previous] = new_last - previous + 1 #ItemToLength[previous] + ItemToLength[next]
            del self._item_to_length[next]
            del self._start_items[index]
            last = new_last
            if index >= len(self._start_items):
                break
            next =  self._start_items[index]
        return True

    # return the range that has the largest start and for which start<=element
    def _best_start_length_index_contains(self, element):
        index = bisect_left(self._start_items, element)
        if index != len(self._start_items) and self._start_items[index] == element: #element is the start element of some range
            return element, self._item_to_length[element], index, True
        elif index == 0: # element is before any of the ranges
            return element, 0, -1, False
        else:
            index -= 1
            start = self._start_items[index]
            length = self._item_to_length[start]
            return start, length, index, element <= start+length-1 # we already know element is greater than start

    #Returns True iff item is within the ranges of this IntRangeSet.
    def __contains__(self, input):
        if isinstance(input,(int,long)):
            _,_,_,contains = self._best_start_length_index_contains(input)
            return contains
        elif isinstance(input,str):
            return IntRangeSet(input) in self
        elif hasattr(input, 'ranges'):
            # Returns true iff the entire range (start,last) is captured by this int_range_set.
            for start_in, last_in in input.ranges():
                start_self,length_self,index,contains = self._best_start_length_index_contains(start_in)
                if not contains or last_in > start_self+length_self-1:
                    return False
            return True
        elif hasattr(input, '__iter__'):
            start_self, last_self = (0,-1) #init to an range that contains nothing
            for element in input:
                if element < start_self or element > last_self: #not in the current range? Find the best one for this element
                    _,_,_,contains = self._best_start_length_index_contains(element)
                    if not contains:
                        return False #If not, its not in any range, so return False
            return True
        else:
            raise Exception("Don't know how to test if '{0}' is contained in the IntRangeSet".format(input))

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
        assert str(IntRangeSet(7,7)) == "7"
        assert str(IntRangeSet(7,10)) == "7-10"
        assert str(IntRangeSet(xrange(7,11))) == "7-10"
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

        assert IntRangeSet("0-66")-IntRangeSet("30-100") == IntRangeSet("0-29")
        assert IntRangeSet("0-66").difference("30-50","40-100") == IntRangeSet("0-29")
        assert IntRangeSet("0-66,100,200")-IntRangeSet("30-100,300") == IntRangeSet("0-29,200")
        assert IntRangeSet("30-100")-IntRangeSet("0-66") == IntRangeSet("67-100")
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
        assert remove0 == IntRangeSet("9,14,55-59")

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
        assert discard0 == IntRangeSet("9,14,55-59")

        discard0 = IntRangeSet("9-14,55-59")
        discard0.discard("100")
        assert discard0 == "9-14,55-59"

        discard0 = IntRangeSet("9-14,55-59")
        try:
            discard0.discard(IntRangeSet("100"),101)
        except Exception:
            pass

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



    #s[i] ith item of s, origin 0 (3) 
    #s[i:j] slice of s from i to j (3)(4) 
    #s[i:j:k] slice of s from i to j with step k (3)(5) 
    def __getitem__(self, key):
        if isinstance(key, slice):
            start,stop,step = key.start,key.stop,key.step
            if step is None:
                step = 1
            if stop is None:
                stop = len(self)

            if step == 1:
                other = IntRangeSet(self[start],self[stop-1])
                return other & self
            else:
                logging.warn("Slicing with a step other than 1 is implemented slowly") #!!
                return IntRangeSet(self[index] for index in xrange(start,stop,step))
        elif key >= 0:
            for start in self._start_items:
                length = self._item_to_length[start]
                if key < length:
                    return start+key
                key -= length
            raise KeyError()
        else:
            assert key < 0
            key = -key-1
            for start_index in xrange(len(self._start_items)):
                start = self._start_items[-1-start_index]
                length = self._item_to_length[start]
                if key < length:
                    return start+length-1-key
                key -= length
            raise KeyError()

    #max(s) largest item of s   
    def max(self):
        return self[-1]

    #min(s) smallest item of s   
    def min(self):
        return self[0]

    def _make_args_range_set(*args):
        for arg in args:
            if arg is None:
                yield None
            elif isinstance(arg,IntRangeSet):
                yield arg
            else:
                yield IntRangeSet(arg)

    def __concat__(*args):
        args = IntRangeSet._make_args_range_set(*args)
        result = IntRangeSet()
        for arg in args:
            result.add(arg)
        return result

    #s + t the concatenation of s and t (6) 
    __add__ = __concat__



    #s * n, n * s n shallow copies of s concatenated (2) 
    def __mul__(self, n):
        return IntRangeSet(self)

    #s.index(x) index of the first occurrence of x in s
    def index(self, element):
        index = bisect_left(self._start_items, element)

        # Find the position_in_its_range of this element
        if index != len(self._start_items) and self._start_items[index] == element: #element is start value
            position_in_its_range = 0
        elif index == 0:   # item is before any of the ranges
            raise IndexError()
        else:
            index -= 1
            start = self._start_items[index]
            last = start+self._item_to_length[start]-1
            if element > last: # we already know it's greater than start
                raise IndexError()
            else:
                position_in_its_range = element - start

        # Sum up the length of all preceding ranges
        preceeding_starts = self._start_items[0:index]
        preceeding_lengths = (self._item_to_length[start] for start in preceeding_starts)
        result = position_in_its_range + sum(preceeding_lengths)
        return result

    #s.count(x) total number of occurrences of x in s   
    def count(self,x):
        if x in self:
            return 1
        else:
            return 0

    #Return True if the set has no elements in common with other. Sets are disjoint if and only if their intersection is the empty set.
    def isdisjoint(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        intersection = self & other #!! this could be faster by not materializing the full intersection
        return len(intersection._start_items)==0

    #issubset(other)set <= other
    #Test whether every element in the set is in other.
    def __le__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return self in other
    issubset = __le__

    #set < other
    #Test whether the set is a proper subset of other, that is, set <= other and set != other.
    def __lt__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return self != other and self in other

    #issuperset(other)set >= other
    #Test whether every element in other is in the set.
    def __ge__(self,other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return other in self
    issuperset = __ge__

    #set > other
    #Test whether the set is a proper superset of other, that is, set >= other and set != other.
    def __gt__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return self != other and other in self

    #union(other, ...)set | other | ...
    #Return a new set with elements from the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __or__(self, other):
        return self+other
    union = __concat__

    #intersection(other, ...)set & other & ...
    #Return a new set with elements common to the set and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __and__(*args):
        args = sorted(IntRangeSet._make_args_range_set(*args),key=lambda int_range_set:len(int_range_set._start_items))
        result = args[0] #!!!what if no args, emtpy? The universe?
        for arg in args[1:]:
            result = result._binary_intersection(arg)
        return result
    intersection = __and__

    def _binary_intersection(self,other):
        result = IntRangeSet()

        if len(self._start_items) == 0:
            return result

        index = 0
        start0 = self._start_items[index]
        length0 = self._item_to_length[start0]
        last0 = start0+length0-1
        while True:
            start1,length1,index1,contains = other._best_start_length_index_contains(start0)
            last1=start1+length1-1
            if contains:
                if last0 <= last1: #All of range0 fits inside some range1, so add it the intersection and next look at the next range0
                    result._try_add(start0,length0)
                    index+=1
                    if index >= len(self._start_items):
                        break # leave the while loop
                    start0 = self._start_items[index]
                    length0 = self._item_to_length[start0]
                    last0 = start0+length0-1
                else: #Only part of range0 fits inside some range0, so add that part and then next look at the rest of range0
                    result._try_add(start0,last1-start0+1)
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
                length0 = self._item_to_length[start0]
                last0 = start0+length0-1
        return result

    def _universe(*args):
        assert len(args) > 0

        args = IntRangeSet._make_args_range_set(*args)

        start = None
        last = None
        for int_range_set in args:
            start2 = int_range_set[0]
            last2 = int_range_set[-1]
            if start is None or  start2 < start:
                start = start2
            if last is None or  last2 > last:
                last = last2
        return IntRangeSet(start,last)

    def _complement(self,universe):
        result = IntRangeSet()
        assert len(universe._item_to_length) == 1, "The universe must contain a single range"
        startU = universe[0]
        lastU = universe[-1]
        startS = self[0]
        if startU < startS: #the universe has a smaller element than self
            result._try_add(startU, startS-startU)
        else:
            assert startU == startS, "The universe must be a superset of self"

        for i, (startS, lastS) in enumerate(self.ranges()):
            if i+1 < len(self._start_items):
                start_next = self._start_items[i+1]
                result._try_add(lastS+1,start_next-(lastS+1))
            else:
                if lastS < lastU:
                    result._try_add(lastS+1,lastU-lastS)
                else:
                    assert lastS == lastU, "The universe must be a superset of self"
        return result

    @staticmethod
    def _comp_others(universe, args):
        for i, arg in enumerate(args):
            if i == 0:
                yield arg
            else:
                yield arg._complement(universe)

    #difference(other, ...)set - other - ...
    #Return a new set with elements in the set that are not in the others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __sub__(*args):
        args = list(IntRangeSet._make_args_range_set(*args))
        universe = IntRangeSet._universe(*args)
        comp_others = IntRangeSet._comp_others(universe, args)
        return IntRangeSet.intersection(*comp_others)
    difference = __sub__

    #symmetric_difference(other)set ^ other
    #Return a new set with elements in either the set or other but not both.
    def __xor__(self, other):
        self, other = IntRangeSet._make_args_range_set(self, other)
        return IntRangeSet.union(self, other)-IntRangeSet.intersection(self, other)
    symmetric_difference = __xor__

    def _clone_state(self, result):
        self._start_items = result._start_items
        self._item_to_length = result._item_to_length
        return self

    #update(other, ...)set |= other | ...
    #Update the set, adding elements from all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __ior__(*args):
        if len(args)==0:
            return IntRangeSet()

        args = list(IntRangeSet._make_args_range_set(*args))
        for arg in args[1:]:
            args[0].add(arg)
        return args[0]
    update = __ior__
    def __iadd__(*args):
        return IntRangeSet.__ior__(*args)


    #intersection_update(other, ...)set &= other & ...
    #Update the set, keeping only elements found in it and all others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __iand__(*args):
        return args[0]._clone_state(IntRangeSet.intersection(*args))
    intersection_update = __iand__

    #difference_update(other, ...)set -= other | ...
    #Update the set, removing elements found in others.
    #Changed in version 2.6: Accepts multiple input iterables.
    def __isub__(*args):
        return args[0]._clone_state(IntRangeSet.difference(*args))
    difference_update = __isub__


    #symmetric_difference_update(other)set ^= other
    #Update the set, keeping only elements found in either set, but not in both.
    def __ixor__(self, other):
        return self._clone_state(IntRangeSet.symmetric_difference(self, other))
    symmetric_difference_update = __ixor__

    #remove(elem)
    #Remove element elem from the set. Raises KeyError if elem is not contained in the set.
    #!!could implement more efficiently like add/_try_add
    def remove(self, input, input2=None):
        if isinstance(input,IntRangeSet):
            assert input2 is None
        else:
            input = IntRangeSet(input, input2)
        if not input < self:
            raise KeyError()
        self -= input


    #discard(elem)
    #Remove element elem from the set if it is present.
    #!!could implement more efficiently like add/_try_add
    def discard(self, input, input2=None):
        if isinstance(input,IntRangeSet):
            assert input2 is None
        else:
            input = IntRangeSet(input, input2)
        self -= input

    #pop()
    #Remove and return an arbitrary element from the set. Raises KeyError if the set is empty.
    def pop(self):
        if len(self._start_items)==0:
            raise KeyError()
        #Get the last range
        start = self._start_items[-1]
        length = self._item_to_length[start]
        if length == 1:
            del self._item_to_length[start]
            self._start_items.pop()
        else:
            self._item_to_length[start] = length - 1
        return start+length-1


    def __delitem__(self,key):
        if isinstance(key, slice):
            start,stop,step = key.start,key.stop,key.step
            if step is None:
                step = 1
            if stop is None:
                stop = len(self)

            if step == 1:
                self -= IntRangeSet(self[start],self[stop-1])
            else:
                logging.warn("Slicing with a step other than 1 is implemented slowly") #!!
                self -= (self[index] for index in xrange(start,stop,step))
        elif key >= 0:
            for start in self._start_items:
                length = self._item_to_length[start]
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
                length = self._item_to_length[start]
                if key < length:
                    self -= start+length-1-key
                    return 
                key -= length
            raise KeyError()

    def __reversed__(self):
        for start in reversed(self._start_items):
            length = self._item_to_length[start]
            for item in xrange(start+length-1, start-1, -1):
                yield item

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IntRangeSet._test()
