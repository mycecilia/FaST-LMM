#If we want this, we need to up this file. See SnpAndSetnameCollection.py for an example

#class NucRangeSet(object):
#    '''
#    Reads a file nuc ranges. You can iterate the result and get a sequence of NucRange instances suitable for a Bed instances read method.
#    '''
#    def __init__(self, filename,idist=2):
#        logging.info("Reading {0}".format(filename))
#        self.altsets = SP.loadtxt(filename, dtype = 'str',comments=None)
#        self.idist=idist

#    def __len__(self):
#        return len(self.altsets)

#    def __iter__(self):
#        for altset in self.altsets:
#            name = altset[0]
#            chr1 = int(altset[1])
#            pos1 = float(altset[2])
#            chr2 = int(altset[3])
#            pos2 = float(altset[4])

#            yield NucRange(name,chr1,pos1,chr2,pos2,idist=self.idist)


#class NucRange(object):
#    '''
#    When given to a bed reader, tells it to read snps from [startchr,startpos] to [endchar,endpos] (inclusive).
#    '''

#    def __init__(self, name, startchr,startpos,endchr,endpos,idist=2):
#        '''
#        startchr,startpos        : starting position of the loaded genomic region
#        endchr,endpos            : end-position of the loaded genomic region
#        idist :the index of the position index to use (default 2)
#                    1 : genomic distance
#                    2 : base-pair distance

#        '''
#        self.idist = idist
#        self.name = name
#        self.startpos = [startchr,startpos]
#        self.endpos = [endchr,endpos]

#    def get_name(self):
#        return self.name

#    def get_count(self,bed):
#        start, nSNPs = self.__get_start_and_count(bed)
#        return nSNPs

#    # would be nice to not need to call this twice
#    def __get_start_and_count(self,bed):
#        import fastlmm.util.util as ut
#        i_c = bed.pos[:,0]==self.startpos[0]
#        i_largerbp = bed.pos[:,self.idist]>=self.startpos[1]
#        start = ut.which(i_c * i_largerbp)
#        while start-1 >= 0 and bed.pos[start-1,self.idist] == self.startpos[1]:
#            start = start -1
#        i_c = bed.pos[:,0]==self.endpos[0]
#        i_smallerbp = bed.pos[:,self.idist]<self.endpos[1]
#        end = ut.which_opposite(i_c * i_smallerbp)
#        while end+1 < bed.pos.shape[0] and bed.pos[end+1,self.idist] == self.endpos[1]:
#            end = end + 1
#        nSNPs = end - start + 1
#        if end == 0: raise Exception("End should not be 0")
#        return [start, nSNPs]


#    def index_sequence(self,bed):
#        start, nSNPs = self.__get_start_and_count(bed)
#        return xrange(start,start+nSNPs)
