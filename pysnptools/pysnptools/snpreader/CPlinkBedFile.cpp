#define REAL double
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## doubleCAAA
#include "CPlinkBedFileT.cpp"

#define REAL float
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## floatCAAA
#include "CPlinkBedFileT.cpp"

#define REAL double
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## doubleFAAA
#include "CPlinkBedFileT.cpp"

#define REAL float
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## floatFAAA
#include "CPlinkBedFileT.cpp"

