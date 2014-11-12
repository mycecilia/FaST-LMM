#define REAL double
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## doubleCAAA
#include "CPlinkBedFileT.h"

#define REAL float
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## floatCAAA
#include "CPlinkBedFileT.h"

#define REAL double
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## doubleFAAA
#include "CPlinkBedFileT.h"

#define REAL float
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## floatFAAA
#include "CPlinkBedFileT.h"

