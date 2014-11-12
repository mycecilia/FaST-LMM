/*#if !defined( CPlinkBedFileT_h )
#define CPlinkBedFileT_h
*/
/*
 *******************************************************************
 *
 *    Copyright (c) Microsoft. All rights reserved.
 *
 *    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 *    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 *    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 *    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
 *
 ******************************************************************
 */

/*
 * CPlinkBedFile - {PLINK Bed File Access Class}
 *
 *         File Name:   CPlinkBedFile.h
 *           Version:   2.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:   14 Aug 2013
 *
 *    Module Purpose:   This file defines the CPlinkBedFile class 
 *                         for FastLmmC
 *
 *                      A .BED file contains compressed binary genotype values for 
 *                         for individuals by SNPs.  For FastLmm, we prefer and may
 *                         require the file LayoutMode be LayoutGroupGenotypesBySnp
 *
 *                      The .bed header is three bytes followed immediately by data.
 *                      
 *                         bedFileMagic1 | bedFileMagic2 | LayoutMode
 *                         [... data ... ]
 *
 *    Change History:   Version 2.00: Reworked to be wrapped in python version by Chris Widmer (chris@shogun-toolbox.org)
 *
 * Test Files: 
 */

#include <vector>
#include <string>
#include <limits>
//#include <inttypes.h>
 
using namespace std;
typedef unsigned char BYTE;
typedef unsigned long long uint64_t_;

#if !defined(CPlinkBedFileT_h_consts)
#define CPlinkBedFileT_h_consts

const BYTE bedFileMagic1 = 0x6C;       // 0b01101100 or 'l' (lowercase 'L')
const BYTE bedFileMagic2 = 0x1B;       // 0b00011011 or <esc>

enum LayoutMode
   {
   LayoutUnknown     = -1
  ,LayoutRowMajor    = 0                  // all elements of a row are sequential in memory
  ,LayoutColumnMajor = 1                  // all elements of a colomn are sequential in memory
  ,LayoutGroupGenotypesByIndividual = 0   // all SNP genotypes for a specific individual are seqential in memory
  ,LayoutGroupGenotypesBySnp = 1          // all Individual's genotypes for a specific SNP are sequential in memory
   };

enum BedGenotype     // integer representation of genotype values in Plink's binary .BED file
   {
   bedHomozygousMinor = 0
  ,bedMissingGenotype = 1
  ,bedHeterozygous    = 2
  ,bedHomozygousMajor = 3
   };

#endif

extern REAL SUFFIX(unknownOrMissing);  // now used by SnpInfo
extern REAL SUFFIX(homozygousPrimaryAllele);
extern REAL SUFFIX(heterozygousAllele);
extern REAL SUFFIX(homozygousSecondaryAllele);
extern REAL SUFFIX(mapBedGenotypeToRealAllele)[4];

class SUFFIX(CBedFile)
   {
public:
   SUFFIX(CBedFile)();
   SUFFIX(~CBedFile)();

   void Open( const string& filename_, size_t cIndividuals_, size_t cSnps_ );                // validate the file matches the extents we expect

   // return the layout mode of the file.  
   //   We work better with files that are LayoutGroupGenotypesBySnp
   LayoutMode  GetLayoutMode();

   // return the compressed length of one line of SNP data (related to cIndividuals)
   // TODO:  MAKE THIS PRIVATE!  Consumers should be using cIndividuals or cSnps
   size_t      CbStride() { return( cbStride ); }

   // return the filename associated with this CBedFile
   const string& Filename() { return( filename ); }

   // read the data for one SNP (idxSnp) into the BYTE buffer pb
   size_t   ReadLine( BYTE *pb, size_t idxSnp );

   // read the genotype for all the individuals in 'list' at the SNP specified by iSNP
   void     ReadGenotypes( size_t iSnp, const vector< size_t >& iIndividualList, REAL* pvOutSNP, uint64_t_ startpos, uint64_t_  outputNumSNPs);

private:
   int      NextChar();
   size_t   Read( BYTE *pb, size_t cbToRead );

   static const size_t   cbHeader = 3;         // 
   string   filename;
   FILE     *pFile;
   vector< BYTE > rgBytes;
   vector< BedGenotype > rgBedGenotypes;
   
   LayoutMode  layout;        // 0=RowMajor(all snps per individual together);
                              // 1=ColumnMajor(all individuals per SNP together in memory)
   size_t   cIndividuals;
   size_t   cSnps;
   size_t   cbStride;

   };


void SUFFIX(ImputeAndZeroMeanSNPs)( 
   REAL *SNPs, 
   const size_t nIndividuals, 
   const size_t nSNPs, 
	const bool betaNotUnitVariance,
	const REAL betaA,
	const REAL betaB
   );

// to be used by cython wrapper
void SUFFIX(readPlinkBedFile)(std::string bed_fn, int inputNumIndividuals, int inputNumSNPs, std::vector<size_t> individuals_idx, std::vector<int> snpIdxList, REAL* out);

/*#endif      // CPlinkBedFile_h
*/
