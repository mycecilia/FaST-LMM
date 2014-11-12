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
* CPlinkBedFile - {PLINK BED File Access Class}
*
*         File Name:   CPlinkBedFile.cpp
*           Version:   2.00
*            Author:   
*     Creation Date:   18 Nov 2010
*     Revision Date:   14 Aug 2013
*
*    Module Purpose:   This file implements the CPlinkBedFile 
*                      class for FastLmmC
*
*                      A .BED file contains compressed binary genotype 
*                         values for individuals by SNPs.  
*
*    Change History:   Version 2.00: Reworked to be wrapped in python version by Chris Widmer  (chris@shogun-toolbox.org)
*
* Test Files: 
*/

/*
* Include Files
*/
#include "CPlinkBedFileT.h"
#include <iostream>
#include <stdio.h>
#include <math.h> 
#include <stdlib.h>

// 0 and 2 are flipped (wrt C++ fastlmm) in order to agree to python code
REAL SUFFIX(unknownOrMissing) = std::numeric_limits<REAL>::quiet_NaN();  // now used by SnpInfo
REAL SUFFIX(homozygousPrimaryAllele) = 2;                // Major Allele
REAL SUFFIX(heterozygousAllele) = 1;                     
REAL SUFFIX(homozygousSecondaryAllele) = 0;              // Minor Allele ()

REAL SUFFIX(mapBedGenotypeToRealAllele)[4] = { 
	SUFFIX(homozygousSecondaryAllele),       // look-up 0
	SUFFIX(unknownOrMissing),                // look-up 1
	SUFFIX(heterozygousAllele),              // look-up 2
	SUFFIX(homozygousPrimaryAllele),         // look-up 3
};

SUFFIX(CBedFile)::SUFFIX(CBedFile)()
{
	layout = LayoutUnknown;    // layout describes the matrix layout on disk
	// 0=RowMajor(all snps per individual together);
	// 1=ColumnMajor(all individuals per SNP together in memory)
	cIndividuals = 0;
	cSnps        = 0;
	cbStride     = 0;

}

SUFFIX(CBedFile)::SUFFIX(~CBedFile)()
{
	if ( pFile )
	{
		fclose( pFile );
		pFile = NULL;
	}
}


void SUFFIX(CBedFile)::Open( const string& filename_, size_t cIndividuals_, size_t cSnps_ )
{
	if ( filename_.empty() )
	{
		printf( "Could not create BedFile Reader.  Parameter 'filename' is zero length string" );
	}

	filename = filename_;         // TODO: removed FullPath
	cIndividuals = cIndividuals_;
	cSnps = cSnps_;

	pFile = fopen( filename.c_str(), "rb" );  // read in binary to ensure ftell works right
	if ( !pFile )
	{
		printf( "Cannot open input file [%s].\n", filename.c_str()); //TODO: removed errorNO
	}

	//  Verify 'magic' number
	unsigned char rd1 = NextChar();
	unsigned char rd2 = NextChar();
	if ( (bedFileMagic1 != rd1) || (bedFileMagic2 != rd2))
	{
		printf( "Ill-formed BED file [%s]."
			"\n  BED file header is incorrect."
			"\n  Expected magic number of 0x%02x 0x%02x, found 0x%02x 0x%02x", 
			filename.c_str(), bedFileMagic1, bedFileMagic2, rd1, rd2 );
	}

	// Verify 'mode' is valid
	unsigned char rd3 = NextChar();
	switch( rd3 )
	{
	case 0:  // mode = 'IndividualMajor' or RowMajor
		layout = LayoutGroupGenotypesByIndividual;   // all SNPs per individual are sequential in memory
		cbStride = (cSnps + 3)/4;                    // 4 genotypes per byte so round up
		break;
	case 1:  // mode = 'SnpMajor' or ColumnMajor
		layout = LayoutGroupGenotypesBySnp;          // all individuals per SNP are sequential in memory
		cbStride = (cIndividuals + 3)/4;             // 4 genotypes per byte so round up
		break;
	default:
		printf( "Ill-formed BED file [%s].  BED file header is incorrect.  Expected mode to be 0 or 1, found %d", filename.c_str(), rd3 );
		break;
	}

	// allocate the read buffer for a SNP
	rgBytes.resize( cbStride );
	rgBedGenotypes.resize( cIndividuals, bedMissingGenotype );
}

LayoutMode  SUFFIX(CBedFile)::GetLayoutMode()
{
	return( layout );
}

int SUFFIX(CBedFile)::NextChar()
{
	int value = fgetc( pFile );
	if ( value == EOF )
	{
		printf( "Ill-formed BED file [%s]. Encountered EOF before expected.", filename.c_str() );
	}
	return( (unsigned char)value );
}

size_t SUFFIX(CBedFile)::Read( BYTE *pb, size_t cbToRead )
{
	size_t cbRead = fread( pb, 1, cbToRead, pFile );
	if ( cbRead != cbToRead )
	{
		if ( feof( pFile ) )
		{
			printf( "Encountered EOF before expected in BED file. Ill-formed BED file [%s]", filename.c_str() );
		}
		int err = ferror( pFile );
		if ( err )
		{
			printf( "Encountered a file error %d in BED file [%s]", err, filename.c_str() );
		}
	}
	return( cbRead );
}

size_t SUFFIX(CBedFile)::ReadLine( BYTE *pb, size_t idx )
{
	long long fpos = cbHeader + (idx*cbStride);
#ifdef _WIN32
	long long fposCur = _ftelli64( pFile );
#else
	long long fposCur = ftello64( pFile );
#endif
	if ( fpos != fposCur )
	{
#ifdef _WIN32
		_fseeki64( pFile, fpos, SEEK_SET );
#else
		fseeko64( pFile, fpos, SEEK_SET );
#endif
	}

	size_t cbRead = Read( pb, cbStride );
	return( cbRead );
}

/*
* Read the genotype for all the individuals in iidList at the SNP specified by iSNP
*   and store the results in pvOut
*/
void SUFFIX(CBedFile)::ReadGenotypes( size_t iSnp, const vector< size_t >& idxIndividualList, REAL* pvOut, uint64_t_ startpos, uint64_t_  outputNumSNPs)
{
	//fprintf(stdout,"reading iSnp=%d w/ cIndividuals=%d and startpos=%d\n",iSnp,cIndividuals,startpos);
	ReadLine( &rgBytes[0], iSnp );
	// 'decompress' the genotype information
	size_t iIndividual = 0;
	for ( size_t ib = 0; ib < cbStride; ++ib )
	{
		BYTE genotypeByte = rgBytes[ ib ];

		// manually unrolled loop to decompress this byte
		if ( iIndividual < cIndividuals ) rgBedGenotypes[ iIndividual++ ] = (BedGenotype)( genotypeByte       & 0x03);
		if ( iIndividual < cIndividuals ) rgBedGenotypes[ iIndividual++ ] = (BedGenotype)((genotypeByte >> 2) & 0x03);
		if ( iIndividual < cIndividuals ) rgBedGenotypes[ iIndividual++ ] = (BedGenotype)((genotypeByte >> 4) & 0x03);
		if ( iIndividual < cIndividuals ) rgBedGenotypes[ iIndividual++ ] = (BedGenotype)((genotypeByte >> 6) & 0x03);
	}
	for ( size_t i=0; i<idxIndividualList.size(); ++i )
	{
		size_t idx = idxIndividualList[ i ];
		//fprintf(stdout,"iSnp=%d, iIID=%d\n",iSnp,idx);
#ifdef ORDERF
		uint64_t_ out_idx = startpos + i;
#else
		uint64_t_ out_idx = startpos + i * outputNumSNPs;
#endif
		pvOut[ out_idx ] = SUFFIX(mapBedGenotypeToRealAllele)[ rgBedGenotypes[ idx ] ];
	}
}

const REAL SUFFIX(_PI) = 2.0*acos(0.0);
const REAL SUFFIX(_halflog2pi)=(REAL)0.5*log((REAL)2.0*SUFFIX(_PI));
const REAL SUFFIX(coeffsForLogGamma)[] = { 12.0, -360.0, 1260.0, -1680.0, 1188.0 };

const REAL SUFFIX(eps_rank)=(REAL)3E-8;


// Gamma and LogGamma

// Use GalenA's implementation of LogGamma - it's faster!
/// <summary>Returns the log of the gamma function</summary>
/// <param name="x">Argument of function</param>
/// <returns>Log Gamma(x)</returns>
/// <remarks>Accurate to eight digits for all x.</remarks>
REAL SUFFIX(logGamma)(REAL x)
{
	if (x <= (REAL)0.0){
		printf("LogGamma arg=%f must be > 0.",x);
		throw(1);
	}

	REAL res = (REAL)0.0;
	if (x < (REAL)6.0)
	{
		int toAdd = (int)floor(7 - x);
		REAL v2 = (REAL)1.0;
		for (int i = 0; i < toAdd; i++)
		{
			v2 *= (x + i);
		}
		res = -log(v2);
		x += toAdd;
	}
	x -= (REAL)1.0;

	res += SUFFIX(_halflog2pi) + (x + (REAL)0.5) * log(x) - x;

	// correction terms
	REAL xSquared = x * x;
	REAL pow = x;
	for (int i=0; i<5; ++i)   //the length of the coefficient array is 5.
	{
		REAL newRes = res + (REAL)1.0 / (SUFFIX(coeffsForLogGamma)[i] * pow);
		if (newRes == res)
		{
			return res;
		}
		res = newRes;
		pow *= xSquared;
	}

	return res;
}

// Beta and LogBeta
/// <summary>Computes the log beta function</summary>
double SUFFIX(LogBeta)(REAL x, REAL y)
{
	if (x <= 0.0 || y <= 0.0){
		printf("LogBeta args must be > 0.");
		throw(1);
	}
	return SUFFIX(logGamma)(x) + SUFFIX(logGamma)(y) - SUFFIX(logGamma)(x + y);
}

/// <summary>Probability distribution function</summary>
/// <param name="x">Value at which to compute the pdf</param>
/// <param name="a">Shape parameter (alpha)</param>
/// <param name="b">Shape parameter (beta)</param>
REAL SUFFIX(BetaPdf)(REAL x, REAL a, REAL b){
   if (a <= 0 || b <= 0){
      printf("Beta.Pdf parameters, a and b, must be > 0");
      throw(1);
   }

   if (x > 1) return 0;
   if (x < 0) return 0;

   REAL lnb = SUFFIX(LogBeta)(a, b);
   return exp((a - 1) * log(x) + (b - 1) * log(1 - x) - lnb);
}



/*
* Parameters: 
* SNPs [nIndividuals by nSNPs]:
*                       Matrix stored in column-major order. 
*                       This will hold the result.
*                       NaNs will be set to 0.0 in the result.
*/
void SUFFIX(ImputeAndZeroMeanSNPs)( 
	REAL *SNPs, 
	const size_t nIndividuals, 
	const size_t nSNPs, 
	const bool betaNotUnitVariance,
	const REAL betaA,
	const REAL betaB
	)
{
	bool seenSNC = false; //Keep track of this so that only one warning message is reported
#ifdef ORDERF

	for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
	{

		REAL n_observed = 0.0;
		REAL sum_s  = 0.0;      //the sum of a SNP over all observed individuals
		REAL sum2_s = 0.0;      //the sum of the squares of the SNP over all observed individuals

		size_t end = nIndividuals;
		size_t delta = 1;
		for( size_t ind = 0; ind < end; ind+=delta )
		{
			if (SNPs[ind] == SNPs[ind])
			{
				//check for not NaN
				sum_s += SNPs[ ind ];
				sum2_s+= SNPs[ ind ] * SNPs[ ind ];
				++n_observed;
			}
		}

		if ( n_observed < 1.0 )
		{
			printf( "No individual observed for the SNP.\n");
		}

		REAL mean_s  = sum_s  / n_observed;    //compute the mean over observed individuals for the current SNP
		REAL mean2_s = sum2_s / n_observed;    //compute the mean of the squared SNP

		//When beta standardization is being done, check that data is 0,1,2
		if (betaNotUnitVariance && sum_s <= (REAL)0.0)
		{
			REAL freqT = sum_s/n_observed;
			fprintf(stderr, "Observed SNP freq is %.2f. for a SNPs[:][%i]\n", freqT, iSnp );
			exit(1);
		}


		//The SNP frequency
		REAL freq = (sum_s) / (n_observed * (REAL)2.0);   //compute snp-freq as in the Visscher Height paper (Nat Gen, Yang et al 2010).

		if ((freq != freq) || betaNotUnitVariance && ((freq >= (REAL)1.0) || (freq <= (REAL)0.0)))
		{
			if (!seenSNC)
			{
				seenSNC = true;
				fprintf(stderr, "Illegal SNP frequency: %.2f for SNPs[:][%i]\n", freq, iSnp);
			}
		}


		REAL variance = mean2_s-mean_s * mean_s;        //By the Cauchy Shwartz inequality this should always be positive
		REAL std = sqrt( variance );                    //The SNP frequency

		bool isSNC = false;
		if ( (std != std) || (std <= (REAL)0.0) )
		{
			// a std == 0.0 means all SNPs have the same value (no variation or Single Nucleotide Constant (SNC))
			//   however ALL SNCs should have been removed in previous filtering steps
			//   This test now prevents a divide by zero error below
			std = 1.0;
			isSNC = true;
			if (!seenSNC)
			{
				seenSNC = true;
				fprintf(stderr, "std=.%2f has illegal value for SNPs[:][%i]\n", std, iSnp );
			}

		}

		if (betaNotUnitVariance && freq > .5)
		{
			freq = 1.0 - freq;
		}

		for( size_t ind = 0; ind < end; ind+=delta )
		{
			//check for NaN
			if ( (SNPs[ ind ]!=SNPs[ ind ]) || isSNC)
			{
				SNPs[ ind ] = 0.0;
			}
			else
			{
				SNPs[ ind ] -= mean_s;     //subtract the mean from the data
				if (betaNotUnitVariance )
				{
					REAL rT = SUFFIX(BetaPdf)( freq, betaA, betaB );
					//fprintf(stderr, "BetaPdf(%f,%f,%f)=%f\n",  freq, betaA, betaB, rT);
					SNPs[ ind ] *= rT;
				}
				else
				{
					SNPs[ ind ] /= std;        //unit variance as well
				}

			}
		}

		SNPs += nIndividuals;
	}

#else //Order C


	// Make one pass through the data (by individual, because that is how it is laid out), collecting statistics
	std::vector<REAL> n_observed(nSNPs); //                                                C++ inits to 0's
	std::vector<REAL> sum_s(nSNPs);      //the sum of a SNP over all observed individuals. C++ inits to 0's
	std::vector<REAL> sum2_s(nSNPs);     //the sum of the squares of the SNP over all observed individuals.     C++ inits to 0's

	for( size_t ind = 0; ind < nIndividuals; ++ind)
	{
		size_t rowStart = ind * nSNPs;
		for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
		{
			REAL value = SNPs[rowStart+iSnp];
			if ( value == value )
			{
				sum_s[iSnp] += value;
				sum2_s[iSnp] += value * value;
				++n_observed[iSnp];
			}
		}
	}


	std::vector<REAL> mean_s(nSNPs);  //compute the mean over observed individuals for the current SNP
	std::vector<REAL> mean2_s(nSNPs); //compute the mean of the squared SNP
	std::vector<REAL> std(nSNPs); //the standard deviation
	std::vector<REAL> freq(nSNPs); //The SNP frequency
	std::vector<bool> isSNC(nSNPs); // Is this a SNC (C++ inits to false)

	for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
	{
		if ( n_observed[iSnp] < 1.0 )
		{
			printf( "No individual observed for the SNP.\n");
		}

		mean_s[iSnp]  = sum_s[iSnp]  / n_observed[iSnp];    //compute the mean over observed individuals for the current SNP
		mean2_s[iSnp] = sum2_s[iSnp] / n_observed[iSnp];    //compute the mean of the squared SNP

		//When beta standardization is being done, check that data is 0,1,2
		if (betaNotUnitVariance && sum_s[iSnp] <= (REAL)0.0)
		{
			REAL freqT = sum_s[iSnp]/n_observed[iSnp];
			fprintf(stderr, "Observed SNP freq is %.2f. for a SNPs[:][%i]\n", freqT, iSnp );
			exit(1);
		}

		freq[iSnp] = (sum_s[iSnp]) / (n_observed[iSnp] * (REAL)2.0);   //compute snp-freq[iSnp] as in the Visscher Height paper (Nat Gen, Yang et al 2010).

		if ((freq[iSnp] != freq[iSnp]) || betaNotUnitVariance && ((freq[iSnp] >= (REAL)1.0) || (freq[iSnp] <= (REAL)0.0)))
		{
			if (!seenSNC)
			{
				seenSNC = true;
				fprintf(stderr, "Illegal SNP frequency: %.2f for SNPs[:][%i]\n", freq[iSnp], iSnp);
			}
		}


		REAL variance = mean2_s[iSnp]-mean_s[iSnp] * mean_s[iSnp];        //By the Cauchy Shwartz inequality this should always be positive
		std[iSnp] = sqrt( variance );

		if ( (std[iSnp] != std[iSnp]) || (std[iSnp] <= (REAL)0.0) )
		{
			// a std == 0.0 means all SNPs have the same value (no variation or Single Nucleotide Constant (SNC))
			//   however ALL SNCs should have been removed in previous filtering steps
			//   This test now prevents a divide by zero error below
			std[iSnp] = 1.0;
			isSNC[iSnp] = true;
			if (!seenSNC)
			{
				seenSNC = true;
				fprintf(stderr, "std=.%2f has illegal value for SNPs[:][%i]\n", std[iSnp], iSnp );
			}
		}

		if (betaNotUnitVariance && freq[iSnp] > .5)
		{
			freq[iSnp] = 1.0 - freq[iSnp];
		}
	}

	for( size_t ind = 0; ind < nIndividuals; ++ind)
	{
		size_t rowStart = ind * nSNPs;
		for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
		{
			REAL value = SNPs[rowStart+iSnp];
			//check for NaN
			if ( (value != value) || isSNC[iSnp])
			{
				value = 0.0;
			}
			else
			{
				value -= mean_s[iSnp];     //subtract the mean from the data
				if (betaNotUnitVariance )
				{
					REAL rT = SUFFIX(BetaPdf)( freq[iSnp], betaA, betaB );
					//fprintf(stderr, "BetaPdf(%f,%f,%f)=%f\n",  freq, betaA, betaB, rT);
					value *= rT;
				}
				else
				{
					value /= std[iSnp];        //unit variance as well
				}
			}
			SNPs[rowStart+iSnp] = value;
		}
	}
#endif
}

// wrapper to be used from cython
void SUFFIX(readPlinkBedFile)(std::string bed_fn, int inputNumIndividuals, int inputNumSNPs, std::vector<size_t> individuals_idx, std::vector<int> snpIdxList, REAL* out)
{
	uint64_t_ N = inputNumIndividuals;
	uint64_t_ outputNumSNPs = snpIdxList.size();

	SUFFIX(CBedFile) bedFile = SUFFIX(CBedFile)();
	bedFile.Open(bed_fn, inputNumIndividuals, inputNumSNPs);

	for (size_t i=0; i!=snpIdxList.size(); i++){
		int idx = snpIdxList[i];

#ifdef ORDERF
		uint64_t_ startpos = ((uint64_t_)i) * individuals_idx.size();
#else
		uint64_t_ startpos = ((uint64_t_)i);
#endif
		bedFile.ReadGenotypes( idx, individuals_idx, out, startpos, outputNumSNPs);
	}

}
