################################
:mod:`fastlmm` API Documentation
################################

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing both
single-SNP and SNP-set genome-wide association studies (GWAS) on extremely large data sets.
This release contains the improvements described in Widmer *et al.*, *Scientific Reports* 2014, and tests for epistasis.

See the FaST-LMM website for related software:  
http://research.microsoft.com/en-us/um/redmond/projects/MicrosoftGenomics/Fastlmm/

Our main documentation (including live examples) is also available as ipython notebook:
http://nbviewer.ipython.org/github/MicrosoftGenomics/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb


**************************************************
:mod:`single_snp`
**************************************************

.. autofunction:: fastlmm.association.single_snp

**************************************************
:mod:`single_snp_leave_out_one_chrom`
**************************************************

.. autofunction:: fastlmm.association.single_snp_leave_out_one_chrom

**************************************************
:mod:`epistasis`
**************************************************
.. autofunction:: fastlmm.association.epistasis


**************************************************
:mod:`snp_set`
**************************************************
.. autofunction:: fastlmm.association.snp_set



**************************************************
:mod:`compute_auto_pcs`
**************************************************
.. autofunction:: fastlmm.util.compute_auto_pcs

 
.. only:: html 

***********************
Indices and Tables
***********************

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
