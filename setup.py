import platform
import os
import sys
import shutil
from setuptools import setup, Extension 
from setuptools import setup
from distutils.command.clean import clean as Clean
from Cython.Distutils import build_ext
import numpy

# Version number
version = '0.2.14'


def readme():
    with open('README.md') as f:
       return f.read()

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
else:
    macros = [("_UNIX", "1")]

ext_modules = [Extension(name="fastlmm.util.stats.quadform.qfc_src.wrap_qfc",
                         language="c++",
                         sources=["fastlmm/util/stats/quadform/qfc_src/wrap_qfc.pyx", "fastlmm/util/stats/quadform/qfc_src/QFC.cpp"],
                         include_dirs=[numpy.get_include()],
                         define_macros=macros)]


class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (   filename.endswith('.so')
                    or filename.endswith('.pyd')
                    or filename.find("wrap_qfc.cpp") != -1 # remove automatically generated source file
                    #or filename.endswith('.dll')
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print "removing", tmp_fn
                    os.unlink(tmp_fn)

#python setup.py sdist bdist_wininst upload
setup(
    name='fastlmm',
    version=version,
    description='Fast GWAS',
    long_description=readme(),
    keywords='gwas bioinformatics LMMs MLMs',
    url="http://research.microsoft.com/en-us/um/redmond/projects/mscompbio/fastlmm/",
    author='MSR',
    author_email='fastlmm@microsoft.com',
    license='Apache 2.0',
    packages=[
        "fastlmm/association/tests",
        "fastlmm/association",
        "fastlmm/external/util",
        "fastlmm/external",
        "fastlmm/feature_selection",
        "fastlmm/inference",
        "fastlmm/pyplink/altset_list", #old snpreader
        "fastlmm/pyplink/snpreader", #old snpreader
        "fastlmm/pyplink/snpset", #old snpreader
        "fastlmm/pyplink", #old snpreader
        "fastlmm/util/runner",
        "fastlmm/util/stats/quadform",
        "fastlmm/util/stats/quadform/qfc_src",
        "fastlmm/util/standardizer",
        "fastlmm/util/stats",
        "fastlmm/util",
        "fastlmm",
	],
	package_data={"fastlmm/association" : [
                       "Fastlmm_autoselect/FastLmmC.exe",
                       "Fastlmm_autoselect/libiomp5md.dll",
                       "Fastlmm_autoselect/fastlmmc",
                       "Fastlmm_autoselect/FastLmmC.Manual.pdf"],
                  "fastlmm/feature_selection" : [
                       "examples/bronze.txt",
                       "examples/ScanISP.Toydata.config.py",
                       "examples/ScanLMM.Toydata.config.py",
                       "examples/ScanOSP.Toydata.config.py",
                       "examples/toydata.5chrom.bed",
                       "examples/toydata.5chrom.bim",
                       "examples/toydata.5chrom.fam",
                       "examples/toydata.bed",
                       "examples/toydata.bim",
                       "examples/toydata.cov",
                       "examples/toydata.dat",
                       "examples/toydata.fam",
                       "examples/toydata.iidmajor.hdf5",
                       "examples/toydata.map",
                       "examples/toydata.phe",
                       "examples/toydata.shufflePlus.phe",
                       "examples/toydata.sim",
                       "examples/toydata.snpmajor.hdf5",
                       "examples/toydataTest.phe",
                       "examples/toydataTrain.phe"
					   ]
                 },
    install_requires = ['cython', 'numpy', 'scipy', 'pandas', 'scikit-learn>=0.16', 'matplotlib', 'pysnptools'],
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand},
    ext_modules = ext_modules,
  )

