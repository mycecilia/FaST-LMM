"""
file to set up python package, see http://docs.python.org/2/distutils/setupscript.html for details.
"""

import platform
import os
import sys
import shutil

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.clean import clean as Clean

try:
	from Cython.Distutils import build_ext
except Exception:
	print "cython needed for installation, please install cython first"
	sys.exit()

try:
	import numpy
except Exception:
	print "numpy needed for installation, please install numpy first"
	sys.exit()

def readme():
    with open('INSTALL_README.txt') as f:
       return f.read()

class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('fastlmm'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                             #or filename.endswith('.dll')
                             #or filename.endswith('.pyc')
                             ):
                    os.unlink(os.path.join(dirpath, filename))

# set up macro
if "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
else:
    macros = [("_UNIX", "1")]

ext = [Extension("pysnptools.snpreader.wrap_plink_parser", ["pysnptools/snpreader/wrap_plink_parser.pyx", "pysnptools/snpreader/CPlinkBedFile.cpp"], language="c++", define_macros=macros)]

setup(
    name='pysnptools',
    version='0.1',
    description='PySnpTools',
    long_description=readme(),
    keywords='SnpReader SnpData',
    url='',
    author='MSR',
    author_email='...',
    license='non-commercial (MSR-LA)',
    packages=[
        "pysnptools/altset_list",
        "pysnptools/snpreader",
        "pysnptools/standardizer",
        "pysnptools"
	],
	package_data={
                 },
    #install_requires=['cython', 'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib'],
    #zip_safe=False,
    # extensions
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand},
    ext_modules = ext,
	include_dirs = [numpy.get_include()],
  )

