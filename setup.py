from setuptools import setup
import os
import glob
import warnings
import sys
import fnmatch
import subprocess
#from setuptools.command.test import test as TestCommand
from distutils.core import setup
import numpy.distutils.misc_util


if sys.version_info < (3, 0):
    sys.exit('Sorry, Python 2 is not supported')


AUTHOR = 'Justin Pierel'
AUTHOR_EMAIL = 'jpierel@stsci.edu'
VERSION = '0.0.9'
LICENSE = 'BSD'
URL = ''


def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


PACKAGENAME = 'st_phot'
# Add the project-global data
#data_files = []
pkgdatadir = os.path.join(PACKAGENAME, 'wfc3_photometry')
data_files = []
data_files.extend(recursive_glob(pkgdatadir, '*'))
data_files = [f[len(PACKAGENAME)+1:] for f in data_files]



setup(
    name=PACKAGENAME,
    setup_requires=['numpy'],
    install_requires=['numpy', 'astropy','jwst','sncosmo','webbpsf','corner','nestle',
                        'stsci.skypac'],
    packages=[PACKAGENAME],
    package_data={PACKAGENAME: data_files},

    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    # include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    # package_data={'sntd': data_files},
    # include_package_data=True
)
