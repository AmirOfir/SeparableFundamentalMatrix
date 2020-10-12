import glob
import os
from setuptools import setup, Extension
import numpy
this_dir = os.path.dirname(os.path.abspath(__file__))

sources = ['python_entry.cpp', 'line_homography.cpp', 'matching_lines.cpp', 'SFM_finder.cpp', 'sfm_ransac.cpp', "fm_finder.cpp" ]

# site_packages_dir = numpy.__file__[:numpy.__file__.index('site-packages')+13]

main_folder = numpy.__file__[:numpy.__file__.lower().index('lib')]
library_dir = os.path.join( main_folder, 'lib')
include_dirs = [os.path.join(numpy.get_include(), 'numpy'), os.path.join( main_folder, 'include')]
cv_version=''

setup(
    name='sepfm',
    version='1.0',
    description='Separable Fundamental Matrix finder',
    ext_modules=[
        Extension(
            'sepfm',
            sources=sources,
            include_dirs=include_dirs,
            libraries=['opencv_core'+cv_version, 'opencv_imgproc'+cv_version, 'opencv_calib3d' + cv_version],
            define_macros=[('AccessFlag', 'int'), ('VERSION', '3')],
            py_limited_api=True)
    ],
)
