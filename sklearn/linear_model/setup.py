import os
from os.path import join

import numpy

from sklearn._build_utils import get_blas_info


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('linear_model', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()
    def_macros = blas_info.get('define_macros')  # FJ
    def_macros.append(('CYTHON_TRACE_NOGIL', '1'))  # FJ
    def_macros.append(('CYTHON_TRACE', '1'))  # FJ

    if os.name == 'posix':
        cblas_libs.append('m')

    config.add_extension('cd_fast', sources=['cd_fast.c'],
                         libraries=cblas_libs,
                         include_dirs=[join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []), **blas_info)

    config.add_extension('sgd_fast',
                         sources=['sgd_fast.c'],
                         include_dirs=[join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         libraries=cblas_libs,
#                         define_macros=[('CYTHON_TRACE', '1')],  # FJ
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []),
                         **blas_info)

    config.add_extension('sag_fast',
                         sources=['sag_fast.c'],
                         include_dirs=numpy.get_include())

    # add other directories
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
