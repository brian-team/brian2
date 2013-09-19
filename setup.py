#! /usr/bin/env python
'''
Brian2 setup script
'''
import sys
import os
import warnings

# This will automatically download setuptools if it is not already installed
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from setuptools.command.install import install as _install


def generate_preferences(dir):
    '''
    Generate a file in the brian2 installation dictionary containing all the
    preferences with their default values and documentation. This file can be
    used as a starting point for setting user- or project-specific preferences.
    '''
    sys.path.insert(0, dir)
    from brian2.core.preferences import brian_prefs
    # We generate the file directly in the install directory
    try:
        with open(os.path.join(dir,
                               'brian2', 'default_preferences'), 'wt') as f:
            defaults = brian_prefs.defaults_as_file
            f.write(defaults)
    except IOError as ex:
        warnings.warn(('Could not write the default preferences to a '
                       'file: %s' % str(ex)))


class install(_install):
    def run(self):
        # Make sure we first run the build (including running 2to3 for Python3)
        # and then import from the build directory
        _install.run(self)

        self.execute(generate_preferences, (self.install_lib, ),
                     msg='Generating default preferences file')


setup(name='Brian2',
      version='2.0dev',
      packages=find_packages(),
      # include template files
      package_data={'brian2.codegen.runtime.numpy_rt': ['templates/*.py_'],
                    'brian2.codegen.runtime.weave_rt': ['templates/*.cpp',
                                                        'templates/*.h'],
                    'brian2.devices.cpp_standalone': ['templates/*.cpp',
                                                      'templates/*.h']
                    },
      requires=['numpy(>=1.4.1)',
                'scipy(>=0.7.0)',
                'sympy(>=0.7.1)',
                'pyparsing',
                'jinja2(>=2.7)'
                ],
      provides=['brian2'],
      extras_require={'test': ['nosetests>=1.0'],
                      'docs': ['sphinx>=1.0.1']},
      cmdclass={'install': install},
      use_2to3=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ]
      )
