#! /usr/bin/env python
'''
Brian2 setup script
'''
# This will automatically download setuptools if it is not already installed
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages


long_description = '''
Brian2 is a simulator for spiking neural networks available on almost all platforms.
The motivation for this project is that a simulator should not only save the time of
processors, but also the time of scientists.

It is the successor of Brian1 and shares its approach of being highly flexible
and easily extensible. It is based on a code generation framework that allows
to execute simulations using other programming languages and/or on different
devices.

We currently consider this software to be in the alpha status, please report
issues to the github issue tracker (https://github.com/brian-team/brian2/issues) or to the
brian-development mailing list (http://groups.google.com/group/brian-development/)

Documentation for Brian2 can be found at http://brian2.readthedocs.org
'''

setup(name='Brian2',
      version='2.0a5',
      packages=find_packages(),
      # include template files
      package_data={'brian2.codegen.runtime.numpy_rt': ['templates/*.py_'],
                    'brian2.codegen.runtime.weave_rt': ['templates/*.cpp',
                                                        'templates/*.h'],
                    'brian2.devices.cpp_standalone': ['templates/*.cpp',
                                                      'templates/*.h',
                                                      'brianlib/*.cpp',
                                                      'brianlib/*.h']
                    },
      install_requires=['numpy>=1.4.1',
                        'scipy>=0.7.0',
                        'sympy>=0.7.2',
                        'pyparsing',
                        'jinja2>=2.7'
                       ],
      provides=['brian2'],
      extras_require={'test': ['nosetests>=1.0'],
                      'docs': ['sphinx>=1.0.1', 'sphinxcontrib-issuetracker']},
      use_2to3=True,
      url='http://www.briansimulator.org/',
      description='A clock-driven simulator for spiking neural networks',
      long_description=long_description,
      author='Marcel Stimberg, Dan Goodman, Romain Brette',
      author_email='Romain.Brette at ens.fr',
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
