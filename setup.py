#! /usr/bin/env python
'''
Preliminary Brian2 setup script
'''
import sys
import os

from distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

class generate_preferences(build_py):
    def run(self):
        # Make sure we first run the build (including running 2to3 for Python3)
        # and then import from the build directory
        build_py.run(self)
        sys.path.insert(0, self.build_lib)
        from brian2.core.preferences import brian_prefs
        
        # We generate the file directly in the build directory
        try:
            with open(os.path.join(self.build_lib,
                                   'brian2', 'default_preferences'), 'wt') as f:
                defaults = brian_prefs.defaults_as_file
                f.write(defaults)
        except IOError as ex:
            raise IOError(('Could not write the default preferences to a '
                           'file: %s' % str(ex)))


setup(name='Brian2',
      version='2.0dev',
      packages=['brian2',
                'brian2.codegen',
                'brian2.codegen.functions',
                'brian2.codegen.languages',
                'brian2.codegen.languages.cpp',
                'brian2.codegen.languages.python',
                'brian2.core',
                'brian2.equations',
                'brian2.groups',
                'brian2.memory',
                'brian2.monitors',
                'brain2.parsing',
                'brian2.sphinxext',
                'brian2.stateupdaters',
                'brian2.tests',
                'brian2.units',
                'brian2.utils'],
      package_data={'brian2.codegen.languages.cpp': ['templates/*.cpp'],
                    'brian2.codegen.languages.python': ['templates/*.py_']},
      requires=['numpy(>=1.4.1)',
                'scipy(>=0.7.0)',
                'sympy(>=0.7.1)'
                ],
      cmdclass = {'build_py': generate_preferences}                
     )
