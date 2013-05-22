#! /usr/bin/env python
'''
Preliminary Brian2 setup script
'''
import sys

from distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

from distutils.command.install_data import install_data

class install_preferences(install_data):
    def run(self):        
        # Make sure we load the brian2 packages from the installation
        # directory, not from the source directory (important for Python 3)
        sys.path.insert(0, self.install_dir)
        from brian2.core.preferences import brian_prefs
        
        try:
            with open('./brian2/default_preferences', 'wt') as f:
                defaults = brian_prefs.defaults_as_file
                f.write(defaults)
        except IOError as ex:
            raise IOError(('Could not write the default preferences to a '
                           'file: %s' % str(ex)))
        
        # Now, run the original install_data command which copies the
        # generated file to the appropriate place
        install_data.run(self)

setup(name='Brian2',
      version='2.0dev',
      packages=['brian2',
                'brian2.codegen',
                'brian2.codegen.functions',
                'brian2.codegen.languages',
                'brian2.core',
                'brian2.equations',
                'brian2.groups',
                'brian2.memory',
                'brian2.monitors',
                'brian2.sphinxext',
                'brian2.stateupdaters',
                'brian2.tests',
                'brian2.units',
                'brian2.utils'],
      package_data={'brian2': ['default_preferences']},
      # We fake data_files here, because otherwise install_data is not run 
      data_files = [('', [])], 
      requires=['numpy(>=1.4.1)',
                'scipy(>=0.7.0)',
                'sympy(>=0.7.1)'
                ],
      cmdclass = {'build_py': build_py,
                  'install_data': install_preferences}                
     )
