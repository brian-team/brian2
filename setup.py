#! /usr/bin/env python
'''
Preliminary Brian2 setup script
'''

from distutils.core import setup

from brian2.core.preferences import brian_prefs

try:
    with open('./brian2/default_preferences', 'wt') as f:
        defaults = brian_prefs.defaults_as_file
        f.write(defaults)
except IOError as ex:
    raise IOError(('Could not write the default preferences to a file: %s' %
                   str(ex)))

setup(name='Brian2',
      version='2.0dev',
      packages=['brian2',
                'brian2.codegen',
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
      requires=['numpy(>=1.4.1)',
                'scipy(>=0.7.0)',
                'sympy(>=0.7.1)'
                ],                
     )

