'''
Handles loading templates from a directory.
'''
import os
import re

from jinja2 import Template, Environment, FileSystemLoader, PackageLoader

from brian2.utils.stringtools import (indent, deindent, strip_empty_lines,
                                      get_identifiers)


__all__ = ['Templater']

class Templater(object):
    '''
    Class to load and return all the templates a `CodeObject` defines.
    '''
    def __init__(self, package_name, env_globals=None):
        self.env = Environment(loader=PackageLoader(package_name, 'templates'),
                               trim_blocks=True,
                               lstrip_blocks=True,
                               )
        if env_globals is not None:
            self.env.globals.update(env_globals)
        for name in self.env.list_templates():
            template = CodeObjectTemplate(self.env.get_template(name))
            setattr(self, os.path.splitext(name)[0], template)


class CodeObjectTemplate(object):
    def __init__(self, template):
        self.template = template
        res = self([''])
        if isinstance(res, str):
            temps = [res]
        else:
            temps = res._templates.values()
        #: The set of words in this template
        self.words = set([])
        for v in temps:
            self.words.update(get_identifiers(v))
        #: The set of variables in this template
        self.variables = set([])
        #: The indices over which the template iterates completely
        self.iterate_all = set([])
        for v in temps:
            # This is the bit inside {} for USES_VARIABLES { list of words }
            specifier_blocks = re.findall(r'\bUSES_VARIABLES\b\s*\{(.*?)\}',
                                          v, re.M|re.S)
            # Same for ITERATE_ALL
            iterate_all_blocks = re.findall(r'\bITERATE_ALL\b\s*\{(.*?)\}',
                              v, re.M|re.S)
            for block in specifier_blocks:
                self.variables.update(get_identifiers(block))
            for block in iterate_all_blocks:
                self.iterate_all.update(get_identifiers(block))
                
    def __call__(self, code_lines, **kwds):
        kwds['code_lines'] = code_lines
        module = self.template.make_module(kwds)
        if len([k for k in module.__dict__.keys() if not k.startswith('_')]):
            return MultiTemplate(module)
        else:
            return str(module)


class MultiTemplate(object):
    def __init__(self, module):
        self._templates = {}
        for k, f in module.__dict__.items():
            if not k.startswith('_'):
                s = str(f())
                setattr(self, k, s)
                self._templates[k] = s
                
    def __str__(self):
        s = ''
        for k, v in self._templates.items():
            s += k+':\n'
            s += strip_empty_lines(indent(v))+'\n'
        return s
    
    __repr__ = __str__
