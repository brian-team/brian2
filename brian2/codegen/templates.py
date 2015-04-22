'''
Handles loading templates from a directory.
'''
import os
import re
import collections

from jinja2 import Environment, PackageLoader

from brian2.utils.stringtools import (indent, strip_empty_lines,
                                      get_identifiers)


__all__ = ['Templater']

AUTOINDENT_START = '%%START_AUTOINDENT%%'
AUTOINDENT_END = '%%END_AUTOINDENT%%'

def autoindent(code):
    if isinstance(code, list):
        code = '\n'.join(code)
    if not code.startswith('\n'):
        code = '\n'+code
    if not code.endswith('\n'):
        code = code + '\n'
    return AUTOINDENT_START+code+AUTOINDENT_END

def autoindent_postfilter(code):
    lines = code.split('\n')
    outlines = []
    addspaces = 0
    for line in lines:
        if AUTOINDENT_START in line:
            if addspaces>0:
                raise SyntaxError("Cannot nest autoindents")
            addspaces = line.find(AUTOINDENT_START)
            line = line.replace(AUTOINDENT_START, '')
        if AUTOINDENT_END in line:
            line = line.replace(AUTOINDENT_END, '')
            addspaces = 0
        outlines.append(' '*addspaces+line)
    return '\n'.join(outlines)

class Templater(object):
    '''
    Class to load and return all the templates a `CodeObject` defines.
    '''
    def __init__(self, package_name, env_globals=None):
        self.env = Environment(loader=PackageLoader(package_name, 'templates'),
                               trim_blocks=True,
                               lstrip_blocks=True,
                               )
        self.env.globals['autoindent'] = autoindent
        self.env.filters['autoindent'] = autoindent
        if env_globals is not None:
            self.env.globals.update(env_globals)
        for name in self.env.list_templates():
            template = CodeObjectTemplate(self.env.get_template(name),
                                          self.env.loader.get_source(self.env,
                                                                     name)[0])
            setattr(self, os.path.splitext(name)[0], template)


class CodeObjectTemplate(object):
    def __init__(self, template, template_source):
        self.template = template
        self.template_source = template_source
        #: The set of variables in this template
        self.variables = set([])
        #: The indices over which the template iterates completely
        self.iterate_all = set([])
        # This is the bit inside {} for USES_VARIABLES { list of words }
        specifier_blocks = re.findall(r'\bUSES_VARIABLES\b\s*\{(.*?)\}',
                                      template_source, re.M|re.S)
        # Same for ITERATE_ALL
        iterate_all_blocks = re.findall(r'\bITERATE_ALL\b\s*\{(.*?)\}',
                                        template_source, re.M|re.S)
        #: Does this template allow writing to scalar variables?
        self.allows_scalar_write = 'ALLOWS_SCALAR_WRITE' in template_source

        for block in specifier_blocks:
            self.variables.update(get_identifiers(block))
        for block in iterate_all_blocks:
            self.iterate_all.update(get_identifiers(block))
                
    def __call__(self, scalar_code, vector_code, **kwds):
        if scalar_code is not None and len(scalar_code)==1 and scalar_code.keys()[0] is None:
            scalar_code = scalar_code[None]
        if vector_code is not None and len(vector_code)==1 and vector_code.keys()[0] is None:
            vector_code = vector_code[None]
        kwds['scalar_code'] = scalar_code
        kwds['vector_code'] = vector_code
        module = self.template.make_module(kwds)
        if len([k for k in module.__dict__.keys() if not k.startswith('_')]):
            return MultiTemplate(module)
        else:
            return autoindent_postfilter(str(module))


class MultiTemplate(object):
    def __init__(self, module):
        self._templates = {}
        for k, f in module.__dict__.items():
            if not k.startswith('_'):
                s = autoindent_postfilter(str(f()))
                setattr(self, k, s)
                self._templates[k] = s
                
    def __str__(self):
        s = ''
        for k, v in self._templates.items():
            s += k+':\n'
            s += strip_empty_lines(indent(v))+'\n'
        return s
    
    __repr__ = __str__
