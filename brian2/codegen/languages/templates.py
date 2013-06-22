'''
Handles loading templates from a directory.
'''
from brian2.utils.stringtools import indent, deindent, strip_empty_lines
from jinja2 import Template, Environment, FileSystemLoader
import os

__all__ = ['LanguageTemplater']

class LanguageTemplater(object):
    '''
    Class to load and return all the templates a language defines.
    '''
    def __init__(self, basedir):
        self.basedir = basedir
        self.env = Environment(loader=FileSystemLoader(basedir),
                               trim_blocks=True,
                               lstrip_blocks=True,
                               )
        for name in self.env.list_templates():
            template = LanguageTemplate(self.env.get_template(name))
            setattr(self, os.path.splitext(name)[0], template)


class LanguageTemplate(object):
    def __init__(self, template):
        self.template = template
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


if __name__=='__main__':
    lt = LanguageTemplater('python/templates')
    print lt.reset('a=b\nc=d')
