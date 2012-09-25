"""
========
briandoc
========

Sphinx extension that handles docstrings in the Numpy standard format with some
brian-specific tweaks. [1]

It will:

- Convert Parameters etc. sections to field lists.
- Convert See Also section to a See also entry.
- Renumber references.
- Extract the signature from the docstring, if it can't be determined otherwise.

.. [1] https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""

import sphinx

if sphinx.__version__ < '1.0.1':
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

import os, re, pydoc
from docscrape_sphinx import get_doc_object, SphinxDocString

import inspect

import brian2

BRIAN_CLASS_NAMES = [key for key, value in brian2.__dict__.iteritems()
                     if inspect.isclass(value) and
                     value.__module__.startswith('brian2.')]

def mangle_docstrings(app, what, name, obj, options, lines,
                      reference_offset=[0]):
    cfg = dict()

    if what == 'module':
        # Strip top title
        title_re = re.compile(ur'^\s*[#*=]{4,}\n[a-z0-9 -]+\n[#*=]{4,}\s*',
                              re.I|re.S)
        lines[:] = title_re.sub(u'', u"\n".join(lines)).split(u"\n")
    else:
        doc = get_doc_object(obj, what, u"\n".join(lines), config=cfg)
        lines[:] = unicode(doc).split(u"\n")

    # replace reference numbers so that there are no duplicates
    references = []
    for line in lines:
        line = line.strip()
        m = re.match(ur'^.. \[([a-z0-9_.-])\]', line, re.I)
        if m:
            references.append(m.group(1))

    # start renaming from the longest string, to avoid overwriting parts
    references.sort(key=lambda x: -len(x))
    if references:
        for i, line in enumerate(lines):
            for r in references:
                if re.match(ur'^\d+$', r):
                    new_r = u"R%d" % (reference_offset[0] + int(r))
                else:
                    new_r = u"%s%d" % (r, reference_offset[0])
                lines[i] = lines[i].replace(u'[%s]_' % r,
                                            u'[%s]_' % new_r)
                lines[i] = lines[i].replace(u'.. [%s]' % r,
                                            u'.. [%s]' % new_r)

    reference_offset[0] += len(references)

    for class_name in BRIAN_CLASS_NAMES:
        for i, line in enumerate(lines):
            # Quick and dirty implementation that matches class names and
            # ``classname``. Note that we have to do an in-place modification
            # of the list
            
            # skip things like '.. automethod::'
            if line.strip().startswith('..'):
                continue
            lines[i] = re.sub(r'``%s``' % class_name,
                              ':class:`%s`' % class_name, lines[i])
            lines[i] = re.sub(r'(\s|^|[\(|{\[])(%s)(\s|$|[,\)\}\]]|\. )' % class_name,
                  lambda m: m.group(1) + ':class:`' + m.group(2) + r'`' + m.group(3),
                  lines[i])
            # Replace plurals
            lines[i] = re.sub(r'(\s|^|[\(|{\[])(%s)s(\s|$|[,\)\}\]]|\. )' % class_name,
                  lambda m: m.group(1) + ':class:`' + m.group(2) + r'`\ s' + m.group(3),
                  lines[i])            


def mangle_signature(app, what, name, obj, options, sig, retann):
    # Do not try to inspect classes that don't define `__init__`
    if (inspect.isclass(obj) and
        (not hasattr(obj, '__init__') or
        'initializes x; see ' in pydoc.getdoc(obj.__init__))):
        return '', ''

    if not (callable(obj) or hasattr(obj, '__argspec_is_invalid_')): return
    if not hasattr(obj, '__doc__'): return

    doc = SphinxDocString(pydoc.getdoc(obj))
    if doc['Signature']:
        sig = re.sub(u"^[^(]*", u"", doc['Signature'])
        return sig, u''

def setup(app, get_doc_object_=get_doc_object):
    global get_doc_object
    get_doc_object = get_doc_object_

    app.connect('autodoc-process-docstring', mangle_docstrings)
    app.connect('autodoc-process-signature', mangle_signature)

    # Extra mangling domains
    app.add_domain(NumpyPythonDomain)
    app.add_domain(NumpyCDomain)

#------------------------------------------------------------------------------
# Docstring-mangling domains
#------------------------------------------------------------------------------

from docutils.statemachine import ViewList
from sphinx.domains.c import CDomain
from sphinx.domains.python import PythonDomain

class ManglingDomainBase(object):
    directive_mangling_map = {}

    def __init__(self, *a, **kw):
        super(ManglingDomainBase, self).__init__(*a, **kw)
        self.wrap_mangling_directives()

    def wrap_mangling_directives(self):
        for name, objtype in self.directive_mangling_map.items():
            self.directives[name] = wrap_mangling_directive(
                self.directives[name], objtype)

class NumpyPythonDomain(ManglingDomainBase, PythonDomain):
    name = 'np'
    directive_mangling_map = {
        'function': 'function',
        'class': 'class',
        'exception': 'class',
        'method': 'function',
        'classmethod': 'function',
        'staticmethod': 'function',
        'attribute': 'attribute',
    }

class NumpyCDomain(ManglingDomainBase, CDomain):
    name = 'np-c'
    directive_mangling_map = {
        'function': 'function',
        'member': 'attribute',
        'macro': 'function',
        'type': 'class',
        'var': 'object',
    }

def wrap_mangling_directive(base_directive, objtype):
    class directive(base_directive):
        def run(self):
            env = self.state.document.settings.env

            name = None
            if self.arguments:
                m = re.match(r'^(.*\s+)?(.*?)(\(.*)?', self.arguments[0])
                name = m.group(2).strip()

            if not name:
                name = self.arguments[0]

            lines = list(self.content)
            mangle_docstrings(env.app, objtype, name, None, None, lines)
            self.content = ViewList(lines, self.content.parent)

            return base_directive.run(self)

    return directive

