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
import re
import pydoc
import inspect
from docutils import statemachine
from docutils.parsers.rst import directives, Directive

import sphinx
from sphinx.roles import XRefRole
if sphinx.__version__ < '1.0.1':
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

from brian2.core.preferences import brian_prefs

from .docscrape_sphinx import get_doc_object, SphinxDocString


class BrianPrefsDirective(Directive):
    '''
    A sphinx 'Directive' for automatically generated documentation of Brian preferences.
    '''
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        rawtext = brian_prefs.documentation
        include_lines = statemachine.string2lines(rawtext,
                                                  convert_whitespace=True)
        self.state_machine.insert_input(include_lines, 'Brian preferences')
        return []


def brian_prefs_role(role, rawtext, text, lineno, inliner,
                       options={}, content=[]):
    '''
    A sphinx role, allowing to link to Brian preferences using a
    ``:bpref:`preference_name``` syntax.
    '''
    if not text in brian_prefs._values:
        msg = inliner.reporter.error(
            'Unknown brian preference: %s' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    text = '%s <brian-pref-%s>' % (text, text.replace('_', '-'))
    # Use sphinx's cross-reference role
    xref = XRefRole(warn_dangling=True)
    return xref('std:ref', rawtext, text, lineno, inliner, options, content)


def mangle_docstrings(app, what, name, obj, options, lines,
                      reference_offset=[0]):
    cfg = dict()

    if what == 'module':
        # Strip top title
        title_re = re.compile(ur'^\s*[#*=]{4,}\n[a-z0-9 -]+\n[#*=]{4,}\s*',
                              re.I|re.S)
        lines[:] = title_re.sub(u'', u"\n".join(lines)).split(u"\n")
        exported_members = getattr(obj, '__all__', None)
        if exported_members:
            lines.append('*Exported members:* ')
            # do not print more than 25 members            
            lines.append(', '.join(['`%s`' % member for
                                    member in exported_members[:25]]))
            if len(exported_members) > 25:
                lines.append('... (%d more members)' % (len(exported_members) - 25))
            
            lines.append('')
    else:
        doc = get_doc_object(obj, what, u"\n".join(lines), name=name,
                             config=cfg)
        lines[:] = unicode(doc).split(u"\n")

    # This tries to replace links to base classes with the shortest possible
    # import name (the one that is also used elsewhere in the docs)
    # E.g. for brian2.units.fundamentalunits.Quantity it checks whether it is
    # possible to import brian2.Quantity or brian2.units.Quantity and if this
    # is possible, replaces the link. In this case with the first variant 
    # brian2.Quantity
    for i, line in enumerate(lines):
        if line.startswith('Bases: '):
            class_links = re.findall(r':class:`[A-Za-z0-9.]+`', lines[i])
            replacements = {}
            for match in class_links:
                full_name = str(match[8:-1])
                name_parts = full_name.split('.')
                # We only care about brian2 classes
                if not name_parts[0] == 'brian2':
                    continue
                for module_parts in xrange(len(name_parts) - 2):
                    try:
                        # Use the first modules
                        name = '.'.join(name_parts[:module_parts + 1])
                        __import__(name, fromlist=name_parts[-1:])
                        # It worked!
                        replacements[full_name] = name + '.' + name_parts[-1]
                        # no need to search further
                        break
                    except ImportError:
                        pass
            for old, new in replacements.iteritems():
                lines[i] = lines[i].replace('`%s`' % old, '`%s`' % new)

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


def mangle_signature(app, what, name, obj, options, sig, retann):
    # Do not try to inspect classes that don't define `__init__`
    if (inspect.isclass(obj) and
        (not hasattr(obj, '__init__') or
        'initializes x; see ' in pydoc.getdoc(obj.__init__))):
        return '', ''

    if not (callable(obj) or hasattr(obj, '__argspec_is_invalid_')):
        return
    if not hasattr(obj, '__doc__'):
        return

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

    directives.register_directive('document_brian_prefs', BrianPrefsDirective)
    app.add_role('bpref', brian_prefs_role)

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

