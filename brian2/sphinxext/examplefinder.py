'''
Automatically find examples of a Brian object or function.
'''

import os, re
from collections import defaultdict
from generate_examples import GlobDirectoryWalker
from brian2.utils.stringtools import get_identifiers

__all__ = ['auto_find_examples']

the_examples_map = defaultdict(list)

def get_examples_map():
    if the_examples_map:
        return the_examples_map
    if 'BRIAN2_DOCS_EXAMPLE_DIR' in os.environ:
        rootdir = os.environ['BRIAN2_DOCS_EXAMPLE_DIR']
    else:
        rootdir, _ = os.path.split(__file__)
        rootdir = os.path.normpath(os.path.join(rootdir, '../../examples'))
    fnames = [fname for fname in GlobDirectoryWalker(rootdir, '*.py')]
    shortfnames = [os.path.relpath(fname, rootdir) for fname in fnames]
    exnames = [fname.replace('/', '.').replace('\\', '.').replace('.py', '') for fname in shortfnames]
    for fname, shortfname, exname in zip(fnames, shortfnames, exnames):
        ex = open(fname, 'r').read()
        ids = get_identifiers(ex)
        for id in ids:
            the_examples_map[id].append((shortfname.replace('\\', '/'), exname))
    return the_examples_map

def auto_find_examples(obj, headersymbol='='):
    '''
    Returns a restructured text section listing all the examples making use
    of the specified object (as determined by the name being in the list
    of identifiers, which may occasionally make mistakes but is usually
    going to be correct).
    '''
    name = obj.__name__
    examples_map = get_examples_map()
    examples = the_examples_map[name]
    if len(examples)==0:
        return ''
    txt = 'Examples using this'
    txt = txt+'\n'+headersymbol*len(txt)+'\n\n'
    for ex in examples:
        txt += '* :doc:`%s </examples/%s>`\n' % ex
    return txt+'\n'
    
if __name__=='__main__':
    from brian2 import NeuronGroup, SpatialNeuron
    print auto_find_examples(SpatialNeuron)
