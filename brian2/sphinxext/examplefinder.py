"""
Automatically find examples of a Brian object or function.
"""


import os, re
from collections import defaultdict
from .generate_examples import GlobDirectoryWalker
from brian2.utils.stringtools import get_identifiers

__all__ = ["auto_find_examples"]

the_examples_map = defaultdict(list)
the_tutorials_map = defaultdict(list)


def get_map(environ_var, relrootdir, pattern, the_map, path_exclusions=[]):
    if the_map:
        return the_map
    if environ_var in os.environ:
        rootdir = os.environ[environ_var]
    else:
        rootdir, _ = os.path.split(__file__)
        rootdir = os.path.normpath(os.path.join(rootdir, relrootdir))
    fnames = [fname for fname in GlobDirectoryWalker(rootdir, f"*{pattern}")]
    for exclude in path_exclusions:
        fnames = [fname for fname in fnames if exclude not in fname]
    shortfnames = [os.path.relpath(fname, rootdir) for fname in fnames]
    exnames = [
        fname.replace("/", ".").replace("\\", ".").replace(pattern, "")
        for fname in shortfnames
    ]
    for fname, shortfname, exname in zip(fnames, shortfnames, exnames):
        with open(fname, "r") as f:
            ex = f.read()
        ids = get_identifiers(ex)
        for id in ids:
            the_map[id].append((shortfname.replace("\\", "/"), exname))
    return the_map


def get_examples_map():
    return get_map("BRIAN2_DOCS_EXAMPLE_DIR", "../../examples", ".py", the_examples_map)


def get_tutorials_map():
    return get_map(
        "BRIAN2_DOCS_TUTORIALS_DIR",
        "../../tutorials",
        ".ipynb",
        the_tutorials_map,
        path_exclusions=[".ipynb_checkpoints"],
    )


def auto_find_examples(obj, headersymbol="="):
    """
    Returns a restructured text section listing all the examples and
    tutorials making use of the specified object (as determined by
    the name being in the list of identifiers, which may occasionally
    make mistakes but is usually going to be correct).
    """
    name = obj.__name__
    examples_map = get_examples_map()
    examples = sorted(the_examples_map[name])
    tutorials_map = get_tutorials_map()
    tutorials = sorted(the_tutorials_map[name])
    if len(examples + tutorials) == 0:
        return ""
    txt = "Tutorials and examples using this"
    txt = f"{txt}\n{headersymbol * len(txt)}\n\n"
    for tutname, tutloc in tutorials:
        tutname = tutname.replace(".ipynb", "")
        txt += f"* Tutorial :doc:`{tutname} </resources/tutorials/{tutloc}>`\n"
    for exname, exloc in examples:
        exname = exname.replace(".py", "")
        txt += f"* Example :doc:`{exname} </examples/{exloc}>`\n"
    return f"{txt}\n"


if __name__ == "__main__":
    from brian2 import NeuronGroup, SpatialNeuron

    print(auto_find_examples(NeuronGroup))
