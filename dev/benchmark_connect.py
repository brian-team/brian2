import sys

from brian2 import *

prefs.codegen.target = "cython"  # just to be sure
if sys.argv[1] == "debug":
    prefs.codegen.cpp.msvc_debug = True

clear_cache("cython")

g = NeuronGroup(100, "")
syn = Synapses(g, g)
import time

start = time.time()
syn.connect()
print(f"{sys.argv[1]} - Took: {time.time() - start:.02}s")
