import sys

from brian2 import *

prefs.codegen.cpp.extra_compile_args_msvc = [sys.argv[1]]

clear_cache("cython")

g = NeuronGroup(100, "")
syn = Synapses(g, g)
import time

start = time.time()
syn.connect()
print(f"{sys.argv[1]} - Took: {time.time() - start:.02}s")
