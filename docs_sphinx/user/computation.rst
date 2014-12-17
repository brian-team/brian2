Computational methods and efficiency
====================================

Brian has several different methods for running the computations in a
simulation. In particular, Brian uses "runtime code generation" for
efficient computation. This means that it takes the Python code and strings
in your model and generates code in one of several possible different
languages and actually executes that. By default, this generated code is in
Python because it is the only one guaranteed to work in all cases, however
it is not the most computationally efficient language. If you have a C++
compiler installed (currently gcc is the only supported compiler), and you're
running on Python 2.x then you can use this for a speed by setting the
``codegen.target = 'weave'`` preference. The simplest way to do that is to write
the following at the top of your script::

    from brian2 import *
    prefs.codegen.target = 'weave'

See :doc:`../advanced/preferences` for different ways of setting preferences.
If you are running on Python 3.x then the scipy.weave module that we use
for C++ code generation will not work, so you'll need to install Cython
instead and use the preference ``codegen.target = 'cython'``.

Both of these code generation targets are still run via Python, which means
that there are still overheads due to Python. The very fastest way to run
Brian is in "standalone mode" (see :doc:`devices`), although this won't work
for every possible simulation. Note that you can also use multiple threads
with standalone mode, which is not possible in the modes described above.
This doesn't always lead to a huge speed improvement, but can occasionally
give a higher than linear speed up relative to the number of cores.

You might find that running simulations in weave or Cython modes won't work
or is not as efficient as you were expecting. This is probably because you're
using Python functions which are not compatible with weave or Cython. For
example, if you wrote something like this it would not be efficient::

    from brian2 import *
    prefs.codegen.target = 'weave'
    def f(x):
        return abs(x)
    G = NeuronGroup(10000, 'dv/dt = -x*f(x) : 1')
    
The reason is that the function ``f(x)`` is a Python function and so cannot
be called from C++ directly. To solve this problem, you need to provide an
implementation of the function in the target language. See :doc:`functions`.
