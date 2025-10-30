Defensive programming
=====================

One idea for Brian 2 is to make it so that it's more likely that errors are
raised rather than silently causing weird bugs. Some ideas in this line:

Synapses.source should be stored internally as a weakref Synapses._source, and
Synapses.source should be a computed attribute that dereferences this weakref.
Like this, if the source object isn't kept by the user, Synapses won't store a
reference to it, and so won't stop it from being deallocated.

We should write an automated test that takes a piece of correct code like::

    NeuronGroup(N, eqs, reset='V>Vt')

and tries replacing all arguments by nonsense arguments, it should always
raise an error in this case (forcing us to write code to validate the inputs).
For example, you could create a new NonsenseObject class, and do this::

    nonsense = NonsenseObject()
    NeuronGroup(nonsense, eqs, reset='V>Vt')
    NeuronGroup(N, nonsense, reset='V>Vt')
    NeuronGroup(N, eqs, nonsense)

In general, the idea should be to make it hard for something incorrect to run
without raising an error, preferably at the point where the user makes the error
and not in some obscure way several lines later.

The preferred way to validate inputs is one that handles types in a Pythonic
way. For example, instead of doing something like::

    if not isinstance(arg, (float, int)):
        raise TypeError(...)

Do something like::

        arg = float(arg)

(or use try/except to raise a more specific error). In contrast to the
``isinstance`` check it does not make any assumptions about the type except for
its ability to be converted to a float.

This approach is particular useful for numpy arrays::

    arr = np.asarray(arg)

(or ``np.asanyarray`` if you want to allow for array subclasses like arrays
with units or masked arrays). This approach has also the nice advantage that it
allows all "array-like" arguments, e.g. a list of numbers.
