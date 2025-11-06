Older notes on code generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is an outline of how the Brian 2 code generation system works,
with indicators as to which packages to look at and which bits of code to read
for a clearer understanding.

We illustrate the global process with an example, the creation and running of
a single `NeuronGroup` object:

- Parse the equations, add refractoriness to them: this isn't really part of
  code generation.

- Allocate memory for the state variables.

- Create `Thresholder`, `Resetter` and `StateUpdater` objects.

  - Determine all the variable and function names used in the respective
    abstract code blocks and templates

  - Determine the abstract namespace, i.e. determine a `Variable` or `Function`
    object for each name.

  - Create a `CodeObject` based on the abstract code, template and abstract
    namespace. This will generate code in the target language and the namespace
    in which the code will be executed.

- At runtime, each object calls `CodeObject.__call__` to execute the code.

Stages of code generation
=========================

Equations to abstract code
--------------------------

In the case of `Equations`, the set of equations are combined with a
numerical integration method to generate an *abstract code block* (see below)
which represents the integration code for a single time step.

An example of this would be converting the following equations::

    eqs = '''
    dv/dt = (v0-v)/tau : volt (unless refractory)
    v0 : volt
    '''
    group = NeuronGroup(N, eqs, threshold='v>10*mV',
                        reset='v=0*mV', refractory=5*ms)

into the following abstract code using the `exponential_euler` method (which
is selected automatically)::

    not_refractory = 1*((t - lastspike) > 0.005000)
    _BA_v = -v0
    _v = -_BA_v + (_BA_v + v)*exp(-dt*not_refractory/tau)
    v = _v

The code for this stage can be seen in `NeuronGroup.__init__`,
`StateUpdater.__init__`, and `StateUpdater.update_abstract_code`
(in ``brian2.groups.neurongroup``), and the `StateUpdateMethod` classes
defined in the ``brian2.stateupdaters`` package.

For more details, see :doc:`../advanced/state_update`.

Abstract code
-------------

'Abstract code' is just a multi-line string representing a block of code which
should be executed for each item (e.g. each neuron, each synapse). Each item
is independent of the others in abstract code. This allows us to later
generate code either for vectorised languages (like numpy in Python) or
using loops (e.g. in C++).

Abstract code is parsed according to Python syntax, with certain language
constructs excluded. For example, there cannot be any conditional or looping
statements at the moment, although support for this is in principle possible
and may be added later. Essentially, all that is allowed at the moment is a
sequence of arithmetical ``a = b*c`` style statements.

Abstract code is provided directly by the user for threshold and reset
statements in `NeuronGroup` and for pre/post spiking events in `Synapses`.

Abstract code to snippet
------------------------

We convert abstract code into a 'snippet', which is a small segment of
code which is syntactically correct in the target language, although it may
not be runnable on its own (that's handled by insertion into a 'template'
later). This is handled by the `CodeGenerator` object in ``brian2.codegen.generators``.
In the case of converting into python/numpy code this typically doesn't involve
any changes to the code at all because the original code is in Python
syntax. For conversion to C++, we have to do some syntactic transformations
(e.g. ``a**b`` is converted to ``pow(a, b)``), and add declarations for
certain variables (e.g. converting ``x=y*z`` into ``const double x = y*z;``).

An example of a snippet in C++ for the equations above::

    const double v0 = _ptr_array_neurongroup_v0[_neuron_idx];
    const double lastspike = _ptr_array_neurongroup_lastspike[_neuron_idx];
    bool not_refractory = _ptr_array_neurongroup_not_refractory[_neuron_idx];
    double v = _ptr_array_neurongroup_v[_neuron_idx];
    not_refractory = 1 * (t - lastspike > 0.0050000000000000001);
    const double _BA_v = -(v0);
    const double _v = -(_BA_v) + (_BA_v + v) * exp(-(dt) * not_refractory / tau);
    v = _v;
    _ptr_array_neurongroup_not_refractory[_neuron_idx] = not_refractory;
    _ptr_array_neurongroup_v[_neuron_idx] = v;

The code path that includes snippet generation will be discussed in more detail
below, since it involves the concepts of namespaces and variables which we
haven't covered yet.

Snippet to code block
---------------------

The final stage in the generation of a runnable code block is the insertion
of a snippet into a template. These use the Jinja2 template specification
language. This is handled in ``brian2.codegen.templates``.

An example of a template for Python thresholding::

    # USES_VARIABLES { not_refractory, lastspike, t }
    {% for line in code_lines %}
    {{line}}
    {% endfor %}
    _return_values, = _cond.nonzero()
    # Set the neuron to refractory
    not_refractory[_return_values] = False
    lastspike[_return_values] = t

and the output code from the example equations above::

    # USES_VARIABLES { not_refractory, lastspike, t }
    v = _array_neurongroup_v
    _cond = v > 10 * mV
    _return_values, = _cond.nonzero()
    # Set the neuron to refractory
    not_refractory[_return_values] = False
    lastspike[_return_values] = t

Code block to executing code
----------------------------

A code block represents runnable code. Brian operates in two different regimes,
either in runtime or standalone mode. In runtime mode, memory allocation and
overall simulation control is handled by Python and numpy, and code objects
operate on this memory when called directly by Brian. This is the typical
way that Brian is used, and it allows for a rapid development cycle. However,
we also support a standalone mode in which an entire project workspace is
generated for a target language or device by Brian, which can then be
compiled and run independently of Brian. Each mode has different templates,
and does different things with the outputted code blocks. For runtime mode,
in Python/numpy code is executed by simply calling the ``exec`` statement
on the code block in a given namespace. In standalone mode, the templates
will typically each be saved into different files.

Key concepts
============

Namespaces
----------

In general, a namespace is simply a mapping/dict from names to values. In Brian
we use the term 'namespace' in two ways: the high level "abstract namespace"
maps names to objects based on the `Variables` or `Function` class. In the above
example, ``v`` maps to an `ArrayVariable` object, ``tau`` to a `Constant`
object, etc. This namespace has all the information that is needed for checking
the consistency of units, to determine which variables are boolean or scalar,
etc. During the `CodeObject` creation, this abstract namespace is converted into
the final namespace in which the code will be executed. In this namespace, ``v``
maps to the numpy array storing the state variable values (without units) and
``tau`` maps to a concrete value (again, without units).
See :doc:`equations_namespaces` for more details.

Variable
----------

`Variable` objects contain information about the variable
they correspond to, including details like the data type, whether it is a single value
or an array, etc.

See ``brian2.core.variables`` and, e.g. `Group._create_variables`,
`NeuronGroup._create_variables`.

Templates
---------

Templates are stored in Jinja2 format. They come in one of two forms, either they are a single
template if code generation only needs to output a single block of code, or they define multiple
Jinja macros, each of which is a separate code block. The `CodeObject` should define what type of
template it wants, and the names of the macros to define. For examples, see the templates in the
directories in ``brian2/codegen/runtime``. See ``brian2.codegen.templates`` for more details.

Code guide
==========

This section includes a guide to the various relevant packages and subpackages
involved in the code generation process.

``codegen``
    Stores the majority of all code generation related code.

    ``codegen.functions``
        Code related to including functions - built-in and user-defined - in generated code.
    ``codegen.generators``
        Each `CodeGenerator` is defined in a module here.
    ``codegen.runtime``
        Each runtime `CodeObject` and its templates are defined in a package here.
``core``
    ``core.variables``
        The `Variable` types are defined here.
``equations``
    Everything related to `Equations`.
``groups``
    All `Group` related stuff is in here. The `Group.resolve` methods are
    responsible for determining the abstract namespace.
``parsing``
    Various tools using Python's ``ast`` module to parse user-specified code. Includes syntax
    translation to various languages in ``parsing.rendering``.
``stateupdaters``
    Everything related to generating abstract code blocks from integration methods is here.
