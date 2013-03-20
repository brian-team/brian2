State update
============

In Brian 2, a state updater transforms a set of equations into an abstract
state update code (and therefore is automatically target-independent). In
general, any function (or callable object) that takes an `Equations` object
and returns abstract code (as a string) can be used as a state updater and
passed to the `NeuronGroup` constructor as a ``method`` argument.

The more common use case is to specify no state updater at all or chose one by
name, see `Choice of state updaters`_ below.


Explicit state update
---------------------
Explicit state update schemes can be specified in mathematical notation, using
the `ExplicitStateUpdater` class. A state updater scheme contains of a series
of statements, defining temporary variables and a final line (starting with
``return``), giving the updated value for the state variable. The description
can make reference to ``t`` (the current time), ``dt`` (the size of the time
step), ``x`` (value of the state variable), and ``f(x, t)`` (the definition of
the state variable ``x``, assuming ``dx/dt = f(x, t)``. In addition, state
updaters supporting stochastic equations additionally make use of ``dW`` (a
normal distributed random variable with variance ``dt``) and ``g(x, t)``, the
factor multiplied with the noise variable, assuming
``dx/dt = f(x, t) + g(x, t) * xi``.

Using this notation, simple forward Euler integration is specified as::

	return x + dt * f(x, t)

A Runge-Kutta 2 (midpoint) method is specified as::
	
    k = dt * f(x,t)
    return x + dt * f(x +  k/2, t + dt/2)

When creating a new state updater using `ExplicitStateUpdater`, you can
specify the ``stochastic`` keyword argument, determining whether this state
updater does not support any stochastic equations (``None``, the default),
stochastic equations with additive noise only (``'additive'``), or
arbitrary stochastic equations (``'multiplicative'``). The provided state
updaters use the Stratonovich interpretation for stochastic equations (which
is the correct interpretation if the white noise source is seen as the limit
of a coloured noise source with a short time constant). As a result of this,
the simple Euler-Maruyama scheme (``return x + dt*f(x, t) + dW*g(x, t)``) will
only be used for additive noise. You can enforce the Itō interpretation,
however, by simply directly passing such a state updater. For example, if you 
specify `euler` for a system with multiplicative noise it will generate a
warning (because the state updater does not give correct results under the
Stratonovich interpretation) but will work (and give the correct result under
the Itō interpretation).

An example for a general state updater that handles arbitrary multiplicative
noise (under Stratonovich interpretation) is the derivative-free Milstein
method::

    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    return x + dt*f(x,t) + g(x, t) * dW + k 

Note that a single line in these descriptions is only allowed to mention
``g(x, t)``, respectively ``f(x, t)`` only once (and you are not allowed to
write, for example, ``g(f(x, t), t)``). You can work around these restrictions
by using intermediate steps, defining temporary variables, as in the above
examples for `milstein` and `rk2`.


Choice of state updaters
------------------------
As mentioned in the beginning, you can pass arbitrary callables to the
method argument of a `NeuronGroup`, as long as this callable converts an
`Equations` object into abstract code. The best way to add a new state updater,
however, is to register it with brian and provide a method to determine whether
it is appropriate for a given set of equations. This way, it can be
automatically chosen when no method is specified and it can be referred to with
a name (i.e. you can pass a string like ``'euler'`` to the method argument
instead of importing `euler` and passing a reference to the object itself).

If you create a new state updater using the `ExplicitStateUpdater` class, you
have to specify a *priority* for this state updater, a number determining which
state updater is chosen when more than one is appropriate for a given equation.
For example, for equations without stochastic variables, `euler` will be chosen
automatically, because it has a higher priority than the `rk2`, `rk4` and
`milstein`. The same is the case for equations with additive noise, but then
`rk2` and `rk4` are not even considered at all. For equations with
multiplicative noise, however, `milstein` is the only available integrator.

.. note::
	
	This mechanism is not ideal, maybe the "best" integrator for additive noise
	is not the same as for noise-free equations, for example? 

After creating the state updater, it has to be registered with
`StateUpdateMethod`::

    new_state_updater = ExplicitStateUpdater('...', priority=40,
                                             stochastic='additive')
    StateUpdateMethod.register('mymethod', new_state_updater)


The preferred way to do write new general state updaters (i.e. state updaters
that cannot be described using the explicit syntax described above) is to
extend the `StateUpdaterMethod` class (but this is not strictly necessary, all
that is needed is an object that implements a ``get_priority`` and a
``__call__`` method). The new class's ``get_priority`` method gets an
`Equations` object, a `namespace` dictionary for the external
variables/functions and a `specifier` dictionary for the internal state
variables. It has to return an integer value, the priority of the state updater
for the given situation (``0`` if it is not applicable, a positive value if it
is applicable). The method would typically make use of
`Equations.is_stochastic` or `Equations.stochastic_type`, check whether any
external functions are used, etc.. Finally, the state updater has to be registered
with `StateUpdateMethod` as shown above.

Implicit state updates
----------------------

.. note::

	All of the following is just here for future reference, it's not
	implemented yet.


Implicit schemes often use Newton-Raphson or fixed point iterations.
These can also be defined by mathematical statements, but the number of iterations
is dynamic and therefore not easily vectorised. However, this might not be
a big issue in C, GPU or even with Numba.

Backward Euler
^^^^^^^^^^^^^^
Backward Euler is defined as follows::

	x(t+dt)=x(t)+dt*f(x(t+dt),t+dt)

This is not a executable statement because the RHS depends on the future.
A simple way is to perform fixed point iterations::

	x(t+dt)=x(t)
	x(t+dt)=x(t)+dt*dx=f(x(t+dt),t+dt)    until increment<tolerance

This includes a loop with a different number of iterations depending on the
neuron.

Exponential Euler
^^^^^^^^^^^^^^^^^
This applies to conditionally linear equations, and requires the calculation
of matrices. It is not clear whether this can be described simply with
statements as above.

Linear and mixed state updates
------------------------------
If the equations are linear (with the special case of coefficients between
non-constant parameters), then exact state updaters can be used.
Exponential Euler is similar.
Currently, the matrix is calculated by direct evaluation of the expressions
with different vectors. Alternatively, perhaps we could use sympy to calculate
these matrices symbolically. There are several advantages: 1) it is done only
once rather than at each time step, 2) it does not need to be computed on
the target.

I think it would actually be possible to generate target-independent code
for exact updates, although it would not use target-specific matrix operations
(simply explicitly spelling the matrix product). But it would be more general.
