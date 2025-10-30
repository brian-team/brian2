State update
============

In Brian, a state updater transforms a set of equations into an abstract
state update code (and therefore is automatically target-independent). In
general, any function (or callable object) that takes an `Equations` object
and returns abstract code (as a string) can be used as a state updater and
passed to the `NeuronGroup` constructor as a ``method`` argument.

The more common use case is to specify no state updater at all or chose one by
name, see `Choice of state updaters`_ below.


Explicit state update
---------------------
Explicit state update schemes can be specified in mathematical notation, using
the `ExplicitStateUpdater` class. A state updater scheme contains a series
of statements, defining temporary variables and a final line (starting with
``x_new =``), giving the updated value for the state variable. The description
can make reference to ``t`` (the current time), ``dt`` (the size of the time
step), ``x`` (value of the state variable), and ``f(x, t)`` (the definition of
the state variable ``x``, assuming ``dx/dt = f(x, t)``. In addition, state
updaters supporting stochastic equations additionally make use of ``dW`` (a
normal distributed random variable with variance ``dt``) and ``g(x, t)``, the
factor multiplied with the noise variable, assuming
``dx/dt = f(x, t) + g(x, t) * xi``.

Using this notation, simple forward Euler integration is specified as::

	x_new = x + dt * f(x, t)

A Runge-Kutta 2 (midpoint) method is specified as::

    k = dt * f(x,t)
    x_new = x + dt * f(x +  k/2, t + dt/2)

When creating a new state updater using `ExplicitStateUpdater`, you can
specify the ``stochastic`` keyword argument, determining whether this state
updater does not support any stochastic equations (``None``, the default),
stochastic equations with additive noise only (``'additive'``), or
arbitrary stochastic equations (``'multiplicative'``). The provided state
updaters use the Stratonovich interpretation for stochastic equations (which
is the correct interpretation if the white noise source is seen as the limit
of a coloured noise source with a short time constant). As a result of this,
the simple Euler-Maruyama scheme (``x_new = x + dt*f(x, t) + dW*g(x, t)``) will
only be used for additive noise.

An example for a general state updater that handles arbitrary multiplicative
noise (under Stratonovich interpretation) is the derivative-free Milstein
method::

    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    x_new = x + dt*f(x,t) + g(x, t) * dW + k

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
have to specify what kind of stochastic equations it supports. The keyword
argument ``stochastic`` takes the values ``None`` (no stochastic equation
support, the default), ``'additive'`` (support for stochastic equations with
additive noise), ``'multiplicative'`` (support for arbitrary stochastic
equations).

After creating the state updater, it has to be registered with
`StateUpdateMethod`::

    new_state_updater = ExplicitStateUpdater('...', stochastic='additive')
    StateUpdateMethod.register('mymethod', new_state_updater)

The preferred way to do write new general state updaters (i.e. state updaters
that cannot be described using the explicit syntax described above) is to
extend the `StateUpdateMethod` class (but this is not strictly necessary, all
that is needed is an object that implements a ``__call__`` method that
operates on an `Equations` object and a dictionary of variables). Optionally,
the state updater can be registered with `StateUpdateMethod` as shown above.

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
