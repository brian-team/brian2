State update
============

In Brian 2.0, a state updater transforms a set of equations into an abstract
state update code. The development of specific state update schemes is made
easier.
One difference with former versions is the direct handling of stochasticity.
State updaters are target-independent.

In the following, we consider a set of stochastic differential equations written in
vector form::

	dx/dt=f(x,t)+g(x,t)*xi

Proposed mechanism: we allow both writing explicit schemes and generator-based state updaters
as in experimental.codegen2.

Explicit state update
---------------------
In general, explicit state update schemes can be specified as a series of
mathematical statements. Here are a few examples

Euler
^^^^^

::

	x(t+dt)=x(t)+dt*f(x(t),t)

Runge-Kutta 2 (midpoint)
^^^^^^^^^^^^^^^^^^^^^^^^

::

	x(t+dt)=x(t)+dt*f(x(t)+dt/2*f(x(t),t),t+dt/2)

Runge-Kutta 4
^^^^^^^^^^^^^

::

	k1=dt*f(x(t),t)
	k2=dt*f(x(t)+k1/2,t+dt/2)
	k3=df*f(x(t)+k2/2,t+dt/2)
	k4=df*f(x(t)+k3,t+dt)
	x(t+dt)=x(t)+(k1+2*k2+2*k3+k4)/6

Stochastic Euler
^^^^^^^^^^^^^^^^

::

	x(t+dt)=x(t)+dt*f(x(t),t)+dt**.5*g(x(t),t)*randn()

General schemes
^^^^^^^^^^^^^^^^
So explicit non-linear state update schemes are relatively simple to
describe. I suggest that they can be described in a string as above, where
x is taken as the vector of dynamic variables. Static variables are automatically
calculated when f(x,t) is called (that is, we insert a string of update statements
before any such expression). We may use x instead of x(t), and remove the final
assignment. In this case, RK4 will become::

	k1=dt*f(x,t)
	k2=dt*f(x+k1/2,t+dt/2)
	k3=df*f(x+k2/2,t+dt/2)
	k4=df*f(x+k3,t+dt)
	return x+(k1+2*k2+2*k3+k4)/6

Implicit state updates
----------------------
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
