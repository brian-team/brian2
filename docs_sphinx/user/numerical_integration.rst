.. _numerical_integration:

Numerical integration
=====================

By default, Brian
chooses an integration method automatically, trying to solve the equations
exactly first (for linear equations) and then resorting to numerical algorithms.
It will also take care of integrating stochastic differential equations
appropriately.

Note that in some cases, the automatic choice of integration method will not be
appropriate, because of a choice of parameters that couldn't be determined in
advance. In this case, typically you will get nan (not a number) values in the
results, or large oscillations. In this case, Brian will generate a warning to
let you know, but will not raise an error.

Method choice
-------------

You will get an ``INFO`` message telling you which integration method Brian decided to use,
together with information about how much time it took to apply the integration method
to your equations. If other methods have been tried but were not applicable, you will
also see the time it took to try out those other methods. In some cases, checking
other methods (in particular the ``'linear'`` method which attempts to solve the
equations analytically) can take a considerable amount of time -- to avoid wasting
this time, you can always chose the integration method manually (see below). You
can also suppress the message by raising the log level or by explicitly suppressing
``'method_choice'`` log messages -- for details, see :doc:`../advanced/logging`.

If you prefer to chose an integration algorithm yourself, you can do so using
the ``method`` keyword for `NeuronGroup`, `Synapses`, or `SpatialNeuron`.
The complete list of available methods is the following:

* ``'linear'``: exact integration for linear equations
* ``'independent'``: exact integration for a system of independent equations,
  where all the equations can be analytically solved independently
* ``'exponential_euler'``: exponential Euler integration for conditionally
  linear equations
* ``'euler'``: forward Euler integration (for additive stochastic
  differential equations using the Euler-Maruyama method)
* ``'rk2'``: second order Runge-Kutta method (midpoint method)
* ``'rk4'``: classical Runge-Kutta method (RK4)
* ``'heun'``: stochastic Heun method for solving Stratonovich stochastic
  differential equations with non-diagonal multiplicative noise.
* ``'milstein'``: derivative-free Milstein method for solving stochastic
  differential equations with diagonal multiplicative noise

.. admonition:: The following topics are not essential for beginners.

    |

Technical notes
---------------

Each class defines its own list of algorithms it tries to
apply, `NeuronGroup` and `Synapses` will use the first suitable method out of
the methods ``'linear'``, ``'euler'`` and ``'heun'`` while `SpatialNeuron`
objects will use ``'linear'``, ``'exponential_euler'``, ``'rk2'`` or ``'heun'``.

You can also define your own numerical integrators, see
:doc:`../advanced/state_update` for details.
