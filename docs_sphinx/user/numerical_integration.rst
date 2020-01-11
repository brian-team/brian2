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
other methods (in particular the ``'exact'`` method which attempts to solve the
equations analytically) can take a considerable amount of time -- to avoid wasting
this time, you can always chose the integration method manually (see below). You
can also suppress the message by raising the log level or by explicitly suppressing
``'method_choice'`` log messages -- for details, see :doc:`../advanced/logging`.

If you prefer to chose an integration algorithm yourself, you can do so using
the ``method`` keyword for `NeuronGroup`, `Synapses`, or `SpatialNeuron`.
The complete list of available methods is the following:

* ``'exact'``: exact integration for linear equations (alternative name: ``'linear'``)
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

.. note::

  The ``'independent'`` integration method (exact integration for a system of
  independent equations, where all the equations can be analytically solved
  independently) should no longer be used and might be removed in future
  versions of Brian.

.. note:: The following methods are still considered experimental

* ``'gsl'``: default integrator when choosing to integrate equations with
  the GNU Scientific Library ODE solver: the rkf45 method. Uses an adaptable
  time step by default.
* ``'gsl_rkf45'``: Runge-Kutta-Fehlberg method.
  A good general-purpose integrator according to the GSL documentation. Uses an
  adaptable time step by default.
* ``'gsl_rk2'``: Second order Runge-Kutta method using GSL. Uses an adaptable
  time step by default.
* ``'gsl_rk4'``: Fourth order Runge-Kutta method using GSL. Uses an adaptable
  time step by default.
* ``'gsl_rkck'``: Runge-Kutta Cash-Karp method using GSL. Uses an adaptable
  time step by default.
* ``'gsl_rk8pd'``: Runge-Kutta Prince-Dormand method using GSL. Uses an adaptable
  time step by default.

.. admonition:: The following topics are not essential for beginners.

    |

Technical notes
---------------

Each class defines its own list of algorithms it tries to
apply, `NeuronGroup` and `Synapses` will use the first suitable method out of
the methods ``'exact'``, ``'euler'`` and ``'heun'`` while `SpatialNeuron`
objects will use ``'exact'``, ``'exponential_euler'``, ``'rk2'`` or ``'heun'``.

You can also define your own numerical integrators, see
:doc:`../advanced/state_update` for details.

.. _gsl_integration:

GSL stateupdaters
-----------------
The stateupdaters preceded with the gsl tag use ODE solvers defined in the GNU
Scientific Library. The benefit of using these integrators over the ones written
by Brian internally, is that they are implemented with an adaptable timestep.
Integrating with an adaptable timestep comes with two advantages:

* These methods check whether the estimated error of the solutions returned fall
  within a certain error bound. For the non-gsl integrators there is currently no
  such check.
* Systems no longer need to be simulated with just one time step. That is, a bigger
  timestep can be chosen and the integrator will reduce the timestep when increased
  accuracy is required. This is particularly useful for systems where both slow and
  fast time constants coexist, as is the case with for example (networks of neurons
  with) Hodgkin-Huxley equations. Note that Brian's timestep still determines the
  resolution for monitors, spike timing, spike propagation etc. Hence, in a network,
  the simulation error will therefore still be on the order of ``dt``. The benefit
  is that short time constants occurring in equations no longer dictate the network
  time step.

In addition to a choice between different integration methods, there are a few more
options that can be specified when using GSL. These options can be specified by
sending a dictionary as the ``method_options`` key upon initialization of the object
using the integrator (`NeuronGroup`, `Synapses` or `SpatialNeuron`).
The available method options are:

* ``'adaptable_timestep'``: whether or not to let GSL reduce the timestep to
  achieve the accuracy defined with the ``'absolute_error'`` and
  ``'absolute_error_per_variable'`` options described below. If this is set to ``False``,
  the timestep is determined by Brian (i.e. the ``dt`` of the respective clock is used, see :ref:`scheduling`).
  If the resulted estimated error exceeds the set error bounds, the simulation
  is aborted. When using cython this is reported with an `IntegrationError`.
  Defaults to ``True``.
* ``'absolute_error'``: each of the methods has a way of estimating the error that
  is the result of using numerical integration. You can specify the maximum size of this
  error to be allowed for any of the to-be-integrated variables in base units with this
  keyword. Note that giving very small values makes the simulation slow and might result
  in unsuccessful integration. In the case of using the ``'absolute_error_per_variable'``
  option, this is the error for variables that were not specified individually.
  Defaults to 1e-6.
* ``'absolute_error_per_variable'``: specify the absolute error per variable in its own
  units. Variables for which the error is not specified use the error set with the
  ``'absolute_error'`` option.
  Defaults to None.
* ``'max_steps'``: The maximal number of steps that the integrator will take within a
  single "Brian timestep" in order to reach the given error criterion. Can be set to
  0 to not set any limits. Note that without limits, it can take a very long time
  until the integrator figures out that it cannot reach the desired error level. This
  will manifest as a simulation that appears to be stuck.
  Defaults to 100.
* ``'use_last_timestep'``: with the ``'adaptable_timestep'`` option set to True, GSL tries
  different time steps to find a solution that satisfies the set error bounds.
  It is likely that for Brian's next time step the GSL time step
  will be somewhat similar per neuron (e.g. active neurons will have a shorter GSL time step
  than inactive neurons). With this option set to True, the time step GSL found to satisfy
  the set error bounds is saved per neuron and given to GSL again in Brian's next time step.
  This also means that the final time steps are saved in Brian's memory and can thus
  be recorded with the `StateMonitor`: it can be accessed under ``'_last_timestep'``.
  Note that some extra memory is required to keep track of the last time steps.
  Defaults to True.
* ``'save_failed_steps'``: if ``'adaptable_timestep'`` is set to True,
  each time GSL tries a time step and it results in an estimated
  error that exceeds the set bounds, one is added to the ``'_failed_steps'`` variable. For
  purposes of investigating what happens within GSL during an integration step, we offer
  the option of saving this variable.
  Defaults to False.
* ``'save_step_count'``: the same goes for the total number of GSL steps taken in a single
  Brian time step: this is optionally saved in the ``'_step_count'`` variable.
  Defaults to False.

Note that at the moment recording ``'_last_timestep'``, ``'_failed_steps'``, or ``'_step_count'``
requires a call to `run` (e.g. with 0 ms) to trigger the code generation process, before the
call to `StateMonitor`.

More information on the GSL ODE solver itself can be found in its
`documentation <https://www.gnu.org/software/gsl/manual/html_node/Ordinary-Differential-Equations.html>`_.
