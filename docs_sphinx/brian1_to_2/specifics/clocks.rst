Clocks and timesteps
====================

Brian's system of handling clocks has substantially changed. For details about the new system in place see
:ref:`time_steps`. The main differences to Brian 1 are:

* There is no more "clock guessing" -- objects either use the `defaultclock` or a ``dt``/``clock`` value that was
  explicitly specified during their construction
* In Brian 2, the time step is allowed to change after the creation of an object and between runs -- the relevant value
  is the value in place at the point of the `run` call.
* It is rarely necessary to create an explicit `Clock` object, most of the time you should usie the `defaultclock` or
  provide a ``dt`` argument during the construction of the object.
* There's only one `Clock` class, the (deprecated) ``FloatClock``, ``RegularClock``, etc. classes that Brian 1 provided
  no longer exist.
