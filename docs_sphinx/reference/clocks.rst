.. currentmodule:: brian2

Clocks
======

.. data:: defaultclock

	The default Brian clock. If a clock is not specified for a particular
	object, it will use this clock. By default, it has a step size of
	``dt=0.1*ms`` but this can be changed at the beginning of your script
	with, for example, ``defaultclock.dt = 1*ms``.

.. autoclass:: Clock
