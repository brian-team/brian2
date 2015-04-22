Brian 1 Hears bridge
====================

.. currentmodule:: brian2.hears

The "`Brian Hears`_" library is being rewritten for Brian 2 in a more flexible way to allow it to work on multiple devices.
Until this work is complete, it is possible to run the version of Brian Hears from Brian 1.x using the "bridge". To
make this work, you must have a copy of Brian 1 installed (preferably the latest version), and import Brian Hears
using::

	from brian2.hears import *

Many scripts will run without any changes, but there are a few caveats to be aware of. Mostly, the problems are due
to the fact that the units system in Brian 2 is not 100% compatible with the units system of Brian 1.

`FilterbankGroup` now follows the rules for `NeuronGroup` in Brian 2, which means some changes may be
necessary to match the syntax of Brian 2, for example, the following would work in Brian 1 Hears::
  
	# Leaky integrate-and-fire model with noise and refractoriness
	eqs = '''
	dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
	I : 1
	'''
	anf = FilterbankGroup(ihc, 'I', eqs, reset=0, threshold=1, refractory=5*ms)
  
However, in Brian 2 Hears you would need to do::
  
	# Leaky integrate-and-fire model with noise and refractoriness
	eqs = '''
	dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1 (unless refractory)
	I : 1
	'''
	anf = FilterbankGroup(ihc, 'I', eqs, reset='v=0', threshold='v>1', refractory=5*ms)  
  
Slicing sounds no longer works. Previously you could do, e.g. ``sound[:20*ms]`` but with Brian 2 you would need
to do ``sound.slice(0*ms, 20*ms)``.

In addition, some functions may not work correctly with Brian 2 units. In most circumstances, Brian 2 units can be
used interchangeably with Brian 1 units in the bridge, but in some cases it may be necessary to convert units from
one format to another, and to do that you can use the functions `convert_unit_b1_to_b2` and `convert_unit_b2_to_b1`.

.. _`Brian Hears`: http://brian.readthedocs.org/en/latest/hears.html