Brian 2.0
=========
Aims:

* Simplification: fewer modules and classes but more general; refactoring
* More encapsulation
* Easier to develop and maintain

Outline
-------
* Units: more or less ok
	- considering arrays
	- reduction of number of units imported by default
	- runtime disabling of units
	- unpickling units when units are disabled
* Clocks: integer clocks
* Parsing (simplification of Equations)
	- turning strings into structure with flags etc
	- check units
	- get namespaces
* Parsing threshold (boolean conditions) and reset (statements)
	- share things with parsing equations
* Code generation, dealing with equations
	- Intermediate step: producing abstract code (target independent)
	- Translate abstract code into target code (=translation or compilation)
	- constant parameters
	- syntax: flags in brackets
		(event-driven)
		(constant)
	- get rid of aliases
	- integrate randomness into the code
* Equations replaced by string formatting?
	(see examples).
* NeuronGroup
	- new refractoriness (see end of BEP-18)
	- code generation is the core of updates
	- LS is out (replaced by SpikeQueue), instead: an array of last spikes (current timestep)
	- threshold and reset objects are out
	- only string threshold/reset (no guessing)
	
	Methods:
	
	- update()
	- threshold()
	- reset()
* Library: replace by catalogue (copy/paste)
* Synapses
	- modulation disappears
* SpikeQueue
* SpikeMonitor and StateMonitor?
* NetworkOperation becomes the base class (with a when attribute)
	- update() (or __call__()?)
	- when
	- name attribute? (could be automatically generated or optionally given by the user)
* TimedArray
* Linked variables
* Global choice for float vs. double
* Magic with repeated runs

Main loop
---------
(we can change the order)
1) NeuronGroup state updates
2) Threshold
3) Transfer to SpikeQueue
4) Reset
5) Spike propagation
6) Other network operations

Notes
-----
See style guide style.txt

A number of projects to look at:

* GPU for Python: http://www.accelereyes.com/
  (includes IIR filters and convolution)
* Oliphant's note on speed comparisons:
  http://technicaldiscovery.blogspot.fr/2011/07/speeding-up-python-again.html
  Fortran rather than C could perhaps be a target?
* Numba looks amazing: http://numba.pydata.org/
* This is used in Numba: http://www.llvmpy.org/
* Python for Android: https://code.google.com/p/python-for-android/
