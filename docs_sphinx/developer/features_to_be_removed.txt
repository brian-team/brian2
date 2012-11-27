Brian 1.x features to be removed or significantly changed
=========================================================

* Connection, DelayConnection, STDP, STP: these are now handled by Synapses
* 'dynamic' and 'dense' matrix types not represented in Synapses?
* Some unit names will not be included in ``from brian import *``
* The ``Equations`` object will be removed, and so to iteratively construct
  equations you will need to join the strings together. This is because it is
  a nightmare to handle behind the scenes and causes many problems. As a
  replacement, we will offer a module to make joining these strings together
  and substituting variables names easy. Almost everything that could be done
  before should be possible in the new system, without any nasty bugs.
* The model library will be removed. In its place, on the website, we will have
  a catalogue of model definitions that can be cut and pasted.
* Float clocks may be removed and only int-based clocks kept.
* Some things that were previously implicit will now be forced to be explicit,
  for example ``reset=-60*mV`` will not work in NeuronGroup, you will need to
  write ``reset='V=-60*mV'``. We plan to do less guessing of which variable
  is intended, and having everything in string form makes it easier to develop
  and easier for others to understand your code.
* The magic system might change a bit, we haven't quite decided on this yet.
  One option is to get rid of the confusing and complicated execution frame
  based magic, and simply use all created objects when you call run().
* Monitors will probably be simplified.

Some of things will be useful to some users, but we have strong reasons to
believe that the changes will be of substantial long term benefit to everyone.
Some user code will need to be rewritten, but everything that was possible
before should be possible afterwards (although possibly not in the first
release, e.g. 'dynamic' matrix type). As a compensation, it should in the long
run be easier, faster and even more flexible. Plus, we have some new features
planned as well.