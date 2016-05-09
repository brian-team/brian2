Release notes
=============

Brian 2.0rc1
------------
This is a bug fix release that we release only about two weeks after the previous
release because that release introduced a bug that could lead to wrong integration of
stochastic differential equations. Note that standard neuronal noise models were
not affected by this bug, it only concerned differential equations implementing a
"random walk". The release also fixes a few other issues reported by users, see below
for more information.

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix a regression from 2.0b4: stochastic differential equations without any non-stochastic
  part (e.g. ``dx/dt = xi/sqrt(ms)```) were not integrated correctly (see #686).
* Repeatedly calling `restore` (or `Network.restore`) no longer raises an error (see #681).
* Fix an issue that made `PoissonInput` refuse to run after a change of dt (see #684).
* If the ``rates`` argument of `PoissonGroup` is a string, it will now be evaluated at
  every time step instead of once at construction time. This makes time-dependent rate
  expressions work as expected (see #660).

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
anyone we forgot...):

* Cian O'Donnell
* Daniel Bliss
* Ibrahim Ozturk
* Olivia Gozel


Brian 2.0rc
-----------
This is a release candidate for the final Brian 2.0 release, meaning that from
now on we will focus on bug fixes and documentation, without introducing new
major features or changing the syntax for the user. This release candidate itself
*does* however change a few important syntax elements, see "Backwards-incompatible
changes" below.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development mailing
list (brian-development@googlegroups.com).

Major new features
~~~~~~~~~~~~~~~~~~
* New "generator syntax" to efficiently generate synapses (e.g. one-to-one connections), see :ref:`creating_synapses`
  for more details.
* For synaptic connections with multiple synapses between a pair of neurons, the number of the synapse can now be
  stored in a variable, allowing its use in expressions and statements (see :ref:`creating_synapses`).
* `Synapses` can now target other `Synapses` objects, useful for some models of synaptic modulation.
* The `Morphology` object has been completely re-worked and several issues have been fixed. The new `Section` object
  allows to model a section as a series of truncated cones (see :ref:`creating_morphology`).
* Scripts with a single `run` call, no longer need an explicit ``device.build()`` call to run with the C++
  standalone device. A `set_device` in the beginning is enough and will trigger the ``build`` call after the run
  (see :ref:`cpp_standalone`).
* All state variables within a `Network` can now be accessed by `Network.get_states` and `Network.set_states` and the
  `store`/`restore` mechanism can now store the full state of a simulation to disk.
* Stochastic differential equations with multiplicative noise can now be integrated using the Euler-Heun method
  (``heun``). Thanks to Jan-Hendrik Schleimer for this contribution.
* Error messages have been significantly improved: errors for unit mismatches are now much clearer and error messages
  triggered during the intialization phase point back to the line of code where the relevant object (e.g. a
  `NeuronGroup`) was created.
* `PopulationRateMonitor` now provides a `~brian2.monitors.ratemonitor.PopulationRateMonitor.smooth_rate` method for a filtered version of the
  stored rates.

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* In addition to the new synapse creation syntax, sparse probabilistic connections are now created much faster.
* The time for the initialization phase at the beginning of a `run` has been significantly reduced.
* Multicompartmental simulations with a large number of compartments are now simulated more efficiently and are making
  better use of several processor cores when OpenMP is activated in C++ standalone mode. Thanks to Moritz Augustin for
  this contribution.
* Simulations will use compiler settings that optimize performance by default.
* Objects that have user-specified names are better supported for complex simulation scenarios (names no longer have to
  be unique at all times, but only across a network or across a standalone device).
* Various fixes for compatibility with recent versions of numpy and sympy

Important backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The argument names in `Synapses.connect` have changed and the first argument can no longer be an array of indices. To
  connect based on indices, use ``Synapses.connect(i=source_indices, j=target_indices)``. See :ref:`creating_synapses`
  and the documentation of `Synapses.connect` for more details.
* The actions triggered by pre-synaptic and post-synaptic spikes are now described by the ``on_pre`` and ``on_post``
  keyword arguments (instead of ``pre`` and ``post``).
* The `Morphology` object no longer allows to change attributes such as length and diameter after its creation. Complex
  morphologies should instead be created using the `Section` class, allowing for the specification of all details.
* `Morphology` objects that are defined with coordinates need to provide the start point (relative to the end point of
  the parent compartment) as the first coordinate. See :ref:`creating_morphology` for more details.
* For simulations using the C++ standalone mode, no longer call `Device.build` (if using a single `run` call), or
  use `set_device` with ``build_on_run=False`` (see :ref:`cpp_standalone`).

Infrastructure improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Our test suite is now also run on Mac OS-X (on the `Travis CI <https://travis-ci.org/>`_ platform).

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Moritz Augustin (`@moritzaugustin <https://github.com/moritzaugustin>`_)
* Jan-Hendrik Schleimer (`@ttxtea <https://github.com/ttxtea>`_)
* Romain Cazé (`@rcaze <https://github.com/rcaze>`_)
* Konrad Wartke (`@Kwartke <https://github.com/Kwartke>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
anyone we forgot...):

* Chaofei Hong
* Kees de Leeuw
* Luke Y Prince
* Myung Seok Shim
* Owen Mackwood
* Github users: @epaxon, @flinz, @mariomulansky, @martinosorb, @neuralyzer, @oleskiw, @prcastro, @sudoankit


Brian 2.0b4
-----------
This is the fourth (and probably last) beta release for Brian 2.0. This release
adds a few important new features and fixes a number of bugs so we recommend all
users of Brian 2 to upgrade. If you are a user new to Brian, we also recommend
to directly start with Brian 2 instead of using the stable release of Brian 1.
Note that the new recommended way to install Brian 2 is to use the Anaconda
distribution and to install the Brian 2 conda package (see :doc:`install`).

This is however still a Beta release, please report bugs or suggestions to the
github bug tracker (https://github.com/brian-team/brian2/issues) or to the
brian-development mailing list (brian-development@googlegroups.com).

Major new features
~~~~~~~~~~~~~~~~~~
* In addition to the standard threshold/reset, groups can now define "custom
  events". These can be recorded with the new `EventMonitor` (a generalization
  of `SpikeMonitor`) and `Synapses` can connect to these events instead of
  the standard spike event. See :doc:`../advanced/custom_events` for more
  details.
* `SpikeMonitor` and `EventMonitor` can now also record state variable values
  at the time of spikes (or custom events), thereby offering the functionality
  of ``StateSpikeMonitor`` from Brian 1. See
  :ref:`recording_variables_spike_time` for more details.
* The code generation modes that interact with C++ code (weave, Cython, and C++
  standalone) can now be more easily configured to work with external libraries
  (compiler and linker options, header files, etc.). See the documentation of
  the `~brian2.codegen.cpp_prefs` module for more details.

Improvemements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Cython simulations no longer interfere with each other when run in parallel
  (thanks to Daniel Bliss for reporting and fixing this).
* The C++ standalone now works with scalar delays and the spike queue
  implementation deals more efficiently with them in general.
* Dynamic arrays are now resized more efficiently, leading to faster monitors
  in runtime mode.
* The spikes generated by a `SpikeGeneratorGroup` can now be changed between
  runs using the
  `~brian2.input.spikegeneratorgroup.SpikeGeneratorGroup.set_spikes` method.
* Multi-step state updaters now work correctly for non-autonomous differential
  equations
* `PoissonInput` now correctly works with multiple clocks (thanks to Daniel
  Bliss for reporting and fixing this)
* The `~brian2.groups.group.Group.get_states` method now works for
  `StateMonitor`. This method provides a convenient way to access all the data
  stored in the monitor, e.g. in order to store it on disk.
* C++ compilation is now easier to get to work under Windows, see
  :doc:`install` for details.

Important backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The ``custom_operation`` method has been renamed to
  `~brian2.groups.group.Group.run_regularly` and can now be called without the
  need for storing its return value.
* `StateMonitor` will now by default record at the beginning of a time step
  instead of at the end. See :ref:`recording_variables_continuously` for
  details.
* Scalar quantities now behave as python scalars with respect to in-place
  modifications (augmented assignments). This means that
  ``x = 3*mV; y = x; y += 1*mV`` will no longer increase the value of the
  variable ``x`` as well.

Infrastructure improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* We now provide conda packages for Brian 2, making it very easy to install
  when using the Anaconda distribution (see :doc:`install`).

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Daniel Bliss (`@dabliss <https://github.com/dabliss>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
everyone we forgot...):

* Daniel Bliss
* Damien Drix
* Rainer Engelken
* Beatriz Herrera Figueredo
* Owen Mackwood
* Augustine Tan
* Ot de Wiljes


Brian 2.0b3
-----------
This is the third beta release for Brian 2.0. This release does not add many new
features but it fixes a number of important bugs so we recommend all users of
Brian 2 to upgrade. If you are a user new to Brian, we also recommend to
directly start with Brian 2 instead of using the stable release of Brian 1.

This is however still a Beta release, please report bugs or suggestions to the
github bug tracker (https://github.com/brian-team/brian2/issues) or to the
brian-development mailing list (brian-development@googlegroups.com).

Major new features
~~~~~~~~~~~~~~~~~~
* A new `PoissonInput` class for efficient simulation of Poisson-distributed
  input events.

Improvements
~~~~~~~~~~~~
* The order of execution for ``pre`` and ``post`` statements happending in the
  same time step was not well defined (it fell back to the default alphabetical
  ordering, executing ``post`` before ``pre``). It now explicitly specifies the
  ``order`` attribute so that ``pre`` gets executed before ``post`` (as in
  Brian 1). See the :doc:`../user/synapses` documentation for details.
* The default schedule that is used can now be set via a preference
  (`core.network.default_schedule`). New automatically generated scheduling
  slots relative to the explicitly defined ones can be used, e.g.
  ``before_resets`` or ``after_synapses``. See :ref:`scheduling` for details.
* The scipy_ package is no longer a dependency (note that weave_ for
  compiled C code under Python 2 is now available in a separate package). Note
  that multicompartmental models will still benefit from the scipy_ package
  if they are simulated in pure Python (i.e. with the ``numpy`` code generation
  target) -- otherwise Brian 2 will fall back to a numpy-only solution which is
  significantly slower.

Important bug fixes
~~~~~~~~~~~~~~~~~~~
* Fix `SpikeGeneratorGroup` which did not emit all the spikes under certain
  conditions for some code generation targets (#429)
* Fix an incorrect update of pre-synaptic variables in synaptic statements for
  the ``numpy`` code generation target (#435).
* Fix the possibility of an incorrect memory access when recording a subgroup
  with `SpikeMonitor` (#454).
* Fix the storing of results on disk for C++ standalone on Windows -- variables
  that had the same name when ignoring case (e.g. ``i`` and ``I``) where
  overwriting each other (#455).

Infrastructure improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Brian 2 now has a chat room on gitter_: https://gitter.im/brian-team/brian2
* The sphinx documentation can now be built from the release archive file
* After a big cleanup, all files in the repository have now simple LF line
  endings (see https://help.github.com/articles/dealing-with-line-endings/ on
  how to configure your own machine properly if you want to contribute to
  Brian).

.. _scipy: http://scipy.org
.. _weave: https://pypi.python.org/pypi/weave
.. _gitter: http://gitter.im

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Konrad Wartke (`@kwartke <https://github.com/Kwartke>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
everyone we forgot...):

* Daniel Bliss
* Owen Mackwood
* Ankur Sinha
* Richard Tomsett


Brian 2.0b2
-----------
This is the second beta release for Brian 2.0, we recommend all users of Brian 2
to upgrade. If you are a user new to Brian, we also recommend to directly start
with Brian 2 instead of using the stable release of Brian 1.

This is however still a Beta release, please report bugs or suggestions to the
github bug tracker (https://github.com/brian-team/brian2/issues) or to the
brian-development mailing list (brian-development@googlegroups.com).

Major new features
~~~~~~~~~~~~~~~~~~
* Multi-compartmental simulations can now be run using the
  :ref:`cpp_standalone` mode (this is not yet well-tested, though).
* The implementation of `TimedArray` now supports two-dimensional arrays, i.e.
  different input per neuron (or synapse, etc.), see :ref:`timed_arrays` for
  details.
* Previously, not setting a code generation target (using the `codegen.target`
  preference) would mean that the ``numpy`` target was used. Now,
  the default target is ``auto``, which means that a compiled language
  (``weave`` or ``cython``) will be used if possible. See
  :doc:`../user/computation` for details.
* The implementation of `SpikeGeneratorGroup` has been improved and it now
  supports a ``period`` argument to repeatedly generate a spike pattern.

Improvements
~~~~~~~~~~~~
* The selection of a numerical algorithm (if none has been specified by the
  user) has been simplified. See :ref:`numerical_integration` for details.
* Expressions that are shared among neurons/synapses are now updated only once
  instead of for every neuron/synapse which can lead to performance
  improvements.
* On Windows, The Microsoft Visual C compiler is now supported in the
  ``cpp_standalone`` mode, see the respective notes in the :doc:`install` and
  :doc:`../user/computation` documents.
* Simulation runs (using the standard "runtime" device) now collect profiling
  information. See :ref:`profiling` for details.

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :doc:`Tutorials for beginners <../resources/tutorials/index>` in the form of
  ipython notebooks (currently only covering the basics of neurons and synapses)
  are now available.
* The :doc:`../examples/index` in the documentation now include the images
  they generated. Several examples have been adapted from Brian 1.
* The code is now automatically tested on Windows machines, using the
  `appveyor <http://ci.appveyor.com>`_ service. This complements the Linux
  testing on `travis <https://travis-ci.org>`_.
* Using a version of a dependency (e.g. sympy) that we don't support will now
  raise an error when you import ``brian2`` -- see :ref:`dependency_checks` for
  more details.
* Test coverage for the ``cpp_standalone`` mode has been significantly
  increased.

Important bug fixes
~~~~~~~~~~~~~~~~~~~
* The preparation time for complicated equations has been significantly reduced.
* The string representation of small physical quantities has been corrected
  (#361)
* Linking variables from a group of size 1 now works correctly (#383)

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* Pierre Yger (`@yger <https://github.com/yger>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
everyone we forgot...):

* Conor Cox
* Gordon Erlebacher
* Konstantin Mergenthaler


Brian 2.0beta
-------------
This is the first beta release for Brian 2.0 and the first version of Brian 2.0
we recommend for general use. From now on, we will try to keep changes that
break existing code to a minimum. If you are a user new to Brian, we'd
recommend to start with the Brian 2 beta instead of using the stable release of
Brian 1.

This is however still a Beta release, please report bugs or suggestions to the
github bug tracker (https://github.com/brian-team/brian2/issues) or to the
brian-development mailing list (brian-development@googlegroups.com).

Major new features
~~~~~~~~~~~~~~~~~~
* New classes `Morphology` and `SpatialNeuron` for the simulation of
  :doc:`../user/multicompartmental`
* A temporary "bridge" for ``brian.hears`` that allows to use its Brian 1
  version from Brian 2 (:doc:`../user/brian1hears_bridge`)
* Cython is now a new code generation target, therefore the performance benefits
  of compiled code are now also available to users running simulations under
  Python 3.x (where ``scipy.weave`` is not available)
* Networks can now store their current state and return to it at a later time,
  e.g. for simulating multiple trials starting from a fixed network state
  (:ref:`continue_repeat`)
* C++ standalone mode: multiple processors are now supported via OpenMP
  (:ref:`openmp`), although this code has not yet been well tested so may be
  inaccurate.
* C++ standalone mode: after a run, state variables and monitored values can
  be loaded from disk transparently. Most scripts therefore only need two
  additional lines to use standalone mode instead of Brian's default runtime
  mode (:ref:`cpp_standalone`).

Syntax changes
~~~~~~~~~~~~~~
* The syntax and semantics of everything around simulation time steps, clocks,
  and multiple runs have been cleaned up, making ``reinit`` obsolete and also
  making it unnecessary for most users to explicitly generate `Clock` objects --
  instead, a ``dt`` keyword can be specified for objects such as `NeuronGroup`
  (:doc:`../user/running`)
* The ``scalar`` flag for parameters/subexpressions has been renamed to
  ``shared``
* The "unit" for boolean variables has been renamed from ``bool`` to ``boolean``
* C++ standalone: several keywords of
  `CPPStandaloneDevice.build <brian2.devices.cpp_standalone.device.CPPStandaloneDevice.build>`
  have been renamed
* The preferences are now accessible via ``prefs`` instead of ``brian_prefs``
* The ``runner`` method has been renamed to `~brian2.groups.group.Group.custom_operation`

Improvements
~~~~~~~~~~~~
* Variables can now be linked across `NeuronGroup`\ s (:ref:`linked_variables`)
* More flexible progress reporting system, progress reporting also works in the
  C++ standalone mode (:ref:`progress_reporting`)
* State variables can be declared as ``integer`` (:ref:`equation_strings`)

Bug fixes
~~~~~~~~~
57 github issues have been closed since the alpha release, of which 26 had been
labeled as bugs. We recommend all users of Brian 2 to upgrade.

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* Pierre Yger (`@yger <https://github.com/yger>`_)
* Werner Beroux (`@wernight <https://github.com/wernight>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
everyone we forgot…):

* Guillaume Bellec
* Victor Benichoux
* Laureline Logiaco
* Konstantin Mergenthaler
* Maurizio De Pitta
* Jan-Hendrick Schleimer
* Douglas Sterling
* Katharina Wilmes
