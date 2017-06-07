Release notes
=============

Brian 2.0.2
-----------

New features
~~~~~~~~~~~~
* `molar` and `liter` (as well as `litre`, scaled versions of the former, and a
  few useful abbreviations such as `mM`) have been added as new units (#574).
* A new module `brian2.units.constants` provides physical constants such as the
  Faraday constants or the gas constant (see :ref:`constants` for details).
* `SpatialNeuron` now supports non-linear membrane currents (e.g.
  Goldman–Hodgkin–Katz equations) by linearizing them with respect to v.
* Multi-compartmental models can access the capacitive current via `Ic` in
  their equations (#677)
* A new function `scheduling_summary` that displays information about the
  scheduling of all objects (see :ref:`scheduling` for details).
* Introduce a new preference to pass arguments to the ``make``/``nmake`` command
  in C++ standalone mode (`devices.cpp_standalone.extra_make_args_unix` for
  Linux/OS X and `devices.cpp_standalone.extra_make_args_windows` for Windows).
  For Linux/OS X, this enables parallel compilation by default.
* Anaconda packages for Brian 2 are now available for Python 3.6 (but Python 3.4
  support has been removed).

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Work around low performance for certain C++ standalone simulations on Linux,
  due to a bug in glibc (see #803). Thanks to Oleg Strikov
  (`@xj8z <https://github.com/xj8z>`_) for debugging this
  issue and providing the workaround that is now in use.
* Make exact integration of ``event-driven`` synaptic variables use the
  ``linear`` numerical integration algorithm (instead of ``independent``),
  fixing rare occasions where integration failed despite the equations being
  linear (#801).
* Better error messages for incorrect unit definitions in equations.
* Various fixes for the internal representation of physical units and the
  unit registration system.
* Fix a bug in the assignment of state variables in subtrees of `SpatialNeuron`
  (#822)
* Numpy target: fix an indexing error for a `SpikeMonitor` that records from a
  subgroup (#824)
* Summed variables targeting the same post-synaptic variable now raise an error
  (previously, only the one executed last was taken into account, see #766).
* Fix bugs in synapse generation affecting Cython (#781) respectively numpy
  (#835)
* C++ standalone simulations with many objects no longer fail on Windows (#787)

Backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `celsius` has been removed as a unit, because it was ambiguous in its relation
  to `kelvin` and gave wrong results when used as an absolute temperature (and
  not a temperature difference). For temperature differences, you can directly
  replace `celsius` by `kelvin`. To convert an absolute temperature in degree
  Celsius to Kelvin, add the `zero_celsius` constant from
  `brian2.units.constants` (#817).
* State variables are no longer allowed to have names ending in ``_pre`` or
  ``_post`` to avoid confusion with references to pre- and post-synaptic
  variables in `Synapses` (#818)

Changes to default settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* In C++ standalone mode, the ``clean`` argument now defaults to ``False``,
  meaning that ``make clean`` will not be executed by default before building
  the simulation. This avoids recompiling all files for unchanged simulations
  that are executed repeatedly. To return to the previous behaviour, specify
  ``clean=True`` in the ``device.build`` call (or in ``set_device`` if your
  script does not have an explicit ``device.build``).

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Thomas McColgan (`@phreeza <https://github.com/phreeza>`_)
* Daan Sprenkels (`@dsprenkels <https://github.com/dsprenkels>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* Oleg Strikov (`@xj8z <https://github.com/xj8z>`_)
* Charlee Fletterman (`@CharleeSF <https://github.com/CharleeSF>`_)
* Meng Dong (`@whenov <https://github.com/whenov>`_)
* Denis Alevi (`@denisalevi <https://github.com/denisalevi>`_)
* Mihir Vaidya (`@MihirVaidya94 <https://github.com/MihirVaidya94>`_)
* Adam (`@ffa <https://github.com/ffa>`_)
* Sourav Singh (`@souravsingh <https://github.com/souravsingh>`_)
* Nick Hale (`@nik849 <https://github.com/nik849>`_)
* Cody Greer (`@Cody-G <https://github.com/Cody-G>`_)
* Jean-Sébastien Dessureault (`@jsdessureault <https://github.com/jsdessureault>`_)
* Michele Giugliano (`@mgiugliano <https://github.com/mgiugliano>`_)
* Teo Stocco (`@zifeo <https://github.com/zifeo>`_)
* Edward Betts (`@EdwardBetts <https://github.com/EdwardBetts>`_)

Other contributions outside of github (ordered alphabetically, apologies to
anyone we forgot...):

* Christopher Nolan
* Regimantas Jurkus
* Shailesh Appukuttan

Brian 2.0.1
-----------
This is a bug-fix release that fixes a number of important bugs (see below),
but does not introduce any new features. We recommend all users of Brian 2 to
upgrade.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix `PopulationRateMonitor` for recordings from subgroups (#772)
* Fix `SpikeMonitor` for recordings from subgroups (#777)
* Check that string expressions provided as the ``rates`` argument for
  `PoissonGroup` have correct units.
* Fix compilation errors when multiple run statements with different ``report``
  arguments are used in C++ standalone mode.
* Several documentation updates and fixes

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Alex Seeholzer (`@flinz <https://github.com/flinz>`_)
* Meng Dong (`@whenov <https://github.com/whenov>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
anyone we forgot...):

* Myung Seok Shim
* Pamela Hathway


Brian 2.0 (changes since 1.4)
-----------------------------

Major new features
~~~~~~~~~~~~~~~~~~

* Much more flexible model definitions. The behaviour of all model elements
  can now be defined by arbitrary equations specified in standard
  mathematical notation.

* Code generation as standard. Behind the scenes, Brian automatically generates
  and compiles C++ code to simulate your model, making it much faster.

* "Standalone mode". In this mode, Brian generates a complete C++ project tree
  that implements your model. This can be then be compiled and run entirely
  independently of Brian. This leads to both highly efficient code, as well as
  making it much easier to run simulations on non-standard computational
  hardware, for example on robotics platforms.

* Multicompartmental modelling.

* Python 2 and 3 support.

New features
~~~~~~~~~~~~

* Installation should now be much easier, especially if using the
  Anaconda Python distribution. See :doc:`/introduction/install`.
* Many improvements to `Synapses` which replaces the old ``Connection``
  object in Brian 1. This includes:
  synapses that are triggered by non-spike events; synapses that target
  other synapses; huge speed improvements thanks to using code generation;
  new "generator syntax" when creating synapses is much more flexible and
  efficient. See :doc:`/user/synapses`.
* New model definitions allow for much more flexible refractoriness. See
  :doc:`/user/refractoriness`.
* `SpikeMonitor` and `StateMonitor` are now much more flexible, and cover a
  lot of what used to be covered by things like ``MultiStateMonitor``, etc.
  See :doc:`/user/recording`.
* Multiple event types. In addition to the default ``spike`` event, you can
  create arbitrary events, and have these trigger code blocks (like reset)
  or synaptic events. See :doc:`/advanced/custom_events`.
* New units system allows arrays to have units. This eliminates the need for
  a lot of the special casing that was required in Brian 1. See
  :doc:`/user/units`.
* Indexing variable by condition, e.g. you might write ``G.v['x>0']`` to
  return all values of variable ``v`` in `NeuronGroup` ``G`` where the
  group's variable ``x>0``. See :ref:`state_variables`.
* Correct numerical integration of stochastic differential equations.
  See :doc:`/user/numerical_integration`.
* "Magic" `run` system has been greatly simplified and is now much more
  transparent. In addition, if there is any ambiguity about what the user
  wants to run, an erorr will be raised rather than making a guess. This
  makes it much safer. In addition, there is now a `store`/`restore`
  mechanism that simplifies restarting simulations and managing separate
  training/testing runs. See :doc:`/user/running`.
* Changing an external variable between runs now works as expected, i.e.
  something like ``tau=1*ms; run(100*ms); tau=5*ms; run(100*ms)``. In
  Brian 1 this would have used ``tau=1*ms`` for both runs. More generally,
  in Brian 2 there is now better control over namespaces. See
  :doc:`/advanced/namespaces`.
* New "shared" variables with a single value shared between all neurons.
  See :ref:`shared_variables`.
* New `Group.run_regularly` method for a codegen-compatible way of doing
  things that used to be done with `network_operation` (which can still
  be used). See :ref:`regular_operations`.
* New system for handling externally defined functions. They have to specify
  which units they accept in their arguments, and what they return. In
  addition, you can easily specify the implementation of user-defined
  functions in different languages for code generation. See
  :doc:`/advanced/functions`.
* State variables can now be defined as integer or boolean values.
  See :doc:`/user/equations`.
* State variables can now be exported directly to Pandas data frame.
  See :ref:`storing_state_variables`.
* New generalised "flags" system for giving additional information when
  defining models. See :ref:`flags`.
* `TimedArray` now allows for 2D arrays with arbitrary indexing.
  See :ref:`timed_arrays`.
* Better support for using Brian in IPython/Jupyter. See, for example,
  `start_scope`.
* New preferences system. See :doc:`/advanced/preferences`.
* Random number generation can now be made reliably reproducible.
  See :doc:`/advanced/random`.
* New profiling option to see which parts of your simulation are taking
  the longest to run. See :ref:`profiling`.
* New logging system allows for more precise control. See
  :doc:`/advanced/logging`.
* New ways of importing Brian for advanced Python users. See
  :doc:`/user/import`.
* Improved control over the order in which objects are updated during
  a run. See :doc:`/advanced/scheduling`.
* Users can now easily define their own numerical integration methods.
  See :doc:`/advanced/state_update`.
* Support for parallel processing using the OpenMP version of
  standalone mode. Note that all Brian tests pass with this, but it is
  still considered to be experimental. See :ref:`openmp`.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`brian1_to_2/index`.

Behind the scenes changes
~~~~~~~~~~~~~~~~~~~~~~~~~

* All user models are now passed through the code generation system.
  This allows us to be much more flexible about introducing new target
  languages for generated code to make use of non-standard computational
  hardware. See :doc:`/developer/codegen`.
* New standalone/device mode allows generation of a complete project tree
  that can be compiled and built independently of Brian and Python. This
  allows for even more flexible use of Brian on non-standard hardware.
  See :doc:`/developer/devices`.
* All objects now have a unique name, used in code generation. This can
  also be used to access the object through the `Network` object.

Contributions
~~~~~~~~~~~~~
Full list of all Brian 2 contributors, ordered by the time of their first
contribution:

* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* Cyrille Rossant (`@rossant <https://github.com/rossant>`_)
* Victor Benichoux (`@victorbenichoux <https://github.com/victorbenichoux>`_)
* Pierre Yger (`@yger <https://github.com/yger>`_)
* Werner Beroux (`@wernight <https://github.com/wernight>`_)
* Konrad Wartke (`@Kwartke <https://github.com/Kwartke>`_)
* Daniel Bliss (`@dabliss <https://github.com/dabliss>`_)
* Jan-Hendrik Schleimer (`@ttxtea <https://github.com/ttxtea>`_)
* Moritz Augustin (`@moritzaugustin <https://github.com/moritzaugustin>`_)
* Romain Cazé (`@rcaze <https://github.com/rcaze>`_)
* Dominik Krzemiński (`@dokato <https://github.com/dokato>`_)
* Martino Sorbaro (`@martinosorb <https://github.com/martinosorb>`_)
* Benjamin Evans (`@bdevans <https://github.com/bdevans>`_)


Brian 2.0 (changes since 2.0rc3)
--------------------------------

New features
~~~~~~~~~~~~
* A new flag ``constant over dt`` can be applied to subexpressions to have them
  only evaluated once per timestep (see :doc:`../user/models`). This flag is
  mandatory for stateful subexpressions, e.g. expressions using ``rand()`` or
  ``randn()``. (#720, #721)

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix `EventMonitor.values` and `SpikeMonitor.spike_trains` to always return
  sorted spike/event times (#725).
* Respect the ``active`` attribute in C++ standalone mode (#718).
* More consistent check of compatible time and dt values (#730).
* Attempting to set a synaptic variable or to start a simulation with synapses
  without any preceding connect call now raises an error (#737).
* Improve the performance of coordinate calculation for `Morphology` objects,
  which previously made plotting very slow for complex morphologies (#741).
* Fix a bug in `SpatialNeuron` where it did not detect non-linear dependencies
  on v, introduced via point currents (#743).

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* An interactive demo, tutorials, and examples can now be run in an interactive
  jupyter notebook on the `mybinder <http://mybinder.org/>`_ platform, without
  any need for a local Brian installation (#736). Thanks to Ben Evans for the
  idea and help with the implementation.
* A new extensive guide for converting Brian 1 simulations to Brian 2 user
  coming from Brian 1: :doc:`changes`
* A re-organized :doc:`../user/index`, with clearer indications which
  information is important for new Brian users.

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Benjamin Evans (`@bdevans <https://github.com/bdevans>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
anyone we forgot...):

* Chaofei Hong
* Daniel Bliss
* Jacopo Bono
* Ruben Tikidji-Hamburyan


Brian 2.0rc3
------------
This is another "release candidate" for Brian 2.0 that fixes a range of bugs and introduces
better support for random numbers (see below). We are getting close to the final Brian 2.0
release, the remaining work will focus on bug fixes, and better error messages and
documentation.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development mailing
list (brian-development@googlegroups.com).

New features
~~~~~~~~~~~~
* Brian now comes with its own `seed` function, allowing to seed the random number generator
  and thereby to make simulations reproducible. This function works for all code generation
  targets and in runtime and standalone mode. See :doc:`../advanced/random` for details.
* Brian can now export/import state variables of a group or a full network to/from a
  `pandas <http://pandas.pydata.org>`_ ``DataFrame`` and comes with a mechanism to extend
  this to other formats. Thanks to Dominik Krzemiński for this contribution (see #306).

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Use a Mersenne-Twister pseudorandom number generator in C++ standalone mode, replacing the
  previously used low-quality random number generator from the C standard library (see #222,
  #671 and #706).
* Fix a memory leak in code running with the weave code generation target, and a smaller
  memory leak related to units stored repetitively in the `~brian2.units.fundamentalunits.UnitRegistry`.
* Fix a difference of one timestep in the number of simulated timesteps between
  runtime and standalone that could arise for very specific values of dt and t (see #695).
* Fix standalone compilation failures with the most recent gcc version which defaults to
  C++14 mode (see #701)
* Fix incorrect summation in synapses when using the ``(summed)`` flag and writing to
  *pre*-synaptic variables (see #704)
* Make synaptic pathways work when connecting groups that define nested subexpressions,
  instead of failing with a cryptic error message (see #707).

Contributions
~~~~~~~~~~~~~
Code and documentation contributions (ordered by the number of commits):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dominik Krzemiński (`@dokato <https://github.com/dokato>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Martino Sorbaro (`@martinosorb <https://github.com/martinosorb>`_)

Testing, suggestions and bug reports (ordered alphabetically, apologies to
anyone we forgot...):

* Craig Henriquez
* Daniel Bliss
* David Higgins
* Gordon Erlebacher
* Max Gillett
* Moritz Augustin
* Sami Abdul-Wahid


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
  version from Brian 2 (:doc:`brian1_to_2/brian1hears_bridge`)
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
