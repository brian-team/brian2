Release notes
=============

Next version
------------

New features
~~~~~~~~~~~~
TODO

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Removing objects from networks no longer fails (:issue:`1151`). Thanks to Wilhelm
  Braun for reporting the issue.
* Point currents marked as ``constant over dt`` are now correctly handled
  (:issue:`1160`). Thanks to Andrew Brughera for reporting the issue.
* Elapsed and estimated remaining time are now formatted as hours/minutes/etc.
  in standalone mode as well (:issue:`1162`). Thanks to Rahul Kumar Gupta,
  Syed Osama Hussain, Bhuwan Chandra, and Vigneswaran Chandrasekaran for working
  on this issue as part of the GSoC 2020 application process.

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

TODO


Other contributions outside of github (ordered alphabetically, apologies to
anyone we forgot...):

* Andrew Brughera
* Wilhelm Braun
* Willam Xavier


.. _brian2.3:
Brian 2.3
---------
This release contains the usual mix of bug fixes and new features (see below), but
also makes some important changes to the Brian 2 code base to pave the way for
the full Python 2 -> 3 transition (the source code is now directly compatible with
Python 2 and Python 3, without the need for any translation at install time). Please
note that this release will be the last release that supports
Python 2, given that Python 2 reaches end-of-life in January 2020. Brian now also uses
`pytest <https://docs.pytest.org>`_ as its testing framework, since the previously used
``nose`` package is not maintained anymore. Since `brian2hears <https://brian2hears.readthedocs.io>`_
has been released as an independent package, using `brian2.hears` as a "bridge" to
Brian 1's ``brian.hears`` package is now deprecated.

Finally, the Brian project has adopted the "Contributor Covenant"
:doc:`code_of_conduct`, pledging "to make participation in our community a
harassment-free experience for everyone".

New features
~~~~~~~~~~~~
* The `restore` function can now also restore the state of the random number generator,
  allowing for exact reproducibility of stochastic simulations (:issue:`1134`)
* The functions `expm1`, `log1p`, and `exprel` can now be used (:issue:`1133`)
* The system for calling random number generating functions has been generalized (see
  :ref:`function_vectorisation`), and a new `poisson` function for Poisson-distrubted
  random numbers has been added (:issue:`1111`)
* New versions of Visual Studio are now supported for standalone mode on Windows
  (:issue:`1135`)

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `~brian2.groups.group.Group.run_regularly` operations are now included in the network, even if they are
  created after the parent object was added to the network (:issue:`1009`).
  Contributed by `Vigneswaran Chandrasekaran <https://github.com/Vigneswaran-Chandrasekaran>`_.
* No longer incorrectly classify some equations as having "multiplicative noise" (:issue:`968`).
  Contributed by `Vigneswaran Chandrasekaran <https://github.com/Vigneswaran-Chandrasekaran>`_.
* Brian is now compatible with Python 3.8 (:issue:`1130`), and doctests are compatible
  with numpy 1.17 (:issue:`1120`)
* Progress reports for repeated runs have been fixed (:issue:`1116`), thanks to Ronaldo
  Nunes for reporting the issue.
* `SpikeGeneratorGroup` now correctly works with `restore` (:issue:`1084`), thanks to
  Tom Achache for reporting the issue.
* An indexing problem in `PopulationRateMonitor` has been fixed (:issue:`1119`).
* Handling of equations referring to ``-inf`` has been fixed (:issue:`1061`).
* Long simulations recording more than ~2 billion data points no longer crash with a
  segmentation fault (:issue:`1136`), thanks to Rike-Benjamin Schuppner for reporting
  the issue.

Backward-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The fix for `~brian2.groups.group.Group.run_regularly` operations (:issue:`1009`, see above) entails
  a change in how objects are stored within `Network` objects. Previously, `Network.objects` stored a
  complete list of all objects, including objects such as `~brian2.groups.neurongroup.StateUpdater` that
  – often invisible to the user – are a part of major objects such as
  `NeuronGroup`. Now, `Network.objects` only stores the objects directly
  provided by the user (`NeuronGroup`, `Synapses`, `StateMonitor`, ...), the
  dependent objects (`~brian2.groups.neurongroup.StateUpdater`, `~brian2.groups.neurongroup.Thresholder`, ...) are taken into account
  at the time of the run. This might break code in some corner cases, e.g.
  when removing a `~brian2.groups.neurongroup.StateUpdater` from `Network.objects` via `Network.remove`.
* The `brian2.hears` interface to Brian 1's ``brian.hears`` package has been deprecated.

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The same code base is used on Python 2 and Python 3 (:issue:`1073`).
* The test framework uses ``pytest`` (:issue:`1127`).
* We have adapoted a Code of Conduct (:issue:`1113`), thanks to Tapasweni Pathak for the
  suggestion.

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Vigneswaran Chandrasekaran (`@Vigneswaran-Chandrasekaran <https://github.com/Vigneswaran-Chandrasekaran>`_)
* Moritz Orth (`@morth <https://github.com/morth>`_)
* Tristan Stöber (`@tristanstoeber <https://github.com/tristanstoeber>`_)
* `@ulyssek <https://github.com/ulyssek>`_
* Wilhelm Braun (`@wilhelmbraun <https://github.com/wilhelmbraun>`_)
* `@flomlo <https://github.com/flomlo>`_
* Rike-Benjamin Schuppner (`@Debilski <https://github.com/Debilski>`_)
* `@sdeiss <https://github.com/sdeiss>`_
* Ben Evans (`@bdevans <https://github.com/bdevans>`_)
* Tapasweni Pathak (`@tapaswenipathak <https://github.com/tapaswenipathak>`_)
* `@jonathanoesterle <https://github.com/jonathanoesterle>`_
* Richard C Gerkin (`@rgerkin <https://github.com/rgerkin>`_)
* Christian Behrens (`@chbehrens <https://github.com/chbehrens>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* XiaoquinNUDT (`@XiaoquinNUDT <https://github.com/XiaoquinNUDT>`_)
* Dylan Muir (`@DylanMuir <https://github.com/DylanMuir>`_)
* Aleksandra Teska (`@alTeska <https://github.com/alTeska>`_)
* Felix Z. Hoffmann (`@felix11h <https://github.com/felix11h>`__)
* `@baixiaotian63648995 <https://github.com/baixiaotian63648995>`_
* Carlos de la Torre (`@c-torre <https://github.com/c-torre>`_)
* Sam Mathias (`@sammosummo <https://github.com/sammosummo>`_)
* `@Marghepano <https://github.com/Marghepano>`_
* Simon Brodeur (`@sbrodeur <https://github.com/sbrodeur>`_)
* Alex Dimitrov (`@adimitr <https://github.com/adimitr>`_)


Other contributions outside of github (ordered alphabetically, apologies to
anyone we forgot...):

* Ronaldo Nunes
* Tom Achache

Brian 2.2.2.1
-------------
This is a bug-fix release that fixes several bugs and adds a few minor new
features. We recommend all users of Brian 2 to upgrade.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

[Note that the original upload of this release was version 2.2.2, but due to
a mistake in the released archive, it has been uploaded again as version 2.2.2.1]

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix an issue with the synapses generator syntax (:issue:`1037`).
* Fix an incorrect error when using a `SpikeGeneratorGroup` with a long period
  (:issue:`1041`). Thanks to Kévin Cuallado-Keltsch for reporting this issue.
* Improve the performance of `SpikeGeneratorGroup` by avoiding a conversion
  from time to integer time step (:issue:`1043`). This time step is now also
  available to user code as ``t_in_timesteps``.
* Function definitions for weave/Cython/C++ standalone can now declare
  additional header files and libraries. They also support a new ``sources``
  argument to use a function definition from an external file. See the
  :doc:`../advanced/functions` documentation for details.
* For convenience, single-neuron subgroups can now be created with a single
  index instead of with a slice (e.g. ``neurongroup[3]`` instead of
  ``neurongroup[3:4]``).
* Fix an issue when ``-inf`` is used in an equation (:issue:`1061`).

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Felix Z. Hoffmann (`@Felix11H <https://github.com/Felix11H>`_)
* `@wjx0914 <https://github.com/wjx0914>`_
* Kévin Cuallado-Keltsch (`@kevincuallado <https://github.com/kevincuallado>`_)
* Romain Cazé (`@rcaze <https://github.com/rcaze>`_)
* Daphne (`@daphn3cor <https://github.com/daphn3cor>`_)
* Erik (`@parenthetical-e <https://github.com/parenthetical-e>`_)
* `@RahulMaram <https://github.com/RahulMaram>`_
* Eghbal Hosseini (`@eghbalhosseini <https://github.com/eghbalhosseini>`_)
* Martino Sorbaro (`@martinosorb <https://github.com/martinosorb>`_)
* Mihir Vaidya (`@MihirVaidya94 <https://github.com/MihirVaidya94>`_)
* `@hellolingling <https://github.com/hellolingling>`_
* Volodimir Slobodyanyuk (`@vslobody <https://github.com/vslobody>`_)
* Peter Duggins (`@psipeter <https://github.com/psipeter>`_)


Brian 2.2.1
-----------
This is a bug-fix release that fixes a few minor bugs and incompatibilites with
recent versions of the dependencies. We recommend all users of Brian 2 to
upgrade.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Work around problems with the latest version of ``py-cpuinfo`` on Windows
  (:issue:`990`, :issue:`1020`) and no longer require it for Linux and OS X.
* Avoid warnings with newer versions of Cython (:issue:`1030`) and correctly
  build the Cython spike queue for Python 3.7 (:issue:`1026`), thanks to Fleur
  Zeldenrust and Ankur Sinha for reporting these issues.
* Fix error messages for ``SyntaxError`` exceptions in jupyter notebooks
  (:issue:`#964`).

Dependency and packaging changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Conda packages in `conda-forge <https://conda-forge.org/>`_ are now avaible
  for Python 3.7 (but no longer for Python 3.5).
* Linux and OS X no longer depend on the ``py-cpuinfo`` package.
* Source packages on `pypi <https://pypi.org/>`_ now require a recent Cython
  version for installation.

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Christopher (`@Chris-Currin <https://github.com/Chris-Currin>`_)
* Peter Duggins (`@psipeter <https://github.com/psipeter>`_)
* Paola Suárez (`@psrmx <https://github.com/psrmx>`_)
* Ankur Sinha (`@sanjayankur31 <https://github.com/sanjayankur31>`_)
* `@JingjinW <https://github.com/JingjinW>`_
* Denis Alevi (`@denisalevi <https://github.com/denisalevi>`_)
* `@lemonade117 <https://github.com/lemonade117>`_
* `@wjx0914 <https://github.com/wjx0914>`_
* Sven Leach (`@SvennoNito <https://github.com/SvennoNito>`_)
* svadams (`@svadams <https://github.com/svadams>`_)
* `@ghaessig <https://github.com/ghaessig>`_
* Varshith Sreeramdass (`@varshiths <https://github.com/varshiths>`_)


Brian 2.2
---------
This releases fixes a number of important bugs and comes with a number of
performance improvements. It also makes sure that simulation no longer give
platform-dependent results for certain corner cases that involve the division of
integers. These changes can break backwards-compatiblity in certain cases, see
below.  We recommend all users of Brian 2 to upgrade.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Divisions involving integers now use floating point division, independent of
  Python version and code generation target. The `//` operator can now used in
  equations and expressions to denote flooring division (:issue:`984`).
* Simulations can now use single precision instead of double precision floats in
  simulations (:issue:`981`, :issue:`1004`). This is mostly intended for use
  with GPU code generation targets.
* The `~brian2.core.functions.timestep`, introduced in version 2.1.3, was
  further optimized for performance, making the refractoriness calculation
  faster (:issue:`996`).
* The ``lastupdate`` variable is only automatically added to synaptic models
  when event-driven equations are used, reducing the memory and performance
  footprint of simple synaptic models (:issue:`1003`). Thanks to Denis Alevi
  for bringing this up.
* A ``from brian2 import *`` imported names unrelated to Brian, and overwrote
  some Python builtins such as ``dir`` (:issue:`969`). Now, fewer names are
  imported (but note that this still includes numpy and plotting tools:
  :doc:`../user/import`).
* The ``exponential_euler`` state updater is no longer failing for systems of
  equations with differential equations that have trivial, constant
  right-hand-sides (:issue:`1010`). Thanks to Peter Duggins for making us aware
  of this issue.

Backward-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Code that divided integers (e.g. ``N/10``) with a C-based code generation
  target, or with the ``numpy`` target on Python 2, will now use floating point
  division instead of flooring division (i.e., Python 3 semantics). A warning
  will notify the user of this change, use either the flooring division operator
  (``N//10``), or the ``int`` function (``int(N/10)``) to make the expression
  unambiguous.
* Code that directly referred to the ``lastupdate`` variable in synaptic
  statements, without using any event-driven variables, now has to manually add
  ``lastupdate : second`` to the equations and update the variable at the end
  of ``on_pre`` and/or ``on_post`` with ``lastupdate = t``.
* Code that relied on ``from brian2 import *`` also importing unrelated names
  such as ``sympy``, now has to import such names explicitly.

Documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Various small fixes and additions (e.g. installation instructions, available
  functions, fixes in examples)
* A new example, :doc:`Izhikevich 2007 <../examples/frompapers.Izhikevich_2007>`,
  provided by `Guillaume Dumas <https://github.com/deep-introspection>`_.

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Denis Alevi (`@denisalevi <https://github.com/denisalevi>`_)
* Thomas Nowotny (`@tnowotny <https://github.com/tnowotny>`_)
* `@neworderofjamie <https://github.com/neworderofjamie>`_
* Paul Brodersen (`@paulbrodersen <https://github.com/paulbrodersen>`_)
* `@matrec4 <https://github.com/matrec4>`_
* svadams (`@svadams <https://github.com/svadams>`_)
* XiaoquinNUDT (`@XiaoquinNUDT <https://github.com/XiaoquinNUDT>`_)
* Peter Duggins (`@psipeter <https://github.com/psipeter>`_)
* `@nh17937 <https://github.com/nh17937>`_
* Patrick Nave (`@pnave95 <https://github.com/pnave95>`_)
* `@AI-pha <https://github.com/AI-pha>`_
* Guillaume Dumas (`@deep-introspection <https://github.com/deep-introspection>`_)
* `@godelicbach <https://github.com/godelicbach>`_
* `@galharth <https://github.com/galharth>`_


Brian 2.1.3.1
-------------
This is a bug-fix release that fixes two bugs in the recent 2.1.3 release:

* Fix an inefficiency in the newly introduced `~brian2.core.functions.timestep`
  function when using the ``numpy`` target (:issue:`965`)
* Fix inefficiencies in the unit system that could lead to slow operations
  and high memory use (:issue:`967`). Thanks to Kaustab Pal for making us
  aware of the issue.

Brian 2.1.3
-----------
This is a bug-fix release that fixes a number of important bugs (see below),
but does not introduce any new features. We recommend all users of Brian 2 to
upgrade.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The Cython cache on disk now uses significantly less space by deleting
  unnecessary source files (set the `codegen.runtime.cython.delete_source_files`
  preference to ``False`` if you want to keep these files for debugging). In
  addition, a warning will be given when the Cython or weave cache exceeds a
  configurable size (`codegen.max_cache_dir_size`). The
  `~brian2.__init__.clear_cache` function is provided to delete files from the
  cache (:issue:`914`).
- The C++ standalone mode now respects the ``profile`` option and therefore no
  longer collects profiling information by default. This can speed up
  simulations in certain cases (:issue:`935`).
- The exact number of time steps that a neuron stays in the state of
  refractoriness after a spike could vary by up to one time step when the
  requested refractory time was a multiple of the simulation time step. With
  this fix, the number of time steps is ensured to be as expected by making
  use of a new `~brian2.core.functions.timestep` function that avoids floating
  point rounding issues (:issue:`949`, first reported by
  `@zhouyanasd <https://github.com/zhouyanasd>`_ in issue :issue:`943`).
- When `restore` was called twice for a network, spikes that were not yet
  delivered to their target were not restored correctly (:issue:`938`, reported by
  `@zhouyanasd <https://github.com/zhouyanasd>`_).
- `SpikeGeneratorGroup` now uses a more efficient method for sorting spike
  indices and times, leading to a much faster preparation time for groups that
  store many spikes (:issue:`948`).
- Fix a memory leak in `TimedArray` (:issue:`923`, reported by Wilhelm Braun).
- Fix an issue with summed variables targetting subgroups (:issue:`925`,
  reported by `@AI-pha <https://github.com/AI-pha>`_).
- Fix the use of `~brian2.groups.group.Group.run_regularly` on subgroups
  (:issue:`922`, reported by `@AI-pha <https://github.com/AI-pha>`_).
- Improve performance for `SpatialNeuron` by removing redundant computations
  (:issue:`910`, thanks to `Moritz Augustin <https://github.com/moritzaugustin>`_
  for making us aware of the issue).
- Fix linked variables that link to scalar variables (:issue:`916`)
- Fix warnings for numpy 1.14 and avoid compilation issues when switching
  between versions of numpy (:issue:`913`)
- Fix problems when using logical operators in code generated for the numpy
  target which could lead to issues such as wrongly connected synapses
  (:issue:`901`, :issue:`900`).

Backward-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- No longer allow ``delay`` as a variable name in a synaptic model to avoid
  ambiguity with respect to the synaptic delay. Also no longer allow access to
  the ``delay`` variable in synaptic code since there is no way to distinguish
  between pre- and post-synaptic delay (:issue:`927`, reported by Denis Alevi).
- Due to the changed handling of refractoriness (see bug fixes above),
  simulations that make use of refractoriness will possibly no longer give
  exactly the same results. The preference `legacy.refractory_timing` can
  be set to ``True`` to reinstate the previous behaviour.

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- From this version on, conda packages will be available on
  `conda-forge <https://conda-forge.org/>`_. For a limited time, we will copy
  over packages to the ``brian-team`` channel as well.
- Conda packages are no longer tied to a specific numpy version (PR :issue:`954`)
- New example (:doc:`Brunel & Wang, 2001 <../examples/frompapers.Brunel_Wang_2001>`)
  contributed by `Teo Stocco <https://github.com/zifeo>`_ and
  `Alex Seeholzer <https://github.com/flinz>`_.

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Teo Stocco (`@zifeo <https://github.com/zifeo>`_)
* Dylan Muir (`@DylanMuir <https://github.com/DylanMuir>`_)
* scarecrow (`@zhouyanasd <https://github.com/zhouyanasd>`_)
* `@fuadfukhasyi <https://github.com/fuadfukhasyi>`_
* Aditya Addepalli (`@Dyex719 <https://github.com/Dyex719>`_)
* Kapil kumar (`@kapilkd13 <https://github.com/kapilkd13>`_)
* svadams (`@svadams <https://github.com/svadams>`_)
* Vafa Andalibi (`@Vafa-Andalibi <https://github.com/Vafa-Andalibi>`_)
* Sven Leach (`@SvennoNito <https://github.com/SvennoNito>`_)
* `@matrec4 <https://github.com/matrec4>`_
* `@jarishna <https://github.com/jarishna>`_
* `@AI-pha <https://github.com/AI-pha>`_
* `@xdzhangxuejun <https://github.com/xdzhangxuejun>`_
* Denis Alevi (`@denisalevi <https://github.com/denisalevi>`_)
* Paul Pfeiffer (`@pfeffer90 <https://github.com/pfeffer90>`_)
* Romain Brette (`@romainbrette <https://github.com/romainbrette>`_)
* `@hustyanghui <https://github.com/hustyanghui>`_
* Adrien F. Vincent (`@afvincent <https://github.com/afvincent>`_)
* `@ckemere <https://github.com/ckemere>`_
* `@evearmstrong <https://github.com/evearmstrong>`_
* Paweł Kopeć (`@pawelkopec <https://github.com/pawelkopec>`_)
* Moritz Augustin (`@moritzaugustin <https://github.com/moritzaugustin>`_)
* Bart (`@louwers <https://github.com/louwers>`_)
* `@amarsdd <https://github.com/amarsdd>`_
* `@ttxtea <https://github.com/ttxtea>`_
* Maria Cervera (`@MariaCervera <https://github.com/MariaCervera>`_)
* ouyangxinrong (`@longzhixin <https://github.com/longzhixin>`_)

Other contributions outside of github (ordered alphabetically, apologies to
anyone we forgot...):

* Wilhelm Braun

Brian 2.1.2
-----------
This is another bug fix release that fixes a major bug in `Equations`'
substitution mechanism (:issue:`896`). Thanks to Teo Stocco for reporting this issue.

Brian 2.1.1
-----------
This is a bug fix release that re-activates parts of the caching mechanism for
code generation that had been erroneously deactivated in the previous release.

Brian 2.1
---------
This release introduces two main new features: a new "GSL integration" mode for
differential equation that offers to integrate equations with variable-timestep
methods provided by the GNU Scientific Library, and caching for the run
preparation phase that can significantly speed up simulations. It also comes
with a newly written tutorial, as well as additional documentation and examples.

As always, please report bugs or suggestions to the github bug tracker
(https://github.com/brian-team/brian2/issues) or to the brian-development
mailing list (brian-development@googlegroups.com).

New features
~~~~~~~~~~~~
* New numerical integration methods with variable time-step integration, based
  on the GNU Scientific Library (see :ref:`numerical_integration`). Contributed
  by `Charlee Fletterman <https://github.com/CharleeSF>`_, supported by 2017's
  `Google Summer of Code <https://summerofcode.withgoogle.com>`_ program.
* New caching mechanism for the code generation stage (application of numerical
  integration algorithms, analysis of equations and statements, etc.), reducing
  the preparation time before the actual run, in particular for simulations with
  multiple `run` statements.

Selected improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix a rare problem in Cython code generation caused by missing type information (:issue:`893`)
* Fix warnings about improperly closed files on Python 3.6 (:issue:`892`;
  reported and fixed by `Teo Stocco <https://github.com/zifeo>`_)
* Fix an error when using numpy integer types for synaptic indexing (:issue:`888`)
* Fix an error in numpy codegen target, triggered when assigning to a variable with an unfulfilled condition (:issue:`887`)
* Fix an error when repeatedly referring to subexpressions in multiline statements (:issue:`880`)
* Shorten long arrays in warning messages (:issue:`874`)
* Enable the use of ``if`` in the shorthand generator syntax for `Synapses.connect` (:issue:`873`)
* Fix the meaning of ``i`` and ``j`` in synapses connecting to/from other synapses (:issue:`854`)

Backward-incompatible changes and deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* In C++ standalone mode, information about the number of synapses and spikes
  will now only be displayed when built with ``debug=True`` (:issue:`882`).
* The ``linear`` state updater has been renamed to ``exact`` to avoid confusion
  (:issue:`877`). Users are encouraged to use ``exact``, but the name ``linear``
  is still available and does not raise any warning or error for now.
* The ``independent`` state updater has been marked as deprecated and might be
  removed in future versions.

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A new, more advanced, :doc:`tutorial <../resources/tutorials/3-intro-to-brian-simulations>` "about
  managing the slightly more complicated tasks that crop up in research
  problems, rather than the toy examples we’ve been looking at so far."
* Additional documentation on :doc:`../advanced/custom_events` and
  :doc:`../user/converting_from_integrated_form` (including example code for
  typical synapse models).
* New example code reproducing published findings (:doc:`Platkiewicz and Brette, 2011 <../examples/frompapers.Platkiewicz_Brette_2011>`;
  :ref:`Stimberg et al., 2018 <frompapers.stimberg_et_al_2018>`)
* Fixes to the sphinx documentation creation process, the documentation can be downloaded as a PDF once again (705 pages!)
* Conda packages now have support for numpy 1.13 (but support for numpy 1.10 and 1.11 has been removed)

Contributions
~~~~~~~~~~~~~
Github code, documentation, and issue contributions (ordered by the number of
contributions):

* Marcel Stimberg (`@mstimberg <https://github.com/mstimberg>`_)
* Charlee Fletterman (`@CharleeSF <https://github.com/CharleeSF>`_)
* Dan Goodman (`@thesamovar <https://github.com/thesamovar>`_)
* Teo Stocco (`@zifeo <https://github.com/zifeo>`_)
* `@k47h4 <https://github.com/k47h4>`_

Other contributions outside of github (ordered alphabetically, apologies to
anyone we forgot...):

* Chaofei Hong
* Lucas ("lucascdst")


Brian 2.0.2.1
-------------

Fixes a bug in the tutorials' HMTL rendering on readthedocs.org (code blocks
were not displayed). Thanks to Flora Bouchacourt for making us aware of this
problem.

Brian 2.0.2
-----------

New features
~~~~~~~~~~~~
* `molar` and `liter` (as well as `litre`, scaled versions of the former, and a
  few useful abbreviations such as `mM`) have been added as new units (:issue:`574`).
* A new module `brian2.units.constants` provides physical constants such as the
  Faraday constants or the gas constant (see :ref:`constants` for details).
* `SpatialNeuron` now supports non-linear membrane currents (e.g.
  Goldman–Hodgkin–Katz equations) by linearizing them with respect to v.
* Multi-compartmental models can access the capacitive current via `Ic` in
  their equations (:issue:`677`)
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
  due to a bug in glibc (see :issue:`803`). Thanks to Oleg Strikov
  (`@xj8z <https://github.com/xj8z>`_) for debugging this
  issue and providing the workaround that is now in use.
* Make exact integration of ``event-driven`` synaptic variables use the
  ``linear`` numerical integration algorithm (instead of ``independent``),
  fixing rare occasions where integration failed despite the equations being
  linear (:issue:`801`).
* Better error messages for incorrect unit definitions in equations.
* Various fixes for the internal representation of physical units and the
  unit registration system.
* Fix a bug in the assignment of state variables in subtrees of `SpatialNeuron`
  (:issue:`822`)
* Numpy target: fix an indexing error for a `SpikeMonitor` that records from a
  subgroup (:issue:`824`)
* Summed variables targeting the same post-synaptic variable now raise an error
  (previously, only the one executed last was taken into account, see :issue:`766`).
* Fix bugs in synapse generation affecting Cython (:issue:`781`) respectively numpy
  (:issue:`835`)
* C++ standalone simulations with many objects no longer fail on Windows (:issue:`787`)

Backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `celsius` has been removed as a unit, because it was ambiguous in its relation
  to `kelvin` and gave wrong results when used as an absolute temperature (and
  not a temperature difference). For temperature differences, you can directly
  replace `celsius` by `kelvin`. To convert an absolute temperature in degree
  Celsius to Kelvin, add the `zero_celsius` constant from
  `brian2.units.constants` (:issue:`817`).
* State variables are no longer allowed to have names ending in ``_pre`` or
  ``_post`` to avoid confusion with references to pre- and post-synaptic
  variables in `Synapses` (:issue:`818`)

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
* Fix `PopulationRateMonitor` for recordings from subgroups (:issue:`772`)
* Fix `SpikeMonitor` for recordings from subgroups (:issue:`777`)
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
  ``randn()``. (:issue:`720`, :issue:`721`)

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Fix `EventMonitor.values` and `SpikeMonitor.spike_trains` to always return
  sorted spike/event times (:issue:`725`).
* Respect the ``active`` attribute in C++ standalone mode (:issue:`718`).
* More consistent check of compatible time and dt values (:issue:`730`).
* Attempting to set a synaptic variable or to start a simulation with synapses
  without any preceding connect call now raises an error (:issue:`737`).
* Improve the performance of coordinate calculation for `Morphology` objects,
  which previously made plotting very slow for complex morphologies (:issue:`741`).
* Fix a bug in `SpatialNeuron` where it did not detect non-linear dependencies
  on v, introduced via point currents (:issue:`743`).

Infrastructure and documentation improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* An interactive demo, tutorials, and examples can now be run in an interactive
  jupyter notebook on the `mybinder <http://mybinder.org/>`_ platform, without
  any need for a local Brian installation (:issue:`736`). Thanks to Ben Evans for the
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
  this to other formats. Thanks to Dominik Krzemiński for this contribution (see :issue:`306`).

Improvements and bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Use a Mersenne-Twister pseudorandom number generator in C++ standalone mode, replacing the
  previously used low-quality random number generator from the C standard library (see :issue:`222`,
  :issue:`671` and :issue:`706`).
* Fix a memory leak in code running with the weave code generation target, and a smaller
  memory leak related to units stored repetitively in the `~brian2.units.fundamentalunits.UnitRegistry`.
* Fix a difference of one timestep in the number of simulated timesteps between
  runtime and standalone that could arise for very specific values of dt and t (see :issue:`695`).
* Fix standalone compilation failures with the most recent gcc version which defaults to
  C++14 mode (see :issue:`701`)
* Fix incorrect summation in synapses when using the ``(summed)`` flag and writing to
  *pre*-synaptic variables (see :issue:`704`)
* Make synaptic pathways work when connecting groups that define nested subexpressions,
  instead of failing with a cryptic error message (see :issue:`707`).

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
  part (e.g. ``dx/dt = xi/sqrt(ms)```) were not integrated correctly (see :issue:`686`).
* Repeatedly calling `restore` (or `Network.restore`) no longer raises an error (see :issue:`681`).
* Fix an issue that made `PoissonInput` refuse to run after a change of dt (see :issue:`684`).
* If the ``rates`` argument of `PoissonGroup` is a string, it will now be evaluated at
  every time step instead of once at construction time. This makes time-dependent rate
  expressions work as expected (see :issue:`660`).

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
  conditions for some code generation targets (:issue:`429`)
* Fix an incorrect update of pre-synaptic variables in synaptic statements for
  the ``numpy`` code generation target (:issue:`435`).
* Fix the possibility of an incorrect memory access when recording a subgroup
  with `SpikeMonitor` (:issue:`454`).
* Fix the storing of results on disk for C++ standalone on Windows -- variables
  that had the same name when ignoring case (e.g. ``i`` and ``I``) where
  overwriting each other (:issue:`455`).

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
  (:issue:`361`)
* Linking variables from a group of size 1 now works correctly (:issue:`383`)

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
