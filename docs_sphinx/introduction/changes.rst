Changes for Brian 1 users
=========================
.. contents::
    :local:
    :depth: 1

In most cases, Brian 2 works in a very similar way to Brian 1 but there are
some important differences to be aware of. The major distinction is that
in Brian 2 you need to be more explicit about the definition of your
simulation in order to avoid inadvertent errors. In some cases, you will now
get a warning in other even an error -- often the error/warning message
describes a way to resolve the issue.

Specific examples how to convert code from Brian 1 can be found in the document
:doc:`brian1_to_2/index`.

Physical units
--------------
The unit system now extends to arrays, e.g. ``np.arange(5) * mV`` will retain
the units of volts and not discard them as Brian 1 did. Brian 2 is therefore
also more strict in checking the units. For example, if the state variable
``v`` uses the unit of volt, the statement ``G.v = np.rand(len(G)) / 1000.``
will now raise an error. For consistency, units are returned everywhere, e.g.
in monitors. If ``mon`` records a state variable v, ``mon.t`` will return a
time in seconds and ``mon.v`` the stored values of ``v`` in units of volts.

If you need a pure numpy array without units for further processing, there
are several options: if it is a state variable or a recorded variable in a
monitor, appending an underscore will refer to the variable values without
units, e.g. ``mon.t_`` returns pure floating point values. Alternatively, you
can remove units by diving by the unit (e.g. ``mon.t / second``) or by
explicitly converting it (``np.asarray(mon.t)``).

Here's an overview showing a few expressions and their respective values in
Brian 1 and Brian 2:

================================    ================================    =================================
Expression                          Brian 1                             Brian 2
================================    ================================    =================================
1 * mV                              1.0 * mvolt                         1.0 * mvolt
np.array(1) * mV                    0.001                               1.0 * mvolt
np.array([1]) * mV                  array([ 0.001])                     array([1.]) * mvolt
np.mean(np.arange(5) * mV)          0.002                               2.0 * mvolt
np.arange(2) * mV                   array([ 0.   ,  0.001])             array([ 0.,  1.]) * mvolt
(np.arange(2) * mV) >= 1 * mV       array([False, True], dtype=bool)    array([False, True], dtype=bool)
(np.arange(2) * mV)[0] >= 1 * mV    False                               False
(np.arange(2) * mV)[1] >= 1 * mV    DimensionMismatchError              True
================================    ================================    =================================

Unported packages
-----------------
The following packages have not (yet) been ported to Brian 1. If your simulation
critically depends on them, you should consider staying with Brian 1 for now.

* ``brian.tools``
* ``brian.hears``  (the Brian 1 version can be used via `brian2.hears`, though,
  see :ref:`brian_hears`)
* ``brian.library.modelfitting``
* ``brian.library.electrophysilogy``

Removed classes/functions and their replacements
------------------------------------------------
In Brian 2, we have tried to keep the number of classes/functions to a minimum, but make
each of them flexible enough to encompass a large number of use cases. A lot of the classes
and functions that existed in Brian 1 have therefore been removed.
The following table lists (most of) the classes that existed in Brian 1 but do no longer
exist in Brian 2. You can consult it when you get a ``NameError`` while converting an
existing script from Brian 1. The third column links to a document with further explanation
and the second column gives either:

1. the equivalent class in Brian 2 (e.g. `StateMonitor` can record multiple variables now
   and therefore replaces ``MultiStateMonitor``);
2. the name of a Brian 2 class in square brackets (e.g. [`Synapses`] for ``STDP``), this
   means that the class can be used as a replacement but needs some additional
   code (e.g. explicitly specified STDP equations). The "More details" document should
   help you in making the necessary changes;
3. "string expression", if the functionality of a previously existing class can
   be expressed using the general string expression framework (e.g.
   `threshold=VariableThreshold('Vt', 'V')` can be replaced by
   `threshold='V > Vt'`);
4. a link to the relevant github issue if no equivalent class/function does exist so far
   in Brian 2;
5. a remark such as "obsolete" if the particular class/function is no longer needed.

=============================== ================================= ================================================================
Brian 1                         Brian 2                           More details
=============================== ================================= ================================================================
``AdEx``	                    [`Equations`]	                  :doc:`brian1_to_2/library`
``aEIF``	                    [`Equations`]	                  :doc:`brian1_to_2/library`
``AERSpikeMonitor``	            #298	                          :doc:`brian1_to_2/monitors`
``alpha_conductance``	        [`Equations`]	                  :doc:`brian1_to_2/library`
``alpha_current``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``alpha_synapse``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``AutoCorrelogram``             [`SpikeMonitor`]                  :doc:`brian1_to_2/monitors`
``biexpr_conductance``	        [`Equations`]	                  :doc:`brian1_to_2/library`
``biexpr_current``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``biexpr_synapse``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``Brette_Gerstner``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``CoincidenceCounter``          [`SpikeMonitor`]                  :doc:`brian1_to_2/monitors`
``CoincidenceMatrixCounter``    [`SpikeMonitor`]                  :doc:`brian1_to_2/monitors`
``Compartments``	            #443	                          :doc:`brian1_to_2/multicompartmental`
``Connection``	                `Synapses`	                      :doc:`brian1_to_2/synapses`
``Current``	                    #443	                          :doc:`brian1_to_2/multicompartmental`
``CustomRefractoriness``	    [string expression]	              :doc:`brian1_to_2/neurongroup`
``DefaultClock``	            `Clock`	                          :doc:`brian1_to_2/networks_and_clocks`
``EmpiricalThreshold``	        string	expression	              :doc:`brian1_to_2/neurongroup`
``EventClock``	                `Clock`	                          :doc:`brian1_to_2/networks_and_clocks`
``exp_conductance``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``exp_current``	                [`Equations`]	                  :doc:`brian1_to_2/library`
``exp_IF``	                    [`Equations`]	                  :doc:`brian1_to_2/library`
``exp_synapse``	                [`Equations`]	                  :doc:`brian1_to_2/library`
``FileSpikeMonitor``	        #298	                          :doc:`brian1_to_2/monitors`
``FloatClock``	                `Clock`	                          :doc:`brian1_to_2/networks_and_clocks`
``FunReset``	                [string	expression]	              :doc:`brian1_to_2/neurongroup`
``FunThreshold``	            [string	expression]	              :doc:`brian1_to_2/neurongroup`
``hist_plot``                   no equivalent                     --
``HomogeneousPoissonThreshold``	string	expression	              :doc:`brian1_to_2/neurongroup`
``IdentityConnection``	        `Synapses`	                      :doc:`brian1_to_2/synapses`
``IonicCurrent``	            #443	                          :doc:`brian1_to_2/multicompartmental`
``ISIHistogramMonitor``         [`SpikeMonitor`]                  :doc:`brian1_to_2/monitors`
``Izhikevich``	                [`Equations`]	                  :doc:`brian1_to_2/library`
``K_current_HH``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``leak_current``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``leaky_IF``	                [`Equations`]	                  :doc:`brian1_to_2/library`
``MembraneEquation``	        #443	                          :doc:`brian1_to_2/multicompartmental`
``MultiStateMonitor``	        `StateMonitor`	                  :doc:`brian1_to_2/monitors`
``Na_current_HH``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``NaiveClock``	                `Clock`	                          :doc:`brian1_to_2/networks_and_clocks`
``NoReset``	                    obsolete	                      :doc:`brian1_to_2/neurongroup`
``NoThreshold``	                obsolete	                      :doc:`brian1_to_2/neurongroup`
``OfflinePoissonGroup``	        [`SpikeGeneratorGroup`]	          :doc:`brian1_to_2/inputs`
``OrnsteinUhlenbeck``	        [`Equations`]	                  :doc:`brian1_to_2/library`
``perfect_IF``	                [`Equations`]	                  :doc:`brian1_to_2/library`
``PoissonThreshold``	        string expression	              :doc:`brian1_to_2/neurongroup`
``PopulationSpikeCounter``	    `SpikeMonitor`	                  :doc:`brian1_to_2/monitors`
``PulsePacket``	                [`SpikeGeneratorGroup`]	          :doc:`brian1_to_2/inputs`
``quadratic_IF``	            [`Equations`]	                  :doc:`brian1_to_2/library`
``raster_plot``	                ``plot_raster`` (``brian2tools``) `brian2tools documentation <http://brian2tools.readthedocs.io>`_
``RecentStateMonitor``          no direct equivalent              :doc:`brian1_to_2/monitors`
``Refractoriness``	            string expression	              :doc:`brian1_to_2/neurongroup`
``RegularClock``	            `Clock`	                          :doc:`brian1_to_2/networks_and_clocks`
``Reset``	                    string expression	              :doc:`brian1_to_2/neurongroup`
``SimpleCustomRefractoriness``	[string	expression]	              :doc:`brian1_to_2/neurongroup`
``SimpleFunThreshold``	        [string	expression]	              :doc:`brian1_to_2/neurongroup`
``SpikeCounter``	            `SpikeMonitor`	                  :doc:`brian1_to_2/monitors`
``StateHistogramMonitor``       [`StateMonitor`]                  :doc:`brian1_to_2/monitors`
``StateSpikeMonitor``	        `SpikeMonitor`	                  :doc:`brian1_to_2/monitors`
``STDP``	                    [`Synapses`]	                  :doc:`brian1_to_2/synapses`
``STP``	                        [`Synapses`]	                  :doc:`brian1_to_2/synapses`
``StringReset``	                string expression	              :doc:`brian1_to_2/neurongroup`
``StringThreshold``	            string expression	              :doc:`brian1_to_2/neurongroup`
``Threshold``	                string expression	              :doc:`brian1_to_2/neurongroup`
``VanRossumMetric``             [`SpikeMonitor`]                  :doc:`brian1_to_2/monitors`
``VariableReset``	            string expression	              :doc:`brian1_to_2/neurongroup`
``VariableThreshold``	        string expression	              :doc:`brian1_to_2/neurongroup`
=============================== ================================= ================================================================

List of detailed instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    brian1_to_2/index