Overview
========

In most cases, Brian 2 works in a very similar way to Brian 1 but there are
some important differences to be aware of. The major distinction is that
in Brian 2 you need to be more explicit about the definition of your
simulation in order to avoid inadvertent errors. In some cases, you will now
get a warning in other even an error -- often the error/warning message
describes a way to resolve the issue.

Unported packages
-----------------
* ``brian.tools``
* ``brian.hears``  (the Brian 1 version can be used via `brian2.hears`, though)
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
3. a link to the relevant github issue if no equivalent class/function does exist so far
   in Brian 2;
4. a remark such as "obsolete" if the particular class/function is no longer needed.

=============================== ================================= ===========================
Brian 1                         Brian 2                           More details
=============================== ================================= ===========================
``AdEx``	                    [`Equations`]	                  :doc:`specifics/library`
``aEIF``	                    [`Equations`]	                  :doc:`specifics/library`
``AERSpikeMonitor``	            #298	                          :doc:`specifics/monitors`
``alpha_conductance``	        [`Equations`]	                  :doc:`specifics/library`
``alpha_current``	            [`Equations`]	                  :doc:`specifics/library`
``alpha_synapse``	            [`Equations`]	                  :doc:`specifics/library`
``AutoCorrelogram``             [`SpikeMonitor`]                  :doc:`specifics/monitors`
``biexpr_conductance``	        [`Equations`]	                  :doc:`specifics/library`
``biexpr_current``	            [`Equations`]	                  :doc:`specifics/library`
``biexpr_synapse``	            [`Equations`]	                  :doc:`specifics/library`
``Brette_Gerstner``	            [`Equations`]	                  :doc:`specifics/library`
``CoincidenceCounter``          no direct equivalent              :doc:`specifics/monitors`
``CoincidenceMatrixCounter``    no direct equivalent              :doc:`specifics/monitors`
``Compartments``	            #443	                          :doc:`specifics/multicompartmental`
``Connection``	                `Synapses`	                      :doc:`specifics/synapses`
``Current``	                    #443	                          :doc:`specifics/multicompartmental`
``CustomRefractoriness``	    [string expression]	              :doc:`specifics/neurongroup`
``DefaultClock``	            `Clock`	                          :doc:`specifics/clocks`
``EmpiricalThreshold``	        string	expression	              :doc:`specifics/neurongroup`
``EventClock``	                `Clock`	                          :doc:`specifics/clocks`
``exp_conductance``	            [`Equations`]	                  :doc:`specifics/library`
``exp_current``	                [`Equations`]	                  :doc:`specifics/library`
``exp_IF``	                    [`Equations`]	                  :doc:`specifics/library`
``exp_synapse``	                [`Equations`]	                  :doc:`specifics/library`
``FileSpikeMonitor``	        #298	                          :doc:`specifics/monitors`
``FloatClock``	                `Clock`	                          :doc:`specifics/clocks`
``FunReset``	                [string	expression]	              :doc:`specifics/neurongroup`
``FunThreshold``	            [string	expression]	              :doc:`specifics/neurongroup`
``hist_plot``                   no equivalent
``HomogeneousPoissonThreshold``	string	expression	              :doc:`specifics/neurongroup`
``IdentityConnection``	        `Synapses`	                      :doc:`specifics/synapses`
``IonicCurrent``	            #443	                          :doc:`specifics/multicompartmental`
``ISIHistogramMonitor``         [`SpikeMonitor`]                  :doc:`specifics/monitors`
``Izhikevich``	                [`Equations`]	                  :doc:`specifics/library`
``K_current_HH``	            [`Equations`]	                  :doc:`specifics/library`
``leak_current``	            [`Equations`]	                  :doc:`specifics/library`
``leaky_IF``	                [`Equations`]	                  :doc:`specifics/library`
``MembraneEquation``	        #443	                          :doc:`specifics/multicompartmental`
``MultiStateMonitor``	        `StateMonitor`	                  :doc:`specifics/monitors`
``Na_current_HH``	            [`Equations`]	                  :doc:`specifics/library`
``NaiveClock``	                `Clock`	                          :doc:`specifics/clocks`
``NoReset``	                    obsolete	                      :doc:`specifics/neurongroup`
``NoThreshold``	                obsolete	                      :doc:`specifics/neurongroup`
``OfflinePoissonGroup``	        [`SpikeGeneratorGroup`]	          :doc:`specifics/inputs`
``OrnsteinUhlenbeck``	        [`Equations`]	                  :doc:`specifics/library`
``perfect_IF``	                [`Equations`]	                  :doc:`specifics/library`
``PoissonThreshold``	        string expression	              :doc:`specifics/neurongroup`
``PopulationSpikeCounter``	    `SpikeMonitor`	                  :doc:`specifics/monitors`
``PulsePacket``	                [`SpikeGeneratorGroup`]	          :doc:`specifics/inputs`
``quadratic_IF``	            [`Equations`]	                  :doc:`specifics/library`
``raster_plot``	                ``plot_raster`` (``brian2tools``) `brian2tools documentation <http://brian2tools.readthedocs.io>`_
``RecentStateMonitor``          no direct equivalent              :doc:`specifics/monitors`
``Refractoriness``	            string expression	              :doc:`specifics/neurongroup`
``RegularClock``	            `Clock`	                          :doc:`specifics/clocks`
``Reset``	                    string expression	              :doc:`specifics/neurongroup`
``SimpleCustomRefractoriness``	[string	expression]	              :doc:`specifics/neurongroup`
``SimpleFunThreshold``	        [string	expression]	              :doc:`specifics/neurongroup`
``SpikeCounter``	            `SpikeMonitor`	                  :doc:`specifics/monitors`
``StateHistogramMonitor``       [`StateMonitor`]                  :doc:`specifics/monitors`
``StateSpikeMonitor``	        `SpikeMonitor`	                  :doc:`specifics/monitors`
``STDP``	                    [`Synapses`]	                  :doc:`specifics/synapses`
``STP``	                        [`Synapses`]	                  :doc:`specifics/synapses`
``StringReset``	                string expression	              :doc:`specifics/neurongroup`
``StringThreshold``	            string expression	              :doc:`specifics/neurongroup`
``Threshold``	                string expression	              :doc:`specifics/neurongroup`
``VanRossumMetric``             [`SpikeMonitor`]                  :doc:`specifics/monitors`
``VariableReset``	            string expression	              :doc:`specifics/neurongroup`
``VariableThreshold``	        string expression	              :doc:`specifics/neurongroup`
=============================== ================================= ===========================