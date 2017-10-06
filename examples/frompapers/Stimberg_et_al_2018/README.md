These Brian scripts reproduce the figures from the following preprint:

Modeling neuron-glia interactions with the Brian 2 simulator
Marcel Stimberg, Dan F. M. Goodman, Romain Brette, Maurizio De Pittà
bioRxiv 198366; doi: https://doi.org/10.1101/198366

Each file can be run individually to reproduce the respective figure. Note that
most files use the [standalone mode](http://brian2.readthedocs.io/en/stable/user/computation.html#standalone-code-generation)
for faster simulation. If your setup does not support this mode, you can instead
fallback to the runtime mode by removing the `set_device('cpp_standalone)` line.

Note that example 6 ("recurrent neuron-glial network") takes a relatively long
time (~15min on a reasonably fast desktop machine) to run.
