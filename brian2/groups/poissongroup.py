import numpy as np

from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables
from brian2.units.fundamentalunits import check_units, Unit
from brian2.units.allunits import second
from brian2.units.stdunits import Hz

from .group import Group
from .neurongroup import Thresholder

__all__ = ['PoissonGroup']


class PoissonGroup(Group, SpikeSource):
    '''
    Poisson spike source
    
    Parameters
    ----------
    N : int
        Number of neurons
    rates : `Quantity`, str
        Single rate, array of rates of length N, or a string expression
        evaluating to a rate
    clock : Clock, optional
        The update clock to be used, or defaultclock if not specified.
    name : str, optional
        Unique name, or use poissongroup, poissongroup_1, etc.

    '''
    @check_units(rates=Hz)
    def __init__(self, N, rates, clock=None, name='poissongroup*',
                 codeobj_class=None):

        Group.__init__(self, when=clock, name=name)

        self.codeobj_class = codeobj_class

        self._N = N = int(N)

        # TODO: In principle, it would be nice to support Poisson groups with
        # refactoriness, but we can't currently, since the refractoriness
        # information is reset in the state updater which we are not using
        # We could either use a specific template or simply not bother and make
        # users write their own NeuronGroup (with threshold rand() < rates*dt)
        # for more complex use cases.

        self.variables = Variables(self)
        # standard variables
        self.variables.add_clock_variables(self.clock)
        self.variables.add_constant('N', unit=Unit(1), value=self._N)
        self.variables.add_arange('i', self._N, constant=True, read_only=True)
        self.variables.add_array('_spikespace', size=N+1, unit=Unit(1),
                                 dtype=np.int32)

        # The firing rates
        self.variables.add_array('rates', size=N, unit=Hz)

        self.start = 0
        self.stop = N

        self._refractory = False
        # this prevents using the local namespace and showing a warning about#
        # the local variable 'rates'
        self.namespace = {}
        self.threshold = 'rand() < rates * dt'
        self.thresholder = Thresholder(self)
        self.contained_objects.append(self.thresholder)

        self._enable_group_attributes()
        # Set the rates according to the argument (allow the use of the local
        # namespace again)
        self.namespace = None
        self.rates.set_item(slice(None), rates, level=2)

    @property
    def spikes(self):
        '''
        The spikes returned by the most recent thresholding operation.
        '''
        # Note that we have to directly access the ArrayVariable object here
        # instead of using the Group mechanism by accessing self._spikespace
        # Using the latter would cut _spikespace to the length of the group
        spikespace = self.variables['_spikespace'].get_value()
        return spikespace[:spikespace[-1]]

    def __len__(self):
        return self.N

    def __repr__(self):
        description = '{classname}({N}, rates=<...>)'
        return description.format(classname=self.__class__.__name__,
                                  N=self.N)

