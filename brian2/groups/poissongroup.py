import numpy as np

from brian2.core.namespace import create_namespace
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import ArrayVariable
from brian2.devices.device import get_device
from brian2.equations import Equations
from brian2.units.fundamentalunits import check_units
from brian2.units.stdunits import Hz

from .group import Group
from .neurongroup import Thresholder, StateUpdater

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

        self.variables = Group._create_variables(self)
        self.variables.update({'i': get_device().arange(self, 'i', N,
                                                        constant=True,
                                                        read_only=True),
                               'rates': get_device().array(self, 'rates',
                                                           size=N, unit=Hz),
                               '_spikespace': get_device().array(self,
                                                                 '_spikespace',
                                                                 size=N+1,
                                                                 unit=1,
                                                                 dtype=np.int32)})
        self.start = 0
        self.stop = N
        self.namespace = create_namespace(None)

        self.threshold = 'rand() < rates * dt'
        self._refractory = False
        self.thresholder = Thresholder(self)
        self.contained_objects.append(self.thresholder)

        self._enable_group_attributes()

        # Set the rates according to the argument (make sure to use the correct
        # namespace)
        rate_value = self.variables['rates'].get_addressable_value_with_unit(name,
                                                                             self,
                                                                             level=2)
        rate_value[:] = rates

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

