import numpy as np

from brian2.core.base import BrianObject
from brian2.core.namespace import create_namespace
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import ArrayVariable
from brian2.devices.device import get_device
from brian2.equations import Equations
from brian2.units.fundamentalunits import check_units, Unit
from brian2.units.allunits import second
from brian2.units.stdunits import Hz

from .group import Group
from .neurongroup import Thresholder, StateUpdater

__all__ = ['PoissonGroup']


class PoissonGroup(Group, BrianObject, SpikeSource):
    '''
    Poisson spike source
    
    Parameters
    ----------
    N : int
        Number of neurons
    rates : `Quantity`, str
        Single rate or array of rates of length N
        evaluating to a rate
    clock : Clock, optional
        The update clock to be used, or defaultclock if not specified.
    name : str, optional
        Unique name, or use poissongroup, poissongroup_1, etc.

    Notes
    -----

    TODO: make rates not have to be a value/array, use code generation for str
    '''
    @check_units(rates=Hz)
    def __init__(self, N, rates, clock=None, name='poissongroup*',
                 codeobj_class=None):

        BrianObject.__init__(self, when=clock, name=name)

        self.codeobj_class = codeobj_class

        self.N = N = int(N)
        #: The array holding the spikes
        self._spikespace = get_device().array(self, '_spikespace', N+1, 1, dtype=np.int32)

        #: The array holding the rates
        self._rates = np.asarray(rates)
        if self._rates.ndim == 0:
            self._rates = np.repeat(self._rates, N)

        # TODO: In principle, it would be nice to support Poisson groups with
        # refactoriness, but we can't currently, since the refractoriness
        # information is reset in the state updater which we are not using
        # We could either use a specific template or simply not bother and make
        # users write their own NeuronGroup (with threshold rand() < rates*dt)
        # for more complex use cases.

        #: The array storing the refractoriness information (not used, currently)
        self._not_refractory = get_device().array(self, '_not_refractory', N, 1,
                                                  dtype=np.bool)
        self._lastspike = get_device().array(self, '_lastspike', N, 1)

        self.variables = Group._create_variables(self)
        self.variables.update({'rates': ArrayVariable('rates', Hz, self._rates,
                                                      group_name=self.name,
                                                      constant=True),
                               '_spikespace': ArrayVariable('_spikespace', Unit(1),
                                                            self._spikespace,
                                                            group_name=self.name),
                               'not_refractory': ArrayVariable('not_refractory',
                                                               Unit(1),
                                                               self._not_refractory,
                                                               group_name=self.name,
                                                               is_bool=True),
                               'lastspike': ArrayVariable('lastspike',
                                                          second,
                                                          self._lastspike,
                                                          group_name=self.name)})

        self.namespace = create_namespace(None)

        self.threshold = 'rand() < rates * dt'
        self.thresholder = Thresholder(self)
        self.contained_objects.append(self.thresholder)

        # This is quite inefficient, we need a state updater to reset
        # not_refractory after every time step
        self.equations = Equations([])
        self._refractory = False
        self.state_updater = StateUpdater(self, method='independent')
        self.contained_objects.append(self.state_updater)
        Group.__init__(self)

    @property
    def spikes(self):
        '''
        The spikes returned by the most recent thresholding operation.
        '''
        return self._spikespace[:self._spikespace[-1]]

    def __len__(self):
        return self.N

    def __repr__(self):
        description = '{classname}({N}, rates={rates})'
        return description.format(classname=self.__class__.__name__,
                                  N=self.N,
                                  rates=repr(self._rates))

