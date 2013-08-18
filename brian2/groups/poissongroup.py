import numpy as np
from numpy.random import rand

from brian2.core.base import BrianObject
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.core.variables import ArrayVariable
from brian2.units.fundamentalunits import check_units
from brian2.units.stdunits import Hz

from .group import Group

__all__ = ['PoissonGroup']

class PoissonGroup(Group, BrianObject, SpikeSource):
    '''
    Poisson spike source
    
    Parameters
    ----------
    N : int
        Number of neurons
    rates : `Quantity`
        Single rate or array of rates of length N.
    when : Scheduler, optional
        When the `spikes` should be updated, will always be in the
        'thresholds' slot.
    name : str, optional
        Unique name, or use poissongroup, poissongroup_1, etc.
    
    Notes
    -----
    
    TODO: make rates not have to be a value/array, use code generation for str
    '''
    @check_units(rates=Hz)
    def __init__(self, N, rates, when=None, name='poissongroup*'):
        # TODO: sort out the default values in Scheduler
        scheduler = Scheduler(when)
        scheduler.when = 'thresholds'
        BrianObject.__init__(self, when=scheduler, name=name)

        #: The array of spikes from the most recent time step
        self.spikes = np.array([], dtype=int)
        
        self._rates = np.asarray(rates)
        self.N = N = int(N)
        
        self.pthresh = self._calc_threshold()

        self.variables = {'rates': ArrayVariable('rates', Hz, self._rates,
                                                 group_name=self.name,
                                                 constant=True)}
        Group.__init__(self)
        
    def __len__(self):
        return self.N
    
    def _calc_threshold(self):
        return np.array(self._rates*self.clock.dt_)
    
    def pre_run(self, namespace):
        self.pthresh = self._calc_threshold()
    
    def update(self):
        self.spikes, = (rand(self.N)<self.pthresh).nonzero()

    def __repr__(self):
        description = '{classname}({N}, rates={rates})'
        return description.format(classname=self.__class__.__name__,
                                        N=self.N,
                                        rates=repr(self._rates))

