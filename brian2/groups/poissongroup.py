import numpy as np
from numpy.random import rand

from brian2.core.base import BrianObject
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.units.fundamentalunits import check_units
from brian2.units.stdunits import Hz

__all__ = ['PoissonGroup']

class PoissonGroup(BrianObject, SpikeSource):
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
        Unique name, or use poisson_group_0, etc.
    
    Notes
    -----
    
    TODO: make rates not have to be a value/array, use code generation for str
    '''
    basename = 'poisson_group'
    @check_units(rates=Hz)
    def __init__(self, N, rates, when=None, name=None):
        # TODO: sort out the default values in Scheduler
        scheduler = Scheduler(when)
        scheduler.when = 'thresholds'
        BrianObject.__init__(self, when=scheduler, name=name)

        #: The array of spikes from the most recent time step
        self.spikes = np.array([], dtype=int)
        
        self.rates = rates
        self.N = N = int(N)
        
        self.pthresh = self._calc_threshold()
        
    def __len__(self):
        return self.N
    
    def _calc_threshold(self):
        return np.array(self.rates*self.clock.dt)
    
    def pre_run(self, namespace):
        self.pthresh = self._calc_threshold()
    
    def update(self):
        self.spikes, = (rand(self.N)<self.pthresh).nonzero()


if __name__=='__main__':
    from pylab import *
    from brian2 import *
    P = PoissonGroup(1000, rates=100*Hz)
    M = SpikeMonitor(P)
    run(100*ms)
    plot(M.t, M.i, '.k')
    print 'Estimated rate:', M.num_spikes/(defaultclock.t*len(P))
    show()
