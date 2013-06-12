import weakref

from numpy import array, logical_and

from brian2.core.base import BrianObject
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler

__all__ = ['Subgroup']


class Subgroup(BrianObject, SpikeSource):
    '''
    Subgroup of any `SpikeSource`
    
    Parameters
    ----------
    source : SpikeSource
        The source object to subgroup.
    start, end : int
        Select only spikes with indices from ``start`` to ``end-1``.
    name : str, optional
        A unique name for the group, or use ``source.name+'_subgroup_0'``, etc.
    
    Notes
    -----
    
    Functions differently to Brian 1.x subgroup in that:
    
    * It works for any spike source
    * You need to keep a reference to it
    * It makes a copy of the spikes, and there is no direct support for
      subgroups in `Connection` (or rather `Synapses`)
    
    TODO: Group state variable access
    '''
    def __init__(self, source, start, end, name=None):
        self.source = weakref.proxy(source)
        # We want to update the spikes attribute after it has been updated
        # by the parent, we do this in slot 'thresholds' with an order
        # one higher than the parent order to ensure it takes place after the
        # parent threshold operation
        schedule = Scheduler(clock=source.clock, when='thresholds',
                             order=source.order+1)
        BrianObject.__init__(self, when=schedule, name=name)
        self.spikes = array([], dtype=int)
        self.N = end-start
        self.start = start
        self.end = end
        
    def __len__(self):
        return self.N
        
    def update(self):
        spikes = self.source.spikes
        # TODO: improve efficiency with bisect?
        spikes = spikes[logical_and(spikes>=self.start, spikes<self.end)]
        self.spikes = spikes-self.start
        
    def __repr__(self):
        description = '<{classname} {name} of {source} from {start} to {end}>'
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  source=repr(self.source.name),
                                  start=self.start,
                                  end=self.end)


if __name__=='__main__':
    from pylab import *
    from brian2 import *
    
    G = PoissonGroup(1000, rates=arange(1000)*Hz)
    subgp = Subgroup(G, 200, 400)
    
    M = SpikeMonitor(subgp)
    
    run(100*ms)
    
    subplot(121)
    plot(M.t, M.i, '.k')
    subplot(122)
    plot(M.count/defaultclock.t)
    show()
    
