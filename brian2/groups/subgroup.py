import weakref
from collections import defaultdict

from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variable
from brian2.groups.group import Group
from brian2.devices.device import get_device
from brian2.units.fundamentalunits import Unit
from brian2.core.variables import Subexpression

__all__ = ['Subgroup']


class Subgroup(Group, SpikeSource):
    '''
    Subgroup of any `Group`
    
    Parameters
    ----------
    source : SpikeSource
        The source object to subgroup.
    start, stop : int
        Select only spikes with indices from ``start`` to ``stop-1``.
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
    def __init__(self, source, start, stop, name=None):
        self.source = weakref.proxy(source)
        if name is None:
            name = source.name + '_subgroup*'
        # We want to update the spikes attribute after it has been updated
        # by the parent, we do this in slot 'thresholds' with an order
        # one higher than the parent order to ensure it takes place after the
        # parent threshold operation
        schedule = Scheduler(clock=source.clock, when='thresholds',
                             order=source.order+1)
        Group.__init__(self, when=schedule, name=name)
        self._N = stop-start
        self.start = start
        self.stop = stop

        self.variables = dict(self.source.variables)
        # overwrite the meaning of N and i
        self.variables['_offset'] = Variable(Unit(1), value=self.start,
                                             scalar=True, constant=True,
                                             read_only=True)
        self.variables['_source_i'] = self.source.variables['i']
        self.variables['i'] = Subexpression('i', Unit(1),
                                            dtype=source.variables['i'].dtype,
                                            expr='_source_i - _offset',
                                            group=self)
        self.variables['N'] = Variable(Unit(1), value=self._N, constant=True,
                                       read_only=True)
        # All variables refer to the original group and have to use a special
        # index
        self.variable_indices = defaultdict(lambda: '_sub_idx')
        # Only the variable _sub_idx itself is stored in the subgroup
        # and needs the normal index for this group
        self.variable_indices['_sub_idx'] = '_idx'

        for key, value in self.source.variable_indices.iteritems():
            if value != '_idx':
                raise ValueError(('Do not how to deal with variable %s using '
                                  'index %s in a subgroup') % (key, value))
        self.variables['_sub_idx'] = get_device().arange(self, '_sub_idx',
                                                         self._N,
                                                         start=self.start,
                                                         constant=True,
                                                         read_only=True)
        self.namespace = self.source.namespace
        self.codeobj_class = self.source.codeobj_class

        self._enable_group_attributes()

    spikes = property(lambda self: self.source.spikes)

    def __len__(self):
        return self._N
        
    def __repr__(self):
        description = '<{classname} {name} of {source} from {start} to {end}>'
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  source=repr(self.source.name),
                                  start=self.start,
                                  end=self.stop)
