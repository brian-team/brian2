from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.core.variables import Variables
from brian2.groups.group import Group
from brian2.units.fundamentalunits import Unit
from brian2.utils.proxy import Proxy

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
        self.source = Proxy(source)
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

        # All the variables have to go via the _sub_idx to refer to the
        # appropriate values in the source group
        self.variables = Variables(self, default_index='_sub_idx')

        # overwrite the meaning of N and i
        self.variables.add_constant('_offset', unit=Unit(1), value=self.start)
        self.variables.add_reference('_source_i', source.variables['i'])
        self.variables.add_subexpression('i', unit=Unit(1),
                                         dtype=source.variables['i'].dtype,
                                         expr='_source_i - _offset')
        self.variables.add_constant('N', unit=Unit(1), value=self._N)
        # add references for all variables in the original group
        self.variables.add_references(source.variables)

        # Only the variable _sub_idx itself is stored in the subgroup
        # and needs the normal index for this group
        self.variables.add_arange('_sub_idx', size=self._N, start=self.start,
                                  index='_idx')

        for key, value in self.source.variables.indices.iteritems():
            if value != '_idx':
                raise ValueError(('Do not how to deal with variable %s using '
                                  'index %s in a subgroup') % (key, value))

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
