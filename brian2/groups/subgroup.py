from brian2.core.base import weakproxy_with_fallback
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables
from brian2.groups.group import Group
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second

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
    '''
    def __init__(self, source, start, stop, name=None):
        # First check if the source is itself a Subgroup
        # If so, then make this a Subgroup of the original Group
        if isinstance(source, Subgroup):
            source = source.source
            start = start + source.start
            stop = stop + source.start
            self.source = source
        else:
            self.source = weakproxy_with_fallback(source)

        if name is None:
            name = source.name + '_subgroup*'
        # We want to update the spikes attribute after it has been updated
        # by the parent, we do this in slot 'thresholds' with an order
        # one higher than the parent order to ensure it takes place after the
        # parent threshold operation
        Group.__init__(self,
                       dt=None if source.dt is None else source.dt*second,
                       when='thresholds',
                       order=source.order+1, name=name)
        self._N = stop-start
        self.start = start
        self.stop = stop

        # All the variables have to go via the _sub_idx to refer to the
        # appropriate values in the source group
        self.variables = Variables(self, default_index='_sub_idx')

        # overwrite the meaning of N and i
        if self.start > 0:
            self.variables.add_constant('_offset', unit=Unit(1), value=self.start)
            self.variables.add_reference('_source_i', source, 'i')
            self.variables.add_subexpression('i', unit=Unit(1),
                                             dtype=source.variables['i'].dtype,
                                             expr='_source_i - _offset')
        else:
            # no need to calculate anything if this is a subgroup starting at 0
            self.variables.add_reference('i', source)

        self.variables.add_constant('N', unit=Unit(1), value=self._N)
        # add references for all variables in the original group
        self.variables.add_references(source, source.variables.keys())

        # Only the variable _sub_idx itself is stored in the subgroup
        # and needs the normal index for this group
        self.variables.add_arange('_sub_idx', size=self._N, start=self.start,
                                  index='_idx')

        for key, value in self.source.variables.indices.iteritems():
            if value not in ('_idx', '0'):
                raise ValueError(('Do not know how to deal with variable %s '
                                  'using  index %s in a subgroup') % (key,
                                                                      value))

        self.namespace = self.source.namespace
        self.codeobj_class = self.source.codeobj_class

        self._enable_group_attributes()

    spikes = property(lambda self: self.source.spikes)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError('Subgroups can only be constructed using slicing syntax')
        start, stop, step = item.indices(self._N)
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')
        if start >= stop:
            raise IndexError('Illegal start/end values for subgroup, %d>=%d' %
                             (start, stop))
        return Subgroup(self.source, self.start + start, self.start + stop)

    def __len__(self):
        return self._N

    def __repr__(self):
        description = '<{classname} {name} of {source} from {start} to {end}>'
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  source=repr(self.source.name),
                                  start=self.start,
                                  end=self.stop)
