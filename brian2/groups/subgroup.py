from brian2.core.base import weakproxy_with_fallback
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables

from .group import Group, Indexing

__all__ = ["Subgroup"]


class Subgroup(Group, SpikeSource):
    """
    Subgroup of any `Group`

    Parameters
    ----------
    source : SpikeSource
        The source object to subgroup.
    start, stop : int
        Select only spikes with indices from ``start`` to ``stop-1``.
    name : str, optional
        A unique name for the group, or use ``source.name+'_subgroup_0'``, etc.
    """

    def __init__(self, source, start, stop, name=None):
        # A Subgroup should never be constructed from another Subgroup
        # Instead, use Subgroup(source.source,
        #                       start + source.start, stop + source.start)
        assert not isinstance(source, Subgroup)
        self.source = weakproxy_with_fallback(source)

        # Store a reference to the source's equations (if any)
        self.equations = None
        if hasattr(self.source, "equations"):
            self.equations = weakproxy_with_fallback(self.source.equations)

        if name is None:
            name = f"{source.name}_subgroup*"
        # We want to update the spikes attribute after it has been updated
        # by the parent, we do this in slot 'thresholds' with an order
        # one higher than the parent order to ensure it takes place after the
        # parent threshold operation
        Group.__init__(
            self,
            clock=source._clock,
            when="thresholds",
            order=source.order + 1,
            name=name,
        )
        self._N = stop - start
        self.start = start
        self.stop = stop

        self.events = self.source.events

        # All the variables have to go via the _sub_idx to refer to the
        # appropriate values in the source group
        self.variables = Variables(self, default_index="_sub_idx")

        # overwrite the meaning of N and i
        if self.start > 0:
            self.variables.add_constant("_offset", value=self.start)
            self.variables.add_reference("_source_i", source, "i")
            self.variables.add_subexpression(
                "i",
                dtype=source.variables["i"].dtype,
                expr="_source_i - _offset",
                index="_idx",
            )
        else:
            # no need to calculate anything if this is a subgroup starting at 0
            self.variables.add_reference("i", source)

        self.variables.add_constant("N", value=self._N)
        self.variables.add_constant("_source_N", value=len(source))
        # add references for all variables in the original group
        self.variables.add_references(source, list(source.variables.keys()))

        # Only the variable _sub_idx itself is stored in the subgroup
        # and needs the normal index for this group
        self.variables.add_arange(
            "_sub_idx", size=self._N, start=self.start, index="_idx"
        )

        # special indexing for subgroups
        self._indices = Indexing(self, self.variables["_sub_idx"])

        # Deal with special indices
        for key, value in self.source.variables.indices.items():
            if value == "0":
                self.variables.indices[key] = "0"
            elif value == "_idx":
                continue  # nothing to do, already uses _sub_idx correctly
            else:
                raise ValueError(
                    f"Do not know how to deal with variable '{key}' "
                    f"using index '{value}' in a subgroup."
                )

        self.namespace = self.source.namespace
        self.codeobj_class = self.source.codeobj_class

        self._enable_group_attributes()

    spikes = property(lambda self: self.source.spikes)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("Subgroups can only be constructed using slicing syntax")
        start, stop, step = item.indices(self._N)
        if step != 1:
            raise IndexError("Subgroups have to be contiguous")
        if start >= stop:
            raise IndexError(
                f"Illegal start/end values for subgroup, {int(start)}>={int(stop)}"
            )
        return Subgroup(self.source, self.start + start, self.start + stop)

    def __repr__(self):
        classname = self.__class__.__name__
        return (
            f"<{classname} {self.name!r} of {self.source.name!r} "
            f"from {self.start} to {self.stop}>"
        )
