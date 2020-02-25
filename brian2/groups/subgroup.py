import numbers
from collections.abc import Sequence

import numpy as np

from brian2.core.base import weakproxy_with_fallback
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables
from brian2.utils.logger import get_logger

from .group import Group, Indexing

__all__ = ["Subgroup"]

logger = get_logger(__name__)


def to_start_stop_or_index(item, N):
    """
    Helper function to transform a single number, a slice or an array of
    indices to a start and stop value (if possible), or to an index of positive
    indices (interpreting negative indices correctly). This is used to allow for
    some flexibility in the syntax of specifying subgroups in `.NeuronGroup`
    and `.SpatialNeuron`.

    Parameters
    ----------
    item : slice, int or sequence
        The slice, index, or sequence of indices to use.
    N : int
        The total number of elements in the group.

    Returns
    -------
    start : int or None
        The start value of the slice.
    stop : int or None
        The stop value of the slice.
    indices : `np.ndarray` or None
        The indices.

    Examples
    --------
    >>> from brian2.groups.neurongroup import to_start_stop_or_index
    >>> to_start_stop_or_index(slice(3, 6), 10)
    (3, 6, None)
    >>> to_start_stop_or_index(slice(3, None), 10)
    (3, 10, None)
    >>> to_start_stop_or_index(5, 10)
    (5, 6, None)
    >>> to_start_stop_or_index(slice(None, None, 2), 10)
    (None, None, array([0, 2, 4, 6, 8]))
    >>> to_start_stop_or_index([3, 4, 5], 10)
    (3, 6, None)
    >>> to_start_stop_or_index([3, 5, 7], 10)
    (None, None, array([3, 5, 7]))
    >>> to_start_stop_or_index([-1, -2, -3], 10)
    (None, None, array([7, 8, 9]))
    """
    start = stop = indices = None
    if isinstance(item, slice):
        start, stop, step = item.indices(N)
        if step != 1:
            indices = np.arange(start, stop, step)
            start = stop = None
    elif isinstance(item, numbers.Integral):
        start = item
        stop = item + 1
        step = 1
    elif isinstance(item, (Sequence, np.ndarray)) and not isinstance(item, str):
        item = np.asarray(item)
        if not np.issubdtype(item.dtype, np.integer):
            raise TypeError("Subgroups can only be constructed using integer values.")
        if not len(item) > 0:
            raise IndexError("Cannot create an empty subgroup")
        if np.min(item) < -N:
            raise IndexError(f"Illegal index {np.min(item)} for a group of size {N}")
        elif np.max(item) >= N:
            raise IndexError(f"Illegal index {np.min(item)} for a group of size {N}")
        indices = item % N  # interpret negative indices correctly

        if not np.all(item[:-1] <= item[1:]):
            logger.warn(
                "The indices provided to create the subgroup were "
                "not sorted. They will be sorted before use.",
                name_suffix="unsorted_subgroup_indices",
            )
            indices.sort()

        if np.all(np.diff(item) == 1):
            start = int(item[0])
            stop = int(item[-1]) + 1
            indices = None

    else:
        raise TypeError(
            f"Cannot interpret object of type '{type(item)}' as index or slice"
        )

    if indices is None:
        if start >= stop:
            raise IndexError(
                "Illegal start/end values for subgroup, %d>=%d" % (start, stop)
            )
        if start >= N:
            raise IndexError("Illegal start value for subgroup, %d>=%d" % (start, N))
        if stop > N:
            raise IndexError("Illegal stop value for subgroup, %d>%d" % (stop, N))

    return start, stop, indices


class Subgroup(Group, SpikeSource):
    """
    Subgroup of any `Group`

    Parameters
    ----------
    source : SpikeSource
        The source object to subgroup.
    start, stop : int, optional
        Select only spikes with indices from ``start`` to ``stop-1``. Cannot
        be specified at the same time as ``indices``.
    indices : `np.ndarray`, optional
        The indices of the subgroup. Note that subgroups with non-contiguous
        indices cannot be used everywhere. Cannot be specified at the same time
        as ``start`` and ``stop``.
    name : str, optional
        A unique name for the group, or use ``source.name+'_subgroup_0'``, etc.
    """

    def __init__(self, source, start=None, stop=None, indices=None, name=None):
        if start is stop is indices is None:
            raise TypeError("Need to specify either start and stop or indices.")
        if start != stop and (start is None or stop is None):
            raise TypeError("start and stop have to be specified together.")
        if indices is not None and (start is not None):
            raise TypeError("Cannot specify both sub_indices and start and stop.")
        if start is not None:
            self.contiguous = True
        else:
            self.contiguous = False
            if not len(indices):
                raise IndexError("Cannot create an empty subgroup.")
            max_index = np.max(indices)
            if max_index >= len(source):
                raise IndexError(
                    "Index {} cannot be >= the size of the group ({})".format(
                        max_index, len(source)
                    )
                )
            if len(indices) != len(np.unique(indices)):
                raise IndexError("sub_indices cannot contain repeated values.")

        # First check if the source is itself a Subgroup
        # If so, then make this a Subgroup of the original Group
        if isinstance(source, Subgroup):
            if source.contiguous:
                if self.contiguous:
                    source = source.source
                    start = start + source.start
                    stop = stop + source.start
                else:
                    if np.max(indices) >= source.stop - source.start:
                        raise IndexError("Index exceeds range")
                    indices += source.start
            else:
                if self.contiguous:
                    indices = source.sub_indices[start:stop]
                    self.contiguous = False
                else:
                    indices = source.sub_indices[indices]
            self.source = source
        else:
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
        if self.contiguous:
            self._N = stop - start
        else:
            self._N = len(indices)
        self.start = start
        self.stop = stop
        self.sub_indices = indices

        self.events = self.source.events

        # All the variables have to go via the _sub_idx to refer to the
        # appropriate values in the source group
        self.variables = Variables(self, default_index="_sub_idx")

        # overwrite the meaning of N and i
        if self.contiguous and self.start > 0:
            self.variables.add_constant("_offset", value=self.start)
            self.variables.add_reference("_source_i", source, "i")
            self.variables.add_subexpression(
                "i",
                dtype=source.variables["i"].dtype,
                expr="_source_i - _offset",
                index="_idx",
            )
        elif self.contiguous:
            # no need to calculate anything if this is a subgroup starting at 0
            self.variables.add_reference("i", source)
        else:
            # We need an array to invert the indexing, i.e. an array where you
            # can use the sub_indices and get back 0, 1, 2, ...
            inv_idx = np.zeros(np.max(indices) + 1)
            inv_idx[indices] = np.arange(len(indices))
            self.variables.add_array(
                "i",
                size=len(inv_idx),
                dtype=source.variables["i"].dtype,
                values=inv_idx,
                constant=True,
                read_only=True,
                unique=True,
            )

        self.variables.add_constant("N", value=self._N)
        self.variables.add_constant("_source_N", value=len(source))
        # add references for all variables in the original group
        self.variables.add_references(source, list(source.variables.keys()))

        # Only the variable _sub_idx itself is stored in the subgroup
        # and needs the normal index for this group
        if self.contiguous:
            self.variables.add_arange(
                "_sub_idx", size=self._N, start=self.start, index="_idx"
            )
        else:
            self.variables.add_array(
                "_sub_idx",
                size=self._N,
                dtype=np.int32,
                values=indices,
                index="_idx",
                constant=True,
                read_only=True,
                unique=True,
            )

        # special indexing for subgroups
        self._indices = Indexing(self, self.variables["_sub_idx"])

        # Deal with special sub_indices
        for key, value in self.source.variables.indices.items():
            if value == "0":
                self.variables.indices[key] = "0"
            elif value == "_idx":
                continue  # nothing to do, already uses _sub_idx correctly
            else:
                raise ValueError(
                    "Do not know how to deal with variable %s "
                    "using index %s in a subgroup" % (key, value)
                )

        self.namespace = self.source.namespace
        self.codeobj_class = self.source.codeobj_class

        self._enable_group_attributes()

    spikes = property(lambda self: self.source.spikes)

    def __getitem__(self, item):
        start, stop, indices = to_start_stop_or_index(item, self._N)
        if self.contiguous:
            if start is not None:
                return Subgroup(self.source, self.start + start, self.start + stop)
            else:
                return Subgroup(self.source, indices=indices + self.start)
        else:
            indices = self.sub_indices[item]
            return Subgroup(self.source, indices=indices)

    def __repr__(self):
        if self.contiguous:
            description = "<{classname} {name} of {source} from {start} to {end}>"
            str_indices = None
        else:
            description = "<{classname} {name} of {source} with indices {indices}>"
            str_indices = np.array2string(
                self.sub_indices, threshold=10, separator=", "
            )
        return description.format(
            classname=self.__class__.__name__,
            name=repr(self.name),
            source=repr(self.source.name),
            start=self.start,
            indices=str_indices,
            end=self.stop,
        )
