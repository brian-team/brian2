'''
Neuronal morphology module.
This module defines classes to load and build neuronal morphologies.
'''
import abc
import numbers
from collections import OrderedDict

from brian2.units.allunits import meter
from brian2.utils.logger import get_logger
from brian2.units.stdunits import um
from brian2.units.fundamentalunits import (have_same_dimensions, Quantity,
                                           check_units)
from brian2 import numpy as np

logger = get_logger(__name__)

__all__ = ['Morphology', 'Section', 'Cylinder', 'Soma']

# TODO: Missing:
# * loading from SWC files
# * plotting (goes directly to brian2tools?)
# * calculation of coordinates for morphologies without coordinates
# * [later?] re-segmentation


class MorphologyIndexWrapper(object):
    '''
    A simpler version of `~brian2.groups.group.IndexWrapper`, not allowing for
    string indexing (`Morphology` is not a `Group`). It allows to use
    ``morphology.indices[...]`` instead of ``morphology[...]._indices()``.
    '''
    def __init__(self, morphology):
        self.morphology = morphology

    def __getitem__(self, item):
        if isinstance(item, basestring):
            raise NotImplementedError(('Morphologies do not support string '
                                       'indexing'))
        assert isinstance(self.morphology, (SubMorphology, Morphology))
        return self.morphology._indices(item)


def _calc_start_idx(section):
    '''
    Calculate the absolute start index that will be used by a flattened
    representation.
    '''
    # calculate the absolute start index of this section
    # 1. find the root of the tree
    root = section
    while root._parent is not None:
        root = root._parent
    # 2. go down from the root and advance the indices until we find
    # the current section
    start_idx, found = _find_start_index(root, section)
    assert found
    return start_idx


def _find_start_index(current, target_section, index=0):
    if current == target_section:
        return index, True
    index += current.n
    for child in current.children:
        if child == target_section:
            return index, True
        else:
            index, found = _find_start_index(child, target_section, index)
            if found:
                return index, True
    return index, False


class Children(object):
    def __init__(self, owner):
        self._owner = owner
        self._counter = 0
        self._children = []
        self._named_children = {}

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def __contains__(self, item):
        return item in self._named_children

    def values(self):
        return self._named_children.values()

    def __getitem__(self, item):
        if isinstance(item, basestring):
            return self._named_children[item]
        else:
            raise TypeError('Index has to be an integer or a string.')

    def add(self, name, subtree):
        if subtree._parent is not None:
            raise TypeError(('Cannot add subtree as "%s", it already has a '
                             'parent.') % name)
        if name in self._named_children:
            raise AttributeError('The subtree %s already exists' % name)
        self._counter += 1
        self._children.append(subtree)
        self._named_children[name] = subtree
        self._named_children[str(self._counter)] = subtree
        subtree._parent = self._owner

    def remove(self, item):
        if item not in self:
            raise AttributeError('The subtree ' + item + ' does not exist')
        subtree = self._named_children[item]
        del self._named_children[item]
        self._children.remove(subtree)
        subtree.parent = None

    # TODO: Useful __str__ and __repr__
    def __str__(self):
        return str(self._named_children)
    __repr__ = __str__


class Morphology(object):
    '''
    Neuronal morphology (tree structure).

    The data structure is a tree where each node is a segment consisting of a
    number of connected compartments, each one defined by its geometrical
    properties (length, area, diameter, position).

    Notes
    -----
    You cannot create objects of this class, create a `Soma`, a `Section`, or
    a `Cylinder` instead.
    '''
    __metaclass__ = abc.ABCMeta

    @check_units(n=1)
    def __init__(self, n, type=None):
        self._n = int(n)
        if self._n != n:
            raise TypeError('The number of compartments n has to be an integer '
                            'value.')
        if n <= 0:
            raise ValueError('The number of compartments n has to be at least 1.')
        self.type = type
        self.children = Children(self)
        self._parent = None
        self.indices = MorphologyIndexWrapper(self)

    def __getitem__(self, item):
        '''
        Returns the subtree named item.
        Ex.: ```neuron['axon']``` or ```neuron['11213']```
        ```neuron[10*um:20*um]``` returns the subbranch from 10 um to 20 um.
        ```neuron[10*um]``` returns one compartment.
        ```neuron[5]``` returns compartment number 5.
        '''
        if isinstance(item, slice):  # neuron[10*um:20*um] or neuron[1:3]
            using_lengths = all([arg is None or have_same_dimensions(arg, meter)
                                 for arg in [item.start, item.stop]])
            using_ints = all([arg is None or int(arg) == float(arg)
                              for arg in [item.start, item.stop]])
            if not (using_lengths or using_ints):
                raise TypeError('Index slice has to use lengths or integers')

            if using_lengths:
                if item.step is not None:
                    raise TypeError(('Cannot provide a step argument when '
                                     'slicing with lengths'))
                l = np.cumsum(np.asarray(self.length))  # coordinate on the branch
                # We use a special handling for values very close to the points
                # between the compartments to avoid non-intuitive rounding
                # effects: a point closer than 1e-12*length of section will be
                # considered to be within the following section (for a start
                # index), respectively within the previous section (for an end
                # index)
                if item.start is None:
                    i = 0
                else:
                    diff = np.abs(float(item.start) - l)
                    if min(diff) < 1e-12 * l[-1]:
                        i = np.argmin(diff) + 1
                    else:
                        i = np.searchsorted(l, item.start)
                if item.stop is None:
                    j = len(l)
                else:
                    diff = np.abs(float(item.stop) - l)
                    if min(diff) < 1e-12 * l[-1]:
                        j = np.argmin(diff) + 1
                    else:
                        j = np.searchsorted(l, item.stop) + 1
            else:  # integers
                i, j, step = item.indices(self.n)
                if step != 1:
                    raise TypeError('Can only slice a contiguous segment')
        elif isinstance(item, Quantity) and have_same_dimensions(item, meter):
            l = np.hstack([0, np.cumsum(np.asarray(self.length))])  # coordinate on the branch
            if float(item) < 0 or float(item) > (1 + 1e-12) * l[-1]:
                raise IndexError(('Invalid index %s, has to be in the interval '
                                  '[%s, %s].' % (item, 0*meter, l[-1]*meter)))
            diff = np.abs(float(item) - l)
            if min(diff) < 1e-12 * l[-1]:
                i = np.argmin(diff)
            else:
                i = np.searchsorted(l, item) - 1
            j = i + 1
        elif isinstance(item, numbers.Integral):  # int: returns one compartment
            if item < 0:  # allows e.g. to use -1 to get the last compartment
                item += self.n
            if item >= self.n:
                raise IndexError(('Invalid index %d '
                                  'for %d compartments') % (item, self.n))
            i = item
            j = i + 1
        elif isinstance(item, basestring):
            item = str(item)  # convert int to string
            if (len(item) > 1) and all([c in 'LR123456789' for c in
                                     item]):  # binary string of the form LLLRLR or 1213 (or mixed)
                return self.children[item[0]][item[1:]]
            elif item in self.children:
                return self.children[item]
            else:
                raise AttributeError('The subtree ' + item + ' does not exist')
        else:
            raise TypeError('Index of type %s not understood' % type(item))

        return SubMorphology(self, i, j)

    def __setitem__(self, item, child):
        '''
        Inserts the subtree and name it item.
        Ex.: ``neuron['axon']`` or ``neuron['11213']``
        If the tree already exists with another name, then it creates a synonym
        for this tree.
        The coordinates of the subtree are relative before function call,
        and are absolute after function call.
        '''
        item = str(item)  # convert int to string
        if (len(item) > 1) and all([c in 'LR123456789' for c in item]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            self.children[item[0]][item[1:]] = child
        else:
            self.children.add(item, child)

    def __delitem__(self, item):
        '''
        Removes the subtree `item`.
        '''
        item = str(item)  # convert int to string
        if (len(item) > 1) and all([c in 'LR123456789' for c in item]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            del self.children[item[0]][item[1:]]
        self.children.remove(item)

    def __getattr__(self, item):
        '''
        Returns the subtree named `item`.
        Ex.: ``axon=neuron.axon``
        '''
        if item.startswith('_'):
            return super(object, self).__getattr__(item)
        else:
            return self[item]

    def __setattr__(self, item, child):
        '''
        Attach a subtree and name it `item`.
        Ex.: ``neuron.axon = Soma(diameter=10*um)``
        Ex.: ``neuron.axon = None``
        '''
        if isinstance(child, Morphology) and not item.startswith('_'):
            self[item] = child
        else:  # If it is not a subtree, then it's a normal class attribute
            object.__setattr__(self, item, child)

    def _indices(self, item=None, index_var='_idx'):
        '''
        Returns compartment indices for the main branch, relative to the
        original morphology.
        '''
        if index_var != '_idx':
            raise AssertionError('Unexpected index %s' % index_var)
        if not (item is None or item == slice(None)):
            if isinstance(item, slice):
                # So that this always returns an array of values, even if it is
                # just a single value
                return self[item]._indices(slice(None))
            else:
                return self[item]._indices(None)
        else:
            start_idx = _calc_start_idx(self)
            if self.n == 1 and item is None:
                return start_idx
            else:
                return np.arange(start_idx, start_idx + self.n)

    @property
    def n(self):
        return self._n

    def __len__(self):
        return self.n + sum(len(c) for c in self.children)

    @property
    def n_sections(self):
        return 1 + sum(c.n_sections for c in self.children)

    @property
    def parent(self):
        return self._parent

    @abc.abstractproperty
    def total_distance(self):
        pass

    # Per-compartment attributes
    @abc.abstractproperty
    def area(self):
        pass

    @abc.abstractproperty
    def start_diameter(self):
        pass

    @abc.abstractproperty
    def diameter(self):
        pass

    @abc.abstractproperty
    def end_diameter(self):
        pass

    @abc.abstractproperty
    def volume(self):
        pass

    @abc.abstractproperty
    def length(self):
        pass

    @abc.abstractproperty
    def r_length(self):
        pass

    @abc.abstractproperty
    def electrical_center(self):
        pass

    # At-electrical-midpoint attributes
    @abc.abstractproperty
    def distance(self):
        pass

    @abc.abstractproperty
    def x(self):
        pass

    @abc.abstractproperty
    def y(self):
        pass

    @abc.abstractproperty
    def z(self):
        pass

    @property
    def coordinates(self):
        return np.vstack([self.x, self.y, self.z]).T

    @abc.abstractproperty
    def end_x(self):
        pass

    @abc.abstractproperty
    def end_y(self):
        pass

    @abc.abstractproperty
    def end_z(self):
        pass

    @abc.abstractproperty
    def start_x(self):
        pass

    @abc.abstractproperty
    def start_y(self):
        pass

    @abc.abstractproperty
    def start_z(self):
        pass

    @staticmethod
    def from_points(points, spherical_soma=True):
        '''
        Format:

        `index name x y z diameter parent`

        Note that the values should not use units, but are instead all taken
        to be in micrometers.

        Parameters
        ----------
        points : sequence of 7-tuples
            The points of the morphology.

        Returns
        -------
        morphology : `Morphology`

        Notes
        -----
        This format closely follows the SWC format (see `Morphology.from_file`)
        with two differences: the ``type`` should be a string (e.g. ``'soma'``)
        instead of an integer and the 6-th element should be the diameter and
        not the radius.
        '''
        # First pass through all points to get the dependency structure
        compartments = OrderedDict()
        for counter, point in enumerate(points):
            index, comp_name, x, y, z, diameter, parent = point
            if index in compartments:
                raise ValueError('Two compartments with index %d' % index)
            if parent == index:
                raise ValueError('Compartment %d lists itself as the parent '
                                 'compartment.' % index)
            compartments[index] = (comp_name, x, y, z, diameter, parent, [])  # empty list for the children
            if counter == 0 and parent == -1:
                continue  # The first compartment does not have a parent
            if parent not in compartments:
                raise ValueError(('Did not find the compartment %d (parent '
                                  'compartment of compartment %d). Make sure '
                                  'that parent compartments are listed before '
                                  'their children.') % (parent, index))
            compartments[parent][-1].append(index)

        # Merge all unbranched segments of the same type into a single section
        sections = dict()
        previous_name = None
        current_compartments = []
        previous_index = None
        for index, compartment in compartments.iteritems():
            comp_name, x, y, z, diameter, parent, children = compartment
            if len(current_compartments) and comp_name != previous_name or len(children) != 1:
                if spherical_soma and previous_name == 'soma':
                    if len(current_compartments) > 1:
                        raise NotImplementedError('Only spherical somas '
                                                  'described by a single point '
                                                  'and diameter are supported.')
                    soma_x, soma_y, soma_z, soma_diameter, soma_parent = current_compartments[0]
                    section = Soma(diameter=soma_diameter*um, x=soma_x*um, y=soma_y*um, z=soma_z*um)
                    sections[previous_index] = section, soma_parent, 'soma'
                    # We did not yet deal with the current compartment
                    current_compartments = [(x, y, z, diameter, parent)]
                else:
                    current_compartments.append((x, y, z, diameter, parent))
                    sec_x, sec_y, sec_z, sec_diameter, parents = zip(*current_compartments)
                    # Add a point for the end of the parent compartment
                    if parents[0] != -1:
                        n = len(current_compartments)
                        _, parent_x, parent_y, parent_z, parent_diameter, _, _ = compartments[parents[0]]
                        sec_x = [parent_x] + list(sec_x)
                        sec_y = [parent_y] + list(sec_y)
                        sec_z = [parent_z] + list(sec_z)
                        if isinstance(sections[parents[0]][0], Soma):
                            # For a Soma, we don't use its diameter
                            parent_diameter = sec_diameter[0]
                        sec_diameter = [parent_diameter] + list(sec_diameter)
                    else:
                        n = len(current_compartments) - 1
                    section = Section(n, diameter=sec_diameter*um,
                                      x=sec_x*um, y=sec_y*um, z=sec_z*um,
                                      type=previous_name)
                    sections[index] = section, current_compartments[0][4], previous_name  # parent of the first compartment in the section
                    current_compartments = []
            else:
                current_compartments.append((x, y, z, diameter, parent))

            previous_name = comp_name
            previous_index = index

        # There should be no compartments left
        assert len(current_compartments) == 0

        # Connect the sections
        for index, (section, parent, name) in sections.iteritems():
            # Add section to its parent
            if parent != -1:
                children_list = sections[parent][0].children
                n_children = len(children_list)
                if name is None:
                    children_list.add(name='child%d' % (n_children+1),
                                      subtree=section)
                else:
                    counter = 2
                    basename = name
                    while name in children_list:
                        name = basename + str(counter)
                        counter += 1
                    children_list.add(name=name,
                                      subtree=section)

        # There should only be one section without parents
        root = [sec for sec, _, _ in sections.itervalues() if sec.parent is None]
        assert len(root) == 1
        return root[0]


class SubMorphology(object):
    '''
    A view on a subset of a section in a morphology.
    '''
    def __init__(self, morphology, i, j):
        self._morphology = morphology
        self.indices = MorphologyIndexWrapper(self)
        self._i = i
        self._j = j

    def _indices(self, item=None):
        if not (item is None or item == slice(None)):
            raise IndexError('Cannot index a view on a subset of a section further')
        # Start index of the main section
        start_idx = _calc_start_idx(self._morphology)
        if item is None and self.n == 1:
            return start_idx + self._i
        else:
            return np.arange(start_idx + self._i, start_idx + self._j)

    @property
    def n(self):
        return self._j - self._i

    def __len__(self):
        return self.n

    @property
    def n_sections(self):
        return 1

    # Per-compartment attributes
    @property
    def area(self):
        return self._morphology.area[self._i:self._j]

    @property
    def diameter(self):
        return self._morphology.diameter[self._i:self._j]

    @property
    def volume(self):
        return self._morphology.volume[self._i:self._j]

    @property
    def length(self):
        return self._morphology.length[self._i:self._j]

    @property
    def r_length(self):
        return self._morphology.r_length[self._i:self._j]

    @property
    def electrical_center(self):
        return self._morphology.electrical_center[self._i:self._j]

    # At-electrical-midpoint attributes
    @property
    def distance(self):
        return self._morphology.distance[self._i:self._j]

    @property
    def x(self):
        if self._morphology.x is None:
            return None
        return self._morphology.x[self._i:self._j]

    @property
    def y(self):
        if self._morphology.y is None:
            return None
        return self._morphology.y[self._i:self._j]

    @property
    def z(self):
        if self._morphology.z is None:
            return None
        return self._morphology.z[self._i:self._j]

    @property
    def end_x(self):
        if self._morphology.end_x is None:
            return None
        return self._morphology.end_x[self._i:self._j]

    @property
    def end_y(self):
        if self._morphology.end_y is None:
            return None
        return self._morphology.end_y[self._i:self._j]

    @property
    def end_z(self):
        if self._morphology.end_z is None:
            return None
        return self._morphology.end_z[self._i:self._j]

    @property
    def start_x(self):
        if self._morphology.start_x is None:
            return None
        return self._morphology.start_x[self._i:self._j]

    @property
    def start_y(self):
        if self._morphology.start_y is None:
            return None
        return self._morphology.start_y[self._i:self._j]

    @property
    def start_z(self):
        if self._morphology.start_z is None:
            return None
        return self._morphology.start_z[self._i:self._j]


class Soma(Morphology):
    '''
    A spherical, iso-potential soma.

    Parameters
    ----------
    diameter : `Quantity`, optional
        Diameter of the sphere.
    '''

    @check_units(diameter=meter, x=meter, y=meter, z=meter)
    def __init__(self, diameter, x=None, y=None, z=None, type='soma'):
        Morphology.__init__(self, n=1, type=type)
        if diameter.shape != ():
            raise TypeError('Diameter has to be a scalar value.')
        for coord in [x, y, z]:
            if coord is not None and coord.shape != ():
                raise TypeError('Coordinates have to be scalar values.')
        self._diameter = np.ones(1) * diameter
        if any(coord is not None for coord in (x, y, z)):
            default_value = 0*um
        else:
            default_value = None
        self._x = np.atleast_1d(x) if x is not None else default_value
        self._y = np.atleast_1d(y) if y is not None else default_value
        self._z = np.atleast_1d(z) if z is not None else default_value

    # Note that the per-compartment properties should always return 1D arrays,
    # i.e. for the soma arrays of length 1 instead of scalar values
    @property
    def area(self):
        return np.pi * self.diameter ** 2

    @property
    def start_diameter(self):
        return [0]*um  # TODO: best value?

    @property
    def diameter(self):
        return self._diameter

    @property
    def end_diameter(self):
        return [0]*um  # TODO: best value?

    @property
    def volume(self):
        return (np.pi * self.diameter ** 3)/6

    @property
    def length(self):
        return self.diameter

    @property
    def r_length(self):
        # The soma does not have any resistance
        return [0]*um

    @property
    def electrical_center(self):
        return np.array([0.0])

    @property
    def distance(self):
        dist = self._parent.distance if self._parent else 0*um
        return dist

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def start_x(self):
        return self._x

    @property
    def start_y(self):
        return self._y

    @property
    def start_z(self):
        return self._z

    @property
    def end_x(self):
        return self._x

    @property
    def end_y(self):
        return self._y

    @property
    def end_z(self):
        return self._z

    @property
    def total_distance(self):
        dist = self._parent.total_distance if self._parent is not None else 0*um
        return dist  # TODO: + self.diameter/2 ?

class Section(Morphology):
    '''
    A section (unbranched structure), described as a sequence of truncated
    cones.

    Parameters
    ----------
    n : int
        The number of compartments in this section.
    diameter : `Quantity`
        Either a single value (the constant diameter along the whole section),
        or a value of length ``n`` or ``n+1``. When ``n`` values are given, they
        will be interpreted as the diameter at the ends of each compartment. In
        this case, the diameter at the start of the first compartment will be
        taken as the diameter at the end of the parent compartment (i.e., the
        connection will be continuous). When ``n+1`` values are given, then the
        first value specifies the diameter at the start of the compartment and
        the following values the diameter at the ends of each compartment.
    length : `Quantity`
        Either a single value (the total length of the section), or a value of
        length ``n``, the length of each individual compartment. Cannot be
        combined with the specification of coordinates.
    x : `Quantity`
        ``n`` values, specifying the x coordinates of the end-points of the
        compartments, or ``n``+1 values, specifying the x coordinate at the
        start-point of the first compartment and the x coordinates at the
        end-points of all compartments. If only ``n`` points are specified, the
        start point of the first compartment is considered to be the end-point
        of the previous compartment. In that case, all coordinates are
        considered to be relative to this point.
        You can specify all of ``x``, ``y``, or ``z`` to specify
        a morphology in 3D, or only one or two out of them to specify a
        morphology in 1D or 2D (the non-specified components will be considered
        as 0)
    y : `Quantity`
        See ``x``
    z : `Quantity`
        See ``x``
    parent : `Morphology`, optional
        The parent of this section.
    '''
    @check_units(n=1, length=meter, diameter=meter, x=meter, y=meter, z=meter)
    def __init__(self, n, diameter, length=None, x=None, y=None, z=None,
                 type=None):
        n = int(n)
        Morphology.__init__(self, n=n, type=type)

        if diameter.ndim > 1:
            raise TypeError('The diameter argument has to be a single value '
                            'or a one-dimensional array.')
        if diameter.shape == ():
            self._start_diameter = diameter  # TODO: or None?
            self._diameter = np.ones(n) * diameter
        elif len(diameter) == n:
            self._start_diameter = None
            self._diameter = diameter
        elif len(diameter) == n+1:
            self._start_diameter = diameter[0]
            self._diameter = diameter[1:]
        else:
            raise TypeError(('Need to specify a single value or %d or %d values '
                 'for the diameter, got %d values '
                 'instead') % (n, n+1, len(diameter)))

        if length is not None:
            # Specification by length
            if x is not None or y is not None or z is not None:
                raise TypeError('Cannot use both lengths and coordinates to '
                                'specify a section.')
            length = np.atleast_1d(length)
            if length.ndim > 1:
                raise TypeError('The length argument has to be a single value '
                                'or a one-dimensional array.')
            if len(length) != 1 and len(length) != n:
                raise TypeError(('Need to specify a single value or %d values '
                                 'for the length, got %d values '
                                 'instead.') % (n, len(length)))
            if len(length) == 1:
                # This is the *total* length of the whole section
                length = np.ones(n) * length/n
        else:
            if x is None and y is None and z is None:
                raise TypeError('No length specified, need to specify at least '
                                'one out of x, y, or z.')
            value_shape = None
            include_startpoints = False
            for name, value in [('x', x), ('y', y), ('z', z)]:
                if value is not None:
                    if value_shape is not None and value.shape != value_shape:
                        raise TypeError('All coordinate arrays have to have '
                                        'the same length.')
                    if value.shape not in [(), (n,), (n+1, )]:
                        raise TypeError(('Coordinates need to be a single '
                                         'value or one-dimensional arrays of '
                                         'length %d or %d, but the array '
                                         'provided for %s has shape '
                                         '%s') % (n, n+1, name, value.shape))
                    elif value.shape == (n+1, ):
                        include_startpoints = True
                        if self._start_diameter is None:
                            raise TypeError('If start coordinates have been given,'
                                            'then the diameter at the start has to '
                                            'be given as well.')
            if (include_startpoints and any([coord is not None and coord.shape != (n+1, )
                         for coord in x, y, z])):
                raise TypeError('If one coordinate is specified with absolute '
                                'coordinates, then all coordinates have to be '
                                'specified in this way.')
            default_value = np.zeros(n+1)*meter if include_startpoints else np.zeros(n)*meter
            x = x if x is not None else default_value
            y = y if y is not None else default_value
            z = z if z is not None else default_value
            x = x if x.shape != () else np.linspace(float(x)/n, float(x), n)*meter
            y = y if y.shape != () else np.linspace(float(y)/n, float(y), n)*meter
            z = z if z.shape != () else np.linspace(float(z)/n, float(z), n)*meter
            if len(x) == n:
                # Relative to start of the section
                start_x = np.hstack([0, np.asarray(x)[:-1]])*meter
                start_y = np.hstack([0, np.asarray(y)[:-1]])*meter
                start_z = np.hstack([0, np.asarray(z)[:-1]])*meter
                end_x = x
                end_y = y
                end_z = z
            else:
                # Absolute coordinates
                start_x = x[:-1]
                start_y = y[:-1]
                start_z = z[:-1]
                end_x = x[1:]
                end_y = y[1:]
                end_z = z[1:]
            length = np.sqrt((end_x - start_x)**2 +
                             (end_y - start_y)**2 +
                             (end_z - start_z)**2)

            x = x if x is not None else np.zeros(n)*meter
            y = y if y is not None else np.zeros(n)*meter
            z = z if z is not None else np.zeros(n)*meter

        self._x = x
        self._y = y
        self._z = z

        self._length = length

    @property
    def area(self):
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return np.pi/2*(d_1 + d_2)*np.sqrt(((d_1 - d_2)**2)/4 + self._length**2)

    @property
    def start_diameter(self):
        if self._start_diameter is None:
             start_diameter = (self.parent.end_diameter[-1]
                               if self.parent is not None else np.nan*um)
        else:
            start_diameter = self._start_diameter
        return Quantity(np.hstack([np.asarray(start_diameter),
                                   np.asarray(self._diameter[:-1])]),
                        dim=meter.dim)

    @property
    def diameter(self):
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        # TODO: Rather the diameter ath the electrical center?
        return 0.5*(d_1 + d_2)

    @property
    def end_diameter(self):
        return self._diameter

    @property
    def volume(self):
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return np.pi * self._length * (d_1**2 + d_1*d_2 + d_2**2)/12

    @property
    def length(self):
        return self._length

    @property
    def distance(self):
        dist = self._parent.total_distance if self._parent is not None else 0*um
        return dist + np.cumsum(self.length) - (1 - self.electrical_center) * self.length

    @property
    def total_distance(self):
        return self.distance[-1] + (1 - self.electrical_center[-1]) * self.length[-1]

    @property
    def electrical_center(self):
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return d_1 / (d_1 + d_2)

    @property
    def r_length(self):
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return np.pi/4 * (d_1 * d_2)/self._length

    @property
    def x(self):
        if self._x is None:
            return None
        diff_x = (self.end_x - self.start_x)
        return self.start_x + self.electrical_center*diff_x

    @property
    def y(self):
        if self._y is None:
            return None
        diff_y = (self.end_y - self.start_y)
        return self.start_y + self.electrical_center*diff_y

    @property
    def z(self):
        if self._z is None:
            return None
        diff_z = (self.end_z - self.start_z)
        return self.start_z + self.electrical_center*diff_z

    @property
    def start_x(self):
        if self._x is None:
            return None
        if len(self._x) == self.n:
            if self._parent is not None and self._parent.end_x is not None:
                parent_x = self._parent.end_x[-1]
            else:
                parent_x = 0*meter
            # Note that numpy's hstack function does not conserve units
            return np.hstack([0, np.asarray(self._x)[:-1]])*meter + parent_x
        else:
            # Do not return the last point (end point of the last compartment)
            return self._x[:self.n]

    @property
    def start_y(self):
        if self._y is None:
            return None
        if len(self._y) == self.n:
            if self._parent is not None and self._parent.end_y is not None:
                parent_y = self._parent.end_y[-1]
            else:
                parent_y = 0*meter
            # Note that numpy's hstack function does not conserve units
            return np.hstack([0, np.asarray(self._y)[:-1]])*meter + parent_y
        else:
            # Do not return the last point (end point of the last compartment)
            return self._y[:self.n]

    @property
    def start_z(self):
        if self._z is None:
            return None
        if len(self._z) == self.n:
            if self._parent is not None and self._parent.end_z is not None:
                parent_z = self._parent.end_z[-1]
            else:
                parent_z = 0*meter
            # Note that numpy's hstack function does not conserve units
            return np.hstack([0, np.asarray(self._z)[:-1]])*meter + parent_z
        else:
            # Do not return the last point (end point of the last compartment)
            return self._z[:self.n]

    @property
    def end_x(self):
        if self._x is None:
            return None
        elif len(self._x) == self.n:
            if self._parent is not None and self._parent.end_x is not None:
                parent_x = self._parent.end_x[-1]
            else:
                parent_x = 0*meter
            return self._x + parent_x
        else:
            # Do not return the first point (start point)
            return self._x[1:]

    @property
    def end_y(self):
        if self._y is None:
            return None
        elif len(self._y) == self.n:
            if self._parent is not None and self._parent.end_y is not None:
                parent_y = self._parent.end_y[-1]
            else:
                parent_y = 0*meter
            return self._y + parent_y
        else:
            # Do not return the first point (start point)
            return self._y[1:]

    @property
    def end_z(self):
        if self._z is None:
            return None
        elif len(self._z) == self.n:
            if self._parent is not None and self._parent.end_z is not None:
                parent_z = self._parent.end_z[-1]
            else:
                parent_z = 0*meter
            return self._z + parent_z
        else:
            # Do not return the first point (start point)
            return self._z[1:]


class Cylinder(Section):
    '''
    A section (unbranched structure), described as a sequence of cylinders

    Parameters
    ----------
    n : int
        The number of compartments in this section.
    diameter : `Quantity`
        Either a single value (the constant diameter along the whole section),
        or a value of length ``n``, giving the diameter for each compartment.
    length : `Quantity`, optional
        Either a single value (the total length of the section), or a value of
        length ``n``, the length of each individual compartment. Cannot be
        combined with the specification of coordinates.
    x : `Quantity`, optional
        ``n`` values, specifying the x coordinates of the end-points of the
        compartments, or ``n``+1 values, specifying the x coordinate at the
        start-point of the first compartment and the x coordinates at the
        end-points of all compartments. If only ``n`` points are specified, the
        start point of the first compartment is considered to be the end-point
        of the previous compartment. In that case, all coordinates are
        considered to be relative to this point.
        You can specify all of ``x``, ``y``, or ``z`` to specify
        a morphology in 3D, or only one or two out of them to specify a
        morphology in 1D or 2D (the non-specified components will be considered
        as 0). Cannot be combined with the specification of ``length``.
    y : `Quantity`, optional
        See ``x``
    z : `Quantity`, optional
        See ``x``
    parent : `Morphology`, optional
        The parent of this section.
    '''
    @check_units(n=1, length=meter, diameter=meter, x=meter, y=meter, z=meter)
    def __init__(self, n, diameter, length=None, x=None, y=None, z=None,
                 type=None):
        n = int(n)
        Morphology.__init__(self, n=n, type=type)

        diameter = np.atleast_1d(diameter)
        if diameter.ndim > 1:
            raise TypeError('The diameter argument has to be a single value '
                            'or a one-dimensional array.')
        if len(diameter) != 1 and len(diameter) != n:
            raise TypeError(('Need to specify a single value or %d values '
                             'for the diameter, got %d values '
                             'instead') % (n, len(diameter)))
        diameter = np.ones(n) * diameter
        self._diameter = diameter

        if length is not None:
            # Specification by length
            if x is not None or y is not None or z is not None:
                raise TypeError('Cannot use both lengths and coordinates to '
                                'specify a section.')
            length = np.atleast_1d(length)
            if length.ndim > 1:
                raise TypeError('The length argument has to be a single value '
                                'or a one-dimensional array.')
            if len(length) != 1 and len(length) != n:
                raise TypeError(('Need to specify a single value or %d values '
                                 'for the length, got %d values '
                                 'instead.') % (n, len(length)))
            if len(length) == 1:
                # This is the *total* length of the whole section
                length = np.ones(n) * (length/n)
        else:
            if x is None and y is None and z is None:
                raise TypeError('No length specified, need to specify at least '
                                'one out of x, y, or z.')
            include_startpoints = False
            for name, value in [('x', x), ('y', y), ('z', z)]:
                if value is not None and value.shape not in [(), (n, ), (n+1, )]:
                    raise TypeError(('Coordinates need to be single values or '
                                     'one-dimensional arrays of length %d or '
                                     '%d but the array provided for %s has '
                                     'shape %s') % (n, n+1, name, value.shape))
                elif value is not None and value.shape == (n+1,):
                    include_startpoints = True
            if (include_startpoints and any([coord is not None and coord.shape != (n+1, )
                                             for coord in x, y, z])):
                raise TypeError('If one coordinate is specified with absolute '
                                'coordinates, then all coordinates have to be '
                                'specified in this way.')
            default_value = np.zeros(n+1)*meter if include_startpoints else np.zeros(n)*meter
            x = x if x is not None else default_value
            y = y if y is not None else default_value
            z = z if z is not None else default_value
            x = x if x.shape != () else np.linspace(float(x)/n, float(x), n)*meter
            y = y if y.shape != () else np.linspace(float(y)/n, float(y), n)*meter
            z = z if z.shape != () else np.linspace(float(z)/n, float(z), n)*meter
            if len(x) == n:
                # Relative to start of the section
                start_x = np.hstack([0, np.asarray(x)[:-1]])*meter
                start_y = np.hstack([0, np.asarray(y)[:-1]])*meter
                start_z = np.hstack([0, np.asarray(z)[:-1]])*meter
                end_x = x
                end_y = y
                end_z = z
            else:
                # Absolute coordinates
                start_x = x[:-1]
                start_y = y[:-1]
                start_z = z[:-1]
                end_x = x[1:]
                end_y = y[1:]
                end_z = z[1:]
            length = np.sqrt((end_x - start_x)**2 +
                             (end_y - start_y)**2 +
                             (end_z - start_z)**2)

        self._length = length
        self._x = x
        self._y = y
        self._z = z

    # Overwrite the properties that differ from `Section`

    @property
    def area(self):
        return np.pi * self._diameter * self.length

    @property
    def start_diameter(self):
        return self._diameter

    @property
    def diameter(self):
        return self._diameter

    @property
    def end_diameter(self):
        return self._diameter

    @property
    def volume(self):
        return np.pi * (self._diameter/2)**2 * self.length

    @property
    def electrical_center(self):
        return np.ones(self.n)*0.5

    @property
    def r_length(self):
        return np.pi/4 * (self._diameter**2)/self.length
