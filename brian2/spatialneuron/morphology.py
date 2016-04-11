'''
Neuronal morphology module.
This module defines classes to load and build neuronal morphologies.
'''
import abc
import numbers
from abc import abstractmethod
from collections import OrderedDict, defaultdict, namedtuple
import os

from brian2.units.allunits import meter
from brian2.utils.logger import get_logger
from brian2.units.stdunits import um
from brian2.units.fundamentalunits import (have_same_dimensions, Quantity,
                                           check_units, DimensionMismatchError)
from brian2 import numpy as np

logger = get_logger(__name__)

__all__ = ['Morphology', 'Section', 'Cylinder', 'Soma']


Node = namedtuple('Node',
                  field_names='index,comp_name,x,y,z,diameter,parent,children')


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


class Topology(object):
    '''
    A representation of the topology of a `Morphology`. Has a useful string
    representation, inspired by NEURON's ``topology`` function.
    '''
    def __init__(self, morphology):
        self.morphology = morphology

    def __str__(self):
        # TODO: Make sure that the shown compartments do not get out of hand
        divisor = 1
        return Topology._str_topology(self.morphology, compartments_divisor=divisor)

    @staticmethod
    def _str_topology(morphology, indent=0, named_path='',
                      compartments_divisor=1, parent=None):
        '''
        A simple string-based representation of a morphology. Inspired by
        NEURON's ``topology`` function.
        '''
        description = ' '*indent
        length = max([1, morphology.n//compartments_divisor])
        if parent is not None:
            description += '`'
        if isinstance(morphology, Soma):
            description += '( )'
        else:
            description += '-' * length
            description += '|'
        if len(named_path) == 0:
            description += '  [root] \n'
        else:
            description += '  ' + named_path + '\n'
        for child in morphology.children:
            name = morphology.children.name(child)
            description += Topology._str_topology(child,
                                                  indent=indent+2+length,
                                                  named_path=named_path+'.'+name,
                                                  compartments_divisor=compartments_divisor,
                                                  parent=morphology)
        return description

    __repr__ = __str__


def _rotate(vec, axis, angle):
    '''
    Rotate a vector around an arbitrary axis.

    Parameters
    ----------
    vec : `ndarray`
        The vector to rotate.
    axis : `ndarray`
        The axis around which the vector should be rotated.
    angle : float
        The rotation angle (in radians).

    Returns
    -------
    rotated : `ndarray`
        The rotated vector.
    '''
    return (vec*np.cos(angle) -
            np.cross(axis, vec)*np.sin(angle) +
            axis*np.dot(axis, vec)*(1 - np.cos(angle)))


def _perturb(vec, sigma):
    if sigma == 0:
        return vec
    # Get an arbitrary orthogonal vector
    if vec[1] != 0 or vec[0] != 0:
        orthogonal = np.hstack([vec[1], vec[0], 0])
    else:  # special case for the [0, 0, 1] vector
        orthogonal = np.array([1, 0, 0])

    # Rotate the orthogonal vector
    orthogonal = _rotate(orthogonal, vec, np.random.rand()*np.pi*2)

    # Use an exponentially distributed angle for the perturbation
    perturbation = np.random.exponential(sigma, 1)
    return _rotate(vec, orthogonal, perturbation)


def _add_coordinates(orig_morphology, root=None, parent=None, name=None,
                     section_randomness=0.0, compartment_randomness=0.0,
                     n_th_child=0, total_children=0,
                     overwrite_existing=False):
    # Note that in the following, all values are without physical units

    # The new direction is based on the direction of the parent section
    if parent is None:
        section_dir = np.array([0, 0, 0])
    else:
        section_dir = np.hstack([np.asarray(parent.end_x[-1] - parent.start_x[0]),
                                np.asarray(parent.end_y[-1] - parent.start_y[0]),
                                np.asarray(parent.end_z[-1] - parent.start_z[0])])
        parent_dir_norm = np.sqrt(np.sum(section_dir**2))
        if parent_dir_norm != 0:
            section_dir /= parent_dir_norm
        else:
            section_dir = np.array([0, 0, 0])
    if not overwrite_existing and orig_morphology.x is not None:
        section = orig_morphology.copy_section()
    elif isinstance(orig_morphology, Soma):
        # No perturbation for the soma
        section = Soma(diameter=orig_morphology.diameter,
                       x=section_dir[0]*meter,
                       y=section_dir[1]*meter,
                       z=section_dir[2]*meter)
    else:
        if np.sum(section_dir**2) == 0:
            # We don't have any direction to base this section on (most common
            # case is that the root section is a soma)
            # We stay in the x-y plane and distribute all children in a 360 degree
            # circle around (0, 0, 0)
            section_dir = np.array([1, 0, 0])
            rotation_axis = np.array([0, 0, 1])
            angle_increment = 2*np.pi/total_children
            rotation_angle = np.pi/2 + angle_increment * n_th_child
            section_dir = _rotate(section_dir, rotation_axis, rotation_angle)
        else:
            if section_randomness == 0 and section_dir[2] == 0:  # If we are in the x-y plane, stay there
                rotation_axis = np.array([0, 0, 1])
            else:
                rotation_axis = np.array([-section_dir[1], section_dir[2], 0])
            if section_randomness == 0:
                angle_increment = np.pi/(total_children + 1)
                rotation_angle = -np.pi/2 + angle_increment * (n_th_child + 1)
                section_dir = _rotate(section_dir, rotation_axis, rotation_angle)
        if section_randomness > 0:
            # Rotate randomly
            section_dir = _perturb(section_dir, section_randomness)

        section_dir_norm = np.sqrt(np.sum(section_dir**2))
        section_dir /= section_dir_norm

        # For a soma, we let child sections begin at the surface of the sphere
        if isinstance(parent, Soma):
            origin = parent.diameter/2*section_dir
        else:
            origin = (0, 0, 0)*um
        coordinates = np.zeros((orig_morphology.n + 1, 3))*meter
        start_coords = origin
        coordinates[0, :] = origin
        # Perturb individual compartments as well
        for idx, length in enumerate(orig_morphology.length):
            compartment_dir = _perturb(section_dir, compartment_randomness)
            compartment_dir_norm = np.sqrt(np.sum(compartment_dir**2))
            compartment_dir /= compartment_dir_norm
            current_coords = start_coords + length*compartment_dir
            coordinates[idx + 1, :] = current_coords
            start_coords = current_coords

        if isinstance(orig_morphology, Cylinder) and compartment_randomness == 0:
            section = Cylinder(n=orig_morphology.n,
                               diameter=orig_morphology.diameter[0],
                               x=coordinates[[0, -1], 0],
                               y=coordinates[[0, -1], 1],
                               z=coordinates[[0, -1], 2],
                               type=orig_morphology.type)
        elif isinstance(orig_morphology, Section):
            section = Section(n=orig_morphology.n,
                              diameter=np.hstack([orig_morphology.start_diameter[0],
                                                  orig_morphology.end_diameter])*meter,
                              x=coordinates[:, 0],
                              y=coordinates[:, 1],
                              z=coordinates[:, 2],
                              type=orig_morphology.type)
        else:
            raise NotImplementedError(('Do not know how to deal with section of '
                                       'type %s.' % type(orig_morphology)))
    if parent is None:
        root = section
    else:
        parent.children.add(name, section)

    for idx, child in enumerate(orig_morphology.children):
        _add_coordinates(child, root=root, parent=section,
                         name=orig_morphology.children.name(child),
                         n_th_child=idx, total_children=len(orig_morphology.children),
                         section_randomness=section_randomness,
                         compartment_randomness=compartment_randomness,
                         overwrite_existing=overwrite_existing)
    return section


class Children(object):
    '''
    Helper class to represent the children (sub trees) of a section. Can be
    used like a dictionary (mapping names to `Morphology` objects), but iterates
    over the values (sub trees) instead of over the keys (names).
    '''
    def __init__(self, owner):
        self._owner = owner
        self._counter = 0
        self._children = []
        self._named_children = {}
        self._given_name = defaultdict(lambda: None)

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def __contains__(self, item):
        return item in self._named_children

    def name(self, child):
        '''
        Return the given name (i.e. not the automatic name such as ``1``) for a
        child subtree.

        Parameters
        ----------
        child : `Morphology`

        Returns
        -------
        name : str
            The given name for the ``child``.
        '''
        return self._given_name[child]

    def __getitem__(self, item):
        if isinstance(item, basestring):
            return self._named_children[item]
        else:
            raise TypeError('Index has to be an integer or a string.')

    def add(self, name, subtree, automatic_name=False):
        '''
        Add a new child to the morphology.

        Parameters
        ----------
        name : str
            The name (e.g. ``"axon"``, ``"soma"``) to use for this sub tree.
        subtree : `Morphology`
            The subtree to link as a child.
        automatic_name : bool, optional
            Whether to chose a new name automatically, if a subtree of the same
            name already exists (uses e.g. ``"dend2"`` instead ``"dend"``).
            Defaults to ``False`` and will raise an error instead.
        '''
        if (name in self._named_children and
                    self._named_children[name] is not subtree):
            if automatic_name:
                basename = name
                counter = 1
                while name in self._named_children:
                    counter += 1
                    name = basename + str(counter)
            else:
                raise AttributeError('The name %s is already used for a '
                                     'subtree.' % name)

        if subtree not in self._children:
            self._counter += 1
            self._children.append(subtree)
            self._named_children[str(self._counter)] = subtree
            self._given_name[subtree] = name

        if name is not None:
            self._named_children[name] = subtree

        subtree._parent = self._owner

    def remove(self, name):
        '''
        Remove a subtree from this morphology.

        Parameters
        ----------
        name : str
            The name of the sub tree to remove.
        '''
        if name not in self:
            raise AttributeError('The subtree ' + name + ' does not exist')
        subtree = self._named_children[name]
        del self._named_children[name]
        self._children.remove(subtree)
        subtree._parent = None

    def __repr__(self):
        n = len(self._children)
        s = '<%d children' % n
        if n > 0:
            name_dict = {self.name(sec): sec for sec in self._children}
            s += ': %r' % name_dict
        return s + '>'


class Morphology(object):
    '''
    Neuronal morphology (tree structure).

    The data structure is a tree where each node is an un-branched section
    consisting of a number of connected compartments, each one defined by its
    geometrical properties (length, area, diameter, position).

    Notes
    -----
    You cannot create objects of this class, create a `Soma`, a `Section`, or
    a `Cylinder` instead.
    '''
    __metaclass__ = abc.ABCMeta

    @check_units(n=1)
    def __init__(self, n, type=None):
        if isinstance(n, basestring):
            raise TypeError('Need the number of compartments, not a string. '
                            'If you want to load a morphology from a file, '
                            'use Morphology.from_file instead.')
        self._n = int(n)
        if self._n != n:
            raise TypeError('The number of compartments n has to be an integer '
                            'value.')
        if n <= 0:
            raise ValueError('The number of compartments n has to be at least 1.')
        self.type = type
        self._children = Children(self)
        self._parent = None
        self.indices = MorphologyIndexWrapper(self)

    def __getitem__(self, item):
        '''
        Return the subtree with the given name/index.

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
                l = np.cumsum(np.asarray(self.length))  # coordinate on the section
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
            l = np.hstack([0, np.cumsum(np.asarray(self.length))])  # coordinate on the section
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
                return self._children[item[0]][item[1:]]
            elif item in self._children:
                return self._children[item]
            else:
                raise AttributeError('The subtree ' + item + ' does not exist')
        else:
            raise TypeError('Index of type %s not understood' % type(item))

        return SubMorphology(self, i, j)

    def __setitem__(self, item, child):
        '''
        Inserts the subtree and name it ``item``.
        Ex.: ``neuron['axon']`` or ``neuron['11213']``
        '''
        item = str(item)  # convert int to string
        if (len(item) > 1) and all([c in 'LR123456789' for c in item]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            self.children[item[0]][item[1:]] = child
        else:
            self.children.add(item, child)

    def __delitem__(self, item):
        '''
        Remove the subtree ``item``.
        '''
        item = str(item)  # convert int to string
        if (len(item) > 1) and all([c in 'LR123456789' for c in item]):
            # binary string of the form LLLRLR or 1213 (or mixed)
            del self._children[item[0]][item[1:]]
        else:
            self._children.remove(item)

    def __getattr__(self, item):
        '''
        Return the subtree named ``item``.

        Ex.: ``axon = neuron.axon``
        '''
        if item.startswith('_'):
            return super(object, self).__getattr__(item)
        else:
            return self[item]

    def __setattr__(self, item, child):
        '''
        Attach a subtree and name it ``item``.

        Ex.: ``neuron.axon = Soma(diameter=10*um)``
        '''
        if isinstance(child, Morphology) and not item.startswith('_'):
            self[item] = child
        else:  # If it is not a subtree, then it's a normal class attribute
            object.__setattr__(self, item, child)

    def __delattr__(self, item):
        '''
        Remove the subtree ``item``.
        '''
        del self[item]

    def _indices(self, item=None, index_var='_idx'):
        '''
        Return compartment indices for the main section, relative to the
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

    def topology(self):
        '''
        Return a representation of the topology

        Returns
        -------
        topology : `Topology`
            An object representing the topology (can be converted to a string
            by using ``str(...)`` or simply by printing it with `print`.)
        '''
        return Topology(self)

    def generate_coordinates(self,
                             section_randomness=0.0,
                             compartment_randomness=0.0,
                             overwrite_existing=False):
        '''
        Create a new `Morphology`, with coordinates filled in place where the
        previous morphology did not have any. This is mostly useful for
        plotting a morphology, it does not affect its electrical properties.

        Parameters
        ----------
        section_randomness : float, optional
            The randomness when deciding the direction vector for each new
            section. The given number is the :math:`\beta` parameter of an
            exponential distribution (in degrees) which will be used to
            determine the deviation from the direction of the parent section.
            If the given value equals 0 (the default), then a deterministic
            algorithm will be used instead.
        compartment_randomness : float, optional
            The randomness when deciding the direction vector for each
            compartment within a section. The given number is the :math:`\beta`
            parameter of an exponential distribution (in degrees) which will be
            used to determine the deviation from the main direction of the
            current section. If the given value equals 0 (the default), then all
            compartments will be along a straight line.
        overwrite_existing : bool, optional
            Whether to overwrite existing coordinates in the morphology. This
            is by default set to ``False``, meaning that only sections that do
            not currently have any coordinates set will get new coordinates.
            This allows to conveniently generate a morphology that can be
            plotted for a morphology that is based on points but also has
            artificially added sections (the most common case: an axon added
            to a reconstructed morphology). If set to ``True``, all sections
            will get new coordinates. This can be useful to either get a
            schematic representation of the morphology (with
            ``section_randomness`` and ``compartment_randomness`` both 0) or to
            simply generate a new random variation of a morphology (which will
            still be electrically equivalent, of course).

        Returns
        -------
        morpho_with_coordinates : `Morphology`
            The same morphology, but with coordinates
        '''
        # Convert to radians
        section_randomness *= np.pi/180
        compartment_randomness *= np.pi/180
        return _add_coordinates(self, section_randomness=section_randomness,
                                compartment_randomness=compartment_randomness,
                                overwrite_existing=overwrite_existing)

    @abstractmethod
    def copy_section(self):
        '''
        Create a copy of the current section (attributes of this section only,
        not re-creating the parent/children relation)

        Returns
        -------
        copy : `Morphology`
            A copy of this section (without the links to the parent/children)
        '''
        raise NotImplementedError()

    @property
    def n(self):
        '''
        The number of compartments in this section.
        '''
        return self._n

    def __len__(self):
        '''
        This is not well-defined, use `Morphology.n` or
        `Morphology.total_compartments` instead.
        '''
        raise TypeError('The "length" of a Morphology is ambiguous, use its '
                        '"n" attribute for the number of compartments in this '
                        'section or the "total_compartments" attribute for the '
                        'total number of compartments in the whole sub-tree.')

    @property
    def total_compartments(self):
        '''
        The total number of compartments in this subtree (i.e. the number of
        compartments in this section plus all the compartments in the sections
        deeper in the tree).
        '''
        return self.n + sum(c.total_compartments for c in self.children)

    @property
    def total_sections(self):
        '''
        The total number of sections in this subtree.
        '''
        return 1 + sum(c.total_sections for c in self.children)

    @property
    def parent(self):
        '''
        The parent section of this section.
        '''
        return self._parent

    @property
    def children(self):
        '''
        The children (as a `Children` object) of this section.
        '''
        return self._children

    @abc.abstractproperty
    def end_distance(self):
        '''
        The distance to the root of the morphology at the end of this section.
        '''
        raise NotImplementedError()

    # Per-compartment attributes
    @abc.abstractproperty
    def area(self):
        '''
        The membrane surface area of each compartment in this section.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def volume(self):
        '''
        The volume of each compartment in this section.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def length(self):
        '''
        The length of each compartment in this section.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def r_length_1(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        start and the midpoint of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def r_length_2(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        midpoint and the end of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        raise NotImplementedError()

    # At-midpoint attributes
    @abc.abstractproperty
    def diameter(self):
        '''
        The diameter at the middle of each compartment in this section.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def distance(self):
        '''
        The total distance between the midpoint of each compartment and the root
        of the morphology.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def start_x(self):
        '''
        The x coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def start_y(self):
        '''
        The y coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def start_z(self):
        '''
        The z coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def x(self):
        '''
        The x coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def y(self):
        '''
        The y coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def z(self):
        '''
        The y coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def end_x(self):
        '''
        The x coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def end_y(self):
        '''
        The y coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @abc.abstractproperty
    def end_z(self):
        '''
        The z coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        raise NotImplementedError()

    @property
    def coordinates(self):
        r'''
        Array with all coordinates at the start- and end-points of each
        compartment in this section. The array has size :math:`(n+1) \times 3`,
        where :math:`n` is the number of compartments in this section. Each
        row is one point (start point of first compartment, end point of first
        compartment, end point of second compartment, ...), with the columns
        being the x, y, and z coordinates. Returns ``None`` for morphologies
        without coordinates.
        '''
        if self.x is None:
            return None
        else:
            return Quantity(np.vstack([np.hstack([self.start_x[0], self.end_x[:]]),
                                       np.hstack([self.start_y[0], self.end_y[:]]),
                                       np.hstack([self.start_z[0], self.end_z[:]])]).T,
                            dim=meter.dim)

    @staticmethod
    def _create_section(compartments, name, parent, sections,
                        spherical_soma):

        if (spherical_soma and
                    len(compartments) == 1 and
                    compartments[0].comp_name == 'soma'):
            soma = compartments[0]
            section = Soma(diameter=soma.diameter * um,
                           x=soma.x * um, y=soma.y * um, z=soma.z * um)
        else:
            sec_x, sec_y, sec_z, sec_diameter = zip(*[(c.x, c.y, c.z,
                                                       c.diameter)
                                                      for c in compartments])
            # Add a point for the end of the parent_idx compartment
            if parent is not None:
                n = len(compartments)
                if (parent.comp_name is not None and
                            parent.comp_name.lower() == 'soma'):
                    # For a Soma, we don't use its diameter
                    start_diameter = sec_diameter[0]
                else:
                    start_diameter = parent.diameter
                # Use relative coordinates
                sec_x = np.array(sec_x) - parent.x
                sec_y = np.array(sec_y) - parent.y
                sec_z = np.array(sec_z) - parent.z
                start_x = start_y = start_z = 0.
            else:
                n = len(compartments) - 1
                start_diameter = sec_diameter[0]
                sec_diameter = sec_diameter[1:]
                start_x = sec_x[0]
                start_y = sec_y[0]
                start_z = sec_z[0]
                sec_x = sec_x[1:]
                sec_y = sec_y[1:]
                sec_z = sec_z[1:]

            diameter = np.hstack([start_diameter, sec_diameter])*um
            x = np.hstack([start_x, sec_x])*um
            y = np.hstack([start_y, sec_y])*um
            z = np.hstack([start_z, sec_z])*um
            section = Section(n=n, diameter=diameter, x=x, y=y, z=z,
                              type=name)

        # Add the section as a child to its parent
        if parent is not None:
            parent_sec = sections[parent.index]
            parent_sec.children.add(name, section, automatic_name=True)

        return section

    @staticmethod
    def _compartments_to_sections(compartment, spherical_soma,
                                  current_compartments=None, sections=None):
        # Merge all unbranched compartments of the same type into a single
        # section
        if sections is None:
            sections = OrderedDict()
        if current_compartments is None:
            current_compartments = []

        current_compartments.append(compartment)

        # We have to create a new section, if we are either
        # 1. at a leaf of the tree or at a branching point, or
        # 2. if the compartment type changes
        if (len(compartment.children) != 1 or
                    compartment.comp_name != compartment.children[0].comp_name):
            parent = current_compartments[0].parent
            section = Morphology._create_section(current_compartments,
                                                 compartment.comp_name,
                                                 parent=parent,
                                                 sections=sections,
                                                 spherical_soma=spherical_soma)
            sections[current_compartments[-1].index] = section
            # If we are at a branching point, recurse into all subtrees
            for child in compartment.children:
                Morphology._compartments_to_sections(child,
                                                     spherical_soma=spherical_soma,
                                                     current_compartments=None,
                                                     sections=sections)
        else:
            # A single child of the same type, continue (recursive call)
            Morphology._compartments_to_sections(compartment.children[0],
                                                 spherical_soma=spherical_soma,
                                                 current_compartments=current_compartments,
                                                 sections=sections)

        return sections

    @staticmethod
    def _replace_three_point_soma(compartment, all_compartments):
        # Replace a three-point/two-cylinder soma by a single spherical soma
        # if possible (see http://neuromorpho.org/SomaFormat.html for some
        # details)

        # We are looking for a node with two children of the soma type (and
        # other childen of other types), where the two children don't have any
        # children of their own
        soma_children = [c for c in compartment.children
                         if c.comp_name == 'soma']
        if (compartment.comp_name == 'soma' and len(soma_children) == 2 and
                all(len(c.children) == 0 for c in soma_children)):
            # We've found a 3-point soma to replace
            soma_c = [compartment] + soma_children
            if not all(abs(c.diameter - soma_c[0].diameter) < 1e-15
                       for c in soma_c):
                indices = ', '.join(str(c.index) for c in soma_c)
                raise ValueError('Found a "3-point-soma" (lines: %s), but not '
                                 'all the diameters are '
                                 'identical.' % indices)
            diameter = soma_c[0].diameter
            point_0 = np.array([soma_c[0].x, soma_c[0].y, soma_c[0].z])
            point_1 = np.array([soma_c[1].x, soma_c[1].y, soma_c[1].z])
            point_2 = np.array([soma_c[2].x, soma_c[2].y, soma_c[2].z])
            length_1 = np.sqrt(np.sum((point_1 - point_0) ** 2))
            length_2 = np.sqrt(np.sum((point_2 - point_0) ** 2))
            if (np.abs(length_1 - diameter / 2) > 0.01 or
                        np.abs(length_2 - diameter / 2) > 0.01):
                raise ValueError(('Cannot replace "3-point-soma" by a single '
                                  'point, the second and third points should '
                                  'be positioned one radius away from the '
                                  'first point. Distances are %.3fum and '
                                  '%.3fum, respectively, while the '
                                  'radius is %.3fum.') % (length_1,
                                                          length_2,
                                                          diameter / 2))
            children = [c for c in compartment.children
                        if not c in soma_c]
            compartment = Node(index=compartment.index, comp_name='soma',
                               x=point_0[0], y=point_0[1], z=point_0[2],
                               diameter=diameter, parent=compartment.parent,
                               children=children)
            all_compartments[compartment.index] = compartment
            del all_compartments[soma_children[0].index]
            del all_compartments[soma_children[1].index]

        # Recurse further down the tree
        all_compartments[compartment.index] = compartment
        for child in compartment.children:
            Morphology._replace_three_point_soma(child,
                                                 all_compartments)

    @staticmethod
    def from_points(points, spherical_soma=True):
        '''
        Create a morphology from a sequence of points (similar to the ``SWC``
        format, see `Morphology.from_swc_file`). Each point has to be
        a 7-tuple: ``(index, name, x, y, z, diameter, parent)``

        Note that the values should not use units, but are instead all taken
        to be in micrometers.

        Parameters
        ----------
        points : sequence of 7-tuples
            The points of the morphology.
        spherical_soma : bool, optional
            Whether to model a soma as a sphere.
        Returns
        -------
        morphology : `Morphology`

        Notes
        -----
        This format closely follows the SWC format (see
        `Morphology.from_swc_file`) with two differences: the ``type`` should
        be a string (e.g. ``'soma'``) instead of an integer and the 6-th element
        should be the diameter and not the radius.
        '''
        # First pass through all points to get the dependency structure
        compartments = OrderedDict()
        for counter, point in enumerate(points):
            if len(point) != 7:
                raise ValueError('Each point needs to be described by 7 '
                                 'values, got %d instead.' % len(point))
            index, name, x, y, z, diameter, parent_idx = point

            if index in compartments:
                raise ValueError('Two compartments with index %d' % index)
            if parent_idx == index:
                raise ValueError('Compartment %d lists itself as the parent '
                                 'compartment.' % index)

            if counter == 0 and parent_idx == -1:
                parent = None  # The first compartment does not have a parent
            elif parent_idx not in compartments:
                raise ValueError(('Did not find the compartment %d (parent '
                                  'compartment of compartment %d). Make sure '
                                  'that parent compartments are listed before '
                                  'their children.') % (parent_idx, index))
            else:
                parent = compartments[parent_idx]
            children = []
            node = Node(index, name, x, y, z, diameter, parent, children)
            compartments[index] = node
            if parent is not None:
                parent.children.append(node)

        if spherical_soma:
            Morphology._replace_three_point_soma(compartments.values()[0],
                                                 compartments)

        sections = Morphology._compartments_to_sections(compartments.values()[0],
                                                        spherical_soma)

        # Go through all the sections again and add standard names for all
        # sections (potentially in addition to the name they already have):
        # "L" + "R" for one or two children, "1", "2", "3", etc. otherwise
        children_counter = defaultdict(int)
        for section in sections.itervalues():
            parent = section.parent
            if parent is not None:
                children_counter[parent] += 1
                children = parent.children
                nth_child = children_counter[parent]
                if len(children) <= 2:
                    name = 'L' if nth_child == 1 else 'R'
                else:
                    name = '%d' % nth_child
                children.add(name, section)

        # There should only be one section without parents
        root = [sec for sec in sections.itervalues() if sec.parent is None]
        assert len(root) == 1
        return root[0]

    @staticmethod
    def from_swc_file(filename, spherical_soma=True):
        '''
        Load a morphology from a ``SWC`` file. A large database of morphologies
        in this format can be found at http://neuromorpho.org

        The format consists of an optional header of lines starting with ``#``
        (ignored), followed by a sequence of points, each described in a line
        following the format::

            index type x y z radius parent

        ``index`` is an integer label (starting at 1) that identifies the
        current point and increases by one each line. ``type`` is an integer
        representing the type of the neural segment. The only type that changes
        the interpretation by Brian is the type ``1`` which signals a soma.
        Types ``2`` (axon), ``3`` (dendrite), and ``4`` (apical dendrite) are
        used to give corresponding names to the respective sections. All other
        types are ignored. ``x``, ``y``, and ``z`` are the cartesian coordinates
        at each point and ``r`` is its radius. ``parent`` refers to the index
        of the parent point or is ``-1`` for the root point.

        Parameters
        ----------
        filename : str
            The name of the ``SWC`` file.
        spherical_soma : bool, optional
            Whether to model the soma as a sphere.

        Returns
        -------
        morpho : `Morphology`
            The morphology stored in the given file.
        '''
        swc_types = defaultdict(lambda: None)
        # The following names will be translated into names, all other will be
        # ignored
        swc_types.update({'1': 'soma', '2': 'axon', '3': 'dend', '4': 'apic'})

        with open(filename, 'r') as f:
            points = []
            for line_no, line in enumerate(f):
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    # Ignore comments or empty lines
                    continue
                splitted = line.split()
                if len(splitted) != 7:
                    raise ValueError('Each line of an SWC file has to contain '
                                     '7 space-separated entries, but line %d '
                                     'contains %d.' % (line_no + 1,
                                                       len(splitted)))
                index, comp_type, x, y, z, radius, parent = splitted
                points.append((int(index),
                               swc_types[comp_type],
                               float(x),
                               float(y),
                               float(z),
                               2*float(radius),
                               int(parent)))

        return Morphology.from_points(points, spherical_soma=spherical_soma)

    @staticmethod
    def from_file(filename, spherical_soma=True):
        '''
        Convencience method to load a morphology from a given file. At the
        moment, only ``SWC`` files are supported, calling this function is
        therefore equivalent to calling `Morphology.from_swc_file` directly.

        Parameters
        ----------
        filename : str
            The name of a file storing a morphology.
        spherical_soma : bool, optional
            Whether to model the soma as a sphere.
        Returns
        -------
        morphology : `Morphology`
            The morphology stored in the given file.
        '''
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.swc':
            return Morphology.from_swc_file(filename,
                                            spherical_soma=spherical_soma)
        else:
            raise NotImplementedError('Currently, SWC is the only supported '
                                      'file format.')


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
        '''
        The number of compartments in this sub-section.
        '''
        return self._j - self._i

    def __len__(self):
        return self.n

    @property
    def n_sections(self):
        '''
        The number of sections in this sub-section (always 1).
        '''
        return 1

    # Per-compartment attributes
    @property
    def area(self):
        '''
        The membrane surface area of each compartment in this sub-section.
        '''
        return self._morphology.area[self._i:self._j]

    @property
    def volume(self):
        '''
        The volume of each compartment in this sub-section.
        '''
        return self._morphology.volume[self._i:self._j]

    @property
    def length(self):
        '''
        The length of each compartment in this sub-section.
        '''
        return self._morphology.length[self._i:self._j]

    @property
    def r_length_1(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        start and the midpoint of each compartment in this sub-section.
        Dividing this value by the Intracellular resistivity gives the
        conductance.
        '''
        return self._morphology.r_length_1[self._i:self._j]

    @property
    def r_length_2(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        midpoint and the end of each compartment in this sub-section. Dividing
        this value by the Intracellular resistivity gives the conductance.
        '''
        return self._morphology.r_length_2[self._i:self._j]

    # At-midpoint attributes
    @property
    def diameter(self):
        '''
        The diameter at the middle of each compartment in this sub-section.
        '''
        return self._morphology.diameter[self._i:self._j]

    @property
    def distance(self):
        '''
        The total distance between the midpoint of each compartment in this
        sub-section and the root of the morphology.
        '''
        return self._morphology.distance[self._i:self._j]

    @property
    def start_x(self):
        '''
        The x coordinate at the beginning of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.start_x is None:
            return None
        return self._morphology.start_x[self._i:self._j]

    @property
    def start_y(self):
        '''
        The y coordinate at the beginning of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.start_y is None:
            return None
        return self._morphology.start_y[self._i:self._j]

    @property
    def start_z(self):
        '''
        The z coordinate at the beginning of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.start_z is None:
            return None
        return self._morphology.start_z[self._i:self._j]

    @property
    def x(self):
        '''
        The x coordinate at the midpoint of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.x is None:
            return None
        return self._morphology.x[self._i:self._j]

    @property
    def y(self):
        '''
        The y coordinate at the midpoint of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.y is None:
            return None
        return self._morphology.y[self._i:self._j]

    @property
    def z(self):
        '''
        The z coordinate at the midpoint of each compartment in this
        sub-section. Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.z is None:
            return None
        return self._morphology.z[self._i:self._j]

    @property
    def end_x(self):
        '''
        The x coordinate at the end of each compartment in this sub-section.
        Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.end_x is None:
            return None
        return self._morphology.end_x[self._i:self._j]

    @property
    def end_y(self):
        '''
        The y coordinate at the end of each compartment in this sub-section.
        Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.end_y is None:
            return None
        return self._morphology.end_y[self._i:self._j]

    @property
    def end_z(self):
        '''
        The z coordinate at the end of each compartment in this sub-section.
        Returns ``None`` for morphologies without coordinates.
        '''
        if self._morphology.end_z is None:
            return None
        return self._morphology.end_z[self._i:self._j]


class Soma(Morphology):
    '''
    A spherical, iso-potential soma.

    Parameters
    ----------
    diameter : `Quantity`
        Diameter of the sphere.
    x : `Quantity`, optional
        The x coordinate of the position of the soma.
    y : `Quantity`, optional
        The y coordinate of the position of the soma.
    z : `Quantity`, optional
        The z coordinate of the position of the soma.
    type : str, optional
        The ``type`` of this section, defaults to ``'soma'``.
    '''

    @check_units(diameter=meter, x=meter, y=meter, z=meter)
    def __init__(self, diameter, x=None, y=None, z=None, type='soma'):
        Morphology.__init__(self, n=1, type=type)
        if diameter.shape != () and len(diameter) != 1:
            raise TypeError('Diameter has to be a scalar value.')
        for coord in [x, y, z]:
            if coord is not None and coord.shape != () and len(coord) != 1:
                raise TypeError('Coordinates have to be scalar values.')
        self._diameter = np.ones(1) * diameter
        if any(coord is not None for coord in (x, y, z)):
            default_value = [0]*um
        else:
            default_value = None
        self._x = np.atleast_1d(x) if x is not None else default_value
        self._y = np.atleast_1d(y) if y is not None else default_value
        self._z = np.atleast_1d(z) if z is not None else default_value

    def __repr__(self):
        s = '{klass}(diameter={diam!r}'.format(klass=self.__class__.__name__,
                                                 diam=self.diameter[0])
        if self._x is not None:
            s += ', x={x!r}, y={y!r}, z={z!r}'.format(x=self.x[0],
                                                      y=self.y[0],
                                                      z=self.z[0])
        if self.type != 'soma':
            s += ', type={type!r}'.format(type=self.type)
        return s + ')'

    def copy_section(self):
        return Soma(self.diameter, x=self.x, y=self.y, z=self.z,
                    type=self.type)

    # Note that the per-compartment properties should always return 1D arrays,
    # i.e. for the soma arrays of length 1 instead of scalar values
    @property
    def area(self):
        '''
        The membrane surface area of this section (as an array of length 1).
        '''
        return np.pi * self.diameter ** 2

    @property
    def volume(self):
        '''
        The volume of this section (as an array of length 1).
        '''
        return (np.pi * self.diameter ** 3)/6

    @property
    def length(self):
        '''
        The "length" (equal to `diameter`) of this section (as an array of
        length 1).
        '''
        return self.diameter

    @property
    def r_length_1(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        start and the midpoint of each compartment. Returns a fixed (high)
        value for a `Soma`, corresponding to a section with very low
        intracellular resistance.
        '''
        return [1]*meter

    @property
    def r_length_2(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        midpoint and the end of each compartment. Returns a fixed (high)
        value for a `Soma`, corresponding to a section with very low
        intracellular resistance.
        '''
        return [1]*meter

    @property
    def diameter(self):
        '''
        The diameter of this section (as an array of length 1).
        '''
        return self._diameter

    @property
    def distance(self):
        '''
        The total distance between the midpoint of this section and the root
        of the morphology. The `Soma` is most likely the root of the
        morphology, and therefore the `distance` is 0.
        '''
        dist = self._parent.distance if self._parent else 0*um
        return dist

    @property
    def start_x(self):
        '''
        The x-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._x

    @property
    def start_y(self):
        '''
        The y-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._y

    @property
    def start_z(self):
        '''
        The z-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._z

    @property
    def x(self):
        '''
        The x-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._x

    @property
    def y(self):
        '''
        The y-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._y

    @property
    def z(self):
        '''
        The z-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._z

    @property
    def end_x(self):
        '''
        The x-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._x

    @property
    def end_y(self):
        '''
        The y-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._y

    @property
    def end_z(self):
        '''
        The z-coordinate of the current section (as an array of length 1). Note
        that a `Soma` is modelled as a "point" with finite surface/volume,
        equivalent to that of a sphere with the given `diameter`. It's start-,
        midpoint-, and end-coordinates are therefore identical.
        '''
        return self._z

    @property
    def end_distance(self):
        '''
        The distance to the root of the morphology at the end of this section.
        Note that since a `Soma` is modeled as a point (see docs of `x`, etc.),
        it does not add anything to the total distance, e.g. a section
        connecting to a `Soma` has a `distance` of 0 um at its start.
        '''
        dist = self._parent.end_distance if self._parent is not None else 0 * um
        return dist


class Section(Morphology):
    '''
    A section (unbranched structure), described as a sequence of truncated
    cones with potentially varying diameters and lengths per compartment.

    Parameters
    ----------
    diameter : `Quantity`
        Either a single value (the constant diameter along the whole section),
        or a value of length ``n+1``. When ``n+1`` values are given, they
        will be interpreted as the diameters at the start of the first
        compartment and the diameters at the end of each compartment (which is
        equivalent to: the diameter at the start of each compartment and the
        diameter at the end of the last compartment.
    n : int, optional
        The number of compartments in this section. Defaults to 1.
    length : `Quantity`, optional
        Either a single value (the total length of the section), or a value of
        length ``n``, the length of each individual compartment. Cannot be
        combined with the specification of coordinates.
    x : `Quantity`, optional
        ``n+1`` values, specifying the x coordinates of the start point of the
        first compartment and the end-points of all compartments (which is
        equivalent to: the start point of all compartments and the end point of
        the last compartment). The coordinates are interpreted as relative to
        the end point of the parent compartment (if any), so in most cases the
        start point should be ``0*um``. The common exception is a cylinder
        connecting to a `Soma`, here the start point can be used to make the
        cylinder start at the surface of the sphere instead of at its center.
        You can specify all of ``x``, ``y``, or ``z`` to specify
        a morphology in 3D, or only one or two out of them to specify a
        morphology in 1D or 2D.
    y : `Quantity`, optional
        See ``x``
    z : `Quantity`, optional
        See ``x``
    type : str, optional
        The type (e.g. ``"axon"``) of this `Section`.
    '''
    @check_units(n=1, length=meter, diameter=meter, start_diameter=meter,
                 x=meter, y=meter, z=meter)
    def __init__(self, diameter, n=1, length=None, x=None, y=None, z=None,
                 start_diameter=None, origin=None, type=None):
        n = int(n)
        Morphology.__init__(self, n=n, type=type)

        if diameter.ndim != 1 or len(diameter) != n+1:
            raise TypeError('The diameter argument has to be a one-dimensional '
                            'array of length %d' % (n + 1))
        self._diameter = Quantity(diameter, copy=True).reshape((n+1, ))

        if ((x is not None or y is not None or z is not None) and
                length is not None):
            raise TypeError('Cannot specify coordinates and length at the same '
                            'time.')

        if length is not None:
            # Length
            if length.ndim != 1 or len(length) != n:
                raise TypeError('The length argument has to be a '
                                'one-dimensional array of length %d' % n)
            self._length = Quantity(length, copy=True).reshape((n, ))
            self._x = self._y = self._z = None
        else:
            # Coordinates
            if x is None and y is None and z is None:
                raise TypeError('No length specified, need to specify at least '
                                'one out of x, y, or z.')
            for name, value in [('x', x), ('y', y), ('z', z)]:
                if value is not None and (value.ndim != 1 or len(value) != n + 1):
                    raise TypeError(('%s needs to be a 1-dimensional array '
                                     'of length %d.') % (name, n + 1))
            self._x = Quantity(x, copy=True).reshape((n+1, )) if x is not None else np.zeros(n + 1)*um
            self._y = Quantity(y, copy=True).reshape((n+1, )) if y is not None else np.zeros(n + 1)*um
            self._z = Quantity(z, copy=True).reshape((n+1, )) if z is not None else np.zeros(n + 1)*um

            length = np.sqrt((self.end_x - self.start_x) ** 2 +
                             (self.end_y - self.start_y) ** 2 +
                             (self.end_z - self.start_z) ** 2)
            self._length = length

    def __repr__(self):
        if all(np.abs(self.end_diameter - self.end_diameter[0]) < self.end_diameter[0]*1e-12):
            # Constant diameter
            diam = self.end_diameter[0]
        else:
            diam = np.hstack([np.asarray(self.start_diameter[0]),
                              np.asarray(self.end_diameter)])*meter
        s = '{klass}(diameter={diam!r}'.format(klass=self.__class__.__name__,
                                               diam=diam)
        if self.n != 1:
            s += ', n={n}'.format(n=self.n)
        if self._x is not None:
            s += ', x={x!r}, y={y!r}, z={z!r}'.format(x=self._x,
                                                      y=self._y,
                                                      z=self._z)
        else:
            s += ', length={length!r}'.format(length=sum(self._length))
        if self.type is not None:
            s += ', type={type!r}'.format(type=self.type)
        return s + ')'

    def copy_section(self):
        if self.x is None:
            x, y, z = None, None, None
            length = self.length
        else:
            x, y, z = self._x, self._y, self._z
            length = None
        return Section(diameter=self._diameter, n=self.n, x=x, y=y, z=z,
                       length=length, type=self.type)

    @property
    def area(self):
        r'''
        The membrane surface area of each compartment in this section. The
        surface area of each compartment is calculated as
        :math:`\frac{\pi}{2}(d_1 + d_2)\sqrt{\frac{(d_1 - d_2)^2}{4} + l^2)}`,
        where :math:`l` is the length of the compartment, and :math:`d_1` and
        :math:`d_2` are the diameter at the start and end of the compartment,
        respectively. Note that this surface area does not contain the area of
        the two disks at the two sides of the truncated cone.
        '''
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return np.pi/2*(d_1 + d_2)*np.sqrt(((d_1 - d_2)**2)/4 + self._length**2)

    @property
    def volume(self):
        r'''
        The volume of each compartment in this section. The volume of each
        compartment is calculated as
        :math:`\frac{\pi}{12} l (d_1^2 + d_1 d_2 + d_2^2)`,
        where :math:`l` is the length of the compartment, and :math:`d_1` and
        :math:`d_2` are the diameter at the start and end of the compartment,
        respectively.
        '''
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        return np.pi * self._length * (d_1**2 + d_1*d_2 + d_2**2)/12

    @property
    def length(self):
        '''
        The length of each compartment in this section.
        '''
        return self._length

    @property
    def start_diameter(self):
        '''
        The diameter at the start of each compartment in this section.
        '''
        return Quantity(self._diameter[:-1], copy=True)

    @property
    def end_diameter(self):
        '''
        The diameter at the end of each compartment in this section.
        '''
        return Quantity(self._diameter[1:], copy=True)

    @property
    def diameter(self):
        '''
        The diameter at the middle of each compartment in this section.
        '''
        d_1 = self.start_diameter
        d_2 = self.end_diameter
        # Diameter at the center
        return 0.5*(d_1 + d_2)

    @property
    def distance(self):
        '''
        The total distance between the midpoint of each compartment and the root
        of the morphology.
        '''
        dist = self._parent.end_distance if self._parent is not None else 0 * um
        return dist + np.cumsum(self.length) - 0.5 * self.length

    @property
    def end_distance(self):
        '''
        The distance to the root of the morphology at the end of this section.
        '''
        return self.distance[-1] + 0.5 * self.length[-1]

    @property
    def r_length_1(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        start and the midpoint of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        d_1 = self.start_diameter
        d_2 = (self.start_diameter + self.end_diameter)*0.5
        return np.pi/2 * (d_1 * d_2)/self._length

    @property
    def r_length_2(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        midpoint and the end of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        d_1 = (self.start_diameter + self.end_diameter)*0.5
        d_2 = self.end_diameter
        return np.pi/2 * (d_1 * d_2)/self._length

    @property
    def start_x(self):
        '''
        The x coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._x is None:
            return None
        if self.parent is not None and self.parent.end_x is not None:
            parent_x = self.parent.end_x[-1]
        else:
            parent_x = 0*um
        return parent_x + self._x[:-1]

    @property
    def start_y(self):
        '''
        The y coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._y is None:
            return None
        if self.parent is not None and self.parent.end_y is not None:
            parent_y = self.parent.end_y[-1]
        else:
            parent_y = 0*um
        return parent_y + self._y[:-1]

    @property
    def start_z(self):
        '''
        The z coordinate at the beginning of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._z is None:
            return None
        if self.parent is not None and self.parent.end_z is not None:
            parent_z = self.parent.end_z[-1]
        else:
            parent_z = 0*um
        return parent_z + self._z[:-1]

    @property
    def x(self):
        '''
        The x coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._x is None:
            return None
        diff_x = (self.end_x - self.start_x)
        return self.start_x + 0.5*diff_x

    @property
    def y(self):
        '''
        The y coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._y is None:
            return None
        diff_y = (self.end_y - self.start_y)
        return self.start_y + 0.5*diff_y

    @property
    def z(self):
        '''
        The z coordinate at the midpoint of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._z is None:
            return None
        diff_z = (self.end_z - self.start_z)
        return self.start_z + 0.5*diff_z

    @property
    def end_x(self):
        '''
        The x coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._x is None:
            return None
        if self.parent is not None and self.parent.end_x is not None:
            parent_x = self.parent.end_x[-1]
        else:
            parent_x = 0*um
        return parent_x + self._x[1:]

    @property
    def end_y(self):
        '''
        The y coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._y is None:
            return None
        if self.parent is not None and self.parent.end_y is not None:
            parent_y = self.parent.end_y[-1]
        else:
            parent_y = 0*um
        return parent_y + self._y[1:]

    @property
    def end_z(self):
        '''
        The z coordinate at the end of each compartment. Returns ``None``
        for morphologies without coordinates.
        '''
        if self._z is None:
            return None
        if self.parent is not None and self.parent.end_z is not None:
            parent_z = self.parent.end_z[-1]
        else:
            parent_z = 0*um
        return parent_z + self._z[1:]


class Cylinder(Section):
    '''
    A cylindrical section. For sections with more complex geometry (varying
    length and/or diameter of each compartment), use the `Section` class.

    Parameters
    ----------
    diameter : `Quantity`
        The diameter of the cylinder.
    n : int, optional
        The number of compartments in this section. Defaults to 1.
    length : `Quantity`, optional
        The length of the cylinder. Cannot be combined with the specification
        of coordinates.
    x : `Quantity`, optional
        A sequence of two values, the start and the end point of the cylinder.
        The coordinates are interpreted as relative to the end point of the
        parent compartment (if any), so in most cases the start point should
        be ``0*um``. The common exception is a cylinder connecting to a `Soma`,
        here the start point can be used to make the cylinder start at the
        surface of the sphere instead of at its center.
        You can specify all of ``x``, ``y``, or ``z`` to specify
        a morphology in 3D, or only one or two out of them to specify a
        morphology in 1D or 2D.
    y : `Quantity`, optional
        See ``x``
    z : `Quantity`, optional
        See ``x``
    type : str, optional
        The type (e.g. ``"axon"``) of this `Cylinder`.
    '''
    @check_units(n=1, length=meter, diameter=meter, x=meter, y=meter, z=meter)
    def __init__(self, diameter, n=1, length=None, x=None, y=None, z=None,
                 type=None):
        n = int(n)
        Morphology.__init__(self, n=n, type=type)

        # Diameter
        if diameter.shape != () and (diameter.ndim > 1 or len(diameter) != 1):
            raise TypeError('The diameter argument has to be a single value.')
        diameter = np.ones(n) * diameter
        self._diameter = diameter

        if ((x is not None or y is not None or z is not None) and
                    length is not None):
            raise TypeError('Cannot specify coordinates and length at the same '
                            'time.')

        if length is not None:
            # Length
            if length.shape != () and (length.ndim > 1 or len(length) != 1):
                raise TypeError('The length argument has to be a single value.')
            self._length = np.ones(n) * (length/n)  # length was total length
            self._x = self._y = self._z = None
        else:
            # Coordinates
            if x is None and y is None and z is None:
                raise TypeError('No length specified, need to specify at least '
                                'one out of x, y, or z.')
            for name, value in [('x', x), ('y', y), ('z', z)]:
                if value is not None and (value.ndim != 1 or len(value) != 2):
                    raise TypeError('%s needs to be a 1-dimensional array of '
                                    'length 2 (start and end point)' % name)
            self._x = np.linspace(x[0], x[1], n+1) if x is not None else np.zeros(n+1)*um
            self._y = np.linspace(y[0], y[1], n+1) if y is not None else np.zeros(n+1)*um
            self._z = np.linspace(z[0], z[1], n+1) if z is not None else np.zeros(n+1)*um
            length = np.sqrt((self.end_x - self.start_x) ** 2 +
                             (self.end_y - self.start_y) ** 2 +
                             (self.end_z - self.start_z) ** 2)
            self._length = length

    def __repr__(self):
        s = '{klass}(diameter={diam!r}'.format(klass=self.__class__.__name__,
                                               diam=self.diameter[0])
        if self.n != 1:
            s += ', n={n}'.format(n=self.n)
        if self._x is not None:
            s += ', x={x!r}, y={y!r}, z={z!r}'.format(x=self._x[[0, -1]],
                                                      y=self._y[[0, -1]],
                                                      z=self._z[[0, -1]])
        else:
            s += ', length={length!r}'.format(length=sum(self._length))
        if self.type is not None:
            s += ', type={type!r}'.format(type=self.type)
        return s + ')'

    def copy_section(self):
        if self.x is None:
            return Cylinder(self.diameter[0], n=self.n, length=self.length,
                            type=self.type)
        else:
            return Cylinder(self.diameter[0], n=self.n,
                            x=self._x[[0, -1]], y=self._y[[0, -1]], z=self._z[[0, -1]],
                            type=self.type)

    # Overwrite the properties that differ from `Section`
    @property
    def area(self):
        r'''
        The membrane surface area of each compartment in this section. The
        surface area of each compartment is calculated as
        :math:`\pi d l`,
        where :math:`l` is the length of the compartment, and :math:`d` is its
        diameter. Note that this surface area does not contain the area of
        the two disks at the two sides of the cylinder.
        '''
        return np.pi * self._diameter * self.length

    @property
    def start_diameter(self):
        '''
        The diameter at the start of each compartment in this section.
        '''
        return self._diameter

    @property
    def diameter(self):
        '''
        The diameter at the middle of each compartment in this section.
        '''
        return self._diameter

    @property
    def end_diameter(self):
        '''
        The diameter at the end of each compartment in this section.
        '''
        return self._diameter

    @property
    def volume(self):
        r'''
        The volume of each compartment in this section. The volume of each
        compartment is calculated as
        :math:`\pi \frac{d}{2}^2 l` ,
        where :math:`l` is the length of the compartment, and :math:`d` is its
        diameter.
        '''
        return np.pi * (self._diameter/2)**2 * self.length

    @property
    def r_length_1(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        start and the midpoint of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        return np.pi/2 * (self._diameter**2)/self.length

    @property
    def r_length_2(self):
        '''
        The geometry-dependent term to calculate the conductance between the
        midpoint and the end of each compartment. Dividing this value by the
        Intracellular resistivity gives the conductance.
        '''
        return np.pi/2 * (self._diameter**2)/self.length
