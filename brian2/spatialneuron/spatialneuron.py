'''
Compartmental models.
This module defines the SpatialNeuron class, which defines multicompartmental models.
'''
import weakref
import copy

import sympy as sp
import numpy as np

from brian2.core.variables import Variables
from brian2.equations.equations import (Equations, PARAMETER, SUBEXPRESSION,
                                        DIFFERENTIAL_EQUATION, SingleEquation,
                                        extract_constant_subexpressions)
from brian2.groups.group import Group, CodeRunner, create_runner_codeobj
from brian2.units.allunits import ohm, siemens, amp, meter, volt
from brian2.units.fundamentalunits import Quantity, Unit, fail_for_dimension_mismatch, have_same_dimensions, DimensionMismatchError
from brian2.units.stdunits import uF, cm
from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.utils.logger import get_logger
from brian2.groups.neurongroup import NeuronGroup, SubexpressionUpdater
from brian2.groups.subgroup import Subgroup
from brian2.equations.codestrings import Expression

__all__ = ['SpatialNeuron']

logger = get_logger(__name__)


class FlatMorphology(object):
    '''
    Container object to store the flattened representation of a morphology.
    Note that all values are stored as numpy arrays without unit information
    (i.e. in base units).
    '''
    def __init__(self, morphology):
        self.n = n = morphology.total_compartments  # Total number of compartments
        # Per-compartment attributes
        self.length = np.zeros(n)
        self.distance = np.zeros(n)
        self.area = np.zeros(n)
        self.diameter = np.zeros(n)
        self.volume = np.zeros(n)
        self.r_length_1 = np.zeros(n)
        self.r_length_2 = np.zeros(n)
        self.start_x = np.zeros(n)
        self.start_y = np.zeros(n)
        self.start_z = np.zeros(n)
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.z = np.zeros(n)
        self.end_x = np.zeros(n)
        self.end_y = np.zeros(n)
        self.end_z = np.zeros(n)
        self.depth = np.zeros(n, dtype=np.int32)
        self.sections = sections = morphology.total_sections
        self.end_distance = np.zeros(sections)
        # Index of the parent for each section (-1 for the root)
        self.morph_parent_i = np.zeros(sections, dtype=np.int32)
        # The children indices for each section (list of lists, will be later
        # transformed into an array representation)
        self.morph_children = []
        # each section is child of exactly one parent, this stores the index in
        # the parents list of children
        self.morph_idxchild = np.zeros(sections, dtype=np.int32)
        self.starts = np.zeros(sections, dtype=np.int32)
        self.ends = np.zeros(sections, dtype=np.int32)

        # recursively fill the data structures
        self._sections_without_coordinates = False
        self.has_coordinates = False
        self._offset = 0
        self._section_counter = 0
        self._insert_data(morphology)
        if self.has_coordinates and self._sections_without_coordinates:
            logger.info('The morphology has a mix of sections with and '
                        'without coordinates. The SpatialNeuron object '
                        'will store NaN values for the coordinates of '
                        'the sections that do not specify coordinates. '
                        'Call generate_coordinates on the morphology '
                        'before creating the SpatialNeuron object to fill '
                        'in the missing coordinates.')
        # Do not store coordinates for morphologies that don't define them
        if not self.has_coordinates:
            self.start_x = self.start_y = self.start_z = None
            self.x = self.y = self.z = None
            self.end_x = self.end_y = self.end_z = None

        # Transform the list of list of children into a 2D array (stored as
        # 1D) -- note that this wastes space if the number of children per
        # section is very different. In practice, this should not be much of a
        # problem since most sections have 0, 1, or 2 children (e.g. SWC files
        # on neuromorpho.org are all binary trees)
        self.morph_children_num = np.array([len(c)
                                            for c in self.morph_children] + [0])
        max_children = max(self.morph_children_num)
        morph_children = np.zeros((sections+1, max_children), dtype=np.int32)
        for idx, section_children in enumerate(self.morph_children):
            morph_children[idx, :len(section_children)] = section_children
        self.morph_children = morph_children.reshape(-1)

    def _insert_data(self, section, parent_idx=-1, depth=0):
        n = section.n
        start = self._offset
        end = self._offset + n
        # Compartment attributes
        self.depth[start:end] = depth
        self.length[start:end] = np.asarray(section.length)
        self.distance[start:end] = np.asarray(section.distance)
        self.area[start:end] = np.asarray(section.area)
        self.diameter[start:end] = np.asarray(section.diameter)
        self.volume[start:end] = np.asarray(section.volume)
        self.r_length_1[start:end] = np.asarray(section.r_length_1)
        self.r_length_2[start:end] = np.asarray(section.r_length_2)
        if section.x is None:
            self._sections_without_coordinates = True
            self.start_x[start:end] = np.ones(n)*np.nan
            self.start_y[start:end] = np.ones(n)*np.nan
            self.start_z[start:end] = np.ones(n)*np.nan
            self.x[start:end] = np.ones(n)*np.nan
            self.y[start:end] = np.ones(n)*np.nan
            self.z[start:end] = np.ones(n)*np.nan
            self.end_x[start:end] = np.ones(n)*np.nan
            self.end_y[start:end] = np.ones(n)*np.nan
            self.end_z[start:end] = np.ones(n)*np.nan
        else:
            self.has_coordinates = True
            self.start_x[start:end] = np.asarray(section.start_x)
            self.start_y[start:end] = np.asarray(section.start_y)
            self.start_z[start:end] = np.asarray(section.start_z)
            self.x[start:end] = np.asarray(section.x)
            self.y[start:end] = np.asarray(section.y)
            self.z[start:end] = np.asarray(section.z)
            self.end_x[start:end] = np.asarray(section.end_x)
            self.end_y[start:end] = np.asarray(section.end_y)
            self.end_z[start:end] = np.asarray(section.end_z)

        # Section attributes
        idx = self._section_counter
        # We start counting from 1 for the parent indices, since the index 0
        # is used for the (virtual) root compartment
        self.morph_parent_i[idx] = parent_idx + 1
        self.morph_children.append([])
        self.starts[idx] = start
        self.ends[idx] = end
        # Append ourselves to the children list of our parent
        self.morph_idxchild[idx] = len(self.morph_children[parent_idx+1])
        self.morph_children[parent_idx + 1].append(idx + 1)
        self.end_distance[idx] = section.end_distance
        # Recurse down the tree
        self._offset += n
        self._section_counter += 1
        for child in section.children:
            self._insert_data(child, parent_idx=idx, depth=depth+1)


class SpatialNeuron(NeuronGroup):
    '''
    A single neuron with a morphology and possibly many compartments.

    Parameters
    ----------
    morphology : `Morphology`
        The morphology of the neuron.
    model : (str, `Equations`)
        The equations defining the group.
    method : (str, function), optional
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function that receives an
        `Equations` object and returns the corresponding abstract code. If no
        method is specified, a suitable method will be chosen automatically.
    threshold : str, optional
        The condition which produces spikes. Should be a single line boolean
        expression.
    threshold_location : (int, `Morphology`), optional
        Compartment where the threshold condition applies, specified as an
        integer (compartment index) or a `Morphology` object corresponding to
        the compartment (e.g. ``morpho.axon[10*um]``).
        If unspecified, the threshold condition applies at all compartments.
    Cm : `Quantity`, optional
        Specific capacitance in uF/cm**2 (default 0.9). It can be accessed and
        modified later as a state variable. In particular, its value can differ
        in different compartments.
    Ri : `Quantity`, optional
        Intracellular resistivity in ohm.cm (default 150). It can be accessed
        as a shared state variable, but modified only before the first run.
        It is uniform across the neuron.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on reset.
    events : dict, optional
        User-defined events in addition to the "spike" event defined by the
        ``threshold``. Has to be a mapping of strings (the event name) to
        strings (the condition) that will be checked.
    refractory : {str, `Quantity`}, optional
        Either the length of the refractory period (e.g. ``2*ms``), a string
        expression that evaluates to the length of the refractory period
        after each spike (e.g. ``'(1 + rand())*ms'``), or a string expression
        evaluating to a boolean value, given the condition under which the
        neuron stays refractory after a spike (e.g. ``'v > -20*mV'``)
    namespace : dict, optional
        A dictionary mapping identifier names to objects. If not given, the
        namespace will be filled in at the time of the call of `Network.run`,
        with either the values from the ``namespace`` argument of the
        `Network.run` method or from the local context, if no such argument is
        given.
    dtype : (`dtype`, `dict`), optional
        The `numpy.dtype` that will be used to store the values, or a
        dictionary specifying the type for variable names. If a value is not
        provided for a variable (or no value is provided at all), the preference
        setting `core.default_float_dtype` is used.
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the group, otherwise use ``spatialneuron_0``, etc.
    '''

    def __init__(self, morphology=None, model=None, threshold=None,
                 refractory=False, reset=None, events=None,
                 threshold_location=None,
                 dt=None, clock=None, order=0, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm,
                 name='spatialneuron*', dtype=None, namespace=None,
                 method=('exact', 'exponential_euler', 'rk2', 'heun'),
                 method_options=None):

        # #### Prepare and validate equations
        if isinstance(model, basestring):
            model = Equations(model)
        if not isinstance(model, Equations):
            raise TypeError(('model has to be a string or an Equations '
                             'object, is "%s" instead.') % type(model))

        # Insert the threshold mechanism at the specified location
        if threshold_location is not None:
            if hasattr(threshold_location,
                       '_indices'):  # assuming this is a method
                threshold_location = threshold_location._indices()
                # for now, only a single compartment allowed
                if len(threshold_location) == 1:
                    threshold_location = threshold_location[0]
                else:
                    raise AttributeError(('Threshold can only be applied on a '
                                          'single location'))
            threshold = '(' + threshold + ') and (i == ' + str(threshold_location) + ')'

        # Check flags (we have point currents)
        model.check_flags({DIFFERENTIAL_EQUATION: ('point current',),
                           PARAMETER: ('constant', 'shared', 'linked', 'point current'),
                           SUBEXPRESSION: ('shared', 'point current',
                                           'constant over dt')})
        #: The original equations as specified by the user (i.e. before
        #: inserting point-currents into the membrane equation, before adding
        #: all the internally used variables and constants, etc.).
        self.user_equations = model

        # Separate subexpressions depending whether they are considered to be
        # constant over a time step or not (this would also be done by the
        # NeuronGroup initializer later, but this would give incorrect results
        # for the linearity check)
        model, constant_over_dt = extract_constant_subexpressions(model)

        # Extract membrane equation
        if 'Im' in model:
            if len(model['Im'].flags):
                raise TypeError('Cannot specify any flags for the transmembrane '
                                'current Im.')
            membrane_expr = model['Im'].expr  # the membrane equation
        else:
            raise TypeError('The transmembrane current Im must be defined')

        model_equations = []
        # Insert point currents in the membrane equation
        for eq in model.itervalues():
            if eq.varname == 'Im':
                continue  # ignore -- handled separately
            if 'point current' in eq.flags:
                fail_for_dimension_mismatch(eq.dim, amp,
                                            "Point current " + eq.varname + " should be in amp")
                membrane_expr = Expression(
                    str(membrane_expr.code) + '+' + eq.varname + '/area')
                eq = SingleEquation(eq.type, eq.varname, eq.dim, expr=eq.expr,
                                    flags=list(set(eq.flags)-set(['point current'])))
            model_equations.append(eq)

        model_equations.append(SingleEquation(SUBEXPRESSION, 'Im',
                                              dimensions=(amp/meter**2).dim,
                                              expr=membrane_expr))
        model_equations.append(SingleEquation(PARAMETER, 'v', volt.dim))
        model = Equations(model_equations)

        ###### Process model equations (Im) to extract total conductance and the remaining current
        # Expand expressions in the membrane equation
        for var, expr in model.get_substituted_expressions(include_subexpressions=True):
            if var == 'Im':
                Im_expr = expr
                break
        else:
            raise AssertionError('Model equations did not contain Im!')

        # Differentiate Im with respect to v
        Im_sympy_exp = str_to_sympy(Im_expr.code)
        v_sympy = sp.Symbol('v', real=True)
        diffed = sp.diff(Im_sympy_exp, v_sympy)

        unevaled_derivatives = diffed.atoms(sp.Derivative)
        if len(unevaled_derivatives):
            raise TypeError('Cannot take the derivative of "{Im}" with respect '
                            'to v.'.format(Im=Im_expr.code))

        gtot_str = sympy_to_str(sp.simplify(-diffed))
        I0_str = sympy_to_str(sp.simplify(Im_sympy_exp - diffed*v_sympy))

        if gtot_str == '0':
            gtot_str += '*siemens/meter**2'
        if I0_str == '0':
            I0_str += '*amp/meter**2'
        gtot_str = "gtot__private=" + gtot_str + ": siemens/meter**2"
        I0_str = "I0__private=" + I0_str + ": amp/meter**2"

        model += Equations(gtot_str + "\n" + I0_str)

        # Insert morphology (store a copy)
        self.morphology = copy.deepcopy(morphology)

        # Flatten the morphology
        self.flat_morphology = FlatMorphology(morphology)

        # Equations for morphology
        # TODO: check whether Cm and Ri are already in the equations
        #       no: should be shared instead of constant
        #       yes: should be constant (check)
        eqs_constants = Equations("""
        length : meter (constant)
        distance : meter (constant)
        area : meter**2 (constant)
        volume : meter**3
        Ic : amp/meter**2
        diameter : meter (constant)
        Cm : farad/meter**2 (constant)
        Ri : ohm*meter (constant, shared)
        r_length_1 : meter (constant)
        r_length_2 : meter (constant)
        time_constant = Cm/gtot__private : second
        space_constant = (2/pi)**(1.0/3.0) * (area/(1/r_length_1 + 1/r_length_2))**(1.0/6.0) /
                         (2*(Ri*gtot__private)**(1.0/2.0)) : meter
        """)
        if self.flat_morphology.has_coordinates:
            eqs_constants += Equations('''
            x : meter (constant)
            y : meter (constant)
            z : meter (constant)
            ''')

        NeuronGroup.__init__(self, morphology.total_compartments,
                             model=model + eqs_constants,
                             method_options=method_options,
                             threshold=threshold, refractory=refractory,
                             reset=reset, events=events,
                             method=method, dt=dt, clock=clock, order=order,
                             namespace=namespace, dtype=dtype, name=name)
        # Parameters and intermediate variables for solving the cable equations
        # Note that some of these variables could have meaningful physical
        # units (e.g. _v_star is in volt, _I0_all is in amp/meter**2 etc.) but
        # since these variables should never be used in user code, we don't
        # assign them any units
        self.variables.add_arrays(['_ab_star0', '_ab_star1', '_ab_star2',
                                   '_b_plus', '_b_minus',
                                   '_v_star', '_u_plus', '_u_minus',
                                   '_v_previous', '_c',
                                   # The following two are only necessary for
                                   # C code where we cannot deal with scalars
                                   # and arrays interchangeably:
                                   '_I0_all', '_gtot_all'],
                                  size=self.N, read_only=True)

        self.Cm = Cm
        self.Ri = Ri
        # These explict assignments will load the morphology values from disk
        # in standalone mode
        self.distance_ = self.flat_morphology.distance
        self.length_ = self.flat_morphology.length
        self.area_ = self.flat_morphology.area
        self.diameter_ = self.flat_morphology.diameter
        self.r_length_1_ = self.flat_morphology.r_length_1
        self.r_length_2_ = self.flat_morphology.r_length_2
        if self.flat_morphology.has_coordinates:
            self.x_ = self.flat_morphology.x
            self.y_ = self.flat_morphology.y
            self.z_ = self.flat_morphology.z

        # Performs numerical integration step
        self.add_attribute('diffusion_state_updater')
        self.diffusion_state_updater = SpatialStateUpdater(self, method,
                                                           clock=self.clock,
                                                           order=order)

        # Update v after the gating variables to obtain consistent Ic and Im
        self.diffusion_state_updater.order = 1

        # Creation of contained_objects that do the work
        self.contained_objects.extend([self.diffusion_state_updater])

        if len(constant_over_dt):
            self.subexpression_updater = SubexpressionUpdater(self,
                                                              constant_over_dt)
            self.contained_objects.append(self.subexpression_updater)

    def __getattr__(self, name):
        '''
        Subtrees are accessed by attribute, e.g. neuron.axon.
        '''
        return self.spatialneuron_attribute(self, name)

    def __getitem__(self, item):
        '''
        Selects a segment, where x is a slice of either compartment
        indexes or distances.
        Note a: segment is not a SpatialNeuron, only a Group.
        '''
        return self.spatialneuron_segment(self, item)

    @staticmethod
    def _find_subtree_end(morpho):
        '''
        Go down a morphology recursively to find the (absolute) index of the
        "final" compartment (i.e. the one with the highest index) of the
        subtree.

        Parameters
        ----------
        morpho : `Morphology`
            The morphology for which to find the index.

        Returns
        -------
        index : int
            The highest index within the subtree.
        '''
        indices = [morpho.indices[-1]]
        for child in morpho.children:
            indices.append(SpatialNeuron._find_subtree_end(child))
        return max(indices)

    @staticmethod
    def spatialneuron_attribute(neuron, name):
        '''
        Selects a subtree from `SpatialNeuron` neuron and returns a `SpatialSubgroup`.
        If it does not exist, returns the `Group` attribute.
        '''
        if name == 'main':  # Main section, without the subtrees
            indices = neuron.morphology.indices[:]
            start, stop = indices[0], indices[-1]
            return SpatialSubgroup(neuron, start, stop + 1,
                                   morphology=neuron.morphology)
        elif (name != 'morphology') and ((name in getattr(neuron.morphology, 'children', [])) or
                                      all([c in 'LR123456789' for c in name])):  # subtree
            morpho = neuron.morphology[name]
            start = morpho.indices[0]
            stop = SpatialNeuron._find_subtree_end(morpho)
            return SpatialSubgroup(neuron, start, stop + 1, morphology=morpho)
        else:
            return Group.__getattr__(neuron, name)

    @staticmethod
    def spatialneuron_segment(neuron, item):
        '''
        Selects a segment from `SpatialNeuron` neuron, where item is a slice of
        either compartment indexes or distances.
        Note a: segment is not a `SpatialNeuron`, only a `Group`.
        '''
        if not isinstance(item, slice):
            raise TypeError(
                'Subgroups can only be constructed using slicing syntax')
        start, stop, step = item.start, item.stop, item.step
        if step is None:
            step = 1
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')

        if isinstance(start, Quantity):
            if not have_same_dimensions(start, meter) or not have_same_dimensions(stop, meter):
                raise DimensionMismatchError('Start and stop should have units of meter', start, stop)
            # Convert to integers (compartment numbers)
            indices = neuron.morphology.indices[item]
            start, stop = indices[0], indices[-1] + 1

        if start >= stop:
            raise IndexError('Illegal start/end values for subgroup, %d>=%d' %
                             (start, stop))

        return Subgroup(neuron, start, stop)


class SpatialSubgroup(Subgroup):
    '''
    A subgroup of a `SpatialNeuron`.

    Parameters
    ----------
    source : int
        First compartment.
    stop : int
        Ending compartment, not included (as in slices).
    morphology : `Morphology`
        Morphology corresponding to the subgroup (not the full
        morphology).
    name : str, optional
        Name of the subgroup.
    '''

    def __init__(self, source, start, stop, morphology, name=None):
        self.morphology = morphology
        Subgroup.__init__(self, source, start, stop, name)

    def __getattr__(self, name):
        return SpatialNeuron.spatialneuron_attribute(self, name)

    def __getitem__(self, item):
        return SpatialNeuron.spatialneuron_segment(self, item)


class SpatialStateUpdater(CodeRunner, Group):
    '''
    The `CodeRunner` that updates the state variables of a `SpatialNeuron`
    at every timestep.
    '''

    def __init__(self, group, method, clock, order=0):
        # group is the neuron (a group of compartments)
        self.method_choice = method
        self.group = weakref.proxy(group)

        compartments = group.flat_morphology.n
        sections = group.flat_morphology.sections

        CodeRunner.__init__(self, group,
                            'spatialstateupdate',
                            code='''_gtot = gtot__private
                                    _I0 = I0__private''',
                            clock=clock,
                            when='groups',
                            order=order,
                            name=group.name + '_spatialstateupdater*',
                            check_units=False,
                            template_kwds={'number_sections': sections})

        self.variables = Variables(self, default_index='_section_idx')
        self.variables.add_reference('N', group)
        # One value per compartment
        self.variables.add_arange('_compartment_idx', size=compartments)
        self.variables.add_array('_invr', dimensions=siemens.dim,
                                 size=compartments, constant=True,
                                 index='_compartment_idx')
        # one value per section
        self.variables.add_arange('_section_idx', size=sections)
        self.variables.add_array('_P_parent', size=sections,
                                 constant=True)  # elements below diagonal
        self.variables.add_arrays(['_morph_idxchild', '_morph_parent_i',
                                   '_starts', '_ends'], size=sections,
                                  dtype=np.int32, constant=True)
        self.variables.add_arrays(['_invr0', '_invrn'], dimensions=siemens.dim,
                                  size=sections, constant=True)
        # one value per section + 1 value for the root
        self.variables.add_arange('_section_root_idx', size=sections+1)
        self.variables.add_array('_P_diag', size=sections+1,
                                 constant=True, index='_section_root_idx')
        self.variables.add_array('_B', size=sections+1,
                                 constant=True, index='_section_root_idx')
        self.variables.add_array('_morph_children_num',
                                 size=sections+1, dtype=np.int32,
                                 constant=True, index='_section_root_idx')
        # 2D matrices of size (sections + 1) x max children per section
        self.variables.add_arange('_morph_children_idx',
                                  size=len(group.flat_morphology.morph_children))
        self.variables.add_array('_P_children',
                                 size=len(group.flat_morphology.morph_children),
                                 index='_morph_children_idx',
                                 constant=True)  # elements above diagonal
        self.variables.add_array('_morph_children',
                                 size=len(group.flat_morphology.morph_children),
                                 dtype=np.int32, constant=True,
                                 index='_morph_children_idx')
        self._enable_group_attributes()

        self._morph_parent_i = group.flat_morphology.morph_parent_i
        self._morph_children_num = group.flat_morphology.morph_children_num
        self._morph_children = group.flat_morphology.morph_children
        self._morph_idxchild = group.flat_morphology.morph_idxchild
        self._starts = group.flat_morphology.starts
        self._ends = group.flat_morphology.ends
        self._prepare_codeobj = None

    def before_run(self, run_namespace):
        # execute code to initalize the data structures
        if self._prepare_codeobj is None:
            self._prepare_codeobj = create_runner_codeobj(self.group,
                                                          '', # no code,
                                                          'spatialneuron_prepare',
                                                          name=self.name+'_spatialneuron_prepare',
                                                          check_units=False,
                                                          additional_variables=self.variables,
                                                          run_namespace=run_namespace)
        self._prepare_codeobj()
        # Raise a warning if the slow pure numpy version is used
        # For simplicity, we check which CodeObject class the _prepare_codeobj
        # is using, this will be the same as the main state updater
        from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
        if isinstance(self._prepare_codeobj, NumpyCodeObject):
            # If numpy is used, raise a warning if scipy is not present
            try:
                import scipy
            except ImportError:
                logger.info(('SpatialNeuron will use numpy to do the numerical '
                             'integration -- this will be very slow. Either '
                             'switch to a different code generation target '
                             '(e.g. weave or cython) or install scipy.'),
                            once=True)
        CodeRunner.before_run(self, run_namespace)
