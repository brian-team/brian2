'''
Compartmental models.
This module defines the SpatialNeuron class, which defines multicompartmental models.
'''
import weakref

import sympy as sp
import numpy as np

from brian2.core.variables import Variables
from brian2.equations.equations import (Equations, PARAMETER, SUBEXPRESSION,
                                        DIFFERENTIAL_EQUATION)
from brian2.groups.group import Group, CodeRunner, create_runner_codeobj
from brian2.units.allunits import ohm, siemens, amp, meter
from brian2.units.fundamentalunits import Unit, fail_for_dimension_mismatch
from brian2.units.stdunits import uF, cm
from brian2.parsing.sympytools import sympy_to_str, str_to_sympy
from brian2.utils.logger import get_logger
from brian2.groups.neurongroup import NeuronGroup
from brian2.groups.subgroup import Subgroup
from brian2.equations.codestrings import Expression

from .morphology import MorphologyData

__all__ = ['SpatialNeuron']

logger = get_logger(__name__)


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
        A dictionary mapping variable/function names to the respective objects.
        If no `namespace` is given, the "implicit" namespace, consisting of
        the local and global namespace surrounding the creation of the class,
        is used.
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
                 method=('linear', 'exponential_euler', 'rk2', 'heun')):

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
                           SUBEXPRESSION: ('shared', 'point current')})

        # Add the membrane potential
        model += Equations('''
        v:volt # membrane potential
        ''')

        # Extract membrane equation
        if 'Im' in model:
            membrane_eq = model['Im']  # the membrane equation
        else:
            raise TypeError('The transmembrane current Im must be defined')

        # Insert point currents in the membrane equation
        for eq in model.itervalues():
            if 'point current' in eq.flags:
                fail_for_dimension_mismatch(eq.unit, amp,
                                            "Point current " + eq.varname + " should be in amp")
                eq.flags.remove('point current')
                membrane_eq.expr = Expression(
                    str(membrane_eq.expr.code) + '+' + eq.varname + '/area')

        ###### Process model equations (Im) to extract total conductance and the remaining current
        # Check conditional linearity with respect to v
        # Match to _A*v+_B
        var = sp.Symbol('v', real=True)
        wildcard = sp.Wild('_A', exclude=[var])
        constant_wildcard = sp.Wild('_B', exclude=[var])
        pattern = wildcard * var + constant_wildcard

        # Expand expressions in the membrane equation
        membrane_eq.type = DIFFERENTIAL_EQUATION
        for var, expr in model.get_substituted_expressions():
            if var == 'Im':
                Im_expr = expr
        membrane_eq.type = SUBEXPRESSION

        # Factor out the variable
        s_expr = sp.collect(str_to_sympy(Im_expr.code).expand(), var)
        matches = s_expr.match(pattern)

        if matches is None:
            raise TypeError, "The membrane current must be linear with respect to v"
        a, b = (matches[wildcard],
                matches[constant_wildcard])

        # Extracts the total conductance from Im, and the remaining current
        minusa_str, b_str = sympy_to_str(-a), sympy_to_str(b)
        # Add correct units if necessary
        if minusa_str == '0':
            minusa_str += '*siemens/meter**2'
        if b_str == '0':
            b_str += '*amp/meter**2'
        gtot_str = "gtot__private=" + minusa_str + ": siemens/meter**2"
        I0_str = "I0__private=" + b_str + ": amp/meter**2"
        model += Equations(gtot_str + "\n" + I0_str)

        # Equations for morphology
        # TODO: check whether Cm and Ri are already in the equations
        #       no: should be shared instead of constant
        #       yes: should be constant (check)
        eqs_constants = Equations("""
        diameter : meter (constant)
        length : meter (constant)
        x : meter (constant)
        y : meter (constant)
        z : meter (constant)
        distance : meter (constant)
        area : meter**2 (constant)
        Cm : farad/meter**2 (constant)
        Ri : ohm*meter (constant, shared)
        space_constant = (diameter/(4*Ri*gtot__private))**.5 : meter
        """)
        # Insert morphology
        self.morphology = morphology

        # Link morphology variables to neuron's state variables
        self.morphology_data = MorphologyData(len(morphology))
        self.morphology.compress(self.morphology_data)

        NeuronGroup.__init__(self, len(morphology), model=model + eqs_constants,
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
                                   '_a_minus0', '_a_minus1', '_a_minus2',
                                   '_a_plus0', '_a_plus1', '_a_plus2',
                                   '_b_plus', '_b_minus',
                                   '_v_star', '_u_plus', '_u_minus',
                                   # The following three are for solving the
                                   # three tridiag systems in parallel
                                   '_c1', '_c2', '_c3',
                                   # The following two are only necessary for
                                   # C code where we cannot deal with scalars
                                   # and arrays interchangeably:
                                   '_I0_all', '_gtot_all'], unit=1,
                                  size=self.N, read_only=True)

        self.Cm = Cm
        self.Ri = Ri
        # TODO: View instead of copy for runtime?
        self.diameter_ = self.morphology_data.diameter
        self.distance_ = self.morphology_data.distance
        self.length_ = self.morphology_data.length
        self.area_ = self.morphology_data.area
        self.x_ = self.morphology_data.x
        self.y_ = self.morphology_data.y
        self.z_ = self.morphology_data.z

        # Performs numerical integration step
        self.add_attribute('diffusion_state_updater')
        self.diffusion_state_updater = SpatialStateUpdater(self, method,
                                                           clock=self.clock,
                                                           order=order)

        # Creation of contained_objects that do the work
        self.contained_objects.extend([self.diffusion_state_updater])

    def __getattr__(self, x):
        '''
        Subtrees are accessed by attribute, e.g. neuron.axon.
        '''
        return self.spatialneuron_attribute(self, x)

    def __getitem__(self, x):
        '''
        Selects a segment, where x is a slice of either compartment
        indexes or distances.
        Note a: segment is not a SpatialNeuron, only a Group.
        '''
        return self.spatialneuron_segment(self, x)

    @staticmethod
    def spatialneuron_attribute(neuron, x):
        '''
        Selects a subtree from `SpatialNeuron` neuron and returns a `SpatialSubgroup`.
        If it does not exist, returns the `Group` attribute.
        '''
        if x == 'main':  # Main segment, without the subtrees
            origin = neuron.morphology._origin
            return Subgroup(neuron, origin, origin + len(neuron.morphology.x))
        elif (x != 'morphology') and ((x in neuron.morphology._namedkid) or
                                      all([c in 'LR123456789' for c in x])):  # subtree
            morpho = neuron.morphology[x]
            return SpatialSubgroup(neuron, morpho._origin,
                                   morpho._origin + len(morpho),
                                   morphology=morpho)
        else:
            return Group.__getattr__(neuron, x)

    @staticmethod
    def spatialneuron_segment(neuron, x):
        '''
        Selects a segment from `SpatialNeuron` neuron, where x is a slice of
        either compartment indexes or distances.
        Note a: segment is not a `SpatialNeuron`, only a `Group`.
        '''
        if not isinstance(x, slice):
            raise TypeError(
                'Subgroups can only be constructed using slicing syntax')
        start, stop, step = x.start, x.stop, x.step
        if step is None:
            step = 1
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')

        if type(start) == type(1 * cm):  # e.g. 10*um:20*um
            # Convert to integers (compartment numbers)
            morpho = neuron.morphology[x]
            start = morpho._origin
            stop = morpho._origin + len(morpho)

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

    def __getattr__(self, x):
        return SpatialNeuron.spatialneuron_attribute(self, x)

    def __getitem__(self, x):
        return SpatialNeuron.spatialneuron_segment(self, x)


class SpatialStateUpdater(CodeRunner, Group):
    '''
    The `CodeRunner` that updates the state variables of a `SpatialNeuron`
    at every timestep.
    '''

    def __init__(self, group, method, clock, order=0):
        # group is the neuron (a group of compartments)
        self.method_choice = method
        self.group = weakref.proxy(group)

        compartments = len(group) # total number of compartments
        branches = self.number_branches(group.morphology)

        CodeRunner.__init__(self, group,
                            'spatialstateupdate',
                            code='''_gtot = gtot__private
                                    _I0 = I0__private''',
                            clock=clock,
                            when='groups',
                            order=order,
                            name=group.name + '_spatialstateupdater*',
                            check_units=False,
                            template_kwds={'number_branches': branches})

        # The morphology is considered fixed (length etc. can still be changed,
        # though)
        # Traverse it once to get a flattened representation
        self._temp_morph_i = np.zeros(branches, dtype=np.int32)
        self._temp_morph_parent_i = np.zeros(branches, dtype=np.int32)
        # for the following: a smaller array of size no_segments x max_no_children would suffice...
        self._temp_morph_children = np.zeros((branches+1, branches), dtype=np.int32)
        # children count per branch: determines the no of actually used elements of the array above
        self._temp_morph_children_num = np.zeros(branches+1, dtype=np.int32)
        # each branch is child of exactly one parent (and we say the first branch i=1 is child of branch i=0)
        # here we store the indices j-1->k of morph_children_i[i,k] = j 
        self._temp_morph_idxchild = np.zeros(branches, dtype=np.int32)
        self._temp_starts = np.zeros(branches, dtype=np.int32)
        self._temp_ends = np.zeros(branches, dtype=np.int32)
        self._pre_calc_iteration(self.group.morphology)
        # flattened and reduce children indices
        max_children = max(self._temp_morph_children_num)
        self._temp_morph_children = self._temp_morph_children[:,:max_children].reshape(-1)
        
        self.variables = Variables(self, default_index='_branch_idx')
        self.variables.add_reference('N', group)
        # One value per compartment
        self.variables.add_arange('_compartment_idx', size=compartments)
        self.variables.add_array('_invr', unit=siemens, size=compartments,
                                 constant=True, index='_compartment_idx')
        # one value per branch
        self.variables.add_arange('_branch_idx', size=branches)
        self.variables.add_array('_P_parent', unit=Unit(1), size=branches,
                                 constant=True) # elements below diagonal
        self.variables.add_array('_morph_idxchild', unit=Unit(1), size=branches,
                                 dtype=np.int32, constant=True)
        self.variables.add_arrays(['_morph_i', '_morph_parent_i',
                                   '_starts', '_ends'], unit=Unit(1),
                                  size=branches, dtype=np.int32, constant=True)
        self.variables.add_arrays(['_invr0', '_invrn'], unit=siemens,
                                  size=branches, constant=True)
        # one value per branch + 1 value for the root
        self.variables.add_arange('_branch_root_idx', size=branches+1)
        self.variables.add_array('_P_diag', unit=Unit(1), size=branches+1,
                                 constant=True, index='_branch_root_idx')
        self.variables.add_array('_B', unit=Unit(1), size=branches+1,
                                 constant=True, index='_branch_root_idx')
        self.variables.add_arange('_morph_children_num_idx', size=branches+1)
        self.variables.add_array('_morph_children_num', unit=Unit(1),
                                 size=branches+1, dtype=np.int32, constant=True,
                                 index='_morph_children_num_idx')
        # 2D matrices of size (branches + 1) x max children per branch
        # Note that this data structure wastes space if the number of children
        # per branch is very different. In practice, however, this should not
        # matter much, since branches will normally have 0, 1, or 2 children
        # (e.g. SWC files on neuromporh are strictly binary trees)
        self.variables.add_array('_P_children', unit=Unit(1),
                                 size=(branches+1)*max_children,
                                 constant=True)  # elements above diagonal
        self.variables.add_arange('_morph_children_idx',
                                  size=(branches+1)*max_children)
        self.variables.add_array('_morph_children', unit=Unit(1),
                                 size=(branches+1)*max_children,
                                 dtype=np.int32, constant=True,
                                 index='_morph_children_idx')
        self._enable_group_attributes()
        
        self._morph_i = self._temp_morph_i
        self._morph_parent_i = self._temp_morph_parent_i
        self._morph_children_num = self._temp_morph_children_num
        self._morph_children = self._temp_morph_children
        self._morph_idxchild = self._temp_morph_idxchild
        self._starts = self._temp_starts
        self._ends = self._temp_ends
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

    def _pre_calc_iteration(self, morphology, counter=0):
        self._temp_morph_i[counter] = morphology.index + 1
        self._temp_morph_parent_i[counter] = morphology.parent + 1
        
        # add to parent's children list
        if counter>0:
            parent_i = self._temp_morph_parent_i[counter]
            child_num = self._temp_morph_children_num[parent_i]
            self._temp_morph_children[parent_i, child_num] = counter+1
            self._temp_morph_children_num[parent_i] += 1 # increase parent's children count
            self._temp_morph_idxchild[counter] = child_num
        else:
            self._temp_morph_children_num[0] = 1
            self._temp_morph_children[0, 0] = 1
            self._temp_morph_idxchild[0] = 0
        
        self._temp_starts[counter] = morphology._origin
        self._temp_ends[counter] = morphology._origin + len(morphology.x) - 1
        total_count = 1
        for child in morphology.children:
            total_count += self._pre_calc_iteration(child, counter+total_count)
        return total_count

    def number_branches(self, morphology, n=0, parent=-1):
        '''
        Recursively number the branches and return their total number.
        n is the index number of the current branch.
        parent is the index number of the parent branch.
        '''
        morphology.index = n
        morphology.parent = parent
        nbranches = 1
        for kid in (morphology.children):
            nbranches += self.number_branches(kid, n + nbranches, n)
        return nbranches
