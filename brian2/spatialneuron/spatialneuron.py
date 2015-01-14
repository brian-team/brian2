'''
Compartmental models.
This module defines the SpatialNeuron class, which defines multicompartmental models.
'''
from itertools import izip

import sympy as sp
from numpy import zeros, pi
from numpy.linalg import solve
import numpy as np

from brian2.core.variables import Variables
from brian2.equations.equations import (Equations, PARAMETER, SUBEXPRESSION,
                                        DIFFERENTIAL_EQUATION)
from brian2.groups.group import Group, CodeRunner
from brian2.units.allunits import ohm, siemens, amp
from brian2.units.fundamentalunits import Unit, fail_for_dimension_mismatch
from brian2.units.stdunits import uF, cm
from brian2.parsing.sympytools import sympy_to_str
from brian2.utils.logger import get_logger
from brian2.groups.neurongroup import NeuronGroup
from brian2.groups.subgroup import Subgroup
from brian2.equations.codestrings import Expression

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
                 refractory=False, reset=None,
                 threshold_location=None,
                 dt=None, clock=None, order=0, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm,
                 name='spatialneuron*', dtype=None, namespace=None,
                 method=None):

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
        for var, expr in model._get_substituted_expressions():  # this returns substituted expressions for diff eqs
            if var == 'Im':
                Im_expr = expr
        membrane_eq.type = SUBEXPRESSION

        # Factor out the variable
        s_expr = sp.collect(Im_expr.sympy_expr.expand(), var)
        matches = s_expr.match(pattern)

        if matches is None:
            raise TypeError, "The membrane current must be linear with respect to v"
        a, b = (matches[wildcard].simplify(),
                matches[constant_wildcard].simplify())

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
        space_constant = (diameter/(4*Ri*gtot__private))**.5 : meter # Not so sure about the name

        ### Parameters and intermediate variables for solving the cable equation
        ab_star0 : siemens/meter**2
        ab_plus0 : siemens/meter**2
        ab_minus0 : siemens/meter**2
        ab_star1 : siemens/meter**2
        ab_plus1 : siemens/meter**2
        ab_minus1 : siemens/meter**2
        ab_star2 : siemens/meter**2
        ab_plus2 : siemens/meter**2
        ab_minus2 : siemens/meter**2
        b_plus : siemens/meter**2
        b_minus : siemens/meter**2
        v_star : volt
        u_plus : 1
        u_minus : 1
        """)
        # Possibilities for the name: characteristic_length, electrotonic_length, length_constant, space_constant

        NeuronGroup.__init__(self, len(morphology), model=model + eqs_constants,
                             threshold=threshold, refractory=refractory,
                             reset=reset,
                             method=method, dt=dt, clock=clock, order=order,
                             namespace=namespace, dtype=dtype, name=name)

        self.Cm = Cm
        self.Ri = Ri

        # Insert morphology
        self.morphology = morphology
        # Link morphology variables to neuron's state variables
        self.morphology.compress(
            diameter=self.variables['diameter'].get_value(),
            length=self.variables['length'].get_value(),
            x=self.variables['x'].get_value(),
            y=self.variables['y'].get_value(),
            z=self.variables['z'].get_value(),
            area=self.variables['area'].get_value(),
            distance=self.variables['distance'].get_value())

        # Performs numerical integration step
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
        Subgroup.__init__(self, source, start, stop, name)
        self.morphology = morphology

    def __getattr__(self, x):
        return SpatialNeuron.spatialneuron_attribute(self, x)

    def __getitem__(self, x):
        return SpatialNeuron.spatialneuron_segment(self, x)


class SpatialStateUpdater(CodeRunner, Group):
    '''
    The `CodeRunner` that updates the state variables of a `SpatialNeuron`
    at every timestep.

    TODO: all internal variables (u_minus etc) could be inserted in the SpatialNeuron.
    '''

    def __init__(self, group, method, clock, order=0):
        # group is the neuron (a group of compartments) 
        self.method_choice = method
        self._isprepared = False
        CodeRunner.__init__(self, group,
                            'spatialstateupdate',
                            code='''_gtot = gtot__private
                                    _I0 = I0__private''',
                            clock=clock,
                            when='groups',
                            order=order,
                            name=group.name + '_spatialstateupdater*',
                            check_units=False)
        n = len(group) # total number of compartments
        segments = self.number_branches(group.morphology)
        self.variables = Variables(self, default_index='_segment_idx')
        self.variables.add_reference('N', group)
        self.variables.add_arange('_compartment_idx', size=n)
        self.variables.add_arange('_segment_idx', size=segments)
        self.variables.add_arange('_segment_root_idx', size=segments+1)
        self.variables.add_arange('_P_idx', size=(segments+1)**2)

        self.variables.add_array('_invr', unit=siemens, size=n, constant=True,
                                 index='_compartment_idx')
        self.variables.add_array('_P', unit=Unit(1), size=(segments+1)**2,
                                 constant=True, index='_P_idx')
        self.variables.add_array('_B', unit=Unit(1), size=segments+1,
                                 constant=True, index='_segment_root_idx')
        self.variables.add_array('_V', unit=Unit(1), size=segments+1,
                                 constant=True, index='_segment_root_idx')
        self.variables.add_array('_morph_i', unit=Unit(1), size=segments,
                                 dtype=np.int32, constant=True)
        self.variables.add_array('_morph_parent_i', unit=Unit(1), size=segments,
                                 dtype=np.int32, constant=True)
        self.variables.add_array('_starts', unit=Unit(1), size=segments,
                                 dtype=np.int32, constant=True)
        self.variables.add_array('_ends', unit=Unit(1), size=segments,
                                 dtype=np.int32, constant=True)
        self.variables.add_array('_invr0', unit=siemens, size=segments,
                                 constant=True)
        self.variables.add_array('_invrn', unit=siemens, size=segments,
                                 constant=True)
        self._enable_group_attributes()

        # A 2d view on P for convenience
        self._P_2d = self.variables['_P'].get_value().reshape((segments + 1,
                                                               segments + 1))

    def before_run(self, run_namespace=None, level=0):
        if not self._isprepared:  # this is done only once even if there are multiple runs
            self.prepare()
            self._isprepared = True
        CodeRunner.before_run(self, run_namespace, level=level + 1)

    def run(self):
        CodeRunner.run(self)
        # Solve the linear system connecting branches
        self._P[:] = 0
        self._B[:] = 0

        self.fill_matrix()
        self._V = solve(self._P_2d, self._B)  # This code could be generated at initialization
        # Calculate solutions by linear combination
        self.linear_combination()

    def prepare(self):
        '''
        Preparation of data structures.
        See the relevant document.
        '''
        # Correction for soma (a bit of a hack), so that it has negligible axial resistance
        if self.group.morphology.type == 'soma':
            self.group.length[0] = self.group.diameter[0] * 0.01
        # Inverse axial resistance
        self._invr[1:] = (pi / (2 * self.group.Ri) * (self.group.diameter[:-1] *
                                                      self.group.diameter[1:]) /
                          (self.group.length[:-1] + self.group.length[1:]))
        # Note: this would give nan for the soma
        self.cut_branches(self.group.morphology)

        # Linear systems
        # The particular solution
        '''a[i,j]=ab[u+i-j,j]'''  # u is the number of upper diagonals = 1
        self.group.ab_star0[1:] = self._invr[1:] / self.group.area[:-1]
        self.group.ab_star2[:-1] = self._invr[1:] / self.group.area[1:]
        self.group.ab_star1[:] = (-(self.group.Cm / self.group.clock.dt) -
                                  self._invr / self.group.area)
        self.group.ab_star1[:-1] -= self._invr[1:] / self.group.area[:-1]
        # Homogeneous solutions
        self.group.ab_plus0[:] = self.group.ab_star0
        self.group.ab_minus0[:] = self.group.ab_star0
        self.group.ab_plus1[:] = self.group.ab_star1
        self.group.ab_minus1[:] = self.group.ab_star1
        self.group.ab_plus2[:] = self.group.ab_star2
        self.group.ab_minus2[:] = self.group.ab_star2

        # Boundary conditions
        self.boundary_conditions(self.group.morphology)

        # Pre-calculate the iteration over the segments (to replace the
        # recursion during runtime)

        self.pre_calc_iteration(self.group.morphology)

    def pre_calc_iteration(self, morphology, counter=0):
        self._morph_i[counter] = morphology.index + 1
        self._morph_parent_i[counter] = morphology.parent + 1
        self._starts[counter] = morphology._origin
        self._ends[counter] = morphology._origin + len(morphology.x) - 1
        self._invr0[counter] = morphology.invr0
        self._invrn[counter] = morphology.invrn
        for child in morphology.children:
            counter += 1
            self.pre_calc_iteration(child, counter)

    def cut_branches(self, morphology):
        '''
        Recursively cut the branches by setting zero axial resistances.
        '''
        self._invr[morphology._origin] = 0
        for kid in (morphology.children):
            self.cut_branches(kid)

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

    def boundary_conditions(self, morphology):
        '''
        Recursively sets the boundary conditions in the linear systems.
        '''
        first = morphology._origin  # first compartment
        last = first + len(morphology.x) - 1  # last compartment
        # Inverse axial resistances at the ends: r0 and rn
        morphology.invr0 = (pi / (2 * self.group.Ri) *
                                 self.group.diameter[first] ** 2 /
                                 self.group.length[first])
        morphology.invrn = (pi / (2 * self.group.Ri) *
                                 self.group.diameter[last] ** 2 /
                                 self.group.length[last])
        # Correction for boundary conditions
        self.group.ab_star1[first] -= (morphology.invr0 / self.group.area[first])  # because of units problems
        self.group.ab_star1[last] -= (morphology.invrn / self.group.area[last])
        self.group.ab_plus1[first] -= (morphology.invr0 / self.group.area[first])  # because of units problems
        self.group.ab_plus1[last] -= (morphology.invrn / self.group.area[last])
        self.group.ab_minus1[first] -= (morphology.invr0 / self.group.area[first])  # because of units problems
        self.group.ab_minus1[last] -= (morphology.invrn / self.group.area[last])
        # RHS for homogeneous solutions
        self.group.b_plus[last] = -(morphology.invrn / self.group.area[last])
        self.group.b_minus[first] = -(morphology.invr0 / self.group.area[first])
        # Recursive call
        for kid in (morphology.children):
            self.boundary_conditions(kid)

    # ### The two methods below should be written in C
    #### In each one there is a static function, plus a call for each segment
    #### Code for the latter could be generated at initialization
    def linear_combination(self):
        '''
        Calculates solutions by linear combination
        '''
        # Directly access the underlying arrays (as in a template) for better
        # performance
        v_star = self.group.variables['v_star'].get_value()
        u_minus = self.group.variables['u_minus'].get_value()
        u_plus = self.group.variables['u_plus'].get_value()
        for i, i_parent, first, last in izip(self.variables['_morph_i'].get_value(),
                                             self.variables['_morph_parent_i'].get_value(),
                                             self.variables['_starts'].get_value(),
                                             self.variables['_ends'].get_value()):
            self.group.v_[first:last + 1] = (v_star[first:last + 1]
                                             + self._V_[i_parent] * u_minus[first:last + 1]
                                             + self._V_[i] * u_plus[first:last + 1])

    def fill_matrix(self):
        '''
        Fills the matrix of the linear system that connects branches together.
        '''
        # Directly access the underlying arrays (as in a template) for better
        # performance
        v_star = self.group.variables['v_star'].get_value()
        u_minus = self.group.variables['u_minus'].get_value()
        u_plus = self.group.variables['u_plus'].get_value()
        for i, i_parent, first, last, invr0, invrn in izip(self.variables['_morph_i'].get_value(),
                                                           self.variables['_morph_parent_i'].get_value(),
                                                           self.variables['_starts'].get_value(),
                                                           self.variables['_ends'].get_value(),
                                                           self.variables['_invr0'].get_value(),
                                                           self.variables['_invrn'].get_value()):
            # Towards parent
            if i == 1:  # first branch, sealed end
                self._P_2d[0, 0] = u_minus[first] - 1
                self._P_2d[0, 1] = u_plus[first]
                self._B_[0] = -v_star[first]
            else:
                self._P_2d[i_parent, i_parent] += (1 - u_minus[first]) * invr0
                self._P_2d[i_parent, i] -= u_plus[first] * invr0
                self._B_[i_parent] += v_star[first] * invr0
            # Towards children
            self._P_2d[i, i] = (1 - u_plus[last]) * invrn
            self._P_2d[i, i_parent] = -u_minus[last] * invrn
            self._B_[i] = v_star[last] * invrn
