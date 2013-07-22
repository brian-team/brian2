'''
Spatial neuron!
'''
from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        STATIC_EQUATION, PARAMETER)
from brian2.memory import allocate_array
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.core.preferences import brian_prefs
from brian2.groups.group import Group, GroupCodeRunner
from brian2.core.base import BrianObject
from brian2.units.allunits import ohm
from brian2.units.fundamentalunits import Unit
from brian2.units.stdunits import uF,cm
from brian2.codegen.parsing import sympy_to_str
from brian2.utils.logger import get_logger
from brian2.core.namespace import create_namespace
from brian2.core.specifiers import (ReadOnlyValue, AttributeValue, ArrayVariable,
                                    StochasticVariable, Subexpression, Index)
import sympy as sp
from numpy import ones
import numpy as np

__all__ = ['SpatialNeuron']

logger = get_logger(__name__)

class SpatialNeuron(Group,BrianObject):
    def __init__(self, morphology=None, model=None, clock=None, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm,
                 name='spatialneuron*', dtype=None, namespace=None, method=None):
        BrianObject.__init__(self, when=clock, name=name)

        self.N = len(morphology) # number of compartments

        ##### Prepare and validate equations
        if isinstance(model, basestring):
            model = Equations(model)
        if not isinstance(model, Equations):
            raise TypeError(('model has to be a string or an Equations '
                             'object, is "%s" instead.') % type(model))

        # Check flags
        model.check_flags({PARAMETER: ('constant')})

        model += Equations('''
        v:volt # membrane potential
        ''')

        # Process model equations (Im) to extract total conductance and the remaining current
        if 'Im' in model:
            membrane_eq=model['Im'] # the membrane equation
        else:
            raise TypeError,"The transmembrane current Im must be defined"
        # Check conditional linearity
        # Match to _A*v+_B
        var = sp.Symbol('v', real=True)        
        wildcard = sp.Wild('_A', exclude=[var])
        constant_wildcard = sp.Wild('_B', exclude=[var])
        pattern = wildcard*var + constant_wildcard   
        # Factor out the variable
        s_expr = sp.collect(membrane_eq.expr.sympy_expr.expand(), var)
        matches = s_expr.match(pattern)
        
        if matches is None:
            raise TypeError,"The membrane current must be linear with respect to v"
        a,b = (matches[wildcard].simplify(),
                        matches[constant_wildcard].simplify())
        
        # Extracts the total conductance from Im, and the remaining current
        gtot_str="gtot__="+sympy_to_str(-a)+": siemens/meter**2"
        I0_str="I0__="+sympy_to_str(b)+": amp/meter**2"
        model+=Equations(gtot_str+"\n"+I0_str)

        # Equations for morphology (isn't it a duplicate??)
        eqs_morphology = Equations("""
        diameter : meter (constant)
        length : meter (constant)
        x : meter (constant)
        y : meter (constant)
        z : meter (constant)
        area : meter**2 (constant)
        """)

        self.equations = model + eqs_morphology
        
        # Create the state updater
        self.Cm = ones(self.N)*Cm #  Temporary hack - so that it can be a vector, later
        self.Ri = Ri
        #self._state_updater = SpatialStateUpdater(self, clock)

        ##### Setup the memory
        self.arrays = self._allocate_memory(dtype=dtype)
        
        # Setup the namespace
        self.namespace = create_namespace(namespace)

        # Setup specifiers
        self.specifiers = self._create_specifiers()

        # Activate name attribute access
        Group.__init__(self)

        # Insert morphology
        self.morphology = morphology
        self.morphology.compress(diameter=self.diameter, length=self.length, x=self.x, y=self.y, z=self.z, area=self.area)

        #: The state update method selected by the user
        self.method_choice = method
        
        #: Performs numerical integration step
        self.state_updater = SpatialStateUpdater(self, method)

        # Creation of contained_objects that do the work
        self.contained_objects.append(self.state_updater)

        # We try to run a pre_run already now. This might fail because of an
        # incomplete namespace but if the namespace is already complete we
        # can spot unit or syntax errors already here, at creation time.
        try:
            self.pre_run(None)
        except KeyError:
            pass

    def __len__(self):
        '''
        Return number of neurons in the group.
        '''
        return self.N

    def _allocate_memory(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)

        arrays = {}
        for eq in self.equations.itervalues():
            if eq.type == STATIC_EQUATION:
                # nothing to do
                continue
            name = eq.varname
            if isinstance(dtype, dict):
                curdtype = dtype[name]
            else:
                curdtype = dtype
            if curdtype is None:
                curdtype = brian_prefs['core.default_scalar_dtype']
            if eq.is_bool:
                arrays[name] = allocate_array(self.N, dtype=np.bool)
            else:
                arrays[name] = allocate_array(self.N, dtype=curdtype)
        logger.debug("NeuronGroup memory allocated successfully.")
        return arrays

    def _create_specifiers(self):
        '''
        Create the specifiers dictionary for this `SpatialNeuron`, containing
        entries for the equation variables and some standard entries.
        '''
        # Get the standard specifiers for all groups
        s = Group._create_specifiers(self)

        # Standard specifiers always present
        s.update({'_num_neurons': ReadOnlyValue('_num_neurons', Unit(1),
                                                np.int, self.N)})

        # First add all the differential equations and parameters, because they
        # may be referred to by static equations
        for eq in self.equations.itervalues():
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                array = self.arrays[eq.varname]
                constant = ('constant' in eq.flags)
                s.update({eq.varname: ArrayVariable(eq.varname,
                                                    eq.unit,
                                                    array.dtype,
                                                    array,
                                                    '_neuron_idx',
                                                    self,                                                    
                                                    constant=constant,
                                                    is_bool=eq.is_bool)})
        
            elif eq.type == STATIC_EQUATION:
                s.update({eq.varname: Subexpression(eq.varname, eq.unit,
                                                    brian_prefs['core.default_scalar_dtype'],
                                                    str(eq.expr),
                                                    s,
                                                    self.namespace,
                                                    is_bool=eq.is_bool)})
            else:
                raise AssertionError('Unknown type of equation: ' + eq.eq_type)

        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            s.update({xi: StochasticVariable(xi)})

        return s

    def pre_run(self, namespace):
    
        # Update the namespace information in the specifiers in case the
        # namespace was not specified explicitly defined at creation time
        # Note that values in the explicit namespace might still change
        # between runs, but the Subexpression stores a reference to 
        # self.namespace so these changes are taken into account automatically
        if not self.namespace.is_explicit:
            for spec in self.specifiers.itervalues():
                if isinstance(spec, Subexpression):
                    spec.additional_namespace = namespace

        # Check units
        self.equations.check_units(self.namespace, self.specifiers,
                                   namespace)

class SpatialStateUpdater(GroupCodeRunner):
    '''
    The `GroupCodeRunner` that updates the state variables of a `NeuronGroup`
    at every timestep.
    '''
    def __init__(self, group, method):
        self.method_choice = method
        indices = {'_neuron_idx': Index('_neuron_idx', True)}
        
        GroupCodeRunner.__init__(self, group,
                                       'stateupdate',
                                       indices=indices,
                                       when=(group.clock, 'groups'),
                                       name=group.name + '_stateupdater*',
                                       check_units=False)

        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.specifiers,
                                                               method)
    
    def update_abstract_code(self):
        self.method = StateUpdateMethod.determine_stateupdater(self.group.equations,
                                                               self.group.specifiers,
                                                               self.method_choice)
        self.abstract_code = self.method(self.group.equations,
                                         self.group.specifiers)

    def post_update(self, return_value):
        pass