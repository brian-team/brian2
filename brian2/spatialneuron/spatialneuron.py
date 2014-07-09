'''
Compartmental models

TODO:
* subgroups (neuron.axon etc)
* access with metric indexes
* faster (C?)
* clean up
* point processes
'''
from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        SUBEXPRESSION, PARAMETER)
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.core.preferences import brian_prefs
from brian2.groups.group import Group, CodeRunner
from brian2.groups.neurongroup import StateUpdater
from brian2.core.base import BrianObject
from brian2.units.allunits import ohm,siemens
from brian2.units.fundamentalunits import Unit
from brian2.units.stdunits import uF,cm
from brian2.parsing.sympytools import sympy_to_str
from brian2.utils.logger import get_logger
from brian2.core.variables import (AttributeVariable, ArrayVariable,
                                   Subexpression)
from brian2.core.variables import Variables, LinkedVariable, DynamicArrayVariable
from scipy.linalg import solve_banded
from brian2.groups.neurongroup import NeuronGroup
import sympy as sp
from numpy import zeros, ones, pi
from numpy.linalg import solve
import numpy as np

__all__ = ['SpatialNeuron']

logger = get_logger(__name__)

class SpatialNeuron(NeuronGroup):
    def __init__(self, morphology=None, model=None, clock=None, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm,
                 name='spatialneuron*', dtype=None, namespace=None, method=None):

        #self.N = len(morphology) # number of compartments

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
        gtot_str="gtot__private="+sympy_to_str(-a)+": siemens/meter**2"
        I0_str="I0__private="+sympy_to_str(b)+": amp/meter**2"
        model+=Equations(gtot_str+"\n"+I0_str)

        # Equations for morphology (isn't it a duplicate??)
        eqs_constants = Equations("""
        diameter : meter (constant)
        length : meter (constant)
        x : meter (constant)
        y : meter (constant)
        z : meter (constant)
        area : meter**2 (constant)
        Cm : farad/meter**2 (constant)
        """)

        NeuronGroup.__init__(self,len(morphology),model=model + eqs_constants,method=method,clock=clock,
                             namespace=namespace,dtype=dtype,name=name)

        self.Cm = Cm
        self.Ri = Ri # this could also be a state variable

        # Insert morphology
        self.morphology = morphology
        self.morphology.compress(diameter=self.diameter, length=self.length, x=self.x, y=self.y, z=self.z, area=self.area)

        #: Performs numerical integration step
        self.diffusion_state_updater = SpatialStateUpdater(self, method)

        # Creation of contained_objects that do the work
        self.contained_objects.extend([self.diffusion_state_updater])

class SpatialStateUpdater(CodeRunner,Group):
    '''
    The `CodeRunner` that updates the state variables of a `SpatialGroup`
    at every timestep.
    '''
    def __init__(self, group, method):
        # group is the neuron (a group of compartments) 
        self.method_choice = method
        self._isprepared=False
        CodeRunner.__init__(self, group,
                                       'spatialstateupdate',
                                       when=(group.clock, 'groups', 1),
                                       name=group.name + '_spatialstateupdater*',
                                       check_units=False)
        self.abstract_code='''
        _gtot = gtot__private
        _I0 = I0__private
        '''
        N=len(self.group)
        self.ab_star=zeros((3,N))
        self.ab_plus=zeros((3,N))
        self.ab_minus=zeros((3,N))
        self.b_plus=zeros(N)
        self.b_minus=zeros(N)
        self.v_star=zeros(N)
        self.u_plus=zeros(N)
        self.u_minus=zeros(N)
        self.variables=Variables(self)
        # These 5 variables are constant after prepare()
        self.variables.add_array('ab_star',Unit(1),3*N,values=self.ab_star.flatten(),dtype=self.ab_star.dtype)
        self.variables.add_array('ab_plus',Unit(1),3*N,values=self.ab_plus.flatten(),dtype=self.ab_plus.dtype)
        self.variables.add_array('ab_minus',Unit(1),3*N,values=self.ab_minus.flatten(),dtype=self.ab_minus.dtype)
        self.variables.add_array('b_plus',Unit(1),N,values=self.b_plus,dtype=self.b_plus.dtype)
        self.variables.add_array('b_minus',Unit(1),N,values=self.b_minus,dtype=self.b_minus.dtype)
        # These 3 variables change every time step
        self.variables.add_array('v_star',Unit(1),N,values=self.v_star,dtype=self.v_star.dtype)
        self.variables.add_array('u_plus',Unit(1),N,values=self.u_plus,dtype=self.u_plus.dtype)
        self.variables.add_array('u_minus',Unit(1),N,values=self.u_minus,dtype=self.u_minus.dtype)

    def before_run(self, run_namespace = None, level=0):
        if not self._isprepared: # this is done only once even if there are multiple runs
            self.prepare()
            self._isprepared=True
        CodeRunner.before_run(self, run_namespace, level=level+1)
        
    def after_update(self):
        # *** Apparently this is never called !!! ***
        # Solve the linear system connecting branches
        self.P[:]=0
        self.B[:]=0
        self.fill_matrix(self.group.morphology)
        self.V = solve(self.P,self.B)
        # Calculate solutions by linear combination
        self.linear_combination(self.group.morphology)
        
    def prepare(self):
        '''
        Preparation of data structures.
        See the relevant document.
        '''
        # Correction for soma (a bit of a hack), so that it has negligible axial resistance
        if self.group.morphology.type=='soma':
            self.group.length[0]=self.group.diameter[0]*0.01
        # Inverse axial resistance
        self.invr=zeros(len(self.group))*siemens
        self.invr[1:]=pi/(2*self.group.Ri)*(self.group.diameter[:-1]*self.group.diameter[1:])/\
                   (self.group.length[:-1]+self.group.length[1:])
        # Note: this would give nan for the soma
        self.cut_branches(self.group.morphology)
        
        # Linear systems
        # The particular solution
        '''a[i,j]=ab[u+i-j,j]''' # u is the number of upper diagonals = 1
        self.ab_star[0,1:]=self.invr[1:]/self.group.area[:-1]
        self.ab_star[2,:-1]=self.invr[1:]/self.group.area[1:]
        self.ab_star[1,:]=-(self.group.Cm/self.group.clock.dt)-self.invr/self.group.area
        self.ab_star[1,:-1]-=self.invr[1:]/self.group.area[:-1]
        # Homogeneous solutions
        self.ab_plus[:]=self.ab_star
        self.ab_minus[:]=self.ab_star
        
        # Boundary conditions
        self.boundary_conditions(self.group.morphology)
        
        # Linear system for connecting branches
        n=1+self.number_branches(self.group.morphology) # number of nodes (2 for the root)
        self.P=zeros((n,n)) # matrix
        self.B=zeros(n) # vector RHS
        self.V=zeros(n) # solution = voltages at nodes

        self.variables['ab_star'].set_value(self.ab_star.flatten())
        self.variables['ab_plus'].set_value(self.ab_plus.flatten())
        self.variables['ab_minus'].set_value(self.ab_minus.flatten())
        self.variables['b_plus'].set_value(self.b_plus)
        self.variables['b_minus'].set_value(self.b_minus)

    def cut_branches(self,morphology):
        '''
        Recursively cut the branches by setting zero axial resistances.
        '''
        self.invr[morphology._origin]=0
        for kid in (morphology.children):
            self.cut_branches(kid)
    
    def number_branches(self,morphology,n=0,parent=-1):
        '''
        Recursively number the branches and return their total number.
        n is the index number of the current branch.
        parent is the index number of the parent branch.
        '''
        morphology.index=n
        morphology.parent=parent
        nbranches=1
        for kid in (morphology.children):
            nbranches+=self.number_branches(kid,n+nbranches,n)
        return nbranches

    def boundary_conditions(self,morphology):
        '''
        Recursively sets the boundary conditions in the linear systems.
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        # Inverse axial resistances at the ends: r0 and rn
        morphology.invr0=float(pi/(2*self.group.Ri)*self.group.diameter[first]**2/self.group.length[first])
        morphology.invrn=float(pi/(2*self.group.Ri)*self.group.diameter[last]**2/self.group.length[last])
        # Correction for boundary conditions
        self.ab_star[1,first]-=float(morphology.invr0/self.group.area[first]) # because of units problems
        self.ab_star[1,last]-=float(morphology.invrn/self.group.area[last])
        self.ab_plus[1,first]-=float(morphology.invr0/self.group.area[first]) # because of units problems
        self.ab_plus[1,last]-=float(morphology.invrn/self.group.area[last])
        self.ab_minus[1,first]-=float(morphology.invr0/self.group.area[first]) # because of units problems
        self.ab_minus[1,last]-=float(morphology.invrn/self.group.area[last])
        # RHS for homogeneous solutions
        self.b_plus[last]=-float(morphology.invrn/self.group.area[last])
        self.b_minus[first]=-float(morphology.invr0/self.group.area[first])
        # Recursive call
        for kid in (morphology.children):
            self.boundary_conditions(kid)
        
    def linear_combination(self,morphology):
        '''
        Calculates solutions by linear combination
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        i=morphology.index+1
        i_parent=morphology.parent+1
        v_star=self.variables['v_star'].get_value()
        u_minus=self.variables['u_minus'].get_value()
        u_plus=self.variables['u_plus'].get_value()
        self.group.v_[first:last+1]=v_star[first:last+1]+self.V[i_parent]*u_minus[first:last+1]\
                                                             +self.V[i]*u_plus[first:last+1]
        # Recursive call
        for kid in (morphology.children):
            self.linear_combination(kid)

    def fill_matrix(self,morphology):
        '''
        Recursively fills the matrix of the linear system that connects branches together.
        Apparently this is quick.
        '''
        first=morphology._origin # first compartment
        last=first+len(morphology.x)-1 # last compartment
        i=morphology.index+1
        i_parent=morphology.parent+1
        v_star=self.variables['v_star'].get_value()
        u_minus=self.variables['u_minus'].get_value()
        u_plus=self.variables['u_plus'].get_value()
        # Towards parent
        if i==1: # first branch, sealed end
            self.P[0,0]=u_minus[first]-1
            self.P[0,1]=u_plus[first]
            self.B[0]=-v_star[first]
        else:
            self.P[i_parent,i_parent]+=(1-u_minus[first])*morphology.invr0
            self.P[i_parent,i]-=u_plus[first]*morphology.invr0
            self.B[i_parent]+=v_star[first]*morphology.invr0
        # Towards children
        self.P[i,i]=(1-u_plus[last])*morphology.invrn
        self.P[i,i_parent]=-u_minus[last]*morphology.invrn
        self.B[i]=v_star[last]*morphology.invrn
        # Recursive call
        for kid in (morphology.children):
            self.fill_matrix(kid)
