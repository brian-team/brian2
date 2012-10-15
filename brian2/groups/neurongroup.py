import numpy as np

from brian2.equations import Equations
from brian2.stateupdaters.integration import euler
from brian2.codegen.languages import PythonLanguage
from brian2.codegen.specifiers import (Value, ArrayVariable, Subexpression,
                                       Index)
from brian2.codegen.translation import translate
from brian2.memory import allocate_array
from brian2.core.preferences import brian_prefs
from brian2.core.base import BrianObject
from brian2.utils.logger import get_logger
from brian2.groups.group import Group

__all__ = ['NeuronGroup']

logger = get_logger(__name__)

class NeuronGroup(BrianObject, Group):
    '''
    This is currently only a placeholder object, it does not even save states.
    
    Its only purpose is to provide an entry point for code generation, i.e.
    you can give it an equations string (or an Equations object) and it can
    generate abstract code (via a state updater) and "specifiers", needed for
    the next steps of code generation.
    '''
    basename = 'neurongroup'
    def __init__(self, N, equations, method=euler,
                 dtype=None, language=None,
                 when=None, name=None):
        BrianObject.__init__(self, when=when, name=name)
        ##### VALIDATE ARGUMENTS AND STORE ATTRIBUTES
        self.method = method
        try:
            self.N = N = int(N)
        except ValueError:
            if isinstance(N, str):
                raise TypeError("First NeuronGroup argument should be size, not equations.")
            raise
        if N<1:
            raise ValueError("NeuronGroup size should be at least 1, was "+str(N))
        # Validate equations
        if isinstance(equations, basestring):
            equations = Equations(equations, level=1)
        if not isinstance(equations, Equations):
            raise ValueError(('equations has to be a string or an Equations '
                              'object, is "%s" instead.') % type(equations))
        self.equations = equations
        
        logger.debug("Creating NeuronGroup of size {self.N}, "
                     "equations {self.equations}.".format(self=self))
        
        # Check flags
        equations.check_flags({'diff_equation': ('active'),
                               'parameter': ('constant')})
        
        # Set dtypes and units
        self.prepare_dtypes(dtype=dtype)
        self.units = equations.units
        
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        self.allocate_memory()
        
        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set language to Python explicitly
        self.language = PythonLanguage()
        self.create_state_updater()
        
        # Activate name attribute access
        Group.__init__(self)
        
    def prepare_dtypes(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        arrayvarnames = set(eq.varname for eq in self.equations.equations.itervalues() if eq.eq_type in ('diff_equation', 'parameter'))
        self.dtypes = {}
        for name in arrayvarnames:
            if isinstance(dtype, dict):
                curdtype = dtype[name]
            else:
                curdtype = dtype
            if curdtype is None:
                curdtype = brian_prefs.default_scalar_dtype
            self.dtypes[name] = curdtype
        logger.debug("NeuronGroup dtypes: "+", ".join(name+'='+str(dtype) for name, dtype in self.dtypes.iteritems()))

    def allocate_memory(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        self.arrays = {}
        for name, curdtype in self.dtypes.iteritems():
            self.arrays[name] = allocate_array(self.N, dtype=curdtype)
        logger.debug("NeuronGroup memory allocated successfully.")
            
    def create_state_updater(self):
        lang = self.language
        specs = self.specifiers
        abstract_code = self.abstract_code
        logger.debug("NeuronGroup state updater abstract code:\n"+abstract_code)
        innercode = translate(self.abstract_code, specs,
                              brian_prefs.default_scalar_dtype,
                              lang)
        logger.debug("NeuronGroup state updater inner code:\n"+innercode)
        code = lang.apply_template(innercode, lang.template_state_update())
        logger.debug("NeuronGroup state updater code:\n"+code)
        codeobj = lang.code_object(code, specs)
        self.namespace = {}
        for name, arr in self.arrays.iteritems():
            self.namespace['_array_'+name] = arr
        self.namespace['_num_neurons'] = self.N
        self.namespace['dt'] = self.clock.dt_
        self.namespace['t'] = self.clock.t_
        codeobj.compile(self.namespace)
        self.state_update_codeobj = codeobj

    def get_specifiers(self):
        '''
        Returns a dictionary of `Specifier` objects.
        '''
        # Standard specifiers always present
        s = {'_neuron_idx': Index(all=True),
             'dt': Value(np.float64),
             't': Value(np.float64)}
        # TODO: What about xi?
        for eq in self.equations.equations.itervalues():
            if eq.eq_type in ('diff_equation', 'parameter'):
                s.update({eq.varname: ArrayVariable('_array_'+eq.varname,
                          '_neuron_idx', self.dtypes[eq.varname])})
            elif eq.eq_type == 'static_equation':                
                s.update({eq.varname: Subexpression(str(eq.expr.frozen()))})
            else:
                raise AssertionError('Unknown equation type "%s"' % eq.eq_type)
        
        return s        
    
    abstract_code = property(lambda self: self.method(self.equations))
    specifiers = property(get_specifiers)
    
    def update(self):
        self.state_update_codeobj(t=self.clock.t_)

if __name__=='__main__':
    from pylab import *
    from brian2 import *
    import time
    #log_level_debug()
    N = 100
    tau = 10*ms
    eqs = '''
    dV/dt = -V/tau : 1
    '''
    G = NeuronGroup(N, eqs)
    G.V = 1.0
    recvals = []
    times = []
    @network_operation
    def recup(self):
        recvals.append(G.V[0])
        times.append(G.clock.t_)
    start = time.time()
    run(1*ms)
    print 'Initialise time:', time.time()-start
    start = time.time()
    run(99*ms)
    print 'Runtime:', time.time()-start
    plot(times, recvals)
    show()
    