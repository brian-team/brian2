import weakref

import numpy as np
from numpy import array, zeros

from brian2.equations.codestrings import Statements
from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        STATIC_EQUATION, PARAMETER) 
from brian2.equations.refractory import add_refractoriness
from brian2.stateupdaters.integration import euler
from brian2.codegen.languages import PythonLanguage
from brian2.codegen.specifiers import (Value, ArrayVariable, Subexpression,
                                       Index)
from brian2.codegen.translation import translate
from brian2.memory import allocate_array
from brian2.core.preferences import brian_prefs
from brian2.core.base import BrianObject
from brian2.core.namespace import ObjectWithNamespace
from brian2.core.spikesource import SpikeSource
from brian2.core.scheduler import Scheduler
from brian2.utils.logger import get_logger
from brian2.groups.group import Group
from brian2.units.allunits import second

__all__ = ['NeuronGroup',
           'CodeRunner']

logger = get_logger(__name__)

class CodeRunner(BrianObject):
    '''
    Runs a code object on an update schedule.
    
    Inserts the current time and dt into the namespace at each step.
    '''
    basename = 'code_runner'
    def __init__(self, codeobj, init=None, pre=None, post=None,
                 when=None, name=None):
        BrianObject.__init__(self, when=when, name=name)
        self.codeobj = codeobj
        self.pre = pre
        self.post = post
        if init is not None:
            init(self)
        
    def update(self):
        if self.pre is not None:
            self.pre(self)
        self.codeobj(t=self.clock.t_, dt=self.clock.dt_)
        if self.post is not None:
            self.post(self)


class NeuronGroupCodeRunner(CodeRunner):
    def __init__(self, group, codeobj, when=None, name=None):
        CodeRunner.__init__(self, codeobj, when=when, name=name)
        self.group = weakref.proxy(group)
        self.prepared = False
        
    def prepare(self):
        if not self.prepared:
            self.is_active = self.group.is_active_
            self.refractory_until = self.group.refractory_until_
            self.refractory = self.group.refractory_
            self.prepared = True


class StateUpdater(NeuronGroupCodeRunner):
    def update(self):
        self.prepare()
        self.is_active[:] = self.clock.t_>=self.refractory_until
        NeuronGroupCodeRunner.update(self)
        
        
class Thresholder(NeuronGroupCodeRunner):
    def update(self):
        self.prepare()
        NeuronGroupCodeRunner.update(self)
        ns = self.codeobj.namespace
        spikesarray = ns['_spikes_space']
        numspikes = ns['_array_num_spikes'][0]
        spikes = spikesarray[:numspikes]
        spikes = spikes[array(self.is_active[spikes], dtype=bool)]
        self.group.spikes = spikes
        self.refractory_until[spikes] = self.clock.t_+self.refractory[spikes]


class Resetter(NeuronGroupCodeRunner):
    def update(self):
        self.prepare()
        spikes = self.group.spikes
        self.codeobj.namespace['_spikes'] = spikes
        self.codeobj.namespace['_num_spikes'] = len(spikes)
        NeuronGroupCodeRunner.update(self)


class NeuronGroup(ObjectWithNamespace, BrianObject, Group, SpikeSource):
    '''
    Group of neurons
    
    In addition to the variable names you create, `NeuronGroup` will have an
    additional state variable ``refractory`` (in units of seconds) which 
    gives the absolute refractory period of the neuron. This value can be
    modified in the reset code. (TODO: more modifiability)
    
    Parameters
    ----------
    N : int
        Number of neurons in the group.
    equations : (str, `Equations`)
        The differential equations defining the group
    method : ?, optional
        The numerical integration method.
    threshold : str, optional
        The condition which produces spikes. Should be a single line boolean
        expression.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on reset.
    namespace: dict, optional
        A dictionary mapping variable/function names to the respective objects.
        If no `namespace` is given, the "implicit" namespace, consisting of
        the local and global namespace surrounding the creation of the class,
        is used.
    dtype : (`dtype`, `dict`), optional
        The `numpy.dtype` that will be used to store the values, or
        :bpref:`default_scalar_dtype` if not specified (`numpy.float64` by
        default).
    clock : Clock, optional
        The update clock to be used, or defaultclock if not specified.
    name : str, optional
        A unique name for the group, otherwise use ``neurongroup_0``, etc.
        
    Notes
    -----
    
    `NeuronGroup` contains a `StateUpdater`, `Thresholder` and `Resetter`, and
    these are run at the 'groups', 'thresholds' and 'resets' slots (i.e. the
    values of `Scheduler.when` take these values). The `Scheduler.order`
    attribute is set to 0 initially, but this can be modified using the
    attributes `state_updater`, `thresholder` and `resetter`.    
    '''
    basename = 'neurongroup'
    def __init__(self, N, equations, method=euler,
                 threshold=None,
                 reset=None,
                 namespace=None,
                 dtype=None, language=None,
                 clock=None, name=None):
        BrianObject.__init__(self, when=clock, name=name)
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
                
        ##### Prepare and validate equations
        if isinstance(equations, basestring):
            equations = Equations(equations)
        if not isinstance(equations, Equations):
            raise ValueError(('equations has to be a string or an Equations '
                              'object, is "%s" instead.') % type(equations))
        # add refractoriness
        equations = add_refractoriness(equations)
        self.equations = equations
        
        logger.debug("Creating NeuronGroup of size {self.N}, "
                     "equations {self.equations}.".format(self=self))
        
        # Check flags
        equations.check_flags({DIFFERENTIAL_EQUATION: ('active'),
                               PARAMETER: ('constant')})

        ##### Setup the memory
        self.arrays = self.allocate_memory(dtype=dtype)
                
        # Setup specifiers
        self.specifiers = self.create_specifiers()        
        
        # Setup the namespace
        self._namespace = self.create_namespace(self.specifiers, namespace)
        
        # Setup units
        self.units = self.equations.units
        
        #: The array of spikes from the most recent threshold operation
        self.spikes = array([], dtype=int)

        # Code generation (TODO: this should be refactored and modularised)
        # Temporary, set default language to Python
        if language is None:
            language = PythonLanguage()
        self.language = language

        #: Performs numerical integration step
        self.state_updater = self.create_state_updater() 
        #: Performs thresholding step, sets the value of `spikes`
        self.thresholder = self.create_thresholder(threshold)
        #: Resets neurons which have spiked (`spikes`)
        self.resetter = self.create_resetter(reset)
        
        # Creation of contained_objects that do the work
        self.contained_objects.append(self.state_updater)
        if self.thresholder is not None:
            self.contained_objects.append(self.thresholder)
        if self.resetter is not None:
            self.contained_objects.append(self.resetter)

        # Activate name attribute access
        Group.__init__(self)
    

    def __len__(self):
        '''
        Return number of neurons in the group.
        '''
        return self.N
    

    def allocate_memory(self, dtype=None):
        # Allocate memory (TODO: this should be refactored somewhere at some point)
        arrayvarnames = set(eq.varname for eq in self.equations.itervalues() if
                            eq.eq_type in (DIFFERENTIAL_EQUATION,
                                           PARAMETER))
        arrays = {}
        for name in arrayvarnames:
            if isinstance(dtype, dict):
                curdtype = dtype[name]
            else:
                curdtype = dtype
            if curdtype is None:
                curdtype = brian_prefs.default_scalar_dtype
            arrays[name] = allocate_array(self.N, dtype=curdtype)         
        logger.debug("NeuronGroup memory allocated successfully.")
        return arrays

    def create_codeobj(self, name, abstract_code, specs, template_method,
                       additional_namespace={}):
        lang = self.language
        logger.debug("NeuronGroup "+name+" abstract code:\n"+abstract_code)
        innercode = translate(abstract_code, specs,
                              brian_prefs.default_scalar_dtype,
                              lang)
        logger.debug("NeuronGroup "+name+" inner code:\n"+str(innercode))
        code = lang.apply_template(innercode, template_method())
        logger.debug("NeuronGroup "+name+" code:\n"+str(code))
        codeobj = lang.code_object(code, specs)
        # all variable names in the equations that are not part of this model        
        identifiers = set()
        for equation in self.equations.itervalues():            
            for identifier in equation.identifiers:
                if not identifier in self.specifiers:
                    identifiers.add(identifier)
        resolved_namespace = self.namespace.resolve_all(list(identifiers))
        for name, spec in self.specifiers.iteritems():
            if isinstance(spec, ArrayVariable):
                resolved_namespace[spec.array] = self.arrays[name]
        resolved_namespace.update(additional_namespace)
        resolved_namespace.update({'_num_neurons': self.N})
        codeobj.compile(resolved_namespace)
        return codeobj
            
    def create_state_updater(self):
        
        codeobj = self.create_codeobj("state updater",
                                      self.abstract_code,
                                      self.specifiers,
                                      self.language.template_state_update
                                      )
        state_updater = StateUpdater(self, codeobj,
                                     name=self.name+'_state_updater',
                                     when=(self.clock, 'groups'))
        return state_updater
        
    def runner(self, code, init=None, pre=None, post=None,
               when=None, name=None,
               level=0):
        '''
        Returns a `CodeRunner` that runs abstract code in the groups namespace
        
        Parameters
        ----------
        code : str
            The abstract code to run.
        init, pre, post : function, optional
            See `CodeRunner`
        when : Scheduler
            When to run, by default in the 'start' slot with the same clock as
            the group.
        name : str
            A unique name, by default the name of the group appended with
            'runner_0', 'runner_1', etc.
        level : int, optional
            How many levels up the stack to go to find values of variables.
        '''
        if when is None: # TODO: make this better with default values
            when = Scheduler(clock=self.clock)
        else:
            raise NotImplementedError
        if name is None:
            if not hasattr(self, 'num_runners'):
                self.num_runners = 0
            name = self.name+'_runner_'+str(self.num_runners)
            self.num_runners += 1      
        smt = Statements(code)
        identifiers = [identifier for identifier in smt.identifiers
                       if not identifier in self.specifiers]
        additional_ns = self.namespace.resolve_all(identifiers)  
        codeobj = self.create_codeobj("runner",
                                      code,
                                      self.specifiers,
                                      self.language.template_state_update,
                                      additional_ns
                                      )
        runner = CodeRunner(codeobj, name=name, when=when,
                            init=init, pre=pre, post=post)
        return runner
        
    def create_thresholder(self, threshold):
        if threshold is None:
            return None

        abstract_code = '_cond = ' + threshold
        
        additional_ns = {
            '_spikes': self.spikes,
            '_spikes_space': zeros(self.N, dtype=int),
            '_array_num_spikes': zeros(1, dtype=int),
            }
        codeobj = self.create_codeobj("thresholder",
                                      abstract_code,
                                      self.specifiers,
                                      self.language.template_threshold,
                                      additional_ns,
                                      )
        thresholder = Thresholder(self, codeobj,
                                  name=self.name+'_thresholder',
                                  when=(self.clock, 'thresholds'))
        return thresholder
        
    def create_resetter(self, reset):
        if reset is None:
            return None
        specs = dict(self.specifiers)
        specs['_neuron_idx'] = Index(all=False)
        abstract_code = reset        
        additional_ns = {
            '_spikes': self.spikes,
            '_num_spikes': len(self.spikes),
            }
        codeobj = self.create_codeobj("resetter",
                                      abstract_code,
                                      specs,
                                      self.language.template_reset,
                                      additional_ns,
                                      )
        resetter = Resetter(self, codeobj,
                            name=self.name+'_resetter',
                            when=(self.clock, 'resets'))
        return resetter
        
    def create_specifiers(self):
        # Standard specifiers always present
        s = {'_neuron_idx': Index(all=True),
             't': Value(np.float64),
             'dt': Value(np.float64)}
        # TODO: What about xi?
        for eq in self.equations.itervalues():
            if eq.eq_type in (DIFFERENTIAL_EQUATION, PARAMETER):
                array = self.arrays[eq.varname]
                s.update({eq.varname: ArrayVariable('_array_' + eq.varname,
                          '_neuron_idx', array.dtype)})
            elif eq.eq_type == STATIC_EQUATION:                
                s.update({eq.varname: Subexpression(str(eq.expr))})
            else:
                raise AssertionError('Unknown equation type "%s"' % eq.eq_type)
        
        return s        
    
    abstract_code = property(lambda self: self.method(self.equations))


if __name__=='__main__':
    from pylab import *
    from brian2 import *
    from brian2.codegen.languages import *    
    import time

    N = 100
    tau = 10*ms
    eqs = '''
    dV/dt = (2*volt-V)/tau : volt (active)
    Vt : volt
    '''
    threshold = 'V>Vt'
    reset = 'V = 0*volt'
    G = NeuronGroup(N, eqs,
                    threshold=threshold,
                    reset=reset,
                    language=CPPLanguage()
                    #language=NumexprPythonLanguage(),
                    )
    G.refractory = 5*ms
    Gmid = Subgroup(G, 40, 80)
    
    runner = G.runner('Vt = 1*volt-(t/second)*5*volt')

    G.V = rand(N)
    
    statemon = StateMonitor(G, 'V', record=range(3))
    spikemon = SpikeMonitor(Gmid)
    
    start = time.time()
    run(1*ms)
    print 'Initialise time:', time.time()-start
    start = time.time()
    run(99*ms)
    print 'Runtime:', time.time()-start
    subplot(121)
    plot(statemon.t, statemon.V)
    subplot(122)
    plot(spikemon.t, spikemon.i, '.k')
    show()
