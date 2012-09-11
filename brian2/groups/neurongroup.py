import numpy as np

from brian2.equations import Equations
from brian2.stateupdaters.integration import euler
from brian2.codegen.specifiers import (Value, ArrayVariable, Subexpression,
                                       Index)

class NeuronGroup(object):
    '''
    This is currently only a placeholder object, it does not even save states.
    
    Its only purpose is to provide an entry point for code generation, i.e.
    you can give it an equations string (or an Equations object) and it can
    generate abstract code (via a state updater) and "specifiers", needed for
    the next steps of code generation.
    '''
    
    def __init__(self, N, model, method=euler):
        if isinstance(model, basestring):
            model = Equations(model, level=1)
        if not isinstance(model, Equations):
            raise ValueError(('model has to be a string or an Equations '
                              'object, is "%s" instead.') % type(model))
        
        # Check flags and identifiers        
        model.check_identifiers(('t', 'dt', 'xi')) # not necessary...
        model.check_flags({'diff_equation': ('active'),
                           'parameter': ('constant')})
        
        self.model = model
        self.method = method
    
    def get_specifiers(self):
        '''
        Returns a dictionary of :class:`Specifier` objects.
        '''
        # Standard specifiers always present
        s = {'_neuron_idx': Index(all=True),
             'dt': Value(np.float64),
             't':Value(np.float64)}
        # TODO: What about xi?
        for eq in self.model.equations.itervalues():
            if eq.eq_type in ('diff_equation', 'parameter'):
                s.update({eq.varname: ArrayVariable('_array_'+eq.varname,
                          '_neuron_idx', np.float64)})
            elif eq.eq_type == 'static_equation':                
                s.update({eq.varname: Subexpression(str(eq.expr.frozen()))})
            else:
                raise AssertionError('Unknown equation type "%s"' % eq.eq_type)
        
        return s        
        
    
    abstract_code = property(lambda self: self.method(self.model))
    specifiers = property(get_specifiers)
    