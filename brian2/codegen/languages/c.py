'''
TODO: restrict keyword optimisations
'''
import numpy
from base import Language, CodeObject
import sympy
from sympy.printing.ccode import CCodePrinter
from brian2.codegen.templating import apply_code_template
from scipy import weave

__all__ = ['CLanguage', 'CCodeObject',
           'c_data_type',
           ]

def c_data_type(dtype):
    '''
    Gives the C language specifier for numpy data types. For example,
    ``numpy.int32`` maps to ``int32_t`` in C.
    '''
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = array([1]).dtype.type
        
    if dtype==numpy.float32:
        dtype = 'float'
    elif dtype==numpy.float64:
        dtype = 'double'
    elif dtype==numpy.int32:
        dtype = 'int32_t'
    elif dtype==numpy.int64:
        dtype = 'int64_t'
    elif dtype==numpy.uint16:
        dtype = 'uint16_t'
    elif dtype==numpy.uint32:
        dtype = 'uint32_t'
    elif dtype==numpy.bool_ or dtype is bool:
        dtype = 'bool'
    else:
        raise ValueError("dtype "+str(dtype)+" not known.")
    return dtype

class CLanguage(Language):
    '''
    Initialisation arguments:
    
    ``compiler``
        The distutils name of the compiler.
    ``extra_compile_args``
        Extra compilation arguments, e.g. for optimisation. Best performance is
        often gained by using
        ``extra_compile_args=['-O3', '-ffast-math', '-march=native']`` 
        however the ``'-march=native'`` is not compatible with all versions of
        gcc so is switched off by default.
    ``restrict``
        The keyword used for the given compiler to declare pointers as
        restricted (different on different compilers).
    ``flush_denormals``
        Adds code to flush denormals to zero, but the code is gcc and
        architecture specific, so may not compile on all platforms, therefore
        it is off by default. The code, for reference is::

            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            
        Found at `<http://stackoverflow.com/questions/2487653/avoiding-denormal-values-in-c>`_.
    '''
    def __init__(self, compiler='gcc', extra_compile_args=['-O3', '-ffast-math'],
                 restrict='__restrict__', flush_denormals=False):
        self.compiler = compiler
        self.extra_compile_args = extra_compile_args
        self.restrict = restrict+' '
        self.flush_denormals = flush_denormals
    
    def translate_expression(self, expr):
        expr = sympy.sympify(expr)
        return CCodePrinter().doprint(expr)

    def translate_statement(self, statement):
        var, op, expr = statement.var, statement.op, statement.expr
        if op==':=':
            decl = c_data_type(statement.dtype)+' '
            op = '='
            if statement.constant:
                decl = 'const '+decl
        else:
            decl = ''
        return decl+var+' '+op+' '+self.translate_expression(expr)+';'

    def translate_statement_sequence(self, statements, specifiers):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            index_var = specifiers[var].index
            index_spec = specifiers[index_var]
            spec = specifiers[var]
            if var not in write:
                line = 'const '
            else:
                line = ''
            line = line+c_data_type(spec.dtype)+' '+var+' = '
            line = line+'_ptr'+spec.array+'['+index_var+'];'
            lines.append(line)
        # simply declare variables that will be written but not read
        for var in write:
            if var not in read:
                spec = specifiers[var]
                line = c_data_type(spec.dtype)+' '+var+';'
                lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            index_var = specifiers[var].index
            index_spec = specifiers[index_var]
            spec = specifiers[var]
            line = '_ptr'+spec.array+'['+index_var+'] = '+var+';'
            lines.append(line)
        code = '\n'.join(lines)
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        lines = []
        for var in read.union(write):
            spec = specifiers[var]
            line = c_data_type(spec.dtype)+' * '+self.restrict+'_ptr'+spec.array+' = '+spec.array+';'
            lines.append(line)
        pointers = '\n'.join(lines)
        translation = {'%CODE%': code,
                       '%POINTERS%': pointers,
                       }
        return translation
    
    def code_object(self, code):
        return CCodeObject(code, compiler=self.compiler,
                           extra_compile_args=self.extra_compile_args)
        
    def denormals_to_zero_code(self):
        if self.flush_denormals:
            return '''
        #define CSR_FLUSH_TO_ZERO         (1 << 15)
        unsigned csr = __builtin_ia32_stmxcsr();
        csr |= CSR_FLUSH_TO_ZERO;
        __builtin_ia32_ldmxcsr(csr);
            '''
        else:
            return ''

    def template_iterate_all(self, index, size):
        return self.denormals_to_zero_code()+'''
        %POINTERS%
        for(int {index}=0; {index}<{size}; {index}++)
        {{
            %CODE%
        }}
        '''.format(index=index, size=size)
    
    def template_iterate_index_array(self, index, array, size):
        return self.denormals_to_zero_code()+'''
        %POINTERS%
        for(int _index_{array}=0; _index_{array}<{size}; _index_{array}++)
        {{
            const int {index} = {array}[_index_{array}];
            %CODE%
        }}
        '''.format(index=index, array=array, size=size)

    def template_threshold(self):
        return self.denormals_to_zero_code()+'''
        %POINTERS%
        int _numspikes = 0;
        for(int _neuron_idx=0; _neuron_idx<_num_neurons; _neuron_idx++)
        {
            %CODE%
            if(_cond) {
                _spikes[_numspikes++] = _neuron_idx;
            }
        }
        '''

    def template_synapses(self):
        return self.denormals_to_zero_code()+'''
        %POINTERS%
        for(int _spiking_synapse_idx=0;
            _spiking_synapse_idx<_num_spiking_synapses;
            _spiking_synapse_idx++)
        {
                const int _synapse_idx = _spiking_synapses[_spiking_synapse_idx];
                const int _postsynaptic_idx = _postsynaptic[_synapse_idx];
                const int _presynaptic_idx = _presynaptic[_synapse_idx];
                %CODE%
        }
        '''

class CCodeObject(CodeObject):
    def __init__(self, code, compiler='gcc', extra_compile_args=['-O3']):
        self.code = code+'  '
        self.compiler = compiler
        self.extra_compile_args = extra_compile_args
        
    def compile(self, namespace):
        self.namespace = namespace
        
    def __call__(self, **kwds):
        self.namespace.update(kwds)
        weave.inline(self.code, self.namespace.keys(),
                     local_dict = self.namespace,
                     #support_code=c_support_code,
                     compiler=self.compiler,
                     extra_compile_args=self.extra_compile_args)
