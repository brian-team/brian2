import numpy
from base import Language
import sympy
from sympy.printing.ccode import CCodePrinter
from brian2.codegen.templating import apply_code_template

__all__ = ['CLanguage',
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
            line = line+spec.array+'['+index_var+'];'
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
            line = spec.array+'['+index_var+'] = '+var+';'
            lines.append(line)
        return '\n'.join(lines)

    def template_iterate_all(self, index, size):
        return '''
        for(int {index}=0; {index}<{size}; {index}++)
        {{
            %CODE%
        }}
        '''.format(index=index, size=size)
    
    def template_iterate_index_array(self, index, array, size):
        return '''
        for(int _index_{array}=0; _index_{array}<{size}; _index_{array}++)
        {{
            const int {index} = {array}[_index_{array}];
            %CODE%
        }}
        '''.format(index=index, array=array, size=size)

    def template_threshold(self):
        return '''
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
        return '''
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
