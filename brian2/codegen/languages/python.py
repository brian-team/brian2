from base import Language, CodeObject
import sympy

__all__ = ['PythonLanguage', 'PythonCodeObject']

class PythonLanguage(Language):
    
    def translate_expression(self, expr):
        return expr.strip()
    
    def translate_statement(self, statement):
        # TODO: optimisation, translate arithmetic to a sequence of inplace
        # operations like a=b+c -> add(b, c, a)
        var, op, expr = statement.var, statement.op, statement.expr
        if op==':=':
            op = '='
        return var+' '+op+' '+self.translate_expression(expr)

    def translate_statement_sequence(self, statements, specifiers):
        read, write = self.array_read_write(statements, specifiers)
        lines = []
        # read arrays
        for var in read:
            spec = specifiers[var]
            index_spec = specifiers[spec.index]
            line = var+' = '+spec.array
            if not index_spec.all:
                line = line+'['+spec.index+']'
            lines.append(line)
        # the actual code
        lines.extend([self.translate_statement(stmt) for stmt in statements])
        # write arrays
        for var in write:
            index_var = specifiers[var].index
            index_spec = specifiers[index_var]
            # check if all operations were inplace and we're operating on the
            # whole vector, if so we don't need to write the array back
            if not index_spec.all:
                all_inplace = False
            else:
                all_inplace = True
                for stmt in statements:
                    if stmt.var==var and not stmt.inplace:
                        all_inplace = False
                        break
            if not all_inplace:
                line = specifiers[var].array
                if index_spec.all:
                    line = line+'[:]'
                else:
                    line = line+'['+index_var+']'
                line = line+' = '+var
                lines.append(line)
        return '\n'.join(lines)
    
    def code_object(self, code):
        return PythonCodeObject(code)

    def template_iterate_all(self, index, size):
        return '''
        %CODE%
        '''
    
    def template_iterate_index_array(self, index, array, size):
        return '''
        {index} = {array}
        %CODE%
        '''.format(index=index, array=array)

    def template_threshold(self):
        return '''
        %CODE%
        return _cond.nonzero()[0]
        '''

    def template_synapses(self):
        return '''
        # TODO: check and improve this
        _post_neurons = _postsynaptic.data.take(_spiking_synapses)
        _perm = _post_neurons.argsort()
        _aux = _post_neurons.take(_perm)
        _flag = empty(len(_aux)+1, dtype=bool)
        _flag[0] = _flag[-1] = 1
        not_equal(_aux[1:], _aux[:-1], _flag[1:-1])
        _F = _flag.nonzero()[0][:-1]
        logical_not(_flag, _flag)
        while len(_F):
            _u = _aux.take(_F)
            _i = _perm.take(_F)
            _postsynaptic_idx = _u
            _synapse_idx = _spiking_synapse[_i]
            # TODO: how do we get presynaptic indices? do we need to?
        
            %CODE%
        
            _F += 1
            _F = extract(_flag.take(_F), _F)
        '''


class PythonCodeObject(CodeObject):
    def compile(self, namespace):
        self.namespace = namespace
        self.compiled_code = compile(self.code, '(string)', 'exec')
    
    def __call__(self, **kwds):
        self.namespace.update(kwds)
        exec self.compiled_code in self.namespace

# THIS DOESN'T WORK
#def convert_expr_to_inplace(expr):
#    lines = []
#    expr = symbolic_eval(expr)
#    curstep = 0
#    def step(subexpr):
#        myargs = map(step, subexpr.args)
#        name = '_temp_'+str(curstep)
#        curstep += 1
#        if isinstance(subexpr, sympy.Add):
#            args = subexpr.args
#            while len(args)>=2:
#                lines.append('add({')
#        return name
#    return '\n'.join(lines)
#
#if __name__=='__main__':
#    print convert_expr_to_inplace('x+y*z')
    