'''
This module translates a series of statements into a language-specific
syntactically correct code block that can be inserted into a template.

It infers whether or not a variable can be declared as
constant, etc. It should handle common subexpressions, and so forth.

The input information needed:

* The sequence of statements (a multiline string) in standard mathematical form
* The list of known variables, common subexpressions and functions, and for each
  variable whether or not it is a value or an array, and if an array what the
  dtype is.
* The dtype to use for newly created variables
* The language to translate to
'''

from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       OutputVariable, Subexpression, Index)
from brian2.codegen.statements import Statement
from brian2.utils.stringtools import (deindent, strip_empty_lines, indent,
                                      get_identifiers)
import re
from collections import defaultdict

__all__ = ['translate', 'make_statements']

DEBUG = False

class LineInfo(object):
    '''
    A helper class, just used to store attributes.
    '''
    def __init__(self, **kwds):
        for k, v in kwds.iteritems():
            setattr(self, k, v)

def make_statements(code, specifiers, dtype):
    '''
    Turn a series of abstract code statements into Statement objects, inferring
    whether each line is a set/declare operation, whether the variables are
    constant or not, and handling the cacheing of subexpressions. Returns a
    list of Statement objects. For arguments, see documentation for
    :func:`translate`.
    '''
    code = strip_empty_lines(deindent(code))
    lines = code.split('\n')
    lines = [LineInfo(code=line) for line in lines]
    if DEBUG:
        print 'INPUT CODE:'
        print code
    dtypes = dict((name, value.dtype) for name, value in specifiers.items() if hasattr(value, 'dtype'))
    # we will do inference to work out which lines are := and which are =
    #defined = set(specifiers.keys()) # variables which are already defined
    defined = set(var for var, spec in specifiers.items() if not isinstance(spec, OutputVariable))
    for line in lines:
        # parse statement into "var op expr"
        m = re.search(r'[^><=]=', line.code)
        if not m:
            raise ValueError("Could not extract statement from: "+line.code)
        start, end = m.start(), m.end()
        op = line.code[start:end].strip()
        var = line.code[:start].strip()
        expr = line.code[end:].strip()
        # var should be a single word
        if len(re.findall(r'^[A-Za-z_][A-Za-z0-9_]*$', var))!=1:
            raise ValueError("LHS in statement must be single variable name, line: "+line.code)
        if op=='=' and var not in defined:
            op = ':='
            defined.add(var)
            if var not in dtypes:
                dtypes[var] = dtype
        statement = Statement(var, op, expr, dtypes[var])
        line.statement = statement
        # for each line will give the variable being written to
        line.write = var 
        # each line will give a set of variables which are read
        line.read = set(get_identifiers(expr))
        
    if DEBUG:
        print 'PARSED STATEMENTS:'
        for line in lines:
            print line.statement, 'Read:'+str(line.read), 'Write:'+line.write
    
    # all variables which are written to at some point in the code block
    # used to determine whether they should be const or not
    all_write = set(line.write for line in lines)
    if DEBUG:
        print 'ALL WRITE:', all_write
        
    # backwards compute whether or not variables will be read again
    # note that will_read for a line gives the set of variables it will read
    # on the current line or subsequent ones. will_write gives the set of
    # variables that will be written after the current line
    will_read = set()
    will_write = set()
    for line in lines[::-1]:
        will_read = will_read.union(line.read)
        line.will_read = will_read.copy()
        line.will_write = will_write.copy()
        will_write.add(line.write)

    if DEBUG:
        print 'WILL READ/WRITE:'
        for line in lines:
            print line.statement, 'Read:'+str(line.will_read), 'Write:'+str(line.will_write)
        
    # generate cacheing statements for common subexpressions
    # cached subexpressions need to be recomputed whenever they are to be used
    # on the next line, and currently invalid (meaning that the current value
    # stored in the subexpression variable is no longer accurate because one
    # of the variables appearing in it has changed). All subexpressions start
    # as invalid, and are invalidated whenever one of the variables appearing
    # in the RHS changes value.
    subexpressions = dict((name, val) for name, val in specifiers.items() if isinstance(val, Subexpression))
    if DEBUG:
        print 'SUBEXPRESSIONS:', subexpressions.keys()
    statements = []
    # all start as invalid
    valid = dict((name, False) for name in subexpressions.keys())
    # none are yet defined (or declared)
    subdefined = dict((name, False) for name in subexpressions.keys())
    for line in lines:
        stmt = line.statement
        read = line.read
        write = line.write
        will_read = line.will_read
        will_write = line.will_write
        # check that all subexpressions in expr are valid, and if not
        # add a definition/set its value, and set it to be valid
        for var in read:
            # if subexpression, and invalid
            if not valid.get(var, True): # all non-subexpressions are valid
                # if already defined/declared
                if subdefined[var]:
                    op = '='
                    constant = False
                else:
                    op = ':='
                    subdefined[var] = True
                    dtypes[var] = dtype # default dtype
                    # set to constant only if we will not write to it again
                    constant = var not in will_write
                    # check all subvariables are not written to again as well
                    if constant:
                        ids = subexpressions[var].identifiers
                        constant = all(v not in will_write for v in ids)
                valid[var] = True
                statement = Statement(var, op, subexpressions[var].expr,
                                      dtype, constant=constant,
                                      subexpression=True)
                statements.append(statement)
        var, op, expr = stmt.var, stmt.op, stmt.expr
        # invalidate any subexpressions including var
        for subvar, spec in subexpressions.items():
            if var in spec.identifiers:
                valid[subvar] = False
        # constant only if we are declaring a new variable and we will not
        # write to it again
        constant = op==':=' and var not in will_write
        statement = Statement(var, op, expr, dtypes[var],
                              constant=constant)
        statements.append(statement)

    if DEBUG:
        print 'OUTPUT STATEMENTS:'
        for stmt in statements:
            print stmt

    return statements


def translate(code, specifiers, dtype, language):
    '''
    Translates an abstract code block into the target language.
    
    ``code``
        The abstract code block, a series of one-line statements.
    ``specifiers``
        A dict of ``(var, spec)`` where ``var`` is a variable name whose type
        is specified by ``spec``, a :class:`Specifier` object. These include
        :class:`Value` for a single (non-vector) value that will be inserted
        into the namespace at runtime, :class:`Function` for a function,
        :class:`ArrayVariable` for a value coming from an array of values,
        :class:`Index` for the name of the index into these arrays, and
        :class:`Subexpression` for a common subexpression used in the code.
        There should only be a single :class:`Index` specifier, and the name
        should correspond to that given in the :class:`ArrayVariable`
        specifiers.
    ``dtype``
        The default dtype for newly created variables (usually float64).
    ``language``
        The :class:`Language` to translate to.
    
    Returns a multi-line string.
    '''
    statements = make_statements(code, specifiers, dtype)
    return language.translate_statement_sequence(statements, specifiers)
    

if __name__=='__main__':
    from numpy import float64
    from languages import CLanguage, PythonLanguage, NumexprPythonLanguage
    DEBUG = True
    # switch between these two to invalidate x on the last line
    if 1:
        # x invalid on last line
        code = '''
        _tmp_V = x
        I += 1
        V += _tmp_V*x*dt
        _cond = V>x
        '''
    else:
        # x valid on last line
        code = '''
        _tmp_V = x
        V += _tmp_V*x*dt
        '''
    specifiers = {
        'V':ArrayVariable('_array_V', '_neuron_idx', float64),
        'I':ArrayVariable('_array_I', '_neuron_idx', float64),
        'x':Subexpression('-(V+I)/tau'),
        'tau':Value(float64),
        'dt':Value(float64),
        '_cond':OutputVariable(bool),
        #'_neuron_idx':Index(),
        '_neuron_idx':Index(all=False),
        }
    for lang in [
                 CLanguage(),
                 PythonLanguage(),
                 NumexprPythonLanguage()
                 ]:
        print lang.__class__.__name__
        print '='*len(lang.__class__.__name__)
        output = translate(code, specifiers, float64, lang)
        print 'OUTPUT CODE:'
        if isinstance(output, str):
            print indent(output)
        else:
            for k, v in output.items():
                print k+':'
                print indent(v)
        print
