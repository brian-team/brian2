from collections import namedtuple
import os

import numpy as np
from nose.plugins.skip import SkipTest
from numpy.testing.utils import assert_raises
from nose.plugins.attrib import attr

from brian2 import prefs, clear_cache, _cache_dirs_and_extensions
from brian2.codegen.optimisation import optimise_statements
from brian2.codegen.translation import (analyse_identifiers,
                                        get_identifiers_recursively,
                                        parse_statement,
                                        make_statements,
                                        )
from brian2.codegen.statements import Statement
from brian2.codegen.codeobject import CodeObject
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.core.variables import Subexpression, Variable, Constant, ArrayVariable
from brian2.core.functions import Function, DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS
from brian2.devices.device import auto_target, device
from brian2.units.fundamentalunits import Unit
from brian2.units import second, ms

FakeGroup = namedtuple('FakeGroup', ['variables'])

@attr('codegen-independent')
def test_auto_target():
    # very basic test that the "auto" codegen target is useable
    assert issubclass(auto_target(), CodeObject)


@attr('codegen-independent')
def test_analyse_identifiers():
    '''
    Test that the analyse_identifiers function works on a simple clear example.
    '''
    code = '''
    a = b+c
    d = e+f
    '''
    known = {'b': Variable(name='b'),
             'c': Variable(name='c'),
             'd': Variable(name='d'),
             'g': Variable(name='g')}
    
    defined, used_known, dependent = analyse_identifiers(code, known)
    assert 'a' in defined  # There might be an additional constant added by the
                           # loop-invariant optimisation
    assert used_known == {'b', 'c', 'd'}
    assert dependent == {'e', 'f'}


@attr('codegen-independent')
def test_get_identifiers_recursively():
    '''
    Test finding identifiers including subexpressions.
    '''
    variables = {'sub1': Subexpression(name='sub1',
                                       dtype=np.float32, expr='sub2 * z',
                                       owner=FakeGroup(variables={}),
                                       device=None),
                 'sub2': Subexpression(name='sub2',
                                       dtype=np.float32, expr='5 + y',
                                       owner=FakeGroup(variables={}),
                                       device=None),
                 'x': Variable(name='x')}
    identifiers = get_identifiers_recursively(['_x = sub1 + x'],
                                              variables)
    assert identifiers == {'x', '_x', 'y', 'z', 'sub1', 'sub2'}


@attr('codegen-independent')
def test_write_to_subexpression():
    variables = {
        'a': Subexpression(name='a', dtype=np.float32,
                           owner=FakeGroup(variables={}), device=None,
                           expr='2*z'),
        'z': Variable(name='z')
    }

    # Writing to a subexpression is not allowed
    code = 'a = z'
    assert_raises(SyntaxError, make_statements, code, variables, np.float32)


@attr('codegen-independent')
def test_repeated_subexpressions():
    variables = {
        'a': Subexpression(name='a', dtype=np.float32,
                           owner=FakeGroup(variables={}), device=None,
                           expr='2*z'),
        'x': Variable(name='x'),
        'y': Variable(name='y'),
        'z': Variable(name='z')
    }
    # subexpression a (referring to z) is used twice, but can be reused the
    # second time (no change to z)
    code = '''
    x = a
    y = a
    '''
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ['a', 'x', 'y']
    assert vector_stmts[0].constant

    code = '''
    x = a
    z *= 2
    '''
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ['a', 'x', 'z']
    # Note that we currently do not mark the subexpression as constant in this
    # case, because its use after the "z *=2" line would actually redefine it.
    # Our algorithm is currently not smart enough to detect that it is actually
    # not used afterwards

    # a refers to z, therefore we have to redefine a after z changed, and a
    # cannot be constant
    code = '''
    x = a
    z *= 2
    y = a
    '''
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    assert [stmt.var for stmt in vector_stmts] == ['a', 'x', 'z', 'a', 'y']
    assert not any(stmt.constant for stmt in vector_stmts)


@attr('codegen-independent')
def test_nested_subexpressions():
    '''
    This test checks that code translation works with nested subexpressions.
    '''
    code = '''
    x = a + b + c
    c = 1
    x = a + b + c
    d = 1
    x = a + b + c
    '''
    variables = {
        'a': Subexpression(name='a', dtype=np.float32, owner=FakeGroup(variables={}), device=None,
                           expr='b*b+d'),
        'b': Subexpression(name='b', dtype=np.float32, owner=FakeGroup(variables={}), device=None,
                           expr='c*c*c'),
        'c': Variable(name='c'),
        'd': Variable(name='d'),
        }
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    evalorder = ''.join(stmt.var for stmt in vector_stmts)
    # This is the order that variables ought to be evaluated in (note that
    # previously this test did not expect the last "b" evaluation, because its
    # value did not change (c was not changed). We have since removed this
    # subexpression caching, because it did not seem to apply in practical
    # use cases)
    assert evalorder == 'baxcbaxdbax'

@attr('codegen-independent')
def test_apply_loop_invariant_optimisation():
    variables = {'v': Variable('v', scalar=False),
                 'w': Variable('w', scalar=False),
                 'dt': Constant('dt', dimensions=second.dim, value=0.1*ms),
                 'tau': Constant('tau', dimensions=second.dim, value=10*ms),
                 'exp': DEFAULT_FUNCTIONS['exp']}
    statements = [Statement('v', '=', 'dt*w*exp(-dt/tau)/tau + v*exp(-dt/tau)', '', np.float32),
                  Statement('w', '=', 'w*exp(-dt/tau)', '', np.float32)]
    scalar, vector = optimise_statements([], statements, variables)
    # The optimisation should pull out at least exp(-dt / tau)
    assert len(scalar) >= 1
    assert np.issubdtype(scalar[0].dtype, np.floating)
    assert scalar[0].var == '_lio_1'
    assert len(vector) == 2
    assert all('_lio_' in stmt.expr for stmt in vector)

@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_integer():
    variables = {'v': Variable('v', scalar=False),
                 'N': Constant('N', 10),
                 'b': Variable('b', scalar=True, dtype=int),
                 'c': Variable('c', scalar=True, dtype=int),
                 'd': Variable('d', scalar=True, dtype=int),
                 'y': Variable('y', scalar=True, dtype=float),
                 'z': Variable('z', scalar=True, dtype=float),
                 'w': Variable('w', scalar=True, dtype=float),
                 }
    statements = [Statement('v', '=', 'v % (2*3*N)', '', np.float32),
                  # integer version doesn't get rewritten but float version does
                  Statement('a', ':=', 'b//(c//d)', '', int),
                  Statement('x', ':=', 'y/(z/w)', '', float),
                  ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 3
    assert np.issubdtype(scalar[0].dtype, np.signedinteger)
    assert scalar[0].var == '_lio_1'
    expr = scalar[0].expr.replace(' ', '')
    assert expr=='6*N' or expr=='N*6'
    assert np.issubdtype(scalar[1].dtype, np.signedinteger)
    assert scalar[1].var == '_lio_2'
    expr = scalar[1].expr.replace(' ', '')
    assert expr=='b//(c//d)'
    assert np.issubdtype(scalar[2].dtype, np.floating)
    assert scalar[2].var == '_lio_3'
    expr = scalar[2].expr.replace(' ', '')
    assert expr=='(y*w)/z' or expr=='(w*y)/z'

@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_boolean():
    variables = {'v1': Variable('v1', scalar=False),
                 'v2': Variable('v2', scalar=False),
                 'N': Constant('N', 10),
                 'b': Variable('b', scalar=True, dtype=bool),
                 'c': Variable('c', scalar=True, dtype=bool),
                 'int': DEFAULT_FUNCTIONS['int'],
                 'foo': Function(lambda x: None,
                                 arg_units=[Unit(1)], return_unit=Unit(1),
                                 arg_types=['boolean'], return_type='float',
                                 stateless=False)
                 }
    # The calls for "foo" cannot be pulled out, since foo is marked as stateful
    statements = [Statement('v1', '=', '1.0*int(b and c)', '', np.float32),
                  Statement('v1', '=', '1.0*foo(b and c)', '', np.float32),
                  Statement('v2', '=', 'int(not b and True)', '', np.float32),
                  Statement('v2', '=', 'foo(not b and True)', '', np.float32)
                  ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 4
    assert scalar[0].expr == '1.0 * int(b and c)'
    assert scalar[1].expr == 'b and c'
    assert scalar[2].expr == 'int((not b) and True)'
    assert scalar[3].expr == '(not b) and True'
    assert len(vector) == 4
    assert vector[0].expr == '_lio_1'
    assert vector[1].expr == 'foo(_lio_2)'
    assert vector[2].expr == '_lio_3'
    assert vector[3].expr == 'foo(_lio_4)'

@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_no_optimisation():
    variables = {'v1': Variable('v1', scalar=False),
                 'v2': Variable('v2', scalar=False),
                 'N': Constant('N', 10),
                 's1': Variable('s1', scalar=True, dtype=float),
                 's2': Variable('s2', scalar=True, dtype=float),
                 'rand': DEFAULT_FUNCTIONS['rand']
                 }
    statements = [
        # This hould not be simplified to 0!
        Statement('v1', '=', 'rand() - rand()', '', np.float),
        Statement('v1', '=', '3*rand() - 3*rand()', '', np.float),
        Statement('v1', '=', '3*rand() - ((1+2)*rand())', '', np.float),
        # This should not pull out rand()*N
        Statement('v1', '=', 's1*rand()*N', '', np.float),
        Statement('v1', '=', 's2*rand()*N', '', np.float),
        # This is not important mathematically, but it would change the numbers
        # that are generated
        Statement('v1', '=', '0*rand()*N', '', np.float),
        Statement('v1', '=', '0/rand()*N', '', np.float)
    ]
    scalar, vector = optimise_statements([], statements, variables)
    for vs in vector[:3]:
        assert vs.expr.count('rand()') == 2, 'Expression should still contain two rand() calls, but got ' + str(vs)
    for vs in vector[3:]:
        assert vs.expr.count('rand()') == 1, 'Expression should still contain a rand() call, but got ' + str(vs)

@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_simplification():
    variables = {'v1': Variable('v1', scalar=False),
                 'v2': Variable('v2', scalar=False),
                 'i1': Variable('i1', scalar=False, dtype=int),
                 'N': Constant('N', 10)
                 }
    statements = [
        # Should be simplified to 0.0
        Statement('v1', '=', 'v1 - v1', '', np.float),
        Statement('v1', '=', 'N*v1 - N*v1', '', np.float),
        Statement('v1', '=', 'v1*N * 0', '', np.float),
        Statement('v1', '=', 'v1 * 0', '', np.float),
        Statement('v1', '=', 'v1 * 0.0', '', np.float),
        Statement('v1', '=', '0.0 / (v1*N)', '', np.float),
        # Should be simplified to 0
        Statement('i1', '=', 'i1*N * 0', '', np.int),
        Statement('i1', '=', '0 * i1', '', np.int),
        Statement('i1', '=', '0 * i1*N', '', np.int),
        Statement('i1', '=', 'i1 * 0', '', np.int),
        # Should be simplified to v1*N
        Statement('v2', '=', '0 + v1*N', '', np.float),
        Statement('v2', '=', 'v1*N + 0.0', '', np.float),
        Statement('v2', '=', 'v1*N - 0', '', np.float),
        Statement('v2', '=', 'v1*N - 0.0', '', np.float),
        Statement('v2', '=', '1 * v1*N', '', np.float),
        Statement('v2', '=', '1.0 * v1*N', '', np.float),
        Statement('v2', '=', 'v1*N / 1.0', '', np.float),
        Statement('v2', '=', 'v1*N / 1', '', np.float),
        # Should be simplified to i1
        Statement('i1', '=', 'i1*1', '', int),
        Statement('i1', '=', 'i1//1', '', int),
        Statement('i1', '=', 'i1+0', '', int),
        Statement('i1', '=', '0+i1', '', int),
        Statement('i1', '=', 'i1-0', '', int),
        # Should *not* be simplified (because it would change the type,
        # important for integer division, for example)
        Statement('v1', '=', 'i1*1.0', '', float),
        Statement('v1', '=', '1.0*i1', '', float),
        Statement('v1', '=', 'i1/1.0', '', float),
        Statement('v1', '=', 'i1/1'  , '', float),
        Statement('v1', '=', 'i1+0.0', '', float),
        Statement('v1', '=', '0.0+i1', '', float),
        Statement('v1', '=', 'i1-0.0', '', float),
        ## Should *not* be simplified, flooring division by 1 changes the value
        Statement('v1', '=', 'v2//1.0', '', float),
        Statement('i1', '=', 'i1//1.0', '', float)  # changes type
    ]
    scalar, vector = optimise_statements([], statements, variables)
    assert len(scalar) == 0
    for s in vector[:6]:
        assert s.expr == '0.0'
    for s in vector[6:10]:
        assert s.expr == '0',s.expr  # integer
    for s in vector[10:18]:
        expr = s.expr.replace(' ', '')
        assert expr == 'v1*N' or expr == 'N*v1'
    for s in vector[18:23]:
        expr = s.expr.replace(' ', '')
        assert expr == 'i1'
    for s in vector[23:27]:
        expr = s.expr.replace(' ', '')
        assert expr == '1.0*i1' or expr == 'i1*1.0' or expr == 'i1/1.0'
    for s in vector[27:30]:
        expr = s.expr.replace(' ', '')
        assert expr == '0.0+i1' or expr == 'i1+0.0'
    for s in vector[30:31]:
        expr = s.expr.replace(' ', '')
        assert expr == 'v2//1.0' or expr == 'v2//1'
    for s in vector[31:]:
        expr = s.expr.replace(' ', '')
        assert expr == 'i1//1.0'


@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_constant_evaluation():
    variables = {'v1': Variable('v1', scalar=False),
                 'v2': Variable('v2', scalar=False),
                 'i1': Variable('i1', scalar=False, dtype=int),
                 'N': Constant('N', 10),
                 's1': Variable('s1', scalar=True, dtype=float),
                 's2': Variable('s2', scalar=True, dtype=float),
                 'exp': DEFAULT_FUNCTIONS['exp']
                 }
    statements = [
        Statement('v1', '=', 'v1 * (1 + 2 + 3)', '', np.float),
        Statement('v1', '=', 'exp(N)*v1', '', np.float),
        Statement('v1', '=', 'exp(0)*v1', '', np.float),
    ]
    scalar, vector = optimise_statements([], statements, variables)
    # exp(N) should be pulled out of the vector statements, the rest should be
    # evaluated in place
    assert len(scalar) == 1
    assert scalar[0].expr == 'exp(N)'
    assert len(vector) == 3
    expr = vector[0].expr.replace(' ', '')
    assert expr == '_lio_1*v1' or 'v1*_lio_1'
    expr = vector[1].expr.replace(' ', '')
    assert expr == '6.0*v1' or 'v1*6.0'
    assert vector[2].expr == 'v1'


@attr('codegen-independent')
def test_automatic_augmented_assignments():
    # We test that statements that could be rewritten as augmented assignments
    # are correctly rewritten (using sympy to test for symbolic equality)
    variables = {
        'x': ArrayVariable('x', owner=None, size=10,
                           device=device),
        'y': ArrayVariable('y', owner=None, size=10,
                           device=device),
        'z': ArrayVariable('y', owner=None, size=10,
                           device=device),
        'b': ArrayVariable('b', owner=None, size=10,
                           dtype=np.bool, device=device),
        'clip': DEFAULT_FUNCTIONS['clip'],
        'inf': DEFAULT_CONSTANTS['inf']
    }
    statements = [
        # examples that should be rewritten
        # Note that using our approach, we will never get -= or /= but always
        # the equivalent += or *= statements
        ('x = x + 1', 'x += 1'),
        ('x = 2 * x', 'x *= 2'),
        ('x = x - 3', 'x += -3'),
        ('x = x/2', 'x *= 0.5'),
        ('x = y + (x + 1)', 'x += y + 1'),
        ('x = x + x', 'x *= 2'),
        ('x = x + y + z', 'x += y + z'),
        ('x = x + y + z', 'x += y + z'),
        # examples that should not be rewritten
        ('x = 1/x', 'x = 1/x'),
        ('x = 1', 'x = 1'),
        ('x = 2*(x + 1)', 'x = 2*(x + 1)'),
        ('x = clip(x + y, 0, inf)', 'x = clip(x + y, 0, inf)'),
        ('b = b or False', 'b = b or False')
    ]
    for orig, rewritten in statements:
        scalar, vector = make_statements(orig, variables, np.float32)
        try:  # we augment the assertion error with the original statement
            assert len(scalar) == 0, 'Did not expect any scalar statements but got ' + str(scalar)
            assert len(vector) == 1, 'Did expect a single statement but got ' + str(vector)
            statement = vector[0]
            expected_var, expected_op, expected_expr, _ = parse_statement(rewritten)
            assert expected_var == statement.var, 'expected write to variable %s, not to %s' % (expected_var, statement.var)
            assert expected_op == statement.op, 'expected operation %s, not %s' % (expected_op, statement.op)
            # Compare the two expressions using sympy to allow for different order etc.
            sympy_expected = str_to_sympy(expected_expr)
            sympy_actual = str_to_sympy(statement.expr)
            assert sympy_expected == sympy_actual, ('RHS expressions "%s" and "%s" are not identical' % (sympy_to_str(sympy_expected),
                                                                                                         sympy_to_str(sympy_actual)))
        except AssertionError as ex:
            raise AssertionError('Transformation for statement "%s" gave an unexpected result: %s' % (orig, str(ex)))


def test_clear_cache_numpy():
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')
    assert 'numpy' not in _cache_dirs_and_extensions
    assert_raises(ValueError, clear_cache, 'numpy')


def test_clear_cache_weave():
    if prefs.codegen.target != 'weave':
        raise SkipTest('weave-only test')
    assert 'weave' in _cache_dirs_and_extensions
    cache_dir, _ = _cache_dirs_and_extensions['weave']
    # Create a file that should not be there
    fname = os.path.join(cache_dir, 'some_file.py')
    open(fname, 'w').close()
    # clear_cache should refuse to clear the directory
    assert_raises(IOError, clear_cache, 'weave')

    os.remove(fname)


def test_clear_cache_cython():
    if prefs.codegen.target != 'cython':
        raise SkipTest('Cython-only test')
    assert 'cython' in _cache_dirs_and_extensions
    cache_dir, _ = _cache_dirs_and_extensions['cython']
    # Create a file that should not be there
    fname = os.path.join(cache_dir, 'some_file.py')
    open(fname, 'w').close()
    # clear_cache should refuse to clear the directory
    assert_raises(IOError, clear_cache, 'cython')

    os.remove(fname)


if __name__ == '__main__':
    test_auto_target()
    test_analyse_identifiers()
    test_get_identifiers_recursively()
    test_write_to_subexpression()
    test_repeated_subexpressions()
    test_nested_subexpressions()
    test_apply_loop_invariant_optimisation()
    test_apply_loop_invariant_optimisation_integer()
    test_apply_loop_invariant_optimisation_boolean()
    test_apply_loop_invariant_optimisation_no_optimisation()
    test_apply_loop_invariant_optimisation_simplification()
    test_apply_loop_invariant_optimisation_constant_evaluation()
    test_automatic_augmented_assignments()
    for t in [test_clear_cache_numpy,
              test_clear_cache_weave,
              test_clear_cache_cython]:
        try:
            t()
        except SkipTest:
            pass
