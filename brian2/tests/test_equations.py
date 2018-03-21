# encoding: utf8
import sys
from StringIO import StringIO

from numpy.testing import assert_raises
import numpy as np
try:
    from IPython.lib.pretty import pprint
except ImportError:
    pprint = None
from nose import SkipTest
from nose.plugins.attrib import attr

from brian2 import volt, mV, second, ms, Hz, farad, metre
from brian2 import Unit, Equations, Expression
from brian2.units.fundamentalunits import (DIMENSIONLESS, get_dimensions,
                                           DimensionMismatchError)
from brian2.core.namespace import DEFAULT_UNITS
from brian2.equations.equations import (check_identifier_basic,
                                        check_identifier_reserved,
                                        check_identifier_functions,
                                        check_identifier_constants,
                                        check_identifier_units,
                                        parse_string_equations,
                                        dimensions_and_type_from_string,
                                        SingleEquation,
                                        DIFFERENTIAL_EQUATION, SUBEXPRESSION,
                                        PARAMETER, FLOAT, BOOLEAN, INTEGER,
                                        EquationError,
                                        extract_constant_subexpressions
                                        )
from brian2.equations.refractory import check_identifier_refractory
from brian2.groups.group import Group


# a simple Group for testing
class SimpleGroup(Group):
    def __init__(self, variables, namespace=None):
        self.variables = variables
        self.namespace = namespace

@attr('codegen-independent')
def test_utility_functions():
    unit_namespace = DEFAULT_UNITS

    # Some simple tests whether the namespace returned by
    # get_default_namespace() makes sense
    assert 'volt' in unit_namespace
    assert 'ms' in unit_namespace
    assert unit_namespace['ms'] is ms
    assert unit_namespace['ms'] is unit_namespace['msecond']
    for unit in unit_namespace.itervalues():
        assert isinstance(unit, Unit)

    assert dimensions_and_type_from_string('second') == (second.dim, FLOAT)
    assert dimensions_and_type_from_string('1') == (DIMENSIONLESS, FLOAT)
    assert dimensions_and_type_from_string('volt') == (volt.dim, FLOAT)
    assert dimensions_and_type_from_string('second ** -1') == (Hz.dim, FLOAT)
    assert dimensions_and_type_from_string('farad / metre**2') == ((farad / metre ** 2).dim, FLOAT)
    assert dimensions_and_type_from_string('boolean') == (DIMENSIONLESS, BOOLEAN)
    assert dimensions_and_type_from_string('integer') == (DIMENSIONLESS, INTEGER)
    assert_raises(ValueError, lambda: dimensions_and_type_from_string('metr / second'))
    assert_raises(ValueError, lambda: dimensions_and_type_from_string('metre **'))
    assert_raises(ValueError, lambda: dimensions_and_type_from_string('5'))
    assert_raises(ValueError, lambda: dimensions_and_type_from_string('2 / second'))
    # Only the use of base units is allowed
    assert_raises(ValueError, lambda: dimensions_and_type_from_string('farad / cm**2'))


@attr('codegen-independent')
def test_identifier_checks():
    legal_identifiers = ['v', 'Vm', 'V', 'x', 'ge', 'g_i', 'a2', 'gaba_123']
    illegal_identifiers = ['_v', '1v', u'ü', 'ge!', 'v.x', 'for', 'else', 'if']

    for identifier in legal_identifiers:
        try:
            check_identifier_basic(identifier)
            check_identifier_reserved(identifier)
        except ValueError as ex:
            raise AssertionError('check complained about '
                                 'identifier "%s": %s' % (identifier, ex))

    for identifier in illegal_identifiers:
        assert_raises(SyntaxError, lambda: check_identifier_basic(identifier))

    for identifier in ('t', 'dt', 'xi', 'i', 'N'):
        assert_raises(SyntaxError, lambda: check_identifier_reserved(identifier))

    for identifier in ('not_refractory', 'refractory', 'refractory_until'):
        assert_raises(SyntaxError, lambda: check_identifier_refractory(identifier))

    for identifier in ('exp', 'sin', 'sqrt'):
        assert_raises(SyntaxError, lambda: check_identifier_functions(identifier))

    for identifier in ('e', 'pi', 'inf'):
        assert_raises(SyntaxError, lambda: check_identifier_constants(identifier))

    for identifier in ('volt', 'second', 'mV', 'nA'):
        assert_raises(SyntaxError, lambda: check_identifier_units(identifier))

    # Check identifier registry
    assert check_identifier_basic in Equations.identifier_checks
    assert check_identifier_reserved in Equations.identifier_checks
    assert check_identifier_refractory in Equations.identifier_checks
    assert check_identifier_functions in Equations.identifier_checks
    assert check_identifier_constants in Equations.identifier_checks
    assert check_identifier_units in Equations.identifier_checks

    # Set up a dummy identifier check that disallows the variable name
    # gaba_123 (that is otherwise valid)
    def disallow_gaba_123(identifier):
        if identifier == 'gaba_123':
            raise SyntaxError('I do not like this name')

    Equations.check_identifier('gaba_123')
    old_checks = set(Equations.identifier_checks)
    Equations.register_identifier_check(disallow_gaba_123)
    assert_raises(SyntaxError, lambda: Equations.check_identifier('gaba_123'))
    Equations.identifier_checks = old_checks

    # registering a non-function should not work
    assert_raises(ValueError,
                  lambda: Equations.register_identifier_check('no function'))

@attr('codegen-independent')
def test_parse_equations():
    ''' Test the parsing of equation strings '''
    # A simple equation
    eqs = parse_string_equations('dv/dt = -v / tau : 1')
    assert len(eqs.keys()) == 1 and 'v' in eqs and eqs['v'].type == DIFFERENTIAL_EQUATION
    assert eqs['v'].dim is DIMENSIONLESS

    # A complex one
    eqs = parse_string_equations('''dv/dt = -(v +
                                             ge + # excitatory conductance
                                             I # external current
                                             )/ tau : volt
                                    dge/dt = -ge / tau_ge : volt
                                    I = sin(2 * pi * f * t) : volt
                                    f : Hz (constant)
                                    b : boolean
                                    n : integer
                                 ''')
    assert len(eqs.keys()) == 6
    assert 'v' in eqs and eqs['v'].type == DIFFERENTIAL_EQUATION
    assert 'ge' in eqs and eqs['ge'].type == DIFFERENTIAL_EQUATION
    assert 'I' in eqs and eqs['I'].type == SUBEXPRESSION
    assert 'f' in eqs and eqs['f'].type == PARAMETER
    assert 'b' in eqs and eqs['b'].type == PARAMETER
    assert 'n' in eqs and eqs['n'].type == PARAMETER
    assert eqs['f'].var_type == FLOAT
    assert eqs['b'].var_type == BOOLEAN
    assert eqs['n'].var_type == INTEGER
    assert eqs['v'].dim is volt.dim
    assert eqs['ge'].dim is volt.dim
    assert eqs['I'].dim is volt.dim
    assert eqs['f'].dim is Hz.dim
    assert eqs['v'].flags == []
    assert eqs['ge'].flags == []
    assert eqs['I'].flags == []
    assert eqs['f'].flags == ['constant']

    duplicate_eqs = '''
    dv/dt = -v / tau : 1
    v = 2 * t : 1
    '''
    assert_raises(EquationError, lambda: parse_string_equations(duplicate_eqs))
    parse_error_eqs = [
    '''dv/d = -v / tau : 1
        x = 2 * t : 1''',
    '''dv/dt = -v / tau : 1 : volt
    x = 2 * t : 1''',
    ''' dv/dt = -v / tau : 2 * volt''',
    'dv/dt = v / second : boolean']
    for error_eqs in parse_error_eqs:
        assert_raises((ValueError, EquationError, TypeError),
                      lambda: parse_string_equations(error_eqs))


@attr('codegen-independent')
def test_correct_replacements():
    ''' Test replacing variables via keyword arguments '''
    # replace a variable name with a new name
    eqs = Equations('dv/dt = -v / tau : 1', v='V')
    # Correct left hand side
    assert ('V' in eqs) and not ('v' in eqs)
    # Correct right hand side
    assert ('V' in eqs['V'].identifiers) and not ('v' in eqs['V'].identifiers)

    # replace a variable name with a value
    eqs = Equations('dv/dt = -v / tau : 1', tau=10 * ms)
    assert not 'tau' in eqs['v'].identifiers


@attr('codegen-independent')
def test_wrong_replacements():
    '''Tests for replacements that should not work'''
    # Replacing a variable name with an illegal new name
    assert_raises(SyntaxError, lambda: Equations('dv/dt = -v / tau : 1',
                                                 v='illegal name'))
    assert_raises(SyntaxError, lambda: Equations('dv/dt = -v / tau : 1',
                                                 v='_reserved'))
    assert_raises(SyntaxError, lambda: Equations('dv/dt = -v / tau : 1',
                                                 v='t'))

    # Replacing a variable name with a value that already exists
    assert_raises(EquationError, lambda: Equations('''
                                                    dv/dt = -v / tau : 1
                                                    dx/dt = -x / tau : 1
                                                    ''',
                                                   v='x'))

    # Replacing a model variable name with a value
    assert_raises(ValueError, lambda: Equations('dv/dt = -v / tau : 1',
                                                v=3 * mV))

    # Replacing with an illegal value
    assert_raises(SyntaxError, lambda: Equations('dv/dt = -v/tau : 1',
                                                 tau=np.arange(5)))


@attr('codegen-independent')
def test_substitute():
    # Check that Equations.substitute returns an independent copy
    eqs = Equations('dx/dt = x : 1')
    eqs2 = eqs.substitute(x='y')

    # First equation should be unaffected
    assert len(eqs) == 1 and 'x' in eqs
    assert eqs['x'].expr == Expression('x')

    # Second equation should have x substituted by y
    assert len(eqs2) == 1 and 'y' in eqs2
    assert eqs2['y'].expr == Expression('y')



@attr('codegen-independent')
def test_construction_errors():
    '''
    Test that the Equations constructor raises errors correctly
    '''
    # parse error
    assert_raises(EquationError, lambda: Equations('dv/dt = -v / tau volt'))
    assert_raises(EquationError, lambda: Equations('dv/dt = -v / tau : volt second'))

    # incorrect unit definition
    assert_raises(EquationError, lambda: Equations('dv/dt = -v / tau : mvolt'))
    assert_raises(EquationError, lambda: Equations('dv/dt = -v / tau : voltage'))
    assert_raises(EquationError, lambda: Equations('dv/dt = -v / tau : 1.0*volt'))

    # Only a single string or a list of SingleEquation objects is allowed
    assert_raises(TypeError, lambda: Equations(None))
    assert_raises(TypeError, lambda: Equations(42))
    assert_raises(TypeError, lambda: Equations(['dv/dt = -v / tau : volt']))

    # duplicate variable names
    assert_raises(EquationError, lambda: Equations('''dv/dt = -v / tau : volt
                                                    v = 2 * t/second * volt : volt'''))

    eqs = [SingleEquation(DIFFERENTIAL_EQUATION, 'v', volt.dim,
                          expr=Expression('-v / tau')),
           SingleEquation(SUBEXPRESSION, 'v', volt.dim,
                          expr=Expression('2 * t/second * volt'))
           ]
    assert_raises(EquationError, lambda: Equations(eqs))

    # illegal variable names
    assert_raises(SyntaxError, lambda: Equations('ddt/dt = -dt / tau : volt'))
    assert_raises(SyntaxError, lambda: Equations('dt/dt = -t / tau : volt'))
    assert_raises(SyntaxError, lambda: Equations('dxi/dt = -xi / tau : volt'))
    assert_raises(SyntaxError, lambda: Equations('for : volt'))
    assert_raises((EquationError, SyntaxError),
                  lambda: Equations('d1a/dt = -1a / tau : volt'))
    assert_raises(SyntaxError, lambda: Equations('d_x/dt = -_x / tau : volt'))

    # xi in a subexpression
    assert_raises(EquationError,
                  lambda: Equations('''dv/dt = -(v + I) / (5 * ms) : volt
                                       I = second**-1*xi**-2*volt : volt'''))

    # more than one xi
    assert_raises(EquationError,
                  lambda: Equations('''dv/dt = -v / tau + xi/tau**.5 : volt
                                       dx/dt = -x / tau + 2*xi/tau : volt
                                       tau : second'''))
    # using not-allowed flags
    eqs = Equations('dv/dt = -v / (5 * ms) : volt (flag)')
    eqs.check_flags({DIFFERENTIAL_EQUATION: ['flag']})  # allow this flag
    assert_raises(ValueError, lambda: eqs.check_flags({DIFFERENTIAL_EQUATION: []}))
    assert_raises(ValueError, lambda: eqs.check_flags({}))
    assert_raises(ValueError, lambda: eqs.check_flags({SUBEXPRESSION: ['flag']}))
    assert_raises(ValueError, lambda: eqs.check_flags({DIFFERENTIAL_EQUATION: ['otherflag']}))
    eqs = Equations('dv/dt = -v / (5 * ms) : volt (flag1, flag2)')
    eqs.check_flags({DIFFERENTIAL_EQUATION: ['flag1', 'flag2']})  # allow both flags
    # Don't allow the two flags in combination
    assert_raises(ValueError,
                  lambda: eqs.check_flags({DIFFERENTIAL_EQUATION: ['flag1', 'flag2']},
                                          incompatible_flags=[('flag1', 'flag2')]))
    eqs = Equations('''dv/dt = -v / (5 * ms) : volt (flag1)
                       dw/dt = -w / (5 * ms) : volt (flag2)''')
    # They should be allowed when used independently
    eqs.check_flags({DIFFERENTIAL_EQUATION: ['flag1', 'flag2']},
                    incompatible_flags=[('flag1', 'flag2')])

    # Circular subexpression
    assert_raises(ValueError, lambda: Equations('''dv/dt = -(v + w) / (10 * ms) : 1
                                                   w = 2 * x : 1
                                                   x = 3 * w : 1'''))

    # Boolean/integer differential equations
    assert_raises(TypeError, lambda: Equations('dv/dt = -v / (10*ms) : boolean'))
    assert_raises(TypeError, lambda: Equations('dv/dt = -v / (10*ms) : integer'))


@attr('codegen-independent')
def test_unit_checking():
    # dummy Variable class
    class S(object):
        def __init__(self, dimensions):
            self.dim = get_dimensions(dimensions)

    # inconsistent unit for a differential equation
    eqs = Equations('dv/dt = -v : volt')
    group = SimpleGroup({'v': S(volt)})
    assert_raises(DimensionMismatchError,
                  lambda: eqs.check_units(group, {}))

    eqs = Equations('dv/dt = -v / tau: volt')
    group = SimpleGroup(namespace={'tau': 5*mV}, variables={'v': S(volt)})
    assert_raises(DimensionMismatchError,
                  lambda: eqs.check_units(group, {}))
    group = SimpleGroup(namespace={'I': 3*second}, variables={'v': S(volt)})
    eqs = Equations('dv/dt = -(v + I) / (5 * ms): volt')
    assert_raises(DimensionMismatchError,
                  lambda: eqs.check_units(group, {}))

    eqs = Equations('''dv/dt = -(v + I) / (5 * ms): volt
                       I : second''')
    group = SimpleGroup(variables={'v': S(volt),
                                   'I': S(second)}, namespace={})
    assert_raises(DimensionMismatchError,
                  lambda: eqs.check_units(group, {}))
    
    # inconsistent unit for a subexpression
    eqs = Equations('''dv/dt = -v / (5 * ms) : volt
                       I = 2 * v : amp''')
    group = SimpleGroup(variables={'v': S(volt),
                                   'I': S(second)}, namespace={})
    assert_raises(DimensionMismatchError,
                  lambda: eqs.check_units(group, {}))
    

@attr('codegen-independent')
def test_properties():
    '''
    Test accessing the various properties of equation objects
    '''
    tau = 10 * ms
    eqs = Equations('''dv/dt = -(v + I)/ tau : volt
                       I = sin(2 * 22/7. * f * t)* volt : volt
                       f = freq * Hz: Hz
                       freq : 1''')
    assert (len(eqs.diff_eq_expressions) == 1 and
            eqs.diff_eq_expressions[0][0] == 'v' and
            isinstance(eqs.diff_eq_expressions[0][1], Expression))
    assert eqs.diff_eq_names == {'v'}
    assert (len(eqs.eq_expressions) == 3 and
            set([name for name, _ in eqs.eq_expressions]) == {'v', 'I', 'f'} and
            all((isinstance(expr, Expression) for _, expr in eqs.eq_expressions)))
    assert len(eqs.eq_names) == 3 and eqs.eq_names == {'v', 'I', 'f'}
    assert set(eqs.keys()) == {'v', 'I', 'f', 'freq'}
    # test that the equations object is iterable itself
    assert all((isinstance(eq, SingleEquation) for eq in eqs.itervalues()))
    assert all((isinstance(eq, basestring) for eq in eqs))
    assert (len(eqs.ordered) == 4 and
            all((isinstance(eq, SingleEquation) for eq in eqs.ordered)) and
            [eq.varname for eq in eqs.ordered] == ['f', 'I', 'v', 'freq'])
    assert [eq.unit for eq in eqs.ordered] == [Hz, volt, volt, 1]
    assert eqs.names == {'v', 'I', 'f', 'freq'}
    assert eqs.parameter_names == {'freq'}
    assert eqs.subexpr_names == {'I', 'f'}
    dimensions = eqs.dimensions
    assert set(dimensions.keys()) == {'v', 'I', 'f', 'freq'}
    assert dimensions['v'] is volt.dim
    assert dimensions['I'] is volt.dim
    assert dimensions['f'] is Hz.dim
    assert dimensions['freq'] is DIMENSIONLESS
    assert eqs.names == set(eqs.dimensions.keys())
    assert eqs.identifiers == {'tau', 'volt', 'Hz', 'sin', 't'}

    # stochastic equations
    assert len(eqs.stochastic_variables) == 0
    assert eqs.stochastic_type is None
    
    eqs = Equations('''dv/dt = -v / tau + 0.1*second**-.5*xi : 1''')
    assert eqs.stochastic_variables == {'xi'}
    assert eqs.stochastic_type == 'additive'
    
    eqs = Equations('''dv/dt = -v / tau + 0.1*second**-.5*xi_1 +  0.1*second**-.5*xi_2: 1''')
    assert eqs.stochastic_variables == {'xi_1', 'xi_2'}
    assert eqs.stochastic_type == 'additive'
    
    eqs = Equations('''dv/dt = -v / tau + 0.1*second**-1.5*xi*t : 1''')
    assert eqs.stochastic_type == 'multiplicative'

    eqs = Equations('''dv/dt = -v / tau + 0.1*second**-1.5*xi*v : 1''')
    assert eqs.stochastic_type == 'multiplicative'


@attr('codegen-independent')
def test_concatenation():
    eqs1 = Equations('''dv/dt = -(v + I) / tau : volt
                        I = sin(2*pi*freq*t) : volt
                        freq : Hz''')

    # Concatenate two equation objects
    eqs2 = (Equations('dv/dt = -(v + I) / tau : volt') +
            Equations('''I = sin(2*pi*freq*t) : volt
                         freq : Hz'''))

    # Concatenate using "in-place" addition (which is not actually in-place)
    eqs3 = Equations('dv/dt = -(v + I) / tau : volt')
    eqs3 += Equations('''I = sin(2*pi*freq*t) : volt
                         freq : Hz''')

    # Concatenate with a string (will be parsed first)
    eqs4 = Equations('dv/dt = -(v + I) / tau : volt')
    eqs4 += '''I = sin(2*pi*freq*t) : volt
               freq : Hz'''

    # Concatenating with something that is not a string should not work
    assert_raises(TypeError, lambda: eqs4 + 5)

    # The string representation is canonical, therefore it should be identical
    # in all cases
    assert str(eqs1) == str(eqs2)
    assert str(eqs2) == str(eqs3)
    assert str(eqs3) == str(eqs4)


@attr('codegen-independent')
def test_extract_subexpressions():
    eqs = Equations('''dv/dt = -v / (10*ms) : 1
                       s1 = 2*v : 1
                       s2 = -v : 1 (constant over dt)
                    ''')
    variable, constant = extract_constant_subexpressions(eqs)
    assert [var in variable for var in ['v', 's1', 's2']]
    assert variable['s1'].type == SUBEXPRESSION
    assert variable['s2'].type == PARAMETER
    assert constant['s2'].type == SUBEXPRESSION


@attr('codegen-independent')
def test_repeated_construction():
    eqs1 = Equations('dx/dt = x : 1')
    eqs2 = Equations('dx/dt = x : 1', x='y')
    assert len(eqs1) == 1
    assert 'x' in eqs1
    assert eqs1['x'].expr == Expression('x')
    assert len(eqs2) == 1
    assert 'y' in eqs2
    assert eqs2['y'].expr == Expression('y')


@attr('codegen-independent')
def test_str_repr():
    '''
    Test the string representation (only that it does not throw errors).
    '''
    tau = 10 * ms
    eqs = Equations('''dv/dt = -(v + I)/ tau : volt (unless refractory)
                       I = sin(2 * 22/7. * f * t)* volt : volt
                       f : Hz''')
    assert len(str(eqs)) > 0
    assert len(repr(eqs)) > 0

    # Test str and repr of SingleEquations explicitly (might already have been
    # called by Equations
    for eq in eqs.itervalues():
        assert(len(str(eq))) > 0
        assert(len(repr(eq))) > 0

@attr('codegen-independent')
def test_ipython_pprint():
    if pprint is None:
        raise SkipTest('ipython is not available')
    eqs = Equations('''dv/dt = -(v + I)/ tau : volt (unless refractory)
                       I = sin(2 * 22/7. * f * t)* volt : volt
                       f : Hz''')
    # Test ipython's pretty printing
    old_stdout = sys.stdout
    string_output = StringIO()
    sys.stdout = string_output
    pprint(eqs)
    assert len(string_output.getvalue()) > 0
    sys.stdout = old_stdout


if __name__ == '__main__':
    test_utility_functions()
    test_identifier_checks()
    test_parse_equations()
    test_correct_replacements()
    test_substitute()
    test_wrong_replacements()
    test_construction_errors()
    test_concatenation()
    test_unit_checking()
    test_properties()
    test_extract_subexpressions()
    test_repeated_construction()
    test_str_repr()
