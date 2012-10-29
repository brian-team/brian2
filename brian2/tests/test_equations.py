# encoding: utf8
import sys
from StringIO import StringIO

from numpy.testing import assert_raises
from IPython.lib.pretty import pprint

from brian2 import volt, mV, second, ms, Hz, farad, metre, cm
from brian2 import Unit, Equations, Expression, sin
from brian2.units.fundamentalunits import (DIMENSIONLESS, get_dimensions,
                                           DimensionMismatchError,
                                           have_same_dimensions)
from brian2.equations.unitcheck import (get_default_unit_namespace,
                                        get_unit_from_string)
from brian2.equations.equations import (check_identifier,
                                        check_identifier_basic,
                                        check_identifier_reserved,
                                        parse_string_equations,
                                        SingleEquation,
                                        DIFFERENTIAL_EQUATION, STATIC_EQUATION,
                                        PARAMETER)



def test_utility_functions():
    unit_namespace = get_default_unit_namespace()
    
    # Some simple tests whether the namespace returned by
    # get_default_namespace() makes sense
    assert 'volt' in unit_namespace
    assert 'ms' in unit_namespace
    assert unit_namespace['ms'] is ms
    assert unit_namespace['ms'] is unit_namespace['msecond']
    for unit in unit_namespace.itervalues():
        assert isinstance(unit, Unit)
    
    assert get_unit_from_string('second') == second
    assert get_unit_from_string('1') == Unit(1, DIMENSIONLESS)
    assert get_unit_from_string('volt') == volt
    assert get_unit_from_string('second ** -1') == Hz
    assert get_unit_from_string('farad / metre**2') == farad / metre ** 2
    assert get_unit_from_string('m / s',
                                unit_namespace={'m': metre,
                                                's': second}) == metre/second
    assert_raises(ValueError, lambda: get_unit_from_string('metr / second'))
    assert_raises(ValueError, lambda: get_unit_from_string('metre **'))
    assert_raises(ValueError, lambda: get_unit_from_string('5'))
    assert_raises(ValueError, lambda: get_unit_from_string('2 / second'))        
    # Make sure that namespace overrides the default namespace
    assert_raises(ValueError, lambda: get_unit_from_string('metre / s',
                                                           unit_namespace={'m': metre,
                                                                           's': second}))
    assert get_unit_from_string('farad / cm**2') == farad / cm**2
    assert_raises(ValueError, lambda: get_unit_from_string('farad / cm**2',
                                                           only_base_units=True))


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
        assert_raises(ValueError, lambda: check_identifier_basic(identifier))

    for identifier in ['t', 'dt', 'xi']:
        assert_raises(ValueError, lambda: check_identifier_reserved(identifier))
    
    # Check identifier registry
    assert check_identifier_basic in Equations.identifier_checks
    assert check_identifier_reserved in Equations.identifier_checks
    
    # Set up a dummy identifier check that disallows the variable name
    # gaba_123 (that is otherwise valid)
    def disallow_gaba_123(identifier):
        if identifier == 'gaba_123':
            raise ValueError('I do not like this name')
    
    check_identifier('gaba_123')
    old_checks = Equations.identifier_checks
    Equations.register_identifier_check(disallow_gaba_123)
    assert_raises(ValueError, lambda: check_identifier('gaba_123'))
    Equations.identifier_checks = old_checks
    
    # registering a non-function should now work
    assert_raises(ValueError, lambda: Equations.register_identifier_check('no function'))

def test_parse_equations():
    ''' Test the parsing of equation strings '''
    # A simple equation
    eqs = parse_string_equations('dv/dt = -v / tau : 1', {}, False, 0)    
    assert len(eqs.keys()) == 1 and 'v' in eqs and eqs['v'].eq_type == DIFFERENTIAL_EQUATION
    assert get_dimensions(eqs['v'].unit) == DIMENSIONLESS
    
    # A complex one
    eqs = parse_string_equations('''dv/dt = -(v +
                                             ge + # excitatory conductance
                                             I # external current
                                             )/ tau : volt
                                    dge/dt = -ge / tau_ge : volt
                                    I = sin(2 * pi * f * t) : volt
                                    f : Hz (constant)
                                 ''', 
                                 {}, False, 0)
    assert len(eqs.keys()) == 4
    assert 'v' in eqs and eqs['v'].eq_type == DIFFERENTIAL_EQUATION
    assert 'ge' in eqs and eqs['ge'].eq_type == DIFFERENTIAL_EQUATION
    assert 'I' in eqs and eqs['I'].eq_type == STATIC_EQUATION
    assert 'f' in eqs and eqs['f'].eq_type == PARAMETER
    assert get_dimensions(eqs['v'].unit) == volt.dim
    assert get_dimensions(eqs['ge'].unit) == volt.dim
    assert get_dimensions(eqs['I'].unit) == volt.dim
    assert get_dimensions(eqs['f'].unit) == Hz.dim
    assert eqs['v'].flags == []
    assert eqs['ge'].flags == []
    assert eqs['I'].flags == []
    assert eqs['f'].flags == ['constant']
    
    duplicate_eqs = '''
    dv/dt = -v / tau : 1
    v = 2 * t : 1
    '''
    assert_raises(SyntaxError, lambda: parse_string_equations(duplicate_eqs,
                                                              {}, False, 0))
    parse_error_eqs = [
    '''dv/d = -v / tau : 1
        x = 2 * t : 1''',
    '''dv/dt = -v / tau : 1 : volt
    x = 2 * t : 1''',
    ''' dv/dt = -v / tau : 2 * volt''']
    for error_eqs in parse_error_eqs:
        assert_raises((ValueError, SyntaxError), lambda: parse_string_equations(error_eqs,
                                                                                {}, False, 0))

def test_construction_errors():
    '''
    Test that the Equations constructor raises errors correctly
    '''
    # parse error
    assert_raises(SyntaxError, lambda: Equations('dv/dt = -v / tau volt'))
    
    # duplicate variable names
    assert_raises(SyntaxError, lambda: Equations('''dv/dt = -v / tau : volt
                                                    v = 2 * t/second * volt : volt'''))
    
    # illegal variable names
    assert_raises(ValueError, lambda: Equations('ddt/dt = -dt / tau : volt'))
    assert_raises(ValueError, lambda: Equations('dt/dt = -t / tau : volt'))
    assert_raises(ValueError, lambda: Equations('dxi/dt = -xi / tau : volt'))
    assert_raises(ValueError, lambda: Equations('for : volt'))
    assert_raises((SyntaxError, ValueError),
                  lambda: Equations('d1a/dt = -1a / tau : volt'))
    assert_raises(ValueError, lambda: Equations('d_x/dt = -_x / tau : volt'))
    
    # inconsistent unit for a differential equation
    assert_raises(DimensionMismatchError,
                  lambda: Equations('dv/dt = -v : volt'))
    assert_raises(DimensionMismatchError,
                  lambda: Equations('dv/dt = -v / tau: volt',
                                    namespace={'tau': 5 * mV}))
    assert_raises(DimensionMismatchError,
                  lambda: Equations('dv/dt = -(v + I) / (5 * ms): volt',
                                    namespace={'I': 3 * second}))    
    
    # inconsistent unit for a static equation
    assert_raises(DimensionMismatchError,
                  lambda: Equations('''dv/dt = -v / (5 * ms) : volt
                                       I = 2 * v : amp'''))
    
    # xi in a static equation
    assert_raises(SyntaxError,
                  lambda: Equations('''dv/dt = -(v + I) / (5 * ms) : volt
                                       I = second**-1*xi**-2*volt : volt''' ))
    
    # more than one xi    
    assert_raises(SyntaxError,                  
                  lambda: Equations('''dv/dt = -v / tau + xi/tau**.5 : volt
                                       dx/dt = -x / tau + 2*xi/tau : volt
                                       tau : second'''))
    # using not-allowed flags
    eqs = Equations('dv/dt = -v / (5 * ms) : volt (flag)')    
    eqs.check_flags({DIFFERENTIAL_EQUATION: ['flag']}) # allow this flag
    assert_raises(ValueError, lambda: eqs.check_flags({DIFFERENTIAL_EQUATION: []}))
    assert_raises(ValueError, lambda: eqs.check_flags({}))
    assert_raises(ValueError, lambda: eqs.check_flags({STATIC_EQUATION: ['flag']}))
    assert_raises(ValueError, lambda: eqs.check_flags({DIFFERENTIAL_EQUATION: ['otherflag']}))
    
    # Circular static equations
    assert_raises(ValueError, lambda: Equations('''dv/dt = -(v + w) / (10 * ms) : 1
                                                   w = 2 * x : 1
                                                   x = 3 * w : 1'''))
    

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
    assert eqs.diff_eq_names == ['v']
    assert (len(eqs.eq_expressions) == 3 and
            set([name for name, _ in eqs.eq_expressions]) == set(['v', 'I', 'f']) and
            all((isinstance(expr, Expression) for _, expr in eqs.eq_expressions)))
    assert len(eqs.eq_names) == 3 and set(eqs.eq_names) == set(['v', 'I', 'f'])
    assert set(eqs.equations.keys()) == set(['v', 'I', 'f', 'freq'])
    assert all((isinstance(eq, SingleEquation) for eq in eqs.equations.itervalues()))
    # test that the equations object is iterable itself
    assert all((isinstance(eq, SingleEquation) for (_, eq) in eqs))
    assert (len(eqs.equations_ordered) == 4 and
            all((isinstance(eq, SingleEquation) for eq in eqs.equations_ordered)) and
            [eq.varname for eq in eqs.equations_ordered] == ['f', 'I', 'v', 'freq'])
    assert set(eqs.names) == set(['v', 'I', 'f', 'freq'])
    assert set(eqs.parameter_names) == set(['freq'])
    assert set(eqs.static_eq_names) == set(['I', 'f'])
    units = eqs.units
    assert set(units.keys()) == set(['v', 'I', 'f', 'freq', 't', 'dt', 'xi'])
    assert units['v'] == volt
    assert units['I'] == volt
    assert units['f'] == Hz
    assert have_same_dimensions(units['freq'], 1)
    assert units['t'] == second
    assert units['dt'] == second
    assert units['xi'] == second**-0.5
    assert set(eqs.variables) == set(eqs.units.keys())


def test_mathematical_properties():
    tau = 10 * ms
    eqs = Equations('''dv/dt = -v / tau : volt ''')
    assert eqs.is_linear
    
    # depending on constant parameters is ok
    eqs = Equations('''dv/dt = -v*c / tau : volt
                       c : 1 (constant)''')
    assert eqs.is_linear

    # depending on non-constant parameters isn't
    eqs = Equations('''dv/dt = -v*c / tau : volt
                       c : 1''')
    assert not eqs.is_linear
    
    eqs = Equations('''dv/dt = -sin(v) / tau : 1''')
    assert not eqs.is_linear
    
    # equations depending on time are never linear
    eqs = Equations('''dv/dt = -(v * t / second) / tau : 1''')
    assert not eqs.is_linear
    
    
    eqs = Equations('''dv/dt = -(v + v2) / tau : 1
                       v2 = sin(v) : 1''')
    assert not eqs.is_linear
    assert not eqs.is_conditionally_linear
    
    eqs = Equations('''dv/dt = -(v + v2) / tau : 1
                       dv2/dt = sin(v) / tau : 1''')
    assert not eqs.is_linear
    assert eqs.is_conditionally_linear


def test_str_repr():
    '''
    Test the string representation (only that it does not throw errors).
    '''
    tau = 10 * ms
    eqs = Equations('''dv/dt = -(v + I)/ tau : volt (active)
                       I = sin(2 * 22/7. * f * t)* volt : volt
                       f : Hz''')
    assert len(str(eqs)) > 0
    assert len(repr(eqs)) > 0
    
    # Test str and repr of SingleEquations explicitly (might already have been
    # called by Equations
    for eq in eqs.equations.itervalues():
        assert(len(str(eq))) > 0
        assert(len(repr(eq))) > 0
    
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
    test_construction_errors()
    test_properties()
    test_mathematical_properties()    
    test_str_repr()
