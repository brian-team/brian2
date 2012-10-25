from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION)
from brian2.units import DimensionMismatchError

__all__ = ['add_refractoriness']

def add_refractoriness(eqs):
    neweqs = ''
    for eq in eqs.equations.values():
        if eq.eq_type == DIFFERENTIAL_EQUATION:
            s = 'd' + eq.varname + '/dt'
        else:
            s = eq.varname
        
        expr = eq.expr
        
        if not expr is None:
            expr = str(expr)
            if eq.eq_type==DIFFERENTIAL_EQUATION and 'active' in eq.flags:
                expr = 'is_active*('+expr+')'
            s += ' = ' + str(expr)
        
        u = eq.unit
        try:
            1+u
            su = '1'
        except DimensionMismatchError:
            su = repr(u)
        s += ' : ' + su
        
        if len(eq.flags):
            s += ' (' + ', '.join(eq.flags) + ')'
        
        neweqs += s+'\n'
        
    neweqs += 'is_active : 1\n'
    neweqs += 'refractory : second\n'
    neweqs += 'refractory_until : second\n'
        
    return neweqs
        
if __name__=='__main__':
    eqs = Equations('''
    dv/dt = -x*v/second : volt (active)
    dw/dt = -w/second : amp
    x : 1
    ''')
    eqs = add_refractoriness(eqs)
    eqs = Equations(eqs)
    print eqs