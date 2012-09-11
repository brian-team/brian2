def euler(eqs):
    '''
    Euler integration
    '''
    s = []
    for var, expr in eqs.diff_eq_expressions.iteritems():
        s.append('_temp_{var} = {expr}'.format(var=var, expr=expr.frozen()))
    for var, expr in eqs.diff_eq_expressions.iteritems():
        s.append('{var} += _temp_{var}*dt'.format(var=var, expr=expr.frozen()))
    return '\n'.join(s)
