def euler(eqs):
    '''
    Euler integration
    '''
    s = []
    for var, expr in eqs.diff_eq_expressions:
        s.append('_temp_{var} = {expr}'.format(var=var, expr=expr))
    for var, expr in eqs.diff_eq_expressions:
        s.append('{var} += _temp_{var}*dt'.format(var=var, expr=expr))
    return '\n'.join(s)
