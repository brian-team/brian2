{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { Cm, dt, v, N,
                        ab_star, b_plus, ab_plus, b_minus, ab_minus,
                        v_star, u_plus, u_minus} #}


    cdef double[:] _gtot_all = _numpy.zeros(N, dtype=_numpy.double)
    cdef double[:] c = _numpy.zeros(N, dtype=_numpy.double)
    cdef double ai
    cdef double bi
    cdef double _m

    cdef int _i

    ## MAIN CODE ######
    # Tridiagonal solving
    # Pass 1
    for _i in range(N):
        _idx = _i
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}
        _gtot_all[_idx] = _gtot

        {{v_star}}[_i] = -({{Cm}}[_i]/dt*{{v}}[_i])-_I0 # RHS -> v_star (solution)
        bi={{ab_star}}[N + _i] - _gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i]= {{ab_star}}[0 + _i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_star}}[2*N + _i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[_i-1])
            c[_i] = c[_i]*_m
            {{v_star}}[_i] = ({{v_star}}[_i] - ai*{{v_star}}[_i-1])*_m
        else:
            c[0]=c[0]/bi
            {{v_star}}[0]={{v_star}}[0]/bi
    for _i in range(N-2, -1, -1):
        {{v_star}}[_i]={{v_star}}[_i] - c[_i]*{{v_star}}[_i+1]
    
    # Pass 2
    for _i in range(N):
        {{u_plus}}[_i] = {{b_plus}}[_i] # RHS -> v_star (solution)
        bi = {{ab_plus}}[N + _i]-_gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i] = {{ab_plus}}[0 + _i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_plus}}[2*N + _i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[_i-1])
            c[_i] = c[_i]*_m
            {{u_plus}}[_i] = ({{u_plus}}[_i] - ai*{{u_plus}}[_i-1])*_m
        else:
            c[0]=c[0]/bi
            {{u_plus}}[0] = {{u_plus}}[0]/bi
    for _i in range(N-2, -1, -1):
        {{u_plus}}[_i] = {{u_plus}}[_i] - c[_i]*{{u_plus}}[_i+1]
    
    # Pass 3
    for _i in range(N):
        {{u_minus}}[_i] = {{b_minus}}[_i] # RHS -> v_star (solution)
        bi = {{ab_minus}}[N + _i] - _gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i] = {{ab_minus}}[0 + _i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_minus}}[2*N + _i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[_i-1])
            c[_i] = c[_i]*_m
            {{u_minus}}[_i] = ({{u_minus}}[_i] - ai*{{u_minus}}[_i-1])*_m
        else:
            c[0] = c[0]/bi
            {{u_minus}}[0] = {{u_minus}}[0]/bi
    for _i in range(N-2, -1, -1):
        {{u_minus}}[_i] = {{u_minus}}[_i] - c[_i]*{{u_minus}}[_i+1]

{% endblock %}
