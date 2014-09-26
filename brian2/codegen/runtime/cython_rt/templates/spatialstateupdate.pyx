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

    cdef int i

    ## MAIN CODE ######
    # Tridiagonal solving
    # Pass 1
    for i in range(N):
        _idx = i
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}
        _gtot_all[_idx] = _gtot

        {{v_star}}[i] = -({{Cm}}[i]/dt*{{v}}[i])-_I0 # RHS -> v_star (solution)
        bi={{ab_star}}[N + i] - _gtot_all[i] # main diagonal
        if (i<N-1):
            c[i]= {{ab_star}}[0 + i+1] # superdiagonal
        if (i>0):
            ai = {{ab_star}}[2*N + i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[i-1])
            c[i] = c[i]*_m
            {{v_star}}[i] = ({{v_star}}[i] - ai*{{v_star}}[i-1])*_m
        else:
            c[0]=c[0]/bi
            {{v_star}}[0]={{v_star}}[0]/bi
    for i in range(N-2, 0, -1):
        {{v_star}}[i]={{v_star}}[i] - c[i]*{{v_star}}[i+1]
    
    # Pass 2
    for i in range(N):
        {{u_plus}}[i] = {{b_plus}}[i] # RHS -> v_star (solution)
        bi = {{ab_plus}}[N + i]-_gtot_all[i] # main diagonal
        if (i<N-1):
            c[i] = {{ab_plus}}[0 + i+1] # superdiagonal
        if (i>0):
            ai = {{ab_plus}}[2*N + i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[i-1])
            c[i] = c[i]*_m
            {{u_plus}}[i] = ({{u_plus}}[i] - ai*{{u_plus}}[i-1])*_m
        else:
            c[0]=c[0]/bi
            {{u_plus}}[0] = {{u_plus}}[0]/bi
    for i in range(N-2, 0, -1):
        {{u_plus}}[i] = {{u_plus}}[i] - c[i]*{{u_plus}}[i+1]
    
    # Pass 3
    for i in range(N):
        {{u_minus}}[i] = {{b_minus}}[i] # RHS -> v_star (solution)
        bi = {{ab_minus}}[N + i] - _gtot_all[i] # main diagonal
        if (i<N-1):
            c[i] = {{ab_minus}}[0 + i+1] # superdiagonal
        if (i>0):
            ai = {{ab_minus}}[2*N + i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[i-1])
            c[i] = c[i]*_m
            {{u_minus}}[i] = ({{u_minus}}[i] - ai*{{u_minus}}[i-1])*_m
        else:
            c[0] = c[0]/bi
            {{u_minus}}[0] = {{u_minus}}[0]/bi
    for i in range(N-2, 0, -1):
        {{u_minus}}[i] = {{u_minus}}[i] - c[i]*{{u_minus}}[i+1]

{% endblock %}
