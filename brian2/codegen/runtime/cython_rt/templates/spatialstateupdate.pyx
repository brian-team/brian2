{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { Cm, dt, v, N,
                        ab_star0, ab_star1, ab_star2, b_plus,
                        ab_plus0, ab_plus1, ab_plus2, b_minus,
                        ab_minus0, ab_minus1, ab_minus2,
                        v_star, u_plus, u_minus,
                        _P, _B, _morph_i, _morph_parent_i, _starts, _ends,
                        _invr0, _invrn} #}


    cdef double[:] _gtot_all = _numpy.zeros(N, dtype=_numpy.double)
    cdef double[:] c = _numpy.zeros(N, dtype=_numpy.double)
    cdef double ai
    cdef double bi
    cdef double _m

    cdef int _i
    cdef int _j

    cdef int _n_segments = _num{{_B}}
    cdef int __morph_i
    cdef int __i_parent
    cdef int __first
    cdef int __last
    cdef double __invr0
    cdef double __invrn

    ## MAIN CODE ######
    # Tridiagonal solving
    # Pass 1
    for _i in range(N):
        _idx = _i
        _vectorisation_idx = _idx

        {{vector_code|autoindent}}
        _gtot_all[_idx] = _gtot

        {{v_star}}[_i] = -({{Cm}}[_i]/dt*{{v}}[_i])-_I0 # RHS -> v_star (solution)
        bi={{ab_star1}}[_i] - _gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i]= {{ab_star0}}[_i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_star2}}[_i-1] # subdiagonal
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
        bi = {{ab_plus1}}[_i]-_gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i] = {{ab_plus0}}[_i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_plus2}}[_i-1] # subdiagonal
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
        bi = {{ab_minus1}}[_i] - _gtot_all[_i] # main diagonal
        if (_i<N-1):
            c[_i] = {{ab_minus0}}[_i+1] # superdiagonal
        if (_i>0):
            ai = {{ab_minus2}}[_i-1] # subdiagonal
            _m = 1.0/(bi-ai*c[_i-1])
            c[_i] = c[_i]*_m
            {{u_minus}}[_i] = ({{u_minus}}[_i] - ai*{{u_minus}}[_i-1])*_m
        else:
            c[0] = c[0]/bi
            {{u_minus}}[0] = {{u_minus}}[0]/bi
    for _i in range(N-2, -1, -1):
        {{u_minus}}[_i] = {{u_minus}}[_i] - c[_i]*{{u_minus}}[_i+1]

    # Prepare matrix for solving the linear system
    for _i in range(_num{{_B}}):
        {{_B}}[_i] = 0.
    for _i in range(_num{{_P}}):
        {{_P}}[_i] = 0.
    for _i in range(_n_segments - 1):
        __morph_i = {{_morph_i}}[_i]
        __i_parent = {{_morph_parent_i}}[_i]
        __first = {{_starts}}[_i]
        __last = {{_ends}}[_i]
        __invr0 = {{_invr0}}[_i]
        __invrn = {{_invrn}}[_i]
        # Towards parent
        if __morph_i == 1: # first branch, sealed end
            {{_P}}[0] = {{u_minus}}[__first] - 1
            {{_P}}[0 + 1] = {{u_plus}}[__first]
            {{_B}}[0] = -{{v_star}}[__first]
        else:
            {{_P}}[__i_parent*_n_segments + __i_parent] += (1 - {{u_minus}}[__first]) * __invr0
            {{_P}}[__i_parent*_n_segments + __morph_i] -= {{u_plus}}[__first] * __invr0
            {{_B}}[__i_parent] += {{v_star}}[__first] * __invr0
        # Towards children
        {{_P}}[__morph_i*_n_segments + __morph_i] = (1 - {{u_plus}}[__last]) * __invrn
        {{_P}}[__morph_i*_n_segments + __i_parent] = -{{u_minus}}[__last] * __invrn
        {{_B}}[__morph_i] = {{v_star}}[__last] * __invrn

{% endblock %}
