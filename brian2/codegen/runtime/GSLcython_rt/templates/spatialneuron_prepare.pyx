{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { N, _invr, Ri, Cm, dt, area, r_length_1, r_length_2,
                        _ab_star0, _ab_star1, _ab_star2,
                        _a_plus0, _a_plus1, _a_plus2,
                        _a_minus0, _a_minus1, _a_minus2,
                        _starts, _ends, _invr0, _invrn, _b_plus, _b_minus } #}
    cdef int _i
    cdef int _counter
    cdef int _first
    cdef int _last
    cdef double __invr0
    cdef double __invrn

    cdef double _Ri = {{Ri}}  # Ri is a shared variable

    # Inverse axial resistance
    for _i in range(1, N):
        {{_invr}}[_i] =1.0/(_Ri*(1/{{r_length_2}}[_i-1] + 1/{{r_length_1}}[_i]))
    # Cut sections
    for _i in range(_num{{_starts}}):
        {{_invr}}[{{_starts}}[_i]] = 0

    # Linear systems
    # The particular solution
    # a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
    for _i in range(N):
        {{_ab_star1}}[_i] = (-({{Cm}}[_i] / {{dt}}) - {{_invr}}[_i] / {{area}}[_i])
    for _i in range(1, N):
        {{_ab_star0}}[_i] = {{_invr}}[_i] / {{area}}[_i-1]
        {{_ab_star2}}[_i-1] = {{_invr}}[_i] / {{area}}[_i]
        {{_ab_star1}}[_i-1] -= {{_invr}}[_i] / {{area}}[_i-1]
    for _i in range(N):
        # Homogeneous solutions
        {{_a_plus0}}[_i] = {{_ab_star0}}[_i]
        {{_a_minus0}}[_i] = {{_ab_star0}}[_i]
        {{_a_plus1}}[_i] = {{_ab_star1}}[_i]
        {{_a_minus1}}[_i] = {{_ab_star1}}[_i]
        {{_a_plus2}}[_i] = {{_ab_star2}}[_i]
        {{_a_minus2}}[_i] = {{_ab_star2}}[_i]

    # Set the boundary conditions
    for _counter in range(_num{{_starts}}):
        _first = {{_starts}}[_counter]
        _last = {{_ends}}[_counter] -1  # the compartment indices are in the interval [starts, ends[
        # Inverse axial resistances at the ends: r0 and rn
        __invr0 = {{r_length_1}}[_first]/_Ri
        __invrn = {{r_length_2}}[_last]/_Ri
        {{_invr0}}[_counter] = __invr0
        {{_invrn}}[_counter] = __invrn
        # Correction for boundary conditions
        {{_ab_star1}}[_first] -= (__invr0 / {{area}}[_first])
        {{_ab_star1}}[_last] -= (__invrn / {{area}}[_last])
        {{_a_plus1}}[_first] -= (__invr0 / {{area}}[_first])
        {{_a_plus1}}[_last] -= (__invrn / {{area}}[_last])
        {{_a_minus1}}[_first] -= (__invr0 / {{area}}[_first])
        {{_a_minus1}}[_last] -= (__invrn / {{area}}[_last])
        # RHS for homogeneous solutions
        {{_b_plus}}[_last] = -(__invrn / {{area}}[_last])
        {{_b_minus}}[_first] = -(__invr0 / {{area}}[_first])

{% endblock %}
