{% extends 'common.pyx' %}

{% block maincode %}
    {# USES_VARIABLES { N, _invr, Ri, Cm, dt, area, diameter, length,
                        ab_star0, ab_star1, ab_star2,
                        ab_plus0, ab_plus1, ab_plus2,
                        ab_minus0, ab_minus1, ab_minus2,
                        _starts, _ends, _invr0, _invrn, b_plus, b_minus } #}
    cdef int _i
    cdef int _counter
    cdef int _first
    cdef int _last
    cdef double __invr0
    cdef double __invrn

    cdef double _Ri = {{Ri}}[0]  # Ri is a shared variable

    {% if owner.morphology.type == 'soma' %}
    # Correction for soma (a bit of a hack),
    #  so that it has negligible axial resistance
    {{length}}[0] = {{diameter}}[0] * 0.01;
    {% endif %}

    # Inverse axial resistance
    for _i in range(1, N):
        {{_invr}}[_i] = (M_PI / (2 * _Ri) * ({{diameter}}[_i-1] * {{diameter}}[_i]) /
                       ({{length}}[_i-1] + {{length}}[_i]))
    # Note: this would give nan for the soma
    # Cut branches
    for _i in range(_num{{_starts}}):
        {{_invr}}[{{_starts}}[_i]] = 0

    # Linear systems
    # The particular solution
    # a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
    for _i in range(N):
        {{ab_star1}}[_i] = (-({{Cm}}[_i] / dt) - {{_invr}}[_i] / {{area}}[_i])
    for _i in range(1, N):
        {{ab_star0}}[_i] = {{_invr}}[_i] / {{area}}[_i-1]
        {{ab_star2}}[_i-1] = {{_invr}}[_i] / {{area}}[_i]
        {{ab_star1}}[_i-1] -= {{_invr}}[_i] / {{area}}[_i-1]
    for _i in range(N):
        # Homogeneous solutions
        {{ab_plus0}}[_i] = {{ab_star0}}[_i]
        {{ab_minus0}}[_i] = {{ab_star0}}[_i]
        {{ab_plus1}}[_i] = {{ab_star1}}[_i]
        {{ab_minus1}}[_i] = {{ab_star1}}[_i]
        {{ab_plus2}}[_i] = {{ab_star2}}[_i]
        {{ab_minus2}}[_i] = {{ab_star2}}[_i]

    # Set the boundary conditions
    for _counter in range(_num{{_starts}}):
        _first = {{_starts}}[_counter]
        _last = {{_ends}}[_counter]
        # Inverse axial resistances at the ends: r0 and rn
        __invr0 = (M_PI / (2 * _Ri) * {{diameter}}[_first] * {{diameter}}[_first]  /
                              {{length}}[_first])
        __invrn = (M_PI / (2 * _Ri) * {{diameter}}[_last] * {{diameter}}[_last] /
                              {{length}}[_last])
        {{_invr0}}[_counter] = __invr0
        {{_invrn}}[_counter] = __invrn
        # Correction for boundary conditions
        {{ab_star1}}[_first] -= (__invr0 / {{area}}[_first])
        {{ab_star1}}[_last] -= (__invrn / {{area}}[_last])
        {{ab_plus1}}[_first] -= (__invr0 / {{area}}[_first])
        {{ab_plus1}}[_last] -= (__invrn / {{area}}[_last])
        {{ab_minus1}}[_first] -= (__invr0 / {{area}}[_first])
        {{ab_minus1}}[_last] -= (__invrn / {{area}}[_last])
        # RHS for homogeneous solutions
        {{b_plus}}[_last] = -(__invrn / {{area}}[_last])
        {{b_minus}}[_first] = -(__invr0 / {{area}}[_first])

{% endblock %}
