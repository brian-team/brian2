{# USES_VARIABLES { N, _invr, Ri, Cm, dt, area, diameter, length,
                    _ab_star0, _ab_star1, _ab_star2,
                    _a_plus0, _a_plus1, _a_plus2,
                    _a_minus0, _a_minus1, _a_minus2,
                    _starts, _ends, _invr0, _invrn, _b_plus, _b_minus } #}
{% extends 'common_group.cpp' %}
{% block maincode %}
    const double _Ri = {{Ri}};  // Ri is a shared variable

    {% if owner.morphology.type == 'soma' %}
    // Correction for soma (a bit of a hack),
    //  so that it has negligible axial resistance
    {{length}}[0] = {{diameter}}[0] * 0.01;
    {% endif %}

    // Inverse axial resistance
    {{ openmp_pragma('parallel-static') }}
    for (int _i=1; _i<N; _i++)
        {{_invr}}[_i] = (M_PI / (2 * _Ri) * ({{diameter}}[_i-1] * {{diameter}}[_i]) /
                       ({{length}}[_i-1] + {{length}}[_i]));
    // Note: this would give nan for the soma
    // Cut branches
    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<_num_starts; _i++)
        {{_invr}}[{{_starts}}[_i]] = 0;

    // Linear systems
    // The particular solution
    // a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<N; _i++)
        {{_ab_star1}}[_i] = (-({{Cm}}[_i] / {{dt}}) - {{_invr}}[_i] / {{area}}[_i]);
    {{ openmp_pragma('parallel-static') }}
    for (int _i=1; _i<N; _i++)
    {
        {{_ab_star0}}[_i] = {{_invr}}[_i] / {{area}}[_i-1];
        {{_ab_star2}}[_i-1] = {{_invr}}[_i] / {{area}}[_i];
        {{_ab_star1}}[_i-1] -= {{_invr}}[_i] / {{area}}[_i-1];
    }
    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<N; _i++)
    {
        // Homogeneous solutions
        {{_a_plus0}}[_i] = {{_ab_star0}}[_i];
        {{_a_minus0}}[_i] = {{_ab_star0}}[_i];
        {{_a_plus1}}[_i] = {{_ab_star1}}[_i];
        {{_a_minus1}}[_i] = {{_ab_star1}}[_i];
        {{_a_plus2}}[_i] = {{_ab_star2}}[_i];
        {{_a_minus2}}[_i] = {{_ab_star2}}[_i];
    }

    // Set the boundary conditions
    for (int _counter=0; _counter<_num_starts; _counter++)
    {
        const int _first = {{_starts}}[_counter];
        const int _last = {{_ends}}[_counter];
        // Inverse axial resistances at the ends: r0 and rn
        const double _invr0 = (M_PI / (2 * _Ri) * {{diameter}}[_first] * {{diameter}}[_first]  /
                              {{length}}[_first]);
        const double _invrn = (M_PI / (2 * _Ri) * {{diameter}}[_last] * {{diameter}}[_last] /
                              {{length}}[_last]);
        {{_invr0}}[_counter] = _invr0;
        {{_invrn}}[_counter] = _invrn;
        // Correction for boundary conditions
        {{_ab_star1}}[_first] -= (_invr0 / {{area}}[_first]);
        {{_ab_star1}}[_last] -= (_invrn / {{area}}[_last]);
        {{_a_plus1}}[_first] -= (_invr0 / {{area}}[_first]);
        {{_a_plus1}}[_last] -= (_invrn / {{area}}[_last]);
        {{_a_minus1}}[_first] -= (_invr0 / {{area}}[_first]);
        {{_a_minus1}}[_last] -= (_invrn / {{area}}[_last]);
        // RHS for homogeneous solutions
        {{_b_plus}}[_last] = -(_invrn / {{area}}[_last]);
        {{_b_minus}}[_first] = -(_invr0 / {{area}}[_first]);
    }
{% endblock %}

