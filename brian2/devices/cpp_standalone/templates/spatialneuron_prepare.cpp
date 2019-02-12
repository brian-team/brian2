{# USES_VARIABLES { N, _invr, Ri, Cm, dt, area, r_length_1, r_length_2,
                    _ab_star0, _ab_star1, _ab_star2,
                    _starts, _ends, _invr0, _invrn, _b_plus, _b_minus,
                    _n_neurons, _n_compartments, _n_sections} #}
{% extends 'common_group.cpp' %}
{% block maincode %}
    #define _INDEX(_neuron, _compartment) ((_neuron)*_n_compartments + (_compartment))
    #define _INDEX_SEC(_neuron, _section) ((_neuron)*_n_sections + (_section))
    const double _Ri = {{Ri}};  // Ri is a shared variable

    for (int _neuron=0; _neuron < _n_neurons; _neuron++)
    {
        // Inverse axial resistance
        {{ openmp_pragma('parallel-static') }}
        for (int _i=1; _i<_n_compartments; _i++)
            {{_invr}}[_INDEX(_neuron, _i)] = 1.0/(_Ri*(1/{{r_length_2}}[_INDEX(_neuron, _i-1)] + 1/{{r_length_1}}[_INDEX(_neuron, _i)]));
        // Cut sections
        {{ openmp_pragma('parallel-static') }}
        for (int _i=0; _i<_n_sections; _i++)
            {{_invr}}[{{_starts}}[_INDEX_SEC(_neuron, _i)]] = 0;

        // Linear systems
        // The particular solution
        // a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
        {{ openmp_pragma('parallel-static') }}
        for (int _i=0; _i<_n_compartments; _i++)
            {{_ab_star1}}[_INDEX(_neuron, _i)] = (-({{Cm}}[_INDEX(_neuron, _i)] / {{dt}}) - {{_invr}}[_INDEX(_neuron, _i)] / {{area}}[_INDEX(_neuron, _i)]);
        {{ openmp_pragma('parallel-static') }}
        for (int _i=1; _i<_n_compartments; _i++)
        {
            {{_ab_star0}}[_INDEX(_neuron, _i)] = {{_invr}}[_INDEX(_neuron, _i)] / {{area}}[_INDEX(_neuron, _i-1)];
            {{_ab_star2}}[_INDEX(_neuron, _i-1)] = {{_invr}}[_INDEX(_neuron, _i)] / {{area}}[_INDEX(_neuron, _i)];
            {{_ab_star1}}[_INDEX(_neuron, _i-1)] -= {{_invr}}[_INDEX(_neuron, _i)] / {{area}}[_INDEX(_neuron, _i-1)];
        }

        // Set the boundary conditions
        for (int _counter=0; _counter<_n_sections; _counter++)
        {
            const int _first = {{_starts}}[_INDEX_SEC(_neuron, _counter)];
            const int _last = {{_ends}}[_INDEX_SEC(_neuron, _counter)] - 1;  // the compartment indices are in the interval [starts, ends[
            // Inverse axial resistances at the ends: r0 and rn
            const double _invr0 = {{r_length_1}}[_first]/_Ri;
            const double _invrn = {{r_length_2}}[_last]/_Ri;
            {{_invr0}}[_INDEX_SEC(_neuron, _counter)] = _invr0;
            {{_invrn}}[_INDEX_SEC(_neuron, _counter)] = _invrn;
            // Correction for boundary conditions
            {{_ab_star1}}[_first] -= (_invr0 / {{area}}[_first]);
            {{_ab_star1}}[_last] -= (_invrn / {{area}}[_last]);
            // RHS for homogeneous solutions
            {{_b_plus}}[_last] = -(_invrn / {{area}}[_last]);
            {{_b_minus}}[_first] = -(_invr0 / {{area}}[_first]);
        }
    }
{% endblock %}

