{# USES_VARIABLES { Cm, dt, v, N, Ic, Ri,
                  _ab_star0, _ab_star1, _ab_star2, _b_plus, _b_minus,
                  _v_star, _u_plus, _u_minus,
                  _v_previous,
                  _gtot_all, _I0_all,
                  _c,
                  _P_diag, _P_parent, _P_children,
                  _B, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn, _invr,
                  r_length_1, r_length_2, area } #}
{% extends 'common_group.cpp' %}

{% block before_code %}
    const double _Ri = {{Ri}};  // Ri is a shared variable

    // Inverse axial resistance
    {{ openmp_pragma('parallel-static') }}
    for (int _i=1; _i<N; _i++)
        {{_invr}}[_i] = 1.0/(_Ri*(1/{{r_length_2}}[_i-1] + 1/{{r_length_1}}[_i]));
    // Cut sections
    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<(int)_num_starts; _i++)
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

    // Set the boundary conditions
    for (size_t _counter=0; _counter<_num_starts; _counter++)
    {
        const int _first = {{_starts}}[_counter];
        const int _last = {{_ends}}[_counter] - 1;  // the compartment indices are in the interval [starts, ends[
        // Inverse axial resistances at the ends: r0 and rn
        const double _invr0 = {{r_length_1}}[_first]/_Ri;
        const double _invrn = {{r_length_2}}[_last]/_Ri;
        {{_invr0}}[_counter] = _invr0;
        {{_invrn}}[_counter] = _invrn;
        // Correction for boundary conditions
        {{_ab_star1}}[_first] -= (_invr0 / {{area}}[_first]);
        {{_ab_star1}}[_last] -= (_invrn / {{area}}[_last]);
        // RHS for homogeneous solutions
        {{_b_plus}}[_last] = -(_invrn / {{area}}[_last]);
        {{_b_minus}}[_first] = -(_invr0 / {{area}}[_first]);
    }
{% endblock %}

{% block maincode %}

    int _vectorisation_idx = 1;

    //// MAIN CODE ////////////
    {{scalar_code|autoindent}}

    // STEP 1: compute g_total and I_0 (independent: compartments)
    {{ openmp_pragma('parallel-static') }}
    for(int i=0; i<N; i++)
    {
        const int _idx = i;
        _vectorisation_idx = _idx;

        {{vector_code|autoindent}}
        {{_gtot_all}}[_idx] = _gtot;
        {{_I0_all}}[_idx] = _I0;

        {{_v_previous}}[_idx] = {{v}}[_idx];
    }

    // STEP 2: for each section: solve three tridiagonal systems
    // (independent: branches)

    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<(int)_num_B - 1; _i++)
    {
        // first and last index of the i-th section
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];

        double _ai, _bi, _m; // helper variables

        // upper triangularization of tridiagonal system for _v_star, _u_plus, and _u_minus
        for(int _j=_j_start; _j<_j_end; _j++)
        {
            {{_v_star}}[_j]=-({{Cm}}[_j]/{{dt}}*{{v}}[_j])-{{_I0_all}}[_j]; // RHS -> _v_star (solution)
            {{_u_plus}}[_j]={{_b_plus}}[_j]; // RHS -> _u_plus (solution)
            {{_u_minus}}[_j]={{_b_minus}}[_j]; // RHS -> _u_minus (solution)
            _bi={{_ab_star1}}[_j]-{{_gtot_all}}[_j]; // main diagonal
            if (_j<N-1)
                {{_c}}[_j]={{_ab_star0}}[_j+1]; // superdiagonal
            if (_j>0)
            {
                _ai={{_ab_star2}}[_j-1]; // subdiagonal
                _m=1.0/(_bi-_ai*{{_c}}[_j-1]);
                {{_c}}[_j]={{_c}}[_j]*_m;
                {{_v_star}}[_j]=({{_v_star}}[_j] - _ai*{{_v_star}}[_j-1])*_m;
                {{_u_plus}}[_j]=({{_u_plus}}[_j] - _ai*{{_u_plus}}[_j-1])*_m;
                {{_u_minus}}[_j]=({{_u_minus}}[_j] - _ai*{{_u_minus}}[_j-1])*_m;
            } else
            {
                {{_c}}[0]={{_c}}[0]/_bi;
                {{_v_star}}[0]={{_v_star}}[0]/_bi;
                {{_u_plus}}[0]={{_u_plus}}[0]/_bi;
                {{_u_minus}}[0]={{_u_minus}}[0]/_bi;
            }
        }
        // backwards substituation of the upper triangularized system for _v_star
        for(int _j=_j_end-2; _j>=_j_start; _j--)
        {
            {{_v_star}}[_j]={{_v_star}}[_j] - {{_c}}[_j]*{{_v_star}}[_j+1];
            {{_u_plus}}[_j]={{_u_plus}}[_j] - {{_c}}[_j]*{{_u_plus}}[_j+1];
            {{_u_minus}}[_j]={{_u_minus}}[_j] - {{_c}}[_j]*{{_u_minus}}[_j+1];
        }
    }

    // STEP 3: solve the coupling system

    // indexing for _P_children which contains the elements above the diagonal of the coupling matrix _P
    const int _children_rowlength = _num_morph_children/_num_morph_children_num;
    #define _IDX_C(idx_row,idx_col) _children_rowlength * idx_row + idx_col

    // STEP 3a: construct the coupling system with matrix _P in sparse form. s.t.
    // _P_diag contains the diagonal elements
    // _P_children contains the super diagonal entries
    // _P_parent contains the single sub diagonal entry for each row
    // _B contains the right hand side
    for (size_t _i=0; _i<_num_B - 1; _i++)
    {
        const int _i_parent = {{_morph_parent_i}}[_i];
        const int _i_childind = {{_morph_idxchild}}[_i];
        const int _first = {{_starts}}[_i];
        const int _last = {{_ends}}[_i] - 1;  // the compartment indices are in the interval [starts, ends[
        const double _invr0 = {{_invr0}}[_i];
        const double _invrn = {{_invrn}}[_i];

        // Towards parent
        if (_i == 0) // first section, sealed end
        {
            // sparse matrix version
            {{_P_diag}}[0] = {{_u_minus}}[_first] - 1;
            {{_P_children}}[_IDX_C(0,0)] = {{_u_plus}}[_first];

            // RHS
            {{_B}}[0] = -{{_v_star}}[_first];
        }
        else
        {
            // sparse matrix version
            {{_P_diag}}[_i_parent] += (1 - {{_u_minus}}[_first]) * _invr0;
            {{_P_children}}[_IDX_C(_i_parent, _i_childind)] = -{{_u_plus}}[_first] * _invr0;

            // RHS
            {{_B}}[_i_parent] += {{_v_star}}[_first] * _invr0;
        }

        // Towards children

        // sparse matrix version
        {{_P_diag}}[_i+1] = (1 - {{_u_plus}}[_last]) * _invrn;
        {{_P_parent}}[_i] = -{{_u_minus}}[_last] * _invrn;

        // RHS
        {{_B}}[_i+1] = {{_v_star}}[_last] * _invrn;
    }


    // STEP 3b: solve the linear system (the result will be stored in the former rhs _B in the end)
    // use efficient O(n) solution of the sparse linear system (structure-specific Gaussian elemination)

    // part 1: lower triangularization
    for (int _i=_num_B-1; _i>=0; _i--) {
        const int _num_children = {{_morph_children_num}}[_i];

        // for every child eliminate the corresponding matrix element of row i
        for (size_t _k=0; _k<_num_children; _k++) {
            int _j = {{_morph_children}}[_IDX_C(_i,_k)]; // child index

            // subtracting _subfac times the j-th from the i-th row
            double _subfac = {{_P_children}}[_IDX_C(_i,_k)] / {{_P_diag}}[_j]; // element i,j appears only here

            // the following commented (superdiagonal) element is not used in the following anymore since
            // it is 0 by definition of (lower) triangularization; we keep it here for algorithmic clarity
            //{{_P_children}}[_IDX_C(_i,_k)] = {{_P_children}}[_IDX_C(_i,_k)]  - _subfac * {{_P_diag}}[_j]; // = 0;

            {{_P_diag}}[_i] = {{_P_diag}}[_i]  - _subfac * {{_P_parent}}[_j-1]; // note: element j,i is only used here
            {{_B}}[_i] = {{_B}}[_i] - _subfac * {{_B}}[_j];

        }
    }

    // part 2: forwards substitution
    {{_B}}[0] = {{_B}}[0] / {{_P_diag}}[0]; // the first section does not have a parent
    for (int _i=1; _i<_num_B; _i++) {
        const int _j = {{_morph_parent_i}}[_i-1]; // parent index
        {{_B}}[_i] = {{_B}}[_i] - {{_P_parent}}[_i-1] * {{_B}}[_j];
        {{_B}}[_i] = {{_B}}[_i] / {{_P_diag}}[_i];

    }

    // STEP 4: for each section compute the final solution by linear
    // combination of the general solution (independent: sections & compartments)
    for (size_t _i=0; _i<_num_B - 1; _i++)
    {
        const int _i_parent = {{_morph_parent_i}}[_i];
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];
        for (int _j=_j_start; _j<_j_end; _j++)
            if (_j < _numv)  // don't go beyond the last element
                {{v}}[_j] = {{_v_star}}[_j] + {{_B}}[_i_parent] * {{_u_minus}}[_j]
                                           + {{_B}}[_i+1] * {{_u_plus}}[_j];
    }

    {{ openmp_pragma('parallel-static') }}
    for (int _i=0; _i<N; _i++)
    {
        {{Ic}}[_i] = {{Cm}}[_i]*({{v}}[_i] - {{_v_previous}}[_i])/{{dt}};
    }

{% endblock %}
