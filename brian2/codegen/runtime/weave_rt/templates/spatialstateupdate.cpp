////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////
{# USES_VARIABLES { Cm, dt, v, N,
                  _ab_star0, _ab_star1, _ab_star2, _b_plus,
                  _a_plus0, _a_plus1, _a_plus2, _b_minus,
                  _a_minus0, _a_minus1, _a_minus2, _v_star, _u_plus, _u_minus,
                  _gtot_all, _I0_all,
                  _c1, _c2, _c3,
                  _P_diag, _P_parent, _P_children,
                  _B, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn} #}

{% extends 'common_group.cpp' %}
{% block maincode %}

    int _vectorisation_idx = 1;

    //// MAIN CODE ////////////
    {{scalar_code|autoindent}}

    // STEP 1: compute g_total and I_0
    for(int i=0; i<N; i++)
    {
        const int _idx = i;
        _vectorisation_idx = _idx;

        {{vector_code|autoindent}}
        {{_gtot_all}}[_idx] = _gtot;
        {{_I0_all}}[_idx] = _I0;
    }

    // STEP 2: for each section: solve three tridiagonal systems

    // system 2a: solve for _v_star
    for (int _i=0; _i<_num_B - 1; _i++)
    {
        // first and last index of the i-th section
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];

        double _ai, _bi, _m; // helper variables

        // upper triangularization of tridiagonal system for _v_star
        for(int _j=_j_start; _j<_j_end; _j++)
        {
            {{_v_star}}[_j]=-({{Cm}}[_j]/{{dt}}*{{v}}[_j])-{{_I0_all}}[_j]; // RHS -> _v_star (solution)
            _bi={{_ab_star1}}[_j]-{{_gtot_all}}[_j]; // main diagonal
            if (_j<N-1)
                {{_c1}}[_j]={{_ab_star0}}[_j+1]; // superdiagonal
            if (_j>0)
            {
                _ai={{_ab_star2}}[_j-1]; // subdiagonal
                _m=1.0/(_bi-_ai*{{_c1}}[_j-1]);
                {{_c1}}[_j]={{_c1}}[_j]*_m;
                {{_v_star}}[_j]=({{_v_star}}[_j] - _ai*{{_v_star}}[_j-1])*_m;
            } else
            {
                {{_c1}}[0]={{_c1}}[0]/_bi;
                {{_v_star}}[0]={{_v_star}}[0]/_bi;
            }
        }
        // backwards substitution of the upper triangularized system for _v_star
        for(int _j=_j_end-2; _j>=_j_start; _j--)
            {{_v_star}}[_j]={{_v_star}}[_j] - {{_c1}}[_j]*{{_v_star}}[_j+1];
    }
    for (int _i=0; _i<_num_B - 1; _i++)
    {
        // first and last index of the i-th section
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];  // the compartment indices are in the interval [starts, ends[

        double _ai, _bi, _m; // helper variables

        // upper triangularization of tridiagonal system for _u_plus
        for(int _j=_j_start; _j<_j_end; _j++)
        {
            {{_u_plus}}[_j]={{_b_plus}}[_j]; // RHS -> _u_plus (solution)
            _bi={{_a_plus1}}[_j]-{{_gtot_all}}[_j]; // main diagonal
            if (_j<N-1)
                {{_c2}}[_j]={{_a_plus0}}[_j+1]; // superdiagonal
            if (_j>0)
            {
                _ai={{_a_plus2}}[_j-1]; // subdiagonal
                _m=1.0/(_bi-_ai*{{_c2}}[_j-1]);
                {{_c2}}[_j]={{_c2}}[_j]*_m;
                {{_u_plus}}[_j]=({{_u_plus}}[_j] - _ai*{{_u_plus}}[_j-1])*_m;
            } else
            {
                {{_c2}}[0]={{_c2}}[0]/_bi;
                {{_u_plus}}[0]={{_u_plus}}[0]/_bi;
            }
        }
        // backwards substitution of the upper triangularized system for _u_plus
        for(int _j=_j_end-2; _j>=_j_start; _j--)
            {{_u_plus}}[_j]={{_u_plus}}[_j] - {{_c2}}[_j]*{{_u_plus}}[_j+1];
    }

    for (int _i=0; _i<_num_B - 1; _i++)
    {
        // first and last index of the i-th section
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];

        double _ai, _bi, _m; // helper variables

        // upper triangularization of tridiagonal system for _u_minus
        for(int _j=_j_start; _j<_j_end; _j++)
        {
            {{_u_minus}}[_j]={{_b_minus}}[_j]; // RHS -> _u_minus (solution)
            _bi={{_a_minus1}}[_j]-{{_gtot_all}}[_j]; // main diagonal
            if (_j<N-1)
                {{_c3}}[_j]={{_a_minus0}}[_j+1]; // superdiagonal
            if (_j>0)
            {
                _ai={{_a_minus2}}[_j-1]; // subdiagonal
                _m=1.0/(_bi-_ai*{{_c3}}[_j-1]);
                {{_c3}}[_j]={{_c3}}[_j]*_m;
                {{_u_minus}}[_j]=({{_u_minus}}[_j] - _ai*{{_u_minus}}[_j-1])*_m;
            } else
            {
                {{_c3}}[0]={{_c3}}[0]/_bi;
                {{_u_minus}}[0]={{_u_minus}}[0]/_bi;
            }
        }
        // backwards substitution of the upper triangularized system for _u_minus
        for(int _j=_j_end-2; _j>=_j_start; _j--)
            {{_u_minus}}[_j]={{_u_minus}}[_j] - {{_c3}}[_j]*{{_u_minus}}[_j+1];
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
    for (int _i=0; _i<_num_B - 1; _i++)
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
            {{_P_diag}}[0] = {{_u_minus}}[_first] - 1;
            {{_P_children}}[_IDX_C(0,0)] = {{_u_plus}}[_first];

            // RHS
            {{_B}}[0] = -{{_v_star}}[_first];
        }
        else
        {
            {{_P_diag}}[_i_parent] += (1 - {{_u_minus}}[_first]) * _invr0;
            {{_P_children}}[_IDX_C(_i_parent, _i_childind)] = -{{_u_plus}}[_first] * _invr0;

            // RHS
            {{_B}}[_i_parent] += {{_v_star}}[_first] * _invr0;
        }

        // Towards children
        {{_P_diag}}[_i+1] = (1 - {{_u_plus}}[_last]) * _invrn;
        {{_P_parent}}[_i] = -{{_u_minus}}[_last] * _invrn;

        // RHS
        {{_B}}[_i+1] = {{_v_star}}[_last] * _invrn;
    }


    // STEP 3b: solve the linear system (the result will be stored in the former rhs _B in the end)
    // use efficient O(n) solution of the sparse linear system (structure-specific Gaussian elemination)

    // part 1: lower triangularization
    for (int _i=_num_B-1; _i>=0; _i--) {
        const int num_children = {{_morph_children_num}}[_i];

        // for every child eliminate the corresponding matrix element of row i
        for (int _k=0; _k<num_children; _k++) {
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
    for (int _i=0; _i<_num_B - 1; _i++)
    {
        const int _i_parent = {{_morph_parent_i}}[_i];
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];
        for (int _j=_j_start; _j<_j_end; _j++)
            if (_j < _numv)  // don't go beyond the last element
                {{v}}[_j] = {{_v_star}}[_j] + {{_B}}[_i_parent] * {{_u_minus}}[_j]
                                           + {{_B}}[_i+1] * {{_u_plus}}[_j];
    }

{% endblock %}
