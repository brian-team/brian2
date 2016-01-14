////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  gtot_all, I0_all,
                  c1, c2, c3,
                  _P_diag, _P_parent, _P_children,
                  _B, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn} #}
                  
{% extends 'common_group.cpp' %}
{% block maincode %}

    {% set strategy = prefs.devices.cpp_standalone.openmp_spatialneuron_strategy %}
    {% if strategy == None %}
        {% if prefs.devices.cpp_standalone.openmp_threads <= 3 or number_branches < 3%}
            {% set strategy = 'systems' %}
        {% else %}
            {% set strategy = 'branches' %}
        {% endif %}
    {% endif %}

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
        {{gtot_all}}[_idx] = _gtot;
        {{I0_all}}[_idx] = _I0;
    }

    // STEP 2: for each branch: solve three tridiagonal systems
    // (independent: branches and also the three tridiagonal systems)

    {% if strategy == 'systems' %}
    {{ openmp_pragma('parallel') }}
    {
    {% endif %}
    {{ openmp_pragma('sections') }}
    {
    // system 2a: solve for v_star
    {{ openmp_pragma('section') }}
    {
        {% if strategy == 'branches' %}
        {{ openmp_pragma('parallel-static') }}
        {% endif %}
        for (int _i=0; _i<_num_B - 1; _i++)
        {
            // first and last index of the i-th branch
            const int _j_start = {{_starts}}[_i];
            const int _j_end = {{_ends}}[_i];
            
            double ai, bi, _m; // helper variables

            // upper triangularization of tridiagonal system for v_star
            for(int _j=_j_start; _j<_j_end+1; _j++)
            {
                {{v_star}}[_j]=-({{Cm}}[_j]/{{dt}}*{{v}}[_j])-{{I0_all}}[_j]; // RHS -> v_star (solution)
                bi={{ab_star1}}[_j]-{{gtot_all}}[_j]; // main diagonal
                if (_j<N-1)
                    {{c1}}[_j]={{ab_star0}}[_j+1]; // superdiagonal
                if (_j>0)
                {
                    ai={{ab_star2}}[_j-1]; // subdiagonal
                    _m=1.0/(bi-ai*{{c1}}[_j-1]);
                    {{c1}}[_j]={{c1}}[_j]*_m;
                    {{v_star}}[_j]=({{v_star}}[_j] - ai*{{v_star}}[_j-1])*_m;
                } else
                {
                    {{c1}}[0]={{c1}}[0]/bi;
                    {{v_star}}[0]={{v_star}}[0]/bi;
                }
            }
            // backwards substituation of the upper triangularized system for v_star
            for(int _j=_j_end-1; _j>=_j_start; _j--)
                {{v_star}}[_j]={{v_star}}[_j] - {{c1}}[_j]*{{v_star}}[_j+1];
        }
    }
    // system 2b: solve for u_plus
    {{ openmp_pragma('section') }}
    {
        {% if strategy == 'branches' %}
        {{ openmp_pragma('parallel-static') }}
        {% endif %}
        for (int _i=0; _i<_num_B - 1; _i++)
        {
            // first and last index of the i-th branch
            const int _j_start = {{_starts}}[_i];
            const int _j_end = {{_ends}}[_i];
            
            double ai, bi, _m; // helper variables

            // upper triangularization of tridiagonal system for u_plus
            for(int _j=_j_start; _j<_j_end+1; _j++)
            {
                {{u_plus}}[_j]={{b_plus}}[_j]; // RHS -> u_plus (solution)
                bi={{ab_plus1}}[_j]-{{gtot_all}}[_j]; // main diagonal
                if (_j<N-1)
                    {{c2}}[_j]={{ab_plus0}}[_j+1]; // superdiagonal
                if (_j>0)
                {
                    ai={{ab_plus2}}[_j-1]; // subdiagonal
                    _m=1.0/(bi-ai*{{c2}}[_j-1]);
                    {{c2}}[_j]={{c2}}[_j]*_m;
                    {{u_plus}}[_j]=({{u_plus}}[_j] - ai*{{u_plus}}[_j-1])*_m;
                } else
                {
                    {{c2}}[0]={{c2}}[0]/bi;
                    {{u_plus}}[0]={{u_plus}}[0]/bi;
                }
            }
            // backwards substituation of the upper triangularized system for u_plus
            for(int _j=_j_end-1; _j>=_j_start; _j--)
                {{u_plus}}[_j]={{u_plus}}[_j] - {{c2}}[_j]*{{u_plus}}[_j+1];
        }
    }
    // system 2c: solve for u_minus
    {{ openmp_pragma('section') }}
    {
        {% if strategy == 'branches' %}
        {{ openmp_pragma('parallel-static') }}
        {% endif %}
        for (int _i=0; _i<_num_B - 1; _i++)
        {
            // first and last index of the i-th branch
            const int _j_start = {{_starts}}[_i];
            const int _j_end = {{_ends}}[_i];

            double ai, bi, _m; // helper variables
            
            // upper triangularization of tridiagonal system for u_minus
            for(int _j=_j_start; _j<_j_end+1; _j++)
            {
                {{u_minus}}[_j]={{b_minus}}[_j]; // RHS -> u_minus (solution)
                bi={{ab_minus1}}[_j]-{{gtot_all}}[_j]; // main diagonal
                if (_j<N-1)
                    {{c3}}[_j]={{ab_minus0}}[_j+1]; // superdiagonal
                if (_j>0)
                {
                    ai={{ab_minus2}}[_j-1]; // subdiagonal
                    _m=1.0/(bi-ai*{{c3}}[_j-1]);
                    {{c3}}[_j]={{c3}}[_j]*_m;
                    {{u_minus}}[_j]=({{u_minus}}[_j] - ai*{{u_minus}}[_j-1])*_m;
                } else
                {
                    {{c3}}[0]={{c3}}[0]/bi;
                    {{u_minus}}[0]={{u_minus}}[0]/bi;
                }
            }
            // backwards substituation of the upper triangularized system for u_minus
            for(int _j=_j_end-1; _j>=_j_start; _j--)
                {{u_minus}}[_j]={{u_minus}}[_j] - {{c3}}[_j]*{{u_minus}}[_j+1];
        }
    } // (OpenMP section)
    } // (OpenMP sections)
    {% if strategy == 'systems' %}
    } // (OpenMP parallel)
    {% endif %}


    // STEP 3: solve the coupling system

    // indexing for _P_children which contains the elements above the diagonal of the coupling matrix _P
    const int children_rowlength = _num_morph_children/_num_morph_children_num;
    #define IDX_C(idx_row,idx_col) children_rowlength * idx_row + idx_col

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
        const int _last = {{_ends}}[_i];
        const double _invr0 = {{_invr0}}[_i];
        const double _invrn = {{_invrn}}[_i];

        // Towards parent
        if (_i == 0) // first branch, sealed end
        {
            // sparse matrix version
            {{_P_diag}}[0] = {{u_minus}}[_first] - 1;
            {{_P_children}}[IDX_C(0,0)] = {{u_plus}}[_first];

            // RHS
            {{_B}}[0] = -{{v_star}}[_first];
        }
        else
        {
            // sparse matrix version
            {{_P_diag}}[_i_parent] += (1 - {{u_minus}}[_first]) * _invr0;
            {{_P_children}}[IDX_C(_i_parent, _i_childind)] = -{{u_plus}}[_first] * _invr0;

            // RHS
            {{_B}}[_i_parent] += {{v_star}}[_first] * _invr0;
        }

        // Towards children

        // sparse matrix version
        {{_P_diag}}[_i+1] = (1 - {{u_plus}}[_last]) * _invrn;
        {{_P_parent}}[_i] = -{{u_minus}}[_last] * _invrn;

        // RHS
        {{_B}}[_i+1] = {{v_star}}[_last] * _invrn;
    }


    // STEP 3b: solve the linear system (the result will be stored in the former rhs _B in the end)
    // use efficient O(n) solution of the sparse linear system (structure-specific Gaussian elemination)

    // part 1: lower triangularization
    for (int _i=_num_B-1; _i>=0; _i--) {
        const int num_children = {{_morph_children_num}}[_i];
        
        // for every child eliminate the corresponding matrix element of row i
        for (int _k=0; _k<num_children; _k++) {
            int _j = {{_morph_children}}[IDX_C(_i,_k)]; // child index
            
            // subtracting subfac times the j-th from the i-th row
            double subfac = {{_P_children}}[IDX_C(_i,_k)] / {{_P_diag}}[_j]; // element i,j appears only here

            // the following commented (superdiagonal) element is not used in the following anymore since 
            // it is 0 by definition of (lower) triangularization; we keep it here for algorithmic clarity
            //{{_P_children}}[IDX_C(_i,_k)] = {{_P_children}}[IDX_C(_i,_k)]  - subfac * {{_P_diag}}[_j]; // = 0;

            {{_P_diag}}[_i] = {{_P_diag}}[_i]  - subfac * {{_P_parent}}[_j-1]; // note: element j,i is only used here
            {{_B}}[_i] = {{_B}}[_i] - subfac * {{_B}}[_j];
        
        }
    }

    // part 2: forwards substitution
    {{_B}}[0] = {{_B}}[0] / {{_P_diag}}[0]; // the first branch does not have a parent
    for (int _i=1; _i<_num_B; _i++) {
        const int j = {{_morph_parent_i}}[_i-1]; // parent index
        {{_B}}[_i] = {{_B}}[_i] - {{_P_parent}}[_i-1] * {{_B}}[j];
        {{_B}}[_i] = {{_B}}[_i] / {{_P_diag}}[_i];
        
    }

    // STEP 4: for each branch compute the final solution by linear 
    // combination of the general solution (independent: branches & compartments)
    for (int _i=0; _i<_num_B - 1; _i++)
    {
        const int _i_parent = {{_morph_parent_i}}[_i];
        const int _j_start = {{_starts}}[_i];
        const int _j_end = {{_ends}}[_i];
        for (int _j=_j_start; _j<_j_end + 1; _j++)
            if (_j < _numv)  // don't go beyond the last element
                {{v}}[_j] = {{v_star}}[_j] + {{_B}}[_i_parent] * {{u_minus}}[_j]
                                           + {{_B}}[_i+1] * {{u_plus}}[_j];
    }

{% endblock %}
