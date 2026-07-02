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
    double _Ri = {{ Ri }};

    // Inverse axial resistance
    for (int _i = 1; _i < N; _i++) {
        {{ _invr }}[_i] = 1.0 / (_Ri * (1.0 / {{ r_length_2 }}[_i - 1] + 1.0 / {{ r_length_1 }}[_i]));
    }
    // Cut sections
    for (int _i = 0; _i < _num_starts; _i++) {
        {{ _invr }}[{{ _starts }}[_i]] = 0;
    }

    // Linear systems
    // The particular solution
    // a[i,j]=ab[u+i-j,j]   --  u is the number of upper diagonals = 1
    for (int _i = 0; _i < N; _i++) {
        {{ _ab_star1 }}[_i] = -({{ Cm }}[_i] / {{ dt }}) - {{ _invr }}[_i] / {{ area }}[_i];
    }
    for (int _i = 1; _i < N; _i++) {
        {{ _ab_star0 }}[_i] = {{ _invr }}[_i] / {{ area }}[_i - 1];
        {{ _ab_star2 }}[_i - 1] = {{ _invr }}[_i] / {{ area }}[_i];
        {{ _ab_star1 }}[_i - 1] -= {{ _invr }}[_i] / {{ area }}[_i - 1];
    }

    // Set the boundary conditions
    for (int _counter = 0; _counter < _num_starts; _counter++) {
        int _first = {{ _starts }}[_counter];
        int _last = {{ _ends }}[_counter] - 1;  // compartment indices in [starts, ends[
        // Inverse axial resistances at the ends: r0 and rn
        double __invr0 = {{ r_length_1 }}[_first] / _Ri;
        double __invrn = {{ r_length_2 }}[_last] / _Ri;
        {{ _invr0 }}[_counter] = __invr0;
        {{ _invrn }}[_counter] = __invrn;
        // Correction for boundary conditions
        {{ _ab_star1 }}[_first] -= (__invr0 / {{ area }}[_first]);
        {{ _ab_star1 }}[_last] -= (__invrn / {{ area }}[_last]);
        // RHS for homogeneous solutions
        {{ _b_plus }}[_last] = -(__invrn / {{ area }}[_last]);
        {{ _b_minus }}[_first] = -(__invr0 / {{ area }}[_first]);
    }
{% endblock %}

{% block maincode %}
    // MAIN CODE
    const size_t _vectorisation_idx_scalar = 1;
    {{ scalar_code | autoindent }}

    // STEP 1: compute g_total and I_0
    for (int _i = 0; _i < N; _i++) {
        int _idx = _i;
        const size_t _vectorisation_idx = _idx;
        {{ vector_code | autoindent }}
        {{ _gtot_all }}[_idx] = _gtot;
        {{ _I0_all }}[_idx] = _I0;
        {{ _v_previous }}[_idx] = {{ v }}[_idx];
    }

    // STEP 2: for each section: solve three tridiagonal systems

    // system 2a: solve for _v_star
    for (int _i = 0; _i < _num_B - 1; _i++) {
        // first and last index of the i-th section
        int _j_start = {{ _starts }}[_i];
        int _j_end = {{ _ends }}[_i];

        // upper triangularization of tridiagonal system for _v_star, _u_plus, and _u_minus
        for (int _j = _j_start; _j < _j_end; _j++) {
            {{ _v_star }}[_j] = -({{ Cm }}[_j] / {{ dt }} * {{ v }}[_j]) - {{ _I0_all }}[_j];  // RHS -> _v_star
            {{ _u_plus }}[_j] = {{ _b_plus }}[_j];  // RHS -> _u_plus
            {{ _u_minus }}[_j] = {{ _b_minus }}[_j];  // RHS -> _u_minus
            double _bi = {{ _ab_star1 }}[_j] - {{ _gtot_all }}[_j];  // main diagonal
            if (_j < N - 1) {
                {{ _c }}[_j] = {{ _ab_star0 }}[_j + 1];  // superdiagonal
            }
            if (_j > 0) {
                double _ai = {{ _ab_star2 }}[_j - 1];  // subdiagonal
                double _m = 1.0 / (_bi - _ai * {{ _c }}[_j - 1]);
                {{ _c }}[_j] = {{ _c }}[_j] * _m;
                {{ _v_star }}[_j] = ({{ _v_star }}[_j] - _ai * {{ _v_star }}[_j - 1]) * _m;
                {{ _u_plus }}[_j] = ({{ _u_plus }}[_j] - _ai * {{ _u_plus }}[_j - 1]) * _m;
                {{ _u_minus }}[_j] = ({{ _u_minus }}[_j] - _ai * {{ _u_minus }}[_j - 1]) * _m;
            } else {
                {{ _c }}[0] = {{ _c }}[0] / _bi;
                {{ _v_star }}[0] = {{ _v_star }}[0] / _bi;
                {{ _u_plus }}[0] = {{ _u_plus }}[0] / _bi;
                {{ _u_minus }}[0] = {{ _u_minus }}[0] / _bi;
            }
        }
        // backwards substitution of the upper triangularized system
        for (int _j = _j_end - 2; _j >= _j_start; _j--) {
            {{ _v_star }}[_j] = {{ _v_star }}[_j] - {{ _c }}[_j] * {{ _v_star }}[_j + 1];
            {{ _u_plus }}[_j] = {{ _u_plus }}[_j] - {{ _c }}[_j] * {{ _u_plus }}[_j + 1];
            {{ _u_minus }}[_j] = {{ _u_minus }}[_j] - {{ _c }}[_j] * {{ _u_minus }}[_j + 1];
        }
    }

    // STEP 3: solve the coupling system

    // indexing for _P_children
    int _children_rowlength = _num_morph_children / _num_morph_children_num;

    // STEP 3a: construct the coupling system with matrix _P in sparse form
    for (int _i = 0; _i < _num_B - 1; _i++) {
        int _i_parent = {{ _morph_parent_i }}[_i];
        int _i_childind = {{ _morph_idxchild }}[_i];
        int _first = {{ _starts }}[_i];
        int _last = {{ _ends }}[_i] - 1;
        double _this_invr0 = {{ _invr0 }}[_i];
        double _this_invrn = {{ _invrn }}[_i];

        // Towards parent
        if (_i == 0) {  // first section, sealed end
            {{ _P_diag }}[0] = {{ _u_minus }}[_first] - 1;
            {{ _P_children }}[0 + 0] = {{ _u_plus }}[_first];
            // RHS
            {{ _B }}[0] = -{{ _v_star }}[_first];
        } else {
            {{ _P_diag }}[_i_parent] += (1 - {{ _u_minus }}[_first]) * _this_invr0;
            {{ _P_children }}[_i_parent * _children_rowlength + _i_childind] = -{{ _u_plus }}[_first] * _this_invr0;
            // RHS
            {{ _B }}[_i_parent] += {{ _v_star }}[_first] * _this_invr0;
        }

        // Towards children
        {{ _P_diag }}[_i + 1] = (1 - {{ _u_plus }}[_last]) * _this_invrn;
        {{ _P_parent }}[_i] = -{{ _u_minus }}[_last] * _this_invrn;
        // RHS
        {{ _B }}[_i + 1] = {{ _v_star }}[_last] * _this_invrn;
    }

    // STEP 3b: solve the linear system (O(n) sparse Gaussian elimination)

    // part 1: lower triangularization
    for (int _i = _num_B - 1; _i >= 0; _i--) {
        int _num_children = {{ _morph_children_num }}[_i];
        // for every child eliminate the corresponding matrix element of row i
        for (int _k = 0; _k < _num_children; _k++) {
            int _j = {{ _morph_children }}[_i * _children_rowlength + _k];  // child index
            // subtracting _subfac times the j-th from the i-th row
            double _subfac = {{ _P_children }}[_i * _children_rowlength + _k] / {{ _P_diag }}[_j];
            {{ _P_diag }}[_i] = {{ _P_diag }}[_i] - _subfac * {{ _P_parent }}[_j - 1];
            {{ _B }}[_i] = {{ _B }}[_i] - _subfac * {{ _B }}[_j];
        }
    }

    // part 2: forwards substitution
    {{ _B }}[0] = {{ _B }}[0] / {{ _P_diag }}[0];  // first section has no parent
    for (int _i = 1; _i < _num_B; _i++) {
        int _j = {{ _morph_parent_i }}[_i - 1];  // parent index
        {{ _B }}[_i] = {{ _B }}[_i] - {{ _P_parent }}[_i - 1] * {{ _B }}[_j];
        {{ _B }}[_i] = {{ _B }}[_i] / {{ _P_diag }}[_i];
    }

    // STEP 4: for each section compute the final solution by linear combination
    for (int _i = 0; _i < _num_B - 1; _i++) {
        int _i_parent = {{ _morph_parent_i }}[_i];
        int _j_start = {{ _starts }}[_i];
        int _j_end = {{ _ends }}[_i];
        for (int _j = _j_start; _j < _j_end; _j++) {
            if (_j < _numv) {  // don't go beyond the last element
                {{ v }}[_j] = ({{ _v_star }}[_j] + {{ _B }}[_i_parent] * {{ _u_minus }}[_j]
                                                  + {{ _B }}[_i + 1] * {{ _u_plus }}[_j]);
            }
        }
    }

    for (int _i = 0; _i < N; _i++) {
        {{ Ic }}[_i] = {{ Cm }}[_i] * ({{ v }}[_i] - {{ _v_previous }}[_i]) / {{ dt }};
    }
{% endblock %}
