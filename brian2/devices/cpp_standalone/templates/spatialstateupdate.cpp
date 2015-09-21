////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  gtot_all, I0_all,
                  c1, c2, c3,
                  _P, _B, _morph_i, _morph_parent_i, _starts, _ends,
                  _morph_children_i, _morph_children_num_i
                  _invr0, _invrn} #}
{% extends 'common_group.cpp' %}
{% block maincode %}

	double ai,bi,_m;

    int _vectorisation_idx = 1;

	//// MAIN CODE ////////////
	{{scalar_code|autoindent}}

	// integration step 1: compute g_total and I_0 (independent: compartments)
	{{ openmp_pragma('parallel-static') }}
	for(int i=0;i<N;i++) // computing g_total and I_0
	{
		const int _idx = i;
		_vectorisation_idx = _idx;

		{{vector_code|autoindent}}
		{{gtot_all}}[_idx] = _gtot;
		{{I0_all}}[_idx] = _I0;
	}


	// integration step 2: for each branch: solve three tridiagonal systems (independent: branches & nested: the 3 linear systems)
    {{ openmp_pragma('enable_nested') }} // allow inner parallelization of the three linear systems

    {{ openmp_pragma('parallel') }}
	{{ openmp_pragma('sections') }} // system 2a: solve for v_star
	{
		{{ openmp_pragma('parallel-static') }} // nested parallelism
		for (int _i=0; _i<_num_morph_i; _i++)
		{
			// first and last index of the i-th branch
			const int i_start = {{_starts}}[_i];
			const int i_end = {{_ends}}[_i];

			// upper triangularization of tridiagonal system for v_star
			for(int i=i_start;i<i_end+1;i++)
			{
				{{v_star}}[i]=-({{Cm}}[i]/{{dt}}*{{v}}[i])-{{I0_all}}[i]; // RHS -> v_star (solution)
				bi={{ab_star1}}[i]-{{gtot_all}}[i]; // main diagonal
				if (i<N-1)
					{{c1}}[i]={{ab_star0}}[i+1]; // superdiagonal
				if (i>0)
				{
					ai={{ab_star2}}[i-1]; // subdiagonal
					_m=1.0/(bi-ai*{{c1}}[i-1]);
					{{c1}}[i]={{c1}}[i]*_m;
					{{v_star}}[i]=({{v_star}}[i] - ai*{{v_star}}[i-1])*_m;
				} else
				{
					{{c1}}[0]={{c1}}[0]/bi;
					{{v_star}}[0]={{v_star}}[0]/bi;
				}
			}
			// backwards substituation of the upper triangularized system for v_star
			for(int i=i_end-1;i>=i_start;i--)
				{{v_star}}[i]={{v_star}}[i] - {{c1}}[i]*{{v_star}}[i+1];
		}
	}
	{{ openmp_pragma('sections') }} // system 2b: solve for u_plus
	{
		{{ openmp_pragma('parallel-static') }}
		for (int _i=0; _i<_num_morph_i; _i++)
		{
			// first and last index of the i-th branch
			const int i_start = {{_starts}}[_i];
			const int i_end = {{_ends}}[_i];

			// upper triangularization of tridiagonal system for u_plus
			for(int i=i_start;i<i_end+1;i++)
			{
				{{u_plus}}[i]={{b_plus}}[i]; // RHS -> u_plus (solution)
				bi={{ab_plus1}}[i]-{{gtot_all}}[i]; // main diagonal
				if (i<N-1)
					{{c2}}[i]={{ab_plus0}}[i+1]; // superdiagonal
				if (i>0)
				{
					ai={{ab_plus2}}[i-1]; // subdiagonal
					_m=1.0/(bi-ai*{{c2}}[i-1]);
					{{c2}}[i]={{c2}}[i]*_m;
					{{u_plus}}[i]=({{u_plus}}[i] - ai*{{u_plus}}[i-1])*_m;
				} else
				{
					{{c2}}[0]={{c2}}[0]/bi;
					{{u_plus}}[0]={{u_plus}}[0]/bi;
				}
			}
			// backwards substituation of the upper triangularized system for u_plus
			for(int i=i_end-1;i>=i_start;i--)
				{{u_plus}}[i]={{u_plus}}[i] - {{c2}}[i]*{{u_plus}}[i+1];
		}
	}
	{{ openmp_pragma('sections') }} // system 2c: solve for u_minus
	{
		{{ openmp_pragma('parallel-static') }}
		for (int _i=0; _i<_num_morph_i; _i++)
		{
			// first and last index of the i-th branch
			const int i_start = {{_starts}}[_i];
			const int i_end = {{_ends}}[_i];

			// upper triangularization of tridiagonal system for u_minus
			for(int i=i_start;i<i_end-1;i++)
			{
				{{u_minus}}[i]={{b_minus}}[i]; // RHS -> u_minus (solution)
				bi={{ab_minus1}}[i]-{{gtot_all}}[i]; // main diagonal
				if (i<N-1)
					{{c3}}[i]={{ab_minus0}}[i+1]; // superdiagonal
				if (i>0)
				{
					ai={{ab_minus2}}[i-1]; // subdiagonal
					_m=1.0/(bi-ai*{{c3}}[i-1]);
					{{c3}}[i]={{c3}}[i]*_m;
					{{u_minus}}[i]=({{u_minus}}[i] - ai*{{u_minus}}[i-1])*_m;
				} else
				{
					{{c3}}[0]={{c3}}[0]/bi;
					{{u_minus}}[0]={{u_minus}}[0]/bi;
				}
			}
			// backwards substituation of the upper triangularized system for u_minus
			for(int i=i_end-1;i>=i_start;i--)
				{{u_minus}}[i]={{u_minus}}[i] - {{c3}}[i]*{{u_minus}}[i+1];
		}
	}

    {{ openmp_pragma('disable_nested') }} // really necessary?

    // integration step 3: solve the coupling system (no parallelism)

    // step 3a: construct the coupling matrix _B
    for (int _j=0; _j<_num_B - 1; _j++)
    {
        const int _i = {{_morph_i}}[_j];
        const int _i_parent = {{_morph_parent_i}}[_j];
        const int _first = {{_starts}}[_j];
        const int _last = {{_ends}}[_j];
        const double _invr0 = {{_invr0}}[_j];
        const double _invrn = {{_invrn}}[_j];
        // Towards parent
        if (_i == 1) // first branch, sealed end
        {
            {{_P}}[0] = {{u_minus}}[_first] - 1;
            {{_P}}[0 + 1] = {{u_plus}}[_first];
            {{_B}}[0] = -{{v_star}}[_first];
        }
        else
        {
            {{_P}}[_i_parent*_num_B + _i_parent] += (1 - {{u_minus}}[_first]) * _invr0;
            {{_P}}[_i_parent*_num_B + _i] -= {{u_plus}}[_first] * _invr0;
            {{_B}}[_i_parent] += {{v_star}}[_first] * _invr0;
        }
        // Towards children
        {{_P}}[_i*_num_B + _i] = (1 - {{u_plus}}[_last]) * _invrn;
        {{_P}}[_i*_num_B + _i_parent] = -{{u_minus}}[_last] * _invrn;
        {{_B}}[_i] = {{v_star}}[_last] * _invrn;
    }


    // step 3b: solve the linear system (the result will be in _B in the end)

    // easier indexing
	#define IDX_B(idx_row,idx_col) _num_B * idx_row + idx_col // indexing for the coupling matrix _B

    // check for assumption of proper ordering of the branches, i.e. _i_parent < _i
	bool proper_branch_ordering = true;
	for (int _j=0; _j < _num_B-1; _j++) {
		const int _i = {{_morph_i}}[_j];
		const int _i_parent = {{_morph_parent_i}}[_j];
		if (_i_parent >= _i) {
			proper_branch_ordering = false;
			cout << "WARNING: the branch ordering is wrong for _j= " << _j
				 << "(_i=" << _i << " with _i_parent=" << _i_parent << endl;
		}
	}
	// check: if not ordered properly (e.g. for a cyclic branch graph)
	//        we might still use the dense matrix version?

	// TODO: remove this check about proper ordering and the dense matrix version (else branch)
	if (proper_branch_ordering) { // use efficient O(n) solution of the linear system (structure-specific Gaussian elemination)
								  // exploiting the sparse structural symmetric matrix from branch tree

		const int children_rowlength = _num_morph_children_i/_num_morph_children_num_i;

		// part 1: lower triangularization
		for (int i=_num_B; i>=0; i--) {
			const int num_children = {{_morph_children_num_i}}[i+1]; // using the morph_i[i] = i+1

			// for every child eliminate the corresponding matrix element of row i
			for (int k=0; k<num_children; k++) {
				int j = {{_morph_children_i}}[i*children_rowlength+k]; // child index
				// subtracting subfac times the j-th from the i-th row
				double subfac = {{_P}}[IDX_B(i,j)] / {{_P}}[IDX_B(j,j)]; // P[i,j] appears only here
				{{_P}}[IDX_B(i,j)] = 0; // not used in the following anymore, just for clarity
				{{_P}}[IDX_B(i,i)] = {{_P}}[IDX_B(i,i)]  - subfac * {{_P}}[IDX_B(j,i)]; // note: element j,i is ONLY used here; maybe omit it if a sparse matrix format is introduced
				{{_B}}[i] = {{_B}}[i] - subfac * {{_B}}[i];
			}
		}

		// part 2: forwards substitution
		{{_B}}[0] = {{_B}}[0] / {{_P}}[IDX_B(0,0)]; // the first branch does not have a parent
		for (int i=1; i<=_num_B; i++) {
			const int j = {{_morph_parent_i}}[i]; // parent index
			{{_B}}[i] = {{_B}}[i] - {{_P}}[IDX_B(i,j)] * {{_B}}[j];
			{{_B}}[i] = {{_B}}[i] / {{_P}}[IDX_B(i,i)];
		}
	} // efficient solution
	else { // use inefficient O(n^3) solution (dense Gaussian elemination)

		for (int i=0; i<_num_B; i++)
		{
			// find pivot element
			int i_pivot = i;
			double pivot_magnitude = fabs({{_P}}[i*_num_B + i]);
			for (int j=i+1; j<_num_B; j++)
			   if (fabs({{_P}}[j*_num_B + i]) > pivot_magnitude)
			   {
				   i_pivot = j;
				   pivot_magnitude = fabs({{_P}}[j*_num_B + i]);
			   }

			if (pivot_magnitude == 0)
			{
				std::cerr << "Singular!" << std::endl;
			}

			// swap rows
			if (i != i_pivot)
			{
				for (int col=i; col<_num_B; col++)
				{
					const double tmp = {{_P}}[i*_num_B + col];
					{{_P}}[i*_num_B + col] = {{_P}}[i_pivot*_num_B + col];
					{{_P}}[i_pivot*_num_B + col] = tmp;
				}
			}

			// Deal with rows below
			for (int j=i+1; j<_num_B; j++)
			{
				const double pivot_element = {{_P}}[j*_num_B + i];
				if (pivot_element == 0.0)
					continue;
				const double pivot_factor = pivot_element/{{_P}}[i*_num_B + i];
				for (int k=i+1; k<_num_B; k++)
				{
					{{_P}}[j*_num_B + k] -= {{_P}}[i*_num_B + k]*pivot_factor;
				}
				{{_B}}[j] -= {{_B}}[i]*pivot_factor;
				{{_P}}[j*_num_B + i] = 0;
			}

		}

		// Back substitution
		for (int i=_num_B-1; i>=0; i--)
		{
			// substitute all the known values
			for (int j=_num_B-1; j>i; j--)
			{
				{{_B}}[i] -= {{_P}}[i*_num_B + j]*{{_B}}[j];
				{{_P}}[i*_num_B + j] = 0;
			}
			// divide by the diagonal element
			{{_B}}[i] /= {{_P}}[i*_num_B + i];
			{{_P}}[i*_num_B + i] = 1;
		}
	}

    // integration step 4: for each branch compute the final solution by linear combination of the general solution (independent: branches & compartments)
    for (int _j=0; _j<_num_B - 1; _j++)
    {
        const int _i = {{_morph_i}}[_j];
        const int _i_parent = {{_morph_parent_i}}[_j];
        const int _first = {{_starts}}[_j];
        const int _last = {{_ends}}[_j];
        for (int _k=_first; _k<_last + 1; _k++)
            if (_k < _numv)  // don't go beyond the last element
                {{v}}[_k] = {{v_star}}[_k] + {{_B}}[_i_parent] * {{u_minus}}[_k]
                                           + {{_B}}[_i] * {{u_plus}}[_k];
    }

{% endblock %}
