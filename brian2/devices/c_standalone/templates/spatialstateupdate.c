////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  _P, _B, _morph_i, _morph_parent_i, _starts, _ends,
                  _invr0, _invrn} #}
{% extends 'common_group.c' %}
{% block maincode %}
	double _gtot_all[N];
	double c[N] = {0,};
	double ai,bi,_m;

    int _vectorisation_idx = 1;

    int _i, _j;

	//// MAIN CODE ////////////
	{{scalar_code|autoindent}}

	// Tridiagonal solving
	// Pass 1
	for(_i=0;_i<N;_i++)
	{
		const int _idx = _i;
	    _vectorisation_idx = _idx;
		{{vector_code|autoindent}}
		_gtot_all[_idx]=_gtot;

		{{v_star}}[_i]=-({{Cm}}[_i]/dt*{{v}}[_i])-_I0; // RHS -> v_star (solution)
		bi={{ab_star1}}[_i]-_gtot_all[_i]; // main diagonal
		if (_i<N-1)
			c[_i]={{ab_star0}}[_i+1]; // superdiagonal
		if (_i>0)
		{
			ai={{ab_star2}}[_i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[_i-1]);
			c[_i]=c[_i]*_m;
			{{v_star}}[_i]=({{v_star}}[_i] - ai*{{v_star}}[_i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{v_star}}[0]={{v_star}}[0]/bi;
		}
	}
	for(_i=N-2;_i>=0;_i--)
	{
		{{v_star}}[_i]={{v_star}}[_i] - c[_i]*{{v_star}}[_i+1];
    }
	// Pass 2
	for(_i=0;_i<N;_i++)
	{
		{{u_plus}}[_i]={{b_plus}}[_i]; // RHS -> v_star (solution)
		bi={{ab_plus1}}[_i]-_gtot_all[_i]; // main diagonal
		if (_i<N-1)
			c[_i]={{ab_plus0}}[_i+1]; // superdiagonal
		if (_i>0)
		{
			ai={{ab_plus2}}[_i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[_i-1]);
			c[_i]=c[_i]*_m;
			{{u_plus}}[_i]=({{u_plus}}[_i] - ai*{{u_plus}}[_i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{u_plus}}[0]={{u_plus}}[0]/bi;
		}
	}
	for(_i=N-2;_i>=0;_i--)
		{{u_plus}}[_i]={{u_plus}}[_i] - c[_i]*{{u_plus}}[_i+1];
	
	// Pass 3
	for(_i=0;_i<N;_i++)
	{
		{{u_minus}}[_i]={{b_minus}}[_i]; // RHS -> v_star (solution)
		bi={{ab_minus1}}[_i]-_gtot_all[_i]; // main diagonal
		if (_i<N-1)
			c[_i]={{ab_minus0}}[_i+1]; // superdiagonal
		if (_i>0)
		{
			ai={{ab_minus2}}[_i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[_i-1]);
			c[_i]=c[_i]*_m;
			{{u_minus}}[_i]=({{u_minus}}[_i] - ai*{{u_minus}}[_i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{u_minus}}[0]={{u_minus}}[0]/bi;
		}
	}
	for(_i=N-2;_i>=0;_i--)
		{{u_minus}}[_i]={{u_minus}}[_i] - c[_i]*{{u_minus}}[_i+1];

    // Prepare matrix for solving the linear system

    for (_j=0; _j<_num_B - 1; _j++)
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

    // Solve the linear system (the result will be in _B in the end)
    for (_i=0; _i<_num_B; _i++)
    {
        // find pivot element
        int i_pivot = _i;
        double pivot_magnitude = fabs({{_P}}[_i*_num_B + _i]);
        for (int j=_i+1; j<_num_B; j++)
           if (fabs({{_P}}[j*_num_B + _i]) > pivot_magnitude)
           {
               i_pivot = j;
               pivot_magnitude = fabs({{_P}}[j*_num_B + _i]);
           }

        if (pivot_magnitude == 0)
        {
            fprintf(stderr, "Singular!\n");
        }

        // swap rows
        if (_i != i_pivot)
        {
            for (int col=_i; col<_num_B; col++)
            {
                const double tmp = {{_P}}[_i*_num_B + col];
                {{_P}}[_i*_num_B + col] = {{_P}}[i_pivot*_num_B + col];
                {{_P}}[i_pivot*_num_B + col] = tmp;
            }
        }

        // Deal with rows below
        for (_j=_i+1; _j<_num_B; _j++)
        {
            const double pivot_factor = {{_P}}[_j*_num_B + _i]/{{_P}}[_i*_num_B + _i];
            for (int k=_i+1; k<_num_B; k++)
            {
                {{_P}}[_j*_num_B + k] -= {{_P}}[_i*_num_B + k]*pivot_factor;
            }
            {{_B}}[_j] -= {{_B}}[_i]*pivot_factor;
            {{_P}}[_j*_num_B + _i] = 0;
        }

    }

    // Back substitution
    for (_i=_num_B-1; _i>=0; _i--)
    {
        // substitute all the known values
        for (_j=_num_B-1; _j>_i; _j--)
        {
            {{_B}}[_i] -= {{_P}}[_i*_num_B + _j]*{{_B}}[_j];
            {{_P}}[_i*_num_B + _j] = 0;
        }
        // divide by the diagonal element
        {{_B}}[_i] /= {{_P}}[_i*_num_B + _i];
        {{_P}}[_i*_num_B + _i] = 1;
    }

    // Linear combination
    for (_j=0; _j<_num_B - 1; _j++)
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
