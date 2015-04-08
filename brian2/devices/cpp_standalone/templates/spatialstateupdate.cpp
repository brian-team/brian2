////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  gtot_all, I0_all
                  _P, _B, _morph_i, _morph_parent_i, _starts, _ends,
                  _invr0, _invrn} #}
{% extends 'common_group.cpp' %}
{% block maincode %}

	double ai,bi,_m;
    static double *c = (double*)malloc(N * sizeof(double));
    int _vectorisation_idx = 1;

	//// MAIN CODE ////////////
	{{scalar_code|autoindent}}

	// Tridiagonal solving
	// Pass 1
	for(int i=0;i<N;i++)
	{
		const int _idx = i;
	    _vectorisation_idx = _idx;

		{{vector_code|autoindent}}
		{{gtot_all}}[_idx] = _gtot;
        {{I0_all}}[_idx] = _I0;
    }

    for(int i=0;i<N;i++)
    {
		{{v_star}}[i]=-({{Cm}}[i]/dt*{{v}}[i])-{{I0_all}}[i]; // RHS -> v_star (solution)
		bi={{ab_star1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			c[i]={{ab_star0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_star2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			{{v_star}}[i]=({{v_star}}[i] - ai*{{v_star}}[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{v_star}}[0]={{v_star}}[0]/bi;
		}
	}
	for(int i=N-2;i>=0;i--)
	{
		{{v_star}}[i]={{v_star}}[i] - c[i]*{{v_star}}[i+1];
    }
	// Pass 2
	for(int i=0;i<N;i++)
	{
		{{u_plus}}[i]={{b_plus}}[i]; // RHS -> v_star (solution)
		bi={{ab_plus1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			c[i]={{ab_plus0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_plus2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			{{u_plus}}[i]=({{u_plus}}[i] - ai*{{u_plus}}[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{u_plus}}[0]={{u_plus}}[0]/bi;
		}
	}
	for(int i=N-2;i>=0;i--)
		{{u_plus}}[i]={{u_plus}}[i] - c[i]*{{u_plus}}[i+1];

	// Pass 3
	for(int i=0;i<N;i++)
	{
		{{u_minus}}[i]={{b_minus}}[i]; // RHS -> v_star (solution)
		bi={{ab_minus1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			c[i]={{ab_minus0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_minus2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			{{u_minus}}[i]=({{u_minus}}[i] - ai*{{u_minus}}[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			{{u_minus}}[0]={{u_minus}}[0]/bi;
		}
	}
	for(int i=N-2;i>=0;i--)
		{{u_minus}}[i]={{u_minus}}[i] - c[i]*{{u_minus}}[i+1];


    // Prepare matrix for solving the linear system

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

    // Solve the linear system (the result will be in _B in the end)
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
            const double pivot_factor = {{_P}}[j*_num_B + i]/{{_P}}[i*_num_B + i];
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

    // Linear combination
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
