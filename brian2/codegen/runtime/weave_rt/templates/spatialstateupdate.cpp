////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////
{% import 'common_macros.cpp' as common with context %}

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  _P, _B, _morph_i, _morph_parent_i, _starts, _ends,
                  _invr0, _invrn} #}

{% macro main() %}
    {{ common.insert_group_preamble() }}

	double *_gtot_all=(double *)malloc(N*sizeof(double));
	double *c=(double *)malloc(N*sizeof(double));
	double ai,bi,_m;

	//// MAIN CODE ////////////
	// Tridiagonal solving
	// Pass 1
	for(int i=0;i<N;i++)
	{
		const int _idx = i;
	    const int _vectorisation_idx = _idx;

		{{vector_code|autoindent}}
		_gtot_all[_idx]=_gtot;

		{{v_star}}[i]=-({{Cm}}[i]/dt*{{v}}[i])-_I0; // RHS -> v_star (solution)
		bi={{ab_star1}}[i]-_gtot_all[i]; // main diagonal
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
		{{v_star}}[i]={{v_star}}[i] - c[i]*{{v_star}}[i+1];
	
	// Pass 2
	for(int i=0;i<N;i++)
	{
		{{u_plus}}[i]={{b_plus}}[i]; // RHS -> v_star (solution)
		bi={{ab_plus1}}[i]-_gtot_all[i]; // main diagonal
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
		bi={{ab_minus1}}[i]-_gtot_all[i]; // main diagonal
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

	free(_gtot_all);
	free(c);

    // Prepare matrix for solving the linear system
    std::fill_n({{_B}}, _num_B, 0.0);
    std::fill_n({{_P}}, _num_P, 0.0);
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

{% endmacro %}

{% macro support_code() %}
{% block support_code_block %}
	{{ common.support_code() }}
{% endblock %}
{% endmacro %}
