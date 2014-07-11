////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////
{% import 'common_macros.cpp' as common with context %}

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star, b_plus, ab_plus, b_minus, ab_minus, v_star, u_plus, u_minus} #}

{% macro main() %}
    {{ common.insert_group_preamble() }}

    #define AB_STAR2(i,j) (*((double*)({{ab_star}} + i*N + j)))
    #define AB_MINUS2(i,j) (*((double*)({{ab_minus}} + i*N + j)))
    #define AB_PLUS2(i,j) (*((double*)({{ab_plus}} + i*N + j)))

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
		bi=AB_STAR2(1,i)-_gtot_all[i]; // main diagonal
		if (i<N-1)
			c[i]=AB_STAR2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_STAR2(2,i-1); // subdiagonal
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
		bi=AB_PLUS2(1,i)-_gtot_all[i]; // main diagonal
		if (i<N-1)
			c[i]=AB_PLUS2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_PLUS2(2,i-1); // subdiagonal
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
		bi=AB_MINUS2(1,i)-_gtot_all[i]; // main diagonal
		if (i<N-1)
			c[i]=AB_MINUS2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_MINUS2(2,i-1); // subdiagonal
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
{% endmacro %}

{% macro support_code() %}
{% block support_code_block %}
	{{ common.support_code() }}
{% endblock %}
{% endmacro %}
