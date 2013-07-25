////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro main() %}
	// USE_SPECIFIERS {  Cm, dt, v, _num_neurons,
    //                   ab_star, b_plus, ab_plus, b_minus, ab_minus, v_star, u_plus, u_minus }

    ////// SUPPORT CODE ///
	{% for line in support_code_lines %}
	//{{line}}
	{% endfor %}

	////// HANDLE DENORMALS ///
	{% for line in denormals_code_lines %}
	{{line}}
	{% endfor %}

	////// HASH DEFINES ///////
	{% for line in hashdefine_lines %}
	{{line}}
	{% endfor %}

	///// POINTERS ////////////
	{% for line in pointers_lines %}
	{{line}}
	{% endfor %}

	double *_gtot_all=malloc(_num_neurons*sizeof(double));

	//// MAIN CODE ////////////
	for(int _neuron_idx=0; _neuron_idx<_num_neurons; _neuron_idx++)
	{
	    const int _vectorisation_idx = _neuron_idx;
		double m;

		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		_gtot_all[_neuron_idx]=_gtot;
	}
	
	// Tridiagonal solving
	c[0]=c[0]/b[0];
	v_star[0]=((Cm[0]/dt*v[0])-_I0[0])/b[0];	
	for(int i=1;i<_num_neurons;i++)
	{
		double m=1.0/(b[i]-a[i]*c[i-1]);
		c[i]=c[i]*m;
		v_star[i]=-((Cm[i]/dt*v[i])-_I0[0] - a*v_star[i-1])*m;
	}
	for(int i=_num_neurons-1;i>0;i--)
	{
		v_star[i]=v_star[i] - c[i]*x[i+1];
	}
	
	free(_gtot_all);
	free(_I0_all);
	free(b);
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
{% endmacro %}
