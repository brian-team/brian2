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

	double *_gtot_all=(double *)malloc(_num_neurons*sizeof(double));
	double *c=(double *)malloc(_num_neurons*sizeof(double));
	double ai,bi,_m;

	//// MAIN CODE ////////////
	// Tridiagonal solving
	// Pass 1
	for(int i=0;i<_num_neurons;i++)
	{
		const int _neuron_idx = i;
	    const int _vectorisation_idx = _neuron_idx;

		{% for line in code_lines %}
		{{line}}
		{% endfor %}
		_gtot_all[_neuron_idx]=_gtot;

		v_star[i]=-(Cm[i]/dt*v[i])-_I0; // RHS -> v_star (solution)
		bi=AB_STAR2(1,i)-_gtot_all[i]; // main diagonal
		if (i<_num_neurons-1)
			c[i]=AB_STAR2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_STAR2(2,i-1); // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			v_star[i]=(v_star[i] - ai*v_star[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			v_star[0]=v_star[0]/bi;
		}
	}
	for(int i=_num_neurons-2;i>=0;i--)
		v_star[i]=v_star[i] - c[i]*v_star[i+1];
	
	// Pass 2
	for(int i=0;i<_num_neurons;i++)
	{
		u_plus[i]=b_plus[i]; // RHS -> v_star (solution)
		bi=AB_PLUS2(1,i)-_gtot_all[i]; // main diagonal
		if (i<_num_neurons-1)
			c[i]=AB_PLUS2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_PLUS2(2,i-1); // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			u_plus[i]=(u_plus[i] - ai*u_plus[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			u_plus[0]=u_plus[0]/bi;
		}
	}
	for(int i=_num_neurons-2;i>=0;i--)
		u_plus[i]=u_plus[i] - c[i]*u_plus[i+1];
	
	// Pass 3
	for(int i=0;i<_num_neurons;i++)
	{
		u_minus[i]=b_minus[i]; // RHS -> v_star (solution)
		bi=AB_MINUS2(1,i)-_gtot_all[i]; // main diagonal
		if (i<_num_neurons-1)
			c[i]=AB_MINUS2(0,i+1); // superdiagonal
		if (i>0)
		{
			ai=AB_MINUS2(2,i-1); // subdiagonal
			_m=1.0/(bi-ai*c[i-1]);
			c[i]=c[i]*_m;
			u_minus[i]=(u_minus[i] - ai*u_minus[i-1])*_m;
		} else
		{
			c[0]=c[0]/bi;
			u_minus[0]=u_minus[0]/bi;
		}
	}
	for(int i=_num_neurons-2;i>=0;i--)
		u_minus[i]=u_minus[i] - c[i]*u_minus[i+1];
	
	free(_gtot_all);
	free(c);
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// SUPPORT CODE //////////////////////////////////////////////////////////

{% macro support_code() %}
	{% for line in support_code_lines %}
	{{line}}
	{% endfor %}
{% endmacro %}
