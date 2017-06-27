struct parameters p;
double y[2];

gsl_odeiv2_system sys;
sys.function = func;
sys.dimension = 2;
sys.params = &p;

double h = 1e-6;
double eps_abs = 1e-8;
double eps_rel = 1e-10;

gsl_odeiv2_driver * d = 
    gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, h, eps_abs, eps_rel);

    //double* __restrict  _ptr_array_neurongroup_vt = _array_neurongroup_vt;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    //double* __restrict  _ptr_array_neurongroup_v = _array_neurongroup_v;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    p.vt = _array_neurongroup_vt;
    p.v = _array_neurongroup_v;
    //// MAIN CODE ////////////
    // This allows everything to work correctly for synapses where N is not a
    // constant
    const int _N = N;
    // scalar code
    const int _vectorisation_idx = 1;
    
 double dt = _array_defaultclock_dt[0];
 //const double _lio_1 = 0.0666666666666667 / ms;
 //const double _lio_2 = 10.0 * mV;
 //const double _lio_3 = (-0.1) / ms;
 p._lio_1 = 0.0666666666666667 / ms;    
 p._lio_2 = 10.0 * mV;
 p._lio_3 = (-0.1) / ms;

    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const int _vectorisation_idx = _idx;
        double t = _array_defaultclock_t[0];
        double t1 = t + dt;

        fill_y_vector(&p, y, _idx);

        int status = gsl_odeiv2_driver_apply(d, &t, t1, y);
        gsl_odeiv2_driver_reset(d);

        empty_y_vector(&p, y, _idx);
                
    }
