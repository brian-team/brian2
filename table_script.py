from brian2 import *
from table_functions import *
from numpy.random import randint, rand
from brian2.core.network import ProfilingSummary

runtime = 1*second
plotf = True

prefs.codegen.target = 'weave'
#prefs.codegen.target = 'numpy'
#prefs.codegen.target = 'cython'


# Parameters
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area

El = -60*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area

# Time constants
taue = 5*ms
taui = 10*ms
# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 6*nS  # excitatory synaptic weight
wi = 67*nS  # inhibitory synaptic weight

eqs_interpolation = Equations('''
dv/dt = (I_input+gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = alpha_m_interpolation(v_idx) : Hz (constant over dt)
beta_m = beta_m_interpolation(v_idx) : Hz (constant over dt)
alpha_h = alpha_h_interpolation(v_idx) : Hz (constant over dt)
beta_h = beta_h_interpolation(v_idx) : Hz (constant over dt)
alpha_n = alpha_n_interpolation(v_idx) : Hz (constant over dt)
beta_n = beta_n_interpolation(v_idx) : Hz (constant over dt)
v_idx = int_calculator(v) : 1 (constant over dt)
I_input : amp
''')

eqs_table = Equations('''
dv/dt = (I_input+gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m  *(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = alpha_m_table(v_idx) : Hz (constant over dt)
beta_m = beta_m_table(v_idx) : Hz (constant over dt)
alpha_h = alpha_h_table(v_idx) : Hz (constant over dt)
beta_h = beta_h_table(v_idx) : Hz (constant over dt)
alpha_n = alpha_n_table(v_idx) : Hz (constant over dt)
beta_n = beta_n_table(v_idx) : Hz (constant over dt)
v_idx = int_calculator(v) : 1 (constant over dt)
I_input : amp
''')


eqs_interpolation = Equations('''
dv/dt = (I_input+gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = alpha_m_interpolation(v_idx) : Hz 
beta_m = beta_m_interpolation(v_idx) : Hz 
alpha_h = alpha_h_interpolation(v_idx) : Hz 
beta_h = beta_h_interpolation(v_idx) : Hz 
alpha_n = alpha_n_interpolation(v_idx) : Hz 
beta_n = beta_n_interpolation(v_idx) : Hz 
v_idx = int_calculator_interpolation(v) : 1
I_input : amp
''')

eqs_table = Equations('''
dv/dt = (I_input+gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m  *(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = alpha_m_table(v_idx) : Hz 
beta_m = beta_m_table(v_idx) : Hz 
alpha_h = alpha_h_table(v_idx) : Hz 
beta_h = beta_h_table(v_idx) : Hz 
alpha_n = alpha_n_table(v_idx) : Hz 
beta_n = beta_n_table(v_idx) : Hz 
v_idx = int_calculator_table(v) : 1
I_input : amp
''')

eqs_calc = Equations('''
dv/dt = (I_input+gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m_val*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m_val = 0.32*(mV**-1)*(13*mV-v+VT)/
         (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
         (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
I_input : amp
''')


def run_net(equations, N, v_res, integrationmethod, dt=.1*ms):
    VT = -63*mV

    v = Quantity(arange(-100*mV, 65*mV, v_res), dim=volt.dim)
    _values_alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/(exp((13*mV-v+VT)/(4*mV))-1.)/ms
    _values_beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/(exp((v-VT-40*mV)/(5*mV))-1)/ms
    _values_alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms
    _values_beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms
    _values_alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/(exp((15*mV-v+VT)/(5*mV))-1.)/ms
    _values_beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms
    del v

    @implementation('cpp',
                    code='''
    double int_calculator_table(double v)
    {
        return ((v + 100*0.001)/(%f) + .5);
    }
    '''%(v_res))
    @implementation('cython',
                    code='''
    cdef int int_calculator_table(double v):
        return int((v + 100*0.001)/(%f)+0.5)
    '''%(v_res))
    @check_units(v=volt,result=1)
    def int_calculator_table(v):
        return int_((v + 100*mV)/(v_res)+0.5)

    @implementation('cpp', namespace={'_values_alpha_m': array(_values_alpha_m)},
                    code='''
    double alpha_m_table(double index)
    {
        return _namespace_values_alpha_m[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_m_table(double index):
        global _namespace_values_alpha_m
        return _namespace_values_alpha_m[int(index)]
    ''', namespace={'_values_alpha_m':_values_alpha_m}
    )
    @check_units(idx=1,result=Hz)
    def alpha_m_table(idx):
        return _values_alpha_m[idx]

    @implementation('cpp', namespace={'_values_beta_m': array(_values_beta_m)},
                    code='''
    double beta_m_table(double index)
    {
        return _namespace_values_beta_m[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_m_table(double index):
        global _namespace_values_beta_m
        return _namespace_values_beta_m[int(index)]
    ''', namespace={'_values_beta_m':_values_beta_m}
    )
    @check_units(idx=1,result=Hz)
    def beta_m_table(idx):
        return _values_beta_m[idx]

    @implementation('cpp', namespace={'_values_alpha_h': array(_values_alpha_h)},
                    code='''
    double alpha_h_table(double index)
    {
        return _namespace_values_alpha_h[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_h_table(double index):
        global _namespace_values_alpha_h
        return _namespace_values_alpha_h[int(index)]
    ''', namespace={'_values_alpha_h':_values_alpha_h}
    )
    @check_units(idx=1,result=Hz)
    def alpha_h_table(idx):
        return _values_alpha_h[idx]

    @implementation('cpp', namespace={'_values_beta_h': array(_values_beta_h)},
                    code='''
    double beta_h_table(double index)
    {
        return _namespace_values_beta_h[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_h_table(double index):
        global _namespace_values_beta_h
        return _namespace_values_beta_h[int(index)]
    ''', namespace={'_values_beta_h':_values_beta_h}
    )
    @check_units(idx=1,result=Hz)
    def beta_h_table(idx):
        return _values_beta_h[idx]

    @implementation('cpp', namespace={'_values_alpha_n': array(_values_alpha_n)},
                    code='''
    double alpha_n_table(double index)
    {
        return _namespace_values_alpha_n[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_n_table(double index):
        global _namespace_values_alpha_n
        return _namespace_values_alpha_n[int(index)]
    ''', namespace={'_values_alpha_n':_values_alpha_n}
    )
    @check_units(idx=1,result=Hz)
    def alpha_n_table(idx):
        return _values_alpha_n[idx]

    @implementation('cpp', namespace={'_values_beta_n': array(_values_beta_n)},
                    code='''
    double beta_n_table(double index)
    {
        return _namespace_values_beta_n[int(index)];
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_n_table(double index):
        global _namespace_values_beta_n
        return _namespace_values_beta_n[int(index)]
    ''', namespace={'_values_beta_n':_values_beta_n}
    )
    @check_units(idx=1,result=Hz)
    def beta_n_table(idx):
        return _values_beta_n[idx]

    ###########
    @implementation('cpp',
                    code='''
    double int_calculator_interpolation(double v)
    {
        return ((v + 100*0.001)/(%f));
    }
    '''%(v_res))
    @implementation('cython',
                    code='''
    cdef int int_calculator_interpolation(double v):
        return int((v + 100*0.001)/(%f))
    '''%(v_res))
    @check_units(v=volt,result=1)
    def int_calculator_interpolation(v):
        return int_((v + 100*mV)/(v_res))

    @implementation('cpp', namespace={'_values_alpha_m': array(_values_alpha_m)},
                    code='''
    #include <math.h>
    double alpha_m_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_alpha_m[floor_ind];
        double v1 = _namespace_values_alpha_m[floor_ind+1];
        double vfrac = index-floor_ind;

        return (1 - vfrac) * v0 + vfrac * v1;
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_m_interpolation(double index):
        global _namespace_values_alpha_m
        return _namespace_values_alpha_m[int(index)]
    ''', namespace={'_values_alpha_m':_values_alpha_m}
    )
    @check_units(idx=1,result=Hz)
    def alpha_m_interpolation(idx):
        return _values_alpha_m[idx]

    @implementation('cpp', namespace={'_values_beta_m': array(_values_beta_m)},
                    code='''
    #include <math.h>
    double beta_m_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_beta_m[floor_ind];
        double v1 = _namespace_values_beta_m[floor_ind+1];
        double vfrac = index-floor_ind;

        return (1 - vfrac) * v0 + vfrac * v1;
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_m_interpolation(double index):
        global _namespace_values_beta_m
        return _namespace_values_beta_m[int(index)]
    ''', namespace={'_values_beta_m':_values_beta_m}
    )
    @check_units(idx=1,result=Hz)
    def beta_m_interpolation(idx):
        return _values_beta_m[idx]

    @implementation('cpp', namespace={'_values_alpha_h': array(_values_alpha_h)},
                    code='''
    #include <math.h>
    double alpha_h_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_alpha_h[floor_ind];
        double v1 = _namespace_values_alpha_h[floor_ind+1];
        double vfrac = index-floor_ind;

        return (1 - vfrac) * v0 + vfrac * v1;
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_h_interpolation(double index):
        global _namespace_values_alpha_h
        return _namespace_values_alpha_h[int(index)]
    ''', namespace={'_values_alpha_h':_values_alpha_h}
    )
    @check_units(idx=1,result=Hz)
    def alpha_h_interpolation(idx):
        return _values_alpha_h[idx]

    @implementation('cpp', namespace={'_values_beta_h': array(_values_beta_h)},
                    code='''
    #include <math.h>
    double beta_h_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_beta_h[floor_ind];
        double v1 = _namespace_values_beta_h[floor_ind+1];
        double vfrac = index-floor_ind;

        return (1 - vfrac) * v0 + vfrac * v1;
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_h_interpolation(double index):
        global _namespace_values_beta_h
        return _namespace_values_beta_h[int(index)]
    ''', namespace={'_values_beta_h':_values_beta_h}
    )
    @check_units(idx=1,result=Hz)
    def beta_h_interpolation(idx):
        return _values_beta_h[idx]

    @implementation('cpp', namespace={'_values_alpha_n': array(_values_alpha_n)},
                    code='''
    #include <math.h>
    double alpha_n_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_alpha_n[floor_ind];
        double v1 = _namespace_values_alpha_n[floor_ind+1];
        double vfrac = index-floor_ind;

        return (1 - vfrac) * v0 + vfrac * v1;
    }
    ''')
    @implementation('cython', code='''
    cdef double alpha_n_interpolation(double index):
        global _namespace_values_alpha_n
        return _namespace_values_alpha_n[int(index)]
    ''', namespace={'_values_alpha_n':_values_alpha_n}
    )
    @check_units(idx=1,result=Hz)
    def alpha_n_interpolation(idx):
        return _values_alpha_n[idx]

    @implementation('cpp', namespace={'_values_beta_n': array(_values_beta_n)},
                    code='''
    #include <math.h>
    double beta_n_interpolation(double index)
    {
        int floor_ind = int(floor(index));
		double v0 = _namespace_values_beta_n[floor_ind];
        double v1 = _namespace_values_beta_n[floor_ind+1];
        double vfrac = index-floor_ind;
        double v = (1 - vfrac) * v0 + vfrac * v1;

        return v;
    }
    ''')
    @implementation('cython', code='''
    cdef double beta_n_interpolation(double index):
        global _namespace_values_beta_n
        return _namespace_values_beta_n[int(index)]
    ''', namespace={'_values_beta_n':_values_beta_n}
    )
    @check_units(idx=1,result=Hz)
    def beta_n_interpolation(idx):
        return _values_beta_n[idx]

    P = NeuronGroup(N, model=equations, threshold='v>-20*mV', refractory=3*ms,
                    method=integrationmethod, dt=dt)

    P.v = -70*mV
    P.I_input = linspace(0,1,len(P))*namp

    # Record a few traces
    trace = StateMonitor(P, 'v', record=True, dt=1*ms)

    # define network:
    net = Network(P, trace)

    net.run(runtime, report='text')

    summary = ProfilingSummary(net)

    return (summary, trace)

### main

N = 100
dt = .01*ms
integrationmethod = 'exponential_euler'

summary_calc, trace_calc = run_net(eqs_calc, N, .01*mV, integrationmethod, dt=dt)
print summary_calc


times_table = []
err_table = []
times_inter = []
err_inter = []

resolution_list = [0.001,0.005,0.01,0.05,0.1,.5,1]*mV

for res in resolution_list:

    summary_table, trace_table = run_net(eqs_table, N, res, integrationmethod, dt=dt)
    summary_interpolation, trace_interpolation = run_net(eqs_interpolation, N, res, integrationmethod, dt=dt)

    print summary_table
    print summary_interpolation

    sum_tab = 0*second
    sum_int = 0*second
    for ind, (tab, inter) in enumerate(zip(summary_table.names,summary_interpolation.names)):
        if 'stateupdater' in tab or 'subexpression_update' in tab:
            sum_tab += summary_table.times[ind]
        if 'stateupdater' in inter or 'subexpression_update' in inter:
            sum_int += summary_interpolation.times[ind]
    times_table += [sum_tab]
    times_inter += [sum_int]

    err_table += [sum([sum(((trace_calc.v[i]-trace_table.v[i]))**2)/len(trace_calc.v[i]) for i in range(N)])/N]
    err_inter += [sum([sum(((trace_calc.v[i]-trace_interpolation.v[i]))**2)/len(trace_calc.v[i]) for i in range(N)])/N]

print times_inter, err_inter, times_table, err_table

sum_calc = 0*second
for ind, calc in enumerate(summary_calc.names):
    if 'stateupdater' in calc:
        sum_calc += summary_calc.times[ind]

calc_time = sum([summary_calc.times[i] for i in range(len(summary_calc.names)) if 'stateupdater' in summary_calc.names[i]])

f, ax = subplots(1)

ax.plot(times_inter, err_inter, '.', label='interpolation')
ax.plot(times_table, err_table, '.', label='table')

alltimes = array(times_table + times_inter + [calc_time])

xmin = min(alltimes) - .2
xma = max(alltimes) + .2

xlim([xmin, xma])

ax.annotate("Calculated", xy=((calc_time-xmin)/(xma - xmin),0), xycoords='axes fraction', xytext=((calc_time-xmin)/(xma - xmin),.1), textcoords='axes fraction', arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))



#plot([calc_time], [1e-10], '.', label='calculate')
for ind,(x,y) in enumerate(zip(times_inter,err_inter)):
    text(x, y, str(ind))
for ind,(x,y) in enumerate(zip(times_table,err_table)):
    text(x, y, str(ind))

xlabel('time spent in stateupater (s)')
ylabel(r'MSE (volt$^2$)')
title(integrationmethod)

yscale('log')
legend()
show()

