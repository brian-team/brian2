from brian2 import *

#set_device('cpp_standalone', debug=True, build_on_run=False, directory='cpp')
prefs.codegen.target = 'cython'

def runner(method, runtime, method_options=None, dt=.1*ms, print_code=False):
    device.reinit()
    device.activate()
    print('Running with method = %s'%method)
    eqs = '''
    dv/dt = (stimulus(t) + -v)/(.1*ms) : volt
    '''
    neuron = NeuronGroup(1, model=eqs, reset='v=0*mV', threshold='v>10*volt',
                         method=method, method_options=method_options, dt=dt)
    net = Network(neuron)
    net.run(0*ms)
    if method=='gsl':
        rec_vars = ('v', '_failed_steps', '_step_count')
    else:
        rec_vars = ('v',)
    mon = StateMonitor(neuron, rec_vars, record=True)
    net.add(mon)
    net.run(runtime)
    if print_code:
        print neuron.state_updater.codeobj.code
    return (mon.t/ms, [getattr(mon, var)[0]/volt for var in rec_vars])

seed(0)

runtime = 4*second
runtime = 40*ms

stimulus = TimedArray(rand(int(runtime/(10*ms)))*3*volt, dt=10*ms)

dt = 1*ms
track_timestep = True

error1 = 1e-2*volt
error2 = 1e-5*volt
error3 = 1e-6*volt

times, M_gsl1 = runner('gsl', runtime, dt=dt,
               method_options={'absolute_error': 1., 'absolute_error_per_variable' : {'v':error1},
                               'use_last_timestep' : track_timestep, 'save_failed_steps' : True, 'save_step_count' : True})
times, M_gsl2 = runner('gsl', runtime, dt=dt,
               method_options={'absolute_error': 1., 'absolute_error_per_variable' : {'v':error2},
                               'use_last_timestep' : track_timestep, 'save_failed_steps' : True, 'save_step_count' : True})
times, M_gsl3 = runner('gsl', runtime, dt=dt,
               method_options={'absolute_error': 1., 'absolute_error_per_variable' : {'v':error3},
                               'use_last_timestep' : track_timestep, 'save_failed_steps' : True, 'save_step_count' : True})
times, M_lin = runner('linear', runtime, dt=dt)

print max((M_gsl1[0]-M_lin[0])/(error1/volt))
print mean((M_gsl2[0]-M_lin[0])/(error2/volt))
print max((M_gsl3[0]-M_lin[0])/(error1/volt))

subplot(311)
plot(times, M_lin[0])
plot(times, M_gsl1[0])
plot(times, M_gsl2[0])
xticks([])
ylabel('Potential (volt)')
subplot(312)
plot(times, (M_gsl3[0]-M_lin[0])/error3, label=r'$\epsilon_v$=%.0e'%(error3/volt))
plot(times, (M_gsl2[0]-M_lin[0])/error2, label=r'$\epsilon_v$=%.0e'%(error2/volt))
plot(times, (M_gsl1[0]-M_lin[0])/error1, label=r'$\epsilon_v$=%.0e'%(error1/volt))
#plot(M_lin.t/ms, M_lin.v[0]/volt-M_gsl9.v[0]/volt)
#plot(M_lin.t/ms, M_lin.v[0]-M_exp.v[0])
#yscale('log')
ylabel(r'($v_{GSL} - v_{linear})/\epsilon_{v}$')
legend()
xticks([])
subplot(313)
ax = gca()
ax.plot(times, M_gsl3[2])
ax.plot(times, M_gsl2[2])
ax.plot(times, M_gsl1[2])
ylabel('Steps per dt')
yscale('log')
yticks([1, 2, 3, 10, 20, 30])
ax.set_yticklabels(["1", "2", "3", "10", "20", "30"])
xlabel('time (ms)')
suptitle('Illustation effect varying error bounds')
savefig('Tryout_error_bounds.pdf')
show()
