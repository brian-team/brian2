'''
Run standard examples with C++ standalone

Prints some comparison output and plots some side-by-side comparison figures (not the original plots though)
'''
import os, re, shutil, multiprocessing, numpy, time
import traceback

from pylab import *

basepath = '../../../examples/'

# Uncomment the example you want to run, double commented ones don't work yet

#example = 'IF_curve_Hodgkin_Huxley.py'
#example = 'IF_curve_LIF.py'
#example = 'non_reliability.py'
#example = 'phase_locking.py'
##example = 'stochastic_odes.py' # too complicated
#example = 'synapses.py'
#example = 'synapses_gapjunctions.py'
example = 'synapses_jeffress.py'
#example = 'synapses_licklider.py'
##example = 'synapses_nonlinear.py' # S.w = [1., 10.] should work but doesn't
##example = 'synapses_spatial_connections.py' # doesn't actually run anything
##example = 'synapses_state_variables.py' # doesn't actually run anything
#example = 'synapses_STDP.py'

neuron_group_variables = ['v', 'V', 'x']
synapse_variables = ['w']


def dorunit((code, standalone)):
    try:  # wrap everything in try except to get a nicer stacktrace
        ns = {}
        start = time.time()
        exec code in ns
        tottime = time.time()-start
        rv = {}
        for k, v in ns.iteritems():
            if isinstance(v, ns['SpikeMonitor']):
                if standalone:
                    S = loadtxt('output/results/%s_codeobject.txt' % v.name, delimiter=',',
                                dtype=[('i', int), ('t', float)])
                    i = S['i']
                    t = S['t']
                else:
                    i, t = v.it
                rv[k] = ('SpikeMonitor', v.name, (i, t))
            if isinstance(v, ns['NeuronGroup']):
                found = False
                for var in neuron_group_variables:
                    if var in v.variables:
                        found = True
                        break
                if found:
                    if standalone:
                        val = loadtxt('output/results/%s.txt' % v.variables[var].arrayname, delimiter=',', dtype=float)
                    else:
                        val = asarray(getattr(v, var))
                    rv[k] = ('NeuronGroup', v.name+'.'+var, val)
            if isinstance(v, ns['Synapses']):
                found = False
                for var in synapse_variables:
                    if var in v.variables:
                        found = True
                        break
                if found:
                    if standalone:
                        val = loadtxt('output/results/_dynamic%s.txt' % v.variables[var].arrayname, delimiter=',', dtype=float)
                    else:
                        val = asarray(getattr(v, var))
                    rv[k] = ('Synapses', v.name+'.'+var, val)
        return tottime, rv
    except Exception:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    
def runit(code, standalone):
    pool = multiprocessing.Pool(1)
    return pool.map(dorunit, [(code, standalone)])[0]

if __name__=='__main__':    
    lines = open(os.path.join(basepath, example), 'r').read().split('\n')
    runtime_lines = []
    standalone_lines = []
    for line in lines:
        skip = False
        if line.startswith('BrianLogger'):
            skip = True
        if not skip:
            standalone_lines.append(line)
            runtime_lines.append(line)
        if line.strip()=='from brian2 import *':
            standalone_lines.append('from brian2.devices.cpp_standalone import *')
            standalone_lines.append('set_device("cpp_standalone")')
        if line.startswith('run'):
            standalone_lines.append("build(project_dir='output', compile_project=True, run_project=True)")        
            break
        
    standalone_code = '\n'.join(standalone_lines)
    runtime_code = '\n'.join(runtime_lines)
    
    if os.path.exists('output'):
        shutil.rmtree('output')
    
    t_standalone, rv_standalone = runit(standalone_code, True)
    t_runtime, rv_runtime = runit(runtime_code, False)
    
    print '------', example, '------------------------'
    
    n = len(rv_runtime)
    
    for i_plot, (k, v) in enumerate(rv_runtime.items()):
        subplot(2, n, i_plot+1)
        plot_type, name, plot_args = v
        title(name+' runtime')
        if plot_type=='SpikeMonitor':
            i, t = plot_args
            print name+' num spikes:'
            print '  Runtime    ', len(i)
            plot(t, i, '.k')
            i, t = rv_standalone[k][2]
            subplot(2, n, n+i_plot+1)
            plot(t, i, '.k')
            print '  Standalone ', len(i)
        elif plot_type=='NeuronGroup':
            val = plot_args
            plot(val, '.')
            print name+' (mean/std):'
            print '  Runtime     %f / %f' % (mean(val), std(val))
            val = rv_standalone[k][2]
            subplot(2, n, n+i_plot+1)
            plot(val, '.')
            print '  Standalone  %f / %f' % (mean(val), std(val))
        elif plot_type=='Synapses':
            val = plot_args
            hist(val, 20)
            print name+' (mean/std):'
            print '  Runtime     %f / %f' % (mean(val), std(val))
            val = rv_standalone[k][2]
            subplot(2, n, n+i_plot+1)
            hist(val, 20)
            print '  Standalone  %f / %f' % (mean(val), std(val))
        title(name+' standalone')
    
    print 'Total time (build+run):'
    print '  Runtime              %.2f' % t_runtime
    print '  Standalone           %.2f' % t_standalone
    print '  Runtime/standalone   %.2fx' % (t_runtime/t_standalone)
    
    show()
    