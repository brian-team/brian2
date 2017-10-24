'''
Run standard examples with C++ standalone

Prints some comparison output and plots some side-by-side comparison figures (not the original plots though)
'''
import os, re, shutil, multiprocessing, numpy, time
import traceback

from brian2.devices import device
from pylab import *

only_run_standalone = False

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
                    i = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables['i'], access_data=False), dtype=int32)
                    t = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables['t'], access_data=False), dtype=float64)
                else:
                    i, t = v.it
                rv[k] = ('SpikeMonitor', v.name, (i, t))
            if isinstance(v, ns['StateMonitor']):
                if standalone:
                    t = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables['t'], access_data=False), dtype=float64)
                    rec = {}
                    for var in v.record_variables:
                        vals = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables['_recorded_'+var], access_data=False), dtype=float64)
                        vals.shape = (t.size, -1)
                        rec[var] = vals.T
                else:
                    t = v.variables['t'].get_value().copy()
                    rec = {}
                    for var in v.record_variables:
                        rec[var] = v.variables['_recorded_'+var].get_value().T.copy()
                rv[k] = ('StateMonitor', v.name, (t, rec))
            if isinstance(v, ns['NeuronGroup']):
                found = False
                for var in neuron_group_variables:
                    if var in v.variables:
                        found = True
                        break
                if found:
                    if standalone:
                        val = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables[var]), dtype=float64)
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
                        val = fromfile('output/results/%s' % ns['device'].get_array_name(v.variables[var], access_data=False), dtype=float64)
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
    with open(os.path.join(basepath, example), 'r') as f:
        lines = f.read().split('\n')
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
            standalone_lines.append('set_device("cpp_standalone")')
        if line.startswith('run'):
            standalone_lines.append("device.build(project_dir='output', compile_project=True, run_project=True, debug=False)")
            break
        
    standalone_code = '\n'.join(standalone_lines)
    runtime_code = '\n'.join(runtime_lines)
    
    if os.path.exists('output'):
        shutil.rmtree('output')
    
    t_standalone, rv_standalone = runit(standalone_code, True)
    if only_run_standalone:
        exit()
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
        elif plot_type=='StateMonitor':
            t, rec = plot_args
            for varname, vals in rec.items():
                plot(t, vals.T)
            subplot(2, n, n+i_plot+1)
            t, rec = rv_standalone[k][2]
            for varname, vals in rec.items():
                plot(t, vals.T)
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
    