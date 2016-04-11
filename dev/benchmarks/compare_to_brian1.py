import time
from processify import processify

def halfway_timer():
    global halftime
    halftime = time.time()

@processify
def brian1_CUBA(N, duration=1,
                do_refractory=False, exact_method=False, do_monitor=False, do_synapses=False,
                **ignored_opts):
    global halftime
    import time
    results = dict()
    start_time = time.time()
    from brian import *
    finished_importing_brian_time = time.time()
    results['import_brian'] = finished_importing_brian_time-start_time

    duration *= second
    Ne = int(0.8*N)
    Ni = N-Ne
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    kwds = {}
    if do_refractory:
        kwds['refractory'] = 5*ms
    if not exact_method:
        kwds['method'] = 'Euler'
    P = NeuronGroup(N, model=eqs, threshold=Vt, reset=Vr, **kwds)
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    p = min(80./N, 1)
    if do_synapses:
        Ce = Connection(Pe, P, 'ge', weight=we, sparseness=p)
        Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=p)
    P.v = Vr + rand(len(P)) * (Vt - Vr)

    if do_monitor:
        M = SpikeMonitor(P)

    netop_halfway_timer = network_operation(clock=EventClock(dt=duration*0.51))(halfway_timer)

    objects_created_time = time.time()
    results['object_creation'] = objects_created_time-finished_importing_brian_time

    run(1 * msecond)

    initial_run_time = time.time()
    results['initial_run'] = initial_run_time-objects_created_time


    run(duration)

    main_run_time = time.time()
    results['main_run'] = main_run_time-initial_run_time
    results['second_half_main_run'] = main_run_time-halftime
    results['total'] = main_run_time-start_time

    return results

@processify
def brian2_CUBA(N, duration=1,
                do_refractory=False, exact_method=False, do_monitor=False, do_synapses=False,
                codegen_target='numpy',
                **ignored_opts):
    global halftime
    import time
    results = dict()
    start_time = time.time()
    from brian2 import *
    prefs.codegen.target = codegen_target
    finished_importing_brian_time = time.time()
    results['import_brian'] = finished_importing_brian_time-start_time

    duration *= second
    Ne = int(0.8*N)
    Ni = N-Ne
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''
    if not do_refractory:
        eqs = eqs.replace('(unless refractory)', '')

    kwds = {}
    if do_refractory:
        kwds['refractory'] = 5*ms
    if not exact_method:
        kwds['method'] = 'euler'
    P = NeuronGroup(N, model=eqs, threshold='v>Vt', reset='v=Vr', **kwds)
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    p = min(80./N, 1.0)
    if do_synapses:
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        Ce.connect('i<Ne', p=p)
        Ci.connect('i>=Ne', p=p)
    P.v = Vr + rand(len(P)) * (Vt - Vr)

    if do_monitor:
        M = SpikeMonitor(P)

    netop_halfway_timer = NetworkOperation(halfway_timer, dt=duration*0.51)

    objects_created_time = time.time()
    results['object_creation'] = objects_created_time-finished_importing_brian_time

    run(1 * msecond)

    initial_run_time = time.time()
    results['initial_run'] = initial_run_time-objects_created_time

    run(duration)

    main_run_time = time.time()
    results['main_run'] = main_run_time-initial_run_time
    results['second_half_main_run'] = main_run_time-halftime
    results['total'] = main_run_time-start_time

    return results

def brian2_CUBA_weave(*args, **opts):
    opts['codegen_target'] = 'weave'
    return brian2_CUBA(*args, **opts)

def brian2_CUBA_cython(*args, **opts):
    opts['codegen_target'] = 'cython'
    return brian2_CUBA(*args, **opts)

if __name__=='__main__':
    import functools
    from pylab import *
    numfigs = 0

    funcs = [
        brian1_CUBA,
        brian2_CUBA,
        brian2_CUBA_weave,
        brian2_CUBA_cython,
        ]
    options = dict(duration=1,
                   do_monitor=False,
                   do_refractory=False,
                   do_synapses=False,
                   exact_method=False,
                   )
    N = [1, 10, 100, 1000,
         10000,
         #100000,
         ]

    for name in options.keys()+['']:
    #for name in ['']:
        if name:
            if not isinstance(options[name], bool):
                continue
            options[name] = True
        figure(figsize=(16, 8))
        for func in funcs:
            pfunc = functools.partial(func, **options)
            all_results = map(pfunc, N)
            times = collections.defaultdict(list)
            for res in all_results:
                for k, v in res.items():
                    times[k].append(v)
            for i, k in enumerate(sorted(times.keys())):
                subplot(2, 3, i+1)
                if i>numfigs:
                    numfigs = i
                title(k)
                v = times[k]
                loglog(N[1:], v[1:], label=func.__name__)
        for i in range(numfigs+1):
            subplot(2, 3, i+1)
            legend(loc='best')
        suptitle(', '.join('%s=%s' % (k, options[k]) for k in sorted(options.keys())))
        tight_layout()
        subplots_adjust(top=0.9)
        if name:
            savefig('compare_to_brian1.%s.png'%name)
            options[name] = False
        else:
            savefig('compare_to_brian1.png')
    show()
