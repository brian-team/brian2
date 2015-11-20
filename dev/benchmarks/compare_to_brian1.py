import time

def halfway_timer():
    global halftime
    halftime = time.time()


def brian1_CUBA(N, duration=10):
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

    eqs = Equations('''
    dv/dt  = (ge+gi-(v-El))/taum : volt
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    ''')

    P = NeuronGroup(N, model=eqs, threshold=Vt, reset=Vr,
                    #refractory=5*ms,
                    method='Euler')
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    Pe = P.subgroup(Ne)
    Pi = P.subgroup(Ni)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    p = min(80./N, 1)
    # Ce = Connection(Pe, P, 'ge', weight=we, sparseness=p)
    # Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=p)
    P.v = Vr + rand(len(P)) * (Vt - Vr)

    # M = SpikeMonitor(P)

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

def brian2_CUBA(N, duration=10):
    global halftime
    import time
    results = dict()
    start_time = time.time()
    from brian2 import *
    prefs.codegen.target = 'numpy'
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

    eqs = Equations('''
    dv/dt  = (ge+gi-(v-El))/taum : volt #(unless refractory)
    dge/dt = -ge/taue : volt #(unless refractory)
    dgi/dt = -gi/taui : volt #(unless refractory)
    ''')

    P = NeuronGroup(N, model=eqs, threshold='v>Vt', reset='v=Vr',
                    #refractory=5*ms,
                    method='Euler')
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    p = min(80./N, 1.0)
    # Ce = Synapses(P, P, pre='ge += we')
    # Ci = Synapses(P, P, pre='gi += wi')
    # Ce.connect('i<Ne', p=p)
    # Ci.connect('i>=Ne', p=p)
    P.v = Vr + rand(len(P)) * (Vt - Vr)

    # M = SpikeMonitor(P)

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

if __name__=='__main__':
    import multiprocessing, collections
    pool = multiprocessing.Pool(1, maxtasksperchild=1)
    funcs = [brian1_CUBA, brian2_CUBA]
    from pylab import *
    numfigs = 0
    for func in funcs:
        N = [10, 100, 1000, 10000]
        all_results = pool.map(func, N)
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
            loglog(N, v, label=func.__name__)
    for i in range(numfigs+1):
        subplot(2, 3, i+1)
        legend(loc='best')
    show()
