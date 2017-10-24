import sys, os, numpy, time, pylab

filename = 'example_standalone.py'
datapath = 'data_example'
threads  = [0, 1, 2, 4 ,6]
results  = {}
results['duration'] = []

for t in threads:
    start = time.time()
    os.system('python %s 1 %d' %(filename, t))
    with open('%s_%d/speed.txt' %(datapath, t), 'r') as f:
        results['duration'] += [float(f.read())]
    results[t] = {}

for t in threads:
    results[t]  = {}
    path        = datapath + '_%d/' %t
    ids         = numpy.fromfile(path+'results/_dynamic_array_spikemonitor_i', dtype=numpy.int32)
    times       = numpy.fromfile(path+'results/_dynamic_array_spikemonitor_t', dtype=numpy.float64)
    w           = numpy.fromfile(path+'results/_dynamic_array_synapses_w', dtype=numpy.float64)
    times_w     = numpy.fromfile(path+'results/_dynamic_array_statemonitor_t', dtype=numpy.float64)
    w_over_time = numpy.fromfile(path+'results/_dynamic_array_statemonitor__recorded_w', dtype=numpy.float64)
    v_over_time = numpy.fromfile(path+'results/_dynamic_array_statemonitor_1__recorded_v', dtype=numpy.float64)
    times_v     = numpy.fromfile(path+'results/_dynamic_array_statemonitor_1_t', dtype=numpy.float64)
    results[t]['spikes']  = (times, ids)
    results[t]['w']       = w
    results[t]['trace_w'] = w_over_time.reshape(len(times_w), len(w_over_time)/len(times_w))
    results[t]['trace_v'] = v_over_time.reshape(len(times_v), len(v_over_time)/len(times_v))

results['colors'] = ['b', 'g', 'r', 'c', 'k']

pylab.figure()
pylab.subplot(321)
pylab.title('Raster plots')
pylab.xlabel('Time [s]')
pylab.ylabel('# cell')
for t in threads:
    pylab.plot(results[t]['spikes'][0], results[t]['spikes'][1], '.')
pylab.legend(map(str, threads))
#pylab.xlim(0.5, 0.6)

pylab.subplot(322)
pylab.title('Final Distribution')
pylab.xlabel('Weight [ns]')
pylab.ylabel('Number of synapses')
for t in threads:
    x, y = numpy.histogram(results[t]['w'], 100)
    pylab.plot(y[1:], x)
pylab.legend(map(str, threads))

pylab.subplot(323)
pylab.title('Weight Evolution')
pylab.xlabel('Time [s]')
pylab.ylabel('Weight [ns]')
for t in threads:
    pylab.plot(results[t]['trace_w'].T[0])
pylab.legend(map(str, threads))

pylab.subplot(324)
pylab.title('Weight Evolution')
pylab.xlabel('Time [s]')
pylab.ylabel('Weight [ns]')
for count, t in enumerate(threads):
    for i in xrange(3):
        pylab.plot(results[t]['trace_w'].T[i], c=results['colors'][count])
#pylab.legend(map(str, threads))

pylab.subplot(325)
pylab.title('Voltage Evolution')
pylab.xlabel('Time [s]')
pylab.ylabel('Voltage [mv]')
for count, t in enumerate(threads):
    for i in xrange(3):
        pylab.plot(results[t]['trace_v'].T[i], c=results['colors'][count])
#pylab.legend(map(str, threads))

pylab.subplot(326)
pylab.title('Speed')
pylab.plot(threads, results['duration'])
pylab.xlabel('# threads')
pylab.ylabel('Time [s]')

pylab.tight_layout()


pylab.savefig('net1_openmp.png')

pylab.show()
