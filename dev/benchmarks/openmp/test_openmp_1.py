import sys, os, numpy, time, pylab

filename = 'CUBA_standalone.py'
datapath = 'data_cuba'
threads  = [0, 1, 2, 4, 6]
results  = {}
results['duration'] = []

for t in threads:
    start = time.time()
    os.system('python %s 1 %d' %(filename, t))
    file = open('%s_%d/speed.txt' %(datapath, t), 'r')
    results['duration'] += [float(file.read())]


for t in threads:
    results[t] = {}
    path       = datapath + '_%d/' %t
    ids        = numpy.fromfile(path+'results/_dynamic_array_spikemonitor_i', dtype=numpy.int32)
    times      = numpy.fromfile(path+'results/_dynamic_array_spikemonitor_t', dtype=numpy.float64)
    vms        = numpy.fromfile(path+'results/_array_neurongroup_v', dtype=numpy.float64)
    w          = numpy.fromfile(path+'results/_dynamic_array_synapses_1_w', dtype=numpy.float64)
    results[t]['spikes'] = (times, ids)
    results[t]['vms']    = vms
    results[t]['w']      = w

pylab.figure()
pylab.subplot(221)
pylab.title('Raster plots')
pylab.xlabel('Time [s]')
pylab.ylabel('# cell')
for t in threads:
    pylab.plot(results[t]['spikes'][0], results[t]['spikes'][1], '.')
pylab.legend(map(str, threads))
pylab.xlim(0.5, 0.55)

pylab.subplot(222)
pylab.title('Mean Rates')
pylab.xlabel('Time [s]')
pylab.ylabel('Rate [Hz]')
for t in threads:
    x,  y = numpy.histogram(results[t]['spikes'][0], 100)
    pylab.plot(y[1:], x)
pylab.legend(map(str, threads))


pylab.subplot(223)
pylab.title('Network')
pylab.xlabel('# threads')
pylab.ylabel('Number of synapses')
r = []
for t in threads:
    r += [len(results[t]['w'])]
pylab.bar(threads, r)
pylab.ylim(min(r)-1000, max(r)+1000)

pylab.subplot(224)
pylab.title('Speed')
pylab.plot(threads, results['duration'])
pylab.xlabel('# threads')
pylab.ylabel('Time [s]')

pylab.tight_layout()
pylab.savefig('cuba_openmp.png')


pylab.show()

