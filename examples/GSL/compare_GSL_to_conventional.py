'''
August 31st, 2017
Example using GSL ODE solver and comparing it to the Brian solver.

For this example the error has to be very small (<=1e-5) for GSL to be faster
than conventional Brian, when both are using the rk2 method. Even if Brian has
to do 64 more steps/ms, it does so faster than GSL running 1 'Brian'-step/ms in
which it does 1.4 'GSL'-steps/ms on average.
'''
from brian2 import *
import time

# Run settings
start_dt = 1*ms
meth='rk2'
error = 1.e-6 # requested accuracy
step_count = False # should GSL keep track of the amount of steps taken to reach requested accuracy?

def runner(method, dt, error=None):
    seed(0)
    group = NeuronGroup(100, '''dv/dt = (-v + s)/tau : 1
                              s : 1
                              tau : second''',
                      method=method,
                      method_options={'save_step_count':step_count,
                                      'absolute_error':error},
                      dt=dt)
    group.run_regularly('''s = rand()
                           tau = 0.01*ms + rand()*10*ms''', dt=10*ms)

    if 'gsl' in method and step_count:
        rec_vars = ('v', '_step_count')
    else:
        rec_vars = ('v')
    net = Network(group)
    net.run(0*ms)
    mon = StateMonitor(group, rec_vars, record=True, dt=start_dt)
    net.add(mon)
    start = time.time()
    net.run(1*second)
    mon.add_attribute('run_time')
    mon.run_time = time.time() - start
    return mon

# Doing the
lin = runner('linear', start_dt)
gsl = runner('gsl_%s'%meth, start_dt, error)

print 'gsl time: ', gsl.run_time

# check gsl error
assert max(flatten(abs(lin.v - gsl.v))) < error, "Maximum error gsl integration too large: %f"%max(flatten(abs(lin.v - gsl.v)))
if step_count:
    print "average step count: ", mean(gsl._step_count)
print "average error: ", mean(abs(gsl.v - lin.v))

dt = start_dt
count = 0
while True:
    print count, dt
    brian = runner(meth, dt)
    if max(flatten(abs(brian.v - lin.v))) > error:
        dt *= .5
        count += 1
    else:
        break

print "average step count: ", (1*ms/dt)
print "average error: ", mean(abs(brian.v - lin.v))

print 'brian time: ', brian.run_time
