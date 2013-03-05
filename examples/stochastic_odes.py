import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

# setting a random seed makes all variants use exactly the same Wiener process
seed = 12347  

X0 = 1
mu = 0.5/second # drift
sigma = 0.1/second #diffusion 

runtime = 1*second


def simulate(method, dt):
    '''
    simulate geometrical Brownian with the given method
    ''' 
    np.random.seed(seed)
    G = NeuronGroup(1, 'dX/dt = mu*X + X*sigma*xi*second**.5: 1',
                    clock=Clock(dt=dt), method=method)
    G.X = X0
    mon = StateMonitor(G, 'X', record=True)
    net = Network(G, mon)
    net.run(runtime)
    return mon.t_, mon.X.flatten()


def exact_solution(t, dt):
    '''
    Return the exact solution for geometrical Brownian motion at the given
    time points
    '''
    # Remove units for simplicity
    my_mu = float(mu)
    my_sigma = float(sigma)
    dt = float(dt)
    t = np.asarray(t)
    
    np.random.seed(seed)
    # We are calculating the values at the *end* of a time step, as when using
    # a StateMonitor. Therefore also the Brownian motion starts not with zero
    # but with a random value.
    brownian = np.cumsum(np.sqrt(dt) * np.random.randn(len(t)))
    
    return (X0 * np.exp((my_mu - 0.5*my_sigma**2)*(t+dt) + my_sigma*brownian))


methods = ['euler', 'milstein']
dts = [1*ms, 0.5*ms, 0.2*ms, 0.1*ms, 0.05*ms, 0.025*ms, 0.01*ms, 0.005*ms]

rows = np.floor(np.sqrt(len(dts)))
cols = np.ceil(1.0 * len(dts) / rows)
errors = dict([(method, np.zeros(len(dts))) for method in methods])
for dt_idx, dt in enumerate(dts):
    print 'dt: ', dt
    trajectories = {}
    # Test the numerical methods
    for method in methods:
        t, trajectories[method] = simulate(method, dt)
    # Calculate the exact solution
    exact = exact_solution(t, dt)    
    
    for method in methods:
        # plot the trajectories
        plt.figure(1)
        plt.subplot(rows, cols, dt_idx+1)
        plt.plot(t, trajectories[method], label=method, alpha=0.75)
        
        # determine the mean absolute error
        errors[method][dt_idx] = np.mean(np.abs(trajectories[method] - exact))
        # plot the difference to the real trajectory
        plt.figure(2)
        plt.subplot(rows, cols, dt_idx+1)
        plt.plot(t, trajectories[method] - exact, label=method, alpha=0.75)
        
    plt.figure(1)
    plt.plot(t, exact, color='gray', lw=2, label='exact', alpha=0.75)
    plt.title('dt = %s' % str(dt))
    plt.xticks([])

plt.figure(1)
plt.legend(frameon=False, loc='best')
plt.figure(2)
plt.legend(frameon=False, loc='best')

plt.figure(3)
for method in methods:
    plt.plot(np.array(dts) / ms, errors[method], 'o', label=method)
plt.legend(frameon=False, loc='best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('dt (ms)')
plt.ylabel('mean absolute error')
plt.show()
