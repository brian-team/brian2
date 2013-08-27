'''
A simple example showing how string expressions can be used to implement
spatial (deterministic or stochastic) connection patterns.
'''
import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

# brian_prefs.codegen.target = 'weave'

rows, cols = 20, 20
G = NeuronGroup(rows * cols, '''x : meter
                                y : meter''')
# initialize the grid positions
grid_dist = 25*umeter
G.x = '(i / rows) * grid_dist - rows/2.0 * grid_dist'
G.y = '(i % rows) * grid_dist - cols/2.0 * grid_dist'


# Deterministic connections
distance = 120*umeter
S_deterministic = Synapses(G, G)
S_deterministic.connect('sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < distance')

# Random connections (no self-connections)
S_stochastic = Synapses(G, G)
S_stochastic.connect('i != j',
                     p='1.5 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2)/(2*(60*umeter)**2))')

# Show the connections for some neurons in different colors
for color in ['g', 'b', 'm']:
    plt.subplot(1, 2, 1)
    neuron_idx = np.random.randint(0, rows*cols)
    plt.plot(G.x[neuron_idx] / umeter, G.y[neuron_idx] / umeter, 'o', mec=color,
             mfc='none')
    plt.plot(G.x[S_deterministic.j[neuron_idx, :]] / umeter,
             G.y[S_deterministic.j[neuron_idx, :]] / umeter, color + '.')
    plt.subplot(1, 2, 2)
    plt.plot(G.x[neuron_idx] / umeter, G.y[neuron_idx] / umeter, 'o', mec=color,
             mfc='none')
    plt.plot(G.x[S_stochastic.j[neuron_idx, :]] / umeter,
             G.y[S_stochastic.j[neuron_idx, :]] / umeter, color + '.')

for idx, title in enumerate(['determininstic connections',
                             'random connections']):
    plt.subplot(1, 2, idx + 1)
    plt.xlim((-rows/2.0 * grid_dist) / umeter, (rows/2.0 * grid_dist) / umeter)
    plt.ylim((-cols/2.0 * grid_dist) / umeter, (cols/2.0 * grid_dist) / umeter)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y', rotation='horizontal')
    plt.axis('equal')
plt.show()
