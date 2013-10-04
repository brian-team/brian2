from brian2 import *
import numpy as np

G = NeuronGroup(10, 'x:1')
G.x = np.arange(10)

SG = G[4:9]

S = Synapses(G, SG, 'w:1', connect='x_pre>2')
S.w = 'i + 5*j'
print S.i
print S.j
print S.x_post['x_post > 5']
