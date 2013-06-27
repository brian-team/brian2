'''
Set state variable values with a string (using code generation)
'''

from brian2 import *
import numpy as np
G = NeuronGroup(100, 'v:volt')
S = Synapses(G, G, 'w:volt', pre='v+=w')
ii, jj = np.meshgrid(np.arange(len(G)), np.arange(len(G)))
S.connect(ii.flatten(), jj.flatten())

space_constant = 200.0
S.w['i > j'] = 'exp(-(i - j)**2/space_constant) * mV'

# Generate a matrix for display
w_matrix = np.zeros((len(G), len(G)))
w_matrix[S.indices.i[:], S.indices.j[:]] = S.w[:]

import matplotlib.pyplot as plt
plt.imshow(w_matrix)
plt.show()