'''
Set state variable values with a string (using code generation)
'''

from brian2 import *
import numpy as np

language = CPPLanguage()

G = NeuronGroup(100, 'v:volt', language=language)
G.v = '(sin(2*pi*i/_num_neurons) - 70 + 0.25*randn()) * mV'
S = Synapses(G, G, 'w:volt', pre='v+=w', language=language)
S.connect('True')

space_constant = 200.0
S.w['i > j'] = 'exp(-(i - j)**2/space_constant) * mV'

# Generate a matrix for display
w_matrix = np.zeros((len(G), len(G)))
w_matrix[S.i[:], S.j[:]] = S.w[:]

import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.plot(G.v[:] / mV)
plt.subplot(1, 2, 2)
plt.imshow(w_matrix)
plt.show()