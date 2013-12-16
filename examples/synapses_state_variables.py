'''
Set state variable values with a string (using code generation)
'''

from brian2 import *

#brian_prefs.codegen.target = 'weave'

G = NeuronGroup(100, 'v:volt')
N = len(G)
G.v = '(sin(2*pi*i/N) - 70 + 0.25*randn()) * mV'
S = Synapses(G, G, 'w:volt', pre='v+=w')
S.connect('True')

space_constant = 200.0
S.w['i > j'] = 'exp(-(i - j)**2/space_constant) * mV'

# Generate a matrix for display
w_matrix = np.zeros((len(G), len(G)))
w_matrix[S.i[:], S.j[:]] = S.w[:]

subplot(1, 2, 1)
plot(G.v[:] / mV)
subplot(1, 2, 2)
imshow(w_matrix)
show()