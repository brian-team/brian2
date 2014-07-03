# WORKS!
#!/usr/bin/env python
'''
Example of the use of the class :class:`~brian.hears.LinearGammachirp` available
in the library. It implements a filterbank of FIR gammatone filters with linear
frequency sweeps as described in Wagner et al. 2009, "Auditory responses in the
barn owl's nucleus laminaris to clicks: impulse response and signal analysis of
neurophonic potential", J. Neurophysiol. In this example, a white noise is
filtered by a gammachirp filterbank and the resulting cochleogram is plotted.
The different impulse responses are also plotted.
'''
from brian2 import *
from brian2.hears import *

sound = whitenoise(100*ms).ramp()
sound.level = 50*dB

nbr_center_frequencies = 10  #number of frequency channels in the filterbank
#center frequencies with a spacing following an ERB scale
center_frequencies = erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)

c = 0.0 #glide slope
time_constant = linspace(3, 0.3, nbr_center_frequencies)*ms

gamma_chirp = LinearGammachirp(sound, center_frequencies, time_constant, c) 

gamma_chirp_mon = gamma_chirp.process()

figure()

imshow(gamma_chirp_mon.T, aspect='auto')    
figure()
plot(gamma_chirp.impulse_response.T)
show()
