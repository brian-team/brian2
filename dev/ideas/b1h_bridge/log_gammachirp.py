# WORKS
#!/usr/bin/env python
'''
Example of the use of the class :class:`~brian.hears.LogGammachirp` available in
the library. It implements a filterbank of IIR gammachirp filters as 
Unoki et al. 2001, "Improvement of an IIR asymmetric compensation gammachirp
filter". In this example, a white noise is filtered by a linear gammachirp
filterbank and the resulting cochleogram is plotted. The different impulse
responses are also plotted.
'''
from brian2 import *
from brian2.hears import *

sound = whitenoise(100*ms).ramp()
sound.level = 50*dB

nbr_center_frequencies = 50  #number of frequency channels in the filterbank

c1 = -2.96 #glide slope
b1 = 1.81  #factor determining the time constant of the filters

#center frequencies with a spacing following an ERB scale
cf = erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)

gamma_chirp = LogGammachirp(sound, cf, c=c1, b=b1) 

gamma_chirp_mon = gamma_chirp.process()

figure()
imshow(flipud(gamma_chirp_mon.T), aspect='auto')    
show()    
