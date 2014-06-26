# WORKS!
#!/usr/bin/env python
'''
Example of the use of the class :class:`~brian.hears.ApproximateGammatone`
available in the library. It implements a filterbank of approximate gammatone
filters as  described in Hohmann, V., 2002, "Frequency analysis and synthesis
using a Gammatone filterbank", Acta Acustica United with Acustica. 
In this example, a white noise is filtered by a gammatone filterbank and the
resulting cochleogram is plotted.
'''
from brian2 import *
from brian2.hears import *

level=50*dB # level of the input sound in rms dB SPL
sound = whitenoise(100*ms).ramp() # generation of a white noise
sound = sound.atlevel(level) # set the sound to a certain dB level

nbr_center_frequencies = 50  # number of frequency channels in the filterbank
# center frequencies with a spacing following an ERB scale
center_frequencies = erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)
# bandwidth of the filters (different in each channel) 
bw = 10**(0.037+0.785*log10(center_frequencies))

gammatone = ApproximateGammatone(sound, center_frequencies, bw, order=3) 

gt_mon = gammatone.process()

figure()
imshow(flipud(gt_mon.T), aspect='auto')    
show()
