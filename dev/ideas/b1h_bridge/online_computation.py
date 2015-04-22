# WORKS!
#!/usr/bin/env python
'''
Example of online computation using :meth:`~brian.hears.Filterbank.process`.
Plots the RMS value of each channel output by a gammatone filterbank.
'''
from brian2 import *
from brian2.hears import *

sound1 = tone(1*kHz, .1*second)
sound2 = whitenoise(.1*second)

sound = sound1+sound2
sound = sound.ramp()

sound.level = 60*dB

cf = erbspace(20*Hz, 20*kHz, 3000)
fb = Gammatone(sound, cf)

def sum_of_squares(input, running):
    return running+sum(input**2, axis=0)

rms = sqrt(fb.process(sum_of_squares)/sound.nsamples)

sound_rms = sqrt(mean(sound**2))

axhline(sound_rms, ls='--')
plot(cf, rms)
xlabel('Frequency (Hz)')
ylabel('RMS')
show()
