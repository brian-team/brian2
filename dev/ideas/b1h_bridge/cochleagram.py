# WORKS!
#!/usr/bin/env python
'''
Example of basic filtering of a sound with Brian hears.
This example implements a cochleagram based on a gammatone filterbank
followed by halfwave rectification, cube root compression and 10 Hz
low pass filtering.
'''
from brian2 import *
from brian2.hears import *

sound1 = tone(1*kHz, .1*second)
sound2 = whitenoise(.1*second)

sound = sound1+sound2
sound = sound.ramp()

cf = erbspace(20*Hz, 20*kHz, 3000)
gammatone = Gammatone(sound, cf)
cochlea = FunctionFilterbank(gammatone, lambda x: clip(x, 0, Inf)**(1.0/3.0))
lowpass = LowPass(cochlea, 10*Hz)
output = lowpass.process()

imshow(output.T, origin='lower left', aspect='auto', vmin=0)
show()
