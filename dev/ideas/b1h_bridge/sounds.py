# HALF WORKS
# You need to replace sound[:20*ms] with sound.slice(None, 20*ms)

#!/usr/bin/env python
'''
Example of basic use and manipulation of sounds with Brian hears.
'''
from brian2 import *
from brian2.hears import *

sound1 = tone(1*kHz, 1*second)
sound2 = whitenoise(1*second)

sound = sound1+sound2
#sound = sound.ramp()

# Comment this line out if you don't have pygame installed
#sound.play()

# The first 20ms of the sound
#startsound = sound[:20*ms] # doesn't work
startsound = sound.slice(None, 20*ms) # new Sound method designed to make this work

subplot(121)
plot(startsound.times, startsound)
subplot(122)
sound.spectrogram()
show()
