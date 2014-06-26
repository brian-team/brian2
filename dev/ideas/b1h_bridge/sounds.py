# DOESN'T WORK!
# The reason is that somehow, classes derived from numpy.ndarray do not use the correct method resolution order
# for __getitem__ and __getslice__, and the fact that ms is derived from ndarray in Brian 2 causes it to fuck
# up - the same behaviour can be reproduced very simply, see the example below.

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

import numpy
class X(numpy.ndarray):
    def __getitem__(self, item):
        print item
        return numpy.ndarray.__getitem__(self, item)
    def __getslice__(self, start, stop):
        print start, stop
        return numpy.ndarray.__getslice__(self, start, stop)
x = X([1, 2])
x[:array([1,2])]
exit()

print Sound
s = Sound(array([1, 2]))
print s.__class__
print s.__setitem__
print s.__setslice__
s[:20*ms]
exit()

# The first 20ms of the sound
#startsound = sound.__getslice__(None, 20*ms)
startsound = sound[:20*ms]

subplot(121)
plot(startsound.times, startsound)
subplot(122)
sound.spectrogram()
show()
