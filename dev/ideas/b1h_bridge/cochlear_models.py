# WORKS!
#!/usr/bin/env python
'''
Example of the use of the cochlear models (:class:`~brian.hears.DRNL`,
:class:`~brian.hears.DCGC` and :class:`~brian.hears.TanCarney`) available in the library.
'''
from brian2 import *
from brian2.hears import *

simulation_duration = 50*ms
set_default_samplerate(50*kHz)
sound = whitenoise(simulation_duration)
sound = sound.atlevel(50*dB) # level in rms dB SPL
cf = erbspace(100*Hz, 1000*Hz, 50) # centre frequencies
interval = 16  #update interval of the time varying filters
## DNRL
#param_drnl = {}
#param_drnl['lp_nl_cutoff_m'] = 1.1
#drnl_filter=DRNL(sound, cf, type='human', param=param_drnl)
#out = drnl_filter.process()

## DCGC
#param_dcgc = {}
#param_dcgc['c1'] = -2.96

#dcgc_filter = DCGC(sound, cf, interval, param=param_dcgc)
#out = dcgc_filter.process()

## Tan and Carney 2003
tan_filter = TanCarney(sound, cf, interval)
out = tan_filter.process()

figure()
imshow(flipud(out.T), aspect='auto')
show()
