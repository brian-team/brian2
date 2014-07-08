# WORKS!

#!/usr/bin/env python
'''
Example of the use of the class :class:`~brian.hears.Butterworth` available in
the library. In this example, a white noise is filtered by a bank of butterworth
bandpass filters and lowpass filters which are different for every channels. The
centre or cutoff frequency of the filters are linearly taken between 100kHz and
1000kHz and its bandwidth frequency increases linearly with frequency.
'''
from brian2 import *
from brian2.hears import *

level = 50*dB  # level of the input sound in rms dB SPL
sound = whitenoise(100*ms).ramp()
sound = sound.atlevel(level)
order = 2 #order of the filters

#### example of a bank of bandpass filter ################
nchannels = 50
center_frequencies = linspace(100*Hz, 1000*Hz, nchannels) 
bw = linspace(50*Hz, 300*Hz, nchannels) # bandwidth of the filters
#arrays of shape (2 x nchannels) defining the passband frequencies (Hz)
fc = vstack((center_frequencies-bw/2, center_frequencies+bw/2))

filterbank = Butterworth(sound, nchannels, order, fc, 'bandpass')

filterbank_mon = filterbank.process()

figure()
subplot(211)
imshow(flipud(filterbank_mon.T), aspect='auto')    

### example of a bank of lowpass filter ################
nchannels = 50
cutoff_frequencies = linspace(200*Hz, 1000*Hz, nchannels) 

filterbank = Butterworth(sound, nchannels, order, cutoff_frequencies, 'low')

filterbank_mon = filterbank.process()

subplot(212)
imshow(flipud(filterbank_mon.T), aspect='auto')    
show()
