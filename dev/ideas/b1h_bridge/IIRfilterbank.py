# WORKS!
#!/usr/bin/env python
'''
Example of the use of the class :class:`~brian.hears.IIRFilterbank` available in
the library.  In this example, a white noise is filtered by a bank of chebyshev
bandpass filters and lowpass filters which are different for every channels.
The centre frequencies of  the filters are linearly taken between 100kHz and
1000kHz and its bandwidth or cutoff frequency increases linearly with frequency.
'''
from brian2 import *
from brian2.hears import *

sound = whitenoise(100*ms).ramp()
sound.level = 50*dB

### example of a bank of bandpass filter ################
nchannels = 50
center_frequencies = linspace(200*Hz, 1000*Hz, nchannels)  #center frequencies 
bw = linspace(50*Hz, 300*Hz, nchannels)  #bandwidth of the filters
# The maximum loss in the passband in dB. Can be a scalar or an array of length
# nchannels
gpass = 1.*dB
# The minimum attenuation in the stopband in dB. Can be a scalar or an array
# of length nchannels
gstop = 10.*dB
#arrays of shape (2 x nchannels) defining the passband frequencies (Hz)
passband = vstack((center_frequencies-bw/2, center_frequencies+bw/2))
#arrays of shape (2 x nchannels) defining the stopband frequencies (Hz)
stopband = vstack((center_frequencies-1.1*bw, center_frequencies+1.1*bw))

filterbank = IIRFilterbank(sound, nchannels, passband, stopband, gpass, gstop,
                           'bandstop', 'cheby1')
filterbank_mon = filterbank.process()

figure()
subplot(211)
imshow(flipud(filterbank_mon.T), aspect='auto')    

#### example of a bank of lowpass filter ################
nchannels = 50
cutoff_frequencies = linspace(100*Hz, 1000*Hz, nchannels)
#bandwidth of the transition region between the en of the pass band and the
#begin of the stop band 
width_transition = linspace(50*Hz, 300*Hz, nchannels)
# The maximum loss in the passband in dB. Can be a scalar or an array of length
# nchannels
gpass = 1*dB
# The minimum attenuation in the stopband in dB. Can be a scalar or an array of
# length nchannels
gstop = 10*dB
passband = cutoff_frequencies-width_transition/2
stopband = cutoff_frequencies+width_transition/2

filterbank = IIRFilterbank(sound, nchannels, passband, stopband, gpass, gstop,
                           'low','cheby1')
filterbank_mon=filterbank.process()

subplot(212)
imshow(flipud(filterbank_mon.T), aspect='auto')    
show()
