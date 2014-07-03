# WORKS!
#!/usr/bin/env python
'''
Implementation example of the dual resonance nonlinear (DRNL) filter with
parameters fitted for human as described in Lopez-Paveda, E. and Meddis, R., A
human nonlinear cochlear filterbank, JASA 2001.

A class called :class:`~brian.hears.DRNL` implementing this model is available
in the library.

The entire pathway consists of the sum of a linear and a nonlinear pathway.

The linear path consists of a bank of bandpass filters (second order gammatone),
a low pass function, and a gain/attenuation factor, g, in a cascade.

The nonlinear path is  a cascade consisting of a bank of gammatone filters, a
compression function, a second bank of gammatone filters, and a low
pass function, in that order.

The parameters are given in the form ``10**(p0+mlog10(cf))``.
'''
from brian2 import *
from brian2.hears import *

simulation_duration = 50*ms
samplerate = 50*kHz
level = 50*dB  # level of the input sound in rms dB SPL
sound = whitenoise(simulation_duration, samplerate).ramp()
sound.level = level
 
nbr_cf = 50  #number of centre frequencies
#center frequencies with a spacing following an ERB scale
center_frequencies = erbspace(100*Hz,1000*Hz, nbr_cf)

#conversion to stape velocity (which are the units needed by the following centres)
sound = sound*0.00014

#### Linear Pathway ####

#bandpass filter (second order gammatone filter)
center_frequencies_linear = 10**(-0.067+1.016*log10(center_frequencies))
bandwidth_linear = 10**(0.037+0.785*log10(center_frequencies))
order_linear = 3
gammatone = ApproximateGammatone(sound, center_frequencies_linear,
                                 bandwidth_linear, order=order_linear)

#linear gain
g = 10**(4.2-0.48*log10(center_frequencies))
func_gain = lambda x:g*x
gain = FunctionFilterbank(gammatone, func_gain)

#low pass filter(cascade of 4 second order lowpass butterworth filters)
cutoff_frequencies_linear = center_frequencies_linear
order_lowpass_linear = 2
lp_l = LowPass(gain, cutoff_frequencies_linear)
lowpass_linear = Cascade(gain, lp_l, 4)

#### Nonlinear Pathway ####

#bandpass filter (third order gammatone filters)
center_frequencies_nonlinear = center_frequencies
bandwidth_nonlinear = 10**(-0.031+0.774*log10(center_frequencies))
order_nonlinear = 3
bandpass_nonlinear1 = ApproximateGammatone(sound, center_frequencies_nonlinear,
                                           bandwidth_nonlinear,
                                           order=order_nonlinear)

#compression (linear at low level, compress at high level)
a = 10**(1.402+0.819*log10(center_frequencies))  #linear gain
b = 10**(1.619-0.818*log10(center_frequencies))  
v = .2 #compression exponent
func_compression = lambda x: sign(x)*minimum(a*abs(x), b*abs(x)**v)
compression = FunctionFilterbank(bandpass_nonlinear1, func_compression)

#bandpass filter (third order gammatone filters)
bandpass_nonlinear2 = ApproximateGammatone(compression,
                                           center_frequencies_nonlinear,
                                           bandwidth_nonlinear,
                                           order=order_nonlinear)

#low pass filter
cutoff_frequencies_nonlinear = center_frequencies_nonlinear
order_lowpass_nonlinear = 2
lp_nl = LowPass(bandpass_nonlinear2, cutoff_frequencies_nonlinear)
lowpass_nonlinear = Cascade(bandpass_nonlinear2, lp_nl, 3)

#adding the two pathways
dnrl_filter = lowpass_linear+lowpass_nonlinear

dnrl = dnrl_filter.process()

figure()
imshow(flipud(dnrl.T), aspect='auto')    
show()
