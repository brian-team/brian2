# WORKS!
#!/usr/bin/env python
'''
This example implements a band pass filter whose center frequency is modulated by
a sinusoid function. This modulator is implemented as a
:class:`~brian.hears.FunctionFilterbank`. One  state variable (here time) must
be kept; it is therefore implemented with a class.
The bandpass filter coefficients update is an example of how to use a
:class:`~brian.hears.ControlFilterbank`. The bandpass filter is a basic
biquadratic filter for which the Q factor and the center
frequency must be given. The input is a white noise.
'''

from brian2 import *
from brian2.hears import *


samplerate = 20*kHz
SoundDuration = 300*ms
sound = whitenoise(SoundDuration, samplerate).ramp() 

#number of frequency channel (here it must be one as a spectrogram of the
#output is plotted)
nchannels = 1   

fc_init = 5000*Hz   #initial center frequency of the band pass filter
Q = 5               #quality factor of the band pass filter
update_interval = 1 # the filter coefficients are updated every sample

mean_center_freq = 4*kHz #mean frequency around which the CF will oscillate
amplitude = 1500*Hz      #amplitude of the oscillation
frequency = 10*Hz        #frequency of the oscillation

#this class is used in a FunctionFilterbank (via its __call__). It outputs the
#center frequency of the band pass filter. Its output is thus later passed as
#input to the controler. 
class CenterFrequencyGenerator(object):
    def __init__(self): 
        self.t=0*second
   
    def __call__(self, input):
        #update of the center frequency
        fc = mean_center_freq+amplitude*sin(2*pi*frequency*self.t)
        #update of the state variable
        self.t = self.t+1./samplerate 
        return fc

center_frequency = CenterFrequencyGenerator()      

fc_generator = FunctionFilterbank(sound, center_frequency)

#the updater of the controller generates new filter coefficient of the band pass
#filter based on the center frequency it receives from the fc_generator
#(its input)
class CoeffController(object):
    def __init__(self, target):
        self.BW = 2*arcsinh(1./2/Q)*1.44269
        self.target=target
        
    def __call__(self, input):
        fc = input[-1,:] #the control variables are taken as the last of the buffer
        w0 = 2*pi*fc/array(samplerate)    
        alpha = sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
        
        self.target.filt_b[:, 0, 0] = sin(w0)/2
        self.target.filt_b[:, 1, 0] = 0
        self.target.filt_b[:, 2, 0] = -sin(w0)/2
     
        self.target.filt_a[:, 0, 0] = 1+alpha
        self.target.filt_a[:, 1, 0] = -2*cos(w0)
        self.target.filt_a[:, 2, 0] = 1-alpha

# In the present example the time varying filter is a LinearFilterbank therefore
#we must initialise the filter coefficients; the one used for the first buffer computation
w0 = 2*pi*fc_init/samplerate
BW = 2*arcsinh(1./2/Q)*1.44269
alpha = sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))

filt_b = zeros((nchannels, 3, 1))
filt_a = zeros((nchannels, 3, 1))

filt_b[:, 0, 0] = sin(w0)/2
filt_b[:, 1, 0] = 0
filt_b[:, 2, 0] = -sin(w0)/2

filt_a[:, 0, 0] = 1+alpha
filt_a[:, 1, 0] = -2*cos(w0)
filt_a[:, 2, 0] = 1-alpha

#the filter which will have time varying coefficients
bandpass_filter = LinearFilterbank(sound, filt_b, filt_a)
#the updater
updater = CoeffController(bandpass_filter)

#the controller. Remember it must be the last of the chain
control = ControlFilterbank(bandpass_filter, fc_generator, bandpass_filter,
                            updater, update_interval)   
      
time_varying_filter_mon = control.process()

figure(1)
pxx, freqs, bins, im = specgram(squeeze(time_varying_filter_mon),
                                NFFT=256, Fs=float(samplerate), noverlap=240)
imshow(flipud(pxx), aspect='auto')

show()
