# WORKS!
#!/usr/bin/env python
'''
Implementation example of the compressive gammachirp auditory filter as
described in Irino, T. and Patterson R., "A compressive gammachirp auditory
filter for both physiological and psychophysical data", JASA 2001.

A class called :class:`~brian.hears.DCGC` implementing this model is available
in the library.

Technical implementation details and notation can be found in Irino, T. and
Patterson R., "A Dynamic Compressive Gammachirp Auditory Filterbank",
IEEE Trans Audio Speech Lang Processing.
'''
from brian2 import *
from brian2.hears import *

simulation_duration = 50*ms
samplerate = 50*kHz
level = 50*dB # level of the input sound in rms dB SPL
sound = whitenoise(simulation_duration, samplerate).ramp()
sound = sound.atlevel(level)

nbr_cf = 50 # number of centre frequencies
# center frequencies with a spacing following an ERB scale
cf = erbspace(100*Hz, 1000*Hz, nbr_cf)

c1 = -2.96 #glide slope of the first filterbank
b1 = 1.81  #factor determining the time constant of the first filterbank
c2 = 2.2   #glide slope of the second filterbank
b2 = 2.17  #factor determining the time constant of the second filterbank

order_ERB = 4
ERBrate = 21.4*log10(4.37*cf/1000+1)
ERBwidth = 24.7*(4.37*cf/1000 + 1)
ERBspace = mean(diff(ERBrate))

# the filter coefficients are updated every update_interval (here in samples)
update_interval = 1

                  

#bank of passive gammachirp filters. As the control path uses the same passive
#filterbank than the signal path (but shifted in frequency)
#this filterbank is used by both pathway.
pGc = LogGammachirp(sound, cf, b=b1, c=c1)

fp1 = cf + c1*ERBwidth*b1/order_ERB #centre frequency of the signal path

#### Control Path ####

#the first filterbank in the control path consists of gammachirp filters
#value of the shift in ERB frequencies of the control path with respect to the signal path
lct_ERB = 1.5
n_ch_shift = round(lct_ERB/ERBspace) #value of the shift in channels
#index of the channel of the control path taken from pGc
indch1_control = minimum(maximum(1, arange(1, nbr_cf+1)+n_ch_shift), nbr_cf).astype(int)-1 
fp1_control = fp1[indch1_control]
#the control path bank pass filter uses the channels of pGc indexed by indch1_control
pGc_control = RestructureFilterbank(pGc, indexmapping=indch1_control)

#the second filterbank in the control path consists of fixed asymmetric compensation filters
frat_control = 1.08
fr2_control = frat_control*fp1_control
asym_comp_control = AsymmetricCompensation(pGc_control, fr2_control, b=b2, c=c2)

#definition of the pole of the asymmetric comensation filters
p0 = 2
p1 = 1.7818*(1-0.0791*b2)*(1-0.1655*abs(c2))
p2 = 0.5689*(1-0.1620*b2)*(1-0.0857*abs(c2))
p3 = 0.2523*(1-0.0244*b2)*(1+0.0574*abs(c2))
p4 = 1.0724

#definition of the parameters used in the control path output levels computation
#(see IEEE paper for details)
decay_tcst = .5*ms
order = 1.
lev_weight = .5
level_ref = 50.
level_pwr1 = 1.5
level_pwr2 = .5
RMStoSPL = 30.
frat0 = .2330
frat1 = .005 
exp_deca_val = exp(-1/(decay_tcst*samplerate)*log(2))
level_min = 10**(-RMStoSPL/20)

#definition of the controller class. What is does it take the outputs of the
#first and second fitlerbanks of the control filter as input, compute an overall
#intensity level for each frequency channel. It then uses those level to update
#the filter coefficient of its target, the asymmetric compensation filterbank of
#the signal path.
class CompensensationFilterUpdater(object): 
    def __init__(self, target):
        self.target = target
        self.level1_prev = -100
        self.level2_prev = -100
        
    def __call__(self, *input):
         value1 = input[0][-1,:]
         value2 = input[1][-1,:]
         #the current level value is chosen as the max between the current
         #output and the previous one decreased by a decay
         level1 = maximum(maximum(value1, 0), self.level1_prev*exp_deca_val) 
         level2 = maximum(maximum(value2, 0), self.level2_prev*exp_deca_val)

         self.level1_prev = level1 #the value is stored for the next iteration
         self.level2_prev = level2
         #the overall intensity is computed between the two filterbank outputs
         level_total = lev_weight*level_ref*(level1/level_ref)**level_pwr1+\
                   (1-lev_weight)*level_ref*(level2/level_ref)**level_pwr2
         #then it is converted in dB
         level_dB = 20*log10(maximum(level_total, level_min))+RMStoSPL
         #the frequency factor is calculated           
         frat = frat0 + frat1*level_dB
         #the centre frequency of the asymmetric compensation filters are updated       
         fr2 = fp1*frat
         coeffs = asymmetric_compensation_coeffs(samplerate, fr2,
                        self.target.filt_b, self.target.filt_a, b2, c2,
                        p0, p1, p2, p3, p4)
         self.target.filt_b, self.target.filt_a = coeffs                 

#### Signal Path ####
#the signal path consists of the passive gammachirp filterbank pGc previously
#defined followed by a asymmetric compensation filterbank
fr1 = fp1*frat0
varyingfilter_signal_path = AsymmetricCompensation(pGc, fr1, b=b2, c=c2)
updater = CompensensationFilterUpdater(varyingfilter_signal_path)
 #the controler which takes the two filterbanks of the control path as inputs
 #and the varying filter of the signal path as target is instantiated
control = ControlFilterbank(varyingfilter_signal_path,
                            [pGc_control, asym_comp_control],
                            varyingfilter_signal_path, updater, update_interval)  

#run the simulation
#Remember that the controler are at the end of the chain and the output of the
#whole path comes from them
signal = control.process() 

figure()
imshow(flipud(signal.T), aspect='auto')    
show()
