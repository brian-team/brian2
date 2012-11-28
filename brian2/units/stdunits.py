######### PHYSICAL UNIT NAMES #####################
#------------------------------------ Dan Goodman -
# These are optional shorthand unit names which in
# most circumstances shouldn't clash with local names

"""Optional short unit names

This module defines the following short unit names:

mV, mA, uA (micro_amp), nA, pA, mF, uF, nF, mS, uS, ms,
Hz, kHz, MHz, cm, cm2, cm3, mm, mm2, mm3, um, um2, um3
"""

from .allunits import (mvolt, mamp, uamp, namp, pamp, pfarad, ufarad, nfarad,
                       nsiemens, usiemens, msecond, usecond, hertz, khertz,
                       Mhertz, cmetre, cmetre2, cmetre3, mmetre, mmetre2,
                       mmetre3, umetre, umetre2, umetre3)
from .allunits import all_units

__all__ = ['mV', 'mA', 'uA', 'nA', 'pA', 'pF', 'uF', 'nF', 'nS', 'uS', 'ms',
           'us', 'Hz', 'kHz', 'MHz', 'cm', 'cm2', 'cm3', 'mm', 'mm2', 'mm3',
           'um', 'um2', 'um3']
mV = mvolt

mA = mamp
uA = uamp
nA = namp
pA = pamp

pF = pfarad
uF = ufarad
nF = nfarad

nS = nsiemens
uS = usiemens

ms = msecond
us = usecond

Hz = hertz
kHz = khertz
MHz = Mhertz

cm = cmetre
cm2 = cmetre2
cm3 = cmetre3
mm = mmetre
mm2 = mmetre2
mm3 = mmetre3
um = umetre
um2 = umetre2
um3 = umetre3

stdunits = {'mV': mV, 'mA': mA, 'uA': uA, 'nA': nA, 'pA': pA, 'pF': pF,
            'uF': uF, 'nF': nF, 'nS': nS, 'uS': uS, 'ms': ms, 'us': us,
            'Hz' : Hz, 'kHz': kHz, 'MHz': MHz, 'cm': cm, 'cm2': cm2,
            'cm3': cm3, 'mm': mm, 'mm2': mm2, 'mm3': mm3, 'um': um, 'um2': um2,
            'um3': um3}
all_units.extend(stdunits)
