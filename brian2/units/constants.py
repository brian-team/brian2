'''
A module providing some physical units as `Quantity` objects. Note that these
units are not imported by wildcard imports (e.g. `from brian2 import *`), since
their names are likely to clash with local variables. Import them explicitly
instead.
'''
import numpy as np

from .allunits import (amp, coulomb, farad, gram, joule, kelvin, kilogram,
                       meter, mole, newton)
from .fundamentalunits import Unit

# FIXME: I am not quite sure why this is necessary
Unit.automatically_register_units = False

# Boltzmann constant
k = k_B = Boltzmann = 1.38064852e-23*joule/kelvin
# gas constant (http://physics.nist.gov/cgi-bin/cuu/Value?r)
R = gas_constant = 8.3144598*joule/mole/kelvin
# Avogadro constant (http://physics.nist.gov/cgi-bin/cuu/Value?na)
L = N_A = Avogadro = 6.022140857e23/mole
# Faraday constant (http://physics.nist.gov/cgi-bin/cuu/Value?f)
F = Faraday = 96485.33289*coulomb/mole
# Elementary charge (physics.nist.gov/cgi-bin/cuu/Value?e)
e = elementary_charge = 1.6021766208e-19*coulomb
# zero degree Celsius
zero_celsius = 273.15*kelvin
# Magnetic constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu0)
mu_0 = magnetic_constant = vacuum_permeability = 4*np.pi*1e-7*newton/amp**2
# Electron rest mass (physics.nist.gov/cgi-bin/cuu/Value?me)
m_e = electron_mass = 9.10938356e-31*kilogram
# Molar mass (http://physics.nist.gov/cgi-bin/cuu/Value?mu)
M_u = molar_mass = 1*gram/mole
# electric constant (http://physics.nist.gov/cgi-bin/cuu/Value?ep0)
epsilon_0 = electric_constant = vacuum_permittivity = 8.854187817e-12*farad/meter

Unit.automatically_register_units = True
