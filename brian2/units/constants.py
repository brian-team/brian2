r"""
A module providing some physical units as `Quantity` objects. Note that these
units are not imported by wildcard imports (e.g. `from brian2 import *`), they
have to be imported explicitly. You can use ``import ... as ...`` to import them
with shorter names, e.g.::

    from brian2.units.constants import faraday_constant as F

The available constants are:

==================== ================== ======================= ==================================================================
Constant             Symbol(s)          Brian name              Value
==================== ================== ======================= ==================================================================
Avogadro constant    :math:`N_A, L`     ``avogadro_constant``   :math:`6.022140857\times 10^{23}\,\mathrm{mol}^{-1}`
Boltzmann constant   :math:`k`          ``boltzmann_constant``  :math:`1.38064852\times 10^{-23}\,\mathrm{J}\,\mathrm{K}^{-1}`
Electric constant    :math:`\epsilon_0` ``electric_constant``   :math:`8.854187817\times 10^{-12}\,\mathrm{F}\,\mathrm{m}^{-1}`
Electron mass        :math:`m_e`        ``electron_mass``       :math:`9.10938356\times 10^{-31}\,\mathrm{kg}`
Elementary charge    :math:`e`          ``elementary_charge``   :math:`1.6021766208\times 10^{-19}\,\mathrm{C}`
Faraday constant     :math:`F`          ``faraday_constant``    :math:`96485.33289\,\mathrm{C}\,\mathrm{mol}^{-1}`
Gas constant         :math:`R`          ``gas_constant``        :math:`8.3144598\,\mathrm{J}\,\mathrm{mol}^{-1}\,\mathrm{K}^{-1}`
Magnetic constant    :math:`\mu_0`      ``magnetic_constant``   :math:`12.566370614\times 10^{-7}\,\mathrm{N}\,\mathrm{A}^{-2}`
Molar mass constant  :math:`M_u`        ``molar_mass_constant`` :math:`1\times 10^{-3}\,\mathrm{kg}\,\mathrm{mol}^{-1}`
0°C                                     ``zero_celsius``        :math:`273.15\,\mathrm{K}`
==================== ================== ======================= ==================================================================
"""

import numpy as np

from .allunits import (
    amp,
    coulomb,
    farad,
    gram,
    joule,
    kelvin,
    kilogram,
    meter,
    mole,
    newton,
)
from .fundamentalunits import Unit

Unit.automatically_register_units = False

#: Avogadro constant (http://physics.nist.gov/cgi-bin/cuu/Value?na)
avogadro_constant = 6.022140857e23 / mole
#: Boltzmann constant (physics.nist.gov/cgi-bin/cuu/Value?k)
boltzmann_constant = 1.38064852e-23 * joule / kelvin
#: electric constant (http://physics.nist.gov/cgi-bin/cuu/Value?ep0)
electric_constant = 8.854187817e-12 * farad / meter
#: Electron rest mass (physics.nist.gov/cgi-bin/cuu/Value?me)
electron_mass = 9.10938356e-31 * kilogram
#: Elementary charge (physics.nist.gov/cgi-bin/cuu/Value?e)
elementary_charge = 1.6021766208e-19 * coulomb
#: Faraday constant (http://physics.nist.gov/cgi-bin/cuu/Value?f)
faraday_constant = 96485.33289 * coulomb / mole
#: gas constant (http://physics.nist.gov/cgi-bin/cuu/Value?r)
gas_constant = 8.3144598 * joule / mole / kelvin
#: Magnetic constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu0)
magnetic_constant = 4 * np.pi * 1e-7 * newton / amp**2
#: Molar mass constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu)
molar_mass_constant = 1 * gram / mole
#: zero degree Celsius
zero_celsius = 273.15 * kelvin

Unit.automatically_register_units = True
