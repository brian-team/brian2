"""This module is used by the package Brian for turning units on and off.

For the user, you can ignore this unless you want to turn units off,
in which case you simply put at the top of your program (before
importing anything from brian):

from brian_unit_prefs import turn_off_units
turn_off_units()

Or to turn units off and not have a warning printed:

from brian_unit_prefs import turn_off_units
turn_off_units(warn=False)
"""


class _unit_prefs():
    pass

bup = _unit_prefs()

bup.use_units = True
bup.warn_about_units = True

def turn_off_units(warn=True):
    bup.use_units = False
    bup.warn_about_units = warn
