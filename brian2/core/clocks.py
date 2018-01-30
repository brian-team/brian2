"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

import numpy as np

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.variables import Variables
from brian2.groups.group import VariableOwner
from brian2.units.fundamentalunits import check_units, Quantity, DIMENSIONLESS
from brian2.units.allunits import second

__all__ = ['Clock', 'defaultclock']

logger = get_logger(__name__)


def check_dt(new_dt, old_dt, target_t):
    '''
    Check that the target time can be represented equally well with the new
    dt.

    Parameters
    ----------
    new_dt : float
        The new dt value
    old_dt : float
        The old dt value
    target_t : float
        The target time

    Raises
    ------
    ValueError
        If using the new dt value would lead to a difference in the target
        time of more than `Clock.epsilon_dt` times ``new_dt`` (by default,
        0.01% of the new dt).

    Examples
    --------
    >>> from brian2 import *
    >>> check_dt(float(17*ms), float(0.1*ms), float(0*ms))  # For t=0s, every dt is fine
    >>> check_dt(float(0.05*ms), float(0.1*ms), float(10*ms))  # t=10*ms can be represented with the new dt
    >>> check_dt(float(0.2*ms), float(0.1*ms), float(10.1*ms))  # t=10.1ms cannot be represented with dt=0.2ms # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Cannot set dt from 100. us to 200. us, the time 10.1 ms is not a multiple of 200. us
    '''
    old_t = np.int64(np.round(target_t / old_dt)) * old_dt
    new_t = np.int64(np.round(target_t / new_dt)) * new_dt
    error_t = target_t
    if abs(new_t - old_t)/new_dt > Clock.epsilon_dt:
        raise ValueError(('Cannot set dt from {old} to {new}, the '
                          'time {t} is not a multiple of '
                          '{new}').format(old=str(old_dt * second),
                                          new=str(new_dt * second),
                                          t=str(error_t * second)))


class Clock(VariableOwner):
    '''
    An object that holds the simulation time and the time step.
    
    Parameters
    ----------
    dt : float
        The time step of the simulation as a float
    name : str, optional
        An explicit name, if not specified gives an automatically generated name

    Notes
    -----
    Clocks are run in the same `Network.run` iteration if `~Clock.t` is the
    same. The condition for two
    clocks to be considered as having the same time is
    ``abs(t1-t2)<epsilon*abs(t1)``, a standard test for equality of floating
    point values. The value of ``epsilon`` is ``1e-14``.
    '''

    def __init__(self, dt, name='clock*'):
        # We need a name right away because some devices (e.g. cpp_standalone)
        # need a name for the object when creating the variables
        Nameable.__init__(self, name=name)
        self._old_dt = None
        self.variables = Variables(self)
        self.variables.add_array('timestep', size=1, dtype=np.int64,
                                 read_only=True, scalar=True)
        self.variables.add_array('t', dimensions=second.dim, size=1,
                                 dtype=np.double, read_only=True, scalar=True)
        self.variables.add_array('dt', dimensions=second.dim, size=1, values=float(dt),
                                 dtype=np.float, read_only=True, constant=True,
                                 scalar=True)
        self.variables.add_constant('N', value=1)
        self._enable_group_attributes()
        self.dt = dt
        logger.diagnostic("Created clock {name} with dt={dt}".format(name=self.name,
                                                                     dt=self.dt))

    @check_units(t=second)
    def _set_t_update_dt(self, target_t=0*second):
        new_dt = self.dt_
        old_dt = self._old_dt
        target_t = float(target_t)
        if old_dt is not None and new_dt != old_dt:
            self._old_dt = None
            # Only allow a new dt which allows to correctly set the new time step
            check_dt(new_dt, old_dt, target_t)

        new_timestep = self._calc_timestep(target_t)
        # Since these attributes are read-only for normal users, we have to
        # update them via the variables object directly
        self.variables['timestep'].set_value(new_timestep)
        self.variables['t'].set_value(new_timestep * new_dt)
        logger.diagnostic("Setting Clock {name} to t={t}, dt={dt}".format(name=self.name,
                                                                          t=self.t,
                                                                          dt=self.dt))

    def _calc_timestep(self, target_t):
        '''
        Calculate the integer time step for the target time. If it cannot be
        exactly represented (up to 0.01% of dt), round up.

        Parameters
        ----------
        target_t : float
            The target time in seconds

        Returns
        -------
        timestep : int
            The target time in integers (based on dt)
        '''
        new_i = np.int64(np.round(target_t / self.dt_))
        new_t = new_i * self.dt_
        if (new_t == target_t or
                        np.abs(new_t - target_t)/self.dt_ <= Clock.epsilon_dt):
            new_timestep = new_i
        else:
            new_timestep = np.int64(np.ceil(target_t / self.dt_))
        return new_timestep

    def __repr__(self):
        return 'Clock(dt=%r, name=%r)' % (self.dt, self.name)

    def _get_dt_(self):
        return self.variables['dt'].get_value().item()

    @check_units(dt_=1)
    def _set_dt_(self, dt_):
        self._old_dt = self._get_dt_()
        self.variables['dt'].set_value(dt_)

    @check_units(dt=second)
    def _set_dt(self, dt):
        self._set_dt_(float(dt))

    dt = property(fget=lambda self: Quantity(self.dt_, dim=second.dim),
                  fset=_set_dt,
                  doc='''The time step of the simulation in seconds.''',
                  )
    dt_ = property(fget=_get_dt_, fset=_set_dt_,
                   doc='''The time step of the simulation as a float (in seconds)''')

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        '''
        set_interval(self, start, end)

        Set the start and end time of the simulation.

        Sets the start and end value of the clock precisely if
        possible (using epsilon) or rounding up if not. This assures that
        multiple calls to `Network.run` will not re-run the same time step.      
        '''
        self._set_t_update_dt(target_t=start)
        end = float(end)
        self._i_end = self._calc_timestep(end)
        if self._i_end > 2**40:
            logger.warn('The end time of the simulation has been set to {}, '
                        'which based on the dt value of {} means that {} '
                        'time steps will be simulated. This can lead to '
                        'numerical problems, e.g. the times t will not '
                        'correspond to exact multiples of '
                        'dt.'.format(str(end*second),
                                     str(self.dt),
                                     self._i_end),
                        'many_timesteps')

    #: The relative difference for times (in terms of dt) so that they are
    #: considered identical.
    epsilon_dt = 1e-4


class DefaultClockProxy(object):
    '''
    Method proxy to access the defaultclock of the currently active device
    '''
    def __getattr__(self, name):
        if name == '_is_proxy':
            return True
        from brian2.devices.device import active_device
        return getattr(active_device.defaultclock, name)

    def __setattr__(self, key, value):
        from brian2.devices.device import active_device
        setattr(active_device.defaultclock, key, value)

#: The standard clock, used for objects that do not specify any clock or dt
defaultclock = DefaultClockProxy()
