"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

import numpy as np

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.units.fundamentalunits import check_units, Quantity
from brian2.units.allunits import second, msecond

__all__ = ['Clock', 'defaultclock']

logger = get_logger(__name__)


class Clock(Nameable):
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
        self._i = 0
        #: The internally used dt. Note that right after a change of dt, this
        #: will not equal the new dt (which is stored in `Clock._new_dt`). Call
        #: `Clock._set_t_update_t` to update the internal clock representation.
        self._dt = float(dt)
        self._new_dt = None
        Nameable.__init__(self, name=name)
        logger.debug("Created clock {self.name} with dt={self._dt}".format(self=self))

    @check_units(t=second)
    def _set_t_update_dt(self, t=0*second):
        dt = self._new_dt if self._new_dt is not None else self._dt
        t = float(t)
        if dt != self._dt:
            self._new_dt = None  # i.e.: i is up-to-date for the dt
            # Only allow a new dt which allows to correctly set the new time step
            if t != self.t_:
                old_t = np.uint64(np.round(t / self._dt)) * self._dt
                new_t = np.uint64(np.round(t / dt)) * dt
                error_t = t
            else:
                old_t = np.uint64(np.round(self.t_ / self._dt)) * self._dt
                new_t = np.uint64(np.round(self.t_ / dt)) * dt
                error_t = self.t_
            if abs(new_t - old_t) > self.epsilon:
                raise ValueError(('Cannot set dt from {old} to {new}, the '
                                  'time {t} is not a multiple of '
                                  '{new}').format(old=self.dt,
                                                  new=dt*second,
                                                  t=error_t*second))
            self._dt = dt

        new_i = np.uint64(np.round(t/dt))
        new_t = new_i*self.dt_
        if new_t==t or np.abs(new_t-t)<=self.epsilon*np.abs(new_t):
            self._i = new_i
        else:
            self._i = np.uint64(np.ceil(t/dt))
        logger.debug("Setting Clock {self.name} to t={self.t}, dt={self.dt}".format(self=self))

    def __str__(self):
        if self._new_dt is None:
            return 'Clock ' + self.name + ': t = ' + str(self.t) + ', dt = ' + str(self.dt)
        else:
            return 'Clock ' + self.name + ': t = ' + str(self.t) + ', (new) dt = ' + str(self._new_dt*second)
    
    def __repr__(self):
        return 'Clock(dt=%r, name=%r)' % (self._new_dt*second
                                          if self._new_dt is not None
                                          else self.dt,
                                          self.name)

    def tick(self):
        '''
        Advances the clock by one time step.
        '''
        self._i += 1

    @check_units(end=second)
    def _set_t_end(self, end):
        self._i_end = np.uint64(float(end) / self.dt_)

    @property
    def t_(self):
        'The simulation time as a float (in seconds)'
        return float(self._i*self._dt)

    @property
    def t(self):
        'The simulation time in seconds'
        return self.t_*second

    def _get_dt_(self):
        if self._new_dt is None:
            return self._dt
        else:
            return self._new_dt

    @check_units(dt_=1)
    def _set_dt_(self, dt_):
        self._new_dt = dt_
    
    @check_units(dt=second)
    def _set_dt(self, dt):
        self._new_dt = float(dt)
    
    dt = property(fget=lambda self: Quantity(self.dt_, dim=second.dim),
                  fset=_set_dt,
                  doc='''The time step of the simulation in seconds.''',
                  )
    dt_ = property(fget=_get_dt_, fset=_set_dt_,
                   doc='''The time step of the simulation as a float (in seconds)''')
    t_end = property(fget=lambda self: self._i_end*self.dt_*second,
                     doc='The time the simulation will end (in seconds)')

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        '''
        set_interval(self, start, end)
        
        Set the start and end time of the simulation.
        
        Sets the start and end value of the clock precisely if
        possible (using epsilon) or rounding up if not. This assures that
        multiple calls to `Network.run` will not re-run the same time step.      
        '''
        self._set_t_update_dt(t=start)
        end = float(end)
        i_end = np.uint64(np.round(end/self.dt_))
        t_end = i_end*self.dt_
        if t_end==end or np.abs(t_end-end)<=self.epsilon*np.abs(t_end):
            self._i_end = i_end
        else:
            self._i_end = np.uint64(np.ceil(end/self.dt_))

    @property
    def running(self):
        '''
        A ``bool`` to indicate whether the current simulation is running.
        '''
        return self._i < self._i_end

    epsilon = 1e-14

defaultclock = Clock(dt=0.1*msecond, name='defaultclock')
