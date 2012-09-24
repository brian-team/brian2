"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

__all__ = ['Clock', 'defaultclock']

from brian2.units import second, msecond, check_units
from time import time
from numpy import ceil


class Clock(object):
    '''
    Clock(dt=0.1*ms, order=0)
    
    An object that holds the simulation time and the time step.
    
    Parameters
    ----------
    dt : Quantity, optional
        The time step of the simulation, will be set to ``0.1*ms`` if
        unspecified.
    order : int, optional
        If two clocks have the same time, the order of the clock is used to
        resolve which clock is processed first, lower orders first.

    Notes
    -----
    In order to make sure that certain operations happen in the correct
    sequence, you can use the ``order`` attribute, clocks with a lower order
    will be processed first if the time is the same. The condition for two
    clocks to be considered as having the same time is
    ``abs(t1-t2)<epsilon*abs(t1)``, a standard test for equality of floating
    point values. The value of ``epsilon`` is ``1e-14``.
    '''
    
    @check_units(dt=second, t=second)
    def __init__(self, dt=None, order=0):
        self._dt_spec = dt
        self.i = 0  #: The time step of the simulation as an integer.
        self.i_end = 0  #: The time step the simulation will end as an integer

        #: In which order two clocks should be processed when they have the
        #: same time. Lower orders will be processed first.
        self.order = order

    def reinit(self):
        '''
        Reinitialises the clock time to zero.
        '''
        self.i = 0

    def __str__(self):
        return 'Clock: t = ' + str(self.t) + ', dt = ' + str(self.dt)
    
    def __repr__(self):
        return 'Clock(dt=%s, order=%s)' % (repr(self.dt), repr(self.order))

    def tick(self):
        '''
        Advances the clock by one time step.
        '''
        self.i += 1

    @check_units(t=second)
    def _set_t(self, t):
        self.i = int(float(t) / self.dt_)

    def _set_t_(self, t):
        self.i = int(t/self.dt_)

    @check_units(end=second)
    def _set_t_end(self, end):
        self.i_end = int(float(end) / self.dt_)
        
    def _get_dt_(self):
        if hasattr(self, '_dt'):
            return self._dt
        else:
            dtspec = self._dt_spec
            if dtspec is None:
                dtspec = 0.1*msecond
            self._dt = float(dtspec)
            return self._dt
            
    def _set_dt_(self, dt_):
        if hasattr(self, '_dt'):
            raise RuntimeError("Cannot change dt, it has already been set to "+str(self.dt))
        self._dt = dt_
    
    @check_units(dt=second)
    def _set_dt(self, dt):
        self.dt_ = float(dt)
    
    dt = property(fget=lambda self: self.dt_*second,
                  fset=_set_dt,
                  doc='''The time step of the simulation in seconds
                         Returns a Quantity, and can only
                         be set once. Defaults to ``0.1*ms``.''',
                  )
    
    dt_ = property(fget=_get_dt_, fset=_set_dt_,
                   doc='''The time step of the simulation as a float (in seconds)''')
    t = property(fget=lambda self: self.i*self.dt_*second,
                 fset=_set_t,
                 doc='The simulation time in seconds')
    t_ = property(fget=lambda self: self.i*self.dt_,
                  fset=_set_t_,
                  doc='The simulation time as a float (in seconds)')
    t_end = property(fget=lambda self: self.i_end*self.dt_*second,
                     fset=_set_t_end,
                     doc='The time the simulation will end (in seconds)')

    @check_units(duration=second)
    def set_duration(self, duration):
        '''
        set_duration(self, duration)
        
        Set the time until the current simulation ends. Some more text for test.
        '''
        self.i_end = self.i+int(float(duration)/self.dt_)

    @property
    def still_running(self):
        '''
        A ``bool`` to indicate whether the current simulation is still running.
        '''
        return self.i<self.i_end

    epsilon = 1e-14

    def __lt__(self, other):
        selft = self._t
        othert = other._t
        if selft==othert or abs(selft-othert)<=self.epsilon*abs(selft):
            return self.order<other.order
        return selft<othert
    
    
defaultclock = Clock()
