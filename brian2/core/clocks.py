"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

from numpy import ceil 

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.units.fundamentalunits import check_units, Quantity
from brian2.units.allunits import second, msecond

__all__ = ['Clock', 'defaultclock']

logger = get_logger(__name__)


class Clock(Nameable):
    '''
    Clock(dt=0.1*ms, name=None)
    
    An object that holds the simulation time and the time step.
    
    Parameters
    ----------
    dt : `Quantity`, optional
        The time step of the simulation, will be set to ``0.1*ms`` if
        unspecified.
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
    
    @check_units(dt=second)
    def __init__(self, dt=0.1*msecond, name='clock*'):
        self._force_reinit(dt=dt)
        Nameable.__init__(self, name=name)
        logger.debug("Created clock {self.name} with dt={self._dt}".format(self=self))
        
    def _force_reinit(self, dt=0.1*msecond):
        self._dt = float(dt)
        self.i = 0  #: The time step of the simulation as an integer.
        self.i_end = 0  #: The time step the simulation will end as an integer

    def reinit(self):
        '''
        Reinitialises the clock time to zero.
        '''
        self.i = 0

    def __str__(self):
        return 'Clock ' + self.name + ': t = ' + str(self.t) + ', dt = ' + str(self.dt)
    
    def __repr__(self):
        return 'Clock(dt=%r, name=%r)' % (self.dt, self.name)

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
        return self._dt
            
    def _set_dt_(self, dt_):
        self._dt = dt_
        logger.debug("Set dt for clock {self.name} to {self.dt}".format(self=self))
    
    @check_units(dt=second)
    def _set_dt(self, dt):
        self.dt_ = float(dt)
    
    dt = property(fget=lambda self: Quantity(self.dt_, dim=second.dim),
                  fset=_set_dt,
                  doc='''The time step of the simulation in seconds.''',
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

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        '''
        set_interval(self, start, end)
        
        Set the start and end time of the simulation.
        
        Sets the start and end value of the clock precisely if
        possible (using epsilon) or rounding up if not. This assures that
        multiple calls to `Network.run` will not re-run the same time step.      
        '''
        start = float(start)
        end = float(end)
        i_start = int(round(start/self.dt_))
        t_start = i_start*self.dt_
        if t_start==start or abs(t_start-start)<=self.epsilon*abs(t_start):
            self.i = i_start
        else:
            self.i = int(ceil(start/self.dt_))
        i_end = int(round(end/self.dt_))
        t_end = i_end*self.dt_
        if t_end==end or abs(t_end-end)<=self.epsilon*abs(t_end):
            self.i_end = i_end
        else:
            self.i_end = int(ceil(end/self.dt_))

    @property
    def running(self):
        '''
        A ``bool`` to indicate whether the current simulation is running.
        '''
        return self.i<self.i_end

    epsilon = 1e-14
    
defaultclock = Clock(name='defaultclock')
