"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

from numpy import ceil 

from brian2 import second, msecond, check_units

__all__ = ['Clock', 'defaultclock']


class Clock(object):
    '''
    Clock(dt=0.1*ms)
    
    An object that holds the simulation time and the time step.
    
    Parameters
    ----------
    dt : `Quantity`, optional
        The time step of the simulation, will be set to ``0.1*ms`` if
        unspecified.

    Notes
    -----
    Clocks are run in the same `Network.run` iteration if `~Clock.t` is the
    same. The condition for two
    clocks to be considered as having the same time is
    ``abs(t1-t2)<epsilon*abs(t1)``, a standard test for equality of floating
    point values. The value of ``epsilon`` is ``1e-14``.
    '''
    
    @check_units(dt=second, t=second)
    def __init__(self, dt=None):
        self._dt_spec = dt
        self.i = 0  #: The time step of the simulation as an integer.
        self.i_end = 0  #: The time step the simulation will end as an integer

    def reinit(self):
        '''
        Reinitialises the clock time to zero.
        '''
        self.i = 0

    def __str__(self):
        return 'Clock: t = ' + str(self.t) + ', dt = ' + str(self.dt)
    
    def __repr__(self):
        return 'Clock(dt=%s)' % (repr(self.dt),)

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
                         Returns a `Quantity`, and can only
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

    def __lt__(self, other):
        selft = self.t_
        othert = other.t_
        if selft==othert or abs(selft-othert)<=self.epsilon*abs(selft):
            return False
        return selft<othert
    
    def __eq__(self, other):
        selft = self.t_
        othert = other.t_
        if selft==othert or abs(selft-othert)<=self.epsilon*abs(selft):
            return True
        else:
            return False
    
    
defaultclock = Clock()
