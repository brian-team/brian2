"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

import numpy as np

from brian2.utils.logger import get_logger
from brian2.core.names import Nameable
from brian2.core.variables import Variables
from brian2.groups.group import Group, CodeRunner
from brian2.units.fundamentalunits import check_units, Quantity, Unit
from brian2.units.allunits import second, msecond

__all__ = ['Clock', 'defaultclock']

logger = get_logger(__name__)


class Clock(Group, CodeRunner):
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
    add_to_magic_network = False
    def __init__(self, dt, name='clock*'):
        # We need a name right away because some devices (e.g. cpp_standalone)
        # need a name for the object when creating the variables
        Nameable.__init__(self, name=name)
        #: The internally used dt. Note that right after a change of dt, this
        #: will not equal the new dt (which is stored in `Clock._new_dt`). Call
        #: `Clock._set_t_update_t` to update the internal clock representation.
        self._dt = float(dt)
        #: The "pure Python" copy of the time t -- needed so we can do checks of
        #: the time in Python, even in standalone mode
        self._t = 0.0
        self._new_dt = None
        self.variables = Variables(self)
        self.variables.add_array('timestep', unit=Unit(1), size=1,
                                 dtype=np.uint64, read_only=True, scalar=True)
        self.variables.add_array('t', unit=second, size=1,
                                 dtype=np.double, read_only=True, scalar=True)
        self.variables.add_array('dt', unit=second, size=1, values=float(dt),
                                 dtype=np.float, read_only=False, constant=True,
                                 scalar=True)
        self.variables.add_constant('N', unit=Unit(1), value=1)
        self.codeobj_class = None
        CodeRunner.__init__(self, group=self, template='stateupdate',
                            code='''timestep += 1
                                    t = timestep * dt''',
                            user_code='',
                            clock=self, when='after_end',
                            name=None)  # Name as already been set
        self._enable_group_attributes()
        logger.debug("Created clock {name} with dt={dt}".format(name=self.name,
                                                                dt=self.dt))

    @check_units(t=second)
    def _set_t_update_dt(self, target_t=0*second):
        the_dt = self._new_dt if self._new_dt is not None else self._dt
        target_t = float(target_t)
        if the_dt != self._dt:
            self._new_dt = None  # i.e.: i is up-to-date for the dt
            # Only allow a new dt which allows to correctly set the new time step
            if target_t != self._t:
                old_t = np.uint64(np.round(target_t / self._dt)) * self._dt
                new_t = np.uint64(np.round(target_t / the_dt)) * the_dt
                error_t = target_t
            else:
                old_t = np.uint64(np.round(self._t / self._dt)) * self._dt
                new_t = np.uint64(np.round(self._t / the_dt)) * the_dt
                error_t = self._t
            if abs(new_t - old_t) > self.epsilon:
                raise ValueError(('Cannot set dt from {old} to {new}, the '
                                  'time {t} is not a multiple of '
                                  '{new}').format(old=self.dt,
                                                  new=the_dt*second,
                                                  t=error_t*second))
            self._dt = the_dt

        new_i = np.uint64(np.round(target_t/the_dt))
        new_t = new_i*self.dt_
        if new_t==target_t or np.abs(new_t-target_t)<=self.epsilon*np.abs(new_t):
            new_timestep = new_i
        else:
            new_timestep = np.uint64(np.ceil(target_t/the_dt))
        self.variables['timestep'].set_value(new_timestep)
        self.state('t')[:] = 'timestep * dt'
        self._t = new_timestep * the_dt
        logger.debug("Setting Clock {name} to t={t}, dt={dt}".format(name=self.name,
                                                                     t=Quantity(self._t, dim=second.dim),
                                                                     dt=Quantity(self._dt, dim=second.dim)))

    def __repr__(self):
        return 'Clock(dt=%r, name=%r)' % (
            # self._new_dt*second
            #                               if self._new_dt is not None
            #                               else
                                          self.dt,
                                          self.name)

    @check_units(end=second)
    def _set_t_end(self, end):
        self._i_end = np.uint64(float(end) / self.dt_)

    @property
    def t_(self):
        'The simulation time as a float (in seconds)'
        try:
            return float(self.timestep*self._dt)
        except NotImplementedError:
            # Standalone mode
            return self._t

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
        self.variables['dt'].set_value(dt_)

    @check_units(dt=second)
    def _set_dt(self, dt):
        self._new_dt = float(dt)
        self.variables['dt'].set_value(float(dt))

    dt = property(fget=lambda self: Quantity(self.dt_, dim=second.dim),
                  fset=_set_dt,
                  doc='''The time step of the simulation in seconds.''',
                  )
    dt_ = property(fget=_get_dt_, fset=_set_dt_,
                   doc='''The time step of the simulation as a float (in seconds)''')
    _t_end = property(fget=lambda self: self._i_end*self._dt,
                      doc='The time the simulation will end as a float (in seconds)')

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
        return self.timestep < self._i_end

    epsilon = 1e-14


class DefaultClockProxy(object):
    '''
    Method proxy for access to the currently active device
    '''
    def __getattr__(self, name):
        if name == '_is_proxy':
            return True
        from brian2.devices.device import active_device
        return getattr(active_device.defaultclock, name)

    def __setattr__(self, key, value):
        from brian2.devices.device import active_device
        # TODO: Why should this happend?
        if active_device.defaultclock is not None:
            return setattr(active_device.defaultclock, key, value)

defaultclock = DefaultClockProxy()
