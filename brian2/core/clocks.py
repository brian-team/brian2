"""
Clocks for the simulator.
"""

__docformat__ = "restructuredtext en"

import numpy as np

from brian2.core.names import Nameable
from brian2.core.variables import Variables
from brian2.groups.group import VariableOwner
from brian2.units.allunits import second
from brian2.units.fundamentalunits import Quantity, check_units
from brian2.utils.logger import get_logger

__all__ = ["BaseClock", "Clock", "defaultclock", "EventClock"]

logger = get_logger(__name__)


def check_dt(new_dt, old_dt, target_t):
    """
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
        time of more than Clock.epsilon_dt times `new_dt (by default,
        0.01% of the new dt).

    Examples
    --------
    >>> from brian2 import *
    >>> check_dt(float(17*ms), float(0.1*ms), float(0*ms))  # For t=0s, every dt is fine
    >>> check_dt(float(0.05*ms), float(0.1*ms), float(10*ms))  # t=10*ms can be represented with the new dt
    >>> check_dt(float(0.2*ms), float(0.1*ms), float(10.1*ms))  # t=10.1ms cannot be represented with dt=0.2ms # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Cannot set dt from 100. us to 200. us, the time 10.1 ms is not a multiple of 200. us.
    """
    old_t = np.int64(np.round(target_t / old_dt)) * old_dt
    new_t = np.int64(np.round(target_t / new_dt)) * new_dt
    error_t = target_t
    if abs(new_t - old_t) / new_dt > Clock.epsilon_dt:
        old = str(old_dt * second)
        new = str(new_dt * second)
        t = str(error_t * second)
        raise ValueError(
            f"Cannot set dt from {old} to {new}, the "
            f"time {t} is not a multiple of {new}."
        )


class BaseClock(VariableOwner):
    """
    Base class for all clocks in the simulator.

    Parameters
    ----------
    name : str, optional
        An explicit name, if not specified gives an automatically generated name
    """

    epsilon = 1e-14

    def __init__(self, name):
        Nameable.__init__(self, name=name)
        self.variables = Variables(self)
        self.variables.add_array(
            "timestep", size=1, dtype=np.int64, read_only=True, scalar=True
        )
        self.variables.add_array(
            "t",
            dimensions=second.dim,
            size=1,
            dtype=np.float64,
            read_only=True,
            scalar=True,
        )
        self.variables["timestep"].set_value(0)

        self.variables.add_constant("N", value=1)

        self._enable_group_attributes()

        self._i_end = None
        logger.diagnostic(f"Created clock {self.name}")

    def advance(self):
        """
        Advance the clock to the next time step.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        """
        Set the start and end time of the simulation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def __lt__(self, other):
        return (
            self.variables["t"].get_value().item()
            < other.variables["t"].get_value().item()
        )

    def __gt__(self, other):
        return (
            self.variables["t"].get_value().item()
            > other.variables["t"].get_value().item()
        )

    def __le__(self, other):
        return self.__lt__(other) or self.same_time(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.same_time(other)

    def same_time(self, other):
        """
        Check if two clocks are at the same time (within epsilon).

        Parameters
        ----------
        other : BaseClock
            The other clock to compare with

        Returns
        -------
        bool
            True if both clocks are at the same time
        """
        t1 = self.variables["t"].get_value().item()
        t2 = other.variables["t"].get_value().item()

        return abs(t1 - t2) < self.epsilon


class EventClock(BaseClock):
    """
    A clock that advances through a predefined sequence of times.

    Parameters
    ----------
    times : array-like
        The sequence of times for the clock to advance through
    name : str, optional
        An explicit name, if not specified gives an automatically generated name
    """

    def __init__(self, times, name="eventclock*"):
        super().__init__(name=name)

        self.times = sorted(times)
        if len(self.times) != len(set(self.times)):
            raise ValueError(
                "The times provided to EventClock must not contain duplicates"
            )

        self.variables["t"].set_value(self.times[0])

        logger.diagnostic(f"Created event clock {self.name}")

    def advance(self):
        """
        Advance to the next time in the sequence.
        """
        new_ts = self.variables["timestep"].get_value().item() + 1
        if self._i_end is not None and new_ts > self._i_end:
            raise StopIteration("Clock has reached the end of its available times.")

        self.variables["timestep"].set_value(new_ts)
        self.variables["t"].set_value(self.times[new_ts])

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        """
        Set the start and end time of the simulation.

        Parameters
        ----------
        start : second
            The start time of the simulation
        end : second
            The end time of the simulation
        """
        start = float(start)
        end = float(end)

        start_idx = np.searchsorted(self.times, start)
        end_idx = np.searchsorted(self.times, end)

        self.variables["timestep"].set_value(start_idx)
        self.variables["t"].set_value(self.times[start_idx])

        self._i_end = end_idx - 1

    def __getitem__(self, timestep):
        """
        Get the time at a specific timestep.

        Parameters
        ----------
        timestep : int
            The timestep to get the time for

        Returns
        -------
        float
            The time at the specified timestep
        """
        return self.times[timestep]

    def same_time(self, other):
        """
        Check if two clocks are at the same time.

        For comparisons with Clock objects, uses the Clock's dt and epsilon_dt.
        For comparisons with other EventClocks or BaseClock objects, uses the base
        epsilon value.

        Parameters
        ----------
        other : BaseClock
            The other clock to compare with

        Returns
        -------
        bool
            True if both clocks are at the same time
        """
        t1 = self.variables["t"].get_value().item()
        t2 = other.variables["t"].get_value().item()

        if isinstance(other, Clock):
            return abs(t1 - t2) / other.dt_ < other.epsilon_dt
        else:
            # Both are pure EventClocks without dt.
            return abs(t1 - t2) < self.epsilon

    def __le__(self, other):
        return self.__lt__(other) or self.same_time(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.same_time(other)


class Clock(BaseClock):
    """
    An object that holds the simulation time and the time step.

    Parameters
    ----------
    dt : float
        The time step of the simulation as a float
    name : str, optional
        An explicit name, if not specified gives an automatically generated name

    Notes
    -----
    Clocks are run in the same Network.run iteration if ~Clock.t is the
    same. The condition for two
    clocks to be considered as having the same time is
    `abs(t1-t2)<epsilon*abs(t1), a standard test for equality of floating
    point values. The value of `epsilon is 1e-14.
    """

    #: The relative difference for times (in terms of dt) so that they are
    #: considered identical.
    epsilon_dt = 1e-4

    def __init__(self, dt, name="clock*"):
        super().__init__(name=name)

        self._old_dt = None

        self.variables.add_array(
            "dt",
            dimensions=second.dim,
            size=1,
            values=float(dt),
            dtype=np.float64,
            read_only=True,
            constant=True,
            scalar=True,
        )

        self.dt = dt

        logger.diagnostic(f"Created clock {self.name} with dt={self.dt}")

    def __repr__(self):
        return f"Clock(dt={self.dt!r}, name={self.name!r})"

    def advance(self):
        """
        Advance to the next time step.
        """
        new_ts = self.variables["timestep"].get_value().item() + 1
        if self._i_end is not None and new_ts > self._i_end:
            raise StopIteration("Clock has reached the end of its available times.")

        self.variables["timestep"].set_value(new_ts)
        new_t = new_ts * self.dt_
        self.variables["t"].set_value(new_t)

    def _get_dt_(self):
        return self.variables["dt"].get_value().item()

    @check_units(dt_=1)
    def _set_dt_(self, dt_):
        self._old_dt = self._get_dt_()
        self.variables["dt"].set_value(dt_)

    @check_units(dt=second)
    def _set_dt(self, dt):
        self._set_dt_(float(dt))

    dt = property(
        fget=lambda self: Quantity(self.dt_, dim=second.dim),
        fset=_set_dt,
        doc="""The time step of the simulation in seconds.""",
    )
    dt_ = property(
        fget=_get_dt_,
        fset=_set_dt_,
        doc="""The time step of the simulation as a float (in seconds)""",
    )

    def _calc_timestep(self, target_t):
        """
        Calculate the integer time step for the target time. If it cannot be
        exactly represented (up to epsilon_dt of dt), round up.

        Parameters
        ----------
        target_t : float
            The target time in seconds

        Returns
        -------
        timestep : int
            The target time in integers (based on dt)
        """
        new_i = np.int64(np.round(target_t / self.dt_))
        new_t = new_i * self.dt_
        if new_t == target_t or np.abs(new_t - target_t) / self.dt_ <= Clock.epsilon_dt:
            new_timestep = new_i
        else:
            new_timestep = np.int64(np.ceil(target_t / self.dt_))
        return new_timestep

    @check_units(target_t=second)
    def _set_t_update_dt(self, target_t=0 * second):
        """
        Set the time to a specific value, checking if dt has changed.

        Parameters
        ----------
        target_t : second
            The target time to set
        """
        new_dt = self.dt_
        old_dt = self._old_dt
        target_t = float(target_t)

        if old_dt is not None and new_dt != old_dt:
            self._old_dt = None
            check_dt(new_dt, old_dt, target_t)

        new_timestep = self._calc_timestep(target_t)

        self.variables["timestep"].set_value(new_timestep)
        self.variables["t"].set_value(new_timestep * self.dt_)
        set_t = self.variables["t"].get_value().item()

        logger.diagnostic(f"Setting Clock {self.name} to t={set_t}, dt={new_dt}")

    @check_units(start=second, end=second)
    def set_interval(self, start, end):
        """
        Set the start and end time of the simulation.

        Sets the start and end value of the clock precisely if
        possible (using epsilon_dt) or rounding up if not. This assures that
        multiple calls to `Network.run` will not re-run the same time step.

        Parameters
        ----------
        start : second
            The start time of the simulation
        end : second
            The end time of the simulation
        """
        self._set_t_update_dt(target_t=start)
        end = float(end)
        self._i_end = self._calc_timestep(end)

        if self._i_end > 2**40:
            logger.warn(
                "The end time of the simulation has been set to "
                f"{str(end*second)}, which based on the dt value of "
                f"{str(self.dt)} means that {self._i_end} "
                "time steps will be simulated. This can lead to "
                "numerical problems, e.g. the times t will not "
                "correspond to exact multiples of "
                "dt.",
                "many_timesteps",
            )

    def same_time(self, other):
        """
        Check if two clocks are at the same time (within epsilon_dt * dt).

        Parameters
        ----------
        other : BaseClock
            The other clock to compare with

        Returns
        -------
        bool
            True if both clocks are at the same time
        """
        t1 = self.variables["t"].get_value().item()
        t2 = other.variables["t"].get_value().item()

        if isinstance(other, Clock):
            # Both are pure Clocks with dt so we  take the min.
            dt = min(self.dt_, other.dt_)
            return abs(t1 - t2) / dt < self.epsilon_dt
        else:
            return abs(t1 - t2) / self.dt_ < self.epsilon_dt

    def __le__(self, other):
        return self.__lt__(other) or self.same_time(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.same_time(other)


class DefaultClockProxy:
    """
    Method proxy to access the defaultclock of the currently active device
    """

    def __getattr__(self, name):
        if name == "_is_proxy":
            return True
        from brian2.devices.device import active_device

        return getattr(active_device.defaultclock, name)

    def __setattr__(self, key, value):
        from brian2.devices.device import active_device

        setattr(active_device.defaultclock, key, value)


#: The standard clock, used for objects that do not specify any clock or dt
defaultclock = DefaultClockProxy()
