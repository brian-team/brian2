import numpy as np

from brian2.core.functions import Function
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units, get_unit

__all__ = ['TimedArray']


class TimedArray(Function):

    @check_units(dt=second)
    def __init__(self, name, values, dt):
        self.name = name
        self.unit = get_unit(values)
        values = np.asarray(values)
        self.values = values
        dt = float(dt)
        self.dt = dt

        def timed_array_func(t):
            i = np.clip(int(float(t) / dt + 0.5), 0, len(values)-1)
            return values[i]

        Function.__init__(self, timed_array_func, name=self.name,
                          arg_units=[second],
                          return_unit=self.unit)


