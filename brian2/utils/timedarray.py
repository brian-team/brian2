import numpy as np

from brian2.core.functions import Function
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units, get_unit
from brian2.core.names import Nameable
from brian2.codegen.functions import (add_implementations,
                                      add_numpy_implementation)

__all__ = ['TimedArray']


class TimedArray(Function, Nameable):
    '''
    TimedArray(values, dt, name=None)

    A function of time built from an array of values. The returned object can
    be used as a function, including in model equations etc.

    Parameters
    ----------
    values : ndarray or `Quantity`
        An array of values providing the values at various points in time
    dt : `Quantity`
        The time distance between values in the `values` array.
    name : str, optional
        A unique name for this object, see `Nameable` for details. Defaults
        to ``'_timedarray*'``.

    Notes
    -----
    For time values corresponding to elements outside of the range of `values`
    provided, the first respectively last element is returned.

    Examples
    --------
    >>> from brian2 import *
    >>> ta = TimedArray([1, 2, 3, 4] * mV, dt=0.1*ms)
    >>> print(ta(0.3*ms))
    4.0 mV
    >>> G = NeuronGroup(1, 'v = ta(t) : volt')
    >>> mon = StateMonitor(G, 'v', record=True)
    >>> net = Network(G, mon)
    >>> net.run(1*ms)
    >>> print(mon[0].v)
    [ 1.  2.  3.  4.  4.  4.  4.  4.  4.  4.] mV
    '''
    @check_units(dt=second)
    def __init__(self, values, dt, name=None):
        if name is None:
            name = '_timedarray*'
        Nameable.__init__(self, name)
        unit = get_unit(values)
        values = np.asarray(values)
        self.values = values
        dt = float(dt)
        self.dt = dt

        # Python implementation
        @check_units(t=second, result=unit)
        def timed_array_func(t):
            i = np.clip(np.int_(np.float_(t) / dt + 0.5), 0, len(values)-1)
            return values[i] * unit

        Function.__init__(self, pyfunc=timed_array_func)

        # Implementation for C++
        code = {'support_code': '''
        inline double _timedarray_%NAME%(const double t, const double _dt, const int _num_values, const double* _values)
        {
            int i = (int)(t/_dt + 0.5); // rounds to nearest int for positive values
            if(i<0)
                i = 0;
            if(i>=_num_values)
                i = _num_values-1;
            return _values[i];
        }
        '''.replace('%NAME%', self.name),
                                       'hashdefine_code': '''
        #define %NAME%(t) _timedarray_%NAME%(t, _%NAME%_dt, _%NAME%_num_values, _%NAME%_values)
        '''.replace('%NAME%', self.name)}
        namespace = {'_%s_dt' % self.name: self.dt,
                     '_%s_num_values' % self.name: len(self.values),
                     '_%s_values' % self.name: self.values}

        add_implementations(self, codes={'cpp': code},
                            namespaces={'cpp': namespace},
                            names={'cpp': self.name})

        # Since the function does not internally use any units, always discard
        # the units
        add_numpy_implementation(self, timed_array_func, discard_units=True)

