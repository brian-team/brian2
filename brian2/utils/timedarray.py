import numpy as np

from brian2.core.functions import Function
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units, get_unit
from brian2.core.names import Nameable
from brian2.utils.logger import get_logger

__all__ = ['TimedArray']


logger = get_logger(__name__)


def _find_K(group_dt, dt):
    dt_ratio = dt / group_dt
    if dt_ratio > 1 and np.floor(dt_ratio) != dt_ratio:
        logger.warn(('Group uses a dt of %s while TimedArray uses dt '
                     'of %s') % (group_dt*second, dt*second), once=True)
    # Find an upsampling factor that should avoid rounding issues even
    # for multistep methods
    K = int(2**np.ceil(np.log2(8/group_dt*dt)))
    return K


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

        # Python implementation (with units), used when calling the TimedArray
        # directly, outside of a simulation
        @check_units(t=second, result=unit)
        def timed_array_func(t):
            i = np.clip(np.int_(np.float_(t) / dt + 0.5), 0, len(values)-1)
            return values[i] * unit

        Function.__init__(self, pyfunc=timed_array_func)

        # we use dynamic implementations because we want to do upsampling
        # in a way that avoids rounding problems with the group's dt
        def create_numpy_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            epsilon = dt / K
            n_values = len(values)
            def unitless_timed_array_func(t):
                timestep = np.clip(np.int_(np.round(t/epsilon)) / K, 0, n_values-1)
                return values[timestep]
            unitless_timed_array_func._arg_units = [second]
            unitless_timed_array_func._return_unit = unit

            return unitless_timed_array_func

        self.implementations.add_dynamic_implementation('numpy',
                                                        create_numpy_implementation)

        def create_cpp_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            epsilon = dt / K
            cpp_code = {'support_code': '''
            inline double _timedarray_%NAME%(const double t, const int _num_values, const double* _values)
            {
                int i = (int)(t/%EPSILON% + 0.5)/%K%; // rounds to nearest int for positive values
                if(i<0)
                    i = 0;
                if(i>=_num_values)
                    i = _num_values-1;
                return _values[i];
            }
            '''.replace('%NAME%', self.name).replace('%EPSILON%', '%.12f' %epsilon).replace('%K%', str(K)),
                                           'hashdefine_code': '''
            #define %NAME%(t) _timedarray_%NAME%(t, _%NAME%_num_values, _%NAME%_values)
            '''.replace('%NAME%', self.name)}

            return cpp_code

        def create_cpp_namespace(owner):
            return {'_%s_num_values' % self.name: len(self.values),
                    '_%s_values' % self.name: self.values}

        self.implementations.add_dynamic_implementation('cpp',
                                                        create_cpp_implementation,
                                                        create_cpp_namespace,
                                                        name=self.name)

    def is_locally_constant(self, dt):
        if dt > self.dt:
            return False
        dt_ratio = self.dt / float(dt)
        if np.floor(dt_ratio) != dt_ratio:
            logger.warn(("dt of the TimedArray is not an integer multiple of "
                         "the group's dt, the TimedArray's return value can "
                         "therefore not be considered constant over one "
                         "timestep, making linear integration impossible."),
                        once=True)
            return False
        return True