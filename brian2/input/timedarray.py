'''
Implementation of `TimedArray`.
'''

import numpy as np

from brian2.core.clocks import defaultclock
from brian2.core.functions import Function
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units, get_unit
from brian2.core.names import Nameable
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import replace


__all__ = ['TimedArray']


logger = get_logger(__name__)


def _find_K(group_dt, dt):
    dt_ratio = dt / group_dt
    if dt_ratio > 1 and np.floor(dt_ratio) != dt_ratio:
        logger.warn(('Group uses a dt of %s while TimedArray uses dt '
                     'of %s') % (group_dt*second, dt*second), once=True)
    # Find an upsampling factor that should avoid rounding issues even
    # for multistep methods
    K = max(int(2**np.ceil(np.log2(8/group_dt*dt))), 1)
    return K


class TimedArray(Function, Nameable):
    '''
    TimedArray(values, dt, name=None)

    A function of time built from an array of values. The returned object can
    be used as a function, including in model equations etc. The resulting
    function has to be called as `funcion_name(t)` if the provided value array
    is one-dimensional and as `function_name(t, i)` if it is two-dimensional.

    Parameters
    ----------
    values : ndarray or `Quantity`
        An array of values providing the values at various points in time. This
        array can either be one- or two-dimensional. If it is two-dimensional
        it's first dimension should be the time.
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
    4. mV
    >>> G = NeuronGroup(1, 'v = ta(t) : volt')
    >>> mon = StateMonitor(G, 'v', record=True)
    >>> net = Network(G, mon)
    >>> net.run(1*ms)  # doctest: +ELLIPSIS
    ...
    >>> print(mon[0].v)
    [ 1.  2.  3.  4.  4.  4.  4.  4.  4.  4.] mV
    >>> ta2d = TimedArray([[1, 2], [3, 4], [5, 6]]*mV, dt=0.1*ms)
    >>> G = NeuronGroup(4, 'v = ta2d(t, i%2) : volt')
    >>> mon = StateMonitor(G, 'v', record=True)
    >>> net = Network(G, mon)
    >>> net.run(0.2*ms)  # doctest: +ELLIPSIS
    ...
    >>> print mon.v[:]
    [[ 1.  3.]
     [ 2.  4.]
     [ 1.  3.]
     [ 2.  4.]] mV
    '''
    @check_units(dt=second)
    def __init__(self, values, dt, name=None):
        if name is None:
            name = '_timedarray*'
        Nameable.__init__(self, name)
        unit = get_unit(values)
        self.unit = unit
        values = np.asarray(values, dtype=np.double)
        self.values = values
        dt = float(dt)
        self.dt = dt
        if values.ndim == 1:
            self._init_1d()
        elif values.ndim == 2:
            self._init_2d()
        else:
            raise NotImplementedError(('Only 1d and 2d arrays are supported '
                                       'for TimedArray'))

    def _init_1d(self):
        unit = self.unit
        values = self.values
        dt = self.dt

        # Python implementation (with units), used when calling the TimedArray
        # directly, outside of a simulation
        @check_units(t=second, result=unit)
        def timed_array_func(t):
            # We round according to the current defaultclock.dt
            K = _find_K(float(defaultclock.dt), dt)
            epsilon = dt / K
            i = np.clip(np.int_(np.round(np.asarray(t/epsilon)) / K),
                        0, len(values)-1)
            return values[i] * unit

        Function.__init__(self, pyfunc=timed_array_func)

        # we use dynamic implementations because we want to do upsampling
        # in a way that avoids rounding problems with the group's dt
        def create_numpy_implementation(owner):
            group_dt = owner.clock.dt_

            K = _find_K(group_dt, dt)
            n_values = len(values)
            epsilon = dt / K
            def unitless_timed_array_func(t):
                timestep = np.clip(np.int_(np.round(t/epsilon) / K),
                                   0, n_values-1)
                return values[timestep]

            unitless_timed_array_func._arg_units = [second]
            unitless_timed_array_func._return_unit = unit

            return unitless_timed_array_func

        self.implementations.add_dynamic_implementation('numpy',
                                                        create_numpy_implementation)

        def create_cpp_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            support_code = '''
            inline double %NAME%(const double t)
            {
                const double epsilon = %DT% / %K%;
                int i = (int)((t/epsilon + 0.5)/%K%);
                if(i < 0)
                   i = 0;
                if(i >= %NUM_VALUES%)
                    i = %NUM_VALUES%-1;
                return _namespace%NAME%_values[i];
            }
            '''.replace('%NAME%', self.name).replace('%DT%', '%.18f' % dt).replace('%K%', str(K)).replace('%NUM_VALUES%', str(len(self.values)))
            cpp_code = {'support_code': support_code}

            return cpp_code

        def create_cpp_namespace(owner):
            return {'%s_values' % self.name: self.values}

        self.implementations.add_dynamic_implementation('cpp',
                                                        code=create_cpp_implementation,
                                                        namespace=create_cpp_namespace,
                                                        name=self.name)
        def create_cython_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            code = '''
            cdef double %NAME%(const double t):
                global _namespace%NAME%_values
                cdef double epsilon = %DT% / %K%
                cdef int i = (int)((t/epsilon + 0.5)/%K%)
                if i < 0:
                   i = 0
                if i >= %NUM_VALUES%:
                    i = %NUM_VALUES% - 1
                return _namespace%NAME%_values[i]
            '''.replace('%NAME%', self.name).replace('%DT%', '%.18f' % dt).replace('%K%', str(K)).replace('%NUM_VALUES%', str(len(self.values)))

            return code

        def create_cython_namespace(owner):
            return {'%s_values' % self.name: self.values}

        self.implementations.add_dynamic_implementation('cython',
                                                        code=create_cython_implementation,
                                                        namespace=create_cython_namespace,
                                                        name=self.name)


    def _init_2d(self):
        unit = self.unit
        values = self.values
        dt = self.dt

        # Python implementation (with units), used when calling the TimedArray
        # directly, outside of a simulation
        @check_units(i=1, t=second, result=unit)
        def timed_array_func(t, i):
            # We round according to the current defaultclock.dt
            K = _find_K(float(defaultclock.dt), dt)
            epsilon = dt / K
            time_step = np.clip(np.int_(np.round(np.asarray(t/epsilon)) / K),
                        0, len(values)-1)
            return values[time_step, i] * unit

        Function.__init__(self, pyfunc=timed_array_func)

        # we use dynamic implementations because we want to do upsampling
        # in a way that avoids rounding problems with the group's dt
        def create_numpy_implementation(owner):
            group_dt = owner.clock.dt_

            K = _find_K(group_dt, dt)
            n_values = len(values)
            epsilon = dt / K
            def unitless_timed_array_func(t, i):
                timestep = np.clip(np.int_(np.round(t/epsilon) / K),
                                   0, n_values-1)
                return values[timestep, i]

            unitless_timed_array_func._arg_units = [second]
            unitless_timed_array_func._return_unit = unit

            return unitless_timed_array_func

        self.implementations.add_dynamic_implementation('numpy',
                                                        create_numpy_implementation)


        def create_cpp_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            support_code = '''
            inline double %NAME%(const double t, const int i)
            {
                const double epsilon = %DT% / %K%;
                if (i < 0 || i >= %COLS%)
                    return NAN;
                int timestep = (int)((t/epsilon + 0.5)/%K%);
                if(timestep < 0)
                   timestep = 0;
                else if(timestep >= %ROWS%)
                    timestep = %ROWS%-1;
                return _namespace%NAME%_values[timestep*%COLS% + i];
            }
            '''
            support_code = replace(support_code, {'%NAME%': self.name,
                                                  '%DT%': '%.18f' % dt,
                                                  '%K%': str(K),
                                                  '%COLS%': str(self.values.shape[1]),
                                                  '%ROWS%': str(self.values.shape[0])})
            cpp_code = {'support_code': support_code}

            return cpp_code

        def create_cpp_namespace(owner):
            return {'%s_values' % self.name: self.values.astype(np.double,
                                                                order='C',
                                                                copy=False).ravel()}

        self.implementations.add_dynamic_implementation('cpp',
                                                        code=create_cpp_implementation,
                                                        namespace=create_cpp_namespace,
                                                        name=self.name)

        def create_cython_implementation(owner):
            group_dt = owner.clock.dt_
            K = _find_K(group_dt, dt)
            code = '''
            cdef double %NAME%(const double t, const int i):
                global _namespace%NAME%_values
                cdef double epsilon = %DT% / %K%;
                if i < 0 or i >= %COLS%:
                    return _numpy.nan
                cdef int timestep = (int)((t/epsilon + 0.5)/%K%)
                if timestep < 0:
                   timestep = 0
                elif timestep >= %ROWS%:
                    timestep = %ROWS%-1
                return _namespace%NAME%_values[timestep*%COLS% + i]
            '''
            code = replace(code, {'%NAME%': self.name,
                                  '%DT%': '%.18f' % dt,
                                  '%K%': str(K),
                                  '%COLS%': str(self.values.shape[1]),
                                  '%ROWS%': str(self.values.shape[0])})

            return code

        def create_cython_namespace(owner):
            return {'%s_values' % self.name: self.values.astype(np.double,
                                                                order='C',
                                                                copy=False).ravel()}

        self.implementations.add_dynamic_implementation('cython',
                                                        code=create_cython_implementation,
                                                        namespace=create_cython_namespace,
                                                        name=self.name)


    def is_locally_constant(self, dt):
        if dt > self.dt:
            return False
        dt_ratio = self.dt / float(dt)
        if np.floor(dt_ratio) != dt_ratio:
            logger.info(("dt of the TimedArray is not an integer multiple of "
                         "the group's dt, the TimedArray's return value can "
                         "therefore not be considered constant over one "
                         "timestep, making linear integration impossible."),
                        once=True)
            return False
        return True