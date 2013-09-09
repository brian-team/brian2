import numpy as np

from brian2.core.functions import Function, FunctionImplementation
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units, get_unit
from brian2.core.names import Nameable

__all__ = ['TimedArray']


class TimedArray(Function, Nameable):

    @check_units(dt=second)
    def __init__(self, values, dt, name=None):
        if name is None:
            name = '_timedarray*'
        Nameable.__init__(self, name)
        self.unit = get_unit(values)
        values = np.asarray(values)
        self.values = values
        dt = float(dt)
        self.dt = dt

        # Python implementation
        def timed_array_func(t):
            i = np.clip(np.int_(np.float_(t) / dt + 0.5), 0, len(values)-1)
            return values[i]

        Function.__init__(self, pyfunc=timed_array_func,
                          arg_units=[second],
                          return_unit=self.unit)

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
        self.implementations['cpp'] = FunctionImplementation(self.name,
                                                             code=code,
                                                             namespace=namespace)
