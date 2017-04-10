'''
Module providing `NumpyCodeObject`.
'''
import sys
from collections import Iterable

import numpy as np

from brian2.core.base import brian_object_exception
from brian2.core.preferences import prefs, BrianPreference
from brian2.core.variables import (DynamicArrayVariable, ArrayVariable,
                                   AuxiliaryVariable, Subexpression)
from brian2.core.functions import Function

from ...codeobject import CodeObject, constant_or_scalar

from ...templates import Templater
from ...generators.numpy_generator import NumpyCodeGenerator
from ...targets import codegen_targets

__all__ = ['NumpyCodeObject']

# Preferences
prefs.register_preferences(
    'codegen.runtime.numpy',
    'Numpy runtime codegen preferences',
    discard_units = BrianPreference(
        default=False,
        docs='''
        Whether to change the namespace of user-specifed functions to remove
        units.
        '''
        )
    )


class LazyArange(Iterable):
    '''
    A class that can be used as a `~numpy.arange` replacement (with an implied
    step size of 1) but does not actually create an array of values until
    necessary. It is somewhat similar to the ``range()`` function in Python 3,
    but does not use a generator. It is tailored to a special use case, the
    ``_vectorisation_idx`` variable in numpy templates, and not meant for
    general use. The ``_vectorisation_idx`` is used for stateless function
    calls such as ``rand()`` and for the numpy codegen target determines the
    number of values produced by such a call. This will often be the number of
    neurons or synapses, and this class avoids creating a new array of that size
    at every code object call when all that is needed is the *length* of the
    array.

    Examples
    --------
    >>> from brian2.codegen.runtime.numpy_rt.numpy_rt import LazyArange
    >>> ar = LazyArange(10)
    >>> len(ar)
    10
    >>> len(ar[:5])
    5
    >>> type(ar[:5])
    <class 'brian2.codegen.runtime.numpy_rt.numpy_rt.LazyArange'>
    >>> ar[5]
    5
    >>> for value in ar[3:7]:
    ...     print(value)
    ...
    3
    4
    5
    6
    >>> len(ar[np.array([1, 2, 3])])
    3
    '''
    def __init__(self, stop, start=0, indices=None):
        self.start = start
        self.stop = stop
        self.indices = indices

    def __len__(self):
        if self.indices is None:
            return self.stop - self.start
        else:
            return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if self.indices is None:
                start, stop, step = item.start, item.stop, item.step
                if step not in [None, 1]:
                    raise NotImplementedError('Step should be 1')
                if start is None:
                    start = 0
                if stop is None:
                    stop = len(self)
                return LazyArange(start=self.start+start,
                                  stop=min([self.start+stop, self.stop]))
            else:
                raise NotImplementedError('Cannot slice LazyArange with indices')
        elif isinstance(item, np.ndarray):
            if item.dtype == np.dtype(bool):
                item = np.nonzero(item)[0]  # convert boolean array into integers
            if len(item) == 0:
                return np.array([], dtype=np.int32)
            if np.min(item) < 0 or np.max(item) > len(self):
                raise IndexError('Indexing array contains out-of-bounds values')
            return LazyArange(start=self.start, stop=self.stop, indices=item)
        elif isinstance(item, int):
            if self.indices is None:
                index = self.start + item
                if index >= self.stop:
                    raise IndexError(index)
                return index
            else:
                return self.indices[item]
        else:
            raise TypeError('Can only index with integer, numpy array, or slice.')

    def __iter__(self):
        if self.indices is None:
            return iter(np.arange(self.start, self.stop))
        else:
            return iter(self.indices)

    # Allow conversion to a proper array with np.array(...)
    def __array__(self, dtype=None):
        if self.indices is None:
            return np.arange(self.start, self.stop)
        else:
            return self.indices + self.start

    # Allow basic arithmetics (used when shifting stuff for subgroups)
    def __add__(self, other):
        if isinstance(other, int):
            return LazyArange(start=self.start + other, stop=self.stop + other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return LazyArange(start=self.start - other, stop=self.stop - other)
        else:
            return NotImplemented


class NumpyCodeObject(CodeObject):
    '''
    Execute code using Numpy
    
    Default for Brian because it works on all platforms.
    '''
    templater = Templater('brian2.codegen.runtime.numpy_rt', '.py_',
                          env_globals={'constant_or_scalar': constant_or_scalar})
    generator_class = NumpyCodeGenerator
    class_name = 'numpy'

    def __init__(self, owner, code, variables, variable_indices,
                 template_name, template_source, name='numpy_code_object*'):
        from brian2.devices.device import get_device
        self.device = get_device()
        self.namespace = {'_owner': owner,
                          # TODO: This should maybe go somewhere else
                          'logical_not': np.logical_not}
        CodeObject.__init__(self, owner, code, variables, variable_indices,
                            template_name, template_source, name=name)
        self.variables_to_namespace()

    @classmethod
    def is_available(cls):
        # no test necessary for numpy
        return True

    def variables_to_namespace(self):
        # Variables can refer to values that are either constant (e.g. dt)
        # or change every timestep (e.g. t). We add the values of the
        # constant variables here and add the names of non-constant variables
        # to a list

        # A list containing tuples of name and a function giving the value
        self.nonconstant_values = []

        for name, var in self.variables.iteritems():
            if isinstance(var, (AuxiliaryVariable, Subexpression)):
                continue

            try:
                if not hasattr(var, 'get_value'):
                    raise TypeError()
                value = var.get_value()
            except TypeError:
                # Either a dummy Variable without a value or a Function object
                if isinstance(var, Function):
                    impl = var.implementations[self.__class__].get_code(self.owner)
                    self.namespace[name] = impl
                else:
                    self.namespace[name] = var
                continue

            if isinstance(var, ArrayVariable):
                self.namespace[self.generator_class.get_array_name(var)] = value
                if var.scalar and var.constant:
                    self.namespace[name] = value[0]
            else:
                self.namespace[name] = value

            if isinstance(var, DynamicArrayVariable):
                dyn_array_name = self.generator_class.get_array_name(var,
                                                                    access_data=False)
                self.namespace[dyn_array_name] = self.device.get_value(var,
                                                                       access_data=False)

            # Also provide the Variable object itself in the namespace (can be
            # necessary for resize operations, for example)
            self.namespace['_var_'+name] = var

            # There is one type of objects that we have to inject into the
            # namespace with their current value at each time step: dynamic
            # arrays that change in size during runs (i.e. not synapses but
            # e.g. the structures used in monitors)
            if (isinstance(var, DynamicArrayVariable) and
                    var.needs_reference_update):
                self.nonconstant_values.append((self.generator_class.get_array_name(var,
                                                                                   self.variables),
                                                var.get_value))

    def update_namespace(self):
        # update the values of the non-constant values in the namespace
        for name, func in self.nonconstant_values:
            self.namespace[name] = func()

    def compile(self):
        super(NumpyCodeObject, self).compile()
        self.compiled_code = compile(self.code, '(string)', 'exec')

    def run(self):
        try:
            exec self.compiled_code in self.namespace
        except Exception as exc:
            message = ('An exception occured during the execution of code '
                       'object {}.\n').format(self.name)
            lines = self.code.split('\n')
            message += 'The error was raised in the following line:\n'
            _, _, tb = sys.exc_info()
            tb = tb.tb_next  # Line in the code object's code
            message += lines[tb.tb_lineno - 1] + '\n'
            raise brian_object_exception(message, self.owner, exc)
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']

codegen_targets.add(NumpyCodeObject)
