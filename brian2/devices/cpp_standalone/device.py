import numpy
import os
import inspect
from collections import defaultdict

from brian2.units import second
from brian2.core.clocks import defaultclock
from brian2.devices.device import Device, set_device, all_devices
from brian2.core.preferences import brian_prefs
from brian2.core.variables import *
from brian2.utils.filetools import copy_directory
from brian2.utils.stringtools import word_substitute
from brian2.memory.dynamicarray import DynamicArray1D
from brian2.codegen.languages.cpp_lang import c_data_type
from brian2.codegen.codeobject import CodeObjectUpdater

from .codeobject import CPPStandaloneCodeObject

__all__ = ['build']

def freeze(code, ns):
    # this is a bit of a hack, it should be passed to the template somehow
    for k, v in ns.items():
        if isinstance(v, (int, float)):
            code = word_substitute(code, {k: repr(v)})
    return code


class CPPStandaloneDevice(Device):
    '''
    '''
    def __init__(self):
        self.arrays = {}
        self.dynamic_arrays = {}
        self.code_objects = {}
        
    def array(self, owner, name, size, unit, dtype=None):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        arr = numpy.zeros(size, dtype=dtype)
        self.arrays['_array_%s_%s' % (owner.name, name)] = arr
        return arr

    def dynamic_array_1d(self, owner, name, size, unit, dtype):
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        arr = DynamicArray1D(size, dtype=dtype)
        self.dynamic_arrays['_array_%s_%s' % (owner.name, name)] = arr
        return arr
    
    def dynamic_array(self):
        raise NotImplentedError

    def code_object_class(self, codeobj_class=None):
        if codeobj_class is not None:
            raise ValueError("Cannot specify codeobj_class for C++ standalone device.")
        return CPPStandaloneCodeObject

    def code_object(self, name, abstract_code, namespace, variables, template_name,
                    indices, variable_indices, codeobj_class=None,
                    template_kwds=None):
        codeobj = super(CPPStandaloneDevice, self).code_object(name, abstract_code, namespace, variables,
                                                               template_name, indices, variable_indices,
                                                               codeobj_class=codeobj_class,
                                                               template_kwds=template_kwds,
                                                               )
        self.code_objects[codeobj.name] = codeobj
        return codeobj

    def build(self, net):
        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        updaters = []
        for obj in net.objects:
            for updater in obj.updaters:
                updaters.append(updater)
        
        # Extract the arrays information
        vars = {}
        for obj in net.objects:
            if hasattr(obj, 'variables'):
                for k, v in obj.variables.iteritems():
                    vars[(obj, k)] = v

        if not os.path.exists('output'):
            os.mkdir('output')

        # Write the arrays            
        array_specs = [(k, c_data_type(v.dtype), len(v)) for k, v in self.arrays.iteritems()]
        dynamic_array_specs = [(k, c_data_type(v.dtype)) for k, v in self.dynamic_arrays.iteritems()]
        arr_tmp = CPPStandaloneCodeObject.templater.arrays(None, array_specs=array_specs,
                                                           dynamic_array_specs=dynamic_array_specs)
        open('output/arrays.cpp', 'w').write(arr_tmp.cpp_file)
        open('output/arrays.h', 'w').write(arr_tmp.h_file)

        # Generate data for non-constant values
        code_object_defs = defaultdict(list)
        for codeobj in self.code_objects.values():
            for k, v in codeobj.nonconstant_values:
                if k=='t':
                    pass
                elif v.im_class is ArrayVariable:
                    # find the corresponding array
                    arr = v.im_self
                    arr_k = arr.name
                    arr_dtype = arr.dtype
                    arr_N = len(arr.value)
                    val = arr.value
                    if isinstance(val, int):
                        code_object_defs[id(codeobj)].append('int %s = %s;' % (k, arr_N))
                    elif k=='_spikespace':
                        code_object_defs[id(codeobj)].append('%s *%s = %s;' % (arr_dtype, k, arr_k))
                    elif isinstance(val, numpy.ndarray):
                        pass
                    else:
                        raise ValueError("Unknown")
                elif v.im_class is Variable:
                    vv = v.im_self.value
                    if vv.__class__ is DynamicArray1D:
                        print 'Found a dynamic array', k
                    else:
                        raise ValueError("Unknown")
                else:
                    raise ValueError("Unknown")
                    
        # Generate the updaters
        run_lines = []
        for updater in updaters:
            cls = updater.__class__
            if cls is CodeObjectUpdater:
                codeobj = updater.owner
                ns = codeobj.namespace
                # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant. 
                code = freeze(codeobj.code.cpp_file, ns)
                code = code.replace('%CONSTANTS%', '\n'.join(code_object_defs[id(codeobj)]))
                code = '#include "arrays.h"\n'+code
                
                open('output/'+codeobj.name+'.cpp', 'w').write(code)
                open('output/'+codeobj.name+'.h', 'w').write(codeobj.code.h_file)
                
                run_lines.append('_run_%s(t);' % codeobj.name)
            else:
                raise NotImplementedError("C++ standalone device has not implemented "+cls.__name__)
        
        # The code_objects are passed in the right order to run them because they were
        # sorted by the Network object. To support multiple clocks we'll need to be
        # smarter about that.
        main_tmp = CPPStandaloneCodeObject.templater.main(None,
                                                          run_lines=run_lines,
                                                          code_objects=self.code_objects.values(),
                                                          num_steps=1000,
                                                          dt=float(defaultclock.dt),
                                                          )
        open('output/main.cpp', 'w').write(main_tmp)

        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(CPPStandaloneCodeObject))[0],
                                    'brianlib')
        copy_directory(brianlib_dir, 'output/brianlib')


cpp_standalone_device = CPPStandaloneDevice()

all_devices['cpp_standalone'] = cpp_standalone_device

def build(net):
    cpp_standalone_device.build(net)
    
