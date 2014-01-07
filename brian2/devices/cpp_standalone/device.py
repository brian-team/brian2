'''
Module implementing the C++ "standalone" device.
'''
import numpy
import os
import shutil
import subprocess
import inspect
from collections import defaultdict

from brian2.core.clocks import defaultclock
from brian2.core.magic import magic_network
from brian2.core.network import Network as OrigNetwork
from brian2.core.namespace import get_local_namespace
from brian2.devices.device import Device, all_devices
from brian2.core.preferences import brian_prefs
from brian2.core.variables import *
from brian2.synapses.synapses import Synapses
from brian2.utils.filetools import copy_directory, ensure_directory, in_directory
from brian2.utils.stringtools import word_substitute
from brian2.codegen.languages.cpp_lang import c_data_type
from brian2.codegen.codeobject import CodeObjectUpdater
from brian2.units.fundamentalunits import (Quantity, Unit, is_scalar_type,
                                           fail_for_dimension_mismatch,
                                           have_same_dimensions,
                                           )
from brian2.units import second
from brian2.utils.logger import get_logger

from .codeobject import CPPStandaloneCodeObject


__all__ = ['build', 'Network', 'run', 'stop',
           'insert_code_into_main',
           ]

logger = get_logger(__name__)

def freeze(code, ns):
    # this is a bit of a hack, it should be passed to the template somehow
    for k, v in ns.items():
        if isinstance(v, (int, float)):
            code = word_substitute(code, {k: repr(v)})
    return code

class StandaloneVariableView(VariableView):
    '''
    Will store information about how the variable was set in the original
    `ArrayVariable` object.
    '''
    def __init__(self, name, variable, group, unit=None, level=0):
        super(StandaloneVariableView, self).__init__(name, variable, group,
                                                     unit=unit, level=level)

    # Overwrite methods to signal that they are not available for standalone
    def set_array_with_array_index(self, item, value):
        if isinstance(item, list):
            item = numpy.array(item)
        if isinstance(value, list):
            value = numpy.array(value)
        if isinstance(item, slice) and item==slice(None) and isinstance(value, numpy.ndarray):
            arrayname = self.group.variables[self.name].arrayname
            staticarrayname = cpp_standalone_device.static_array(arrayname, value)
            cpp_standalone_device.main_queue.append(('set_by_array', (arrayname, staticarrayname, item, value)))
        elif isinstance(item, numpy.ndarray) and isinstance(value, numpy.ndarray):
            arrayname = self.group.variables[self.name].arrayname
            staticarrayname_index = cpp_standalone_device.static_array('_index_'+arrayname, item)
            staticarrayname_value = cpp_standalone_device.static_array('_value_'+arrayname, value)
            cpp_standalone_device.main_queue.append(('set_array_by_array', (arrayname, staticarrayname_index,
                                                                            staticarrayname_value, item, value)))
        else:
            raise NotImplementedError(('Cannot set variable "%s" this way in '
                                       'standalone, try using string '
                                       'expressions.') % self.name)

    def __getitem__(self, item):
        raise NotImplementedError()


class StandaloneArrayVariable(ArrayVariable):

    def __init__(self, name, unit, size, dtype, group_name=None, constant=False,
                 is_bool=False):
        self.assignments = []
        self.size = size
        super(StandaloneArrayVariable, self).__init__(name, unit, value=None,
                                                      group_name=group_name,
                                                      constant=constant,
                                                      is_bool=is_bool)
        self.dtype = dtype

    def get_len(self):
        return self.size

    def get_value(self):
        raise NotImplementedError('Getting value for variable %s is not '
                                  'supported in standalone.' % self.name)

    def set_value(self, value, index=None):
        if index is None:
            index = slice(None)
        self.assignments.append((index, value))

    def get_addressable_value(self, name, group, level=0):
        return StandaloneVariableView(name, self, group, unit=None,
                                      level=level+1)

    def get_addressable_value_with_unit(self, name, group, level=0):
        return StandaloneVariableView(name, self, group, unit=self.unit,
                                      level=level+1)


class StandaloneDynamicArrayVariable(StandaloneArrayVariable):

    def __init__(self, name, unit, dimensions,
                 size, dtype, group_name=None, constant=False, is_bool=False):
        self.dimensions = dimensions
        super(StandaloneDynamicArrayVariable, self).__init__(name, unit,
                                                             size=size,
                                                             dtype=dtype,
                                                             group_name=group_name,
                                                             constant=constant,
                                                             is_bool=is_bool)
    def resize(self, new_size):
        self.assignments.append(('resize', new_size))

        
class CPPStandaloneDevice(Device):
    '''
    The `Device` used for C++ standalone simulations.
    '''
    def __init__(self):
        #: List of all regular arrays with their type and size
        self.array_specs = []
        #: List of all dynamic arrays with their type
        self.dynamic_array_specs = []
        #: List of all 2d dynamic arrays with their type
        self.dynamic_array_2d_specs = []
        #: List of all arrays to be filled with zeros (with type and size)
        self.zero_specs = []
        #: List of all arrays to be filled with numbers (with type, size,
        #: start, and stop)
        self.arange_specs = []
        
        #: Dict of all static saved arrays
        self.static_arrays = {}
        
        self.code_objects = {}
        self.main_queue = []
        
        self.synapses = []
        
        self.clocks = set([])
        
    def reinit(self):
        self.__init__()

    def insert_device_code(self, slot, code):
        '''
        Insert code directly into main.cpp
        '''
        if slot=='main.cpp':
            self.main_queue.append(('insert_code', code))
        else:
            logger.warn("Ignoring device code, unknown slot: %s, code: %s" % (slot, code))
            
    def static_array(self, name, arr):
        name = '_static_array_' + name
        basename = name
        i = 0
        while name in self.static_arrays:
            i += 1
            name = basename+'_'+str(i)
        self.static_arrays[name] = arr.copy()
        return name
        
    def array(self, owner, name, size, unit, value=None, dtype=None, constant=False,
              is_bool=False, read_only=False):
        if is_bool:
            dtype = numpy.bool
        elif value is not None:
            dtype = value.dtype
        elif dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        self.array_specs.append(('_array_%s_%s' % (owner.name, name),
                                 c_data_type(dtype), size))
        self.zero_specs.append(('_array_%s_%s' % (owner.name, name),
                                c_data_type(dtype), size))

        var = StandaloneArrayVariable(name, unit, size=size, dtype=dtype,
                                      group_name=owner.name,
                                      constant=constant, is_bool=is_bool)

        if value is not None:
            arrayname = var.arrayname
            staticarrayname = self.static_array(arrayname, value)
            self.main_queue.append(('set_by_array', (arrayname,
                                                     staticarrayname,
                                                     slice(None),
                                                     value)))
        return var


    def arange(self, owner, name, size, start=0, dtype=numpy.int32, constant=True,
               read_only=True):
        self.array_specs.append(('_array_%s_%s' % (owner.name, name),
                                 c_data_type(dtype), size))
        self.arange_specs.append(('_array_%s_%s' % (owner.name, name),
                                  c_data_type(dtype), start, size))
        return StandaloneArrayVariable(name, Unit(1), size=size, dtype=dtype,
                                       group_name=owner.name,
                                       constant=constant, is_bool=False)

    def dynamic_array_1d(self, owner, name, size, unit, dtype=None,
                         constant=False, constant_size=True, is_bool=False,
                         read_only=False):
        if is_bool:
            dtype = numpy.bool
        elif dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        self.dynamic_array_specs.append(('_dynamic_array_%s_%s' % (owner.name, name),
                                         c_data_type(dtype)))
        return StandaloneDynamicArrayVariable(name, unit, dimensions=1,
                                              size=size,
                                              dtype=dtype,
                                              group_name=owner.name,
                                              constant=constant, is_bool=is_bool)
        
    def dynamic_array(self, owner, name, size, unit, dtype=None,
                      constant=False, constant_size=True, is_bool=False,
                      read_only=False):
        if isinstance(size, int):
            return self.dynamic_array_1d(owner, name, size, unit, dtype=dtype, constant=constant,
                                         constant_size=constant_size, is_bool=is_bool, read_only=read_only)
        if not isinstance(size, tuple) or not len(size)==2:
            raise NotImplementedError("Only 1D and 2D dynamic arrays are implemented for C++ standalone.")
        if is_bool:
            dtype = numpy.bool
        if dtype is None:
            dtype = brian_prefs['core.default_scalar_dtype']
        self.dynamic_array_2d_specs.append(('_dynamic_2d_array_%s_%s' % (owner.name, name),
                                            c_data_type(dtype)))
        return StandaloneDynamicArrayVariable(name, unit, dimensions=2,
                                              size=size,
                                              dtype=dtype,
                                              group_name=owner.name,
                                              constant=constant, is_bool=is_bool)

    def code_object_class(self, codeobj_class=None):
        if codeobj_class is not None:
            raise ValueError("Cannot specify codeobj_class for C++ standalone device.")
        return CPPStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, namespace, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None):
        codeobj = super(CPPStandaloneDevice, self).code_object(owner, name, abstract_code, namespace, variables,
                                                               template_name, variable_indices,
                                                               codeobj_class=codeobj_class,
                                                               template_kwds=template_kwds,
                                                               )
        self.code_objects[codeobj.name] = codeobj
        return codeobj

    def build(self, project_dir='output', compile_project=True, run_project=False, debug=True,
              with_output=True):
        ensure_directory(project_dir)
        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(project_dir, d))
            
        logger.debug("Writing C++ standalone project to directory "+os.path.normpath(project_dir))

        # Find numpy arrays in the namespaces and convert them into static
        # arrays. Hopefully they are correctly used in the code: For example,
        # this works for the namespaces for functions with C++ (e.g. TimedArray
        # treats it as a C array) but does not work in places that are
        # implicitly vectorized (state updaters, resets, etc.). But arrays
        # shouldn't be used there anyway.
        for code_object in self.code_objects.itervalues():
            for name, value in code_object.namespace.iteritems():
                if isinstance(value, numpy.ndarray):
                    self.static_arrays[name] = value

        # write the static arrays
        logger.debug("static arrays: "+str(sorted(self.static_arrays.keys())))
        static_array_specs = []
        for name, arr in self.static_arrays.iteritems():
            arr.tofile(os.path.join(project_dir, 'static_arrays', name))
            static_array_specs.append((name, c_data_type(arr.dtype), arr.size, name))

        # Write the global objects
        networks = [net() for net in Network.__instances__() if net().name!='_fake_network']
        synapses = [S() for S in Synapses.__instances__()]
        arr_tmp = CPPStandaloneCodeObject.templater.objects(None,
                                                            array_specs=self.array_specs,
                                                            dynamic_array_specs=self.dynamic_array_specs,
                                                            dynamic_array_2d_specs=self.dynamic_array_2d_specs,
                                                            zero_specs=self.zero_specs,
                                                            arange_specs=self.arange_specs,
                                                            synapses=synapses,
                                                            clocks=self.clocks,
                                                            static_array_specs=static_array_specs,
                                                            networks=networks,
                                                            )
        logger.debug("objects: "+str(arr_tmp))
        open(os.path.join(project_dir, 'objects.cpp'), 'w').write(arr_tmp.cpp_file)
        open(os.path.join(project_dir, 'objects.h'), 'w').write(arr_tmp.h_file)

        main_lines = []
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
                main_lines.append('_run_%s(t);' % codeobj.name)
            elif func=='run_network':
                net, netcode = args
                main_lines.extend(netcode)
            elif func=='set_by_array':
                arrayname, staticarrayname, item, value = args
                code = '''
                for(int i=0; i<_num_{staticarrayname}; i++)
                {{
                    {arrayname}[i] = {staticarrayname}[i];
                }}
                '''.format(arrayname=arrayname, staticarrayname=staticarrayname)
                main_lines.extend(code.split('\n'))
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value, item, value = args
                code = '''
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                '''.format(arrayname=arrayname, staticarrayname_index=staticarrayname_index,
                           staticarrayname_value=staticarrayname_value)
                main_lines.extend(code.split('\n'))
            elif func=='insert_code':
                main_lines.append(args)
            else:
                raise NotImplementedError("Unknown main queue function type "+func)

        # generate the finalisations
        for codeobj in self.code_objects.itervalues():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise);

        # Generate data for non-constant values
        code_object_defs = defaultdict(list)
        already_deffed = defaultdict(set)
        for codeobj in self.code_objects.itervalues():
            for k, v in codeobj.variables.iteritems():
                if k=='t':
                    pass
                elif isinstance(v, Subexpression):
                    pass
                elif isinstance(v, AttributeVariable):
                    c_type = c_data_type(v.dtype)
                    # TODO: Handle dt in the correct way
                    if v.attribute == 'dt_':
                        code = ('const {c_type} {k} = '
                                '{value};').format(c_type=c_type,
                                                  k=k,
                                                  value=v.get_value())
                    else:
                        code = ('const {c_type} {k} = '
                                '{name}.{attribute};').format(c_type=c_type,
                                                             k=k,
                                                             name=v.obj.name,
                                                             attribute=v.attribute)
                    code_object_defs[codeobj.name].append(code)
                elif not v.scalar:
                    if hasattr(v, 'arrayname') and v.arrayname in already_deffed[codeobj.name]:
                        continue
                    try:
                        if isinstance(v, StandaloneDynamicArrayVariable) and v.dimensions==1:
                            code_object_defs[codeobj.name].append('const int _num{k} = _dynamic{arrayname}.size();'.format(k=k, arrayname=v.arrayname))
                            c_type = c_data_type(v.dtype)

                            # Create an alias name for the underlying array
                            code = ('{c_type}* {arrayname} = '
                                    '&(_dynamic{arrayname}[0]);').format(c_type=c_type,
                                                                          arrayname=v.arrayname)
                            code_object_defs[codeobj.name].append(code)
                            already_deffed[codeobj.name].add(v.arrayname)
                        else:
                            N = v.get_len()#len(v)
                            code_object_defs[codeobj.name].append('const int _num%s = %s;' % (k, N))
                    except TypeError:
                        pass

        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.namespace
            # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant.
            code = freeze(codeobj.code.cpp_file, ns)
            code = code.replace('%CONSTANTS%', '\n'.join(code_object_defs[codeobj.name]))
            code = '#include "objects.h"\n'+code
            
            open(os.path.join(project_dir, 'code_objects', codeobj.name+'.cpp'), 'w').write(code)
            open(os.path.join(project_dir, 'code_objects', codeobj.name+'.h'), 'w').write(codeobj.code.h_file)
                    
        # The code_objects are passed in the right order to run them because they were
        # sorted by the Network object. To support multiple clocks we'll need to be
        # smarter about that.
        main_tmp = CPPStandaloneCodeObject.templater.main(None,
                                                          main_lines=main_lines,
                                                          code_objects=self.code_objects.values(),
                                                          dt=float(defaultclock.dt),
                                                          )
        logger.debug("main: "+str(main_tmp))
        open(os.path.join(project_dir, 'main.cpp'), 'w').write(main_tmp)

        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(CPPStandaloneCodeObject))[0],
                                    'brianlib')
        copy_directory(brianlib_dir, os.path.join(project_dir, 'brianlib'))

        # Copy the CSpikeQueue implementation
        shutil.copy(os.path.join(os.path.split(inspect.getsourcefile(Synapses))[0],
                                    'cspikequeue.cpp'),
                    os.path.join(project_dir, 'brianlib', 'spikequeue.h'))

        # build the project
        if compile_project:
            with in_directory(project_dir):
                if debug:
                    x = os.system('g++ -I. -g *.cpp code_objects/*.cpp brianlib/*.cpp -o main')
                else:
                    x = os.system('g++ -I. -O3 -ffast-math -march=native *.cpp code_objects/*.cpp brianlib/*.cpp -o main')
                if x==0:
                    if run_project:
                        if not with_output:
                            stdout = open(os.devnull, 'w')
                        else:
                            stdout = None
                        if os.name=='nt':
                            x = subprocess.call('main', stdout=stdout)
                        else:
                            x = subprocess.call('./main', stdout=stdout)
                        if x:
                            raise RuntimeError("Project run failed")
                else:
                    raise RuntimeError("Project compilation failed")


cpp_standalone_device = CPPStandaloneDevice()

all_devices['cpp_standalone'] = cpp_standalone_device

build = cpp_standalone_device.build

    
class Network(OrigNetwork):
    def run(self, duration, report=None, report_period=60*second,
            namespace=None, level=0, _magic_network=None):
        
        if _magic_network is not None:
            self = _magic_network
            
        if namespace is not None:
            self.before_run(('explicit-run-namespace', namespace))
        else:
            namespace = get_local_namespace(1 + level)
            self.before_run(('implicit-run-namespace', namespace))
            
        cpp_standalone_device.clocks.update(self._clocks)
            
        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        updaters = []
        for obj in self.objects:
            for updater in obj.updaters:
                updaters.append((obj.clock, updater))
        
        # Generate the updaters
        run_lines = ['{self.name}.clear();'.format(self=self)]
        for clock, updater in updaters:
            cls = updater.__class__
            if cls is CodeObjectUpdater:
                codeobj = updater.owner
                run_lines.append('{self.name}.add(&{clock.name}, _run_{codeobj.name});'.format(clock=clock, self=self,
                                                                                               codeobj=codeobj));
            else:
                raise NotImplementedError("C++ standalone device has not implemented "+cls.__name__)
        run_lines.append('{self.name}.run({duration});'.format(self=self, duration=float(duration)))
        cpp_standalone_device.main_queue.append(('run_network', (self, run_lines)))

    def __repr__(self):
        return '<Network for C++ standalone>'


#: `Network` object that is "run" in standalone
fake_network = Network(name='_fake_network')

def run(*args, **kwds):
    kwds['_magic_network'] = magic_network
    kwds['level'] = kwds.pop('level', 0)+1
    fake_network.run(*args, **kwds)
    
def stop(*args, **kwds):
    raise NotImplementedError("stop() function not supported in standalone mode")
