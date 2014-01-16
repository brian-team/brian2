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
        if (isinstance(v, Variable) and not isinstance(v, AttributeVariable) and
                v.scalar and v.constant and v.read_only):
            code = word_substitute(code, {k: repr(v.get_value())})
    return code


class CPPStandaloneDevice(Device):
    '''
    The `Device` used for C++ standalone simulations.
    '''
    def __init__(self):
        super(CPPStandaloneDevice, self).__init__()
        #: Dictionary mapping `ArrayVariable` objects to their globally
        #: unique name
        self.arrays = {}
        #: List of all dynamic arrays
        #: Dictionary mapping `DynamicArrayVariable` objects with 1 dimension to
        #: their globally unique name
        self.dynamic_arrays = {}
        #: Dictionary mapping `DynamicArrayVariable` objects with 2 dimensions
        #: to their globally unique name
        self.dynamic_arrays_2d = {}
        #: List of all arrays to be filled with zeros
        self.zero_arrays = []
        #: List of all arrays to be filled with numbers (tuple with
        #: `ArrayVariable` object and start value)
        self.arange_arrays = []

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

    def get_array_name(self, var, access_data=True):
        '''
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        '''
        if isinstance(var, ArrayVariable) and not isinstance(var, DynamicArrayVariable):
            return self.arrays[var]
        elif isinstance(var, DynamicArrayVariable) and access_data:
            return self.arrays[var]
        elif isinstance(var, DynamicArrayVariable):
            return self.dynamic_arrays[var]
        else:
            raise TypeError(('Do not have a name for variable of type '
                             '%s') % type(var))

    def add_array(self, var):
        # Note that a dynamic array variable is added to both the arrays and
        # the _dynamic_array dictionary
        if isinstance(var, DynamicArrayVariable):
            name = '_dynamic_array_%s_%s' % (var.owner.name, var.name)
            if var.dimensions == 1:
                self.dynamic_arrays[var] = name
            elif var.dimensions == 2:
                self.dynamic_arrays_2d[var] = name
            else:
                raise AssertionError(('Did not expect a dynamic array with %d '
                                      'dimensions.') % var.dimensions)

        name = '_array_%s_%s' % (var.owner.name, var.name)
        self.arrays[var] = name

    def init_with_zeros(self, var):
        self.zero_arrays.append(var)

    def init_with_arange(self, var, start):
        self.arange_arrays.append((var, start))

    def fill_with_array(self, var, arr):
        array_name = self.arrays[var]
        static_array_name = self.static_array(array_name, arr)
        self.main_queue.append(('set_by_array', (array_name,
                                                 static_array_name)))


    def code_object_class(self, codeobj_class=None):
        if codeobj_class is not None:
            raise ValueError("Cannot specify codeobj_class for C++ standalone device.")
        return CPPStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None):
        codeobj = super(CPPStandaloneDevice, self).code_object(owner, name, abstract_code, variables,
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

        # # Find numpy arrays in the namespaces and convert them into static
        # # arrays. Hopefully they are correctly used in the code: For example,
        # # this works for the namespaces for functions with C++ (e.g. TimedArray
        # # treats it as a C array) but does not work in places that are
        # # implicitly vectorized (state updaters, resets, etc.). But arrays
        # # shouldn't be used there anyway.
        # for code_object in self.code_objects.itervalues():
        #     for name, value in code_object.variables.iteritems():
        #         if isinstance(value, numpy.ndarray):
        #             self.static_arrays[name] = value

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
                                                            array_specs=self.arrays,
                                                            dynamic_array_specs=self.dynamic_arrays,
                                                            dynamic_array_2d_specs=self.dynamic_arrays_2d,
                                                            zero_arrays=self.zero_arrays,
                                                            arange_arrays=self.arange_arrays,
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
                arrayname, staticarrayname = args
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
                main_lines.append(codeobj.code.main_finalise)

        # Generate data for non-constant values
        handled_arrays = defaultdict(set)
        code_object_defs = defaultdict(list)
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
                                                             name=v.owner.name,
                                                             attribute=v.attribute)
                    code_object_defs[codeobj.name].append(code)
                elif isinstance(v, ArrayVariable):
                    try:
                        if isinstance(v, DynamicArrayVariable):
                            if v.dimensions == 1:
                                dyn_array_name = self.dynamic_arrays[v]
                                array_name = self.arrays[v]
                                code_object_defs[codeobj.name].append('{c_type}* const {array_name} = &{dyn_array_name}[0];'.format(c_type=c_data_type(v.dtype),
                                                                                                                                    array_name=array_name,
                                                                                                                                    dyn_array_name=dyn_array_name))
                                code_object_defs[codeobj.name].append('const int _num{k} = {dyn_array_name}.size();'.format(k=k,
                                                                                                                       dyn_array_name=dyn_array_name))
                        else:
                            code_object_defs[codeobj.name].append('const int _num%s = %s;' % (k, v.size))
                    except TypeError:
                        pass

        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
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
        code_objects = []
        for obj in self.objects:
            for codeobj in obj._code_objects:
                code_objects.append((obj.clock, codeobj))
        
        # Generate the updaters
        run_lines = ['{self.name}.clear();'.format(self=self)]
        for clock, codeobj in code_objects:
            run_lines.append('{self.name}.add(&{clock.name}, _run_{codeobj.name});'.format(clock=clock, self=self,
                                                                                               codeobj=codeobj))
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
