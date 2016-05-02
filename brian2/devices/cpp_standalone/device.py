'''
Module implementing the C++ "standalone" device.
'''
import os
import shutil
import subprocess
import inspect
import platform
from collections import defaultdict, Counter
import numbers
import tempfile
from distutils import ccompiler

import numpy as np
from cpuinfo import cpuinfo

from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.core.network import Network
from brian2.devices.device import Device, all_devices, set_device, reset_device
from brian2.core.variables import *
from brian2.core.namespace import get_local_namespace
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.synapses.synapses import Synapses
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.filetools import copy_directory, ensure_directory, in_directory
from brian2.utils.stringtools import word_substitute
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.units.fundamentalunits import Quantity, have_same_dimensions
from brian2.units import second, ms
from brian2.utils.logger import get_logger, std_silent

from .codeobject import CPPStandaloneCodeObject, openmp_pragma


__all__ = []

logger = get_logger(__name__)


# Preferences
prefs.register_preferences(
    'devices.cpp_standalone',
    'C++ standalone preferences ',
    openmp_threads=BrianPreference(
        default=0,
        docs='''
        The number of threads to use if OpenMP is turned on. By default, this value is set to 0 and the C++ code
        is generated without any reference to OpenMP. If greater than 0, then the corresponding number of threads
        are used to launch the simulation.
        ''',
        ),
    openmp_spatialneuron_strategy=BrianPreference(
        default=None,
        validator=lambda val: val in [None, 'branches', 'systems'],
        docs='''
        Which strategy to chose for solving the three tridiagonal systems with
        OpenMP: `'branches'` means to solve the three systems sequentially, but
        for all the branches in parallel, `'systems'` means to solve the three
        systems in parallel, but all the branches within each system
        sequentially. The `'branches'` approach is usually better for
        morphologies with many branches and a large number of threads, while the
        `'systems'` strategy should be better for morphologies with few
        branches (e.g. cables) and/or simulations with no more than three
        threads. If not specified (the default), the `'systems'` strategy will
        be used when using no more than three threads or when the morphology
        has less than three branches in total.
        '''
    )
    )


class CPPWriter(object):
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.source_files = []
        self.header_files = []
        
    def write(self, filename, contents):
        logger.diagnostic('Writing file %s:\n%s' % (filename, contents))
        if filename.lower().endswith('.cpp'):
            self.source_files.append(filename)
        elif filename.lower().endswith('.h'):
            self.header_files.append(filename)
        elif filename.endswith('.*'):
            self.write(filename[:-1]+'cpp', contents.cpp_file)
            self.write(filename[:-1]+'h', contents.h_file)
            return
        fullfilename = os.path.join(self.project_dir, filename)
        if os.path.exists(fullfilename):
            if open(fullfilename, 'r').read()==contents:
                return
        open(fullfilename, 'w').write(contents)


def invert_dict(x):
    return dict((v, k) for k, v in x.iteritems())


class CPPStandaloneDevice(Device):
    '''
    The `Device` used for C++ standalone simulations.
    '''
    def __init__(self):
        super(CPPStandaloneDevice, self).__init__()
        #: Dictionary mapping `ArrayVariable` objects to their globally
        #: unique name
        self.arrays = {}
        #: Dictionary mapping `ArrayVariable` objects to their value or to
        #: ``None`` if the value (potentially) depends on executed code. This
        #: mechanism allows to access state variables in standalone mode if
        #: their value is known at run time
        self.array_cache = {}
        #: List of all dynamic arrays
        #: Dictionary mapping `DynamicArrayVariable` objects with 1 dimension to
        #: their globally unique name
        self.dynamic_arrays = {}
        #: Dictionary mapping `DynamicArrayVariable` objects with 2 dimensions
        #: to their globally unique name
        self.dynamic_arrays_2d = {}
        #: List of all arrays to be filled with zeros
        self.zero_arrays = []
        #: Dictionary of all arrays to be filled with numbers (mapping
        #: `ArrayVariable` objects to start value)
        self.arange_arrays = {}

        #: Whether the simulation has been run
        self.has_been_run = False

        #: Whether a run should trigger a build
        self.build_on_run = False

        #: build options
        self.build_options = None

        #: Dict of all static saved arrays
        self.static_arrays = {}

        self.code_objects = {}
        self.main_queue = []
        self.runfuncs = {}
        self.networks = []
        self.net_synapses = [] 
        self.static_array_specs =[]
        self.report_func = ''
        self.synapses = []
        
        self.clocks = set([])
        
    def reinit(self):
        # Remember the build_on_run setting and its options -- important during
        # testing
        build_on_run = self.build_on_run
        build_options = self.build_options
        self.__init__()
        super(CPPStandaloneDevice, self).reinit()
        self.build_on_run = build_on_run
        self.build_options = build_options

    def freeze(self, code, ns):
        # this is a bit of a hack, it should be passed to the template somehow
        for k, v in ns.items():
            if (isinstance(v, Variable) and
                  v.scalar and v.constant and v.read_only):
                try:
                    v = v.get_value()
                except NotImplementedError:
                    continue
            if isinstance(v, basestring):
                code = word_substitute(code, {k: v})
            elif isinstance(v, numbers.Number):
                # Use a renderer to correctly transform constants such as True or inf
                renderer = CPPNodeRenderer()
                string_value = renderer.render_expr(repr(v))
                if v < 0:
                    string_value = '(%s)' % string_value
                code = word_substitute(code, {k: string_value})
            else:
                pass  # don't deal with this object
        return code

    def insert_code(self, slot, code):
        '''
        Insert code directly into main.cpp
        '''
        if slot=='main':
            self.main_queue.append(('insert_code', code))
        else:
            logger.warn("Ignoring device code, unknown slot: %s, code: %s" % (slot, code))
            
    def static_array(self, name, arr):
        arr = np.atleast_1d(arr)
        assert len(arr), 'length for %s: %d' % (name, len(arr))
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
        if isinstance(var, DynamicArrayVariable):
            if access_data:
                return self.arrays[var]
            elif var.dimensions == 1:
                return self.dynamic_arrays[var]
            else:
                return self.dynamic_arrays_2d[var]
        elif isinstance(var, ArrayVariable):
            return self.arrays[var]
        else:
            raise TypeError(('Do not have a name for variable of type '
                             '%s') % type(var))

    def get_array_filename(self, var, basedir='results'):
        '''
        Return a file name for a variable.

        Parameters
        ----------
        var : `ArrayVariable`
            The variable to get a filename for.
        basedir : str
            The base directory for the filename, defaults to ``'results'``.
        Returns
        -------
        filename : str
            A filename of the form
            ``'results/'+varname+'_'+str(hash(varname))``, where varname is the
            name returned by `get_array_name`.

        Notes
        -----
        The reason that the filename is not simply ``'results/' + varname`` is
        that this could lead to file names that are not unique in file systems
        that are not case sensitive (e.g. on Windows).
        '''
        varname = self.get_array_name(var, access_data=False)
        return os.path.join(basedir, varname + '_' + str(hash(varname)))

    def add_array(self, var):
        # Note that a dynamic array variable is added to both the arrays and
        # the _dynamic_array dictionary
        if isinstance(var, DynamicArrayVariable):
            # The code below is slightly more complicated than just looking
            # for a unique name as above for static_array, the name has
            # potentially to be unique for more than one dictionary, with
            # different prefixes. This is because dynamic arrays are added to
            # a ``dynamic_arrays`` dictionary (with a `_dynamic` prefix) and to
            # the general ``arrays`` dictionary. We want to make sure that we
            # use the same name in the two dictionaries, not for example
            # ``_dynamic_array_source_name_2`` and ``_array_source_name_1``
            # (this would work fine, but it would make the code harder to read).
            orig_dynamic_name = dynamic_name = '_dynamic_array_%s_%s' % (var.owner.name, var.name)
            orig_array_name = array_name = '_array_%s_%s' % (var.owner.name, var.name)
            suffix = 0

            if var.dimensions == 1:
                dynamic_dict = self.dynamic_arrays
            elif var.dimensions == 2:
                dynamic_dict = self.dynamic_arrays_2d
            else:
                raise AssertionError(('Did not expect a dynamic array with %d '
                                      'dimensions.') % var.dimensions)
            while (dynamic_name in dynamic_dict.values() or
                   array_name in self.arrays.values()):
                suffix += 1
                dynamic_name = orig_dynamic_name + '_%d' % suffix
                array_name = orig_array_name + '_%d' % suffix
            dynamic_dict[var] = dynamic_name
            self.arrays[var] = array_name
        else:
            orig_array_name = array_name = '_array_%s_%s' % (var.owner.name, var.name)
            suffix = 0
            while (array_name in self.arrays.values()):
                suffix += 1
                array_name = orig_array_name + '_%d' % suffix
            self.arrays[var] = array_name

    def init_with_zeros(self, var, dtype):
        self.zero_arrays.append(var)
        self.array_cache[var] = np.zeros(var.size, dtype=dtype)

    def init_with_arange(self, var, start, dtype):
        self.arange_arrays[var] = start
        self.array_cache[var] = np.arange(0, var.size, dtype=dtype) + start

    def fill_with_array(self, var, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return  # nothing to do
        array_name = self.get_array_name(var, access_data=False)
        if isinstance(var, DynamicArrayVariable):
            # We can never be sure about the size of a dynamic array, so
            # we can't do correct broadcasting. Therefore, we do not cache
            # them at all for now.
            self.array_cache[var] = None
        else:
            new_arr = np.empty(var.size, dtype=var.dtype)
            new_arr[:] = arr
            self.array_cache[var] = new_arr

        if arr.size == 1:
            if var.size == 1:
                value = CPPNodeRenderer().render_expr(repr(arr.item(0)))
                # For a single assignment, generate a code line instead of storing the array
                self.main_queue.append(('set_by_single_value', (array_name,
                                                                0,
                                                                value)))
            else:
                self.main_queue.append(('set_by_constant', (array_name,
                                                            arr.item(),
                                                            isinstance(var, DynamicArrayVariable))))
        else:
            # Using the std::vector instead of a pointer to the underlying
            # data for dynamic arrays is fast enough here and it saves us some
            # additional work to set up the pointer
            static_array_name = self.static_array(array_name, arr)
            self.main_queue.append(('set_by_array', (array_name,
                                                     static_array_name,
                                                     isinstance(var, DynamicArrayVariable))))

    def resize(self, var, new_size):
        array_name = self.get_array_name(var, access_data=False)
        self.main_queue.append(('resize_array', (array_name, new_size)))

    def variableview_set_with_index_array(self, variableview, item,
                                          value, check_units):
        if isinstance(item, slice) and item == slice(None):
            item = 'True'
        value = Quantity(value)

        if (isinstance(item, int) or (isinstance(item, np.ndarray) and item.shape==())) and value.size == 1:
            array_name = self.get_array_name(variableview.variable, access_data=False)
            value_str = CPPNodeRenderer().render_expr(repr(np.asarray(value).item(0)))
            if self.array_cache.get(variableview.variable, None) is not None:
                self.array_cache[variableview.variable][item] = value
            # For a single assignment, generate a code line instead of storing the array
            self.main_queue.append(('set_by_single_value', (array_name,
                                                            item,
                                                            value_str)))
        # Simple case where we don't have to do any indexing
        elif (item == 'True' and variableview.index_var in ('_idx', '0')):
            self.fill_with_array(variableview.variable, value)
        else:
            # We have to calculate indices. This will not work for synaptic
            # variables
            try:
                indices = np.asarray(variableview.indexing(item,
                                                           index_var=variableview.index_var))
            except NotImplementedError:
                raise NotImplementedError(('Cannot set variable "%s" this way in '
                                           'standalone, try using string '
                                           'expressions.') % variableview.name)
            # Using the std::vector instead of a pointer to the underlying
            # data for dynamic arrays is fast enough here and it saves us some
            # additional work to set up the pointer
            arrayname = self.get_array_name(variableview.variable,
                                            access_data=False)
            if (indices.shape != () and
                    (value.shape == () or
                         (value.size == 1 and indices.size > 1))):
                value = np.repeat(value, indices.size)
            elif (value.shape != indices.shape and len(value) != len(indices)):
                raise ValueError(('Provided values do not match the size '
                                  'of the indices, '
                                  '%d != %d.') % (len(value),
                                                  len(indices)))

            staticarrayname_index = self.static_array('_index_'+arrayname,
                                                      indices)
            staticarrayname_value = self.static_array('_value_'+arrayname,
                                                      value)
            self.array_cache[variableview.variable] = None
            self.main_queue.append(('set_array_by_array', (arrayname,
                                                           staticarrayname_index,
                                                           staticarrayname_value)))

    def get_value(self, var, access_data=True):
        # Usually, we cannot retrieve the values of state variables in
        # standalone scripts since their values might depend on the evaluation
        # of expressions at runtime. For some variables we do know the value
        # however (values that have been set with explicit values and not
        # changed in code objects)
        if self.array_cache.get(var, None) is not None:
            return self.array_cache[var]
        else:
            # After the network has been run, we can retrieve the values from
            # disk
            if self.has_been_run:
                dtype = var.dtype
                fname = os.path.join(self.project_dir,
                                     self.get_array_filename(var))
                with open(fname, 'rb') as f:
                    data = np.fromfile(f, dtype=dtype)
                # This is a bit of an heuristic, but our 2d dynamic arrays are
                # only expanding in one dimension, we assume here that the
                # other dimension has size 0 at the beginning
                if isinstance(var.size, tuple) and len(var.size) == 2:
                    if var.size[0] * var.size[1] == len(data):
                        size = var.size
                    elif var.size[0] == 0:
                        size = (len(data)/var.size[1], var.size[1])
                    elif var.size[0] == 0:
                        size = (var.size[0], len(data)/var.size[0])
                    else:
                        raise IndexError(('Do not now how to deal with 2d '
                                          'array of size %s, the array on disk '
                                          'has length %d') % (str(var.size),
                                                              len(data)))

                    var.size = size
                    return data.reshape(var.size)
                var.size = len(data)
                return data
            raise NotImplementedError('Cannot retrieve the values of state '
                                      'variables in standalone code before the '
                                      'simulation has been run.')

    def variableview_get_subexpression_with_index_array(self, variableview,
                                                        item, level=0,
                                                        run_namespace=None):
        if not self.has_been_run:
            raise NotImplementedError('Cannot retrieve the values of state '
                                      'variables in standalone code before the '
                                      'simulation has been run.')
        # Temporarily switch to the runtime device to evaluate the subexpression
        # (based on the values stored on disk)
        set_device('runtime')
        result = VariableView.get_subexpression_with_index_array(variableview, item,
                                                                 level=level+2,
                                                                 run_namespace=run_namespace)
        reset_device()
        return result

    def variableview_get_with_expression(self, variableview, code, level=0,
                                         run_namespace=None):
        raise NotImplementedError('Cannot retrieve the values of state '
                                  'variables with string expressions in '
                                  'standalone scripts.')

    def code_object_class(self, codeobj_class=None):
        # Ignore the requested codeobj_class
        return CPPStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None,
                    override_conditional_write=None):
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = dict(template_kwds)
        template_kwds['user_headers'] = prefs['codegen.cpp.headers']
        codeobj = super(CPPStandaloneDevice, self).code_object(owner, name, abstract_code, variables,
                                                               template_name, variable_indices,
                                                               codeobj_class=codeobj_class,
                                                               template_kwds=template_kwds,
                                                               override_conditional_write=override_conditional_write,
                                                               )
        self.code_objects[codeobj.name] = codeobj

        # Mark all the non-read-only or non-constant variables used in this code
        # object as "dirty". This is almost certainly too much, most of these
        # variables will only be read. However, the templates for synapse
        # creation will write to read-only variables.. This is noted in the
        # WRITES_TO_READ_ONLY_VARIABLES comment in the template.
        template = getattr(codeobj.templater, template_name)
        written_readonly_vars = {codeobj.variables[varname]
                                 for varname in template.writes_read_only}
        for var in codeobj.variables.itervalues():
            if (isinstance(var, ArrayVariable) and
                    (not var.read_only or not var.constant or
                             var in written_readonly_vars)):
                self.array_cache[var] = None

        return codeobj

    def check_openmp_compatible(self, nb_threads):
        if nb_threads > 0:
            logger.warn("OpenMP code is not yet well tested, and may be inaccurate.", "openmp", once=True)
            logger.diagnostic("Using OpenMP with %d threads " % nb_threads)
    
    def generate_objects_source(self, writer, arange_arrays, synapses, static_array_specs, networks):
        arr_tmp = CPPStandaloneCodeObject.templater.objects(
                        None, None,
                        array_specs=self.arrays,
                        dynamic_array_specs=self.dynamic_arrays,
                        dynamic_array_2d_specs=self.dynamic_arrays_2d,
                        zero_arrays=self.zero_arrays,
                        arange_arrays=arange_arrays,
                        synapses=synapses,
                        clocks=self.clocks,
                        static_array_specs=static_array_specs,
                        networks=networks,
                        get_array_filename=self.get_array_filename,
                        get_array_name=self.get_array_name,
                        code_objects=self.code_objects.values())
        writer.write('objects.*', arr_tmp)
        
    def generate_main_source(self, writer):
        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
                main_lines.append('_run_%s();' % codeobj.name)
            elif func=='run_network':
                net, netcode = args
                main_lines.extend(netcode)
            elif func=='set_by_constant':
                arrayname, value, is_dynamic = args
                size_str = arrayname+'.size()' if is_dynamic else '_num_'+arrayname
                code = '''
                {pragma}
                for(int i=0; i<{size_str}; i++)
                {{
                    {arrayname}[i] = {value};
                }}
                '''.format(arrayname=arrayname, size_str=size_str,
                           value=CPPNodeRenderer().render_expr(repr(value)),
                           pragma=openmp_pragma('static'))
                main_lines.extend(code.split('\n'))
            elif func=='set_by_array':
                arrayname, staticarrayname, is_dynamic = args
                size_str = arrayname+'.size()' if is_dynamic else '_num_'+arrayname
                code = '''
                {pragma}
                for(int i=0; i<{size_str}; i++)
                {{
                    {arrayname}[i] = {staticarrayname}[i];
                }}
                '''.format(arrayname=arrayname, size_str=size_str,
                           staticarrayname=staticarrayname,
                           pragma=openmp_pragma('static'))
                main_lines.extend(code.split('\n'))
            elif func=='set_by_single_value':
                arrayname, item, value = args
                code = '{arrayname}[{item}] = {value};'.format(arrayname=arrayname,
                                                               item=item,
                                                               value=value)
                main_lines.extend([code])
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value = args
                code = '''
                {pragma}
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                '''.format(arrayname=arrayname, staticarrayname_index=staticarrayname_index,
                           staticarrayname_value=staticarrayname_value, pragma=openmp_pragma('static'))
                main_lines.extend(code.split('\n'))
            elif func=='resize_array':
                array_name, new_size = args
                main_lines.append("{array_name}.resize({new_size});".format(array_name=array_name,
                                                                            new_size=new_size))
            elif func=='insert_code':
                main_lines.append(args)
            elif func=='start_run_func':
                name, include_in_parent = args
                if include_in_parent:
                    main_lines.append('%s();' % name)
                main_lines = []
                procedures.append((name, main_lines))
            elif func=='end_run_func':
                name, include_in_parent = args
                name, main_lines = procedures.pop(-1)
                runfuncs[name] = main_lines
                name, main_lines = procedures[-1]
            else:
                raise NotImplementedError("Unknown main queue function type "+func)
        
        self.runfuncs = runfuncs

        # generate the finalisations
        for codeobj in self.code_objects.itervalues():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise)
                
                # The code_objects are passed in the right order to run them because they were
        # sorted by the Network object. To support multiple clocks we'll need to be
        # smarter about that.
        main_tmp = CPPStandaloneCodeObject.templater.main(None, None,
                                                          main_lines=main_lines,
                                                          code_objects=self.code_objects.values(),
                                                          report_func=self.report_func,
                                                          dt=float(self.defaultclock.dt),
                                                          user_headers=prefs['codegen.cpp.headers']
                                                          )
        writer.write('main.cpp', main_tmp)
        
    def generate_codeobj_source(self, writer):
                # Generate data for non-constant values
        code_object_defs = defaultdict(list)
        for codeobj in self.code_objects.itervalues():
            lines = []
            for k, v in codeobj.variables.iteritems():
                if isinstance(v, ArrayVariable):
                    try:
                        if isinstance(v, DynamicArrayVariable):
                            if v.dimensions == 1:
                                dyn_array_name = self.dynamic_arrays[v]
                                array_name = self.arrays[v]
                                line = '{c_type}* const {array_name} = {dyn_array_name}.empty()? 0 : &{dyn_array_name}[0];'
                                line = line.format(c_type=c_data_type(v.dtype), array_name=array_name,
                                                   dyn_array_name=dyn_array_name)
                                lines.append(line)
                                line = 'const int _num{k} = {dyn_array_name}.size();'
                                line = line.format(k=k, dyn_array_name=dyn_array_name)
                                lines.append(line)
                        else:
                            lines.append('const int _num%s = %s;' % (k, v.size))
                    except TypeError:
                        pass
            for line in lines:
                # Sometimes an array is referred to by to different keys in our
                # dictionary -- make sure to never add a line twice
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)

        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
            # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant.
            code = self.freeze(codeobj.code.cpp_file, ns)
            code = code.replace('%CONSTANTS%', '\n'.join(code_object_defs[codeobj.name]))
            code = '#include "objects.h"\n'+code
            
            writer.write('code_objects/'+codeobj.name+'.cpp', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)
        
    def generate_network_source(self, writer, compiler):
        maximum_run_time = self._maximum_run_time
        if maximum_run_time is not None:
            maximum_run_time = float(maximum_run_time)
        network_tmp = CPPStandaloneCodeObject.templater.network(None, None, maximum_run_time=maximum_run_time)
        writer.write('network.*', network_tmp)
        
    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = CPPStandaloneCodeObject.templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', synapses_classes_tmp)
        
    def generate_run_source(self, writer):
        run_tmp = CPPStandaloneCodeObject.templater.run(None, None, run_funcs=self.runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        user_headers=prefs['codegen.cpp.headers'],
                                                        array_specs=self.arrays,
                                                        clocks=self.clocks
                                                        )
        writer.write('run.*', run_tmp)
        
    def generate_makefile(self, writer, compiler, compiler_flags, linker_flags, nb_threads):
        if compiler=='msvc':
            if nb_threads>1:
                openmp_flag = '/openmp'
            else:
                openmp_flag = ''
            # Generate the visual studio makefile
            source_bases = [fname.replace('.cpp', '').replace('/', '\\') for fname in writer.source_files]
            win_makefile_tmp = CPPStandaloneCodeObject.templater.win_makefile(
                None, None,
                source_bases=source_bases,
                compiler_flags=compiler_flags,
                linker_flags=linker_flags,
                openmp_flag=openmp_flag,
                )
            writer.write('win_makefile', win_makefile_tmp)
        else:
            # Generate the makefile
            if os.name=='nt':
                rm_cmd = 'del *.o /s\n\tdel main.exe $(DEPS)'
            else:
                rm_cmd = 'rm $(OBJS) $(PROGRAM) $(DEPS)'
            makefile_tmp = CPPStandaloneCodeObject.templater.makefile(None, None,
                source_files=' '.join(writer.source_files),
                header_files=' '.join(writer.header_files),
                compiler_flags=compiler_flags,
                linker_flags=linker_flags,
                rm_cmd=rm_cmd)
            writer.write('makefile', makefile_tmp)
    
    def copy_source_files(self, writer, directory):
        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(CPPStandaloneCodeObject))[0],
                                    'brianlib')
        brianlib_files = copy_directory(brianlib_dir, os.path.join(directory, 'brianlib'))
        for file in brianlib_files:
            if file.lower().endswith('.cpp'):
                writer.source_files.append('brianlib/'+file)
            elif file.lower().endswith('.h'):
                writer.header_files.append('brianlib/'+file)

        # Copy the CSpikeQueue implementation
        shutil.copy2(os.path.join(os.path.split(inspect.getsourcefile(Synapses))[0], 'cspikequeue.cpp'),
                     os.path.join(directory, 'brianlib', 'spikequeue.h'))
        shutil.copy2(os.path.join(os.path.split(inspect.getsourcefile(Synapses))[0], 'stdint_compat.h'),
                     os.path.join(directory, 'brianlib', 'stdint_compat.h'))
        
    def write_static_arrays(self, directory):
        # # Find np arrays in the namespaces and convert them into static
        # # arrays. Hopefully they are correctly used in the code: For example,
        # # this works for the namespaces for functions with C++ (e.g. TimedArray
        # # treats it as a C array) but does not work in places that are
        # # implicitly vectorized (state updaters, resets, etc.). But arrays
        # # shouldn't be used there anyway.
        for code_object in self.code_objects.itervalues():
            for name, value in code_object.variables.iteritems():
                if isinstance(value, np.ndarray):
                    self.static_arrays[name] = value
                    
        logger.diagnostic("static arrays: "+str(sorted(self.static_arrays.keys())))
        
        static_array_specs = []
        for name, arr in sorted(self.static_arrays.items()):
            arr.tofile(os.path.join(directory, 'static_arrays', name))
            static_array_specs.append((name, c_data_type(arr.dtype), arr.size, name))
        self.static_array_specs = static_array_specs
            
    def find_synapses(self):
        # Write the global objects
        networks = [net() for net in Network.__instances__()
                    if net().name != '_fake_network']
        synapses = []
        for net in networks:
            net_synapses = [s for s in net.objects if isinstance(s, Synapses)]
            synapses.extend(net_synapses)
        self.networks = networks
        self.net_synapses = synapses
    
    def compile_source(self, directory, compiler, debug, clean):
        num_threads = prefs.devices.cpp_standalone.openmp_threads
        with in_directory(directory):
            if compiler == 'msvc':
                from distutils import msvc9compiler
                # TODO: handle debug
                if debug:
                    logger.warn('Debug flag currently ignored for MSVC', once=True)
                vcvars_loc = prefs['codegen.cpp.msvc_vars_location']
                if vcvars_loc == '':
                    for version in xrange(16, 8, -1):
                        fname = msvc9compiler.find_vcvarsall(version)
                        if fname:
                            if version==14 and num_threads>0:
                                logger.warn("Found Visual Studio 2015, but due to a bug in OpenMP support in "
                                            "that version it is being ignored. We will use another version if "
                                            "we find one, or you can switch OpenMP support off.", once=True)
                            else:
                                vcvars_loc = fname
                                break
                if vcvars_loc == '':
                    raise IOError("Cannot find vcvarsall.bat on standard "
                                  "search path. Set the "
                                  "codegen.cpp.msvc_vars_location preference "
                                  "explicitly.")
                # TODO: copy vcvars and make replacements for 64 bit automatically
                arch_name = prefs['codegen.cpp.msvc_architecture']
                if arch_name == '':
                    mach = platform.machine()
                    if mach == 'AMD64':
                        arch_name = 'x86_amd64'
                    else:
                        arch_name = 'x86'
                
                vcvars_cmd = '"{vcvars_loc}" {arch_name}'.format(
                        vcvars_loc=vcvars_loc, arch_name=arch_name)
                make_cmd = 'nmake /f win_makefile'
                if os.path.exists('winmake.log'):
                    os.remove('winmake.log')
                with std_silent(debug):
                    if clean:
                        os.system('%s >>winmake.log 2>&1 && %s clean >>winmake.log 2>&1' % (vcvars_cmd, make_cmd))
                    x = os.system('%s >>winmake.log 2>&1 && %s >>winmake.log 2>&1' % (vcvars_cmd, make_cmd))
                    if x!=0:
                        if os.path.exists('winmake.log'):
                            print open('winmake.log', 'r').read()
                        raise RuntimeError("Project compilation failed")
            else:
                with std_silent(debug):
                    if clean:
                        os.system('make clean')
                    if debug:
                        x = os.system('make debug')
                    else:
                        x = os.system('make')
                    if x!=0:
                        raise RuntimeError("Project compilation failed")

    def run(self, directory, with_output, run_args):
        with in_directory(directory):
            if not with_output:
                stdout = open('results/stdout.txt', 'w')
            else:
                stdout = None
            if os.name=='nt':
                x = subprocess.call(['main'] + run_args, stdout=stdout)
            else:
                x = subprocess.call(['./main'] + run_args, stdout=stdout)
            if x:
                if stdout is not None:
                    stdout.close()
                if os.path.exists('results/stdout.txt'):
                    print open('results/stdout.txt', 'r').read()
                raise RuntimeError("Project run failed (project directory: "
                                   "%s)" % os.path.abspath(directory))
            self.has_been_run = True
            if os.path.isfile('results/last_run_info.txt'):
                last_run_info = open('results/last_run_info.txt', 'r').read()
                self._last_run_time, self._last_run_completed_fraction = map(float, last_run_info.split())

    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=True,
              with_output=True, additional_source_files=None,
              run_args=None, direct_call=True, **kwds):
        '''
        Build the project

        TODO: more details

        Parameters
        ----------
        directory : str, optional
            The output directory to write the project to, any existing files
            will be overwritten. If the given directory name is ``None``, then
            a temporary directory will be used (used in the test suite to avoid
            problems when running several tests in parallel). Defaults to
            ``'output'``.
        compile : bool, optional
            Whether or not to attempt to compile the project. Defaults to
            ``True``.
        run : bool, optional
            Whether or not to attempt to run the built project if it
            successfully builds. Defaults to ``True``.
        debug : bool, optional
            Whether to compile in debug mode. Defaults to ``False``.
        with_output : bool, optional
            Whether or not to show the ``stdout`` of the built program when run.
            Output will be shown in case of compilation or runtime error.
            Defaults to ``True``.
        clean : bool, optional
            Whether or not to clean the project before building. Defaults to
            ``True``.
        additional_source_files : list of str, optional
            A list of additional ``.cpp`` files to include in the build.
        direct_call : bool, optional
            Whether this function was called directly. Is used internally to
            distinguish an automatic build due to the ``build_on_run`` option
            from a manual ``device.build`` call.
        '''
        if self.build_on_run and direct_call:
            raise RuntimeError('You used set_device with build_on_run=True '
                               '(the default option), which will automatically '
                               'build the simulation at the first encountered '
                               'run call - do not call device.build manually '
                               'in this case. If you want to call it manually, '
                               'e.g. because you have multiple run calls, use '
                               'set_device with build_on_run=False.')
        renames = {'project_dir': 'directory',
                   'compile_project': 'compile',
                   'run_project': 'run'}
        if len(kwds):
            msg = ''
            for kwd in kwds:
                if kwd in renames:
                    msg += ("Keyword argument '%s' has been renamed to "
                            "'%s'. ") % (kwd, renames[kwd])
                else:
                    msg += "Unknown keyword argument '%s'. " % kwd
            raise TypeError(msg)

        if additional_source_files is None:
            additional_source_files = []
        if run_args is None:
            run_args = []
        if directory is None:
            directory = tempfile.mkdtemp()
        self.project_dir = directory
        ensure_directory(directory)

        compiler, extra_compile_args = get_compiler_and_args()
        compiler_obj = ccompiler.new_compiler(compiler=compiler)
        compiler_flags = (ccompiler.gen_preprocess_options(prefs['codegen.cpp.define_macros'],
                                                           prefs['codegen.cpp.include_dirs']) +
                          extra_compile_args)
        linker_flags = (ccompiler.gen_lib_options(compiler_obj,
                                                  library_dirs=prefs['codegen.cpp.library_dirs'],
                                                  runtime_library_dirs=prefs['codegen.cpp.runtime_library_dirs'],
                                                  libraries=prefs['codegen.cpp.libraries']) +
                        prefs['codegen.cpp.extra_link_args'])

        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(directory, d))

        writer = CPPWriter(directory)

        # Get the number of threads if specified in an openmp context
        nb_threads = prefs.devices.cpp_standalone.openmp_threads
        # If the number is negative, we need to throw an error
        if (nb_threads < 0):
            raise ValueError('The number of OpenMP threads can not be negative !')

        logger.diagnostic("Writing C++ standalone project to directory "+os.path.normpath(directory))

        self.check_openmp_compatible(nb_threads)

        arange_arrays = sorted([(var, start)
                                for var, start in self.arange_arrays.iteritems()],
                               key=lambda (var, start): var.name)

        self.write_static_arrays(directory)
        self.find_synapses()

        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in self.networks:
            net.after_run()

        # Check that all names are globally unique
        names = [obj.name for net in self.networks for obj in net.objects]
        non_unique_names = [name for name, count in Counter(names).iteritems()
                            if count > 1]
        if len(non_unique_names):
            formatted_names = ', '.join("'%s'" % name
                                        for name in non_unique_names)
            raise ValueError('All objects need to have unique names in '
                             'standalone mode, the following name(s) were used '
                             'more than once: %s' % formatted_names)

        self.generate_objects_source(writer, arange_arrays, self.net_synapses, self.static_array_specs, self.networks)
        self.generate_main_source(writer)
        self.generate_codeobj_source(writer)
        self.generate_network_source(writer, compiler)
        self.generate_synapses_classes_source(writer)
        self.generate_run_source(writer)
        self.copy_source_files(writer, directory)

        writer.source_files.extend(additional_source_files)

        self.generate_makefile(writer, compiler,
                               compiler_flags=' '.join(compiler_flags),
                               linker_flags=' '.join(linker_flags),
                               nb_threads=nb_threads)

        if compile:
            self.compile_source(directory, compiler, debug, clean)
            if run:
                self.run(directory, with_output, run_args)

    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):
        if kwds:
            logger.warn(('Unsupported keyword argument(s) provided for run: '
                         '%s') % ', '.join(kwds.keys()))
        net._clocks = {obj.clock for obj in net.objects}
        t_end = net.t+duration
        for clock in net._clocks:
            clock.set_interval(net.t, t_end)

        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level+2)

        net.before_run(namespace)

        self.clocks.update(net._clocks)
        net.t_ = float(t_end)

        # TODO: remove this horrible hack
        for clock in self.clocks:
            if clock.name=='clock':
                clock._name = '_clock'
            
        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        code_objects = []
        for obj in net.objects:
            for codeobj in obj._code_objects:
                code_objects.append((obj.clock, codeobj))

        # Code for a progress reporting function
        standard_code = '''
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                %STREAMNAME% << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                %STREAMNAME% << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << elapsed << " s";
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    %STREAMNAME% << ", estimated " << remaining << " s remaining.";
                }
            }

            %STREAMNAME% << std::endl << std::flush;
        }
        '''
        if report is None:
            self.report_func = ''
        elif report == 'text' or report == 'stdout':
            self.report_func = standard_code.replace('%STREAMNAME%', 'std::cout')
        elif report == 'stderr':
            self.report_func = standard_code.replace('%STREAMNAME%', 'std::cerr')
        elif isinstance(report, basestring):
            self.report_func = '''
            void report_progress(const double elapsed, const double completed, const double start, const double duration)
            {
            %REPORT%
            }
            '''.replace('%REPORT%', report)
        else:
            raise TypeError(('report argument has to be either "text", '
                             '"stdout", "stderr", or the code for a report '
                             'function'))

        if report is not None:
            report_call = 'report_progress'
        else:
            report_call = 'NULL'

        # Generate the updaters
        run_lines = ['{net.name}.clear();'.format(net=net)]
        all_clocks = set()
        for clock, codeobj in code_objects:
            run_lines.append('{net.name}.add(&{clock.name}, _run_{codeobj.name});'.format(clock=clock, net=net,
                                                                                               codeobj=codeobj))
            all_clocks.add(clock)

        # Under some rare circumstances (e.g. a NeuronGroup only defining a
        # subexpression that is used by other groups (via linking, or recorded
        # by a StateMonitor) *and* not calculating anything itself *and* using a
        # different clock than all other objects) a clock that is not used by
        # any code object should nevertheless advance during the run. We include
        # such clocks without a code function in the network.
        for clock in net._clocks:
            if clock not in all_clocks:
                run_lines.append('{net.name}.add(&{clock.name}, NULL);'.format(clock=clock, net=net))

        run_lines.append('{net.name}.run({duration}, {report_call}, {report_period});'.format(net=net,
                                                                                              duration=float(duration),
                                                                                              report_call=report_call,
                                                                                              report_period=float(report_period)))
        self.main_queue.append(('run_network', (net, run_lines)))

        # Manually set the cache for the clocks, simulation scripts might
        # want to access the time (which has been set in code and is therefore
        # not accessible by the normal means until the code has been built and
        # run)
        for clock in net._clocks:
            self.array_cache[clock.variables['timestep']] = np.array([clock._i_end])
            self.array_cache[clock.variables['t']] = np.array([clock._i_end * clock.dt_])

        if self.build_on_run:
            if self.has_been_run:
                raise RuntimeError('The network has already been built and run '
                                   'before. Use set_device with '
                                   'build_on_run=False and an explicit '
                                   'device.build call to use multiple run '
                                   'statements with this device.')
            self.build(direct_call=False, **self.build_options)

    def network_store(self, net, name='default'):
        raise NotImplementedError(('The store/restore mechanism is not '
                                   'supported in the C++ standalone'))

    def network_restore(self, net, name='default'):
        raise NotImplementedError(('The store/restore mechanism is not '
                                   'supported in the C++ standalone'))

    def network_get_profiling_info(self, net):
        if net._profiling_info is None:
            net._profiling_info = []
            fname = os.path.join(self.project_dir, 'results', 'profiling_info.txt')
            with open(fname) as f:
                for line in f:
                    (key, val) = line.split()
                    net._profiling_info.append((key, float(val)*second))
        return sorted(net._profiling_info, key=lambda item: item[1],
                      reverse=True)

    def run_function(self, name, include_in_parent=True):
        '''
        Context manager to divert code into a function
        
        Code that happens within the scope of this context manager will go into the named function.
        
        Parameters
        ----------
        name : str
            The name of the function to divert code into.
        include_in_parent : bool
            Whether or not to include a call to the newly defined function in the parent context.
        '''
        return RunFunctionContext(name, include_in_parent)


class RunFunctionContext(object):
    def __init__(self, name, include_in_parent):
        self.name = name
        self.include_in_parent = include_in_parent
    def __enter__(self):
        cpp_standalone_device.main_queue.append(('start_run_func', (self.name, self.include_in_parent)))
    def __exit__(self, type, value, traceback):
        cpp_standalone_device.main_queue.append(('end_run_func', (self.name, self.include_in_parent)))

cpp_standalone_device = CPPStandaloneDevice()
all_devices['cpp_standalone'] = cpp_standalone_device
