from Cython.Build.Inline import *
from Cython.Build.Inline import _get_build_extension, _create_context

__all__ = ['modified_cython_inline']

def modified_cython_inline(
                  code,
                  get_type=unsafe_type,
                  lib_dir=os.path.join(get_cython_cache_dir(), 'inline'),
                  cython_include_dirs=['.'],
                  force=False,
                  quiet=False,
                  locals=None,
                  globals=None,
                  **kwds):
    if get_type is None:
        get_type = lambda x: 'object'
    code = to_unicode(code)
    orig_code = code
    code, literals = strip_string_literals(code)
    code = strip_common_indent(code)
    ctx = _create_context(tuple(cython_include_dirs))
    if locals is None:
        locals = inspect.currentframe().f_back.f_back.f_locals
    if globals is None:
        globals = inspect.currentframe().f_back.f_back.f_globals
    try:
        for symbol in unbound_symbols(code):
            if symbol in kwds:
                continue
            elif symbol in locals:
                kwds[symbol] = locals[symbol]
            elif symbol in globals:
                kwds[symbol] = globals[symbol]
            elif symbol=='sin' or symbol=='exp':
                pass
# elif symbol=='numpy':
# pass
            else:
                print("Couldn't find ", symbol)
    except AssertionError:
        if not quiet:
            # Parsing from strings not fully supported (e.g. cimports).
            print("Could not parse code as a string (to extract unbound symbols).")
    cimports = []
    for name, arg in kwds.items():
        if arg is cython_module:
            cimports.append('\ncimport cython as %s' % name)
            del kwds[name]
    arg_names = kwds.keys()
    arg_names.sort()
    arg_sigs = tuple([(get_type(kwds[arg], ctx), arg) for arg in arg_names])
    key = orig_code, arg_sigs, sys.version_info, sys.executable, Cython.__version__
    module_name = "_cython_inline_" + hashlib.md5(str(key).encode('utf-8')).hexdigest()
    
    if module_name in sys.modules:
        module = sys.modules[module_name]
    
    else:
        build_extension = None
        if cython_inline.so_ext is None:
            # Figure out and cache current extension suffix
            build_extension = _get_build_extension()
            cython_inline.so_ext = build_extension.get_ext_filename('')

        module_path = os.path.join(lib_dir, module_name + cython_inline.so_ext)

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        if force or not os.path.isfile(module_path):
            cflags = []
            c_include_dirs = []
            qualified = re.compile(r'([.\w]+)[.]')
            for type, _ in arg_sigs:
                m = qualified.match(type)
                if m:
                    cimports.append('\ncimport %s' % m.groups()[0])
                    # one special case
                    if m.groups()[0] == 'numpy':
                        import numpy
                        c_include_dirs.append(numpy.get_include())
                        # cflags.append('-Wno-unused')
            module_body, func_body = extract_func_code(code)
            params = ', '.join(['%s %s' % a for a in arg_sigs])
            module_code = """
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
%(module_body)s
%(cimports)s
#import numpy
from libc.math cimport sin, exp
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def __invoke(%(params)s):
%(func_body)s
""" % {'cimports': '\n'.join(cimports), 'module_body': module_body, 'params': params, 'func_body': func_body }
            for key, value in literals.items():
                module_code = module_code.replace(key, value)
# print module_code
            pyx_file = os.path.join(lib_dir, module_name + '.pyx')
            fh = open(pyx_file, 'w')
            try:
                fh.write(module_code)
            finally:
                fh.close()
            extension = Extension(
                name = module_name,
                sources = [pyx_file],
                include_dirs = c_include_dirs,
                #extra_compile_args = cflags,
                extra_compile_args = ['-O3', '-ffast-math', '-march=native'],
                )
            if build_extension is None:
                build_extension = _get_build_extension()
            build_extension.extensions = cythonize([extension], include_path=cython_include_dirs, quiet=quiet)
            build_extension.build_temp = os.path.dirname(pyx_file)
            build_extension.build_lib = lib_dir
            build_extension.run()

        module = imp.load_dynamic(module_name, module_path)

    arg_list = [kwds[arg] for arg in arg_names]
    return module, arg_list
    #return lambda: module.__invoke(*arg_list)
    #return module.__invoke(*arg_list)
