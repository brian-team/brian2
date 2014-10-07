# -*- coding: utf-8 -*-
"""
    Automatically generate Brian's reference documentation.
    
    Based on sphinx-apidoc, published under a BSD license: http://sphinx-doc.org/
"""
import inspect
import sys
import os
from os import path

from .examplefinder import auto_find_examples

INITPY = '__init__.py'

OPTIONS = ['show-inheritance']


def makename(package, module):
    """Join package and module with a dot."""
    # Both package and module can be None/empty.
    if package:
        name = package
        if module:
            name += '.' + module
    else:
        name = module
    return name


def write_file(name, text, destdir, suffix):
    """Write the output file for module/package <name>."""
    fname = path.join(destdir, '%s.%s' % (name, suffix))
    print 'Creating file %s.' % fname
    f = open(fname, 'w')
    try:
        f.write(text)
    finally:
        f.close()


def format_heading(level, text):
    """Create a heading of <level> [1, 2 or 3 supported]."""
    underlining = ['=', '-', '~', ][level-1] * len(text)
    return '%s\n%s\n\n' % (text, underlining)


def format_directive(module, destdir, package=None, basename='brian2'):
    """Create the automodule directive and add the options."""
    directive = '.. automodule:: %s\n' % makename(package, module)
    for option in OPTIONS:
        directive += '    :%s:\n' % option
    directive += '\n'
    # document all the classes in the modules
    full_name = basename + '.' + module
    __import__(full_name)
    mod = sys.modules[full_name]
    dir_members = dir(mod)
    classes = []
    functions = []
    variables = []
    for member in dir_members:
        _temp = __import__(full_name, {}, {}, [member], 0)
        member_obj = getattr(_temp, member)
        member_module = getattr(member_obj, '__module__', None)
        # only document members that where defined in this module
        if member_module == full_name and not member.startswith('_'):
            if inspect.isclass(member_obj):
                classes.append((member, member_obj))
            elif inspect.isfunction(member_obj):
                functions.append((member, member_obj))
            else:
                variables.append((member, member_obj))
    
    if classes:
        directive += '**Classes**\n\n'
        for member, member_obj in classes:
            directive += '.. autosummary:: %s\n' % (member)
            directive += '    :toctree:\n\n'
            create_member_file(full_name, member, member_obj, destdir)
    if functions:
        directive += '**Functions**\n\n'
        for member, member_obj in functions:
            directive += '.. autosummary:: %s\n' % (member)
            directive += '    :toctree:\n\n'
            create_member_file(full_name, member, member_obj, destdir)
    if variables:
        directive += '**Objects**\n\n'
        for member, member_obj in variables:
            directive += '.. autosummary:: %s\n' % (member)
            directive += '    :toctree:\n\n'
            create_member_file(full_name, member, member_obj, destdir)
                
    return directive


def find_shortest_import(module_name, obj_name):
    parts = module_name.split('.')
    for idx in range(1, len(parts) + 1):
        try:
            result = __import__('.'.join(parts[:idx]), globals(), {},
                                fromlist=[str(obj_name)], level=0)
            result_obj = getattr(result, obj_name, None)
            if result_obj is not None and getattr(result_obj,
                                                  '__module__',
                                                  None) == module_name:
                # import seems to have worked
                return '.'.join(parts[:idx])
        except ImportError:
            pass
    raise AssertionError("Couldn't import " + module_name + '.' + obj_name)


def create_member_file(module_name, member, member_obj, destdir, suffix='rst'):
    """Build the text of the file and write the file."""
    
    text = '.. currentmodule:: ' + module_name + '\n\n'

    shortest_import = find_shortest_import(module_name, member)
    import_text = '(*Shortest import*: ``from {} import {})``\n\n'.format(shortest_import,
                                                                          member)
    if inspect.isclass(member_obj):
        text += format_heading(1, '%s class' % member)
        text += import_text
        text += '.. autoclass:: %s\n\n' % member
        text += auto_find_examples(member_obj, headersymbol='-')
    elif inspect.isfunction(member_obj):
        text += format_heading(1, '%s function' % member)
        text += import_text
        text += '.. autofunction:: %s\n\n' % member
    else:
        text += format_heading(1, '%s object' % member)
        text += import_text
        text += '.. autodata:: %s\n' % member

    write_file(makename(module_name, member), text, destdir, suffix)


def create_package_file(root, master_package, subroot, py_files, subs,
                        destdir, excludes, suffix='rst'):
    """Build the text of the file and write the file."""
    package = path.split(root)[-1]
    text = format_heading(1, '%s package' % package)
    # add each module in the package
    for py_file in py_files:
        if shall_skip(path.join(root, py_file)):
            continue
        is_package = py_file == INITPY
        py_file = path.splitext(py_file)[0]
        py_path = makename(subroot, py_file)
        # we don't want an additional header for the package,
        if not is_package:
            heading = ':mod:`%s` module' % py_file
            text += format_heading(2, heading)
        text += format_directive(is_package and subroot or py_path, destdir,
                                 master_package)
        text += '\n'

    # build a list of directories that are packages (contain an INITPY file)
    subs = [sub for sub in subs if path.isfile(path.join(root, sub, INITPY))]
    # if there are some package directories, add a TOC for theses subpackages
    if subs:
        text += format_heading(2, 'Subpackages')
        text += '.. toctree::\n'
        text += '    :maxdepth: 2\n\n'
        for sub in subs:
            if not is_excluded(os.path.join(root, sub), excludes):
                text += '    %s.%s\n' % (makename(master_package, subroot), sub)
        text += '\n'

    write_file(makename(master_package, subroot), text, destdir, suffix)


def shall_skip(module):
    """Check if we want to skip this module."""
    # skip it if there is nothing (or just \n or \r\n) in the file
    return path.getsize(module) <= 2


def recurse_tree(rootpath, excludes, destdir):
    """
    Look for every file in the directory tree and create the corresponding
    ReST files.
    """
    # use absolute path for root, as relative paths like '../../foo' cause
    # 'if "/." in root ...' to filter out *all* modules otherwise
    rootpath = path.normpath(path.abspath(rootpath))
    # check if the base directory is a package and get its name
    if INITPY in os.listdir(rootpath):
        root_package = rootpath.split(path.sep)[-1]
    else:
        # otherwise, the base is a directory with packages
        root_package = None

    toplevels = []
    for root, subs, files in os.walk(rootpath):
        if is_excluded(root, excludes):
            del subs[:]
            continue
        # document only Python module files
        py_files = sorted([f for f in files if path.splitext(f)[1] == '.py'])
        is_pkg = INITPY in py_files
        if is_pkg:
            py_files.remove(INITPY)
            py_files.insert(0, INITPY)
        elif root != rootpath:
            # only accept non-package at toplevel
            del subs[:]
            continue
        # remove hidden ('.') and private ('_') directories
        subs[:] = sorted(sub for sub in subs if sub[0] not in ['.', '_'])

        if is_pkg:
            # we are in a package with something to document
            if subs or len(py_files) > 1 or not \
                shall_skip(path.join(root, INITPY)):
                subpackage = root[len(rootpath):].lstrip(path.sep).\
                    replace(path.sep, '.')
                create_package_file(root, root_package, subpackage,
                                    py_files, subs, destdir, excludes)
                toplevels.append(makename(root_package, subpackage))
        else:
            raise AssertionError('Expected it to be a package')

    return toplevels


def normalize_excludes(rootpath, excludes):
    """
    Normalize the excluded directory list:
    * must be either an absolute path or start with rootpath,
    * otherwise it is joined with rootpath
    * with trailing slash
    """
    f_excludes = []
    for exclude in excludes:
        if not path.isabs(exclude) and not exclude.startswith(rootpath):
            exclude = path.join(rootpath, exclude)
        f_excludes.append(path.normpath(exclude) + path.sep)
    return f_excludes


def is_excluded(root, excludes):
    """
    Check if the directory is in the exclude list.

    Note: by having trailing slashes, we avoid common prefix issues, like
          e.g. an exlude "foo" also accidentally excluding "foobar".
    """
    sep = path.sep
    if not root.endswith(sep):
        root += sep
    for exclude in excludes:
        if root.startswith(exclude):
            return True
    return False


def main(rootpath, excludes, destdir):
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    excludes = normalize_excludes(rootpath, excludes)
    modules = recurse_tree(rootpath, excludes, destdir)

