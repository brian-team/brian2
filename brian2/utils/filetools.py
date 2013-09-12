'''
File system tools
'''

import os

__all__ = ['ensure_directory_of_file', 'copy_directory']


def ensure_directory_of_file(f):
    '''
    Ensures that a directory exists for filename to go in (creates if
    necessary), and returns the directory path.
    '''
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def copy_directory(source, target):
    '''
    Copies directory source to target.
    '''
    sourcebase = os.path.normpath(source)+os.path.sep
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            fullname = os.path.normpath(os.path.join(root, filename))
            relname = fullname.replace(sourcebase, '')
            tgtname = os.path.join(target, relname)
            ensure_directory_of_file(tgtname)
            open(tgtname, 'w').write(open(fullname).read())
