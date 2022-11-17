"""
File system tools
"""


import os

__all__ = [
    "ensure_directory",
    "ensure_directory_of_file",
    "in_directory",
    "copy_directory",
]


def ensure_directory_of_file(f):
    """
    Ensures that a directory exists for filename to go in (creates if
    necessary), and returns the directory path.
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def ensure_directory(d):
    """
    Ensures that a given directory exists (creates it if necessary)
    """
    if not os.path.exists(d):
        os.makedirs(d)
    return d


class in_directory(object):
    """
    Safely temporarily work in a subdirectory

    Usage::

        with in_directory(directory):
            ... do stuff here

    Guarantees that the code in the with block will be executed in directory,
    and that after the block is completed we return to the original directory.
    """

    def __init__(self, new_dir):
        self.orig_dir = os.getcwd()
        self.new_dir = new_dir

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *exc_info):
        os.chdir(self.orig_dir)


def copy_directory(source, target):
    """
    Copies directory source to target.
    """
    relnames = []
    sourcebase = os.path.normpath(source) + os.path.sep
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            fullname = os.path.normpath(os.path.join(root, filename))
            relname = fullname.replace(sourcebase, "")
            relnames.append(relname)
            tgtname = os.path.join(target, relname)
            ensure_directory_of_file(tgtname)
            with open(fullname) as f:
                contents = f.read()
            if os.path.exists(tgtname):
                with open(tgtname) as f:
                    if f.read() == contents:
                        continue
            with open(tgtname, "w") as f:
                f.write(contents)
    return relnames
