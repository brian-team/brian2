import os, sys, subprocess, warnings, unittest
import tempfile, pickle
import pytest

from brian2 import device, set_device

import numpy as np
warnings.simplefilter('ignore')


class ExampleRun(pytest.Item):
    '''
    A test case that simply executes a python script
    '''
    @classmethod
    def from_parent(cls, filename, codegen_target, dtype, parent):
        super_class = super(ExampleRun, cls)
        if hasattr(super_class, 'from_parent'):
            new_node = super_class.from_parent(parent=parent,
                                               name=ExampleRun.id(filename))
        else:
            # For pytest < 6
            new_node = cls(parent=parent, name=ExampleRun.id(filename))
        new_node.filename = filename
        new_node.codegen_target = codegen_target
        new_node.dtype = dtype
        return new_node

    @staticmethod
    def id(filename):
        # Remove the .py and pretend the dirname is a package and the filename
        # is a class.
        name = os.path.splitext(os.path.split(filename)[1])[0]
        pkgname = os.path.split(os.path.split(filename)[0])[1]
        return pkgname + '.' + name.replace('.', '_')

    def shortDescription(self):
        return str(self)

    def runtest(self):
        import matplotlib as _mpl
        _mpl.use('Agg')
        import numpy as np
        from brian2 import prefs
        from brian2.utils.filetools import ensure_directory_of_file
        prefs.codegen.target = self.codegen_target
        prefs.core.default_float_dtype = self.dtype
        # Move to the file's directory for the run, so that it can do relative
        # imports and load files (e.g. figure style files)
        curdir = os.getcwd()
        os.chdir(os.path.dirname(self.filename))
        sys.path.append(os.path.dirname(self.filename))
        import warnings
        warnings.simplefilter('ignore')
        try:
            with open(self.filename, 'r') as f:
                exec(f.read())
            if self.codegen_target == 'cython' and self.dtype == np.float64:
                for fignum in _mpl.pyplot.get_fignums():
                    fname = os.path.relpath(self.filename, self.example_dir)
                    fname = fname.replace('/', '.').replace('\\\\', '.')
                    fname = fname.replace('.py', '.%d.png' % fignum)
                    fname = os.path.abspath(os.path.join(self.example_dir,
                                                         '../docs_sphinx/resources/examples_images/',
                                                         fname))
                    ensure_directory_of_file(fname)
                    _mpl.pyplot.figure(fignum).savefig(fname)
        finally:
            os.chdir(curdir)
            sys.path.remove(os.path.dirname(self.filename))
            device.reinit()
            set_device('runtime')

    def __str__(self):
        return 'Example: %s (%s, %s)' % (self.filename, self.codegen_target,
                                         self.dtype.__name__)


class ExampleCollector(pytest.Collector):
    @classmethod
    def from_parent(cls, example_dir, parent):
        collector = super(ExampleCollector, cls)
        if hasattr(collector, 'from_parent'):
            new_collector = collector.from_parent(parent,
                                                  name='example_collector')
        else:
            # For pytest < 6
            new_collector = cls(parent=parent, name='example_collector')
        new_collector.example_dir = example_dir
        return new_collector

    def collect(self):
        for dirname, dirs, files in os.walk(self.example_dir):
            for filename in files:
                if filename.endswith('.py'):
                    run = ExampleRun.from_parent(os.path.join(dirname, filename),
                                                 'cython',
                                                 np.float64,
                                                 parent=self)
                    run.example_dir = self.example_dir
                    yield run


class Plugin:
    def __init__(self, example_dir):
        self.example_dir = example_dir

    def pytest_collect_file(self, path, parent):
        return ExampleCollector.from_parent(self.example_dir, parent=parent)


if __name__ == '__main__':
    example_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'examples'))
    if not pytest.main([__file__, '--verbose'], plugins=[Plugin(example_dir)]):
        sys.exit(1)
