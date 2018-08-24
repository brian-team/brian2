import os, nose, sys, subprocess, warnings, unittest
import tempfile, pickle

from nose.plugins import Plugin
from nose.plugins.capture import Capture
from nose.plugins.xunit import Xunit
import numpy as np
warnings.simplefilter('ignore')


class RunTestCase(unittest.TestCase):
    '''
    A test case that simply executes a python script
    '''
    def __init__(self, filename, codegen_target, dtype):
        unittest.TestCase.__init__(self)
        self.filename = filename
        self.codegen_target = codegen_target
        self.dtype = dtype

    def id(self):
        # Remove the .py and pretend the dirname is a package and the filename
        # is a class.
        name = os.path.splitext(os.path.split(self.filename)[1])[0]
        pkgname = os.path.split(os.path.split(self.filename)[0])[1]
        return pkgname + '.' + name.replace('.', '_')

    def shortDescription(self):
        return str(self)

    def runTest(self):
        # a simpler version of what the nosepipe plugin achieves:
        # isolate test execution in a subprocess:
        tempfilename = tempfile.mktemp('exception')

        # Catch any exception and save it to a temporary file
        code_string = """
# needed for some scripts that load data
__file__ = '{fname}'
import matplotlib as _mpl
_mpl.use('Agg')
import warnings, traceback, pickle, sys, os
warnings.simplefilter('ignore')
try:
    import numpy as np
    from brian2 import prefs
    from brian2.utils.filetools import ensure_directory_of_file
    prefs.codegen.target = '{target}'
    prefs.core.default_float_dtype = np.{dtype}
    # Move to the file's directory for the run, so that it can do relative
    # imports and load files (e.g. figure style files)
    curdir = os.getcwd()
    os.chdir(os.path.dirname(r'{fname}'))
    rel_fname = os.path.basename(r'{fname}')
    with open(rel_fname, "rb") as f:
        exec(compile(f.read(), rel_fname, 'exec'))
    os.chdir(curdir)
    if '{target}'=='cython' and {dtype} == np.float64:
        for fignum in _mpl.pyplot.get_fignums():
            fname = r'{fname}'
            fname = os.path.relpath(fname, '../../examples')
            fname = fname.replace('/', '.').replace('\\\\', '.')
            fname = fname.replace('.py', '.%d.png' % fignum)
            fname = '../../docs_sphinx/resources/examples_images/'+fname
            print(fname)
            ensure_directory_of_file(fname)
            _mpl.pyplot.figure(fignum).savefig(fname)
except Exception as ex:
    traceback.print_exc(file=sys.stdout)
    f = open(r'{tempfname}', 'wb')
    pickle.dump(ex, f, -1)
    f.close()
""".format(fname=self.filename,
           tempfname=tempfilename,
           target=self.codegen_target,
           dtype=self.dtype.__name__)

        args = [sys.executable, '-c',
                code_string]
        # Run the example in a new process and make sure that stdout gets
        # redirected into the capture plugin
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        # Write both stdout and stderr to stdout so it gets captured by the
        # Capture plugin
        sys.stdout.write(stdout)
        sys.stdout.write(stderr)

        # Re-raise any exception that occured
        if os.path.exists(tempfilename):
            with open(tempfilename, 'rb') as f:
                ex = pickle.load(f)
            self.successful = False
            raise ex
        else:
            self.successful = True

    def __str__(self):
        return 'Example: %s (%s, %s)' % (self.filename, self.codegen_target,
                                         self.dtype.__name__)


class SelectFilesPlugin(Plugin):
    '''
    This plugin makes nose descend into all directories and exectue all files.
    '''
    # no command line arg needed to activate plugin
    enabled = True
    name = "select-files"

    def configure(self, options, conf):
        pass # always on

    def wantDirectory(self, dirname):
        # we want all directories
        return True

    def find_examples(self, name):
        examples = []
        if os.path.isdir(name):
            for subname in os.listdir(name):
                examples.extend(self.find_examples(os.path.join(name, subname)))
            return examples
        elif name.endswith('.py'):  # only execute Python scripts
            return [name]
        else:
            return []

    def loadTestsFromName(self, name, module=None, discovered=False):
        all_examples = self.find_examples(name)
        all_tests = []
        for target in ['numpy', 'cython']:
            for dtype in [np.float32, np.float64]:
                for example in all_examples:
                    all_tests.append(RunTestCase(example, target, dtype))
        return all_tests


if __name__ == '__main__':
    argv = [__file__, '-v', '--with-xunit', '--verbose', '--exe', '../../examples']

    if not nose.main(argv=argv,
                     plugins=[SelectFilesPlugin(), Capture(), Xunit()]):
        sys.exit(1)
