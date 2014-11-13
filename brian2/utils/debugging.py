'''
Some tools for debugging.
'''
import os, sys, tempfile
from cStringIO import StringIO

__all__ = ['RedirectStdStreams', 'std_silent']

class RedirectStdStreams(object):
    '''
    Context manager which allows you to temporarily redirect stdin/stdout and restore them on exit.
    '''
    def __init__(self, stdout=None, stderr=None):
        if stdout=='null':
            stdout = open(os.devnull, 'w')
        if stderr=='null':
            stderr = open(os.devnull, 'w')
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


# See http://stackoverflow.com/questions/26126160/redirecting-standard-out-in-err-back-after-os-dup2
# for an explanation of how this function works. Note that 1 and 2 are the file
# numbers for stdout and stderr
class std_silent(object):
    '''
    Context manager that temporarily silences stdout and stderr but keeps the
    output saved in a temporary file and writes it if an exception is raised.
    '''
    def __init__(self, alwaysprint=False):
        self.alwaysprint = alwaysprint
        if not hasattr(std_silent, 'dest_stdout'):
            std_silent.dest_fname_stdout = tempfile.mktemp()
            std_silent.dest_fname_stderr = tempfile.mktemp()
            std_silent.dest_stdout = open(std_silent.dest_fname_stdout, 'w')
            std_silent.dest_stderr = open(std_silent.dest_fname_stderr, 'w')
        
    def __enter__(self):
        if not self.alwaysprint:
            sys.stdout.flush()
            sys.stderr.flush()
            self.orig_out_fd = os.dup(1)
            self.orig_err_fd = os.dup(2)
            os.dup2(std_silent.dest_stdout.fileno(), 1)
            os.dup2(std_silent.dest_stderr.fileno(), 2)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.alwaysprint:
            std_silent.dest_stdout.flush()
            std_silent.dest_stderr.flush()
            if exc_type is not None:
                out = open(std_silent.dest_fname_stdout, 'r').read()
                err = open(std_silent.dest_fname_stderr, 'r').read()
            os.dup2(self.orig_out_fd, 1)
            os.dup2(self.orig_err_fd, 2)
            os.close(self.orig_out_fd)
            os.close(self.orig_err_fd)
            if exc_type is not None:
                sys.stdout.write(out)
                sys.stderr.write(err)
        