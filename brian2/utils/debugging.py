'''
Some tools for debugging.
'''
import os, sys
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


class std_silent(RedirectStdStreams):
    '''
    Context manager which temporarily silences stdin/stdout unless there is an exception.
    '''
    def __init__(self, alwaysprint=False):
        self.alwaysprint = alwaysprint
        self.newout = StringIO()
        self.newerr = StringIO()
        RedirectStdStreams.__init__(self,
                                    stdout=self.newout,
                                    stderr=self.newerr)
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None or self.alwaysprint:
            self.newout.flush()
            self.newerr.flush()
            self.old_stdout.write(self.newout.getvalue())
            self.old_stderr.write(self.newerr.getvalue())
            self.old_stdout.flush()
            self.old_stderr.flush()
        RedirectStdStreams.__exit__(self, exc_type, exc_value, traceback)
        