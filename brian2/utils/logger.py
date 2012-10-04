import atexit
import logging
import os
import shutil
import sys
import tempfile
import time
from warnings import warn

import numpy
import scipy
import sympy

import brian2
from brian2.core.preferences import brian_prefs

__all__ = ['get_logger', 'BrianLogger']

#===============================================================================
# Global options for logging
#===============================================================================
brian_prefs.define('delete_log_on_exit', True,
    '''
    Whether to delete the log and script file on exit.
    
    If set to ``True`` (the default), log files (and the copy of the main
    script) will be deleted after the brian process has exited, unless an
    uncaught exception occured. If set to ``False``, all log files will be kept.
    ''')

#===============================================================================
# Initial setup
#===============================================================================

# get the root logger
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

# Log to a file
try:
    # Temporary filename used for logging
    TMP_LOG = tempfile.NamedTemporaryFile(prefix='brian_debug_', suffix='.log',
                                          delete=False)
    TMP_LOG = TMP_LOG.name
    FILE_HANDLER = logging.FileHandler(TMP_LOG, mode='w+b')
    FILE_HANDLER.setLevel(logging.DEBUG)
    FILE_HANDLER.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s'))
    logger.addHandler(FILE_HANDLER)
except IOError as ex:
    warn('Could not create log file: %s' % ex)
    TMP_LOG = None

# Save a copy of the script
try:
    tmp_file = tempfile.NamedTemporaryFile(prefix='brian_script_', suffix='.py',
                                           delete=False)
    with tmp_file:
        # Timestamp
        tmp_file.write('# %s\n' % time.asctime())
        # Command line arguments
        tmp_file.write('# Run as: %s\n\n' % (' '.join(sys.argv)))
        # The actual script file
        with open(os.path.abspath(sys.argv[0])) as script_file:
            shutil.copyfileobj(script_file, tmp_file)    
        TMP_SCRIPT = tmp_file.name
except IOError as ex:
    warn('Could not copy script file to temp directory: %s' % ex)
    TMP_SCRIPT = None

# create console handler with a higher log level
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(logging.WARN)
CONSOLE_HANDLER.setFormatter(logging.Formatter('%(levelname)-8s %(name)s: %(message)s'))

# add the handler to the logger
logger.addHandler(CONSOLE_HANDLER)

# We want to log all warnings
if hasattr(logging, 'captureWarnings'):
    # This function was added in Python 2.7
    logging.captureWarnings(True)

# Put some standard info into the log file
logger = logging.getLogger('brian2')
logger.debug('Logging to file: %s, copy of main script saved as: %s' %
             (TMP_LOG, TMP_SCRIPT))
logger.debug('Python interpreter: %s' % sys.executable)
logger.debug('Platform: %s' % sys.platform)
version_infos = {'brian': brian2.__version__,
                 'numpy': numpy.__version__,
                 'scipy': scipy.__version__,
                 'sympy': sympy.__version__,
                 'python': sys.version,
                 }
for name, version in version_infos.iteritems():
    logger.debug('{name} version is: {version}'.format(name=name,
                                                       version=str(version)))


UNHANDLED_ERROR_MESSAGE =  ('Brian encountered an unexpected error. '
'If you think this is bug in Brian, please report this issue either to the '
'mailing list at <http://groups.google.com/group/brian-support/>, '
'or to the issue tracker at <http://neuralensemble.org/trac/brian/report>. '
'Please include this file with debug information in your report: {logfile} '
' Additionally, you can also include a copy of the script that was run, '
'available at: {filename} Thanks!').format(logfile=TMP_LOG, filename=TMP_SCRIPT)


def brian_excepthook(exc_type, exc_obj, exc_tb):
    BrianLogger.exception_occured = True
    
    logger.error(UNHANDLED_ERROR_MESSAGE,
                 exc_info=(exc_type, exc_obj, exc_tb))

def clean_up_logging():
    logging.shutdown()
    if not BrianLogger.exception_occured and brian_prefs.delete_log_on_exit:
        os.remove(TMP_LOG)
        os.remove(TMP_SCRIPT)

sys.excepthook = brian_excepthook
atexit.register(clean_up_logging)

class InvertedFilter(object):
    '''
    A class for suppressing log messages. Does exactly the opposite as the
    `logging.Filter` class, which allows messages in a certain name hierarchy
    to *pass*.
    
    Parameters
    ----------
    name : str
        The name hiearchy to suppress. See `BrianLogger.suppress_messages` for
        details.
    '''

    def __init__(self, name):
        self.orig_filter = logging.Filter(name)
    
    def filter(self, record):
        # do the opposite of what the standard filter class would do
        return not self.orig_filter.filter(record)


class BrianLogger(object):
    '''
    Convenience object for logging. Call `get_logger` to get an instance of
    this class.
    
    Parameters
    ----------
    name : str
        The name used for logging, normally the name of the module.
    
    Methods
    -------
    debug
    info
    warn
    error
    log_level_debug
    log_level_info
    log_level_warn
    log_level_error
    suppress_messages
    '''
    
    # : Class attribute to remember whether any exception occured
    exception_occured = False
    
    # : Class attribute for remembering log messages that should only be
    # : displayed once
    _log_messages = set()

    def __init__(self, name):
        self.name = name


    def _log(self, log_level, msg, name_suffix, once):
        name = self.name
        if name_suffix:
            name += '.' + name_suffix
        
        if once:
            # Check whether this exact message has already been displayed 
            log_tuple = (name, log_level, msg)
            if log_tuple in BrianLogger._log_messages:
                return
            else:
                BrianLogger._log_messages.add(log_tuple)
        
        logger = logging.getLogger(name)
        {'debug': logger.debug,
         'info': logger.info,
         'warn': logger.warn,
         'error': logger.error}.get(log_level)(msg)

    def debug(self, msg, name_suffix=None, once=False):
        '''
        Log a debug message.
        
        Parameters
        ----------
        msg : str
            The message to log.
        name_suffix : str, optional
            A suffix to add to the name, e.g. a class or function name.
        once : bool, optional
            Whether this message should be logged only once and not repeated
            if sent another time. 
        '''
        self._log('debug', msg, name_suffix, once)

    def info(self, msg, name_suffix=None, once=False):
        '''
        Log an info message.
        
        Parameters
        ----------
        msg : str
            The message to log.
        name_suffix : str, optional
            A suffix to add to the name, e.g. a class or function name.
        once : bool, optional
            Whether this message should be logged only once and not repeated
            if sent another time. 
        '''
        self._log('info', msg, name_suffix, once)

    def warn(self, msg, name_suffix=None, once=False):
        '''
        Log a warn message.
        
        Parameters
        ----------
        msg : str
            The message to log.
        name_suffix : str, optional
            A suffix to add to the name, e.g. a class or function name.
        once : bool, optional
            Whether this message should be logged only once and not repeated
            if sent another time. 
        '''
        self._log('warn', msg, name_suffix, once)

    def error(self, msg, name_suffix=None, once=False):
        '''
        Log an error message.
        
        Parameters
        ----------
        msg : str
            The message to log.
        name_suffix : str, optional
            A suffix to add to the name, e.g. a class or function name.
        once : bool, optional
            Whether this message should be logged only once and not repeated
            if sent another time. 
        '''
        self._log('error', msg, name_suffix, once)

    @staticmethod
    def suppress_messages(name, filter_log_file=False):
        '''
        Suppress a subset of log messages.
        
        Parameters
        ----------
        name : str
            Suppress all log messages in the given `name` hierarchy. For
            example, specifying ``'brian2'`` suppresses all messages logged
            by Brian, specifying ``'brian2.codegen'`` suppresses all messages
            generated by the code generation modules.
        filter_log_file : bool, optional
            Whether to suppress the messages also in the log file. Defaults to
            ``False`` meaning that suppressed messages are not displayed on
            the console but are still saved to the log file.
        '''
        
        suppress_filter = InvertedFilter(name)
        
        CONSOLE_HANDLER.addFilter(suppress_filter)
        
        if filter_log_file:
            FILE_HANDLER.addFilter(suppress_filter)

    @staticmethod
    def log_level_debug():
        '''
        Set the log level to "debug".
        '''
        CONSOLE_HANDLER.setLevel(logging.DEBUG)

    @staticmethod
    def log_level_info():
        '''
        Set the log level to "info".
        '''        
        CONSOLE_HANDLER.setLevel(logging.INFO)

    @staticmethod
    def log_level_warn():
        '''
        Set the log level to "warn".
        '''        
        CONSOLE_HANDLER.setLevel(logging.INFO)

    @staticmethod
    def log_level_error():
        '''
        Set the log level to "error".
        '''        
        CONSOLE_HANDLER.setLevel(logging.INFO)


def get_logger(module_name='brian2'):
    '''
    Get an object that can be used for logging.
    
    Parameters
    ----------
    module_name : str
        The name used for logging, should normally be the module name as
        returned by ``__name__``.
    
    Returns
    -------
    logger : `BrianLogger`
    '''

    return BrianLogger(module_name)
