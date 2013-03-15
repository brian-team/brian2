'''
Brian's logging module.
'''

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

from .environment import running_from_ipython

__all__ = ['get_logger', 'BrianLogger']

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
TMP_SCRIPT = None
if len(sys.argv[0]) and not running_from_ipython():
    try:
        tmp_file = tempfile.NamedTemporaryFile(prefix='brian_script_',
                                               suffix='.py',
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

# create console handler with a higher log level
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(logging.WARN)
CONSOLE_HANDLER.setFormatter(logging.Formatter('%(levelname)-8s %(name)s: %(message)s'))

# add the handler to the logger
logger.addHandler(CONSOLE_HANDLER)

# We want to log all warnings
if hasattr(logging, 'captureWarnings'):
    # This function was added in Python 2.7
    logging.captureWarnings(True) # pylint: disable=E1101

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
for _name, _version in version_infos.iteritems():
    logger.debug('{name} version is: {version}'.format(name=_name,
                                                       version=str(_version)))


UNHANDLED_ERROR_MESSAGE = ('Brian encountered an unexpected error. '
'If you think this is bug in Brian, please report this issue either to the '
'mailing list at <http://groups.google.com/group/brian-support/>, '
'or to the issue tracker at <http://neuralensemble.org/trac/brian/report>. '
'Please include this file with debug information in your report: {logfile} '
' Additionally, you can also include a copy of the script that was run, '
'available at: {filename} Thanks!').format(logfile=TMP_LOG, filename=TMP_SCRIPT)


def brian_excepthook(exc_type, exc_obj, exc_tb):
    '''
    Display a message mentioning the debug log in case of an uncaught
    exception.
    '''
    BrianLogger.exception_occured = True
    
    logger.error(UNHANDLED_ERROR_MESSAGE,
                 exc_info=(exc_type, exc_obj, exc_tb))

def clean_up_logging():
    '''
    Shutdown the logging system and delete the debug log file if no error
    occured.
    '''
    logging.shutdown()
    if not BrianLogger.exception_occured and brian_prefs['core.delete_log_on_exit']:
        if not TMP_LOG is None:
            try:
                os.remove(TMP_LOG)
            except IOError as exc:
                warn('Could not delete log file: %s' % exc)
        if not TMP_SCRIPT is None:
            try:
                os.remove(TMP_SCRIPT)
            except IOError as exc:
                warn('Could not delete copy of script file: %s' % exc)

sys.excepthook = brian_excepthook
atexit.register(clean_up_logging)


class HierarchyFilter(object):
    '''
    A class for suppressing all log messages in a subtree of the name hierarchy.
    Does exactly the opposite as the `logging.Filter` class, which allows
    messages in a certain name hierarchy to *pass*.
    
    Parameters
    ----------
    name : str
        The name hiearchy to suppress. See `BrianLogger.suppress_hierarchy` for
        details.
    '''

    def __init__(self, name):
        self.orig_filter = logging.Filter(name)
    
    def filter(self, record):
        '''
        Filter out all messages in a subtree of the name hierarchy.
        '''
        # do the opposite of what the standard filter class would do
        return not self.orig_filter.filter(record)


class NameFilter(object):
    '''
    A class for suppressing log messages ending with a certain name.
    
    Parameters
    ----------
    name : str
        The name to suppress. See `BrianLogger.suppress_name` for details.
    '''
    
    def __init__(self, name):
        self.name = name
    
    def filter(self, record):
        '''
        Filter out all messages ending with a certain name.
        '''
        # The last part of the name
        record_name = record.name.split('.')[-1]
        return self.name != record_name


class BrianLogger(object):
    '''
    Convenience object for logging. Call `get_logger` to get an instance of
    this class.
    
    Parameters
    ----------
    name : str
        The name used for logging, normally the name of the module.
    '''
    
    # : Class attribute to remember whether any exception occured
    exception_occured = False
    
    # : Class attribute for remembering log messages that should only be
    # : displayed once
    _log_messages = set()

    def __init__(self, name):
        self.name = name

    def _log(self, log_level, msg, name_suffix, once):
        '''
        Log an entry.
        
        Parameters
        ----------
        log_level : {'debug', 'info', 'warn', 'error'}
            The level with which to log the message.
        msg : str
            The log message.
        name_suffix : str
            A suffix that will be added to the logger name.
        once : bool
            Whether to suppress identical messages if they are logged again.
        '''
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
        
        the_logger = logging.getLogger(name)
        {'debug': the_logger.debug,
         'info': the_logger.info,
         'warn': the_logger.warn,
         'error': the_logger.error}.get(log_level)(msg)

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
    def _suppress(filterobj, filter_log_file):
        '''
        Apply a filter object to log messages.
        
        Parameters
        ----------
        filterobj : `logging.Filter`
            A filter object to apply to log messages.
        filter_log_file : bool
            Whether the filter also applies to log messages in the log file.
        '''
        CONSOLE_HANDLER.addFilter(filterobj)
        
        if filter_log_file:
            FILE_HANDLER.addFilter(filterobj)

    @staticmethod
    def suppress_hierarchy(name, filter_log_file=False):
        '''
        Suppress all log messages in a given hiearchy.
        
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
        
        suppress_filter = HierarchyFilter(name)
        
        BrianLogger._suppress(suppress_filter, filter_log_file)

    @staticmethod
    def suppress_name(name, filter_log_file=False):
        '''
        Suppress all log messages with a given name.
        
        Parameters
        ----------
        name : str
            Suppress all log messages ending in the given `name`. For
            example, specifying ``'resolution_conflict'`` would suppress
            messages with names such as
            ``brian2.equations.codestrings.CodeString.resolution_conflict`` or
            ``brian2.equations.equations.Equations.resolution_conflict``.
        filter_log_file : bool, optional
            Whether to suppress the messages also in the log file. Defaults to
            ``False`` meaning that suppressed messages are not displayed on
            the console but are still saved to the log file.
        '''
        suppress_filter = NameFilter(name)
        
        BrianLogger._suppress(suppress_filter, filter_log_file)

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


class catch_logs(object):
    '''
    A context manager for catching log messages. Use this for testing the
    messages that are logged. Defaults to catching warning/error messages and
    this is probably the only real use case for testing. Note that while this
    context manager is active, *all* log messages are suppressed. Using this
    context manager returns a list of (log level, name, message) tuples.
    
    Parameters
    ----------
    log_level : int or str, optional
        The log level above which messages are caught.
    
    Examples
    --------
    >>> logger = get_logger('brian2.logtest')
    >>> logger.warn('An uncaught warning') # doctest: +SKIP
    WARNING  brian2.logtest: An uncaught warning
    >>> with catch_logs() as l:
    ...    logger.warn('a caught warning')
    ...    print 'l contains:', l
    ... 
    l contains: [('WARNING', 'brian2.logtest', 'a caught warning')]

    '''
    _entered = False
    
    def __init__(self, log_level=logging.WARN):
        self.log_list = []
        self.handler = LogCapture(self.log_list, log_level)
        self._entered = False
    
    def __enter__(self):
        if self._entered:
            raise RuntimeError('Cannot enter %r twice' % self)
        self._entered = True
        return self.log_list
    
    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError('Cannot exit %r without entering first' % self)
        self.handler.uninstall()


class LogCapture(logging.Handler):
    '''
    A class for capturing log warnings. This class is used by
    `~brian2.utils.logger.catch_logs` to allow testing in a similar
    way as with `warnings.catch_warnings`.
    '''
    
    def __init__(self, log_list, log_level=logging.WARN):
        logging.Handler.__init__(self, level=log_level)
        self.log_list = log_list
        # make a copy of the previous handlers
        self.old_handlers = list(logging.getLogger().handlers)
        self.install()

    def emit(self, record):
        # Append a tuple consisting of (level, name, msg) to the list of
        # warnings
        self.log_list.append((record.levelname, record.name, record.msg))
    
    def install(self):
        '''
        Install this handler to catch all warnings. Temporarily disconnect all
        other handlers.
        '''
        the_logger = logging.getLogger()
        for handler in self.old_handlers:
            the_logger.removeHandler(handler)
        # make sure everything gets logged by the root logger
        the_logger.setLevel(logging.DEBUG)
        the_logger.addHandler(self)
    
    def uninstall(self):
        '''
        Uninstall this handler and re-connect the previously installed
        handlers.
        '''
        the_logger = logging.getLogger()
        the_logger.removeHandler(self)
        for handler in self.old_handlers:
            the_logger.addHandler(handler)
