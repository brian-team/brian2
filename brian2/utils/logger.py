import atexit
import logging
import os
import sys
import tempfile
from warnings import warn

import numpy
import scipy
import sympy

import brian2
from brian2.core.preferences import brian_prefs

__all__ = ['get_logger']

#===============================================================================
# Initial setup
#===============================================================================

try:
    # Temporary filename used for logging
    TMP_LOG = tempfile.NamedTemporaryFile(prefix='brian_debug_', suffix='.log',
                                          delete=False)
    TMP_LOG = TMP_LOG.name
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
                    filename=TMP_LOG,
                    filemode='w')
except IOError as ex:
    warn('Could not create log file: %s' % ex)
    TMP_LOG = None

# create console handler with a higher log level
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(logging.WARN)
CONSOLE_HANDLER.setFormatter(logging.Formatter('%(levelname)-8s %(name)s: %(message)s'))

# add the handler to the logger
logger = logging.getLogger('brian2')
logger.addHandler(CONSOLE_HANDLER)


# Put some standard info into the log file
logger.debug('Python interpreter: %s' % sys.executable)
logger.debug('Platform: %s' % sys.platform)
version_infos = {'brian': brian2.__version__,
                 'numpy': numpy.__version__,
                 'scipy': scipy.__version__,
                 'sympy': sympy.__version__}
for name, version in version_infos.iteritems():
    logger.debug('{name} version is: {version}'.format(name=name,
                                                       version=str(version)))


UNHANDLED_ERROR_MESSAGE = '''
Brian encountered an unexpected error. If you think this is bug in Brian, please
report this issue either to the mailing list at <http://groups.google.com/group/brian-support/>,
or to the issue tracker at <http://neuralensemble.org/trac/brian/report>.
Please include this file with debug information in your report: {filename}
Thanks!
'''.format(filename=TMP_LOG)


def brian_excepthook(exc_type, exc_obj, exc_tb):
    BrianLogger.exception_occured = True
    logger.error(UNHANDLED_ERROR_MESSAGE,
                 exc_info=(exc_type, exc_obj, exc_tb))

# A global option to switch off the automatic deletion
brian_prefs.define('delete_log_on_exit', True,
    '''
    Whether to delete the log file on exit.
    
    If set to ``True`` (the default), log files will be deleted after the
    brian process has exited, unless an uncaught exception occured. If set to
    ``False``, all log files will be kept.
    ''')

def clean_up_logging():
    logging.shutdown()
    if not BrianLogger.exception_occured and brian_prefs.delete_log_on_exit:
        os.remove(TMP_LOG)

sys.excepthook = brian_excepthook
atexit.register(clean_up_logging)


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
    logger : `~brian2.utils.logger.BrianLogger`
    '''    
    return BrianLogger(module_name)
