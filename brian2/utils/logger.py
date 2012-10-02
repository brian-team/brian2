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


def clean_up_logging():
    logging.shutdown()
    if not BrianLogger.exception_occured:
        os.remove(TMP_LOG)

sys.excepthook = brian_excepthook
atexit.register(clean_up_logging)


class BrianLogger(object):

    exception_occured = False

    def __init__(self, name):
        self.name = name

    def debug(self, msg, name_suffix=None):
        name = self.name
        if name_suffix:
            name += '.' + name_suffix
        logging.getLogger(name).debug(msg)

    def info(self, msg, name_suffix=None):
        name = self.name
        if name_suffix:
            name += '.' + name_suffix
        logging.getLogger(name).info(msg)

    def warn(self, msg, name_suffix=None):
        name = self.name
        if name_suffix:
            name += '.' + name_suffix
        logging.getLogger(name).warn(msg)

    def error(self, msg, name_suffix=None):
        name = self.name
        if name_suffix:
            name += '.' + name_suffix
        logging.getLogger(name).error(msg)

    @staticmethod
    def log_level_debug():
        CONSOLE_HANDLER.setLevel(logging.DEBUG)

    @staticmethod
    def log_level_info():
        CONSOLE_HANDLER.setLevel(logging.INFO)

    @staticmethod
    def log_level_warn():
        CONSOLE_HANDLER.setLevel(logging.INFO)

    @staticmethod
    def log_level_error():
        CONSOLE_HANDLER.setLevel(logging.INFO)


def get_logger(module_name):
    return BrianLogger(module_name)
