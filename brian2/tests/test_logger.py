import os

from brian2.utils.logger import BrianLogger, get_logger

import pytest

logger = get_logger('brian2.tests.test_logger')

@pytest.mark.codegen_independent
def test_file_logging():
    logger.error('error message xxx')
    logger.warn('warning message xxx')
    logger.info('info message xxx')
    logger.debug('debug message xxx')
    logger.diagnostic('diagnostic message xxx')
    BrianLogger.file_handler.flush()
    # By default, only >= debug messages should show up
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, 'r') as f:
        log_content = f.readlines()
    for level, line in zip(['error', 'warning', 'info', 'debug'], log_content[-4:]):
        assert 'brian2.tests.test_logger' in line
        assert f"{level} message xxx" in line
        assert level.upper() in line


if __name__ == '__main__':
    test_file_logging()
