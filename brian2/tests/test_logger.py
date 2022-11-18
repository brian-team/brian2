import multiprocessing
import os

from brian2.utils.logger import BrianLogger, get_logger
from brian2.core.preferences import prefs

import pytest

logger = get_logger("brian2.tests.test_logger")


@pytest.mark.codegen_independent
def test_file_logging():
    logger.error("error message xxx")
    logger.warn("warning message xxx")
    logger.info("info message xxx")
    logger.debug("debug message xxx")
    logger.diagnostic("diagnostic message xxx")
    BrianLogger.file_handler.flush()
    # By default, only >= debug messages should show up
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, "r") as f:
        log_content = f.readlines()
    for level, line in zip(["error", "warning", "info", "debug"], log_content[-4:]):
        assert "brian2.tests.test_logger" in line
        assert f"{level} message xxx" in line
        assert level.upper() in line


def run_in_process(x):
    logger.info(f"subprocess info message {x}")


def run_in_process_with_logger(x):
    prefs.logging.delete_log_on_exit = False
    BrianLogger.initialize()
    logger.info(f"subprocess info message {x}")
    BrianLogger.file_handler.flush()
    return BrianLogger.tmp_log


@pytest.mark.codegen_independent
def test_file_logging_multiprocessing():
    logger.info("info message before multiprocessing")

    with multiprocessing.Pool() as p:
        p.map(run_in_process, range(3))

    BrianLogger.file_handler.flush()
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, "r") as f:
        log_content = f.readlines()
    # The subprocesses should not have written to the log file
    assert "info message before multiprocessing" in log_content[-1]


@pytest.mark.codegen_independent
def test_file_logging_multiprocessing_with_loggers():
    logger.info("info message before multiprocessing")

    with multiprocessing.Pool() as p:
        log_files = p.map(run_in_process_with_logger, range(3))

    BrianLogger.file_handler.flush()
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, "r") as f:
        log_content = f.readlines()
    # The subprocesses should not have written to the main log file
    assert "info message before multiprocessing" in log_content[-1]

    # Each subprocess should have their own log file
    for x, log_file in enumerate(log_files):
        assert os.path.isfile(log_file)
        with open(log_file, "r") as f:
            log_content = f.readlines()
        assert f"subprocess info message {x}" in log_content[-1]

    prefs.logging.delete_log_on_exit = True


if __name__ == "__main__":
    test_file_logging()
    test_file_logging_multiprocessing()
    test_file_logging_multiprocessing_with_loggers()
