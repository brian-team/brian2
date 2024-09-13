import multiprocessing
import os

import pytest

from brian2.core.preferences import prefs
from brian2.utils.logger import BrianLogger, catch_logs, get_logger

logger = get_logger("brian2.tests.test_logger")


@pytest.mark.codegen_independent
def test_file_logging():
    BrianLogger.initialize()
    logger.error("error message xxx")
    logger.warn("warning message xxx")
    logger.info("info message xxx")
    logger.debug("debug message xxx")
    logger.diagnostic("diagnostic message xxx")
    BrianLogger.file_handler.flush()
    # By default, only >= debug messages should show up
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    for level, line in zip(["error", "warning", "info", "debug"], log_content[-4:]):
        assert "brian2.tests.test_logger" in line
        assert f"{level} message xxx" in line
        assert level.upper() in line


@pytest.mark.codegen_independent
def test_file_logging_special_characters():
    BrianLogger.initialize()
    # Test logging with special characters that could occur in log messages and
    # require UTF-8
    special_chars = "→ ≠ ≤ ≥ ← ∞ µ ∝ ∂ ∅"
    logger.debug(special_chars)
    BrianLogger.file_handler.flush()
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    last_line = log_content[-1]
    assert "brian2.tests.test_logger" in last_line
    assert special_chars in last_line


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
    p = multiprocessing.Pool()

    try:
        p.map(run_in_process, range(3))
    finally:
        p.close()
        p.join()

    BrianLogger.file_handler.flush()
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    # The subprocesses should not have written to the log file
    assert "info message before multiprocessing" in log_content[-1]


@pytest.mark.codegen_independent
def test_file_logging_multiprocessing_with_loggers():
    logger.info("info message before multiprocessing")

    p = multiprocessing.Pool()
    try:
        log_files = p.map(run_in_process_with_logger, range(3))
    finally:
        p.close()
        p.join()

    BrianLogger.file_handler.flush()
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    # The subprocesses should not have written to the main log file
    assert "info message before multiprocessing" in log_content[-1]

    # Each subprocess should have their own log file
    for x, log_file in enumerate(log_files):
        assert os.path.isfile(log_file)
        with open(log_file, encoding="utf-8") as f:
            log_content = f.readlines()
        assert f"subprocess info message {x}" in log_content[-1]

    prefs.logging.delete_log_on_exit = True


@pytest.mark.codegen_independent
def test_submodule_logging():
    submodule_logger = get_logger("submodule.dummy")
    BrianLogger.initialize()
    submodule_logger.error("error message xxx")
    submodule_logger.warn("warning message xxx")
    submodule_logger.info("info message xxx")
    submodule_logger.debug("debug message xxx")
    submodule_logger.diagnostic("diagnostic message xxx")
    BrianLogger.file_handler.flush()
    # By default, only >= debug messages should show up
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    for level, line in zip(["error", "warning", "info", "debug"], log_content[-4:]):
        assert "submodule.dummy" in line
        # The logger name has brian2 internally prefixed, but this shouldn't show up in logs
        assert not "brian2.submodule.dummy" in line
        assert f"{level} message xxx" in line
        assert level.upper() in line

    with catch_logs() as l:
        logger.warn("warning message from Brian")
        submodule_logger.warn("warning message from submodule")
    # only the warning from Brian should be logged
    assert len(l) == 1
    assert "warning message from Brian" in l[0]

    with catch_logs(only_from=("submodule",)) as l:
        logger.warn("warning message from Brian")
        submodule_logger.warn("warning message from submodule")
    # only the warning from submodule should be logged
    assert len(l) == 1
    assert "warning message from submodule" in l[0]

    # Make sure that a submodule with a name starting with "brian2" gets handled correctly
    submodule_logger = get_logger("brian2submodule.dummy")
    BrianLogger.initialize()
    submodule_logger.error("error message xxx")
    submodule_logger.warn("warning message xxx")
    submodule_logger.info("info message xxx")
    submodule_logger.debug("debug message xxx")
    submodule_logger.diagnostic("diagnostic message xxx")
    BrianLogger.file_handler.flush()
    # By default, only >= debug messages should show up
    assert os.path.isfile(BrianLogger.tmp_log)
    with open(BrianLogger.tmp_log, encoding="utf-8") as f:
        log_content = f.readlines()
    for level, line in zip(["error", "warning", "info", "debug"], log_content[-4:]):
        assert "submodule.dummy" in line
        # The logger name has brian2 internally prefixed, but this shouldn't show up in logs
        assert not "brian2.submodule.dummy" in line
        assert f"{level} message xxx" in line
        assert level.upper() in line

    with catch_logs() as l:
        logger.warn("warning message from Brian")
        submodule_logger.warn("warning message from submodule")
    # only the warning from Brian should be logged
    assert len(l) == 1
    assert "warning message from Brian" in l[0]


if __name__ == "__main__":
    test_file_logging()
    test_file_logging_special_characters()
    test_file_logging_multiprocessing()
    test_file_logging_multiprocessing_with_loggers()
