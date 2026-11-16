import logging
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

# utils/logger.py -> project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logs_path = PROJECT_ROOT / "logs"
logs_path.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = logs_path / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(filename)s:%(lineno)d %(funcName)s() - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)


def _fmt(values):
    return " | ".join(f"{k}={v}" for k, v in values.items())


@contextmanager
def log_step(name, **inputs):
    """Log the START (with inputs), END (with outputs + duration) or FAILED of one pipeline step.

    Usage:
        with log_step("Detection", frames=100) as out:
            ...
            out["players"] = 230      # anything put in `out` is logged as the step's output
    """
    # stacklevel=3 -> the log line shows the caller's file:line, not logger.py / contextlib
    logging.info(f"[START] {name}" + (f" | input: {_fmt(inputs)}" if inputs else ""), stacklevel=3)
    outputs = {}
    t0 = time.perf_counter()
    try:
        yield outputs
    except Exception as e:
        logging.error(f"[FAILED] {name} after {time.perf_counter() - t0:.2f}s | {type(e).__name__}: {e}",
                      stacklevel=3)
        raise
    logging.info(f"[END] {name} ({time.perf_counter() - t0:.2f}s)"
                 + (f" | output: {_fmt(outputs)}" if outputs else ""), stacklevel=3)


if __name__ == "__main__":
    logging.info("Logging has started")
