import signal
import time
from typing import Optional

from loguru import logger

TIMEOUT_INIT = False


class Timeout:
    def __init__(self, seconds: float, name: Optional[str] = None, mode: str = "error"):
        """
        Initialize the Timeout context manager.

        :param seconds: Timeout duration in seconds.
        :param name: Optional name for the operation.
        :param mode: Timeout mode. Either 'error' to raise an exception or 'warning' to print a warning.
        """
        self.seconds = seconds
        self.name = name
        self.mode = mode.lower()
        if self.mode not in {"error", "warning"}:
            raise ValueError("Mode must be either 'error' or 'warning'")

    def handle_timeout(self, signum: int, frame: Optional[object]) -> None:
        """
        Handle the timeout event.
        """
        if self.mode == "error":
            if self.name:
                raise TimeoutError(f"Operation '{self.name}' timed out after {self.seconds} seconds")
            else:
                raise TimeoutError(f"Operation timed out after {self.seconds} seconds")
        elif self.mode == "warning":
            op = f"'{self.name}' " if self.name else ""
            logger.warning(f"Operation {op}exceeded {self.seconds} seconds but continues.")

    def __enter__(self):
        """
        Enter the context and set the timeout alarm.
        """
        global TIMEOUT_INIT
        if not TIMEOUT_INIT:
            TIMEOUT_INIT = True
        else:
            raise NotImplementedError("Nested timeouts are not supported")
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)  # type: ignore

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context and clear the timeout alarm.
        """
        global TIMEOUT_INIT
        TIMEOUT_INIT = False
        signal.alarm(0)  # Disable the alarm


class Rate:
    def __init__(self, rate: Optional[float], rate_name: Optional[str] = None, warn_tolerance: float = 0.1):
        self.last = time.time()
        self.rate = rate
        self.rate_name = rate_name
        self.warn_tolerance = warn_tolerance
        self._first = True

    @property
    def dt(self) -> float:
        if self.rate is None:
            return 0.0
        return 1.0 / self.rate

    def sleep(self) -> None:
        if self.rate is None:
            return
        overrun = time.time() - (self.last + self.dt)
        if overrun > self.warn_tolerance and not self._first:
            logger.debug(f"Behind schedule {self.rate_name} by {overrun:.4f}s (tolerance {self.warn_tolerance}s)")
        else:
            needed_sleep = max(0, self.last + self.dt - time.time() - 0.0001)
            time.sleep(needed_sleep)
        self._first = False
        self.last = time.time()


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2
