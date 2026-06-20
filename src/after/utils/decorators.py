"""
Reusable decorators for timing, logging, and validation.

Applied to WFC and terrain generation functions to separate cross-cutting
concerns (measurement, validation) from core research logic.
"""
from __future__ import annotations

import functools
import logging
import statistics
import time
from typing import Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def timing(repeat: int = 1, logger_name: Optional[str] = None):
    """Measure mean and std-dev execution time over `repeat` runs.

    Uses functools.wraps to preserve the original function's __name__,
    __doc__, and __module__ so introspection tools see the real function.

    Args:
        repeat: Number of times to call the function per invocation.
        logger_name: Logger name; defaults to the function's own module.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logging.getLogger(logger_name or func.__module__)
            times: list[float] = []
            result = None
            for _ in range(repeat):
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - t0)
            mean_ms = statistics.mean(times) * 1_000
            stdev_ms = (statistics.stdev(times) if len(times) > 1 else 0.0) * 1_000
            _logger.info(
                "%s | mean=%.2f ms  stdev=%.2f ms  (n=%d)",
                func.__qualname__, mean_ms, stdev_ms, repeat,
            )
            return result

        wrapper._timing_repeat = repeat
        return wrapper

    return decorator


def log_call(func: Callable) -> Callable:
    """Log function entry, successful return, and any exception.

    Never suppresses exceptions — re-raises after logging so the caller
    still sees the original error and stack trace.
    """
    _logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _logger.debug("→ %s called", func.__qualname__)
        try:
            result = func(*args, **kwargs)
            _logger.debug("← %s returned", func.__qualname__)
            return result
        except Exception as exc:
            _logger.error(
                "✗ %s raised %s: %s", func.__qualname__, type(exc).__name__, exc
            )
            raise

    return wrapper


def validate_grid(min_size: int = 2, max_size: int = 200):
    """Validate that the first positional argument (grid `size`) is in bounds.

    Raises TypeError for non-integer input and ValueError for out-of-range
    values, giving callers an explicit error before the algorithm starts.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(size, *args, **kwargs):
            if not isinstance(size, int):
                raise TypeError(
                    f"{func.__qualname__}: size must be int, got {type(size).__name__}"
                )
            if not (min_size <= size <= max_size):
                raise ValueError(
                    f"{func.__qualname__}: size={size} out of valid range "
                    f"[{min_size}, {max_size}]"
                )
            return func(size, *args, **kwargs)

        return wrapper

    return decorator
