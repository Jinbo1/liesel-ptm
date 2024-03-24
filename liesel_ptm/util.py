import logging
from collections.abc import Sequence
from contextlib import contextmanager

from .custom_types import Array


@contextmanager
def log_exception_and_pass(
    logger: str | logging.Logger, exception: type[Exception] = Exception
):
    try:
        yield
    except exception:
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
            logger.exception("Exception in run.")
        else:
            logger.exception("Exception in run.")


def standardize(
    a: Array,
    center: bool = True,
    scale: bool = True,
    axis: Sequence[int] | int | None = None,
) -> Array:
    if center:
        a = a - a.mean(axis=axis, keepdims=True)
    if scale:
        a = a / a.std(axis=axis, keepdims=True)
    return a
