import logging
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, overload

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@overload
def beta[F: Callable[..., Any]](func: F) -> F: ...


@overload
def beta[F: Callable[..., Any]](
    *, message: str | None = None, log_level: int = logging.WARNING
) -> Callable[[F], F]: ...


def beta(
    func: Any = None,
    *,
    message: str | None = None,
    log_level: int = logging.WARNING,
) -> Any:
    def decorator[F: Callable[..., Any]](fn: F) -> F:
        msg = message or f"{fn.__name__}() is a beta feature and may change."

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logging.getLogger(fn.__module__).log(
                log_level, f"Beta: {fn.__name__}"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator(func) if callable(func) else decorator
