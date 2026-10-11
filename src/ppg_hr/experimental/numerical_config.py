"""Thread-safe numerical snapshot scope used by all shared signal helpers."""
from contextvars import ContextVar
from functools import wraps
from ppg_hr.params import DEFAULT_NUMERICS, NumericalParams

NUMERICS = ContextVar('ppg_numerics', default=DEFAULT_NUMERICS)


def with_numerics(function):
    """Restore the explicit snapshot for a call, without changing code defaults."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        config = kwargs.pop('numerical_config', None)
        fixed = kwargs.get('fixed_config')
        if config is None and fixed is not None:
            config = fixed['numerical']
        values = {key: tuple(value) if isinstance(value, list) else value for key, value in config.items()} if config is not None else None
        token = NUMERICS.set(NumericalParams(**values) if values is not None else DEFAULT_NUMERICS)
        try:
            return function(*args, **kwargs)
        finally:
            NUMERICS.reset(token)
    return wrapped
