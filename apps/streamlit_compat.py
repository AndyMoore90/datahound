"""Small Streamlit compatibility helpers for cross-version UI kwargs.

Primary goal: avoid runtime crashes when newer kwargs (e.g. width='stretch',
use_container_width) are unavailable in the installed Streamlit version.
"""

from __future__ import annotations

from typing import Any, Callable


def call_compat(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call Streamlit API with graceful fallback for width-related kwargs."""
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" not in msg:
            raise
        fallback = dict(kwargs)
        fallback.pop("width", None)
        fallback.pop("use_container_width", None)
        return fn(*args, **fallback)


def stretch_kwargs() -> dict[str, bool]:
    """Preferred full-width kwargs for modern Streamlit."""
    return {"use_container_width": True}
