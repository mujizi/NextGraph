"""NextGraph backend application package."""

from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def create_app(*args: Any, **kwargs: Any):
    from backend.app.main import create_app as _create_app

    return _create_app(*args, **kwargs)


def __getattr__(name: str):
    if name == "app":
        from backend.app.main import app as _app

        return _app
    if name == "create_app":
        return create_app
    raise AttributeError(name)
