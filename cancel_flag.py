"""Lightweight per-project cancel flag — avoids circular imports."""

_flags: dict[str, bool] = {}


def set_cancel(project: str):
    _flags[project] = True


def clear(project: str):
    _flags.pop(project, None)


def is_cancelled(project: str) -> bool:
    return _flags.get(project, False)
