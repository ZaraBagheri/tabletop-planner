"""Construct simulation backends without importing PyBullet for symbolic runs."""
from .symbolic import SymbolicBackend


def make_backend(episode, kind="symbolic", config=None, gui=False, record=False):
    if kind == "bullet":
        from .bullet_backend import BulletBackend
        return BulletBackend(episode, gui=gui, record=record, config=config)
    if kind == "symbolic" and not (gui or record):
        return SymbolicBackend(episode, config=config)
    raise ValueError(f"Unsupported backend/options: {kind}")
