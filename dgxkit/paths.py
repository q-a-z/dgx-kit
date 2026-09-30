"""`~` in a path the person typed means their home, not the service's.

Installed as the installer does it, DGX-kit runs in a container where ~ is /root (or /tmp for a user
install) while the models and llmctl folders are in the person's home, which the container can read at /home.
"""
from __future__ import annotations

import glob
import os


def expand_home(path: str, homes: str = "/home") -> str:
    if not path.startswith("~"):
        return path
    rest = path[1:]
    if rest and not rest.startswith("/"):  # ~someone/...: the system knows who that is
        return os.path.expanduser(path)
    home = os.environ.get("DGXKIT_HOME") or os.path.expanduser("~")  # the installer records the real one
    guess = home + rest
    if os.path.exists(guess):
        return guess
    # Not there (an install made before DGXKIT_HOME existed): use the one home that has it.
    found = [p for p in glob.glob(f"{homes}/*{rest}") if os.path.exists(p)]
    return found[0] if len(found) == 1 else guess
