"""Paired-video root. Set WHOODEO_DATA to the directory that contains orig/ and low/."""

import os
from pathlib import Path


class EnvPath:
    """Default path under $WHOODEO_DATA.

    argparse keeps this object when the flag is omitted, and converts a
    passed path with Path. The help text prints the environment form.
    """

    def __init__(self, child):
        self.child = child

    def __str__(self):
        return f"$WHOODEO_DATA/{self.child}"

    def resolve(self):
        return data_root() / self.child


def data_root():
    raw = os.environ.get("WHOODEO_DATA")
    if not raw:
        raise SystemExit(
            "WHOODEO_DATA is not set. "
            "Point it at the directory that contains orig/ and low/."
        )
    return Path(raw).expanduser()


def resolve_data(path):
    if isinstance(path, EnvPath):
        return path.resolve()
    return path
