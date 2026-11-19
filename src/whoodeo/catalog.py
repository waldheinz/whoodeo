"""Directories for the films and the runs.

The user config is $XDG_CONFIG_HOME/whoodeo/config.yaml. When that variable
is unset or empty, the file is ~/.config/whoodeo/config.yaml. A path named
in the file is used after expanding ~, and it has to be absolute. A missing
file or a missing key uses the XDG directory for that kind of file: films
under $XDG_DATA_HOME/whoodeo, runs under $XDG_STATE_HOME/whoodeo. An empty
or relative value in those variables is ignored, and the spec default
under the home directory is used instead.

`events` is optional: the directory for the TensorBoard events of each run,
one folder per run name. Without it the events stay in the run directory.
On a network mount each TensorBoard event is its own append, which made the
training step wait (about 5 s a step over the VPN mount of spark-03,
19.11.2026), so there the events go to a local disk.

`live` is the host:port the live page of training and whoodeo-apply binds,
127.0.0.1:8765 when absent; 0.0.0.0:8765 serves it to the network, port 0
picks a free port.

Rungs and the degrade recipe live in the data directory's catalog.json,
not in this file.
"""

import os
from pathlib import Path

import yaml

_PATH_KEYS = ("data", "runs", "events")
_LIVE_KEY = "live"
DEFAULT_LIVE = "127.0.0.1:8765"


class EnvPath:
    """Default path under the data directory.

    argparse keeps this object when the flag is omitted, and converts a
    passed path with Path. The help text prints the resolved directory.
    """

    def __init__(self, child):
        self.child = child

    def __str__(self):
        return str(self.resolve())

    def resolve(self):
        return data_root() / self.child


def config_path():
    """User config file. It may be absent."""
    return _xdg_home("XDG_CONFIG_HOME", ".config") / "whoodeo" / "config.yaml"


def data_root():
    """Directory that contains orig/ and low/."""
    found = _configured("data")
    if found is not None:
        return found
    return _xdg_home("XDG_DATA_HOME", ".local/share") / "whoodeo"


def runs_root():
    """Directory that contains the train runs and latest."""
    found = _configured("runs")
    if found is not None:
        return found
    return _xdg_home("XDG_STATE_HOME", ".local/state") / "whoodeo"


def events_root():
    """Directory for the TensorBoard events of the runs, or None for the run directory."""
    return _configured("events")


def live_bind():
    """host:port for the live page, from the config or the default."""
    data = _document()
    if _LIVE_KEY not in data:
        return DEFAULT_LIVE
    value = data[_LIVE_KEY]
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{config_path()}: {_LIVE_KEY} must be host:port")
    return value.strip()


def resolve_data(path):
    if isinstance(path, EnvPath):
        return path.resolve()
    return path


def _configured(key):
    """Path written for `key`, or None when the file or the key is absent."""
    return _paths().get(key)


def _document():
    """The config mapping. Only data, runs, events, and live are read."""
    path = config_path()
    if not path.exists():
        return {}
    if not path.is_file():
        raise SystemExit(f"{path}: expected a file")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping")
    unknown = [str(name) for name in data if name not in _PATH_KEYS and name != _LIVE_KEY]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SystemExit(f"{path}: unknown keys: {names}")
    return data


def _paths():
    data = _document()
    path = config_path()
    found = {}
    for key in _PATH_KEYS:
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"{path}: {key} must be a path")
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            raise SystemExit(f"{path}: {key} must be an absolute path")
        found[key] = resolved
    return found


def _xdg_home(name, default):
    """Absolute base directory from an XDG variable.

    An empty or relative value is ignored. `default` is the path under the
    home directory that the spec uses then, such as `.config`.
    """
    raw = os.environ.get(name, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
    return Path.home() / default
