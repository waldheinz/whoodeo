"""Serve TensorBoard for the configured runs directory.

Each training run writes its event file into that run's directory. This
listens on every interface and reads those files from the runs directory.
"""

import sys

from tensorboard.main import run_main

from whoodeo.catalog import runs_root


def main():
    logdir = runs_root()
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"tensorboard  {logdir}  port 6006, every interface", flush=True)
    sys.argv = [
        "tensorboard",
        "--logdir", str(logdir),
        "--bind_all",
    ]
    run_main()
