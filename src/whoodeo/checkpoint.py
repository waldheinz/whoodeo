"""Generator file and training state written by a run.

model.pt is the generator. It carries the architecture, the weights, and the
step those weights were saved at. whoodeo-apply and a fine-tune read only
this file.

train.pt is everything else the same run needs in order to continue: the
step again, both optimizers, the discriminator, the loss recipe, and the
loss scales. --resume reads it. A fine-tune does not.

An older model.pt that still holds the optimizer in one file loads too. The
next save of that run writes the two files.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from whoodeo.config import parse_model
from whoodeo.models import build_model

_ARCH_KEYS = (
    "arch", "blocks", "channels", "in_frames",
    "radius", "stem", "sharpness", "reject", "levels",
)


@dataclass(frozen=True)
class GeneratorCheckpoint:
    path: Path
    architecture: dict
    state_dict: dict
    step: int | None
    legacy: bool


@dataclass(frozen=True)
class TrainState:
    path: Path
    step: int
    optimizer: dict
    loss: dict
    objective: dict
    discriminator: dict | None
    disc_optimizer: dict | None


def save_run(
    directory, model, architecture, step, optimizer, loss, objective,
    discriminator=None, disc_optimizer=None,
):
    """Write train.pt first, then model.pt. A crash keeps the previous model.pt."""
    directory = Path(directory)
    step = int(step)
    train_payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "objective": objective,
    }
    if discriminator is not None:
        train_payload["discriminator"] = discriminator.state_dict()
        train_payload["disc_optimizer"] = disc_optimizer.state_dict()
    _atomic_save(directory / "train.pt", train_payload)
    _atomic_save(directory / "model.pt", {
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "step": step,
    })


def load_generator(path):
    """Generator weights from a model file or a run directory."""
    spec, _payload = _read_model(_model_file(path))
    return spec


def load_resume(path):
    """Generator plus the training state for --resume.

    `path` may be a run directory, model.pt, or train.pt.
    """
    model_path, train_path = _run_files(path)
    if not model_path.is_file():
        raise SystemExit(f"no model: {model_path}")
    spec, payload = _read_model(model_path)
    if train_path.is_file():
        state = _read_train(train_path)
        _same_step(spec, state, model_path, train_path)
        return spec, state
    if spec.legacy:
        return spec, _train_from_legacy(payload, model_path)
    raise SystemExit(
        f"no training state: {train_path}. Fine-tune this model with --model."
    )


def build_generator(spec, device):
    """The saved generator, eval mode, on `device`."""
    arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels = parse_model(
        spec.architecture, f"{spec.path}: architecture",
    )
    model, label = build_model(
        arch, blocks, channels, in_frames,
        radius=radius, stem=stem, sharpness=sharpness, reject=reject, levels=levels,
    )
    model.load_state_dict(spec.state_dict)
    model.eval().to(device)
    return model, label, in_frames, spec.step, spec.path


def _model_file(path):
    path = Path(path).expanduser()
    if path.is_file() and path.name == "train.pt":
        model_path = path.parent / "model.pt"
        if not model_path.is_file():
            raise SystemExit(f"{path} is training state, and {model_path} is missing")
        return model_path
    if path.is_dir():
        path = path / "model.pt"
    if not path.is_file():
        raise SystemExit(f"no model: {path}")
    return path


def _run_files(path):
    path = Path(path).expanduser()
    if path.is_dir():
        return path / "model.pt", path / "train.pt"
    if not path.is_file():
        if path.name == "train.pt":
            raise SystemExit(f"no training state: {path}")
        raise SystemExit(f"no model: {path}")
    if path.name == "train.pt":
        return path.parent / "model.pt", path
    return path, path.parent / "train.pt"


def _read_model(path):
    data = _load(path)
    if "architecture" in data and "state_dict" in data:
        architecture = data["architecture"]
        if not isinstance(architecture, dict):
            raise SystemExit(f"{path}: architecture must be a mapping")
        spec = GeneratorCheckpoint(
            path, architecture, data["state_dict"], _optional_step(data.get("step"), path), False,
        )
        return spec, None
    args = data.get("args")
    if "model" in data and isinstance(args, dict) and "arch" in args:
        architecture = {
            key: args[key]
            for key in _ARCH_KEYS
            if key in args and args[key] is not None
        }
        spec = GeneratorCheckpoint(
            path, architecture, data["model"], _optional_step(data.get("step"), path), True,
        )
        return spec, data
    raise SystemExit(f"{path}: not a whoodeo model")


def _read_train(path):
    return _train_state(path, _load(path))


def _train_from_legacy(data, path):
    args = data.get("args") or {}
    return _train_state(path, {
        "step": data.get("step"),
        "optimizer": data.get("optimizer"),
        "loss": args.get("loss"),
        "objective": args.get("objective"),
        "discriminator": data.get("discriminator"),
        "disc_optimizer": data.get("disc_optimizer"),
    })


def _train_state(path, data):
    missing = [key for key in ("step", "optimizer", "loss", "objective") if key not in data or data[key] is None]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    if not isinstance(data["loss"], dict) or not isinstance(data["objective"], dict):
        raise SystemExit(f"{path}: loss and objective must be mappings")
    discriminator = data.get("discriminator")
    disc_optimizer = data.get("disc_optimizer")
    if (discriminator is None) != (disc_optimizer is None):
        raise SystemExit(f"{path}: discriminator and its optimizer must be saved together")
    return TrainState(
        path,
        _require_step(data["step"], path),
        data["optimizer"],
        data["loss"],
        data["objective"],
        discriminator,
        disc_optimizer,
    )


def _same_step(spec, state, model_path, train_path):
    if spec.step is None or spec.step == state.step:
        return
    raise SystemExit(
        f"{model_path} is step {spec.step}, but {train_path} is step {state.step}. "
        f"The last save did not finish. Move {train_path.name} aside to resume from {model_path.name}."
    )


def _optional_step(value, path):
    if value is None:
        return None
    return _require_step(value, path)


def _require_step(value, path):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{path}: step must be an integer")
    return value


def _load(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping")
    return data


def _atomic_save(path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
