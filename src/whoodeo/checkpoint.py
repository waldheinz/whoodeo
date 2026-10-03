"""Generator file and training state written by a run.

Each save lands in checkpoints/<step>/, with model.pt and train.pt beside
each other. A later step adds a directory. The same step may replace its
own two files. latest inside the run points at the newest directory, and
latest in the runs directory points at the newest checkpoint written by any run.

model.pt is the generator. It carries the architecture, the weights, and the
step those weights were saved at. whoodeo-apply and a fine-tune read only
this file.

train.pt is everything else the same run needs in order to continue: the
step again, both optimizers, the discriminator, the loss recipe, and the
loss scales. --resume reads it. A fine-tune does not.

A run directory follows its latest link. A checkpoint directory, model.pt,
or train.pt names one step exactly. An older run that still keeps model.pt
directly in the run directory loads too. An older model.pt that still holds
the optimizer in one file loads too. The next save of that run writes
checkpoints/.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from whoodeo.catalog import runs_root
from whoodeo.config import parse_model
from whoodeo.models import build_model

_ARCH_KEYS = (
    "arch", "blocks", "channels", "in_frames",
    "radius", "stem", "sharpness", "reject", "levels", "bilinear", "codes", "patch",
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


def latest_link():
    """Symlink that follows the newest checkpoint in the runs directory."""
    return runs_root() / "latest"


def run_directory(model_path):
    """Run that owns this model.pt: config, loss, and validation live there."""
    parent = Path(model_path).parent
    if parent.name.isdigit() and parent.parent.name == "checkpoints":
        return parent.parent.parent
    return parent


def save_run(
    directory, model, architecture, step, optimizer, loss, objective,
    discriminator=None, disc_optimizer=None,
):
    """Write checkpoints/<step>/ and point both latest links at it.

    A new step is published by renaming its directory into place, so a crash
    leaves the previous checkpoint alone. The same step replaces its own two
    files, train.pt first. Returns the checkpoint directory.
    """
    directory = Path(directory)
    step = int(step)
    name = f"{step:06d}"
    train_payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "objective": objective,
    }
    if discriminator is not None:
        train_payload["discriminator"] = discriminator.state_dict()
        train_payload["disc_optimizer"] = disc_optimizer.state_dict()
    model_payload = {
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "step": step,
    }
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    partial = checkpoints / f".{name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        _atomic_save(partial / "train.pt", train_payload)
        _atomic_save(partial / "model.pt", model_payload)
        dest = checkpoints / name
        if dest.is_dir():
            os.replace(partial / "train.pt", dest / "train.pt")
            os.replace(partial / "model.pt", dest / "model.pt")
            partial.rmdir()
        else:
            os.replace(partial, dest)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        raise
    _retarget(directory / "latest", Path("checkpoints") / name)
    _publish_runs_latest(dest)
    return dest


def load_generator(path):
    """Generator weights from a model file, a checkpoint, or a run directory."""
    spec, _payload = _read_model(_model_file(path))
    return spec


def load_resume(path):
    """Generator plus the training state for --resume.

    `path` may be a run directory, a checkpoint directory, model.pt, or train.pt.
    A run directory opens its newest checkpoint.
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


def continue_run(path):
    """Run directory to keep writing into.

    None means `path` is an older checkpoint than the one its run calls latest,
    so the caller starts a new run and leaves the later checkpoints in place.
    """
    model_path, _train_path = _run_files(path)
    root = run_directory(model_path)
    current = _latest_checkpoint_dir(root)
    if current is not None and current.resolve() != Path(model_path).parent.resolve():
        return None
    return root


def reconstruction_path(model_path, step, video):
    """Default apply output: <stem>-recon-<step>.mkv in the run directory."""
    if step is None:
        name = f"{Path(video).stem}-recon.mkv"
    else:
        name = f"{Path(video).stem}-recon-{int(step):06d}.mkv"
    return run_directory(model_path) / name


def build_generator(spec, device):
    """The saved generator, eval mode, on `device`."""
    (
        arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels,
        bilinear, codes, patch,
    ) = parse_model(spec.architecture, f"{spec.path}: architecture")
    model, label = build_model(
        arch, blocks, channels, in_frames,
        radius=radius, stem=stem, sharpness=sharpness, reject=reject, levels=levels,
        bilinear=bilinear, codes=codes, patch=patch,
    )
    model.load_state_dict(spec.state_dict)
    model.eval().to(device)
    return model, label, in_frames, spec.step, spec.path


def _publish_runs_latest(checkpoint_dir):
    runs = runs_root()
    runs.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir.resolve()
    try:
        dest.relative_to(runs.resolve())
        target = Path(os.path.relpath(dest, runs.resolve()))
    except ValueError:
        target = dest
    _retarget(runs / "latest", target)


def _retarget(link, target):
    """Point `link` at `target`. The replacement itself is one rename."""
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(link.name + ".tmp")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _latest_checkpoint_dir(run_dir):
    """Newest checkpoint of one run, or the legacy model.pt in the run itself."""
    run_dir = Path(run_dir)
    link = run_dir / "latest"
    if link.is_symlink():
        raw = Path(os.readlink(link))
        target = raw if raw.is_absolute() else run_dir / raw
        target = target.resolve()
        if (target / "model.pt").is_file():
            return target
    best = _highest_step(run_dir / "checkpoints")
    if best is not None:
        return best.resolve()
    if (run_dir / "model.pt").is_file():
        return run_dir.resolve()
    return None


def _highest_step(checkpoints):
    if not checkpoints.is_dir():
        return None
    found = []
    for child in checkpoints.iterdir():
        if child.name.isdigit() and (child / "model.pt").is_file():
            found.append((int(child.name), child))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def _checkpoint_dir(path):
    """Directory that directly holds the model.pt `path` refers to.

    A run directory follows latest. A checkpoint directory is itself.
    """
    if not path.is_dir():
        return None
    if (path / "latest").is_symlink() or (path / "checkpoints").is_dir():
        found = _latest_checkpoint_dir(path)
        if found is not None:
            return found
    if (path / "model.pt").is_file():
        return path.resolve()
    return None


def _model_file(path):
    path = Path(path).expanduser()
    if path.is_file() and path.name == "train.pt":
        model_path = path.parent / "model.pt"
        if not model_path.is_file():
            raise SystemExit(f"{path} is training state, and {model_path} is missing")
        return model_path
    if path.is_dir():
        found = _checkpoint_dir(path)
        if found is None:
            raise SystemExit(f"no model: {path / 'model.pt'}")
        return found / "model.pt"
    if not path.is_file():
        raise SystemExit(f"no model: {path}")
    return path


def _run_files(path):
    path = Path(path).expanduser()
    if path.is_dir():
        found = _checkpoint_dir(path)
        if found is None:
            raise SystemExit(f"no model: {path / 'model.pt'}")
        return found / "model.pt", found / "train.pt"
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
