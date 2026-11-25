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
plateau schedule when that schedule is in use. --resume reads it. A
fine-tune does not. With a weight average (ema in the config), model.pt
holds the averaged weights, which validation and the preview use, and
train.pt adds the weights the optimizer works on (`weights`) and the
average's decay and update count (`ema`).

A run directory follows its latest link. A checkpoint directory, model.pt,
or train.pt names one step exactly.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from whoodeo.catalog import runs_root
from whoodeo.config import parse_model
from whoodeo.models import build_model


@dataclass(frozen=True)
class GeneratorCheckpoint:
    path: Path
    architecture: dict
    state_dict: dict
    step: int | None


@dataclass(frozen=True)
class TrainState:
    path: Path
    step: int
    optimizer: dict
    loss: dict
    discriminator: dict | None
    disc_optimizer: dict | None
    schedule: dict | None
    # With a weight average: the trained weights (model.pt holds the average)
    # and {"decay", "updates"}. None for a run without one.
    weights: dict | None = None
    ema: dict | None = None


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
    directory, model, architecture, step, optimizer, loss,
    discriminator=None, disc_optimizer=None, schedule=None, extra=None,
    publish_latest=True,
):
    """Write checkpoints/<step>/ and point both latest links at it.

    A new step is published by renaming its directory into place, so a crash
    leaves the previous checkpoint alone. The same step replaces its own two
    files, train.pt first. `extra` adds keys to train.pt. Returns the
    checkpoint directory.
    """
    directory = Path(directory)
    step = int(step)
    name = f"{step:06d}"
    train_payload = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }
    if discriminator is not None:
        train_payload["discriminator"] = discriminator.state_dict()
        train_payload["disc_optimizer"] = disc_optimizer.state_dict()
    if schedule is not None:
        train_payload["schedule"] = schedule
    if extra:
        train_payload.update(extra)
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
    if publish_latest:
        _publish_runs_latest(dest)
    return dest


def load_generator(path):
    """Generator weights from a model file, a checkpoint, or a run directory."""
    return _read_model(_model_file(path))


def load_resume(path):
    """Generator plus the training state for --resume.

    `path` may be a run directory, a checkpoint directory, model.pt, or train.pt.
    A run directory opens its newest checkpoint.
    """
    model_path, train_path = _run_files(path)
    if not model_path.is_file():
        raise SystemExit(f"no model: {model_path}")
    spec = _read_model(model_path)
    if train_path.is_file():
        state = _read_train(train_path)
        _same_step(spec, state, model_path, train_path)
        return spec, state
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
        arch, blocks, channels, in_frames, latent, factor,
        encoder_blocks, decoder_blocks, widths, _vae, activation, condition,
    ) = parse_model(spec.architecture, f"{spec.path}: architecture")
    model, label = build_model(
        arch, blocks, channels, in_frames,
        latent=latent, factor=factor,
        encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks, widths=widths,
        activation=activation, condition=condition,
    )
    model.load_state_dict(spec.state_dict)
    model.eval().to(device)
    return model, label, in_frames, spec.step, spec.path


def load_weights(model, spec):
    """spec's weights into `model`. Returns the names of the weights that start fresh, empty when none.

    A latent_up with condition takes the weights of one without: its FiLM
    weights are missing and keep their start, the last layer at zero, so the
    network gives what the loaded one gave. Anything else must match exactly,
    in_frames too.
    """
    # Only latent_up has condition_size.
    frames = model.in_frames if hasattr(model, "condition_size") else None
    saved_frames = int(spec.architecture.get("in_frames", 1))
    if frames is not None and frames != saved_frames:
        raise SystemExit(
            f"{spec.path}: reads in_frames {saved_frames}, this model in_frames {frames}; "
            "--model needs the same in_frames"
        )
    fresh = []
    if getattr(model, "film", None) is not None and spec.architecture.get("condition") is None:
        fresh.append("film.")
    if not fresh:
        model.load_state_dict(spec.state_dict)
        return []
    result = model.load_state_dict(spec.state_dict, strict=False)
    foreign = [name for name in result.missing_keys if not name.startswith(tuple(fresh))]
    if foreign or result.unexpected_keys:
        raise SystemExit(
            f"{spec.path}: does not load into this latent_up: "
            f"missing {foreign or 'none'}, unexpected {result.unexpected_keys or 'none'}"
        )
    return list(result.missing_keys)


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
    """Newest checkpoint of one run."""
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
    if "architecture" not in data or "state_dict" not in data:
        raise SystemExit(f"{path}: not a whoodeo model")
    architecture = data["architecture"]
    if not isinstance(architecture, dict):
        raise SystemExit(f"{path}: architecture must be a mapping")
    return GeneratorCheckpoint(
        path, architecture, data["state_dict"], _optional_step(data.get("step"), path),
    )


def _read_train(path):
    return _train_state(path, _load(path))


def _train_state(path, data):
    missing = [key for key in ("step", "optimizer", "loss") if key not in data or data[key] is None]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    if not isinstance(data["loss"], dict):
        raise SystemExit(f"{path}: loss must be a mapping")
    discriminator = data.get("discriminator")
    disc_optimizer = data.get("disc_optimizer")
    if (discriminator is None) != (disc_optimizer is None):
        raise SystemExit(f"{path}: discriminator and its optimizer must be saved together")
    return TrainState(
        path,
        _require_step(data["step"], path),
        data["optimizer"],
        data["loss"],
        discriminator,
        disc_optimizer,
        data.get("schedule"),
        data.get("weights"),
        data.get("ema"),
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
