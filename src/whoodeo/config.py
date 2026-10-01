"""Training recipe loaded from one YAML file."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from whoodeo.nets import PRESETS

VGG_LAYERS = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}

TERM_NAMES = ("pixel", "vgg", "gan")


@dataclass(frozen=True)
class Term:
    name: str
    weight: float
    kind: str | None = None
    layers: tuple[str, ...] = ()
    crop: int | None = None
    disc_lr: float | None = None

    def as_dict(self):
        # disc_lr stays off the matched recipe: a resume may change learning rates.
        item = {"name": self.name, "weight": round(self.weight, 8)}
        if self.name == "pixel":
            item["kind"] = self.kind
        elif self.name == "vgg":
            item["layers"] = list(self.layers)
        else:
            item["crop"] = self.crop
        return item


@dataclass(frozen=True)
class TrainConfig:
    source: Path
    arch: str
    blocks: int | None
    channels: int | None
    in_frames: int
    batch: int
    lr: float
    holdout: float
    log_every: int
    val_every: int
    val_count: int
    seed: int
    steps: int | None
    out: Path | None
    resume: Path | None
    balance: str
    terms: tuple[Term, ...]

    def arch_key(self):
        return (self.arch, self.blocks, self.channels, self.in_frames)

    def loss_recipe(self):
        return {"balance": self.balance, "terms": [term.as_dict() for term in self.terms]}

    def term(self, name):
        for item in self.terms:
            if item.name == name:
                return item
        return None


def shape_text(arch, blocks, channels, in_frames):
    if blocks is None:
        return f"{arch} in_frames {in_frames}"
    return f"{arch} {blocks}x{channels} in_frames {in_frames}"


def assert_resume_matches(cfg, saved, checkpoint):
    """The checkpoint must be the same network and the same loss. Learning rates may change."""
    if not isinstance(saved, dict) or "arch" not in saved or "loss" not in saved or "objective" not in saved:
        raise SystemExit(f"checkpoint {checkpoint} has no training recipe")
    for key in ("blocks", "channels", "in_frames"):
        if key not in saved:
            raise SystemExit(f"checkpoint {checkpoint} has no {key}")
    saved_key = (saved["arch"], saved["blocks"], saved["channels"], saved["in_frames"])
    if saved_key != cfg.arch_key():
        found = shape_text(*saved_key)
        wanted = shape_text(*cfg.arch_key())
        raise SystemExit(f"checkpoint architecture is {found}, config asks for {wanted}")
    if saved["loss"] != cfg.loss_recipe():
        raise SystemExit(f"checkpoint loss does not match {cfg.source}")


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no config: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping")
    where = str(path)

    arch = _take_choice(data, "arch", where, PRESETS)
    preset = PRESETS[arch]
    in_frames = _take_int(data, "in_frames", where, positive=True)
    if in_frames % 2 != 1:
        raise SystemExit(f"{where}: in_frames must be odd")
    fixed = preset["frames"]
    if fixed is not None and in_frames != fixed:
        raise SystemExit(f"{where}: {arch} takes {fixed} input frame, not {in_frames}")
    if arch == "espcn":
        if "blocks" in data or "channels" in data:
            raise SystemExit(f"{where}: espcn has a fixed size")
        blocks = None
        channels = None
    else:
        blocks = _take_int(data, "blocks", where, positive=True)
        channels = _take_int(data, "channels", where, positive=True)

    batch = _take_int(data, "batch", where, positive=True)
    lr = _take_float(data, "lr", where, positive=True)
    holdout = _take_float(data, "holdout", where, positive=True)
    if holdout >= 1:
        raise SystemExit(f"{where}: holdout must be below 1")
    log_every = _take_int(data, "log_every", where, positive=True)
    val_every = _take_int(data, "val_every", where, positive=True)
    val_count = _take_int(data, "val_count", where, positive=True)
    seed = _take_int(data, "seed", where, default=0)
    steps = _take_int(data, "steps", where, positive=True, default=None)
    out = _take_path(data, "out", where)
    resume = _take_path(data, "resume", where)
    balance, terms = _take_loss(data, where)
    _reject_unknown(data, where)
    return TrainConfig(
        source=path,
        arch=arch,
        blocks=blocks,
        channels=channels,
        in_frames=in_frames,
        batch=batch,
        lr=lr,
        holdout=holdout,
        log_every=log_every,
        val_every=val_every,
        val_count=val_count,
        seed=seed,
        steps=steps,
        out=out,
        resume=resume,
        balance=balance,
        terms=tuple(terms),
    )


def _take_loss(data, where):
    if "loss" not in data:
        raise SystemExit(f"{where}: missing loss")
    loss = data.pop("loss")
    if not isinstance(loss, dict):
        raise SystemExit(f"{where}: loss must be a mapping")
    balance = _take_choice(loss, "balance", f"{where}: loss", ("start", "running"))
    if "terms" not in loss:
        raise SystemExit(f"{where}: loss is missing terms")
    raw_terms = loss.pop("terms")
    _reject_unknown(loss, f"{where}: loss")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise SystemExit(f"{where}: loss.terms must be a non-empty list")
    terms = []
    seen = set()
    for index, raw in enumerate(raw_terms, start=1):
        label = f"{where}: loss term {index}"
        if not isinstance(raw, dict):
            raise SystemExit(f"{label} must be a mapping")
        name = _take_choice(raw, "name", label, TERM_NAMES)
        if name in seen:
            raise SystemExit(f"{where}: duplicate loss term {name}")
        seen.add(name)
        weight = _take_float(raw, "weight", label, positive=True)
        if weight > 1:
            raise SystemExit(f"{label}: weight must be at most 1")
        if name == "pixel":
            kind = _take_choice(raw, "kind", label, ("mse", "l1"))
            terms.append(Term(name=name, weight=weight, kind=kind))
        elif name == "vgg":
            layers = _take_layers(raw, label)
            terms.append(Term(name=name, weight=weight, layers=tuple(layers)))
        else:
            crop = _take_int(raw, "crop", label, positive=True)
            if crop % 8 != 0:
                raise SystemExit(f"{label}: crop must be a multiple of 8")
            disc_lr = _take_float(raw, "disc_lr", label, positive=True)
            terms.append(Term(name=name, weight=weight, crop=crop, disc_lr=disc_lr))
        _reject_unknown(raw, label)
    total = sum(term.weight for term in terms)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"{where}: loss weights sum to {total:.6g}, expected 1")
    return balance, terms


def _take_layers(data, where):
    if "layers" not in data:
        raise SystemExit(f"{where}: missing layers")
    layers = data.pop("layers")
    if not isinstance(layers, list) or not layers:
        raise SystemExit(f"{where}: layers must be a non-empty list")
    known = ", ".join(VGG_LAYERS)
    seen = set()
    for layer in layers:
        if not isinstance(layer, str) or layer not in VGG_LAYERS:
            raise SystemExit(f"{where}: unknown vgg layer {layer!r}. choices: {known}")
        if layer in seen:
            raise SystemExit(f"{where}: duplicate vgg layer {layer}")
        seen.add(layer)
    return layers


def _take_choice(data, key, where, choices):
    if key not in data:
        raise SystemExit(f"{where}: missing {key}")
    value = data.pop(key)
    if value not in choices:
        known = ", ".join(str(item) for item in choices)
        raise SystemExit(f"{where}: unknown {key} {value!r}. choices: {known}")
    return value


def _take_int(data, key, where, positive=False, default=...):
    value = _pop_value(data, key, where, default)
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, int):
        raise SystemExit(f"{where}: {key} must be an integer")
    if positive and value <= 0:
        raise SystemExit(f"{where}: {key} must be positive")
    return value


def _take_float(data, key, where, positive=False, default=...):
    value = _pop_value(data, key, where, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{where}: {key} must be a number")
    value = float(value)
    if positive and value <= 0:
        raise SystemExit(f"{where}: {key} must be positive")
    return value


def _take_path(data, key, where):
    if key not in data or data[key] is None:
        data.pop(key, None)
        return None
    value = data.pop(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{where}: {key} must be a path")
    return Path(value).expanduser()


def _pop_value(data, key, where, default):
    if key not in data:
        if default is ...:
            raise SystemExit(f"{where}: missing {key}")
        return default
    value = data.pop(key)
    if value is None:
        if default is None:
            return None
        raise SystemExit(f"{where}: {key} is empty")
    return value


def _reject_unknown(data, where):
    if data:
        names = ", ".join(sorted(str(key) for key in data))
        raise SystemExit(f"{where}: unknown keys: {names}")
