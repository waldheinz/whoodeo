"""Training recipe loaded from one YAML file."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from whoodeo.models import PRESETS

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
    balance: str
    terms: tuple[Term, ...]
    radius: int | None = None
    stem: int | None = None
    sharpness: float | None = None
    reject: bool | None = None
    levels: int | None = None

    def arch_key(self):
        return model_key(
            self.arch, self.blocks, self.channels, self.in_frames,
            self.radius, self.stem, self.sharpness, self.reject, self.levels,
        )

    def loss_recipe(self):
        return {"balance": self.balance, "terms": [term.as_dict() for term in self.terms]}

    def term(self, name):
        for item in self.terms:
            if item.name == name:
                return item
        return None


def shape_text(arch, blocks, channels, in_frames, *extra):
    if blocks is None:
        text = f"{arch} in_frames {in_frames}"
    else:
        text = f"{arch} {blocks}x{channels} in_frames {in_frames}"
    if arch == "shift":
        radius, stem, sharpness, reject = extra
        text += (
            f" radius {radius} stem {stem} "
            f"sharpness {sharpness:g} reject {str(reject).lower()}"
        )
    elif arch == "pyramid":
        levels, stem, radius = extra
        text += f" levels {levels} stem {stem} radius {radius}"
    return text


def model_key(arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels):
    key = (arch, blocks, channels, in_frames)
    if arch == "shift":
        return key + (radius, stem, float(sharpness), reject)
    if arch == "pyramid":
        return key + (levels, stem, radius)
    return key


def architecture_dict(cfg):
    """The model mapping stored next to the generator weights."""
    item = {"arch": cfg.arch, "in_frames": cfg.in_frames}
    if cfg.blocks is not None:
        item["blocks"] = cfg.blocks
    if cfg.channels is not None:
        item["channels"] = cfg.channels
    if cfg.arch == "shift":
        item["radius"] = cfg.radius
        item["stem"] = cfg.stem
        item["sharpness"] = cfg.sharpness
        item["reject"] = cfg.reject
    elif cfg.arch == "pyramid":
        item["levels"] = cfg.levels
        item["stem"] = cfg.stem
        item["radius"] = cfg.radius
    return item


def assert_resume_matches(cfg, architecture, loss, checkpoint):
    """The checkpoint must be the same network and the same loss. Learning rates may change."""
    saved_key = model_key(*parse_model(architecture, f"{checkpoint}: architecture"))
    if saved_key != cfg.arch_key():
        found = shape_text(*saved_key)
        wanted = shape_text(*cfg.arch_key())
        raise SystemExit(f"checkpoint architecture is {found}, config asks for {wanted}")
    if loss != cfg.loss_recipe():
        raise SystemExit(f"checkpoint loss does not match {cfg.source}")


def load_config(path, architecture=None, architecture_from=None):
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

    arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels = _take_model(
        data, where, architecture, architecture_from,
    )
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
        balance=balance,
        terms=tuple(terms),
        radius=radius,
        stem=stem,
        sharpness=sharpness,
        reject=reject,
        levels=levels,
    )


def _take_model(data, where, architecture, architecture_from):
    source = str(architecture_from) if architecture_from is not None else "model file"
    if "model" in data:
        raw = data.pop("model")
        if not isinstance(raw, dict):
            raise SystemExit(f"{where}: model must be a mapping")
        chosen = parse_model(raw, f"{where}: model")
        if architecture is not None:
            from_file = parse_model(architecture, f"{source}: architecture")
            if model_key(*from_file) != model_key(*chosen):
                raise SystemExit(
                    f"{where}: model is {shape_text(*model_key(*chosen))}, "
                    f"{source} is {shape_text(*model_key(*from_file))}"
                )
        return chosen
    if architecture is None:
        raise SystemExit(
            f"{where}: missing model. Add a model: section, or pass --model to fine-tune."
        )
    return parse_model(architecture, f"{source}: architecture")


def parse_model(model, where):
    """Read one model mapping. `where` is the location named in errors."""
    if not isinstance(model, dict):
        raise SystemExit(f"{where} must be a mapping")
    model = dict(model)
    arch = _take_choice(model, "arch", where, PRESETS)
    preset = PRESETS[arch]
    in_frames = _take_int(model, "in_frames", where, positive=True)
    if arch == "espcn":
        if "blocks" in model or "channels" in model:
            raise SystemExit(f"{where}: espcn has a fixed size")
        blocks = None
        channels = None
    else:
        blocks = _take_int(model, "blocks", where, positive=True)
        channels = _take_int(model, "channels", where, positive=True)
    radius, stem, sharpness, reject = _take_shift(model, arch, where)
    levels = None
    if arch == "pyramid":
        levels, stem, radius = _take_pyramid(model, where)
    _reject_unknown(model, where)
    if in_frames % 2 != 1:
        raise SystemExit(f"{where}: in_frames must be odd")
    fixed = preset["frames"]
    if fixed is not None and in_frames != fixed:
        raise SystemExit(f"{where}: {arch} takes {fixed} input frame, not {in_frames}")
    if arch in ("shift", "pyramid") and in_frames < 3:
        raise SystemExit(f"{where}: {arch} needs at least 3 input frames")
    return arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels


def _take_shift(data, arch, where):
    if arch != "shift":
        return None, None, None, None
    radius = _take_int(data, "radius", where)
    if radius < 0 or radius > 16:
        raise SystemExit(f"{where}: radius must be from 0 to 16")
    stem = _take_int(data, "stem", where)
    if stem < 0:
        raise SystemExit(f"{where}: stem must be zero or positive")
    sharpness = _take_float(data, "sharpness", where, positive=True)
    reject = _take_bool(data, "reject", where)
    return radius, stem, sharpness, reject


def _take_pyramid(data, where):
    levels = _take_int(data, "levels", where, positive=True)
    if levels > 6:
        raise SystemExit(f"{where}: levels must be from 1 to 6")
    stem = _take_int(data, "stem", where)
    if stem < 0:
        raise SystemExit(f"{where}: stem must be zero or positive")
    radius = _take_int(data, "radius", where)
    if radius < 0 or radius > 8:
        raise SystemExit(f"{where}: radius must be from 0 to 8")
    return levels, stem, radius


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


def _take_bool(data, key, where):
    value = _pop_value(data, key, where, ...)
    if not isinstance(value, bool):
        raise SystemExit(f"{where}: {key} must be true or false")
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
