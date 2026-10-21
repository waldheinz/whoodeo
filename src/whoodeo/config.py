"""Training recipe loaded from one YAML file."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from whoodeo.models import PRESETS
from whoodeo.models.vae2 import layout, layout_text

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

TERM_NAMES = ("pixel", "vgg", "gan", "color", "mu_l2", "scale_eq")
# vae2 terms that stand beside the objective: in the step loss, not in val.
SIDE_TERMS = ("mu_l2", "scale_eq")
SCALE_EQ_FACTORS = (2, 4)
# Spaces the color term compares in, the first the default.
COLOR_SPACES = ("ycbcr", "oklab")
# Pooling factor of the color term when the file leaves pool out.
COLOR_POOL = 8

# Architectures that reconstruct a master frame through a latent.
VAE_ARCHS = ("vae", "vae2")


@dataclass(frozen=True)
class Term:
    """One loss term. weight multiplies its raw value.

    vgg has no term weight: `layers` pairs each layer with its own weight.
    color compares in `space` after a 1/`pool` average pool.
    """

    name: str
    weight: float | None
    kind: str | None = None
    layers: tuple[tuple[str, float], ...] = ()
    crop: int | None = None
    disc_lr: float | None = None
    start: int | None = None
    factors: tuple[int, ...] | None = None
    space: str | None = None
    pool: int | None = None

    def as_dict(self):
        # disc_lr stays off the matched recipe: a resume may change learning rates.
        if self.name == "vgg":
            return {
                "name": self.name,
                "layers": {layer: round(weight, 8) for layer, weight in self.layers},
            }
        item = {"name": self.name, "weight": round(self.weight, 8)}
        if self.name == "pixel":
            item["kind"] = self.kind
        elif self.name == "scale_eq":
            item["factors"] = list(self.factors)
        elif self.name == "color":
            item["space"] = self.space
            item["pool"] = self.pool
        elif self.name == "gan":
            item["crop"] = self.crop
            if self.start is not None:
                item["start"] = self.start
        return item


@dataclass(frozen=True)
class Schedule:
    """How `lr` moves. `patience`, `factor`, `min_delta`, and `min_lr` are set for plateau only."""

    kind: str
    patience: int | None = None
    factor: float | None = None
    min_delta: float | None = None
    min_lr: float | None = None


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
    val_every: int
    val_count: int
    seed: int
    steps: int | None
    out: Path | None
    terms: tuple[Term, ...]
    schedule: Schedule
    radius: int | None = None
    stem: int | None = None
    vote: int | None = None
    sharpness: float | None = None
    reject: bool | None = None
    levels: int | None = None
    bilinear: bool | None = None
    codes: int | None = None
    patch: int | None = None
    res_scale: float | None = None
    variants: str | None = None
    latent: int | None = None
    factor: int | None = None
    beta: float | None = None
    encoder_blocks: tuple[int, ...] | None = None
    decoder_blocks: tuple[int, ...] | None = None
    widths: tuple[int, ...] | None = None
    decoder_widths: tuple[int, ...] | None = None
    posterior: str | None = None
    crop: int | None = None
    freeze_encoder: bool = False
    # Training noise on the vae2 latent. Not part of the loss recipe, so a
    # resume may change it. 0 is off.
    latent_noise_tau: float = 0.0

    def arch_key(self):
        return model_key(
            self.arch, self.blocks, self.channels, self.in_frames,
            self.radius, self.stem, self.sharpness, self.reject, self.levels,
            self.bilinear, self.codes, self.patch, self.res_scale, self.vote,
            self.latent, self.factor, self.beta,
            self.encoder_blocks, self.decoder_blocks, self.widths, self.decoder_widths,
            self.posterior,
        )

    def loss_recipe(self):
        return {"terms": [term.as_dict() for term in self.terms]}

    def term(self, name):
        for item in self.terms:
            if item.name == name:
                return item
        return None

    def objective_terms(self):
        """Terms summed into the objective as weight * raw.

        A vae2 gan term stands outside: its weight multiplies the adaptive
        weight, and the validation objective leaves it out. So do mu_l2 and
        scale_eq, which the training step adds.
        """
        return tuple(
            item for item in self.terms
            if not (self.arch == "vae2" and item.name == "gan")
            and item.name not in SIDE_TERMS
        )


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
    elif arch == "vq":
        codes, patch, bilinear = extra
        text += f" codes {codes} patch {patch} bilinear {str(bilinear).lower()}"
    elif arch == "edsr":
        res_scale, = extra
        text += f" res_scale {res_scale:g}"
    elif arch == "vote":
        stem, vote, res_scale = extra
        text = (
            f"vote {blocks}x{channels} stem {stem} vote {vote} "
            f"res_scale {res_scale:g}"
        )
    elif arch in VAE_ARCHS:
        latent, factor, *levels = extra
        text = f"{arch} {blocks}x{channels} latent {latent} factor {factor}"
        if levels:
            *levels, posterior = levels
            text += layout_text(factor, *levels)
            if posterior != "gaussian":
                text += f" posterior {posterior}"
    return text


def model_key(
    arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels,
    bilinear, codes, patch, res_scale, vote, latent, factor, beta,
    encoder_blocks=None, decoder_blocks=None, widths=None, decoder_widths=None,
    posterior=None,
):
    key = (arch, blocks, channels, in_frames)
    if arch == "shift":
        return key + (radius, stem, float(sharpness), reject)
    if arch == "pyramid":
        return key + (levels, stem, radius)
    if arch == "vq":
        return key + (codes, patch, bilinear)
    if arch == "edsr":
        return key + (float(res_scale),)
    if arch == "vote":
        return key + (stem, vote, float(res_scale))
    if arch == "vae2":
        # beta weighs the KL in the loss. It does not shape the weights.
        # The levels compare with their defaults filled in.
        return key + (latent, factor) + layout(
            factor, encoder_blocks, decoder_blocks, widths, decoder_widths,
        ) + (posterior or "gaussian",)
    if arch in VAE_ARCHS:
        return key + (latent, factor)
    return key


def architecture_dict(cfg):
    """The model mapping stored next to the generator weights."""
    item = {"arch": cfg.arch}
    if cfg.arch not in ("vote", *VAE_ARCHS):
        item["in_frames"] = cfg.in_frames
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
    elif cfg.arch == "vq":
        item["bilinear"] = cfg.bilinear
        item["codes"] = cfg.codes
        item["patch"] = cfg.patch
    elif cfg.arch == "edsr":
        item["res_scale"] = cfg.res_scale
    elif cfg.arch == "vote":
        item["stem"] = cfg.stem
        item["vote"] = cfg.vote
        item["res_scale"] = cfg.res_scale
    elif cfg.arch in VAE_ARCHS:
        item["latent"] = cfg.latent
        item["factor"] = cfg.factor
        if cfg.beta is not None:
            item["beta"] = cfg.beta
        if cfg.posterior is not None:
            item["posterior"] = cfg.posterior
        for name in ("encoder_blocks", "decoder_blocks", "widths", "decoder_widths"):
            value = getattr(cfg, name)
            if value is not None:
                item[name] = list(value)
    return item


def assert_resume_matches(cfg, architecture, loss, checkpoint):
    """The checkpoint must be the same network and the same loss.

    Learning rates, the schedule, and a vae beta may change. beta comes from the config.
    """
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

    (
        arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels,
        bilinear, codes, patch, res_scale, vote, latent, factor, beta,
        encoder_blocks, decoder_blocks, widths, decoder_widths, posterior,
    ) = _take_model(
        data, where, architecture, architecture_from,
    )
    batch = _take_int(data, "batch", where, positive=True)
    lr = _take_float(data, "lr", where, positive=True)
    holdout = _take_float(data, "holdout", where, positive=True)
    if holdout >= 1:
        raise SystemExit(f"{where}: holdout must be below 1")
    val_every = _take_int(data, "val_every", where, positive=True)
    val_count = _take_int(data, "val_count", where, positive=True)
    seed = _take_int(data, "seed", where, default=0)
    steps = _take_int(data, "steps", where, positive=True, default=None)
    out = _take_path(data, "out", where)
    variants = _take_variants(data, where)
    crop = _take_crop(data, where)
    freeze_encoder = _take_flag(data, "freeze_encoder", where)
    latent_noise_tau = _take_latent_noise(data, where, arch)
    terms = _take_loss(data, where, arch)
    if arch in VAE_ARCHS:
        if variants is not None:
            raise SystemExit(f"{where}: {arch} does not take variants")
        if arch == "vae" and any(term.name == "gan" for term in terms):
            raise SystemExit(f"{where}: vae does not take a gan term")
        if crop is not None and crop % factor != 0:
            raise SystemExit(f"{where}: crop must be a multiple of factor {factor}")
    elif crop is not None:
        raise SystemExit(f"{where}: only vae and vae2 use crop")
    if arch not in VAE_ARCHS and any(term.name == "color" for term in terms):
        raise SystemExit(f"{where}: only vae and vae2 use a color term")
    if arch != "vae2" and any(term.start is not None for term in terms):
        raise SystemExit(f"{where}: only a vae2 gan term takes start")
    if freeze_encoder and arch != "vae2":
        raise SystemExit(f"{where}: only vae2 takes freeze_encoder")
    schedule, schedule_steps = _take_schedule(data, where, lr)
    if schedule_steps is not None and steps is not None:
        raise SystemExit(f"{where}: steps is set twice, at the top and under schedule")
    if schedule_steps is not None:
        steps = schedule_steps
    if schedule.kind == "cosine" and steps is None:
        raise SystemExit(f"{where}: cosine needs steps")
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
        val_every=val_every,
        val_count=val_count,
        seed=seed,
        steps=steps,
        out=out,
        terms=tuple(terms),
        radius=radius,
        stem=stem,
        sharpness=sharpness,
        reject=reject,
        levels=levels,
        bilinear=bilinear,
        codes=codes,
        patch=patch,
        res_scale=res_scale,
        vote=vote,
        variants=variants,
        latent=latent,
        factor=factor,
        beta=beta,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        widths=widths,
        decoder_widths=decoder_widths,
        posterior=posterior,
        latent_noise_tau=latent_noise_tau,
        crop=crop,
        freeze_encoder=freeze_encoder,
        schedule=schedule,
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
    if arch == "vote":
        if "in_frames" in model:
            raise SystemExit(
                f"{where}: vote reads the previous, middle, and next frame "
                "and does not take in_frames"
            )
        in_frames = 3
    elif arch in VAE_ARCHS:
        if "in_frames" in model:
            raise SystemExit(
                f"{where}: {arch} reads one master frame and does not take in_frames"
            )
        in_frames = 1
    elif arch in ("vq", "swinir", "swinir_light", "swinir_real", "edsr") and "in_frames" not in model:
        in_frames = 1
    else:
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
    bilinear, codes, patch = _take_vq(model, arch, where)
    posterior = _take_posterior(model, arch, where)
    latent, factor, beta = _take_vae(model, arch, where, posterior)
    encoder_blocks, decoder_blocks, widths, decoder_widths = _take_levels(model, arch, factor, where)
    vote = None
    if arch == "vote":
        stem = _take_int(model, "stem", where)
        if stem < 0:
            raise SystemExit(f"{where}: stem must be zero or positive")
        vote = _take_int(model, "vote", where)
        if vote < 0:
            raise SystemExit(f"{where}: vote must be zero or positive")
    res_scale = _take_res_scale(model, arch, where)
    _reject_unknown(model, where)
    if in_frames % 2 != 1:
        raise SystemExit(f"{where}: in_frames must be odd")
    fixed = preset["frames"]
    if fixed is not None and in_frames != fixed:
        raise SystemExit(f"{where}: {arch} takes {fixed} input frame, not {in_frames}")
    if arch in ("shift", "pyramid") and in_frames < 3:
        raise SystemExit(f"{where}: {arch} needs at least 3 input frames")
    return (
        arch, blocks, channels, in_frames, radius, stem, sharpness, reject, levels,
        bilinear, codes, patch, res_scale, vote, latent, factor, beta,
        encoder_blocks, decoder_blocks, widths, decoder_widths, posterior,
    )


def _take_posterior(data, arch, where):
    """vae2 posterior as written, None when left out (gaussian)."""
    if "posterior" not in data:
        return None
    if arch != "vae2":
        raise SystemExit(f"{where}: only vae2 takes posterior")
    return _take_choice(data, "posterior", where, ("gaussian", "fixed"))


def _take_latent_noise(data, where, arch):
    """latent_noise: {tau: t}. Each frame adds s * std_c * N(0, 1), s ~ |N(0, t^2)|."""
    if "latent_noise" not in data:
        return 0.0
    if arch != "vae2":
        raise SystemExit(f"{where}: only vae2 takes latent_noise")
    raw = data.pop("latent_noise")
    label = f"{where}: latent_noise"
    if not isinstance(raw, dict):
        raise SystemExit(f"{label} must be a mapping with tau")
    tau = _take_float(raw, "tau", label)
    if tau < 0:
        raise SystemExit(f"{label}: tau must be zero or positive")
    _reject_unknown(raw, label)
    return tau


_LEVEL_KEYS = ("encoder_blocks", "decoder_blocks", "widths", "decoder_widths")


def _take_levels(data, arch, factor, where):
    """vae2 blocks and width multiples per level, as written. None where left out."""
    if arch != "vae2":
        foreign = [name for name in _LEVEL_KEYS if name in data]
        if foreign:
            raise SystemExit(f"{where}: only vae2 uses {', '.join(foreign)}")
        return None, None, None, None
    raw = [data.pop(name, None) for name in _LEVEL_KEYS]
    layout(factor, *raw, where=where)
    return tuple(None if value is None else tuple(value) for value in raw)


def _take_res_scale(data, arch, where):
    if arch not in ("edsr", "vote"):
        if "res_scale" in data:
            raise SystemExit(f"{where}: only edsr and vote use res_scale")
        return None
    return _take_float(data, "res_scale", where, positive=True)


def _take_vq(data, arch, where):
    if arch != "vq":
        foreign = [name for name in ("bilinear", "codes", "patch") if name in data]
        if foreign:
            names = ", ".join(foreign)
            raise SystemExit(f"{where}: only vq uses {names}")
        return None, None, None
    if "bilinear" not in data:
        bilinear = True
    else:
        bilinear = _take_bool(data, "bilinear", where)
    codes = _take_int(data, "codes", where, positive=True, default=1024)
    if codes > 65536:
        raise SystemExit(f"{where}: codes must be from 1 to 65536")
    patch = _take_int(data, "patch", where, positive=True, default=4)
    if patch > 16:
        raise SystemExit(f"{where}: patch must be from 1 to 16")
    return bilinear, codes, patch


def _take_vae(data, arch, where, posterior=None):
    if arch not in VAE_ARCHS:
        foreign = [name for name in ("latent", "factor", "beta") if name in data]
        if foreign:
            names = ", ".join(foreign)
            raise SystemExit(f"{where}: only vae and vae2 use {names}")
        return None, None, None
    latent = _take_int(data, "latent", where, positive=True)
    factor = _take_int(data, "factor", where, positive=True)
    if factor < 2 or factor > 16 or factor & (factor - 1):
        raise SystemExit(f"{where}: factor must be a power of two from 2 to 16")
    if posterior == "fixed":
        if "beta" in data:
            raise SystemExit(
                f"{where}: posterior fixed has no KL, so beta does nothing. Remove beta"
            )
        return latent, factor, None
    beta = _take_float(data, "beta", where, positive=True)
    return latent, factor, beta


def _take_flag(data, key, where):
    """An optional true or false. Missing means false."""
    if key not in data:
        return False
    return _take_bool(data, key, where)


def _take_crop(data, where):
    """A fixed window for a vae. Missing means the whole frame. Empty is an error."""
    if "crop" not in data:
        return None
    value = data.pop("crop")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{where}: crop must be a positive integer")
    return value


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


SCHEDULE_KINDS = ("const", "cosine", "plateau")


def cosine_learning_rate(step, steps, peak):
    """One hump over steps 1..steps. Halfway is `peak`, and both ends sit just off zero.

    Each step takes the middle of its slot on a cosine that is flat at the
    bottom, the top, and the bottom again, so the rise and the fall leave
    and arrive gently. The first step and the last step are the same rate.
    """
    if step <= 0 or step > steps:
        return 0.0
    progress = (step - 0.5) / steps
    return peak * 0.5 * (1.0 - math.cos(2.0 * math.pi * progress))


class Plateau:
    """Drop `lr` when validation misses, and stop once that rate is already the floor.

    A miss is a validation that does not beat the best by `min_delta`, a
    fraction of that best. After `patience` misses the rate is multiplied
    by `factor`, and never set below `min_lr`. The next miss at `min_lr`
    asks the run to stop. A real improvement clears the miss count and
    keeps going, including at the floor.
    """

    def __init__(self, lr, patience, factor, min_delta, min_lr, best=None, bad=0):
        self.lr = float(lr)
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        self.min_lr = float(min_lr)
        self.best = None if best is None else float(best)
        self.bad = bad

    def observe(self, val_loss):
        """Return 'ok', 'drop', or 'stop'."""
        if self.best is None or val_loss < self.best * (1.0 - self.min_delta):
            self.best = float(val_loss)
            self.bad = 0
            return "ok"
        self.bad += 1
        if self.bad < self.patience:
            return "ok"
        if self.lr <= self.min_lr:
            return "stop"
        nxt = self.lr * self.factor
        if nxt < self.min_lr:
            nxt = self.min_lr
        self.lr = nxt
        self.bad = 0
        return "drop"

    def state_dict(self, peak):
        return {
            "kind": "plateau",
            "peak": float(peak),
            "lr": self.lr,
            "best": self.best,
            "bad": self.bad,
        }


def restore_plateau(cfg, saved, where):
    """Continue a plateau, or start at `lr` when the checkpoint has nothing to continue.

    The note is 'missing' when the checkpoint has no schedule, 'reset' when
    `lr` changed, and 'restored' when the miss count and the current rate
    come from the checkpoint. `where` is the checkpoint path named in errors.
    """
    schedule = cfg.schedule
    fresh = Plateau(cfg.lr, schedule.patience, schedule.factor, schedule.min_delta, schedule.min_lr)
    if saved is None:
        return fresh, "missing"
    if not isinstance(saved, dict):
        raise SystemExit(f"{where}: schedule must be a mapping")
    if saved.get("kind") != "plateau":
        return fresh, "missing"
    peak = saved.get("peak")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)):
        raise SystemExit(f"{where}: schedule is missing peak")
    if not math.isclose(float(peak), cfg.lr, rel_tol=1e-6, abs_tol=0.0):
        return fresh, "reset"
    lr = saved.get("lr")
    bad = saved.get("bad")
    best = saved.get("best")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0:
        raise SystemExit(f"{where}: schedule has no learning rate")
    if isinstance(bad, bool) or not isinstance(bad, int) or bad < 0:
        raise SystemExit(f"{where}: schedule has no miss count")
    if best is not None and (
        isinstance(best, bool) or not isinstance(best, (int, float)) or not math.isfinite(float(best))
    ):
        raise SystemExit(f"{where}: schedule has no best validation")
    if float(lr) < schedule.min_lr:
        raise SystemExit(
            f"{where}: learning rate {float(lr):g} is below min_lr {schedule.min_lr:g}"
        )
    return Plateau(
        float(lr), schedule.patience, schedule.factor, schedule.min_delta, schedule.min_lr,
        best=best, bad=bad,
    ), "restored"


def _take_schedule(data, where, lr):
    """The schedule section. A bare name is the kind. Missing means const.

    `steps` may sit here or at the top of the file, not in both. Cosine
    requires it. Plateau uses patience 8, factor 0.5, min_delta 0.01, and
    min_lr at one hundredth of `lr` when those keys are left out.
    """
    if "schedule" not in data:
        return Schedule("const"), None
    raw = data.pop("schedule")
    label = f"{where}: schedule"
    if isinstance(raw, str):
        kind = raw
        raw = {}
        if kind not in SCHEDULE_KINDS:
            known = ", ".join(SCHEDULE_KINDS)
            raise SystemExit(f"{label}: unknown kind {kind!r}. choices: {known}")
    elif isinstance(raw, dict):
        kind = _take_choice(raw, "kind", label, SCHEDULE_KINDS)
    else:
        raise SystemExit(f"{label} must be a name or a mapping")
    steps = _take_int(raw, "steps", label, positive=True, default=None)
    if kind != "plateau":
        _reject_unknown(raw, label)
        return Schedule(kind), steps
    patience = _take_int(raw, "patience", label, positive=True, default=8)
    factor = _take_float(raw, "factor", label, positive=True, default=0.5)
    if factor >= 1:
        raise SystemExit(f"{label}: factor must be below 1")
    min_delta = _take_float(raw, "min_delta", label, default=0.01)
    if min_delta < 0 or min_delta >= 1:
        raise SystemExit(f"{label}: min_delta must be at least 0 and below 1")
    min_lr = _take_float(raw, "min_lr", label, positive=True, default=lr * 0.01)
    if min_lr >= lr:
        raise SystemExit(f"{label}: min_lr must be below lr")
    _reject_unknown(raw, label)
    return Schedule(kind, patience, factor, min_delta, min_lr), steps


def _take_loss(data, where, arch=None):
    """Loss terms. The loss is the sum of weight * raw. A vae2 gan term stands apart.

    A weight is any positive multiplier. vgg weighs each layer under layers.

    color is L1 on average-pooled color, for vae and vae2. `space` is ycbcr
    (the default) or oklab, `pool` the pooling factor (default 8). A vae2 gan
    term may name `start`, the first step whose generator loss includes it.
    """
    if "loss" not in data:
        raise SystemExit(f"{where}: missing loss")
    loss = data.pop("loss")
    if not isinstance(loss, dict):
        raise SystemExit(f"{where}: loss must be a mapping")
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
        if name in SIDE_TERMS and arch != "vae2":
            raise SystemExit(f"{label}: only vae2 takes {name}")
        if name == "vgg":
            if "weight" in raw:
                raise SystemExit(
                    f"{label}: vgg has no weight. Weigh each layer: "
                    f"layers: {{relu2_2: <weight>, relu3_4: <weight>}}"
                )
            weight = None
        else:
            weight = _take_float(raw, "weight", label, positive=True)
        if name == "pixel":
            kind = _take_choice(raw, "kind", label, ("mse", "l1"))
            terms.append(Term(name=name, weight=weight, kind=kind))
        elif name == "vgg":
            layers = _take_layers(raw, label)
            terms.append(Term(name=name, weight=None, layers=layers))
        elif name == "color":
            space = raw.pop("space", COLOR_SPACES[0])
            if space not in COLOR_SPACES:
                known = ", ".join(COLOR_SPACES)
                raise SystemExit(f"{label}: unknown space {space!r}. choices: {known}")
            pool = _take_int(raw, "pool", label, positive=True, default=COLOR_POOL)
            terms.append(Term(name=name, weight=weight, space=space, pool=pool))
        elif name == "mu_l2":
            terms.append(Term(name=name, weight=weight))
        elif name == "scale_eq":
            factors = raw.pop("factors", list(SCALE_EQ_FACTORS))
            if (
                not isinstance(factors, list) or not factors
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in factors)
            ):
                raise SystemExit(f"{label}: factors must be a list of integers from 2 up")
            terms.append(Term(name=name, weight=weight, factors=tuple(factors)))
        else:
            crop = _take_int(raw, "crop", label, positive=True)
            if crop % 8 != 0:
                raise SystemExit(f"{label}: crop must be a multiple of 8")
            disc_lr = _take_float(raw, "disc_lr", label, positive=True)
            start = _take_int(raw, "start", label, default=None)
            if start is not None and start < 0:
                raise SystemExit(f"{label}: start must be zero or positive")
            terms.append(Term(
                name=name, weight=weight, crop=crop, disc_lr=disc_lr, start=start,
            ))
        _reject_unknown(raw, label)
    summed = [
        term for term in terms
        if not (arch == "vae2" and term.name == "gan") and term.name not in SIDE_TERMS
    ]
    if not summed:
        raise SystemExit(f"{where}: vae2 needs a term beside gan")
    return terms


def _take_layers(data, where):
    """VGG layers mapped to their weights, in file order."""
    if "layers" not in data:
        raise SystemExit(f"{where}: missing layers")
    layers = data.pop("layers")
    if not isinstance(layers, dict) or not layers:
        raise SystemExit(f"{where}: layers must map vgg layers to weights")
    known = ", ".join(VGG_LAYERS)
    pairs = []
    for layer, weight in layers.items():
        if not isinstance(layer, str) or layer not in VGG_LAYERS:
            raise SystemExit(f"{where}: unknown vgg layer {layer!r}. choices: {known}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise SystemExit(f"{where}: vgg layer {layer} weight must be a positive number")
        pairs.append((layer, float(weight)))
    return tuple(pairs)


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


def _take_variants(data, where):
    """Optional regex matched against each path relative to low/."""
    if "variants" not in data:
        return None
    value = data.pop("variants")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{where}: variants must be a regex")
    try:
        re.compile(value)
    except re.error as exc:
        raise SystemExit(f"{where}: variants: {exc}") from exc
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
