"""Train on random full frames from paired videos.

The recipe is a YAML file. Masters live in the data directory's orig/ and
degraded variants in its low/. catalog.json beside those folders names the
pairs, their sizes, and the frame count of each title. Training reads that
file once at startup. An optional variants regex is searched against each
path relative to low/, such as x264-crf28/film.720.mkv. Without it, every
pair is used. A step draws one film to pick a resolution, then fills the
batch with films of that resolution. The draw is without replacement when
that resolution has at least as many films as the batch, and with replacement
when it has fewer, so a lone size still fills whole batches. Each film draws
one variant per sample it received, with the same rule. Two spans of frames
are held out for validation.

`arch: vae` reconstructs master frames and does not read a low variant.
Without crop each sample is the whole frame, and a batch stays inside one
master size. With crop every sample is that window, so sizes can share a batch.
`arch: vae2` reads the same batches.

A vae2 may add a gan term. Its discriminator trains from the first step. Its
generator term joins at `start` with the adaptive weight of latent diffusion:
the gradient norm of the objective over that of the gan loss, both at the last
decoder convolution, times the term's weight. That term stays out of the
validation objective. freeze_encoder: true trains the decoder alone, for a
fine-tune from a vae2 checkpoint. A vae2 validation also stores the latent
shift and scale in the model and logs val/latent/scale.

Training batches are decoded on one side thread into a queue of three CPU
batches. A frame read opens the video and closes it before the next read.
An unreadable training clip is logged and the loader draws a new batch.
The step copies a batch onto the device. Validation reads on the main thread.

The loss is the weighted sum of the terms in the file. The first term anchors
the magnitude. balance start freezes the scales on the validation frames
before the first step. balance running keeps the weights as shares.

schedule chooses how lr moves. Left out, it is const: lr stays put. steps,
at the top of the file or under schedule, stops const and plateau after
that many new updates; without it those runs wait for Ctrl-C. A resume
counts that cap from zero again. cosine needs steps. Its rate follows one
hump on the run's step counter: the first step is just above zero, the
halfway step is lr, and the last step is just above zero. A resume
continues that same hump.
plateau reads the validation loss. After patience misses it multiplies lr
by factor, and never goes below min_lr. A miss at that floor stops the
run. A resume keeps the dropped rate and the miss count when lr is the
same peak. The discriminator, when there is one, keeps the same fraction
of disc_lr.

A fresh run validates once at step 0, before any update. Where the last
layer starts at zero, that pass is the bilinear upscale. Modified leaves
that layer at the default initialization, so step 0 already includes a
correction. A fine-tune's step 0 is the generator that was loaded. A vae
has no bilinear base, so that first pass is the untrained reconstruction.

A gan term adds a U-Net discriminator. Each checkpoint is a directory
checkpoints/<step>/ holding two files. model.pt is the generator and the
architecture that builds it. whoodeo-apply and a fine-tune read only that
file. train.pt is the step, the optimizers, the discriminator, the loss
recipe, and the loss scales. latest in the run points at the newest of
those directories, and latest in the runs directory follows the newest
checkpoint written. The run directory also gets a TensorBoard event file for the losses.
The vote model adds neighbor/weight and neighbor/grad on each step, and
neighbor/image when validation runs. weight and grad cover every kernel
in the shared vote. A vae adds train/kl and val/kl. The step loss includes
beta times that KL. The validation loss the plateau reads is the pixel and
VGG objective. It also records latent/mu_mean, latent/mu_std, latent/sigma_mean
and latent/spread for the batch, the same four on the validation set, and the
channel summaries latent/active and latent/kl_min. active counts the channels
whose μ varies with std above 0.1, kl_min is the KL of the weakest channel.
Each validation also writes the per-channel table (μ std, σ mean, KL) to the
text tab as latent/channels. spread is the square root of
var(μ) + mean(σ²) and sits near 1 when the code matches the unit Gaussian.
The validation histogram is latent/mu.
whoodeo-board serves every run.

--resume continues a run from its newest checkpoint: the same architecture,
the same loss, the Adam state, and the loss scales. The schedule comes
from the file. A plateau resume keeps its rate and its miss count when lr
is unchanged. Resuming an older checkpoint starts a new run, so the later
checkpoints stay where they are. An older model.pt that still carries the
optimizer loads the same way, and the next save writes checkpoints/.

--model fine-tunes that generator under the loss in the file. The optimizer
and the discriminator start over, the loss scales are measured again, and
the result is a new run. The recipe may omit the model: section; the
architecture then comes from the model file. A model: section that is
still present has to match the file.
"""

import argparse
import queue
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import torch
from torch.utils.tensorboard import SummaryWriter
from whoodeo.catalog import data_root, runs_root
from whoodeo.checkpoint import continue_run, load_generator, load_resume, run_directory, save_run
from whoodeo.config import (
    VAE_ARCHS, architecture_dict, assert_resume_matches, cosine_learning_rate,
    load_config, restore_plateau, shape_text,
)
from whoodeo.films import (
    UnreadableSample, draw_indexes, load_films, load_masters, require_films,
    require_masters, sample_index, unreadable,
)
from whoodeo.live import add_live_args, open_live
from whoodeo.models import build_model
from whoodeo.models.common import pad_to
from whoodeo.models.discriminator import UNetDiscriminatorSN, gan_bce
from whoodeo.models.vae import LatentTally, latent_scalars, latent_table
from whoodeo.objective import Objective, color_loss, pixel_loss
from whoodeo.video import png_bytes, rgb_image


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sample(film, variant, index, radius):
    try:
        low_frames = variant.low.frames_around(index, radius)
    except (RuntimeError, OSError, av.error.FFmpegError) as exc:
        where = f"low {variant.name}/{variant.low.path.name}"
        raise unreadable(where, index, exc) from exc
    try:
        hr = film.orig.frames_around(index, radius)[radius]
    except (RuntimeError, OSError, av.error.FFmpegError) as exc:
        raise unreadable(f"master {film.orig.path.name}", index, exc) from exc
    stacked = torch.cat(low_frames, dim=0)
    return stacked, hr


def frame_size(film):
    """Low resolution shared by every variant of this film."""
    low = film.variants[0].low
    return low.width, low.height


def make_batch_specs(rng, films, split, count):
    """count frame indexes that share a low resolution.

    One uniform film picks the resolution. The samples are films of that
    resolution, without replacement when the group has at least count films
    and with replacement when it has fewer. A resolution with a single film
    therefore fills the batch by itself and keeps its share of frames. Each
    film draws one variant per sample it received, under the same rule.
    """
    anchor = rng.randrange(len(films))
    size = frame_size(films[anchor])
    group = [index for index, film in enumerate(films) if frame_size(film) == size]
    film_indexes = [group[pick] for pick in draw_indexes(rng, len(group), count)]

    slots = {}
    for slot, film_index in enumerate(film_indexes):
        slots.setdefault(film_index, []).append(slot)
    variant_indexes = [None] * count
    for film_index, positions in slots.items():
        drawn = draw_indexes(rng, len(films[film_index].variants), len(positions))
        for slot, variant_index in zip(positions, drawn):
            variant_indexes[slot] = variant_index

    specs = []
    for slot, film_index in enumerate(film_indexes):
        film = films[film_index]
        variant_index = variant_indexes[slot]
        specs.append((
            film_index,
            variant_index,
            sample_index(rng, film, split),
        ))
    return specs


def batch_from_specs(films, specs, radius):
    lows = []
    highs = []
    for film_index, variant_index, index in specs:
        film = films[film_index]
        low, high = load_sample(film, film.variants[variant_index], index, radius)
        lows.append(low)
        highs.append(high)
    return torch.stack(lows), torch.stack(highs)


def batch_on_device(films, specs, radius, device):
    low, high = batch_from_specs(films, specs, radius)
    return low.to(device), high.to(device)


_QUEUE_DEPTH = 3
_QUEUE_POLL = 0.2
# One bad clip is skipped. This many failed batches in a row means the
# draw can no longer produce a readable batch, so the run stops.
_UNREADABLE_LIMIT = 32


@dataclass
class PreparedBatch:
    low: torch.Tensor
    high: torch.Tensor
    width: int
    height: int
    box: tuple | None


class BatchLoader:
    """One thread of CPU training batches.

    Each frame read opens its video and closes it. The thread never moves
    tensors onto a device. An unreadable clip is logged and the batch is
    drawn again. After too many failures in a row the error stops the
    loader. ``put`` times out so a full queue cannot block shutdown.
    ``get`` raises an error from the thread instead of waiting on a dead
    loader.
    """

    def __init__(self, produce):
        self._produce = produce
        self._queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="whoodeo-batch", daemon=False)

    def start(self):
        self._thread.start()

    def get(self):
        while True:
            try:
                item = self._queue.get(timeout=_QUEUE_POLL)
            except queue.Empty:
                if not self._thread.is_alive():
                    raise RuntimeError("batch loader stopped")
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_QUEUE_POLL)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        failed = 0
        try:
            while not self._stop.is_set():
                try:
                    prepared = self._produce()
                except UnreadableSample as exc:
                    failed += 1
                    print(f"skip batch: {exc}", file=sys.stderr, flush=True)
                    if failed >= _UNREADABLE_LIMIT:
                        raise RuntimeError(
                            f"{failed} batches in a row could not be read"
                        ) from exc
                    continue
                failed = 0
                if not self._put(prepared):
                    return
        except BaseException as exc:
            self._put(exc)


def write_scalars(writer, step, train_loss, val_loss, train_parts, val_parts, usage, neighbor_image, lr, extras=None):
    """One TensorBoard point per value that this step actually has."""
    def emit(tag, value):
        if value is not None:
            writer.add_scalar(tag, value, step)

    emit("loss/train", train_loss)
    emit("loss/val", val_loss)
    emit("lr", lr)
    for prefix, parts in (("train", train_parts), ("val", val_parts)):
        if parts is None:
            continue
        pixel, vgg, gan, d_loss = parts
        emit(f"{prefix}/pixel", pixel)
        emit(f"{prefix}/vgg", vgg)
        emit(f"{prefix}/gan", gan)
        emit(f"{prefix}/d", d_loss)
    if usage is not None:
        emit("neighbor/weight", usage.get("weight"))
        emit("neighbor/grad", usage.get("grad"))
    emit("neighbor/image", neighbor_image)
    if extras:
        for tag, value in extras.items():
            emit(tag, value)
    writer.flush()


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def exact_box(image, size, rng):
    """A window of exactly `size` on each side. `rng` None keeps the center."""
    _batch, _channels, height, width = image.shape
    if height < size or width < size:
        raise SystemExit(f"frame {width}x{height} is smaller than a {size} crop")
    if rng is None:
        top = (height - size) // 2
        left = (width - size) // 2
    else:
        top = rng.randrange(0, height - size + 1)
        left = rng.randrange(0, width - size + 1)
    return top, left, size, size


def sample_box(image, size, rng):
    """A crop whose sides are multiples of 8, so the U-Net skips line up."""
    _batch, _channels, height, width = image.shape
    crop_h = min(size, height - height % 8)
    crop_w = min(size, width - width % 8)
    if crop_h < 8 or crop_w < 8:
        raise SystemExit(f"frame {width}x{height} is smaller than an 8-pixel discriminator crop")
    if rng is None:
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
    else:
        top = rng.randrange(0, height - crop_h + 1)
        left = rng.randrange(0, width - crop_w + 1)
    return top, left, crop_h, crop_w


def take(image, box):
    top, left, crop_h, crop_w = box
    return image[:, :, top:top + crop_h, left:left + crop_w]


def part_text(pixel, vgg, gan, d_loss):
    parts = []
    if pixel is not None:
        parts.append(f"pixel {pixel:.6f}")
    if vgg is not None:
        parts.append(f"vgg {vgg:.6f}")
    if gan is not None:
        parts.append(f"gan {gan:.6f}")
    if d_loss is not None:
        parts.append(f"d {d_loss:.4f}")
    return "  ".join(parts)


def describe_loss(cfg, objective):
    print(f"loss balance {cfg.balance}  anchor {cfg.terms[0].name}", flush=True)
    for term in cfg.terms:
        if term.name not in objective.names:
            start = 0 if term.start is None else term.start
            print(
                f"  {term.name} crop {term.crop}  weight {term.weight:.2f}  "
                f"adaptive  from step {start}",
                flush=True,
            )
            continue
        raw = objective.term_ref[term.name]
        scale = objective.scale[term.name]
        weighted = scale * raw
        if term.name == "pixel":
            detail = term.kind
        elif term.name == "vgg":
            detail = "+".join(term.layers)
        elif term.name == "color":
            detail = "ycbcr/8"
        else:
            detail = f"crop {term.crop}"
        print(
            f"  {term.name} {detail}  weight {term.weight:.2f}  "
            f"scale {scale:.6g}  ref {raw:.6g}  weighted {weighted:.6g}",
            flush=True,
        )


def raw_terms(objective, cfg, pred, high, discriminator, box):
    raw = {}
    layer_raw = {}
    pixel = cfg.term("pixel")
    if pixel is not None:
        raw["pixel"] = pixel_loss(pred, high, pixel.kind)
    if cfg.term("vgg") is not None:
        raw["vgg"], layer_raw = objective.vgg_raw(pred, high)
    if cfg.term("color") is not None:
        raw["color"] = color_loss(pred, high)
    if "gan" in objective.names:
        raw["gan"] = gan_bce(discriminator(take(pred, box)), True)
    return raw, layer_raw


def measure_raw_means(model, films, specs, radius, batch_size, device, cfg, objective, discriminator):
    model.eval()
    if discriminator is not None:
        discriminator.eval()
    count = 0
    pixel_acc = 0.0
    gan_acc = 0.0
    layer_acc = {name: 0.0 for name in objective.vgg_layers}
    pixel = cfg.term("pixel")
    gan = cfg.term("gan")
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            low, high = batch_on_device(films, chunk, radius, device)
            pred = model(low)
            frames = len(chunk)
            if pixel is not None:
                pixel_acc += pixel_loss(pred, high, pixel.kind).item() * frames
            if objective.vgg is not None:
                for name, loss in objective.vgg.layer_losses(pred, high).items():
                    layer_acc[name] += loss.item() * frames
            if gan is not None:
                box = sample_box(high, gan.crop, None)
                fake = discriminator(take(pred, box))
                gan_acc += gan_bce(fake, True).item() * frames
            count += frames
    model.train()
    if discriminator is not None:
        discriminator.train()
    term_means = {}
    if pixel is not None:
        term_means["pixel"] = pixel_acc / count
    if gan is not None:
        term_means["gan"] = gan_acc / count
    layer_means = {name: value / count for name, value in layer_acc.items()}
    return layer_means, term_means


def save_still(folder, name, tensor):
    """Write a reference image once. A later step leaves the file in place."""
    path = folder / f"{name}.png"
    if not path.is_file():
        path.write_bytes(png_bytes(rgb_image(tensor)))


def save_step_image(folder, stem, step, tensor):
    (folder / f"{stem}-{step:06d}.png").write_bytes(png_bytes(rgb_image(tensor)))


def save_val_images(directory, step, index, low, pred, high, width):
    """Write one validation sample. low and master stay; the upscale is per step."""
    folder = directory / f"{index:0{width}d}"
    folder.mkdir(parents=True, exist_ok=True)
    save_still(folder, "low", low)
    save_still(folder, "master", high)
    save_step_image(folder, "upscale", step, pred)


def neighbor_image_effect(model, low, pred):
    """Mean absolute change when both neighbors are replaced by the middle frame.

    The bias of the vote is present in both forwards, so this is the part that
    comes from the frames actually differing. None for a model without a vote.
    """
    if not hasattr(model, "neighbor_usage"):
        return None
    count = model.in_frames
    mid = count // 2
    center = low[:, mid * 3:(mid + 1) * 3]
    flat = center.repeat(1, count, 1, 1)
    alt = model(flat)
    return (pred - alt).abs().mean().item()


def evaluate(model, films, specs, radius, batch_size, device, cfg, objective, discriminator, image_dir=None, step=None):
    """Mean of the weighted objective. The GAN part uses the center crop.

    With image_dir set, each sample is written under image_dir/<index>/.
    The last return is the neighbor image effect, or None when the model
    has no vote.
    """
    model.eval()
    disc_training = discriminator is not None and discriminator.training
    if discriminator is not None:
        discriminator.eval()
    loss_acc = 0.0
    pixel_acc = 0.0
    vgg_acc = 0.0
    gan_acc = 0.0
    d_acc = 0.0
    image_acc = 0.0
    saw_d = False
    saw_image = False
    count = 0
    saved = 0
    width = max(2, len(str(max(len(specs) - 1, 0))))
    center = radius * 3
    gan = cfg.term("gan")
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            low, high = batch_on_device(films, chunk, radius, device)
            pred = model(low)
            effect = neighbor_image_effect(model, low, pred)
            if effect is not None:
                image_acc += effect * len(chunk)
                saw_image = True
            if image_dir is not None:
                for offset in range(len(chunk)):
                    save_val_images(
                        image_dir, step, saved + offset,
                        low[offset, center:center + 3],
                        pred[offset],
                        high[offset],
                        width,
                    )
                saved += len(chunk)
            box = None
            d_here = None
            if gan is not None:
                box = sample_box(high, gan.crop, None)
                fake = discriminator(take(pred, box))
                real = discriminator(take(high, box))
                d_here = 0.5 * (gan_bce(real, True).item() + gan_bce(fake, False).item())
            raw, _layer_raw = raw_terms(objective, cfg, pred, high, discriminator, box)
            total, weighted = objective.combine(raw)
            frames = len(chunk)
            loss_acc += total.item() * frames
            if "pixel" in weighted:
                pixel_acc += weighted["pixel"].item() * frames
            if "vgg" in weighted:
                vgg_acc += weighted["vgg"].item() * frames
            if "gan" in weighted:
                gan_acc += weighted["gan"].item() * frames
            if d_here is not None:
                d_acc += d_here * frames
                saw_d = True
            count += frames
    model.train()
    if disc_training:
        discriminator.train()
    pixel_mean = pixel_acc / count if cfg.term("pixel") is not None else None
    vgg_mean = vgg_acc / count if cfg.term("vgg") is not None else None
    gan_mean = gan_acc / count if gan is not None else None
    d_mean = d_acc / count if saw_d else None
    image_mean = image_acc / count if saw_image else None
    return loss_acc / count, pixel_mean, vgg_mean, gan_mean, d_mean, image_mean


def apply_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def learning_rate(cfg, step, plateau):
    """The generator rate for this step. Plateau carries its own current rate."""
    if cfg.schedule.kind == "cosine":
        return cosine_learning_rate(step, cfg.steps, cfg.lr)
    if cfg.schedule.kind == "plateau":
        return plateau.lr
    return cfg.lr


def describe_schedule(cfg, plateau, note):
    schedule = cfg.schedule
    if schedule.kind == "const":
        span = "until Ctrl-C" if cfg.steps is None else f"{cfg.steps} steps"
        print(f"schedule const  lr {cfg.lr:g}  {span}", flush=True)
        return
    if schedule.kind == "cosine":
        print(f"schedule cosine  peak {cfg.lr:g}  {cfg.steps} steps", flush=True)
        return
    line = (
        f"schedule plateau  peak {cfg.lr:g}  patience {schedule.patience}  "
        f"factor {schedule.factor:g}  min_delta {schedule.min_delta:g}  "
        f"min_lr {schedule.min_lr:g}"
    )
    if note == "restored":
        best = "none" if plateau.best is None else f"{plateau.best:.6f}"
        line += f"  resumed lr {plateau.lr:g}  best {best}  missed {plateau.bad}"
    elif note == "reset":
        line += "  lr changed, starting over"
    elif note == "missing":
        line += "  checkpoint has no plateau, starting over"
    print(line, flush=True)


def master_size(film):
    return film.orig.width, film.orig.height


def make_vae_specs(rng, films, split, count, crop):
    """count (film, frame) pairs.

    With a crop any master can share the batch, because every window is the
    same size. Without one, a film picks the master size and the batch stays
    inside that size.
    """
    if crop is not None:
        film_indexes = draw_indexes(rng, len(films), count)
    else:
        anchor = rng.randrange(len(films))
        size = master_size(films[anchor])
        group = [index for index, film in enumerate(films) if master_size(film) == size]
        film_indexes = [group[pick] for pick in draw_indexes(rng, len(group), count)]
    return [
        (film_index, sample_index(rng, films[film_index], split))
        for film_index in film_indexes
    ]


def load_master(film, index):
    try:
        return film.orig.frames_around(index, 0)[0]
    except (RuntimeError, OSError, av.error.FFmpegError) as exc:
        raise unreadable(f"master {film.orig.path.name}", index, exc) from exc


def vae_batch(films, specs, crop, rng):
    """Stack master frames. The crop is drawn only after every frame is read."""
    frames = [load_master(films[film_index], index) for film_index, index in specs]
    if crop is not None:
        cropped = []
        for frame in frames:
            batched = frame.unsqueeze(0)
            cropped.append(take(batched, exact_box(batched, crop, rng))[0])
        frames = cropped
    image = torch.stack(frames)
    return FrameBatch(image, int(image.shape[-1]), int(image.shape[-2]))


def prior_image(model, image):
    """One decode from the unit Gaussian at this frame's latent size.

    Goes through `decode`, so the KL stored by `forward` stays put.
    """
    padded, height, width = pad_to(image, model.factor)
    latent_h = padded.shape[-2] // model.factor
    latent_w = padded.shape[-1] // model.factor
    draw = torch.randn(
        1, model.latent, latent_h, latent_w, device=image.device, dtype=image.dtype,
    )
    with torch.no_grad():
        recon = model.decode(draw)
    return recon[0, :, :height, :width]


def measure_vae(model, films, specs, crop, batch_size, device, cfg, objective):
    """Raw term means on master frames, for the loss scales."""
    model.eval()
    count = 0
    pixel_acc = 0.0
    color_acc = 0.0
    layer_acc = {name: 0.0 for name in objective.vgg_layers}
    pixel = cfg.term("pixel")
    color = cfg.term("color")
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            image = vae_batch(films, chunk, crop, None).image.to(device)
            pred = model(image)
            frames = len(chunk)
            if pixel is not None:
                pixel_acc += pixel_loss(pred, image, pixel.kind).item() * frames
            if color is not None:
                color_acc += color_loss(pred, image).item() * frames
            if objective.vgg is not None:
                for name, loss in objective.vgg.layer_losses(pred, image).items():
                    layer_acc[name] += loss.item() * frames
            count += frames
    model.train()
    term_means = {}
    if pixel is not None:
        term_means["pixel"] = pixel_acc / count
    if color is not None:
        term_means["color"] = color_acc / count
    layer_means = {name: value / count for name, value in layer_acc.items()}
    return layer_means, term_means


def evaluate_vae(model, films, specs, crop, batch_size, device, cfg, objective, image_dir=None, step=None):
    """Mean of the weighted objective. The KL is returned beside it.

    master.png is written once. recon-<step>.png is the reconstruction.
    """
    model.eval()
    loss_acc = 0.0
    pixel_acc = 0.0
    vgg_acc = 0.0
    color_acc = 0.0
    kl_acc = 0.0
    count = 0
    saved = 0
    tally = LatentTally()
    width = max(2, len(str(max(len(specs) - 1, 0))))
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            image = vae_batch(films, chunk, crop, None).image.to(device)
            pred = model(image)
            tally.add(model.mu, model.logvar, sample=True)
            if image_dir is not None:
                for offset in range(len(chunk)):
                    folder = image_dir / f"{saved + offset:0{width}d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_still(folder, "master", image[offset])
                    save_step_image(folder, "recon", step, pred[offset])
                saved += len(chunk)
            raw, _layer_raw = raw_terms(objective, cfg, pred, image, None, None)
            total, weighted = objective.combine(raw)
            frames = len(chunk)
            loss_acc += total.item() * frames
            if "pixel" in weighted:
                pixel_acc += weighted["pixel"].item() * frames
            if "vgg" in weighted:
                vgg_acc += weighted["vgg"].item() * frames
            if "color" in weighted:
                color_acc += weighted["color"].item() * frames
            kl_acc += float(model.kl) * frames
            count += frames
    model.train()
    pixel_mean = pixel_acc / count if cfg.term("pixel") is not None else None
    vgg_mean = vgg_acc / count if cfg.term("vgg") is not None else None
    extras = None
    if cfg.term("color") is not None:
        extras = {"val/color": color_acc / count}
    return EvalResult(
        loss_acc / count,
        (pixel_mean, vgg_mean, None, None),
        None,
        kl_acc / count,
        tally.report(),
        extras,
    )


@dataclass
class FrameBatch:
    image: torch.Tensor
    width: int
    height: int


@dataclass
class StepOut:
    """One forward pass. `loss` is added to the objective, `gan` is its logged value."""

    raw: dict
    layer_raw: dict
    d_value: float | None
    images: dict | None
    width: int
    height: int
    loss: torch.Tensor | None = None
    gan: float | None = None
    extras: dict | None = None


@dataclass
class EvalResult:
    loss: float
    parts: tuple
    neighbor: float | None
    kl: float | None
    latent: dict | None = None
    extras: dict | None = None


def _scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


@dataclass
class Run:
    cfg: object
    out_dir: Path
    device: torch.device
    rng: random.Random
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    start_step: int
    plateau: object
    plateau_note: str | None
    train_state: object
    films: list
    discriminator: torch.nn.Module | None
    disc_optimizer: torch.optim.Optimizer | None
    gan: object

    def apply_rate(self, at_step):
        rate = learning_rate(self.cfg, at_step, self.plateau)
        apply_learning_rate(self.optimizer, rate)
        if self.disc_optimizer is not None:
            apply_learning_rate(
                self.disc_optimizer, self.gan.disc_lr * rate / self.cfg.lr,
            )
        return rate


def open_run(cfg, resume, finetune, load):
    """Device, films, model, and optimizer. None when a cosine run is already over.

    `load` prints its film report and returns the films.
    """
    if resume is not None and finetune is not None:
        raise SystemExit("pass either --resume or --model")
    generator = finetune
    train_state = None
    if resume is not None:
        generator, train_state = load_resume(resume)
        assert_resume_matches(cfg, generator.architecture, train_state.loss, generator.path)
        if cfg.schedule.kind == "cosine" and train_state.step >= cfg.steps:
            print(
                f"cosine sweep ended at step {cfg.steps}, "
                f"{train_state.path} is step {train_state.step}",
                flush=True,
            )
            return None

    out_dir = cfg.out
    if out_dir is None and train_state is not None:
        out_dir = continue_run(resume)
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = runs_root() / f"train-{stamp}"
    if (
        generator is not None
        and train_state is None
        and run_directory(generator.path).resolve() == out_dir.resolve()
    ):
        raise SystemExit(
            f"fine-tune would overwrite {out_dir}. Leave out unset, or choose a new directory."
        )

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"device {device}", flush=True)
    print(f"config {cfg.source}", flush=True)
    print(f"network {shape_text(*cfg.arch_key())}", flush=True)
    if generator is not None and train_state is None:
        shown = "unknown" if generator.step is None else str(generator.step)
        print(f"fine-tune {generator.path}  loaded step {shown}", flush=True)
    films = load()

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = out_dir / "config.yaml"
    if cfg.source.resolve() != copied.resolve():
        shutil.copyfile(cfg.source, copied)
    same_run = (
        generator is not None
        and out_dir.resolve() == run_directory(generator.path).resolve()
    )
    if cfg.schedule.kind == "cosine":
        limit = str(cfg.steps)
    elif cfg.steps is None and cfg.schedule.kind == "plateau":
        limit = "until the plateau stops or Ctrl-C"
    elif cfg.steps is None:
        limit = "until Ctrl-C"
    else:
        limit = str(cfg.steps)
    if train_state is not None and not same_run:
        print(f"new run from {generator.path}", flush=True)
    print(f"run {out_dir}  steps {limit}", flush=True)

    model, label = build_model(
        cfg.arch, cfg.blocks, cfg.channels, cfg.in_frames,
        radius=cfg.radius, stem=cfg.stem, sharpness=cfg.sharpness, reject=cfg.reject,
        levels=cfg.levels, bilinear=cfg.bilinear, codes=cfg.codes, patch=cfg.patch,
        res_scale=cfg.res_scale, vote=cfg.vote,
        latent=cfg.latent, factor=cfg.factor, beta=cfg.beta,
    )
    model = model.train().to(device)
    trainable = model.parameters()
    if cfg.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    print(label, flush=True)
    if cfg.freeze_encoder:
        print("encoder frozen", flush=True)
    if generator is not None:
        model.load_state_dict(generator.state_dict)
    if train_state is not None:
        optimizer.load_state_dict(train_state.optimizer)
        move_optimizer(optimizer, device)
        print(f"resume step {train_state.step}", flush=True)

    gan = cfg.term("gan")
    discriminator = None
    disc_optimizer = None
    if gan is not None:
        discriminator = UNetDiscriminatorSN().train().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan.disc_lr)
        count = sum(parameter.numel() for parameter in discriminator.parameters())
        if train_state is not None:
            if train_state.discriminator is None or train_state.disc_optimizer is None:
                raise SystemExit(f"{train_state.path} has no discriminator")
            discriminator.load_state_dict(train_state.discriminator)
            disc_optimizer.load_state_dict(train_state.disc_optimizer)
            move_optimizer(disc_optimizer, device)
            print("loaded discriminator", flush=True)
        print(
            f"discriminator unet-sn  {count} parameters  crop {gan.crop}  "
            f"lr {gan.disc_lr:g}",
            flush=True,
        )

    start_step = train_state.step if train_state is not None else 0
    plateau = None
    plateau_note = None
    if cfg.schedule.kind == "plateau":
        if train_state is None:
            plateau, _note = restore_plateau(cfg, None, "checkpoint")
        else:
            plateau, plateau_note = restore_plateau(cfg, train_state.schedule, train_state.path)

    run = Run(
        cfg, out_dir, device, rng, model, optimizer, start_step, plateau, plateau_note,
        train_state, films, discriminator, disc_optimizer, gan,
    )
    run.apply_rate(start_step)
    return run


def drive(run, preview, live_bind, make_val_specs, make_loader, measure, one_step, evaluate_split):
    """Loss scales, the live page, and the step loop.

    Validation specs are drawn before the loader starts. The plateau follows
    the objective from `evaluate_split`. Commitment and KL are added on the
    training step and logged beside that objective.
    """
    cfg = run.cfg
    objective = Objective(cfg, run.device)
    val_specs = make_val_specs()
    loader = None
    live = None
    writer = None
    try:
        loader = make_loader()
        loader.start()
        if run.train_state is not None:
            objective.load_state_dict(run.train_state.objective)
            print("loss scales from checkpoint", flush=True)
        else:
            layer_means, term_means = measure(objective, val_specs)
            objective.calibrate(layer_means, term_means)
        describe_loss(cfg, objective)
        describe_schedule(cfg, run.plateau, run.plateau_note)

        live = open_live(live_bind, preview)
        writer = SummaryWriter(log_dir=run.out_dir)

        def logged_parts(weighted, work):
            gan_value = work.gan
            if "gan" in weighted:
                gan_value = weighted["gan"].item()
            return (
                weighted["pixel"].item() if cfg.term("pixel") is not None else None,
                weighted["vgg"].item() if cfg.term("vgg") is not None else None,
                gan_value if run.gan is not None else None,
                work.d_value if run.gan is not None else None,
            )

        def step_extras(weighted, work):
            """Weighted terms beyond pixel, vgg, and gan, and what the step reports."""
            extras = {
                f"train/{name}": value.item()
                for name, value in weighted.items()
                if name not in ("pixel", "vgg", "gan")
            }
            if work.extras:
                extras.update(work.extras)
            return extras

        def batch_latent():
            mu = getattr(run.model, "mu", None)
            if mu is None:
                return None
            tally = LatentTally()
            tally.add(mu, run.model.logvar)
            return tally.report()

        def log(step, train_loss, val_loss, train_parts, val_parts, usage, neighbor_image, lr, train_kl, val_kl, train_latent, val_latent, train_extras=None, val_extras=None):
            extras = {}
            if train_extras:
                extras.update(train_extras)
            if val_extras:
                extras.update(val_extras)
            if train_kl is not None:
                extras["train/kl"] = train_kl
            if val_kl is not None:
                extras["val/kl"] = val_kl
            if train_latent is not None:
                extras.update(latent_scalars(train_latent, ""))
            if val_latent is not None:
                extras.update(latent_scalars(val_latent, "val/"))
            write_scalars(
                writer, step, train_loss, val_loss, train_parts, val_parts, usage,
                neighbor_image, lr, extras or None,
            )
            hist = None if val_latent is None else val_latent.get("hist")
            if hist is not None:
                writer.add_histogram("latent/mu", hist, step)
            if val_latent is not None:
                writer.add_text("latent/channels", latent_table(val_latent), step)
            if hist is not None or val_latent is not None:
                writer.flush()
            if val_loss is None:
                return
            message = f"step {step:06d}  val {val_loss:.6f}  lr {lr:.4g}"
            if val_parts is not None:
                message += "  " + part_text(*val_parts)
            if neighbor_image is not None:
                message += f"  neighbors {neighbor_image:.6f}"
            if usage is not None:
                message += f"  weight {usage['weight']:.5f}"
                if usage.get("grad") is not None:
                    message += f"  grad {usage['grad']:.4f}"
            if val_extras and "val/color" in val_extras:
                message += f"  color {val_extras['val/color']:.6f}"
            if val_kl is not None:
                message += f"  kl {val_kl:.6f}"
            if val_latent is not None:
                message += (
                    f"  mu {val_latent['mu_mean']:+.3f} ± {val_latent['mu_std']:.3f}"
                    f"  sigma {val_latent['sigma_mean']:.3f}"
                    f"  spread {val_latent['spread']:.3f}"
                )
            print(message, flush=True)
            if live is not None:
                live.status(message)

        saved_at = None

        def checkpoint():
            nonlocal saved_at
            schedule_state = None if run.plateau is None else run.plateau.state_dict(cfg.lr)
            saved_at = save_run(
                run.out_dir, run.model, architecture_dict(cfg), step, run.optimizer,
                cfg.loss_recipe(), objective.state_dict(),
                run.discriminator, run.disc_optimizer, schedule_state,
            )

        step = run.start_step
        steps_done = 0

        def updates_done():
            # cosine follows the run's step counter. const and plateau count
            # new updates, so a resume can ask for more.
            if cfg.steps is None:
                return False
            if cfg.schedule.kind == "cosine":
                return step >= cfg.steps
            return steps_done >= cfg.steps

        def validate():
            """Validation loss, images, and a checkpoint at the current step."""
            result = evaluate_split(objective, val_specs, run.out_dir / "val", step)
            old = None if run.plateau is None else run.plateau.lr
            decision = None if run.plateau is None else run.plateau.observe(result.loss)
            if decision == "drop":
                run.apply_rate(step)
            checkpoint()
            return result, decision, old

        def note_plateau(decision, old):
            if decision == "drop":
                print(f"step {step:06d}  lr {old:.4g} -> {run.plateau.lr:.4g}", flush=True)
                return False
            if decision == "stop":
                print(
                    f"step {step:06d}  stop  validation stood still at lr {run.plateau.lr:.4g}"
                    f"  min_lr {cfg.schedule.min_lr:.4g}  best val {run.plateau.best:.6f}",
                    flush=True,
                )
                return True
            return False

        try:
            # A new run validates before any update. A fine-tune's step 0 is
            # the generator that was loaded.
            stopped = False
            if run.train_state is None:
                result, decision, old_lr = validate()
                log(
                    step, None, result.loss, None, result.parts, None, result.neighbor,
                    learning_rate(cfg, step, run.plateau), None, result.kl, None, result.latent,
                    None, result.extras,
                )
                stopped = note_plateau(decision, old_lr)
            while not stopped and not updates_done():
                step += 1
                steps_done += 1
                lr = run.apply_rate(step)
                started = time.perf_counter()
                work = one_step(loader, objective, live is not None)
                train_kl = _scalar(getattr(run.model, "kl", None))
                train_latent = batch_latent()
                loss, weighted = objective.combine(work.raw)
                align_loss = getattr(run.model, "align_loss", None)
                if align_loss is not None:
                    loss = loss + align_loss
                kl_loss = getattr(run.model, "kl_loss", None)
                if kl_loss is not None:
                    loss = loss + kl_loss
                if work.loss is not None:
                    loss = loss + work.loss
                run.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                usage = run.model.neighbor_usage() if hasattr(run.model, "neighbor_usage") else None
                run.optimizer.step()
                if run.discriminator is not None:
                    for parameter in run.discriminator.parameters():
                        parameter.requires_grad_(True)
                objective.observe(
                    work.layer_raw, {name: value.detach() for name, value in work.raw.items()},
                )
                batch_parts = logged_parts(weighted, work)
                train_extras = step_extras(weighted, work)
                line = (
                    f"step {step:06d}  {time.perf_counter() - started:.1f}s  "
                    f"lr {lr:.4g}  loss {loss.item():.6f}  {part_text(*batch_parts)}  "
                    f"{work.width}x{work.height}"
                )
                if "train/color" in train_extras:
                    line += f"  color {train_extras['train/color']:.6f}"
                if "train/gan_weight" in train_extras:
                    line += f"  gan_weight {train_extras['train/gan_weight']:.4g}"
                if train_kl is not None:
                    line += f"  kl {train_kl:.6f}"
                print(line, flush=True)

                if live is not None:
                    live.status(line)
                    if work.images is not None:
                        live.show({
                            name: rgb_image(tensor) for name, tensor in work.images.items()
                        })

                finished = updates_done()
                result = None
                decision = None
                old_lr = None
                if step % cfg.val_every == 0 or finished:
                    result, decision, old_lr = validate()
                log(
                    step, loss.item(),
                    None if result is None else result.loss,
                    batch_parts,
                    None if result is None else result.parts,
                    usage,
                    None if result is None else result.neighbor,
                    lr, train_kl,
                    None if result is None else result.kl,
                    train_latent,
                    None if result is None else result.latent,
                    train_extras,
                    None if result is None else result.extras,
                )
                if note_plateau(decision, old_lr):
                    break
        except KeyboardInterrupt:
            checkpoint()
            print(f"interrupted, saved {saved_at / 'model.pt'}", flush=True)
            return
    finally:
        if writer is not None:
            writer.close()
        if loader is not None:
            loader.close()
        if live is not None:
            live.close()
    shown = saved_at / "model.pt" if saved_at is not None else run.out_dir / "model.pt"
    print(f"done  {shown}", flush=True)


def sr_hooks(run):
    """Paired low and master batches, and the discriminator step."""
    cfg = run.cfg
    films = run.films
    rng = run.rng
    radius = cfg.in_frames // 2
    gan = run.gan

    def make_val_specs():
        specs = []
        while len(specs) < cfg.val_count:
            need = min(cfg.batch, cfg.val_count - len(specs))
            specs.extend(make_batch_specs(rng, films, "val", need))
        return specs

    def make_loader():
        def produce():
            specs = make_batch_specs(rng, films, "train", cfg.batch)
            film_index, variant_index, _index = specs[0]
            variant = films[film_index].variants[variant_index]
            low, high = batch_from_specs(films, specs, radius)
            box = None
            if gan is not None:
                box = sample_box(high, gan.crop, rng)
            return PreparedBatch(low, high, variant.low.width, variant.low.height, box)

        return BatchLoader(produce)

    def measure(objective, specs):
        return measure_raw_means(
            run.model, films, specs, radius, cfg.batch, run.device, cfg, objective,
            run.discriminator,
        )

    def one_step(loader, objective, show):
        prepared = loader.get()
        low = prepared.low.to(run.device)
        high = prepared.high.to(run.device)
        pred = run.model(low)
        box = prepared.box
        d_value = None
        if gan is not None:
            d_loss = 0.5 * (
                gan_bce(run.discriminator(take(high, box)), True)
                + gan_bce(run.discriminator(take(pred.detach(), box)), False)
            )
            run.disc_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            run.disc_optimizer.step()
            d_value = d_loss.item()
            for parameter in run.discriminator.parameters():
                parameter.requires_grad_(False)
        raw, layer_raw = raw_terms(objective, cfg, pred, high, run.discriminator, box)
        images = None
        if show:
            center = radius * 3
            images = {
                "low": low[0, center:center + 3],
                "upscale": pred[0],
                "master": high[0],
            }
        return StepOut(raw, layer_raw, d_value, images, prepared.width, prepared.height)

    def evaluate_split(objective, specs, image_dir, step):
        loss, pixel, vgg, gan_value, d_value, neighbor = evaluate(
            run.model, films, specs, radius, cfg.batch, run.device, cfg, objective,
            run.discriminator, image_dir=image_dir, step=step,
        )
        return EvalResult(loss, (pixel, vgg, gan_value, d_value), neighbor, None)

    return make_val_specs, make_loader, measure, one_step, evaluate_split


def adaptive_gan_weight(objective_loss, gan_loss, layer):
    """Latent diffusion's ratio of gradient norms at `layer`, kept within 0 and 1e4."""
    objective_grad = torch.autograd.grad(objective_loss, layer, retain_graph=True)[0]
    gan_grad = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    weight = torch.linalg.vector_norm(objective_grad) / (torch.linalg.vector_norm(gan_grad) + 1e-4)
    return weight.clamp(0.0, 1e4).detach()


def vae_hooks(run):
    """Master frames in, the same frames as the target.

    A vae2 gan term trains its discriminator on one window per step. The
    generator term starts at the term's `start` step.
    """
    cfg = run.cfg
    films = run.films
    rng = run.rng
    gan = run.gan
    # Windows for the discriminator come from their own draw, on this thread.
    gan_rng = random.Random(cfg.seed + 1)
    counter = {"step": run.start_step}

    def make_val_specs():
        specs = []
        while len(specs) < cfg.val_count:
            need = min(cfg.batch, cfg.val_count - len(specs))
            specs.extend(make_vae_specs(rng, films, "val", need, cfg.crop))
        return specs

    def make_loader():
        def produce():
            specs = make_vae_specs(rng, films, "train", cfg.batch, cfg.crop)
            return vae_batch(films, specs, cfg.crop, rng)

        return BatchLoader(produce)

    def measure(objective, specs):
        return measure_vae(
            run.model, films, specs, cfg.crop, cfg.batch, run.device, cfg, objective,
        )

    def adversarial(objective, raw, pred, image):
        """Discriminator update, then the weighted generator term once it has started."""
        box = sample_box(image, gan.crop, gan_rng)
        discriminator = run.discriminator
        d_loss = 0.5 * (
            gan_bce(discriminator(take(image, box)), True)
            + gan_bce(discriminator(take(pred.detach(), box)), False)
        )
        run.disc_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        run.disc_optimizer.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        start = 0 if gan.start is None else gan.start
        if counter["step"] < start:
            return d_loss.item(), None, None, None
        gan_loss = gan_bce(discriminator(take(pred, box)), True)
        objective_loss, _weighted = objective.combine(raw)
        weight = gan.weight * adaptive_gan_weight(
            objective_loss, gan_loss, run.model.last_layer(),
        )
        added = weight * gan_loss
        return d_loss.item(), added, added.item(), {
            "train/gan_weight": float(weight),
            "train/gan_raw": gan_loss.item(),
        }

    def one_step(loader, objective, show):
        counter["step"] += 1
        prepared = loader.get()
        image = prepared.image.to(run.device)
        pred = run.model(image)
        raw, layer_raw = raw_terms(objective, cfg, pred, image, None, None)
        d_value = None
        added = None
        gan_value = None
        extras = None
        if gan is not None:
            d_value, added, gan_value, extras = adversarial(objective, raw, pred, image)
        images = None
        if show:
            images = {
                "sample": prior_image(run.model, image),
                "recon": pred[0],
                "master": image[0],
            }
        return StepOut(
            raw, layer_raw, d_value, images, prepared.width, prepared.height,
            added, gan_value, extras,
        )

    def evaluate_split(objective, specs, image_dir, step):
        result = evaluate_vae(
            run.model, films, specs, cfg.crop, cfg.batch, run.device, cfg, objective,
            image_dir=image_dir, step=step,
        )
        if hasattr(run.model, "set_latent_stats") and result.latent is not None:
            run.model.set_latent_stats(result.latent["mu_mean"], result.latent["spread"])
            extras = dict(result.extras or {})
            extras["val/latent/shift"] = float(run.model.latent_shift)
            extras["val/latent/scale"] = float(run.model.latent_scale)
            result.extras = extras
        return result

    return make_val_specs, make_loader, measure, one_step, evaluate_split


def load_sr(cfg):
    root = data_root()
    found = load_films(root, cfg.holdout, cfg.variants)
    for problem in found.problems:
        print(problem, flush=True)
    require_films(found, root, cfg.variants)
    print(f"pairs {found.kept} of {found.total}", flush=True)
    return found.films


def load_vae(cfg):
    root = data_root()
    found = load_masters(root, cfg.holdout, cfg.crop)
    for problem in found.problems:
        print(problem, flush=True)
    require_masters(found, root)
    print(f"masters {found.kept} of {found.total}", flush=True)
    return found.films


def train(cfg, preview=True, live_bind="127.0.0.1:8765", resume=None, finetune=None):
    if cfg.arch in VAE_ARCHS:
        def load():
            return load_vae(cfg)

        hooks = vae_hooks
    else:
        def load():
            return load_sr(cfg)

        hooks = sr_hooks
    run = open_run(cfg, resume, finetune, load)
    if run is None:
        return
    drive(run, preview, live_bind, *hooks(run))


def main():
    parser = argparse.ArgumentParser(description="train on random paired video frames")
    parser.add_argument("config", type=Path, help="training recipe")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "continue this run from its newest checkpoint "
            "(directory, latest, a checkpoint, model.pt, or train.pt). "
            "An older checkpoint starts a new run. "
            "Same network and loss. The schedule comes from the file"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "fine-tune this generator (a checkpoint, model.pt, or a run directory). "
            "New run and a new optimizer. The recipe may omit its model section"
        ),
    )
    add_live_args(parser)
    args = parser.parse_args()
    if args.resume is not None and args.model is not None:
        raise SystemExit("pass either --resume or --model")
    finetune = None
    architecture = None
    architecture_from = None
    if args.model is not None:
        finetune = load_generator(args.model)
        architecture = finetune.architecture
        architecture_from = finetune.path
    train(
        load_config(args.config, architecture, architecture_from),
        preview=not args.no_preview,
        live_bind=args.live_bind,
        resume=args.resume,
        finetune=finetune,
    )


if __name__ == "__main__":
    main()
