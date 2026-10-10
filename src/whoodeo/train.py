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

Training batches are decoded on one side thread into a queue of three CPU
batches. A frame read opens the video and closes it before the next read.
An unreadable training clip is logged and the loader draws a new batch.
The step copies a batch onto the device. Validation reads on the main thread.

The loss is the weighted sum of the terms in the file. The first term anchors
the magnitude. balance start freezes the scales on the validation frames
before the first step. balance running keeps the weights as shares.

A fresh run validates once at step 0, before any update. Where the last
layer starts at zero, that pass is the bilinear upscale. Modified leaves
that layer at the default initialization, so step 0 already includes a
correction. A fine-tune's step 0 is the generator that was loaded.

A gan term adds a U-Net discriminator. Each checkpoint is a directory
checkpoints/<step>/ holding two files. model.pt is the generator and the
architecture that builds it. whoodeo-apply and a fine-tune read only that
file. train.pt is the step, the optimizers, the discriminator, the loss
recipe, and the loss scales. latest in the run points at the newest of
those directories, and latest in the runs directory follows the newest
checkpoint written. The run directory also gets a TensorBoard event file for the losses.
The vote model adds neighbor/weight, neighbor/grad, and neighbor/vote
on each step, and neighbor/image when validation runs. weight and grad
cover every kernel in the shared vote.
whoodeo-board serves every run.

--resume continues a run from its newest checkpoint: the same architecture,
the same loss, the Adam state, and the loss scales. Learning rates come
from the file. Resuming an older checkpoint starts a new run, so the later
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
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import torch
from torch.utils.tensorboard import SummaryWriter
from whoodeo.catalog import data_root, runs_root
from whoodeo.library import load_catalog, master_filename, titles_of
from whoodeo.checkpoint import continue_run, load_generator, load_resume, run_directory, save_run
from whoodeo.config import architecture_dict, assert_resume_matches, load_config, shape_text
from whoodeo.live import add_live_args, open_live
from whoodeo.models import build_model
from whoodeo.models.discriminator import UNetDiscriminatorSN, gan_bce
from whoodeo.objective import Objective, pixel_loss
from whoodeo.video import frame_tensor, png_bytes, rgb_image

SCALE = 2


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def holdout_ranges(count, fraction):
    span = count * fraction / 2
    ranges = []
    for start in (0.30 * count, 0.70 * count):
        end = min(count, start + span)
        if end > start:
            ranges.append((start, end))
    return ranges


def in_ranges(index, ranges):
    return any(start <= index < end for start, end in ranges)


def frame_index(stamp, rate):
    """Display index of a presentation time at this frame rate."""
    return round(stamp * float(rate))


class Clip:
    """A video named in the catalog. Nothing stays open between reads."""

    def __init__(self, path, width, height, frames):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.frames = frames

    def frames_around(self, index, radius):
        """Frames centered on this display index.

        The seek timestamp only reaches a keyframe. The index is the frame
        rate times the presentation time, and each kept step has to be one
        frame. A gap, or a landing past the requested index, stops the read.
        """
        if index < 0 or index >= self.frames:
            raise RuntimeError(
                f"{self.path.name}: frame {index} outside 0..{self.frames - 1}"
            )
        container = av.open(str(self.path))
        try:
            stream = container.streams.video[0]
            width = stream.codec_context.width
            height = stream.codec_context.height
            if (width, height) != (self.width, self.height):
                raise RuntimeError(
                    f"{self.path.name} is {width}x{height}, "
                    f"catalog says {self.width}x{self.height}"
                )
            rate = stream.average_rate
            if not rate:
                raise RuntimeError(f"no frame rate: {self.path}")
            first = max(0, index - radius)
            last = min(self.frames - 1, index + radius)
            hint = max(0.0, (first - 1) / float(rate))
            container.seek(int(hint * av.time_base), backward=True, any_frame=False)
            found = {}
            previous = None
            for frame in container.decode(stream):
                if frame.time is None:
                    raise RuntimeError(f"no timestamp: {self.path.name}")
                current = frame_index(frame.time, rate)
                if previous is None:
                    if current > first:
                        raise RuntimeError(
                            f"{self.path.name}: seek landed on frame {current}, "
                            f"needed {first}"
                        )
                elif current != previous + 1:
                    raise RuntimeError(
                        f"{self.path.name}: frame {previous} is followed by {current}"
                    )
                previous = current
                if current < first:
                    continue
                found[current] = frame_tensor(frame)
                if current >= last:
                    break
            if first not in found or last not in found:
                raise RuntimeError(f"no frame {index} in {self.path.name}")
            return [
                found[min(max(index + delta, 0), self.frames - 1)]
                for delta in range(-radius, radius + 1)
            ]
        finally:
            container.close()


@dataclass
class Variant:
    name: str
    low: Clip


@dataclass
class Film:
    name: str
    orig: Clip
    frames: int
    holdouts: list
    variants: list = field(default_factory=list)


@dataclass
class FoundFilms:
    """Pairs named in the catalog. A master with no low is left out."""

    films: list
    total: int
    kept: int
    problems: list


def _number(record, key, label):
    value = record.get(key) if isinstance(record, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label}: missing {key}")
    return value


def _count(record, key, label):
    value = record.get(key) if isinstance(record, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"{label}: missing {key}")
    return value


def load_films(root, holdout, variants=None):
    """Pairs from catalog.json. No video is opened."""
    orig_dir = root / "orig"
    low_dir = root / "low"
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    if not low_dir.is_dir():
        raise SystemExit(f"no low folder: {low_dir}")
    catalog = load_catalog(root)
    titles = titles_of(catalog, root / "catalog.json")
    pattern = re.compile(variants) if variants is not None else None
    total = 0
    kept = 0
    problems = []
    ordered = []
    for title in titles:
        entry = titles[title]
        masters = entry.get("masters") if isinstance(entry, dict) else None
        if not isinstance(masters, dict):
            problems.append(f"skip {title}: no masters")
            continue
        frames = _count(entry, "frames", title)
        holdouts = holdout_ranges(frames, holdout)
        for rung, master in masters.items():
            name = master_filename(title, rung)
            orig_path = orig_dir / name
            if not isinstance(master, dict):
                problems.append(f"skip {name}: bad catalog entry")
                continue
            if not orig_path.is_file():
                problems.append(f"skip {name}: no master file")
                continue
            orig_w = int(_number(master, "width", name))
            orig_h = int(_number(master, "height", name))
            lows = master.get("low")
            if not isinstance(lows, list):
                problems.append(f"skip {name}: no lows")
                continue
            names = []
            for variant_name in lows:
                if (
                    not isinstance(variant_name, str)
                    or not variant_name
                    or "/" in variant_name
                    or variant_name in {".", ".."}
                ):
                    problems.append(f"skip {name}: bad variant name")
                    continue
                if variant_name not in names:
                    names.append(variant_name)
            low_w = orig_w // SCALE
            low_h = orig_h // SCALE
            variants_here = []
            for variant_name in sorted(names):
                rel = f"{variant_name}/{name}"
                low_path = low_dir / variant_name / name
                if not low_path.is_file():
                    problems.append(f"skip {rel}: file missing")
                    continue
                total += 1
                if pattern is not None and pattern.search(rel) is None:
                    continue
                if orig_w % SCALE or orig_h % SCALE:
                    raise SystemExit(
                        f"{name} is {orig_w}x{orig_h}, "
                        f"both sides must be divisible by {SCALE}"
                    )
                kept += 1
                variants_here.append(Variant(
                    name=variant_name,
                    low=Clip(low_path, low_w, low_h, frames),
                ))
            if not variants_here:
                continue
            film = Film(
                name=f"{title}.{rung}",
                orig=Clip(orig_path, orig_w, orig_h, frames),
                frames=frames,
                holdouts=holdouts,
                variants=variants_here,
            )
            ordered.append((f"{title}.mkv", orig_w * orig_h, film))
    ordered.sort(key=lambda item: (item[0], -item[1]))
    films = [item[2] for item in ordered]
    return FoundFilms(films, total, kept, problems)


def require_films(found, root, variants):
    if found.kept:
        return
    if variants:
        raise SystemExit(
            f"no pairs match {variants!r}: {found.kept} of {found.total}"
        )
    raise SystemExit(f"no video pairs in {root / 'orig'} and {root / 'low'}")


def sample_index(rng, film, split):
    for _ in range(10000):
        index = rng.randrange(film.frames)
        inside = in_ranges(index, film.holdouts)
        if split == "val" and inside:
            return index
        if split == "train" and not inside:
            return index
    raise RuntimeError(f"could not sample a {split} frame in {film.name}")


class UnreadableSample(RuntimeError):
    """A training clip could not be read. The loader draws another batch."""


def _unreadable(where, index, exc):
    return UnreadableSample(f"{where} frame {index}: {exc}")


def load_sample(film, variant, index, radius):
    try:
        low_frames = variant.low.frames_around(index, radius)
    except (RuntimeError, OSError, av.error.FFmpegError) as exc:
        where = f"low {variant.name}/{variant.low.path.name}"
        raise _unreadable(where, index, exc) from exc
    try:
        hr = film.orig.frames_around(index, radius)[radius]
    except (RuntimeError, OSError, av.error.FFmpegError) as exc:
        raise _unreadable(f"master {film.orig.path.name}", index, exc) from exc
    stacked = torch.cat(low_frames, dim=0)
    return stacked, hr


def draw_indexes(rng, available, count):
    """count indexes into a collection of `available` items.

    The draw is without replacement when there are at least count items, and
    with replacement when there are fewer.
    """
    if count > available:
        return rng.choices(range(available), k=count)
    return rng.sample(range(available), count)


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

    def __init__(self, rng, films, batch, radius, crop):
        self._rng = rng
        self._films = films
        self._batch = batch
        self._radius = radius
        self._crop = crop
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
        films = self._films
        failed = 0
        try:
            while not self._stop.is_set():
                specs = make_batch_specs(self._rng, films, "train", self._batch)
                film_index, variant_index, _index = specs[0]
                variant = films[film_index].variants[variant_index]
                try:
                    low, high = batch_from_specs(films, specs, self._radius)
                except UnreadableSample as exc:
                    failed += 1
                    print(f"skip batch: {exc}", file=sys.stderr, flush=True)
                    if failed >= _UNREADABLE_LIMIT:
                        raise RuntimeError(
                            f"{failed} batches in a row could not be read"
                        ) from exc
                    continue
                failed = 0
                box = None
                if self._crop is not None:
                    box = sample_box(high, self._crop, self._rng)
                prepared = PreparedBatch(
                    low=low,
                    high=high,
                    width=variant.low.width,
                    height=variant.low.height,
                    box=box,
                )
                if not self._put(prepared):
                    return
        except BaseException as exc:
            self._put(exc)


def write_scalars(writer, step, train_loss, val_loss, train_parts, val_parts, usage, neighbor_image):
    """One TensorBoard point per value that this step actually has."""
    def emit(tag, value):
        if value is not None:
            writer.add_scalar(tag, value, step)

    emit("loss/train", train_loss)
    emit("loss/val", val_loss)
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
        emit("neighbor/vote", usage.get("vote"))
        emit("neighbor/grad", usage.get("grad"))
    emit("neighbor/image", neighbor_image)
    writer.flush()


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


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
        raw = objective.term_ref[term.name]
        scale = objective.scale[term.name]
        weighted = scale * raw
        if term.name == "pixel":
            detail = term.kind
        elif term.name == "vgg":
            detail = "+".join(term.layers)
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
    if cfg.term("gan") is not None:
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


def save_val_images(directory, step, index, low, pred, high, width):
    """Write one validation sample. low and master stay; the upscale is per step."""
    folder = directory / f"{index:0{width}d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("low", low), ("master", high)):
        path = folder / f"{name}.png"
        if not path.is_file():
            path.write_bytes(png_bytes(rgb_image(tensor)))
    (folder / f"upscale-{step:06d}.png").write_bytes(png_bytes(rgb_image(pred)))


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


def train(cfg, preview=True, live_bind="127.0.0.1:8765", resume=None, finetune=None):
    if resume is not None and finetune is not None:
        raise SystemExit("pass either --resume or --model")
    generator = finetune
    train_state = None
    if resume is not None:
        generator, train_state = load_resume(resume)
        assert_resume_matches(cfg, generator.architecture, train_state.loss, generator.path)

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
    radius = cfg.in_frames // 2
    device = get_device()
    root = data_root()

    found = load_films(root, cfg.holdout, cfg.variants)
    print(f"device {device}", flush=True)
    print(f"config {cfg.source}", flush=True)
    print(f"network {shape_text(*cfg.arch_key())}", flush=True)
    if generator is not None and train_state is None:
        shown = "unknown" if generator.step is None else str(generator.step)
        print(f"fine-tune {generator.path}  loaded step {shown}", flush=True)
    for problem in found.problems:
        print(problem, flush=True)
    require_films(found, root, cfg.variants)
    print(f"pairs {found.kept} of {found.total}", flush=True)
    films = found.films

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = out_dir / "config.yaml"
    if cfg.source.resolve() != copied.resolve():
        shutil.copyfile(cfg.source, copied)
    same_run = (
        generator is not None
        and out_dir.resolve() == run_directory(generator.path).resolve()
    )
    limit = "until Ctrl-C" if cfg.steps is None else str(cfg.steps)
    if train_state is not None and not same_run:
        print(f"new run from {generator.path}", flush=True)
    print(f"run {out_dir}  steps {limit}", flush=True)

    model, label = build_model(
        cfg.arch, cfg.blocks, cfg.channels, cfg.in_frames,
        radius=cfg.radius, stem=cfg.stem, sharpness=cfg.sharpness, reject=cfg.reject,
        levels=cfg.levels, bilinear=cfg.bilinear, codes=cfg.codes, patch=cfg.patch,
        res_scale=cfg.res_scale, vote=cfg.vote,
    )
    model = model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(label, flush=True)
    if generator is not None:
        model.load_state_dict(generator.state_dict)
    if train_state is not None:
        optimizer.load_state_dict(train_state.optimizer)
        move_optimizer(optimizer, device)
        apply_learning_rate(optimizer, cfg.lr)
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
            apply_learning_rate(disc_optimizer, gan.disc_lr)
            print("loaded discriminator", flush=True)
        print(
            f"discriminator unet-sn  {count} parameters  crop {gan.crop}  "
            f"lr {gan.disc_lr:g}",
            flush=True,
        )

    objective = Objective(cfg, device)
    val_specs = []
    while len(val_specs) < cfg.val_count:
        need = min(cfg.batch, cfg.val_count - len(val_specs))
        val_specs.extend(make_batch_specs(rng, films, "val", need))
    loader = None
    live = None
    writer = None
    try:
        loader = BatchLoader(
            rng,
            films,
            cfg.batch,
            radius,
            None if gan is None else gan.crop,
        )
        loader.start()
        if train_state is not None:
            objective.load_state_dict(train_state.objective)
            print("loss scales from checkpoint", flush=True)
        else:
            layer_means, term_means = measure_raw_means(
                model, films, val_specs, radius, cfg.batch, device, cfg, objective, discriminator,
            )
            objective.calibrate(layer_means, term_means)
        describe_loss(cfg, objective)

        live = open_live(live_bind, preview)
        writer = SummaryWriter(log_dir=out_dir)
        has_pixel = cfg.term("pixel") is not None
        has_vgg = cfg.term("vgg") is not None

        def logged_parts(pixel, vgg, gan_value, d_value):
            return (
                pixel if has_pixel else None,
                vgg if has_vgg else None,
                gan_value if gan is not None else None,
                d_value if gan is not None else None,
            )

        def log(step, train_loss, val_loss, train_parts, val_parts, usage, neighbor_image):
            write_scalars(
                writer, step, train_loss, val_loss, train_parts, val_parts, usage, neighbor_image,
            )
            if val_loss is None:
                return
            message = f"step {step:06d}  val {val_loss:.6f}"
            if val_parts is not None:
                message += "  " + part_text(*val_parts)
            if neighbor_image is not None:
                message += f"  neighbors {neighbor_image:.6f}"
            if usage is not None:
                message += f"  vote {usage['vote']:.4f}  weight {usage['weight']:.5f}"
                if usage.get("grad") is not None:
                    message += f"  grad {usage['grad']:.4f}"
            print(message, flush=True)
            if live is not None:
                live.status(message)

        saved_at = None

        def checkpoint():
            nonlocal saved_at
            saved_at = save_run(
                out_dir, model, architecture_dict(cfg), step, optimizer,
                cfg.loss_recipe(), objective.state_dict(),
                discriminator, disc_optimizer,
            )

        start_step = train_state.step if train_state is not None else 0
        step = start_step
        steps_done = 0

        def validate():
            """Validation loss, images, and a checkpoint at the current step."""
            loss, pixel, vgg, gan_value, d_value, neighbor_image = evaluate(
                model, films, val_specs, radius, cfg.batch, device,
                cfg, objective, discriminator,
                image_dir=out_dir / "val", step=step,
            )
            checkpoint()
            return loss, logged_parts(pixel, vgg, gan_value, d_value), neighbor_image

        try:
            # A new run validates before any update. A fine-tune's step 0 is
            # the generator that was loaded.
            if train_state is None:
                val_loss, val_parts, neighbor_image = validate()
                log(step, None, val_loss, None, val_parts, None, neighbor_image)
            while cfg.steps is None or steps_done < cfg.steps:
                step += 1
                steps_done += 1
                started = time.perf_counter()
                prepared = loader.get()
                low = prepared.low.to(device)
                high = prepared.high.to(device)
                pred = model(low)
                box = prepared.box
                d_value = None
                if gan is not None:
                    d_loss = 0.5 * (
                        gan_bce(discriminator(take(high, box)), True)
                        + gan_bce(discriminator(take(pred.detach(), box)), False)
                    )
                    disc_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    disc_optimizer.step()
                    d_value = d_loss.item()
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                raw, layer_raw = raw_terms(objective, cfg, pred, high, discriminator, box)
                loss, weighted = objective.combine(raw)
                align_loss = getattr(model, "align_loss", None)
                if align_loss is not None:
                    loss = loss + align_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                usage = model.neighbor_usage() if hasattr(model, "neighbor_usage") else None
                optimizer.step()
                if discriminator is not None:
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)
                objective.observe(layer_raw, {name: value.detach() for name, value in raw.items()})
                batch_parts = logged_parts(
                    weighted["pixel"].item() if has_pixel else None,
                    weighted["vgg"].item() if has_vgg else None,
                    weighted["gan"].item() if gan is not None else None,
                    d_value,
                )
                line = (
                    f"step {step:06d}  {time.perf_counter() - started:.1f}s  "
                    f"loss {loss.item():.6f}  {part_text(*batch_parts)}  "
                    f"{prepared.width}x{prepared.height}"
                )
                print(line, flush=True)

                if live is not None:
                    center = radius * 3
                    live.status(line)
                    live.show({
                        "low": rgb_image(low[0, center:center + 3]),
                        "upscale": rgb_image(pred[0]),
                        "master": rgb_image(high[0]),
                    })

                finished = cfg.steps is not None and steps_done == cfg.steps
                val_loss = None
                val_parts = None
                neighbor_image = None
                if step % cfg.val_every == 0 or finished:
                    val_loss, val_parts, neighbor_image = validate()
                log(step, loss.item(), val_loss, batch_parts, val_parts, usage, neighbor_image)
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
    shown = saved_at / "model.pt" if saved_at is not None else out_dir / "model.pt"
    print(f"done  {shown}", flush=True)


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
            "Same network and loss; learning rates come from the file"
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
