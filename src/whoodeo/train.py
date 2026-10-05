"""Train on random full frames from paired videos.

The recipe is a YAML file. Masters live in the data directory's orig/ and
degraded variants in its low/. A step picks one film, then one variant, then
as many center times as the batch size. Two time spans per variant are held
out for validation.

Training batches are decoded on one side thread into a queue of three CPU
batches. The step copies a batch onto the device. Validation keeps its own
clips and reads them on the main thread.

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
checkpoint written. The run directory also gets a TensorBoard event file
for the training and validation losses. whoodeo-board serves every run.

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
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import torch
from torch.utils.tensorboard import SummaryWriter
from whoodeo.catalog import data_root, runs_root
from whoodeo.checkpoint import continue_run, load_generator, load_resume, run_directory, save_run
from whoodeo.config import architecture_dict, assert_resume_matches, load_config, shape_text
from whoodeo.live import add_live_args, open_live
from whoodeo.models import build_model
from whoodeo.models.discriminator import UNetDiscriminatorSN, gan_bce
from whoodeo.objective import Objective, pixel_loss
from whoodeo.video import frame_tensor, png_bytes, rgb_image

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
SCALE = 2


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def holdout_ranges(duration, fraction):
    span = duration * fraction / 2
    ranges = []
    for start in (0.30 * duration, 0.70 * duration):
        end = min(duration, start + span)
        if end > start:
            ranges.append((start, end))
    return ranges


def in_ranges(time_sec, ranges):
    return any(start <= time_sec < end for start, end in ranges)


class Clip:
    def __init__(self, path):
        self.path = Path(path)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate)
        if not self.container.duration:
            raise RuntimeError(f"no duration: {self.path}")
        self.duration = self.container.duration / av.time_base
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height

    def close(self):
        self.container.close()

    def frames_around(self, time_sec, radius):
        pad = (radius + 1) / self.fps
        start = max(0.0, time_sec - pad)
        self.container.seek(int(start * av.time_base), backward=True, any_frame=False)
        frames = []
        times = []
        for frame in self.container.decode(self.stream):
            if frame.time is None or frame.time + 1e-3 < start:
                continue
            frames.append(frame_tensor(frame))
            times.append(frame.time)
            if frame.time >= time_sec + pad and len(frames) >= radius * 2 + 1:
                break
        if not frames:
            raise RuntimeError(f"no frames near {time_sec:.3f}s in {self.path.name}")
        center = min(range(len(times)), key=lambda i: abs(times[i] - time_sec))
        picked = []
        last = len(frames) - 1
        for delta in range(-radius, radius + 1):
            picked.append(frames[min(max(center + delta, 0), last)])
        return picked


@dataclass
class Variant:
    name: str
    low: Clip
    duration: float
    holdouts: list = field(default_factory=list)


@dataclass
class Film:
    name: str
    orig: Clip
    variants: list = field(default_factory=list)


def find_films(orig_dir, low_dir):
    if not orig_dir.is_dir():
        raise SystemExit(f"no orig folder: {orig_dir}")
    if not low_dir.is_dir():
        raise SystemExit(f"no low folder: {low_dir}")
    found = {}
    variant_dirs = sorted(
        path for path in low_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    for variant_dir in variant_dirs:
        for low_path in sorted(variant_dir.iterdir()):
            if low_path.suffix.lower() not in VIDEO_EXTS or not low_path.is_file():
                continue
            orig_path = orig_dir / low_path.name
            if not orig_path.is_file():
                print(
                    f"skip {variant_dir.name}/{low_path.name}: no match in {orig_dir}",
                    flush=True,
                )
                continue
            found.setdefault(low_path.name, (orig_path, []))
            found[low_path.name][1].append((variant_dir.name, low_path))
    if not found:
        raise SystemExit(f"no video pairs in {orig_dir} and {low_dir}")
    films = []
    for name in sorted(found):
        orig_path, variants = found[name]
        variants.sort(key=lambda item: item[0])
        films.append((name, orig_path, variants))
    return films


def open_films(orig_dir, low_dir, holdout):
    opened = []
    for name, orig_path, variant_paths in find_films(orig_dir, low_dir):
        orig = Clip(orig_path)
        variants = []
        for variant_name, low_path in variant_paths:
            low = Clip(low_path)
            if (orig.height, orig.width) != (low.height * SCALE, low.width * SCALE):
                raise SystemExit(
                    f"{variant_name}/{low_path.name} is {low.width}x{low.height}, "
                    f"orig is {orig.width}x{orig.height}, expected exactly {SCALE}x"
                )
            duration = min(orig.duration, low.duration)
            variants.append(Variant(
                name=variant_name,
                low=low,
                duration=duration,
                holdouts=holdout_ranges(duration, holdout),
            ))
        opened.append(Film(name=name, orig=orig, variants=variants))
    return opened


def sample_time(rng, film, variant, split):
    for _ in range(10000):
        time_sec = rng.uniform(0.0, variant.duration)
        inside = in_ranges(time_sec, variant.holdouts)
        if split == "val" and inside:
            return time_sec
        if split == "train" and not inside:
            return time_sec
    raise RuntimeError(f"could not sample a {split} time in {film.name} {variant.name}")


def load_sample(film, variant, time_sec, radius):
    low_frames = variant.low.frames_around(time_sec, radius)
    hr = film.orig.frames_around(time_sec, radius)[radius]
    stacked = torch.cat(low_frames, dim=0)
    return stacked, hr


def make_batch_specs(rng, films, split, count):
    """count times from one film and one variant, so the batch shares a size."""
    film_index = rng.randrange(len(films))
    film = films[film_index]
    variant_index = rng.randrange(len(film.variants))
    variant = film.variants[variant_index]
    return [
        (film_index, variant_index, sample_time(rng, film, variant, split))
        for _ in range(count)
    ]


def batch_from_specs(films, specs, radius):
    lows = []
    highs = []
    for film_index, variant_index, time_sec in specs:
        film = films[film_index]
        low, high = load_sample(film, film.variants[variant_index], time_sec, radius)
        lows.append(low)
        highs.append(high)
    return torch.stack(lows), torch.stack(highs)


def batch_on_device(films, specs, radius, device):
    low, high = batch_from_specs(films, specs, radius)
    return low.to(device), high.to(device)


def close_films(films):
    for film in films:
        film.orig.close()
        for variant in film.variants:
            variant.low.close()


_QUEUE_DEPTH = 3
_QUEUE_POLL = 0.2


@dataclass
class PreparedBatch:
    low: torch.Tensor
    high: torch.Tensor
    film_name: str
    variant_name: str
    width: int
    height: int
    box: tuple | None


class BatchLoader:
    """One thread of CPU training batches.

    The thread opens its own clips and never moves tensors onto a device.
    ``put`` times out so a full queue cannot block shutdown. ``get`` raises
    an error from the thread instead of waiting on a dead loader.
    """

    def __init__(self, rng, orig_dir, low_dir, holdout, batch, radius, crop):
        self._rng = rng
        self._orig_dir = orig_dir
        self._low_dir = low_dir
        self._holdout = holdout
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
        films = None
        try:
            films = open_films(self._orig_dir, self._low_dir, self._holdout)
            while not self._stop.is_set():
                specs = make_batch_specs(self._rng, films, "train", self._batch)
                film_index, variant_index, _time_sec = specs[0]
                film = films[film_index]
                variant = film.variants[variant_index]
                low, high = batch_from_specs(films, specs, self._radius)
                box = None
                if self._crop is not None:
                    box = sample_box(high, self._crop, self._rng)
                prepared = PreparedBatch(
                    low=low,
                    high=high,
                    film_name=film.name,
                    variant_name=variant.name,
                    width=variant.low.width,
                    height=variant.low.height,
                    box=box,
                )
                if not self._put(prepared):
                    return
        except BaseException as exc:
            self._put(exc)
        finally:
            if films is not None:
                close_films(films)


def write_scalars(writer, step, train_loss, val_loss, train_parts, val_parts):
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


def evaluate(model, films, specs, radius, batch_size, device, cfg, objective, discriminator, image_dir=None, step=None):
    """Mean of the weighted objective. The GAN part uses the center crop.

    With image_dir set, each sample is written under image_dir/<index>/.
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
    saw_d = False
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
    return loss_acc / count, pixel_mean, vgg_mean, gan_mean, d_mean


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

    films = open_films(root / "orig", root / "low", cfg.holdout)
    print(f"device {device}", flush=True)
    print(f"config {cfg.source}", flush=True)
    print(f"network {shape_text(*cfg.arch_key())}", flush=True)
    if generator is not None and train_state is None:
        shown = "unknown" if generator.step is None else str(generator.step)
        print(f"fine-tune {generator.path}  loaded step {shown}", flush=True)
    for film in films:
        for variant in film.variants:
            spans = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in variant.holdouts)
            print(
                f"pair {film.name}  {variant.name}  "
                f"{variant.low.width}x{variant.low.height}  "
                f"{variant.duration:.1f}s  holdout {spans}",
                flush=True,
            )

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
            root / "orig",
            root / "low",
            cfg.holdout,
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

        def log(step, train_loss, val_loss, train_parts, val_parts):
            write_scalars(writer, step, train_loss, val_loss, train_parts, val_parts)
            if val_loss is None:
                return
            message = f"step {step:06d}  val {val_loss:.6f}"
            if val_parts is not None:
                message += "  " + part_text(*val_parts)
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
            loss, pixel, vgg, gan_value, d_value = evaluate(
                model, films, val_specs, radius, cfg.batch, device,
                cfg, objective, discriminator,
                image_dir=out_dir / "val", step=step,
            )
            checkpoint()
            return loss, logged_parts(pixel, vgg, gan_value, d_value)

        try:
            # A new run validates before any update. A fine-tune's step 0 is
            # the generator that was loaded.
            if train_state is None:
                val_loss, val_parts = validate()
                log(step, None, val_loss, None, val_parts)
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
                    f"{prepared.width}x{prepared.height}  "
                    f"{prepared.variant_name}  {prepared.film_name}"
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
                if step % cfg.val_every == 0 or finished:
                    val_loss, val_parts = validate()
                log(step, loss.item(), val_loss, batch_parts, val_parts)
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
        close_films(films)
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
