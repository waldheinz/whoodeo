"""Train on random full frames from paired videos.

The recipe is a YAML file. Masters live in $WHOODEO_DATA/orig and degraded
variants in $WHOODEO_DATA/low. A step picks one film, then one variant, then
as many center times as the batch size. Two time spans per variant are held
out for validation.

The loss is the weighted sum of the terms in the file. The first term anchors
the magnitude. balance start freezes the scales on the validation frames
before the first step. balance running keeps the weights as shares.

A gan term adds a U-Net discriminator. Its weights and Adam state are stored
in the checkpoint. whoodeo-apply reads only the generator. Resuming requires
the same architecture and the same loss. Learning rates come from the file.
"""

import argparse
import csv
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from whoodeo.catalog import data_root
from whoodeo.config import assert_resume_matches, load_config, shape_text
from whoodeo.discriminator import UNetDiscriminatorSN, gan_bce
from whoodeo.live import add_live_args, open_live
from whoodeo.nets import build_model
from whoodeo.objective import Objective, pixel_loss
from whoodeo.video import rgb_image

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
            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)
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


def batch_from_specs(films, specs, radius, device):
    lows = []
    highs = []
    for film_index, variant_index, time_sec in specs:
        film = films[film_index]
        low, high = load_sample(film, film.variants[variant_index], time_sec, radius)
        lows.append(low)
        highs.append(high)
    return torch.stack(lows).to(device), torch.stack(highs).to(device)


CSV_COLUMNS = [
    "step", "train_loss", "val_loss",
    "train_pixel", "train_vgg", "train_gan", "train_d",
    "val_pixel", "val_vgg", "val_gan", "val_d",
]


def write_plot(rows, path):
    train_steps = [row[0] for row in rows if row[1] is not None]
    train_loss = [row[1] for row in rows if row[1] is not None]
    val_steps = [row[0] for row in rows if row[2] is not None]
    val_loss = [row[2] for row in rows if row[2] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if train_steps:
        ax.plot(train_steps, train_loss, color="#1f4e79", linewidth=1.5, label="train")
    if val_steps:
        ax.plot(val_steps, val_loss, color="#c45911", marker="o", linewidth=1.2, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_cell(value):
    return "" if value is None else f"{value:.8f}"


def parts_or_blank(parts):
    if parts is None:
        return (None, None, None, None)
    return parts


def append_row(csv_path, rows, step, train_loss, val_loss, train_parts, val_parts):
    train_mse, train_vgg, train_gan, train_d = parts_or_blank(train_parts)
    val_mse, val_vgg, val_gan, val_d = parts_or_blank(val_parts)
    row = (
        step, train_loss, val_loss,
        train_mse, train_vgg, train_gan, train_d,
        val_mse, val_vgg, val_gan, val_d,
    )
    rows.append(row)
    with csv_path.open("a", newline="") as handle:
        csv.writer(handle).writerow([step, *(format_cell(value) for value in row[1:])])
    write_plot(rows, csv_path.with_name("loss.png"))


def write_loss_csv(csv_path, rows):
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([row[0], *(format_cell(value) for value in row[1:])])


def save_checkpoint(path, step, model, optimizer, saved, discriminator=None, disc_optimizer=None):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": saved,
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
        payload["disc_optimizer"] = disc_optimizer.state_dict()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_rows(csv_path):
    rows = []
    with csv_path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            def cell(name):
                text = record.get(name) or ""
                return float(text) if text else None

            rows.append((
                int(record["step"]),
                cell("train_loss"),
                cell("val_loss"),
                cell("train_pixel"),
                cell("train_vgg"),
                cell("train_gan"),
                cell("train_d"),
                cell("val_pixel"),
                cell("val_vgg"),
                cell("val_gan"),
                cell("val_d"),
            ))
    return rows


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


def checkpoint_args(cfg, objective):
    gan = cfg.term("gan")
    return {
        "arch": cfg.arch,
        "blocks": cfg.blocks,
        "channels": cfg.channels,
        "in_frames": cfg.in_frames,
        "batch": cfg.batch,
        "lr": cfg.lr,
        "disc_lr": None if gan is None else gan.disc_lr,
        "loss": cfg.loss_recipe(),
        "objective": objective.state_dict(),
    }


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
            low, high = batch_from_specs(films, chunk, radius, device)
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


def evaluate(model, films, specs, radius, batch_size, device, cfg, objective, discriminator):
    """Mean of the weighted objective. The GAN part uses the center crop."""
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
    gan = cfg.term("gan")
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            low, high = batch_from_specs(films, chunk, radius, device)
            pred = model(low)
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


_LOSS_SERIES = (
    ("train", 1),
    ("val", 2),
    ("pixel", 3),
    ("vgg", 4),
    ("gan", 5),
    ("d", 6),
    ("val pixel", 7),
    ("val vgg", 8),
    ("val gan", 9),
    ("val d", 10),
)


def push_loss_series(live, rows):
    if live is None:
        return
    for name, index in _LOSS_SERIES:
        xs = [row[0] for row in rows if row[index] is not None]
        ys = [row[index] for row in rows if row[index] is not None]
        if xs:
            live.series(name, xs, ys)


def train(cfg, preview=True, live_bind="127.0.0.1:8765"):
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    radius = cfg.in_frames // 2
    device = get_device()
    root = data_root()

    resumed = None
    resume_path = None
    saved_args = {}
    if cfg.resume is not None:
        resume_path = cfg.resume if cfg.resume.is_file() else cfg.resume / "model.pt"
        if not resume_path.is_file():
            raise SystemExit(f"no checkpoint: {resume_path}")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_args = resumed.get("args") or {}
        assert_resume_matches(cfg, saved_args, resume_path)

    films = open_films(root / "orig", root / "low", cfg.holdout)
    print(f"device {device}", flush=True)
    print(f"config {cfg.source}", flush=True)
    print(f"network {shape_text(*cfg.arch_key())}", flush=True)
    for film in films:
        for variant in film.variants:
            spans = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in variant.holdouts)
            print(
                f"pair {film.name}  {variant.name}  "
                f"{variant.low.width}x{variant.low.height}  "
                f"{variant.duration:.1f}s  holdout {spans}",
                flush=True,
            )

    out_dir = cfg.out
    if out_dir is None and resume_path is not None:
        out_dir = resume_path.parent
    elif out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path.cwd() / "runs" / f"train-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = out_dir / "config.yaml"
    if cfg.source.resolve() != copied.resolve():
        shutil.copyfile(cfg.source, copied)
    csv_path = out_dir / "loss.csv"
    ckpt_path = out_dir / "model.pt"
    continuing = (
        resume_path is not None
        and out_dir.resolve() == resume_path.parent.resolve()
        and csv_path.is_file()
    )
    rows = load_rows(csv_path) if continuing else []
    if not continuing:
        write_loss_csv(csv_path, [])
    else:
        with csv_path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != CSV_COLUMNS:
            write_loss_csv(csv_path, rows)
    limit = "until Ctrl-C" if cfg.steps is None else str(cfg.steps)
    print(f"run {out_dir}  steps {limit}", flush=True)

    model, label = build_model(cfg.arch, cfg.blocks, cfg.channels, cfg.in_frames)
    model = model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(label, flush=True)
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        move_optimizer(optimizer, device)
        apply_learning_rate(optimizer, cfg.lr)
        print(f"resume step {int(resumed['step'])}", flush=True)

    gan = cfg.term("gan")
    discriminator = None
    disc_optimizer = None
    if gan is not None:
        discriminator = UNetDiscriminatorSN().train().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=gan.disc_lr)
        count = sum(parameter.numel() for parameter in discriminator.parameters())
        if resumed is not None:
            if "discriminator" not in resumed or "disc_optimizer" not in resumed:
                raise SystemExit(f"checkpoint {resume_path} has no discriminator")
            discriminator.load_state_dict(resumed["discriminator"])
            disc_optimizer.load_state_dict(resumed["disc_optimizer"])
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
    if resumed is not None:
        objective.load_state_dict(saved_args["objective"])
        print("loss scales from checkpoint", flush=True)
    else:
        layer_means, term_means = measure_raw_means(
            model, films, val_specs, radius, cfg.batch, device, cfg, objective, discriminator,
        )
        objective.calibrate(layer_means, term_means)
    describe_loss(cfg, objective)

    live = open_live(live_bind, preview)
    if live is not None:
        push_loss_series(live, rows)
    running = 0.0
    running_pixel = 0.0
    running_vgg = 0.0
    running_gan = 0.0
    running_d = 0.0
    running_n = 0
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
        append_row(csv_path, rows, step, train_loss, val_loss, train_parts, val_parts)
        push_loss_series(live, rows)
        if val_loss is None:
            return
        message = f"step {step:06d}  val {val_loss:.6f}"
        if val_parts is not None:
            message += "  " + part_text(*val_parts)
        print(message, flush=True)
        if live is not None:
            live.status(message)

    def checkpoint():
        save_checkpoint(
            ckpt_path, step, model, optimizer, checkpoint_args(cfg, objective),
            discriminator, disc_optimizer,
        )

    start_step = int(resumed["step"]) if resumed is not None else 0
    step = start_step
    steps_done = 0
    try:
        while cfg.steps is None or steps_done < cfg.steps:
            step += 1
            steps_done += 1
            started = time.perf_counter()
            specs = make_batch_specs(rng, films, "train", cfg.batch)
            film_index, variant_index, _time_sec = specs[0]
            batch_film = films[film_index]
            batch_variant = batch_film.variants[variant_index]
            low, high = batch_from_specs(films, specs, radius, device)
            pred = model(low)
            box = None
            d_value = None
            if gan is not None:
                box = sample_box(high, gan.crop, rng)
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if discriminator is not None:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            objective.observe(layer_raw, {name: value.detach() for name, value in raw.items()})
            running += loss.item()
            if has_pixel:
                running_pixel += weighted["pixel"].item()
            if has_vgg:
                running_vgg += weighted["vgg"].item()
            if gan is not None:
                running_gan += weighted["gan"].item()
            if d_value is not None:
                running_d += d_value
            running_n += 1
            batch_parts = logged_parts(
                weighted["pixel"].item() if has_pixel else None,
                weighted["vgg"].item() if has_vgg else None,
                weighted["gan"].item() if gan is not None else None,
                d_value,
            )
            line = (
                f"step {step:06d}  {time.perf_counter() - started:.1f}s  "
                f"loss {loss.item():.6f}  {part_text(*batch_parts)}  "
                f"{batch_variant.low.width}x{batch_variant.low.height}  "
                f"{batch_variant.name}  {batch_film.name}"
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
            do_log = step % cfg.log_every == 0 or finished
            do_val = step % cfg.val_every == 0 or finished
            if not (do_log or do_val):
                continue
            train_loss = None
            train_parts = None
            if do_log:
                train_loss = running / running_n
                train_parts = logged_parts(
                    running_pixel / running_n if has_pixel else None,
                    running_vgg / running_n if has_vgg else None,
                    running_gan / running_n if gan is not None else None,
                    running_d / running_n if gan is not None else None,
                )
                running = 0.0
                running_pixel = 0.0
                running_vgg = 0.0
                running_gan = 0.0
                running_d = 0.0
                running_n = 0
            val_loss = None
            val_parts = None
            if do_val:
                val_loss, val_pixel, val_vgg, val_gan, val_d = evaluate(
                    model, films, val_specs, radius, cfg.batch, device,
                    cfg, objective, discriminator,
                )
                val_parts = logged_parts(val_pixel, val_vgg, val_gan, val_d)
                checkpoint()
            log(step, train_loss, val_loss, train_parts, val_parts)
    except KeyboardInterrupt:
        checkpoint()
        print(f"interrupted, saved {ckpt_path}", flush=True)
        return
    finally:
        if live is not None:
            live.close()

    for film in films:
        film.orig.close()
        for variant in film.variants:
            variant.low.close()
    print(f"done  {csv_path}  {ckpt_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="train on random paired video frames")
    parser.add_argument("config", type=Path, help="training recipe")
    add_live_args(parser)
    args = parser.parse_args()
    train(load_config(args.config), preview=not args.no_preview, live_bind=args.live_bind)


if __name__ == "__main__":
    main()
