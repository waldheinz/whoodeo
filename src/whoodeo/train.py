"""Train on random full frames from paired videos.

Masters live in --orig. Each subdirectory of --low is one degradation
variant and keeps the source filename. A step picks one film, then one of
its variants, then as many center times as the batch size. Two time spans
per variant are held out for validation.

When --gan is above zero, a U-Net discriminator scores one HD crop per step.
Its weights and Adam state are stored in the checkpoint and restored by
--resume. whoodeo-apply reads only the generator.

A fresh run trains modified ESPCN at 32 blocks, 256 channels, batch 1,
generator learning rate 2e-5, discriminator learning rate 1e-4, and GAN
share 0.1. --resume keeps the checkpoint's architecture, width, batch, and
generator learning rate unless those flags are passed. The discriminator
learning rate is the current --disc-lr, including after a resume.
"""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from whoodeo.apply import detect_arch, infer_deform, infer_edsr, infer_modified
from whoodeo.catalog import EnvPath, resolve_data
from whoodeo.discriminator import UNetDiscriminatorSN, gan_bce
from whoodeo.nets import add_model_args, build_model
from whoodeo.video import Preview

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
SCALE = 2


class VGGLoss(nn.Module):
    """Mean L1 on frozen VGG19 relu2_2 and relu3_4. The classifier head is dropped."""

    LAYERS = (8, 17)

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: self.LAYERS[-1] + 1]
        for layer in features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.features = features.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def maps(self, image):
        hidden = (image - self.mean) / self.std
        found = []
        for index, layer in enumerate(self.features):
            hidden = layer(hidden)
            if index in self.LAYERS:
                found.append(hidden)
        return found

    def forward(self, pred, target):
        pred_maps = self.maps(pred.clamp(0, 1))
        with torch.no_grad():
            target_maps = self.maps(target.clamp(0, 1))
        loss = pred.new_zeros(())
        for pred_map, target_map in zip(pred_maps, target_maps):
            loss = loss + F.l1_loss(pred_map, target_map)
        return loss / len(self.LAYERS)


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
    "train_mse", "train_vgg", "train_gan", "train_d",
    "val_mse", "val_vgg", "val_gan", "val_d",
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


def save_checkpoint(path, step, model, optimizer, args, discriminator=None, disc_optimizer=None):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
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
                cell("train_mse"),
                cell("train_vgg"),
                cell("train_gan"),
                cell("train_d"),
                cell("val_mse"),
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


def objective_parts(pred, high, perceptual, vgg_scale):
    mse = F.mse_loss(pred, high)
    if perceptual is None:
        return mse, mse, mse.new_zeros(())
    vgg_term = vgg_scale * perceptual(pred, high)
    return mse + vgg_term, mse, vgg_term


def evaluate(model, films, specs, radius, batch_size, device, perceptual, vgg_scale,
             discriminator, gan_scale, gan_crop):
    """Mean of the generator objective and its MSE, VGG, and weighted GAN parts.

    The GAN part uses the center crop. Validation does not update either network.
    """
    model.eval()
    disc_training = discriminator is not None and discriminator.training
    if discriminator is not None:
        discriminator.eval()
    loss_acc = 0.0
    mse_acc = 0.0
    vgg_acc = 0.0
    gan_acc = 0.0
    d_acc = 0.0
    saw_d = False
    count = 0
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            low, high = batch_from_specs(films, chunk, radius, device)
            pred = model(low)
            mse_total, mse, vgg_term = objective_parts(pred, high, perceptual, vgg_scale)
            gan_term = pred.new_zeros(())
            d_here = None
            if discriminator is not None and gan_scale != 0:
                box = sample_box(high, gan_crop, None)
                fake_logits = discriminator(take(pred, box))
                real_logits = discriminator(take(high, box))
                gan_term = gan_scale * gan_bce(fake_logits, True)
                d_here = 0.5 * (
                    gan_bce(real_logits, True).item() + gan_bce(fake_logits, False).item()
                )
            frames = len(chunk)
            loss_acc += (mse_total + gan_term).item() * frames
            mse_acc += mse.item() * frames
            vgg_acc += vgg_term.item() * frames
            gan_acc += gan_term.item() * frames
            if d_here is not None:
                d_acc += d_here * frames
                saw_d = True
            count += frames
    model.train()
    if disc_training:
        discriminator.train()
    d_mean = d_acc / count if saw_d else None
    return loss_acc / count, mse_acc / count, vgg_acc / count, gan_acc / count, d_mean


def option_passed(name):
    flag = f"--{name}"
    for token in sys.argv[1:]:
        if token.split("=", 1)[0] == flag:
            return True
    return False


def checkpoint_shape(arch, state, saved_args):
    """Blocks, channels, and frames stored in a checkpoint, filled in from the weights."""
    blocks = saved_args.get("blocks")
    channels = saved_args.get("channels")
    in_frames = saved_args.get("in_frames")
    if arch == "modified":
        found_blocks, found_channels, found_frames = infer_modified(state)
    elif arch == "deform":
        found_blocks, found_channels, found_frames = infer_deform(state)
    elif arch == "edsr":
        found_blocks, found_channels = infer_edsr(state)
        found_frames = 1
    else:
        return None, None, 1 if in_frames is None else in_frames
    if blocks is None:
        blocks = found_blocks
    if channels is None:
        channels = found_channels
    if in_frames is None:
        in_frames = found_frames
    return blocks, channels, in_frames


def adopt_checkpoint_run(args, state, saved_args):
    """Continue the saved run. Flags on the command line still win."""
    if not option_passed("arch"):
        args.arch = saved_args.get("arch") or detect_arch(state)
    blocks, channels, in_frames = checkpoint_shape(args.arch, state, saved_args)
    if not option_passed("blocks"):
        args.blocks = blocks
    if not option_passed("channels"):
        args.channels = channels
    if not option_passed("in-frames") and in_frames is not None:
        args.in_frames = in_frames
    if not option_passed("batch") and saved_args.get("batch") is not None:
        args.batch = int(saved_args["batch"])
    if not option_passed("lr") and saved_args.get("lr") is not None:
        args.lr = float(saved_args["lr"])


def part_text(mse, vgg, gan, d_loss):
    text = f"mse {mse:.6f}"
    if vgg is not None:
        text += f"  vgg {vgg:.6f}"
    if gan is not None:
        text += f"  gan {gan:.6f}"
    if d_loss is not None:
        text += f"  d {d_loss:.4f}"
    return text


def train(args):
    if args.vgg < 0 or args.vgg >= 1:
        raise SystemExit("--vgg must be in [0, 1)")
    if args.gan < 0 or args.gan >= 1:
        raise SystemExit("--gan must be in [0, 1)")
    if args.gan > 0 and (args.gan_crop < 8 or args.gan_crop % 8 != 0):
        raise SystemExit("--gan-crop must be a positive multiple of 8")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if args.in_frames % 2 != 1:
        raise SystemExit("--in-frames must be odd")
    radius = args.in_frames // 2
    device = get_device()

    resumed = None
    resume_path = None
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_file() else args.resume / "model.pt"
        if not resume_path.is_file():
            raise SystemExit(f"no checkpoint: {resume_path}")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_args = resumed.get("args") or {}
    else:
        saved_args = {}

    films = open_films(args.orig, args.low, args.holdout)
    print(f"device {device}", flush=True)
    for film in films:
        for variant in film.variants:
            spans = ", ".join(f"{start:.1f}-{end:.1f}s" for start, end in variant.holdouts)
            print(
                f"pair {film.name}  {variant.name}  "
                f"{variant.low.width}x{variant.low.height}  "
                f"{variant.duration:.1f}s  holdout {spans}",
                flush=True,
            )

    out_dir = args.out
    if out_dir is None and resume_path is not None:
        out_dir = resume_path.parent
    elif out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path.cwd() / "runs" / f"train-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "loss.csv"
    ckpt_path = out_dir / "model.pt"
    continuing = resume_path is not None and out_dir == resume_path.parent and csv_path.is_file()
    rows = load_rows(csv_path) if continuing else []
    if not continuing:
        write_loss_csv(csv_path, [])
    else:
        with csv_path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != CSV_COLUMNS:
            write_loss_csv(csv_path, rows)
    limit = "until Ctrl-C" if args.steps is None else str(args.steps)
    print(f"run {out_dir}  steps {limit}", flush=True)

    if resumed is not None:
        adopt_checkpoint_run(args, resumed["model"], saved_args)
    model, label = build_model(args.arch, args.blocks, args.channels, args.in_frames)
    model = model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(label, flush=True)
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        move_optimizer(optimizer, device)
        print(f"resume step {int(resumed['step'])}", flush=True)

    discriminator = None
    disc_optimizer = None
    if args.gan > 0:
        discriminator = UNetDiscriminatorSN().train().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)
        count = sum(parameter.numel() for parameter in discriminator.parameters())
        if resumed is not None and resumed.get("discriminator"):
            discriminator.load_state_dict(resumed["discriminator"])
            if resumed.get("disc_optimizer"):
                disc_optimizer.load_state_dict(resumed["disc_optimizer"])
                move_optimizer(disc_optimizer, device)
            print("loaded discriminator", flush=True)
        for group in disc_optimizer.param_groups:
            group["lr"] = args.disc_lr
        print(
            f"discriminator unet-sn  {count} parameters  crop {args.gan_crop}  "
            f"lr {args.disc_lr:g}",
            flush=True,
        )

    val_specs = []
    while len(val_specs) < args.val_count:
        need = min(args.batch, args.val_count - len(val_specs))
        val_specs.extend(make_batch_specs(rng, films, "val", need))

    perceptual = None
    vgg_scale = 0.0
    args.vgg_scale = 0.0
    if args.vgg > 0 and saved_args.get("vgg_scale") is not None:
        perceptual = VGGLoss().to(device)
        vgg_scale = float(saved_args["vgg_scale"])
        args.vgg_scale = vgg_scale
        print(f"vgg scale {vgg_scale:.6g} from checkpoint", flush=True)
    elif args.vgg > 0:
        perceptual = VGGLoss().to(device)
        # Raw feature L1 is orders of magnitude above pixel MSE. One scale,
        # measured on the val frames before any step, makes `args.vgg` the
        # VGG share of MSE plus VGG at the start. It stays fixed afterwards.
        _total, mse_mean, perc_mean, _gan, _d = evaluate(
            model, films, val_specs, radius, args.batch, device,
            perceptual, 1.0, None, 0.0, args.gan_crop,
        )
        if perc_mean <= 0:
            raise SystemExit("VGG loss on the val frames is zero")
        vgg_scale = (args.vgg / (1.0 - args.vgg)) * (mse_mean / perc_mean)
        args.vgg_scale = vgg_scale
        print(
            f"vgg relu2_2+relu3_4  share {args.vgg:.2f}  scale {vgg_scale:.6g}  "
            f"val mse {mse_mean:.6f}  val vgg {vgg_scale * perc_mean:.6f}",
            flush=True,
        )
    else:
        mse_mean = None
        print("loss mse", flush=True)

    gan_scale = 0.0
    args.gan_scale = 0.0
    saved_disc = resumed is not None and bool(resumed.get("discriminator"))
    if args.gan > 0 and saved_disc and saved_args.get("gan_scale") is not None:
        gan_scale = float(saved_args["gan_scale"])
        args.gan_scale = gan_scale
        print(f"gan scale {gan_scale:.6g} from checkpoint", flush=True)
    elif args.gan > 0:
        if args.vgg > 0 and saved_args.get("vgg_scale") is None:
            base = mse_mean + args.vgg_scale * perc_mean
        else:
            _total, base_mse, base_vgg, _gan, _d = evaluate(
                model, films, val_specs, radius, args.batch, device,
                perceptual, vgg_scale, None, 0.0, args.gan_crop,
            )
            base = base_mse + base_vgg
        _total, _mse, _vgg, raw_gan, _d = evaluate(
            model, films, val_specs, radius, args.batch, device,
            None, 0.0, discriminator, 1.0, args.gan_crop,
        )
        if raw_gan <= 0:
            raise SystemExit("GAN loss on the val frames is zero")
        gan_scale = (args.gan / (1.0 - args.gan)) * (base / raw_gan)
        args.gan_scale = gan_scale
        print(
            f"gan unet-sn  share {args.gan:.2f}  scale {gan_scale:.6g}  "
            f"val gan {gan_scale * raw_gan:.6f}",
            flush=True,
        )

    preview = None if args.no_preview else Preview()
    running = 0.0
    running_mse = 0.0
    running_vgg = 0.0
    running_gan = 0.0
    running_d = 0.0
    running_n = 0
    show_parts = perceptual is not None or discriminator is not None

    def log(step, train_loss, val_loss, train_parts, val_parts):
        append_row(csv_path, rows, step, train_loss, val_loss, train_parts, val_parts)
        message = f"step {step:06d}"
        if train_loss is not None:
            message += f"  train {train_loss:.6f}"
            if train_parts is not None:
                message += "  " + part_text(*train_parts)
        if val_loss is not None:
            message += f"  val {val_loss:.6f}"
            if val_parts is not None:
                message += "  " + part_text(*val_parts)
        print(message, flush=True)

    def checkpoint():
        save_checkpoint(
            ckpt_path, step, model, optimizer, args, discriminator, disc_optimizer,
        )

    start_step = int(resumed["step"]) if resumed is not None else 0
    step = start_step
    steps_done = 0
    try:
        while args.steps is None or steps_done < args.steps:
            step += 1
            steps_done += 1
            started = time.perf_counter()
            specs = make_batch_specs(rng, films, "train", args.batch)
            film_index, variant_index, _time_sec = specs[0]
            batch_film = films[film_index]
            batch_variant = batch_film.variants[variant_index]
            low, high = batch_from_specs(films, specs, radius, device)
            pred = model(low)
            mse_total, mse, vgg_term = objective_parts(pred, high, perceptual, vgg_scale)
            gan_term = pred.new_zeros(())
            d_value = None
            if discriminator is not None:
                box = sample_box(high, args.gan_crop, rng)
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
                gan_term = gan_scale * gan_bce(discriminator(take(pred, box)), True)
            loss = mse_total + gan_term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if discriminator is not None:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            running += loss.item()
            running_mse += mse.item()
            running_vgg += vgg_term.item()
            running_gan += gan_term.item()
            if d_value is not None:
                running_d += d_value
            running_n += 1
            batch_parts = (
                mse.item(),
                vgg_term.item() if perceptual is not None else None,
                gan_term.item() if discriminator is not None else None,
                d_value,
            )
            print(
                f"step {step:06d}  {time.perf_counter() - started:.1f}s  "
                f"loss {loss.item():.6f}  {part_text(*batch_parts)}  "
                f"{batch_variant.low.width}x{batch_variant.low.height}  "
                f"{batch_variant.name}  {batch_film.name}",
                flush=True,
            )

            if preview is not None:
                center = radius * 3
                preview.show((
                    low[0, center:center + 3].detach(),
                    pred[0].detach(),
                ))

            finished = args.steps is not None and steps_done == args.steps
            do_log = step % args.log_every == 0 or finished
            do_val = step % args.val_every == 0 or finished
            if not (do_log or do_val):
                continue
            train_loss = None
            train_parts = None
            if do_log:
                train_loss = running / running_n
                if show_parts:
                    train_parts = (
                        running_mse / running_n,
                        running_vgg / running_n if perceptual is not None else None,
                        running_gan / running_n if discriminator is not None else None,
                        running_d / running_n if discriminator is not None else None,
                    )
                running = 0.0
                running_mse = 0.0
                running_vgg = 0.0
                running_gan = 0.0
                running_d = 0.0
                running_n = 0
            val_loss = None
            val_parts = None
            if do_val:
                print(f"step {step:06d}  val", flush=True)
                val_loss, val_mse, val_vgg, val_gan, val_d = evaluate(
                    model, films, val_specs, radius, args.batch, device,
                    perceptual, vgg_scale, discriminator, gan_scale, args.gan_crop,
                )
                if show_parts:
                    val_parts = (
                        val_mse,
                        val_vgg if perceptual is not None else None,
                        val_gan if discriminator is not None else None,
                        val_d if discriminator is not None else None,
                    )
                checkpoint()
            log(step, train_loss, val_loss, train_parts, val_parts)
    except KeyboardInterrupt:
        checkpoint()
        print(f"interrupted, saved {ckpt_path}", flush=True)
        if preview is not None:
            preview.close()
        return

    if preview is not None:
        preview.close()
    for film in films:
        film.orig.close()
        for variant in film.variants:
            variant.low.close()
    print(f"done  {csv_path}  {ckpt_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="train on random paired video frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orig", type=Path, default=EnvPath("orig"), help="master videos")
    parser.add_argument("--low", type=Path, default=EnvPath("low"), help="degraded variants")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="stop after this many steps; default runs until Ctrl-C",
    )
    parser.add_argument("--batch", type=int, default=1, help="frames per step, all from one film")
    parser.add_argument("--lr", type=float, default=2e-5, help="Adam learning rate of the generator")
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
        help="Adam learning rate of the discriminator; applied again after --resume",
    )
    add_model_args(parser)
    parser.set_defaults(blocks=32, channels=256)
    for action in parser._actions:
        if action.dest == "arch":
            action.help = "network"
        elif action.dest == "blocks":
            action.help = "residual blocks; 32 for a fresh modified run, otherwise that arch's preset"
        elif action.dest == "channels":
            action.help = "feature channels; 256 for a fresh modified run, otherwise that arch's preset"
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint or run directory to continue")
    parser.add_argument("--no-preview", action="store_true", help="do not open the live window")
    parser.add_argument(
        "--vgg",
        type=float,
        default=0.5,
        help="VGG share of MSE+VGG at the start; 0.5 matches MSE, 0 uses MSE only",
    )
    parser.add_argument(
        "--gan",
        type=float,
        default=0.1,
        help="GAN share of MSE+VGG+GAN at the start; 0 leaves the discriminator out",
    )
    parser.add_argument(
        "--gan-crop",
        type=int,
        default=256,
        help="HD crop scored by the discriminator; must be a multiple of 8",
    )
    args = parser.parse_args()
    args.orig = resolve_data(args.orig)
    args.low = resolve_data(args.low)
    # 32x256 is the fresh modified default. Other archs keep their preset, and
    # a resumed run takes its shape from the checkpoint unless the flag was passed.
    fresh_modified = args.resume is None and args.arch == "modified"
    if not fresh_modified:
        if not option_passed("blocks"):
            args.blocks = None
        if not option_passed("channels"):
            args.channels = None
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
