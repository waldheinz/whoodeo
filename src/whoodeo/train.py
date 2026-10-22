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
shift and scale per channel in the model and logs the mean scale as
val_latent/scale. The mean shift is val_latent/mu_mean.

latent_noise: {tau: t} perturbs the vae2 latent in training, encoder and
decoder together, also with freeze_encoder. Each frame adds s * std_c * N(0, 1)
with s ~ |N(0, t^2)| and std_c the running std of μ per channel. It is not part
of the loss recipe, so a resume may change it; tau 0 is off. Validation decodes
μ. Two vae2 terms stand beside the objective, in the step loss and not in val:
mu_l2 is weight * mean(μ^2) against scale drift, and scale_eq picks one of its
factors k per step, decodes the k x k average-pooled latent, and compares it
with the k x k average-pooled frame under the pixel, color, and VGG weights of
the objective, times its weight.

Each vae2 validation also logs val_noise/psnr_<s> for s 0.1, 0.3, 0.5: the PSNR
of decode(μ + s * std_c * N(0, 1)) with a fixed seed, std_c the std of the
validation μ per channel. val_noise/scale_eq_psnr decodes the 2 x 2 pooled μ against
the 2 x 2 pooled frame. `python -m whoodeo.eval_latent <checkpoint>` computes
the same numbers for a saved checkpoint.

Training batches are decoded on one side thread into a queue of three CPU
batches. A frame read opens the video and closes it before the next read.
An unreadable training clip is logged and the loader draws a new batch.
The step copies a batch onto the device. Validation reads on the main thread.

faces: {weight, score, expand, feather} (vae, vae2, and latent_up) runs
OpenCV YuNet on the CPU on every final training frame, in the loader thread,
and on the validation frames; for latent_up on the master of each pair. The soft face mask rides in the batch; the step turns it
into 1 + (weight - 1) x mask divided by its mean per frame and weighs oklab,
ms_ssim, and vgg with it (scale_eq, mu_l2 and the KL stay unweighted).
train_raw/face_fraction and val_raw/face_fraction are the share of face
pixels, val/psnr_face and val/psnr_nonface the PSNR over face and other
pixels. The live page draws the boxes on recon (upscale) and master. latent_up
also weighs its latent L1 with the map area-pooled by the vae2 factor 8 to
the latent grid, again divided by its mean per frame, and logs
val_ref/vae_psnr_face and val_ref/vae_psnr_nonface beside val_ref/vae_psnr.
Weight 1 or no faces: section runs no detector. See whoodeo.faces.

The loss is the sum of weight * raw over the terms in the file. A weight
is a multiplier, not a share, and nothing is calibrated on the network.
vgg weighs each of its layers. A vae adds beta times the KL, a gan term
adds as above.

TensorBoard groups tags by their first segment, so the headline numbers sit
in train/ and val/ and the details in groups of their own:
train/loss, train/lr, and the weighted terms (train/pixel, train/vgg, ...);
val/loss, val/psnr, and the weighted terms on the validation frames;
train_raw/ and val_raw/ the unweighted terms, vgg as vgg_<layer>, the KL as
kl, and for vae2 train_raw/mu_l2, scale_eq, scale_eq_factor, gan, and
gan_weight; train_latent/ and val_latent/ the latent statistics, with the
val_latent/mu histogram and the val_latent/channels table; val_noise/ the
vae2 robustness PSNRs; val_ref/ the latent_up reference PSNRs; and for vote
train_neighbor/ and val_neighbor/. val/psnr is the mean PSNR of the clamped
output over the validation frames, comparable across loss weights.

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
file. train.pt is the step, the optimizers, the discriminator, and the loss
recipe. latest in the run points at the newest of
those directories, and latest in the runs directory follows the newest
checkpoint written. The run directory also gets a TensorBoard event file for the losses.
The vote model adds train_neighbor/weight and train_neighbor/grad on each
step, and val_neighbor/image when validation runs. weight and grad cover every kernel
in the shared vote. A vae adds train_raw/kl and val_raw/kl. The step loss includes
beta times that KL. The validation loss the plateau reads is the sum of the
weighted terms, without the KL and without the gan term. It also records train_latent/mu_mean, mu_std, sigma_mean
and spread for the batch, the same four under val_latent/ for the validation
set, and the channel summaries active and kl_min. active counts the channels
whose μ varies with std above 0.1, kl_min is the KL of the weakest channel.
Each validation also writes the per-channel table (μ std, σ mean, KL) to the
text tab as val_latent/channels. spread is the square root of
var(μ) + mean(σ²) and sits near 1 when the code matches the unit Gaussian.
The validation histogram is val_latent/mu.
whoodeo-board serves every run.

`arch: latent_up` reads the same low and master pairs as the upscalers and
predicts the normalized code z = (μ - shift) * scale of a frozen vae2 at 1/4
of the low frame, which is the vae2 latent of the 2x master. `vae` under
model: names that vae2, a run directory (its latest checkpoint) or a
checkpoint; the config stores the checkpoint it resolved to, so a resume
decodes with the same one. The vae2 stays in eval mode with no gradient on
its weights and no latent noise. Each step encodes the master with it under
no_grad for the target code. The `latent` term is L1 between the predicted
and that target code. The image terms (pixel, vgg, color, oklab, ms_ssim) compare
decode(z / scale + shift), cut to the master size, with the master, and their
gradient runs through the frozen decoder into z. The step skips the decode
when no image term is set and the live page is off. train_raw/latent and
val_raw/latent are logged in every case. The first validation of a
process also logs three references on the same frames, val_ref/bicubic_psnr, val_ref/bilinear_psnr, and
val_ref/vae_psnr, the vae2 roundtrip decode(encode(master)), which is
the ceiling of this decoder. The live page shows low (bicubic 2x), upscale,
and master. init_from_vae: true starts the last blocks at 1/4 and the head
from the vae2 encoder, the head with shift and scale folded in, on a fresh
run only.

--resume continues a run from its newest checkpoint: the same architecture,
the same loss weights, and the Adam state. The schedule comes from the
file. A plateau resume keeps its rate and its miss count when lr
is unchanged. A vae beta comes from the file too, on resume and fine-tune.
Resuming an older checkpoint starts a new run, so the later
checkpoints stay where they are. An older model.pt that still carries the
optimizer loads the same way, and the next save writes checkpoints/.

--model fine-tunes that generator under the loss in the file. The optimizer
and the discriminator start over, and the result is a new run. The recipe may omit the model: section; the
architecture then comes from the model file. A model: section that is
still present has to match the file, beta aside. Keep it to change beta.
"""

import argparse
import math
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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from whoodeo.catalog import data_root, runs_root
from whoodeo.checkpoint import (
    build_generator, continue_run, load_generator, load_resume, run_directory, save_run,
)
from whoodeo.config import (
    IMAGE_TERMS, LATENT_ARCH, VAE_ARCHS, architecture_dict, assert_resume_matches, cosine_learning_rate,
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
from whoodeo.models.latent_up import FACTOR as LATENT_FACTOR
from whoodeo.models.latent_up import init_from_vae
from whoodeo.models.vae import LatentTally, latent_scalars, latent_table
from whoodeo.faces import FACE_LEVEL, FaceDetector, detect_batch, face_fraction, weight_map
from whoodeo.objective import (
    Objective, color_loss, ms_ssim_loss, oklab_full, oklab_loss, pixel_loss, raw_scalars,
)
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
    # latent_up faces: the soft mask (B, 1, H, W) of the masters and the YuNet boxes.
    faces: torch.Tensor | None = None
    boxes: list | None = None


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

    emit("train/loss", train_loss)
    emit("val/loss", val_loss)
    emit("train/lr", lr)
    for prefix, parts in (("train", train_parts), ("val", val_parts)):
        if parts is None:
            continue
        pixel, vgg, gan, d_loss = parts
        emit(f"{prefix}/pixel", pixel)
        emit(f"{prefix}/vgg", vgg)
        emit(f"{prefix}/gan", gan)
        emit(f"{prefix}/d", d_loss)
    if usage is not None:
        emit("train_neighbor/weight", usage.get("weight"))
        emit("train_neighbor/grad", usage.get("grad"))
    emit("val_neighbor/image", neighbor_image)
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
    print("loss  sum of weight x raw", flush=True)
    for term in cfg.terms:
        if term.name == "mu_l2":
            print(f"  mu_l2  weight {term.weight:g}  step only", flush=True)
            continue
        if term.name == "latent":
            print(f"  latent l1 on the normalized code  weight {term.weight:g}", flush=True)
            continue
        if term.name == "scale_eq":
            factors = "/".join(str(k) for k in term.factors)
            print(f"  scale_eq factors {factors}  weight {term.weight:g}  step only", flush=True)
            continue
        if term.name not in objective.names:
            start = 0 if term.start is None else term.start
            print(
                f"  {term.name} crop {term.crop}  weight {term.weight:g}  "
                f"adaptive  from step {start}",
                flush=True,
            )
            continue
        if term.name == "vgg":
            layers = "  ".join(f"{layer} {weight:g}" for layer, weight in term.layers)
            print(f"  vgg  {layers}", flush=True)
            continue
        if term.name == "pixel":
            detail = term.kind
        elif term.name == "color":
            detail = f"{term.space}/{term.pool}"
        elif term.name == "oklab":
            detail = "L/a/b " + "/".join(f"{value:g}" for value in term.channels)
        elif term.name == "ms_ssim":
            detail = f"oklab L, {term.scales} scales"
        else:
            detail = f"crop {term.crop}"
        print(f"  {term.name} {detail}  weight {term.weight:g}", flush=True)


def raw_terms(objective, cfg, pred, high, discriminator, box, pixel_weight=None):
    """Unweighted terms, and the vgg L1 of each layer. All still in the graph.

    `pixel_weight` (B, 1, H, W), the faces map, weighs oklab, ms_ssim, and vgg.
    """
    raw = {}
    layer_raw = {}
    pixel = cfg.term("pixel")
    if pixel is not None:
        raw["pixel"] = pixel_loss(pred, high, pixel.kind)
    if cfg.term("vgg") is not None:
        layer_raw = objective.vgg_layers(pred, high, pixel_weight)
    color = cfg.term("color")
    if color is not None:
        raw["color"] = color_loss(pred, high, color.space, color.pool)
    lab = cfg.term("oklab")
    structure = cfg.term("ms_ssim")
    if lab is not None or structure is not None:
        pred_lab = oklab_full(pred)
        with torch.no_grad():
            high_lab = oklab_full(high)
        if lab is not None:
            raw["oklab"] = oklab_loss(pred_lab, high_lab, lab.channels, pixel_weight)
        if structure is not None:
            raw["ms_ssim"] = ms_ssim_loss(pred_lab, high_lab, structure.scales, pixel_weight)
    if "gan" in objective.names:
        raw["gan"] = gan_bce(discriminator(take(pred, box)), True)
    return raw, layer_raw


def psnr_sum(pred, target):
    """Sum over the batch of each frame's PSNR in dB, the prediction clamped to [0, 1]."""
    error = (pred.clamp(0, 1) - target).square().flatten(1).mean(1)
    return (-10.0 * torch.log10(error.clamp_min(1e-10))).sum().item()


def masked_error(pred, target, inside):
    """Squared error summed over the pixels of `inside` (B, 1, H, W), and their count."""
    error = (pred.clamp(0, 1) - target).square().mean(1, keepdim=True)
    weight = inside.to(error.dtype)
    return float((error * weight).sum()), float(weight.sum())


def pooled_psnr(error, count):
    if count <= 0:
        return None
    return -10.0 * math.log10(max(error / count, 1e-10))


# Noise levels of val_noise/psnr_<s>, in units of the per-channel std of μ.
NOISE_LEVELS = (0.1, 0.3, 0.5)
NOISE_SEED = 1234


def pooled_pair(model, image, latent, k):
    """decode(avg_pool(latent, k)) and avg_pool(frame, k), or None when too small.

    Both are cut to a whole number of k x k latent cells inside the unpadded frame.
    """
    factor = model.factor
    lat_h = (image.shape[-2] // factor) // k * k
    lat_w = (image.shape[-1] // factor) // k * k
    if lat_h == 0 or lat_w == 0:
        return None, None
    code = F.avg_pool2d(latent[:, :, :lat_h, :lat_w], k)
    frame = F.avg_pool2d(image[:, :, :lat_h * factor, :lat_w * factor], k)
    return model.decode(code), frame


def robustness(model, mus, images, std):
    """val_noise/psnr_<s> and val_noise/scale_eq_psnr over the validation frames.

    `mus` and `images` are the batches of one validation, μ on the device and the
    frames on the CPU. `std` is the per-channel std of μ. The noise has a fixed seed.
    """
    device = mus[0].device
    generator = torch.Generator(device=device)
    generator.manual_seed(NOISE_SEED)
    scale = torch.as_tensor(std, dtype=mus[0].dtype, device=device).view(1, -1, 1, 1)
    sums = {f"val_noise/psnr_{level:g}": 0.0 for level in NOISE_LEVELS}
    sums["val_noise/scale_eq_psnr"] = 0.0
    count = 0
    pooled_count = 0
    with torch.no_grad():
        for mu, image in zip(mus, images):
            image = image.to(device)
            height, width = image.shape[-2:]
            for level in NOISE_LEVELS:
                eps = torch.randn(mu.shape, device=device, dtype=mu.dtype, generator=generator)
                recon = model.decode(mu + level * scale * eps)[:, :, :height, :width]
                sums[f"val_noise/psnr_{level:g}"] += psnr_sum(recon, image)
            recon, frame = pooled_pair(model, image, mu, 2)
            if recon is not None:
                sums["val_noise/scale_eq_psnr"] += psnr_sum(recon, frame)
                pooled_count += image.shape[0]
            count += image.shape[0]
    found = {tag: value / count for tag, value in sums.items() if tag != "val_noise/scale_eq_psnr"}
    if pooled_count:
        found["val_noise/scale_eq_psnr"] = sums["val_noise/scale_eq_psnr"] / pooled_count
    return found


def add_raw(acc, prefix, raw, layer_raw, frames):
    for tag, value in raw_scalars(prefix, raw, layer_raw).items():
        acc[tag] = acc.get(tag, 0.0) + value * frames


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
    Then the neighbor image effect, or None when the model has no vote, and
    the raw term means as val_raw/ tags beside val/psnr.
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
    raw_acc = {}
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
            raw, layer_raw = raw_terms(objective, cfg, pred, high, discriminator, box)
            total, weighted = objective.combine(raw, layer_raw)
            frames = len(chunk)
            loss_acc += total.item() * frames
            add_raw(raw_acc, "val", raw, layer_raw, frames)
            raw_acc["val/psnr"] = raw_acc.get("val/psnr", 0.0) + psnr_sum(pred, high)
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
    raw_means = {tag: value / count for tag, value in raw_acc.items()}
    return loss_acc / count, pixel_mean, vgg_mean, gan_mean, d_mean, image_mean, raw_means


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


def vae_batch(films, specs, crop, rng, detector=None, faces=None):
    """Stack master frames. The crop is drawn only after every frame is read.

    With a detector, YuNet runs on the final frames (after the crop), and the
    batch carries the boxes and the soft face mask of `faces`.
    """
    frames = [load_master(films[film_index], index) for film_index, index in specs]
    if crop is not None:
        cropped = []
        for frame in frames:
            batched = frame.unsqueeze(0)
            cropped.append(take(batched, exact_box(batched, crop, rng))[0])
        frames = cropped
    image = torch.stack(frames)
    boxes = None
    mask = None
    if detector is not None:
        boxes, mask = detect_batch(detector, image, faces.expand, faces.feather)
    return FrameBatch(image, int(image.shape[-1]), int(image.shape[-2]), mask, boxes)


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


def evaluate_vae(model, films, specs, crop, batch_size, device, cfg, objective, image_dir=None, step=None, detector=None):
    """Mean of the weighted objective. The KL is returned beside it.

    The extras hold the raw term means and val/psnr of the clamped reconstruction,
    and for a vae2 the noise and scale_eq PSNRs of `robustness`. With faces the
    objective is face-weighted as in training, and val/psnr_face and
    val/psnr_nonface are the PSNR pooled over all face and all other pixels.

    master.png is written once. recon-<step>.png is the reconstruction.
    """
    model.eval()
    loss_acc = 0.0
    pixel_acc = 0.0
    vgg_acc = 0.0
    image_acc = {}
    kl_acc = 0.0
    saw_kl = False
    raw_acc = {}
    count = 0
    saved = 0
    tally = LatentTally()
    robust = hasattr(model, "latent_ema_std")
    mus = []
    frames_cpu = []
    width = max(2, len(str(max(len(specs) - 1, 0))))
    faces = cfg.faces
    if faces is not None and detector is None:
        detector = FaceDetector(faces.score)
    face_acc = [0.0, 0.0, 0.0, 0.0]  # face error, face pixels, other error, other pixels
    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            chunk = specs[start:start + batch_size]
            batch = vae_batch(films, chunk, crop, None, detector if faces is not None else None, faces)
            image = batch.image.to(device)
            pixel_weight = None
            pred = model(image)
            tally.add(model.mu, model.logvar, sample=True)
            if robust:
                mus.append(model.mu)
                frames_cpu.append(image.cpu())
            if image_dir is not None:
                for offset in range(len(chunk)):
                    folder = image_dir / f"{saved + offset:0{width}d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_still(folder, "master", image[offset])
                    save_step_image(folder, "recon", step, pred[offset])
                saved += len(chunk)
            if batch.faces is not None:
                mask = batch.faces.to(device)
                pixel_weight = weight_map(mask, faces.weight)
                inside = mask >= FACE_LEVEL
                error, pixels = masked_error(pred, image, inside)
                face_acc[0] += error
                face_acc[1] += pixels
                error, pixels = masked_error(pred, image, ~inside)
                face_acc[2] += error
                face_acc[3] += pixels
            raw, layer_raw = raw_terms(objective, cfg, pred, image, None, None, pixel_weight)
            total, weighted = objective.combine(raw, layer_raw)
            frames = len(chunk)
            loss_acc += total.item() * frames
            add_raw(raw_acc, "val", raw, layer_raw, frames)
            raw_acc["val/psnr"] = raw_acc.get("val/psnr", 0.0) + psnr_sum(pred, image)
            if "pixel" in weighted:
                pixel_acc += weighted["pixel"].item() * frames
            if "vgg" in weighted:
                vgg_acc += weighted["vgg"].item() * frames
            for name in IMAGE_TERMS:
                if name in weighted:
                    image_acc[name] = image_acc.get(name, 0.0) + weighted[name].item() * frames
            if model.kl is not None:
                kl_acc += float(model.kl) * frames
                saw_kl = True
            count += frames
    report = tally.report()
    extras = {tag: value / count for tag, value in raw_acc.items()}
    if faces is not None:
        total_pixels = face_acc[1] + face_acc[3]
        extras["val_raw/face_fraction"] = face_acc[1] / total_pixels if total_pixels else 0.0
        for tag, value in (
            ("val/psnr_face", pooled_psnr(face_acc[0], face_acc[1])),
            ("val/psnr_nonface", pooled_psnr(face_acc[2], face_acc[3])),
        ):
            if value is not None:
                extras[tag] = value
    if robust and report is not None:
        std = [row[0] for row in report["channels"]]
        extras.update(robustness(model, mus, frames_cpu, std))
    model.train()
    pixel_mean = pixel_acc / count if cfg.term("pixel") is not None else None
    vgg_mean = vgg_acc / count if cfg.term("vgg") is not None else None
    for name, value in image_acc.items():
        extras[f"val/{name}"] = value / count
    return EvalResult(
        loss_acc / count,
        (pixel_mean, vgg_mean, None, None),
        None,
        kl_acc / count if saw_kl else None,
        report,
        extras,
    )


@dataclass
class FrameBatch:
    image: torch.Tensor
    width: int
    height: int
    # faces: the soft mask (B, 1, H, W) and the YuNet boxes per frame.
    faces: torch.Tensor | None = None
    boxes: list | None = None


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
    # Face boxes per shown image, for the live page.
    boxes: dict | None = None


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
    # The generator weights came from --resume or --model.
    loaded: bool = False

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
        encoder_blocks=cfg.encoder_blocks, decoder_blocks=cfg.decoder_blocks,
        widths=cfg.widths, decoder_widths=cfg.decoder_widths, posterior=cfg.posterior,
        activation=cfg.activation,
    )
    model = model.train().to(device)
    if hasattr(model, "noise_tau"):
        model.noise_tau = cfg.latent_noise_tau
    trainable = model.parameters()
    if cfg.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    print(label, flush=True)
    if cfg.latent_noise_tau > 0:
        print(f"latent noise tau {cfg.latent_noise_tau:g}", flush=True)
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
        loaded=generator is not None,
    )
    run.apply_rate(start_step)
    return run


def drive(run, preview, live_bind, make_val_specs, make_loader, one_step, evaluate_split):
    """The live page and the step loop.

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
            """Weighted terms beyond pixel, vgg, and gan, the raw terms, and what the step reports."""
            extras = {
                f"train/{name}": value.item()
                for name, value in weighted.items()
                if name not in ("pixel", "vgg", "gan")
            }
            extras.update(raw_scalars("train", work.raw, work.layer_raw))
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
                extras["train_raw/kl"] = train_kl
            if val_kl is not None:
                extras["val_raw/kl"] = val_kl
            if train_latent is not None:
                extras.update(latent_scalars(train_latent, "train"))
            if val_latent is not None:
                extras.update(latent_scalars(val_latent, "val"))
            write_scalars(
                writer, step, train_loss, val_loss, train_parts, val_parts, usage,
                neighbor_image, lr, extras or None,
            )
            hist = None if val_latent is None else val_latent.get("hist")
            if hist is not None:
                writer.add_histogram("val_latent/mu", hist, step)
            if val_latent is not None:
                writer.add_text("val_latent/channels", latent_table(val_latent), step)
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
            for name in IMAGE_TERMS:
                if val_extras and f"val/{name}" in val_extras:
                    message += f"  {name} {val_extras[f'val/{name}']:.6f}"
            if val_extras and "val_raw/latent" in val_extras:
                message += f"  latent {val_extras['val_raw/latent']:.5f}"
            if val_extras and "val/psnr" in val_extras:
                message += f"  psnr {val_extras['val/psnr']:.2f}"
            if val_extras and "val/psnr_face" in val_extras:
                message += f"  face {val_extras['val/psnr_face']:.2f}"
            if val_extras and "val/psnr_nonface" in val_extras:
                message += f"  nonface {val_extras['val/psnr_nonface']:.2f}"
            if val_extras and "val_noise/psnr_0.3" in val_extras:
                message += (
                    "  noise " + "/".join(
                        f"{val_extras[f'val_noise/psnr_{level:g}']:.2f}" for level in NOISE_LEVELS
                    )
                )
            if val_extras and "val_noise/scale_eq_psnr" in val_extras:
                message += f"  pooled {val_extras['val_noise/scale_eq_psnr']:.2f}"
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
                cfg.loss_recipe(),
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
                loss, weighted = objective.combine(work.raw, work.layer_raw)
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
                batch_parts = logged_parts(weighted, work)
                train_extras = step_extras(weighted, work)
                line = (
                    f"step {step:06d}  {time.perf_counter() - started:.1f}s  "
                    f"lr {lr:.4g}  loss {loss.item():.6f}  {part_text(*batch_parts)}  "
                    f"{work.width}x{work.height}"
                )
                for name in IMAGE_TERMS:
                    if f"train/{name}" in train_extras:
                        line += f"  {name} {train_extras[f'train/{name}']:.6f}"
                if "train_raw/latent" in train_extras:
                    line += f"  latent {train_extras['train_raw/latent']:.5f}"
                if "train_raw/gan_weight" in train_extras:
                    line += f"  gan_weight {train_extras['train_raw/gan_weight']:.4g}"
                if "train/scale_eq" in train_extras:
                    line += f"  scale_eq {train_extras['train/scale_eq']:.6f}"
                if "train/mu_l2" in train_extras:
                    line += f"  mu_l2 {train_extras['train/mu_l2']:.3g}"
                if "train_raw/face_fraction" in train_extras:
                    line += f"  faces {train_extras['train_raw/face_fraction']:.3f}"
                if train_kl is not None:
                    line += f"  kl {train_kl:.6f}"
                print(line, flush=True)

                if live is not None:
                    live.status(line)
                    if work.images is not None:
                        live.show({
                            name: rgb_image(tensor) for name, tensor in work.images.items()
                        }, boxes=work.boxes)

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
        loss, pixel, vgg, gan_value, d_value, neighbor, raw_means = evaluate(
            run.model, films, specs, radius, cfg.batch, run.device, cfg, objective,
            run.discriminator, image_dir=image_dir, step=step,
        )
        return EvalResult(
            loss, (pixel, vgg, gan_value, d_value), neighbor, None, None, raw_means,
        )

    return make_val_specs, make_loader, one_step, evaluate_split


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
    # The scale_eq factor of each step, from its own draw too.
    eq_rng = random.Random(cfg.seed + 2)
    counter = {"step": run.start_step}
    mu_l2 = cfg.term("mu_l2")
    scale_eq = cfg.term("scale_eq")
    faces = cfg.faces
    detector = None
    if faces is not None:
        detector = FaceDetector(faces.score)
        print(
            f"faces weight {faces.weight:g}  score {faces.score:g}  expand {faces.expand:g}  "
            f"feather {faces.feather:g}  oklab, ms_ssim, vgg  mean 1 per frame  {detector.path}",
            flush=True,
        )

    def side_terms(objective, image):
        """mu_l2 and scale_eq: the weighted sum for the step loss, and what to log."""
        added = None
        extras = {}
        if mu_l2 is not None:
            value = run.model.mu_grad.square().mean()
            weighted = mu_l2.weight * value
            added = weighted
            extras["train/mu_l2"] = weighted.item()
            extras["train_raw/mu_l2"] = value.item()
        if scale_eq is not None:
            k = eq_rng.choice(scale_eq.factors)
            recon, frame = pooled_pair(run.model, image, run.model.code, k)
            if recon is not None:
                raw, layer_raw = raw_terms(objective, cfg, recon, frame, None, None)
                value, _parts = objective.combine(raw, layer_raw)
                weighted = scale_eq.weight * value
                added = weighted if added is None else added + weighted
                extras["train/scale_eq"] = weighted.item()
                extras["train_raw/scale_eq"] = value.item()
                extras["train_raw/scale_eq_factor"] = float(k)
        return added, extras

    def make_val_specs():
        specs = []
        while len(specs) < cfg.val_count:
            need = min(cfg.batch, cfg.val_count - len(specs))
            specs.extend(make_vae_specs(rng, films, "val", need, cfg.crop))
        return specs

    def make_loader():
        def produce():
            specs = make_vae_specs(rng, films, "train", cfg.batch, cfg.crop)
            return vae_batch(films, specs, cfg.crop, rng, detector, faces)

        return BatchLoader(produce)

    def adversarial(objective, raw, layer_raw, pred, image):
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
        objective_loss, _weighted = objective.combine(raw, layer_raw)
        weight = gan.weight * adaptive_gan_weight(
            objective_loss, gan_loss, run.model.last_layer(),
        )
        added = weight * gan_loss
        return d_loss.item(), added, added.item(), {
            "train_raw/gan_weight": float(weight),
            "train_raw/gan": gan_loss.item(),
        }

    def one_step(loader, objective, show):
        counter["step"] += 1
        prepared = loader.get()
        image = prepared.image.to(run.device)
        pixel_weight = None
        face_extras = {}
        if prepared.faces is not None:
            mask = prepared.faces.to(run.device)
            pixel_weight = weight_map(mask, faces.weight)
            face_extras["train_raw/face_fraction"] = face_fraction(mask).item()
        pred = run.model(image)
        raw, layer_raw = raw_terms(objective, cfg, pred, image, None, None, pixel_weight)
        d_value = None
        added = None
        gan_value = None
        extras = None
        if gan is not None:
            d_value, added, gan_value, extras = adversarial(objective, raw, layer_raw, pred, image)
        side, side_extras = side_terms(objective, image)
        if side is not None:
            added = side if added is None else added + side
        if side_extras:
            extras = {**(extras or {}), **side_extras}
        if face_extras:
            extras = {**(extras or {}), **face_extras}
        images = None
        boxes = None
        if show:
            images = {
                "sample": prior_image(run.model, image),
                "recon": pred[0],
                "master": image[0],
            }
            if prepared.boxes is not None:
                boxes = {"recon": prepared.boxes[0], "master": prepared.boxes[0]}
        return StepOut(
            raw, layer_raw, d_value, images, prepared.width, prepared.height,
            added, gan_value, extras, boxes,
        )

    def evaluate_split(objective, specs, image_dir, step):
        result = evaluate_vae(
            run.model, films, specs, cfg.crop, cfg.batch, run.device, cfg, objective,
            image_dir=image_dir, step=step, detector=detector,
        )
        if hasattr(run.model, "set_latent_stats") and result.latent is not None:
            channels = result.latent["channels"]
            run.model.set_latent_stats(
                [row[3] for row in channels], [row[0] for row in channels],
            )
            extras = dict(result.extras or {})
            extras["val_latent/scale"] = float(run.model.latent_scale.mean())
            result.extras = extras
        return result

    return make_val_specs, make_loader, one_step, evaluate_split


def load_frozen_vae(cfg, device):
    """The vae2 that latent_up decodes with, in eval mode, its weights frozen."""
    spec = load_generator(cfg.vae)
    arch = spec.architecture.get("arch")
    if arch != "vae2":
        raise SystemExit(f"{spec.path}: latent_up decodes with a vae2, not {arch}")
    vae, label, _frames, step, path = build_generator(spec, device)
    if vae.factor != 2 * LATENT_FACTOR:
        raise SystemExit(
            f"{path}: vae2 factor {vae.factor}, latent_up needs {2 * LATENT_FACTOR} "
            f"(its own {LATENT_FACTOR} times the 2x scale)"
        )
    if vae.latent != cfg.latent:
        raise SystemExit(f"{path}: vae2 has {vae.latent} latent channels, the config says {cfg.latent}")
    if (
        bool((vae.latent_scale == 1).all()) and bool((vae.latent_shift == 0).all())
    ):
        raise SystemExit(f"{path}: the vae2 has no latent shift and scale yet, validate it once")
    vae.eval()
    vae.noise_tau = 0.0
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"vae {path}  step {step}  {label}  frozen", flush=True)
    print(
        f"vae latent shift {float(vae.latent_shift.mean()):+.3f} mean  "
        f"scale {float(vae.latent_scale.min()):.3f}..{float(vae.latent_scale.max()):.3f}",
        flush=True,
    )
    return vae


def latent_hooks(run):
    """Low frames in, the frozen vae2 code of the master as the target.

    The image terms compare the decoded code with the master. The first
    validation of the process adds the bicubic, bilinear, and vae2 roundtrip
    PSNR of the same frames.
    """
    cfg = run.cfg
    films = run.films
    rng = run.rng
    device = run.device
    vae = load_frozen_vae(cfg, device)
    if cfg.init_from_vae:
        if run.loaded:
            print("init_from_vae skipped: the weights come from a checkpoint", flush=True)
        else:
            print(init_from_vae(run.model, vae), flush=True)
    shift = vae.latent_shift.view(1, -1, 1, 1)
    scale = vae.latent_scale.view(1, -1, 1, 1)
    need_image = any(cfg.term(name) is not None for name in ("pixel", "vgg", *IMAGE_TERMS))
    pending = {"baselines": True}
    faces = cfg.faces
    detector = None
    if faces is not None:
        detector = FaceDetector(faces.score)
        print(
            f"faces weight {faces.weight:g}  score {faces.score:g}  expand {faces.expand:g}  "
            f"feather {faces.feather:g}  oklab, ms_ssim, vgg, latent (area 1/{vae.factor})  "
            f"mean 1 per frame  {detector.path}",
            flush=True,
        )

    def face_weights(mask, code):
        """The pixel weight map of the master and its area-pooled copy on the latent grid."""
        pixel_weight = weight_map(mask, faces.weight)
        # ceil_mode: the last cell of a master that is not a multiple of the
        # factor (852 -> 107 cells) averages the pixels it has.
        latent_weight = F.avg_pool2d(pixel_weight, vae.factor, ceil_mode=True)
        latent_weight = latent_weight / latent_weight.mean(dim=(-2, -1), keepdim=True)
        if latent_weight.shape[-2:] != code.shape[-2:]:
            raise SystemExit(
                f"faces: latent weight {tuple(latent_weight.shape[-2:])} does not match "
                f"the code {tuple(code.shape[-2:])}"
            )
        return pixel_weight, latent_weight

    def target_code(high):
        """(μ - shift) * scale of the master, no gradient."""
        with torch.no_grad():
            padded, _height, _width = pad_to(high, vae.factor)
            mu, _logvar = vae.encode(padded)
            return (mu - shift) * scale, mu

    def decode(code, height, width):
        return vae.decode(code / scale + shift)[:, :, :height, :width]

    def predict(low, high):
        code = run.model(low)
        target, mu = target_code(high)
        if code.shape != target.shape:
            raise SystemExit(
                f"code {tuple(code.shape)} does not match the vae2 code {tuple(target.shape)} "
                f"of a {high.shape[-1]}x{high.shape[-2]} master"
            )
        return code, target, mu

    def bicubic(low):
        return F.interpolate(low, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)

    def terms(objective, code, target, pred, high, pixel_weight=None, latent_weight=None):
        if latent_weight is None:
            raw = {"latent": F.l1_loss(code, target)}
        else:
            raw = {"latent": ((code - target).abs() * latent_weight).mean()}
        layer_raw = {}
        if need_image:
            image_raw, layer_raw = raw_terms(objective, cfg, pred, high, None, None, pixel_weight)
            raw.update(image_raw)
        return raw, layer_raw

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
            low, high = batch_from_specs(films, specs, 0)
            boxes = None
            mask = None
            if detector is not None:
                boxes, mask = detect_batch(detector, high, faces.expand, faces.feather)
            return PreparedBatch(
                low, high, variant.low.width, variant.low.height, None, mask, boxes,
            )

        return BatchLoader(produce)

    def one_step(loader, objective, show):
        prepared = loader.get()
        low = prepared.low.to(device)
        high = prepared.high.to(device)
        height, width = high.shape[-2:]
        code, target, _mu = predict(low, high)
        pixel_weight = None
        latent_weight = None
        extras = {"train_latent/z_std": float(code.detach().std())}
        if prepared.faces is not None:
            mask = prepared.faces.to(device)
            pixel_weight, latent_weight = face_weights(mask, code)
            extras["train_raw/face_fraction"] = face_fraction(mask).item()
        pred = None
        if need_image:
            pred = decode(code, height, width)
        elif show:
            with torch.no_grad():
                pred = decode(code, height, width)
        raw, layer_raw = terms(objective, code, target, pred, high, pixel_weight, latent_weight)
        if cfg.term("latent") is None:
            raw["latent"] = raw["latent"].detach()
        images = None
        boxes = None
        if show:
            images = {
                "low": bicubic(low[:1])[0],
                "upscale": pred[0].detach(),
                "master": high[0],
            }
            if prepared.boxes is not None:
                boxes = {"upscale": prepared.boxes[0], "master": prepared.boxes[0]}
        return StepOut(
            raw, layer_raw, None, images, prepared.width, prepared.height,
            extras=extras, boxes=boxes,
        )

    def evaluate_split(objective, specs, image_dir, step):
        started = time.perf_counter()
        run.model.eval()
        baselines = pending["baselines"]
        loss_acc = 0.0
        weighted_acc = {}
        raw_acc = {}
        z_std_acc = 0.0
        count = 0
        saved = 0
        # face error, face pixels, other error, other pixels; the same for the vae2 roundtrip
        face_acc = [0.0, 0.0, 0.0, 0.0]
        vae_face_acc = [0.0, 0.0, 0.0, 0.0]
        digits = max(2, len(str(max(len(specs) - 1, 0))))
        with torch.no_grad():
            for start in range(0, len(specs), cfg.batch):
                chunk = specs[start:start + cfg.batch]
                low_cpu, high_cpu = batch_from_specs(films, chunk, 0)
                low = low_cpu.to(device)
                high = high_cpu.to(device)
                height, width = high.shape[-2:]
                code, target, mu = predict(low, high)
                pred = decode(code, height, width)
                pixel_weight = None
                latent_weight = None
                inside = None
                if detector is not None:
                    _boxes, mask = detect_batch(detector, high_cpu, faces.expand, faces.feather)
                    mask = mask.to(device)
                    pixel_weight, latent_weight = face_weights(mask, code)
                    inside = mask >= FACE_LEVEL
                    error, pixels = masked_error(pred, high, inside)
                    face_acc[0] += error
                    face_acc[1] += pixels
                    error, pixels = masked_error(pred, high, ~inside)
                    face_acc[2] += error
                    face_acc[3] += pixels
                raw, layer_raw = terms(objective, code, target, pred, high, pixel_weight, latent_weight)
                total, weighted = objective.combine(raw, layer_raw)
                frames = len(chunk)
                loss_acc += total.item() * frames
                for name, value in weighted.items():
                    weighted_acc[name] = weighted_acc.get(name, 0.0) + value.item() * frames
                add_raw(raw_acc, "val", raw, layer_raw, frames)
                raw_acc["val/psnr"] = raw_acc.get("val/psnr", 0.0) + psnr_sum(pred, high)
                z_std_acc += float(code.std()) * frames
                roundtrip = None
                if baselines:
                    roundtrip = vae.decode(mu)[:, :, :height, :width]
                    for tag, image in (
                        ("val_ref/bicubic_psnr", bicubic(low)),
                        ("val_ref/bilinear_psnr", F.interpolate(
                            low, scale_factor=2, mode="bilinear", align_corners=False,
                        )),
                        ("val_ref/vae_psnr", roundtrip),
                    ):
                        raw_acc[tag] = raw_acc.get(tag, 0.0) + psnr_sum(image, high)
                    if inside is not None:
                        error, pixels = masked_error(roundtrip, high, inside)
                        vae_face_acc[0] += error
                        vae_face_acc[1] += pixels
                        error, pixels = masked_error(roundtrip, high, ~inside)
                        vae_face_acc[2] += error
                        vae_face_acc[3] += pixels
                if image_dir is not None:
                    for offset in range(frames):
                        save_val_images(
                            image_dir, step, saved + offset,
                            low[offset], pred[offset], high[offset], digits,
                        )
                        if roundtrip is not None:
                            folder = image_dir / f"{saved + offset:0{digits}d}"
                            save_still(folder, "vae", roundtrip[offset])
                    saved += frames
                count += frames
        run.model.train()
        extras = {tag: value / count for tag, value in raw_acc.items()}
        extras["val_latent/z_std"] = z_std_acc / count
        if detector is not None:
            total_pixels = face_acc[1] + face_acc[3]
            extras["val_raw/face_fraction"] = face_acc[1] / total_pixels if total_pixels else 0.0
            found = [
                ("val/psnr_face", pooled_psnr(face_acc[0], face_acc[1])),
                ("val/psnr_nonface", pooled_psnr(face_acc[2], face_acc[3])),
            ]
            if baselines:
                found += [
                    ("val_ref/vae_psnr_face", pooled_psnr(vae_face_acc[0], vae_face_acc[1])),
                    ("val_ref/vae_psnr_nonface", pooled_psnr(vae_face_acc[2], vae_face_acc[3])),
                ]
            for tag, value in found:
                if value is not None:
                    extras[tag] = value
        for name, value in weighted_acc.items():
            if name not in ("pixel", "vgg"):
                extras[f"val/{name}"] = value / count
        if baselines:
            pending["baselines"] = False
            print(
                f"step {step:06d}  baselines on {count} val frames  psnr  "
                f"bicubic {extras['val_ref/bicubic_psnr']:.2f}  "
                f"bilinear {extras['val_ref/bilinear_psnr']:.2f}  "
                f"vae2 roundtrip {extras['val_ref/vae_psnr']:.2f}"
                + (
                    f"  face {extras['val_ref/vae_psnr_face']:.2f}"
                    if "val_ref/vae_psnr_face" in extras else ""
                )
                + (
                    f"  nonface {extras['val_ref/vae_psnr_nonface']:.2f}"
                    if "val_ref/vae_psnr_nonface" in extras else ""
                )
                + (
                    f"  faces {extras['val_raw/face_fraction']:.3f}"
                    if "val_raw/face_fraction" in extras else ""
                ),
                flush=True,
            )
        print(f"step {step:06d}  validation {count} frames  {time.perf_counter() - started:.1f}s", flush=True)
        pixel = weighted_acc["pixel"] / count if "pixel" in weighted_acc else None
        vgg = weighted_acc["vgg"] / count if "vgg" in weighted_acc else None
        return EvalResult(loss_acc / count, (pixel, vgg, None, None), None, None, None, extras)

    return make_val_specs, make_loader, one_step, evaluate_split


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
    elif cfg.arch == LATENT_ARCH:
        def load():
            return load_sr(cfg)

        hooks = latent_hooks
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
