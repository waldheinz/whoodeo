"""Latent robustness of a saved vae2 checkpoint, on the validation frames of its run.

    uv run --no-sync python -m whoodeo.eval_latent <checkpoint> [--loss configs/vae2d-spark.yaml]

The run's config.yaml gives the data, the seed, the batch, and val_count, so the
frames are the ones that run validated on. The loss section comes from --loss,
so runs trained under different weights compare under one loss. Prints the validation
loss, val/psnr, val/noise_psnr_<s> for s 0.1, 0.3, 0.5 (decode of μ + s * std_c
* N(0, 1), fixed seed, std_c the per-channel std of the validation μ),
val/scale_eq_psnr, and the per-channel latent table.
"""

import argparse
import random
import tempfile
from pathlib import Path

import torch
import yaml

from whoodeo.checkpoint import build_generator, load_generator, run_directory
from whoodeo.config import load_config
from whoodeo.models.vae import latent_table
from whoodeo.objective import Objective
from whoodeo.train import evaluate_vae, get_device, load_vae, make_vae_specs

DEFAULT_LOSS = Path(__file__).resolve().parents[2] / "configs" / "vae2d-spark.yaml"


def recipe(run_config, loss_config):
    """The run's file with its model section dropped and the loss from `loss_config`."""
    data = yaml.safe_load(Path(run_config).read_text())
    loss = yaml.safe_load(Path(loss_config).read_text())["loss"]
    data.pop("model", None)
    data.pop("out", None)
    data["loss"] = loss
    return data


def main():
    parser = argparse.ArgumentParser(description="latent robustness of a vae2 checkpoint")
    parser.add_argument("checkpoint", type=Path, help="run directory, checkpoint, or model.pt")
    parser.add_argument("--loss", type=Path, default=DEFAULT_LOSS, help="config whose loss: section is used")
    args = parser.parse_args()
    spec = load_generator(args.checkpoint)
    if spec.architecture.get("arch") != "vae2":
        raise SystemExit(f"{spec.path}: eval_latent reads vae2, not {spec.architecture.get('arch')}")
    run_config = run_directory(spec.path) / "config.yaml"
    if not run_config.is_file():
        raise SystemExit(f"no run config: {run_config}")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "eval.yaml"
        path.write_text(yaml.safe_dump(recipe(run_config, args.loss)))
        cfg = load_config(path, spec.architecture, spec.path)
    device = get_device()
    model, label, _frames, step, _path = build_generator(spec, device)
    print(f"{label}  step {step}", flush=True)
    print(f"data from {run_config}  loss from {args.loss}", flush=True)
    films = load_vae(cfg)
    rng = random.Random(cfg.seed)
    specs = []
    while len(specs) < cfg.val_count:
        need = min(cfg.batch, cfg.val_count - len(specs))
        specs.extend(make_vae_specs(rng, films, "val", need, cfg.crop))
    objective = Objective(cfg, device)
    result = evaluate_vae(model, films, specs, cfg.crop, cfg.batch, device, cfg, objective)
    model.eval()
    extras = result.extras or {}
    print(f"frames {len(specs)}  loss/val {result.loss:.6f}", flush=True)
    for tag in sorted(extras):
        if "psnr" in tag:
            print(f"{tag} {extras[tag]:.3f}", flush=True)
    if result.kl is not None:
        print(f"val/kl {result.kl:.6f}", flush=True)
    if result.latent is not None:
        print(latent_table(result.latent), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
