"""Export frozen training assets for the Swift MLX trainer.

    python -m whoodeo.mlxport.export_train

Writes into ~/Library/Application Support/Whoodeo/Train/:
  vae2-044750.safetensors + vae2-044750.json  (encoder, decoder, shift, scale)
  vgg19.safetensors                           (VGG19 features up to relu3_4)
  spynet.safetensors                          (official SPyNet, MLX layout, align.spynet.* keys)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

APP_TRAIN = Path.home() / "Library/Application Support/Whoodeo/Train"
VAE_CANDIDATES = [
    Path("/Users/trem/storinator/home/whoodeo/runs/train-20261119-180935/checkpoints/044750"),
    Path("/Volumes/Black Elem/whoodeo/runs/train-20261119-180935/checkpoints/044750"),
]
SPYNET_PTH = Path.home() / "Arbeitsplatz/Meins/whoodeo/weights/spynet_20210409-c6c1bd09.pth"
# Fallback: copy SPyNet out of an existing app package.
SPYNET_MLXMODEL = Path.home() / (
    "Library/Application Support/Whoodeo/Models/"
    "train-20261124-100406-latent_up-12x128-step006000-3f-hd-align-cond.mlxmodel/"
    "model.safetensors"
)


def _mx(tensor):
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return mx.array(np.ascontiguousarray(array))


def export_vae(out_dir: Path) -> Path:
    import torch

    from whoodeo.checkpoint import build_generator, load_generator

    path = next((p for p in VAE_CANDIDATES if (p / "model.pt").is_file()), None)
    if path is None:
        raise SystemExit("no vae2 checkpoint 044750 found (storinator or Black Elem)")
    device = torch.device("cpu")
    spec = load_generator(path)
    vae, label, _frames, step, resolved = build_generator(spec, device)
    vae.eval()
    params = {}
    for key, value in vae.encoder.state_dict().items():
        params["encoder." + key] = _mx(value)
    for key, value in vae.decoder.state_dict().items():
        params["decoder." + key] = _mx(value)
    params["vae.latent_scale"] = _mx(vae.latent_scale)
    params["vae.latent_shift"] = _mx(vae.latent_shift)
    weights = out_dir / "vae2-044750.safetensors"
    meta = out_dir / "vae2-044750.json"
    mx.save_safetensors(str(weights), params)
    config = {
        "arch": "vae2",
        "step": step,
        "checkpoint": str(resolved),
        "label": label,
        "activation": vae.activation,
        "channels": vae.channels,
        "blocks": len(vae.decoder.mid),
        "latent": vae.latent,
        "factor": vae.factor,
        "encoder_blocks": list(vae.encoder_blocks),
        "decoder_blocks": list(vae.decoder_blocks),
        "widths": [w // vae.channels for w in vae.widths],
    }
    meta.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {weights}  {len(params)} tensors  step {step}  {label}")
    return weights


def export_vgg(out_dir: Path) -> Path:
    import torch
    from torchvision.models import VGG19_Weights, vgg19

    features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: 17 + 1].eval()
    params = {}
    for name, value in features.state_dict().items():
        # torchvision keys are "0.weight"; store as "features.0.weight"
        params["features." + name] = _mx(value)
    weights = out_dir / "vgg19.safetensors"
    mx.save_safetensors(str(weights), params)
    print(f"wrote {weights}  {len(params)} tensors")
    return weights


def export_spynet(out_dir: Path) -> Path:
    out = out_dir / "spynet.safetensors"
    if SPYNET_PTH.is_file():
        import torch

        from whoodeo.spynet import SPyNet, load_weights

        net = SPyNet()
        load_weights(net, SPYNET_PTH)
        params = {("align.spynet." + key): _mx(value) for key, value in net.state_dict().items()}
        mx.save_safetensors(str(out), params)
        print(f"wrote {out}  {len(params)} tensors from {SPYNET_PTH}")
        return out
    if SPYNET_MLXMODEL.is_file():
        loaded = mx.load(str(SPYNET_MLXMODEL))
        params = {k: v for k, v in loaded.items() if k.startswith("align.spynet.")}
        mx.save_safetensors(str(out), params)
        print(f"wrote {out}  {len(params)} tensors from {SPYNET_MLXMODEL}")
        return out
    raise SystemExit(f"no SPyNet weights at {SPYNET_PTH}")


def main():
    APP_TRAIN.mkdir(parents=True, exist_ok=True)
    export_vae(APP_TRAIN)
    export_vgg(APP_TRAIN)
    export_spynet(APP_TRAIN)


if __name__ == "__main__":
    main()
