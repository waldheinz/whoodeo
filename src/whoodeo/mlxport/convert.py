"""Write a 3-frame latent_up checkpoint as MLX weights: <out>.safetensors + <out>.json.

    python -m whoodeo.mlxport.convert RUN/checkpoints/STEP weights/mlx/align-STEP
    python -m whoodeo.mlxport.convert RUN/checkpoints/STEP --app

An `out` ending in .mlxmodel is a folder with model.safetensors and
config.json, the format the Mac app reads. `--app` writes that folder into
~/Library/Application Support/Whoodeo/Models under the name an export of
the same checkpoint gets, with .mlxmodel instead of .mlpackage.

Loads the same model whoodeo-apply builds (latent_up with its frozen SPyNet
inside, and its vae2) on the CPU. Conv kernels go from torch
(O, I, kH, kW) to MLX (O, kH, kW, I); everything else keeps its name.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_reference(checkpoint, vae=None, codec=None):
    """(LatentUpReference, latent_up, vae2, step) as whoodeo-apply loads them, on the CPU."""
    import torch

    from whoodeo.apply import load_checkpoint

    torch.set_grad_enabled(False)
    reference, _label, in_frames, step, path = load_checkpoint(checkpoint, torch.device("cpu"), vae, codec)
    if in_frames != 3 or getattr(reference.model, "align", None) is None:
        raise SystemExit(f"{checkpoint}: not a 3-frame latent_up with feature alignment")
    return reference, reference.model, reference.vae, step


def _mx(tensor):
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return mx.array(np.ascontiguousarray(array))


def convert(checkpoint, out, vae=None):
    _ref, model, vae_model, step = load_reference(checkpoint, vae)
    params = {}
    for key, value in model.state_dict().items():
        params[key] = _mx(value)
    for key, value in vae_model.decoder.state_dict().items():
        params["decoder." + key] = _mx(value)
    params["vae.latent_scale"] = _mx(vae_model.latent_scale)
    params["vae.latent_shift"] = _mx(vae_model.latent_shift)
    from whoodeo.models.latent_up import block_widths

    config = {
        "arch": "latent_up",
        "align": "feature",
        "step": step,
        "checkpoint": str(checkpoint),
        "activation": model.activation,
        "in_frames": model.in_frames,
        "condition": model.condition,
        "condition_size": model.condition_size,
        "encoder_blocks": list(model.encoder_blocks),
        "blocks": model.blocks,
        "block_widths": block_widths(model.encoder),
        "latent": model.latent,
        "vae_activation": vae_model.activation,
        "vae_blocks": len(vae_model.decoder.mid),
        "decoder_blocks_run": [len(stage.blocks) for stage in vae_model.decoder.ups],
        # Precision the app computes latent_up and the decoder in, as the Core ML
        # export did; SPyNet and its flow stay fp32. The weights are stored fp32.
        "compute": "float16",
    }
    if config["vae_activation"] != config["activation"]:
        raise SystemExit("the MLX port assumes one activation for latent_up and vae2")
    if out is None:
        run = Path(checkpoint).resolve().parent.parent.name
        name = (
            f"{run}-latent_up-{model.blocks}x{model.channels}-step{step:06d}"
            f"-{model.in_frames}f-hd-align" + ("-cond" if model.condition else "") + ".mlxmodel"
        )
        out = Path.home() / "Library/Application Support/Whoodeo/Models" / name
    out = Path(out)
    if out.suffix == ".mlxmodel":
        out.mkdir(parents=True, exist_ok=True)
        weights, meta = out / "model.safetensors", out / "config.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        weights, meta = out.with_suffix(".safetensors"), out.with_suffix(".json")
    mx.save_safetensors(str(weights), params)
    meta.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {weights}  {len(params)} tensors  step {step}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out", nargs="?", help="path without suffix, or a .mlxmodel folder")
    parser.add_argument("--app", action="store_true", help="write the .mlxmodel into the app's model folder")
    parser.add_argument("--vae", default=None)
    args = parser.parse_args(argv)
    if (args.out is None) == (not args.app):
        parser.error("give out or --app")
    convert(args.checkpoint, args.out, args.vae)


if __name__ == "__main__":
    main()
