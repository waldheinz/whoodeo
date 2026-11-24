"""The Mac app's MLX path (Whoodeo --check) against PyTorch and the Python MLX port.

    python -m whoodeo.mlxport.check_app RUN/checkpoints/STEP MODEL.mlxmodel \
        [--video low.mkv] [--sizes 384x288 200x150]

`--check` feeds the unknown codec (all-zero condition), so the references
do too. Inputs: the shifted smooth test picture of check.py at each size,
and with --video three consecutive frames of that video at its own size.
Tensor files go to mlx-check/ in the repo.
"""

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "macos/Whoodeo/build/Release/Whoodeo.app/Contents/MacOS/Whoodeo"
WORK = REPO / "mlx-check"


def write_f32(path, array):
    array = np.ascontiguousarray(array, dtype="<f4")
    path.write_bytes(struct.pack("<4i", *array.shape) + array.tobytes())


def read_f32(path):
    data = path.read_bytes()
    shape = struct.unpack("<4i", data[:16])
    return np.frombuffer(data[16:], dtype="<f4").reshape(shape)


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def video_frames(path, start=30):
    import av

    with av.open(str(path)) as container:
        frames = []
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == 3:
                break
    stack = np.concatenate([f.transpose(2, 0, 1) for f in frames], 0)[None].astype(np.float32) / 255
    return torch.from_numpy(stack)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("model", help="the .mlxmodel folder")
    parser.add_argument("--video")
    parser.add_argument("--sizes", nargs="*", default=["384x288", "200x150"])
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    import mlx.core as mx

    from whoodeo.mlxport.check import test_frames
    from whoodeo.mlxport.convert import load_reference
    from whoodeo.mlxport.model import AlignModel

    reference, model, vae, spynet, _step = load_reference(args.checkpoint)
    reference.inner.condition = torch.zeros(1, model.condition_size)
    import json, os

    compute = os.environ.get("WHOODEO_MLX_PRECISION") or json.loads((Path(args.model) / "config.json").read_text()).get("compute", "float32")
    ours = AlignModel.load(args.model, mx.float16 if compute == "float16" else mx.float32)
    print(f"app precision (latent_up + decoder): {compute}")
    WORK.mkdir(exist_ok=True)
    cases = []
    for size in args.sizes:
        width, height = (int(v) for v in size.lower().split("x"))
        cases.append((f"test {width}x{height}", test_frames(height, width)))
    if args.video:
        clip = video_frames(args.video)
        cases.append((f"video {clip.shape[3]}x{clip.shape[2]}", clip))
    for name, low in cases:
        tag = name.replace(" ", "-")
        src, dst = WORK / f"{tag}-in.f32", WORK / f"{tag}-app.f32"
        write_f32(src, low.numpy())
        out = subprocess.run([str(APP), "--check", args.model, str(src), str(dst)], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"app failed: {out.stderr or out.stdout}")
        app = read_f32(dst)
        torch_out = reference(low).numpy()
        py_mlx = np.array(ours(mx.array(low.numpy().transpose(0, 2, 3, 1)), mx.zeros((1, model.condition_size))))
        py_mlx = py_mlx.transpose(0, 3, 1, 2)
        print(
            f"{name:18s} app vs PyTorch {psnr(app, torch_out):6.1f} dB (max {np.abs(app - torch_out).max():.1e})"
            f"  app vs Python-MLX {psnr(app, py_mlx):6.1f} dB  ({out.stdout.strip()} incl. first-run setup)"
        )


if __name__ == "__main__":
    main()
