"""Time the MLX 3-frame latent_up on the GPU. Only with the Whoodeo app closed.

    python -m whoodeo.mlxport.bench weights/mlx/align-STEP --size 1280x720 --iters 20

Reports ms per frame for SPyNet (both neighbors) plus warp, for
latent_up + decoder, and for the whole path, each after warm-up runs.
`--compile` wraps the stages in mx.compile.
"""

import argparse
import time

import mlx.core as mx


def _time(fn, iters, warmup):
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - start) / iters * 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="latent_up and decoder in float16, as the app")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    from whoodeo.mlxport.model import AlignModel

    model = AlignModel.load(args.weights, mx.float16 if args.fp16 else mx.float32)
    width, height = (int(v) for v in args.size.lower().split("x"))
    mx.random.seed(42)
    low = mx.random.uniform(shape=(1, height, width, 3 * model.in_frames))
    condition = mx.zeros((1, model.condition_size))
    condition[0, 0] = 1.0

    def flows():
        frames = [low[..., 3 * i:3 * i + 3] for i in range(model.in_frames)]
        supp = mx.concatenate([frames[0], frames[2]], axis=0)
        return model.spynet(mx.concatenate([frames[1]] * 2, axis=0), supp)

    def full():
        return model(low, condition)

    if args.compile:
        flows, full = mx.compile(flows), mx.compile(full)
    print(f"MLX {mx.__version__} on {mx.default_device()}  {width}x{height}  compile {args.compile}  fp16 {args.fp16}")
    print(f"  SPyNet x2                 {_time(flows, args.iters, args.warmup):8.1f} ms")
    print(f"  whole path                {_time(full, args.iters, args.warmup):8.1f} ms")
    print(f"  peak memory               {mx.get_peak_memory() / 2**30:8.2f} GiB")


if __name__ == "__main__":
    main()
