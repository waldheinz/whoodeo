# The shift model

`arch: shift` is a 2× video upscaler. It looks at a few low-resolution frames and predicts the high-resolution frame in the middle. The picture it returns is the middle frame, enlarged with bilinear interpolation, plus a correction. That correction starts at zero, so an untrained model is just the bilinear enlargement.

`configs/shift.yaml` is a worked example. The keys under `model:` are the ones this section is about.

## What goes in

`in_frames` is an odd number, at least 3. With 3, the input is the frame before, the frame being reconstructed, and the frame after. With 5 it is two frames on either side of the middle. Only the middle frame is enlarged into the base image. The others can only contribute to the correction.

## Stem: one look at each frame, before any shifting

The stem runs on each frame alone, and every frame uses the same weights. Nothing is shifted yet, and the frames are not mixed.

It is one 3×3 convolution, from the 3 RGB values to `channels` numbers per pixel, followed by `stem` residual blocks. A residual block is two 3×3 convolutions whose result is added back onto their input, so the block only has to learn a change. `stem: 1` is that first convolution plus one residual block. `stem: 0` is only the convolution.

There is one stem. The possible shifts do not each get a stem, and the frames do not each get their own either. The stem is applied three times for `in_frames: 3`, always with the same filters.

## Radius: where a neighbor is allowed to have moved

The search happens after the stem, on those feature maps, at the resolution of the low frame. It does not slide the original RGB pictures.

The middle frame stays put. Each other frame is lined up onto it. `radius` is how far that search may look, in low-resolution pixels, left, right, up, and down. `radius: 4` tries every whole-pixel offset from −4 to +4 in both directions: a 9×9 window, 81 placements. It is the whole window, not one hop of 4 pixels. Four low-resolution pixels are eight pixels in the 2× output.

At every pixel the model scores each placement by how similar the middle features are to the neighbor features sitting at that offset. The best placement can differ from pixel to pixel, so one part of the frame can move right while another stays still. `sharpness` multiplies those scores before they are turned into a mix. It stays at the value in the file. A high value makes the best placement win cleanly. A low value blends several placements.

The aligned neighbor is that mix of shifted feature maps. A mix of two neighboring offsets is how a movement of half a pixel shows up. Anything past `radius` is invisible to the search.

`reject: true` adds one extra option: take nothing from this neighbor. A similarity of zero starts tied with that option, so a cut, or motion larger than the window, can drop the neighbor instead of dragging the nearest bad placement in.

The same search is used for every neighbor. The previous frame and the next frame do not have separate shifts or separate filters.

## Blocks and channels: the network after the frames are lined up

The lined-up frames are stacked and mixed back down to `channels` by one 3×3 convolution. Then the trunk runs: `blocks` residual blocks, the same kind as in the stem. `blocks: 8` means eight of those, after the search. It does not repeat the stem and it does not repeat the shifts.

`channels` is the width of the whole model. After the stem, every pixel is described by that many numbers instead of 3. The search compares those numbers, and the trunk processes them. `channels: 64` is a 64-number description. More channels can tell edges, texture, and flat skin apart more finely, and they cost memory and time throughout. More blocks make the trunk deeper, so it can turn the lined-up frames into a more complicated correction. They do not widen the search window. Only `radius` does that.

The last layer turns the trunk's features into the 2× correction and starts at zero. The neighbors influence the image only through that correction.

# The pyramid model

`arch: pyramid` lines the neighbor frames up by searching at a few scales, then moves each neighbor in one step. The rest of the network is the shift model. The picture is still the middle frame enlarged with bilinear interpolation, plus a correction that starts at zero.

`configs/pyramid.yaml` is a worked example. `blocks`, `channels`, `in_frames`, and `stem` mean what they mean for the shift model.

## Levels and radius

The line-up is a flow: two numbers at every pixel, how far to read the neighbor from, to the right and down. The guess at one scale is at most `radius` pixels of that scale, so it cannot run away. Every guess starts at zero, and the neighbors all share the same networks.

The smallest scale does not guess from the raw pictures. It scores every integer offset inside the radius by how well the neighbor points the same way as the middle, and a small network reads those scores. Training also asks that flow to land on the offset that scored best. On a smooth frame the picture itself has almost no slope until the guess is already close, but the best offset is visible on the first step.

`levels` is how many scales the guess uses, counting the full low-resolution frame. `levels: 1` has no smaller scale, so only the correction below runs. `levels: 3` starts at a quarter of the width and height, then half, then the full frame. The smallest scale guesses first. That flow is enlarged to the next scale and then held fixed, the neighbor is moved by it, and that scale's own network, which sees the two pictures, adds a correction of at most `radius` pixels. Holding the enlarged flow fixed keeps a fine score from pulling the coarse guess off a shift it already found. One pixel on the quarter-size frame is four low-resolution pixels, so with `levels: 3` and `radius: 4` the first guess reaches sixteen low-resolution pixels, and the finer scales correct what is left.

While training, every scale is also scored on whether the moved neighbor points the same way as the middle. That score is added to the loss. It does not train the stem, so the stem cannot satisfy it by forgetting the picture. It gives the flow a reason to move before the mix below has opened.

A movement past that first window is hard to see. The gate can drop the pixel.

## The gate

A moved neighbor can still be dropped, pixel by pixel. The gate compares the direction of the middle features with the direction of the moved neighbor. A match starts open. The opposite direction starts closed. An unrelated neighbor starts half open. A convolution can learn a different choice later. That convolution starts at zero, so at the beginning only the comparison decides.

## The mix

The middle features and the gated neighbors are stacked and mixed back down to `channels` by one 3×3 convolution. The weights that read a neighbor start at zero, so an untrained model does not use the neighbors. The trunk then runs for `blocks` residual blocks, and the last layer turns that into the 2× correction. That layer starts at zero, so the picture starts as the bilinear enlargement.

# The vote model

`arch: vote` is a 2× upscaler derived from EDSR. It reads the frame before, the frame being reconstructed, and the frame after, and predicts the middle frame. There is no `in_frames` key. The count is three.

The stem runs on each frame alone, and every frame uses the same weights. It is one 3×3 convolution from the 3 RGB values to `channels`, followed by `stem` residual blocks. `stem: 0` is only the convolution. `stem: 2` is the convolution plus two residual blocks.

The vote sits on the difference between a neighbor and the middle. The same weights are used for the previous frame and the next frame. The path is one 3×3 convolution, a ReLU, `vote` residual blocks, and a closing 3×3 convolution. The raw difference is not added back on, so the decision cannot stay linear. The two results are added onto the middle features. `vote: 0` is the convolution, the ReLU, and the closing convolution.

`blocks` is the trunk after the vote: that many residual blocks, then one convolution. The long skip adds the middle stem output back on after the trunk, so a neighbor reaches the picture only through the trunk. `res_scale` multiplies the residual inside each block, in the stem, in the vote, and in the trunk, the same way it does in EDSR. The mean subtracted at the input and added at the output is the DIV2K mean, and the tail is the EDSR 2× upsampler.

`configs/vote.yaml` is a worked example. The x264 pair filter matches the EDSR baseline recipe. The VGG layers are `relu2_2` and `relu3_4`. `blocks` there is the trunk only, not the stem and not the vote.

While a vote run trains, TensorBoard records three neighbor numbers. `train_neighbor/weight` is the mean absolute value of the kernels in the shared vote. `train_neighbor/grad` is the mean absolute gradient of those kernels divided by the same quantity on the first convolution, so a value near zero means the loss is not asking the vote to change. `val_neighbor/image` is written when validation runs. It is the mean absolute change in the picture when both neighbors are replaced by a copy of the middle frame.

# The SwinIR models

`arch: swinir`, `arch: swinir_light`, and `arch: swinir_real` read the low-resolution frame and predict the 2× picture. The three names share one trunk. `configs/swinir.yaml` is the classical recipe and says how to switch to the other two.

The trunk is a stack of residual Swin Transformer blocks. Attention looks at an 8×8 window, and every other layer shifts that window. A side that is not a multiple of 8 is padded by reflection, and that extra strip is cut off the output.

`blocks` is how many of those residual blocks there are. Each one contains six Swin layers. `channels` is the width of the features inside them. It has to be divisible by 6, so each of the six attention heads gets an equal share.

`swinir` is the classical head: a convolution to 64 channels, a pixel shuffle, and a 3×3 back to RGB. The recipe uses 6 blocks and 180 channels.

`swinir_light` is the small head: one convolution straight into the pixel shuffle. It is meant to be run as 4 blocks and 60 channels.

`swinir_real` keeps the classical trunk and enlarges with a nearest-neighbor step followed by convolutions.

The last convolution keeps PyTorch's ordinary initialization. Before the trunk runs, the DIV2K mean is subtracted, and that mean is added back onto the picture the network predicts.

# The vae model

`arch: vae` reconstructs a master frame. It does not read a low variant. The encoder is a stack of residual blocks and stride-2 convolutions. The last convolution yields the mean and the log variance of a diagonal Gaussian. `latent` is how many channels that Gaussian has. `factor` is the spatial shrink, a power of two from 2 to 16, so factor 8 turns a 256×256 window into a 32×32 latent. The decoder starts from that latent alone. There is no skip from the encoder.

The loss is the sum of `weight` times each raw term. The weights are multipliers, not shares, and need not add up to 1. VGG has no term weight and is weighed per layer under `layers`, for example `layers: {relu2_2: 0.230, relu3_4: 0.155}`. `beta` multiplies the mean KL divergence. The mean runs over the batch, the latent grid, and the channels, so a beta near 1 is a real weight. The step loss adds that term to the weighted pixel, VGG, and color objective when those terms are present. The validation loss the plateau reads is that objective without the KL. `beta` is not part of the architecture key, so a `--resume` or `--model` can change it from the file. TensorBoard records the unweighted KL as `train_raw/kl` and `val_raw/kl`. `train_latent/mu_mean`, `mu_std`, `sigma_mean` and `spread` describe the code of the batch, with the same four under `val_latent/` for the held-out frames, plus `active` and `kl_min`. Each validation also writes a per-channel table (`μ` mean and std, `σ` mean, SNR, KL) to the text tab as `val_latent/channels`. `spread` is the square root of var(μ) + mean(σ²) and sits near 1 when the code matches the unit Gaussian. The validation histogram is `val_latent/mu`.

A side that `factor` does not divide is padded on the bottom and the right, and the reconstruction is cut back to the original size. Training draws a latent sample. Evaluation uses the mean, so a validation image stays put between two passes.

`crop` is optional. Left out, each sample is the whole master frame, and the batch stays inside one master size. Set to a positive multiple of `factor`, every sample is that window. A training window is placed at random. A validation window is the center, and it stays there, so `recon-<step>.png` can be compared across steps. `master.png` is written once. The live page shows a draw from the unit Gaussian (`sample`), the reconstruction (`recon`), and the master.

# The vae2 model

`arch: vae2` reconstructs a master frame like `vae`, with a network shaped for the Neural Engine. The layers are 3×3 convolutions, an activation, additions, and nearest-neighbor doubling. There is no normalization and no attention. `activation` under `model:` is `silu` (the default, and what a checkpoint without the key means) or `relu`; it is part of the saved architecture, so loading, resuming, and `whoodeo-coreml` pick it up, and a resume refuses the other one. `channels` is the width at full resolution. By default the width doubles at half resolution and is four times `channels` from a quarter down, so `configs/vae2.yaml` runs 64, 128, 256, 256 from the frame to the latent. Full resolution has one residual block, each level below it two. `blocks` is the number of residual blocks at the latent, with dilations 1, 2, and 4 in turn. Encoder and decoder mirror each other. The layout can be set apart: `encoder_blocks` and `decoder_blocks` name the residual count per level above the latent (full resolution first), and `widths` names a multiple of `channels` per level including the latent; `configs/vae2.yaml` uses `encoder_blocks: [1, 2, 2]`, `decoder_blocks: [2, 3, 2]`, `widths: [1, 2, 4, 4]`. `decoder_widths` can give the decoder its own multiples. The last convolution of every residual branch starts at zero. The default layout has 22.2M parameters, and the decoder costs about 1.2 TFLOP for a 720p frame, against 3.2 for `vae` 4×128.

`decode(latent, clamp=True)` limits the frame to [0, 1]. Training compares the unclamped output. `beta` 1e-4 keeps the KL near one part in a thousand of the objective; like `vae`, beta is not part of the architecture key, so it can change on `--resume`. Each validation stores per-channel `latent_shift` and `latent_scale` in the model: (μ − shift) × scale has mean 0 and std 1 in each channel. A model that works on latents should see that scaled code and hand z / scale + shift back to the decoder. TensorBoard logs the mean scale as `val_latent/scale`. The mean shift is `val_latent/mu_mean`.

The `color` loss term is L1 between average-pooled frames, the chroma channels weighted twice. `space: ycbcr`, the default, pools the gamma sRGB frames and compares BT.709 Y, Cb, Cr. `space: oklab` pools in linear light and compares Oklab L, a, b. `pool` is the pooling factor, 8 by default. Both are part of the loss recipe. The term is for color drift in flat areas, and `vae` takes it too. The current vae2 weights are pixel 1.0, VGG `relu2_2` 0.230 / `relu3_4` 0.155, and color 5.5 in Oklab with pool 16.

Two terms can take the place of pixel and color. `oklab` is L1 in Oklab at full resolution: gamma sRGB to linear light to LMS, the cube root, and L, a, b, each weighed by `channels: [L, a, b]` (default `[1, 3, 3]`; on vae2d mean |ΔL| is about 3.2 times |Δa| or |Δb|, so at 3 each channel weighs about the same). Below (6/29)³ in LMS the cube root becomes its tangent line, as in CIE L\*a\*b\*, so the slope near black stays at 7.8 and an overshoot below black keeps a gradient. `ms_ssim` is 1 − MS-SSIM (Wang 2003: Gaussian window 11, σ 1.5, the five standard scale weights) on Oklab L, `scales` 1 to 5 (default 5); scales whose shorter side falls below the window are dropped and the rest renormalized, so the pooled `scale_eq` frames work too. Both log `train_raw/`, `val_raw/` and the weighted `train/`, `val/` values like the other terms. `configs/vae2f-spark.yaml` (and vae2g) uses oklab 4.5 with `channels: [1, 1, 1]` (raw 0.00313 on vae2d, the mean of the channels) and ms_ssim 1 beside the VGG weights, calibrated on 32 val frames so that they hold the share against VGG that pixel + color held: 23.3% against 23.4% on vae2d step 15755, 85% against 86% at step 0, but 50% against 41% at vae2e step 250, where 1 − MS-SSIM is still large.

`posterior: fixed` makes the encoder output only μ: no log-variance, no sample, and no KL. Leave it out for the usual diagonal Gaussian.

`latent_noise: {tau: ...}` adds per-channel-relative noise to the latent in training, after any sample. Each frame draws s from |N(0, τ²)| and gets s × std_c × N(0, 1), where std_c is a running std of μ per channel. It sits outside the loss recipe, so a resume may change it; `tau: 0` turns it off for a clean final fine-tune. Validation still decodes μ.

Two more terms stand beside the objective, in the step loss only: `mu_l2` is `weight` times mean(μ²), against scale drift, and `scale_eq` picks one of its `factors` (for example `[2, 4]`) each step, decodes the average-pooled latent, and compares it with the pooled frame under the same pixel, VGG, color, oklab, and ms_ssim weights.

`faces: {weight, score, expand, feather}` (vae, vae2, and latent_up) weighs the image terms up inside faces. OpenCV YuNet (`cv2.FaceDetectorYN`, the `opencv-python-headless` dependency) runs on the CPU on every frame exactly as the batch hands it to the model, after any crop: in the loader thread for training, on the main thread for validation, on the frame scaled to at most 640 pixels on its longer side, about 6 ms a frame on the Spark. There are no cached boxes. The model file `face_detection_yunet_2023mar.onnx` (opencv_zoo, MIT) sits in `~/.cache/whoodeo/` (`$XDG_CACHE_HOME/whoodeo`); a missing file is downloaded on first use and its SHA-256 checked. Boxes at `score` (default 0.6) or above grow by `expand` (1.2) about their center; the soft mask is that box blurred by a Gaussian of `feather` (0.1) times its shorter side, the maximum where faces overlap. The weight map is 1 + (`weight` − 1) × mask, divided by its mean per frame: every frame keeps a mean weight of 1, so the loss scale and the term balance stay those of the recipe, a frame without faces counts as before, and in a frame with faces the rest gives up what the faces gain. `oklab` multiplies its L1 by the map, `ms_ssim` takes the weighted mean of the SSIM and contrast-structure maps at each scale (the map pooled with the frames), and `vgg` multiplies each layer's L1 by the map area-pooled to that layer; `pixel`, `color`, `scale_eq`, `mu_l2`, and the KL stay unweighted. Validation weighs its objective the same way and logs `val/psnr_face` and `val/psnr_nonface`, the PSNR pooled over all face pixels (mask ≥ 0.5) and all others, beside `train_raw/face_fraction` and `val_raw/face_fraction`. The live page draws the boxes on `recon` and `master`. For latent_up see its section below. `faces` stays outside the loss recipe, like `latent_noise`; weight 1 or no `faces:` runs no detector at all.

Each vae2 validation also logs `val_noise/psnr_{0.1,0.3,0.5}` (PSNR of decode(μ + s × std_c × N(0, 1)) with a fixed seed) and `val_noise/scale_eq_psnr` (decode of the 2×2 pooled μ against the 2×2 pooled frame). The per-channel SNR sits in the `val_latent/channels` table. `uv run --no-sync python -m whoodeo.eval_latent <checkpoint> [--loss cfg]` prints the same numbers for a saved checkpoint.

`configs/vae2.yaml` is the current recipe: the layout above with 16 latent channels, `posterior: fixed`, latent noise with tau 0.4, Oklab color, `mu_l2` at 5e-4, and `scale_eq` at 0.25, at batch 2. `configs/vae2d-spark.yaml` is the same recipe at batch 6 for the DGX Spark, and its comments say where each choice comes from. `configs/vae2e-spark.yaml` is vae2d with `activation: relu` and nothing else changed, a fresh run to see what relu costs in quality; relu is for the Neural Engine, see the Core ML section below. A 40-step smoke run on the Spark (21.10.2026) took 2.3–2.6 s/step at 480p and 5.2–5.9 s at 720p, left about 60 GB RAM available, and went from val/psnr 5.9 to 19.0.

A vae2 may add a `gan` term with the U-Net discriminator. Its weight is not a share of the objective. The generator term is the latent diffusion adaptive weight, the gradient norm of the objective over that of the GAN loss at the last decoder convolution, times `weight`. The discriminator trains from the first step. The generator term joins at `start`. The validation objective leaves the GAN out. `freeze_encoder: true` trains the decoder alone. `configs/vae2-gan.yaml` combines the two as a fine-tune of a vae2 run:

```
uv run whoodeo-train configs/vae2-gan.yaml --model <run>/latest
```

`whoodeo-coreml` turns a vae2 checkpoint into its decoder: an fp16 latent in, the clamped fp16 frame out. The input takes the latent sizes 60×107, 90×160, and 135×240, which are 480p, 720p, and 1080p at factor 8. `--latent 90x160` gives one fixed size instead. The metadata carries the per-channel latent shift and scale.

# The latent_up model

`arch: latent_up` is a 2× upscaler that predicts the code of a frozen vae2 instead of pixels. The low frame is h × w and its master 2h × 2w. A vae2 at factor 8 codes the master at h/4 × w/4, so the network maps the low RGB frame to that grid: the normalized code z = (μ − shift) × scale, which has mean 0 and std 1 per channel. The frozen vae2 decoder turns z / scale + shift into the 2× picture. A side that 4 does not divide is padded like vae2 pads, and the picture is cut back to 2h × 2w.

`vae` under `model:` names the vae2, a run directory (its latest checkpoint, as `--model` reads it), a checkpoint, or a model.pt. The config stores the checkpoint directory it resolved to, so a `--resume` decodes with the same vae2 and refuses another. The vae2 stays in eval mode, its weights get no gradient, and there is no latent noise. `latent` must match its channels. The network has the vae2 encoder's building blocks: a 3×3 stem, residual blocks of conv and activation (`activation`, `silu` or `relu` as in vae2, independent of the vae2's) whose last convolution starts at zero, a stride-2 convolution to each coarser level, `blocks` residual blocks at 1/4 with dilations 1, 2, 4 in turn, and a 3×3 head. `channels` is the width at the low resolution, `widths` the multiples at low, 1/2, and 1/4, `encoder_blocks` the residual count at low and at 1/2. `init_from_vae: true` starts the last blocks at 1/4 and the head from the vae2 encoder, with shift and scale folded into the head, on a fresh run.

The `latent` loss term is L1 between the predicted z and the vae2 code of the master, encoded in the same step without gradient. The image terms (`pixel`, `vgg`, `color`, `oklab`, `ms_ssim`) compare the decoded picture with the master, and their gradient runs through the frozen decoder into z. Every term is optional. `train_raw/latent` and `val_raw/latent` are logged even when the term is left out. The first validation of a process logs three references on the same frames: `val_ref/bicubic_psnr`, `val_ref/bilinear_psnr`, and `val_ref/vae_psnr`, the vae2 roundtrip decode(encode(master)), which is the ceiling of this decoder. The live page shows `low` (bicubic 2×), `upscale`, and `master`; each validation frame folder also gets `vae.png`, that roundtrip. With `faces:` YuNet runs on the master of each pair (in the loader thread for training, on the main thread for validation), and the weight map weighs `oklab`, `ms_ssim`, and `vgg` exactly as for the vae2. The `latent` L1 is weighted too: the map is area-pooled by the vae2 factor 8 onto the latent grid (a last partial cell averages the pixels it has, 852 → 107) and divided again by its mean per frame. `train_raw/latent` and `val_raw/latent` are then the face-weighted L1. Validation adds `val/psnr_face`, `val/psnr_nonface`, and `val_raw/face_fraction`, and the first one also `val_ref/vae_psnr_face` and `val_ref/vae_psnr_nonface`, the roundtrip ceiling on the same pixels. The live page draws the boxes on `upscale` and `master`. `configs/latent-up-spark.yaml` is the run on the DGX Spark: on vae2g, with the vae2g image recipe (oklab 4.5, ms_ssim 1, VGG 0.230 / 0.155) beside latent 1.0, and the vae2g faces. `whoodeo-apply` and `whoodeo-coreml` both take a latent_up checkpoint, see below.

## latent_up in Core ML and the macOS app

`whoodeo-coreml` turns a latent_up checkpoint into one program with its vae2 decoder behind it: the low frame in, the clamped 2× frame out, both fp16, input `frames` and output `image`. The vae2 is the checkpoint the latent_up config names; a path under `/mnt/home-storinator-trem` is looked up under `/Volumes/home`, where the Mac mounts the same share, and `--vae` overrides it. The latent shift and scale fold into the last convolution of latent_up, so the graph hands the decoder μ directly. The input is flexible, any size from 64×64 to 1136×640 (`--max-low HxW`); a whole 1280×720 low frame aborts on the Neural Engine. `--low HxW`, repeatable, gives enumerated sizes instead. `--plan` prints the Neural Engine placement at 428×240, 640×360, 852×480, and 1136×640. On the Mac:

    uv pip install --python .venv/bin/python 'coremltools==9.1.dev1'
    .venv/bin/whoodeo-coreml /Volumes/home/whoodeo/runs/<run> --plan --bench low1.mkv low2.mkv

The Neural Engine loses convolution products below about 2^-14, and the residual convolutions that start at zero make many of them. `--conv-scale k` (16 by default for vae2 and latent_up, off for vote and edsr, 1 turns it off) multiplies each convolution's weight and bias by k and puts a separate multiply by 1/k behind it, with coremltools' `fuse_conv_scale` pass off so the multiply stays apart. It costs nothing measurable. The conversion checks the fold and the scale against the plain fp32 path. The rest of the Neural Engine error is its SiLU, an approximation that it fuses into the convolution, and only a different activation removes it. ReLU is exact there: with `activation: relu` the conversion emits a plain relu, and with k=16 a relu latent_up + vae2 with random weights scaled to the activations of the trained pair reaches 65.6 dB against fp32 at 426×240 on the Neural Engine (241/241 ops there, about 99 ms), where the same net with silu gives 44.6 dB (110 ms). Earlier probes at 640×360 gave about 70 dB for relu against 47 for silu and 226 ms against 251. LeakyReLU and ReLU6 are about as exact but as slow as SiLU, hardswish and GELU are not exact. relu needs a model trained with it: `configs/vae2e-spark.yaml`, then latent_up on top (`configs/latent-up-spark.yaml` is `activation: relu` and decodes with the relu vae2g).

Without `-o` the package goes to `~/Library/Application Support/Whoodeo/Models`, the app's collection. `--bench` decodes frame 100 of each video, resizes it to each probe size, and compares GPU and Neural Engine against PyTorch fp32 on MPS: worst PSNR, largest difference, and milliseconds per size. A unit passes at 50 dB or more (one 8-bit step of rounding noise is about 59 dB), and the faster passing unit becomes the app's default; the numbers go into the package metadata. vote and edsr use the same criterion. On the M1 Ultra, step 2669 of train-20261021-064120 took 73, 156, 265, and 454 ms per frame on the GPU at 428×240, 640×360, 852×480, and 1136×640, at 82–84 dB, and 116, 253, 457, and 791 ms on the Neural Engine with every op placed there, at 48–55 dB, so the GPU is the default.

Step 500 of the vae2g latent_up train-20261022-013506 (relu, k=16, every op on the Neural Engine at all four sizes) took 73, 150, 258, and 450 ms on the GPU at 76.6–78.1 dB, and 106, 233, 417, and 717 ms on the Neural Engine at 75.2–76.1 dB: with relu both units pass, and the GPU stays the default because it is faster.

`whoodeo-apply` runs latent_up on a video, in PyTorch or through a package. A latent_up checkpoint loads the vae2 its config names (`/mnt/home-storinator-trem` looked up under `/Volumes/home`, `--vae` overrides it) and runs the path validation takes, code, z / scale + shift, decode, clamp, crop to 2h × 2w, on MPS in fp32. `--coreml <package>`, or a `.mlpackage` in place of the checkpoint, runs the Core ML program instead, on `--unit gpu` (CPU_AND_GPU, the default), `ane` (CPU_AND_NE), `cpu`, or `all`. The frame is padded to the package's `whoodeo.pad` and checked against its range or sizes as in the app, so a 1280×720 low frame stops with the range in the message; without `--out` a package writes `<stem>-recon-<package>-<unit>.mkv` into the current directory. The end of the run prints the median model time per frame. On the Mac, without `uv run`, so the training dependencies (OpenCV) need not be in the venv:

    .venv/bin/whoodeo-apply /Volumes/home/whoodeo/runs/<run>/checkpoints/<step> low.mkv --out up.mkv
    .venv/bin/whoodeo-apply ~/Library/Application\ Support/Whoodeo/Models/<package>.mlpackage low.mkv --out up.mkv
    .venv/bin/whoodeo-apply ~/Library/Application\ Support/Whoodeo/Models/<package>.mlpackage low.mkv --unit ane --out up.mkv

On ten frames of a 640×360 clip the package was 76.9 dB on the GPU and 75.4 dB on the Neural Engine against the PyTorch checkpoint (worst frame 76.0 and 74.9 dB), 77.7 and 76.0 dB at 426×238, which pads to 428×240. Per frame at 640×360: 148 ms on the GPU, 234 ms on the Neural Engine, 173 ms in PyTorch on MPS.

The macOS app (`macos/Whoodeo`, open `Whoodeo.xcodeproj`, or `xcodebuild -scheme Whoodeo -configuration Release`) lists the packages in that folder. It reads `whoodeo.pad`, the size range, and `whoodeo.sizes` for an enumerated package from the metadata: a frame is padded on the bottom and the right to the next multiple of 4 (reflect, as `pad_to`), must then lie in the range or match one of the sizes, and the result is cropped to 2h × 2w. A 426×240 film runs as 428×240. Larger films get an error that names the range. fp16 and fp32 inputs and outputs both work, so vote and edsr packages run as before. Without the window, `Whoodeo.app/Contents/MacOS/Whoodeo --check model.mlpackage in.f32 out.f32 --unit ane` runs one tensor file.

# The loss and the configs

The loss is the sum of `weight` times each raw term under `loss.terms`. A weight is a plain multiplier: nothing is normalized or calibrated on the network, and the weights need not add up to anything. `pixel` is L1 or MSE, `kind: l1` or `kind: mse`. `vgg` has no term weight; `layers` maps each VGG19 layer to its weight on the L1 of that layer. `color`, `oklab`, and `ms_ssim` are for `vae`, `vae2`, and `latent_up`, `mu_l2` and `scale_eq` for `vae2` only, `latent` for `latent_up` only. A vae adds `beta` times the KL, and a gan term adds as described with its model. TensorBoard groups tags by their first segment, so `train/` and `val/` hold the overview and the details sit in groups that collapse on their own. `train/` has `loss`, `lr`, and each weighted term; `val/` has `loss`, `psnr` (which compares across loss weights), and each weighted term. The unweighted values are under `train_raw/` and `val_raw/` (VGG as `vgg_<layer>`, the KL as `kl`, the vae2 side terms and GAN weight in `train_raw/`). Latent statistics are under `train_latent/` and `val_latent/`, with the histogram `val_latent/mu` and the text table `val_latent/channels`. The vae2 robustness PSNRs are under `val_noise/`, the latent_up reference PSNRs under `val_ref/`, and the vote neighbor numbers under `train_neighbor/` and `val_neighbor/`. A `--resume` needs the same loss as the checkpoint.

The weights of the super-resolution configs were estimated from the earlier balanced loss: pixel weight 1, and each VGG layer weighed so that the VGG part had its old share against pixel L1 on the bilinear upscale of the validation frames at step 0. Each config says which numbers it used.

- `configs/modified.yaml`, `configs/mse.yaml`, `configs/vgg.yaml`: the `modified` network with pixel L1, pixel MSE, and L1 plus VGG.
- `configs/finetune.yaml`: fine-tune a trained generator with VGG and a GAN, `--model <run>`.
- `configs/shift.yaml`, `configs/pyramid.yaml`, `configs/vote.yaml`: the multi-frame models above.
- `configs/edsr.yaml`, `configs/x264.yaml`: EDSR on every pair, and on the x264 pairs only.
- `configs/swinir.yaml`, `configs/vq.yaml`: SwinIR and the codebook model.
- `configs/vae2.yaml`, `configs/vae2d-spark.yaml`: the current vae2 recipe, at batch 2 and on the DGX Spark at batch 6.
- `configs/vae2-gan.yaml`: the GAN fine-tune of a vae2 decoder.
- `configs/vae2e-spark.yaml`: vae2d with relu, for the Neural Engine.
- `configs/vae2f-spark.yaml`: vae2e with oklab and ms_ssim instead of pixel and color.
- `configs/vae2g-spark.yaml`: vae2f with face-weighted image terms, `faces: {weight: 3.0, score: 0.6, expand: 1.2, feather: 0.1}`.
- `configs/latent-up-spark.yaml`: latent_up (relu) on the frozen relu vae2g, on the DGX Spark, with the vae2g loss recipe beside the latent L1 and face weighting.
