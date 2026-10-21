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

`arch: vae2` reconstructs a master frame like `vae`, with a network shaped for the Neural Engine. The layers are 3×3 convolutions, SiLU, additions, and nearest-neighbor doubling. There is no normalization and no attention. `channels` is the width at full resolution. By default the width doubles at half resolution and is four times `channels` from a quarter down, so `configs/vae2.yaml` runs 64, 128, 256, 256 from the frame to the latent. Full resolution has one residual block, each level below it two. `blocks` is the number of residual blocks at the latent, with dilations 1, 2, and 4 in turn. Encoder and decoder mirror each other. The layout can be set apart: `encoder_blocks` and `decoder_blocks` name the residual count per level above the latent (full resolution first), and `widths` names a multiple of `channels` per level including the latent; `configs/vae2.yaml` uses `encoder_blocks: [1, 2, 2]`, `decoder_blocks: [2, 3, 2]`, `widths: [1, 2, 4, 4]`. `decoder_widths` can give the decoder its own multiples. The last convolution of every residual branch starts at zero. The default layout has 22.2M parameters, and the decoder costs about 1.2 TFLOP for a 720p frame, against 3.2 for `vae` 4×128.

`decode(latent, clamp=True)` limits the frame to [0, 1]. Training compares the unclamped output. `beta` 1e-4 keeps the KL near one part in a thousand of the objective; like `vae`, beta is not part of the architecture key, so it can change on `--resume`. Each validation stores per-channel `latent_shift` and `latent_scale` in the model: (μ − shift) × scale has mean 0 and std 1 in each channel. A model that works on latents should see that scaled code and hand z / scale + shift back to the decoder. TensorBoard logs the mean scale as `val_latent/scale`. The mean shift is `val_latent/mu_mean`.

The `color` loss term is L1 between average-pooled frames, the chroma channels weighted twice. `space: ycbcr`, the default, pools the gamma sRGB frames and compares BT.709 Y, Cb, Cr. `space: oklab` pools in linear light and compares Oklab L, a, b. `pool` is the pooling factor, 8 by default. Both are part of the loss recipe. The term is for color drift in flat areas, and `vae` takes it too. The current vae2 weights are pixel 1.0, VGG `relu2_2` 0.230 / `relu3_4` 0.155, and color 5.5 in Oklab with pool 16.

`posterior: fixed` makes the encoder output only μ: no log-variance, no sample, and no KL. Leave it out for the usual diagonal Gaussian.

`latent_noise: {tau: ...}` adds per-channel-relative noise to the latent in training, after any sample. Each frame draws s from |N(0, τ²)| and gets s × std_c × N(0, 1), where std_c is a running std of μ per channel. It sits outside the loss recipe, so a resume may change it; `tau: 0` turns it off for a clean final fine-tune. Validation still decodes μ.

Two more terms stand beside the objective, in the step loss only: `mu_l2` is `weight` times mean(μ²), against scale drift, and `scale_eq` picks one of its `factors` (for example `[2, 4]`) each step, decodes the average-pooled latent, and compares it with the pooled frame under the same pixel, VGG, and color weights.

Each vae2 validation also logs `val_noise/psnr_{0.1,0.3,0.5}` (PSNR of decode(μ + s × std_c × N(0, 1)) with a fixed seed) and `val_noise/scale_eq_psnr` (decode of the 2×2 pooled μ against the 2×2 pooled frame). The per-channel SNR sits in the `val_latent/channels` table. `uv run --no-sync python -m whoodeo.eval_latent <checkpoint> [--loss cfg]` prints the same numbers for a saved checkpoint.

`configs/vae2.yaml` is the current recipe: the layout above with 16 latent channels, `posterior: fixed`, latent noise with tau 0.4, Oklab color, `mu_l2` at 5e-4, and `scale_eq` at 0.25, at batch 2. `configs/vae2d-spark.yaml` is the same recipe at batch 6 for the DGX Spark, and its comments say where each choice comes from.

A vae2 may add a `gan` term with the U-Net discriminator. Its weight is not a share of the objective. The generator term is the latent diffusion adaptive weight, the gradient norm of the objective over that of the GAN loss at the last decoder convolution, times `weight`. The discriminator trains from the first step. The generator term joins at `start`. The validation objective leaves the GAN out. `freeze_encoder: true` trains the decoder alone. `configs/vae2-gan.yaml` combines the two as a fine-tune of a vae2 run:

```
uv run whoodeo-train configs/vae2-gan.yaml --model <run>/latest
```

`whoodeo-coreml` turns a vae2 checkpoint into its decoder: an fp16 latent in, the clamped fp16 frame out. The input takes the latent sizes 60×107, 90×160, and 135×240, which are 480p, 720p, and 1080p at factor 8. `--latent 90x160` gives one fixed size instead. The metadata carries the per-channel latent shift and scale.

# The latent_up model

`arch: latent_up` is a 2× upscaler that predicts the code of a frozen vae2 instead of pixels. The low frame is h × w and its master 2h × 2w. A vae2 at factor 8 codes the master at h/4 × w/4, so the network maps the low RGB frame to that grid: the normalized code z = (μ − shift) × scale, which has mean 0 and std 1 per channel. The frozen vae2 decoder turns z / scale + shift into the 2× picture. A side that 4 does not divide is padded like vae2 pads, and the picture is cut back to 2h × 2w.

`vae` under `model:` names the vae2, a run directory (its latest checkpoint, as `--model` reads it), a checkpoint, or a model.pt. The config stores the checkpoint directory it resolved to, so a `--resume` decodes with the same vae2 and refuses another. The vae2 stays in eval mode, its weights get no gradient, and there is no latent noise. `latent` must match its channels. The network has the vae2 encoder's building blocks: a 3×3 stem, residual blocks of conv and SiLU whose last convolution starts at zero, a stride-2 convolution to each coarser level, `blocks` residual blocks at 1/4 with dilations 1, 2, 4 in turn, and a 3×3 head. `channels` is the width at the low resolution, `widths` the multiples at low, 1/2, and 1/4, `encoder_blocks` the residual count at low and at 1/2. `init_from_vae: true` starts the last blocks at 1/4 and the head from the vae2 encoder, with shift and scale folded into the head, on a fresh run.

The `latent` loss term is L1 between the predicted z and the vae2 code of the master, encoded in the same step without gradient. `pixel`, `vgg`, and `color` compare the decoded picture with the master, and their gradient runs through the frozen decoder into z. Every term is optional. `train_raw/latent` and `val_raw/latent` are logged even when the term is left out. The first validation of a process logs three references on the same frames: `val_ref/bicubic_psnr`, `val_ref/bilinear_psnr`, and `val_ref/vae_psnr`, the vae2 roundtrip decode(encode(master)), which is the ceiling of this decoder. The live page shows `low` (bicubic 2×), `upscale`, and `master`; each validation frame folder also gets `vae.png`, that roundtrip. `configs/latent-up-spark.yaml` is the first trial on the DGX Spark. Only training exists so far: `whoodeo-apply` and `whoodeo-coreml` do not take a latent_up checkpoint yet.

# The loss and the configs

The loss is the sum of `weight` times each raw term under `loss.terms`. A weight is a plain multiplier: nothing is normalized or calibrated on the network, and the weights need not add up to anything. `pixel` is L1 or MSE, `kind: l1` or `kind: mse`. `vgg` has no term weight; `layers` maps each VGG19 layer to its weight on the L1 of that layer. `color` is for `vae`, `vae2`, and `latent_up`, `mu_l2` and `scale_eq` for `vae2` only, `latent` for `latent_up` only. A vae adds `beta` times the KL, and a gan term adds as described with its model. TensorBoard groups tags by their first segment, so `train/` and `val/` hold the overview and the details sit in groups that collapse on their own. `train/` has `loss`, `lr`, and each weighted term; `val/` has `loss`, `psnr` (which compares across loss weights), and each weighted term. The unweighted values are under `train_raw/` and `val_raw/` (VGG as `vgg_<layer>`, the KL as `kl`, the vae2 side terms and GAN weight in `train_raw/`). Latent statistics are under `train_latent/` and `val_latent/`, with the histogram `val_latent/mu` and the text table `val_latent/channels`. The vae2 robustness PSNRs are under `val_noise/`, the latent_up reference PSNRs under `val_ref/`, and the vote neighbor numbers under `train_neighbor/` and `val_neighbor/`. A `--resume` needs the same loss as the checkpoint.

The weights of the super-resolution configs were estimated from the earlier balanced loss: pixel weight 1, and each VGG layer weighed so that the VGG part had its old share against pixel L1 on the bilinear upscale of the validation frames at step 0. Each config says which numbers it used.

- `configs/modified.yaml`, `configs/mse.yaml`, `configs/vgg.yaml`: the `modified` network with pixel L1, pixel MSE, and L1 plus VGG.
- `configs/finetune.yaml`: fine-tune a trained generator with VGG and a GAN, `--model <run>`.
- `configs/shift.yaml`, `configs/pyramid.yaml`, `configs/vote.yaml`: the multi-frame models above.
- `configs/edsr.yaml`, `configs/x264.yaml`: EDSR on every pair, and on the x264 pairs only.
- `configs/swinir.yaml`, `configs/vq.yaml`: SwinIR and the codebook model.
- `configs/vae2.yaml`, `configs/vae2d-spark.yaml`: the current vae2 recipe, at batch 2 and on the DGX Spark at batch 6.
- `configs/vae2-gan.yaml`: the GAN fine-tune of a vae2 decoder.
- `configs/latent-up-spark.yaml`: latent_up on the frozen vae2d, on the DGX Spark.
