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
