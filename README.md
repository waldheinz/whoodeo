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
