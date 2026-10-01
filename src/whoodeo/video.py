

import cv2
import av
import torch
import numpy as np

def read_video_frames(video_path, start_sec=0):
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if start_sec > 0:
            container.seek(int(start_sec * av.time_base), backward=True, any_frame=False)
        for frame in container.decode(stream):
            if start_sec > 0 and (frame.time is None or frame.time + 1e-3 < start_sec):
                continue
            array = frame.to_ndarray(format='rgb24')  # Shape: (height, width, 3)
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            yield tensor


def chw_to_bgr(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (np.clip(image, 0, 1) * 255).astype('uint8')
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def label(image, text, x):
    origin = (x, 32)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)


def preview_image(item):
    if not isinstance(item, tuple):
        return chw_to_bgr(item)
    low, recon = item
    _, recon_h, recon_w = recon.shape
    low_bgr = chw_to_bgr(low)
    low_bgr = cv2.resize(low_bgr, (recon_w, recon_h), interpolation=cv2.INTER_NEAREST)
    recon_bgr = chw_to_bgr(recon)
    gap = np.zeros((recon_h, 4, 3), dtype=np.uint8)
    image = np.concatenate([low_bgr, gap, recon_bgr], axis=1)
    label(image, "low-res", 16)
    label(image, "recon", recon_w + 4 + 16)
    return image


def frame_for_display(item, min_height=0):
    image = preview_image(item)
    height, width = image.shape[:2]
    if min_height and height < min_height:
        scale = min_height / height
        image = cv2.resize(
            image,
            (int(round(width * scale)), min_height),
            interpolation=cv2.INTER_LINEAR,
        )
        height, width = image.shape[:2]
    if width > 1800:
        scale = 1800 / width
        image = cv2.resize(
            image,
            (1800, int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


class Preview:
    def __init__(self, min_height=0):
        self.min_height = min_height
        self.ready = False

    def show(self, item):
        image = frame_for_display(item, self.min_height)
        if not self.ready:
            cv2.namedWindow("Live Display", cv2.WINDOW_NORMAL)
            self.ready = True
        cv2.imshow("Live Display", image)
        cv2.waitKey(1)

    def close(self):
        if self.ready:
            cv2.destroyAllWindows()


def show_video_window(gen):
    preview = Preview()
    for item in gen:
        preview.show(item)
    preview.close()


def write_video(generator, output_path, fps=30):
    """
    Writes video frames from a generator of tensors to a video file using PyAV.

    Args:
        generator (generator): A generator yielding tensors of shape [C, H, W] or [1, C, H, W],
                               where C=3 (RGB), H is height, and W is width.
        output_path (str): Path to save the video (e.g., 'output.mp4').
        fps (int): Frame rate of the output video (default: 30).
    """
    # Open the output video container
    container = av.open(output_path, 'w')

    # Initialize video stream (width and height will be set after first frame)
    stream = container.add_stream('libx264', rate=fps)
    stream.options = { 'crf' : '25' }

    # Flag to track if the stream has been initialized with frame dimensions
    stream_initialized = False

    # Iterate over frames yielded by the generator
    for frame_tensor in generator:
        if isinstance(frame_tensor, tuple):
            frame_tensor = frame_tensor[-1]
        # Handle tensor shape: ensure it's [C, H, W]
        if frame_tensor.dim() == 4:  # Case: [1, C, H, W]
            frame_tensor = frame_tensor.squeeze(0)  # Remove batch dimension
        if frame_tensor.dim() != 3:
            raise ValueError("Frame tensor must have shape [C, H, W] or [1, C, H, W]")

        # Extract dimensions and validate channels
        C, H, W = frame_tensor.shape
        if C != 3:
            raise ValueError("Frame tensor must have 3 channels (RGB)")

        # Set stream dimensions based on the first frame
        if not stream_initialized:
            stream.width = W
            stream.height = H
            stream_initialized = True

        # Convert tensor to NumPy array: [H, W, C], uint8
        frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
        frame_np = (np.clip(frame_np, 0, 1) * 255).astype(np.uint8)

        # Create a PyAV VideoFrame from the NumPy array
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')

        # Encode the frame and write packets to the container
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush any remaining packets in the encoder
    for packet in stream.encode():
        container.mux(packet)

    # Close the container to finalize the video file
    container.close()
