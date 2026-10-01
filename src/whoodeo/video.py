import av
import numpy as np
import torch


def rgb_image(tensor):
    """CHW or NCHW float image in 0..1 as HWC uint8 RGB."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected RGB channels first, got {tuple(tensor.shape)}")
    image = tensor.detach().permute(1, 2, 0).cpu().numpy()
    return (np.clip(image, 0, 1) * 255.0).astype(np.uint8)


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
