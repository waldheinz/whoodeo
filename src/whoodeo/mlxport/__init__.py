"""MLX port of the align path: SPyNet, warp, latent_up with FiLM, vae2 decoder.

Layouts are NHWC as MLX convolutions want them. Weights come from a
PyTorch checkpoint through convert.py; nothing here trains.
"""
