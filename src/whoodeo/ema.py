"""A moving average of the generator weights (EMA).

The trained weights jitter from step to step. On latent_up that showed as a
global brightness offset of 2 to 3 of 255 at some validations, about 1 dB of
PSNR (21.11.2026). The average beside them is smoother: validation, the
preview, and model.pt use it, the optimizer keeps working on the trained
weights.

After each optimizer step every parameter moves its average by
(1 - d) * (weight - average). d is the configured decay, lower at the start:
min(decay, (1 + n) / (10 + n)) after n updates, so a fresh average follows
the weights quickly before it settles. Buffers are not averaged; the
networks that use an average have none that train.
"""

from contextlib import contextmanager

import torch


class WeightAverage:
    def __init__(self, model, decay, updates=0, average=None):
        """Start from `model`'s weights, or from `average` (a state dict) when given."""
        self.decay = float(decay)
        self.updates = int(updates)
        self.names = [name for name, _param in model.named_parameters()]
        params = dict(model.named_parameters())
        self.shadow = {}
        for name in self.names:
            source = params[name].detach() if average is None else average[name]
            self.shadow[name] = source.to(device=params[name].device, dtype=params[name].dtype).clone()

    def current_decay(self):
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model):
        decay = self.current_decay()
        params = dict(model.named_parameters())
        shadows = [self.shadow[name] for name in self.names]
        weights = [params[name].detach() for name in self.names]
        torch._foreach_lerp_(shadows, weights, 1.0 - decay)
        self.updates += 1

    @contextmanager
    def applied(self, model):
        """`model` with the averaged weights inside the block, the trained ones after.

        The tensors are exchanged, not copied; the optimizer keeps its parameters.
        """
        params = dict(model.named_parameters())
        self._exchange(params)
        try:
            yield model
        finally:
            self._exchange(params)

    def _exchange(self, params):
        for name in self.names:
            param = params[name]
            kept = param.data
            param.data = self.shadow[name]
            self.shadow[name] = kept

    def state(self):
        """The decay and the update count for train.pt."""
        return {"decay": self.decay, "updates": self.updates}
