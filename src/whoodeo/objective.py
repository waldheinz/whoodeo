"""Pixel, VGG, and GAN terms, scaled so their weights are shares of the sum.

The first term anchors the magnitude: its scale is 1. The other scales are
chosen so that weight_i * raw_i lines up with that anchor. `start` freezes
the scales after the validation pass. `running` updates them from an EMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoodeo.config import VGG_LAYERS

EMA_DECAY = 0.99
REF_FLOOR = 1e-8


def pixel_loss(pred, high, kind):
    if kind == "l1":
        return F.l1_loss(pred, high)
    return F.mse_loss(pred, high)


def scales_from_refs(names, weights, term_ref):
    anchor = names[0]
    anchor_weight = weights[anchor]
    anchor_ref = term_ref[anchor]
    return {
        name: (weights[name] / anchor_weight) * anchor_ref / term_ref[name]
        for name in names
    }


class VGGLoss(nn.Module):
    """L1 on frozen VGG19 layers. The classifier head is dropped."""

    def __init__(self, layers):
        super().__init__()
        self.layer_names = tuple(layers)
        indexes = [VGG_LAYERS[name] for name in self.layer_names]
        self.indexes = tuple(indexes)
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(indexes) + 1]
        for layer in features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.features = features.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def layer_losses(self, pred, target):
        pred_maps = self._maps(pred.clamp(0, 1))
        with torch.no_grad():
            target_maps = self._maps(target.clamp(0, 1))
        return {
            name: F.l1_loss(pred_maps[name], target_maps[name])
            for name in self.layer_names
        }

    def _maps(self, image):
        hidden = (image - self.mean) / self.std
        found = {}
        wanted = set(self.indexes)
        for index, layer in enumerate(self.features):
            hidden = layer(hidden)
            if index in wanted:
                found[index] = hidden
        return {name: found[VGG_LAYERS[name]] for name in self.layer_names}


class Objective:
    def __init__(self, cfg, device):
        self.names = tuple(term.name for term in cfg.terms)
        self.weights = {term.name: term.weight for term in cfg.terms}
        self.balance = cfg.balance
        self.vgg_layers = ()
        self.vgg = None
        vgg = cfg.term("vgg")
        if vgg is not None:
            self.vgg_layers = vgg.layers
            self.vgg = VGGLoss(vgg.layers).to(device)
        self.layer_ref = {}
        self.term_ref = {}
        self.scale = {}

    def calibrate(self, layer_means, term_means):
        for name in self.vgg_layers:
            value = float(layer_means[name])
            if value <= 0:
                raise SystemExit(f"VGG {name} on the val frames is zero")
            self.layer_ref[name] = value
        self.term_ref = {}
        if "vgg" in self.names:
            self.term_ref["vgg"] = 1.0
        for name in self.names:
            if name == "vgg":
                continue
            value = float(term_means[name])
            if value <= 0:
                raise SystemExit(f"{name} loss on the val frames is zero")
            self.term_ref[name] = value
        self.scale = scales_from_refs(self.names, self.weights, self.term_ref)

    def vgg_raw(self, pred, high):
        losses = self.vgg.layer_losses(pred, high)
        detached = {name: losses[name].detach() for name in self.vgg_layers}
        normed = [losses[name] / self.layer_ref[name] for name in self.vgg_layers]
        return torch.stack(normed).mean(), detached

    def combine(self, raw):
        total = None
        weighted = {}
        for name in self.names:
            value = self.scale[name] * raw[name]
            weighted[name] = value
            total = value if total is None else total + value
        return total, weighted

    def observe(self, layer_raw, term_raw):
        """Move the running references. `start` keeps the calibrated scales."""
        if self.balance != "running":
            return
        for name, value in layer_raw.items():
            self.layer_ref[name] = _ema(self.layer_ref[name], float(value))
        for name, value in term_raw.items():
            self.term_ref[name] = _ema(self.term_ref[name], float(value))
        self.scale = scales_from_refs(self.names, self.weights, self.term_ref)

    def state_dict(self):
        return {
            "layer_ref": dict(self.layer_ref),
            "term_ref": dict(self.term_ref),
            "scale": dict(self.scale),
        }

    def load_state_dict(self, state):
        if set(state.get("term_ref", {})) != set(self.names) or set(state.get("scale", {})) != set(self.names):
            raise SystemExit("checkpoint objective does not match the loss terms")
        if set(state.get("layer_ref", {})) != set(self.vgg_layers):
            raise SystemExit("checkpoint objective does not match the vgg layers")
        self.layer_ref = {name: float(state["layer_ref"][name]) for name in self.vgg_layers}
        self.term_ref = {name: float(state["term_ref"][name]) for name in self.names}
        self.scale = {name: float(state["scale"][name]) for name in self.names}


def _ema(old, value):
    mixed = EMA_DECAY * old + (1.0 - EMA_DECAY) * value
    return max(mixed, REF_FLOOR)
