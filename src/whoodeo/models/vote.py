"""EDSR trunk with a shared nonlinear vote from the previous and next frame.

Three low-resolution frames go in. The picture that comes out is the middle
frame at 2×. One stem, the head convolution plus `stem` residual blocks, runs
on each frame with the same weights. The difference of each neighbor from the
middle then goes through one shared vote: a convolution, a ReLU, `vote`
residual blocks, and a closing convolution. Nothing in that path adds the raw
difference back on, so the decision cannot stay linear. The two results are
added onto the middle features. The trunk (`blocks` residual blocks and the
body convolution) runs on the sum. The long skip adds the middle stem output,
so the neighbors reach the picture only through the trunk.

The convolutions keep PyTorch's default initialization. The DIV2K mean shift
matches EDSR: frames are already in 0..1, and rgb_range is 1.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from whoodeo.models.edsr import MeanShift, ResBlock, Upsampler, default_conv


class VoteEDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, res_scale, stem, vote, conv=default_conv):
        super().__init__()
        if n_resblocks < 1 or n_feats < 1:
            raise SystemExit("blocks and channels must be positive")
        if stem < 0:
            raise SystemExit("stem must be zero or positive")
        if vote < 0:
            raise SystemExit("vote must be zero or positive")

        kernel_size = 3
        scale = 2
        rgb_range = 1
        self.in_frames = 3
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = conv(3, n_feats, kernel_size)
        self.stem = nn.Sequential(*[
            ResBlock(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=res_scale)
            for _ in range(stem)
        ])
        self.vote_in = conv(n_feats, n_feats, kernel_size)
        self.vote = nn.Sequential(*[
            ResBlock(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=res_scale)
            for _ in range(vote)
        ])
        self.vote_out = conv(n_feats, n_feats, kernel_size)
        self.body = nn.Sequential(
            *[
                ResBlock(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=res_scale)
                for _ in range(n_resblocks)
            ],
            conv(n_feats, n_feats, kernel_size),
        )
        self.tail = nn.Sequential(
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size),
        )
        self.last_vote = None
        self.last_center = None

    def _features(self, frame):
        return self.stem(self.head(self.sub_mean(frame)))

    def _decide(self, delta):
        return self.vote_out(self.vote(F.relu(self.vote_in(delta))))

    def _kernels(self):
        found = []
        for module in (self.vote_in, self.vote, self.vote_out):
            for param in module.parameters():
                if param.ndim == 4:
                    found.append(param)
        return found

    def forward(self, x):
        frames = x.chunk(self.in_frames, dim=1)
        feats = [self._features(frame) for frame in frames]
        mid = self.in_frames // 2
        center = feats[mid]
        vote = None
        for index, feat in enumerate(feats):
            if index == mid:
                continue
            term = self._decide(feat - center)
            vote = term if vote is None else vote + term
        res = self.body(center + vote) + center
        out = self.add_mean(self.tail(res))
        # A zero difference still produces a resting vote. Subtract it so the
        # logged number is the part that depends on the frames.
        with torch.no_grad():
            rest = self._decide(torch.zeros_like(center))
            content = vote.detach() - rest * (self.in_frames - 1)
        self.last_vote = content.abs().mean()
        self.last_center = center.detach().abs().mean()
        return out

    def neighbor_usage(self):
        """Weight, vote, and gradient ratios from the forward just run.

        weight is the mean absolute kernel of the shared vote. vote is the
        mean absolute frame-dependent vote divided by the mean absolute
        middle features. grad is the mean absolute gradient of those kernels
        divided by the same quantity on the head. grad is absent until the
        first backward. None before the first forward.
        """
        if self.last_vote is None or self.last_center is None:
            return None
        kernels = self._kernels()
        weight = torch.cat([param.detach().reshape(-1) for param in kernels]).abs().mean()
        vote = self.last_vote / self.last_center.clamp_min(1e-8)
        parts = [weight, vote]
        head_grad = self.head.weight.grad
        grads = [param.grad for param in kernels]
        has_grad = head_grad is not None and all(grad is not None for grad in grads)
        if has_grad:
            grad = torch.cat([param.grad.detach().reshape(-1) for param in kernels]).abs().mean()
            parts.append(grad / head_grad.detach().abs().mean().clamp_min(1e-12))
        values = torch.stack(parts).detach().float().cpu().tolist()
        stats = {"weight": values[0], "vote": values[1]}
        if has_grad:
            stats["grad"] = values[2]
        return stats
