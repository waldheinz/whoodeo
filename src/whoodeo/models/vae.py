"""Latent statistics of a vae2 over a split, for the logs."""

import torch


class LatentTally:
    """Sums of μ and σ, so a split can be compared with the unit Gaussian.

    `spread` is the square root of var(μ) + mean(σ²). It sits near 1 when the
    code matches that Gaussian. A mu_std near 0 means the latent carries
    nothing. A sigma_mean near 0 means a deterministic code.

    Per channel it keeps the mean and std of μ, the mean σ, the SNR std(μ) /
    mean(σ), and the KL to the unit Gaussian per latent position. A channel
    counts as active when its μ varies with std above ACTIVE_STD; a collapsed
    channel copies the prior. A fixed posterior (logvar None) has no σ, so
    SNR and KL are None.
    """

    def __init__(self):
        self.count = 0
        self.sum_mu = 0.0
        self.sumsq_mu = 0.0
        self.sum_sigma = 0.0
        self.sumsq_sigma = 0.0
        self.channels = None
        self.fixed = False
        self._hist = []

    def add(self, mu, logvar, sample=False):
        mu = mu.detach().float()
        if logvar is None:
            self.fixed = True
            logvar = torch.zeros_like(mu)
            sigma = torch.zeros_like(mu)
        else:
            sigma = torch.exp(0.5 * logvar.detach().float())
        self.count += mu.numel()
        self.sum_mu += float(mu.sum())
        self.sumsq_mu += float(mu.square().sum())
        self.sum_sigma += float(sigma.sum())
        self.sumsq_sigma += float(sigma.square().sum())
        logvar = logvar.detach().float()
        summed = torch.stack((
            mu.sum(dim=(0, 2, 3)),
            mu.square().sum(dim=(0, 2, 3)),
            sigma.sum(dim=(0, 2, 3)),
            sigma.square().sum(dim=(0, 2, 3)),
            logvar.sum(dim=(0, 2, 3)),
        )).cpu()
        self.channels = summed if self.channels is None else self.channels + summed
        if sample:
            flat = mu.reshape(-1).cpu()
            take = min(4096, flat.numel())
            if flat.numel() > take:
                flat = flat[torch.randint(flat.numel(), (take,))]
            self._hist.append(flat)

    def report(self):
        if self.count == 0:
            return None
        mean = self.sum_mu / self.count
        var = max(0.0, self.sumsq_mu / self.count - mean * mean)
        mu_std = var ** 0.5
        sigma_mean = self.sum_sigma / self.count
        sigma_sq = self.sumsq_sigma / self.count
        width = self.channels.shape[1]
        each = self.count // width
        per_channel = []
        for index in range(width):
            total, squares, sig, sig_sq, logs = (
                float(self.channels[row, index]) for row in range(5)
            )
            center = total / each
            channel_var = max(0.0, squares / each - center * center)
            mu_std = channel_var ** 0.5
            sigma_mean = sig / each
            if self.fixed:
                kl = None
                snr = None
            else:
                kl = 0.5 * (squares / each + sig_sq / each - 1.0 - logs / each)
                snr = mu_std / max(sigma_mean, 1e-12)
            per_channel.append((mu_std, sigma_mean, kl, center, snr))
        hist = None
        if self._hist:
            hist = torch.cat(self._hist)
            if hist.numel() > 65536:
                hist = hist[torch.randint(hist.numel(), (65536,))]
        return {
            "mu_mean": mean,
            "mu_std": mu_std,
            "sigma_mean": sigma_mean,
            "spread": (var + sigma_sq) ** 0.5,
            "channels": per_channel,
            "hist": hist,
        }


ACTIVE_STD = 0.1   # std of μ across the data; var 0.01 is the usual active-unit line


def latent_scalars(report, prefix):
    """TensorBoard tags for one tally report. `prefix` is `train` or `val`, the tags `<prefix>_latent/...`.

    Per channel there are only two summaries: how many channels are active,
    and the KL of the weakest one. The full table goes to `latent_table`.
    """
    channels = report["channels"]
    found = {
        f"{prefix}_latent/mu_mean": report["mu_mean"],
        f"{prefix}_latent/mu_std": report["mu_std"],
        f"{prefix}_latent/sigma_mean": report["sigma_mean"],
        f"{prefix}_latent/spread": report["spread"],
        f"{prefix}_latent/active": sum(1 for row in channels if row[0] > ACTIVE_STD),
    }
    kls = [row[2] for row in channels if row[2] is not None]
    if kls:
        found[f"{prefix}_latent/kl_min"] = min(kls)
    return found


def latent_table(report):
    """Markdown table of every channel, for TensorBoard's text tab."""
    rows = [
        "| channel | μ mean | μ std | σ mean | SNR | KL | |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for index, (mu_std, sigma_mean, kl, mean, snr) in enumerate(report["channels"]):
        mark = "" if mu_std > ACTIVE_STD else "collapsed"
        snr_text = "n/a" if snr is None else f"{snr:.1f}"
        kl_text = "n/a" if kl is None else f"{kl:.3f}"
        rows.append(
            f"| {index} | {mean:+.3f} | {mu_std:.3f} | {sigma_mean:.3f} | {snr_text} | {kl_text} | {mark} |"
        )
    return "\n".join(rows)
