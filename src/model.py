"""
model.py
========
MuonNet: two-detector 1D residual CNN with three outputs per element
  * presence logit          (multi-label: which elements are there, >= 1 wt%)
  * composition logit       (softmax over elements -> wt fractions, metal basis)
  * composition uncertainty (log sigma of the predicted wt fraction)

Design notes (what changed w.r.t. v1)
-------------------------------------
* v1 ended in global average pooling, which makes a CNN translation
  invariant: it can see *that* a peak exists but not *where* it is -- yet the
  energy of a muonic line is exactly what identifies the element. MuonNet
  appends fixed energy-coordinate channels (CoordConv) and keeps the spatial
  axis until a position-aware flatten + linear layer.
* The low-energy (0.5 keV bins) and high-energy (2 keV bins) HPGe spectra
  get their own branches, like the two detector types at RIKEN-RAL.
* Beam momentum and observed count totals enter as auxiliary scalars
  (momentum decides how many muons stop in air / the Al packet).
* Residual blocks with dilations give a large receptive field (a muonic
  series spans hundreds of keV) at modest cost on a CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int, k: int = 5, drop: float = 0.0):
        super().__init__()
        p = dilation * (k // 2)
        self.c1 = nn.Conv1d(c, c, k, padding=p, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, k, padding=p, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = F.gelu(self.b1(self.c1(x)))
        y = self.drop(y)
        y = self.b2(self.c2(y))
        return F.gelu(x + y)


class Branch(nn.Module):
    def __init__(self, length: int, widths, n_down: int, embed: int, drop: float):
        super().__init__()
        e = torch.linspace(-1, 1, length)
        self.register_buffer("coords", torch.stack([e, e.abs() * 2 - 1])[None])   # (1,2,L)
        # stride-2 stem: peaks span 2-4 bins; the stem's 9-tap filters keep
        # their shape information in channels while halving the compute
        layers = [nn.Conv1d(3, widths[0], 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(widths[0]), nn.GELU()]
        c = widths[0]
        for i in range(n_down):
            w = widths[min(i + 1, len(widths) - 1)]
            layers += [ResBlock(c, 1 + (i % 3), drop=drop),
                       nn.Conv1d(c, w, 4, stride=2, padding=1, bias=False), nn.BatchNorm1d(w), nn.GELU()]
            c = w
        layers += [ResBlock(c, 2, drop=drop)]
        self.body = nn.Sequential(*layers)
        with torch.no_grad():
            L = self.body(torch.zeros(1, 3, length)).shape[-1]
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(c * L, embed), nn.GELU())

    def forward(self, x):
        x = torch.cat([x[:, None, :], self.coords.expand(x.shape[0], -1, -1)], dim=1)
        return self.head(self.body(x))


class MuonNet(nn.Module):
    def __init__(self, n_elements: int, le_len: int, he_len: int, width: int = 24, embed: int = 256,
                 hidden: int = 384, dropout: float = 0.15, n_aux: int = 3):
        super().__init__()
        w = [width, int(width * 1.5), width * 2, int(width * 2.5), width * 3, width * 3]
        self.le = Branch(le_len, w, n_down=5, embed=embed, drop=dropout * 0.5)
        self.he = Branch(he_len, w, n_down=6, embed=embed, drop=dropout * 0.5)
        self.trunk = nn.Sequential(
            nn.Linear(2 * embed + n_aux, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.presence = nn.Linear(hidden, n_elements)
        self.comp = nn.Linear(hidden, n_elements)
        self.logsig = nn.Linear(hidden, n_elements)
        self.config = dict(n_elements=n_elements, le_len=le_len, he_len=he_len, width=width, embed=embed,
                           hidden=hidden, dropout=dropout, n_aux=n_aux)

    def forward(self, le, he, aux):
        h = self.trunk(torch.cat([self.le(le), self.he(he), aux], dim=1))
        return self.presence(h), self.comp(h), self.logsig(h).clamp(-7.0, 0.0)


def multitask_loss(out, frac, present, mask, w_comp: float = 3.0, w_nll: float = 0.1):
    """
    presence  : masked BCE (elements between 0.2 and 1 wt% are not penalised)
    comp      : KL(true || predicted) on the wt-fraction simplex; its gradient
                w.r.t. the logits is (p - f), i.e. an absolute-fraction error
    uncertainty: Gaussian NLL of the (detached) prediction error -> sigma
    """
    p_logit, c_logit, logsig = out
    bce = F.binary_cross_entropy_with_logits(p_logit, present, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1)
    logp = F.log_softmax(c_logit, dim=1)
    kl = (frac * (torch.log(frac.clamp(min=1e-8)) - logp)).sum(1).mean()
    pred = logp.exp().detach()
    sig = logsig.exp() + 1e-3
    nll = (0.5 * ((frac - pred) / sig) ** 2 + torch.log(sig)).mean()
    total = bce + w_comp * kl + w_nll * nll
    return total, dict(bce=bce.item(), kl=kl.item(), nll=nll.item())


if __name__ == "__main__":
    import time
    from config import HE_NBINS, LE_NBINS, NUM_TARGETS
    m = MuonNet(NUM_TARGETS, LE_NBINS, HE_NBINS)
    n = sum(p.numel() for p in m.parameters())
    le, he, aux = torch.randn(32, LE_NBINS), torch.randn(32, HE_NBINS), torch.randn(32, 3)
    t0 = time.time()
    out = m(le, he, aux)
    loss, parts = multitask_loss(out, torch.softmax(torch.randn(32, NUM_TARGETS), 1),
                                 torch.randint(0, 2, (32, NUM_TARGETS)).float(), torch.ones(32, NUM_TARGETS))
    loss.backward()
    t1 = time.time()
    for _ in range(3):
        out = m(le, he, aux)
        multitask_loss(out, torch.softmax(torch.randn(32, NUM_TARGETS), 1),
                       torch.randint(0, 2, (32, NUM_TARGETS)).float(), torch.ones(32, NUM_TARGETS))[0].backward()
    print("outputs:", [tuple(o.shape) for o in out], "params: %.2fM" % (n / 1e6),
          "steady fwd+bwd batch32: %.2fs" % ((time.time() - t1) / 3), parts)
