import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
 
class HuberLoss(nn.Module):
    """Standard Huber loss, kept for baseline comparison."""
 
    def __init__(self, delta=3.0):
        super().__init__()
        self.delta = delta
 
    def forward(self, pred, target):
        if pred.dim() == target.dim() + 1:
            pred = pred.squeeze(1)
        mask = target != 0
        pred, target = pred[mask], target[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        error = target - pred
        loss = torch.where(
            error.abs() < self.delta,
            0.5 * error ** 2,
            self.delta * (error.abs() - 0.5 * self.delta)
        )
        return loss.mean()
 
 
def _norm_huber(error, delta):
    """Huber loss normalized by delta: gradient magnitude is bounded by 1
    regardless of delta, so classes with different height scales produce
    comparable gradients."""
    abs_e = error.abs()
    loss = torch.where(abs_e < delta, 0.5 * error ** 2, delta * (abs_e - 0.5 * delta))
    return loss / delta
 
 
def _gaussian_kernel1d(ks, sigma):
    x = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2.0
    k = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
    return k / k.sum()
 
 
class VegHeightLoss(nn.Module):
    """
    Three-class canopy height regression loss.
 
    Label classes:
        label == 0                          -> invalid pixel, excluded
        0 < label < low_min                 -> I  non-vegetated
        low_min <= label < forest_thr       -> II  low vegetation
        label >= forest_thr                 -> III forest
 
    Each class uses a scale-normalized Huber loss with its own delta,
    averaged within the class and combined with class weights. The forest
    class is additionally reweighted with Label Distribution Smoothing (LDS)
    to counter the imbalance between the dominant 10-20 m range and the
    sparse upper tail.
    """
 
    def __init__(self,
                 low_min=0.1,
                 forest_thr=3.0,
                 delta_nonveg=0.5,
                 delta_low=1.0,
                 delta_forest=3.0,
                 w_nonveg=1.0,
                 w_low=1.0,
                 w_forest=1.0,
                 use_lds=True,
                 lds_hmin=3.0,
                 lds_hmax=50.0,
                 lds_bins=47,
                 lds_ks=5,
                 lds_sigma=2.0,
                 lds_alpha=0.5,
                 lds_wmax=8.0,
                 lds_momentum=0.02):
        super().__init__()
        self.low_min = low_min
        self.forest_thr = forest_thr
 
        self.delta_nonveg = delta_nonveg
        self.delta_low = delta_low
        self.delta_forest = delta_forest
 
        self.w_nonveg = w_nonveg
        self.w_low = w_low
        self.w_forest = w_forest
 
        self.use_lds = use_lds
        self.lds_hmin = lds_hmin
        self.lds_hmax = lds_hmax
        self.lds_bins = int(lds_bins)
        self.lds_alpha = lds_alpha
        self.lds_wmax = lds_wmax
        self.lds_momentum = lds_momentum
        self.bin_w = (lds_hmax - lds_hmin) / float(self.lds_bins)
 
        self.register_buffer('lds_hist', torch.ones(self.lds_bins) / self.lds_bins)
        self.register_buffer('lds_kernel', _gaussian_kernel1d(lds_ks, lds_sigma))
        self.register_buffer('lds_fitted', torch.zeros(1))
 
        self.last_terms = {}
 
    def _bin_index(self, h):
        idx = ((h - self.lds_hmin) / self.bin_w).long()
        return idx.clamp_(0, self.lds_bins - 1)
 
    @torch.no_grad()
    def fit_lds(self, dataloader, max_batches=None, verbose=True):
        """Estimate the forest height histogram from the training set before
        training starts (recommended over online estimation)."""
        hist = torch.zeros(self.lds_bins)
        total = 0
        for i, batch in enumerate(dataloader):
            labels = batch[1] if isinstance(batch, (list, tuple)) else batch
            t = labels.detach().float().reshape(-1)
            t = t[t >= self.forest_thr]
            if t.numel() == 0:
                continue
            idx = self._bin_index(t.cpu())
            hist += torch.bincount(idx, minlength=self.lds_bins).float()
            total += t.numel()
            if max_batches is not None and (i + 1) >= max_batches:
                break
        if total > 0:
            self.lds_hist.copy_((hist / hist.sum()).to(self.lds_hist.device))
            self.lds_fitted.fill_(1.0)
            if verbose:
                print(f"[LDS] fitted on {total} forest pixels")
        elif verbose:
            print("[LDS] warning: no forest pixels found, falling back to online estimate")
        return self
 
    def _smoothed_density(self):
        h = self.lds_hist.view(1, 1, -1)
        k = self.lds_kernel.view(1, 1, -1).to(h.dtype)
        p = F.conv1d(h, k, padding=self.lds_kernel.numel() // 2).view(-1)
        return p / p.sum().clamp_min(1e-12)
 
    @torch.no_grad()
    def _update_hist(self, idx):
        b = torch.bincount(idx, minlength=self.lds_bins).float()
        s = b.sum()
        if s <= 0:
            return
        b = (b / s).to(self.lds_hist.device)
        self.lds_hist.mul_(1 - self.lds_momentum).add_(b, alpha=self.lds_momentum)
 
    def _lds_weight(self, t_forest):
        """Inverse-density weight, mean-normalized to 1 before and after
        truncation so the overall loss scale stays fixed."""
        idx = self._bin_index(t_forest.detach())
        if self.training and self.lds_fitted.item() < 0.5:
            self._update_hist(idx)
        p = self._smoothed_density()
        w = (1.0 / p[idx].clamp_min(1e-6)) ** self.lds_alpha
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(max=self.lds_wmax)
        w = w / w.mean().clamp_min(1e-12)
        return w.detach()
 
    def forward(self, pred, target):
        if pred.dim() == target.dim() + 1:
            pred = pred.squeeze(1)
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
 
        valid = target > 0
        pred, target = pred[valid], target[valid]
        zero = pred.sum() * 0.0
        if pred.numel() == 0:
            self.last_terms = {}
            return zero
 
        m_nonveg = target < self.low_min
        m_forest = target >= self.forest_thr
        m_low = (~m_nonveg) & (~m_forest)
 
        terms = {}
        total = zero
 
        if m_nonveg.any():
            e = target[m_nonveg] - pred[m_nonveg]
            l_nonveg = _norm_huber(e, self.delta_nonveg).mean()
            total = total + self.w_nonveg * l_nonveg
            terms['nonveg'] = float(l_nonveg.detach())
 
        if m_low.any():
            e = target[m_low] - pred[m_low]
            l_low = _norm_huber(e, self.delta_low).mean()
            total = total + self.w_low * l_low
            terms['low'] = float(l_low.detach())
 
        if m_forest.any():
            p_f, t_f = pred[m_forest], target[m_forest]
            e = t_f - p_f
            base = _norm_huber(e, self.delta_forest)
            if self.use_lds:
                w = self._lds_weight(t_f)
                l_forest = (base * w).mean()
            else:
                l_forest = base.mean()
            total = total + self.w_forest * l_forest
            terms['forest'] = float(l_forest.detach())
 
        self.last_terms = terms
        return total
 
 
@torch.no_grad()
def height_metrics(pred, target, low_min=0.1, forest_thr=3.0):
    """Per-batch squared-error sums and pixel counts, for accumulation
    into RMSE across an epoch."""
    if pred.dim() == target.dim() + 1:
        pred = pred.squeeze(1)
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    valid = target > 0
    pred, target = pred[valid], target[valid]
 
    out = {}
    if pred.numel() == 0:
        return out
 
    err2 = (pred - target) ** 2
    out['all'] = (float(err2.sum()), int(err2.numel()))
 
    m_nonveg = target < low_min
    m_forest = target >= forest_thr
    m_low = (~m_nonveg) & (~m_forest)
 
    if m_low.any():
        out['low'] = (float(err2[m_low].sum()), int(m_low.sum()))
    if m_forest.any():
        out['forest'] = (float(err2[m_forest].sum()), int(m_forest.sum()))
    if m_nonveg.any():
        correct = (pred[m_nonveg] < low_min).sum()
        out['nonveg_acc'] = (float(correct), int(m_nonveg.sum()))
 
    return out
 
 
def merge_metrics(acc, new):
    for k, v in new.items():
        a = acc.setdefault(k, [0.0, 0])
        a[0] += v[0]
        a[1] += v[1]
    return acc
 
 
def format_metrics(acc):
    parts = []
    for k in ['all', 'low', 'forest']:
        if k in acc and acc[k][1] > 0:
            parts.append(f"RMSE_{k}: {np.sqrt(acc[k][0] / acc[k][1]):.3f}m")
    if 'nonveg_acc' in acc and acc['nonveg_acc'][1] > 0:
        parts.append(f"NonVegAcc: {acc['nonveg_acc'][0] / acc['nonveg_acc'][1]:.3f}")
    return ", ".join(parts)