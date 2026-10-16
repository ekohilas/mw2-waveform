"""Initialisation for the strand fit: pixel deconvolution + DP ridge tracing.

deconvolve(): non-negative core image L (960x540) + low-res haze such that
  tonemap(K * L + haze) ~= reference (projected Adam on RGB MSE + L1).
trace(): repeatedly extracts the best path through L by dynamic programming
  over columns (state = row, plus an 'off' state, so a path is a set of runs);
  a transition (x, y) -> (x+1, y') collects the core mass along the vertical
  span between the two rows, so near-vertical spikes are followed, minus a
  per-pixel threshold tau and a small slope penalty. Each found path is erased
  from L (within +-r px) before the next one.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src import traces as T


def deconvolve(ref_rgb: torch.Tensor, look_t: dict, iters: int = 200, lam: float = 0.4,
               haze_lam: float = 0.05, lr: float = 0.05, L0=None, E_fixed=None):
    """ref_rgb [3, 540, 960] 0..255 on device. Returns (L [540, 960], haze [h, w])."""
    dev = ref_rgb.device
    cv = T.Canvas(1, device=dev)
    p = cv.pad
    L = torch.zeros(cv.Hp, cv.Wp, device=dev)
    if L0 is not None:
        L[p:p + cv.H, p:p + cv.W] = L0
    L.requires_grad_(True)
    hs = 8
    hh, hw = math.ceil(T.TH / hs), math.ceil(T.TW / hs)
    hz = torch.zeros(hh, hw, device=dev, requires_grad=True)
    gh = [cv.gauss_hat(look_t["sigma"][k]) for k in range(3)]
    Kc = [sum(look_t["A"][k, c] * gh[k] for k in range(3)) for c in range(3)]
    opt = torch.optim.Adam([L, hz], lr=lr)
    tgt = ref_rgb / 255.0
    for it in range(iters):
        FL = torch.fft.rfft2(L)
        E = cv.crop(torch.stack([torch.fft.irfft2(FL * Kc[c], s=(cv.Hp, cv.Wp)) for c in range(3)]))
        hzu = F.interpolate(hz.clamp(min=0)[None, None], size=(cv.H, cv.W), mode="bicubic",
                            align_corners=False)[0, 0].clamp(min=0)
        E = E + look_t["haze_rgb"].view(3, 1, 1) * hzu
        if E_fixed is not None:
            E = E + E_fixed
        out = (1 - torch.exp(-E) - look_t["offset"]).clamp(max=1)
        # below-black is free where the target is black (no clamp: keeps gradients alive)
        out = torch.where((out < 0) & (tgt <= 0), tgt, out)
        loss = ((out - tgt) ** 2).mean() * 100 + lam * 1e-3 * L.clamp(min=0).sum() / 1000 \
            + haze_lam * 1e-3 * hz.clamp(min=0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            L.clamp_(min=0)
            hz.clamp_(min=0)
    return L.detach()[p:p + cv.H, p:p + cv.W], hz.detach()


ANG_BINS = 13                     # path-direction bins over (-90, 90) degrees


def _bin_of_dy(dy):
    ang = torch.atan(dy.float())                            # -pi/2..pi/2
    return torch.clamp(((ang / math.pi + 0.5) * ANG_BINS).long(), 0, ANG_BINS - 1)


def ridge_maps(L: torch.Tensor, sigma: float = 1.2, cos_min: float = 0.85):
    """Per-direction-bin gain maps [ANG_BINS, H, W]: ridge-centre core mass x
    alignment of the local line tangent with the bin's path direction."""
    import cv2
    Ln = L.cpu().numpy()
    Lb = cv2.GaussianBlur(Ln, (0, 0), sigma)
    Lxx = cv2.Sobel(Lb, cv2.CV_32F, 2, 0, ksize=3)
    Lyy = cv2.Sobel(Lb, cv2.CV_32F, 0, 2, ksize=3)
    Lxy = cv2.Sobel(Lb, cv2.CV_32F, 1, 1, ksize=3)
    a, b, c = Lxx, Lxy, Lyy
    mu = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2)     # most negative eigenvalue
    nx, ny = b, mu - a                                             # its eigenvector = line normal
    alt = np.abs(b) < 1e-9
    nx = np.where(alt, np.where(a <= c, 1.0, 0.0), nx)
    ny = np.where(alt, np.where(a <= c, 0.0, 1.0), ny)
    nrm = np.sqrt(nx ** 2 + ny ** 2) + 1e-12
    nx, ny = (nx / nrm).astype(np.float32), (ny / nrm).astype(np.float32)
    tx, ty = -ny, nx                                               # tangent, canonical tx >= 0
    sg = np.where(tx < 0, -1.0, 1.0).astype(np.float32)
    tx, ty = tx * sg, ty * sg
    H, W = Lb.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    a1 = cv2.remap(Lb, gx + nx, gy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a2 = cv2.remap(Lb, gx - nx, gy - ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ridge = (Lb >= a1) & (Lb >= a2) & (mu < 0)
    mass = cv2.boxFilter(Ln, -1, (3, 3), normalize=False) / 3.0 * ridge
    out = np.zeros((ANG_BINS, H, W), np.float32)
    for k in range(ANG_BINS):
        ang = ((k + 0.5) / ANG_BINS - 0.5) * math.pi
        dx, dy = math.cos(ang), math.sin(ang)
        al = np.abs(tx * dx + ty * dy)
        out[k] = mass * np.clip((al - cos_min) / (1 - cos_min), 0, 1)
    return torch.tensor(out, device=L.device), torch.tensor(mass, device=L.device)


def _dp_path(Q: torch.Tensor, lam: float, kappa: float, ylo: int, yhi: int):
    """Best path restricted to rows [ylo, yhi). Q [ANG_BINS, H, W]: per-pixel
    gain for each path-direction bin (already minus tau). A transition
    (x, y) -> (x+1, y') collects the pixels of the vertical span between the
    rows (half in each column) from the bin of its direction atan(y' - y).

    Returns rows (int array, -1 where off) and the path gain."""
    dev = Q.device
    Hh = yhi - ylo
    W = Q.shape[2]
    S = torch.cat([torch.zeros(ANG_BINS, 1, W, device=dev), torch.cumsum(Q[:, ylo:yhi], 1)], 1)
    yy = torch.arange(Hh, device=dev)
    Y0 = yy[:, None].expand(Hh, Hh)                         # from row y
    Y1 = yy[None, :].expand(Hh, Hh)                         # to row y'
    K = _bin_of_dy(Y1 - Y0)
    up = Y1 >= Y0
    mu = (Y0 + Y1) // 2
    md = (Y0 + Y1 + 1) // 2
    same = Y0 == Y1
    pen = lam * (torch.abs(Y1 - Y0) - 1).clamp(min=0).float()
    k0 = ANG_BINS // 2
    bp = torch.zeros(W, Hh, dtype=torch.long, device=dev)       # on-state predecessor row, -1 = off
    bpo = torch.zeros(W, dtype=torch.long, device=dev)          # off-state predecessor row, -1 = off
    V = Q[k0, ylo:yhi, 0] - kappa
    O = torch.zeros((), device=dev)
    for x in range(W - 1):
        Sx, Sn = S[:, :, x], S[:, :, x + 1]
        # y' > y: col x (y, m], col x+1 (m, y'];  y' == y: col x+1 [y, y]
        g_up = (Sx[K, mu + 1] - Sx[K, Y0 + 1]) + (Sn[K, Y1 + 1] - Sn[K, torch.where(same, Y1, mu + 1)])
        # y' < y: col x [m, y), col x+1 [y', m)
        g_dn = (Sx[K, Y0] - Sx[K, md]) + (Sn[K, md] - Sn[K, Y1])
        G = torch.where(up, g_up, g_dn) - pen
        best, arg = (V[:, None] + G).max(0)
        enter = O + Q[k0, ylo:yhi, x + 1] - kappa
        te = enter > best
        bp[x + 1] = torch.where(te, torch.full_like(arg, -1), arg)
        vmax, varg = V.max(0)
        tl = (vmax - kappa) > O
        bpo[x + 1] = torch.where(tl, varg, torch.full_like(varg, -1))
        O = torch.where(tl, vmax - kappa, O)
        V = torch.where(te, enter, best)
    vmax, varg = V.max(0)
    bp = bp.cpu().numpy()
    bpo = bpo.cpu().numpy()
    vmax, varg, O = float(vmax), int(varg), float(O)
    rows = np.full(W, -1, np.int64)
    if vmax - kappa > O:
        state, total = varg, vmax - kappa
    else:
        state, total = -1, O
    for x in range(W - 1, -1, -1):
        if state >= 0:
            rows[x] = state + ylo
            state = int(bp[x, state]) if x > 0 else -1
        else:
            state = int(bpo[x]) if x > 0 else -1
    return rows, total


def path_pixels(xs, rows):
    """Pixels covered by a run (columns xs, rows), incl. vertical spans: list of (y0, y1, x)."""
    out = []
    for i, x in enumerate(xs):
        y = rows[i]
        if i + 1 < len(xs):
            y2 = rows[i + 1]
            if y2 >= y:
                m = (y + y2) // 2
                out.append((y, m, x))
                out.append((m + 1 if y2 > y else y2, y2, x + 1))
            else:
                m = (y + y2 + 1) // 2
                out.append((m, y, x))
                out.append((y2, m - 1, x + 1))
        else:
            out.append((y, y, x))
    return out


def trace(L: torch.Tensor, tau: float = 0.03, lam: float = 0.005, kappa: float = 0.5,
          erase_r: int = 1, max_strands: int = 40, min_gain: float = 1.0, min_len: int = 6,
          verbose=False):
    """Greedy DP strand extraction from a core image L [540, 960] (device tensor).

    Returns a list of runs: dicts with integer columns xs and rows."""
    Hh, W = L.shape
    Qb, mass = ridge_maps(L)
    lit = (mass > tau).any(1).nonzero()
    if len(lit) == 0:
        return []
    ylo = max(0, int(lit.min()) - 3)
    yhi = min(Hh, int(lit.max()) + 4)
    erased = np.zeros((Hh, W), bool)
    runs = []
    for k in range(max_strands):
        Qk = Qb.masked_fill(torch.tensor(erased, device=L.device)[None], 0) - tau
        rows, total = _dp_path(Qk, lam, kappa, ylo, yhi)
        if total < min_gain:
            break
        on = rows >= 0
        xs = np.nonzero(on)[0]
        if len(xs) == 0:
            break
        brk = np.nonzero(np.diff(xs) > 1)[0]
        for seg in np.split(xs, brk + 1):
            if len(seg) >= min_len:
                runs.append({"xs": seg, "rows": rows[seg]})
            for (y0, y1, x) in path_pixels(seg, rows[seg]):
                erased[max(0, y0 - erase_r):y1 + erase_r + 1, max(0, x - erase_r):x + erase_r + 1] = True
        if verbose:
            print(f"  path {k}: gain {total:.2f}, on {on.sum()} cols, runs {len(runs)}")
    return runs
