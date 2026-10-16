"""Per-content-frame inverse rendering of the reference into vector strands.

    .venv/bin/python -m tools.trace.fit --frames 21 101 345     # prototype frames (content idx)
    .venv/bin/python -m tools.trace.fit --all                   # all 270 content frames
    .venv/bin/python -m tools.trace.fit --all --fit-look        # per-frame look (used for the full run)
    .venv/bin/python -m tools.trace.fit --span 1 269 --fit-look # one parallel chunk

Writes out/traces/cNNNN.npz (strands + haze, see src/traces.py) and
out/traces/look.json, and prints per-frame metrics (exact skimage SSIM of the
960x540 render vs ref/frames). Content frame c (odd) is shown at output
frames c and c+1; the fit uses ref/frames/c.

Per frame:
 1. init: warm start from the previous content frame's fitted strands +
    haze (re-optimised briefly; strands that no longer explain anything die),
    then pixel deconvolution of what is still unexplained and DP ridge tracing
    of it (tools/trace/init.py) adds new strands;
 2. coarse-to-fine optimisation of all vertices (y, small x offset,
    brightness, width), the haze grid (and optionally the look) with Adam on
    the differentiable scorer (tools/trace/metrics.py), first at render scale
    1 (960x540) then at scale 2 (1920x1080 render, 2x2 area downscale: what
    the real 1080p video goes through);
 3. residual re-trace (repeat 1's deconvolution + tracing on the residual),
    re-optimise, prune dead strands / trim dark ends.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src import traces as T  # noqa: E402
from tools.trace import init as I  # noqa: E402
from tools.trace import metrics as M  # noqa: E402

REF_DIR = os.path.join(ROOT, "ref", "frames")
CONTENT = list(range(1, 540, 2))
B_MAX = 3.0                     # brightness cap (real strands are <~1.5)
WARM_VERTEX_CAP = 8000         # warm-started strands kept per frame (by mass)


def isp(v):
    """inverse softplus"""
    v = np.maximum(v, 1e-4)
    return np.where(v > 20, v, np.log(np.expm1(v)))


class FrameFit:
    def __init__(self, c: int, look: dict, device, verbose=True):
        self.c = c
        self.dev = device
        self.look = look
        self.look_t = T.look_tensors(look, device)
        self.ref_bgr = cv2.imread(os.path.join(REF_DIR, f"{c:04d}.png"))
        self.ref = M.RefFrame(self.ref_bgr, device)
        self.verbose = verbose
        self.strands = []            # list of numpy dicts x, y, b, u
        hs = look["haze_step"]
        self.haze = np.zeros((math.ceil(T.TH / hs), math.ceil(T.TW / hs)), np.float32)
        self.canvases = {}
        self.log = []

    def canvas(self, s):
        if s not in self.canvases:
            self.canvases[s] = T.Canvas(s, device=self.dev)
        return self.canvases[s]

    # ------------------------------------------------------------ params
    def _tensors(self):
        st = self.strands
        dev = self.dev
        cat = lambda k: np.concatenate([s[k] for s in st]).astype(np.float32)
        P = {}
        if st:
            # vertices live on integer columns x0 (strictly increasing within a
            # strand) plus a bounded offset, so x stays monotone across warm starts
            x = cat("x")
            x0 = np.round(x)
            P["x0"] = torch.tensor(x0, device=dev)
            off = np.clip((x - x0) / 0.45, -0.99, 0.99)
            P["xi"] = torch.tensor(np.arctanh(off).astype(np.float32), device=dev, requires_grad=True)
            P["y"] = torch.tensor(cat("y"), device=dev, requires_grad=True)
            P["beta"] = torch.tensor(isp(cat("b")).astype(np.float32), device=dev, requires_grad=True)
            u = np.clip(cat("u") / 2, 1e-3, 1 - 1e-3)
            P["om"] = torch.tensor(np.log(u / (1 - u)).astype(np.float32), device=dev, requires_grad=True)
            P["first"] = torch.tensor(np.concatenate([np.arange(len(s["x"])) == 0 for s in st]), device=dev)
            P["last"] = torch.tensor(np.concatenate([np.arange(len(s["x"])) == len(s["x"]) - 1 for s in st]),
                                     device=dev)
        P["hz"] = torch.tensor(isp(self.haze).astype(np.float32), device=dev, requires_grad=True)
        return P

    def _strand_dict(self, P):
        if "x0" not in P:
            return None
        return {"X": P["x0"] + 0.45 * torch.tanh(P["xi"]), "Y": P["y"], "B": F.softplus(P["beta"]).clamp(max=B_MAX),
                "U": 2 * torch.sigmoid(P["om"]), "first": P["first"], "last": P["last"]}

    def _store(self, P):
        if "x0" in P:
            sd = self._strand_dict(P)
            X, Y, B, U = (sd[k].detach().cpu().numpy() for k in ("X", "Y", "B", "U"))
            out, o = [], 0
            for s in self.strands:
                n = len(s["x"])
                out.append({"x": X[o:o + n], "y": Y[o:o + n], "b": B[o:o + n], "u": U[o:o + n]})
                o += n
            self.strands = out
        self.haze = F.softplus(P["hz"]).detach().cpu().numpy()

    # ------------------------------------------------------------ render / loss
    def render(self, P, s, look_t=None):
        look_t = look_t or self.look_t
        cv = self.canvas(s)
        E = T.exposure(cv, self._strand_dict(P), look_t, F.softplus(P["hz"]))
        out = T.tonemap(E, look_t["offset"])
        return T.downscale(out, s)

    def lit(self, out):
        ly_o = cv2.cvtColor(M.quantize_bgr(out), cv2.COLOR_BGR2GRAY)
        return torch.tensor(M.lit_mask(self.ref.ly_u8, ly_o), device=self.dev)

    def loss(self, out, lit, P):
        m = M.metrics(out, self.ref, lit)
        L = (1 - m["ssim"]) + 0.3 * (1 - m["ssim_blur"]) + 0.3 * (1 - m["color"]) + 0.1 * (1 - m["energy"])
        # weak regularisers
        L = L + 2e-4 * F.softplus(P["hz"]).mean()
        if "y" in P:
            y = P["y"]
            inner = ~(P["first"][1:-1] | P["last"][1:-1])
            d2 = (y[2:] - 2 * y[1:-1] + y[:-2])[inner]
            L = L + 1e-6 * d2.abs().sum() / max(1, len(self.strands))
        return L, m

    def optimise(self, iters, s, lr_scale=1.0, fit_look=False, tag=""):
        P = self._tensors()
        groups = [{"params": [P["hz"]], "lr": 0.05 * lr_scale}]
        if "x0" in P:
            groups += [{"params": [P["y"]], "lr": 0.15 * lr_scale},
                       {"params": [P["xi"]], "lr": 0.05 * lr_scale},
                       {"params": [P["beta"]], "lr": 0.05 * lr_scale},
                       {"params": [P["om"]], "lr": 0.1 * lr_scale}]
        look_p = None
        if fit_look:
            look_p = {k: v.clone().requires_grad_(True) for k, v in self.look_t.items()}
            groups.append({"params": [look_p["A"], look_p["haze_rgb"]], "lr": 0.003 * lr_scale})
            groups.append({"params": [look_p["sigma"], look_p["soft_sigma"]], "lr": 0.003 * lr_scale})
            groups.append({"params": [look_p["offset"]], "lr": 0.0002 * lr_scale})
        opt = torch.optim.Adam(groups)
        lit = None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.1 * groups[0]["lr"])
        for it in range(iters):
            lt = look_p if fit_look else self.look_t
            out = self.render(P, s, lt)
            if lit is None or it % 20 == 0:
                lit = self.lit(out)
            L, m = self.loss(out, lit, P)
            opt.zero_grad()
            L.backward()
            if "y" in P:                      # dark vertices have no image gradient: freeze them
                with torch.no_grad():
                    yk = P["y"].clone()
            opt.step()
            if "y" in P:
                with torch.no_grad():
                    dark = F.softplus(P["beta"]) < 0.004
                    P["y"].copy_(torch.where(dark, yk, P["y"]))   # (no data-dependent shapes on MPS)
            if fit_look:
                with torch.no_grad():         # keep the look physical
                    look_p["A"].clamp_(min=0)
                    look_p["sigma"].clamp_(min=0.3)
                    look_p["soft_sigma"].clamp_(min=0.5)
                    look_p["haze_rgb"].clamp_(min=0)
                    look_p["offset"].clamp_(0, 0.05)
            sched.step()
        self._store(P)
        if fit_look:
            self.look_t = {k: v.detach() for k, v in look_p.items()}
        if self.verbose:
            nv = sum(len(q["x"]) for q in self.strands)
            print(f"    [{tag} s{s} x{iters} nv {nv}] " + " ".join(f"{k} {float(v.detach()):.4f}" for k, v in m.items()),
                  flush=True)
        return m

    # ------------------------------------------------------------ init / trace
    def current_exposure(self, s=1):
        with torch.no_grad():
            P = self._tensors()
            cv = self.canvas(s)
            return T.exposure(cv, self._strand_dict(P), self.look_t, F.softplus(P["hz"]))

    def residual_core(self, iters=200):
        E0 = self.current_exposure(1) if self.strands else None
        L, hz = I.deconvolve(self.ref.rgb, self.look_t, iters=iters, E_fixed=E0)
        return L, hz

    def add_traced(self, L, min_gain=0.5, max_strands=30):
        runs = I.trace(L, min_gain=min_gain, max_strands=max_strands)
        new = runs_to_strands(runs, L)
        self.strands += new
        return len(new)

    def prune(self, bmin=0.004, min_len=4, gap=20):
        out = []
        for s in self.strands:
            on = np.nonzero(s["b"] > bmin)[0]
            if len(on) < min_len:
                continue
            # split at long interior dark gaps
            gaps = np.nonzero(np.diff(on) > gap)[0]
            for piece in np.split(on, gaps + 1):
                if len(piece) < min_len:
                    continue
                a, b = max(0, piece[0] - 1), min(len(s["x"]), piece[-1] + 2)
                out.append({k: v[a:b] for k, v in s.items()})
        n = len(self.strands) - len(out)
        self.strands = out
        return n

    def cap_vertices(self, cap):
        """Keep the highest-mass strands within a total vertex budget."""
        mass = [float((s["b"] * np.hypot(np.gradient(s["x"]), np.gradient(s["y"]))).sum())
                if len(s["x"]) > 1 else 0.0 for s in self.strands]
        keep, tot = [], 0
        for i in np.argsort(mass)[::-1]:
            n = len(self.strands[i]["x"])
            if tot + n <= cap:
                keep.append(i)
                tot += n
        self.strands = [self.strands[i] for i in sorted(keep)]

    def exact_metrics(self, s=2):
        """Exact tools/score.py metrics of the quantised render (no codec)."""
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import score
        with torch.no_grad():
            out = self.render(self._tensors(), s)
        return score.frame_metrics(self.ref_bgr, M.quantize_bgr(out))

    def save(self, d=T.TRACE_DIR, info=None):
        T.save_trace(T.frame_path(self.c, d), self.strands, self.haze, info)


def runs_to_strands(runs, L):
    """DP runs (integer columns/rows) -> strand dicts with sub-pixel y and initial b."""
    Ln = L.cpu().numpy()
    H, W = Ln.shape
    mass = cv2.boxFilter(Ln, -1, (3, 3), normalize=False) / 3.0
    out = []
    for r in runs:
        xs = r["xs"].astype(np.float32)
        rows = r["rows"].astype(np.int64)
        y = rows.astype(np.float32)
        b = np.zeros(len(xs), np.float32)
        for i, (x, row) in enumerate(zip(r["xs"], rows)):
            # sub-pixel centroid for shallow parts
            dyl = abs(rows[i] - rows[i - 1]) if i > 0 else 0
            dyr = abs(rows[i + 1] - rows[i]) if i + 1 < len(rows) else 0
            if max(dyl, dyr) <= 1:
                lo, hi = max(0, row - 2), min(H, row + 3)
                col = Ln[lo:hi, x]
                if col.sum() > 1e-6:
                    y[i] = float((col * np.arange(lo, hi)).sum() / col.sum())
            b[i] = mass[row, x]
        b = np.maximum(b, 0.02)
        out.append({"x": xs, "y": y, "b": b, "u": np.full(len(xs), 0.3, np.float32)})
    return out


# ------------------------------------------------------------------ driver
class Timer:
    def __init__(self):
        self.t = time.time()
        self.d = {}

    def __call__(self, k):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        now = time.time()
        self.d[k] = self.d.get(k, 0) + now - self.t
        self.t = now


def fit_frame(c, look, dev, prev=None, verbose=True, fit_look=False, budget=1.0):
    t0 = time.time()
    tm = Timer()
    if prev is not None and fit_look:
        look = prev.info["look"]            # per-frame look, warm-started
    ff = FrameFit(c, look, dev, verbose)
    n_warm = 0
    if prev is not None:
        ff.strands = [dict(s) for s in prev.strands]
        ff.haze = prev.haze.copy()
        n_warm = len(ff.strands)
        ff.optimise(int(60 * budget), 1, tag="warm")
        ff.prune(bmin=0.02)
        ff.cap_vertices(WARM_VERTEX_CAP)
        tm("warm")
    L, _ = ff.residual_core()
    tm("deconv")
    n1 = ff.add_traced(L)
    tm("trace")
    ff.optimise(int(150 * budget), 1, tag="fit1", fit_look=fit_look)
    ff.prune()
    tm("opt1")
    L, _ = ff.residual_core(150)
    tm("deconv")
    n2 = ff.add_traced(L, min_gain=0.4)
    tm("trace")
    ff.optimise(int(150 * budget), 1, tag="fit2", fit_look=fit_look)
    ff.prune()
    tm("opt1")
    m = ff.optimise(int(150 * budget), 2, lr_scale=0.5, tag="fine", fit_look=fit_look)
    ff.prune()
    tm("opt2")
    ex = ff.exact_metrics()
    if verbose:
        print("    time: " + " ".join(f"{k} {v:.1f}s" for k, v in tm.d.items()))
    dt = time.time() - t0
    if verbose:
        print(f"  c{c:04d}: warm {n_warm} +{n1} +{n2} -> {len(ff.strands)} strands, "
              f"exact ssim {ex['ssim']:.4f} blur {ex['ssim_blur']:.4f} color {ex['color']:.4f} "
              f"energy {ex['energy']:.4f}  ({dt:.0f}s)", flush=True)
    ff.info = {"c": c, "n_strands": len(ff.strands), "time": dt, **{k: float(v) for k, v in ex.items()}}
    ff.info["look"] = look_dict(ff.look_t, look)
    return ff


def load_fit(c, look, dev, d):
    """A FrameFit holding a saved trace (warm start across process restarts)."""
    strands, haze, info = T.load_trace(T.frame_path(c, d))
    ff = FrameFit(c, info.get("look", look), dev, verbose=False)
    ff.strands, ff.haze, ff.info = strands, haze, info
    if "look" not in ff.info:
        ff.info["look"] = look
    return ff


def look_dict(look_t, base):
    d = dict(base)
    for k in ("sigma", "A", "soft_sigma", "offset", "haze_rgb"):
        d[k] = look_t[k].detach().cpu().numpy().tolist()
    return d


def report(d):
    rows = []
    for c in CONTENT:
        p = T.frame_path(c, d)
        if os.path.exists(p):
            rows.append(T.load_trace(p)[2])
    keys = ["ssim", "ssim_blur", "color", "energy"]
    print(f"{len(rows)} content frames fitted; exact metrics of the 2x render (no codec)")
    for s0 in range(9):
        seg = [r for r in rows if s0 * 60 <= r["c"] < (s0 + 1) * 60]
        if seg:
            print(f"  t={12 + s0}-{13 + s0}s  " + "  ".join(f"{k}={np.mean([r[k] for r in seg]):.4f}" for k in keys)
                  + f"  strands={np.mean([r['n_strands'] for r in seg]):.0f}  time={np.mean([r['time'] for r in seg]):.0f}s")
    print("ALL  " + "  ".join(f"{k}={np.mean([r[k] for r in rows]):.4f}" for k in keys)
          + f"  min ssim {min(r['ssim'] for r in rows):.4f} (c{min(rows, key=lambda r: r['ssim'])['c']})"
          + f"  total fit time {sum(r['time'] for r in rows) / 3600:.2f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, nargs="*", help="content frames (odd output indices)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--span", type=int, nargs=2, help="content frames in [a, b] (for parallel chunks)")
    ap.add_argument("--out", default=T.TRACE_DIR)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="keep frames already in --out (resume)")
    ap.add_argument("--fit-look", action="store_true", help="also fit the global look per frame")
    ap.add_argument("--budget", type=float, default=1.0, help="iteration multiplier")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--report", action="store_true", help="summarise the saved per-frame fit metrics")
    args = ap.parse_args()
    if args.report:
        return report(args.out)
    dev = torch.device("cpu") if args.cpu else T.default_device()
    look = T.load_look(args.out)
    if args.span:
        frames = [c for c in CONTENT if args.span[0] <= c <= args.span[1]]
    else:
        frames = CONTENT if args.all else [T.content_index(f) for f in args.frames]
    prev = None
    rows = []
    for c in frames:
        use_prev = prev if (prev is not None and not args.no_warm and prev.c == c - 2) else None
        if use_prev is None and not args.no_warm and os.path.exists(T.frame_path(c - 2, args.out)):
            use_prev = load_fit(c - 2, look, dev, args.out)      # resume / chunk boundary
        if args.skip_done and os.path.exists(T.frame_path(c, args.out)):
            prev = load_fit(c, look, dev, args.out)
            continue
        ff = fit_frame(c, look, dev, use_prev, fit_look=args.fit_look, budget=args.budget)
        ff.save(args.out, ff.info)
        rows.append(ff.info)
        prev = ff
        if dev.type == "mps":
            torch.mps.empty_cache()
        if args.fit_look:
            look = ff.info["look"]          # chain the look estimate to the next frame
    if args.fit_look and rows:
        med = dict(look)
        for k in ("sigma", "A", "soft_sigma", "offset", "haze_rgb"):
            med[k] = np.median(np.array([r["look"][k] for r in rows]), 0).tolist()
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "look.json"), "w") as f:
            json.dump(med, f, indent=1)
        print("look.json:", json.dumps(med))
    if rows:
        print("MEAN exact ssim %.4f over %d frames" % (np.mean([r["ssim"] for r in rows]), len(rows)))


if __name__ == "__main__":
    main()
