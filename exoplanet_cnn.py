"""
Kepler Transit Vetting with a Two-Branch 1D CNN
------------------------------------------------
Classifies Kepler Threshold Crossing Events (TCEs) as PLANET vs FALSE POSITIVE
(eclipsing binaries, stellar variability, instrument artifacts).

Pipeline
  1. Data    : real Kepler light curves via `lightkurve` + labels from the NASA
               Exoplanet Archive KOI table (CONFIRMED vs FALSE POSITIVE).
               Falls back to a physics-based SYNTHETIC generator for offline demo.
  2. Prep    : flatten -> phase-fold on period -> bin into a GLOBAL view (2001 bins)
               and a LOCAL view (201 bins zoomed on transit)  [AstroNet-style].
  3. Model   : two-branch 1D CNN (global + local) -> fused dense head.
  4. Features: ROC/PR curves, confusion matrix, odd/even-style augmentation,
               Grad-CAM saliency (which part of the dip drove the decision),
               and an `predict_star()` helper for any KIC ID.

Usage
  python exoplanet_cnn.py --mode synthetic          # offline demo (works anywhere)
  python exoplanet_cnn.py --mode real --n 400       # real Kepler data (needs internet)
  python exoplanet_cnn.py --predict "Kepler-90"     # vet a single star (real data)
"""
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_BINS, LOCAL_BINS = 2001, 201
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)


# ----------------------------------------------------------------- preprocessing
def bin_curve(phase, flux, n_bins, lo, hi):
    """Median-bin a folded curve over [lo, hi]."""
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    out = np.ones(n_bins)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            out[i] = np.median(flux[m])
    return out


def make_views(time, flux, period, t0, duration):
    """Fold on period; return (global_view, local_view), normalised."""
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    g = bin_curve(phase, flux, GLOBAL_BINS, -period / 2, period / 2)
    w = max(duration * 4, 1e-3)
    l = bin_curve(phase, flux, LOCAL_BINS, -w / 2, w / 2)
    def norm(x):
        x = x - np.median(x)
        s = np.abs(x.min()) + 1e-8
        return (x / s).astype(np.float32)
    return norm(g), norm(l)


# ----------------------------------------------------------------- synthetic data
def _transit_model(phase, depth, dur, shape="U"):
    x = np.abs(phase) / (dur / 2)
    f = np.zeros_like(phase)
    if shape == "U":                      # planet: flat-bottomed, limb-darkened
        m = x < 1
        f[m] = -depth * np.clip(1 - 0.3 * x[m] ** 2, 0, 1)
    else:                                 # eclipsing binary: V-shaped
        m = x < 1
        f[m] = -depth * (1 - x[m])
    return f


def synth_example(is_planet):
    period = np.random.uniform(1.0, 30.0)
    dur = period * np.random.uniform(0.01, 0.05)
    t0 = np.random.uniform(0, period)
    time = np.sort(np.random.uniform(0, 90, 4000))
    noise = np.random.uniform(2e-4, 8e-4)
    flux = 1 + np.random.normal(0, noise, time.size)
    flux += 1e-3 * np.sin(2 * np.pi * time / np.random.uniform(5, 20))  # starspots
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    if is_planet:
        flux += _transit_model(phase, np.random.uniform(5e-4, 4e-3), dur, "U")
    else:
        kind = np.random.choice(["EB", "noise", "var"])
        if kind == "EB":   # V-shape, deeper, plus a secondary eclipse at phase 0.5
            d = np.random.uniform(3e-3, 3e-2)
            flux += _transit_model(phase, d, dur * 1.5, "V")
            sec = ((time - t0) % period) - 0.5 * period
            flux += _transit_model(sec, d * np.random.uniform(0.3, 0.9), dur * 1.5, "V")
        elif kind == "var":
            flux += 2e-3 * np.sin(2 * np.pi * time / period)
        else:
            flux += np.random.normal(0, 1.5e-3, time.size) * (np.random.rand(time.size) < 0.02)
    return make_views(time, flux, period, t0, dur)


def build_synthetic(n=3000):
    G, L, y = [], [], []
    for i in range(n):
        lab = i % 2
        g, l = synth_example(lab)
        G.append(g); L.append(l); y.append(lab)
    return np.array(G), np.array(L), np.array(y)


# ----------------------------------------------------------------- real Kepler data
def build_real(n=400):
    """Real Kepler light curves + NASA Exoplanet Archive KOI labels."""
    import pandas as pd, lightkurve as lk
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
           "select+kepid,koi_disposition,koi_period,koi_time0bk,koi_duration+"
           "from+cumulative+where+koi_disposition+in+('CONFIRMED','FALSE+POSITIVE')"
           "&format=csv")
    koi = pd.read_csv(url).dropna()
    koi = koi.groupby("koi_disposition").head(n // 2).reset_index(drop=True)
    G, L, y = [], [], []
    for _, r in koi.iterrows():
        try:
            lc = lk.search_lightcurve(f"KIC {int(r.kepid)}", mission="Kepler",
                                      author="Kepler", cadence="long").download_all()
            lc = lc.stitch().remove_nans().flatten(window_length=801)
            g, l = make_views(lc.time.value, lc.flux.value, r.koi_period,
                              r.koi_time0bk, r.koi_duration / 24.0)
            G.append(g); L.append(l); y.append(int(r.koi_disposition == "CONFIRMED"))
        except Exception as e:
            print("skip", r.kepid, str(e)[:60])
    return np.array(G), np.array(L), np.array(y)


# ----------------------------------------------------------------- model
def conv_branch(n_layers, in_ch=1):
    layers, ch = [], in_ch
    for i in range(n_layers):
        out = 16 * 2 ** i
        layers += [nn.Conv1d(ch, out, 5, padding=2), nn.BatchNorm1d(out), nn.ReLU(),
                   nn.Conv1d(out, out, 5, padding=2), nn.BatchNorm1d(out), nn.ReLU(),
                   nn.MaxPool1d(5 if i == 0 else 3, stride=2)]
        ch = out
    return nn.Sequential(*layers), ch


class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g, gc = conv_branch(4)
        self.l, lc = conv_branch(2)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(gc + lc, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))
        self.feat = None

    def forward(self, g, l):
        gf = self.g(g)
        self.feat = gf; gf.retain_grad() if gf.requires_grad else None
        gp = gf.mean(-1)
        lp = self.l(l).mean(-1)
        return self.head(torch.cat([gp, lp], 1)).squeeze(1)


def to_t(a): return torch.tensor(a).unsqueeze(1)


def augment(g, l):
    """Time-reversal + small noise (transits are symmetric -> label preserving)."""
    if np.random.rand() < 0.5:
        g, l = torch.flip(g, [-1]), torch.flip(l, [-1])
    return g + 0.01 * torch.randn_like(g), l + 0.01 * torch.randn_like(l)


def train(G, L, y, epochs=25):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    tr, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=0)
    Gt, Lt, yt = to_t(G), to_t(L), torch.tensor(y, dtype=torch.float32)
    model = TransitCNN()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            g, l = augment(Gt[b], Lt[b])
            loss = F.binary_cross_entropy_with_logits(model(g, l), yt[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(Gt[va], Lt[va]), yt[va]).item()
        hist.append(vl)
        if vl < best: best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep+1:02d} val_loss {vl:.4f}")
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), f"{OUT}/transit_cnn.pt")
    return model, (Gt, Lt, yt), te, hist


# ----------------------------------------------------------------- evaluation
def evaluate(model, data, te, hist):
    Gt, Lt, yt = data
    with torch.no_grad():
        p = torch.sigmoid(model(Gt[te], Lt[te])).numpy()
    y = yt[te].numpy()
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    m = dict(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
             accuracy=float((pred == y).mean()), precision=float(tp / max(tp + fp, 1)),
             recall=float(tp / max(tp + fn, 1)), n_test=int(len(y)))
    json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=2); print(m)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--")
    ax[0].set_title(f"ROC (AUC={m['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr); ax[1].set_title(f"PR (AP={m['pr_auc']:.3f})")
    ax[2].imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]): ax[2].text(j, i, v, ha="center", va="center")
    ax[2].set_xticks([0, 1], ["FP", "Planet"]); ax[2].set_yticks([0, 1], ["FP", "Planet"])
    ax[2].set_title("Confusion matrix"); ax[3].plot(hist); ax[3].set_title("Validation loss")
    plt.tight_layout(); plt.savefig(f"{OUT}/evaluation.png", dpi=130); plt.close()
    return m


# ----------------------------------------------------------------- explainability
def gradcam(model, g, l):
    """Grad-CAM on the global branch: where along the orbit did the model look?"""
    model.eval(); g = g.clone().requires_grad_(True)
    out = model(g, l); model.zero_grad(); out.sum().backward()
    f, gr = model.feat, model.feat.grad
    cam = F.relu((gr.mean(-1, keepdim=True) * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=g.shape[-1], mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().numpy(); return cam / (cam.max() + 1e-8), torch.sigmoid(out).item()


def saliency_figure(model, data, te, n=6):
    Gt, Lt, yt = data
    fig, axes = plt.subplots(2, n // 2, figsize=(15, 6)); axes = axes.ravel()
    for a, i in zip(axes, te[:n]):
        cam, p = gradcam(model, Gt[i:i + 1], Lt[i:i + 1])
        a.plot(Gt[i, 0].numpy(), lw=.8); a.fill_between(range(len(cam)), -1, 0.1, where=cam > .5, alpha=.3, color="r")
        a.set_title(f"true={'Planet' if yt[i] else 'FP'}  P(planet)={p:.2f}", fontsize=9)
    plt.suptitle("Grad-CAM: red = region driving the decision"); plt.tight_layout()
    plt.savefig(f"{OUT}/gradcam.png", dpi=130); plt.close()


# ----------------------------------------------------------------- vet a single star
def predict_star(name, model_path=f"{OUT}/transit_cnn.pt"):
    import lightkurve as lk
    lc = lk.search_lightcurve(name, mission="Kepler", author="Kepler").download_all().stitch()
    lc = lc.remove_nans().flatten(801)
    bls = lc.to_periodogram("bls", period=np.linspace(0.8, 40, 20000), duration=[.05, .1, .2])
    P, t0, d = bls.period_at_max_power.value, bls.transit_time_at_max_power.value, bls.duration_at_max_power.value
    g, l = make_views(lc.time.value, lc.flux.value, P, t0, d)
    model = TransitCNN(); model.load_state_dict(torch.load(model_path)); model.eval()
    with torch.no_grad(): p = torch.sigmoid(model(to_t(g[None]), to_t(l[None]))).item()
    print(f"{name}: BLS period={P:.3f} d  ->  P(planet)={p:.3f}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--predict", type=str, default=None)
    a = ap.parse_args()
    if a.predict:
        predict_star(a.predict)
    else:
        G, L, y = build_synthetic(a.n) if a.mode == "synthetic" else build_real(a.n)
        print("dataset:", G.shape, "planets:", int(y.sum()))
        model, data, te, hist = train(G, L, y, a.epochs)
        evaluate(model, data, te, hist); saliency_figure(model, data, te)
        print("saved results to ./outputs")
