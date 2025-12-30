# ============================================================
# mymodel1.py
# 3-mask features + topology + simple graph stats + averaged perceptron
# - Masks: raw / core / edge
# - Features: counts + moments + projections + grid + radial + transitions + topology + graph(deg, comp, euler)
# - Learning: margin perceptron + int64 clipped weights + lazy averaged
# - Cache: auto-named by config + shape-check (rebuild if mismatch)
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.datasets import fetch_openml

# =========================
# MNIST loader
# =========================
def load_mnist_numpy():
    mnist = fetch_openml("mnist_784", version=1, cache=True)
    X = mnist.data.to_numpy().astype(np.float32) / 255.0
    y = mnist.target.to_numpy().astype(np.int64)
    Xtr, Xte = X[:60000].reshape(-1, 28, 28), X[60000:].reshape(-1, 28, 28)
    ytr, yte = y[:60000], y[60000:]
    print(f"[MNIST] loaded: train={len(Xtr)} test={len(Xte)}")
    return Xtr, ytr, Xte, yte

# =========================
# Metrics / plots
# =========================
def confusion_matrix(y_true, y_pred, K=10):
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def top_confusions(cm, k=12):
    K = cm.shape[0]
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    return " ".join([f"{i}->{j}:{c}" for c, i, j in pairs[:k]])

def plot_cm(cm, normalize=False, title="CM", path="cm.png", show=False):
    data = cm.astype(np.float64)
    if normalize:
        row = data.sum(axis=1, keepdims=True)
        data = np.divide(data, np.maximum(row, 1), dtype=np.float64)
    plt.figure(figsize=(8, 7))
    plt.imshow(data, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.xticks(np.arange(10)); plt.yticks(np.arange(10))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close()
    print(f"[saved] {path}")

def quick_pred_hist(preds, K=10):
    h = np.bincount(preds.astype(np.int64), minlength=K)
    return " ".join([f"{i}:{int(h[i])}" for i in range(K)])

def try_print_classification_report(y_true, y_pred):
    try:
        from sklearn.metrics import classification_report
        print("\nClassification report:\n")
        print(classification_report(y_true, y_pred, digits=4))
    except:
        print("[warn] classification_report unavailable")

# =========================
# Binary ops + morphology
# =========================
def binarize(img28, th):
    return (img28 > th).astype(np.uint8)

def shift01(B, dx, dy):
    H, W = B.shape
    out = np.zeros_like(B)
    x0 = max(0, dx); x1 = min(W, W + dx)
    y0 = max(0, dy); y1 = min(H, H + dy)
    sx0 = x0 - dx; sx1 = x1 - dx
    sy0 = y0 - dy; sy1 = y1 - dy
    out[y0:y1, x0:x1] = B[sy0:sy1, sx0:sx1]
    return out

def edge_map(B):
    D = B | shift01(B, 1, 0) | shift01(B, -1, 0) | shift01(B, 0, 1) | shift01(B, 0, -1)
    E = B & shift01(B, 1, 0) & shift01(B, -1, 0) & shift01(B, 0, 1) & shift01(B, 0, -1)
    return (D ^ E).astype(np.uint8)

def recenter_masks(B_raw, B_core, B_edge, target=(13, 13)):
    ys, xs = np.nonzero(B_raw)
    if xs.size == 0:
        return B_raw, B_core, B_edge
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))
    dx = target[0] - cx
    dy = target[1] - cy
    return shift01(B_raw, dx, dy), shift01(B_core, dx, dy), shift01(B_edge, dx, dy)

# =========================
# Topology (Euler via 2x2)
# =========================
def euler_2x2_features(B01):
    P00 = B01[:-1, :-1]
    P01 = B01[:-1, 1:]
    P10 = B01[1:, :-1]
    P11 = B01[1:, 1:]
    s = (P00 + P01 + P10 + P11).astype(np.uint8)
    n1 = np.count_nonzero(s == 1)
    n3 = np.count_nonzero(s == 3)
    diag = ((P00 == 1) & (P11 == 1) & (P01 == 0) & (P10 == 0)) | \
           ((P01 == 1) & (P10 == 1) & (P00 == 0) & (P11 == 0))
    n_diag = np.count_nonzero(diag)
    chi_num = n1 - n3 - 2 * n_diag
    chi = chi_num // 4
    return np.array([chi, n_diag, n1, n3], dtype=np.int16)

# =========================
# Graph features (no Laplacian)
# =========================
def graph_features_simple(B):
    """
    9-dim int16:
      deg_hist(5), comp_feat(3), euler(V-E+F)(1)
    """
    H, W = B.shape
    ys, xs = np.nonzero(B)
    n = int(len(xs))
    if n == 0:
        return np.zeros(9, dtype=np.int16)

    # degree on 4-neighborhood
    deg = np.zeros_like(B, dtype=np.int16)
    deg[:, 1:]  += B[:, :-1]
    deg[:, :-1] += B[:, 1:]
    deg[1:, :]  += B[:-1, :]
    deg[:-1, :] += B[1:, :]
    deg_vals = deg[B > 0]
    deg_hist = np.bincount(deg_vals, minlength=5)[:5].astype(np.int16)

    # connected components
    visited = np.zeros_like(B, dtype=bool)
    comp_sizes = []
    for y, x in zip(ys, xs):
        y = int(y); x = int(x)
        if visited[y, x]:
            continue
        size = 0
        dq = deque([(y, x)])
        while dq:
            cy, cx = dq.popleft()
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            size += 1
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and B[ny, nx] and not visited[ny, nx]:
                    dq.append((ny, nx))
        comp_sizes.append(size)

    num_comp = int(len(comp_sizes))
    mean_size = int(np.mean(comp_sizes)) if comp_sizes else 0
    max_size  = int(max(comp_sizes)) if comp_sizes else 0
    comp_feat = np.array([num_comp, mean_size, max_size], dtype=np.int16)

    # Euler V - E + F
    V = n
    E = int(deg_vals.sum() // 2)
    F = num_comp
    euler = V - E + F
    euler_feat = np.int16(np.clip(euler, -32768, 32767))

    return np.concatenate((deg_hist, comp_feat, np.array([euler_feat], dtype=np.int16)))

# =========================
# Featurizer
# =========================
class MultiMaskFeaturizer:
    def __init__(self, grid=12, r_bins=12, div2=50, div3=400, div4=3000):
        self.grid = int(grid)
        self.r_bins = int(r_bins)
        self.div2 = int(div2)
        self.div3 = int(div3)
        self.div4 = int(div4)

        self.hx_len = 28
        self.hy_len = 28
        self.hd_len = 55
        self.moment_dim = 10
        self.trans_dim = 2
        self.topo_dim = 4
        self.graph_dim = 9  # simplified

        self.dim_one = (
            3 +
            self.moment_dim +
            (self.hx_len + self.hy_len + self.hd_len + self.hd_len) +
            (self.grid * self.grid) +
            self.r_bins +
            self.trans_dim +
            self.topo_dim +
            self.graph_dim
        )

    @property
    def dim(self):
        return 3 * self.dim_one

    def _one_mask_features(self, B):
        ys, xs = np.nonzero(B)
        n = xs.size
        if n == 0:
            return np.zeros(self.dim_one, dtype=np.int16)

        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)

        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))

        u = xs - cx
        v = ys - cy
        u2 = u * u
        v2 = v * v

        m20 = int(u2.sum())
        m02 = int(v2.sum())
        m11 = int((u * v).sum())
        m30 = int((u2 * u).sum())
        m03 = int((v2 * v).sum())
        m21 = int((u2 * v).sum())
        m12 = int((u * v2).sum())
        m40 = int((u2 * u2).sum())
        m04 = int((v2 * v2).sum())
        m22 = int((u2 * v2).sum())

        mom = np.array([
            np.clip(m20 // self.div2, -32768, 32767),
            np.clip(m02 // self.div2, -32768, 32767),
            np.clip(m11 // self.div2, -32768, 32767),
            np.clip(m30 // self.div3, -32768, 32767),
            np.clip(m03 // self.div3, -32768, 32767),
            np.clip(m21 // self.div3, -32768, 32767),
            np.clip(m12 // self.div3, -32768, 32767),
            np.clip(m40 // self.div4, -32768, 32767),
            np.clip(m04 // self.div4, -32768, 32767),
            np.clip(m22 // self.div4, -32768, 32767),
        ], dtype=np.int16)

        hx = np.bincount(xs, minlength=28).astype(np.int16)
        hy = np.bincount(ys, minlength=28).astype(np.int16)
        hd1 = np.bincount(np.clip(xs - ys + 27, 0, 54), minlength=55).astype(np.int16)
        hd2 = np.bincount(np.clip(xs + ys, 0, 54), minlength=55).astype(np.int16)

        grid_feat = np.zeros(self.grid ** 2, dtype=np.int16)
        binw = 28 / self.grid
        gx = np.floor(xs / binw).astype(int)
        gy = np.floor(ys / binw).astype(int)
        gx = np.clip(gx, 0, self.grid - 1)
        gy = np.clip(gy, 0, self.grid - 1)
        np.add.at(grid_feat, gy * self.grid + gx, 1)

        r = np.sqrt(u2 + v2).astype(int)
        rad = np.bincount(np.clip(r, 0, self.r_bins - 1), minlength=self.r_bins).astype(np.int16)

        # transitions: abs diff of row/col sums (signed to avoid underflow)
        col = B.sum(axis=0, dtype=np.int32)
        row = B.sum(axis=1, dtype=np.int32)
        tx = int(np.abs(np.diff(col)).sum())
        ty = int(np.abs(np.diff(row)).sum())
        tx = int(np.clip(tx, -32768, 32767))
        ty = int(np.clip(ty, -32768, 32767))
        trans = np.array([tx, ty], dtype=np.int16)

        topo = euler_2x2_features(B)
        graph = graph_features_simple(B)

        header = np.array([n, cx, cy], dtype=np.int16)
        return np.concatenate((header, mom, hx, hy, hd1, hd2, grid_feat, rad, trans, topo, graph))

    def extract(self, B_raw, B_core, B_edge):
        return np.concatenate((
            self._one_mask_features(B_raw),
            self._one_mask_features(B_core),
            self._one_mask_features(B_edge),
        ))

# =========================
# Perceptron
# =========================
class AveragedClippedPerceptron3:
    def __init__(self, dim_one, C=10, step=1, wmax=2500, margin=50):
        self.D1 = int(dim_one)
        self.C = int(C)
        self.step = int(step)
        self.wmax = int(wmax)
        self.margin = int(margin)

        self.W = np.zeros((self.C, 3, self.D1), dtype=np.int64)
        self.Wsum = np.zeros_like(self.W)
        self.last_t = np.zeros(self.C, dtype=np.int64)
        self.t = 0
        self.updates = 0

    def _close_row(self, c):
        dt = self.t - self.last_t[c]
        if dt:
            self.Wsum[c] += dt * self.W[c]
            self.last_t[c] = self.t

    def score(self, f_int16):
        # f_int16 must have length 3*self.D1
        f3 = f_int16.reshape(3, self.D1).astype(np.int64, copy=False)
        return (self.W[:, 0, :] @ f3[0]) + (self.W[:, 1, :] @ f3[1]) + (self.W[:, 2, :] @ f3[2])

    def predict(self, f_int16):
        return int(np.argmax(self.score(f_int16)))

    def update(self, f_int16, y_true):
        s = self.score(f_int16)
        y_hat = int(np.argmax(s))

        best = None
        y2 = -1
        for k in range(self.C):
            if k == y_true:
                continue
            v = int(s[k])
            if (best is None) or (v > best):
                best = v
                y2 = k

        s_true = int(s[y_true])
        margin_ok = (s_true >= best + self.margin)
        correct = (y_hat == y_true)

        if not margin_ok:
            self._close_row(y_true)
            self._close_row(y2)
            f3 = f_int16.reshape(3, self.D1).astype(np.int64, copy=False)
            self.W[y_true] += self.step * f3
            self.W[y2] -= self.step * f3
            np.clip(self.W, -self.wmax, self.wmax, out=self.W)
            self.updates += 1

        self.t += 1
        return y_hat, correct, margin_ok

    def finalize(self):
        for c in range(self.C):
            self._close_row(c)
        if self.t > 0:
            self.W = (self.Wsum // self.t).astype(np.int64)

# =========================
# Feature cache
# =========================
def build_cache_all(X, fe, th_raw, th_core, recenter=True, report_every=5000, tag="train"):
    N = len(X)
    F = np.zeros((N, fe.dim), dtype=np.int16)
    t0 = time.perf_counter()
    for i in range(N):
        B_raw  = binarize(X[i], th_raw)
        B_core = binarize(X[i], th_core)
        B_edge = edge_map(B_raw)
        if recenter:
            B_raw, B_core, B_edge = recenter_masks(B_raw, B_core, B_edge, target=(13, 13))
        F[i] = fe.extract(B_raw, B_core, B_edge)

        if (i + 1) % report_every == 0:
            print(f"[cache:{tag}] {i+1}/{N} elapsed={time.perf_counter() - t0:.2f}s")
    return F

def cache_name(prefix, cfg, fe):
    thr_raw_u8  = int(cfg["TH_RAW"] * 255)
    thr_core_u8 = int(cfg["TH_CORE"] * 255)
    rec = int(bool(cfg["RECENTER"]))
    return f"{prefix}_d{fe.dim}_one{fe.dim_one}_g{cfg['GRID']}_rb{cfg['R_BINS']}_thr{thr_raw_u8}_{thr_core_u8}_rec{rec}.npy"

def load_or_build_cache_checked(path, X, fe, th_raw, th_core, recenter, report_every, tag):
    if path and os.path.exists(path):
        print(f"[cache] loading {path} ...")
        F = np.load(path, mmap_mode=None)
        if F.ndim == 2 and F.shape[0] == len(X) and F.shape[1] == fe.dim:
            return F
        print(f"[cache] mismatch: file_shape={getattr(F, 'shape', None)} expected=({len(X)},{fe.dim}) -> rebuild")

    F = build_cache_all(
        X, fe, th_raw, th_core,
        recenter=recenter,
        report_every=report_every,
        tag=tag
    )
    if path:
        np.save(path, F)
        print(f"[cache] saved {path}")
    return F

# =========================
# Main
# =========================
def main():
    cfg = dict(
        TH_RAW=0.35,
        TH_CORE=0.60,
        GRID=12,
        R_BINS=12,
        DIV2=683410889 // 10000000,
        DIV3=394567463 // 1000000,
        DIV4=683410889 // 100000,
        RECENTER=True,
        TRAIN_N=60000,
        TEST_N=10000,
        EPOCHS=128,
        STEP=1,
        MARGIN=50,
        WMAX_START=3000,
        WMAX_END=12000,
        WMAX_SWITCH_EPOCH=5,
        REPORT_EVERY=5000,
        CACHE=True,
        CACHE_REPORT_TRAIN=5000,
        CACHE_REPORT_TEST=2000,
        SHOW_PLOTS=False,
        PRINT_REPORT=True,
        SEED=0,
    )

    Xtr, ytr, Xte, yte = load_mnist_numpy()
    Xtr, ytr = Xtr[:cfg["TRAIN_N"]], ytr[:cfg["TRAIN_N"]]
    Xte, yte = Xte[:cfg["TEST_N"]],  yte[:cfg["TEST_N"]]

    fe = MultiMaskFeaturizer(
        grid=cfg["GRID"], r_bins=cfg["R_BINS"],
        div2=cfg["DIV2"], div3=cfg["DIV3"], div4=cfg["DIV4"],
    )

    print(
        f"[model] dim={fe.dim} (one={fe.dim_one} x3 masks) grid={cfg['GRID']} r_bins={cfg['R_BINS']} "
        f"thr_raw_u8={int(cfg['TH_RAW']*255)} thr_core_u8={int(cfg['TH_CORE']*255)} "
        f"recenter={cfg['RECENTER']} step={cfg['STEP']} margin={cfg['MARGIN']} "
        f"wmax={cfg['WMAX_START']}->{cfg['WMAX_END']}@ep{cfg['WMAX_SWITCH_EPOCH']}"
    )

    if cfg["CACHE"]:
        path_tr = cache_name("Ftr", cfg, fe)
        path_te = cache_name("Fte", cfg, fe)

        Ftr = load_or_build_cache_checked(
            path_tr, Xtr, fe, cfg["TH_RAW"], cfg["TH_CORE"],
            cfg["RECENTER"], cfg["CACHE_REPORT_TRAIN"], "train"
        )
        Fte = load_or_build_cache_checked(
            path_te, Xte, fe, cfg["TH_RAW"], cfg["TH_CORE"],
            cfg["RECENTER"], cfg["CACHE_REPORT_TEST"], "test"
        )
    else:
        Ftr = Fte = None

    model = AveragedClippedPerceptron3(
        dim_one=fe.dim_one,
        step=cfg["STEP"],
        wmax=cfg["WMAX_START"],
        margin=cfg["MARGIN"]
    )
    rng = np.random.default_rng(cfg["SEED"])

    t0 = time.perf_counter()

    for ep in range(cfg["EPOCHS"]):
        model.wmax = cfg["WMAX_END"] if (ep + 1) >= cfg["WMAX_SWITCH_EPOCH"] else cfg["WMAX_START"]

        idx = np.arange(len(Xtr))
        rng.shuffle(idx)

        correct = 0
        margin_ok_cnt = 0

        for it, i in enumerate(idx, 1):
            if cfg["CACHE"]:
                f = Ftr[i]
            else:
                B_raw  = binarize(Xtr[i], cfg["TH_RAW"])
                B_core = binarize(Xtr[i], cfg["TH_CORE"])
                B_edge = edge_map(B_raw)
                if cfg["RECENTER"]:
                    B_raw, B_core, B_edge = recenter_masks(B_raw, B_core, B_edge)
                f = fe.extract(B_raw, B_core, B_edge)

            _, ok, mok = model.update(f, int(ytr[i]))
            correct += int(ok)
            margin_ok_cnt += int(mok)

            if it % cfg["REPORT_EVERY"] == 0:
                print(
                    f"[train] ep={ep+1} {it}/{len(Xtr)} acc={correct/it:.4f} "
                    f"margin_ok={margin_ok_cnt/it:.4f} wmax={model.wmax} updates~={model.updates}"
                )

        print(
            f"[train] ep={ep+1} done acc={correct/len(Xtr):.4f} margin_ok={margin_ok_cnt/len(Xtr):.4f} "
            f"wmax={model.wmax}"
        )

    model.finalize()
    train_time = time.perf_counter() - t0
    print(f"[train] total time {train_time:.2f}s  total_updates={model.updates}")

    t1 = time.perf_counter()
    preds = np.zeros(len(Xte), dtype=np.int64)
    correct = 0
    for i in range(len(Xte)):
        if cfg["CACHE"]:
            f = Fte[i]
        else:
            B_raw  = binarize(Xte[i], cfg["TH_RAW"])
            B_core = binarize(Xte[i], cfg["TH_CORE"])
            B_edge = edge_map(B_raw)
            if cfg["RECENTER"]:
                B_raw, B_core, B_edge = recenter_masks(B_raw, B_core, B_edge)
            f = fe.extract(B_raw, B_core, B_edge)

        yhat = model.predict(f)
        preds[i] = yhat
        correct += int(yhat == int(yte[i]))

        if (i + 1) % 1000 == 0:
            print(f"[test] {i+1}/{len(Xte)} acc={correct/(i+1):.4f} elapsed={time.perf_counter() - t1:.2f}s")

    acc = correct / len(Xte)
    test_time = time.perf_counter() - t1

    print(f"Test accuracy: {acc*100:.2f}% ({len(Xte)} samples)")
    print(f"Test time: {test_time:.2f}s")
    print(f"[pred hist] {quick_pred_hist(preds)}")

    cm = confusion_matrix(yte, preds)
    print(f"[top confusions] {top_confusions(cm, k=12)}")

    plot_cm(cm, normalize=False, title=f"CM acc={acc*100:.2f}%", path="confusion_matrix.png", show=cfg["SHOW_PLOTS"])
    plot_cm(cm, normalize=True,  title=f"CM norm acc={acc*100:.2f}%", path="confusion_matrix_norm.png", show=cfg["SHOW_PLOTS"])
    np.save("confusion_matrix.npy", cm)
    print("[saved] confusion_matrix.npy")

    if cfg["PRINT_REPORT"]:
        try_print_classification_report(yte, preds)

if __name__ == "__main__":
    main()



