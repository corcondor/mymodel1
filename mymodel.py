# ============================================================
# 3-mask (raw/core/edge) feature + margin averaged perceptron
# - Features: counts + moments(compressed) + projections + grid + radial + transitions + topology
# - Masks: raw / core / edge
# - Learning: margin perceptron + int64 clipped weights + lazy averaged
# - Cache features for speed
# ============================================================

import os, shutil, time
import numpy as np
import matplotlib.pyplot as plt

# =========================
# MNIST loader
# =========================
def load_mnist_numpy(data_root=r"C:\mnist_data", max_attempts=2):
    from torchvision import datasets
    root = os.path.abspath(data_root)
    mnist_dir = os.path.join(root, "MNIST")
    print(f"[MNIST] root={root}", flush=True)

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            train = datasets.MNIST(root=root, train=True,  download=True)
            test  = datasets.MNIST(root=root, train=False, download=True)

            Xtr = train.data.numpy().astype(np.float32) / 255.0
            ytr = train.targets.numpy().astype(np.int64)
            Xte = test.data.numpy().astype(np.float32) / 255.0
            yte = test.targets.numpy().astype(np.int64)

            print(f"[MNIST] loaded attempt {attempt}: train={len(Xtr)} test={len(Xte)}", flush=True)
            return Xtr, ytr, Xte, yte
        except OSError as e:
            last_err = e
            print(f"[MNIST] attempt {attempt} failed: {repr(e)}", flush=True)
            shutil.rmtree(mnist_dir, ignore_errors=True)

    raise RuntimeError(f"MNIST load failed: {repr(last_err)}")

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
            if i == j:
                continue
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
    print(f"[saved] {path}", flush=True)

def quick_pred_hist(preds, K=10):
    h = np.bincount(preds.astype(np.int64), minlength=K)
    return " ".join([f"{i}:{int(h[i])}" for i in range(K)])

def try_print_classification_report(y_true, y_pred):
    try:
        from sklearn.metrics import classification_report
        print("\nClassification report:\n", flush=True)
        print(classification_report(y_true, y_pred, digits=4), flush=True)
    except Exception as e:
        print(f"[warn] sklearn classification_report unavailable: {repr(e)}", flush=True)

# =========================
# Binary ops + morphology (NO マル照)
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
    # 4-neighborhood gradient-ish (boolean ops)
    D = B | shift01(B, 1, 0) | shift01(B, -1, 0) | shift01(B, 0, 1) | shift01(B, 0, -1)
    E = B & shift01(B, 1, 0) & shift01(B, -1, 0) & shift01(B, 0, 1) & shift01(B, 0, -1)
    return (D ^ E).astype(np.uint8)

def recenter_masks(B_raw, B_core, B_edge, target=(13, 13)):
    ys, xs = np.nonzero(B_raw)
    if xs.size == 0:
        return B_raw, B_core, B_edge
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))
    tx, ty = int(target[0]), int(target[1])
    dx = tx - cx
    dy = ty - cy
    return shift01(B_raw, dx, dy), shift01(B_core, dx, dy), shift01(B_edge, dx, dy)

# =========================
# Topology (Euler via 2x2 patterns)
# =========================
def euler_2x2_features(B01):
    P00 = B01[:-1, :-1]
    P01 = B01[:-1,  1:]
    P10 = B01[ 1:, :-1]
    P11 = B01[ 1:,  1:]
    s = (P00 + P01 + P10 + P11).astype(np.uint8)

    n1 = int(np.count_nonzero(s == 1))
    n3 = int(np.count_nonzero(s == 3))
    diag = ((P00 == 1) & (P11 == 1) & (P01 == 0) & (P10 == 0)) | \
           ((P01 == 1) & (P10 == 1) & (P00 == 0) & (P11 == 0))
    n_diag = int(np.count_nonzero(diag))

    chi_num = n1 - n3 - 2 * n_diag
    chi = int(chi_num // 4)
    return np.array([chi, n_diag, n1, n3], dtype=np.int16)

# =========================
# Feature extractor
# =========================
class MultiMaskFeaturizer:
    """
    1 mask -> features:
      [n, cx, cy]
      moments(10): m20,m02,m11,m30,m03,m21,m12,m40,m04,m22 (compressed)
      projections: hx(28), hy(28), hd1(55), hd2(55)
      grid: g*g
      radial: r_bins
      transitions: tx, ty
      topology(4): chi, n_diag, n1, n3
    dim_one default = 278 (grid=9 r_bins=12)
    dim = 3*dim_one for raw/core/edge
    """
    def __init__(
        self,
        grid=9, r_bins=12,
        div2=50, div3=400, div4=3000,
        q_n=1, q_proj=2, q_grid=1, q_rad=1, q_tr=2, q_topo=1
    ):
        self.grid = int(grid)
        self.r_bins = int(r_bins)
        self.div2 = int(div2)
        self.div3 = int(div3)
        self.div4 = int(div4)

        self.q_n = int(max(1, q_n))
        self.q_proj = int(max(1, q_proj))
        self.q_grid = int(max(1, q_grid))
        self.q_rad = int(max(1, q_rad))
        self.q_tr = int(max(1, q_tr))
        self.q_topo = int(max(1, q_topo))

        self.hx_len = 28
        self.hy_len = 28
        self.hd_len = 55
        self.r2_max = 1458
        self.moment_dim = 10
        self.trans_dim = 2
        self.topo_dim = 4

        self.dim_one = (
            1 + 2 +
            self.moment_dim +
            (self.hx_len + self.hy_len + self.hd_len + self.hd_len) +
            (self.grid * self.grid) +
            self.r_bins +
            self.trans_dim +
            self.topo_dim
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
        u2 = u*u
        v2 = v*v

        m20 = int(u2.sum())
        m02 = int(v2.sum())
        m11 = int((u*v).sum())

        m30 = int((u2*u).sum())
        m03 = int((v2*v).sum())
        m21 = int((u2*v).sum())
        m12 = int((u*v2).sum())

        m40 = int((u2*u2).sum())
        m04 = int((v2*v2).sum())
        m22 = int((u2*v2).sum())

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

        hx = (np.bincount(xs, minlength=self.hx_len) // self.q_proj).astype(np.int16)
        hy = (np.bincount(ys, minlength=self.hy_len) // self.q_proj).astype(np.int16)

        d1 = xs + ys
        d2 = xs - ys + 27
        hd1 = (np.bincount(d1, minlength=self.hd_len) // self.q_proj).astype(np.int16)
        hd2 = (np.bincount(d2, minlength=self.hd_len) // self.q_proj).astype(np.int16)

        g = self.grid
        bx = (xs * g) // 28
        by = (ys * g) // 28
        idx = by * g + bx
        gc = (np.bincount(idx, minlength=g*g) // self.q_grid).astype(np.int16)

        r2 = u2 + v2
        rb = (r2 * self.r_bins) // (self.r2_max + 1)
        rb = np.clip(rb, 0, self.r_bins - 1)
        rh = (np.bincount(rb, minlength=self.r_bins) // self.q_rad).astype(np.int16)

        Bi = B.astype(np.int16)
        tx = (int(np.abs(np.diff(Bi, axis=1)).sum()) // self.q_tr)
        ty = (int(np.abs(np.diff(Bi, axis=0)).sum()) // self.q_tr)
        tr = np.array([np.clip(tx, 0, 32767), np.clip(ty, 0, 32767)], dtype=np.int16)

        topo = (euler_2x2_features(B) // self.q_topo).astype(np.int16)

        head = np.array([n // self.q_n, cx, cy], dtype=np.int16)

        return np.concatenate([head, mom, hx, hy, hd1, hd2, gc, rh, tr, topo], axis=0)

    def extract(self, B_raw, B_core, B_edge):
        f_raw  = self._one_mask_features(B_raw)
        f_core = self._one_mask_features(B_core)
        f_edge = self._one_mask_features(B_edge)
        return np.concatenate([f_raw, f_core, f_edge], axis=0)

# =========================
# Model: margin perceptron + lazy averaged + clipping (int64 safe)
# =========================
class AveragedClippedPerceptron3:
    def __init__(self, dim_one, num_classes=10, step=1, wmax=2500, margin=50):
        self.C = int(num_classes)
        self.D1 = int(dim_one)
        self.step = int(step)
        self.wmax = int(wmax)
        self.margin = int(margin)

        # W: (C, 3, D1)
        self.W = np.zeros((self.C, 3, self.D1), dtype=np.int64)

        # Lazy averaging
        self.Wsum = np.zeros_like(self.W)
        self.last_t = np.zeros(self.C, dtype=np.int64)
        self.t = 0

        self.updates = 0

    def _close_row(self, c):
        dt = self.t - self.last_t[c]
        if dt != 0:
            self.Wsum[c] += dt * self.W[c]
            self.last_t[c] = self.t

    def score(self, f_int16):
        f3 = f_int16.reshape(3, self.D1).astype(np.int64, copy=False)
        s = np.zeros(self.C, dtype=np.int64)
        s += self.W[:, 0, :] @ f3[0]
        s += self.W[:, 1, :] @ f3[1]
        s += self.W[:, 2, :] @ f3[2]
        return s

    def predict(self, f_int16):
        return int(np.argmax(self.score(f_int16)))

    def update(self, f_int16, y_true):
        s = self.score(f_int16)
        y_hat = int(np.argmax(s))

        # competitor best among k != y_true (safe: int64, no -1e30 hacks)
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
            # close rows before mutate
            self._close_row(y_true)
            self._close_row(y2)

            f3 = f_int16.reshape(3, self.D1).astype(np.int64, copy=False)
            self.W[y_true, :, :] += self.step * f3
            self.W[y2,    :, :] -= self.step * f3

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

        if (i+1) % report_every == 0:
            print(f"[cache:{tag}] {i+1}/{N} elapsed={time.perf_counter()-t0:.2f}s", flush=True)
    return F

# =========================
# Main
# =========================
def main():
    cfg = dict(
        DATA_ROOT=r"C:\mnist_data",

        TH_RAW=0.35,
        TH_CORE=0.60,

        GRID=9,
        R_BINS=12,

        DIV2=50,
        DIV3=400,
        DIV4=3000,

        Q_N=1,
        Q_PROJ=1,
        Q_GRID=1,
        Q_RAD=1,
        Q_TR=1,
        Q_TOPO=1,

        RECENTER=True,

        TRAIN_N=60000,
        TEST_N=10000,
        EPOCHS=8,

        STEP=1,
        MARGIN=50,

        WMAX_START=2500,
        WMAX_END=10000,
        WMAX_SWITCH_EPOCH=5,

        REPORT_EVERY=5000,

        CACHE=True,
        CACHE_REPORT_TRAIN=5000,
        CACHE_REPORT_TEST=2000,

        SHOW_PLOTS=False,
        PRINT_REPORT=True,
        SEED=0,
    )

    # data
    Xtr, ytr, Xte, yte = load_mnist_numpy(cfg["DATA_ROOT"])
    Xtr, ytr = Xtr[:cfg["TRAIN_N"]], ytr[:cfg["TRAIN_N"]]
    Xte, yte = Xte[:cfg["TEST_N"]],  yte[:cfg["TEST_N"]]

    # featurizer
    fe = MultiMaskFeaturizer(
        grid=cfg["GRID"], r_bins=cfg["R_BINS"],
        div2=cfg["DIV2"], div3=cfg["DIV3"], div4=cfg["DIV4"],
        q_n=cfg["Q_N"], q_proj=cfg["Q_PROJ"], q_grid=cfg["Q_GRID"],
        q_rad=cfg["Q_RAD"], q_tr=cfg["Q_TR"], q_topo=cfg["Q_TOPO"]
    )
    dim = fe.dim
    dim_one = fe.dim_one

    print(
        f"[model] dim={dim} (one={dim_one} x3 masks) grid={cfg['GRID']} r_bins={cfg['R_BINS']} "
        f"thr_raw_u8={int(cfg['TH_RAW']*255)} thr_core_u8={int(cfg['TH_CORE']*255)} "
        f"quant(n={cfg['Q_N']},proj={cfg['Q_PROJ']},grid={cfg['Q_GRID']},rad={cfg['Q_RAD']},tr={cfg['Q_TR']},topo={cfg['Q_TOPO']}) "
        f"recenter={cfg['RECENTER']} step={cfg['STEP']} margin={cfg['MARGIN']} "
        f"wmax={cfg['WMAX_START']}->{cfg['WMAX_END']}@ep{cfg['WMAX_SWITCH_EPOCH']}",
        flush=True
    )

    # cache
    if cfg["CACHE"]:
        print("[cache] building train features...", flush=True)
        Ftr = build_cache_all(
            Xtr, fe, cfg["TH_RAW"], cfg["TH_CORE"],
            recenter=cfg["RECENTER"],
            report_every=cfg["CACHE_REPORT_TRAIN"],
            tag="train"
        )
        print("[cache] building test features...", flush=True)
        Fte = build_cache_all(
            Xte, fe, cfg["TH_RAW"], cfg["TH_CORE"],
            recenter=cfg["RECENTER"],
            report_every=cfg["CACHE_REPORT_TEST"],
            tag="test"
        )
    else:
        Ftr = Fte = None

    # model
    model = AveragedClippedPerceptron3(dim_one=dim_one, step=cfg["STEP"], wmax=cfg["WMAX_START"], margin=cfg["MARGIN"])
    rng = np.random.default_rng(cfg["SEED"])

    # train
    t0 = time.perf_counter()
    total_updates = 0

    for ep in range(cfg["EPOCHS"]):
        model.wmax = cfg["WMAX_END"] if (ep + 1) >= cfg["WMAX_SWITCH_EPOCH"] else cfg["WMAX_START"]

        idx = np.arange(len(Xtr))
        rng.shuffle(idx)

        correct = 0
        margin_ok_cnt = 0

        for it, i in enumerate(idx, 1):
            f = Ftr[i] if cfg["CACHE"] else fe.extract(
                *recenter_masks(
                    binarize(Xtr[i], cfg["TH_RAW"]),
                    binarize(Xtr[i], cfg["TH_CORE"]),
                    edge_map(binarize(Xtr[i], cfg["TH_RAW"])),
                    target=(13, 13)
                )
            )

            _, ok, mok = model.update(f, int(ytr[i]))
            correct += int(ok)
            margin_ok_cnt += int(mok)

            if it % cfg["REPORT_EVERY"] == 0:
                print(
                    f"[train] ep={ep+1} {it}/{len(Xtr)} acc={correct/it:.4f} "
                    f"margin_ok={margin_ok_cnt/it:.4f} wmax={model.wmax} updates~={model.updates}",
                    flush=True
                )

        print(
            f"[train] ep={ep+1} done acc={correct/len(Xtr):.4f} margin_ok={margin_ok_cnt/len(Xtr):.4f} "
            f"wmax={model.wmax}",
            flush=True
        )

    model.finalize()
    train_time = time.perf_counter() - t0
    print(f"[train] total time {train_time:.2f}s  total_updates={model.updates}", flush=True)

    # test
    t1 = time.perf_counter()
    preds = np.zeros(len(Xte), dtype=np.int64)
    correct = 0
    for i in range(len(Xte)):
        f = Fte[i] if cfg["CACHE"] else fe.extract(
            *recenter_masks(
                binarize(Xte[i], cfg["TH_RAW"]),
                binarize(Xte[i], cfg["TH_CORE"]),
                edge_map(binarize(Xte[i], cfg["TH_RAW"])),
                target=(13, 13)
            )
        )

        yhat = model.predict(f)
        preds[i] = yhat
        correct += int(yhat == int(yte[i]))

        if (i+1) % 1000 == 0:
            print(f"[test] {i+1}/{len(Xte)} acc={correct/(i+1):.4f} elapsed={time.perf_counter()-t1:.2f}s",
                  flush=True)

    acc = correct / len(Xte)
    test_time = time.perf_counter() - t1

    print(f"Test accuracy: {acc*100:.2f}% ({len(Xte)} samples)", flush=True)
    print(f"Test time: {test_time:.2f}s", flush=True)
    print(f"[pred hist] {quick_pred_hist(preds)}", flush=True)

    cm = confusion_matrix(yte, preds)
    print(f"[top confusions] {top_confusions(cm, k=12)}", flush=True)

    plot_cm(cm, normalize=False, title=f"CM acc={acc*100:.2f}%", path="confusion_matrix.png", show=cfg["SHOW_PLOTS"])
    plot_cm(cm, normalize=True,  title=f"CM norm acc={acc*100:.2f}%", path="confusion_matrix_norm.png", show=cfg["SHOW_PLOTS"])
    np.save("confusion_matrix.npy", cm)
    print("[saved] confusion_matrix.npy", flush=True)

    if cfg["PRINT_REPORT"]:
        try_print_classification_report(yte, preds)

if __name__ == "__main__":
    main()
