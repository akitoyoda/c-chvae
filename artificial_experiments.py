"""
Reproduce the artificial data experiments from Pawelczyk et al. (WWW'20) using a
PyTorch reimplementation of C-CHVAE.

This script focuses on Experiment 1 (Example 1: "make blobs"):
- x = [x1, x2] from a mixture of 3 Gaussians (3 modes)
- classifier is stylized: y = I(x2 > boundary) with boundary=6
- goal: generate counterfactuals for test points with f(x)=0

Key fixes to better match the paper:
- DGP: left & middle clusters straddle boundary; right cluster never crosses boundary
- Search: choose the closest counterfactual in *latent space* (Algorithm 1 intuition),
         and stop at the smallest radius where a counterfactual exists.
- Plots: add a Figure 2(f)-style visualization (test points + counterfactuals only)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import Evaluation  # noqa: E402
import Graph  # noqa: E402

MIN_VARIANCE = 1e-6


# -----------------------------
# Small sklearn-like utilities
# -----------------------------
class IdentityScaler:
    """Drop-in replacement for sklearn's scaler interface."""
    def fit(self, X: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)


class StepClassifier:
    """
    Stylized classifier used in Pawelczyk et al. Example 1:
      y = I(x2 > boundary)
    Exposes sklearn-like predict/predict_proba/score methods.
    """
    def __init__(self, boundary: float = 6.0, feature_index: int = 1):
        self.boundary = float(boundary)
        self.feature_index = int(feature_index)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X[:, self.feature_index] > self.boundary).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y = self.predict(X).astype(float)
        return np.column_stack([1.0 - y, y])

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))


def _split_tensor_by_types(batch_tensor: torch.Tensor, types_list: List[Dict[str, Any]]) -> List[torch.Tensor]:
    parts = []
    start = 0
    for t in types_list:
        dim = int(t["dim"])
        parts.append(batch_tensor[:, start : start + dim])
        start += dim
    return parts


# -----------------------------
# Data generation (Example 1)
# -----------------------------
def _generate_blobs1_dgp(
    n_samples: int,
    boundary: float,
    seed: int,
    centers: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    right_margin: float = 0.05,
):
    """
    Pawelczyk et al. (WWW'20) Example 1 ("make blobs") に寄せたDGP。

    目的:
      - 左2クラスタ: decision boundary x2=boundary をまたいで y=0/1 を含む
      - 右クラスタ: boundary を絶対にまたがない (常に x2 < boundary) -> 常に y=0
      - 縦長: y方向の標準偏差をx方向より大きくする

    実装:
      - 3つのガウスからサンプリング
      - 右クラスタは rejection sampling で x2 < boundary - right_margin を保証
    """
    rng = np.random.RandomState(seed)

    if centers is None:
        # NOTE: middle x2 mean is lifted a bit so that it also has enough y=1 mass
        centers = np.array(
            [
                [-9.0, 5.7],   # left (straddles boundary)
                [0.0, 4.9],    # middle (straddles boundary, but less than left)
                [9.5, 1.5],    # right (must stay below boundary)
            ],
            dtype=float,
        )
    else:
        centers = np.asarray(centers, dtype=float)
        assert centers.shape == (3, 2), "blobs_centers must be shape (3,2)"

    if stds is None:
        # 縦長: y方向のstdを大きめ / x方向を小さめ（左2クラスタの分離を強める）
        stds = np.array(
            [
                [0.85, 1.85],  # left: vertical & separated
                [0.85, 1.85],  # middle: vertical & separated
                [1.00, 1.20],  # right: moderate (and clipped)
            ],
            dtype=float,
        )
    else:
        stds = np.asarray(stds, dtype=float)
        assert stds.shape == (3, 2), "stds must be shape (3,2)"

    # サンプル数配分
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    ns = (w * n_samples).astype(int)
    ns[0] += n_samples - ns.sum()  # 端数調整

    X_parts = []
    cluster_ids = []

    for k in range(3):
        mu = centers[k]
        sd = stds[k]
        nk = int(ns[k])

        if k < 2:
            Xk = rng.normal(loc=mu, scale=sd, size=(nk, 2))
        else:
            # right: guarantee x2 < boundary
            collected = []
            need = nk
            while need > 0:
                batch_n = max(need * 3, 512)
                batch = rng.normal(loc=mu, scale=sd, size=(batch_n, 2))
                batch = batch[batch[:, 1] < (boundary - right_margin)]
                if batch.shape[0] == 0:
                    continue
                take = batch[:need]
                collected.append(take)
                need -= take.shape[0]
            Xk = np.vstack(collected)

        X_parts.append(Xk)
        cluster_ids.append(np.full((Xk.shape[0],), k, dtype=int))

    X = np.vstack(X_parts)
    cluster = np.concatenate(cluster_ids)

    perm = rng.permutation(X.shape[0])
    X = X[perm]
    cluster = cluster[perm]
    return X, cluster


def _prepare_data(
    dataset: str,
    samples: int,
    noise: float,
    seed: int,
    boundary: float,
    blobs_centers: Optional[np.ndarray] = None,
    test_size: float = 0.2,
):
    if dataset == "moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        scaler = IdentityScaler().fit(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(X_train), y_train)
        return X_train, X_test, y_train, y_test, scaler, clf

    if dataset == "blobs1":
        X, _cluster = _generate_blobs1_dgp(
            n_samples=samples,
            boundary=boundary,
            seed=seed,
            centers=blobs_centers,
        )
        y = (X[:, 1] > boundary).astype(int)

        stratify = y if (np.unique(y).size == 2 and min(np.bincount(y)) >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )

        scaler = IdentityScaler().fit(X_train)
        clf = StepClassifier(boundary=boundary, feature_index=1)
        return X_train, X_test, y_train, y_test, scaler, clf

    raise ValueError(f"Unknown dataset: {dataset}")


# -----------------------------
# CHVAE training helpers
# -----------------------------
def _train_chvae(
    X_train: np.ndarray,
    types_list: List[Dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> "Graph.CHVAEModel":
    y_partition = [args.latent_y for _ in types_list]
    model = Graph.CHVAEModel(
        types_list=types_list,
        z_dim=args.latent_z,
        y_dim=args.latent_y,
        s_dim=args.latent_s,
        y_dim_partition=y_partition,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_recon = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_list = _split_tensor_by_types(batch, types_list)
            tau = max(args.tau_min, 1.0 - 0.001 * epoch)

            outputs = model(x_list, tau)
            loss = -outputs["ELBO"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += outputs["loss_reconstruction"].detach().mean().item()

        if epoch % args.display == 0:
            avg_recon = epoch_recon / max(1, len(loader))
            kl_z = outputs["KL_z"].detach().mean().item()
            kl_s = outputs["KL_s"].detach().mean().item()
            print(f"Epoch {epoch:03d} | recon {avg_recon:.4f} | KL_z {kl_z:.4f} | KL_s {kl_s:.4f}")

    return model


def _loglik_params_to_numpy(normalization_params: List[Tuple[torch.Tensor, torch.Tensor]]):
    params = []
    for mean, var in normalization_params:
        params.append((mean.detach().cpu().numpy(), var.detach().cpu().numpy()))
    return params


def _compute_normalization_from_train(
    X_train: np.ndarray, types_list: List[Dict[str, Any]], device: torch.device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    params = []
    start = 0
    for t in types_list:
        dim = int(t["dim"])
        data = torch.tensor(X_train[:, start : start + dim], dtype=torch.float32, device=device)
        if t["type"] == "real":
            mean = data.mean(dim=0)
            var = data.var(dim=0, unbiased=False).clamp(min=MIN_VARIANCE)
        elif t["type"] in {"pos", "count"}:
            data_log = torch.log1p(data)
            mean = data_log.mean(dim=0)
            var = data_log.var(dim=0, unbiased=False).clamp(min=MIN_VARIANCE)
        else:
            mean = torch.tensor(0.0, device=device)
            var = torch.tensor(1.0, device=device)
        params.append((mean, var))
        start += dim
    return params


def _apply_normalization(
    x_list: List[torch.Tensor],
    types_list: List[Dict[str, Any]],
    normalization_params: List[Tuple[torch.Tensor, torch.Tensor]],
):
    normalized = []
    noisy = []
    for data, stats, t in zip(x_list, normalization_params, types_list):
        mean, var = stats
        if t["type"] == "real":
            aux = (data - mean) / torch.sqrt(var)
            normalized.append(aux)
            noisy.append(aux + torch.randn_like(aux) * 0.05)
        elif t["type"] in {"pos", "count"}:
            data_log = torch.log1p(data)
            aux = (data_log - mean) / torch.sqrt(var)
            normalized.append(aux)
            noisy.append(aux)
        else:
            normalized.append(data)
            noisy.append(data + torch.randn_like(data) * 0.05)
    return normalized, noisy


def _theta_to_numpy(theta: List):
    theta_np = []
    for t in theta:
        if isinstance(t, list):
            theta_np.append([p.detach().cpu().numpy() for p in t])
        else:
            theta_np.append(t.detach().cpu().numpy())
    return theta_np


# -----------------------------
# Counterfactual search (Algorithm 1 style)
# -----------------------------
def _counterfactual_search(
    model: "Graph.CHVAEModel",
    types_list: List[Dict[str, Any]],
    x: np.ndarray,
    clf: Any,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    normalization_params: List[Tuple[torch.Tensor, torch.Tensor]],
):
    """
    Key change vs. your previous version:
    - stop at the smallest latent radius where any counterfactual exists
    - choose the candidate with the smallest *latent* distance ||z_tilde - z_base||_p
    """
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    x_list = _split_tensor_by_types(x_tensor, types_list)

    normalized, noisy = _apply_normalization(x_list, types_list, normalization_params)

    with torch.no_grad():
        _samples, q_params = model.encoder(noisy, tau=args.tau_min)

    base_pred = int(clf.predict(scaler.transform(x.reshape(1, -1)))[0])
    target_label = 1 - base_pred

    # q_params["z"][0] should be mean (following the TF repo convention)
    z_base = q_params["z"][0].detach().cpu().numpy()  # (1, z_dim)

    normalization_params_np = _loglik_params_to_numpy(normalization_params)

    lower = 0.0
    upper = args.search_step

    best = None  # (x_cf, z_cf, latent_dist)

    for _ in range(args.search_max_steps):
        # sample uniformly in a p-norm shell (approx)
        delta = np.random.randn(args.search_samples, z_base.shape[1])
        norm_p = np.linalg.norm(delta, ord=args.search_norm, axis=1)
        norm_p = np.where(norm_p == 0, MIN_VARIANCE, norm_p)

        radii = np.random.rand(args.search_samples) * (upper - lower) + lower
        delta = delta * (radii / norm_p)[:, None]
        z_tilde = z_base + delta  # (N, z_dim)

        z_tilde_tensor = torch.tensor(z_tilde, dtype=torch.float32, device=device)
        theta_perturbed, _ = model.decoder.decode_only(z_tilde_tensor)
        theta_np = _theta_to_numpy(theta_perturbed)

        x_tilde_parts, _ = Evaluation.loglik_evaluation_test(
            normalized, theta_np, normalization_params_np, types_list
        )
        x_tilde = np.concatenate(x_tilde_parts, axis=1)

        preds = clf.predict(scaler.transform(x_tilde))
        candidate_idx = np.where(preds == target_label)[0]

        if candidate_idx.size:
            # choose smallest latent move (Algorithm 1 intuition)
            latent_dists = np.linalg.norm((z_tilde[candidate_idx] - z_base), ord=args.search_norm, axis=1)
            j = int(candidate_idx[np.argmin(latent_dists)])
            x_cf = x_tilde[j]
            z_cf = z_tilde[j]
            best = (x_cf, z_cf, float(np.min(latent_dists)))
            break

        lower = upper
        upper += args.search_step

    if best is None:
        return None, None, None, target_label

    x_cf, z_cf, latent_dist = best
    input_l2 = float(np.linalg.norm(x_cf - x, ord=2))
    return x_cf, z_cf, input_l2, target_label


# -----------------------------
# Plotting
# -----------------------------
def _plot_overview(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    clf: Any,
    scaler: Any,
    counterfactuals: List[Dict[str, Any]],
    out_path: Path,
    boundary: Optional[float],
    title: str,
):
    plt.figure(figsize=(9.2, 6.5))

    # background P(y=1)
    x1_min, x1_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    x2_min, x2_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 250), np.linspace(x2_min, x2_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict_proba(scaler.transform(grid))[:, 1].reshape(xx.shape)
    plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.25)

    # train points
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s=10, alpha=0.35, label="train y=0")
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s=10, alpha=0.35, label="train y=1")

    # test points (we usually explain f=0 points)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=35, c="navy", alpha=0.70, edgecolor="k", linewidth=0.3, label="test")

    # boundary line (for blobs1)
    if boundary is not None:
        plt.axhline(boundary, color="k", linestyle="--", linewidth=1.3, label=f"true boundary x2={boundary:g}")

    # counterfactual arrows
    for cf in counterfactuals:
        if cf["counterfactual"] is None:
            continue
        orig = cf["original"]
        cand = cf["counterfactual"]
        plt.plot([orig[0], cand[0]], [orig[1], cand[1]], "k--", alpha=0.50)
        plt.scatter(cand[0], cand[1], marker="x", c="black", s=70, label="_cf")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def _plot_fig2f_style(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: Any,
    scaler: Any,
    counterfactuals: List[Dict[str, Any]],
    out_path: Path,
    boundary: float,
):
    """
    Paper Figure 2(f) style:
    - show only test points (f=0 pool) and their counterfactuals (f=1)
    """
    # pick test points that are actually f=0 (using classifier)
    f_test = clf.predict(scaler.transform(X_test))
    test0 = X_test[f_test == 0]

    plt.figure(figsize=(6.2, 5.2))

    # test points (f=0)
    plt.scatter(test0[:, 0], test0[:, 1], s=38, c="#7a1020", alpha=0.80, edgecolor="none", label="test data (f=0)")

    # boundary line
    plt.axhline(boundary, color="#1f77b4", linewidth=2.0)

    # CF points (f=1)
    cfs = [c for c in counterfactuals if c["counterfactual"] is not None]
    if cfs:
        cf_xy = np.stack([c["counterfactual"] for c in cfs], axis=0)
        plt.scatter(cf_xy[:, 0], cf_xy[:, 1], s=46, c="#163a9c", alpha=0.90, edgecolor="none", label="counterfactuals (f=1)")

        # lines
        for c in cfs:
            o = c["original"]
            p = c["counterfactual"]
            plt.plot([o[0], p[0]], [o[1], p[1]], color="gray", alpha=0.35, linewidth=1.0)

    plt.xlabel("First Feature (Continuous)")
    plt.ylabel("Second Feature (Continuous)")
    plt.title("(f) Test data and E(x) by our cchvae.")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[saved] {out_path}")


# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.samples = min(args.samples, 1500)
        args.epochs = min(args.epochs, 60)
        args.search_samples = min(args.search_samples, 250)
        args.counterfactuals = min(args.counterfactuals, 12)

    X_train, X_test, y_train, y_test, scaler, clf = _prepare_data(
        dataset=args.dataset,
        samples=args.samples,
        noise=args.noise,
        seed=args.seed,
        boundary=args.boundary,
    )
    print(f"Classifier test accuracy: {clf.score(scaler.transform(X_test), y_test):.3f}")

    # tabular -> all real 1D features
    types_list = [{"type": "real", "dim": 1} for _ in range(X_train.shape[1])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalization_params = _compute_normalization_from_train(X_train, types_list, device)
    model = _train_chvae(X_train, types_list, device, args)

    # choose points with f(x)=0
    preds_test = clf.predict(scaler.transform(X_test))
    pool = np.where(preds_test == 0)[0]
    if pool.size < args.counterfactuals:
        pool = np.arange(min(len(X_test), args.counterfactuals))
    chosen = pool[: args.counterfactuals]

    counterfactuals = []
    for idx in tqdm(chosen, desc="Searching counterfactuals"):
        original = X_test[idx]
        cf, z_cf, input_dist, target_label = _counterfactual_search(
            model, types_list, original, clf, scaler, device, args, normalization_params
        )
        counterfactuals.append(
            {
                "original": original,
                "counterfactual": cf,
                "latent": z_cf,
                "distance": input_dist,
                "target": target_label,
            }
        )

    found = [c for c in counterfactuals if c["counterfactual"] is not None]
    if found:
        avg_dist = float(np.mean([c["distance"] for c in found]))
        print(f"Found {len(found)}/{len(counterfactuals)} counterfactuals. Avg input L2 distance: {avg_dist:.3f}")
    else:
        print("No counterfactuals were found with the current settings.")

    # plots
    out = Path(args.plot_path)
    boundary = args.boundary if args.dataset == "blobs1" else None
    _plot_overview(
        X_train, X_test, y_train, y_test, clf, scaler, counterfactuals,
        out, boundary, title=f"{args.dataset} counterfactuals with C-CHVAE"
    )
    if args.dataset == "blobs1":
        out2 = out.with_name(out.stem + "_fig2f" + out.suffix)
        _plot_fig2f_style(X_test, y_test, clf, scaler, counterfactuals, out2, boundary=args.boundary)


def build_parser():
    p = argparse.ArgumentParser(description="Artificial data experiments with C-CHVAE (PyTorch)")
    p.add_argument("--dataset", type=str, default="blobs1", choices=["blobs1", "moons"])
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--boundary", type=float, default=6.0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--learning-rate", type=float, default=1e-3)

    # latent dims
    p.add_argument("--latent-z", type=int, default=2)
    p.add_argument("--latent-s", type=int, default=3)   # = number of modes (3) for blobs1
    p.add_argument("--latent-y", type=int, default=4)

    # gumbel temp
    p.add_argument("--tau-min", type=float, default=1e-3)
    p.add_argument("--display", type=int, default=20)

    # CF search
    p.add_argument("--counterfactuals", type=int, default=25)
    p.add_argument("--search-samples", type=int, default=1200)
    p.add_argument("--search-step", type=float, default=0.25)
    p.add_argument("--search-max-steps", type=int, default=140)
    p.add_argument("--search-norm", type=int, default=2)

    p.add_argument("--plot-path", type=str, default="outputs/exp1_blobs_cchvae.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args)
