"""
Reproduce the artificial data experiments from Pawelczyk et al. (WWW'20) using a
PyTorch reimplementation of C-CHVAE.

This version adds Experiment 1 (Example 1: "make blobs"):
- x = [x1, x2] from a mixture of 3 Gaussians with sigma=1
- y = I(x2 > boundary) with boundary=6 (stylized / constant classifier)

It keeps the existing two-moons option for quick sanity checks.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons, make_blobs
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
# Data generation
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
        # x1: 左(-9), 中(0), 右(9.5)
        # x2: 左/中は boundary より少し下(4.8)に置き、分散でまたぐようにする
        #     右は十分下(1.5)に置く
        centers = np.array(
            [
                [-9.0, 4.8],   # left (straddles boundary)
                [0.0, 4.8],    # middle (straddles boundary)
                [9.5, 1.5],    # right (must stay below boundary)
            ],
            dtype=float,
        )
    else:
        centers = np.asarray(centers, dtype=float)
        assert centers.shape == (3, 2), "blobs_centers must be shape (3,2)"

    if stds is None:
        # 縦長にする: y方向のstdを大きめに
        stds = np.array(
            [
                [1.0, 1.6],  # left: vertical
                [1.0, 1.6],  # middle: vertical
                [1.0, 1.0],  # right: moderate (and clipped)
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
            # 左・中: 普通にサンプル（boundary をまたぐのは分散に任せる）
            Xk = rng.normal(loc=mu, scale=sd, size=(nk, 2))
        else:
            # 右: boundary をまたがないことを保証
            #     x2 < boundary - right_margin を満たす点だけ採用
            collected = []
            need = nk
            # ループが長引かないようにバッチ生成
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

    # シャッフル
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
    rng = np.random.RandomState(seed)

    if dataset == "moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)

        # For moons we train a classifier (kept for debugging / sanity checks)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        # Keep scaling off here too (your CHVAE already normalizes internally)
        scaler = IdentityScaler().fit(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(X_train), y_train)
        return X_train, X_test, y_train, y_test, scaler, clf

    if dataset == "blobs1":
        # ---- Experiment 1 (WWW'20 Example 1: "make blobs") ----
        X, _cluster = _generate_blobs1_dgp(
            n_samples=samples,
            boundary=boundary,
            seed=seed,
            centers=blobs_centers,  # 渡されなければ上のデフォルト中心を使う
        )

        # Label rule from the paper: y = I(x2 > boundary)
        y = (X[:, 1] > boundary).astype(int)

        # stratify は片側が少なすぎると落ちるので保険
        stratify = y if (np.unique(y).size == 2 and min(np.bincount(y)) >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )

        # Example 1 は「与えられた」定数分類器想定（I(x2>6)）
        scaler = IdentityScaler().fit(X_train)
        clf = StepClassifier(boundary=boundary, feature_index=1)
        return X_train, X_test, y_train, y_test, scaler, clf

    raise ValueError(f"Unknown dataset: {dataset}")


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
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    x_list = _split_tensor_by_types(x_tensor, types_list)

    normalized, noisy = _apply_normalization(x_list, types_list, normalization_params)

    with torch.no_grad():
        _samples, q_params = model.encoder(noisy, tau=args.tau_min)

    base_pred = clf.predict(scaler.transform(x.reshape(1, -1)))[0]
    target_label = 1 - base_pred
    z_base = q_params["z"][0].detach().cpu().numpy()
    normalization_params_np = _loglik_params_to_numpy(normalization_params)

    lower = 0.0
    upper = args.search_step
    scaled_base = scaler.transform(x.reshape(1, -1))

    for _ in range(args.search_max_steps):
        delta_z = np.random.randn(args.search_samples, z_base.shape[1])
        norm_p = np.linalg.norm(delta_z, ord=args.search_norm, axis=1)
        norm_p = np.where(norm_p == 0, MIN_VARIANCE, norm_p)
        distances = np.random.rand(args.search_samples) * (upper - lower) + lower
        delta_z = delta_z * (distances / norm_p)[:, None]
        z_tilde = z_base + delta_z

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
            scaled_candidates = scaler.transform(x_tilde[candidate_idx])
            cf_distances = np.linalg.norm(scaled_candidates - scaled_base, axis=1)
            best_idx = candidate_idx[np.argmin(cf_distances)]
            return x_tilde[best_idx], z_tilde[best_idx], float(cf_distances.min()), int(target_label)

        lower = upper
        upper += args.search_step

    # Fallback: sample broadly from the prior if targeted search failed
    random_z = np.random.randn(args.search_samples * 2, z_base.shape[1])
    z_tilde_tensor = torch.tensor(random_z, dtype=torch.float32, device=device)
    theta_perturbed, _ = model.decoder.decode_only(z_tilde_tensor)
    theta_np = _theta_to_numpy(theta_perturbed)
    x_tilde_parts, _ = Evaluation.loglik_evaluation_test(
        normalized, theta_np, normalization_params_np, types_list
    )
    x_tilde = np.concatenate(x_tilde_parts, axis=1)
    preds = clf.predict(scaler.transform(x_tilde))
    candidate_idx = np.where(preds == target_label)[0]
    if candidate_idx.size:
        scaled_candidates = scaler.transform(x_tilde[candidate_idx])
        cf_distances = np.linalg.norm(scaled_candidates - scaled_base, axis=1)
        best_idx = candidate_idx[np.argmin(cf_distances)]
        return x_tilde[best_idx], random_z[best_idx], float(cf_distances.min()), int(target_label)

    return None, None, None, int(target_label)


def _plot_results(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    clf: Any,
    scaler: Any,
    counterfactuals: List[Dict[str, np.ndarray]],
    plot_path: Path,
    boundary: Optional[float] = None,
    title: str = "Artificial data counterfactual search with C-CHVAE",
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    grid_x, grid_y = np.meshgrid(
        np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 240),
        np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 240),
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    probs = clf.predict_proba(scaler.transform(grid_points))[:, 1].reshape(grid_x.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, probs, levels=20, cmap="RdBu", alpha=0.35)
    plt.colorbar(contour, label="P(y=1)")
    plt.contour(grid_x, grid_y, probs, levels=[0.5], colors="k", linestyles="--", linewidths=1)

    if boundary is not None:
        plt.axhline(boundary, color="k", linestyle=":", linewidth=1, alpha=0.8, label=f"x2 = {boundary}")

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.35, label="train")
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.75, edgecolor="k", linewidth=0.4, label="test"
    )

    for cf in counterfactuals:
        if cf["counterfactual"] is None:
            continue
        orig = cf["original"]
        candidate = cf["counterfactual"]
        plt.plot([orig[0], candidate[0]], [orig[1], candidate[1]], "k--", alpha=0.55)
        plt.scatter(candidate[0], candidate[1], marker="x", c="black", s=60, label="_cf")

    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visualization to {plot_path}")


def run_experiment(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.samples = min(args.samples, 1200)
        args.epochs = min(args.epochs, 40)
        args.search_samples = min(args.search_samples, 200)
        args.counterfactuals = min(args.counterfactuals, 10)

    X_train, X_test, y_train, y_test, scaler, clf = _prepare_data(
        dataset=args.dataset,
        samples=args.samples,
        noise=args.noise,
        seed=args.seed,
        boundary=args.boundary,
    )
    print(f"Classifier test accuracy: {clf.score(scaler.transform(X_test), y_test):.3f}")

    types_list = [{"type": "real", "dim": 1} for _ in range(X_train.shape[1])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _train_chvae(X_train, types_list, device, args)
    normalization_params = _compute_normalization_from_train(X_train, types_list, device)

    preds_test = clf.predict(scaler.transform(X_test))
    target_pool_idx = np.where(preds_test == 0)[0]
    if target_pool_idx.size < args.counterfactuals:
        target_pool_idx = np.arange(min(len(X_test), args.counterfactuals))
    chosen_idx = target_pool_idx[: args.counterfactuals]

    counterfactuals = []
    for idx in tqdm(chosen_idx, desc="Searching counterfactuals"):
        original = X_test[idx]
        cf, z_cf, distance, target_label = _counterfactual_search(
            model, types_list, original, clf, scaler, device, args, normalization_params
        )
        counterfactuals.append(
            {
                "original": original,
                "counterfactual": cf,
                "latent": z_cf,
                "distance": distance,
                "target": target_label,
            }
        )

    found = [c for c in counterfactuals if c["counterfactual"] is not None]
    if found:
        avg_distance = np.mean([c["distance"] for c in found])
        print(f"Found {len(found)}/{len(counterfactuals)} counterfactuals. Avg input L2 distance: {avg_distance:.3f}")
    else:
        print("No counterfactuals were found with the current settings.")

    title = f"{args.dataset} counterfactuals with C-CHVAE"
    boundary = args.boundary if args.dataset == "blobs1" else None
    _plot_results(X_train, X_test, y_train, y_test, clf, scaler, counterfactuals, Path(args.plot_path), boundary, title)


def build_parser():
    parser = argparse.ArgumentParser(description="Artificial data experiments with C-CHVAE (PyTorch)")
    parser.add_argument("--dataset", type=str, default="blobs1", choices=["blobs1", "moons"], help="Synthetic dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Total synthetic samples")
    parser.add_argument("--noise", type=float, default=0.15, help="Noise level for make_moons (ignored for blobs1)")
    parser.add_argument("--boundary", type=float, default=6.0, help="Decision boundary for blobs1: y=I(x2>boundary)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs for CHVAE")
    parser.add_argument("--latent-z", type=int, default=2, help="Latent z dimension")
    parser.add_argument("--latent-s", type=int, default=3, help="Latent s (categorical) dimension")
    parser.add_argument("--latent-y", type=int, default=4, help="Latent y dimension per feature")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tau-min", type=float, default=1e-3, help="Minimum Gumbel-Softmax temperature")
    parser.add_argument("--display", type=int, default=20, help="Epoch display frequency")
    parser.add_argument("--counterfactuals", type=int, default=25, help="Number of test points to explain")
    parser.add_argument("--search-samples", type=int, default=1000, help="Latent perturbations per step")
    parser.add_argument("--search-step", type=float, default=0.25, help="Step size in latent space")
    parser.add_argument("--search-max-steps", type=int, default=120, help="Max search expansions")
    parser.add_argument("--search-norm", type=int, default=2, help="Norm used in latent perturbations")
    parser.add_argument("--plot-path", type=str, default="outputs/exp1_blobs_cchvae.png", help="Path to save plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Use a faster configuration for smoke tests")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
