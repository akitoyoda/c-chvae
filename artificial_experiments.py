"""
Reproduce the artificial data experiments from Pawelczyk et al. (WWW'20) using the
PyTorch reimplementation of C-CHVAE in this repository.

The script trains a CHVAE on a 2D synthetic (two moons) dataset, fits a simple
classifier, and then searches the latent space for counterfactuals that flip the
classifier's prediction. A visualization of the data manifold, decision
boundary, and discovered counterfactuals is saved to disk.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import Evaluation  # noqa: E402
import Graph  # noqa: E402


def _split_tensor_by_types(batch_tensor: torch.Tensor, types_list: List[Dict[str, str]]) -> List[torch.Tensor]:
    parts = []
    start = 0
    for t in types_list:
        dim = int(t["dim"])
        parts.append(batch_tensor[:, start : start + dim])
        start += dim
    return parts


def _prepare_data(samples: int, noise: float, seed: int):
    X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(X_train), y_train)
    return X_train, X_test, y_train, y_test, scaler, clf


def _train_chvae(
    X_train: np.ndarray,
    types_list: List[Dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> Graph.CHVAEModel:
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
    X_train: np.ndarray, types_list: List[Dict[str, str]], device: torch.device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    params = []
    start = 0
    for t in types_list:
        dim = int(t["dim"])
        data = torch.tensor(X_train[:, start : start + dim], dtype=torch.float32, device=device)
        if t["type"] == "real":
            mean = data.mean(dim=0)
            var = data.var(dim=0, unbiased=False).clamp(min=1e-6)
        elif t["type"] in {"pos", "count"}:
            data_log = torch.log1p(data)
            mean = data_log.mean(dim=0)
            var = data_log.var(dim=0, unbiased=False).clamp(min=1e-6)
        else:
            mean = torch.tensor(0.0, device=device)
            var = torch.tensor(1.0, device=device)
        params.append((mean, var))
        start += dim
    return params


def _apply_normalization(
    x_list: List[torch.Tensor],
    types_list: List[Dict[str, str]],
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
    model: Graph.CHVAEModel,
    types_list: List[Dict[str, str]],
    x: np.ndarray,
    clf: LogisticRegression,
    scaler: StandardScaler,
    device: torch.device,
    args: argparse.Namespace,
    normalization_params: List[Tuple[torch.Tensor, torch.Tensor]],
):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    x_list = _split_tensor_by_types(x_tensor, types_list)

    normalized, noisy = _apply_normalization(x_list, types_list, normalization_params)

    with torch.no_grad():
        samples, q_params = model.encoder(noisy, tau=args.tau_min)

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
        norm_p = np.where(norm_p == 0, 1e-6, norm_p)
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
            return x_tilde[best_idx], z_tilde[best_idx], cf_distances.min(), target_label

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
        return x_tilde[best_idx], random_z[best_idx], cf_distances.min(), target_label

    return None, None, None, target_label


def _plot_results(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    clf: LogisticRegression,
    scaler: StandardScaler,
    counterfactuals: List[Dict[str, np.ndarray]],
    plot_path: Path,
):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y = np.meshgrid(
        np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 200),
        np.linspace(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 200),
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    probs = clf.predict_proba(scaler.transform(grid_points))[:, 1].reshape(grid_x.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, probs, levels=20, cmap="RdBu", alpha=0.4)
    plt.colorbar(contour, label="P(y=1)")
    plt.contour(grid_x, grid_y, probs, levels=[0.5], colors="k", linestyles="--", linewidths=1)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.4, label="train")
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.7, edgecolor="k", label="test"
    )

    for cf in counterfactuals:
        if cf["counterfactual"] is None:
            continue
        orig = cf["original"]
        candidate = cf["counterfactual"]
        plt.plot([orig[0], candidate[0]], [orig[1], candidate[1]], "k--", alpha=0.6)
        plt.scatter(candidate[0], candidate[1], marker="x", c="black", s=60, label="_cf")

    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Artificial data counterfactual search with C-CHVAE")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved visualization to {plot_path}")


def run_experiment(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.samples = min(args.samples, 600)
        args.epochs = min(args.epochs, 40)
        args.search_samples = min(args.search_samples, 200)
        args.counterfactuals = min(args.counterfactuals, 5)

    X_train, X_test, y_train, y_test, scaler, clf = _prepare_data(args.samples, args.noise, args.seed)
    print(f"Classifier test accuracy: {clf.score(scaler.transform(X_test), y_test):.3f}")

    types_list = [{"type": "real", "dim": 1}, {"type": "real", "dim": 1}]
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
        print(f"Found {len(found)}/{len(counterfactuals)} counterfactuals. Avg scaled L2 distance: {avg_distance:.3f}")
    else:
        print("No counterfactuals were found with the current settings.")

    _plot_results(X_train, X_test, y_train, y_test, clf, scaler, counterfactuals, Path(args.plot_path))


def build_parser():
    parser = argparse.ArgumentParser(description="Artificial data experiments with C-CHVAE (PyTorch)")
    parser.add_argument("--samples", type=int, default=2000, help="Total synthetic samples")
    parser.add_argument("--noise", type=float, default=0.15, help="Noise level for make_moons")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs for CHVAE")
    parser.add_argument("--latent-z", type=int, default=2, help="Latent z dimension")
    parser.add_argument("--latent-s", type=int, default=3, help="Latent s (categorical) dimension")
    parser.add_argument("--latent-y", type=int, default=4, help="Latent y dimension per feature")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tau-min", type=float, default=1e-3, help="Minimum Gumbel-Softmax temperature")
    parser.add_argument("--display", type=int, default=20, help="Epoch display frequency")
    parser.add_argument("--counterfactuals", type=int, default=10, help="Number of test points to explain")
    parser.add_argument("--search-samples", type=int, default=800, help="Latent perturbations per step")
    parser.add_argument("--search-step", type=float, default=0.25, help="Step size in latent space")
    parser.add_argument("--search-max-steps", type=int, default=120, help="Max search expansions")
    parser.add_argument("--search-norm", type=int, default=2, help="Norm used in latent perturbations")
    parser.add_argument("--plot-path", type=str, default="outputs/artificial_counterfactuals.png", help="Path to save plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Use a faster configuration for smoke tests")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
