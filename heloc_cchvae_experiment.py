# heloc_experiments_cchvae.py
"""
HELOC reproduction (Figure 4 style) for Pawelczyk et al. WWW'20:
- CCHVAE only (no GS/AR/HCLS)
- Metrics:
  (a) LOF: predicted local outliers (%) vs neighbors k
  (b) Connectedness: not connected (%) vs epsilon
PyTorch-only, based on the structure of artificial_experiments.py.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# PyTorch ports (must be torch versions)
import Graph  # noqa: E402
import Evaluation  # noqa: E402


MIN_VARIANCE = 1e-6
KNOWN_TYPES = {"real", "pos", "count", "cat", "ord", "binary"}


def _read_csv_matrix(path: str) -> np.ndarray:
    # robust: assume numeric, headerless
    df = pd.read_csv(path, header=None)
    return df.to_numpy(dtype=np.float32)


def _read_csv_vector(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    y = df.to_numpy().reshape(-1)
    # allow floats but cast to int labels
    return y.astype(int)


def _read_types_csv(path: str) -> List[Dict[str, Any]]:
    """
    Try to parse a types CSV like:
      - columns include 'type' and 'dim', or
      - [feature_name, type, dim], or
      - [type, dim]
    Returns: [{'type': str, 'dim': int}, ...]
    """
    df = pd.read_csv(path)

    # find type column
    type_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.lower()
        if (vals.isin(KNOWN_TYPES).mean() > 0.5) and (vals.nunique() <= 20):
            type_col = c
            break
    if type_col is None and "type" in df.columns:
        type_col = "type"

    # find dim column
    dim_col = None
    for c in df.columns:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().mean() > 0.9:
            dim_col = c
            break
    if dim_col is None and "dim" in df.columns:
        dim_col = "dim"

    # fallback: maybe 2 columns without header
    if type_col is None or dim_col is None:
        df2 = pd.read_csv(path, header=None)
        if df2.shape[1] >= 2:
            # heuristics: col0 or col1 might be type
            cands = []
            for idx in range(df2.shape[1]):
                vals = df2.iloc[:, idx].astype(str).str.lower()
                cands.append((idx, vals.isin(KNOWN_TYPES).mean()))
            type_idx = max(cands, key=lambda x: x[1])[0]
            # dim idx: choose a numeric column != type_idx
            dim_idx = None
            for idx in range(df2.shape[1]):
                if idx == type_idx:
                    continue
                numeric = pd.to_numeric(df2.iloc[:, idx], errors="coerce")
                if numeric.notna().mean() > 0.9:
                    dim_idx = idx
                    break
            if dim_idx is None:
                raise ValueError(f"Could not detect dim column in {path}")
            types = []
            for i in range(len(df2)):
                t = str(df2.iloc[i, type_idx]).lower()
                d = int(pd.to_numeric(df2.iloc[i, dim_idx], errors="raise"))
                types.append({"type": t, "dim": d})
            return types

        raise ValueError(f"Could not parse types file: {path}")

    types = []
    for _, row in df.iterrows():
        t = str(row[type_col]).lower()
        d = int(pd.to_numeric(row[dim_col], errors="raise"))
        if t not in KNOWN_TYPES:
            # allow custom but keep it
            t = str(row[type_col]).lower()
        types.append({"type": t, "dim": d})
    return types


def _split_tensor_by_types(batch: torch.Tensor, types_list: List[Dict[str, Any]]) -> List[torch.Tensor]:
    parts = []
    start = 0
    for t in types_list:
        dim = int(t["dim"])
        parts.append(batch[:, start : start + dim])
        start += dim
    return parts


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
            data_log = torch.log1p(data.clamp_min(0))
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
            data_log = torch.log1p(data.clamp_min(0))
            aux = (data_log - mean) / torch.sqrt(var)
            normalized.append(aux)
            noisy.append(aux)
        else:
            normalized.append(data)
            noisy.append(data + torch.randn_like(data) * 0.05)
    return normalized, noisy


def _loglik_params_to_numpy(normalization_params: List[Tuple[torch.Tensor, torch.Tensor]]):
    return [(m.detach().cpu().numpy(), v.detach().cpu().numpy()) for (m, v) in normalization_params]


def _theta_to_numpy(theta: List):
    theta_np = []
    for t in theta:
        if isinstance(t, list):
            theta_np.append([p.detach().cpu().numpy() for p in t])
        else:
            theta_np.append(t.detach().cpu().numpy())
    return theta_np


def _train_cchvae(
    X_free_train: np.ndarray,
    X_c_train: np.ndarray,
    types_list: List[Dict[str, Any]],
    types_list_c: List[Dict[str, Any]],
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
        types_list_c=types_list_c,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_free_train, dtype=torch.float32),
        torch.tensor(X_c_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        tau = max(args.tau_min, 1.0 - 0.001 * epoch)
        recon_sum = 0.0

        for free_b, c_b in loader:
            free_b = free_b.to(device)
            c_b = c_b.to(device)

            x_list = _split_tensor_by_types(free_b, types_list)
            x_list_c = _split_tensor_by_types(c_b, types_list_c)

            outputs = model(x_list, tau, x_list_c)
            loss = -outputs["ELBO"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_sum += outputs["loss_reconstruction"].detach().mean().item()

        if epoch % args.display == 0:
            avg_recon = recon_sum / max(1, len(loader))
            kl_z = outputs["KL_z"].detach().mean().item()
            kl_s = outputs["KL_s"].detach().mean().item()
            print(f"Epoch {epoch:03d} | recon {avg_recon:.4f} | KL_z {kl_z:.4f} | KL_s {kl_s:.4f}")

    return model


@torch.no_grad()
def _encode_z_mean(
    model: "Graph.CHVAEModel",
    types_list: List[Dict[str, Any]],
    types_list_c: List[Dict[str, Any]],
    x_free: np.ndarray,
    x_c: np.ndarray,
    device: torch.device,
    tau: float,
    normalization_params_free: List[Tuple[torch.Tensor, torch.Tensor]],
    normalization_params_c: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> np.ndarray:
    model.eval()
    xf = torch.tensor(x_free, dtype=torch.float32, device=device)
    xc = torch.tensor(x_c, dtype=torch.float32, device=device)

    x_list = _split_tensor_by_types(xf, types_list)
    x_list_c = _split_tensor_by_types(xc, types_list_c)

    # normalize/noise free side (consistent with your artificial_experiments.py decoding path)
    normalized_free, noisy_free = _apply_normalization(x_list, types_list, normalization_params_free)

    # normalize conditional side if params provided; else pass raw
    if normalization_params_c is not None:
        normalized_c, noisy_c = _apply_normalization(x_list_c, types_list_c, normalization_params_c)
        x_list_c_in = noisy_c
    else:
        x_list_c_in = x_list_c

    _samples, q_params = model.encoder(noisy_free, tau=tau, x_list_c=x_list_c_in, deterministic_s=True)
    z = q_params["z"][0].detach().cpu().numpy()  # (N, z_dim)
    return z


def _counterfactual_search_cchvae(
    model: "Graph.CHVAEModel",
    types_list: List[Dict[str, Any]],
    types_list_c: List[Dict[str, Any]],
    x_free: np.ndarray,
    x_c: np.ndarray,
    clf: LogisticRegression,
    scaler: StandardScaler,
    device: torch.device,
    args: argparse.Namespace,
    normalization_params_free: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[int]]:
    """
    Algorithm-1 style search in latent space:
    - expand latent radius in steps
    - stop at first radius shell where any CF exists
    - choose CF with smallest latent ||Δz||_p among those candidates
    """
    model.eval()

    # base prediction on full vector [x_c, x_free]
    x_full = np.concatenate([x_c, x_free], axis=0).reshape(1, -1)
    base_pred = int(clf.predict(scaler.transform(x_full))[0])
    target_label = 1 - base_pred

    # encode base z
    xf_t = torch.tensor(x_free, dtype=torch.float32, device=device).unsqueeze(0)
    xc_t = torch.tensor(x_c, dtype=torch.float32, device=device).unsqueeze(0)

    x_list = _split_tensor_by_types(xf_t, types_list)
    x_list_c_1 = _split_tensor_by_types(xc_t, types_list_c)
    normalized_free, noisy_free = _apply_normalization(x_list, types_list, normalization_params_free)

    with torch.no_grad():
        _samples, q_params = model.encoder(noisy_free, tau=args.tau_min, x_list_c=x_list_c_1, deterministic_s=True)
    z_base = q_params["z"][0].detach().cpu().numpy()  # (1, z_dim)

    normalization_params_np = _loglik_params_to_numpy(normalization_params_free)

    lower = 0.0
    upper = args.search_step

    for _ in range(args.search_max_steps):
        # sample approx-uniform in p-norm shell
        delta = np.random.randn(args.search_samples, z_base.shape[1]).astype(np.float32)
        norm_p = np.linalg.norm(delta, ord=args.search_norm, axis=1)
        norm_p = np.where(norm_p == 0, MIN_VARIANCE, norm_p)
        radii = np.random.rand(args.search_samples).astype(np.float32) * (upper - lower) + lower
        delta = delta * (radii / norm_p)[:, None]
        z_tilde = z_base + delta  # (S, z_dim)

        z_t = torch.tensor(z_tilde, dtype=torch.float32, device=device)

        # repeat condition for S samples
        xc_rep = np.repeat(x_c.reshape(1, -1), args.search_samples, axis=0)
        xc_rep_t = torch.tensor(xc_rep, dtype=torch.float32, device=device)
        x_list_c = _split_tensor_by_types(xc_rep_t, types_list_c)

        # decode -> theta -> sample x_free candidates
        try:
            theta_perturbed, _ = model.decoder.decode_only(z_t, x_list_c=x_list_c)
        except TypeError:
            # some ports use positional or no keyword
            try:
                theta_perturbed, _ = model.decoder.decode_only(z_t, x_list_c)
            except TypeError:
                theta_perturbed, _ = model.decoder.decode_only(z_t)

        theta_np = _theta_to_numpy(theta_perturbed)

        x_tilde_parts, _ = Evaluation.loglik_evaluation_test(
            normalized_free, theta_np, normalization_params_np, types_list
        )
        x_free_tilde = np.concatenate(x_tilde_parts, axis=1).astype(np.float32)  # (S, D_free)

        x_full_tilde = np.concatenate([xc_rep.astype(np.float32), x_free_tilde], axis=1)
        preds = clf.predict(scaler.transform(x_full_tilde)).astype(int)

        cand = np.where(preds == target_label)[0]
        if cand.size > 0:
            # choose smallest latent distance among candidates in this first-success shell
            dz = z_tilde[cand] - z_base
            dz_norm = np.linalg.norm(dz, ord=args.search_norm, axis=1)
            best = cand[int(np.argmin(dz_norm))]
            return x_free_tilde[best], z_tilde[best], float(dz_norm.min()), int(target_label)

        lower = upper
        upper += args.search_step

    return None, None, None, None


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _connectedness_not_connected_pct(
    Z_hplus: np.ndarray,
    anchor_mask: np.ndarray,
    Z_cf: np.ndarray,
    eps: float,
    p: int = 1,
) -> float:
    """
    Build epsilon-graph over H+ points, find components, then:
    CF is connected if it links (within eps) to a component containing any anchor.
    Returns: not connected percentage.
    """
    n = Z_hplus.shape[0]
    if n == 0 or Z_cf.shape[0] == 0 or anchor_mask.sum() == 0:
        return 100.0

    # radius neighbors among H+
    nn = NearestNeighbors(radius=eps, metric="minkowski", p=p)
    nn.fit(Z_hplus)
    neigh = nn.radius_neighbors(Z_hplus, return_distance=False)

    uf = UnionFind(n)
    for i in range(n):
        for j in neigh[i]:
            if j != i:
                uf.union(i, int(j))

    comp = np.array([uf.find(i) for i in range(n)], dtype=int)
    anchor_comps = set(comp[anchor_mask])

    # for each CF, check neighbors and their component
    neigh_cf = nn.radius_neighbors(Z_cf, return_distance=False)
    connected = np.zeros(Z_cf.shape[0], dtype=bool)
    for i, ns in enumerate(neigh_cf):
        if len(ns) == 0:
            continue
        if any(comp[int(j)] in anchor_comps for j in ns):
            connected[i] = True

    not_connected_pct = 100.0 * (1.0 - connected.mean())
    return float(not_connected_pct)


def run(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- load preprocessed HELOC ---
    X_free = _read_csv_matrix(args.x_free_path)
    X_c = _read_csv_matrix(args.x_c_path)
    y = _read_csv_vector(args.y_path)

    assert X_free.shape[0] == X_c.shape[0] == y.shape[0], "Row count mismatch among x/x_c/y"

    # --- parse types ---
    types_list = _read_types_csv(args.types_free_path)
    types_list_c = _read_types_csv(args.types_c_path)

    # sanity: dims match
    dim_free = sum(int(t["dim"]) for t in types_list)
    dim_c = sum(int(t["dim"]) for t in types_list_c)
    if dim_free != X_free.shape[1]:
        raise ValueError(f"types_free dims sum={dim_free} but X_free has {X_free.shape[1]} cols")
    if dim_c != X_c.shape[1]:
        raise ValueError(f"types_c dims sum={dim_c} but X_c has {X_c.shape[1]} cols")

    # --- split train/test (shared test) ---
    Xf_tr, Xf_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        X_free, X_c, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if (np.unique(y).size == 2) else None,
    )

    # --- classifier: logistic regression (RLR) on full vector [x_c, x_free] ---
    X_tr_full = np.concatenate([Xc_tr, Xf_tr], axis=1)
    X_te_full = np.concatenate([Xc_te, Xf_te], axis=1)

    scaler = StandardScaler().fit(X_tr_full)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        n_jobs=None,
        random_state=args.seed,
    ).fit(scaler.transform(X_tr_full), y_tr)

    acc = clf.score(scaler.transform(X_te_full), y_te)
    print(f"[clf] test accuracy: {acc:.4f}")

    # --- train CCHVAE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_free = _compute_normalization_from_train(Xf_tr, types_list, device=device)

    model = _train_cchvae(
        X_free_train=Xf_tr,
        X_c_train=Xc_tr,
        types_list=types_list,
        types_list_c=types_list_c,
        device=device,
        args=args,
    )

    # --- pick test points with f(x)=0 and search counterfactuals to f(x)=1 ---
    preds_te = clf.predict(scaler.transform(X_te_full)).astype(int)
    pool = np.where(preds_te == 0)[0]
    if pool.size == 0:
        raise RuntimeError("No test points with predicted label 0; cannot run CF-to-1 search.")
    rng = np.random.RandomState(args.seed)
    rng.shuffle(pool)
    chosen = pool[: args.n_counterfactuals]

    cfs_free = []
    cfs_z = []
    valid = 0

    for idx in tqdm(chosen, desc="CF search (CCHVAE)"):
        xf = Xf_te[idx]
        xc = Xc_te[idx]
        xf_cf, z_cf, dz, target = _counterfactual_search_cchvae(
            model, types_list, types_list_c, xf, xc, clf, scaler, device, args, norm_free
        )
        if xf_cf is not None:
            valid += 1
            cfs_free.append(xf_cf)
            cfs_z.append(z_cf)

    validity = valid / len(chosen)
    print(f"[CF] validity: {valid}/{len(chosen)} = {validity:.3f}")

    if valid == 0:
        raise RuntimeError("No counterfactuals found. Try increasing --search-samples/--search-max-steps/--latent dims.")

    Z_cf = np.stack(cfs_z, axis=0).astype(np.float32)

    # --- compute latent Z for training sets for faithfulness metrics ---
    # H+ : predicted positive on train
    preds_tr = clf.predict(scaler.transform(X_tr_full)).astype(int)
    hplus_mask = preds_tr == 1
    anchor_mask = hplus_mask & (y_tr == 1)  # H+ ∩ D+

    # encode Z for all H+ points (and anchors)
    Z_tr_hplus = _encode_z_mean(
        model, types_list, types_list_c,
        x_free=Xf_tr[hplus_mask],
        x_c=Xc_tr[hplus_mask],
        device=device,
        tau=args.tau_min,
        normalization_params_free=norm_free,
        normalization_params_c=None,  # optional
    )
    y_tr_hplus = y_tr[hplus_mask].astype(int)
    anchor_mask_hplus = (y_tr_hplus == 1)

    Z_tr_anchor = Z_tr_hplus[anchor_mask_hplus]
    if Z_tr_anchor.shape[0] < 10:
        print("[warn] Very few anchor points (H+∩D+). LOF/connectedness may be unstable.")

    # --- (a) LOF curve: predicted local outliers (%) vs k ---
    k_list = list(range(1, args.lof_k_max + 1))
    outlier_pct = []
    for k in k_list:
        # novelty=True allows scoring new points
        lof = LocalOutlierFactor(
            n_neighbors=max(2, k),
            novelty=True,
            metric="minkowski",
            p=args.metric_p,
        )
        lof.fit(Z_tr_anchor)
        pred = lof.predict(Z_cf)  # -1 outlier, +1 inlier
        outlier_pct.append(100.0 * (pred == -1).mean())

    # --- (b) connectedness: not connected (%) vs epsilon ---
    eps_list = [float(x) for x in args.eps_list.split(",")]
    not_conn_pct = []
    for eps in eps_list:
        pct = _connectedness_not_connected_pct(
            Z_hplus=Z_tr_hplus,
            anchor_mask=anchor_mask_hplus,
            Z_cf=Z_cf,
            eps=eps,
            p=args.metric_p,
        )
        not_conn_pct.append(pct)

    # --- plot (Figure4 style) ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "heloc_fig4_cchvae.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(k_list, outlier_pct, marker="o")
    axes[0].set_xlabel("Neighbors (k)")
    axes[0].set_ylabel("Predicted local outliers (%)")
    axes[0].set_title("LOF score (CCHVAE)")

    axes[1].plot(eps_list, not_conn_pct, marker="o")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel("Not connected (%)")
    axes[1].set_title("Connectedness (CCHVAE)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"[saved] {fig_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("HELOC Figure4 reproduction (CCHVAE only, PyTorch)")

    # data
    p.add_argument("--x-free-path", type=str, default="./data/heloc/heloc_x.csv", help="mutable/free features (csv)")
    p.add_argument("--x-c-path", type=str, default="./data/heloc/heloc_x_c.csv", help="immutable/protected conditional features (csv)")
    p.add_argument("--y-path", type=str, default="./data/heloc/heloc_y.csv", help="labels (csv)")

    p.add_argument("--types-free-path", type=str, default="./data/heloc/heloc_types.csv", help="types file for x-free")
    p.add_argument("--types-c-path", type=str, default="./data/heloc/heloc_types_c_alt.csv", help="types file for x-c")

    # split
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # cchvae training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--display", type=int, default=20)

    # latent dims (tunable; paper-like defaults are not uniquely specified)
    p.add_argument("--latent-z", type=int, default=10)
    p.add_argument("--latent-s", type=int, default=10)  # mixture components count
    p.add_argument("--latent-y", type=int, default=4)
    p.add_argument("--tau-min", type=float, default=1e-3)

    # CF search
    p.add_argument("--n-counterfactuals", type=int, default=200)
    p.add_argument("--search-samples", type=int, default=2000)
    p.add_argument("--search-step", type=float, default=0.25)
    p.add_argument("--search-max-steps", type=int, default=120)
    p.add_argument("--search-norm", type=int, default=1, help="latent norm p (paper HELOC eps seems consistent with p=1)")

    # metrics
    p.add_argument("--metric-p", type=int, default=1, help="distance metric p for LOF/connectedness in latent")
    p.add_argument("--lof-k-max", type=int, default=20)
    p.add_argument("--eps-list", type=str, default="10,15,20,25,30,35")

    # output
    p.add_argument("--out-dir", type=str, default="./outputs_heloc")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
