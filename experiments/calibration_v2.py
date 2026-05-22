"""
Calibration v2: constructive validation that the diagnostic discriminates
across the full separability spectrum, including the distributed end.

A plain full-parameter fine-tune (the FULL_FT variant) reaches only
aggregate rho_eps ~0.17, far below the distributed limit rho_eps -> 1.
To validate discrimination at the upper end of the spectrum we add a
third variant explicitly designed to reach the distributed regime:
full-parameter training with an auxiliary rank-maximization regularizer
that pushes the per-layer modification matrix toward orthogonal column
structure.

Variants
--------
  STEERING         rank-1 by construction (single fixed direction added on
                   safety-relevant inputs at inference); rho_eps -> 1/d.
  FULL_FT          v1 full-parameter fine-tune from base with safety loss
                   only; rho_eps empirically ~0.16-0.39 at MLP scale.
  DISTRIBUTED      v2: full-parameter fine-tune from base with safety loss
                   + rank-maximization regularizer that penalizes Gram
                   off-diagonal magnitude across safety-relevant inputs.
                   Target: rho_eps >= 0.5 at all hidden layers.
"""
import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Local config (lighter than v1; v2 prioritizes hitting the distributed end)
D_IN = 20
D_HIDDEN = 128
N_LAYERS = 3
N_TRAIN = 4000
N_PROBE = 800
SAFETY_FRAC = 0.3
N_EPOCHS_BASE = 300
N_EPOCHS_FT = 400
N_EPOCHS_DIST = 600
LR = 1e-3
N_SEEDS = 5
EPS = 0.05
DIST_REG_WEIGHT = 50.0  # weight for the rank-maximization regularizer
STABLE_RANK_TARGET = 64.0  # target stable rank (out of d=128)
HERE = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- data + model (copied from v1 with tweaks) ----------

def generate_data(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.randn(n, D_IN)
    is_safety = (X[:, :3] > 1.0).all(dim=1)
    n_safety_target = int(n * SAFETY_FRAC)
    n_current = is_safety.sum().item()
    if n_current < n_safety_target:
        deficit = n_safety_target - n_current
        non_idx = (~is_safety).nonzero(as_tuple=True)[0]
        flip = non_idx[torch.randperm(len(non_idx))[:deficit]]
        X[flip, :3] = torch.abs(X[flip, :3]) + 1.1
        is_safety = (X[:, :3] > 1.0).all(dim=1)
    task_signal = X[:, 3] * X[:, 4] - X[:, 5] * X[:, 6]
    y_task = (task_signal > 0).float()
    y_safe = y_task.clone()
    y_safe[is_safety] = 0.0
    return X, y_task, y_safe, is_safety


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        prev = D_IN
        for _ in range(N_LAYERS):
            layers.append(nn.Linear(prev, D_HIDDEN))
            prev = D_HIDDEN
        self.hidden_layers = nn.ModuleList(layers)
        self.readout = nn.Linear(D_HIDDEN, 1)

    def forward(self, x):
        hiddens = []
        h = x
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
            hiddens.append(h)
        return self.readout(h).squeeze(-1), hiddens


def base_train(X, y, epochs=N_EPOCHS_BASE):
    m = DeepMLP().to(DEVICE)
    X = X.to(DEVICE); y = y.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    L = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = m(X)
        loss = L(logits, y)
        loss.backward()
        opt.step()
    return m


def ft_safety(base, X, y_safe, epochs=N_EPOCHS_FT):
    m = copy.deepcopy(base).to(DEVICE)
    X = X.to(DEVICE); y_safe = y_safe.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    L = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = m(X)
        loss = L(logits, y_safe)
        loss.backward()
        opt.step()
    return m


# ---------- distributed variant: rank-maximization regularizer ----------

def stable_rank_loss(M, target, eps=1e-8):
    """
    Maximize the stable rank ||M||_F^2 / ||M||_2^2 toward `target`.
    Stable rank is differentiable (gradients flow through SVD).
    The loss is squared shortfall, scaled to be O(1) so the regularizer
    weight is interpretable.
    """
    s = torch.linalg.svdvals(M)
    s2 = s ** 2
    sr = s2.sum() / (s2.max() + eps)
    return torch.clamp(target - sr, min=0.0) ** 2 / (target ** 2)


def column_orthogonality_loss(M):
    """
    Encourage columns of M to be orthogonal AND of equal magnitude.
    Penalty = ||G - (||M||_F^2 / n) I||_F^2 / ||M||_F^4, where G = M^T M.
    Reaching zero means M has equal singular values (i.e., maximally
    distributed).
    """
    n = M.shape[1]
    fro2 = (M ** 2).sum().clamp(min=1e-8)
    G = M.T @ M
    target_diag = (fro2 / n) * torch.eye(n, device=M.device, dtype=M.dtype)
    return ((G - target_diag) ** 2).sum() / (fro2 ** 2)


def ft_distributed(base, X_train, y_safe_train, is_safety_train,
                   epochs=N_EPOCHS_DIST, reg=DIST_REG_WEIGHT):
    """
    Distributed-variant fine-tune: standard safety loss plus a regularizer
    that forces the per-layer alignment-modification matrix on safety
    inputs to be orthogonal across columns.

    To compute the modification matrix at training time, we keep a frozen
    copy of the base activations on the safety inputs and compare against
    the live aligned activations each step.
    """
    base.eval()
    X_train = X_train.to(DEVICE)
    y_safe_train = y_safe_train.to(DEVICE)
    is_safety_train = is_safety_train.to(DEVICE)
    with torch.no_grad():
        _, h_base_train = base(X_train)
        h_base_safety = [h[is_safety_train].detach() for h in h_base_train]

    m = copy.deepcopy(base).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    L = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        opt.zero_grad()
        logits, h_align_train = m(X_train)
        safety_loss = L(logits, y_safe_train)
        reg_loss = 0.0
        for h_a, h_b in zip(h_align_train, h_base_safety):
            M = (h_a[is_safety_train] - h_b).T  # (d, n_safety)
            # Encourage equal-magnitude orthogonal columns (maximally
            # distributed across the n_safety inputs)
            reg_loss = reg_loss + column_orthogonality_loss(M)
            # Push the stable rank toward the target
            reg_loss = reg_loss + stable_rank_loss(M, STABLE_RANK_TARGET)
        total = safety_loss + reg * reg_loss
        total.backward()
        opt.step()
    return m


# ---------- evaluation ----------

def eff_rank(M, eps=EPS):
    if np.linalg.norm(M) == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, 1 - eps) + 1)


def per_layer_eval(h_base_all, h_other_all, mask, eps=EPS):
    """Returns per-layer eff_rank and rho_eps."""
    out = []
    for li in range(N_LAYERS):
        h_b = h_base_all[li].detach().cpu().numpy()
        h_o = h_other_all[li].detach().cpu().numpy()
        mn = mask.detach().cpu().numpy()
        h_b_m = h_b[mn]
        h_o_m = h_o[mn]
        n = min(len(h_b_m), len(h_o_m))
        if n < 2:
            out.append({'eff_rank': 0, 'rho_eps': 0.0,
                        'rho_min_dn': 0.0, 'd': h_b.shape[1]})
            continue
        M = (h_o_m[:n] - h_b_m[:n]).T
        d = M.shape[0]
        k = eff_rank(M, eps)
        out.append({'eff_rank': k,
                    'rho_eps': k / d,
                    'rho_min_dn': k / min(d, n),
                    'd': d, 'n': n})
    return out


# ---------- experiment ----------

def run_seed(seed):
    print(f"\n--- seed {seed} ---")
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_task_tr, y_safe_tr, is_s_tr = generate_data(N_TRAIN, seed)
    X_pr, y_task_pr, y_safe_pr, is_s_pr = generate_data(N_PROBE, seed + 7777)

    base = base_train(X_tr, y_task_tr)
    ft = ft_safety(base, X_tr, y_safe_tr)
    dist = ft_distributed(base, X_tr, y_safe_tr, is_s_tr)

    base.eval(); ft.eval(); dist.eval()

    # Move probe data to device
    X_pr = X_pr.to(DEVICE)
    y_safe_pr = y_safe_pr.to(DEVICE)
    sm = is_s_pr.to(DEVICE)
    cm = ~is_s_pr.to(DEVICE)

    with torch.no_grad():
        _, h_base = base(X_pr)
        _, h_ft   = ft(X_pr)
        _, h_dist = dist(X_pr)

        # behavior compliance on D_s
        ft_logits, _ = ft(X_pr)
        dist_logits, _ = dist(X_pr)
        ft_s_compliance = ((ft_logits > 0).float()[sm]
                           == y_safe_pr[sm]).float().mean().item()
        dist_s_compliance = ((dist_logits > 0).float()[sm]
                             == y_safe_pr[sm]).float().mean().item()

    # construct steering activations (rank-1 by construction)
    h_steer = []
    for li in range(N_LAYERS):
        h_b = h_base[li].clone()
        d = h_b.shape[1]
        v = torch.randn(d, device=DEVICE); v = v / v.norm()
        diff_mag = (h_ft[li][sm].mean(0) - h_b[sm].mean(0)).norm()
        steer_vec = (diff_mag * v).to(h_b.dtype)
        h_b[sm] = h_b[sm] + steer_vec
        h_steer.append(h_b)

    eval_steer_s = per_layer_eval(h_base, h_steer, sm)
    eval_steer_c = per_layer_eval(h_base, h_steer, cm)
    eval_ft_s    = per_layer_eval(h_base, h_ft,    sm)
    eval_ft_c    = per_layer_eval(h_base, h_ft,    cm)
    eval_dist_s  = per_layer_eval(h_base, h_dist,  sm)
    eval_dist_c  = per_layer_eval(h_base, h_dist,  cm)

    print(f"  ft  safety-compliance = {ft_s_compliance:.3f}   "
          f"  dist safety-compliance = {dist_s_compliance:.3f}")
    for li in range(N_LAYERS):
        print(f"  layer {li}  rho_eps:  "
              f"steer={eval_steer_s[li]['rho_eps']:.3f}  "
              f"ft={eval_ft_s[li]['rho_eps']:.3f}  "
              f"dist={eval_dist_s[li]['rho_eps']:.3f}   "
              f"(d={eval_steer_s[li]['d']})")

    return {
        'seed': seed,
        'ft_safety_compliance': ft_s_compliance,
        'dist_safety_compliance': dist_s_compliance,
        'steering_safety':    eval_steer_s,
        'steering_control':   eval_steer_c,
        'ft_safety':          eval_ft_s,
        'ft_control':         eval_ft_c,
        'distributed_safety': eval_dist_s,
        'distributed_control': eval_dist_c,
    }


def main():
    print("=" * 64)
    print("Calibration v2 — three-variant constructive calibration")
    print("=" * 64)
    print(f"  d_hidden={D_HIDDEN}, n_layers={N_LAYERS}, "
          f"n_train={N_TRAIN}, n_probe={N_PROBE}, n_seeds={N_SEEDS}")
    print(f"  distributed reg weight: {DIST_REG_WEIGHT}")
    results = [run_seed(s) for s in range(N_SEEDS)]

    # Aggregate
    print(f"\n{'=' * 64}")
    print("AGGREGATE rho_eps (mean over seeds)")
    print(f"{'=' * 64}")
    for li in range(N_LAYERS):
        s = np.mean([r['steering_safety'][li]['rho_eps'] for r in results])
        f = np.mean([r['ft_safety'][li]['rho_eps'] for r in results])
        d = np.mean([r['distributed_safety'][li]['rho_eps'] for r in results])
        print(f"  L{li}: steer={s:.3f}  ft={f:.3f}  distributed={d:.3f}")

    agg_steer = np.mean(
        [r['steering_safety'][li]['rho_eps']
         for r in results for li in range(N_LAYERS)])
    agg_ft = np.mean(
        [r['ft_safety'][li]['rho_eps']
         for r in results for li in range(N_LAYERS)])
    agg_dist = np.mean(
        [r['distributed_safety'][li]['rho_eps']
         for r in results for li in range(N_LAYERS)])
    agg_dist_compliance = np.mean(
        [r['dist_safety_compliance'] for r in results])
    print(f"\nAggregate rho_eps  steer={agg_steer:.3f}  "
          f"ft={agg_ft:.3f}  distributed={agg_dist:.3f}")
    print(f"Distributed safety compliance (D_s) = "
          f"{agg_dist_compliance:.3f}")

    out = {'config': {'d_hidden': D_HIDDEN, 'n_layers': N_LAYERS,
                      'n_train': N_TRAIN, 'n_probe': N_PROBE,
                      'n_seeds': N_SEEDS, 'eps': EPS,
                      'dist_reg_weight': DIST_REG_WEIGHT},
           'aggregate': {'steering': agg_steer, 'ft': agg_ft,
                          'distributed': agg_dist,
                          'dist_safety_compliance': agg_dist_compliance},
           'per_seed': results}
    path = HERE / "calibration_v2_results.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item')
                  else o)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
