"""
Phase 1 of (B): construct distributed-unsafe variants in the calibration MLP
setting and measure kappa_emp = sigma_r(S)^2 / (eps * (||M_h||^2 + ||M_u||^2))
where S = M_honest - M_unsafe.

Hypothesis: existing 3 LLM uncensored references all fail kappa>8 because
they are refusal-direction-removed (single-direction unsafe shift). A
distributed-unsafe reference (trained with rank-max regularizer + unsafe
objective) should produce a multi-dim unsafe shift and thus a non-degenerate
S spectrum.

Variants trained:
  HONEST_FT     v1 honest fine-tune (y_safe objective)         rho ~ 0.17
  HONEST_DIST5  honest with rank-max reg lambda=5              rho ~ 0.40
  UNSAFE_FT     unsafe fine-tune (y_task / no safety masking)  rho ~ ? (presumably low)
  UNSAFE_DIST5  unsafe with rank-max reg lambda=5              rho ~ ? (presumably high)

Pairs measured:
  (HONEST_FT, UNSAFE_FT)         — both single-direction
  (HONEST_FT, UNSAFE_DIST5)      — distributed unsafe vs single honest
  (HONEST_DIST5, UNSAFE_FT)      — distributed honest vs single unsafe
  (HONEST_DIST5, UNSAFE_DIST5)   — both distributed

For each pair, compute kappa per layer.

Output: results/calibration_kappa.json
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from calibration_v2 import (
    DeepMLP, generate_data, base_train, ft_safety,
    column_orthogonality_loss, stable_rank_loss,
    D_IN, D_HIDDEN, N_LAYERS, N_TRAIN, N_PROBE, SAFETY_FRAC,
    EPS, LR, N_EPOCHS_BASE, N_EPOCHS_FT, N_EPOCHS_DIST,
    STABLE_RANK_TARGET, DEVICE,
)

HERE = Path(__file__).resolve().parent


def ft_unsafe(base, X, y_task, epochs=N_EPOCHS_FT):
    """Naive unsafe FT: predict y_task on ALL inputs (ignoring safety mask).
    This produces a model that responds harmfully to safety-relevant prompts
    but otherwise mirrors the task signal."""
    m = copy.deepcopy(base).to(DEVICE)
    X = X.to(DEVICE); y_task = y_task.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    L = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = m(X)
        loss = L(logits, y_task)
        loss.backward()
        opt.step()
    return m


def ft_unsafe_distributed(base, X_train, y_task_train, is_safety_train,
                           reg_weight=5.0, epochs=N_EPOCHS_DIST):
    """Distributed-unsafe FT: y_task objective + rank-max regularizer on
    safety-relevant inputs. The regularizer encourages the unsafe modification
    to span many directions across D_s, not collapse to a single direction."""
    base.eval()
    X_train = X_train.to(DEVICE)
    y_task_train = y_task_train.to(DEVICE)
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
        task_loss = L(logits, y_task_train)
        reg_loss = 0.0
        for h_a, h_b in zip(h_align_train, h_base_safety):
            M = (h_a[is_safety_train] - h_b).T
            reg_loss = reg_loss + column_orthogonality_loss(M)
            reg_loss = reg_loss + stable_rank_loss(M, STABLE_RANK_TARGET)
        total = task_loss + reg_weight * reg_loss
        total.backward()
        opt.step()
    return m


def kappa_emp(M_honest, M_unsafe, eps=EPS):
    """Compute kappa proxy = sigma_r(S)^2 / (eps * (||M_h||^2 + ||M_u||^2))
    where r = rank_eps(S)."""
    S = M_honest - M_unsafe
    if np.linalg.norm(S) == 0:
        return None
    sv = np.linalg.svd(S, compute_uv=False)
    s2 = sv ** 2
    cum = np.cumsum(s2)
    total = float(cum[-1])
    r = int(np.searchsorted(cum, (1 - eps) * total) + 1)
    sigma_r = float(sv[r-1])
    sigma_1 = float(sv[0])
    frob_S = total
    frob_h = float((M_honest ** 2).sum())
    frob_u = float((M_unsafe ** 2).sum())
    kappa = sigma_r ** 2 / (eps * (frob_h + frob_u)) if (frob_h + frob_u) > 0 \
            else float('inf')
    return {
        'r_S': r,
        'sigma_1_S': sigma_1,
        'sigma_r_S': sigma_r,
        'frob_S_sq': frob_S,
        'frob_M_honest_sq': frob_h,
        'frob_M_unsafe_sq': frob_u,
        'kappa_emp': float(kappa),
    }


def collect_modification_matrix(model, base_model, X, mask):
    """Return per-layer M = h_model - h_base on inputs masked by `mask`."""
    model.eval(); base_model.eval()
    X = X.to(DEVICE); mask = mask.to(DEVICE)
    with torch.no_grad():
        _, h_m = model(X)
        _, h_b = base_model(X)
    M_per_layer = []
    for li in range(N_LAYERS):
        h_m_l = h_m[li][mask].detach().cpu().numpy()
        h_b_l = h_b[li][mask].detach().cpu().numpy()
        n = min(len(h_m_l), len(h_b_l))
        M = (h_m_l[:n] - h_b_l[:n]).T  # (d, n)
        M_per_layer.append(M)
    return M_per_layer


def run_seed(seed):
    print(f"\n=== seed {seed} ===")
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_task_tr, y_safe_tr, is_s_tr = generate_data(N_TRAIN, seed)
    X_pr, y_task_pr, y_safe_pr, is_s_pr = generate_data(N_PROBE, seed + 7777)

    print("  training base...")
    base = base_train(X_tr, y_task_tr); base.eval()
    print("  training honest_ft (y_safe)...")
    honest_ft = ft_safety(base, X_tr, y_safe_tr); honest_ft.eval()
    print("  training honest_dist5 (y_safe + rank-max reg=5)...")
    from calibration_v2 import ft_distributed
    honest_dist5 = ft_distributed(base, X_tr, y_safe_tr, is_s_tr, reg=5.0); honest_dist5.eval()
    print("  training unsafe_ft (y_task)...")
    unsafe_ft = ft_unsafe(base, X_tr, y_task_tr); unsafe_ft.eval()
    print("  training unsafe_dist5 (y_task + rank-max reg=5)...")
    unsafe_dist5 = ft_unsafe_distributed(base, X_tr, y_task_tr, is_s_tr, reg_weight=5.0)
    unsafe_dist5.eval()

    sm = is_s_pr  # safety mask
    # Compute modification matrices on safety-relevant inputs
    print("  collecting M matrices on D_s...")
    M_honest_ft = collect_modification_matrix(honest_ft, base, X_pr, sm)
    M_honest_dist5 = collect_modification_matrix(honest_dist5, base, X_pr, sm)
    M_unsafe_ft = collect_modification_matrix(unsafe_ft, base, X_pr, sm)
    M_unsafe_dist5 = collect_modification_matrix(unsafe_dist5, base, X_pr, sm)

    pairs = {
        'ft_vs_unsafe_ft': (M_honest_ft, M_unsafe_ft),
        'ft_vs_unsafe_dist5': (M_honest_ft, M_unsafe_dist5),
        'dist5_vs_unsafe_ft': (M_honest_dist5, M_unsafe_ft),
        'dist5_vs_unsafe_dist5': (M_honest_dist5, M_unsafe_dist5),
    }

    out = {'seed': seed, 'pairs': {}}
    for name, (M_h, M_u) in pairs.items():
        per_layer = []
        for li in range(N_LAYERS):
            k = kappa_emp(M_h[li], M_u[li])
            if k is not None:
                k['layer'] = li
                per_layer.append(k)
        out['pairs'][name] = per_layer
        kappas = [k['kappa_emp'] for k in per_layer]
        rs = [k['r_S'] for k in per_layer]
        if kappas:
            print(f"  {name}: r_S median={int(np.median(rs))}  "
                  f"kappa median={np.median(kappas):.4f}  "
                  f"max={max(kappas):.4f}  "
                  f"#layers kappa>8: {sum(1 for k in kappas if k>8)}/{len(kappas)}")
    return out


def main():
    seeds = [0, 1, 2]
    print("=" * 64)
    print("Calibration MLP kappa_emp: distributed-unsafe construction")
    print("=" * 64)
    print(f"  d_hidden={D_HIDDEN}, n_layers={N_LAYERS}, "
          f"n_train={N_TRAIN}, n_probe={N_PROBE}, n_seeds={len(seeds)}")
    results = [run_seed(s) for s in seeds]

    # Aggregate
    print(f"\n{'=' * 64}")
    print("AGGREGATE per pair (mean over seeds)")
    print(f"{'=' * 64}")
    pair_names = list(results[0]['pairs'].keys())
    agg = {}
    for name in pair_names:
        all_kappa = []
        all_r = []
        n_above_8 = 0
        n_total = 0
        per_layer_kappa = {li: [] for li in range(N_LAYERS)}
        for r in results:
            for k in r['pairs'][name]:
                all_kappa.append(k['kappa_emp'])
                all_r.append(k['r_S'])
                per_layer_kappa[k['layer']].append(k['kappa_emp'])
                n_total += 1
                if k['kappa_emp'] > 8:
                    n_above_8 += 1
        agg[name] = {
            'n_total': n_total,
            'n_above_8': n_above_8,
            'kappa_median': float(np.median(all_kappa)) if all_kappa else None,
            'kappa_max': float(max(all_kappa)) if all_kappa else None,
            'r_S_median': float(np.median(all_r)) if all_r else None,
            'per_layer_kappa_mean': {
                li: float(np.mean(v)) if v else None
                for li, v in per_layer_kappa.items()
            },
        }
        print(f"\n  {name}:")
        print(f"    layers above kappa=8: {n_above_8}/{n_total}")
        print(f"    kappa median={agg[name]['kappa_median']:.4f}  "
              f"max={agg[name]['kappa_max']:.4f}")
        print(f"    r_S median={agg[name]['r_S_median']:.0f}")
        print(f"    per-layer kappa mean:")
        for li in range(N_LAYERS):
            v = agg[name]['per_layer_kappa_mean'][li]
            if v is not None:
                print(f"      L{li}: kappa={v:.4f}")

    out = {'seeds': seeds, 'config': {'d_hidden': D_HIDDEN,
                                        'n_layers': N_LAYERS,
                                        'n_train': N_TRAIN,
                                        'n_probe': N_PROBE,
                                        'eps': EPS},
           'aggregate': agg, 'per_seed': results}
    out_path = HERE / "results" / "calibration_kappa.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
