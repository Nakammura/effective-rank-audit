"""
Calibration MLP: lambda-sweep projection-ablation experiment.

  - Test 2 (projection ablation) on each calibration variant.
    Projects out the top-r principal subspace of each variant's
    modification matrix and measures (a) safety compliance on D_s
    and (b) general task accuracy on the control distribution.
    Low rho_eps predicts ablating top-r removes safety at small r;
    high rho_eps predicts large r is required to remove safety.

  - Intermediate rho_eps sweep.
    Sweep the rank-maximization regularizer weight from 0 to
    DIST_REG_WEIGHT to fill in the rho_eps grid between FullFT
    (~0.17) and Distributed (~0.40).
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from calibration_v2 import (
    DeepMLP, generate_data, base_train, ft_safety, ft_distributed,
    eff_rank, per_layer_eval,
    D_IN, D_HIDDEN, N_LAYERS, N_TRAIN, N_PROBE,
    SAFETY_FRAC, EPS, LR, N_EPOCHS_BASE, N_EPOCHS_FT, N_EPOCHS_DIST,
    STABLE_RANK_TARGET, DEVICE,
)

HERE = Path(__file__).resolve().parent


def topk_projector(M, k):
    """Return P = U_k U_k^T projecting onto top-k left singular subspace."""
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    Uk = U[:, :k]
    return Uk @ Uk.T  # (d, d)


def project_out(h, P):
    """Project out the column space of P (P = projector onto subspace).
    Returns h - h P^T  (treating h as row-vectors, projecting out P)."""
    # h: (n, d); P: (d, d) projector onto subspace to REMOVE
    return h - h @ P.T


def compliance_after_ablation(model, X_pr, y_safe_pr, sm, cm,
                              h_align_layers, h_base_layers,
                              k, mask_for_ablation):
    """
    Approximate Test-2 ablation: replace each layer's aligned activation
    on D_s with the projected-out variant (top-k principal subspace of
    M_safe per layer removed), then re-run the readout from the modified
    final-layer activation.

    For an MLP we cannot replay through the network easily after replacing
    intermediate activations, so we instead apply the ablation at the
    final hidden layer and re-run only the readout.  This matches the
    Test-2 recipe used in Section 6.4 (final-layer projection).
    """
    li = N_LAYERS - 1  # final hidden layer
    h_a = h_align_layers[li].detach().cpu().numpy()
    h_b = h_base_layers[li].detach().cpu().numpy()
    sm_np = sm.detach().cpu().numpy()
    M = (h_a[sm_np] - h_b[sm_np]).T  # (d, n_safety)
    if k <= 0 or M.shape[1] < 2:
        P = np.zeros((D_HIDDEN, D_HIDDEN), dtype=np.float32)
    else:
        P = topk_projector(M, k).astype(np.float32)

    # Apply the projection to the aligned activations
    if mask_for_ablation == 'safety':
        mask_np = sm_np
    elif mask_for_ablation == 'control':
        mask_np = (~sm).detach().cpu().numpy()
    else:
        mask_np = np.ones_like(sm_np, dtype=bool)

    h_a_modified = h_a.copy()
    h_a_modified[mask_np] = project_out(h_a[mask_np], P)

    # Run readout
    readout = model.readout
    h_t = torch.from_numpy(h_a_modified).to(DEVICE)
    with torch.no_grad():
        logits = readout(h_t).squeeze(-1)
    preds = (logits > 0).float()
    safety_compliance = (preds[sm] == y_safe_pr[sm]).float().mean().item()
    return {'k': k, 'safety_compliance': safety_compliance}


def run_test2_ablation_for_variant(model, name, X_pr, y_safe_pr, sm, cm,
                                    h_base, h_align, k_list):
    """Run Test 2 projection ablation for k in k_list on a single variant."""
    rows = []
    base_logits, _ = model(X_pr.to(DEVICE))
    base_preds = (base_logits > 0).float()
    base_compliance = (base_preds[sm] == y_safe_pr[sm]).float().mean().item()
    print(f"  {name} baseline safety compliance: {base_compliance:.3f}")
    rows.append({'k': 0, 'safety_compliance': base_compliance})
    for k in k_list:
        r = compliance_after_ablation(
            model, X_pr, y_safe_pr, sm, cm,
            h_align, h_base, k, mask_for_ablation='safety')
        rows.append(r)
        print(f"    k={k:3d}  post-ablation compliance = "
              f"{r['safety_compliance']:.3f}")
    return rows


def ft_distributed_with_reg(base, X_train, y_safe_train, is_safety_train,
                            reg_weight, epochs=N_EPOCHS_DIST):
    """Fine-tune with arbitrary distributed regularizer weight."""
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
    from calibration_v2 import column_orthogonality_loss, stable_rank_loss
    for ep in range(epochs):
        opt.zero_grad()
        logits, h_align_train = m(X_train)
        safety_loss = L(logits, y_safe_train)
        reg_loss = 0.0
        for h_a, h_b in zip(h_align_train, h_base_safety):
            M = (h_a[is_safety_train] - h_b).T
            reg_loss = reg_loss + column_orthogonality_loss(M)
            reg_loss = reg_loss + stable_rank_loss(M, STABLE_RANK_TARGET)
        total = safety_loss + reg_weight * reg_loss
        total.backward()
        opt.step()
    return m


def run_revision3_calibration(seed=0, reg_grid=(0.0, 0.5, 5.0, 15.0, 50.0),
                              k_list=(1, 3, 5, 10, 20, 40)):
    print(f"\n=== Calibration v2 lambda-sweep (seed={seed}) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, y_task_tr, y_safe_tr, is_s_tr = generate_data(N_TRAIN, seed)
    X_pr, y_task_pr, y_safe_pr, is_s_pr = generate_data(
        N_PROBE, seed + 7777)
    X_pr = X_pr.to(DEVICE)
    y_safe_pr = y_safe_pr.to(DEVICE)
    sm = is_s_pr.to(DEVICE)
    cm = (~is_s_pr).to(DEVICE)

    base = base_train(X_tr, y_task_tr)
    base.eval()
    with torch.no_grad():
        _, h_base = base(X_pr)

    # Standard FT (compatibility)
    ft = ft_safety(base, X_tr, y_safe_tr); ft.eval()
    with torch.no_grad():
        _, h_ft = ft(X_pr)

    out = {'seed': seed, 'reg_grid': list(reg_grid),
           'k_list': list(k_list), 'variants': {}}

    # Standard FT row
    eval_ft_s = per_layer_eval(h_base, h_ft, sm)
    rho_ft = float(np.mean([e['rho_eps'] for e in eval_ft_s]))
    print(f"  FullFT  aggregate rho_eps = {rho_ft:.3f}")
    rows = run_test2_ablation_for_variant(
        ft, 'FullFT', X_pr, y_safe_pr, sm, cm,
        h_base, h_ft, list(k_list))
    out['variants']['FullFT'] = {
        'reg_weight': None, 'rho_eps_aggregate': rho_ft,
        'per_layer': eval_ft_s,
        'test2_ablation': rows,
    }

    # Distributed variants at varying reg weights
    for reg_w in reg_grid:
        if reg_w <= 0:
            continue  # FullFT covered above
        print(f"\n  -- Distributed variant, reg={reg_w} --")
        dist = ft_distributed_with_reg(
            base, X_tr, y_safe_tr, is_s_tr, reg_w,
            epochs=N_EPOCHS_DIST)
        dist.eval()
        with torch.no_grad():
            _, h_dist = dist(X_pr)
        eval_dist_s = per_layer_eval(h_base, h_dist, sm)
        rho_dist = float(np.mean([e['rho_eps'] for e in eval_dist_s]))
        print(f"    aggregate rho_eps = {rho_dist:.3f}")
        rows = run_test2_ablation_for_variant(
            dist, f'Dist_reg{reg_w}', X_pr, y_safe_pr, sm, cm,
            h_base, h_dist, list(k_list))
        out['variants'][f'Dist_reg{reg_w}'] = {
            'reg_weight': float(reg_w),
            'rho_eps_aggregate': rho_dist,
            'per_layer': eval_dist_s,
            'test2_ablation': rows,
        }
    return out


def main():
    print("=" * 64)
    print("Calibration v2 lambda-sweep: Test-2 ablation + intermediate rho sweep")
    print("=" * 64)
    seeds = [0, 1, 2]
    reg_grid = (0.0, 0.5, 5.0, 15.0, 50.0)
    k_list = (1, 3, 5, 10, 20, 40, 80)
    results = []
    for s in seeds:
        results.append(run_revision3_calibration(
            seed=s, reg_grid=reg_grid, k_list=k_list))

    out = {'seeds': seeds, 'reg_grid': list(reg_grid),
           'k_list': list(k_list), 'per_seed': results}
    path = HERE / "results" / "calibration_v2_revision3.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
