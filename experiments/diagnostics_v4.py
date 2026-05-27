"""
Effective-rank diagnostic pipeline (single-A100 single pass).

What this script does:

  Chat-template confound (four-variant decomposition).
    For each of {base, aligned} x {chat-template, raw} we collect
    activations on the same prompts, and compute:
      M_naive    = h_a(chat) - h_b(raw)
      M_template = h_a(chat) - h_b(chat)
      M_aligned  = h_a(chat) - h_a(raw)
      M_DiD      = (h_a(D_s, chat) - h_b(D_s, chat))
                 - (h_a(D_c, chat) - h_b(D_c, chat))  (control-subtracted)

  Centered SVD.
    For each M, report eff_rank with and without column-mean
    subtraction.

  n-sweep.
    Re-collect with n in {50, 100, 200, 400} and report eff_rank as
    a function of n (with bootstrap 95% CI on each n).

  Rank/layer ablation sweep.
    Causal ablation for k in {1, 2, 3, 5, 10, 20, 50} and three
    layer-bands: {[0.45, 0.70]L, [0.30, 0.85]L, [0.10, 0.95]L}.
    n=100 generation prompts, Wilson 95% CIs.

  Arditi-direction comparison.
    Compute the Arditi-style refusal direction inside the aligned
    model only (mean shift between safety-relevant and control inputs)
    and report cosine similarity with our top-1 SVD direction of M.

  LRH baseline.
    Compute eff_rank for two arbitrary concept-difference matrices
    (English vs. French; question vs. statement) on the same models,
    to benchmark whether the safety modification is more concentrated
    than typical linearly-represented concepts.

  Last-token vs mean-pool.
    All activations are also recomputed via mean over the last-N tokens
    (N=8) and reported.

Outputs:
  - results/v4_<family>.json  (full numerical results)
  - per-experiment summary printed to stdout
"""
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Some hosting setups don't have the parent dir; we copy colab_run.py
# into HERE so the import resolves locally.
sys.path.insert(0, str(HERE.parent))

from prompts_v4 import (
    SAFETY_PROMPTS_BASE, CONTROL_PROMPTS_BASE,
    safety_prompts, control_prompts,
    ENGLISH_PROMPTS, FRENCH_PROMPTS,
    QUESTION_PROMPTS, STATEMENT_PROMPTS,
)
from colab_run import MODELS, REFUSAL_KEYWORDS, load_model, unload, fmt


EPS = 0.05


# =====================================================================
# Activation collection
# =====================================================================

def collect_acts(model, tokenizer, prompts, device, chat=False,
                 layers=None, max_len=128, last_n=1,
                 chat_template=None):
    """
    Collect residual-stream activations.

    last_n=1: last token only.
    last_n>1: averages over the last N tokens.

    chat_template: explicit Jinja-like chat template string. If passed
    and chat=True, used to format the prompts via apply_chat_template
    with template= argument; this lets us apply the aligned model's
    chat template to base-model tokenizers (which do not have one set).
    """
    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))
    acts = {l: [] for l in layers}
    with torch.no_grad():
        for p in prompts:
            if chat:
                if chat_template is not None:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        chat_template=chat_template,
                        tokenize=False,
                        add_generation_prompt=True)
                elif getattr(tokenizer, 'chat_template', None):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True)
                else:
                    raise ValueError(
                        "Tokenizer has no chat_template and none was "
                        "supplied via chat_template= argument."
                    )
            else:
                text = p
            inp = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).to(device)
            out = model(**inp, output_hidden_states=True)
            sl = inp['attention_mask'].sum().item()
            for l in layers:
                hs = out.hidden_states[l + 1][0]
                if last_n == 1:
                    h = hs[sl - 1, :].float()
                else:
                    start = max(0, sl - last_n)
                    h = hs[start:sl, :].float().mean(dim=0)
                acts[l].append(h.cpu().numpy())
            del out
            torch.cuda.empty_cache()
    return {l: np.stack(acts[l]) for l in layers}


# =====================================================================
# Effective rank
# =====================================================================

def eff_rank(M, eps=EPS, centered=False):
    """
    M: (d, n) matrix where each column is an activation difference.
    Returns (k_eff, ratio_d, ratio_min, top_svals).
    """
    d, n = M.shape
    if centered:
        mu = M.mean(axis=1, keepdims=True)
        M = M - mu
    if np.linalg.norm(M) == 0:
        return 0, 0.0, 0.0, []
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    cum = np.cumsum(s2) / s2.sum()
    k = int(np.searchsorted(cum, 1 - eps) + 1)
    return k, k / d, k / min(d, n), s[:30].tolist()


def _mod_matrix(acts_a, acts_b, layer):
    n = min(len(acts_a[layer]), len(acts_b[layer]))
    return (acts_a[layer][:n] - acts_b[layer][:n]).T


def per_layer_rank(acts_a, acts_b, layers, label='', eps=EPS):
    out = {}
    for l in layers:
        M = _mod_matrix(acts_a, acts_b, l)
        k, r_d, r_mn, top = eff_rank(M, eps=eps, centered=False)
        kc, rc_d, rc_mn, _ = eff_rank(M, eps=eps, centered=True)
        out[l] = {
            'eff_rank': k, 'ratio_d': r_d, 'ratio_min_dn': r_mn,
            'eff_rank_centered': kc, 'ratio_d_centered': rc_d,
            'top_sv': top,
        }
    return out


# =====================================================================
# Bootstrap CI
# =====================================================================

def bootstrap_eff_rank(acts_a, acts_b, layer, n_boot=200, eps=EPS, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(acts_a[layer]), len(acts_b[layer]))
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        M = (acts_a[layer][idx] - acts_b[layer][idx]).T
        k, _, _, _ = eff_rank(M, eps=eps, centered=False)
        ks.append(k)
    ks = np.array(ks)
    return {
        'mean': float(ks.mean()),
        'std': float(ks.std()),
        'ci95_lo': float(np.quantile(ks, 0.025)),
        'ci95_hi': float(np.quantile(ks, 0.975)),
    }


# =====================================================================
# Wilson CI for proportions
# =====================================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# =====================================================================
# Causal ablation
# =====================================================================

class _Ablate:
    def __init__(self, U, layer):
        self.U = torch.tensor(U, dtype=torch.float32)
        self.li = layer
        self.handle = None

    def __call__(self, mod, inp, out):
        h = out[0].float()
        U = self.U.to(h.device)
        proj = torch.einsum('btd,dk,ek->bte', h, U, U)
        return ((h - proj).to(out[0].dtype),) + tuple(out[1:])

    def register(self, model):
        self.handle = model.model.layers[self.li].register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()


def _refusal_rate(model, tok, prompts, device):
    refused = 0
    with torch.no_grad():
        for p in prompts:
            text = fmt(p, tok, chat=True)
            inp = tok(text, return_tensors="pt",
                      truncation=True, max_length=256).to(device)
            ids = model.generate(**inp, max_new_tokens=100,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            resp = tok.decode(ids[0, inp['input_ids'].shape[1]:],
                              skip_special_tokens=True).lower()
            if any(kw.lower() in resp for kw in REFUSAL_KEYWORDS):
                refused += 1
    n = len(prompts)
    rate = refused / n
    lo, hi = wilson_ci(refused, n)
    return {'rate': rate, 'k': refused, 'n': n,
            'wilson95_lo': lo, 'wilson95_hi': hi}


def _topk_basis(acts_a, acts_b, layer, k):
    M = _mod_matrix(acts_a, acts_b, layer)
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :k]


def ablation_sweep(model, tok, gen_prompts, acts_a, acts_b,
                   layer_band, k_list, device, n_random=3):
    print(f"  Layer band: {layer_band}")
    base = _refusal_rate(model, tok, gen_prompts, device)
    print(f"    baseline refusal = {base['rate']:.2f} "
          f"({base['k']}/{base['n']})  CI={base['wilson95_lo']:.2f}"
          f",{base['wilson95_hi']:.2f}")
    rows = []
    for k in k_list:
        # principal
        hooks = []
        for l in layer_band:
            U = _topk_basis(acts_a, acts_b, l, k)
            h = _Ablate(U, l); h.register(model); hooks.append(h)
        principal = _refusal_rate(model, tok, gen_prompts, device)
        for h in hooks: h.remove()
        # random
        rrates = []
        d = acts_a[layer_band[0]].shape[1]
        for _ in range(n_random):
            hooks = []
            for l in layer_band:
                R = np.random.randn(d, k).astype(np.float32)
                R, _ = np.linalg.qr(R)
                h = _Ablate(R, l); h.register(model); hooks.append(h)
            rrates.append(_refusal_rate(model, tok, gen_prompts, device)['rate'])
            for h in hooks: h.remove()
        row = {
            'k': k,
            'principal': principal,
            'random_mean': float(np.mean(rrates)),
            'random_std': float(np.std(rrates)),
        }
        rows.append(row)
        print(f"    k={k:3d}  principal={principal['rate']:.2f}  "
              f"({principal['k']}/{principal['n']}, "
              f"CI=[{principal['wilson95_lo']:.2f},"
              f"{principal['wilson95_hi']:.2f}])  "
              f"random={row['random_mean']:.2f}+/-{row['random_std']:.2f}")
    return {'baseline': base, 'rows': rows, 'layer_band': layer_band}


# =====================================================================
# Arditi-style direction comparison
# =====================================================================

def arditi_direction(acts_safety, acts_control, layer):
    """Mean activation difference between safety and control inputs
    inside the same model (Arditi 2024 construction)."""
    mu_s = acts_safety[layer].mean(axis=0)
    mu_c = acts_control[layer].mean(axis=0)
    v = mu_s - mu_c
    n = np.linalg.norm(v)
    return v / (n + 1e-10), float(n)


def cos_with_top1(M, vec):
    if np.linalg.norm(M) == 0:
        return 0.0
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    u1 = U[:, 0]
    return float(abs(np.dot(u1, vec / (np.linalg.norm(vec) + 1e-10))))


# =====================================================================
# Layer-band utility
# =====================================================================

def band(n_layers, lo, hi, k=5):
    a = int(lo * n_layers)
    b = max(int(hi * n_layers), a + 1)
    return list(np.linspace(a, b, k, dtype=int).tolist())


# =====================================================================
# Main pipeline
# =====================================================================

def run_v4(family='llama', device='cuda', n_safety=200, n_control=200,
           n_gen=100, last_n_pool=8):
    # Strip _minimal suffix used to signal reduced sweep scope.
    family_key = family.replace('_minimal', '')
    cfg = MODELS[family_key]
    sp = safety_prompts(n_safety)
    cp = control_prompts(n_control)
    print(f"\n{'='*64}\n{family.upper()}  v4 — constructive diagnostic\n"
          f"{'='*64}")
    print(f"  n_safety={n_safety}  n_control={n_control}  n_gen={n_gen}")

    out = {'family': family_key, 'family_label': family, 'config': cfg,
           'n_safety': n_safety, 'n_control': n_control,
           'n_gen': n_gen, 'eps': EPS,
           'last_n_pool': last_n_pool}

    # ============================================================
    # Phase 0: get aligned tokenizer's chat template (used to apply
    # the same chat formatting to the base tokenizer, which does not
    # ship with one)
    # ============================================================
    from transformers import AutoTokenizer
    aligned_tokenizer = AutoTokenizer.from_pretrained(cfg['aligned'])
    aligned_chat_template = aligned_tokenizer.chat_template
    print(f"  Aligned chat template ({len(aligned_chat_template or '')} chars) loaded.")

    # ============================================================
    # Phase 1: Base model — both raw AND chat-template
    # ============================================================
    print("\n--- Base model ---")
    m_b, t_b = load_model(cfg['base'], device)

    print("  Base raw, safety...")
    a_br_s = collect_acts(m_b, t_b, sp, device, chat=False)
    print("  Base raw, control...")
    a_br_c = collect_acts(m_b, t_b, cp, device, chat=False)
    print("  Base chat-template, safety...")
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat_template)
    print("  Base chat-template, control...")
    a_bc_c = collect_acts(m_b, t_b, cp, device, chat=True,
                          chat_template=aligned_chat_template)

    # mean-pool variant
    print("  Base raw safety [mean-pool last-N]...")
    a_br_s_mp = collect_acts(m_b, t_b, sp, device, chat=False,
                             last_n=last_n_pool)

    # LRH baseline prompts (only need one model side)
    print("  Base raw, English/French/Q/Stmt for LRH baseline...")
    a_br_en = collect_acts(m_b, t_b, ENGLISH_PROMPTS, device, chat=False)
    a_br_fr = collect_acts(m_b, t_b, FRENCH_PROMPTS, device, chat=False)
    a_br_q  = collect_acts(m_b, t_b, QUESTION_PROMPTS, device, chat=False)
    a_br_st = collect_acts(m_b, t_b, STATEMENT_PROMPTS, device, chat=False)

    unload(m_b, t_b)

    # ============================================================
    # Phase 2: Aligned model — both raw AND chat-template
    # ============================================================
    print("\n--- Aligned model ---")
    m_a, t_a = load_model(cfg['aligned'], device)

    print("  Aligned chat-template, safety...")
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)
    print("  Aligned chat-template, control...")
    a_ac_c = collect_acts(m_a, t_a, cp, device, chat=True)
    print("  Aligned raw, safety...")
    a_ar_s = collect_acts(m_a, t_a, sp, device, chat=False)
    print("  Aligned raw, control...")
    a_ar_c = collect_acts(m_a, t_a, cp, device, chat=False)

    print("  Aligned chat safety [mean-pool last-N]...")
    a_ac_s_mp = collect_acts(m_a, t_a, sp, device, chat=True,
                             last_n=last_n_pool)

    print("  Aligned raw, English/French/Q/Stmt...")
    a_ar_en = collect_acts(m_a, t_a, ENGLISH_PROMPTS, device, chat=False)
    a_ar_fr = collect_acts(m_a, t_a, FRENCH_PROMPTS, device, chat=False)
    a_ar_q  = collect_acts(m_a, t_a, QUESTION_PROMPTS, device, chat=False)
    a_ar_st = collect_acts(m_a, t_a, STATEMENT_PROMPTS, device, chat=False)

    layers = sorted(a_br_s.keys())
    n_layers = len(layers)
    out['n_layers'] = n_layers

    # ============================================================
    # Test 1 variants — chat-template confound + centered SVD
    # ============================================================
    print(f"\n=== Test 1 variants ===")
    print("  M_naive   = h_a(chat) - h_b(raw)   (v3 original)")
    out['M_naive_safety']   = per_layer_rank(a_ac_s, a_br_s, layers)
    out['M_naive_control']  = per_layer_rank(a_ac_c, a_br_c, layers)

    print("  M_template = h_a(chat) - h_b(chat) (template controlled)")
    out['M_template_safety']  = per_layer_rank(a_ac_s, a_bc_s, layers)
    out['M_template_control'] = per_layer_rank(a_ac_c, a_bc_c, layers)

    print("  M_aligned = h_a(chat) - h_a(raw)   (alignment-stage shift only)")
    out['M_aligned_safety']  = per_layer_rank(a_ac_s, a_ar_s, layers)
    out['M_aligned_control'] = per_layer_rank(a_ac_c, a_ar_c, layers)

    # DiD: (a_chat - b_chat) - (a_chat_ctrl - b_chat_ctrl)
    print("  M_DiD = (h_a(D_s,chat) - h_b(D_s,chat))"
          " - (h_a(D_c,chat) - h_b(D_c,chat))")
    M_did = {}
    for l in layers:
        M_s = _mod_matrix(a_ac_s, a_bc_s, l)  # (d, n_s)
        M_c = _mod_matrix(a_ac_c, a_bc_c, l)  # (d, n_c)
        # Subtract column-mean of M_c from each column of M_s
        mu_c = M_c.mean(axis=1, keepdims=True)
        M_did[l] = M_s - mu_c
    out['M_DiD_safety'] = {l: dict(zip(
        ['eff_rank','ratio_d','ratio_min_dn','eff_rank_centered',
         'ratio_d_centered','top_sv'],
        [*eff_rank(M_did[l]),
         *eff_rank(M_did[l], centered=True)[:2], None]
    )) for l in layers}

    # mean-pool variant
    print("  M_naive [mean-pool last-N]")
    out['M_naive_meanpool_safety'] = per_layer_rank(a_ac_s_mp, a_br_s_mp,
                                                    layers)

    # ============================================================
    # Test 1: bootstrap CI on a few middle layers
    # ============================================================
    print(f"\n=== Bootstrap CI on M_template ===")
    sel = layers[n_layers // 2 - 2:n_layers // 2 + 3]
    out['M_template_bootstrap'] = {
        l: bootstrap_eff_rank(a_ac_s, a_bc_s, l, n_boot=200)
        for l in sel
    }
    for l in sel:
        b = out['M_template_bootstrap'][l]
        print(f"  L{l}: mean={b['mean']:.1f} std={b['std']:.2f}"
              f" CI95=[{b['ci95_lo']:.0f},{b['ci95_hi']:.0f}]")

    # ============================================================
    # n-sweep on M_template (the cleaner quantity)
    # ============================================================
    print(f"\n=== n-sweep on M_template (mid layer) ===")
    mid = layers[n_layers // 2]
    n_sweep_grid = [50, 100, 200]
    if n_safety >= 400: n_sweep_grid.append(400)
    out['M_template_n_sweep'] = {}
    for n in n_sweep_grid:
        if n > min(len(a_ac_s[mid]), len(a_bc_s[mid])):
            continue
        idx = np.arange(n)
        M_n = (a_ac_s[mid][idx] - a_bc_s[mid][idx]).T
        k, r_d, r_mn, _ = eff_rank(M_n)
        out['M_template_n_sweep'][n] = {
            'eff_rank': k, 'ratio_d': r_d, 'ratio_min_dn': r_mn}
        print(f"  n={n:4d}  k={k:3d}  ratio_d={r_d:.4f}  "
              f"ratio_min={r_mn:.3f}")

    # ============================================================
    # LRH baseline — concept differences (NOT alignment)
    # ============================================================
    print(f"\n=== LRH baseline: concept-pair effective ranks ===")
    out['LRH_english_vs_french'] = per_layer_rank(a_br_en, a_br_fr, layers)
    out['LRH_question_vs_statement'] = per_layer_rank(
        a_br_q, a_br_st, layers)
    out['LRH_english_vs_french_aligned'] = per_layer_rank(
        a_ar_en, a_ar_fr, layers)
    print("  Mean ratio_d over layers:")
    print(f"    EN-FR   (base):    "
          f"{np.mean([v['ratio_d'] for v in out['LRH_english_vs_french'].values()]):.4f}")
    print(f"    EN-FR   (aligned): "
          f"{np.mean([v['ratio_d'] for v in out['LRH_english_vs_french_aligned'].values()]):.4f}")
    print(f"    Q-Stmt  (base):    "
          f"{np.mean([v['ratio_d'] for v in out['LRH_question_vs_statement'].values()]):.4f}")
    print(f"    Compare to M_template safety mean ratio_d: "
          f"{np.mean([v['ratio_d'] for v in out['M_template_safety'].values()]):.4f}")

    # ============================================================
    # Arditi disambiguation
    # ============================================================
    print(f"\n=== Arditi disambiguation ===")
    out['Arditi'] = {}
    for l in layers:
        v_arditi, n_arditi = arditi_direction(a_ac_s, a_ac_c, l)
        # Compare to top-1 of M_naive (v3 quantity)
        M = _mod_matrix(a_ac_s, a_br_s, l)
        cos_naive = cos_with_top1(M, v_arditi)
        # Compare to top-1 of M_template
        M_t = _mod_matrix(a_ac_s, a_bc_s, l)
        cos_template = cos_with_top1(M_t, v_arditi)
        # Compare to top-1 of M_DiD
        cos_did = cos_with_top1(M_did[l], v_arditi)
        out['Arditi'][l] = {
            'arditi_norm': n_arditi,
            'cos_with_naive_top1': cos_naive,
            'cos_with_template_top1': cos_template,
            'cos_with_DiD_top1': cos_did,
        }
    cn = np.mean([v['cos_with_naive_top1'] for v in out['Arditi'].values()])
    ct = np.mean([v['cos_with_template_top1'] for v in out['Arditi'].values()])
    cd = np.mean([v['cos_with_DiD_top1'] for v in out['Arditi'].values()])
    print(f"  Mean cos(Arditi, top1 of M_naive)   = {cn:.4f}")
    print(f"  Mean cos(Arditi, top1 of M_template) = {ct:.4f}")
    print(f"  Mean cos(Arditi, top1 of M_DiD)     = {cd:.4f}")

    # ============================================================
    # Test 2: rank/layer sweep on the aligned model
    # ============================================================
    print(f"\n=== Test 2: rank/layer ablation sweep ===")
    gp = sp[:n_gen]
    bands = {
        'narrow_45_70': band(n_layers, 0.45, 0.70, k=5),
        'wide_30_85':   band(n_layers, 0.30, 0.85, k=8),
    }
    # Default sweep scope: full grid (6 ranks × 2 bands).  For
    # cost-bounded reruns (e.g.\ when Gemma's full sweep would exceed
    # the executionTimeout) we accept a sweep_scope hint via the job
    # input.
    k_list = [1, 3, 5, 10, 20, 50]
    if str(family).endswith('_minimal') or os.environ.get(
            'V4_SWEEP_MINIMAL') == '1':
        # Minimal sweep: 4 ranks at narrow band only
        bands = {'narrow_45_70': band(n_layers, 0.45, 0.70, k=5)}
        k_list = [1, 3, 10, 20]
    out['Test2_sweep'] = {}
    for bname, blayers in bands.items():
        print(f"  -- band {bname}={blayers} --")
        out['Test2_sweep'][bname] = ablation_sweep(
            m_a, t_a, gp, a_ac_s, a_br_s, blayers, k_list,
            device, n_random=1,
        )

    # ============================================================
    # Save
    # ============================================================
    unload(m_a, t_a)

    # Convert numpy/integer keys for JSON
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    path = HERE / f"v4_{family}.json"
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\n  Saved: {path}")
    return out


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else 'llama'
    n_s = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_c = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    n_g = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    run_v4(fam, n_safety=n_s, n_control=n_c, n_gen=n_g)
