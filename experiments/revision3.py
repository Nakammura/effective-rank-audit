"""
Supplementary diagnostics: per-direction Llama ablation, saved
generations for LLM-judge cross-check, and kappa measurement
against unsafe-mode references.

  Per-direction ablation on Llama narrow band
    (k=1..5 each direction individually).  Explains the
    non-monotonic refusal in the narrow-band Test 2.

  Saved generations at baseline / rank-3 / rank-5 for
    LLM-judge cross-check of the keyword-based refusal
    classifier.

  Kappa_emp measurement using an abliterated/uncensored
    Llama as the unsafe-mode reference: compute
    S = M_honest - M_unsafe and report sigma_r(S),
    ||S||_F, and the kappa proxy.

Outputs JSON to results/revision3_<family>.json.
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from prompts_v4 import safety_prompts, control_prompts
from colab_run import MODELS, REFUSAL_KEYWORDS, load_model, unload, fmt
from diagnostics_v4 import (
    collect_acts, _mod_matrix, _topk_basis, _Ablate, _refusal_rate,
    band, eff_rank,
)


def _save_generations(model, tok, prompts, device, max_new_tokens=100):
    """Run generation greedy and return list of {prompt, response, keyword_refused}."""
    out = []
    with torch.no_grad():
        for p in prompts:
            text = fmt(p, tok, chat=True)
            inp = tok(text, return_tensors='pt',
                      truncation=True, max_length=256).to(device)
            ids = model.generate(
                **inp, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
            resp = tok.decode(
                ids[0, inp['input_ids'].shape[1]:],
                skip_special_tokens=True)
            refused = any(kw.lower() in resp.lower()
                          for kw in REFUSAL_KEYWORDS)
            out.append({'prompt': p, 'response': resp,
                        'keyword_refused': bool(refused)})
    return out


def run_perdir_and_save_gens(family='llama', device='cuda',
                              n_safety=200, n_gen=100, n_save=50):
    """
    D: per-direction ablation at narrow band.
    C.7: saved generations at baseline + rank-3 + rank-5 cumulative.
    """
    cfg = MODELS[family]
    sp = safety_prompts(n_safety)
    out = {'family': family, 'config': cfg,
           'n_safety': n_safety, 'n_gen': n_gen, 'n_save': n_save}

    from transformers import AutoTokenizer
    aligned_tokenizer = AutoTokenizer.from_pretrained(cfg['aligned'])
    aligned_chat_template = aligned_tokenizer.chat_template

    # Phase 1: collect base activations (chat template)
    print("--- Base chat-template safety acts ---")
    m_b, t_b = load_model(cfg['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat_template)
    unload(m_b, t_b)

    # Phase 2: aligned acts + per-direction ablation + saved generations
    print("--- Aligned chat-template safety acts ---")
    m_a, t_a = load_model(cfg['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)

    layers = sorted(a_ac_s.keys())
    n_layers = len(layers)
    out['n_layers'] = n_layers

    blayers = band(n_layers, 0.45, 0.70, k=5)
    out['narrow_band'] = blayers
    print(f"--- Narrow band layers: {blayers} ---")

    gp = sp[:n_gen]
    save_subset = sp[:n_save]

    # Baseline refusal + saved generations
    base_rate = _refusal_rate(m_a, t_a, gp, device)
    print(f"  baseline refusal = {base_rate['rate']:.3f} "
          f"({base_rate['k']}/{base_rate['n']})")
    out['baseline_refusal'] = base_rate
    out['saved_gens'] = {}
    out['saved_gens']['baseline'] = _save_generations(
        m_a, t_a, save_subset, device)

    # Per-direction ablation: ablate u_i alone for i = 1..5
    out['per_direction'] = {}
    for i in range(5):
        hooks = []
        for l in blayers:
            M = _mod_matrix(a_ac_s, a_bc_s, l)
            U, _, _ = np.linalg.svd(M, full_matrices=False)
            U_i = U[:, i:i+1]
            h = _Ablate(U_i, l)
            h.register(m_a)
            hooks.append(h)
        rate = _refusal_rate(m_a, t_a, gp, device)
        for h in hooks:
            h.remove()
        out['per_direction'][i+1] = rate
        print(f"  ablate u_{i+1} only: refusal = {rate['rate']:.3f} "
              f"({rate['k']}/{rate['n']})")

    # Cumulative rank-k ablation + saved generations for k in {3, 5}
    for k in [3, 5]:
        hooks = []
        for l in blayers:
            U = _topk_basis(a_ac_s, a_bc_s, l, k)
            h = _Ablate(U, l)
            h.register(m_a)
            hooks.append(h)
        out['saved_gens'][f'rank_{k}'] = _save_generations(
            m_a, t_a, save_subset, device)
        for h in hooks:
            h.remove()
        print(f"  saved gens for rank-{k} cumulative ablation")

    unload(m_a, t_a)
    return out


def run_kappa_unsafe(family='llama', device='cuda',
                     n_safety=200,
                     abliterated='mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated'):
    """
    B: kappa measurement using an abliterated Llama as unsafe-mode reference.
    Compute S = M_honest - M_unsafe and the spectral-gap proxy.
    """
    cfg = MODELS[family]
    sp = safety_prompts(n_safety)
    out = {'family': family, 'abliterated': abliterated,
           'n_safety': n_safety}

    from transformers import AutoTokenizer
    aligned_tokenizer = AutoTokenizer.from_pretrained(cfg['aligned'])
    aligned_chat_template = aligned_tokenizer.chat_template

    print("--- Aligned (honest) acts ---")
    m_a, t_a = load_model(cfg['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)
    unload(m_a, t_a)

    print("--- Base acts ---")
    m_b, t_b = load_model(cfg['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat_template)
    unload(m_b, t_b)

    print(f"--- Abliterated (unsafe) acts: {abliterated} ---")
    m_u, t_u = load_model(abliterated, device)
    # If the abliterated tokenizer lacks chat template, fall back to aligned
    if t_u.chat_template is None:
        t_u.chat_template = aligned_chat_template
    a_uc_s = collect_acts(m_u, t_u, sp, device, chat=True)
    unload(m_u, t_u)

    layers = sorted(a_ac_s.keys())
    out['n_layers'] = len(layers)

    eps = 0.05
    out['eps'] = eps
    out['per_layer'] = {}
    for l in layers:
        M_h = (a_ac_s[l] - a_bc_s[l]).T  # (d, n)
        M_u = (a_uc_s[l] - a_bc_s[l]).T
        S = M_h - M_u
        sv = np.linalg.svd(S, compute_uv=False)
        s2 = sv ** 2
        cum = np.cumsum(s2)
        total = float(cum[-1]) if len(cum) > 0 else 0.0
        if total <= 0:
            continue
        r = int(np.searchsorted(cum, (1 - eps) * total) + 1)
        sigma_r = float(sv[r-1])
        sigma_1 = float(sv[0])
        frob_S_sq = total
        frob_h = float((M_h ** 2).sum())
        frob_u = float((M_u ** 2).sum())
        # Kappa proxy: sigma_r(S)^2 / (eps * (||M_h||_F^2 + ||M_u||_F^2))
        # using the bound ||E_m||_F^2 <= eps ||M^(m)||_F^2
        kappa_proxy = sigma_r ** 2 / (eps * (frob_h + frob_u)) \
                      if (frob_h + frob_u) > 0 else float('inf')
        out['per_layer'][int(l)] = {
            'r_S_eff_rank': r,
            'sigma_1_S': sigma_1,
            'sigma_r_S': sigma_r,
            'frob_S_sq': frob_S_sq,
            'frob_M_honest_sq': frob_h,
            'frob_M_unsafe_sq': frob_u,
            'kappa_proxy': float(kappa_proxy),
        }
        print(f"  L{l}: r={r}  sigma_1={sigma_1:.2f}  "
              f"sigma_r={sigma_r:.2f}  kappa_proxy={kappa_proxy:.3f}")

    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_revision3(family='llama', device='cuda',
                  n_safety=200, n_gen=100, n_save=50,
                  do_kappa=True,
                  abliterated='mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated'):
    print(f"\n{'='*64}\nSupplementary diagnostics — {family.upper()}\n{'='*64}")
    out = {'family': family}
    path = HERE / "results" / f"revision3_{family}.json"
    path.parent.mkdir(exist_ok=True)

    out['perdir_savedgens'] = run_perdir_and_save_gens(
        family=family, device=device, n_safety=n_safety,
        n_gen=n_gen, n_save=n_save)
    # Save partial results before kappa (kappa loads a 3rd model and
    # can fail; we want the perdir+gens data preserved).
    with open(path, 'w') as f:
        json.dump(_clean(out), f, indent=2)
    print(f"  partial save (perdir+gens): {path}")

    if do_kappa:
        try:
            out['kappa'] = run_kappa_unsafe(
                family=family, device=device, n_safety=n_safety,
                abliterated=abliterated)
        except Exception as e:
            import traceback
            print(f"  kappa phase failed: {e}")
            print(traceback.format_exc())
            out['kappa_error'] = str(e)
    with open(path, 'w') as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else 'llama'
    n_s = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_g = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    n_save = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    run_revision3(fam, n_safety=n_s, n_gen=n_g, n_save=n_save)
