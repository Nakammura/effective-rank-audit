"""
Supplementary experiments:

  1. M_DiD Test 2 ablation on Llama narrow band.
     The headline Test 2 in diagnostics_v4 uses M_template's
     principal subspace.  M_DiD recovers the Arditi direction at
     high cosine, so we re-run the narrow-band ablation using
     M_DiD's principal subspace to verify that the Arditi-direction
     subspace is also causally privileged.

  2. Qwen-2.5-7B-Instruct rho_eps measurement (rank + Arditi cosine).
     Cheap: no generation/ablation in this entry point.  Test 2
     ablation on Qwen is in run_qwen_test2; bootstrap CI on Qwen
     M_template is in run_qwen_bootstrap.
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
    collect_acts, _mod_matrix, _Ablate, _refusal_rate,
    band, eff_rank, arditi_direction, cos_with_top1,
)


def run_mdid_ablation(family='llama', device='cuda',
                       n_safety=200, n_control=200, n_gen=100):
    """Test 2 ablation using M_DiD principal subspace (instead of M_template)."""
    cfg = MODELS[family]
    sp = safety_prompts(n_safety)
    cp = control_prompts(n_control)
    out = {'family': family, 'n_safety': n_safety,
           'n_control': n_control, 'n_gen': n_gen}

    from transformers import AutoTokenizer
    aligned_tok = AutoTokenizer.from_pretrained(cfg['aligned'])
    aligned_chat = aligned_tok.chat_template

    print("--- Base chat-template safety acts ---")
    m_b, t_b = load_model(cfg['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat)
    a_bc_c = collect_acts(m_b, t_b, cp, device, chat=True,
                          chat_template=aligned_chat)
    unload(m_b, t_b); gc.collect(); torch.cuda.empty_cache()

    print("--- Aligned acts ---")
    m_a, t_a = load_model(cfg['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)
    a_ac_c = collect_acts(m_a, t_a, cp, device, chat=True)

    layers = sorted(a_ac_s.keys())
    n_layers = len(layers)
    out['n_layers'] = n_layers

    blayers = band(n_layers, 0.45, 0.70, k=5)
    out['narrow_band'] = blayers
    print(f"--- Narrow band layers: {blayers} ---")

    # Pre-compute M_DiD per layer of narrow band:
    # M_DiD = M_safety_template - mean(M_control_template)
    # then take SVD to get principal subspace.
    Us_did = {}
    for l in blayers:
        M_s = _mod_matrix(a_ac_s, a_bc_s, l)  # (d, n_s)
        M_c = _mod_matrix(a_ac_c, a_bc_c, l)  # (d, n_c)
        mu_c = M_c.mean(axis=1, keepdims=True)
        M_did = M_s - mu_c
        U, _, _ = np.linalg.svd(M_did, full_matrices=False)
        Us_did[l] = U  # full U; we'll slice top-k per ablation

    gp = sp[:n_gen]
    base = _refusal_rate(m_a, t_a, gp, device)
    print(f"  baseline refusal = {base['rate']:.3f}")
    out['baseline_refusal'] = base

    out['mdid_ablation'] = {}
    for k in [1, 3, 5, 10, 20, 50]:
        hooks = []
        for l in blayers:
            U_k = Us_did[l][:, :k]
            h = _Ablate(U_k, l)
            h.register(m_a)
            hooks.append(h)
        rate = _refusal_rate(m_a, t_a, gp, device)
        for h in hooks:
            h.remove()
        out['mdid_ablation'][k] = rate
        print(f"  M_DiD rank-{k:3d} ablation: refusal = {rate['rate']:.3f} "
              f"(CI=[{rate['wilson95_lo']:.2f},{rate['wilson95_hi']:.2f}])")

    unload(m_a, t_a); gc.collect(); torch.cuda.empty_cache()
    return out


def run_qwen_rank_only(family='qwen', device='cuda', n_safety=200,
                       n_control=200):
    """Test 1 (effective rank) + Test 3 (Arditi cosine) on Qwen-2.5-7B.
    No generation/ablation, just rank measurement + cosine. Cheap."""
    cfg_qwen = {
        'base': 'Qwen/Qwen2.5-7B',
        'aligned': 'Qwen/Qwen2.5-7B-Instruct',
    }
    sp = safety_prompts(n_safety)
    cp = control_prompts(n_control)
    out = {'family': family, 'config': cfg_qwen,
           'n_safety': n_safety, 'n_control': n_control}

    from transformers import AutoTokenizer
    aligned_tok = AutoTokenizer.from_pretrained(cfg_qwen['aligned'])
    aligned_chat = aligned_tok.chat_template

    print("--- Qwen base chat-template acts ---")
    m_b, t_b = load_model(cfg_qwen['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat)
    a_bc_c = collect_acts(m_b, t_b, cp, device, chat=True,
                          chat_template=aligned_chat)
    unload(m_b, t_b); gc.collect(); torch.cuda.empty_cache()

    print("--- Qwen aligned acts ---")
    m_a, t_a = load_model(cfg_qwen['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)
    a_ac_c = collect_acts(m_a, t_a, cp, device, chat=True)
    unload(m_a, t_a); gc.collect(); torch.cuda.empty_cache()

    layers = sorted(a_ac_s.keys())
    out['n_layers'] = len(layers)

    # Test 1: rank on M_template (safety, control), centered + uncentered
    print("\n=== Test 1: M_template effective rank ===")
    out['M_template_safety'] = {}
    out['M_template_control'] = {}
    for l in layers:
        M_s = _mod_matrix(a_ac_s, a_bc_s, l)
        M_c = _mod_matrix(a_ac_c, a_bc_c, l)
        for name, M in [('M_template_safety', M_s),
                        ('M_template_control', M_c)]:
            r, ratio_d, ratio_min_dn, top_sv = eff_rank(M)
            r_cent, ratio_d_cent, ratio_min_dn_cent, _ = eff_rank(
                M, centered=True)
            out[name][int(l)] = {
                'eff_rank': int(r),
                'ratio_d': float(ratio_d),
                'ratio_min_dn': float(ratio_min_dn),
                'eff_rank_centered': int(r_cent),
                'ratio_d_centered': float(ratio_d_cent),
                'top_sv': [float(x) for x in top_sv[:30]],
            }
        if int(l) % 4 == 0:
            print(f"  L{l}: M_template_s rho={out['M_template_safety'][int(l)]['ratio_d']:.4f}  "
                  f"r={out['M_template_safety'][int(l)]['eff_rank']}  "
                  f"r_cent={out['M_template_safety'][int(l)]['eff_rank_centered']}")

    # Test 3: Arditi cosine vs top-1 of M_template, M_DiD
    print("\n=== Test 3: Arditi cosine ===")
    out['Arditi'] = {}
    for l in layers:
        v_arditi, n_arditi = arditi_direction(a_ac_s, a_ac_c, l)
        M_t = _mod_matrix(a_ac_s, a_bc_s, l)
        cos_template = cos_with_top1(M_t, v_arditi)
        # M_DiD
        M_c = _mod_matrix(a_ac_c, a_bc_c, l)
        mu_c = M_c.mean(axis=1, keepdims=True)
        M_did = M_t - mu_c
        cos_did = cos_with_top1(M_did, v_arditi)
        out['Arditi'][int(l)] = {
            'arditi_norm': float(n_arditi),
            'cos_with_template_top1': float(cos_template),
            'cos_with_DiD_top1': float(cos_did),
        }
    cos_t = [v['cos_with_template_top1'] for v in out['Arditi'].values()]
    cos_d = [v['cos_with_DiD_top1'] for v in out['Arditi'].values()]
    print(f"  cos(Arditi, top1 M_template): mean={np.mean(cos_t):.3f}")
    print(f"  cos(Arditi, top1 M_DiD):      mean={np.mean(cos_d):.3f}")

    return out


def run_qwen_bootstrap(family='qwen', device='cuda', n_safety=200,
                        n_control=200, n_boot=200, layer_offsets=None):
    """Bootstrap CI on rank_eps for Qwen M_template at representative
    middle layers. Mirrors what Llama/Gemma have in v4_*.json so that
    the headline 3-LLM table can report bootstrap CIs uniformly.
    n_safety/n_control: SVD sample size (200 to match existing runs).
    n_boot: number of bootstrap resamples (200 to match Llama/Gemma).
    layer_offsets: list of fractional positions in [0, 1] for layer
        selection (default [0.45, 0.55, 0.65, 0.75, 0.85] = 5 layers
        in the middle-to-late band).
    """
    from diagnostics_v4 import bootstrap_eff_rank
    cfg_qwen = {
        'base': 'Qwen/Qwen2.5-7B',
        'aligned': 'Qwen/Qwen2.5-7B-Instruct',
    }
    if layer_offsets is None:
        layer_offsets = [0.45, 0.55, 0.65, 0.75, 0.85]
    sp = safety_prompts(n_safety)
    cp = control_prompts(n_control)  # unused but kept for API symmetry
    out = {'family': family, 'config': cfg_qwen,
           'n_safety': n_safety, 'n_boot': n_boot}

    from transformers import AutoTokenizer
    aligned_tok = AutoTokenizer.from_pretrained(cfg_qwen['aligned'])
    aligned_chat = aligned_tok.chat_template

    print("--- Qwen base chat-template safety acts ---")
    m_b, t_b = load_model(cfg_qwen['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat)
    unload(m_b, t_b); gc.collect(); torch.cuda.empty_cache()

    print("--- Qwen aligned chat-template safety acts ---")
    m_a, t_a = load_model(cfg_qwen['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)
    unload(m_a, t_a); gc.collect(); torch.cuda.empty_cache()

    layers = sorted(a_ac_s.keys())
    L = len(layers)
    out['n_layers'] = L
    sel = sorted(set(min(L - 1, max(0, int(round(o * L)))) for o in layer_offsets))
    out['selected_layers'] = sel

    print(f"\n=== Bootstrap CI on M_template (Qwen), n_boot={n_boot} ===")
    out['M_template_bootstrap'] = {}
    for l in sel:
        ci = bootstrap_eff_rank(a_ac_s, a_bc_s, l, n_boot=n_boot, eps=0.05)
        out['M_template_bootstrap'][int(l)] = ci
        print(f"  L{l}: mean={ci['mean']:.2f}  std={ci['std']:.2f}  "
              f"95% CI [{ci['ci95_lo']:.1f}, {ci['ci95_hi']:.1f}]")

    return out


def run_qwen_test2(family='qwen', device='cuda',
                    n_safety=200, n_control=200, n_gen=100):
    """Test 2 (causal ablation) for Qwen-2.5-7B-Instruct on
    M_template's principal subspace, narrow band [0.45L, 0.70L],
    k in {1, 3, 5, 10, 20, 50}.  Mirrors the Llama/Gemma Test 2 in
    diagnostics_v4.  n_safety=200, n_control=200 for SVD; n_gen=100
    held-out prompts for refusal classification.
    """
    cfg_qwen = {
        'base': 'Qwen/Qwen2.5-7B',
        'aligned': 'Qwen/Qwen2.5-7B-Instruct',
    }
    sp = safety_prompts(n_safety)
    cp = control_prompts(n_control)
    out = {'family': family, 'config': cfg_qwen,
           'n_safety': n_safety, 'n_control': n_control, 'n_gen': n_gen}

    from transformers import AutoTokenizer
    aligned_tok = AutoTokenizer.from_pretrained(cfg_qwen['aligned'])
    aligned_chat = aligned_tok.chat_template

    print("--- Qwen base chat-template safety acts ---")
    m_b, t_b = load_model(cfg_qwen['base'], device)
    a_bc_s = collect_acts(m_b, t_b, sp, device, chat=True,
                          chat_template=aligned_chat)
    unload(m_b, t_b); gc.collect(); torch.cuda.empty_cache()

    print("--- Qwen aligned acts ---")
    m_a, t_a = load_model(cfg_qwen['aligned'], device)
    a_ac_s = collect_acts(m_a, t_a, sp, device, chat=True)

    layers = sorted(a_ac_s.keys())
    n_layers = len(layers)
    out['n_layers'] = n_layers

    blayers = band(n_layers, 0.45, 0.70, k=5)
    out['narrow_band'] = blayers
    print(f"--- Narrow band layers: {blayers} ---")

    # Per-layer M_template SVD (top-k principal subspace)
    Us_template = {}
    for l in blayers:
        M_t = _mod_matrix(a_ac_s, a_bc_s, l)  # (d, n_s)
        U, _, _ = np.linalg.svd(M_t, full_matrices=False)
        Us_template[l] = U

    gp = sp[:n_gen]
    base = _refusal_rate(m_a, t_a, gp, device)
    print(f"  baseline refusal = {base['rate']:.3f}")
    out['baseline_refusal'] = base

    out['template_ablation'] = {}
    for k in [1, 3, 5, 10, 20, 50]:
        hooks = []
        for l in blayers:
            U_k = Us_template[l][:, :k]
            h = _Ablate(U_k, l)
            h.register(m_a)
            hooks.append(h)
        rate = _refusal_rate(m_a, t_a, gp, device)
        for h in hooks:
            h.remove()
        out['template_ablation'][k] = rate
        print(f"  M_template rank-{k:3d} ablation: refusal = {rate['rate']:.3f} "
              f"(CI=[{rate['wilson95_lo']:.2f},{rate['wilson95_hi']:.2f}])")

    # Random rank-k baseline control: instead of top-k of M_template's
    # principal subspace, use k random unit vectors per narrow-band layer.
    # Mirrors the Llama/Gemma random-rank-3 control in tab:diagnosis_abl.
    print("  --- Random rank-k baseline (matched k) ---")
    out['template_ablation_random'] = {}
    rng = np.random.default_rng(0)
    d_hidden = next(iter(Us_template.values())).shape[0]
    for k in [3]:  # narrow-band k=3 to match body table
        hooks = []
        for l in blayers:
            G = rng.standard_normal((d_hidden, k))
            U_k_random, _ = np.linalg.qr(G)
            h = _Ablate(U_k_random, l)
            h.register(m_a)
            hooks.append(h)
        rate = _refusal_rate(m_a, t_a, gp, device)
        for h in hooks:
            h.remove()
        out['template_ablation_random'][k] = rate
        print(f"  Random rank-{k} ablation: refusal = {rate['rate']:.3f} "
              f"(CI=[{rate['wilson95_lo']:.2f},{rate['wilson95_hi']:.2f}])")

    unload(m_a, t_a); gc.collect(); torch.cuda.empty_cache()
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


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else 'mdid'
    if task == 'mdid':
        out = run_mdid_ablation('llama')
        path = HERE / "results" / "revision6_mdid_llama.json"
    elif task == 'qwen':
        out = run_qwen_rank_only('qwen')
        path = HERE / "results" / "revision6_qwen.json"
    elif task == 'qwen_test2':
        out = run_qwen_test2('qwen')
        path = HERE / "results" / "revision6_qwen_test2.json"
    else:
        raise SystemExit(f"unknown task: {task}")
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\nSaved: {path}")
