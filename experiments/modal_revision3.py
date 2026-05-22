"""
Modal wrapper for cloud GPU execution of the diagnostic and
calibration pipelines.  Uses bf16 throughout, unloads models
between phases, and caps per-task timeouts to control cost.

To reproduce on Modal:
  pip install modal
  modal token new
  modal run modal_revision3.py --task <task_name>

Tasks: revision3, kappa_only, mdid_ablation, qwen, qwen_test2,
qwen_bootstrap, kappa_multi, calibration_kappa,
calibration_revision3.
"""
from pathlib import Path
import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "scikit-learn==1.5.2",
        "numpy==1.26.4",
        "huggingface_hub==0.26.2",
        "sentencepiece",
    )
    .add_local_file(HERE / "colab_run.py", remote_path="/root/colab_run.py")
    .add_local_file(HERE / "diagnostics_v4.py",
                    remote_path="/root/diagnostics_v4.py")
    .add_local_file(HERE / "prompts_v4.py",
                    remote_path="/root/prompts_v4.py")
    .add_local_file(HERE / "revision3.py",
                    remote_path="/root/revision3.py")
    .add_local_file(HERE / "calibration_v2.py",
                    remote_path="/root/calibration_v2.py")
    .add_local_file(HERE / "calibration_v2_revision3.py",
                    remote_path="/root/calibration_v2_revision3.py")
    .add_local_file(HERE / "calibration_kappa.py",
                    remote_path="/root/calibration_kappa.py")
    .add_local_file(HERE / "revision6_experiments.py",
                    remote_path="/root/revision6_experiments.py")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("separability-results",
                                      create_if_missing=True)

app = modal.App("separability-revision3")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_revision3_modal(family: str = "llama",
                         n_safety: int = 200,
                         n_gen: int = 100,
                         n_save: int = 50,
                         do_kappa: bool = True,
                         abliterated: str = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"):
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    # write results to volume
    os.makedirs("/results", exist_ok=True)

    from revision3 import run_revision3
    result = run_revision3(
        family=family,
        n_safety=n_safety,
        n_gen=n_gen,
        n_save=n_save,
        do_kappa=do_kappa,
        abliterated=abliterated,
    )
    out_path = f"/results/revision3_{family}.json"
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    results_vol.commit()
    print(f"Saved to volume: {out_path}")
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_kappa_only_modal(family: str = "llama",
                          n_safety: int = 200,
                          abliterated: str = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"):
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)
    from revision3 import run_kappa_unsafe
    result = run_kappa_unsafe(
        family=family,
        n_safety=n_safety,
        abliterated=abliterated,
    )
    # merge with existing perdir+gens result if present
    out_path = f"/results/revision3_{family}.json"
    try:
        with open(out_path) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {'family': family}
    existing['kappa'] = _clean(result)
    if 'kappa_error' in existing:
        del existing['kappa_error']
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    results_vol.commit()
    print(f"Saved kappa to volume: {out_path}")
    return existing


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_mdid_modal():
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)
    from revision6_experiments import run_mdid_ablation, _clean
    result = run_mdid_ablation('llama')
    out_path = "/results/revision6_mdid_llama.json"
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    results_vol.commit()
    print(f"Saved kappa to volume: {out_path}")
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_qwen_modal():
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)
    from revision6_experiments import run_qwen_rank_only, _clean
    result = run_qwen_rank_only('qwen')
    out_path = "/results/revision6_qwen.json"
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    results_vol.commit()
    print(f"Saved qwen to volume: {out_path}")
    return result


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_qwen_test2_modal():
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)
    from revision6_experiments import run_qwen_test2, _clean
    result = run_qwen_test2('qwen')
    out_path = "/results/revision6_qwen_test2.json"
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    results_vol.commit()
    print(f"Saved qwen Test 2 to volume: {out_path}")
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_qwen_bootstrap_modal():
    """Bootstrap CI on rank_eps for Qwen M_template at five
    middle-to-late layers; matches the Llama/Gemma bootstrap blocks
    already present in v4_*.json."""
    import os, sys, json
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)
    from revision6_experiments import run_qwen_bootstrap, _clean
    result = run_qwen_bootstrap('qwen')
    out_path = "/results/revision6_qwen_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    results_vol.commit()
    print(f"Saved qwen bootstrap to volume: {out_path}")
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
    },
)
def run_kappa_multi_modal(family: str = "llama",
                           n_safety: int = 200,
                           abliterated_list = None):
    """Run kappa on multiple unsafe references in a single Modal task.
    Loads honest+base once, then loads each abliterated in turn."""
    import os, sys, json, gc
    import torch
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    sys.path.insert(0, "/root")
    os.makedirs("/results", exist_ok=True)

    if abliterated_list is None:
        abliterated_list = [
            "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
            "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
            "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored",
        ]
    from prompts_v4 import safety_prompts
    from colab_run import MODELS, load_model, unload
    from diagnostics_v4 import collect_acts
    from transformers import AutoTokenizer
    import numpy as np

    cfg = MODELS[family]
    sp = safety_prompts(n_safety)
    aligned_tok = AutoTokenizer.from_pretrained(cfg['aligned'])
    aligned_chat = aligned_tok.chat_template

    # Honest acts (load aligned once)
    print("--- Aligned (honest) acts ---")
    m_a, t_a = load_model(cfg['aligned'], "cuda")
    a_ac_s = collect_acts(m_a, t_a, sp, "cuda", chat=True)
    unload(m_a, t_a); gc.collect(); torch.cuda.empty_cache()

    print("--- Base acts ---")
    m_b, t_b = load_model(cfg['base'], "cuda")
    a_bc_s = collect_acts(m_b, t_b, sp, "cuda", chat=True,
                          chat_template=aligned_chat)
    unload(m_b, t_b); gc.collect(); torch.cuda.empty_cache()

    eps = 0.05
    all_results = {}
    layers = sorted(a_ac_s.keys())

    for ref in abliterated_list:
        print(f"\n=== unsafe ref: {ref} ===")
        try:
            m_u, t_u = load_model(ref, "cuda")
            if t_u.chat_template is None:
                t_u.chat_template = aligned_chat
            a_uc_s = collect_acts(m_u, t_u, sp, "cuda", chat=True)
            unload(m_u, t_u); gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED to load {ref}: {e}")
            all_results[ref] = {"error": str(e)}
            continue

        per_layer = {}
        for l in layers:
            M_h = (a_ac_s[l] - a_bc_s[l]).T
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
            frob_h = float((M_h ** 2).sum())
            frob_u = float((M_u ** 2).sum())
            kappa_proxy = sigma_r ** 2 / (eps * (frob_h + frob_u)) \
                          if (frob_h + frob_u) > 0 else float('inf')
            per_layer[int(l)] = {
                'r_S_eff_rank': r,
                'sigma_1_S': sigma_1,
                'sigma_r_S': sigma_r,
                'frob_M_honest_sq': frob_h,
                'frob_M_unsafe_sq': frob_u,
                'kappa_proxy': float(kappa_proxy),
            }
            print(f"  L{l}: r={r}  sigma_r={sigma_r:.2f}  "
                  f"kappa_proxy={kappa_proxy:.4f}")
        all_results[ref] = {
            'family': family, 'eps': eps,
            'per_layer': per_layer,
        }

    out_path = f"/results/kappa_multi_{family}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    results_vol.commit()
    print(f"\nSaved: {out_path}")
    return all_results


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={
        "/results": results_vol,
    },
)
def run_calibration_kappa_modal():
    import os, sys, json
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from calibration_kappa import main as kappa_main
    kappa_main()
    src = "/root/results/calibration_kappa.json"
    dst = "/results/calibration_kappa.json"
    with open(src) as f:
        data = json.load(f)
    with open(dst, "w") as f:
        json.dump(data, f, indent=2)
    results_vol.commit()
    print(f"Saved to volume: {dst}")
    return data


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={
        "/results": results_vol,
    },
)
def run_calibration_revision3_modal():
    import os, sys, json
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from calibration_v2_revision3 import main as cal_main
    cal_main()
    # main saves to relative results/ — copy to volume
    src = "/root/results/calibration_v2_revision3.json"
    dst = "/results/calibration_v2_revision3.json"
    with open(src) as f:
        data = json.load(f)
    with open(dst, "w") as f:
        json.dump(data, f, indent=2)
    results_vol.commit()
    print(f"Saved to volume: {dst}")
    return data


def _clean(o):
    import numpy as np
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


@app.local_entrypoint()
def main(task: str = "revision3", family: str = "llama",
         n_safety: int = 200, n_gen: int = 100, n_save: int = 50,
         do_kappa: bool = True):
    import json
    if task == "revision3":
        res = run_revision3_modal.remote(
            family=family, n_safety=n_safety,
            n_gen=n_gen, n_save=n_save, do_kappa=do_kappa,
        )
        out = HERE / "results" / f"revision3_{family}_modal.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "kappa_only":
        res = run_kappa_only_modal.remote(
            family=family, n_safety=n_safety,
        )
        out = HERE / "results" / f"revision3_{family}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "mdid_ablation":
        res = run_mdid_modal.remote()
        out = HERE / "results" / "revision6_mdid_llama.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "qwen":
        res = run_qwen_modal.remote()
        out = HERE / "results" / "revision6_qwen.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "qwen_test2":
        res = run_qwen_test2_modal.remote()
        out = HERE / "results" / "revision6_qwen_test2.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "qwen_bootstrap":
        res = run_qwen_bootstrap_modal.remote()
        out = HERE / "results" / "revision6_qwen_bootstrap.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "kappa_multi":
        res = run_kappa_multi_modal.remote(
            family=family, n_safety=n_safety,
        )
        out_path = HERE / "results" / f"kappa_multi_{family}.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2,
                                        default=lambda o: float(o)
                                        if hasattr(o, 'item') else o))
        print(f"Wrote {out_path}")
    elif task == "calibration_kappa":
        res = run_calibration_kappa_modal.remote()
        out = HERE / "results" / "calibration_kappa.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    elif task == "calibration_revision3":
        res = run_calibration_revision3_modal.remote()
        out = HERE / "results" / "calibration_v2_revision3.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2,
                                   default=lambda o: float(o)
                                   if hasattr(o, 'item') else o))
        print(f"Wrote {out}")
    else:
        raise SystemExit(f"unknown task: {task}")
