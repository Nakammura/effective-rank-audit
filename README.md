# An Effective-Rank Audit of Alignment-Induced Activation Shifts

Companion repository for the paper *An Effective-Rank Audit of
Alignment-Induced Activation Shifts: Confound Control, Constructive
Calibration, and Limits* by Yuki Nakamura (The Open University of
Japan).

The paper introduces the separability index
$\rho_\epsilon := \operatorname{rank}_\epsilon(M_{\mathcal{D}_s}) / d$
as a continuous formalization of the single-refusal-direction
observation of Arditi et al. (2024), and reports three contributions:

1. **Confound-controlled measurement** via a four-variant decomposition
   of the alignment modification matrix
   ($M_{\mathrm{naive}}, M_{\mathrm{template}}, M_{\mathrm{aligned}},
   M_{\mathrm{DiD}}$) on three open-weight LLM families
   (Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct).
2. **Constructive calibration** on a 3-layer MLP with sweet-spot vs.
   brittle distinction (same $\rho_\epsilon \approx 0.40$, opposite
   ablation response).
3. **Limits**: SVD-vs-causal mismatch (Llama narrow-band $u_2$
   inert despite ranking second), and a structural Mirsky-route
   obstruction (Lemma: $\kappa_{\mathrm{lb}} \leq 2/(\epsilon r)$) on
   the natural lower-bound proof technique.

## Layout

```
paper/         LaTeX source, TMLR style, references, compiled PDF
experiments/   Diagnostic and calibration code, plus result JSONs
```

## Compiling the paper

```bash
cd paper
tectonic main.tex
# or: latexmk -pdf main.tex
```

The TMLR style file (`tmlr.sty`) is included.  The paper compiles
to 18 pages.

## Reproducing the experiments

### Requirements

- Python 3.11+
- PyTorch 2.5+, Transformers 4.46+, NumPy, scikit-learn
- One A100-80GB or equivalent for the LLM diagnostics
- A HuggingFace token with access to gated Llama 3.1 / Gemma 2 weights
  (export `HF_TOKEN`)
- For the LLM-judge cross-check (`llm_judge_eval.py`): Anthropic API
  key (`export ANTHROPIC_API_KEY`); judge model defaults to
  `claude-haiku-4-5-20251001`, override with `JUDGE_MODEL`.

### Local runs

```bash
cd experiments

# LLM diagnostic (n=200, all four matrix variants, single family).
python diagnostics_v4.py        # default: Llama

# Calibration MLP (CPU acceptable).
python calibration_v2.py
python calibration_v2_revision3.py    # lambda-sweep
python calibration_kappa.py           # kappa_emp in MLP

# Qwen + supplementary experiments.
python revision6_experiments.py qwen
python revision6_experiments.py qwen_test2
python revision3.py                   # per-direction Llama + kappa
```

### Cloud runs (Modal)

`modal_revision3.py` provides Modal wrappers for each task on
A100-80GB:

```bash
pip install modal
modal token new
modal run modal_revision3.py --task qwen_test2
```

Available tasks: `revision3`, `kappa_only`, `mdid_ablation`, `qwen`,
`qwen_test2`, `qwen_bootstrap`, `kappa_multi`, `calibration_kappa`,
`calibration_revision3`.

### Result files

All numerical results used in the paper are pre-computed and
stored under `experiments/results/`:

| File | Contents |
|---|---|
| `v4_llama.json`, `v4_gemma.json` | Llama / Gemma diagnostic (rank, bootstrap, Arditi, LRH, Test 2) |
| `revision6_qwen.json` | Qwen rank and Arditi cosine |
| `revision6_qwen_test2.json` | Qwen narrow-band Test 2 ablation with random control |
| `revision6_qwen_bootstrap.json` | Qwen bootstrap CIs on rank |
| `revision6_mdid_llama.json` | Llama M_DiD narrow-band ablation |
| `revision3_llama.json` | Llama per-direction ablation + saved generations |
| `kappa_multi_llama.json` | $\kappa_{\mathrm{emp}}$ across three Llama unsafe references |
| `calibration_v2_results.json`, `calibration_v2_revision3.json` | Calibration MLP results + lambda-sweep |
| `calibration_kappa.json` | Calibration MLP $\kappa_{\mathrm{emp}}$ across honest/unsafe pairings |

## Citation

Paper preprint: see arXiv link in the repository description (once
posted).  ORCID: [0009-0001-7174-6737](https://orcid.org/0009-0001-7174-6737).

## License

All content in this repository is released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).  Please
cite the paper if you use the code or results.
