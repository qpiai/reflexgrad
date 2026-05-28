# Reproducing Paper Results

Exact commands to reproduce results from **"ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing"** (arXiv:2511.14584).

## Prerequisites

```bash
pip install -r requirements.txt
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld/data

# Choose a provider:
export OPENAI_API_KEY=your_key                 # for GPT-5 results
# or
export OPENROUTER_API_KEY=your_key             # for Qwen-3-8B results
export OPENROUTER_MODEL=qwen/qwen3-8b
export OPENROUTER_FAST_MODEL=qwen/qwen3-8b
```

All headline runs use **no demonstrations**, `max_steps=15`, and the fixed seed list
`{42, 123, 456, 789, 1024, 1337, 2025, 3141, 5926, 7531}` (n=10).

---

## Table 2: Cross-Model Ablation (headline result)

**Paper claim:** GPT-5 46.3% → 88.1%; Qwen-3-8B 35.1% → 75.4% (no demonstrations).

```bash
# Qwen-3-8B (open-weight), 134 tasks → 75.4%
python main.py --model_provider openrouter --num_trials 1 --num_envs 134 \
  --run_name paper_qwen --env_type alfworld

# GPT-5, 134 tasks → 88.1%
python main.py --model_provider openai --num_trials 1 --num_envs 134 \
  --run_name paper_gpt5 --env_type alfworld
```

**Component ablations** (rows of Table 2):

```bash
# TextGrad-only (fast process only): GPT-5 69.4%, Qwen-3-8B 61.2%
python main.py --model_provider openrouter --num_trials 1 --num_envs 134 \
  --ablation_mode textgrad_only --pure_ablation --run_name abl_textgrad --env_type alfworld

# Reflexion-only (slow process only): GPT-5 53.0%, Qwen-3-8B 42.5%
python main.py --model_provider openrouter --num_trials 1 --num_envs 134 \
  --ablation_mode reflexion_only --pure_ablation --run_name abl_reflexion --env_type alfworld
```

**Check results:**
```bash
grep "ACCURACY" paper_qwen/trial_0.log
```

---

## Table 3: Compute-Matched Comparison

**Paper claim:** demo-free ReflexGrad on Qwen-3-8B (75.4%) beats 1-shot LATS (+2.7pp), ToT (+5.7pp),
and Self-Refine (+6.7pp), all at *p* < 0.05.

The three baselines are released as separate re-implementations. Each runs in its published 1-shot
configuration on the same Qwen-3-8B backend and 134-task set. See the baseline directories and their
sanity-check anchors (Appendix B of the paper).

---

## Table 4: Sensitivity to Routing Thresholds

**Paper claim:** success stays within 84.3–88.1% across sweeps; worst setting (m=3) still 84.3%.

```bash
# Sweep gradient cadence k ∈ {2,3,5}, trigger m ∈ {3,5,7}, threshold θ_low ∈ {3,4,7}
# (edit base_config.yaml or pass overrides; defaults k=3, m=5, θ_low=4)
python main.py --model_provider openai --num_trials 1 --num_envs 134 \
  --run_name sweep_m3 --env_type alfworld   # set m=3 to reproduce the 84.3% worst case
```

---

## Table 5: Scaling with Step Budget (GPT-5)

**Paper claim:** 56.0% (5 steps) → 76.1% (10) → 88.1% (15) → 90.3% (20). Headline uses 15.

```bash
for steps in 5 10 15 20; do
  python main.py --model_provider openai --num_trials 1 --num_envs 134 \
    --max_steps $steps --run_name scaling_${steps} --env_type alfworld
done
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| max_steps (headline) | 15 |
| gradient cadence k | 3 |
| slow trigger m | 5 |
| low-progress threshold θ_low | 4 |
| cooldown c | 5 |
| working memory k_M | 10 |

```bash
cat base_config.yaml
```

---

## Notes

1. **Reproducibility:** the fixed n=10 seed list reproduces the reported means; LLM outputs are decoded with each backend's default deterministic settings.
2. **Success criteria:** an episode succeeds only if ALFWorld's deterministic completion oracle confirms task completion within the step budget — the LLM evaluator is used only as the routing signal, never as the success metric.
3. **Runtime:** open-weight (Qwen-3-8B) episodes are slower than frontier API calls; a full 134-task single-seed run can take several hours depending on provider throughput.

---

## Troubleshooting

- **Lower success than expected:** verify the provider/model env vars, confirm `ALFWORLD_DATA` is set, and check `grep "ERROR" {run_name}/trial_0.log`.
- **Hangs / rate limits:** OpenRouter and OpenAI both rate-limit; the wrappers back off automatically, but heavy parallelism may stall — reduce `--num_envs` batch size.

For more, see [README.md](README.md) or open an issue.
