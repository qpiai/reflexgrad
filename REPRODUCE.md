# Reproducing Paper Results

This document provides exact commands to reproduce all results from the paper "ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents."

## Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Download ALFWorld data: `alfworld-download`
3. Set API key: `export OPENAI_API_KEY=your_key_here`

## Table 1: Zero-Shot Performance (Trial 0)

**Paper Claim:** 67% success rate (6/9 environments) on first exposure without demonstrations

```bash
python main.py \
  --num_trials 1 \
  --num_envs 9 \
  --run_name table1_zero_shot
```

**Expected Output:**
- Environments succeeded: 6/9
- Success rate: 67%
- Zero action loops
- 100% component alignment

**Check Results:**
```bash
grep "ACCURACY" table1_zero_shot/trial_0.log
```

---

## Table 2: Cross-Trial Learning

**Paper Claim:** 67% (Trial 0) → 78% (Trial 1) improvement through cross-trial memory transfer

```bash
python main.py \
  --num_trials 2 \
  --num_envs 9 \
  --run_name table2_cross_trial
```

**Expected Output:**
- Trial 0: 6/9 environments (67%)
- Trial 1: 7/9 environments (78%)
- 11 percentage point improvement

**Check Results:**
```bash
# Trial 0 accuracy
grep "ACCURACY" table2_cross_trial/trial_0.log

# Trial 1 accuracy  
grep "ACCURACY" table2_cross_trial/trial_1.log
```

---

## Figure 2: Detailed Execution Trace

**Paper Claim:** 11-step execution showing TextGrad loss, gradients, policy updates, and reflexions

The detailed trace in Section 5.4 comes from Environment 0, Trial 0. To reproduce:

```bash
python main.py \
  --num_trials 1 \
  --num_envs 1 \
  --env_start 0 \
  --run_name figure2_detailed_trace
```

**Verify Components:**
```bash
# Check TextGrad loss computation
grep "Progress:" figure2_detailed_trace/step_gradients.jsonl | head -12

# Check policy updates (steps 3, 6, 9)
grep "OPTIMIZER" figure2_detailed_trace/trial_0.log

# Check reflexion generation (steps 5, 10)
grep "REFLEXION" figure2_detailed_trace/trial_0.log
```

---

## Ablation Studies (Table 2 in Paper)

### Reflexion-Only Baseline

```bash
python main.py \
  --num_trials 1 \
  --num_envs 9 \
  --disable_textgrad \
  --run_name ablation_reflexion_only
```

**Expected:** ~33% success rate, high loop count

### TextGrad-Only Baseline

```bash
python main.py \
  --num_trials 1 \
  --num_envs 9 \
  --disable_reflexion \
  --run_name ablation_textgrad_only
```

**Expected:** ~44% success rate, moderate loops

### Full ReflexGrad (Our System)

```bash
python main.py \
  --num_trials 1 \
  --num_envs 9 \
  --run_name ablation_full_system
```

**Expected:** 67% success rate, zero loops

---

## Hyperparameter Verification

The paper reports these hyperparameters (Appendix C, Table 3):

| Parameter | Value |
|-----------|-------|
| max_steps | 28 |
| learning_rate (η) | 1.0 (LLM-based merge) |
| momentum (β) | 0.9 |
| history_window (k) | 5 |
| memory_retrieval_top_k | 6 |
| policy_update_freq | 3 steps |
| reflexion_freq | 5 steps |

**Verify in code:**
```bash
# Check configuration
cat base_config.yaml

# Or verify in code
grep -E "max_steps|learning_rate|history_window" alfworld_trial.py | head -10
```

---

## Full 4-Trial Benchmark

To run the complete benchmark as in the paper experiments:

```bash
python main.py \
  --num_trials 4 \
  --num_envs 9 \
  --run_name full_benchmark
```

**Expected Results:**
- Trial 0: 6/9 (67%) - zero-shot
- Trial 1: 7/9 (78%) - cross-trial learning
- Trial 2-3: Continued performance ~75-80%

**Analysis:**
```bash
# Check all trial accuracies
for trial in 0 1 2 3; do
  echo "=== Trial $trial ==="
  grep "ACCURACY" full_benchmark/trial_${trial}.log
done
```

---

## Computational Cost Estimation

**Paper Reports:**
- Total API calls: ~15,000 (9 envs × 4 trials)
- Estimated cost per trial: $3.50
- Total experimental cost: ~$125

**Measure Your Run:**
```bash
# Count API calls (approximate)
wc -l full_benchmark/step_gradients.jsonl

# Check actual steps taken
grep "Step" full_benchmark/trial_0.log | wc -l
```

---

## Notes

1. **Randomness:** ALFWorld uses fixed seeds for reproducibility, but LLM outputs may vary slightly between runs
2. **Success Criteria:** An environment is "successful" only if the task is fully completed within 28 steps
3. **API Costs:** Running full experiments (~15K API calls) costs approximately $125 with GPT-5/GPT-4o-mini
4. **Runtime:** Full 4-trial × 9-env benchmark takes approximately 8-10 hours

---

## Troubleshooting

### If success rates are lower than expected:

1. Verify OpenAI API key is set correctly
2. Check you're using GPT-5 (Responses API) not GPT-4
3. Ensure ALFWorld data was downloaded: `alfworld-download`
4. Check logs for API errors: `grep "ERROR" {run_name}/trial_0.log`

### If the system hangs:

1. Check API rate limits
2. Verify network connectivity
3. Monitor system memory (each environment needs ~2GB RAM)

---

For additional questions, see the main [README.md](README.md) or open an issue.
