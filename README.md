# ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization

[![arXiv](https://img.shields.io/badge/arXiv-2511.14584-b31b1b.svg)](https://arxiv.org/abs/2511.14584)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents"** by Ankush Kadu and Ashwanth Krishnan (QpiAI, 2025).

**Paper:** [arXiv:2511.14584](https://arxiv.org/abs/2511.14584) | **DOI:** [10.48550/arXiv.2511.14584](https://doi.org/10.48550/arXiv.2511.14584)

##  Key Results

- **67% zero-shot success rate** on ALFWorld (Trial 0, first exposure, no demonstrations)
- **78% success rate** on Trial 1 (cross-trial learning improvement)
- **Zero action loops** through TODO-guided exploration  
- **100% component alignment** (TODO-TextGrad-Reflexion synergy)
- Competitive with few-shot baselines: Reflexion (91%, 6-shot), REBACT (93%), ReflAct (93%)

##  Architecture Overview

ReflexGrad integrates three complementary mechanisms in a tightly-coupled synergistic system:

1. **LLM-Based Hierarchical TODO Decomposition**: Strategic planning via pure LLM reasoning (no hardcoded rules)
2. **History-Aware Causal Reflexion**: Analyzes recent action-observation history every 5 steps to identify failure patterns
3. **TextGrad-Based Optimization**: Gradient-based policy updates every 3 steps via LLM-based semantic merge

![ReflexGrad Architecture](flowchart.png)

*Figure: ReflexGrad Dual-Loop Self-Evolution Mechanism showing the three-way synergistic coupling between TODO decomposition, TextGrad optimization (Loop 1, every 3 steps), and Reflexion generation (Loop 2, every 5 steps). The central episodic memory system provides context to all components.*

### Dual-Loop Mechanism

- **Loop 1 (Policy Optimization)**: Every 3 steps, accumulated gradients are synthesized to update the policy via LLM-based semantic merge
- **Loop 2 (Reflexion Generation)**: Every 5 steps or on failure, causal insights are generated from recent action-observation history and stored in working reflexion buffer

### Three-Way Synergy

The key innovation is bidirectional coupling:
- **TODO → Gradients**: TODO context structures which experiences to analyze for gradient computation
- **Reflexions → Gradients**: Generated reflexions inform TextGrad backward pass with specific failure patterns  
- **Gradients → TODO**: Computed gradients determine TODO progression and reflexion consolidation priorities

##  Installation

### Option 1: Using Docker (Recommended)

```bash
# Build image
docker build -t reflexgrad .

# Run container
docker run -it --env OPENAI_API_KEY=your_key reflexgrad

# Or pull pre-built image (if available)
docker pull qpiai/reflexgrad:latest
docker run -it --env OPENAI_API_KEY=your_key qpiai/reflexgrad:latest
```

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/qpiai/reflexgrad.git
cd reflexgrad

# Create conda environment
conda create -n reflexgrad python=3.9
conda activate reflexgrad

# Install dependencies
pip install -r requirements.txt

# Download ALFWorld data
alfworld-download
```

##  API Configuration

ReflexGrad uses a two-tier model architecture:

- **GPT-5** (Responses API with configurable reasoning) for strategic operations
  - Minimal reasoning (~100 tokens): Fast action selection
  - Medium reasoning (~1000 tokens): Reflexion generation and TextGrad computation
- **GPT-4o-mini** for auxiliary tasks (TODO verification, memory compression, loss computation)

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

##  Quick Start

### Run Single Environment (Test)

```bash
python main.py --num_trials 1 --num_envs 1 --run_name test_run
```

### Run Full Benchmark (9 environments × 4 trials)

```bash
python main.py --num_trials 4 --num_envs 9 --run_name full_benchmark
```

### Reproduce Paper Results

```bash
# Trial 0 results (67% success - Table 1)
python main.py --num_trials 1 --num_envs 9 --run_name paper_trial0

# Trial 0-1 results (67% → 78% improvement - Table 2)
python main.py --num_trials 2 --num_envs 9 --run_name paper_full
```

See [REPRODUCE.md](REPRODUCE.md) for detailed reproduction instructions.

##  Configuration

Key hyperparameters in `base_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_steps` | 28 | Episode length limit |
| `learning_rate` | 1.0 | LLM-based semantic merge rate (not numeric) |
| `momentum` | 0.9 | Momentum coefficient for gradient updates |
| `history_window` | 5 | Recent reflexions kept in working memory |
| `policy_update_freq` | 3 | Loop 1 frequency (steps) |
| `reflexion_freq` | 5 | Loop 2 frequency (steps) |
| `memory_retrieval_top_k` | 6 | Cross-trial memory retrieval count |

##  Monitoring Results

Results are logged to:

- `{run_name}/trial_{n}.log` - Full execution logs with step-by-step trace
- `{run_name}/step_gradients.jsonl` - TextGrad loss, gradients, and progress scores
- `{run_name}/metrics.json` - Success rates and statistics per trial

### Example Log Analysis

```bash
# Check success rate
grep "ACCURACY" {run_name}/trial_0.log

# View TextGrad progress evolution (0-10 scale)
grep "Progress:" {run_name}/step_gradients.jsonl | head -15

# Check reflexion generation timing
grep "REFLEXION" {run_name}/trial_0.log

# Verify policy updates (Loop 1: steps 3, 6, 9, ...)
grep "OPTIMIZER" {run_name}/trial_0.log
```

##  Project Structure

```
reflexion/
├── README.md                      # This file
├── REPRODUCE.md                   # Detailed reproduction guide
├── Dockerfile                     # Docker container definition
├── requirements.txt               # Python dependencies
├── flowchart.png                  # Architecture diagram
├── base_config.yaml               # Hyperparameters
├── main.py                        # Entry point
├── alfworld_trial.py              # Main algorithm (7500+ lines)
├── generate_reflections.py        # Reflexion & memory system
├── dynamic_prompting.py           # Prompt management
├── env_history.py                 # Episode history tracking
├── environment_discovery.py       # Environment adaptation
├── environment_understanding.py   # State analysis
├── knowledge_classifier.py        # Pattern classification
├── learning_extractor.py          # Insight extraction
├── meta_discovery.py              # Meta-learning components
├── task_classifier.py             # Task type identification
├── universal_env_wrapper.py       # ALFWorld interface
└── universal_state_embeddings.py  # State representation
```

##  Technical Details

### TextGrad Components

1. **Loss Computation** (`textgrad_loss`): Progress scoring on 0-10 scale
   - Evaluates task alignment: "How close are we to completion?"
   - Returns hypothesis (textual loss) and numeric score

2. **Backward Pass** (`textgrad_backward`): Gradient generation with reflexion context
   - Input: Current policy, action, loss, **and all past reflexions**
   - Output: Textual gradient with justification
   - Example: "go to stoveburner 2 | JUSTIFICATION: Moves to high-probability pan location"

3. **Optimizer** (`optimizer.step`): LLM-based semantic merge (every 3 steps)
   - Synthesizes accumulated gradients via GPT-5
   - Integrates gradient direction into policy using natural language composition
   - Learning rate η=1.0 (not numeric - LLM controls integration strength)

### Reflexion System

1. **Trigger Conditions**: 
   - Every 5 steps (milestone check)
   - On task failure (immediate analysis)

2. **Context Window**: Recent 5-15 action-observation pairs with outcomes

3. **Output**: Causal analysis identifying:
   - Root cause of failures (not just symptoms)
   - Generalizable patterns (e.g., "containers must be opened before use")
   - Corrective strategies

4. **Storage**: Three-tier hierarchy
   - **Working Memory**: Recent 5 reflexions (full detail, 350 tokens each)
   - **Consolidated Memory**: Compressed patterns with importance scores
   - **Episodic Archive**: Complete history for offline analysis

### Memory System

**Importance Scoring** (heuristic-based):
- Success signal: +5.0 for successful episodes
- Critical language: +3.0 if contains actionable keywords ("must", "should", "avoid", etc.)
- Conciseness: +1.0 if <500 tokens

**Forgetting Curve**:
- Time-based decay: strength × (0.995)^(hours_since_access)
- Pruning threshold: strength < 0.1
- Balances retention of valuable patterns against memory capacity

**Cross-Task Transfer** (LLM-based semantic retrieval):
- No hardcoded similarity metrics
- Given new task + candidate memories, LLM evaluates utility
- Transfers patterns like "must open containers" from microwave task to fridge task

##  Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{kadu2025reflexgrad,
  title={ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents},
  author={Kadu, Ankush and Krishnan, Ashwanth},
  journal={arXiv preprint arXiv:2511.14584},
  year={2025},
  url={https://arxiv.org/abs/2511.14584},
  doi={10.48550/arXiv.2511.14584},
  organization={QpiAI}
}
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 QpiAI (Ankush Kadu, Ashwanth Krishnan)

##  Issues & Support

- **Bug Reports**: https://github.com/qpiai/reflexgrad/issues
- **Questions**: Contact ankush.k@qpiai.tech or ashwanth.krishnan@qpiai.tech

##  Acknowledgments

This research was conducted at QpiAI. We thank the creators of:

- **ALFWorld** (Shridhar et al., 2021) for the challenging benchmark environment
- **Reflexion** (Shinn et al., 2023) for episodic memory and self-reflection concepts  
- **TextGrad** (Yuksekgonul et al., 2024) for the gradient-based optimization framework

---

## 📊 Paper Abstract

> Enabling agents to learn from experience and generalize across diverse tasks without task-specific training remains a fundamental challenge in reinforcement learning and decision-making. While recent approaches have explored episodic memory (Reflexion), gradient-based prompt optimization (TextGrad), and hierarchical task decomposition independently, their potential for synergistic integration remains unexplored. We introduce ReflexGrad, a novel architecture that tightly couples three complementary mechanisms: (1) LLM-based hierarchical TODO decomposition for strategic planning, (2) history-aware causal reflection that analyzes recent action patterns to identify failure root causes and enable within-trial learning, and (3) gradient-based optimization for systematic improvement. Unlike prior work relying on few-shot demonstrations, our system achieves true zero-shot generalization through pure LLM semantic reasoning, requiring no task-specific examples, fine-tuning, or hardcoded similarity metrics. Evaluated on ALFWorld benchmark tasks, ReflexGrad demonstrates 67% zero-shot success rate on Trial 0 without any prior task experience or demonstrations, establishing effective performance on first exposure.
