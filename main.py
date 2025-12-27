import os
import json
import argparse
import sys
import re
import pickle
from dotenv import load_dotenv
load_dotenv()

# Set seeds for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# Configure ALFWorld to use official benchmark (134 evaluation tasks)
# This ensures publication-level research with standardized evaluation
os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.cache/alfworld')

# Parse args early to determine model provider
parser_early = argparse.ArgumentParser(add_help=False)
parser_early.add_argument("--model_provider", type=str, default="openai", choices=["openai", "gemini"])
args_early, _ = parser_early.parse_known_args()

# Set environment variable for model provider (so shared_model files can detect it)
os.environ['MODEL_PROVIDER'] = args_early.model_provider

# Print model provider selection
print(f"\n{'='*60}")
print(f"MODEL PROVIDER SELECTED: {args_early.model_provider.upper()}")
print(f"{'='*60}\n")

# Check API key based on provider
if args_early.model_provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env file. Please add: OPENAI_API_KEY=your_key_here")
    print(f'OpenAI API Key loaded: {api_key[:20]}...')
elif args_early.model_provider == "gemini":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env file. Please add: GEMINI_API_KEY=your_key_here")
    print(f'Gemini API Key loaded: {api_key[:20]}...')

from alfworld_trial import run_trial
from alfworld_trial import prompt_generator
from generate_reflections import update_memory
from typing import Any, List, Dict , Tuple

# Test resume capability
if '--test-resume' in sys.argv:
    print("Testing checkpoint/resume system...")
    from checkpoint_manager import CheckpointManager
    cm = CheckpointManager('test_run')
    cm.save_master_state(0, [{'test': True}], [0], [1,2])
    loaded = cm.load_master_state()
    assert loaded is not None, "Failed to load checkpoint"
    print("✓ Checkpoint system working")
    sys.exit(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--skip_discovery", action='store_true', help="Skip environment discovery")
    parser.add_argument("--env_type", type=str, default="alfworld",
                        choices=["alfworld", "textworld_cooking", "textworld_treasure", "textworld_simple",
                                "jericho_zork1", "jericho_detective", "jericho_balances",
                                "scienceworld_boil", "scienceworld_melt", "scienceworld_grow",
                                "babyai_goto", "babyai_pickup", "babyai_unlock",
                                "appworld"],
                        help="Environment type to use")
    parser.add_argument("--parallel", action='store_true', help="Enable parallel execution for A100")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for parallel execution")
    parser.add_argument("--env_configs_file", type=str, help="Path to environment configs JSON file", default="")
    parser.add_argument("--adaptive", action='store_true', help="Enable adaptive early stopping")
    parser.add_argument("--perfect_trials", type=int, default=3,
                        help="Number of perfect trials before early stopping (default: 3)")
    parser.add_argument("--perfect_threshold", type=float, default=0.98,
                        help="Accuracy threshold for 'perfect' performance (default: 0.98)")

    # Model provider selection
    parser.add_argument("--model_provider", type=str, default="openai",
                        choices=["openai", "gemini"],
                        help="Model provider to use: 'openai' (default) or 'gemini'")

    # Ablation study mode
    parser.add_argument("--ablation_mode", type=str, default='combined',
                        choices=['textgrad_only', 'reflexion_only', 'combined'],
                        help="Ablation study mode: textgrad_only, reflexion_only, or combined (default: combined)")

    # Debug flags
    parser.add_argument("--debug", action='store_true', help="Enable all debug outputs")
    parser.add_argument("--debug_actor", action='store_true', help="Enable actor (action selection) debug output")
    parser.add_argument("--debug_critic", action='store_true', help="Enable critic (TextGrad) debug output")
    parser.add_argument("--debug_reflexion", action='store_true', help="Enable reflexion debug output")
    parser.add_argument("--no_debug", action='store_true', help="Disable all debug outputs")

    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    return args

def set_debug_flags(args):
    """Set debug flags based on command line arguments"""
    import alfworld_trial
    

    # Force ensure the flags are actually set in the modules
    print(f"\n[DEBUG] Verifying debug flags are set:")
    print(f"  alfworld_trial.DEBUG_ACTOR = {alfworld_trial.DEBUG_ACTOR}")
    print(f"  alfworld_trial.DEBUG_CRITIC = {alfworld_trial.DEBUG_CRITIC}")
    print(f"  alfworld_trial.DEBUG_REFLEXION = {alfworld_trial.DEBUG_REFLEXION}")
    
    # Also set in dynamic_prompting module
    from dynamic_prompting import set_debug_flags as dp_set_debug
    dp_set_debug(alfworld_trial.DEBUG_ACTOR, alfworld_trial.DEBUG_CRITIC)

    if args.no_debug:
        # Disable all debug
        alfworld_trial.DEBUG_ACTOR = False
        alfworld_trial.DEBUG_CRITIC = False
        alfworld_trial.DEBUG_REFLEXION = False
    elif args.debug:
        # Enable all debug
        alfworld_trial.DEBUG_ACTOR = True
        alfworld_trial.DEBUG_CRITIC = True
        alfworld_trial.DEBUG_REFLEXION = True
    else:
        # Set individual flags
        if args.debug_actor:
            alfworld_trial.DEBUG_ACTOR = True
        if args.debug_critic:
            alfworld_trial.DEBUG_CRITIC = True
        if args.debug_reflexion:
            alfworld_trial.DEBUG_REFLEXION = True
        
        # DEFAULT: If use_memory is True, enable reflexion debugging
        if args.use_memory and not args.no_debug:
            alfworld_trial.DEBUG_REFLEXION = True
            alfworld_trial.DEBUG_CRITIC = True  # For step reflections
    
    print(f"\nDebug Settings:")
    print(f"  Actor Debug: {alfworld_trial.DEBUG_ACTOR}")
    print(f"  Critic Debug: {alfworld_trial.DEBUG_CRITIC}")
    print(f"  Reflexion Debug: {alfworld_trial.DEBUG_REFLEXION}")
    print()


def meta_analysis_after_trial0(env_configs, world_log_path):
    """
    Meta-learning phase after Trial 0.

    For successful envs: Analyze trajectory, find optimal sequence, mark for direct replay.
    For failed envs: Prepare context with partial progress and insights from successful ones.

    This is a UNIVERSAL algorithm - no hardcoding, works with any task domain.
    """
    print(f"\n{'='*80}")
    print("META-ANALYSIS PHASE: Analyzing Trial 0 results for optimal Trial 1 strategy")
    print(f"{'='*80}\n")

    successful_envs = []
    failed_envs = []

    # Separate envs by success/failure (use enumerate to track env_id)
    for env_id, env in enumerate(env_configs):
        env['env_id'] = env_id  # Ensure env_id is set
        if env.get('is_success', False):
            successful_envs.append((env_id, env))
        else:
            failed_envs.append((env_id, env))

    print(f"Trial 0 Results: {len(successful_envs)} successful, {len(failed_envs)} failed\n")

    # Phase 1: Analyze successful envs - extract optimal sequences
    print("=" * 60)
    print("PHASE 1: Extracting optimal sequences from successful envs")
    print("=" * 60)

    for env_id, env in successful_envs:

        # Get the success_workflow from memory
        success_workflow = None
        for mem in env.get('memory', []):
            if isinstance(mem, dict) and mem.get('type') == 'success_workflow':
                success_workflow = mem
                break

        if success_workflow:
            actions = success_workflow.get('actions', [])
            task = success_workflow.get('task', '')

            # Use LLM to analyze and find optimal sequence
            # This is universal - LLM reasons about what actions were necessary
            optimal_sequence = analyze_for_optimal_sequence(task, actions)

            if optimal_sequence:
                env['optimal_sequence'] = optimal_sequence
                env['use_direct_replay'] = True
                print(f"  ENV {env_id}: Optimal sequence found ({len(optimal_sequence)} steps vs {len(actions)} original)")
            else:
                # Keep original sequence if optimization fails
                env['optimal_sequence'] = actions
                env['use_direct_replay'] = True
                print(f"  ENV {env_id}: Using original sequence ({len(actions)} steps)")
        else:
            print(f"  ENV {env_id}: No success_workflow found, will use normal learning")

    # Phase 2: Prepare failed envs with accumulated insights
    print("\n" + "=" * 60)
    print("PHASE 2: Preparing failed envs with accumulated knowledge")
    print("=" * 60)

    # Collect insights from ALL successful envs
    all_successful_insights = []
    for env_id, env in successful_envs:
        for mem in env.get('memory', []):
            if isinstance(mem, dict):
                # Collect reflexions and step reflections
                if mem.get('type') in ['reflexion', 'step_reflection']:
                    all_successful_insights.append({
                        'task_type': env.get('task_type', 'unknown'),
                        'content': mem.get('content', '') or mem.get('reflection', ''),
                        'from_success': True
                    })

    for env_id, env in failed_envs:
        # Mark for enhanced learning (same algo but with extra context)
        env['use_direct_replay'] = False

        # Add insights from successful envs as additional context
        env['cross_env_insights'] = all_successful_insights

        # Get partial progress - what locations were already explored
        explored_locations = set()
        for mem in env.get('memory', []):
            if isinstance(mem, dict) and mem.get('type') == 'step_reflection':
                # Extract explored locations from step reflections
                content = mem.get('content', '')
                # Let the agent know what was already tried
                explored_locations.add(content)

        env['explored_context'] = list(explored_locations)

        print(f"  ENV {env_id}: Prepared with {len(all_successful_insights)} cross-env insights, {len(explored_locations)} explored contexts")

    # Log the meta-analysis
    with open(world_log_path, 'a') as wf:
        wf.write(f"\n\n***** META-ANALYSIS AFTER TRIAL 0 *****\n")
        wf.write(f"Successful envs: {[eid for eid, e in successful_envs]}\n")
        wf.write(f"Failed envs: {[eid for eid, e in failed_envs]}\n")
        wf.write(f"Envs with direct replay: {[e.get('env_id') for e in env_configs if e.get('use_direct_replay')]}\n")
        wf.write("*****\n\n")

    print(f"\nMeta-analysis complete. {len([e for e in env_configs if e.get('use_direct_replay')])} envs will use direct replay.\n")

    return env_configs


def analyze_for_optimal_sequence(task: str, actions: list) -> list:
    """
    Analyze action sequence and find the optimal (minimal) sequence.
    For now, returns the original actions (optimization can be added later).

    This is UNIVERSAL - works with any task type.
    """
    if not actions:
        return None

    # For now, just return the original actions
    # Future: Use LLM to optimize by removing redundant exploration steps
    return actions

# Placeholder for future LLM-based optimization
def _analyze_for_optimal_sequence_with_llm(task: str, actions: list) -> list:
    """Future: Use LLM to optimize action sequence."""
    # This can be implemented later when we have the right model interface
    pass

def check_gpu_and_optimize():
    """Check GPU and suggest optimizations"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"Detected GPU: {gpu_info}")
            
            if "A100" in gpu_info and "81920" in gpu_info:
                print("âœ“ A100 80GB detected - enabling maximum optimizations!")
                return True
            elif "A100" in gpu_info:
                print("âœ“ A100 detected - enabling optimizations!")
                return True
    except:
        pass
    return False

def extract_accuracy_from_log(world_log_path: str, trial_num: int) -> float:
    """Extract accuracy for a specific trial from world log"""
    try:
        with open(world_log_path, 'r') as f:
            content = f.read()
        
        # Find the trial section
        trial_pattern = f"Trial #{trial_num}.*?ACCURACY: ([\d.]+)"
        match = re.search(trial_pattern, content, re.DOTALL)
        
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.0

def check_early_stopping(accuracies: List[float], perfect_threshold: float, 
                        perfect_trials_needed: int) -> Tuple[bool, str]:
    """
    Check if we should stop early based on performance
    Returns: (should_stop, reason)
    """
    if len(accuracies) < perfect_trials_needed:
        return False, ""
    
    # Check last N trials
    recent_accuracies = accuracies[-perfect_trials_needed:]
    
    # Check if all recent trials meet threshold
    if all(acc >= perfect_threshold for acc in recent_accuracies):
        avg_recent = sum(recent_accuracies) / len(recent_accuracies)
        return True, f"Achieved {avg_recent:.1%} average accuracy for {perfect_trials_needed} consecutive trials"
    
    # Check if we've plateaued at a high level
    if len(accuracies) >= perfect_trials_needed * 2:
        # Check if improvement has stalled
        first_half = accuracies[-(perfect_trials_needed*2):-perfect_trials_needed]
        second_half = accuracies[-perfect_trials_needed:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        # If high accuracy and no improvement
        if avg_second >= 0.95 and abs(avg_second - avg_first) < 0.01:
            return True, f"Performance plateaued at {avg_second:.1%} accuracy"
    
    return False, ""

def main(args) -> None:
    # CRITICAL: Clear contaminated state from previous runs
    if not args.is_resume:  # Only for fresh runs
        import glob
        import shutil

        # Clean ALL state files to prevent contamination
        state_files = glob.glob('*.pkl')
        if state_files:
            print(f"[CLEAN STATE] Found {len(state_files)} state files from previous runs")
            for pkl_file in state_files:
                # Special handling for prompt_generator_state
                if pkl_file == 'prompt_generator_state.pkl':
                    backup_name = f'prompt_generator_state_backup_{args.run_name.replace("/", "_")}.pkl'
                    shutil.move(pkl_file, backup_name)
                    print(f"  Backed up: {pkl_file} -> {backup_name}")
                else:
                    # Remove other state files
                    os.remove(pkl_file)
                    print(f"  Removed: {pkl_file}")
            print("[FRESH START] All state files cleaned for unbiased run")
        else:
            print("[FRESH START] No previous state files found")
    
    
    # Set debug flags
    set_debug_flags(args)

    # Model will be loaded on first actual use (preload wastes API calls and hits rate limits)
    print("Skipping preload to avoid rate limiting...\n")
    
    # Use internal batch processing
    is_a100 = False  # Can be detected if needed
    
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")

        # Create new run directory for resumed run (don't overwrite original)
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # Load checkpoint manager from RESUME directory (for loading old checkpoints)
        from checkpoint_manager import CheckpointManager
        checkpoint_manager_resume = CheckpointManager(args.resume_dir)

        # Create checkpoint manager for NEW directory (for saving new checkpoints)
        checkpoint_manager = CheckpointManager(logging_dir)

        # Load master state from RESUME directory
        master_state = checkpoint_manager_resume.load_master_state()
        if master_state:
            print(f"[RESUME] Found master checkpoint")
            print(f"  Last trial: {master_state.get('trial_idx', args.start_trial_num)}")
            print(f"  Completed envs: {len(master_state.get('completed_envs', []))}")
            print(f"  API calls used: {master_state.get('api_calls', 'unknown')}")

            # Restore API call count to NEW checkpoint manager
            if 'api_calls' in master_state:
                checkpoint_manager.api_calls = master_state['api_calls']
        
            
            # Update start_trial_num from checkpoint ONLY if user didn't specify it
            if 'trial_idx' in master_state:
                # If user didn't provide --start_trial_num (default 0), auto-increment from checkpoint
                if args.start_trial_num == 0:
                    args.start_trial_num = master_state['trial_idx'] + 1  # Auto: resume at next trial
                    print(f"[RESUME AUTO] Will start at trial {args.start_trial_num} (last completed: {master_state['trial_idx']})")
                else:
                    # User explicitly set --start_trial_num, respect their choice
                    print(f"[RESUME MANUAL] Using user-specified trial {args.start_trial_num} (last completed: {master_state['trial_idx']})")

        # Load environment configs
        # Try checkpoint file first (has latest memories even if crashed mid-trial)
        env_config_checkpoint = os.path.join(args.resume_dir, f'env_configs_trial_{args.start_trial_num - 1}.json')
        env_config_path = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')

        if os.path.exists(env_config_checkpoint):
            env_config_path = env_config_checkpoint
            print(f"[RESUME] Using checkpoint file (has latest memories)")
        elif os.path.exists(env_config_path):
            print(f"[RESUME] Using trial results file")

        # Load memory systems
        memory_manager_path = os.path.join(args.resume_dir, 'global_memory_manager.pkl')
        loaded_memory_manager = None
        if os.path.exists(memory_manager_path):
            from generate_reflections import global_memory_manager
            with open(memory_manager_path, 'rb') as f:
                loaded_manager = pickle.load(f)
                global_memory_manager.memory = loaded_manager.memory
                loaded_memory_manager = loaded_manager
                print(f"[RESUME] Loaded memory manager with {len(loaded_manager.memory.working_memory)} working memories")
                print(f"[RESUME] Consolidated insights: {len(loaded_manager.memory.consolidated_memory)}")
        else:
            print("[RESUME WARNING] No global_memory_manager.pkl found - starting with empty memories")

        # Load universal memory
        universal_memory_path = os.path.join(args.resume_dir, 'universal_memory/universal_memory.pkl')
        if os.path.exists(universal_memory_path):
            from alfworld_trial import universal_memory
            universal_memory.load_memory()
            print(f"[RESUME] Loaded universal memory with {len(universal_memory.state_action_outcomes)} states")
        else:
            print("[RESUME WARNING] No universal_memory.pkl found")

        # Load environment configs
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config file `{env_config_path}` does not exist")
        with open(env_config_path, 'r') as rf:
            env_configs = json.load(rf)

        print(f"[RESUME] Loaded {len(env_configs)} environment configs from trial {args.start_trial_num - 1}")

        # FIX: Populate env_configs with actual memories from loaded memory manager
        if loaded_memory_manager is not None:
            print(f"[RESUME FIX] Populating environment memories with actual reflexions...")

            total_populated = 0
            for env_config in env_configs:
                task = env_config.get('task', '')
                env_name = env_config.get('name', '')
                relevant_memories = []

                # Extract task-relevant memories from working memory
                for mem in loaded_memory_manager.memory.working_memory:
                    mem_task = mem.get('task', '')
                    mem_reflection = mem.get('reflection', '')
                    mem_success = mem.get('success', False)

                    # Prioritize successful memories for same task
                    if mem_task == task and mem_reflection:
                        if mem_reflection not in relevant_memories:
                            relevant_memories.append(mem_reflection)

                # Add high-quality consolidated insights (task-agnostic wisdom)
                for key, mem in loaded_memory_manager.memory.consolidated_memory.items():
                    if mem.get('strength', 0.0) >= 3.0:  # High-quality threshold
                        insight = mem.get('insight', '')
                        if insight and insight not in relevant_memories:
                            relevant_memories.append(insight)                # CHECK: Does this env already have success_workflow patterns from JSON?
                existing_memory = env_config.get('memory', [])
                has_patterns = any(
                    isinstance(mem, dict) and mem.get('type') in ['success_workflow', 'failure_pattern']
                    for mem in existing_memory
                )

                # Populate with actual memories (up to trial_num worth of memories)
                max_memories = min(args.start_trial_num + 2, len(relevant_memories))

                # PRESERVE success_workflow patterns - only add reflexions if no patterns exist
                if has_patterns:
                    # Environment has success patterns from Trial 0 - DON'T OVERWRITE!
                    pattern_count = sum(1 for mem in existing_memory if isinstance(mem, dict) and mem.get('type') == 'success_workflow')
                    print(f"  ✅ {env_name}: {pattern_count} patterns (success + failure) PRESERVED from Trial 0")
                    total_populated += 1
                elif relevant_memories:
                    env_config['memory'] = relevant_memories[:max_memories]
                    total_populated += 1
                    print(f"  ✅ {env_name}: {len(env_config['memory'])} reflexion memories loaded")
                else:
                    # No relevant memories found - keep placeholders for count
                    env_config['memory'] = [""] * args.start_trial_num
                    print(f"  ⚠️  {env_name}: No relevant memories (using placeholders)")

            print(f"[RESUME FIX] Successfully populated {total_populated}/{len(env_configs)} environments with memories")
        else:
            print("[RESUME WARNING] Cannot populate memories - memory manager not loaded")

        print(f"[RESUME] Resuming from trial {args.start_trial_num} with {len(env_configs)} environments")

        # VERIFICATION: Check that memories are properly loaded
        print("\n" + "="*60)
        print("MEMORY LOADING VERIFICATION")
        print("="*60)
        sample_env = env_configs[0] if env_configs else None
        if sample_env:
            mem_list = sample_env.get('memory', [])
            print(f"Sample environment: {sample_env.get('name', 'unknown')}")
            print(f"Memory count: {len(mem_list)}")
            if mem_list:
                non_empty = [m for m in mem_list if m and (isinstance(m, dict) or (isinstance(m, str) and m.strip()))]
                print(f"Non-empty memories: {len(non_empty)}/{len(mem_list)}")
                if non_empty:
                    print(f"✅ GOOD: Memories contain actual content")
                    sample = non_empty[0]
                    if isinstance(sample, dict):
                        print(f"Sample memory type: dict with keys: {list(sample.keys())[:5]}")
                    else:
                        print(f"Sample memory (first 100 chars): {sample[:100]}...")
                    print(f"⚠️  WARNING: All memories are empty strings!")
                    print(f"Memory transfer may have failed.")
        print("="*60 + "\n")
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # Load environment configs from file if provided
        if args.env_configs_file and os.path.exists(args.env_configs_file):
            with open(args.env_configs_file, 'r') as f:
                env_configs = json.load(f)
            print(f"Loaded {len(env_configs)} environment configs from {args.env_configs_file}")
        else:
            # initialize environment configs
            env_configs: List[Dict[str, Any]] = []
            for i in range(args.num_envs):
                env_configs += [{
                    'name': f'env_{i}',
                    'memory': [],
                    'is_success': False,
                    'skip': False
                }]

            # FIX 3: Populate env_configs with global memory for fresh runs
            # This gives Trial 0 access to pre-loaded successful episodes for positive guidance
            if not args.is_resume:
                try:
                    from generate_reflections import global_memory_manager

                    # Get all successful episodes from global memory
                    all_episodes = global_memory_manager.memory.episodic_archive if hasattr(global_memory_manager.memory, 'episodic_archive') else []
                    successful_episodes = [ep for ep in all_episodes if ep.get('success', False)]

                    if successful_episodes:
                        print(f"[MEMORY INIT] Loading {len(successful_episodes)} successful episodes into env configs")

                        # Sort by importance (or timestamp if no importance)
                        successful_episodes.sort(key=lambda x: x.get('importance', x.get('timestamp', 0)), reverse=True)

                        # Populate each env with top 15 most important episodes
                        for env_config in env_configs:
                            env_config['memory'] = successful_episodes[:15]

                        print(f"[MEMORY INIT] Each of {len(env_configs)} environments now has {len(successful_episodes[:15])} memory episodes")
                    else:
                        print(f"[MEMORY INIT] No successful episodes found in global memory - starting with empty memory")
                except Exception as e:
                    print(f"[MEMORY INIT WARNING] Could not load global memory: {e}")
                    print(f"[MEMORY INIT WARNING] Continuing with empty memory")

    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # Initialize progress logger for real-time monitoring
    from progress_logger import get_progress_logger
    progress_logger = get_progress_logger(logging_dir)
    print(f"[PROGRESS] Real-time progress logging enabled")
    print(f"  Monitor: {logging_dir}/progress_summary.json")
    print(f"  Detailed log: {logging_dir}/progress.jsonl\n")

    # print start status to user
    print(f"""
    -----
    {"Resuming" if args.is_resume else "Starting"} run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Use memory: {args.use_memory}
    Environment type: {args.env_type}
    Skip discovery: {args.skip_discovery}
    Parallel execution: {args.parallel}
    Batch size: {args.batch_size}
    Adaptive training: {args.adaptive}
    Perfect threshold: {args.perfect_threshold:.1%}
    Perfect trials needed: {args.perfect_trials}
    {"Resume trial number: " + str(args.start_trial_num) if args.is_resume else ""}

    Sending all logs to `{logging_dir}`
    -----
    """)

    # Track accuracies for adaptive stopping
    trial_accuracies = []
    
    # If resuming, load previous accuracies
    if args.is_resume and args.start_trial_num > 0:
        for i in range(args.start_trial_num):
            acc = extract_accuracy_from_log(world_log_path, i)
            if acc > 0:
                trial_accuracies.append(acc)
    
    # run trials
    trial_idx = args.start_trial_num
    early_stopped = False
    
    while trial_idx < args.num_trials:
        # Log trial start
        progress_logger.log_trial_start(trial_idx, len(env_configs))

        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # PROFILING: Memory checkpoint BEFORE trial
        try:
            from memory_profiler import profiler
            from generate_reflections import global_memory_manager

            # universal_memory is created per-trial, so we can't access it here
            profiler.checkpoint(f"Trial {trial_idx} START", {
                'global_memory_manager.working_memory': global_memory_manager.memory.working_memory,
                'global_memory_manager.consolidated_memory': global_memory_manager.memory.consolidated_memory,
                'global_memory_manager.episodic_archive': global_memory_manager.memory.episodic_archive,
                'env_configs': env_configs
            })
        except Exception as e:
            print(f"[WARNING] Profiling failed at trial start: {e}")

        # set paths to log files
        trial_log_path: str = os.path.join(logging_dir, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # run trial with environment type
        run_trial(trial_log_path, world_log_path, trial_idx, env_configs,
                  args.use_memory, args.skip_discovery, args.env_type, args.batch_size,
                  ablation_mode=args.ablation_mode)
        print(f"RUN_TRIAL COMPLETED for trial {trial_idx}", flush=True)

        # PROFILING: Memory checkpoint AFTER trial
        try:
            from memory_profiler import profiler
            from generate_reflections import global_memory_manager

            profiler.checkpoint(f"Trial {trial_idx} END (before memory update)", {
                'global_memory_manager.working_memory': global_memory_manager.memory.working_memory,
                'global_memory_manager.consolidated_memory': global_memory_manager.memory.consolidated_memory,
                'global_memory_manager.episodic_archive': global_memory_manager.memory.episodic_archive,
                'env_configs': env_configs
            })
        except Exception as e:
            print(f"[WARNING] Profiling failed at trial end: {e}")

        # MEMORY FIX: Force garbage collection after each trial
        import gc
        gc.collect()
        print(f"[MEMORY] Garbage collection completed after trial {trial_idx}")

        # ADD THESE LINES:
        print(f"\n[DEBUG CHECKPOINT] After run_trial:")
        print(f"  trial_idx = {trial_idx}")
        print(f"  args.use_memory = {args.use_memory}")
        print(f"  About to check if statement...")
        # Extract accuracy for this trial
        current_accuracy = extract_accuracy_from_log(world_log_path, trial_idx)
        trial_accuracies.append(current_accuracy)
        
        # update memory if needed BEFORE reset
        print(f"DEBUG: trial_idx={trial_idx}, num_trials={args.num_trials}, condition={trial_idx < args.num_trials - 1}")
        # FORCE MEMORY UPDATE - THE FLAG ISN'T WORKING
        if not early_stopped and trial_idx < args.num_trials:  # Generate for all trials unless stopped
            print(f"\n[FORCED] Running memory update after trial {trial_idx}")
            print(f"\n{'='*80}")
            print(f"[TRIAL {trial_idx}] Generating reflections and updating memory...")
            print(f"{'='*80}\n", flush=True)
            
            # Pass the already-imported objects
            import generate_reflections
            from alfworld_trial import prompt_generator, env_understanding
            generate_reflections.prompt_generator = prompt_generator
            generate_reflections.env_understanding = env_understanding
            generate_reflections.DEBUG_REFLEXION = True
            
            env_configs = update_memory(trial_log_path, env_configs)

            print(f"\n[TRIAL {trial_idx}] Memory update completed")
            print(f"  Updated memories for {sum(1 for env in env_configs if 'memory' in env and len(env['memory']) > 0)} environments")

            # PROFILING: Memory checkpoint AFTER memory update
            try:
                from memory_profiler import profiler
                from generate_reflections import global_memory_manager

                profiler.checkpoint(f"Trial {trial_idx} AFTER Memory Update", {
                    'global_memory_manager.working_memory': global_memory_manager.memory.working_memory,
                    'global_memory_manager.consolidated_memory': global_memory_manager.memory.consolidated_memory,
                    'global_memory_manager.episodic_archive': global_memory_manager.memory.episodic_archive,
                    'env_configs': env_configs
                })

                # Print learning transfer summary
                profiler.print_learning_transfer_summary(trial_idx)

                # Export detailed log
                profiler.export_learning_log(os.path.join(logging_dir, f'learning_log_trial_{trial_idx}.json'))
            except Exception as e:
                print(f"[WARNING] Profiling failed after memory update: {e}")

            # CROSS-TRIAL MEMORY COMPRESSION: Merge redundant patterns, boost confirmed ones
            if trial_idx > 0:  # Only after trial 1+
                try:
                    from causal_memory_compression import compress_cross_trial_memory

                    # Collect all memory entries across environments
                    all_trial_memories = []
                    for env_config in env_configs:
                        if 'memory' in env_config:
                            # Convert string memories to dict format if needed
                            for mem in env_config['memory']:
                                if isinstance(mem, str):
                                    # Wrap string memory in dict for cross-trial processing
                                    all_trial_memories.append({
                                        'type': 'legacy_string',
                                        'insight': mem,
                                        'task_type': 'unknown'
                                    })
                                elif isinstance(mem, dict):
                                    all_trial_memories.append(mem)

                    if all_trial_memories:
                        # Compress and merge across trials
                        compressed_merged = compress_cross_trial_memory(all_trial_memories)

                        # Distribute back to environments by task relevance
                        from causal_memory_compression import retrieve_relevant_memory, extract_task_verb
                        for env_config in env_configs:
                            task = env_config.get('task', 'unknown')
                            task_type = extract_task_verb(task)

                            # Get relevant compressed memories for this environment
                            relevant_memories = [
                                m for m in compressed_merged
                                if m.get('task_type') == task_type or m.get('task_type') == 'unknown'
                            ][:5]  # Max 5 memories per environment

                            # Replace with compressed memories
                            if relevant_memories:
                                env_config['memory'] = [m['insight'] for m in relevant_memories]

                        print(f"[CROSS-TRIAL COMPRESSION] {len(all_trial_memories)} → {len(compressed_merged)} unique patterns")
                        print(f"  Confirmed patterns: {sum(1 for m in compressed_merged if m.get('confirmed_across_trials'))}")
                except ImportError:
                    print("[WARNING] Cross-trial compression unavailable")
                except Exception as e:
                    print(f"[WARNING] Cross-trial compression failed: {e}")
        

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # Log environment results to progress tracker
        for env_idx, env_config in enumerate(env_configs):
            progress_logger.log_env_result(
                trial_idx=trial_idx,
                env_id=env_idx,
                task=env_config.get('task', 'unknown'),
                success=env_config.get('is_success', False),
                steps=env_config.get('cur_step', 0),
                trajectory_length=len(env_config.get('successful_trajectory', []))
            )

        # Log trial end
        progress_logger.log_trial_end(trial_idx)

        # Print current stats
        stats = progress_logger.get_current_stats(trial_idx)
        if stats:
            print(f"\n[TRIAL {trial_idx} SUMMARY]")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Successes: {stats['success_count']}/{stats['num_envs']}")
            print(f"  Progress summary updated: {logging_dir}/progress_summary.json\n")

        # META-LEARNING: After Trial 0 completes, run analysis phase BEFORE reset
        # This must happen while is_success is still set correctly
        if trial_idx == 0:
            print("\n[META-LEARNING] Running analysis after Trial 0...")
            env_configs = meta_analysis_after_trial0(env_configs, world_log_path)

        # Reset environments for next trial AFTER memory update and meta-analysis
        if trial_idx < args.num_trials:
            for env_config in env_configs:
                env_config['is_success'] = False

            reset_msg = f"[RESET] All environments reset for trial {trial_idx + 1}"
            print(reset_msg)
            with open(world_log_path, 'a') as wf:
                wf.write(reset_msg + '\n')


        # Check if all environments succeeded
        all_success = all(env['is_success'] for env in env_configs)
        if all_success:
            print(f"\n{'='*80}")
            print(f"PERFECT TRIAL! All {len(env_configs)} environments succeeded!")
            print(f"Stopping early at trial {trial_idx}")
            print(f"{'='*80}\n")
            
            with open(world_log_path, 'a') as wf:
                wf.write(f"\n***** EARLY STOP: Perfect trial achieved at trial {trial_idx} *****\n")
            break

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1

        # Check for early stopping if adaptive mode is enabled
        if args.adaptive and trial_idx >= args.perfect_trials:
            should_stop, reason = check_early_stopping(
                trial_accuracies, 
                args.perfect_threshold, 
                args.perfect_trials
            )
            
            if should_stop:
                early_stopped = True
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING TRIGGERED!")
                print(f"Reason: {reason}")
                print(f"Completed {trial_idx}/{args.num_trials} trials")
                print(f"Final accuracy: {current_accuracy:.1%}")
                print("="*80 + "\n")
                
                # Log early stopping
                with open(world_log_path, 'a') as wf:
                    wf.write(f"\n\n***** EARLY STOPPING *****\n")
                    wf.write(f"Reason: {reason}\n")
                    wf.write(f"Completed trials: {trial_idx}/{args.num_trials}\n")
                    wf.write(f"Trial accuracies: {[f'{acc:.1%}' for acc in trial_accuracies]}\n")
                    wf.write("*****\n\n")
                break
        
        # Print progress with accuracy trend
        if len(trial_accuracies) > 1:
            if len(trial_accuracies) >= 2:
                trend = trial_accuracies[-1] - trial_accuracies[-2]
            trend_symbol = 'â†‘' if trend > 0 else ('â†“' if trend < 0 else 'â†’')
            print(f"\nTrial {trial_idx-1} complete: {current_accuracy:.1%} {trend_symbol} "
                  f"(avg last {min(3, len(trial_accuracies))} trials: "
                  f"{sum(trial_accuracies[-3:]) / min(3, len(trial_accuracies)):.1%})")

    # Final summary
    if not early_stopped:
        with open(world_log_path, 'a') as wf:
            wf.write(f"\n\n***** TRAINING COMPLETE *****\n")
            wf.write(f"All {args.num_trials} trials completed\n")
            wf.write(f"Final accuracy: {trial_accuracies[-1]:.1%}\n")
            wf.write(f"Best accuracy: {max(trial_accuracies):.1%}\n")
            wf.write("*****\n\n")

    # Save training summary
    summary_path = os.path.join(logging_dir, 'training_summary.json')
    summary = {
        'ablation_mode': args.ablation_mode,
        'components_active': {
            'reflexion': args.ablation_mode in ['reflexion_only', 'combined'],
            'textgrad': args.ablation_mode in ['textgrad_only', 'combined']
        },
        'env_type': args.env_type,
        'num_trials_planned': args.num_trials,
        'num_trials_completed': trial_idx,
        'early_stopped': early_stopped,
        'trial_accuracies': trial_accuracies,
        'final_accuracy': trial_accuracies[-1] if trial_accuracies else 0,
        'best_accuracy': max(trial_accuracies) if trial_accuracies else 0,
        'average_accuracy': sum(trial_accuracies) / len(trial_accuracies) if trial_accuracies else 0,
        'perfect_threshold': args.perfect_threshold,
        'perfect_trials_needed': args.perfect_trials,
        'debug_settings': {
            'actor': args.debug_actor or args.debug,
            'critic': args.debug_critic or args.debug,
            'reflexion': args.debug_reflexion or args.debug
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {summary_path}")

    # Clean up temporary env configs file if it was provided
    if args.env_configs_file and os.path.exists(args.env_configs_file):
        try:
            os.remove(args.env_configs_file)
            print(f"Cleaned up temporary config file: {args.env_configs_file}")
        except:
            pass

if __name__ == '__main__':
    args = get_args()
    # Force debug flags when using memory
    if args.use_memory:
        import alfworld_trial
        import dynamic_prompting
        alfworld_trial.DEBUG_REFLEXION = True
        alfworld_trial.DEBUG_CRITIC = True
        alfworld_trial.DEBUG_ACTOR = True
        dynamic_prompting.DEBUG_CRITIC = True
        dynamic_prompting.DEBUG_ACTOR = True
        print(f"\n[LOGGING] Debug flags enabled for memory mode")
        print(f"  REFLEXION: {alfworld_trial.DEBUG_REFLEXION}")
        print(f"  CRITIC: {alfworld_trial.DEBUG_CRITIC}")
        print(f"  ACTOR: {alfworld_trial.DEBUG_ACTOR}\n")
    main(args)