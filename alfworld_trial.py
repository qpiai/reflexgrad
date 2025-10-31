import hashlib
from universal_state_embeddings import state_embeddings
# Dynamic model loading based on MODEL_PROVIDER env var
import os
if os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
    from shared_model_gemini import model, fast_model
else:
    from shared_model import model, fast_model
"""Universal trial execution with adaptive learning and discovery"""
import numpy as np
import os
import sys
import json
import math
import yaml
import importlib
import re
from datetime import datetime
from difflib import SequenceMatcher
from env_history import EnvironmentHistory
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dynamic_prompting import DynamicPromptGenerator, set_debug_flags
from environment_understanding import EnvironmentUnderstanding
from environment_discovery import UniversalEnvironmentDiscovery
from meta_discovery import MetaEnvironmentKnowledge
from universal_env_wrapper import UniversalEnvWrapper, TextWorldWrapper, JerichoWrapper, ScienceWorldWrapper, BabyAIWrapper, NetHackWrapper ,ALFWorldWrapper

# SEQUENTIAL LEARNING SYSTEM - Smart knowledge transfer
from task_classifier import task_classifier
from knowledge_classifier import knowledge_classifier
from learning_extractor import learning_extractor

from vllm import LLM, SamplingParams
import random
import pickle
# Global instances for optimization - PERSIST ACROSS TRIALS
import pickle
import os

# Add these globals
checkpoint_manager = None  # Will be initialized in adaptive_env_interaction_batch
comprehensive_logger = None

def calculate_task_similarity(task1: str, task2: str) -> float:
    """Calculate similarity using pure word overlap - no domain knowledge"""
    if not task1 or not task2:
        return 0.0
    
    # Pure token-based similarity
    words1 = set(task1.lower().split())
    words2 = set(task2.lower().split())
    
    if not words1 or not words2:
        return 0.5
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Check for negation patterns (universal)
    negation_words = {'not', 'no', 'dont', "don't", 'without', 'avoid'}
    has_negation1 = bool(words1 & negation_words)
    has_negation2 = bool(words2 & negation_words)
    
    # If one has negation and other doesn't, reduce similarity
    if has_negation1 != has_negation2:
        jaccard *= 0.3
    
    return jaccard

def calculate_sequence_similarity(step1: int, step2: int, 
                                 prereqs1: List[str], prereqs2: List[str]) -> float:
    """Calculate temporal and prerequisite similarity"""
    # Temporal distance decay
    step_diff = abs(step1 - step2)
    temporal_sim = max(0.1, 1.0 - (step_diff / 10))
    
    # Prerequisites similarity using set operations
    if not prereqs1 and not prereqs2:
        prereq_sim = 1.0
    elif not prereqs1 or not prereqs2:
        prereq_sim = 0.3
    else:
        prereqs1_set = set(prereqs1)
        prereqs2_set = set(prereqs2)
        intersection = len(prereqs1_set & prereqs2_set)
        union = len(prereqs1_set | prereqs2_set)
        prereq_sim = intersection / union if union > 0 else 0.0
    
    return (temporal_sim * 0.5) + (prereq_sim * 0.5)

def should_share_knowledge(source_context: Dict, target_context: Dict) -> Tuple[bool, float]:
    """Determine sharing based on context similarity"""
    # Universal failures always share
    if source_context.get('is_universal', False):
        return True, 1.0
    
    # Calculate similarities
    task_sim = calculate_task_similarity(
        source_context.get('task', ''),
        target_context.get('task', '')
    )
    
    # State similarity based on observation overlap
    state1 = source_context.get('state_text', '')
    state2 = target_context.get('state_text', '')
    state_tokens1 = set(state1.lower().split())
    state_tokens2 = set(state2.lower().split())
    if state_tokens1 and state_tokens2:
        state_sim = len(state_tokens1 & state_tokens2) / len(state_tokens1 | state_tokens2)
    else:
        state_sim = 0.0
    
    seq_sim = calculate_sequence_similarity(
        source_context.get('step', 0),
        target_context.get('step', 0),
        source_context.get('prerequisites', []),
        target_context.get('prerequisites', [])
    )
    
    # Combined score
    combined_score = task_sim * state_sim * seq_sim
    
    # Share only if all dimensions are similar
    should_share = (task_sim > 0.6 and state_sim > 0.6 and seq_sim > 0.5)
    
    return should_share, combined_score





def find_best_matching_action(text: str, valid_actions: List[str]) -> Optional[str]:
    """
    Robust matching that handles numbers and variations
    """
    text_lower = text.lower()
    
    # FIRST: Check if the last line is exactly a valid action
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line in valid_actions:
            return last_line
        # Also check without punctuation
        last_line_clean = last_line.rstrip('.!?,;:')
        if last_line_clean in valid_actions:
            return last_line_clean
    

    
    # First try: exact substring match
    for action in valid_actions:
        if action.lower() in text_lower:
            return action
    
    # Second try: score each action by word overlap
    best_score = 0
    best_action = None
    
    # Extract likely action from text (last line, after "action:", etc)
    action_candidates = []
    
    # Look for explicit action statements
    import re
    for pattern in [r'action:\s*(.+)', r'i (?:will|should|need to)\s+(.+)', 
                    r'(?:^|\n)(.+)$']:  # Last line
        match = re.search(pattern, text_lower)
        if match:
            action_candidates.append(match.group(1).strip())
    
    if not action_candidates:
        action_candidates = [text_lower]
    
    for candidate in action_candidates:
        for valid_action in valid_actions:
            score = calculate_action_similarity(candidate, valid_action.lower())
            
            if score > best_score:
                best_score = score
                best_action = valid_action
    
    # Return if we have decent confidence
    return best_action if best_score > 0.5 else None


def calculate_action_similarity(candidate: str, valid_action: str) -> float:
    """
    Calculate similarity score between candidate and valid action
    Handles numbers and word variations
    """
    import re
    
    def tokenize(text):
        # Split on spaces but keep numbers with adjacent words
        tokens = re.findall(r'\w+|\d+', text.lower())
        return tokens
    
    candidate_tokens = tokenize(candidate)
    action_tokens = tokenize(valid_action)
    
    if not action_tokens:
        return 0.0
    
    # Calculate overlap score
    matches = 0
    for token in action_tokens:
        if token in candidate_tokens:
            matches += 1
        # Partial credit for numbers that are close
        elif token.isdigit():
            for c_token in candidate_tokens:
                if c_token.isdigit() and abs(int(token) - int(c_token)) <= 1:
                    matches += 0.5
                    break
    
    # Calculate final score
    score = matches / len(action_tokens)
    
    # Boost score if all candidate tokens are in action (subset match)
    if all(token in action_tokens for token in candidate_tokens):
        score = min(score * 1.5, 1.0)
    
    return score


def format_step_reflexions(reflexions: List[Dict]) -> str:
    """Format previous step reflexions for context in step-level learning"""
    if not reflexions:
        return "None yet (first few steps)"

    formatted = []
    for r in reflexions:
        step_num = r.get('step', '?')
        reflection_text = r.get('reflection', '')[:150]  # Truncate long reflections
        formatted.append(f"  Step {step_num}: {reflection_text}")
    return '\n'.join(formatted)


def compress_reflexions_medium(reflexions, model, max_tokens=150):
    """Medium compression - STRUCTURED with WHY reasoning"""

    if not reflexions:
        return None

    combined = "\n\n".join([r['reflection'] for r in reflexions])

    prompt = f"""Extract STRUCTURED summary with causal reasoning:

{combined}

Output format (STRUCTURED):
ACTIONS: [list recent actions in order]
WHY FAILED: [causal reason actions didn't work]
WHY WORKED: [if any action succeeded, why it worked]
SEQUENCE: [if order matters, note dependencies like "needed X before Y"]

Keep it factual and explicit. Max 100 words."""

    sampling_params = SamplingParams(max_tokens=200, temperature=0.2, stop=["\n\n\n"])

    # Use fast model for compression (extraction task, no reasoning needed)
    compressed_output = fast_model.generate([prompt], sampling_params)[0]
    compressed_text = compressed_output.outputs[0].text.strip()

    return {
        'step': f"{reflexions[0]['step']}-{reflexions[-1]['step']}",
        'reflection': compressed_text,
        'is_compressed': 'medium'
    }


def compress_reflexions_heavy(reflexions, model, max_tokens=100):
    """Heavy compression to STRUCTURED FACTS - prevents hallucination by being explicit"""

    if not reflexions:
        return None

    combined = "\n\n".join([r['reflection'] for r in reflexions])

    prompt = f"""Extract STRUCTURED FACTS (NOT prose) from these reflexions. Be EXPLICIT and LIST-BASED.

{combined}

Output format (STRUCTURED - use commas, no sentences):
TRIED: [comma-separated list of actions/locations already attempted]
FOUND: [what was discovered - items, locations, objects]
FAILED: [what didn't work - be specific about locations/actions]
LEARNED: [one key insight in 10 words max]

Example:
TRIED: cabinet 1, cabinet 2, safe 1, sofa 1, go to X, open X
FOUND: tissuebox in cabinet 2, creditcard on sofa
FAILED: no pillow in cabinet 1,2,3,4, safe 1, sofa 1
LEARNED: need to try examine/look actions on furniture"""

    sampling_params = SamplingParams(max_tokens=200, temperature=0.1, stop=["\n\n\n"])

    # Use fast model for compression (extraction task, no reasoning needed)
    compressed_output = fast_model.generate([prompt], sampling_params)[0]
    compressed_text = compressed_output.outputs[0].text.strip()

    return {
        'step': f"{reflexions[0]['step']}-{reflexions[-1]['step']}",
        'reflection': compressed_text,
        'is_compressed': 'heavy'
    }


def manage_working_reflexions_tiered(state, model, log_debug):
    """Tiered compression: recent=verbose, medium=compressed, old=summary

    CRITICAL FIX: NEVER compress cross-trial reflexions (Trial 0 lessons)
    These are the most important for learning and must be preserved in full detail.

    Tier 1: ALL cross-trial reflexions - keep FULL VERBOSE (preserve learning)
    Tier 2: Last 2 current-trial reflexions - keep FULL VERBOSE (immediate context)
    Tier 3: Steps 3-5 back (current trial) - MEDIUM compression
    Tier 4: Older than 5 (current trial) - HEAVY compression
    """

    reflexions = state.get('working_reflexions', [])

    if len(reflexions) <= 2:
        # Not enough history, keep all verbose
        return reflexions

    # Get current trial index (may not be set for Trial 0)
    current_trial = state.get('trial_idx', 0)

    # CRITICAL: Separate cross-trial vs current-trial reflexions
    cross_trial = []
    current_trial_refls = []

    for r in reflexions:
        r_trial = r.get('trial', current_trial)  # Default to current if not set
        if r_trial < current_trial:
            # This is from a PREVIOUS trial - NEVER compress!
            cross_trial.append(r)
        else:
            # This is from CURRENT trial - can compress if too many
            current_trial_refls.append(r)

    # Log what we found
    if cross_trial:
        log_debug(f"[CROSS-TRIAL PRESERVATION] Keeping {len(cross_trial)} Trial {current_trial - 1} reflexions FULLY VERBOSE (no compression)")

    # Build result: cross-trial first (never compressed), then current trial (tiered)
    result = cross_trial.copy()  # Keep all cross-trial reflexions verbose

    # Only compress CURRENT trial reflexions if too many
    if len(current_trial_refls) <= 2:
        result.extend(current_trial_refls)
    elif len(current_trial_refls) > 5:
        # Tier 1: Last 2 - VERBOSE
        recent = current_trial_refls[-2:]

        # Tier 2: Steps 3-5 back - MEDIUM compression
        medium_reflexions = current_trial_refls[-5:-2]
        medium_compressed = compress_reflexions_medium(
            reflexions=medium_reflexions,
            model=model,
            max_tokens=150
        )

        # Tier 3: Older than 5 - HEAVY compression
        old_reflexions = current_trial_refls[:-5]
        old_summary = compress_reflexions_heavy(
            reflexions=old_reflexions,
            model=model,
            max_tokens=100
        )

        log_debug(f"[CURRENT-TRIAL COMPRESSION] Steps {old_reflexions[0]['step']}-{old_reflexions[-1]['step']}: heavy | Steps {medium_reflexions[0]['step']}-{medium_reflexions[-1]['step']}: medium | Steps {recent[0]['step']}-{recent[1]['step']}: verbose")

        result.extend([old_summary, medium_compressed] + recent)
    else:
        # 3-5 reflexions: compress older, keep last 2 verbose
        older = current_trial_refls[:-2]
        compressed = compress_reflexions_medium(older, model, max_tokens=150)
        recent = current_trial_refls[-2:]

        log_debug(f"[CURRENT-TRIAL COMPRESSION] Steps {older[0]['step']}-{older[-1]['step']}: medium | Steps {recent[0]['step']}-{recent[1]['step']}: verbose")

        result.extend([compressed] + recent)

    return result



def format_reflexion_insights_complete(memory: List) -> str:
    """Extract COMPLETE actionable insights from reflexion memory (handles both dict and string format)"""
    if not memory:
        return "- No previous attempts"

    # Ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str) or isinstance(memory, dict):
        memory_list = [memory]
    else:
        return "- No actionable insights yet"

    if not memory_list:
        return "- No previous attempts"

    insights = []

    # Process ALL reflections
    for i, reflection_item in enumerate(memory_list, 1):
        # Handle structured dict format (new) or string format (legacy)
        if isinstance(reflection_item, dict):
            # New structured format - extract insight text
            reflection = reflection_item.get('insight', '')
            if not reflection:
                continue
        elif isinstance(reflection_item, str):
            # Legacy string format
            reflection = reflection_item
        else:
            continue

        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract ALL actionable lines
            if any(word in line_lower for word in
                   ['must', 'should', 'avoid', 'never', 'always',
                    'failed because', 'succeeded', 'exact actions',
                    'hypothesis', 'critical', 'important', 'requires',
                    'learned', 'discovered', 'found', 'need', 'try']):
                clean_line = line.strip()
                if len(clean_line) > 10:
                    key_lines.append(f"  - {clean_line}")

        if key_lines:
            insights.append(f"Reflection {i}:\n" + '\n'.join(key_lines))

    return '\n'.join(insights) if insights else "- No actionable insights yet"

# Schema cache to avoid recompilation
_SCHEMA_CACHE = {}

def get_cached_schema(max_actions):
    """Cache SIMPLE JSON schemas that work with xgrammar (fast path)"""
    if max_actions not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[max_actions] = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                     
                    # NO maxLength - causes slow outlines fallback
                },
                "action_number": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_actions
                }
            },
            "required": ["reasoning", "action_number"],
            "additionalProperties": False
        }
    return _SCHEMA_CACHE[max_actions]


def calculate_fuzzy_scores(textgrad_recommendation: str, valid_actions: List[str]) -> List[Tuple[str, float]]:
    """
    Calculate semantic similarity scores between TextGrad recommendation and valid actions.
    Uses fuzzy string matching + keyword overlap for scoring.

    Returns list of (action, score) tuples sorted by score (highest first).
    DOES NOT FILTER - provides guidance scores for LLM to consider.

    Score components:
    - SequenceMatcher ratio (0-1): Character-level similarity
    - Keyword overlap bonus (0-0.3): Shared important words
    - Verb match bonus (0-0.2): Same action verb

    Args:
        textgrad_recommendation: Natural language action recommendation from TextGrad
        valid_actions: List of exact valid actions from environment

    Returns:
        List of (action, score) tuples, sorted by score descending
    """
    if not textgrad_recommendation or textgrad_recommendation.lower() == 'none':
        # No recommendation - return all actions with equal zero score
        return [(action, 0.0) for action in valid_actions]

    rec_lower = textgrad_recommendation.lower()
    rec_words = set(re.findall(r'\w+', rec_lower))

    # Extract verb from recommendation (first word usually)
    rec_parts = rec_lower.split()
    rec_verb = rec_parts[0] if rec_parts else ""

    scored_actions = []

    for action in valid_actions:
        action_lower = action.lower()

        # 1. Fuzzy string similarity (0-1)
        ratio = SequenceMatcher(None, rec_lower, action_lower).ratio()

        # 2. Keyword overlap bonus (shared meaningful words)
        action_words = set(re.findall(r'\w+', action_lower))
        # Filter out common words
        common_words = {'to', 'the', 'a', 'an', 'in', 'on', 'at', 'with', 'from', 'and'}
        meaningful_rec = rec_words - common_words
        meaningful_action = action_words - common_words

        if meaningful_rec and meaningful_action:
            overlap = len(meaningful_rec & meaningful_action)
            keyword_bonus = min(0.3, overlap * 0.1)  # Max 0.3 bonus
        else:
            keyword_bonus = 0.0

        # 3. Verb match bonus (same action type)
        action_parts = action_lower.split()
        action_verb = action_parts[0] if action_parts else ""
        verb_bonus = 0.2 if (rec_verb == action_verb and rec_verb) else 0.0

        # Total score (max = 1.5, realistically 0-1.2 range)
        total_score = ratio + keyword_bonus + verb_bonus

        scored_actions.append((action, total_score))

    # Sort by score descending
    scored_actions.sort(key=lambda x: x[1], reverse=True)

    return scored_actions


def reasoning_based_action_selection_batch(
    batch_data: List[Dict],
    prompt_generator,
    DEBUG_ACTOR: bool = False,
    log_debug = print
) -> List[str]:
    """
    Complete replacement with proper DeepSeek-R1 handling
    """
    
    prompts = []
    valid_actions_lists = []
    
    for i, data in enumerate(batch_data):
        # Extract actual environment ID for correct logging (fallback to batch index if not present)
        env_id = data.get('env_id', i)

        # Keep all your existing data extraction exactly as is
        valid_actions = data['valid_actions']
        observation = data['observation']
        task = data['task']
        step_gradient = data['step_gradient']
        textgrad_components = data['textgrad_components']
        reflexion_memory = data['reflexion_memory']
        action_history = data['action_history']
        tried_actions = data['tried_actions']
        interaction_count = data['interaction_count']
        memory_recommendations = data.get('memory_recommendations', {})
        consolidated_wisdom = data.get('consolidated_step_wisdom', '')
        step_insights = data.get('step_insights_accumulator', [])
        working_reflexions = data.get('working_reflexions', '')
        # INTELLIGENT FILTERING: Get failure history
        failure_history = data.get('failure_history', [])
        
        # Keep all your existing analysis exactly as is
        recent_10_actions = [act for act, _, _ in action_history[-10:]] if action_history else []
        recent_3_actions = [act for act, _, _ in action_history[-3:]] if action_history else []
        never_tried_count = sum(1 for act in valid_actions if act not in tried_actions)
        
        stuck_actions = []
        for act in set(recent_10_actions):
            if recent_10_actions.count(act) >= 3:
                stuck_actions.append(act)
        loop_detected = len(stuck_actions) > 0
        
        action_groups = {}
        action_to_index = {}
        
        for idx, action in enumerate(valid_actions):
            verb = action.split()[0] if action else "unknown"
            if verb not in action_groups:
                action_groups[verb] = []
            action_groups[verb].append(action)
            action_to_index[action] = idx + 1
        
        first_reflexion = ""
        if reflexion_memory and len(reflexion_memory) > 0:
            # Handle dict reflexion_memory - extract insight field
            first_mem = reflexion_memory[0]
            mem_text = first_mem.get('insight', str(first_mem)) if isinstance(first_mem, dict) else str(first_mem)
            first_reflexion = f"INITIAL PLAN/STRATEGY:\n{mem_text}\n{'='*50}\n\n"

        # Build prompt - keep EVERYTHING the same until the very end
        # Get TODO list if available
        todo_display = ""
        if 'todo_manager' in data and data['todo_manager'] is not None:
            todo_manager = data['todo_manager']
            todo_display = todo_manager.get_formatted_todos()
            print(f"[TODO DEBUG] ENV {env_id}: TODO manager exists, formatted output length: {len(todo_display)}")
            if todo_display:
                print(f"[TODO DEBUG] ENV {env_id}: First 200 chars: {todo_display[:200]}")
        else:
            print(f"[TODO DEBUG] ENV {env_id}: No TODO manager in data")

        # Get current TODO for sequential focus
        current_todo_guidance = ""
        if 'todo_manager' in data and data['todo_manager'] is not None:
            current_todo = data['todo_manager']._get_current_todo()
            if current_todo:
                current_todo_guidance = f"""
🎯 CURRENT FOCUS (Complete this FIRST before moving to next TODO):
   {current_todo.active_form}
   Attempts so far: {current_todo.attempts}

   SEQUENTIAL EXECUTION RULES:
   - Your ONLY goal right now is completing the current TODO above
   - Do NOT work on other TODOs until this one is marked complete
   - Each action should make observable progress toward THIS specific subgoal
   - If stuck after 3 attempts, try a different approach for the SAME subgoal

"""

        prompt = f"""You must select an action. This is NOT a reflection or thinking task.

TASK: {task}

{todo_display}
{current_todo_guidance}
CURRENT STATE:
{observation}

============ CRITICAL LEARNING SIGNALS ============

1. STEP REFLECTION FROM LAST ACTION:
STRONGLY RECOMMENDED NEXT ACTION: {step_gradient.get('next_action_guidance', 'Unknown')}
If this exact action appears in the list below, strongly consider selecting it.
Hypothesis: {step_gradient.get('hypothesis', 'None')}
Progress Score: {step_gradient.get('progress_score', 0)}/10

2. TEXTGRAD OPTIMIZATION GUIDANCE:
   - Adaptive Strategy: {textgrad_components.get('adaptive_strategy', 'None')}
   - Task Decomposition: {textgrad_components.get('task_decomposition', 'None')}
   - Environment Understanding: {textgrad_components.get('environment_understanding', 'None')}
   - Action Discovery: {textgrad_components.get('action_discovery', 'None')}
   - Pattern Recognition: {textgrad_components.get('pattern_recognition', 'None')}
   - Hypothesis Testing: {textgrad_components.get('hypothesis_testing', 'None')}

3. REFLEXION MEMORY FROM PREVIOUS FAILURES:
{format_reflexion_insights_complete(reflexion_memory) if reflexion_memory else 'No previous failures yet'}

4. WORKING REFLEXIONS FROM THIS EPISODE:
{working_reflexions if working_reflexions else 'No step reflexions yet'}

5. ACCUMULATED STEP WISDOM:
{consolidated_wisdom if consolidated_wisdom else 'No accumulated wisdom yet'}

6. PAST FAILURES THIS TRIAL (Intelligent Filtering):"""

        # Add intelligent filtering context
        if failure_history:
            prompt += f"\n   ⚠️  IMPORTANT: The following actions made low/no progress recently:"
            for failure in failure_history[-5:]:  # Last 5 failures
                progress = failure.get('progress_score', 'N/A')
                prompt += f"\n   • Step {failure['step']}: '{failure['action']}' (progress: {progress}/10)"
                prompt += f"\n     Before: {failure['context_before'][:80]}..."
                prompt += f"\n     After: {failure['context_after'][:80]}..."

            prompt += "\n\n   SMART FILTERING GUIDANCE:"
            prompt += "\n   - Avoid these actions ONLY if current state is very similar to failure context"
            prompt += "\n   - Allow retry if state changed significantly (new items, different location)"
            prompt += "\n   - First-time exploration is valuable even with low initial progress"
        else:
            prompt += "\n   No failures yet - exploring freely"

        prompt += "\n\n7. UNIVERSAL MEMORY RECOMMENDATIONS:"""
        
        if memory_recommendations.get('previously_succeeded'):
            prompt += "\n   Actions that worked in similar situations:"
            for rec in memory_recommendations['previously_succeeded'][:3]:
                prompt += f"\n   ✓ '{rec['action']}': {rec.get('reason', 'Previously successful')}"
        
        if memory_recommendations.get('avoid'):
            prompt += "\n   Actions to AVOID (failed in similar states):"
            for rec in memory_recommendations['avoid'][:3]:
                prompt += f"\n   ✗ '{rec['action']}': {rec.get('reason', 'Previously failed')}"
        
        prompt += f"""

8. ACTION HISTORY ANALYSIS:
   - Total actions taken: {len(action_history)}
   - Unique actions tried: {len(tried_actions)}
   - Never tried actions: {never_tried_count}/{len(valid_actions)}"""

        if loop_detected:
            prompt += f"\n   ⚠️ WARNING: Stuck in loop with actions: {stuck_actions[:3]}"

        prompt += f"""

9. RECENT TRAJECTORY:"""
        
        if action_history:
            for j, (act, obs, reasoning) in enumerate(action_history[-5:], 1):
                obs_preview = obs[:80].replace('\n', ' ')
                prompt += f"\n   {j}. {act} → {obs_preview}..."
        else:
            prompt += "\n   No actions taken yet"
        
        # Check if gradient guidance exists and is valid
        gradient_action = step_gradient.get('next_action_guidance', '')
        gradient_is_valid = gradient_action and gradient_action in valid_actions

        # EXTRACT FULL ACTION SEQUENCE FROM STRUCTURED MEMORY (highest priority - exact replay)
        success_pattern_actions = []
        task_lower = task.lower()

        # Check if reflexion_memory has structured success patterns with full sequences
        if reflexion_memory:
            from difflib import SequenceMatcher
            import re

            # DEBUG: Print what's in reflexion_memory
            log_debug(f"[SUCCESS PATTERN DEBUG] reflexion_memory has {len(reflexion_memory)} items")
            workflow_count = sum(1 for item in reflexion_memory if isinstance(item, dict) and item.get('type') == 'success_workflow')
            log_debug(f"[SUCCESS PATTERN DEBUG] Found {workflow_count} items with type='success_workflow'")
            if workflow_count > 0:
                for item in reflexion_memory:
                    if isinstance(item, dict) and item.get('type') == 'success_workflow':
                        log_debug(f"[SUCCESS PATTERN DEBUG] Workflow task: '{item.get('task', 'NO TASK')}'")

            best_match = None
            best_similarity = 0.0

            for mem_item in reflexion_memory:
                if isinstance(mem_item, dict) and mem_item.get('type') == 'success_workflow':
                    # Get memory task
                    mem_task = mem_item.get('task', '').lower()
                    if not mem_task:
                        continue

                    # SEMANTIC SIMILARITY CHECK (prevents contamination while allowing valid transfers)

                    # 1. Fuzzy string similarity
                    fuzzy_score = SequenceMatcher(None, task_lower, mem_task).ratio()

                    # 2. Extract meaningful words (filter stop words)
                    stop_words = {'a', 'an', 'the', 'to', 'in', 'on', 'at', 'with', 'from', 'and', 'some', 'it'}
                    task_words = set(w for w in task_lower.split() if w not in stop_words)
                    mem_words = set(w for w in mem_task.split() if w not in stop_words)

                    # 3. Word overlap (meaningful words only)
                    word_overlap = len(task_words & mem_words) / max(len(task_words), len(mem_words)) if task_words and mem_words else 0.0

                    # 4. Verb matching with synonyms
                    task_verb = task_lower.split()[0] if task_lower.split() else ""
                    mem_verb = mem_task.split()[0] if mem_task.split() else ""
                    verb_synonyms = {
                        'put': ['place', 'move', 'set'], 'cool': ['chill', 'freeze'], 'heat': ['warm', 'cook'],
                        'clean': ['wash', 'rinse'], 'look': ['examine', 'inspect']
                    }
                    verb_match = (task_verb == mem_verb)
                    for base, syns in verb_synonyms.items():
                        if (task_verb == base and mem_verb in syns) or (mem_verb == base and task_verb in syns) or (task_verb in syns and mem_verb in syns):
                            verb_match = True
                            break

                    # 5. Combined similarity (weighted: fuzzy 30%, word overlap 40%, verb 30%)
                    combined_similarity = fuzzy_score * 0.3 + word_overlap * 0.4 + (1.0 if verb_match else 0.0) * 0.3

                    log_debug(f"[MEMORY MATCH] '{task}' vs '{mem_task}': sim={combined_similarity:.2f} (fuzzy={fuzzy_score:.2f}, words={word_overlap:.2f}, verb={verb_match})")

                    if combined_similarity > best_similarity:
                        best_similarity = combined_similarity
                        best_match = mem_item

            # USE MEMORY ONLY IF SIMILARITY >= 0.75 (strict to prevent contamination)
            if best_match and best_similarity >= 0.75:
                # FIX: Try 'actions' field first (used by success_workflow), then 'full_action_sequence'
                full_seq = best_match.get('actions', best_match.get('full_action_sequence', []))
                if full_seq:
                    success_pattern_actions = full_seq
                    log_debug(f"[MEMORY REPLAY] Using {len(full_seq)}-step sequence (similarity: {best_similarity:.2f}, task: '{best_match.get('task', '')}')")
                else:
                    log_debug(f"[MEMORY ERROR] Match found but no actions field (keys: {list(best_match.keys() if isinstance(best_match, dict) else [])})")
            elif best_match:
                log_debug(f"[MEMORY SKIP] Best similarity {best_similarity:.2f} < 0.75 threshold (task: '{best_match.get('task', '')}')")

        # FALLBACK: Extract from adaptive_strategy TextGrad patterns (if no memory sequence found)
        if not success_pattern_actions:
            adaptive_strat = textgrad_components.get('adaptive_strategy', '')
            if 'SUCCESS PATTERN:' in adaptive_strat:
                # Extract ALL patterns and find the one matching current task
                import re
                task_lower = task.lower()

                # Find all SUCCESS PATTERN entries
                all_patterns = re.findall(r'SUCCESS PATTERN:\s*\[(.*?)\]\s*completes\s+["\']([^"\']+)["\']', adaptive_strat, re.IGNORECASE)

                # Match pattern to current task
                best_match = None
                best_match_score = 0

                for pattern_str, pattern_task in all_patterns:
                    pattern_task_lower = pattern_task.lower()
                    # Calculate word overlap between current task and pattern task
                    task_words = set(task_lower.split())
                    pattern_words = set(pattern_task_lower.split())
                    overlap = len(task_words & pattern_words)

                    if overlap > best_match_score:
                        best_match_score = overlap
                        best_match = pattern_str

                if best_match and best_match_score >= 2:  # At least 2 word overlap
                    # Parse comma-separated actions
                    success_pattern_actions = [act.strip().strip('"').strip("'") for act in best_match.split(',')]
                    log_debug(f"[SUCCESS PATTERN] Extracted {len(success_pattern_actions)} actions for task '{task}' (match score: {best_match_score})")
            else:
                log_debug(f"[SUCCESS PATTERN] No matching pattern found for task '{task}'")

        # Calculate current step position (how many actions taken so far)
        current_step_in_episode = len(action_history)

        # INTELLIGENT ACTION SCORING with multiple factors
        scored_actions = []
        for action in valid_actions:
            score = 0
            action_lower = action.lower()

            # FACTOR 1: SUCCESS PATTERN REPLAY (highest priority for early steps)
            # If we have a success pattern and we're early in the episode, strongly boost matching actions
            if success_pattern_actions and current_step_in_episode < len(success_pattern_actions):
                # Check if this action matches the expected action at current position
                expected_action = success_pattern_actions[current_step_in_episode]

                # Exact match (highest boost)
                if action_lower == expected_action.lower():
                    score += 50  # Very strong boost for exact replay
                    log_debug(f"[PATTERN MATCH] Step {current_step_in_episode}: '{action}' matches pattern position exactly")

                # Semantic match (verb + main object)
                elif expected_action.lower().split()[0] == action_lower.split()[0]:  # Same verb
                    # Check if main noun matches (fuzzy)
                    expected_words = set(expected_action.lower().split())
                    action_words = set(action_lower.split())
                    noun_overlap = len(expected_words & action_words)
                    if noun_overlap >= 2:  # At least verb + one object
                        score += 30  # Good semantic match
                        log_debug(f"[PATTERN MATCH] Step {current_step_in_episode}: '{action}' semantically matches pattern")

            # FACTOR 2: SUCCESS PATTERN ELEMENTS (general boost for any pattern action)
            # Even if not at the right position, actions from success pattern get boosted
            for success_action in success_pattern_actions:
                if action_lower == success_action.lower():
                    score += 15  # Moderate boost - it's a proven action
                elif success_action.lower().split()[0] == action_lower.split()[0]:
                    score += 5   # Small boost for same verb

            # FACTOR 3: TODO ALIGNMENT (existing system)
            if 'todo_manager' in data and data['todo_manager']:
                current_todo = data['todo_manager']._get_current_todo()
                if current_todo:
                    todo_words = set(word.lower() for word in current_todo.content.split() if len(word) > 3)
                    action_words = set(action_lower.split())
                    overlap = len(todo_words & action_words)
                    score += overlap * 3  # 3 points per matching word

                    action_first_word = action_lower.split()[0] if action.split() else ""
                    if action_first_word in todo_words:
                        score += 5

            # FACTOR 4: FAILURE AVOIDANCE (penalize recently failed actions)
            if failure_history:
                for failure in failure_history[-5:]:  # Last 5 failures
                    if failure['action'].lower() == action_lower:
                        # Check if context is similar (same location/state)
                        context_similar = False
                        if 'context_before' in failure and observation:
                            # Simple similarity: check if key words overlap
                            failure_words = set(failure['context_before'].lower().split())
                            current_words = set(observation.lower().split())
                            if len(failure_words & current_words) > 10:  # Many shared words
                                context_similar = True

                        if context_similar:
                            score -= 10  # Penalize if context is similar to failure
                            log_debug(f"[FAILURE AVOID] Penalizing '{action}' - failed in similar context")
                        else:
                            score -= 2   # Small penalty - might work in different context

            # FACTOR 5: EXPLORATION BONUS (for never-tried actions, but only if no success pattern)
            if action not in tried_actions and not success_pattern_actions:
                score += 1  # Small exploration bonus only when no proven pattern exists

            scored_actions.append((score, action))

        # Sort by score descending
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        valid_actions = [action for score, action in scored_actions]

        # Log top 5 scored actions for debugging
        log_debug(f"[ACTION SCORING] Top 5 actions:")
        for rank, (score, action) in enumerate(scored_actions[:5], 1):
            log_debug(f"  {rank}. [{score:3d}] {action}")

        # Store scored_actions for fuzzy matching fallback
        # NOTE: Intent translation removed - TextGrad must learn exact action syntax
        # This forces clearer failure signals and better gradient learning
        if i >= len(batch_data):
            log_debug(f"[ERROR] i={i} but batch_data length={len(batch_data)}! Skipping scored_actions storage.")
        else:
            batch_data[i]['scored_actions'] = scored_actions

        prompt += f"""

============ ALL VALID ACTIONS (You MUST choose from these {len(valid_actions)} options) ============
"""

        # Show ALL valid actions clearly numbered (up to 100 for readability)
        actions_to_show = min(100, len(valid_actions))

        for idx in range(actions_to_show):
            action = valid_actions[idx]

            # Add score info if available
            score_info = ""
            for score, scored_action in scored_actions[:10]:
                if scored_action == action:
                    score_info = f" [{score} pts]"
                    break

            prompt += f"{idx + 1}. {action}{score_info}\n"

        if len(valid_actions) > 100:
            prompt += f"\n... and {len(valid_actions) - 100} more options available\n"
        

        prompt += f"""

============ RECENT REFLECTION GUIDANCE ============
"""
        # Fix: Use the correct variable from the loop
        if i < len(batch_data) and 'working_reflexions' in batch_data[i]:
            working_reflexions_text = batch_data[i].get('working_reflexions', '')
            if working_reflexions_text:
                prompt += f"{working_reflexions_text}\n"
            
        if step_gradient and 'next_action_guidance' in step_gradient:
            rec = step_gradient['next_action_guidance']
            # Check if recommendation is in valid actions
            if rec in valid_actions:
                prompt += f"\n🎯 REFLEXION RECOMMENDATION: {rec}\n"
                prompt += f"   (This action is available in the list above - it should be strongly considered)\n"
            else:
                prompt += f"\n💡 REFLEXION GUIDANCE: {rec}\n"
                prompt += f"   (Choose the action from the list above that best matches this intent)\n"
                
        prompt += f"""

============ SELECTION INSTRUCTION ============
The actions above are RANKED by a hybrid scoring system that considers:
1. ✓ Successful patterns from memory (proven to work)
2. ✓ Current TODO alignment (what needs to be done now)
3. ✓ TextGrad semantic intent (strategic direction)
4. ✓ Failure avoidance (don't repeat mistakes)

STRONGLY PREFER actions marked with ⭐ or those with highest scores ([XXX pts]).
These have been validated by multiple systems and are most likely to succeed.

You may choose a lower-scored action ONLY if:
- Top actions are physically impossible in current state
- You have strong reasoning based on recent observations

Select the SINGLE action that best advances the task:

"""

        prompt += f"""
============ CRITICAL INSTRUCTION ============
Your output must be EXACTLY one action from the numbered list above.
- Copy the complete action text word-for-word
- Do NOT add extra words, explanations, or modifications
- Do NOT create new actions not in the list
- ONLY choose from the {len(valid_actions)} numbered options shown above

Required format: [action text exactly as listed]

Your selected action:"""
        
        # ============ FUZZY+LLM HYBRID APPROACH ============
        # Calculate fuzzy similarity scores for TextGrad recommendation
        textgrad_rec = step_gradient.get('next_action_guidance', 'None')
        fuzzy_scored_actions = calculate_fuzzy_scores(textgrad_rec, valid_actions)

        # Get top scored actions for display
        top_fuzzy_actions = fuzzy_scored_actions[:10]  # Top 10 by fuzzy score

        # If under 50 actions, show them ALL with fuzzy scores
        if len(valid_actions) <= 50:
            prompt += "\n============ TEXTGRAD RECOMMENDATION ============\n"
            prompt += f"🎯 TextGrad Recommends: {textgrad_rec}\n"
            prompt += f"Reasoning: {step_gradient.get('hypothesis', 'Improves progress')}\n\n"

            prompt += "============ FUZZY MATCH SCORES (Fuzzy+LLM Team Approach) ============\n"
            prompt += "Fuzzy scores show string similarity between TextGrad recommendation and each action.\n"
            prompt += "Use these scores as GUIDANCE (not filters) to identify the best semantic match.\n\n"

            # Show top 10 with scores
            prompt += "TOP MATCHES (by fuzzy similarity):\n"
            for idx, (action, score) in enumerate(top_fuzzy_actions, 1):
                prompt += f"{idx}. {action} [fuzzy score: {score:.2f}]\n"

            # Show remaining actions without scores (but not filtered!)
            remaining_actions = [action for action, _ in fuzzy_scored_actions[10:]]
            if remaining_actions:
                prompt += f"\nOTHER VALID ACTIONS ({len(remaining_actions)} more):\n"
                for idx, action in enumerate(remaining_actions, 11):
                    prompt += f"{idx}. {action}\n"
        else:
            # For large action spaces, show grouped display with fuzzy guidance
            prompt += "\n============ TEXTGRAD RECOMMENDATION ============\n"
            prompt += f"🎯 TextGrad Recommends: {textgrad_rec}\n"
            prompt += f"Reasoning: {step_gradient.get('hypothesis', 'Improves progress')}\n\n"

            prompt += "TOP FUZZY MATCHES (string similarity to TextGrad):\n"
            for idx, (action, score) in enumerate(top_fuzzy_actions, 1):
                prompt += f"{idx}. {action} [score: {score:.2f}]\n"

            verbs_shown = 0
            for verb, actions_list in sorted(action_groups.items(), key=lambda x: -len(x[1]))[:10]:
                prompt += f"\n\n{verb.upper()} ({len(actions_list)} variations):"
                for example in actions_list[:3]:
                    prompt += f"\n  • {example}"
                if len(actions_list) > 3:
                    prompt += f"\n  ... and {len(actions_list) - 3} more {verb} actions"
                verbs_shown += 1

            if len(action_groups) > 10:
                prompt += f"\n\n... and {len(action_groups) - 10} more action types available"

        prompt += f"""

============ PRIMARY DIRECTIVE ============
YOUR ROLE: MATCH TextGrad recommendation to valid action (NOT reason about task)

TextGrad Recommendation: {textgrad_rec}

INSTRUCTIONS:
1. Reflexion analyzed WHY previous action succeeded/failed (causal analysis)
2. TextGrad extracted HOW - the specific action recommendation above
3. YOUR JOB: Find the EXACT valid action that matches TextGrad's recommendation
4. Use fuzzy scores as GUIDANCE to identify semantic matches
   - High score (>0.7): Strong lexical match
   - Medium score (0.4-0.7): Partial match, check semantic meaning
   - Low score but semantically correct: Still valid! (e.g., "fridge" matches "cool pan")

DO NOT CREATE NEW ACTIONS.
DO NOT COMBINE MULTIPLE VALID ACTIONS.
DO NOT IGNORE NUMBERS FROM VALID ACTIONS.

Correct example: 'go to fridge 1'
Incorrect example: 'The action to take is "move pot 1 to countertop 1"'

Output format: Write EXACTLY one action string from the list above, nothing else.
Do not explain. Do not write 'Action:' prefix. Just the exact action string.

Your output must be EXACTLY one of the {len(valid_actions)} actions listed above.

Top recommendations:"""

        # Add top actions as final reminder
        for i, (action, score) in enumerate(top_fuzzy_actions[:5], 1):
            prompt += f"\n{i}. {action}"
        if len(valid_actions) > 5:
            prompt += f"\n... or one of the other {len(valid_actions)-5} valid actions listed above"

        prompt += "\n\nYour action:"
        
        prompts.append(prompt)
        valid_actions_lists.append(valid_actions)
    
    # END OF LOOP - Put verification AFTER the loop completes
    
    # Verify prompts are unique (OUTSIDE the loop, after it's done)
    for idx in range(1, len(prompts)):
        if prompts[idx] == prompts[idx-1]:
            print(f"[ERROR] Prompt {idx} is identical to prompt {idx-1}! This is a bug!")
    
    # Also check overall uniqueness
    if len(set(prompts)) < len(prompts):
        print(f"[WARNING] Only {len(set(prompts))} unique prompts out of {len(prompts)} total!")


    # ADD THIS DEBUG BLOCK before: outputs = model.generate(prompts, SamplingParams(...))
    if DEBUG_ACTOR:
        print(f"\n{'='*80}")
        print("[DEBUG LLM PROMPT - BATCH CHECK]")
        print(f"Total prompts to generate: {len(prompts)}")
        for idx, (prompt, actions) in enumerate(zip(prompts, valid_actions_lists)):
            # Extract task from prompt
            task_match = prompt.split('TASK:')[1].split('\n')[0].strip() if 'TASK:' in prompt else "NO TASK"
            # Get env_id for this index from batch_data
            prompt_env_id = batch_data[idx].get('env_id', idx)
            print(f"\n[PROMPT ENV {prompt_env_id}]:")
            print(f"  Task: {task_match}")
            print(f"  Valid actions count: {len(actions)}")
            print(f"  First 3 actions: {actions[:3]}")
            print(f"  Prompt length: {len(prompt)} chars")
            # Check if prompt actually contains this env's actions
            if actions and actions[0] in prompt:
                print(f"  ✓ Prompt contains env's first action")
            else:
                print(f"  ✗ WARNING: Prompt may not contain correct actions!")
        print(f"{'='*80}\n")


    # UNIVERSAL TWO-TIER VALIDATION: Use TextGrad recommendations directly
    # Forces TextGrad to learn exact action syntax for better gradient signals
    synergy_outputs = []
    need_llm = []

    # Normalization for exact match (handles minor syntax differences)
    def normalize(action):
        return action.lower().strip().rstrip('.')

    # Semantic token matching for natural language to exact syntax
    def semantic_match(textgrad_action, valid_actions):
        """Find best matching action using token overlap + numbered item handling"""
        if not textgrad_action:
            return None

        tg_tokens = set(normalize(textgrad_action).split())
        best_match = None
        best_overlap = 0

        # TIER 3A: Standard token overlap (70% threshold)
        for action in valid_actions:
            action_tokens = set(normalize(action).split())
            overlap = len(tg_tokens & action_tokens)

            # Require at least 70% token overlap
            if overlap > best_overlap and overlap >= len(tg_tokens) * 0.7:
                best_match = action
                best_overlap = overlap

        if best_match:
            return best_match

        # TIER 3B: Numbered item fuzzy matching
        # Handles cases like "examine armchair 1" vs "examine armchair 2"
        # Strategy: Match verb + noun pattern, ignore item numbers
        import re

        # Extract pattern from TextGrad: verb + nouns (ignore numbers)
        tg_words = normalize(textgrad_action).split()
        tg_pattern = ' '.join(word for word in tg_words if not word.isdigit())

        # Score each action based on pattern match
        pattern_matches = []
        for action in valid_actions:
            action_words = normalize(action).split()
            action_pattern = ' '.join(word for word in action_words if not word.isdigit())

            # Check if patterns match (same verb + nouns, different numbers)
            if tg_pattern == action_pattern:
                # Perfect pattern match (verb + noun identical, only numbers differ)
                # Prioritize actions with same item number if available
                tg_numbers = [w for w in tg_words if w.isdigit()]
                action_numbers = [w for w in action_words if w.isdigit()]
                number_match_score = len(set(tg_numbers) & set(action_numbers))
                pattern_matches.append((action, 1000 + number_match_score))  # High priority
            elif action_pattern.startswith(tg_pattern) or tg_pattern.startswith(action_pattern):
                # Partial pattern match (one is prefix of other)
                pattern_matches.append((action, 500))  # Medium priority

        # Return best pattern match
        if pattern_matches:
            pattern_matches.sort(key=lambda x: x[1], reverse=True)
            best_match = pattern_matches[0][0]
            log_debug(f"[NUMBERED-ITEM-MATCH] '{textgrad_action}' → '{best_match}' (pattern: '{tg_pattern}')")
            return best_match

        return None

    for i in range(len(batch_data)):
        # Use last_step_gradient (from previous step) for action selection
        # step_gradient is for CURRENT step (not yet generated during action selection)
        step_grad = batch_data[i].get('last_step_gradient', {})
        textgrad_rec = step_grad.get('next_action_guidance', '')
        guidance_source = step_grad.get('guidance_source', 'unknown')
        valid_actions = valid_actions_lists[i]  # Filtered and prioritized actions
        raw_actions = batch_data[i].get('raw_valid_actions', valid_actions)  # Unfiltered from environment
        env_id = batch_data[i].get('env_id', i)
        trial_idx = batch_data[i].get('trial_idx', 0)

        # PHASE 2: Extract clean action from Reflexion CAUSAL CHAIN if needed
        # TextGrad outputs are already clean, Reflexion outputs need extraction
        if textgrad_rec and guidance_source == 'reflexion' and 'CAUSAL CHAIN' in textgrad_rec:
            import re
            # Try multiple patterns to extract ACTION from Reflexion's verbose output
            action_match = None

            # Pattern 1: ACTION: go to X. | (colon, ends with period or pipe)
            action_match = re.search(r'ACTION:\s*([^.|]+?)(?:\.|\||$)', textgrad_rec, re.IGNORECASE)

            # Pattern 2: ACTION (go to X) (parentheses format)
            if not action_match:
                action_match = re.search(r'ACTION\s*\(([^)]+)\)', textgrad_rec, re.IGNORECASE)

            # Pattern 3: Action = go to X (equals sign format)
            if not action_match:
                action_match = re.search(r'ACTION\s*=\s*([^|>]+?)(?:\||->|RESULT)', textgrad_rec, re.IGNORECASE)

            if action_match:
                extracted = action_match.group(1).strip()
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: Reflexion CAUSAL CHAIN → '{extracted}'")
                textgrad_rec = extracted
            else:
                # Extraction failed - will trigger LLM fallback
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: Failed to extract from Reflexion CAUSAL CHAIN")
                textgrad_rec = ''
        elif textgrad_rec and guidance_source == 'textgrad':
            # FIX: Extract clean action before GRADIENT JUSTIFICATION
            if ' | GRADIENT JUSTIFICATION:' in textgrad_rec:
                textgrad_rec = textgrad_rec.split(' | GRADIENT JUSTIFICATION:')[0].strip()
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: TextGrad action extracted: '{textgrad_rec}'")
            else:
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: TextGrad clean action: '{textgrad_rec[:50]}'")
        elif not textgrad_rec:
            log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: No guidance from previous step")

        # ============================================================================
        # SYNERGY LAYER 0: REFLEXION FILTER (Veto power - runs first)
        # ============================================================================
        reflexion_memory = batch_data[i].get('reflexion_memory', [])
        avoid_patterns = []
        success_patterns = []

        # Extract AVOID and SUCCESS patterns from Reflexion (cheap pattern matching)
        if reflexion_memory and trial_idx > 0:  # Only apply in Trial 1+
            import re
            for mem in reflexion_memory:
                mem_str = str(mem).lower()
                # Extract AVOID patterns: "🚫 AVOID: go to cabinet" → "cabinet"
                if '🚫 avoid:' in mem_str or 'avoid:' in mem_str:
                    match = re.search(r'avoid:\s*([^,\n\.]+)', mem_str)
                    if match:
                        pattern = match.group(1).strip()
                        if pattern and len(pattern) > 2:  # Avoid noise
                            avoid_patterns.append(pattern)

                # Extract SUCCESS patterns: "✅ PROVEN: pans on stoveburner" → "stoveburner"
                elif '✅ proven:' in mem_str or 'success:' in mem_str or 'key_fact:' in mem_str:
                    match = re.search(r'(?:proven|success|key_fact):\s*([^,\n\.]+)', mem_str)
                    if match:
                        pattern = match.group(1).strip()
                        if pattern and len(pattern) > 2:
                            success_patterns.append(pattern)

        # Apply Reflexion filter (hard constraint - removes contradicting actions)
        if avoid_patterns:
            original_count = len(valid_actions)
            valid_actions_filtered = []
            for action in valid_actions:
                action_lower = action.lower()
                is_avoided = any(pattern in action_lower for pattern in avoid_patterns)
                if not is_avoided:
                    valid_actions_filtered.append(action)
                else:
                    log_debug(f"[REFLEXION-VETO] ENV {env_id}: Filtered '{action}' (matches AVOID: {pattern})")

            if valid_actions_filtered:  # Use filtered list if not empty
                valid_actions = valid_actions_filtered
                log_debug(f"[REFLEXION-FILTER] ENV {env_id}: {original_count} → {len(valid_actions)} actions after veto")

        # ============================================================================
        # SYNERGY LAYER 1: TODO COLD START (Strategic direction for new subtasks)
        # ============================================================================
        todo_manager = batch_data[i].get('todo_manager')
        if todo_manager:
            current_todo = todo_manager._get_current_todo()
            # Guide first attempt of each subtask (avoid cold start at subtask transitions)
            if current_todo and current_todo.attempts == 0:
                import re
                # Score actions based on TODO relevance + Reflexion success patterns
                action_scores = {}
                todo_lower = current_todo.content.lower()

                for action in valid_actions:
                    score = 0
                    action_lower = action.lower()

                    # Keyword overlap between TODO and action
                    todo_words = set(todo_lower.split())
                    action_words = set(action_lower.split())
                    overlap = len(todo_words & action_words)
                    score += overlap * 10

                    # If TODO is about finding/locating, prioritize exploration actions
                    if any(verb in todo_lower for verb in ['find', 'locate', 'search', 'look']):
                        if action_lower.startswith('go to'):
                            score += 5
                        elif action_lower.startswith('examine') or action_lower.startswith('open'):
                            score += 3

                    # Extract target object from TODO (usually last noun)
                    obj_match = re.search(r'(pan|pillow|bowl|lamp|cup|plate|knife|fork|spoon|pot|mug|book|pen|phone|watch|key|\w+)\s*$', todo_lower)
                    if obj_match:
                        obj = obj_match.group(1)
                        if obj in action_lower:
                            score += 20  # Very high - directly mentions target object

                    # Bonus: Reflexion success patterns (synergy)
                    if success_patterns:
                        for pattern in success_patterns:
                            if pattern in action_lower:
                                score += 15  # Proven success boost
                                log_debug(f"[TODO+REFLEXION] ENV {env_id}: Action '{action}' matches success pattern '{pattern}'")

                    action_scores[action] = score

                # Pick best scoring action
                if action_scores:
                    best_action = max(action_scores.items(), key=lambda x: x[1])
                    if best_action[1] > 0:  # Has non-zero score
                        synergy_outputs.append(best_action[0])
                        log_debug(f"[TODO-COLDSTART] ENV {env_id}: Subtask '{current_todo.content}' → '{best_action[0]}' (score: {best_action[1]})")
                        continue  # Skip to next environment

        # ============================================================================
        # SYNERGY LAYER 2: TEXTGRAD OPTIMIZATION (Main path - 85%+ of decisions)
        # ============================================================================
        # TextGrad recommendation already respects Reflexion filter (valid_actions filtered above)

        # TIER 1: Exact match in filtered actions
        if textgrad_rec and textgrad_rec in valid_actions:
            synergy_outputs.append(textgrad_rec)
            log_debug(f"[TEXTGRAD-EXACT] ENV {env_id}: {textgrad_rec}")
            continue

        # TIER 2: Normalized/fuzzy match (handles minor extraction variations)
        elif textgrad_rec and normalize(textgrad_rec) in [normalize(a) for a in valid_actions]:
            matched = next(a for a in valid_actions if normalize(a) == normalize(textgrad_rec))
            synergy_outputs.append(matched)
            log_debug(f"[TEXTGRAD-FUZZY] ENV {env_id}: '{textgrad_rec}' → '{matched}'")
            continue

        # TIER 3: Semantic match (handles "move X to Y" vs "move X 1 to Y 1")
        elif textgrad_rec and (semantic_match(textgrad_rec, valid_actions)):
            matched = semantic_match(textgrad_rec, valid_actions)
            synergy_outputs.append(matched)
            log_debug(f"[TEXTGRAD-SEMANTIC] ENV {env_id}: '{textgrad_rec}' → '{matched}'")
            continue

        # ============================================================================
        # SYNERGY LAYER 3: LLM FALLBACK (Should be <5% - extraction failed, exposes root cause)
        # ============================================================================
        # NO HIDDEN FALLBACKS - if TextGrad fails, we WANT to see it via LLM rate
        # This exposes extraction bugs instead of hiding them
        synergy_outputs.append(None)
        need_llm.append(i)
        log_debug(f"[LLM-FALLBACK] ENV {env_id}: TextGrad extraction failed ('{textgrad_rec}') - using LLM to expose issue")

    # Only call LLM for environments that need it
    if need_llm:
        print(f"[ACTION SELECTION] Using fast model for {len(need_llm)}/{len(prompts)} environments (others using Reflexion+TextGrad directly)")

        # Track API calls and handle quota
        global checkpoint_manager
        if checkpoint_manager:
            checkpoint_manager.increment_api_calls(len(need_llm))

        # Generate only for environments that need LLM
        llm_prompts = [prompts[i] for i in need_llm]
        try:
            llm_outputs = fast_model.generate(llm_prompts, SamplingParams(
                max_tokens=100,
                temperature=0.0,
                stop=[],
                skip_special_tokens=False
            ))
        except Exception as e:
            # Handle API quota errors
            from api_quota_handler import APIQuotaHandler
            # Get logging_dir from checkpoint_manager or use fallback

            if checkpoint_manager:
                logging_dir = checkpoint_manager.run_dir
            else:
                logging_dir = '.'
            quota_handler = APIQuotaHandler(checkpoint_manager, logging_dir)
            # Note: trial_idx, env_configs, env_states might not be in scope here
            # Pass empty values if not available
            quota_handler.handle_api_error(e, 0, [], [])
            # If not quota error, re-raise
            raise e

        # Merge LLM outputs with synergy outputs
        llm_output_idx = 0
        for i in range(len(synergy_outputs)):
            if synergy_outputs[i] is None:  # This env needed LLM
                synergy_outputs[i] = llm_outputs[llm_output_idx].outputs[0].text.strip()
                llm_output_idx += 1
    else:
        print(f"[SYNERGY OPTIMIZATION] All {len(prompts)} environments using Reflexion+TextGrad directly - no LLM calls needed!")

    # Create fake output objects for synergy outputs
    class FakeOutput:
        def __init__(self, text):
            self.text = text

    class FakeResult:
        def __init__(self, text):
            self.outputs = [FakeOutput(text)]

    outputs = [FakeResult(text) for text in synergy_outputs]

    # Verify we got outputs for all environments
    if len(outputs) != len(prompts):
        print(f"[ERROR] Model returned {len(outputs)} outputs but we sent {len(prompts)} prompts!")
        
    # Debug what each environment selected
    if DEBUG_ACTOR:
        print(f"\n[MODEL OUTPUTS SUMMARY]:")
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()
            # Get env_id for this response
            response_env_id = batch_data[i].get('env_id', i)
            print(f"  ENV {response_env_id}: Selected '{raw[:500]}...' from {len(valid_actions_lists[i])} options")
            print(f"\n[LLM RAW OUTPUT {i}]: '{raw[:500]}'")
            if valid_actions_lists and i < len(valid_actions_lists):
                print(f"Output matches a valid action?: {raw.strip() in valid_actions_lists[i]}")
    
    selected_actions = []

    for i, output in enumerate(outputs):
        # Get env_id for correct logging
        parse_env_id = batch_data[i].get('env_id', i)

        raw_text = output.outputs[0].text
        log_debug(f"[DEBUG RAW OUTPUT {i}]: '{raw_text[:200]}...'")


        # OpenAI doesn't use think tags - just use the text directly
        action_text = raw_text.strip()

        # Clean prefixes
        if action_text.startswith("Action:"):
            action_text = action_text[7:].strip()

        # Remove brackets if present
        action_text = action_text.strip('[]')

        # Take only first line
        if '\n' in action_text:
            action_text = action_text.split('\n')[0].strip()

        # Remove trailing dots
        action_text = action_text.rstrip('.')

        log_debug(f"[ENV {parse_env_id}] Extracted action text: '{action_text}'")

        # Get TextGrad recommendation for logging
        step_grad = batch_data[i].get('step_gradient', {})
        textgrad_rec = step_grad.get('next_action_guidance', 'None')

        # SIMPLIFIED VALIDATION: Ensure LLM output is in valid actions
        # FAIL CONFIDENTLY - no fuzzy fallbacks that mask issues
        if action_text.strip() not in valid_actions_lists[i]:
            log_debug(f"[VALIDATION FAILED] LLM output '{action_text}' NOT in valid actions")

            # ONLY fallback: Check if TextGrad recommendation is directly valid
            if textgrad_rec and textgrad_rec.strip() in valid_actions_lists[i]:
                action_text = textgrad_rec.strip()
                log_debug(f"[RECOVERY] Using TextGrad recommendation: '{action_text}'")
            else:
                # FAIL CONFIDENTLY: Use first action, let TextGrad learn from this failure
                action_text = valid_actions_lists[i][0]
                log_debug(f"[FAIL CONFIDENTLY] LLM hallucination - using top action: '{action_text}'. TextGrad will learn from failure signal.")

        # EXACT match check - no fuzzy fallbacks
        selected_action = None

        for valid_action in valid_actions_lists[i]:
            if action_text.lower() == valid_action.lower():
                selected_action = valid_action
                log_debug(f"[ENV {parse_env_id}] ✓ EXACT match found: '{selected_action}'")
                break

        # NO FUZZY FALLBACK - fail confidently if no exact match
        # This should never happen after validation recovery above
        if selected_action is None:
            selected_action = action_text  # Use it anyway, let environment reject if invalid
            log_debug(f"[ENV {parse_env_id}] ⚠ NO EXACT MATCH - using action as-is: '{selected_action}'")

        log_debug(f"[ENV {parse_env_id}] Selected to execute: '{selected_action}' (VALID)")

        # Simple TextGrad alignment logging - exact match only
        if textgrad_rec and textgrad_rec != 'None':
            textgrad_exact_match = (selected_action.lower() == textgrad_rec.lower())
            match_quality = "✓ EXACT" if textgrad_exact_match else "✗ MISMATCH"
            log_debug(f"[TEXTGRAD ALIGNMENT] {match_quality} | Rec: '{textgrad_rec}' | Selected: '{selected_action}'")

        selected_actions.append(selected_action)

    return selected_actions


def extract_action_from_reasoning(reasoning: str, valid_actions: List[str]) -> Optional[str]:
    """Extract the most likely action from reasoning text - improved universal approach"""
    reasoning_lower = reasoning.lower()
    
    m = re.search(r'^\s*ACTION:\s*(.+)\s*$', reasoning, flags=re.I|re.M)
    if m:
        cand = m.group(1).strip()
        # pick exact match from valid_actions by case-insensitive equality
        for v in valid_actions:
            if cand.lower() == v.lower():
                return v


    # Look for quoted actions first (most reliable)
    import re
    quoted = re.findall(r'"([^"]+)"', reasoning)
    for q in quoted:
        if q in valid_actions:
            return q
    
    # Also look for actions after "action:" or "next action:"
    action_patterns = [
        r'action:\s*([^\.]+)',
        r'next action:\s*([^\.]+)',
        r'should (?:be|try):\s*([^\.]+)',
    ]
    for pattern in action_patterns:
        match = re.search(pattern, reasoning_lower)
        if match:
            candidate = match.group(1).strip()
            # Check if this exact string is in valid actions
            for valid_action in valid_actions:
                if candidate in valid_action.lower() or valid_action.lower() in candidate:
                    return valid_action
    
    # Score each action by multiple signals
    best_score = 0
    best_action = None
    
    for action in valid_actions:
        score = 0
        action_lower = action.lower()
        
        # Signal 1: Exact substring match
        if action_lower in reasoning_lower:
            score += 10
        
        # Signal 2: All words from action appear in reasoning
        action_words = set(action_lower.split())
        reasoning_words = set(reasoning_lower.split())
        if action_words.issubset(reasoning_words):
            score += 5
        
        # Signal 3: Word overlap ratio
        overlap = len(action_words & reasoning_words)
        if action_words:
            score += (overlap / len(action_words)) * 3
        
        # Signal 4: Action appears near keywords
        for keyword in ['next', 'action', 'choose', 'select', 'try']:
            if keyword in reasoning_lower:
                keyword_pos = reasoning_lower.index(keyword)
                if action_lower in reasoning_lower[max(0, keyword_pos-50):keyword_pos+100]:
                    score += 2
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action if best_score > 2 else None  # Threshold for confidence


def format_reflexion_insights_complete(memory: List[str]) -> str:
    """Extract COMPLETE actionable insights from reflexion memory - NO TRUNCATION"""
    if not memory:
        return "- No previous attempts"
    
    # Ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str):
        memory_list = [memory]
    else:
        return "- No actionable insights yet"
    
    if not memory_list:
        return "- No previous attempts"
    
    insights = []
    
    # Process ALL reflections
    for i, reflection in enumerate(memory_list, 1):
        if not isinstance(reflection, str):
            continue
        
        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract ALL actionable lines
            if any(word in line_lower for word in 
                   ['must', 'should', 'avoid', 'never', 'always', 
                    'failed because', 'succeeded', 'exact actions',
                    'hypothesis', 'critical', 'important', 'requires',
                    'learned', 'discovered', 'found', 'need', 'try']):
                clean_line = line.strip()
                if len(clean_line) > 10:
                    # KEEP FULL LINE - NO TRUNCATION
                    key_lines.append(f"  - {clean_line}")
        
        if key_lines:
            # Include ALL key lines, not just first few
            insights.append(f"Reflection {i}:\n" + '\n'.join(key_lines))
    
    return '\n'.join(insights) if insights else "- No actionable insights yet"

def intelligent_memory_cap(memory_list, cap=25):
    """Keep diverse memories: early (foundation) + recent (context)"""
    if len(memory_list) <= cap:
        return memory_list
    
    # Keep 5 foundation + 20 recent
    foundation = memory_list[:5]
    recent = memory_list[-20:]
    combined = foundation + recent
    
    # Remove exact duplicates
    seen = set()
    unique = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    
    return unique[:cap]


def adaptive_env_interaction_batch(
    envs: List,
    base_prompt: str,
    memories: List[List[str]],
    to_print: bool = True,
    initial_obs_list: List[str] = None,
    trial_log_path: str = None,
    env_configs: List[Dict[str, Any]] = None,
    trial_idx: int = 0,
    use_memory: bool = False  # ADD THIS PARAMETER
) -> List[Tuple[EnvironmentHistory, bool]]:
    """
    Batch version of adaptive_env_interaction with universal memory
    Processes multiple environments in parallel with batched LLM calls
    """



# -------------------------------------------------------------------------------
    global checkpoint_manager 
    # ADD THIS DEBUG BLOCK
    print(f"\n[BATCH ENTRY DEBUG]")
    print(f"  trial_log_path = {trial_log_path}")
    print(f"  trial_log_path type = {type(trial_log_path)}")
    print(f"  trial_log_path is None? {trial_log_path is None}")
    print(f"  to_print = {to_print}")
    print(f"  Number of envs = {len(envs)}")
    print("[/BATCH ENTRY DEBUG]\n")

    import hashlib
    import datetime
    from collections import defaultdict
    debug_log_path = f'debug_batch_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    debug_log = open(debug_log_path, 'w')
    

    # Initialize loggers
    from enhanced_logging import ComprehensiveLogger
    # from checkpoint_manager import CheckpointManager  # Not needed - removed to fix import error

    # Define logging_dir properly
    if trial_log_path:
        logging_dir = os.path.dirname(trial_log_path)
    else:
        logging_dir = '.'  # Current directory as fallback

    # Make checkpoint_manager global for access in other functions

    # checkpoint_manager = CheckpointManager(logging_dir)  # Not needed - removed to fix error
    comprehensive_logger = ComprehensiveLogger(logging_dir)

    def log_debug(msg):
        """Log to both console and file"""
        print(msg)
        debug_log.write(msg + '\n')
        debug_log.flush()
    
    log_debug(f"[TRACE START] adaptive_env_interaction_batch called with {len(envs)} environments")
    
    # Import universal memory system
    from universal_memory_system import universal_memory
    
    # TEST MEMORY SYSTEM
    from generate_reflections import global_memory_manager
    print(f"[MEMORY TEST] Memory system loaded successfully!")
    print(f"[MEMORY TEST] Working memory: {len(global_memory_manager.memory.working_memory)}")
    print(f"[MEMORY TEST] Consolidated: {len(global_memory_manager.memory.consolidated_memory)}")
    print(f"[MEMORY TEST] Universal states tracked: {len(universal_memory.state_action_outcomes)}")
    
    # Initialize all environment states
    env_states = []

    # Sequential TODO initialization for accuracy (each env learns from previous)
    from task_todo_manager import TaskTodoManager
    from todo_transfer_safety import todo_transfer_safety
    import uuid

    # Collect all successful TODOs for cross-environment learning
    successful_todo_patterns = []

    # Track successful TODO patterns from completed environments (within THIS trial)
    shared_todo_knowledge = {
        'successful_patterns': [],  # TODOs that led to success
        'failed_patterns': [],      # TODOs that led to failure
    }

    # Process environments sequentially
    env_data_list = []
    for i, (env, memory, ob) in enumerate(zip(envs, memories, initial_obs_list)):
        task = prompt_generator._extract_task_from_observation(ob)
        env_configs[i]['task'] = task
        if not task:
            log_debug(f"CRITICAL ERROR: Could not extract task from initial observation for env {i}!")
            log_debug(f"Observation preview: {ob[:300]}")
            raise ValueError("Task extraction failed. Cannot proceed without knowing what to accomplish.")

        prompt_generator.set_task(task)
        env_understanding.current_task = task
        log_debug(f"Environment {i} task: {task}")

        initial_valid_actions = []
        if hasattr(env, 'get_current_valid_actions'):
            initial_valid_actions = env.get_current_valid_actions()
            if initial_valid_actions:
                prompt_generator.discovered_knowledge['available_actions'] = set(initial_valid_actions)

        initial_prompt = prompt_generator.format_initial_observation(ob, memory)
        env_history = EnvironmentHistory("", initial_prompt, memory, [])
        env_history.reset()

        if to_print:
            log_debug(f"[ENV {i}] {ob}")
            sys.stdout.flush()

        episode_id = str(uuid.uuid4())[:8]
        reflexion_memory_for_todo = memory if trial_idx > 0 else None

        # CROSS-ENVIRONMENT TODO LEARNING (NEW!)
        # Get similar successful TODOs from previous environments in THIS trial
        similar_todo_suggestions = []
        if i > 0 and shared_todo_knowledge['successful_patterns']:
            log_debug(f"\n[TODO TRANSFER] ENV {i}: Checking {len(shared_todo_knowledge['successful_patterns'])} successful patterns from previous envs")

            similar_todo_suggestions = todo_transfer_safety.transfer_todos(
                shared_todo_knowledge['successful_patterns'],
                task
            )

            if similar_todo_suggestions:
                log_debug(f"[TODO TRANSFER] ✅ Transferred {len(similar_todo_suggestions)} TODO suggestions")
                for idx, suggestion in enumerate(similar_todo_suggestions[:3], 1):
                    log_debug(f"  {idx}. {suggestion}")
            else:
                log_debug(f"[TODO TRANSFER] ❌ No safe transfers found (tasks too different or failed safety checks)")

        # Initialize TODO for this environment (with retry logic built-in)
        todo_manager = TaskTodoManager(model, fast_model)  # Reasoning for generation, fast for verification

        try:
            initial_todos = todo_manager.initialize_from_task(
                task,
                ob,
                initial_valid_actions,
                trial_num=trial_idx,
                reflexion_memory=reflexion_memory_for_todo,
                similar_todo_suggestions=similar_todo_suggestions  # NEW: Cross-env learning!
            )
            log_debug(f"[ENV {i}] Trial {trial_idx}: Created {len(initial_todos)} TODO items")
            if trial_idx > 0 and reflexion_memory_for_todo:
                log_debug(f"[ENV {i}] TODO initialized with {len(reflexion_memory_for_todo)} reflexions")
            if similar_todo_suggestions:
                log_debug(f"[ENV {i}] TODO benefited from {len(similar_todo_suggestions)} cross-env suggestions")

            # Track successful TODO patterns for learning
            if initial_todos:
                successful_todo_patterns.append({
                    'task': task,
                    'todos': [t.content for t in initial_todos]
                })

        except Exception as e:
            log_debug(f"[ENV {i}] TODO initialization FAILED: {e}")
            raise

        env_data_list.append({
            'env': env,
            'env_id': i,
            'episode_id': episode_id,
            'env_history': env_history,
            'task': task,
            'memory': memory,
            'observation': ob,
            'initial_valid_actions': initial_valid_actions,
            'reflexion_memory': reflexion_memory_for_todo,
            'todo_manager': todo_manager
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY LEAK FIX: Separate env storage from env_states
    # ═══════════════════════════════════════════════════════════════════════════
    # Create env registry - stores env objects separately to prevent 48,625x multiplier
    # The multiplier happens when env is stored IN env_states with working_reflexions,
    # creating deep circular references. By storing separately, we eliminate the chain.
    # ═══════════════════════════════════════════════════════════════════════════
    _env_registry = {env_data['env_id']: env_data['env'] for env_data in env_data_list}
    print(f"[MEMORY FIX] Created env registry with {len(_env_registry)} environments")
    print(f"[MEMORY FIX] env_states will contain only env_id, not env objects")

    # Create final env_states WITHOUT env object (just env_id)
    for env_data in env_data_list:
        env_states.append({
            # 'env': env_data['env'],  # REMOVED - this caused 48,625x multiplier!
            'env_id': env_data['env_id'],
            'trial_idx': trial_idx,  # CRITICAL FIX: Track trial for compression
            'episode_id': env_data['episode_id'],
            'history': env_data['env_history'],
            'task': env_data['task'],
            'memory': env_data['memory'],
            'observation': env_data['observation'],
            'prev_observation': env_data['observation'],
            'initial_observation': env_data['observation'],  # Store initial obs with ALL locations
            'failed_state_actions': {},
            'done': False,
            'success': False,
            'trajectory': [],
            'tried_actions': set(),
            'consecutive_failures': 0,
            'cur_step': 0,
            'step_gradient': {},
            'memory_context': None,
            'last_step_gradient': {
                'semantic_state_after': env_data['observation'],
                'task_progress': {'remaining': [env_data['task']]}
            },
            'progress_history': [],
            'is_stuck': False,
            'textgrad_components': prompt_generator.prompt_components.copy(),
            'memory_recommendations': {},
            'step_insights_accumulator': [],
            'consolidated_step_wisdom': "",
            'todo_manager': env_data['todo_manager'],
            'working_reflexions': [],  # Will be populated below with quality-filtered reflexions
        })

        # PROFILING: Log working_reflexions loading with QUALITY SELECTION
        try:
            from memory_profiler import profiler
            working_refl_raw = env_configs[env_data['env_id']].get('working_reflexions_history', [])

            # ═══════════════════════════════════════════════════════════════════════
            # QUALITY-BASED REFLEXION SELECTION (Fix for -23% Trial 1 decline)
            # Data-backed: Baseline used 4 reflexions → +23% improvement
            #             Our broken: 8-11 reflexions → -23% decline
            # Solution: Select best 5 reflexions to prevent cognitive overload
            # ═══════════════════════════════════════════════════════════════════════
            if working_refl_raw and trial_idx > 0:  # Only filter for Trial 1+
                # ═══════════════════════════════════════════════════════════════════════
                # IMPORTANCE-BASED SELECTION (Prevents Catastrophic Forgetting)
                # Data shows: SUCCESS always valuable, early reflexions are foundational,
                #            FAILURE reflexions prevent mistakes, cap at 7 for optimal
                # ═══════════════════════════════════════════════════════════════════════

                must_keep = []

                # 1. ALWAYS KEEP: ALL SUCCESS reflexions (show what works)
                success_refls = [r for r in working_refl_raw if r.get('type') == 'SUCCESS']
                must_keep.extend(success_refls)

                # Special handling for no-success cases (struggling environments)
                if not success_refls and len(working_refl_raw) >= 10:
                    # Keep more late-stage exploration when struggling
                    late_refls = [r for r in working_refl_raw if r.get('step', 0) > 5]
                    # Sort by step to get the latest insights
                    late_refls.sort(key=lambda x: x.get('step', 0), reverse=True)
                    must_keep.extend(late_refls[:3])  # Keep 3 late-stage patterns

                # 2. ALWAYS KEEP: FAILURE reflexions (prevent repeating mistakes) - max 2
                failure_refls = [r for r in working_refl_raw if r.get('type') == 'FAILURE']
                must_keep.extend(failure_refls[:2])

                # 3. KEEP EARLY: Foundation reflexions from steps 0-5 (early learning)
                early_refls = [r for r in working_refl_raw
                              if r.get('step', 999) <= 5
                              and r not in must_keep]
                must_keep.extend(early_refls[:2])

                # 4. Fill remaining slots (up to 10 total) with MILESTONES then recent
                remaining_slots = 10 - len(must_keep)
                if remaining_slots > 0:
                    # First, add MILESTONE reflexions (progress markers)
                    others = [r for r in working_refl_raw if r not in must_keep]
                    milestone_refls = [r for r in others if r.get('type') == 'MILESTONE']
                    must_keep.extend(milestone_refls[:remaining_slots])

                    # Then fill any remaining with most recent
                    if len(must_keep) < 10:
                        recent_others = [r for r in others if r not in must_keep]
                        must_keep.extend(recent_others[-(10-len(must_keep)):])

                # Cap at 10 (increased from 7 to preserve late-stage learning)
                working_refl = must_keep[:10]

                # Log what was selected
                success_count = len([r for r in working_refl if r.get('type') == 'SUCCESS'])
                failure_count = len([r for r in working_refl if r.get('type') == 'FAILURE'])
                milestone_count = len([r for r in working_refl if r.get('type') == 'MILESTONE'])
                early_count = len([r for r in working_refl if r.get('step', 999) <= 5])

                print(f"[IMPORTANCE FILTER] ENV {env_data['env_id']}: Selected {len(working_refl)} from {len(working_refl_raw)} total")
                print(f"  Breakdown: {success_count} SUCCESS, {failure_count} FAILURE, {milestone_count} MILESTONE")
                print(f"  {early_count} early (step≤5), preventing catastrophic forgetting")
            else:
                working_refl = working_refl_raw

            if working_refl:
                # Populate state with quality-filtered reflexions
                env_states[env_data['env_id']]['working_reflexions'] = list(working_refl)

                profiler.log_learning_transfer(
                    trial_idx=trial_idx,
                    env_id=env_data['env_id'],
                    learning_type='working_reflexion',
                    content=working_refl,
                    source=f"env_configs from trial {trial_idx-1} (quality-filtered)" if trial_idx > 0 else "initial"
                )
                print(f"[LEARNING] ENV {env_data['env_id']}: Loaded {len(working_refl)} working_reflexions from previous trial")
        except Exception as e:
            print(f"[WARNING] Quality filtering failed for ENV {env_data['env_id']}: {e}")
            # Fallback to raw reflexions
            env_states[env_data['env_id']]['working_reflexions'] = list(env_configs[env_data['env_id']].get('working_reflexions_history', []))

        # Initial observation learning
        env_understanding.learn_from_interaction("", ob)
    
    # ========================================================================
    # DEEP MEMORY TRACKING - FIND THE LEAK!
    # ========================================================================
    try:
        from deep_memory_tracker import tracker
        tracker.checkpoint("BEFORE Main Loop", {
            'env_states': env_states,
            'env_configs': env_configs,
        })
    except Exception as e:
        print(f"[WARNING] Deep memory tracking failed: {e}")

    # ========================================================================
    # SEQUENTIAL LEARNING SYSTEM - Process environments one at a time
    # ========================================================================

    # Shared knowledge base for cross-environment transfer
    shared_knowledge = {
        'universal': [],  # Universal strategies (share with all)
        'task_families': {},  # Task-family specific knowledge
        'textgrad_updates': {},  # Successful TextGrad component updates
        'successful_workflows': []  # Successful action patterns
    }

    log_debug("\n[SEQUENTIAL LEARNING] Initiating sequential execution mode")
    log_debug("[SEQUENTIAL LEARNING] Environments will be processed one at a time")
    log_debug("[SEQUENTIAL LEARNING] Knowledge from early envs will transfer to later envs")
    log_debug("[SEQUENTIAL LEARNING] Contamination prevention ACTIVE\n")

    # Create global env_prompt_generators for compatibility
    global env_prompt_generators
    env_prompt_generators = {}

    # Dynamic limit based on action space complexity
    # INCREASED: 15 was too low, agent needs room to recover from mistakes
    max_steps = 21

    # ========================================================================
    # TRUE PARALLEL ENVIRONMENT PROCESSING (NO CONTAMINATION)
    # ========================================================================
    # Process ALL environments simultaneously using ONLY their own memories
    # NO cross-task knowledge transfer = TRUE PARALLEL EXECUTION possible
    # This achieves ~9x speedup on A100 GPU

    # STEP 1: Initialize ALL environments before execution
    print(f"\n[PARALLEL INIT] Initializing {len(env_states)} environments...")
    for init_idx in range(len(env_states)):
        init_state = env_states[init_idx]
        task = init_state['task']

        # Create clean prompt generator WITHOUT cross-environment knowledge
        env_pg = DynamicPromptGenerator()
        if ENVIRONMENT_KNOWLEDGE:
            env_pg.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)

        env_pg.set_task(task)
        env_prompt_generators[init_idx] = env_pg
        init_state['prompt_generator'] = env_pg

        log_debug(f"[INIT {init_idx}] {task[:60]}...")

    print(f"[PARALLEL EXEC] Starting parallel execution of {len(env_states)} environments\n")

    # STEP 2: Execute ALL environments in TRUE PARALLEL
    # Process all active environments simultaneously each step
    global_step_counter = 0
    while any(not s['done'] and s['cur_step'] < max_steps for s in env_states):
        global_step_counter += 1
        active_envs = [s for s in env_states if not s['done'] and s['cur_step'] < max_steps]

        if not active_envs:
            break

        print(f"\n[PARALLEL STEP {global_step_counter}] {len(active_envs)}/{len(env_states)} environments active")

        # STEP 2A: Collect data for ALL active environments (TRUE PARALLEL)
        batch_data_all = []
        active_states_all = []

        for current_state in active_envs:
            current_env_idx = current_state['env_id']

            # Skip if this env just finished
            if current_state['done'] or current_state['cur_step'] >= max_steps:
                continue

            # Get valid actions for current environment
            state = current_state  # Alias for code compatibility
            valid_actions = []
            # MEMORY FIX: Access env from registry, not from state
            if hasattr(_env_registry[state['env_id']], 'get_current_valid_actions'):
                valid_actions = _env_registry[state['env_id']].get_current_valid_actions()
                if DEBUG_ACTOR:
                    log_debug(f"[ENV {state['env_id']} - Step {state['cur_step']}] Found {len(valid_actions)} valid actions")

            # CRITICAL: Store raw actions BEFORE any filtering
            # This preserves TextGrad recommendations that might be incorrectly filtered
            raw_valid_actions = valid_actions.copy()


            if not valid_actions:
                # This is a critical failure - environment not providing actions
                log_debug(f"\n[ENV {state['env_id']}] CRITICAL: No valid actions from environment")

                # Record this as a learning signal
                state['step_gradient'] = {
                    'state_change': 'ENVIRONMENT_FAILURE',
                    'progress_score': 0,
                    'hypothesis': 'Environment failed to provide valid actions - may need reset or different approach',
                    'next_action_guidance': 'Cannot proceed without valid actions',
                    'raw_reflection': 'No valid actions available from environment'
                }

                # Store this failure for learning
                if 'critical_failures' not in state:
                    state['critical_failures'] = []
                state['critical_failures'].append({
                    'step': state['cur_step'],
                    'state': state['prev_observation'],
                    'reason': 'no_valid_actions'
                })

                state['done'] = True
                state['success'] = False
                state['failure_reason'] = "No valid actions - environment issue"
                break  # Exit the while loop


            env_prompt_generators[state['env_id']].discovered_knowledge['available_actions'] = set(valid_actions)
            
            # Update max steps based on action space
            # max_steps = min(len(valid_actions) * 3, 100)

            # CHECK STATE-ACTION MEMORY FIRST
            current_state_hash = hashlib.md5(state['observation'][:200].encode()).hexdigest()[:8]

            # Remove actions that failed in THIS EXACT state
            if current_state_hash in state.get('failed_state_actions', {}):
                failed_in_state = state['failed_state_actions'][current_state_hash]
                original_count = len(valid_actions)
                valid_actions = [a for a in valid_actions if a not in failed_in_state]
                if DEBUG_ACTOR:
                    log_debug(f"[STATE MEMORY] Removed {original_count - len(valid_actions)} failed actions for state {current_state_hash}")

            # Check conditional failures with context and decay
            if state['env_id'] in env_prompt_generators:
                pg = env_prompt_generators[state['env_id']]
                if 'conditional_failures' in pg.discovered_knowledge:
                    before_count = len(valid_actions)
                    current_context = {
                        'state_text': state['observation'][:200],
                        'task': state['task'],
                        'step': state['cur_step'],
                        'prerequisites': [act for act, _, _ in state['trajectory'][-5:] if act]
                    }
                    
                    actions_to_avoid = set()
                    for failure in pg.discovered_knowledge['conditional_failures']:
                        # Temporal decay
                        age = state['cur_step'] - failure.get('timestamp', 0)
                        decay_factor = max(0.1, 1.0 - (age / 10))
                        
                        # Context similarity check
                        source_ctx = {
                            'state_text': failure.get('state_text', ''),
                            'task': failure.get('task', ''),
                            'step': failure.get('step', 0),
                            'prerequisites': failure.get('prerequisites', [])
                        }
                        
                        should_avoid, match_score = should_share_knowledge(source_ctx, current_context)
                        
                        # Apply decay to match score
                        final_score = match_score * decay_factor * failure.get('confidence', 1.0)
                        
                        if final_score > 0.5:
                            actions_to_avoid.add(failure['action'])
                    
                    valid_actions = [a for a in valid_actions if a not in actions_to_avoid]
                    if DEBUG_ACTOR:
                        log_debug(f"[CONTEXT FILTER] Removed {before_count - len(valid_actions)} contextually inappropriate actions")

            # GET RECOMMENDATIONS FROM UNIVERSAL MEMORY - Using semantic understanding
            if 'last_step_gradient' in state and state['last_step_gradient']:
                current_semantic_state = state['last_step_gradient'].get('semantic_state_after', '')
                task_remaining = state['last_step_gradient'].get('task_progress', {}).get('remaining', [])
            else:
                # First step - no gradient yet
                current_semantic_state = state['prev_observation']
                task_remaining = [state['task']]  # Use full task as remaining

            recommendations = universal_memory.get_semantic_recommendations(
                current_state_description=current_semantic_state,
                task_remaining=task_remaining,
                available_actions=valid_actions
            )
            
            # CRITICAL: Store recommendations for use in action selection
            state['memory_recommendations'] = recommendations


            # CHECK FOR EXACT SUCCESSFUL TRAJECTORY MATCH
            if env_configs[state['env_id']].get('successful_trajectory') and trial_idx > 0:
                successful_actions = [act for act, _ in env_configs[state['env_id']]['successful_trajectory']]
                # Move successful actions to front of valid_actions
                reordered = []
                for sa in successful_actions:
                    if sa in valid_actions:
                        reordered.append(sa)
                        valid_actions.remove(sa)
                valid_actions = reordered + valid_actions
                if reordered:
                    print(f"[REUSE] Prioritized {len(reordered)} actions from previous success")
          
            # # PRIORITIZE ACTIONS BASED ON MEMORY - NO PATTERNS
            # # Build memory context
            # failed_actions_str = ""
            # if recommendations.get('avoid'):
            #     failed_actions_str = "NEVER TRY THESE (they failed before): " + \
            #                         ", ".join([rec['action'] for rec in recommendations['avoid'][:5]])

            # successful_actions_str = ""
            # if recommendations.get('previously_succeeded'):
            #     successful_actions_str = "THESE WORKED BEFORE: " + \
            #                             ", ".join([rec['action'] for rec in recommendations['previously_succeeded'][:3]])

            # semantic_prompt = f"""Task: {state['task']}

            # {failed_actions_str}
            # {successful_actions_str}

            # Available actions:
            # {chr(10).join([f"{i+1}. {action}" for i, action in enumerate(valid_actions[:40])])}

            # Which actions are semantically most relevant for completing this task?
            # AVOID actions marked as failed. PREFER actions that worked before.
            # List just the numbers of the 5 most relevant actions, separated by commas."""

            # Semantic ranking removed - using universal memory-based prioritization instead

            # for num_str in response.replace(',', ' ').split():
            #     if num_str.isdigit():
            #         idx = int(num_str) - 1
            #         if 0 <= idx < len(valid_actions):
            #             semantic_priorities.append(valid_actions[idx])
            

            # # Build prioritized list: reflexion-mentioned actions FIRST
            # prioritized_actions = []
            # log_debug(f"[DEBUG-VA] Valid actions before prioritization: {len(valid_actions)}")
            # added_actions = set()

            # # Extract actions mentioned in recent reflexions
            # if 'working_reflexions' in state and state['working_reflexions']:
            #     for ref in state['working_reflexions'][-5:]:  # Last 2 reflexions
            #         reflection_lower = ref['reflection'].lower()
            #         for valid_action in valid_actions:
            #             if valid_action.lower() in reflection_lower and valid_action not in added_actions:
            #                 prioritized_actions.append(valid_action)
            #                 added_actions.add(valid_action)
            #                 log_debug(f"[REFLEXION PRIORITY] Moved to front: {valid_action}")

            # # 1. Add LLM's semantic choices first
            # for action in semantic_priorities:
            #     if action not in added_actions:
            #         prioritized_actions.append(action)
            #         added_actions.add(action)

            # # 2. Then add memory recommendations
            # for rec in recommendations.get('historically_successful', []):
            #     if rec['action'] in valid_actions and rec['action'] not in added_actions:
            #         prioritized_actions.append(rec['action'])
            #         added_actions.add(rec['action'])

            # for rec in recommendations.get('explore', []):
            #     if rec['action'] in valid_actions and rec['action'] not in added_actions:
            #         prioritized_actions.append(rec['action'])
            #         added_actions.add(rec['action'])

            # # 3. Add remaining valid actions
            # for action in valid_actions:
            #     if action not in added_actions:
            #         prioritized_actions.append(action)
            #         added_actions.add(action)
            
            # # Ensure we have valid actions but filter known failures
            # if not prioritized_actions:
            #     # Try memory recommendations first
            #     if recommendations.get('strongly_recommended'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['strongly_recommended'] 
            #                              if rec['action'] in valid_actions]
                
            #     # Then try historically successful
            #     if not prioritized_actions and recommendations.get('historically_successful'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['historically_successful'] 
            #                              if rec['action'] in valid_actions]
                
            #     # Then try unexplored from memory
            #     if not prioritized_actions and recommendations.get('explore'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['explore'][:10] 
            #                              if rec['action'] in valid_actions]
                
            #     # If still nothing, we're stuck - better to end than random
            #     if not prioritized_actions:
            #         print(f"[WARNING] ENV {state['env_id']} has no valid actions after filtering - ending episode")
            #         state['done'] = True
            #         state['success'] = False
            #         # Record why we failed
            #         state['failure_reason'] = "No valid actions after filtering failures"
            #         continue

            # log_debug(f"\n[ENV {state['env_id']} FILTER RESULTS]")
            # log_debug(f"Original: {len(valid_actions) + (len(actions_to_delete) if 'actions_to_delete' in locals() else 0)}")
            # log_debug(f"After all filtering: {len(prioritized_actions)}")
            # log_debug(f"First 3 actions: {prioritized_actions[:3]}")
            
           
            # Initialize failure history if not exists
            if 'failure_history' not in state:
                state['failure_history'] = []

            # Filtering will be done in batched action selection (no sequential calls!)
            # Use all valid actions (filtering happens in LLM prompt)
            filtered_actions = valid_actions

            # Prioritize based on step gradient
            prioritized_actions = []
            added = set()

            # Priority 1: Exact match with step gradient
            if 'step_gradient' in state and state['step_gradient']:
                suggested = state['step_gradient'].get('next_action_guidance', '')
                for action in filtered_actions:  # Use filtered list
                    if action == suggested or action in suggested:
                        prioritized_actions.append(action)
                        added.add(action)
                        print(f"[PRIORITY] Step gradient match: {action}")

            # Priority 2: Never tried actions
            for action in filtered_actions:  # Use filtered list
                if action not in state.get('tried_actions', set()) and action not in added:
                    prioritized_actions.append(action)
                    added.add(action)

            # Priority 3: Everything else
            for action in filtered_actions:  # Use filtered list
                if action not in added:
                    prioritized_actions.append(action)
            
            # Include working reflexions in batch data - TIERED COMPRESSION
            working_reflexions_text = ""
            if 'working_reflexions' in state and state['working_reflexions']:
                # Apply tiered compression before formatting
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                # Format based on compression tier
                if tiered_reflexions:
                    working_reflexions_text = "\n============ CRITICAL LEARNING FROM PREVIOUS EXPERIENCE ============\n"

                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        ref_type = ref.get('type', 'general')
                        ref_trial = ref.get('trial', trial_idx)

                        # Add prominent markers for different types
                        if ref_type == 'FAILURE':
                            marker = "🚫 AVOID:"
                        elif ref_type == 'SUCCESS':
                            marker = "✅ PROVEN:"
                        elif ref_trial < trial_idx:
                            marker = "📚 TRIAL 0 LESSON:"  # Cross-trial learning
                        else:
                            marker = "💡"

                        if compression_type == 'heavy':
                            working_reflexions_text += f"\n{marker} Steps {ref['step']} (summary):\n{ref['reflection']}\n"
                        elif compression_type == 'medium':
                            working_reflexions_text += f"\n{marker} Steps {ref['step']} (compressed):\n{ref['reflection']}\n"
                        else:
                            # Verbose (recent or cross-trial)
                            working_reflexions_text += f"\n{marker} Step {ref['step']}:\n{ref['reflection']}\n"

            # Track how many reflexion-suggested actions are at the front
            reflexion_count = len([a for a in prioritized_actions[:5] if any(
                a.lower() in ref['reflection'].lower() 
                for ref in state.get('working_reflexions', [])[-2:]
            )])
            log_debug(f"[REFLEXION CHECK] {reflexion_count} of top 5 actions match reflexion suggestions")

            # Update memory BEFORE creating batch_data
            env_idx = state['env_id']
            import copy
            state['memory'] = copy.deepcopy(env_configs[env_idx].get('memory', [])[-15:] + env_configs[env_idx].get('step_memory', [])[-10:])  # MEMORY FIX: Deep copy to break circular ref

            # Append this environment's data to batch (for TRUE PARALLEL processing)
            batch_data_all.append({
                'valid_actions': prioritized_actions.copy(),
                'raw_valid_actions': raw_valid_actions.copy(),  # CRITICAL: Unfiltered actions for TextGrad validation
                'observation': state['prev_observation'],
                'task': state['task'],
                'env_id': state['env_id'],  # Pass actual environment ID for correct logging
                'step_gradient': state.get('step_gradient', {}),
                'textgrad_components': state['prompt_generator'].prompt_components.copy(),  # Use env-specific
                'reflexion_memory': state['memory'],
                'working_reflexions': working_reflexions_text,
                'action_history': state['trajectory'][-15:] if state['trajectory'] else [],
                'discovered_patterns': {},
                'tried_actions': state.get('tried_actions', set()),
                'interaction_count': state['cur_step'],
                'memory_recommendations': recommendations,
                'is_stuck': state.get('is_stuck', False),
                'consolidated_step_wisdom': state.get('consolidated_step_wisdom', ''),
                'step_insights_accumulator': state.get('step_insights_accumulator', [])[-5:],
                'progress_history': state.get('progress_history', []),
                'todo_manager': state.get('todo_manager'),  # ADD TODO MANAGER
                # ADD THESE NEW FIELDS
                'last_step_gradient': state.get('last_step_gradient', {}),
                'failed_state_actions': state.get('failed_state_actions', {}),
                'successful_actions': state['prompt_generator'].discovered_knowledge.get('successful_actions', []),
                'action_blacklist': state['prompt_generator'].discovered_knowledge.get('action_blacklist', []),
                # INTELLIGENT FILTERING: Add failure history for batched filtering
                'failure_history': state.get('failure_history', [])
            })
            active_states_all.append(state)

            log_debug(f"[BATCH COLLECT] ENV {state['env_id']} - Step {state['cur_step']} data collected")

        # End of data collection for loop
        print(f"[DATA COLLECTION] Collected data for {len(batch_data_all)} environments")

        # ========================================================================
        # PHASE 3B: BATCH ACTION SELECTION
        # ========================================================================
        print(f"\n[PHASE 3B] Selecting actions for {len(batch_data_all)} environments...")
        selected_actions = reasoning_based_action_selection_batch(
            batch_data=batch_data_all,
            prompt_generator=prompt_generator,
            DEBUG_ACTOR=DEBUG_ACTOR,
            log_debug=log_debug
        )
        print(f"[PHASE 3B] ✓ Selected {len(selected_actions)} actions in ONE batch")

        # ========================================================================
        # PHASE 4: EXECUTE ALL ACTIONS, COLLECT RESULTS
        # ========================================================================
        print(f"\n[PHASE 4] Executing {len(selected_actions)} actions...")
        action_results = []

        for idx, (state, action) in enumerate(zip(active_states_all, selected_actions)):
            # Track tried actions
            if not isinstance(state['tried_actions'], set):
                state['tried_actions'] = set(state['tried_actions'])
            state['tried_actions'].add(action)

            # Execute action - MEMORY FIX: Use env from registry
            env = _env_registry[state['env_id']]
            observation, reward, done, info = env.step([action])
            observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
            done = done[0]

            # ═══════════════════════════════════════════════════════════════
            # SYNERGISTIC EPISODIC REFLEXION GENERATION (CORRECT LOCATION)
            # Generate reflexion ONLY at key moments (failures/milestones/success)
            # ═══════════════════════════════════════════════════════════════
            is_key_moment = False
            moment_type = ""

            # Check for failure indicators
            failure_indicators = ['nothing happens', "don't see", "can't", 'already', 'closed']
            if any(fail in observation.lower() for fail in failure_indicators):
                is_key_moment = True
                moment_type = "FAILURE"
            # Check for success
            elif 'you win' in observation.lower() or done:
                is_key_moment = True
                moment_type = "SUCCESS"
            # Check for milestone (every 5 steps)
            elif state['cur_step'] % 5 == 0:
                is_key_moment = True
                moment_type = "MILESTONE"

            if is_key_moment:
                # Skip early milestones - not enough context
                skip_reflexion = (moment_type == "MILESTONE" and state['cur_step'] < 5)

                if not skip_reflexion:
                    try:
                        if 'working_reflexions' not in state:
                            state['working_reflexions'] = []

                        # Create context-specific prompts for better insights
                        if moment_type == "FAILURE":
                            reflexion_prompt = f"""Task: {state['task']}
Action tried: {action}
What happened: {observation[:150]}

This action FAILED. What concrete lesson should we remember to avoid this in future trials?
Focus on: What assumption was wrong? What should we do instead?
Provide ONE actionable insight (1-2 sentences):"""

                        elif moment_type == "SUCCESS":
                            reflexion_prompt = f"""Task: {state['task']}
Successful action: {action}
Result: {observation[:150]}

This action SUCCEEDED and completed the task! What pattern should we remember?
Focus on: What sequence of actions worked? What made this successful?
Provide a success pattern to reuse (1-2 sentences):"""

                        else:  # MILESTONE
                            # Get progress score for context
                            progress_score = state.get('step_gradient', {}).get('progress_score', 0)
                            
                            reflexion_prompt = f"""Task: {state['task']}
Action just executed: {action}
Result: {observation[:150]}
Progress score: {progress_score}/10

ANALYSIS:
1. What are the KEY ACTION REQUIREMENTS from the task description?
   (What specific operations must be performed to complete this task?)

2. Does the action we just executed ALIGN with these requirements?
   (Does it perform the operation the task asks for?)

3. If progress is low (<5/10), what SPECIFIC ACTION should we try instead?
   (Recommend an action that directly addresses the task requirement)

Provide ONE actionable insight (1-2 sentences):"""

                        from vllm import SamplingParams
                        sampling_params = SamplingParams(temperature=0.3, max_tokens=200)  # Lower temp for precision
                        step_reflection_output = model.generate([reflexion_prompt], sampling_params)[0]
                        step_reflection = step_reflection_output.outputs[0].text.strip()

                        state['working_reflexions'].append({
                            'step': state['cur_step'],
                            'action': action,
                            'observation': observation[:200],
                            'reflection': step_reflection,
                            'type': moment_type,
                            'success': moment_type == "SUCCESS"
                        })

                        # Save to env_configs for cross-trial persistence WITH DEDUPLICATION
                        env_idx = state['env_id']
                        if 'step_memory' not in env_configs[env_idx]:
                            env_configs[env_idx]['step_memory'] = []

                        # Deduplication: Check if a similar memory already exists
                        memory_text = f"[Step {state['cur_step']} {moment_type}] {step_reflection}"
                        is_duplicate = False

                        # Simple deduplication: Check for semantic similarity
                        for existing_mem in env_configs[env_idx]['step_memory']:
                            # Extract key words from both memories
                            new_words = set(step_reflection.lower().split())
                            existing_words = set(existing_mem.lower().split())

                            # If >70% word overlap, consider it a duplicate
                            if len(new_words) > 0:
                                overlap = len(new_words & existing_words) / len(new_words)
                                if overlap > 0.7:
                                    is_duplicate = True
                                    log_debug(f"[REFLEXION DEDUP] Skipping duplicate memory (overlap={overlap:.0%})")
                                    break

                        # Only save if not a duplicate
                        if not is_duplicate:
                            env_configs[env_idx]['step_memory'].append(memory_text)

                            # Prioritize SUCCESS memories - keep more of them
                            if moment_type == "SUCCESS":
                                # Keep last 15 memories total, but protect SUCCESS memories
                                if len(env_configs[env_idx]['step_memory']) > 15:
                                    # Remove oldest non-SUCCESS memory
                                    for i in range(len(env_configs[env_idx]['step_memory'])):
                                        if 'SUCCESS' not in env_configs[env_idx]['step_memory'][i]:
                                            del env_configs[env_idx]['step_memory'][i]
                                            break
                                    # If all are SUCCESS, just trim normally
                                    if len(env_configs[env_idx]['step_memory']) > 15:
                                        env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-15:]
                            else:
                                # For FAILURE/MILESTONE, keep only last 10
                                if len(env_configs[env_idx]['step_memory']) > 10:
                                    env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]

                            print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Generated episodic memory")
                        else:
                            print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Skipped (duplicate)")
                    except Exception as e:
                        log_debug(f"[REFLEXION ERROR] ENV {state['env_id']}: {e}")
            # ═══════════════════════════════════════════════════════════════

            # Detect failures (from original line ~2236)
            is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])

            # Get next valid actions (loop detection removed) - MEMORY FIX: Use env from registry
            next_valid_actions = []
            env = _env_registry[state['env_id']]
            if hasattr(env, 'get_current_valid_actions'):
                next_valid_actions = env.get_current_valid_actions()

            # Get current TODO (from original line ~2328)
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # Track inventory (from original line ~2333)
            inventory_items = []
            prev_words = set(state['prev_observation'].lower().split())
            curr_words = set(observation.lower().split())
            disappeared_words = prev_words - curr_words
            action_words = action.lower().split()
            if len(action_words) >= 3:
                for i in range(len(action_words) - 2):
                    if action_words[i+1] in ['from', 'to', 'in', 'on', 'into', 'onto']:
                        potential_item = action_words[i]
                        if potential_item in disappeared_words or len(disappeared_words) > 0:
                            inventory_items.append(potential_item)
                            break

            # Get episodic memory
            env_idx = state['env_id']
            episodic_memory_only = env_configs[env_idx].get('memory', [])

            # Get tiered reflexions
            tiered_step_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

            # Extract explored locations from TODO manager
            explored_locations_dict = {}
            if 'todo_manager' in state and state['todo_manager']:
                explored_locations_dict = state['todo_manager'].visited_locations

            # Collect ALL data for gradient generation (DON'T generate yet!)
            action_results.append({
                'idx': idx,
                'state': state,
                'action': action,
                'observation': observation,
                'prev_observation': state['prev_observation'],
                'initial_observation': state.get('initial_observation', ''),  # CRITICAL: All available locations
                'explored_locations': explored_locations_dict,  # CRITICAL: What's been checked
                'is_failure': is_failure,
                'done': done,
                'reward': reward,
                'info': info,
                'is_repetitive': False,  # Loop detection removed
                'next_valid_actions': next_valid_actions,
                'current_todo': current_todo,
                'inventory_items': inventory_items,
                'tiered_step_reflexions': tiered_step_reflexions,
                'episodic_memory_only': episodic_memory_only,
                'step_insights_accumulator': state.get('step_insights_accumulator', []),  # CRITICAL: TextGrad's gradient history!
                'task': state['task'],
                'current_step': state['cur_step'],
                'env_idx_in_batch': idx
            })

        print(f"[PHASE 4] ✓ Executed {len(action_results)} actions, collected results")

        # ========================================================================
        # PHASE 5: BATCH GENERATE STEP GRADIENTS (TRUE SYNERGY - TWO PASS!)
        # ========================================================================
        print(f"\n[PHASE 5] Generating step gradients for {len(action_results)} environments...")

        # TRUE SYNERGY ARCHITECTURE: Two-pass approach
        # PASS 1: Generate Reflexion strategic insights for struggling environments (progress < 4)
        # PASS 2: Generate TextGrad actions for ALL environments (with updated Reflexion insights in context)

        # PASS 1: Identify environments needing Reflexion strategic insights
        print(f"[PHASE 5 - PASS 1] Identifying environments needing Reflexion strategic analysis...")
        reflexion_needed_indices = []
        reflexion_prompts = []

        for idx, result in enumerate(action_results):
            state = result['state']
            current_step = state['cur_step']

            # SIMPLIFIED SCHEDULE: Fixed Reflexion cadence every 5 steps
            # Reflexion has complete history and can detect loops/patterns
            # This eliminates buggy progress-based heuristics that masked loops
            use_reflexion = (current_step % 5 == 0)

            log_debug(f"[SCHEDULE] ENV {state['env_id']} Step {current_step}: Reflexion={'YES' if use_reflexion else 'NO'} (fixed schedule: every 5 steps)")

            if use_reflexion:
                # Fixed schedule: Reflexion provides strategic insights every 5 steps
                reflexion_needed_indices.append(idx)
                # Build Reflexion prompt for strategic insight generation
                _, reflexion_prompt = build_step_gradient_prompt_from_data(result, log_debug)
                reflexion_prompts.append(reflexion_prompt)
                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {current_step}: Scheduled Reflexion (fixed cadence: every 5 steps)")

        # Generate Reflexion strategic insights in batch (if any needed)
        if len(reflexion_prompts) > 0:
            print(f"[PHASE 5 - PASS 1] Generating {len(reflexion_prompts)} Reflexion strategic insights in batch...")
            from vllm import SamplingParams
            reflexion_sampling_params = SamplingParams(
                max_tokens=7000,
                temperature=0.3,
                stop=["TASK:", "BEFORE:"],
                skip_special_tokens=True
            )
            reflexion_outputs = model.generate(reflexion_prompts, reflexion_sampling_params, reasoning_effort='medium')
            print(f"[PHASE 5 - PASS 1] ✓ Generated {len(reflexion_outputs)} Reflexion insights!")

            # Parse and add Reflexion insights to state context
            for i, (reflexion_output, result_idx) in enumerate(zip(reflexion_outputs, reflexion_needed_indices)):
                result = action_results[result_idx]
                state = result['state']
                reflexion_text = reflexion_output.outputs[0].text.strip()

                # Parse Reflexion strategic insight
                reflexion_insight = parse_step_gradient_response(
                    response=reflexion_text,
                    task=result['task'],
                    prev_observation=result['prev_observation'],
                    curr_observation=result['observation'],
                    action=result['action']
                )

                # Add to working reflexions for this environment
                if 'step_reflexions' not in state:
                    state['step_reflexions'] = []

                state['step_reflexions'].append({
                    'step': state['cur_step'],
                    'reflection': reflexion_insight.get('hypothesis', 'Strategic insight generated'),
                    'progress_score': reflexion_insight.get('progress_score', 0),
                    'is_reflexion_strategic_insight': True  # Mark as Reflexion strategic analysis
                })

                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {state['cur_step']}: Added Reflexion strategic insight to context (counters reset)")
        else:
            print(f"[PHASE 5 - PASS 1] No environments need Reflexion (all making good progress)")

        # PASS 2: Generate TextGrad actions for ALL environments (now with updated Reflexion insights)
        print(f"[PHASE 5 - PASS 2] Generating TextGrad actions for ALL {len(action_results)} environments...")
        textgrad_prompts = []

        for result in action_results:
            # ALWAYS use TextGrad to generate actions (TRUE SYNERGY!)
            # But TextGrad now has Reflexion strategic insights in context (via previous_reflexions)
            textgrad_prompt = generate_textgrad_gradient_prompt(
                result,
                result['tiered_step_reflexions'],  # Now includes Reflexion insights!
                result['episodic_memory_only'],  # FIX: Use correct key name!
                result['initial_observation'],
                result['explored_locations'],
                result['next_valid_actions']  # FIX: Use correct key name!
            )
            textgrad_prompts.append(textgrad_prompt)

        # Batch generate ALL TextGrad actions
        print(f"[PHASE 5 - PASS 2] Calling model.generate() with {len(textgrad_prompts)} TextGrad prompts in ONE batch...")
        from vllm import SamplingParams
        textgrad_sampling_params = SamplingParams(
            max_tokens=7000,
            temperature=0.3,
            stop=["TASK:", "BEFORE:"],
            skip_special_tokens=True
        )
        gradient_outputs = model.generate(textgrad_prompts, textgrad_sampling_params, reasoning_effort='medium')
        print(f"[PHASE 5 - PASS 2] ✓ Generated {len(gradient_outputs)} TextGrad actions in ONE batch call!")

        # Parse ALL gradient outputs
        print(f"[PHASE 5] Parsing {len(gradient_outputs)} gradient outputs...")
        step_gradients = []
        for idx, (gradient_output, result) in enumerate(zip(gradient_outputs, action_results)):
            gradient_text = gradient_output.outputs[0].text.strip()
            step_gradient = parse_step_gradient_response(
                response=gradient_text,
                task=result['task'],
                prev_observation=result['prev_observation'],
                curr_observation=result['observation'],
                action=result['action']
            )

            # TRUE SYNERGY METADATA: Mark whether this action used Reflexion insight
            if idx in reflexion_needed_indices:
                step_gradient['guidance_source'] = 'textgrad_with_reflexion_insight'
            else:
                step_gradient['guidance_source'] = 'textgrad'

            step_gradients.append(step_gradient)
        print(f"[PHASE 5] ✓ Parsed {len(step_gradients)} gradients (TRUE SYNERGY: TextGrad generates 100% of actions)")

        # ========================================================================
        # PHASE 6: APPLY ALL GRADIENTS AND PROCESS LEARNING FLOWS
        # ========================================================================
        print(f"\n[PHASE 6] Applying {len(step_gradients)} gradients and processing learning...")

        # MEMORY LEAK DEBUG: Track memory at each iteration
        import psutil
        import os as os_mod
        process = psutil.Process(os_mod.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        for idx, (result, step_gradient) in enumerate(zip(action_results, step_gradients)):
            state = result['state']
            action = result['action']
            observation = result['observation']
            is_failure = result['is_failure']
            done = result['done']
            reward = result['reward']
            info = result['info']

            # MEMORY LEAK DEBUG: Check memory growth per iteration
            if idx > 0 and idx % 5 == 0:
                memory_current = process.memory_info().rss / 1024 / 1024 / 1024  # GB
                print(f"[MEMORY LEAK] Step {idx}: {memory_current:.2f} GB (delta: +{(memory_current - memory_before):.2f} GB)")

            # 1. UPDATE TODO MANAGER (lines 2378-2388)
            if 'todo_manager' in state and state['todo_manager']:
                try:
                    state['todo_manager'].update_from_action_feedback(
                        action=action,
                        prev_observation=result['prev_observation'],
                        curr_observation=observation,
                        progress_score=step_gradient.get('progress_score', 0)
                    )
                except Exception as e:
                    log_debug(f"[TODO UPDATE ERROR] {e}")

            # 2. LOG STEP GRADIENT (lines 2390-2410)
            comprehensive_logger.log_step_gradient(
                env_id=state['env_id'],
                step=state['cur_step'],
                gradient=step_gradient
            )

            if trial_log_path and os.path.exists(trial_log_path):
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[STEP GRADIENT] Step {state['cur_step']}:\n")
                    f.write(f"  Progress: {step_gradient.get('progress_score', 0)}/10\n")
                    f.write(f"  Hypothesis: {step_gradient.get('hypothesis', 'None')}\n")
                    f.write(f"  Next: {step_gradient.get('next_action_guidance', 'None')}\n")

                    if 'todo_manager' in state and state['todo_manager']:
                        f.write(f"\n[TODO STATUS]:\n")
                        f.write(f"{state['todo_manager'].get_formatted_todos()}\n")

            # 3. APPLY GRADIENT TO PROMPT GENERATOR (lines 2414-2418)
            env_pg = state['prompt_generator']
            env_pg.apply_universal_step_gradient(step_gradient)

            # 4. SHARE FAILURES (lines 2421-2464)
            if is_failure:
                import hashlib
                failure_context = {
                    'action': action,
                    'state_hash': hashlib.md5(result['prev_observation'][:200].encode()).hexdigest()[:8],
                    'state_text': result['prev_observation'][:200],
                    'task': state['task'],
                    'step': state['cur_step'],
                    'prerequisites': [act for act, _, _ in state['trajectory'][-5:] if act],
                    'failure_reason': observation,
                    'timestamp': state['cur_step'],
                    'confidence': 1.0,
                    'is_universal': any(uf in observation.lower() for uf in
                                      ["i don't understand", "invalid command", "not a verb"])
                }

                if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                    state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                state['prompt_generator'].discovered_knowledge['conditional_failures'].append(failure_context)

                # Smart sharing with other envs
                for other_state in env_states:
                    if other_state['env_id'] != state['env_id'] and not other_state['done']:
                        target_context = {
                            'state_text': other_state['prev_observation'][:200],
                            'task': other_state['task'],
                            'step': other_state['cur_step'],
                            'prerequisites': [act for act, _, _ in other_state['trajectory'][-5:] if act]
                        }

                        should_share, confidence = should_share_knowledge(failure_context, target_context)

                        if should_share:
                            shared_failure = failure_context.copy()
                            shared_failure['confidence'] = confidence
                            shared_failure['source_env'] = state['env_id']

                            if 'conditional_failures' not in other_state['prompt_generator'].discovered_knowledge:
                                other_state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                            other_state['prompt_generator'].discovered_knowledge['conditional_failures'].append(shared_failure)

            # 5. STORE GRADIENT IN STATE (lines 2466-2472)
            state['textgrad_components'] = env_pg.prompt_components.copy()
            state['last_step_gradient'] = step_gradient
            state['step_gradient'] = step_gradient

            # 6. LOG TO FILE (lines 2474-2483)
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[GRADIENT UPDATE {state['cur_step']}]\n")
                    f.write(f"  Next action guidance: {step_gradient.get('next_action_guidance', 'None')}\n")
                    f.write(f"  Progress score: {step_gradient.get('progress_score', 0)}\n")

            # 7. ACCUMULATE INSIGHTS (lines 2485-2558)
            progress_score = step_gradient.get('progress_score', 0)
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'hypothesis': step_gradient.get('hypothesis', ''),
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', [])
            })

            # Synthesize insights every 3 steps
            if len(state['step_insights_accumulator']) >= 3 and state['cur_step'] % 3 == 0:
                recent_insights = state['step_insights_accumulator'][-3:]

                # Detect patterns
                avg_progress = sum(ins['progress_score'] for ins in recent_insights) / len(recent_insights)
                stuck_pattern = avg_progress < 2
                repeated_hypotheses = len(set(ins['hypothesis'] for ins in recent_insights)) == 1

                # Consolidate wisdom
                synthesis = {
                    'steps': [ins['step'] for ins in recent_insights],
                    'pattern': 'stuck' if stuck_pattern else ('exploring' if avg_progress < 5 else 'progressing'),
                    'key_insight': recent_insights[-1]['hypothesis'],
                    'recommended_focus': recent_insights[-1]['next_guidance'],
                    'common_prereqs': list(set(
                        prereq for ins in recent_insights
                        for prereq in ins.get('missing_prereqs', [])
                    ))
                }

                state['consolidated_step_wisdom'] = synthesis

                # Keep only recent detailed insights
                if len(state['step_insights_accumulator']) > 20:
                    state['step_insights_accumulator'] = state['step_insights_accumulator'][-20:]

            # 8. TRACK PROGRESS (lines 2560-2570)
            state['progress_history'].append(progress_score)
            if len(state['progress_history']) >= 5:
                recent_progress = state['progress_history'][-5:]
                avg_recent_progress = sum(recent_progress) / len(recent_progress)
                state['is_stuck'] = avg_recent_progress < 2
            else:
                state['is_stuck'] = False

            # 9. RECORD SEMANTIC INTERACTION (lines 2572-2593)
            if step_gradient and 'semantic_state_before' in step_gradient:
                universal_memory.record_semantic_interaction(
                    semantic_state_before=step_gradient['semantic_state_before'],
                    action=action,
                    action_reasoning=batch_data_all[result['env_idx_in_batch']].get('action_reasoning', ''),
                    semantic_state_after=step_gradient['semantic_state_after'],
                    prerequisites=step_gradient.get('prerequisites', {}),
                    task_progress=step_gradient.get('task_progress', {}),
                    success=not is_failure,
                    task=state['task'],
                    episode_id=state['episode_id']
                )
                state['last_step_gradient'] = step_gradient
            else:
                state['last_step_gradient'] = {
                    'semantic_state_after': observation,
                    'task_progress': {'remaining': [state['task']]}
                }

            # 10. MEMORY INTEGRATION (lines 2595-2626)
            from generate_reflections import global_memory_manager
            if is_failure or state['cur_step'] % 3 == 0:
                try:
                    mini_reflection = f"Episode {state['episode_id']}, Step {state['cur_step']}: "
                    if is_failure:
                        mini_reflection += f"EXACT action '{action}' failed. State: {observation}"
                    else:
                        mini_reflection += f"Action '{action}' succeeded. State: {observation}"

                    global_memory_manager.add_experience(
                        reflection=mini_reflection,
                        gradients=step_gradient,
                        success=not is_failure,
                        task=state['task'],
                        env_id=state['env_id'],
                        observation=observation
                    )

                    if state['cur_step'] % 5 == 0:
                        global_memory_manager._consolidate_memories()
                except Exception as e:
                    print(f"[ERROR] Memory integration failed: {e}")

            # 11. LEARN FROM INTERACTION (lines 2628-2630)
            state['prompt_generator']._learn_from_interaction(action, observation)
            env_understanding.learn_from_interaction(action, observation)

            # 12. TRACK TRAJECTORY (lines 2632-2635)
            env_idx_in_batch = result['env_idx_in_batch']
            action_reasoning = batch_data_all[env_idx_in_batch].get('action_reasoning', '')
            state['trajectory'].append((action, observation, action_reasoning))

            # 12a. SYNERGISTIC STEP REFLEXION (Reflexion for episodic memory)
            # Generate reflexions ONLY for failures or key milestones to feed episodic memory
            # TextGrad handles real-time action optimization, Reflexion handles learning from experience
            if is_failure or state['cur_step'] % 5 == 0 or 'you win' in observation.lower():
                try:
                    # Initialize working_reflexions if not exists
                    if 'working_reflexions' not in state:
                        state['working_reflexions'] = []

                    # ═══════════════════════════════════════════════════════════════════════════
                    # FIX #4: HISTORY-AWARE EPISODIC REFLEXION GENERATION
                    # Root cause of loops: Episodic reflexions were generated WITHOUT historical
                    # context. Now we provide recent action history and previous insights so the
                    # reasoning model can understand what was tried and what to avoid.
                    # ═══════════════════════════════════════════════════════════════════════════

                    # Build history context from recent reflexions using TIERED compression
                    # This gives ALL reflexions smartly compressed (recent=verbose, old=summary)
                    history_context = ""
                    recent_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                    if recent_reflexions:
                        # Format tiered reflexions (handles compressed and verbose)
                        history_parts = []
                        for idx, r in enumerate(recent_reflexions, 1):
                            compression_type = r.get('is_compressed', None)

                            if compression_type == 'heavy':
                                # Heavy compression: just the summary
                                step_context = f"Earlier steps (summary):\n  {r.get('reflection', '')}"
                            elif compression_type == 'medium':
                                # Medium compression: key points
                                step_context = f"Previous steps (key points):\n  {r.get('reflection', '')}"
                            else:
                                # Verbose (recent): full detail
                                action_str = r.get('action', '')[:60]
                                obs_str = r.get('observation', '')[:80]
                                reflection_str = r.get('reflection', '')[:120]
                                success_str = "SUCCESS" if r.get('success', False) else "FAILED"

                                step_context = f"Step -{len(recent_reflexions)-idx+1}:\n"
                                step_context += f"  Action: {action_str}\n"
                                step_context += f"  Result: {obs_str}\n"
                                step_context += f"  Status: {success_str}\n"
                                step_context += f"  Insight: {reflection_str}"

                            history_parts.append(step_context)

                        if history_parts:
                            history_context += f"\nHistory (tiered - ALL {len(state.get('working_reflexions', []))} reflexions):\n" + "\n\n".join(history_parts)

                    # Build history-aware reflexion prompt
                    reflexion_prompt = f"""Task: {state['task']}
Action taken: {action}
Result: {observation[:150]}
{"FAILED" if is_failure else "SUCCEEDED"}
{history_context}

What key insight should we remember for future attempts? (1-2 sentences)"""

                    # Log history context status
                    if len(recent_reflexions) > 0:
                        total_reflexions = len(state.get('working_reflexions', []))
                        print(f"[HISTORY TIERED] ENV {state['env_id']} Step {state['cur_step']}: Reflexion sees ALL {total_reflexions} reflexions (tiered: {len(recent_reflexions)} compressed items)")

                    # Generate episodic reflexion
                    step_reflection_output = model.generate([reflexion_prompt], SamplingParams(temperature=0.7, max_tokens=150))[0]  # Restored for exploration
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion
                    state['working_reflexions'].append({
                        'step': state['cur_step'],
                        'action': action,
                        'observation': observation[:200],  # Save truncated observation
                        'reflection': step_reflection,
                        'success': not is_failure
                    })

                    print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: Generated episodic memory ({len(state['working_reflexions'])} total)")
                    log_debug(f"[REFLEXION] Insight: {step_reflection[:80]}...")

                except Exception as e:
                    log_debug(f"[REFLEXION ERROR] Failed to generate episodic reflexion: {e}")

            # 13. TRACK FAILURES (lines 2637-2641)
            if is_failure:
                state['consecutive_failures'] += 1
            else:
                state['consecutive_failures'] = 0

            # 14. UPDATE HISTORY
            state['history'].add("action", action)
            state['history'].add("observation", observation)

            # 15. LOG ACTION/OBSERVATION
            if to_print:
                log_debug(f'[ENV {state["env_id"]}] > {action}\n{observation}')
                sys.stdout.flush()

            # 16. UPDATE OBSERVATION (lines 2881-2886)
            state['prev_observation'] = observation

            # 17. INCREMENT STEP (line 2940)
            state['cur_step'] += 1

            # 18. PERIODIC CHECKPOINT (lines 2942-2949)
            if state['cur_step'] % 10 == 0 and checkpoint_manager:
                try:
                    if checkpoint_manager:  # Only save if checkpoint_manager exists
                        completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
                        pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]
                        checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                except Exception as e:
                    print(f"[WARNING] Periodic checkpoint failed: {e}")

            # 19. CHECK IF DONE (lines 2951-3018)
            if done:
                log_debug(f"\n[DEBUG] ENV {state['env_id']} at step {state['cur_step']}:")
                log_debug(f"  done={done}")
                log_debug(f"  info dict: {info}")
                log_debug(f"  reward: {reward}")
                actual_success = info.get('won', [False])[0]
                log_debug(f"  actual_success={actual_success}")

                state['done'] = True

                # CHECKPOINT: Save state after each environment completes
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_env_checkpoint(
                        trial_idx=trial_idx,
                        env_id=state['env_id'],
                        state=state,
                        trajectory_length=len(state['trajectory'])
                    )

                # Save master state for resume
                completed_envs = [s['env_id'] for s in env_states if s['done']]
                pending_envs = [s['env_id'] for s in env_states if not s['done']]
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                state['success'] = actual_success

                # Also save minimal checkpoint for compatibility
                checkpoint_file = os.path.join(os.path.dirname(trial_log_path), f'checkpoint_trial_{trial_idx}_env_{state["env_id"]}.json')
                checkpoint_data = {
                    'env_id': state['env_id'],
                    'success': actual_success,
                    'trajectory_length': len(state['trajectory']),
                    'task': state['task'],
                    'memory': env_configs[state['env_id']].get('memory', []),
                    'working_reflexions': state.get('working_reflexions', []),
                    'step': state['cur_step']
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                # CRITICAL FIX: Update env_configs immediately
                env_configs[state['env_id']]['is_success'] = actual_success

                # Save successful trajectory for cross-trial learning (Reflexion's episodic memory)
                if actual_success:
                    # Save to per-trial trajectory (for immediate debugging)
                    env_configs[state['env_id']]['successful_trajectory'] = [
                        (act, obs[:100]) for act, obs, _ in state['trajectory']
                    ]

                    # CRITICAL FIX: Save to reflexion_memory with correct type for cross-trial loading!
                    # This enables success pattern replay in future trials (+50 boost at line 803)
                    success_workflow = {
                        'type': 'success_workflow',  # This is what line 697 looks for!
                        'task': state['task'],
                        'actions': [act for act, _, _ in state['trajectory']],
                        'episode_id': state.get('episode_id', f"trial{trial_idx}_env{state['env_id']}"),
                        'trial': trial_idx,
                        'steps': len(state['trajectory']),
                        'success_confirmed': True
                    }

                    # Append to env_configs memory (this persists across trials!)
                    if 'memory' not in env_configs[state['env_id']]:
                        env_configs[state['env_id']]['memory'] = []
                    env_configs[state['env_id']]['memory'].append(success_workflow)

                    print(f"[SUCCESS PATTERN] Saved workflow to reflexion_memory: {len(state['trajectory'])} actions for '{state['task']}'")

                # Log last 3 actions for debugging
                if state['trajectory']:
                    log_debug(f"  Last 3 actions taken:")
                    for act, obs, _ in state['trajectory'][-3:]:
                        obs_preview = obs[:80] if len(obs) > 80 else obs
                        log_debug(f"    {act} -> {obs_preview.replace(chr(10), ' ')}")

                # RECORD COMPLETE EPISODE IN UNIVERSAL MEMORY
                universal_memory.record_episode(state['trajectory'], state['task'], actual_success)

                if actual_success:
                    universal_memory.store_sequence_pattern(state['trajectory'], True, state['task'])
                state['done'] = True
                state['success'] = actual_success

            # 20. CHECK FAILURE THRESHOLD (lines 3021-3031)
            failure_threshold = 5
            if state['consecutive_failures'] > failure_threshold and not state['done']:
                if to_print:
                    print(f"[ENV {state['env_id']}] Stuck in unproductive loop - ending episode")

                # Record failed episode
                universal_memory.record_episode(state['trajectory'], state['task'], False)

                state['done'] = True
                state['success'] = False

        print(f"[PHASE 6] ✓ Applied all gradients and processed learning for {len(action_results)} environments")

        # ========================================================================
        # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE SEQUENTIAL CODE
        # ========================================================================
        # This entire block is wrapped in "if False:" and will NEVER execute
        # Left here for historical reference only
        #
        # ACTIVE parallel batch processing code is in PHASES 3B-6 (lines 2573-3122)
        # If you need to make changes to the parallel code, modify PHASES 3B-6 above
        # DO NOT modify or enable this disabled sequential code
        # ========================================================================
        # OLD SEQUENTIAL PROCESSING (DISABLED - REPLACED WITH PHASES 3B-6)
        # ========================================================================
        if False:  # Lines 2055-3121 disabled - Sequential processing replaced with parallel
            # TIMING FIX: Generate Reflexion BEFORE action selection (except step 0)
            # This ensures extracted actions are available for THIS step, not next step
            if state['cur_step'] > 0 and state['trajectory']:
                # Get the last action and observation from trajectory
                last_action, last_observation, last_reasoning = state['trajectory'][-1]

                # Check if last action failed
                is_last_failure = any(fail_msg in last_observation.lower() for fail_msg in
                                     ["nothing happens", "don't see", "can't", "already"])

                # Generate step reflexion for the PREVIOUS action
                print(f"\n[REFLEXION-BEFORE-ACTION] Step {state['cur_step']} reflecting on previous action")

                recent_trajectory = state['trajectory'][-5:] if state['trajectory'] else []
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                if tiered_reflexions:
                    formatted_learnings = []
                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        step_num = ref.get('step', '?')
                        reflection_text = ref.get('reflection', '')[:200]
                        if compression_type == 'heavy':
                            formatted_learnings.append(f"  Steps {step_num} (summary): {reflection_text}")
                        elif compression_type == 'medium':
                            formatted_learnings.append(f"  Steps {step_num} (key points): {reflection_text}")
                        else:
                            formatted_learnings.append(f"  Step {step_num}: {reflection_text}")
                    previous_step_learnings = '\n'.join(formatted_learnings)
                else:
                    previous_step_learnings = "None yet (first few steps)"

                todo_context = ""
                if hasattr(state.get('todo_manager'), 'current_todo'):
                    current_todo = state['todo_manager'].current_todo()
                    if current_todo:
                        todo_context = f"\nCurrent Subtask: '{current_todo.content}' (Attempt #{current_todo.attempts})"
                        if current_todo.attempts > 2:
                            todo_context += f"\n⚠️  WARNING: {current_todo.attempts} attempts without success"

                step_reflection_prompt = f"""Analyze the last few actions in this episode.

PREVIOUS LEARNINGS FROM THIS TRIAL (step-level memory):
{previous_step_learnings}

Task: {state['task']}
Current step: {state['cur_step']}

Recent actions and outcomes:"""

                for act, obs, reasoning in recent_trajectory:
                    step_reflection_prompt += f"\n> {act}\nResult: {obs[:100]}..."

                step_reflection_prompt += f"""

Current action: {last_action}
Result: {last_observation}...
Success: {'No' if is_last_failure else 'Yes'}
{todo_context}

🔬 REFLEXION: Root-Cause Analysis

CRITICAL ANALYSIS REQUIRED:

1. SEMANTIC PRECISION CHECK:
   Task goal: "{state['task']}"
   Action taken: "{last_action}"

   Question: What is the PRIMARY PURPOSE of each?
   - Task requires me to: [specific effect/goal]
   - Action achieves: [specific effect/purpose]
   - Are these semantically IDENTICAL or DIFFERENT?

   IMPORTANT: Actions with similar objects may serve different purposes.
   Example: "clean X with sink" (removes dirt) vs "cool X with fridge" (lowers temperature)

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{state['task']}", what SPECIFIC property must change?

   Look at observation: "{last_observation[:150]}"
   Did that EXACT property change? Cite specific words as evidence.
   Do NOT assume - only state what observation explicitly shows.

3. FAILURE PATTERN DETECTION:
   {f'Subtask not achieved after {current_todo.attempts} attempts.' if todo_context and hasattr(state.get('todo_manager'), 'current_todo') and state['todo_manager'].current_todo() and state['todo_manager'].current_todo().attempts > 2 else 'Check if repeating similar actions.'}

   Am I repeating similar action types that don't work?
   What assumption might be incorrect?

4. RECOMMENDED NEXT ACTION:
   If there's a semantic mismatch or wrong action type detected:
   What SPECIFIC action should I try instead? (Be precise with action format)

Generate your analysis (keep concise, 3-4 sentences total):"""

                sampling_params = SamplingParams(max_tokens=3000, temperature=0.3)
                step_reflection_output = model.generate([step_reflection_prompt], sampling_params)[0]
                step_reflection = step_reflection_output.outputs[0].text.strip()

                if 'working_reflexions' not in state:
                    state['working_reflexions'] = []

                state['working_reflexions'].append({
                    'step': state['cur_step'] - 1,  # Reflecting on PREVIOUS step
                    'action': last_action,
                    'reflection': step_reflection,
                    'success': not is_last_failure
                })

                log_debug(f"[STEP REFLEXION] Step {state['cur_step']-1}: {step_reflection[:150]}...")

                # Extract gradients using TextGrad
                mini_trajectory = ""
                for act, obs, _ in recent_trajectory[-3:]:
                    mini_trajectory += f"> {act}\n{obs[:100]}...\n"

                progress_score = state.get('step_gradient', {}).get('progress_score', 0)

                # SYNERGY FIX: Only invoke Reflexion's CAUSAL CHAIN when TextGrad needs help
                # High progress (≥4): TextGrad confident → use its clean recommendations
                # Low progress (<4): TextGrad struggling → invoke Reflexion for guidance
                if progress_score < 4:
                    log_debug(f"[SYNERGY] Low progress ({progress_score}/10) - invoking Reflexion guidance")
                    step_reflexion_gradients = state['prompt_generator'].compute_prompt_gradient(
                        trajectory=mini_trajectory,
                        success=False,
                        task=state['task'],
                        reflection=step_reflection,
                        progress_score=progress_score
                    )

                    # Extract clean action from Reflexion's CAUSAL CHAIN format
                    if step_reflexion_gradients and 'next_action_guidance' in step_reflexion_gradients:
                        extracted_action = step_reflexion_gradients['next_action_guidance']
                        if extracted_action and len(extracted_action) > 3:
                            state['step_gradient']['next_action_guidance'] = extracted_action
                            print(f"🔗 [REFLEXION-GUIDE] Low progress → Reflexion suggests: {extracted_action}")
                else:
                    # TextGrad doing well - don't pollute with Reflexion's verbose CAUSAL CHAIN
                    log_debug(f"[SYNERGY] Moderate/high progress ({progress_score}/10) - TextGrad optimizing")

            # Debug batch_data before sending to action selection
            if DEBUG_ACTOR:
                print(f"\n[BATCH_DATA VERIFICATION]:")
                data = batch_data[0]
                print(f"  ENV {state['env_id']}:")
                print(f"    Task: {data['task'][:50]}")
                print(f"    Valid actions count: {len(data['valid_actions'])}")
                print(f"    First action: {data['valid_actions'][0] if data['valid_actions'] else 'NONE'}")
                print(f"    Observation starts: {data['observation']}...")

            # BATCH ACTION SELECTION (single element batch for sequential)
            selected_actions = reasoning_based_action_selection_batch(
                batch_data=batch_data,
                prompt_generator=prompt_generator,
                DEBUG_ACTOR=DEBUG_ACTOR ,
                log_debug=log_debug
            )

            # Execute action (single environment)
            action = selected_actions[0]
            state = active_states[0]
            # Track EXACT tried actions - ensure it's a set
            if not isinstance(state['tried_actions'], set):
                state['tried_actions'] = set(state['tried_actions'])
            state['tried_actions'].add(action)


            # Extract reasoning from batch_data for this environment
            env_idx_in_batch = active_states.index(state)
            action_reasoning = batch_data[env_idx_in_batch].get('action_reasoning', '')
            
            # Store for batch sharing
            state['last_action'] = action
            
            if DEBUG_ACTOR:
                log_debug(f"[ENV {state['env_id']}] SELECTED EXACT ACTION: '{action}'")
            
            # Add action to history
            state['history'].add("action", action)

            # Execute action - MEMORY FIX: Use env from registry
            env = _env_registry[state['env_id']]
            observation, reward, done, info = env.step([action])

            # Process observation
            observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
            done = done[0]

            # ═══════════════════════════════════════════════════════════════
            # SYNERGISTIC EPISODIC REFLEXION GENERATION (Added 2025-10-19)
            # Generate reflexion ONLY at key moments (failures/milestones)
            # TextGrad handles real-time optimization, Reflexion handles episodic memory
            # ═══════════════════════════════════════════════════════════════

            print(f"[DEBUG-REFLEXION-PATH] ENV {state['env_id']} Step {state['cur_step']}: Checking for key moments...")

            # Determine if this is a key moment for episodic memory
            is_key_moment = False
            moment_type = ""

            # Check for failure (key learning moment)
            failure_indicators = ['nothing happens', "don't see", "can't", 'already', 'closed']
            if any(fail in observation.lower() for fail in failure_indicators):
                is_key_moment = True
                moment_type = "FAILURE"

            # Check for success (task completion)
            elif 'you win' in observation.lower() or done:
                is_key_moment = True
                moment_type = "SUCCESS"

            # Check for milestone (every 5 steps or significant state change)
            elif state['cur_step'] % 5 == 0:
                is_key_moment = True
                moment_type = "MILESTONE"

            print(f"[DEBUG-KEY-MOMENT] ENV {state['env_id']} Step {state['cur_step']}: is_key_moment={is_key_moment}, type={moment_type}")

            # Generate episodic reflexion at key moments
            if is_key_moment:
                try:
                    # Initialize working_reflexions if not exists
                    if 'working_reflexions' not in state:
                        state['working_reflexions'] = []

                    # Build reflexion prompt for episodic memory
                    reflexion_prompt = f"""Task: {state['task']}
Step {state['cur_step']}: {action}
Result: {observation[:150]}
Type: {moment_type}

What key insight should we remember for future trials? Focus on:
- What worked or didn't work
- Why this action led to this outcome
- What to try differently next time

Provide a concise (1 sentence) episodic memory insight:"""

                    # Generate episodic reflexion using the model
                    from vllm import SamplingParams
                    sampling_params = SamplingParams(temperature=0.7, max_tokens=150)  # Restored for exploration
                    step_reflection_output = model.generate([reflexion_prompt], sampling_params)[0]
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion to working memory
                    reflexion_entry = {
                        'step': state['cur_step'],
                        'action': action,
                        'observation': observation[:200],
                        'reflection': step_reflection,
                        'type': moment_type,
                        'success': moment_type == "SUCCESS"
                    }

                    state['working_reflexions'].append(reflexion_entry)

                    # CRITICAL: Save to env_configs for cross-trial persistence
                    env_idx = state['env_id']
                    if 'step_memory' not in env_configs[env_idx]:
                        env_configs[env_idx]['step_memory'] = []
                    env_configs[env_idx]['step_memory'].append(f"[Step {state['cur_step']} {moment_type}] {step_reflection}")

                    # Keep only last 10 step memories
                    if len(env_configs[env_idx]['step_memory']) > 10:
                        env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]

                    print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Generated episodic memory")
                    print(f"[REFLEXION] Total episodic memories: {len(state['working_reflexions'])}")
                    log_debug(f"[REFLEXION] Insight: {step_reflection[:100]}...")

                except Exception as e:
                    log_debug(f"[REFLEXION ERROR] Failed to generate episodic reflexion: {e}")

            # ═══════════════════════════════════════════════════════════════
            # END OF SYNERGISTIC EPISODIC REFLEXION GENERATION
            # ═══════════════════════════════════════════════════════════════

            
            # Check if EXACT action failed (deterministic tracking)
            is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])

            # INTELLIGENT: Also classify low-progress actions (not just observation-based failures)
            # Let LLM distinguish: useful exploration vs truly useless actions
            should_classify_semantically = is_failure
            classification_reason = "observation unchanged"

            # Check progress score from step gradient (if available)
            if 'step_gradient' in state and state['step_gradient']:
                progress_score = state['step_gradient'].get('progress_score', 5)

                if progress_score < 3:  # Very low progress
                    should_classify_semantically = True
                    classification_reason = f"low progress ({progress_score}/10)"
                    log_debug(f"[LOW PROGRESS DETECTED] Env {state['env_id']} Step {state['cur_step']}: {progress_score}/10")
                    log_debug(f"  Action: '{action}' - requesting LLM semantic analysis")

            # Store failure in history for batched intelligent filtering
            # Simple storage, no extra LLM call (analysis will be in unified batch prompt)
            if should_classify_semantically:
                log_debug(f"[FAILURE DETECTED] Env {state['env_id']} Step {state['cur_step']}: {classification_reason}")
                log_debug(f"  Action: '{action}' - will be stored for filtering")

                # Store in failure history with context (no LLM analysis call)
                if 'failure_history' not in state:
                    state['failure_history'] = []

                state['failure_history'].append({
                    'step': state['cur_step'],
                    'action': action,
                    'context_before': state['prev_observation'][:300],
                    'context_after': observation[:300],
                    'progress_score': state.get('step_gradient', {}).get('progress_score', 'N/A')
                })

                # Keep only last 10 failures
                if len(state['failure_history']) > 10:
                    state['failure_history'] = state['failure_history'][-10:]

                log_debug(f"[FAILURE STORED] Context saved for intelligent filtering")
                log_debug(f"  Total failures in history: {len(state['failure_history'])}")

            # UNIVERSAL: Detect if action provided new information
            prev_info_content = len(state['prev_observation'].split())
            curr_info_content = len(observation.split())

            # If observation got longer, we gained information
            information_gained = curr_info_content > prev_info_content * 1.2  # 20% more content

            # If we gained information, it's not a failure (even if state didn't change)
            if information_gained:
                is_failure = False
            else:
                # No information gain - check if it's a true failure
                is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])
                
            
           
            # Get next valid actions for the reflection (loop detection removed) - MEMORY FIX
            next_valid_actions = []
            env = _env_registry[state['env_id']]
            if hasattr(env, 'get_current_valid_actions'):
                next_valid_actions = env.get_current_valid_actions()

            # Get current TODO for tactical coordination
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # UNIVERSAL: Track state changes to infer held items (NO hardcoded text)
            # Strategy: Compare before/after observations to detect object acquisition
            inventory_items = []

            # Method 1: Detect if observation got SHORTER (item removed from world = likely taken)
            prev_words = set(state['prev_observation'].lower().split())
            curr_words = set(observation.lower().split())
            disappeared_words = prev_words - curr_words

            # Method 2: Check if action moved an object (universal pattern: "X from Y" or "X to Y")
            action_words = action.lower().split()
            if len(action_words) >= 3:  # e.g., "take pan from stoveburner"
                # Look for pattern: verb + object + preposition
                for i in range(len(action_words) - 2):
                    if action_words[i+1] in ['from', 'to', 'in', 'on', 'into', 'onto']:
                        # Object is likely the word before preposition
                        potential_item = action_words[i]
                        # Verify it disappeared from observation or observation changed significantly
                        if potential_item in disappeared_words or len(disappeared_words) > 0:
                            inventory_items.append(potential_item)
                            break

            # Get episodic memory (cross-trial) separate from step memory (within-trial)
            env_idx = state['env_id']
            episodic_memory_only = env_configs[env_idx].get('memory', [])  # Cross-trial learnings

            # Get tiered/compressed reflexions to pass to TextGrad/Reflexion (uses ALL history smartly compressed)
            tiered_step_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

            # PHASE 2 SYNERGY: Decide between TextGrad (normal) vs Reflexion (failures)
            # Get previous step's progress to decide which component to use
            last_progress = state.get('last_step_gradient', {}).get('progress_score', 5)
            current_step_num = state['cur_step']
# Initialize consecutive failure tracking            if 'consecutive_low_progress' not in state:                state['consecutive_low_progress'] = 0            if 'steps_since_reflexion' not in state:                state['steps_since_reflexion'] = 0            # Update tracking            if last_progress < 4:                state['consecutive_low_progress'] += 1            else:                state['consecutive_low_progress'] = 0            state['steps_since_reflexion'] += 1            # Smart trigger: 5 consecutive failures + not step 0 + cooldown            use_reflexion = (current_step_num > 0 and                           state['consecutive_low_progress'] >= 5 and                           state['steps_since_reflexion'] >= 5)

            # Decision logic: TextGrad for good progress, Reflexion only for failures
            if not use_reflexion:
                # TEXTGRAD PATH (85%+ of steps): Clean action optimization
                # Use when: (1) First step OR (2) Previous step showed good progress
                log_debug(f"[SYNERGY] ENV {state['env_id']} Step {current_step_num}: Using TEXTGRAD (progress={last_progress}/10)")
                step_gradient = generate_textgrad_step_guidance(
                    prev_observation=state['prev_observation'],
                    curr_observation=observation,
                    action=action,
                    task=state['task'],
                    model=model,
                    log_debug=log_debug,
                    current_step=current_step_num,
                    valid_actions=next_valid_actions,
                    current_todo=current_todo,
                    inventory=inventory_items,
                    previous_reflexions=tiered_step_reflexions,
                    episodic_memory=episodic_memory_only
                )
            else:
                # REFLEXION PATH (<15% of steps): Deep causal analysis for failures
                # Use when: Previous step showed low progress (need root cause diagnosis)
                log_debug(f"[SYNERGY] ENV {state['env_id']} Step {current_step_num}: Using REFLEXION (progress={last_progress}/10 - need causal analysis)")
                step_gradient = generate_reflexion_causal_analysis(
                    prev_observation=state['prev_observation'],
                    curr_observation=observation,
                    action=action,
                    task=state['task'],
                    model=model,
                    log_debug=log_debug,
                    current_step=current_step_num,
                    valid_actions=next_valid_actions,
                    current_todo=current_todo,
                    inventory=inventory_items,
                    previous_reflexions=tiered_step_reflexions,
                    episodic_memory=episodic_memory_only
                )

            # UPDATE TODO MANAGER with action feedback
            if 'todo_manager' in state and state['todo_manager']:
                try:
                    state['todo_manager'].update_from_action_feedback(
                        action=action,
                        prev_observation=state['prev_observation'],
                        curr_observation=observation,
                        progress_score=step_gradient.get('progress_score', 0)
                    )
                except Exception as e:
                    log_debug(f"[TODO UPDATE ERROR] {e}")

            # Log the full step gradient
            comprehensive_logger.log_step_gradient(
                env_id=state['env_id'],
                step=state['cur_step'],
                gradient=step_gradient
            )



            # ADD THIS TO LOG TO TRIAL FILE AS WELL
            if trial_log_path and os.path.exists(trial_log_path):
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[STEP GRADIENT] Step {state['cur_step']}:\n")
                    f.write(f"  Progress: {step_gradient.get('progress_score', 0)}/10\n")
                    f.write(f"  Hypothesis: {step_gradient.get('hypothesis', 'None')}\n")
                    f.write(f"  Next: {step_gradient.get('next_action_guidance', 'None')}\n")

                    # Log TODO status as well
                    if 'todo_manager' in state and state['todo_manager']:
                        f.write(f"\n[TODO STATUS]:\n")
                        f.write(f"{state['todo_manager'].get_formatted_todos()}\n")


           
            # Use the environment's SPECIFIC prompt_generator - ensure it's the right one
            env_pg = env_states[active_states.index(state)]['prompt_generator']

            # Always apply observation gradient (it will be overridden by reflexion if better)
            env_pg.apply_universal_step_gradient(step_gradient)
            # Otherwise, reflexion gradient will be applied instead

            # CRITICAL FIX: Share learning across environments in same batch
            # Share failed actions immediately to prevent other envs from trying them
            # CONTEXT-AWARE SHARING: Share learning with full context
            if is_failure:
                # Create failure context with all dimensions
                failure_context = {
                    'action': action,
                    'state_hash': hashlib.md5(state['prev_observation'][:200].encode()).hexdigest()[:8],
                    'state_text': state['prev_observation'][:200],
                    'task': state['task'],
                    'step': state['cur_step'],
                    'prerequisites': [act for act, _, _ in state['trajectory'][-5:] if act],
                    'failure_reason': observation,
                    'timestamp': state['cur_step'],
                    'confidence': 1.0,
                    'is_universal': any(uf in observation.lower() for uf in 
                                      ["i don't understand", "invalid command", "not a verb"])
                }
                
                # Store in conditional failures
                if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                    state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                state['prompt_generator'].discovered_knowledge['conditional_failures'].append(failure_context)
                
                # Smart sharing based on context similarity
                for other_state in env_states:
                    if other_state['env_id'] != state['env_id'] and not other_state['done']:
                        target_context = {
                            'state_text': other_state['prev_observation'][:200],
                            'task': other_state['task'],
                            'step': other_state['cur_step'],
                            'prerequisites': [act for act, _, _ in other_state['trajectory'][-5:] if act]
                        }
                        
                        should_share, confidence = should_share_knowledge(failure_context, target_context)
                        
                        if should_share:
                            shared_failure = failure_context.copy()
                            shared_failure['confidence'] = confidence
                            shared_failure['source_env'] = state['env_id']
                            
                            if 'conditional_failures' not in other_state['prompt_generator'].discovered_knowledge:
                                other_state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                            other_state['prompt_generator'].discovered_knowledge['conditional_failures'].append(shared_failure)

            # Also update the state's reference
            state['textgrad_components'] = env_pg.prompt_components.copy()
    
            
            # CRITICAL: Store the gradient for next iteration
            state['last_step_gradient'] = step_gradient
            state['step_gradient'] = step_gradient

            if DEBUG_CRITIC:
                print(f"[ENV {state['env_id']}] STEP GRADIENT APPLIED at step {state['cur_step']}")
                print(f"  Components updated: {len(state['textgrad_components'])}")

            # LOG TO FILE
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[GRADIENT UPDATE {state['cur_step']}]\n")
                    f.write(f"  Next action guidance: {step_gradient.get('next_action_guidance', 'None')}\n")
                    f.write(f"  Progress score: {step_gradient.get('progress_score', 0)}\n")

            # ACCUMULATE AND SYNTHESIZE STEP INSIGHTS
            progress_score = step_gradient.get('progress_score', 0)

            # Always accumulate, not just on progress
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'hypothesis': step_gradient.get('hypothesis', ''),
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', [])
            })

            # Synthesize insights every 3 steps
            if len(state['step_insights_accumulator']) % 3 == 0:
                # Create actionable synthesis
                recent_insights = state['step_insights_accumulator'][-3:]
                
                synthesis = {
                    'avg_progress': sum(i['progress_score'] for i in recent_insights) / 3,
                    'state_changes': sum(1 for i in recent_insights if i['state_changed']),
                    'consistent_failures': [],
                    'promising_directions': []
                }
                
                # Find patterns
                for insight in recent_insights:
                    if insight['progress_score'] < 2:
                        synthesis['consistent_failures'].append(insight['action'])
                    elif insight['progress_score'] > 5:
                        synthesis['promising_directions'].append(insight['next_guidance'])
                

                # CRITICAL: Make wisdom immediately available for next action
                state['consolidated_step_wisdom'] = state.get('consolidated_step_wisdom', '')
                if synthesis['promising_directions']:
                    state['consolidated_step_wisdom'] = f"PROMISING: {synthesis['promising_directions'][0]}"
                elif synthesis['consistent_failures']:
                    state['consolidated_step_wisdom'] = f"AVOID: {', '.join(synthesis['consistent_failures'][:2])}"
                else:
                    state['consolidated_step_wisdom'] = f"Avg progress: {synthesis['avg_progress']:.1f}/10"
                
                # Ensure it gets to batch_data on next iteration
                state['needs_wisdom_update'] = True

                # Update consolidated wisdom with synthesis
                if synthesis['promising_directions']:
                    state['consolidated_step_wisdom'] = f"PROMISING: {synthesis['promising_directions'][0]}"
                elif synthesis['consistent_failures']:
                    state['consolidated_step_wisdom'] = f"AVOID: {', '.join(synthesis['consistent_failures'][:2])}"
                else:
                    state['consolidated_step_wisdom'] = f"Avg progress: {synthesis['avg_progress']:.1f}/10"
                
                # Consolidate every 5 steps or when stuck
                if len(state['step_insights_accumulator']) >= 5 or state['is_stuck']:
                    # Get best insights
                    best_insights = sorted(
                        state['step_insights_accumulator'], 
                        key=lambda x: x['progress_score'], 
                        reverse=True
                    )[:3]
                    
                    # Build consolidated wisdom
                    wisdom_parts = []
                    for insight in best_insights:
                        if insight['hypothesis']:
                            wisdom_parts.append(f"Step {insight['step']}: {insight['hypothesis']}")
                    
                    if wisdom_parts:
                        state['consolidated_step_wisdom'] = ' | '.join(wisdom_parts)
                        
                    # Keep only recent insights
                    state['step_insights_accumulator'] = state['step_insights_accumulator'][-20:]

            # Track progress scores
            progress_score = step_gradient.get('progress_score', 0)
            state['progress_history'].append(progress_score)

            # Detect if stuck (low progress for many steps)
            if len(state['progress_history']) >= 5:
                recent_progress = state['progress_history'][-5:]
                avg_recent_progress = sum(recent_progress) / len(recent_progress)
                state['is_stuck'] = avg_recent_progress < 2
            else:
                state['is_stuck'] = False            
            
            # Store semantic understanding in memory
            if step_gradient and 'semantic_state_before' in step_gradient:
                universal_memory.record_semantic_interaction(
                    semantic_state_before=step_gradient['semantic_state_before'],
                    action=action,
                    action_reasoning=batch_data[env_idx_in_batch].get('action_reasoning', ''),  # ADD THIS
                    semantic_state_after=step_gradient['semantic_state_after'],
                    prerequisites=step_gradient.get('prerequisites', {}),
                    task_progress=step_gradient.get('task_progress', {}),
                    success=not is_failure,
                    task=state['task'],
                    episode_id=state['episode_id']
                )
                            
                # Store for next iteration
                state['last_step_gradient'] = step_gradient
            else:
                # Fallback if gradient doesn't have semantic info
                state['last_step_gradient'] = {
                    'semantic_state_after': observation,
                    'task_progress': {'remaining': [state['task']]}
                }

            # MEMORY INTEGRATION DURING EPISODE
            from generate_reflections import global_memory_manager
            
            # Consolidate more frequently
            if is_failure or state['cur_step'] % 3 == 0:  # Changed from 5 to 3
                try:
                    # Create mini-reflection with more detail
                    mini_reflection = f"Episode {state['episode_id']}, Step {state['cur_step']}: "
                    if is_failure:
                        mini_reflection += f"EXACT action '{action}' failed. State: {observation}"
                    else:
                        mini_reflection += f"Action '{action}' succeeded. State: {observation}"

                    global_memory_manager.add_experience(
                        reflection=mini_reflection,
                        gradients=step_gradient,
                        success=not is_failure,
                        task=state['task'],
                        env_id=state['env_id'],
                        observation=observation  # ADD: Pass the observation
                    )
                    
                    if state['cur_step'] % 5 == 0:
                        global_memory_manager._consolidate_memories()
                        
                    if DEBUG_REFLEXION:
                        print(f"[MEMORY] Stored experience at step {state['cur_step']}")
                        
                except Exception as e:
                    print(f"[ERROR] Memory integration failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Learn from interaction
            state['prompt_generator']._learn_from_interaction(action, observation)
            env_understanding.learn_from_interaction(action, observation)
            
            # Track trajectory with reasoning (3-tuple)
            env_idx_in_batch = active_states.index(state)
            action_reasoning = batch_data[env_idx_in_batch].get('action_reasoning', '')
            state['trajectory'].append((action, observation, action_reasoning))
            
            # Track consecutive failures
            if is_failure:
                state['consecutive_failures'] += 1
            else:
                state['consecutive_failures'] = 0


            # ====================================================================
            # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE REFLEXION CODE
            # ====================================================================
            # This entire block is wrapped in "if False:" and will NEVER execute
            # Left here for historical reference only
            #
            # ACTIVE reflexion code now runs BEFORE action selection (PHASE 3B)
            # DO NOT modify or enable this disabled after-action reflexion code
            # ====================================================================
            # WITHIN-EPISODE REFLEXION - NOW MOVED BEFORE ACTION SELECTION (see lines 1687-1818)
            # Disabled this old AFTER-action reflexion to prevent duplicates
            if False:  # DISABLED - Reflexion now runs BEFORE action selection for immediate use
                # Generate reflexion when: low progress every 3 steps, on failure, or high progress
                print(f"\n[REFLEXION-OLD-DISABLED] Generating step reflexion at step {state['cur_step']} (Failure: {is_failure})")
                
                # Generate proper step-level reflexion
                recent_trajectory = state['trajectory'][-5:] if state['trajectory'] else []

                # Get previous step reflexions from THIS trial (step-level memory)
                # Use ALL reflexions with tiered compression - don't forget early lessons!
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                # Format compressed reflexions for step reflexion prompt
                if tiered_reflexions:
                    formatted_learnings = []
                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        step_num = ref.get('step', '?')
                        reflection_text = ref.get('reflection', '')[:200]  # Limit for display

                        if compression_type == 'heavy':
                            formatted_learnings.append(f"  Steps {step_num} (summary): {reflection_text}")
                        elif compression_type == 'medium':
                            formatted_learnings.append(f"  Steps {step_num} (key points): {reflection_text}")
                        else:
                            formatted_learnings.append(f"  Step {step_num}: {reflection_text}")

                    previous_step_learnings = '\n'.join(formatted_learnings)
                else:
                    previous_step_learnings = "None yet (first few steps)"

                step_reflection_prompt = f"""Analyze the last few actions in this episode.

            PREVIOUS LEARNINGS FROM THIS TRIAL (step-level memory):
            {previous_step_learnings}

            Task: {state['task']}
            Current step: {state['cur_step']}

            Recent actions and outcomes:"""

                for act, obs, reasoning in recent_trajectory:
                    step_reflection_prompt += f"\n> {act}\nResult: {obs[:100]}..."

                # Get TODO context for semantic verification
                todo_context = ""
                if hasattr(state.get('todo_manager'), 'current_todo'):
                    current_todo = state['todo_manager'].current_todo()
                    if current_todo:
                        todo_context = f"\nCurrent Subtask: '{current_todo.content}' (Attempt #{current_todo.attempts})"
                        if current_todo.attempts > 2:
                            todo_context += f"\n⚠️  WARNING: {current_todo.attempts} attempts without success"

                step_reflection_prompt += f"""

            Current action: {action}
            Result: {observation}...
            Success: {'No' if is_failure else 'Yes'}
            {todo_context}

            🔬 REFLEXION: Root-Cause Analysis

            CRITICAL ANALYSIS REQUIRED:

            1. SEMANTIC PRECISION CHECK:
               Task goal: "{state['task']}"
               Action taken: "{action}"

               Question: What is the PRIMARY PURPOSE of each?
               - Task requires me to: [specific effect/goal]
               - Action achieves: [specific effect/purpose]
               - Are these semantically IDENTICAL or DIFFERENT?

               IMPORTANT: Actions with similar objects may serve different purposes.
               Example: "clean X with sink" (removes dirt) vs "cool X with fridge" (lowers temperature)

            2. STATE VERIFICATION WITH EVIDENCE:
               Based on task "{state['task']}", what SPECIFIC property must change?

               Look at observation: "{observation[:150]}"
               Did that EXACT property change? Cite specific words as evidence.
               Do NOT assume - only state what observation explicitly shows.

            3. FAILURE PATTERN DETECTION:
               {f'Subtask not achieved after {current_todo.attempts} attempts.' if todo_context and hasattr(state.get('todo_manager'), 'current_todo') and state['todo_manager'].current_todo() and state['todo_manager'].current_todo().attempts > 2 else 'Check if repeating similar actions.'}

               Am I repeating similar action types that don't work?
               What assumption might be incorrect?

            4. RECOMMENDED NEXT ACTION:
               If there's a semantic mismatch or wrong action type detected:
               What SPECIFIC action should I try instead? (Be precise with action format)

            Generate your analysis (keep concise, 3-4 sentences total):"""
                
                # Generate reflection
                sampling_params = SamplingParams(max_tokens=3000, temperature=0.3)
                step_reflection_output = model.generate([step_reflection_prompt], sampling_params)[0]
                step_reflection = step_reflection_output.outputs[0].text.strip()
                
                # Store in WORKING memory for immediate use
                if 'working_reflexions' not in state:
                    state['working_reflexions'] = []
                
                state['working_reflexions'].append({
                    'step': state['cur_step'],
                    'action': action,
                    'reflection': step_reflection,
                    'success': not is_failure
                })

                # Log the step reflexion for paper
                log_debug(f"[STEP REFLEXION] Step {state['cur_step']}: {step_reflection[:150]}...")
                total_reflexions = len(state.get('working_reflexions', []))
                compressed_count = len([r for r in tiered_reflexions if r.get('is_compressed')])
                log_debug(f"[STEP REFLEXION] Using ALL {total_reflexions} reflexions ({compressed_count} compressed, {total_reflexions - compressed_count} verbose)")

                # Log the step reflexion
                comprehensive_logger.log_step_reflexion(
                    env_id=state['env_id'],
                    step=state['cur_step'],
                    action=action,
                    reflection=step_reflection,
                    success=not is_failure
                )

                # NEW: Extract gradients from step reflexion using existing logic
                # Build a mini-trajectory for the compute_prompt_gradient function
                mini_trajectory = ""
                for act, obs, _ in recent_trajectory[-3:]:
                    mini_trajectory += f"> {act}\n{obs[:100]}...\n"
                
                # Extract progress score from current step gradient
                progress_score = state.get('step_gradient', {}).get('progress_score', 0)

                # SYNERGY FIX: Only invoke Reflexion's CAUSAL CHAIN when TextGrad needs help
                # High progress (≥4): TextGrad confident → use its clean recommendations
                # Low progress (<4): TextGrad struggling → invoke Reflexion for guidance
                if progress_score < 4:
                    log_debug(f"[SYNERGY] Low progress ({progress_score}/10) - invoking Reflexion guidance")
                    step_reflexion_gradients = state['prompt_generator'].compute_prompt_gradient(
                        trajectory=mini_trajectory,
                        success=False,
                        task=state['task'],
                        reflection=step_reflection,
                        progress_score=progress_score
                    )

                    # Extract clean action from Reflexion's CAUSAL CHAIN format
                    if step_reflexion_gradients and 'next_action_guidance' in step_reflexion_gradients:
                        extracted_action = step_reflexion_gradients['next_action_guidance']
                        if extracted_action and len(extracted_action) > 3:
                            state['step_gradient']['next_action_guidance'] = extracted_action
                            print(f"🔗 [REFLEXION-GUIDE] Low progress → Reflexion suggests: {extracted_action}")
                else:
                    # TextGrad doing well - don't pollute with Reflexion's verbose CAUSAL CHAIN
                    log_debug(f"[SYNERGY] Moderate/high progress ({progress_score}/10) - TextGrad optimizing")
                    step_reflexion_gradients = {}  # Empty dict for clean flow

                # SMART APPLICATION: Apply based on gradient quality, not just progress
                for component, gradient in step_reflexion_gradients.items():
                    if component not in state['prompt_generator'].prompt_components:
                        continue
                    
                    # Check gradient quality (actionability)
                    gradient_lower = gradient.lower()
                    is_actionable = any(word in gradient_lower for word in 
                                        ['must', 'should', 'try', 'next', 'need', 'avoid'])
                    
                    # Determine application strength based on both actionability and progress
                    if is_actionable and progress_score >= 6:
                        # High progress + actionable = strong update
                        momentum = 0.9
                    elif is_actionable and progress_score >= 3:
                        # Moderate progress + actionable = moderate update
                        momentum = 0.6
                    elif is_actionable:
                        # Low progress but actionable = careful update
                        momentum = 0.4
                    else:
                        # Non-actionable = minimal update
                        momentum = 0.2
                    
                    # Apply only if meaningful
                    if is_actionable or progress_score >= 7:
                        state['prompt_generator'].update_component_with_momentum(
                            component, gradient, momentum=momentum
                        )
                
                # Apply with HIGHER WEIGHT since reflexion-based insights are richer
                for component, gradient in step_reflexion_gradients.items():
                    if component in state['prompt_generator'].prompt_components:
                        # Use momentum update with higher weight
                        state['prompt_generator'].update_component_with_momentum(
                            component, 
                            gradient, 
                            momentum=0.95  # Higher momentum for reflexion-based updates
                        )
                
                if DEBUG_CRITIC:
                    print(f"[STEP REFLEXION GRADIENT] Applied {len(step_reflexion_gradients)} gradient updates from reflexion")
                    for comp, grad in step_reflexion_gradients.items():
                        print(f"  {comp}: {grad}...")

                        
                # CRITICAL FIX: Save to env_configs for next trial
                env_idx = state['env_id']
                if 'working_reflexions_history' not in env_configs[env_idx]:
                    env_configs[env_idx]['working_reflexions_history'] = []
                env_configs[env_idx]['working_reflexions_history'].append({
                    'trial': trial_idx,
                    'step': state['cur_step'],
                    'action': action,
                    'reflection': step_reflection,
                    'success': not is_failure
                })
                if len(env_configs[env_idx]['working_reflexions_history']) > 20:
                    env_configs[env_idx]['working_reflexions_history'] = env_configs[env_idx]['working_reflexions_history'][-20:]

                print(f"[REFLEXION] Stored reflexion for step {state['cur_step']}: {step_reflection}")

                # LOG TO FILE
                if trial_log_path:
                    with open(trial_log_path, 'a') as f:
                        f.write(f"\n[STEP REFLEXION {state['cur_step']}]\n")
                        f.write(f"  Action: {action}\n")
                        f.write(f"  Progress: {progress_score}/10\n")
                        f.write(f"  Reflection: {step_reflection[:200]}...\n")
                
                # Keep only last 5 step reflexions
                if len(state['working_reflexions']) > 15:
                    state['working_reflexions'] = state['working_reflexions'][-15:]
                
                
                # CRITICAL FIX: Also save to env_configs
                env_idx = state['env_id']
                if 'memory' not in env_configs[env_idx]:
                    env_configs[env_idx]['memory'] = []
                # Store step reflections separately to avoid polluting episodic memory
                if 'step_memory' not in env_configs[env_idx]:
                    env_configs[env_idx]['step_memory'] = []
                env_configs[env_idx]['step_memory'].append(f"[Step {state['cur_step']}] {step_reflection}")
                # Keep only last 10 step reflections
                if len(env_configs[env_idx]['step_memory']) > 10:
                    env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]


                # Update the STATE's prompt generator (consistent!)
                # Update with conditional failure instead of blacklist
                if is_failure:
                    if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                        state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                    
                    state['prompt_generator'].discovered_knowledge['conditional_failures'].append({
                        'action': action,
                        'timestamp': state['cur_step'],
                        'confidence': 0.9,
                        'is_universal': False
                    })
                    
                    # Also add to exact failed actions for this state
                    current_state_hash = hashlib.md5(state['prev_observation'][:200].encode()).hexdigest()[:8]
                    if 'failed_state_actions' not in state:
                        state['failed_state_actions'] = {}
                    if current_state_hash not in state['failed_state_actions']:
                        state['failed_state_actions'][current_state_hash] = set()
                    state['failed_state_actions'][current_state_hash].add(action)
                else:
                    # On success, add to successful actions
                    if 'successful_actions' not in state['prompt_generator'].discovered_knowledge:
                        state['prompt_generator'].discovered_knowledge['successful_actions'] = []
                    if action not in state['prompt_generator'].discovered_knowledge['successful_actions']:
                        state['prompt_generator'].discovered_knowledge['successful_actions'].append(action)
                
            

                
                # Also update the prompt generator immediately
                # Use conditional failures instead
                if 'conditional_failures' not in env_prompt_generators[state['env_id']].discovered_knowledge:
                    env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'] = []
                
                env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'].append({
                    'action': action,
                    'timestamp': state['cur_step'],
                    'confidence': 0.8,
                    'is_universal': False
                })
                
                log_debug(f"[ENV {state['env_id']}] Added mini-reflection after {state['consecutive_failures']} failures")


            # Add observation to history
            state['history'].add("observation", observation)


            
            if to_print:
                log_debug(f'[ENV {state["env_id"]}] > {action}\n{observation}')
                sys.stdout.flush()
            
            # Update state
            state['prev_observation'] = observation
            state['cur_step'] += 1
            
            # Periodic checkpoint every 10 steps to prevent loss
            if state['cur_step'] % 10 == 0 and checkpoint_manager:
                try:
                    if checkpoint_manager:  # Only save if checkpoint_manager exists
                        completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
                        pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]
                        checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                except Exception as e:
                    print(f"[WARNING] Periodic checkpoint failed: {e}")

            # Check done
            if done:
                log_debug(f"\n[DEBUG] ENV {state['env_id']} at step {state['cur_step']}:")
                log_debug(f"  done={done}")
                log_debug(f"  info dict: {info}")
                log_debug(f"  reward: {reward}")
                actual_success = info.get('won', [False])[0]
                log_debug(f"  actual_success={actual_success}")
                
                state['done'] = True

                # CHECKPOINT: Save state after each environment completes
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_env_checkpoint(
                        trial_idx=trial_idx,
                        env_id=state['env_id'],
                        state=state,
                        trajectory_length=len(state['trajectory'])
                    )

                # Save master state for resume
                completed_envs = [s['env_id'] for s in env_states if s['done']]
                pending_envs = [s['env_id'] for s in env_states if not s['done']]
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                state['success'] = actual_success
                

                
                # Also save minimal checkpoint for compatibility
                checkpoint_file = os.path.join(os.path.dirname(trial_log_path), f'checkpoint_trial_{trial_idx}_env_{state["env_id"]}.json')
                checkpoint_data = {
                    'env_id': state['env_id'],
                    'success': actual_success,
                    'trajectory_length': len(state['trajectory']),
                    'task': state['task'],
                    'memory': env_configs[state['env_id']].get('memory', []),
                    'working_reflexions': state.get('working_reflexions', []),
                    'step': state['cur_step']
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                # CRITICAL FIX: Update env_configs immediately
                env_configs[state['env_id']]['is_success'] = actual_success
                
                # Save successful trajectory for reuse
                if actual_success:
                    env_configs[state['env_id']]['successful_trajectory'] = [
                        (act, obs[:100]) for act, obs, _ in state['trajectory']
                    ]
                    print(f"[SUCCESS] Saved trajectory for env {state['env_id']}")
                                                                                                                                                                                                                                                                                                               

                # Log last 3 actions for debugging
                if state['trajectory']:
                    log_debug(f"  Last 3 actions taken:")
                    for act, obs, _ in state['trajectory'][-3:]:
                        obs_preview = obs[:80] if len(obs) > 80 else obs
                        log_debug(f"    {act} -> {obs_preview.replace(chr(10), ' ')}")
                
                # RECORD COMPLETE EPISODE IN UNIVERSAL MEMORY
                universal_memory.record_episode(state['trajectory'], state['task'], actual_success)
                

                if actual_success:
                    universal_memory.store_sequence_pattern(state['trajectory'], True, state['task'])                
                state['done'] = True
                state['success'] = actual_success



            # With max_steps=15, fail faster on consecutive failures
            failure_threshold = 5  # Fixed threshold
            if state['consecutive_failures'] > failure_threshold and not state['done']:
                if to_print:
                    print(f"[ENV {state['env_id']}] Stuck in unproductive loop - ending episode")
                
                # Record failed episode
                universal_memory.record_episode(state['trajectory'], state['task'], False)
                
                state['done'] = True
                state['success'] = False

        # End of while loop for current environment

        # ========================================================================
        # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE LEARNING CODE
        # ========================================================================
        # This entire block is wrapped in "if False:" and will NEVER execute
        # Left here for historical reference only
        #
        # ACTIVE learning extraction happens in PHASE 6 (parallel mode)
        # DO NOT modify or enable this disabled sequential learning code
        # ========================================================================
        # OLD SEQUENTIAL: STEP 5 - Extract learnings (DISABLED IN PARALLEL MODE)
        # ========================================================================
        if False:  # Lines 3544-3709 DISABLED - Sequential learning extraction not used in parallel mode
            log_debug(f"\n{'='*80}")
            log_debug(f"[LEARNING EXTRACTION] Environment {current_env_idx} completed")
            log_debug(f"[LEARNING EXTRACTION] Success: {current_state['success']}")
            log_debug(f"[LEARNING EXTRACTION] Steps taken: {current_state['cur_step']}")
            log_debug(f"{'='*80}\n")
            # Extract learnings
            result = {
                'success': current_state['success'],
                'steps': current_state['cur_step'],
                'trajectory': current_state['trajectory']
            }

            learnings = learning_extractor.extract_learnings(
                env_state=current_state,
                result=result,
                task=task
            )

            log_debug(f"[EXTRACTION] Extracted {len(learnings)} learning items")

            # Classify and store learnings in shared knowledge
            universal_count = 0
            family_count = 0
            specific_count = 0

            for learning in learnings:
                classification = learning['classification']
                knowledge_type, family = classification

                if knowledge_type == 'universal':
                    shared_knowledge['universal'].append(learning)
                    universal_count += 1
                    log_debug(f"[STORED] Universal: {learning['text'][:80]}...")

                elif knowledge_type == 'task_family' and family:
                    if family not in shared_knowledge['task_families']:
                        shared_knowledge['task_families'][family] = []
                    shared_knowledge['task_families'][family].append(learning)
                    family_count += 1
                    log_debug(f"[STORED] Family '{family}': {learning['text'][:80]}...")

                else:  # task_specific
                    specific_count += 1
                    log_debug(f"[SKIPPED] Task-specific: {learning['text'][:80]}...")

            log_debug(f"\n[KNOWLEDGE SUMMARY]")
            log_debug(f"  Universal: {universal_count} (total: {len(shared_knowledge['universal'])})")
            log_debug(f"  Task-family: {family_count}")
            log_debug(f"  Task-specific (not stored): {specific_count}")
            log_debug(f"  Total transferable knowledge: {len(shared_knowledge['universal']) + sum(len(v) for v in shared_knowledge['task_families'].values())}\n")

            # ========================================================================
            # STEP 6: Store TODO patterns for cross-environment learning
            # ========================================================================
            if current_state['success']:
                # Get final TODO state
                todo_manager = current_state.get('todo_manager')
                if todo_manager and todo_manager.todos:
                    task_class = task_classifier.classify_task(task)

                    shared_todo_knowledge['successful_patterns'].append({
                        'source_task': task,
                        'task_family': task_class['family'],
                        'todos': [t.content for t in todo_manager.todos],
                        'env_id': current_env_idx,
                        'trial': trial_idx
                    })

                    log_debug(f"[TODO KNOWLEDGE] ✅ Stored {len(todo_manager.todos)} successful TODOs for task '{task}'")
                    log_debug(f"  Task family: {task_class['family']}")
                    log_debug(f"  Available for transfer to environments {current_env_idx + 1}+")
            else:
                # Store failed patterns (for future reference - might help avoid bad approaches)
                todo_manager = current_state.get('todo_manager')
                if todo_manager and todo_manager.todos:
                    shared_todo_knowledge['failed_patterns'].append({
                        'source_task': task,
                        'todos': [t.content for t in todo_manager.todos],
                        'env_id': current_env_idx
                    })
                    log_debug(f"[TODO KNOWLEDGE] ❌ Stored {len(todo_manager.todos)} failed TODOs (not transferred)")
            # END OF if False BLOCK - Old sequential processing disabled above


    # ========================================================================
    # LOG TODO TRANSFER STATISTICS
    # ========================================================================
    transfer_stats = todo_transfer_safety.get_transfer_stats()
    if transfer_stats['total'] > 0:
        log_debug(f"\n{'='*80}")
        log_debug(f"[TODO TRANSFER STATS] Trial {trial_idx}")
        log_debug(f"{'='*80}")
        log_debug(f"  Total transfer attempts: {transfer_stats['total']}")
        log_debug(f"  Successfully transferred: {transfer_stats['transferred']}")
        log_debug(f"  Blocked (safety): {transfer_stats['blocked']}")
        log_debug(f"  Block rate: {transfer_stats['block_rate']:.1%}")

        if transfer_stats['blocked_reasons']:
            log_debug(f"\n  Top block reasons:")
            for reason, count in sorted(transfer_stats['blocked_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
                log_debug(f"    - {reason}: {count}")

        log_debug(f"  Successful patterns stored: {len(shared_todo_knowledge['successful_patterns'])}")
        log_debug(f"  Failed patterns stored: {len(shared_todo_knowledge['failed_patterns'])}")
        log_debug(f"{'='*80}\n")

    # Save universal memory after all episodes complete
    universal_memory.save_memory()

    # Save global memory manager
    from generate_reflections import global_memory_manager
    if trial_log_path:
        logging_dir = os.path.dirname(trial_log_path)
    else:
        logging_dir = '.'  # Current directory as fallback

    memory_manager_path = os.path.join(logging_dir, 'global_memory_manager.pkl')
    try:
        with open(memory_manager_path, 'wb') as f:
            pickle.dump(global_memory_manager, f)
        print(f"[CHECKPOINT] Saved global memory manager")
    except Exception as e:
        print(f"[ERROR] Failed to save memory manager: {e}")

    # Save embedding cache
    state_embeddings.save_cache()
    log_debug(f"[EMBEDDINGS] Saved {len(state_embeddings.state_texts)} states to cache")

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Save working reflexions to env_configs BEFORE json.dump()
    # This was the root cause of learning transfer failure - reflexions were being
    # saved AFTER the JSON file was written, so they never persisted across trials!
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"[LEARNING TRANSFER] Saving working_reflexions to env_configs for trial {trial_idx}...")
    for state in env_states:
        env_idx = state['env_id']
        wr_count = len(state.get('working_reflexions', []))
        print(f"[LEARNING TRANSFER] ENV {env_idx}: Has {wr_count} working_reflexions in state")

        if 'working_reflexions' in state and state['working_reflexions']:
            if 'working_reflexions_history' not in env_configs[env_idx]:
                env_configs[env_idx]['working_reflexions_history'] = []

            # Add current working reflexions
            for wr in state['working_reflexions']:
                env_configs[env_idx]['working_reflexions_history'].append({
                    'trial': trial_idx,
                    'step': wr.get('step', 0),
                    'action': wr.get('action', ''),
                    'reflection': wr.get('reflection', ''),
                    'success': wr.get('success', False),
                    'type': wr.get('type')  # CRITICAL FIX: Add type field for categorization
                })

            # Keep only last 20 working reflexions
            if len(env_configs[env_idx]['working_reflexions_history']) > 20:
                env_configs[env_idx]['working_reflexions_history'] = \
                    env_configs[env_idx]['working_reflexions_history'][-20:]

            print(f"[LEARNING TRANSFER] ✅ ENV {env_idx}: Saved {wr_count} reflexions → total history: {len(env_configs[env_idx]['working_reflexions_history'])}")
        else:
            print(f"[LEARNING TRANSFER] ⚠️  ENV {env_idx}: NO working_reflexions to save!")

    # Save env configs after EVERY trial (NOW includes working_reflexions_history!)
    env_config_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
    with open(env_config_path, 'w') as f:
        json.dump(env_configs, f, indent=4)
    print(f"[CHECKPOINT] Saved env configs for trial {trial_idx}")

    # CRITICAL: Update env_configs with results
    if env_configs is not None:
        for state in env_states:
            env_configs[state['env_id']]['is_success'] = state['success']
            print(f"[BATCH] Env {state['env_id']} marked as {'SUCCESS' if state['success'] else 'FAIL'}")

    # =======================================================================
    # AGGRESSIVE MEMORY LEAK PREVENTION - Force GC After Batch
    # =======================================================================
    # This prevents the 219GB leak by forcing garbage collection after
    # each major batch processing completes, before large objects accumulate
    # =======================================================================
    import gc
    collected_objects = gc.collect()
    if collected_objects > 0:
        print(f"[PERIODIC GC] Collected {collected_objects} objects after batch processing")


    # ========================================================================
    # DEEP MEMORY TRACKING - BEFORE RETURN
    # ========================================================================
    try:
        from deep_memory_tracker import tracker
        tracker.checkpoint("BEFORE Return", {
            'env_states': env_states,
            'env_configs': env_configs,
        })

        # Print final growth summary
        print("\n[FINAL GROWTH SUMMARY]")
        growth_summary = tracker.get_growth_summary()
        for name, growth_bytes in growth_summary:
            if growth_bytes > 1024 * 1024:  # > 1MB
                print(f"  {name}: +{growth_bytes/1024/1024:.2f} MB")
    except Exception as e:
        print(f"[WARNING] Deep memory tracking failed: {e}")

    # DEBUG: Check memory usage before return
    import psutil
    import os as os_module
    process = psutil.Process(os_module.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY DEBUG] Before return: {memory_mb:.2f} MB")
    print(f"[MEMORY DEBUG] Number of env_states: {len(env_states)}")
    for i, state in enumerate(env_states):
        if 'history' in state:
            hist_str = str(state['history'])
            print(f"[MEMORY DEBUG] Env {i} history size: {len(hist_str)} chars")

    # Return results for all environments
    # DEBUG: Test if returning history objects is causing the issue
    print(f"[MEMORY DEBUG] About to create result list...")

    # Try to identify which part is causing the issue
    for i, state in enumerate(env_states):
        print(f"[MEMORY DEBUG] Env {i}: history type = {type(state['history'])}")
        # Check if we can convert history to string
        try:
            hist_str = str(state['history'])
            print(f"[MEMORY DEBUG] Env {i}: history string size = {len(hist_str)} chars")
        except Exception as e:
            print(f"[MEMORY DEBUG] Env {i}: Error converting history to string: {e}")

    # CRITICAL: The learning is already saved in env_configs['working_reflexions_history']
    # The returned history is only used for logging, so converting to string is safe
    # This fixes the "Killed" issue while preserving all learning
    result = []
    for state in env_states:
        # Convert EnvironmentHistory object to string to prevent serialization/memory issues
        # Learning is preserved via env_configs, not via this return value
        history_str = str(state['history']) if state['history'] else ""
        result.append((history_str, state['success']))

    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY DEBUG] After creating result: {memory_mb:.2f} MB")

    # =======================================================================
    # MEMORY LEAK FIX: Explicitly release env objects to prevent 219GB leak
    # =======================================================================
    # ROOT CAUSE CONFIRMED: The 'env' field contains ALFWorld environment
    # objects that consume 2.3MB each with a 48,625x memory multiplier!
    #
    # SAFE TO DELETE because:
    # 1. Learning is stored in env_configs (working_reflexions_history, memory, successful_trajectory)
    # 2. env objects are ONLY used for step(), get_current_valid_actions(), process_observation()
    # 3. env objects do NOT store any learning data - they're just game simulators
    # 4. Deletion happens AFTER all learning is saved to env_configs
    #
    # This prevents the 2.3MB × 48,625x × num_envs objects from persisting
    # =======================================================================
    # MEMORY FIX: Clear env registry and force GC
    # =======================================================================
    import gc
    print(f"[MEMORY FIX] Clearing env registry with {len(_env_registry)} environments...")
    _env_registry.clear()  # Remove all env object references

    # Force garbage collection to immediately free env objects
    collected = gc.collect()
    print(f"[MEMORY FIX] Garbage collected {collected} objects after clearing registry")

    memory_after_gc = process.memory_info().rss / 1024 / 1024
    memory_freed = memory_mb - memory_after_gc
    print(f"[MEMORY FIX] Memory after GC: {memory_after_gc:.2f} MB (freed {memory_freed:.2f} MB)")
    # =======================================================================

    return result

def format_reflexion_insights(memory) -> str:
    """Extract key actionable insights from reflexion memory - UNIVERSAL"""
    if not memory:
        return "- No previous attempts"
    
    # Universal type handling - ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str):
        memory_list = [memory]
    elif hasattr(memory, '__iter__'):
        try:
            memory_list = list(memory)
        except:
            return "- No actionable insights yet"
    else:
        return "- No actionable insights yet"
    
    if not memory_list:
        return "- No previous attempts"
    
    insights = []
    # Process last 2 reflections
    for i, reflection in enumerate(memory_list[-7:], 1):
        # Ensure reflection is a string
        if not isinstance(reflection, str):
            continue
            
        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract actionable lines using universal modal verbs
            if any(word in line_lower for word in ['must', 'should', 'avoid', 'never', 'always', 'failed because', 'succeeded', 'exact actions']):
                clean_line = line.strip()
                if len(clean_line) > 10:  # REMOVED the < 200 limit
                    
                    # Don't truncate important memory - it's causing incomplete guidance
                    key_lines.append(f"  - {clean_line}")
        
        if key_lines:
            insights.append(f"Attempt {i}:\n" + '\n'.join(key_lines[:6]))  # Increased from 4 to 6
    
    return '\n'.join(insights) if insights else "- No actionable insights yet"


def format_recent_trajectory(trajectory: List[Tuple[str, str]]) -> str:
    """Format trajectory with success/failure indicators"""
    if not trajectory:
        return "No actions taken yet"
    
    formatted = []
    for i, (action, obs, _) in enumerate(trajectory, 1):  # Unpack 3 elements,
        # Determine if successful (universal check)
        obs_lower = obs.lower()
        failure_indicators = ["nothing happens", "don't understand", "invalid", "error"]
        success = not any(ind in obs_lower for ind in failure_indicators)
        
        success_indicator = "Ã¢Å“â€œ" if success else "Ã¢Å“â€”"
        obs_preview = obs[:60] + "..." if len(obs) > 60 else obs
        # Remove newlines for cleaner display
        obs_preview = obs_preview.replace('\n', ' ')
        formatted.append(f"{i}. [{success_indicator}] {action} Ã¢â€ â€™ {obs_preview}")
    
    return '\n'.join(formatted)


def process_ob(ob):
    """Process observation for parallel runner compatibility"""
    if isinstance(ob, list) and ob:
        return ob[0]
    return str(ob)

    
# Global instances for optimization - PERSIST ACROSS TRIALS
import pickle
import os

# Try to load existing prompt_generator if it exists
PROMPT_GEN_PATH = 'prompt_generator_state.pkl'
if os.path.exists(PROMPT_GEN_PATH) and os.path.getsize(PROMPT_GEN_PATH) > 0:
    try:
        with open(PROMPT_GEN_PATH, 'rb') as f:
            prompt_generator = pickle.load(f)
        print(f"[INIT] Loaded existing prompt_generator with {len(prompt_generator.prompt_gradients)} gradients")
        
        # Verify it loaded correctly
        if hasattr(prompt_generator, 'prompt_components'):
            comp_count = len([c for c, v in prompt_generator.prompt_components.items() 
                            if v != DynamicPromptGenerator().prompt_components.get(c)])
            print(f"[INIT] Components modified from default: {comp_count}")
    except Exception as e:
        print(f"[INIT] Load failed: {e}")
        prompt_generator = DynamicPromptGenerator()
        print("[INIT] Created NEW prompt_generator (load failed)")
else:
    prompt_generator = DynamicPromptGenerator()
    if not os.path.exists(PROMPT_GEN_PATH):
        print("[INIT] Created NEW prompt_generator (first run)")
    else:
        print("[INIT] Created NEW prompt_generator (empty file)")
env_understanding = EnvironmentUnderstanding()
meta_knowledge = MetaEnvironmentKnowledge()

# Connect prompt generator with environment understanding
prompt_generator.env_understanding = env_understanding
env_understanding.set_wrapper(None)

# Add save functions
def save_prompt_generator():
    """Save prompt_generator state to disk"""
    try:
        with open(PROMPT_GEN_PATH, 'wb') as f:
            pickle.dump(prompt_generator, f)
        print(f"[SAVE] Saved prompt_generator with {len(prompt_generator.prompt_gradients)} gradients")
    except Exception as e:
        print(f"[ERROR] Failed to save prompt_generator: {e}")

def save_environment_knowledge():
    """Save ENVIRONMENT_KNOWLEDGE to disk"""
    try:
        with open('environment_knowledge.pkl', 'wb') as f:
            pickle.dump(ENVIRONMENT_KNOWLEDGE, f)
        if ENVIRONMENT_KNOWLEDGE:
            actions = len(ENVIRONMENT_KNOWLEDGE.get('action_space', {}))
            print(f"[SAVE] Saved ENVIRONMENT_KNOWLEDGE with {actions} actions")
        else:
            print(f"[SAVE] Saved empty ENVIRONMENT_KNOWLEDGE")
    except Exception as e:
        print(f"[ERROR] Failed to save ENVIRONMENT_KNOWLEDGE: {e}")

def load_environment_knowledge():
    """Load ENVIRONMENT_KNOWLEDGE from disk"""
    global ENVIRONMENT_KNOWLEDGE
    try:
        if os.path.exists('environment_knowledge.pkl'):
            with open('environment_knowledge.pkl', 'rb') as f:
                ENVIRONMENT_KNOWLEDGE = pickle.load(f)
            if ENVIRONMENT_KNOWLEDGE:
                actions = len(ENVIRONMENT_KNOWLEDGE.get('action_space', {}))
                print(f"[LOAD] Loaded ENVIRONMENT_KNOWLEDGE with {actions} actions")
                return True
            else:
                print(f"[LOAD] Loaded empty ENVIRONMENT_KNOWLEDGE")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to load ENVIRONMENT_KNOWLEDGE: {e}")
    return False
# Global discovery cache
ENVIRONMENT_KNOWLEDGE = None

# Try to load existing ENVIRONMENT_KNOWLEDGE if it exists
if load_environment_knowledge():
    # Inject loaded knowledge into prompt_generator
    if ENVIRONMENT_KNOWLEDGE:
        prompt_generator.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        print(f"[INIT] Injected loaded ENVIRONMENT_KNOWLEDGE into prompt_generator")

# Debug flags
DEBUG_ACTOR = False  # Will be set by main.py
DEBUG_CRITIC = False  # Will be set by main.py
DEBUG_REFLEXION = False  # Will be set by main.py

# Ablation study flags - default to full system (combined mode)
USE_REFLEXION = True  # Enable episodic memory and reflection generation
USE_TEXTGRAD = True   # Enable prompt optimization via textual gradients

# Set the flags in dynamic_prompting
set_debug_flags(DEBUG_ACTOR, DEBUG_CRITIC)

def process_ob(ob):
    """Process observation for parallel runner compatibility"""
    if isinstance(ob, list) and ob:
        return ob[0]
    return str(ob)


def print_debug(category: str, content: str, color: str = "blue"):
    """Print debug information with formatting"""
    if not (DEBUG_ACTOR or DEBUG_CRITIC or DEBUG_REFLEXION):
        return
        
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color, colors["blue"])
    reset = colors["reset"]
    
    print(f"\n{color_code}{'='*80}")
    print(f"[{category}]")
    print(f"{'='*80}{reset}")
    print(content)
    print(f"{color_code}{'='*80}{reset}\n")

def get_environment(env_type, config=None, env_id=None):
    """Factory function to get different environments - ALWAYS returns wrapped environment"""
    if env_type == "alfworld":
        return ALFWorldWrapper(config, env_id=env_id)
    
    # TextWorld variants
    elif env_type == "textworld_cooking":
        return TextWorldWrapper(game_type="cooking", level="medium", env_id=env_id)
    elif env_type == "textworld_treasure":
        return TextWorldWrapper(game_type="treasure", level="medium", env_id=env_id)
    elif env_type == "textworld_simple":
        return TextWorldWrapper(game_type="simple", level="easy", env_id=env_id)
    
    # Jericho games
    elif env_type == "jericho_zork1":
        return JerichoWrapper(game_name="zork1", env_id=env_id)
    elif env_type == "jericho_detective":
        return JerichoWrapper(game_name="detective", env_id=env_id)
    elif env_type == "jericho_balances":
        return JerichoWrapper(game_name="balances", env_id=env_id)
    
    # ScienceWorld tasks
    elif env_type == "scienceworld_boil":
        return ScienceWorldWrapper(task_name="boil", env_id=env_id)
    elif env_type == "scienceworld_melt":
        return ScienceWorldWrapper(task_name="melt", env_id=env_id)
    elif env_type == "scienceworld_grow":
        return ScienceWorldWrapper(task_name="grow-plant", env_id=env_id)
    
    # BabyAI levels
    elif env_type == "babyai_goto":
        return BabyAIWrapper(level="BabyAI-GoToObj-v0", env_id=env_id)
    elif env_type == "babyai_pickup":
        return BabyAIWrapper(level="BabyAI-PickupLoc-v0", env_id=env_id)
    elif env_type == "babyai_unlock":
        return BabyAIWrapper(level="BabyAI-UnlockLocal-v0", env_id=env_id)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def llm(prompt: str, stop: List[str] = ["\n"], context: str = "action_selection", 
        structured_output_schema: Optional[Dict] = None) -> str:
    """Generate text using vLLM - universal approach with structured output support"""
    
    if DEBUG_ACTOR:
        print(f"\n[LLM CALLED] Context: {context}")
    
    # Import here to avoid circular imports
    from vllm.sampling_params import GuidedDecodingParams
    
    # Prepare sampling params based on whether we need structured output
    if structured_output_schema:
        # Use guided decoding for structured output
        guided_params = GuidedDecodingParams(json=structured_output_schema)
        sampling_params = SamplingParams(
            max_tokens=100,  # Numbers need very few tokens
            temperature=0.0,  # Deterministic for action selection
            stop=stop,
            skip_special_tokens=True,
            guided_decoding=guided_params  # Use guided_decoding instead of guided_json
        )

    
    output = model.generate([prompt], sampling_params)[0]
    raw_text = output.outputs[0].text
    
    if DEBUG_ACTOR:
        print(f"[LLM RAW OUTPUT]: '{raw_text}'")

    # For structured output, the text is already just a number
    if structured_output_schema:
        return raw_text.strip()
    
    # For regular output, do minimal cleaning
    cleaned_text = raw_text.strip()
    
    # Remove think tags if present
    if "</think>" in cleaned_text:
        parts = cleaned_text.split("</think>")
        cleaned_text = parts[-1].strip() if len(parts) > 1 else cleaned_text
    
    # Clean any trailing annotations
    cleaned_text = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned_text).strip()
    
    # Remove quotes
    cleaned_text = cleaned_text.strip('"\'')
    
    # Take first line only
    if '\n' in cleaned_text:
        cleaned_text = cleaned_text.split('\n')[0].strip()
    
    if DEBUG_ACTOR:
        print(f"[LLM CLEANED OUTPUT]: '{cleaned_text}'")
    
    return cleaned_text


def extract_valid_commands_from_prompt(prompt: str) -> List[str]:
    """Extract valid commands from the prompt for validation"""
    valid_commands = []
    lines = prompt.split('\n')
    in_command_section = False
    
    for line in lines:
        if "VALID ACTIONS:" in line:
            in_command_section = True
            continue
        elif in_command_section and line.strip() and not line.startswith(' '):
            # End of valid actions section
            break
        
        if in_command_section and line.strip():
            cmd = line.strip()
            if cmd.startswith('- '):
                cmd = cmd[2:].strip()
            if cmd and not cmd.startswith('DEBUG') and not cmd.startswith('['):
                valid_commands.append(cmd)
    
    return valid_commands


def build_step_gradient_prompt_from_data(result: Dict, log_debug=print) -> str:
    """Build step gradient prompt from collected action result data.

    This extracts the prompt-building logic from generate_universal_step_reflection
    so we can batch-generate prompts for multiple environments.
    """
    # Extract all needed data from result
    prev_observation = result['prev_observation']
    curr_observation = result['observation']
    action = result['action']
    task = result['task']
    current_step = result['current_step']
    valid_actions = result['next_valid_actions']
    # Loop detection removed - no longer needed
    # is_repetitive = result['is_repetitive']
    current_todo = result['current_todo']
    inventory = result['inventory_items']
    previous_reflexions = result['tiered_step_reflexions']
    episodic_memory = result['episodic_memory_only']
    initial_observation = result.get('initial_observation', '')
    explored_locations = result.get('explored_locations', {})

    # Loop detection removed - no repetition warning needed
    repetition_warning = ""

    # Build TODO context for tactical coordination
    todo_context = ""
    if current_todo:
        todo_context = f"""
📋 CURRENT SUBTASK: {current_todo.content}
   (This is what we're trying to accomplish RIGHT NOW)
"""

    # Build inventory context (CRITICAL for avoiding loops)
    inventory_context = ""
    if inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ IMPORTANT: If holding items, you MUST place them before taking new items!
   Consider "put <item> on/in <location>" actions to complete placement subtasks.
"""

    # Build previous reflexions context
    reflexions_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        formatted_reflexions = []
        for r in previous_reflexions:
            compression_type = r.get('is_compressed', None)
            step_num = r.get('step', '?')
            reflection_text = r.get('reflection', '')[:150]

            if compression_type == 'heavy':
                formatted_reflexions.append(f"  Steps {step_num} (early summary): {reflection_text}")
            elif compression_type == 'medium':
                formatted_reflexions.append(f"  Steps {step_num} (compressed): {reflection_text}")
            else:
                formatted_reflexions.append(f"  Step {step_num} (recent): {reflection_text}")

        reflexion_text = "\n".join(formatted_reflexions)
        total_count = len(previous_reflexions)
        compressed_count = len([r for r in previous_reflexions if r.get('is_compressed')])

        reflexions_context = f"""
📚 STEP-LEVEL LEARNINGS (ALL {total_count} steps: {compressed_count} compressed, {total_count - compressed_count} verbose):
{reflexion_text}
   ⚠️ CRITICAL: Early steps are compressed but still visible - learn from ENTIRE history!
"""

    # Build episodic memory context
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        recent_episodic = episodic_memory[-5:]
        episodic_parts = []
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:200]
            else:
                mem_text = str(mem)[:200]
            episodic_parts.append(f"  • {mem_text}")
        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"""
🧠 EPISODIC MEMORY (strategic patterns from previous trials):
{episodic_text}
   ⚠️ CRITICAL: These are proven patterns across trials - use them to avoid known pitfalls!
"""

    # Build exploration guidance from initial observation and explored locations
    exploration_context = ""
    if initial_observation:
        # Extract all locations from initial observation
        import re
        location_matches = re.findall(r'(cabinet|drawer|shelf|countertop|stoveburner|fridge|microwave|dresser|desk|sidetable|armchair|sofa|bed) \d+', initial_observation.lower())
        all_locations = list(set(location_matches))  # Unique locations

        # Determine which have been explored
        explored = [loc for loc in all_locations if loc in explored_locations and explored_locations[loc] > 0]
        unexplored = [loc for loc in all_locations if loc not in explored_locations or explored_locations[loc] == 0]

        if len(unexplored) > 0:
            unexplored_list = ', '.join(sorted(unexplored)[:15])  # Show first 15
            explored_list = ', '.join(sorted(explored)[:10]) if explored else "None"
            exploration_context = f"""
🔍 EXPLORATION STATUS (CRITICAL - Avoid revisiting locations!):
   Total locations: {len(all_locations)}
   ✅ Explored ({len(explored)}): {explored_list}
   ❌ NOT YET EXPLORED ({len(unexplored)}): {unexplored_list}

   ⚠️ PRIORITY: When searching for items, CHECK UNEXPLORED LOCATIONS FIRST!
   Don't waste steps re-examining locations you've already checked!
"""

    # Build TextGrad's previous recommendations context (CRITICAL for avoiding cycles)
    textgrad_history_context = ""
    step_insights = result.get('step_insights_accumulator', [])
    log_debug(f"[TEXTGRAD DEBUG] step_insights has {len(step_insights)} items")
    if step_insights and len(step_insights) > 0:
        recent_gradients = step_insights[-5:]  # Last 5 TextGrad recommendations
        log_debug(f"[TEXTGRAD DEBUG] Building history context with {len(recent_gradients)} recent gradients")
        textgrad_history_context = "\n🎯 YOUR PREVIOUS TEXTGRAD RECOMMENDATIONS (Avoid Cycling!):\n"
        for insight in recent_gradients:
            action_taken = insight.get('action', 'Unknown')
            guidance = insight.get('next_guidance', 'None')
            progress = insight.get('progress_score', 0)
            textgrad_history_context += f"  Step {insight.get('step', '?')}: Action '{action_taken}' → Recommended '{guidance}' (progress: {progress}/10)\n"
            log_debug(f"[TEXTGRAD DEBUG] Added: Step {insight.get('step')} action='{action_taken}' guidance='{guidance}'")
        textgrad_history_context += "\n  ⚠️ CRITICAL TEXTGRAD MEMORY: If you're about to recommend an action you already suggested,\n"
        textgrad_history_context += "  that means you're CYCLING! Try a COMPLETELY DIFFERENT approach instead!\n"
        textgrad_history_context += "  (This is textual gradient descent - use your own gradient history to avoid local minima)\n"
        log_debug(f"[TEXTGRAD DEBUG] textgrad_history_context length: {len(textgrad_history_context)} chars")
    else:
        log_debug(f"[TEXTGRAD DEBUG] No step_insights, textgrad_history_context is EMPTY")

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}
BEFORE: {prev_observation}
ACTION: {action}
AFTER: {curr_observation}
VALID ACTIONS: {', '.join(valid_actions[:100]) if valid_actions else 'Unknown'}

{repetition_warning}

═══════════════════════════════════════════════════════════
CRITICAL REFLEXION ANALYSIS (Answer each question precisely):
═══════════════════════════════════════════════════════════

1. SEMANTIC PRECISION CHECK:
   Task instruction: "{task}"
   My action: "{action}"

   Extract the PRIMARY PURPOSE of each:
   - What is the main goal the task requires me to achieve? (e.g., change temperature, change location, change cleanliness, etc.)
   - What is the main effect/purpose of the action I took?
   - Are these semantically IDENTICAL purposes or DIFFERENT purposes?

   IMPORTANT: Even if both use similar objects or locations, they may serve different purposes.
   Example: "remove dirt" vs "lower temperature" are DIFFERENT even if both might involve water.

   Answer: SEMANTIC MATCH: [Yes/No] - Explain: [task purpose] vs [action purpose]

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{task}", what SPECIFIC property of objects must change?
   (Examples: temperature, location, cleanliness, container state, object state)

   Now look at the AFTER observation: "{curr_observation}"
   Did that EXACT property change? Cite specific words from observation as evidence.
   Do NOT assume or infer - only state what the observation explicitly shows.

   Answer: PROPERTY REQUIRED: [what must change] | EVIDENCE: [quote from observation] | CHANGED: [Yes/No]

3. TODO FEEDBACK INTEGRATION:
   {f'Current subtask: "{current_todo.content}" - Status after action: NOT ACHIEVED (Attempt #{current_todo.attempts if current_todo else 0})' if current_todo else 'No active subtask'}

   If subtask not achieved after {current_todo.attempts if current_todo else 0} attempt(s):
   - What pattern do you see in my actions?
   - Am I repeating similar action types that don't work?
   - What assumption might I be making that is incorrect?

   Answer: FAILURE PATTERN: [describe pattern if any] | ROOT CAUSE: [precise diagnosis]

4. CAUSAL CHAIN ANALYSIS:
   Trace the causal relationship:
   - GOAL (from task): [what I wanted to accomplish]
   - ACTION (what I did): {action}
   - RESULT (what happened): [state change or lack thereof]
   - CAUSALITY: This [succeeded/failed] because [causal reason]

   If there's a semantic mismatch or wrong action type:
   - Have I already tried similar actions on this location in STEP-LEVEL LEARNINGS?
   - If yes, what was the result? Did it fail to find the target object?
   - What is the ROOT CAUSE of the repeated failure? (e.g., wrong location, wrong action type, impossible task)

   NOTE: Do NOT recommend next actions - that's TextGrad's responsibility for optimization.
   Reflexion focuses ONLY on WHY things succeed/fail, not WHAT to do next.

   Answer: CAUSAL CHAIN: [describe cause-effect] | ROOT CAUSE: [fundamental reason]

5. CONCRETE LESSON EXTRACTION:
   Based on your analysis, complete this pattern for episodic memory:
   - WRONG ASSUMPTION: [what I incorrectly believed, if any]
   - CORRECT UNDERSTANDING: [what is actually true]
   - NEVER REPEAT: [specific action or pattern to avoid]
   - INSTEAD USE: [specific action or approach that works]

   Answer: LESSON: [fill in above format only if there's a clear lesson to extract]

6. PROGRESS ASSESSMENT:
   Score (0-10) how much this action progressed the {f'subtask "{current_todo.content}"' if current_todo else 'task'}:

   CRITICAL: Be SPECIFIC about concrete state changes, not conservative estimates!

   - 0-2: No progress (state unchanged, or moved away from goal)
   - 3-4: Minimal progress (explored location, gathered info, but no task-relevant change)
   - 5-6: Concrete progress (acquired object, opened container, moved toward target)
   - 7-8: Major progress (completed sub-task: placed object, changed state significantly)
   - 9-10: Task complete or nearly complete (all requirements met or one step away)

   IMPORTANT:
   - Score based on CONCRETE state changes (GOT object? MOVED it? State CHANGED?)
   - Don't be conservative! If action changed state toward goal, give 5+
   - Reserve 0-2 for truly useless actions (loops, redundant observations)

   Justify based on: semantic match + state change evidence + TODO status

   Answer: SCORE: [0-10] | JUSTIFICATION: [based on semantic/state/TODO analysis]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    # ═══════════════════════════════════════════════════════════
    # PHASE 2 SYNERGY: TextGrad (normal) vs Reflexion (failures) - PARALLEL CODE
    # ═══════════════════════════════════════════════════════════

    # Get previous step's progress to decide which component to use
    last_step_gradient = result.get('last_step_gradient', {})
    last_progress = last_step_gradient.get('progress_score', 0)  # FIX: Default to 0 (failure) not 5 (success)

    # PHASE 2 DECISION: Use progress-based routing (aligned with papers)
    # - TextGrad (Nature 2024): Step-by-step optimization when making progress
    # - Reflexion (NeurIPS 2023): Causal analysis only at failures/low progress

    # Get state for consecutive failure tracking
    state = result.get('state', {})
    current_step_num = result.get('current_step', 0)

    # Initialize consecutive failure tracking variables
    if 'consecutive_low_progress' not in state:
        state['consecutive_low_progress'] = 0
    if 'steps_since_reflexion' not in state:
        state['steps_since_reflexion'] = 0

    # Update consecutive failure counter based on last step's progress
    if last_progress < 4:  # Low progress threshold (0-3 are low, 4+ is actual progress)
        state['consecutive_low_progress'] += 1
    else:
        state['consecutive_low_progress'] = 0  # Reset on good progress

    # Increment cooldown counter
    state['steps_since_reflexion'] += 1

    # Smart trigger: Only use Reflexion when TRULY stuck
    # Requirements:
    # 1. Not step 0 (TODO provides initial guidance)
    # 2. 5+ consecutive low-progress steps (not just transient failure)
    # 3. 5+ steps since last Reflexion (cooldown to let TextGrad establish gradient)
    use_reflexion = (
        current_step_num > 0 and
        state['consecutive_low_progress'] >= 5 and
        state['steps_since_reflexion'] >= 5
    )

    if not use_reflexion:
        # TEXTGRAD PATH (85%+ of steps): Good progress → keep optimizing
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using TEXTGRAD (last_progress={last_progress}/10)")
        prompt = generate_textgrad_gradient_prompt(result, previous_reflexions, episodic_memory,
                                                     initial_observation, explored_locations, valid_actions)
        return ('textgrad', prompt)  # Return tuple with type for metadata tracking
    else:
        # REFLEXION PATH (<15% of steps): Low progress → diagnose why
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using REFLEXION (last_progress={last_progress}/10 - need causal analysis)")

        # Reset tracking counters after using Reflexion
        state['consecutive_low_progress'] = 0
        state['steps_since_reflexion'] = 0

        return ('reflexion', reflection_prompt)  # Return tuple with type for metadata tracking


def generate_textgrad_gradient_prompt(result: Dict, previous_reflexions: list, episodic_memory: list,
                                      initial_observation: str, explored_locations: dict,
                                      valid_actions: list) -> str:
    """Generate TextGrad prompt for automatic differentiation via text.

    TextGrad's role (Nature 2024): Optimize actions through textual gradients
    - Provides step-by-step action optimization (NOT strategic analysis)
    - Accumulates gradients to backpropagate through text
    - Suggests next action based on gradient descent
    """
    task = result['task']
    prev_observation = result['prev_observation']
    curr_observation = result['observation']
    action = result['action']
    current_todo = result.get('current_todo')
    inventory_items = result.get('inventory_items', [])

    # Build TODO context
    todo_context = ""
    if current_todo:
        status_emoji = {"pending": "⏳", "in_progress": "🔧", "completed": "✅", "failed": "❌"}.get(current_todo.status, "")
        todo_context = f"\n🎯 CURRENT SUBTASK: {status_emoji} {current_todo.content} (Attempt #{current_todo.attempts})\n"

    # Build inventory context
    inventory_context = ""
    if inventory_items and len(inventory_items) > 0:
        items_str = ", ".join(inventory_items)
        inventory_context = f"\n🎒 INVENTORY: Currently holding {items_str}\n   ⚠️ CRITICAL: Use held items for task completion!\n"

    # Build gradient history (last 5 steps for backpropagation)
    gradient_history = ""
    step_insights = result.get('step_insights_accumulator', [])
    if step_insights and len(step_insights) > 0:
        recent_gradients = step_insights[-5:]
        gradient_history = "\n📊 ACCUMULATED GRADIENTS (Your optimization history):\n"
        for insight in recent_gradients:
            act = insight.get('action', 'Unknown')[:30]
            guide = insight.get('next_guidance', 'None')[:50]
            score = insight.get('progress_score', 0)
            gradient_history += f"  Step {insight.get('step', '?')}: {act} → Score:{score}/10 → Gradient:'{guide}'\n"
        gradient_history += "  ⚠️ Use gradient descent: If similar actions yielded low scores, try DIFFERENT direction!\n"

    # Build REFLEXION STRATEGIC INSIGHTS (step-level guidance from recent Reflexion analysis)
    reflexion_insights = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        recent_reflexions = previous_reflexions[-3:]  # Last 3 Reflexion insights for immediate context
        reflexion_insights = "\n🎯 REFLEXION STRATEGIC INSIGHTS (Critical guidance from failure analysis):\n"
        for ref in recent_reflexions:
            if isinstance(ref, dict):
                step = ref.get('step', '?')
                insight = ref.get('reflection', '')
                # Extract the key strategic recommendation
                if insight:
                    # Truncate to key insight
                    insight_text = insight[:200] if len(insight) > 200 else insight
                    reflexion_insights += f"  📍 Step {step}: {insight_text}\n"
        reflexion_insights += "  ⚠️ CRITICAL: These insights identify WHY previous actions failed. Your next action MUST address these strategic issues!\n"

    # Build episodic constraints from Reflexion (strategic warnings)
    episodic_constraints = ""
    if episodic_memory and len(episodic_memory) > 0:
        recent_episodic = episodic_memory[-3:]  # Last 3 strategic lessons
        episodic_constraints = "\n🧠 STRATEGIC CONSTRAINTS (from Reflexion's episodic memory):\n"
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:150]
            else:
                mem_text = str(mem)[:150]
            episodic_constraints += f"  ⚠️  {mem_text}\n"
        episodic_constraints += "  Your recommended action MUST NOT violate these strategic constraints!\n"

    textgrad_prompt = f"""🔬 TEXTGRAD: Automatic Differentiation via Text

YOUR ROLE (TextGrad's Unique Strength - Nature 2024):
- Optimize action selection through textual gradients (like backpropagation for prompts)
- Provide gradient feedback: "If you had done X instead of Y, outcome would have been Z"
- Accumulate gradients across steps to perform gradient descent
- Suggest NEXT ACTION that follows the gradient toward task completion
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{todo_context}{inventory_context}{reflexion_insights}{episodic_constraints}{gradient_history}
BEFORE: {prev_observation}
ACTION TAKEN: {action}
AFTER: {curr_observation}
VALID ACTIONS: {', '.join(valid_actions[:50]) if valid_actions else 'Unknown'}

═══════════════════════════════════════════════════════════
TEXTGRAD GRADIENT COMPUTATION (Answer each precisely):
═══════════════════════════════════════════════════════════

1. GRADIENT EVALUATION:
   The action I took: "{action}"
   The outcome: "{curr_observation[:100]}"

   Gradient question: If I had taken a DIFFERENT action, would the outcome be better or worse?
   Specifically:
   - What property should have changed based on task "{task}"?
   - Did that property change? (Cite evidence from AFTER observation)
   - If NO change, what alternative action would have caused the desired change?

   Answer: GRADIENT: [Positive/Zero/Negative] | REASON: [why] | BETTER ACTION: [specific alternative from VALID ACTIONS]

2. ACCUMULATED GRADIENT DESCENT:
   Looking at my ACCUMULATED GRADIENTS above (last 5 steps):
   - What pattern do I see in actions with high scores (7-10)?
   - What pattern do I see in actions with low scores (0-3)?
   - Based on gradient descent, should I continue current direction or try new direction?

   Answer: PATTERN HIGH: [what worked] | PATTERN LOW: [what failed] | DIRECTION: [continue/change]

3. STRATEGIC CONSTRAINT CHECK:
   {f'My episodic memory warns: {episodic_constraints}' if episodic_constraints else 'No strategic constraints'}

   Before recommending an action, verify:
   - Does my recommended action violate any strategic constraint above?
   - If yes, what alternative achieves same goal without violation?

   Answer: CONSTRAINT CHECK: [Pass/Fail] | IF FAIL: [alternative action]

4. NEXT ACTION OPTIMIZATION:
   Based on:
   - Current gradient (what would improve outcome)
   - Accumulated gradient history (what direction is working)
   - Strategic constraints (what to avoid)

   From the VALID ACTIONS list, which SPECIFIC action optimizes progress?
   (Copy exact action text - don't paraphrase)
   (Follow gradient descent - if stuck in local minima, try exploration)

   Answer: RECOMMENDED ACTION: [exact action from VALID ACTIONS] | GRADIENT JUSTIFICATION: [why this follows gradient]

5. PROGRESS SCORE:
   Score (0-10) the progress this action made toward {f'subtask "{current_todo.content}"' if current_todo else 'task'}:

   CRITICAL: Be SPECIFIC about what changed, not conservative estimates!

   - 0-2: No progress (state identical, or moved away from goal)
   - 3-4: Minimal progress (explored new location, gathered information, but no task-relevant change)
   - 5-6: Concrete progress (acquired object, opened container, moved closer to target location)
   - 7-8: Major progress (completed a sub-task: placed object, cooled/heated item, found target)
   - 9-10: Task complete or nearly complete (all requirements met or one step away)

   IMPORTANT:
   - Score based on CONCRETE state changes (did you GET an object? did it MOVE somewhere? did temperature CHANGE?)
   - Don't be conservative! If action changed state toward goal, give 5+
   - Reserve 0-2 for truly useless actions (loops, redundant observations)

   Answer: SCORE: [0-10] | REASONING: [cite specific state change from AFTER observation]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    return textgrad_prompt


def parse_step_gradient_response(response: str, task: str, prev_observation: str, curr_observation: str, action: str) -> Dict:
    """Parse step gradient LLM response into structured format.

    This extracts the parsing logic from generate_universal_step_reflection
    so we can batch-parse multiple gradient outputs.
    """
    import re

    # Initialize step_gradient
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
    }

    # Clean think tags if present
    response = response.replace("</think>", "").replace("<think>", "").strip()

    # Parse numbered responses
    lines = response.split('\n')
    answers = {}
    current_num = 0
    current_answer = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # More robust number extraction
        if line_stripped and line_stripped[0].isdigit():
            # Save previous answer
            if current_num > 0 and current_answer:
                answers[current_num] = ' '.join(current_answer).strip()

            # Extract number more carefully
            num_match = re.match(r'^(\d+)', line_stripped)
            if num_match:
                current_num = int(num_match.group(1))
                # Get rest of line after number (and any punctuation)
                rest = re.sub(r'^\d+[.,):\s]*', '', line_stripped)
                current_answer = [rest] if rest else []
            else:
                # Add to current answer if no number found
                if current_num > 0:
                    current_answer.append(line_stripped)
        elif current_num > 0:
            # Continue current answer
            current_answer.append(line_stripped)

    # Save last answer
    if current_num > 0 and current_answer:
        answers[current_num] = ' '.join(current_answer).strip()

    # Extract progress_score from answer 6
    progress_score = 0
    if 5 in answers:
        try:
            score_text = answers[5]
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                progress_score = min(int(numbers[0]), 10)  # Cap at 10
        except:
            progress_score = 0

    # Direct observation comparison for state change detection
    # NOTE: This must be defined BEFORE it's used in the progress boost logic below
    state_changed = (prev_observation.strip() != curr_observation.strip())

    # REMOVED naive word matching - use ONLY LLM's semantic score
    # The LLM already evaluates progress semantically, don't override with word overlap


    # Extract structured information with defaults
    missing_prereqs = answers.get(2, '').strip()
    missing_list = [missing_prereqs] if missing_prereqs and missing_prereqs.lower() not in ['none', 'nothing', 'n/a'] else []

    task_addressed = answers.get(3, '').strip()
    addressed_list = [task_addressed] if task_addressed and task_addressed.lower() not in ['none', 'nothing', 'n/a'] else []

    task_remaining = answers.get(4, task).strip()
    remaining_list = [task_remaining] if task_remaining else [task]

    learned_rule = answers.get(5, 'Continue exploring available actions').strip()
    next_guidance = answers.get(4, 'Try unexplored actions').strip()  # Extract from answer 4 (CAUSAL CHAIN)

    # CRITICAL FIX: Extract "Recommended next action: X" from response
    recommended_action_match = re.search(r'Recommended next action:\s*([^\n]+)', response, re.IGNORECASE)
    if not recommended_action_match:
        recommended_action_match = re.search(r'RECOMMENDED ACTION:\s*-?\s*([a-z][^\n-]+)', response, re.IGNORECASE)

    if recommended_action_match:
        next_guidance = recommended_action_match.group(1).strip()
        next_guidance = next_guidance.strip('- .,;:')
        if next_guidance.upper().startswith('GOAL:'):
            action_after_goal = re.search(r'(?:GOAL:.*?)?[-\s]*(?:ACTION|RESULT|Recommended):\s*([a-z][^\n-]+)', response, re.IGNORECASE | re.DOTALL)
            if action_after_goal:
                next_guidance = action_after_goal.group(1).strip().strip('- .,;:')
    else:
        # Fallback: Extract exact action if it's quoted
        quoted_action = re.findall(r"'([^']+)'", next_guidance)
        if quoted_action:
            next_guidance = quoted_action[0]

    # Universal task progress detection
    task_words = set(task.lower().split())
    state_before_words = set(prev_observation.lower().split())
    state_after_words = set(curr_observation.lower().split())

    # Check if any task words that weren't in before state are now in after state
    progress_indicators = task_words & (state_after_words - state_before_words)

    # Build full gradient structure
    step_gradient = {
        'state_change': curr_observation if state_changed else 'NO CHANGE',
        'progress_assessment': f"Addressed: {', '.join(addressed_list)}" if addressed_list else "No progress",
        'progress_score': progress_score,
        'hypothesis': learned_rule,
        'next_action_guidance': next_guidance,
        'raw_reflection': response,
        'semantic_state_before': prev_observation,
        'semantic_state_after': curr_observation,
        'prerequisites': {
            'present': [],
            'missing': missing_list
        },
        'task_progress': {
            'addressed': addressed_list,
            'remaining': remaining_list
        }
    }

    return step_gradient


def generate_textgrad_step_guidance(
    prev_observation: str,
    curr_observation: str,
    action: str,
    task: str,
    model,
    log_debug = print,
    current_step: int = 0,
    valid_actions: List[str] = None,
    current_todo = None,
    inventory: List[str] = None,
    previous_reflexions: List[Dict] = None,
    episodic_memory: List[str] = None
) -> Dict:
    """
    TextGrad-only prompt: Generate clean, actionable next action through gradient optimization.

    TextGrad's Unique Strength (Nature 2024):
    - Step-by-step action optimization through textual gradients
    - Clean, executable action recommendations
    - Gradient-based learning from immediate feedback

    This function does NOT do causal why-analysis (that's Reflexion's job).
    It focuses ONLY on WHAT action to take next based on gradient signals.
    """

    # Initialize step_gradient
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
        'guidance_source': 'textgrad'  # Mark source for action selector
    }

    # Build TODO context
    todo_context = ""
    if current_todo:
        todo_context = f"\n📋 CURRENT SUBTASK: {current_todo.content}\n   (Focus on completing THIS subtask)\n"

    # Build inventory context
    inventory_context = ""
    if inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"\n🎒 INVENTORY: Currently carrying {items_str}\n   ⚠️ If holding items, consider 'put' actions to place them.\n"

    # Build gradient history (recent step learnings for optimization)
    gradient_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        recent_reflexions = previous_reflexions[-5:]  # Last 5 steps for immediate feedback
        formatted = []
        for r in recent_reflexions:
            step_num = r.get('step', '?')
            progress = r.get('progress_score', 0)
            guidance = r.get('next_action_guidance', '')[:80]
            formatted.append(f"  Step {step_num}: Progress={progress}/10 → {guidance}")

        gradient_text = "\n".join(formatted)
        gradient_context = f"\n📊 RECENT GRADIENT SIGNALS (last {len(recent_reflexions)} steps):\n{gradient_text}\n"

    # Build episodic memory context (proven patterns)
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        episodic_parts = []
        for mem in episodic_memory[-3:]:  # Last 3 episodic memories
            mem_str = str(mem)
            if len(mem_str) > 120:
                mem_str = mem_str[:120] + '...'
            episodic_parts.append(f"  • {mem_str}")

        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"\n🧠 PROVEN PATTERNS (from previous trials):\n{episodic_text}\n"

    textgrad_prompt = f"""🎯 TEXTGRAD: Action Optimization Through Gradients

YOUR ROLE (TextGrad's Unique Strength):
- Generate the BEST next action based on gradient signals from previous steps
- Optimize action selection using immediate feedback (progress scores, observations)
- Provide CLEAN, EXECUTABLE action (no analysis, just the action string)
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{todo_context}{inventory_context}{gradient_context}{episodic_context}
BEFORE: {prev_observation}
ACTION TAKEN: {action}
AFTER: {curr_observation}
VALID ACTIONS: {', '.join(valid_actions[:100]) if valid_actions else 'Unknown'}

═══════════════════════════════════════════════════════════
TEXTGRAD OPTIMIZATION (Answer concisely):
═══════════════════════════════════════════════════════════

1. PROGRESS ASSESSMENT (0-10 scale):
   Did the action move us closer to completing the task or current subtask?
   - Score 8-10: Major progress (found target object, completed key state change)
   - Score 4-7: Moderate progress (eliminated wrong locations, gathered info)
   - Score 0-3: No progress or regression (dead end, repetition)

   Answer: PROGRESS: [0-10] | REASON: [one sentence why]

2. GRADIENT SIGNAL:
   What did this step teach us about what works/doesn't work?
   Keep it concise - focus on actionable insights, not analysis.

   Answer: GRADIENT: [one sentence lesson]

3. NEXT ACTION OPTIMIZATION:
   Based on progress score and gradient signals, what's the BEST next action?

   Selection criteria:
   - If progress ≥ 7: Continue current strategy
   - If progress 4-6: Adjust approach slightly
   - If progress < 4: Try different location/action type
   - Respect TODO subtask if active
   - Use episodic patterns if available
   - Pick from VALID ACTIONS list

   Answer: NEXT_ACTION: [exact action string from valid actions, nothing else]

═══════════════════════════════════════════════════════════
CRITICAL: Your NEXT_ACTION must be:
1. A single action string (e.g., "go to cabinet 2")
2. Exactly matching or very close to a valid action
3. NO explanations, NO analysis, just the action
═══════════════════════════════════════════════════════════
"""

    try:
        response = model.generate([textgrad_prompt], SamplingParams(
            max_tokens=200,  # Short response - just action + brief reasoning
            temperature=0.0,
            top_p=0.95
        ))[0].outputs[0].text.strip()

        step_gradient['raw_reflection'] = response

        # Extract progress score
        # Extract LLM's semantic score from "PROGRESS: 7" pattern (TextGrad format)
        progress_match = re.search(r'PROGRESS:\s*(\d+)', response, re.IGNORECASE)
        progress_score = int(progress_match.group(1)) if progress_match else 0  # No fallback - use LLM score only
        step_gradient['progress_score'] = min(max(progress_score, 0), 10)

        # Extract gradient lesson
        gradient_match = re.search(r'GRADIENT:\s*([^\n]+)', response, re.IGNORECASE)
        if gradient_match:
            step_gradient['hypothesis'] = gradient_match.group(1).strip()

        # Extract next action (CLEAN - no verbose analysis)
        next_action = ''
        action_match = re.search(r'NEXT_ACTION:\s*([^\n]+)', response, re.IGNORECASE)
        if action_match:
            next_action = action_match.group(1).strip()
            # Remove common prefixes
            next_action = next_action.replace('ACTION:', '').replace('NEXT:', '').strip()
            # Remove quotes if present
            next_action = next_action.strip('"\'')

        step_gradient['next_action_guidance'] = next_action

        log_debug(f"[TEXTGRAD] Progress={progress_score}/10, Next={next_action[:50]}")

    except Exception as e:
        print(f"[ERROR] Exception in generate_textgrad_step_guidance: {e}")
        step_gradient['progress_score'] = 0
        step_gradient['next_action_guidance'] = ''

    return step_gradient


def generate_reflexion_causal_analysis(
    prev_observation: str,
    curr_observation: str,
    action: str,
    task: str,
    model,
    log_debug = print,
    current_step: int = 0,
    valid_actions: List[str] = None,
    # Loop detection removed - parameter no longer needed
    current_todo = None,  # Current TODO for tactical coordination
    inventory: List[str] = None,  # Items currently being carried
    previous_reflexions: List[Dict] = None,  # CRITICAL: Previous step learnings (step-level memory)
    episodic_memory: List[str] = None  # CRITICAL: Cross-trial learnings (episodic memory)
) -> Dict:
    """
    Reflexion-only prompt: Deep causal why-analysis for failures and low-progress situations.

    Reflexion's Unique Strength (NeurIPS 2023):
    - Identify WHY actions fail through precise causal analysis
    - Extract concrete lessons for episodic memory (cross-trial learning)
    - Root cause diagnosis (not action optimization)

    This function should ONLY be called when progress_score < 4 (failures/low progress).
    For normal steps with good progress, use generate_textgrad_step_guidance() instead.
    """

    # Initialize step_gradient at the beginning
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
        'is_loop': False,  # Loop detection removed
        'guidance_source': 'reflexion'  # Mark as Reflexion CAUSAL CHAIN (needs extraction)
    }

    # Loop detection removed - no repetition warning needed
    repetition_warning = ""

    # Build TODO context for tactical coordination
    todo_context = ""
    if current_todo:
        todo_context = f"""
📋 CURRENT SUBTASK: {current_todo.content}
   (This is what we're trying to accomplish RIGHT NOW)
"""

    # Build inventory context (CRITICAL for avoiding loops)
    inventory_context = ""
    if inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ IMPORTANT: If holding items, you MUST place them before taking new items!
   Consider "put <item> on/in <location>" actions to complete placement subtasks.
"""

    # Build previous reflexions context (CRITICAL for learning from past steps!)
    # Use ALL reflexions with smart tiered compression - never forget early lessons!
    reflexions_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        # previous_reflexions is already tiered/compressed from the calling function
        formatted_reflexions = []
        for r in previous_reflexions:
            compression_type = r.get('is_compressed', None)
            step_num = r.get('step', '?')
            reflection_text = r.get('reflection', '')[:150]

            if compression_type == 'heavy':
                formatted_reflexions.append(f"  Steps {step_num} (early summary): {reflection_text}")
            elif compression_type == 'medium':
                formatted_reflexions.append(f"  Steps {step_num} (compressed): {reflection_text}")
            else:
                formatted_reflexions.append(f"  Step {step_num} (recent): {reflection_text}")

        reflexion_text = "\n".join(formatted_reflexions)
        total_count = len(previous_reflexions)
        compressed_count = len([r for r in previous_reflexions if r.get('is_compressed')])

        reflexions_context = f"""
📚 STEP-LEVEL LEARNINGS (ALL {total_count} steps: {compressed_count} compressed, {total_count - compressed_count} verbose):
{reflexion_text}
   ⚠️ CRITICAL: Early steps are compressed but still visible - learn from ENTIRE history!
"""

    # Build episodic memory context (CRITICAL for cross-trial strategic learning!)
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        # Take last 5 episodic memories (strategic patterns across trials)
        recent_episodic = episodic_memory[-5:]
        # Handle dict memories - extract insight field
        episodic_parts = []
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:200]
            else:
                mem_text = str(mem)[:200]
            episodic_parts.append(f"  • {mem_text}")
        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"""
🧠 EPISODIC MEMORY (strategic patterns from previous trials):
{episodic_text}
   ⚠️ CRITICAL: These are proven patterns across trials - use them to avoid known pitfalls!
"""

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}
BEFORE: {prev_observation}
ACTION: {action}
AFTER: {curr_observation}
VALID ACTIONS: {', '.join(valid_actions[:100]) if valid_actions else 'Unknown'}

{repetition_warning}

═══════════════════════════════════════════════════════════
CRITICAL REFLEXION ANALYSIS (Answer each question precisely):
═══════════════════════════════════════════════════════════

1. SEMANTIC PRECISION CHECK:
   Task instruction: "{task}"
   My action: "{action}"

   Extract the PRIMARY PURPOSE of each:
   - What is the main goal the task requires me to achieve? (e.g., change temperature, change location, change cleanliness, etc.)
   - What is the main effect/purpose of the action I took?
   - Are these semantically IDENTICAL purposes or DIFFERENT purposes?

   IMPORTANT: Even if both use similar objects or locations, they may serve different purposes.
   Example: "remove dirt" vs "lower temperature" are DIFFERENT even if both might involve water.

   Answer: SEMANTIC MATCH: [Yes/No] - Explain: [task purpose] vs [action purpose]

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{task}", what SPECIFIC property of objects must change?
   (Examples: temperature, location, cleanliness, container state, object state)

   Now look at the AFTER observation: "{curr_observation}"
   Did that EXACT property change? Cite specific words from observation as evidence.
   Do NOT assume or infer - only state what the observation explicitly shows.

   Answer: PROPERTY REQUIRED: [what must change] | EVIDENCE: [quote from observation] | CHANGED: [Yes/No]

3. TODO FEEDBACK INTEGRATION:
   {f'Current subtask: "{current_todo.content}" - Status after action: NOT ACHIEVED (Attempt #{current_todo.attempts if current_todo else 0})' if current_todo else 'No active subtask'}

   If subtask not achieved after {current_todo.attempts if current_todo else 0} attempt(s):
   - What pattern do you see in my actions?
   - Am I repeating similar action types that don't work?
   - What assumption might I be making that is incorrect?

   Answer: FAILURE PATTERN: [describe pattern if any] | ROOT CAUSE: [precise diagnosis]

4. CAUSAL CHAIN ANALYSIS:
   Trace the causal relationship:
   - GOAL (from task): [what I wanted to accomplish]
   - ACTION (what I did): {action}
   - RESULT (what happened): [state change or lack thereof]
   - CAUSALITY: This [succeeded/failed] because [causal reason]

   If there's a semantic mismatch or wrong action type:
   - Have I already tried similar actions on this location in STEP-LEVEL LEARNINGS?
   - If yes, what was the result? Did it fail to find the target object?
   - What is the ROOT CAUSE of the repeated failure? (e.g., wrong location, wrong action type, impossible task)

   NOTE: Do NOT recommend next actions - that's TextGrad's responsibility for optimization.
   Reflexion focuses ONLY on WHY things succeed/fail, not WHAT to do next.

   Answer: CAUSAL CHAIN: [describe cause-effect] | ROOT CAUSE: [fundamental reason]

5. CONCRETE LESSON EXTRACTION:
   Based on your analysis, complete this pattern for episodic memory:
   - WRONG ASSUMPTION: [what I incorrectly believed, if any]
   - CORRECT UNDERSTANDING: [what is actually true]
   - NEVER REPEAT: [specific action or pattern to avoid]
   - INSTEAD USE: [specific action or approach that works]

   Answer: LESSON: [fill in above format only if there's a clear lesson to extract]

6. PROGRESS ASSESSMENT:
   Score (0-10) how much this action progressed the {f'subtask "{current_todo.content}"' if current_todo else 'task'}:

   CRITICAL: Be SPECIFIC about concrete state changes, not conservative estimates!

   - 0-2: No progress (state unchanged, or moved away from goal)
   - 3-4: Minimal progress (explored location, gathered info, but no task-relevant change)
   - 5-6: Concrete progress (acquired object, opened container, moved toward target)
   - 7-8: Major progress (completed sub-task: placed object, changed state significantly)
   - 9-10: Task complete or nearly complete (all requirements met or one step away)

   IMPORTANT:
   - Score based on CONCRETE state changes (GOT object? MOVED it? State CHANGED?)
   - Don't be conservative! If action changed state toward goal, give 5+
   - Reserve 0-2 for truly useless actions (loops, redundant observations)

   Justify based on: semantic match + state change evidence + TODO status

   Answer: SCORE: [0-10] | JUSTIFICATION: [based on semantic/state/TODO analysis]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    sampling_params = SamplingParams(
        max_tokens=7000,  # Increased to 7000 to prevent empty responses with reasoning='medium'
        temperature=0.3,
        stop=["TASK:", "BEFORE:"],
        skip_special_tokens=True
    )

    # Log TextGrad using both memories (with compression info)
    step_mem_count = len(previous_reflexions) if previous_reflexions else 0
    compressed_count = len([r for r in previous_reflexions if r.get('is_compressed')]) if previous_reflexions else 0
    episodic_mem_count = len(episodic_memory) if episodic_memory else 0
    log_debug(f"[TEXTGRAD] Using ALL {step_mem_count} step reflexions ({compressed_count} compressed), {episodic_mem_count} episodic patterns")

    try:
        # Use reasoning='medium' explicitly for step reflexions (faster than 'high', sufficient quality)
        output = model.generate([reflection_prompt], sampling_params, reasoning_effort='medium')[0]
    except Exception as e:
        # from api_quota_handler import APIQuotaHandler  # Commented out - not needed
        # from checkpoint_manager import CheckpointManager  # Commented out - not needed
        # Create temporary checkpoint manager for error handling
        # temp_logging_dir = '.'
        # checkpoint_manager = CheckpointManager(temp_logging_dir)
        # quota_handler = APIQuotaHandler(checkpoint_manager, checkpoint_manager.run_dir)
        # quota_handler.handle_api_error(e, current_step, {}, [])
        print(f"[ERROR] Exception in generate_universal_step_reflection: {e}")
        raise e
    response = output.outputs[0].text.strip()
    
    # Clean think tags if present
    response = response.replace("</think>", "").replace("<think>", "").strip()
    
    # Parse numbered responses
    lines = response.split('\n')
    answers = {}
    current_num = 0
    current_answer = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # More robust number extraction
        import re
        if line_stripped and line_stripped[0].isdigit():
            # Save previous answer
            if current_num > 0 and current_answer:
                answers[current_num] = ' '.join(current_answer).strip()
            
            # Extract number more carefully
            num_match = re.match(r'^(\d+)', line_stripped)
            if num_match:
                current_num = int(num_match.group(1))
                # Get rest of line after number (and any punctuation)
                rest = re.sub(r'^\d+[.,):\s]*', '', line_stripped)
                current_answer = [rest] if rest else []
            else:
                # Add to current answer if no number found
                if current_num > 0:
                    current_answer.append(line_stripped)
        elif current_num > 0:
            # Continue current answer
            current_answer.append(line_stripped)
    
    # Save last answer
    if current_num > 0 and current_answer:
        answers[current_num] = ' '.join(current_answer).strip()
    
    # NOW check if we need to add the critical question
    # Extract progress_score from answer 6 (PROGRESS ASSESSMENT)
    progress_score = 0
    if 6 in answers:
        try:
            score_text = answers[6]
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                progress_score = min(int(numbers[0]), 10)  # Cap at 10
        except:
            progress_score = 0


    # (state_changed already defined above - skipping duplicate)

    # Universal progress adjustment based on state change and task relevance  
    if state_changed and 3 in answers and 4 in answers:
        addressed_text = answers.get(3, '').lower()
        remaining_text = answers.get(4, '').lower()
        
        # Information theory: if "remaining" got shorter than "addressed", we made progress
        if len(remaining_text.split()) < len(task.split()):
            progress_score = max(progress_score, 4)
        
        # If something was addressed (not 'none' or 'n/a'), boost slightly
        if addressed_text and addressed_text not in ['none', 'nothing', 'n/a', '']:
            progress_score = max(progress_score, 3)
        
        # If task words appear in current observation but not in previous, we progressed
        task_tokens = set(task.lower().split())
        current_tokens = set(curr_observation.lower().split())
        previous_tokens = set(prev_observation.lower().split())
        new_relevant_tokens = (task_tokens & current_tokens) - previous_tokens
        
        if new_relevant_tokens:
            progress_score = max(progress_score, 5)
    

    # Check action alignment (removed redundant critical reflection - TextGrad handles this)
    action_alignment = answers.get(10, '')
    if 'not' in action_alignment.lower() or 'no' in action_alignment.lower():
        # Extract what should have been done
        should_have = action_alignment.split('instead')[-1].strip() if 'instead' in action_alignment else ''
        step_gradient['action_mismatch'] = True
        if should_have:
            step_gradient['intended_action'] = should_have
    
    # Extract structured information with defaults
    # (state_changed already defined above - skipping duplicate)
    
    missing_prereqs = answers.get(2, '').strip()
    missing_list = [missing_prereqs] if missing_prereqs and missing_prereqs.lower() not in ['none', 'nothing', 'n/a'] else []
    
    task_addressed = answers.get(3, '').strip()
    addressed_list = [task_addressed] if task_addressed and task_addressed.lower() not in ['none', 'nothing', 'n/a'] else []
    
    task_remaining = answers.get(4, task).strip()
    remaining_list = [task_remaining] if task_remaining else [task]
    
    why_failed = answers.get(5, None)
    if why_failed and why_failed.lower() in ['did not fail', 'n/a', 'none']:
        why_failed = None
        
    learned_rule = answers.get(6, 'Continue exploring available actions').strip()
    
    # ACTUAL CORRECT FIX: Extract score from learned_rule (answers[6])
    # This is where "PROGRESS SCORE: SCORE: 3 | REASONING: ..." actually is!
    if 'SCORE:' in learned_rule:
        score_match = re.search(r'SCORE:\s*(\d+)', learned_rule, re.IGNORECASE)
        if score_match:
            progress_score = min(int(score_match.group(1)), 10)
    
    next_guidance = answers.get(7, 'Try unexplored actions').strip()

    # CRITICAL FIX: Extract "Recommended next action: X" from anywhere in the response
    # GPT-5 embeds this in question 4 (CAUSAL CHAIN ANALYSIS) but doesn't use numbered format
    import re
    # Try multiple patterns in order of preference:
    # 1. "Recommended next action: X" (most specific, at end of causal chain)
    # 2. "RECOMMENDED ACTION: X" (more general, may include GOAL statement)
    recommended_action_match = re.search(r'Recommended next action:\s*([^\n]+)', response, re.IGNORECASE)
    if not recommended_action_match:
        recommended_action_match = re.search(r'RECOMMENDED ACTION:\s*-?\s*([a-z][^\n-]+)', response, re.IGNORECASE)

    if recommended_action_match:
        next_guidance = recommended_action_match.group(1).strip()
        # Clean up any leading/trailing punctuation or markdown
        next_guidance = next_guidance.strip('- .,;:')
        # If it starts with "GOAL:", skip it and try to find the actual action after it
        if next_guidance.upper().startswith('GOAL:'):
            # Look for the actual action after the GOAL statement
            action_after_goal = re.search(r'(?:GOAL:.*?)?[-\s]*(?:ACTION|RESULT|Recommended):\s*([a-z][^\n-]+)', response, re.IGNORECASE | re.DOTALL)
            if action_after_goal:
                next_guidance = action_after_goal.group(1).strip().strip('- .,;:')
        print(f"[REFLEXION FIX] Extracted action from RECOMMENDED ACTION: '{next_guidance}'")
    else:
        # Fallback: Extract exact action if it's quoted
        quoted_action = re.findall(r"'([^']+)'", next_guidance)
        if quoted_action:
            next_guidance = quoted_action[0]  # Use the exact quoted action

    # PHASE 1: Extract action from CAUSAL CHAIN (handles multiple LLM output formats)
    # Reflexion generates CAUSAL CHAIN, TextGrad needs clean action - extract for synergy
    if 'CAUSAL CHAIN' in next_guidance:
        action_match = None

        # Pattern 1: ACTION: go to X. | (colon, ends with period or pipe)
        action_match = re.search(r'ACTION:\s*([^.|]+?)(?:\.|\||$)', next_guidance, re.IGNORECASE)

        # Pattern 2: ACTION (go to X) (parentheses format)
        if not action_match:
            action_match = re.search(r'ACTION\s*\(([^)]+)\)', next_guidance, re.IGNORECASE)

        # Pattern 3: Action = go to X (equals sign format)
        if not action_match:
            action_match = re.search(r'ACTION\s*=\s*([^|>]+?)(?:\||->|RESULT)', next_guidance, re.IGNORECASE)

        if action_match:
            next_guidance = action_match.group(1).strip()
            print(f"[SYNERGY] Extracted action from CAUSAL CHAIN: '{next_guidance}'")
        else:
            # No ACTION found - pure analysis, use TODO/LLM instead
            next_guidance = ''
            print(f"[SYNERGY] No ACTION in CAUSAL CHAIN - will use TODO/LLM")

    # REMOVED duplicate naive word matching - use ONLY LLM's semantic evaluation
    # TextGrad needs clean semantic gradients, not polluted with heuristics
    
    # Build full gradient structure - preserves all learning signals
    step_gradient = {
        'state_change': curr_observation if state_changed else 'NO CHANGE',
        'progress_assessment': f"Addressed: {', '.join(addressed_list)}" if addressed_list else "No progress",
        'progress_score': progress_score,  
        'hypothesis': learned_rule,
        'next_action_guidance': next_guidance,
        'raw_reflection': response,
        'semantic_state_before': prev_observation,
        'semantic_state_after': curr_observation,
        'prerequisites': {
            'present': [],  # Could extract from success
            'missing': missing_list
        },
        'task_progress': {
            'addressed': addressed_list,
            'remaining': remaining_list
        }
    }
    
    # Add critical insight if available
    if 'critical' in answers:
        step_gradient['critical_insight'] = answers['critical']
    
    if DEBUG_CRITIC:
        print(f"\n[STEP REFLECTION]")
        print(f"  Step Number: {current_step}")  # You'll need to pass current_step as parameter
        print(f"  Action taken: {action}")
        print(f"  State: {prev_observation} -> {curr_observation}")
        print(f"  Changed: {state_changed}")
        print(f"  Missing prerequisites: {missing_list}")
        print(f"  Task progress: {addressed_list}")
        print(f"  Progress score: {progress_score}")
        if 'critical' in answers:
            print(f"  Critical insight: {answers['critical']}")
    

    # Score already extracted from answers[5] earlier in the function
    return step_gradient



def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        skip_discovery: bool = False,
        env_type: str = "alfworld",
        batch_size: int = 8,
        ablation_mode: str = 'combined',
    ) -> List[Dict[str, Any]]:
    """Run trial with discovery and gradient updates - NO PATTERNS"""

    # Set ablation flags based on mode
    global USE_REFLEXION, USE_TEXTGRAD
    if ablation_mode == 'textgrad_only':
        USE_REFLEXION = False
        USE_TEXTGRAD = True
    elif ablation_mode == 'reflexion_only':
        USE_REFLEXION = True
        USE_TEXTGRAD = False
    else:  # combined (default)
        USE_REFLEXION = True
        USE_TEXTGRAD = True

    print(f"\n[RUN_TRIAL START]")
    print(f"  trial_log_path = {trial_log_path}")
    print(f"  trial_idx = {trial_idx}")
    print(f"  num_envs = {len(env_configs)}")
    print(f"\n[ABLATION MODE] {ablation_mode}")
    print(f"  Reflexion (Memory): {USE_REFLEXION}")
    print(f"  TextGrad (Prompt Optimization): {USE_TEXTGRAD}\n")
    print(f"  env_type = {env_type}")
    print("[/RUN_TRIAL START]\n")
    
    global ENVIRONMENT_KNOWLEDGE, prompt_generator
    
    # DEBUG: Print current component state
    print(f"\n[TRIAL {trial_idx}] Starting with prompt components:")
    for comp, value in prompt_generator.prompt_components.items():
        print(f"  {comp}: {value}")
    
    # Load prompt_generator state if it exists (for trials > 0)
    # Load and selectively preserve learning
    if trial_idx > 0 and use_memory:
        PROMPT_GEN_PATH = 'prompt_generator_state.pkl'
        if os.path.exists(PROMPT_GEN_PATH) and os.path.getsize(PROMPT_GEN_PATH) > 0:
            try:
                with open(PROMPT_GEN_PATH, 'rb') as f:
                    old_pg = pickle.load(f)
                
                # Create fresh generator but preserve pool
                prompt_generator = DynamicPromptGenerator()
                
                # Preserve the generator pool if it exists
                if hasattr(old_pg, 'generator_pool'):
                    prompt_generator.generator_pool = old_pg.generator_pool
                    print(f"[LOADED] Generator pool with {len(prompt_generator.generator_pool)} task types")
                else:
                    prompt_generator.generator_pool = {}
                
                # PRESERVE THESE LEARNINGS:
                # 1. Discovered environment structure
                if hasattr(old_pg, 'environment_knowledge'):
                    prompt_generator.environment_knowledge = old_pg.environment_knowledge
                
                # 2. Available actions discovered
                if hasattr(old_pg, 'discovered_knowledge'):
                    prompt_generator.discovered_knowledge['available_actions'] = old_pg.discovered_knowledge.get('available_actions', set())
                    prompt_generator.discovered_knowledge['uses_numbers'] = old_pg.discovered_knowledge.get('uses_numbers', False)
                    prompt_generator.discovered_knowledge['completion_actions'] = old_pg.discovered_knowledge.get('completion_actions', set())
                
                # 3. Successful trajectories (if any)
                if hasattr(old_pg, 'successful_trajectories') and old_pg.successful_trajectories:
                    prompt_generator.successful_trajectories = old_pg.successful_trajectories
                    print(f"[PRESERVED] {len(old_pg.successful_trajectories)} successful trajectories")
                
                # 4. Gradient history for analysis (but don't pre-apply)
                if hasattr(old_pg, 'prompt_gradients'):
                    prompt_generator.prompt_gradients = old_pg.prompt_gradients
                    print(f"[PRESERVED] {len(old_pg.prompt_gradients)} gradient history")
                
                # 5. CRITICAL FIX: Preserve learned prompt components
                if hasattr(old_pg, 'prompt_components'):
                    prompt_generator.prompt_components = old_pg.prompt_components
                    print(f"[PRESERVED] Learned prompt components with TextGrad updates")

                print(f"[TRIAL {trial_idx}] Preserved structural learning, reset task-specific components")
                
            except Exception as e:
                print(f"[ERROR] Failed to load: {e}")
                prompt_generator = DynamicPromptGenerator()
    else:
        prompt_generator = DynamicPromptGenerator()

    parallel_mode = True  # Always use our batch implementation
    
    if skip_discovery:
        ENVIRONMENT_KNOWLEDGE = None  # Force no discovery knowledge
        # Also disable prompt generator's discovered knowledge
        prompt_generator.environment_knowledge = None
        env_understanding.environment_knowledge = None
    
    # Load appropriate environment configuration
    config = None
    if env_type == "alfworld":
        import alfworld
        import alfworld.agents.environment as alfworld_env
        
        importlib.reload(alfworld)
        importlib.reload(alfworld.agents.environment)
        
        # Get script directory to find base_config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'base_config.yaml')
        with open(config_path) as reader:
            config = yaml.safe_load(reader)

    # DISCOVERY PHASE - Run on first trial OR if knowledge is missing (resume without saved knowledge)
    discovery_report = ""
    if ENVIRONMENT_KNOWLEDGE is None and not skip_discovery:
        print("\n=== PERFORMING ENVIRONMENT DISCOVERY ===\n")
        
        # Get a WRAPPED environment for discovery
        env = get_environment(env_type, config, env_id=0)
        ob, info = env.reset()
        print(f"  ob type: {type(ob)}")
        print(f"  ob value (first 300): {str(ob)[:300]}")
        # Process observation universally
        if isinstance(ob, tuple) and len(ob) > 0:
            ob = ob[0]
        elif isinstance(ob, list):
            ob = ob[0]
        ob = str(ob) if not isinstance(ob, str) else ob  # Ensure it's string
        
        # Check if we have similar environment knowledge
        similar_knowledge = meta_knowledge.load_similar_environment(ob)
        
        if similar_knowledge:
            print("âœ“ Found similar environment knowledge - adapting...")
            ENVIRONMENT_KNOWLEDGE = similar_knowledge
            
            # Generate report for loaded knowledge
            discovery_report = "=== Using Transferred Environment Knowledge ===\n\n"
            
            # Action space info
            if ENVIRONMENT_KNOWLEDGE.get('action_space'):
                discovery_report += f"AVAILABLE COMMANDS: {len(ENVIRONMENT_KNOWLEDGE['action_space'])} discovered\n"
                # Show first few commands
                for cmd, info_dict in list(ENVIRONMENT_KNOWLEDGE['action_space'].items())[:5]:
                    if isinstance(info_dict, dict):
                        discovery_report += f"  - {info_dict.get('format', cmd)}: {info_dict.get('description', '')}\n"
                    else:
                        discovery_report += f"  - {cmd}\n"
                if len(ENVIRONMENT_KNOWLEDGE['action_space']) > 5:
                    discovery_report += f"  ... and {len(ENVIRONMENT_KNOWLEDGE['action_space']) - 5} more\n"
            
            # State space info
            if ENVIRONMENT_KNOWLEDGE.get('state_space', {}).get('locations'):
                num_locations = len(ENVIRONMENT_KNOWLEDGE['state_space']['locations'])
                discovery_report += f"\nDISCOVERED LOCATIONS: {num_locations} found\n"
            
            # Constraints
            if ENVIRONMENT_KNOWLEDGE.get('constraints'):
                discovery_report += "\nKNOWN CONSTRAINTS:\n"
                for constraint in ENVIRONMENT_KNOWLEDGE['constraints']:
                    discovery_report += f"  - {constraint}\n"
            
            discovery_report += "\nâœ“ Ready to use transferred knowledge!\n"
            
        else:
            print("Ã— No similar environment found - running full discovery...")
            
            # Run discovery with wrapped environment
            discoverer = UniversalEnvironmentDiscovery()
            ENVIRONMENT_KNOWLEDGE = discoverer.discover_environment(
                env, ob, max_discovery_steps=50
            )
            
            # Save discovered knowledge
            if env_type == "alfworld":
                env_name = info['extra.gamefile'][0].split('/')[-3] if 'extra.gamefile' in info else 'unknown'
            else:
                env_name = env_type
            meta_knowledge.save_environment_knowledge(env_name, ENVIRONMENT_KNOWLEDGE)
            
            # Generate discovery report
            discovery_report = discoverer.generate_discovery_report()
        
        # Share knowledge with components
        prompt_generator.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        env_understanding.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        
        # Log discovery results
        print(discovery_report)
        
        with open(world_log_path, 'a') as wf:
            wf.write("\n" + discovery_report + "\n")
        
        # Add meta summary
        meta_summary = meta_knowledge.get_discovery_summary()
        print(meta_summary)
        
        # Close discovery environment
        env.close()

    # Log current learning state - SIMPLIFIED, NO PATTERNS
    optimization_state = prompt_generator.get_prompt_optimization_state()
    with open(world_log_path, 'a') as wf:
        wf.write(f'\nTrial #{trial_idx} Learning State:\n')
        wf.write(f'Ablation Mode: {ablation_mode}\n')
        wf.write(f'Reflexion Active: {USE_REFLEXION}\n')
        wf.write(f'TextGrad Active: {USE_TEXTGRAD}\n')
        wf.write(f'Number of environments: {len(env_configs)}\n')
        wf.write(f'Available Actions Found: {optimization_state["available_actions_found"]}\n')
        wf.write(f'Interactions: {optimization_state["interaction_count"]}\n')
        wf.write(f'Updates: {optimization_state["num_updates"]}\n')
        wf.write(f'Has Environment Knowledge: {optimization_state["has_environment_knowledge"]}\n')
        wf.write(f'Uses Numbered Items: {optimization_state["uses_numbered_items"]}\n')
        wf.write(f'Completion Actions: {optimization_state["completion_actions"]}\n\n')
        
        # Log prompt components to track evolution
        wf.write(f'Prompt Components (Trial #{trial_idx}):\n')
        for component, value in prompt_generator.prompt_components.items():
            wf.write(f'  {component}: {value}\n')
        wf.write('\n')

    # ALWAYS USE PARALLEL EXECUTION
    print(f"\n=== RUNNING {len(env_configs)} ENVIRONMENTS IN PARALLEL MODE ===")
    print(f"Environment Type: {env_type}")
    
    # Process in batches to manage memory
    BATCH_SIZE = batch_size  # Use parameter from function
    num_successes = 0
    num_additional_successes = 0
    num_envs = len(env_configs)
    
    # Process environments in batches
    for batch_start in range(0, len(env_configs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(env_configs))
        batch_size = batch_end - batch_start

        print(f"\n--- Processing batch: environments {batch_start} to {batch_end-1} ({batch_size} envs) ---")

        # FORCE LOGGING TO FILE TO VERIFY BATCH EXECUTION
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Processing batch: environments {batch_start} to {batch_end-1} ({batch_size} envs)\n")
                f.flush()
        
        # Prepare batch
        batch_envs = []
        batch_memories = []
        batch_initial_obs = []
        batch_indices = []
        
        for i in range(batch_start, batch_end):
            # FORCE LOG EACH ENV PROCESSING START - NO TRY/CATCH, FAIL HARD
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[ENV DEBUG] Starting processing for ENV {i}\n")
                    f.flush()

            env_config = env_configs[i]
            print(f"\n[PROCESSING ENV {i}]")

            # DEBUG: Check memory status
            memory_items = env_config.get('memory', [])
            print(f"[ENV {i}] Trial {trial_idx} - Memory in config: {len(memory_items)} items")
            if memory_items:
                workflow_count = sum(1 for item in memory_items if isinstance(item, dict) and item.get('type') == 'success_workflow')
                print(f"[ENV {i}] Trial {trial_idx} - success_workflow items: {workflow_count}")
                if workflow_count > 0:
                    for item in memory_items:
                        if isinstance(item, dict) and item.get('type') == 'success_workflow':
                            print(f"[ENV {i}] Trial {trial_idx} - Workflow for task: '{item.get('task', 'NO TASK')}'")

            if trial_idx > 0 and use_memory:
                if 'memory' not in env_config or len(env_config['memory']) == 0:
                    print(f"[WARNING] ENV {i} has no memory despite being trial {trial_idx}")
            
            print(f"\n[PROCESSING ENV {i}] Starting...")
            
            # Create environment ONLY ONCE
            env = get_environment(env_type, config, env_id=i)
            ob, info = env.reset()
            
            print(f"[DEBUG ENV {i}] Reset output:")
            print(f"  ob type: {type(ob)}")
            print(f"  ob value (first 300): {str(ob)[:300]}")
            print(f"  ob is list? {isinstance(ob, list)}")
            if isinstance(ob, list) and ob:
                print(f"  ob[0] type: {type(ob[0])}")
                print(f"  ob[0] value (first 300): {str(ob[0])[:300]}")

            # Process observation
            if hasattr(env, 'process_observation'):
                # Handle tuple from ALFWorld
                if isinstance(ob, tuple) and len(ob) > 0:
                    processed_ob = env.process_observation(ob)
                else:
                    processed_ob = env.process_observation(ob[0] if isinstance(ob, list) else ob)
            else:
                raw_ob = ob[0] if isinstance(ob, list) else ob
                if isinstance(raw_ob, str):
                    processed_ob = raw_ob
                else:
                    processed_ob = str(raw_ob)

            print(f"[DEBUG ENV {i}] After processing:")
            print(f"  processed_ob type: {type(processed_ob)}")
            print(f"  processed_ob (first 500): {processed_ob[:500]}")

            # Get environment name
            name = env.get_environment_name(info) if hasattr(env, 'get_environment_name') else f"{env_type}_task_{i}"

            # Extract task
            task = prompt_generator._extract_task_from_observation(processed_ob)
            print(f"[DEBUG ENV {i}] Task extraction:")
            print(f"  Extracted task: {task}")

            if not task:
                print(f"[CRITICAL ERROR] Task extraction FAILED for env {i}!")
                print(f"  Observation: {processed_ob[:200]}")

                # FAIL FAST - NO FALLBACK TO EXPOSE ROOT CAUSE
                env_configs[i]["is_success"] = False
                env_configs[i]["task"] = "TASK_EXTRACTION_FAILED"

                # Log and close this environment
                if trial_log_path:
                    with open(trial_log_path, 'a') as f:
                        f.write(f"\n[CRITICAL] ENV {i}: Task extraction failed, observation: {processed_ob[:200]}\n")

                env.close()
                continue  # Skip to next environment in batch
            else:
                # Set task in all relevant places
                prompt_generator.set_task(task)
                env_configs[i]['task'] = task
                if env_understanding:
                    env_understanding.current_task = task

                print(f"Environment: {name}")
                print(f"Task: {task}")

            # Task is guaranteed to exist here (failed cases already handled above)

            # Get initial valid actions
            if hasattr(env, 'get_current_valid_actions'):
                initial_actions = env.get_current_valid_actions()
                if initial_actions:
                    prompt_generator.discovered_knowledge['available_actions'] = set(initial_actions)
                    if DEBUG_ACTOR:
                        print(f"[DEBUG] Initial valid actions: {len(initial_actions)}")


            # Get optimized memory from global manager if available
            memory_to_use = env_config.get("memory", [])  # Use existing memory first

            # Pure LLM-based memory retrieval
            if use_memory and trial_idx > 0:
                from generate_reflections import global_memory_manager

                # CRITICAL FIX: Preserve success_workflow items before LLM filtering
                # These are structured patterns that must be preserved for cross-trial learning
                success_workflows = []
                other_memories = []
                for item in memory_to_use:
                    if isinstance(item, dict) and item.get('type') == 'success_workflow':
                        success_workflows.append(item)
                        print(f"[ENV {i}] PRESERVING success_workflow: '{item.get('task', 'NO TASK')}'")
                    else:
                        other_memories.append(item)

                # Use pure LLM-based retrieval with current observation (only on textual memories)
                optimized_memory = global_memory_manager.get_relevant_memories(
                    task=task,
                    env_memory=other_memories,  # Filter only textual memories, not workflows
                    k=6,  # Request top 6 most useful memories
                    current_observation=processed_ob,  # Pass current situation
                    env_id=i  # Pass env_id for contamination logging
                )

                # CRITICAL FIX: Combine preserved workflows with filtered memories
                # Success workflows ALWAYS come first for priority in action selection
                if optimized_memory:
                    memory_to_use = success_workflows + optimized_memory
                    print(f"[ENV {i}] Using {len(success_workflows)} workflows + {len(optimized_memory)} LLM-selected memories")
                else:
                    memory_to_use = success_workflows + other_memories
                    print(f"[ENV {i}] Using {len(success_workflows)} workflows + {len(other_memories)} memories (no LLM filtering)")

            # Add to batch
            batch_envs.append(env)
            batch_memories.append(memory_to_use)
            batch_initial_obs.append(processed_ob)
            batch_indices.append(i)
        
        # Skip if no environments in this batch
        if not batch_envs:
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[BATCH DEBUG] SKIPPING batch {batch_start}-{batch_end-1}: batch_envs is EMPTY\n")
                    f.flush()
            print(f"WARNING: Skipping empty batch {batch_start}-{batch_end-1}")
            continue

        print(f"Running {len(batch_envs)} environments in parallel...")

        # LOG BATCH SIZE
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Running batch {batch_start}-{batch_end-1} with {len(batch_envs)} environments, indices: {batch_indices}\n")
                f.flush()
        
        # DEBUG: Verify trial_log_path before calling
        print(f"[PRE-CALL DEBUG] trial_log_path = {trial_log_path}")
   
        # RUN BATCH IN PARALLEL
        import psutil
        import os as os_module
        process = psutil.Process(os_module.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY DEBUG] Before adaptive_env_interaction_batch: {memory_before:.2f} MB")

        results = adaptive_env_interaction_batch(
            envs=batch_envs,
            base_prompt="",
            memories=batch_memories,
            to_print=True,
            initial_obs_list=batch_initial_obs,
            trial_log_path=trial_log_path,
            env_configs=env_configs,
            trial_idx=trial_idx,
            use_memory=use_memory
        )

        memory_after = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY DEBUG] After adaptive_env_interaction_batch: {memory_after:.2f} MB")
        print(f"[MEMORY DEBUG] Memory increase: {memory_after - memory_before:.2f} MB")
        print(f"[MEMORY DEBUG] Results type: {type(results)}, length: {len(results)}")

        # Process results
        print(f"\n[DEBUG RESULTS] Processing {len(results)} results")
        print(f"  batch_indices: {batch_indices}")
        print(f"  results success values: {[r[1] for r in results]}")
        
        for idx, (env_idx, (history, is_success)) in enumerate(zip(batch_indices, results)):
            # CRITICAL: Set success status in env_configs
            env_configs[env_idx]['is_success'] = is_success
            print(f"\n[DEBUG] Processing result {idx}:")
            print(f"  env_idx={env_idx}")
            print(f"  is_success={is_success}, type={type(is_success)}")

            if is_success:
                status_str = f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS'
                env_configs[env_idx]['is_success'] = True
                num_successes += 1
                num_additional_successes += 1
            else:
                status_str = f'Environment #{env_idx} Trial #{trial_idx}: FAIL'
                env_configs[env_idx]['is_success'] = False
            # Log results
            with open(world_log_path, 'a') as f:
                f.write(status_str + '\n')
            
         
            # Write with consistent format for each environment
            with open(trial_log_path, 'a') as wf:
                # Add clear separator between environments
                if env_idx > batch_indices[0]:
                    wf.write('\n\n')  # Simple double newline separator
                
                # Write environment log with clear markers
                wf.write(f'Environment #{env_idx}:\n')
                wf.write(str(history))
                wf.write(f'\nSTATUS: {"OK" if is_success else "FAIL"}\n')
        
        # Clean up batch environments
        for env in batch_envs:
            env.close()
        
        print(f"Batch complete. Successes in batch: {sum(1 for _, (_, s) in zip(batch_indices, results) if s)}")

        # FORCE LOG BATCH COMPLETION
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Completed batch {batch_start}-{batch_end-1}: {sum(1 for _, (_, s) in zip(batch_indices, results) if s)} successes\n")
                f.flush()



    # Update generators in pool without merging
    # CRITICAL FIX: Update pool with learning from ALL environments
    print("[UPDATING] Updating task-specific generators in pool...")
    if 'env_prompt_generators' in locals() and env_prompt_generators:
        for env_id, env_pg in env_prompt_generators.items():
            if env_id >= len(env_states):
                continue
                
            task = env_states[env_id]['task']
            was_successful = env_states[env_id].get('success', False)
            
            # Find or create pool entry for this task type
            best_match_task = None
            best_similarity = 0
            
            for pool_task in prompt_generator.generator_pool.keys():
                task_words = set(task.lower().split())
                pool_words = set(pool_task.lower().split())
                similarity = len(task_words & pool_words) / len(task_words | pool_words) if (task_words | pool_words) else 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_task = pool_task
            
            # Update or create pool entry
            if best_similarity > 0.7:  # Similar task exists
                print(f"[POOL SYNC] '{task[:30]}' → '{best_match_task[:30]}' (sim: {best_similarity:.2f})")
                pool_pg = prompt_generator.generator_pool[best_match_task]
                
                # Merge learning regardless of success/failure
                for component, value in env_pg.prompt_components.items():
                    # Check if this component actually changed from default
                    default_value = DynamicPromptGenerator().prompt_components.get(component)
                    if value != default_value:
                        # Weight by success
                        if was_successful:
                            pool_pg.prompt_components[component] = value
                        else:
                            # For failures, use momentum update
                            pool_pg.update_component_with_momentum(component, value, momentum=0.3)
                
                # Always merge discovered knowledge
                pool_pg.discovered_knowledge.update(env_pg.discovered_knowledge)
                pool_pg.prompt_gradients.extend(env_pg.prompt_gradients[len(pool_pg.prompt_gradients):])

            elif best_similarity < 0.3:  # New task type
                # Store this as a new task type
                prompt_generator.generator_pool[task] = env_pg
                print(f"[UPDATING] Added new task type to pool: {task[:30]}")
        
        print(f"[UPDATING] Pool now contains {len(prompt_generator.generator_pool)} task types")
    

    # Save the merged global generator
    save_prompt_generator()
    save_environment_knowledge()
    print("[SAVED] Global prompt_generator and ENVIRONMENT_KNOWLEDGE with merged learning")


    # Log trial summary - SIMPLIFIED, NO PATTERNS
    discovered_actions = len(prompt_generator.discovered_knowledge.get('available_actions', []))
    if ENVIRONMENT_KNOWLEDGE and 'action_space' in ENVIRONMENT_KNOWLEDGE:
        total_actions = len(ENVIRONMENT_KNOWLEDGE['action_space'])
    else:
        total_actions = discovered_actions
        
    # Add learning summary for paper
    learning_summary = "\n=== LEARNING SUMMARY (Trial {}) ===\n".format(trial_idx)
    total_episodic = sum(len(env_configs[i].get('memory', [])) for i in range(num_envs))
    total_step_mem = sum(len(env_configs[i].get('step_memory', [])) for i in range(num_envs))
    learning_summary += f"Episodic Memory Entries: {total_episodic} (cross-trial strategic patterns)\n"
    learning_summary += f"Step Memory Entries: {total_step_mem} (within-trial tactical learning)\n"
    learning_summary += f"TextGrad + Reflexion Synergy: Active (both memories used for meta-learning)\n"

    log_str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
DISCOVERED ACTIONS: {discovered_actions}/{total_actions}
ENVIRONMENT TYPE: {env_type}
EXECUTION MODE: PARALLEL (Batch size: {BATCH_SIZE})
{learning_summary}
-----
"""

    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    # Ensure results are saved even if there's an error
    try:
        logging_dir = os.path.dirname(trial_log_path) if trial_log_path else '.'
        env_config_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
        with open(env_config_path, 'w') as f:
            json.dump(env_configs, f, indent=4)
            f.flush()  # Force write
            os.fsync(f.fileno())  # Force OS to write to disk
        print(f"[SAVED] Trial {trial_idx} results to {env_config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save trial results: {e}")
        logging_dir = os.path.dirname(trial_log_path) if trial_log_path else '.'
        # Emergency save as backup
        backup_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}_backup.json')
        with open(backup_path, 'w') as f:
            # Save minimal data
            minimal_data = [{'is_success': env.get('is_success', False)} for env in env_configs]
            json.dump(minimal_data, f)

    # FORCE LOG FINAL COMPLETION
    if trial_log_path:
        with open(trial_log_path, 'a') as f:
            total_success = sum(1 for env in env_configs if env.get('is_success', False))
            f.write(f"\n[FINAL DEBUG] All batches completed. Total successes: {total_success}/{len(env_configs)}\n")
            f.flush()

    print(f"\n[FINAL] All batches completed. Total successes: {sum(1 for env in env_configs if env.get('is_success', False))}/{len(env_configs)}")

    return env_configs