"""
Causal Memory Compression System

Extracts causal insights from verbose trajectories to enable efficient cross-trial learning.
Compresses both success patterns and failure anti-patterns without losing critical context.

NO ENVIRONMENT-SPECIFIC HARDCODING - learns action→effect mappings from observation.
"""

import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict


def extract_verb(action: str) -> str:
    """
    Extract the primary verb from an action string.

    Examples:
        "cool pan 1 with fridge 1" → "cool"
        "go to stoveburner 2" → "go"
        "take pan 1 from stoveburner 2" → "take"
    """
    action = action.strip().lower()
    # First word is usually the verb
    parts = action.split()
    if len(parts) > 0:
        return parts[0]
    return ""


def extract_object(action: str) -> str:
    """
    Extract the object being acted upon.

    Examples:
        "cool pan 1 with fridge 1" → "pan 1"
        "take pan 1 from stoveburner 2" → "pan 1"
        "go to fridge 1" → "fridge 1"
    """
    action = action.strip().lower()

    # Pattern: verb object [with/from/to] location
    # Extract the first noun phrase after verb
    parts = action.split()
    if len(parts) >= 2:
        # Usually verb + object + number
        if len(parts) >= 3 and parts[2].isdigit():
            return f"{parts[1]} {parts[2]}"
        return parts[1]
    return ""


def extract_appliance(action: str) -> str:
    """
    Extract the appliance/location from an action.

    Examples:
        "cool pan 1 with fridge 1" → "fridge"
        "clean pan 1 with sinkbasin 1" → "sinkbasin"
        "go to stoveburner 2" → "stoveburner"
    """
    action = action.strip().lower()

    # Look for prepositions that indicate location/appliance
    if " with " in action:
        parts = action.split(" with ")
        if len(parts) > 1:
            # Extract appliance name (without number)
            appliance_part = parts[1].strip().split()
            return appliance_part[0] if appliance_part else ""

    if " from " in action:
        parts = action.split(" from ")
        if len(parts) > 1:
            appliance_part = parts[1].strip().split()
            return appliance_part[0] if appliance_part else ""

    if " to " in action:
        parts = action.split(" to ")
        if len(parts) > 1:
            appliance_part = parts[1].strip().split()
            return appliance_part[0] if appliance_part else ""

    if " in " in action:
        parts = action.split(" in ")
        if len(parts) > 1:
            appliance_part = parts[1].strip().split()
            return appliance_part[0] if appliance_part else ""

    return ""


def extract_task_verb(task: str) -> str:
    """
    Extract the primary action verb from a task description.

    Examples:
        "cool some pan and put it in countertop" → "cool"
        "put a hot apple in fridge" → "put"
        "clean a plate and put it in cabinet" → "clean"
    """
    task = task.strip().lower()

    # Common task verbs to look for
    action_verbs = ['cool', 'heat', 'clean', 'put', 'place', 'move', 'find', 'take']

    for verb in action_verbs:
        if task.startswith(verb) or f" {verb} " in task:
            return verb

    # Fallback: first word
    parts = task.split()
    if parts:
        return parts[0]
    return ""


def is_state_changing(action: str) -> bool:
    """
    Determine if an action changes environment state (vs just navigation/observation).

    State-changing: take, put, cool, clean, open, toggle, use
    Non-state-changing: go, examine, look, inventory
    """
    action = action.strip().lower()
    verb = extract_verb(action)

    navigation_verbs = ['go', 'examine', 'look', 'inventory']
    return verb not in navigation_verbs


def extract_effect(observation: str) -> str:
    """
    Extract the primary effect/result from an observation.

    Examples:
        "You cool the pan 1 using the fridge 1." → "cooled"
        "You clean the pan 1 using the sinkbasin 1." → "cleaned"
        "Nothing happens." → "no effect"
    """
    obs = observation.strip().lower()

    if "nothing happens" in obs or "don't see" in obs or "can't" in obs:
        return "no effect"

    # Look for past tense verbs indicating effect
    effect_patterns = [
        (r'you (cool|heat|clean|wash|open|close|toggle|take|put|move) ', r'\1ed'),
        (r'you arrive at', 'location changed'),
        (r'you (\w+) the', r'\1ed')
    ]

    for pattern, replacement in effect_patterns:
        match = re.search(pattern, obs)
        if match:
            if replacement.startswith('\\'):
                return match.group(1) + 'ed'
            return replacement

    return "unknown effect"


def compress_failure_pattern(
    failed_attempts: List[Tuple[str, str]],
    task: str,
    llm_causal_insight: str = None
) -> Dict[str, Any]:
    """
    Compress multiple repeated failures into a single causal anti-pattern insight.

    Args:
        failed_attempts: List of (action, observation) tuples that failed
        task: The task description
        llm_causal_insight: Optional LLM-generated explanation of WHY failures occurred

    Returns:
        Compressed insight with causal explanation
    """
    if not failed_attempts:
        # REMOVED FALLBACK: Even with no failed_attempts, create a generic failure pattern
        raise ValueError(f"compress_failure_pattern called with empty failed_attempts for task '{task}'. This should not happen!")

    # Detect repetition
    action_counts = defaultdict(int)
    for action, _ in failed_attempts:
        # Normalize action (remove numbers for counting)
        normalized = re.sub(r'\d+', 'X', action)
        action_counts[normalized] += 1

    # Find most repeated action
    most_repeated = max(action_counts.items(), key=lambda x: x[1])
    repeated_pattern, count = most_repeated

    if count < 2:
        # Not enough repetition, but still create a failure pattern
        # REMOVED FALLBACK: Always return a dict
        task_verb = extract_task_verb(task)
        insight = f"Task '{task}' failed. Attempted various actions but no clear repetitive pattern detected. May need better exploration strategy or task understanding."
        return {
            'type': 'failure_pattern',
            'insight': insight,
            'task': task,
            'task_type': task_verb if task_verb else task.split()[0] if task else 'unknown',
            'failed_actions': [action for action, _ in failed_attempts],
            'weight': 2.0  # Lower weight for non-repetitive failures
        }

    # Extract causal information
    sample_action = failed_attempts[0][0]
    action_verb = extract_verb(sample_action)
    task_verb = extract_task_verb(task)
    appliance = extract_appliance(sample_action)

    # Generate causal insight (use LLM insight if provided, otherwise use rule-based)
    if llm_causal_insight:
        # Hybrid: LLM provides WHY, rules provide structure
        insight = f"""ANTI-PATTERN (Avoid Repetition):
❌ Task: '{task}'
❌ Attempted: '{action_verb}' action {count}× with {appliance}
ROOT CAUSE (LLM Analysis): {llm_causal_insight.strip()}
Wasted attempts: {count}"""
    elif action_verb and task_verb and action_verb != task_verb:
        insight = f"""ANTI-PATTERN (Avoid Repetition):
❌ Task: '{task}' requires '{task_verb}' effect
❌ Attempted: '{action_verb}' action {count}× with {appliance}
ROOT CAUSE: '{action_verb}' produces different effect than '{task_verb}'
LESSON: Action verb must match task requirement
Wasted attempts: {count}"""

    elif appliance:
        insight = f"""ANTI-PATTERN (Avoid Repetition):
❌ Tried '{repeated_pattern.replace('X', '[N]')}' {count}× without progress
❌ Used appliance: {appliance}
LESSON: This action-appliance combination doesn't advance task '{task}'
Wasted attempts: {count}"""

    else:
        insight = f"""ANTI-PATTERN (Avoid Repetition):
❌ Repeated '{repeated_pattern.replace('X', '[N]')}' {count}× without progress
LESSON: This action doesn't advance task '{task}'
Wasted attempts: {count}"""

    return {
        'type': 'failure_pattern',
        'insight': insight,
        'pattern': repeated_pattern,
        'count': count,
        'task_type': extract_task_verb(task),
        'action_verb': action_verb,
        'task_verb': task_verb,
        'appliance': appliance,
        'weight': 5.0
    }


def extract_partial_success_from_failure(
    trajectory: List[Tuple[str, str, str]],
    task: str,
    progress_scores: List[int] = None
) -> Dict[str, Any]:
    """
    Extract what DID work from a failed attempt - preserve the good parts!

    Even if the agent failed overall, some steps made progress.
    This function identifies which actions were productive and should be kept.

    Args:
        trajectory: List of (action, observation, reasoning) tuples
        task: The task description
        progress_scores: Optional list of progress scores for each step (0-10)

    Returns:
        Dict with partial success insights, or None if nothing useful
    """
    if not trajectory:
        return None

    productive_steps = []

    # STRATEGY 1: Use progress scores if available
    if progress_scores and len(progress_scores) == len(trajectory):
        for i, ((action, obs, reasoning), score) in enumerate(zip(trajectory, progress_scores)):
            if score >= 5:  # Progress score of 5+ means this step made meaningful progress
                verb = extract_verb(action)
                obj = extract_object(action)
                appliance = extract_appliance(action)
                productive_steps.append({
                    'step': i,
                    'action': action,
                    'verb': verb,
                    'object': obj,
                    'appliance': appliance,
                    'progress_score': score,
                    'observation': obs[:100]  # First 100 chars
                })

    # STRATEGY 2: Detect state-changing actions (likely productive)
    if not productive_steps:
        for i, (action, obs, reasoning) in enumerate(trajectory):
            if is_state_changing(action):
                # Check if observation shows new information or items
                obs_lower = obs.lower()
                if any(indicator in obs_lower for indicator in ['you take', 'you put', 'you open',
                                                                  'you clean', 'you heat', 'you cool',
                                                                  'you see a', 'you find']):
                    verb = extract_verb(action)
                    obj = extract_object(action)
                    appliance = extract_appliance(action)
                    productive_steps.append({
                        'step': i,
                        'action': action,
                        'verb': verb,
                        'object': obj,
                        'appliance': appliance,
                        'observation': obs[:100]
                    })

    if not productive_steps:
        return None  # Nothing useful to extract

    # Build partial success insight
    task_verb = extract_task_verb(task)
    action_sequence = ' → '.join([s['verb'] for s in productive_steps[:5]])

    insight = f"""PARTIAL SUCCESS (Learn from what worked):
✓ Task: {task} (Failed overall, but made progress)
✓ Productive Actions: {action_sequence}
✓ What Worked: {len(productive_steps)} steps made measurable progress
✓ Key Objects Handled: {', '.join(set(s['object'] for s in productive_steps if s['object']))}
✓ Useful Appliances: {', '.join(set(s['appliance'] for s in productive_steps if s['appliance']))}
LESSON: Reuse these productive steps, but identify what was missing for completion"""

    return {
        'type': 'partial_success',
        'insight': insight,
        'task_type': task_verb,
        'productive_steps': productive_steps,
        'productive_action_sequence': [s['action'] for s in productive_steps],
        'weight': 7.0  # Higher than failure (5.0), lower than full success (10.0)
    }


def compress_success_pattern(
    trajectory: List[Tuple[str, str, str]],
    task: str,
    llm_success_insight: str = None
) -> Dict[str, Any]:
    """
    Compress successful trajectory into reusable causal workflow.

    Args:
        trajectory: List of (action, observation, reasoning) tuples
        task: The task description
        llm_success_insight: Optional LLM-generated explanation of WHY success occurred

    Returns:
        Compressed success workflow with causal chain
    """
    if not trajectory:
        return None

    # Extract only state-changing steps (filter navigation)
    key_steps = []
    for action, obs, _ in trajectory:
        if is_state_changing(action):
            effect = extract_effect(obs)
            key_steps.append({
                'action': action,
                'verb': extract_verb(action),
                'appliance': extract_appliance(action),
                'effect': effect
            })

    if not key_steps:
        return None

    # Build causal workflow
    step_descriptions = []
    appliances_used = set()

    for i, step in enumerate(key_steps, 1):
        step_descriptions.append(f"{i}. {step['action']} → {step['effect']}")
        if step['appliance']:
            appliances_used.add(step['appliance'])

    # Generate causal explanation
    task_verb = extract_task_verb(task)
    relevant_steps = [s for s in key_steps if s['verb'] == task_verb or task_verb in s['effect']]

    if llm_success_insight:
        # Use LLM's explanation if provided
        causal_explanation = llm_success_insight.strip()
    elif relevant_steps:
        critical_step = relevant_steps[0]
        causal_explanation = f"Task '{task}' achieved by '{critical_step['verb']}' action with {critical_step['appliance']}"
    else:
        causal_explanation = f"Task '{task}' completed through {len(key_steps)} state-changing actions"

    # CRITICAL: Store FULL action sequence for exact replay
    full_action_sequence = [action for action, obs, _ in trajectory]

    # Build detailed step-by-step breakdown for actionable learning
    detailed_steps = []
    for i, step in enumerate(key_steps[:8], 1):  # Show up to 8 critical steps
        detailed_steps.append(f"  {i}. {step['action']} → {step['effect']}")

    detailed_steps_text = "\n".join(detailed_steps) if detailed_steps else "  No key steps identified"

    workflow = f"""SUCCESS WORKFLOW:
✓ Task: {task}
✓ Critical Appliances: {', '.join(sorted(appliances_used)) if appliances_used else 'none'}
✓ Why It Worked: {causal_explanation}

Step-by-Step Breakdown (state-changing actions):
{detailed_steps_text}

Total: {len(full_action_sequence)} steps ({len(key_steps)} state-changing)"""

    return {
        'type': 'success_workflow',
        'insight': workflow,
        'task': task,  # CRITICAL: Store full task for contamination prevention
        'task_type': task_verb,
        'key_steps': key_steps,
        'appliances': list(appliances_used),
        'full_action_sequence': full_action_sequence,  # For exact replay optimization
        'weight': 10.0
    }


def compress_cross_trial_memory(trial_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge redundant insights across trials, keeping only unique causal patterns.

    Args:
        trial_memories: All memory entries from multiple trials

    Returns:
        Compressed list of unique causal insights with confidence scores
    """
    if not trial_memories:
        return []

    # Group by pattern type and task type
    patterns = defaultdict(list)

    for memory in trial_memories:
        if 'type' not in memory or 'insight' not in memory:
            continue

        mem_type = memory['type']
        task_type = memory.get('task_type', 'unknown')

        # Create grouping key
        if mem_type == 'failure_pattern':
            # Group failures by the specific pattern
            pattern_key = f"{mem_type}_{task_type}_{memory.get('pattern', 'unknown')}"
        elif mem_type == 'success_workflow':
            # Group successes by task type and appliances used
            appliances = '_'.join(sorted(memory.get('appliances', [])))
            pattern_key = f"{mem_type}_{task_type}_{appliances}"
        else:
            pattern_key = f"{mem_type}_{task_type}"

        patterns[pattern_key].append(memory)

    # Merge redundancies
    compressed = []

    for pattern_key, instances in patterns.items():
        if len(instances) > 1:
            # Multiple trials confirmed same pattern → HIGH confidence
            merged = {
                'pattern_key': pattern_key,
                'type': instances[0]['type'],
                'insight': instances[0]['insight'],  # Use first instance's insight
                'task_type': instances[0].get('task_type', 'unknown'),
                'evidence_count': len(instances),
                'confidence': min(1.0, len(instances) * 0.3),  # 30% per confirmation, max 100%
                'weight': instances[0].get('weight', 1.0) * (1 + len(instances) * 0.2),  # Boost weight
                'confirmed_across_trials': True
            }
            compressed.append(merged)
        else:
            # Single instance - keep as-is
            single = instances[0].copy()
            single['evidence_count'] = 1
            single['confidence'] = 0.3  # Low confidence (only one trial)
            single['confirmed_across_trials'] = False
            compressed.append(single)

    # Sort by confidence and weight
    compressed.sort(key=lambda x: (x.get('confidence', 0), x.get('weight', 0)), reverse=True)

    return compressed


def retrieve_relevant_memory(
    compressed_memory: List[Dict[str, Any]],
    current_task: str,
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve most relevant memory insights for current task.

    Args:
        compressed_memory: Compressed causal insights
        current_task: Current task description
        limit: Maximum number of insights to return

    Returns:
        Top N most relevant insights
    """
    if not compressed_memory:
        return []

    current_task_type = extract_task_verb(current_task)

    # Score each memory by relevance
    scored_memories = []
    for memory in compressed_memory:
        score = 0.0

        # Match task type
        if memory.get('task_type') == current_task_type:
            score += 10.0

        # Boost high-confidence patterns
        score += memory.get('confidence', 0) * 5.0

        # Boost high-weight patterns
        score += memory.get('weight', 0) * 0.5

        # Prefer patterns confirmed across trials
        if memory.get('confirmed_across_trials'):
            score += 3.0

        scored_memories.append((score, memory))

    # Sort by score and return top N
    scored_memories.sort(key=lambda x: x[0], reverse=True)

    return [mem for score, mem in scored_memories[:limit]]


def format_memory_for_reflexion(relevant_memories: List[Dict[str, Any]]) -> str:
    """
    Format retrieved memories for inclusion in Reflexion prompt.

    Args:
        relevant_memories: Retrieved causal insights

    Returns:
        Formatted string for prompt inclusion
    """
    if not relevant_memories:
        return ""

    output = "\n" + "="*70 + "\n"
    output += "LEARNED CAUSAL PATTERNS (From Previous Trials):\n"
    output += "="*70 + "\n"

    for i, mem in enumerate(relevant_memories, 1):
        confidence = mem.get('confidence', 0)
        evidence = mem.get('evidence_count', 1)

        output += f"\n[Pattern {i}] "
        if mem.get('confirmed_across_trials'):
            output += f"✓ CONFIRMED (seen in {evidence} trial(s), confidence: {confidence:.0%})\n"
        else:
            output += f"(1 trial, confidence: {confidence:.0%})\n"

        output += mem['insight']
        output += "\n"

    output += "="*70 + "\n"
    output += "Use these patterns to avoid repeating mistakes and replicate successes!\n"
    output += "="*70 + "\n\n"

    return output
