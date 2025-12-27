# Dynamic model loading based on MODEL_PROVIDER env var
import os as _os
if _os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
    from shared_model_gemini import model
else:
    from shared_model import model
"""Universal reflection generation with TextGrad integration"""

from typing import List, Dict, Any, Tuple
# Conditional import - use vllm if available, otherwise use shared_model compatibility layer
try:
    from vllm import LLM, SamplingParams
except ImportError:
    from shared_model import SamplingParams
    LLM = None
import re
import os
import time
from collections import defaultdict
import numpy as np
import json

# Import ablation flags - will be set by alfworld_trial.py
import alfworld_trial

# PURE LLM-BASED SEMANTIC SIMILARITY (Zero hardcoding, zero bias)
def llm_evaluate_memory_relevance(
    current_task: str,
    current_observation: str,
    candidate_memories: List[Dict],
    llm_model: LLM,
    top_k: int = 5
) -> List[int]:
    """
    Use LLM to evaluate which memories are genuinely helpful.
    ZERO hardcoding, ZERO examples, PURE semantic reasoning.

    Returns: List of memory indices sorted by relevance
    """

    if not candidate_memories:
        return []

    # Build prompt that focuses on UTILITY, not similarity
    prompt = f"""You are helping an AI agent decide which past experiences are most useful for solving a new task.

CURRENT TASK:
Task: {current_task}
Current situation: {current_observation[:400]}

AVAILABLE PAST EXPERIENCES:
"""

    # Add each candidate memory
    for idx, mem in enumerate(candidate_memories):
        mem_task = mem.get('task', 'Unknown')
        mem_situation = mem.get('observation', '')[:300]
        # Handle dict reflection (extract insight field)
        reflection = mem.get('reflection', '')
        mem_learning = reflection.get('insight', '')[:250] if isinstance(reflection, dict) else str(reflection)[:250]
        mem_success = "SUCCESS" if mem.get('success') else "FAILURE"

        prompt += f"""
[{idx}] {mem_success}
Past task: {mem_task}
Past situation: {mem_situation}
What was learned: {mem_learning}
"""

    prompt += f"""
Analyze each past experience and determine if it would help solve the CURRENT task.

Consider:
- Does the learning apply to the current situation?
- Are the strategies transferable despite different specifics?
- Would applying this memory improve or harm performance?

Output ONLY a JSON array of indices for the {top_k} MOST USEFUL memories, ranked by utility.
Use this format: [index1, index2, index3, ...]

If fewer than {top_k} memories are useful, return fewer indices.
If NO memories are useful, return an empty array: []

Most useful memories:"""

    # Get LLM response with low temperature for consistency
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=150,
        stop=["\n\n", "Explanation:", "Note:"]
    )

    try:
        response = llm_model.generate([prompt], sampling_params)[0].outputs[0].text.strip()

        # Handle empty response
        if not response or response == "[]":
            print("[INFO] LLM returned no relevant memories")
            return []

        # Extract JSON array (handle various formats)
        # Try to find [number, number, ...]
        json_match = re.search(r'\[[\d,\s]+\]', response)
        if json_match:
            selected_indices = json.loads(json_match.group())

            # Validate indices
            valid_indices = [i for i in selected_indices if 0 <= i < len(candidate_memories)]

            return valid_indices[:top_k]

        # Fallback: try to find individual numbers
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            selected_indices = [int(n) for n in numbers[:top_k]]
            valid_indices = [i for i in selected_indices if 0 <= i < len(candidate_memories)]
            if valid_indices:
                print(f"[INFO] Parsed {len(valid_indices)} indices from text: {valid_indices}")
                return valid_indices

        print(f"[WARNING] Could not parse LLM response (len={len(response)}): {response[:200]}")
        return []

    except Exception as e:
        print(f"[ERROR] LLM memory evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

# ADD THIS NEW MEMORY SYSTEM CLASS (right after imports)
class HierarchicalMemorySystem:
    """Three-tier memory system for better retention"""
    def __init__(self):
        self.working_memory = []  # Last 3-5 experiences
        self.consolidated_memory = {}  # Key patterns/rules
        self.episodic_archive = []  # All experiences with metadata
        self.access_counts = defaultdict(int)  # Track retrieval frequency

class AdaptiveMemoryManager:
    """Manages hierarchical memory with consolidation"""
    def __init__(self, max_working=5, max_consolidated=20):
        self.memory = HierarchicalMemorySystem()
        self.max_working = max_working
        self.max_consolidated = max_consolidated
        self.last_consolidation = time.time()
        
    def add_experience(self, reflection: str, gradients: Dict,
                       success: bool, task: str, env_id: int,
                       observation: str = ""):
        """Add new experience - PURE data, no preprocessing"""

        # Calculate importance score (universal indicators)
        importance = 0.0
        if success:
            importance += 5.0

        # Universal critical language detection
        critical_words = ['must', 'should', 'never', 'always', 'critical', 'key', 'avoid', 'try']
        # Handle both dict and string reflections
        reflection_text = reflection.get('insight', '') if isinstance(reflection, dict) else str(reflection)
        if any(word in reflection_text.lower() for word in critical_words):
            importance += 3.0

        # Shorter reflections are often more actionable
        if len(reflection_text) < 500:
            importance += 1.0

        # Add to working memory
        self.memory.working_memory.append({
            'reflection': reflection,
            'gradients': gradients,
            'importance': importance,
            'timestamp': time.time(),
            'task': task,
            'success': success,
            'env_id': env_id,
            'observation': observation
        })

        # Consolidate if working memory is full
        if len(self.memory.working_memory) > self.max_working:
            self._consolidate_memories()
    
    def _consolidate_memories(self):
        """Consolidate working memory - NO hardcoded grouping"""

        # Sort by importance
        self.memory.working_memory.sort(key=lambda x: x['importance'], reverse=True)

        # Keep top memories in working memory (high importance)
        top_working = self.memory.working_memory[:3]

        # Move others to consolidated storage
        for mem in self.memory.working_memory[3:]:
            key = f"{mem['task'][:40]}_{mem['env_id']}_{int(mem['timestamp'])}"

            self.memory.consolidated_memory[key] = {
                'insight': mem['reflection'],
                'strength': mem['importance'],
                'timestamp': mem['timestamp'],
                'last_access': time.time(),
                'task': mem['task'],
                'observation': mem.get('observation', ''),
                'success': mem.get('success', False)
            }

        # Archive all
        self.memory.episodic_archive.extend(self.memory.working_memory)

        # Reset working memory to top items
        self.memory.working_memory = top_working

        # Apply forgetting curve
        self._apply_forgetting_curve()
    
    def _extract_consolidated_insight(self, memories: List[Dict]) -> str:
        """Extract key insight from multiple memories using intelligent merging"""
        # Memories are now compressed, so we can combine more of them
        # Extract common patterns from multiple similar task attempts

        # Combine reflections (already compressed, so safe to use more)
        # Handle dict reflections - extract insight field
        combined_text = "\n\n".join([
            m['reflection'].get('insight', '') if isinstance(m['reflection'], dict) else str(m['reflection'])
            for m in memories[:5]
        ])  # Up to 5 memories

        # Extract EXACT failed actions that appear multiple times (confirmed patterns)
        failed_actions = {}  # action -> count
        for mem in memories:
            reflection = mem['reflection']
            # Look for action mentions in AVOID sections
            if 'AVOID:' in reflection:
                avoid_section = reflection.split('AVOID:')[1].split('TRY:')[0] if 'TRY:' in reflection else reflection.split('AVOID:')[1]
                # Extract actions in quotes
                import re
                action_matches = re.findall(r"['\"]([^'\"]+)['\"]", avoid_section)
                for match in action_matches:
                    if 3 < len(match) < 100:  # Reasonable action length
                        failed_actions[match] = failed_actions.get(match, 0) + 1

        # Build consolidated insight focused on patterns that appear multiple times
        confirmed_failures = [action for action, count in failed_actions.items() if count >= 2]

        if confirmed_failures:
            failed_list = ', '.join(list(confirmed_failures)[:5])  # Top 5
            insight = f"CROSS-TASK LEARNING ({len(memories)} similar tasks):\n"
            insight += f"CONSISTENTLY FAILED: {failed_list}\n"
            # Add first "TRY:" section from most recent memory
            if memories and 'TRY:' in memories[0]['reflection']:
                try_section = memories[0]['reflection'].split('TRY:')[1].split('KEY INSIGHT:')[0] if 'KEY INSIGHT:' in memories[0]['reflection'] else memories[0]['reflection'].split('TRY:')[1]
                insight += f"SUGGESTED APPROACH: {try_section[:200]}"
            return insight[:500]  # Keep it concise
        else:
            # No strong patterns yet, return most recent compressed memory
            # Handle dict reflection
            refl = memories[0]['reflection']
            refl_text = refl.get('insight', '')[:400] if isinstance(refl, dict) else str(refl)[:400]
            return f"LEARNING FROM {len(memories)} similar tasks:\n{refl_text}"
    
    def _apply_forgetting_curve(self):
        """Apply decay to old consolidated memories"""
        current_time = time.time()
        decay_rate = 0.995  # Per hour
        
        to_remove = []
        for key, memory in self.memory.consolidated_memory.items():
            time_elapsed = (current_time - memory['last_access']) / 3600  # hours
            memory['strength'] *= (decay_rate ** time_elapsed)
            
            # Remove if strength too low
            if memory['strength'] < 0.1:
                to_remove.append(key)
        
        for key in to_remove:
            del self.memory.consolidated_memory[key]
    
    def get_relevant_memories(self, task: str, env_memory: List[str],
                             k=6, current_observation: str = "", env_id: int = -1) -> List[str]:
        """
        Task-specific memory retrieval - NO cross-task contamination.
        Returns only memories from the SAME EXACT task.
        """

        memories_to_return = []

        # 1. Keep last 2 env-specific memories for continuity
        if env_memory:
            memories_to_return.extend(env_memory[-2:])

        # 2. Gather ONLY same-task candidate memories (NO cross-task transfer)
        all_candidates = []

        # From working memory - ONLY EXACT task match
        for mem in self.memory.working_memory:
            mem_task = mem.get('task', '')
            # STRICT: Only exact task match (prevent contamination)
            if mem_task.lower().strip() == task.lower().strip():
                all_candidates.append({
                    'task': mem_task,
                    'observation': mem.get('observation', ''),
                    'reflection': mem.get('reflection', ''),
                    'success': mem.get('success', False),
                    'importance': mem.get('importance', 0.0),
                    'source': 'working'
                })

        # From consolidated memory - ONLY same-task high-quality ones
        for key, mem in self.memory.consolidated_memory.items():
            mem_task = mem.get('task', '')
            # STRICT: Only exact task match + quality threshold
            if (mem_task.lower().strip() == task.lower().strip() and
                mem.get('strength', 0.0) >= 3.0):
                all_candidates.append({
                    'task': mem_task,
                    'observation': mem.get('observation', ''),
                    'reflection': mem.get('insight', ''),
                    'success': mem.get('success', False),
                    'importance': mem.get('strength', 0.0),
                    'source': 'consolidated'
                })

        # 3. SMART PRE-FILTERING: Semantic similarity (no hardcoded thresholds)
        candidate_memories = []
        if len(all_candidates) > 20:
            # Compute semantic similarity for each candidate
            from causal_memory_compression import extract_task_verb

            current_task_lower = task.lower()
            current_words = set(current_task_lower.split())
            current_verb = extract_task_verb(task)

            candidates_with_scores = []
            for mem in all_candidates:
                mem_task = mem.get('task', '').lower()
                mem_words = set(mem_task.split())

                # Compute semantic similarity score
                word_overlap = len(current_words & mem_words) / max(len(current_words), 1)

                # Extract verb from memory task
                try:
                    from causal_memory_compression import extract_task_verb
                    mem_verb = extract_task_verb(mem_task)
                    verb_match = 1.0 if mem_verb == current_verb else 0.0
                except:
                    verb_match = 0.0

                # Combined similarity: weighted combination
                # High word overlap OR verb match indicates relevance
                similarity = max(word_overlap, verb_match)

                # Also consider importance (high importance memories always pass)
                importance_bonus = 0.3 if mem.get('importance', 0.0) >= 7.0 else 0.0
                final_score = similarity + importance_bonus

                candidates_with_scores.append((mem, final_score))

            # Sort by similarity score
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top 20 OR all with score > 0.3 (whichever is smaller)
            # This ensures we keep all semantically relevant memories
            threshold = 0.3
            candidate_memories = []
            for mem, score in candidates_with_scores:
                if score >= threshold or len(candidate_memories) < 10:
                    candidate_memories.append(mem)
                if len(candidate_memories) >= 20:
                    break

            # Log pre-filtering stats
            print(f"[PRE-FILTER] ENV {env_id}: Reduced {len(all_candidates)} → {len(candidate_memories)} candidates for '{task[:40]}...'")
        else:
            # Small candidate pool, use all
            candidate_memories = all_candidates

        # 4. Use LLM to select best memories
        if candidate_memories and current_observation:
            selected_indices = llm_evaluate_memory_relevance(
                current_task=task,
                current_observation=current_observation,
                candidate_memories=candidate_memories,
                llm_model=model,
                top_k=k-1  # k-1 because we already added 1 env memory
            )

            # Add selected memories WITH intelligent contamination tracking
            for idx in selected_indices:
                selected_mem = candidate_memories[idx]
                selected_task = selected_mem.get('task', '')

                # SMART CONTAMINATION DETECTION: Only warn on TRUE contamination
                # (not on beneficial semantic transfer)
                if selected_task and task and env_id >= 0:
                    try:
                        from causal_memory_compression import extract_task_verb
                        current_verb = extract_task_verb(task)
                        selected_verb = extract_task_verb(selected_task)

                        # Calculate semantic similarity
                        current_words = set(task.lower().split())
                        selected_words = set(selected_task.lower().split())
                        word_overlap = len(current_words & selected_words) / max(len(current_words), 1)

                        # Semantic synonyms for verbs (same action type)
                        verb_synonyms = {
                            'cool': ['freeze', 'chill', 'refrigerate'],
                            'heat': ['warm', 'cook', 'microwave'],
                            'clean': ['wash', 'rinse', 'scrub'],
                            'put': ['place', 'move', 'set'],
                        }

                        # Check if verbs are semantically similar
                        verbs_similar = False
                        if current_verb == selected_verb:
                            verbs_similar = True
                        else:
                            # Check synonyms
                            for base_verb, synonyms in verb_synonyms.items():
                                if current_verb in [base_verb] + synonyms and selected_verb in [base_verb] + synonyms:
                                    verbs_similar = True
                                    break

                        # Only flag as contamination if:
                        # 1. Verbs are NOT similar (semantically different actions)
                        # 2. AND word overlap is low (different context)
                        # 3. AND importance is not very high (not a critical universal lesson)
                        is_high_importance = selected_mem.get('importance', 0) >= 8.0

                        if not verbs_similar and word_overlap < 0.3 and not is_high_importance:
                            # TRUE contamination - semantically unrelated memory retrieved
                            print(f"[⚠️  CONTAMINATION] ENV {env_id}: Retrieved '{selected_task[:50]}' for '{task[:50]}'")
                            print(f"   Reason: Different action types ('{current_verb}' vs '{selected_verb}'), low overlap ({word_overlap:.2f})")
                        elif not verbs_similar and word_overlap >= 0.3:
                            # Beneficial transfer - different action but shared context
                            print(f"[✓ TRANSFER] ENV {env_id}: Using '{selected_task[:50]}' for '{task[:50]}'")
                            print(f"   Reason: Shared context (overlap={word_overlap:.2f}), transferable strategy")
                        elif verbs_similar and current_verb != selected_verb:
                            # Synonym transfer - semantically similar actions
                            print(f"[✓ SEMANTIC] ENV {env_id}: Using '{selected_task[:50]}' for '{task[:50]}'")
                            print(f"   Reason: Similar actions ('{current_verb}' ≈ '{selected_verb}'), semantically equivalent")
                    except Exception as e:
                        pass  # Don't fail on logging errors

                memories_to_return.append(selected_mem['reflection'])

                # Update access time for consolidated memories
                if selected_mem['source'] == 'consolidated':
                    for key, mem in self.memory.consolidated_memory.items():
                        if mem.get('insight') == selected_mem['reflection']:
                            mem['last_access'] = time.time()
                            break

        # 5. Deduplicate (handle both dict and string memories)
        seen = set()
        unique_memories = []
        for mem in memories_to_return:
            # Handle dict memories - use insight field as key for deduplication
            if isinstance(mem, dict):
                mem_key = mem.get('insight', str(mem))[:100]  # Use first 100 chars as key
                if mem and mem_key not in seen:
                    seen.add(mem_key)
                    unique_memories.append(mem)
            else:
                # String memory (legacy)
                if mem and mem.strip() and mem not in seen:
                    seen.add(mem)
                    unique_memories.append(mem)

        return unique_memories[:k]

# CREATE GLOBAL INSTANCE (add after the class definition)
global_memory_manager = AdaptiveMemoryManager()



def clean_deepseek_response(text: str) -> str:
    """Clean response - works for both DeepSeek and OpenAI"""
    import re
    text = text.strip()
    text = re.sub(r'\n\n+', '\n\n', text).strip()
    return text


# Moved imports inside functions to avoid circular dependency
prompt_generator = None
env_understanding = None

def _get_prompt_generator():
    global prompt_generator
    if prompt_generator is None:
        from alfworld_trial import prompt_generator as pg
        prompt_generator = pg
    return prompt_generator

def _get_env_understanding():
    global env_understanding
    if env_understanding is None:
        from alfworld_trial import env_understanding as eu
        env_understanding = eu
    return env_understanding

# Import debug function
try:
    from alfworld_trial import print_debug, DEBUG_REFLEXION
except ImportError:
    DEBUG_REFLEXION = True
    def print_debug(category: str, content: str, color: str = "blue"):
        """Fallback debug print"""
        print(f"\n[{category}]\n{content}\n")

def _parse_trajectory(trajectory_str: str) -> List[Tuple[str, str, str]]:
    """Parse trajectory into (action, observation, reasoning) triples"""
    interactions = []
    lines = trajectory_str.split('\n')
    
    current_action = None
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            current_action = line[1:].strip()
        elif current_action and line:
            # Always create 3-tuple with empty reasoning if not available
            interactions.append((current_action, line, ""))
            
            # Simple learning from interaction (no patterns)
            _get_prompt_generator()._learn_from_interaction(current_action, line)
            
            current_action = None
    
    return interactions

def parse_reflection_to_structured_insights(reflection: str) -> Dict[str, List[str]]:
    """Parse reflection into structured, actionable insights - NO PATTERNS"""
    insights = {
        'must_do': [],
        'must_avoid': [],
        'exact_failed_actions': [],  # EXACT actions that failed
        'hypotheses': [],
        'causal_insights': []
    }
    
    lines = reflection.split('\n')
    for line in lines:
        line_lower = line.lower().strip()
        line_clean = line.strip()
        
        # Must do actions
        if 'must' in line_lower and ('try' in line_lower or 'do' in line_lower or 'first' in line_lower):
            insights['must_do'].append(line_clean)
        
        # Must avoid actions  
        elif 'avoid' in line_lower or 'never' in line_lower or "don't" in line_lower:
            insights['must_avoid'].append(line_clean)
            # Try to extract EXACT action from avoid statements
            import re
            action_matches = re.findall(r"'([^']+)'", line_clean)
            for match in action_matches:
                if len(match) > 3 and len(match) < 100:
                    insights['exact_failed_actions'].append(match)
        
        # Failed actions
        elif 'failed' in line_lower or "didn't work" in line_lower:
            # Extract exact actions mentioned
            import re
            action_matches = re.findall(r"'([^']+)'", line_clean)
            for match in action_matches:
                if len(match) > 3 and len(match) < 100:
                    insights['exact_failed_actions'].append(match)
        
        # Hypotheses
        elif 'hypothesis' in line_lower or 'might' in line_lower or 'could' in line_lower:
            insights['hypotheses'].append(line_clean)
        
        # Causal insights
        elif 'because' in line_lower or 'caused' in line_lower or 'led to' in line_lower:
            insights['causal_insights'].append(line_clean)
    
    # Remove duplicates from exact_failed_actions
    insights['exact_failed_actions'] = list(set(insights['exact_failed_actions']))
    
    return insights

def _extract_task_from_log(log_str: str) -> str:
    """Extract task/goal from log - universal approach"""
    lines = log_str.split('\n')
    
    goal_indicators = ['task:', 'goal:', 'objective:', 'mission:', 
                      'your', 'complete', 'achieve', 'instruction:']
    
    for line in lines:
        line_lower = line.lower()
        for indicator in goal_indicators:
            if indicator in line_lower:
                if ':' in line:
                    return line.split(':', 1)[1].strip()
                else:
                    return line.strip()
    
    return "Unknown task - discovered through interaction"


def _evaluate_trajectory(trajectory: List[Tuple[str, str, str]], task: str) -> Dict[str, Any]:
    """Evaluate trajectory quality for Reflexion"""
    
    # Basic evaluation metrics
    evaluation = {
        'score': 0,
        'issues': [],
        'state_changes': 0,
        'cycles': 0,
        'efficiency': 0.0,
        'exact_repeated_actions': []
    }
    
    if not trajectory:
        evaluation['issues'].append('No actions taken')
        return evaluation
    
    # Count state changes and track EXACT action repetitions
    prev_obs = ""
    action_counts = {}
    unique_observations = set()
    
    for i, (action, obs, reasoning) in enumerate(trajectory):  # Now unpacking 3 elements
        # Count EXACT action occurrences
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # Check for state changes
        if obs != prev_obs and i > 0:
            evaluation['state_changes'] += 1
        prev_obs = obs
        
        # Track unique observations
        unique_observations.add(obs)
    
    # Find EXACT actions repeated multiple times
    for action, count in action_counts.items():
        if count >= 3:
            evaluation['exact_repeated_actions'].append(f"{action} ({count}x)")
    
    # Detect cycles (same EXACT action-observation pairs)
    action_obs_pairs = [(a, o[:50]) for a, o, r in trajectory]  # Unpack 3 elements, ignore reasoning
    unique_pairs = len(set(action_obs_pairs))
    if unique_pairs < len(trajectory) * 0.8:
        evaluation['cycles'] = len(trajectory) - unique_pairs
        evaluation['issues'].append('Repetitive cycles detected')
    
    # Calculate efficiency
    evaluation['efficiency'] = unique_pairs / len(trajectory) if trajectory else 0
    
    # Score based on progress
    if evaluation['state_changes'] == 0:
        evaluation['score'] = 2
        evaluation['issues'].append('No state changes detected')
    elif evaluation['state_changes'] < len(trajectory) * 0.3:
        evaluation['score'] = 4
        evaluation['issues'].append('Limited action diversity')
    else:
        evaluation['score'] = 6
    
    # Penalty for repeating EXACT actions
    if evaluation['exact_repeated_actions']:
        evaluation['score'] -= 2
        evaluation['issues'].append(f'Repeated EXACT actions: {len(evaluation["exact_repeated_actions"])}')
    
    # Bonus for task-relevant observations
    if task:
        task_keywords = task.lower().split()
        obs_text = ' '.join(obs.lower() for _, obs, _ in trajectory[-5:])
        if any(keyword in obs_text for keyword in task_keywords):
            evaluation['score'] += 2
    
    return evaluation


def _generate_reflection_with_gradients(log_str: str, memory: List[str], env_config: Dict[str, Any] = None) -> Tuple[str, Dict[str, str]]:
    """Generate reflection using Reflexion framework with TextGrad enhancement - NO PATTERNS"""
    
    # Parse trajectory for Reflexion
    trajectory = _parse_trajectory(log_str)
    
    # Extract task - try multiple sources
    task = None
    if env_config and 'task' in env_config:
        task = env_config['task']
    if not task:
        task = _extract_task_from_log(log_str)
    if not task:
        task = "Complete the given task by selecting appropriate actions"
    
    # Reflexion Step 1: Evaluator scores the trajectory
    evaluation = _evaluate_trajectory(trajectory, task)
    
    if DEBUG_REFLEXION:
        print_debug("REFLEXION EVALUATOR", 
                   f"Task: {task}\n"
                   f"Trajectory length: {len(trajectory)}\n"
                   f"Evaluation score: {evaluation['score']}/10\n"
                   f"Key issues: {', '.join(evaluation['issues'])}\n"
                   f"EXACT repeated actions: {evaluation['exact_repeated_actions']}",
                   "cyan")
    
    # Reflexion Step 2: Self-Reflection generation
    reflection_prompt = f"""You are a self-reflection module in a Reflexion agent.

TASK: {task}

TRAJECTORY EVALUATION:
- Performance Score: {evaluation['score']}/10
- Identified Issues: {', '.join(evaluation['issues'])}
- State Changes Detected: {evaluation['state_changes']}
- Cycles Detected: {evaluation['cycles']}
- EXACT Actions Repeated: {', '.join(evaluation['exact_repeated_actions']) if evaluation['exact_repeated_actions'] else 'None'}

RECENT TRAJECTORY (last 15 steps):
"""
   
    for action, obs, reasoning in trajectory[-25:]:  # Unpack 3 elements
        reflection_prompt += f"\n> {action}"
        if reasoning:  # Include reasoning if available
            reflection_prompt += f" [Reasoning: {reasoning[:50]}]"
        reflection_prompt += f"\n{obs}..."
    
    # Add memory context for Reflexion
    if memory:
        reflection_prompt += "\n\nPREVIOUS REFLECTIONS (memory buffer):"
        # Filter for episodic reflections only (not step reflections)
        episodic_mems = [m for m in memory if 'BEHAVIORAL ADAPTATIONS' in m or 'MISTAKE DIAGNOSIS' in m]
        for i, mem in enumerate(episodic_mems[-2:]):
            # Extract key points from previous reflections
            key_points = []
            for line in mem.split('\n'):
                if any(marker in line for marker in ['MISTAKE:', 'INSIGHT:', 'HYPOTHESIS:', '- ']):
                    key_points.append(line.strip())
            if key_points:
                reflection_prompt += f"\n\nReflection {i+1} key points:\n" + '\n'.join(key_points[:10])

    # CRITICAL FIX: Differential framing based on success/failure status
    # This prevents memory contamination where successes get framed as failures
    is_success = env_config.get('is_success', False) if env_config else False

    if is_success:
        # SUCCESS FRAMING: Ask what worked (positive reinforcement)
        reflection_prompt += """

Generate a self-reflection on this SUCCESSFUL episode using these REQUIRED keywords:
- Use "worked" or "succeeded" when describing what went right
- Use "because" for explanations of success
- Use "key" or "critical" for important factors
- Use "should" for recommendations
- Use "maintain" or "repeat" for future strategies

Format your response as:

1. WHAT WORKED: Which actions succeeded and led to task completion?
2. WHY IT WORKED: Explain because of what reason these actions succeeded
3. KEY SUCCESS FACTORS: What were the critical elements that led to success?
4. WHAT TO MAINTAIN: What should be repeated in similar situations?
5. TRANSFERABLE STRATEGY: What strategy can be applied to other tasks?

Be specific about EXACT action strings when mentioning them.
Focus on the successful workflow and what made it effective.

Make the reflection specific, causal, and actionable for future attempts.
Keep it concise (under 500 words) and avoid thinking tags or metacommentary.
Use single quotes around EXACT action strings when mentioning them."""
    else:
        # FAILURE FRAMING: Ask what failed (learning from mistakes)
        reflection_prompt += """

Generate a self-reflection on this FAILED episode using these REQUIRED keywords:
- Use "failed" or "didn't" when describing what went wrong
- Use "must" or "need" for requirements
- Use "because" for explanations
- Use "should" for recommendations
- Use "try" or "attempt" for future strategies

Format your response as:

1. WHAT FAILED: Which actions failed or didn't work?
2. WHY IT FAILED: Explain because of what reason
3. WHAT YOU NEED: What must be done or need to happen?
4. WHAT YOU SHOULD DO: What should be the approach?
5. WHAT TO TRY: What to try or attempt next?

Be specific about EXACT action strings when mentioning them.
Keep it concise but use the keywords above.

Make the reflection specific, causal, and actionable for future attempts.
Keep it concise (under 500 words) and avoid thinking tags or metacommentary.
Use single quotes around EXACT action strings when mentioning them."""

    if DEBUG_REFLEXION:
        print_debug("REFLEXION PROMPT", reflection_prompt[:500] + "...", "yellow")

    # Generate reflection
    sampling_params = SamplingParams(
        max_tokens=3000,
        temperature=0.7,
        stop=["---", "\n\n\n"]
    )
    
    output = model.generate([reflection_prompt], sampling_params)[0]
    reflection = output.outputs[0].text.strip()
    
    # Clean the reflection
    reflection = clean_deepseek_response(reflection)
    # Remove any thinking artifacts
    reflection = reflection.replace('</think>', '').replace('<think>', '').strip()
    # Limit length
    if len(reflection) > 50000:
        reflection = reflection[:50000] + "..."
    
    if DEBUG_REFLEXION:
        print_debug("REFLEXION OUTPUT", reflection, "green")
    
    # Reflexion Step 3: Convert reflection to gradients for TextGrad
    gradients = _get_prompt_generator().compute_prompt_gradient(
        trajectory=log_str,
        success=evaluation['score'] > 6,
        task=task,
        reflection=reflection  # Pass reflection for gradient computation
    )
    
    # Don't apply gradients globally during batch processing
    # This will be handled per-environment instead
    pass  # Remove the global update
    
    # Create enhanced reflection combining Reflexion + TextGrad
    enhanced_reflection = f"""{reflection}

BEHAVIORAL ADAPTATIONS (via TextGrad optimization):
"""
    
    # Add specific behavioral changes from gradients
    for component, gradient in gradients.items():
        # Keep FULL gradient text - no stripping or truncation
        if component in ['structured_insights', 'raw_reflection']:
            continue  # Skip meta keys
        enhanced_reflection += f"\n- {component}: {gradient}"  # Full gradient, no cleaning
    
    # Add quantitative metrics
    enhanced_reflection += f"\n\nQUANTITATIVE ANALYSIS:"
    enhanced_reflection += f"\n- Unique EXACT actions tried: {len(set(a for a, _, _ in trajectory if a))}"
    enhanced_reflection += f"\n- Exploration efficiency: {evaluation['efficiency']:.2f}"
    
    # Add EXACT failed actions (NO PATTERNS)
    failed_actions = set()
    for action, obs, reasoning in trajectory:  # Unpack 3 elements
        if _get_env_understanding()._is_likely_failure(obs):
            failed_actions.add(action)  # EXACT action string
    
    if failed_actions:
        enhanced_reflection += "\n\nEXACT ACTIONS THAT FAILED:"
        for action in list(failed_actions)[:5]:
            enhanced_reflection += f"\n- '{action}'"
    
    # Add after the "EXACT Actions Repeated:" section
    reflection_prompt += f"""

    CRITICAL LOOP ANALYSIS:
    - Are you stuck repeating the same action? {', '.join(evaluation['exact_repeated_actions']) if evaluation['exact_repeated_actions'] else 'No'}
    - If yes, this is a CRITICAL FAILURE requiring immediate strategy change
    - The reflection MUST emphasize avoiding these exact repeated actions
    """

    # Parse reflection into structured insights
    structured_insights = parse_reflection_to_structured_insights(enhanced_reflection)

    # Store structured insights in gradient
    gradients['structured_insights'] = structured_insights

    # Add to prompt generator's discovered knowledge (EXACT actions only)
    if structured_insights['must_do']:
        env_prompt_generators[state['env_id']].discovered_knowledge['action_priorities'] = structured_insights['must_do']
    if structured_insights['exact_failed_actions']:
        # Store as conditional failures instead of blacklist
        if 'conditional_failures' not in env_prompt_generators[state['env_id']].discovered_knowledge:
            env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'] = []
        
        for failed_action in structured_insights['exact_failed_actions']:
            env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'].append({
                'action': failed_action,
                'confidence': 0.8,
                'timestamp': state['interaction_count'],
                'is_universal': False
            })

    return enhanced_reflection, gradients

def _generate_enhanced_reflection_prompt(log_str: str, success: bool, task: str, analysis: Dict[str, Any]) -> str:
    """Generate enhanced reflection prompt - FOCUS ON EXACT ACTIONS"""
    
    # Count EXACT action repetitions
    failed_commands = {}
    successful_commands = {}
    
    trajectory = _parse_trajectory(log_str)
    for action, obs, reasoning in trajectory:  # Unpack 3 elements
        if _get_env_understanding()._is_likely_failure(obs):
            # Store EXACT action
            failed_commands[action] = failed_commands.get(action, 0) + 1
        else:
            successful_commands[action] = successful_commands.get(action, 0) + 1
    
    # Calculate exploration metrics
    total_actions = len(trajectory)
    unique_actions = len(set(action for action, _, _ in trajectory))
    coverage_percent = (unique_actions / total_actions * 100) if total_actions > 0 else 0
    exploration_status = "POOR" if coverage_percent < 30 else "MODERATE" if coverage_percent < 60 else "GOOD"
    
# Handle both success and failure
    if success:
        prompt = f"""Analyze this SUCCESSFUL completion of the task: "{task}"

SUCCESSFUL TRAJECTORY - Learn what worked:
"""
    else:
        prompt = f"""Analyze this {'successful' if success else 'failed'} attempt at completing the task: "{task}"


Trajectory Statistics:
- Total actions: {total_actions}
- Unique EXACT actions tried: {unique_actions}
- Action diversity: {coverage_percent:.1f}% ({exploration_status})
- Stuck in loop: {'YES' if any(count >= 3 for count in failed_commands.values()) else 'NO'}

EXACT Actions That Failed Multiple Times:
"""
    
    # List EXACT actions that failed repeatedly
    for action, count in sorted(failed_commands.items(), key=lambda x: x[1], reverse=True):
        if count >= 2:
            prompt += f"  - '{action}': failed {count} times\n"
    
    if not any(count >= 2 for count in failed_commands.values()):
        prompt += "  None (no action failed more than once)\n"
    
    prompt += f"""
EXACT Actions That Succeeded:
"""
    
    # List successful EXACT actions
    for action, count in sorted(successful_commands.items(), key=lambda x: x[1], reverse=True)[:5]:
        prompt += f"  - '{action}': succeeded {count} times\n"
    
    if not successful_commands:
        prompt += "  None (no successful actions)\n"
    
    prompt += f"""

Provide a structured reflection with these specific sections:

1. EXACT ACTION ANALYSIS:
   - Which EXACT actions (use single quotes) should NEVER be tried again?
   - Which EXACT actions showed promise but need different timing?
   - Are there variations of successful actions to try?

2. FAILURE DIAGNOSIS:
   - Why did the EXACT actions '{list(failed_commands.keys())[0] if failed_commands else 'N/A'}' fail?
   - Was it wrong action, wrong timing, or wrong state?
   - What prerequisites were missing?

3. TASK DECOMPOSITION:
   - What sub-tasks are needed for "{task}"?
   - What objects/locations are mentioned?
   - What's the logical sequence of actions?

4. HYPOTHESIS FOR NEXT ATTEMPT:
   - Form 2-3 specific hypotheses about EXACT action sequences
   - What specific state changes would enable progress?
   - Which EXACT actions to try first next time?

5. ACTIONABLE STRATEGY:
   - List  EXACT actions to try immediately next attempt
   - List  EXACT actions to NEVER try again
   - What order to try new actions?

Focus on EXACT action strings. Use single quotes when mentioning specific actions.
"""

    return prompt


def _extract_llm_causal_why(trajectory: List[Tuple[str, str]], task: str, success: bool) -> str:
    """
    Use LLM to extract brief causal WHY explanation (for hybrid compression).

    Args:
        trajectory: List of (action, observation) tuples
        task: Task description
        success: Whether episode succeeded

    Returns:
        Brief 1-sentence explanation of WHY success/failure occurred
    """
    # Keep only last 5 steps to minimize tokens
    recent_steps = trajectory[-5:] if len(trajectory) > 5 else trajectory

    # Trajectory can be either (action, obs) or (action, obs, reasoning)
    trajectory_text = ""
    for i, step in enumerate(recent_steps):
        if len(step) == 2:
            act, obs = step
        elif len(step) == 3:
            act, obs, _ = step
        else:
            continue  # Skip malformed steps
        trajectory_text += f"{i+1}. {act} → {obs[:80]}\n"
    trajectory_text = trajectory_text.strip()

    prompt = f"""Analyze this {'successful' if success else 'failed'} episode for task: "{task}"

Last steps:
{trajectory_text}

In ONE sentence (max 100 chars), explain the ROOT CAUSE of {('success' if success else 'failure')}:
Why did it {'work' if success else 'fail'}? What was the key semantic/logical reason?

Output ONLY the explanation, nothing else:"""

    try:
        sampling_params = SamplingParams(
            max_tokens=50,  # Force brevity
            temperature=0.3,
            stop=["\n", "."]
        )

        output = model.generate([prompt], sampling_params)[0]
        why = output.outputs[0].text.strip()

        # Clean and validate
        why = why.replace('</think>', '').replace('<think>', '').strip()
        if len(why) > 200:
            why = why[:200]

        return why if why else None

    except Exception as e:
        print(f"[WARNING] LLM WHY extraction failed: {e}")
        return None


def _compress_memory_with_causal_patterns(
    trajectory: List[Tuple[str, str, str]],  # 3-tuple: (action, observation, reasoning)
    task: str,
    success: bool,
    reflection: str = None
) -> str:
    """
    Hybrid compression: LLM provides WHY, causal patterns provide structure.
    Replaces old _compress_memory_intelligently() with causal approach.

    Args:
        trajectory: Episode trajectory (action, observation pairs)
        task: Task description
        success: Whether episode succeeded
        reflection: Optional verbose reflection (for backward compatibility)

    Returns:
        Compressed causal insight (~200 chars vs old ~2000 chars)
    """
    # Import causal compression functions
    try:
        from causal_memory_compression import (
            compress_failure_pattern,
            compress_success_pattern,
            extract_verb,
            extract_effect
        )
    except ImportError:
        print("[WARNING] Causal compression unavailable, falling back to truncation")
        return reflection[:500] if reflection else "No memory"

    # Extract brief LLM WHY (semantic understanding)
    llm_why = _extract_llm_causal_why(trajectory, task, success)

    if success:
        # Compress success trajectory
        # Trajectory already in correct format: (action, obs, reasoning)
        compressed = compress_success_pattern(
            trajectory=trajectory,
            task=task,
            llm_success_insight=llm_why
        )
    else:
        # Compress failures - detect all non-progressing attempts
        # Universal approach: Track repeated actions and lack of state change
        failed_attempts = []
        prev_obs = None
        action_history = []

        for action, obs, _ in trajectory:  # Trajectory always 3-tuple format
            action_history.append(action)

            # Detect failure through multiple signals:
            # 1. No state change (observation unchanged)
            # 2. Action repetition (same action type tried multiple times)
            # 3. Task still incomplete at trajectory end

            is_non_progressing = False

            # Check for repeated action pattern (universal signal)
            if len(action_history) >= 3:
                # Extract action verb (first word) for comparison
                current_verb = action.split()[0] if action else ''
                recent_verbs = [a.split()[0] if a else '' for a in action_history[-3:]]
                if recent_verbs.count(current_verb) >= 2:
                    is_non_progressing = True

            # Check for observation stagnation (universal signal)
            if prev_obs and obs == prev_obs:
                is_non_progressing = True

            if is_non_progressing:
                failed_attempts.append((action, obs))

            prev_obs = obs

        # ALWAYS use detailed compression if we have trajectory
        # The compress_failure_pattern function will extract patterns
        if trajectory:
            compressed = compress_failure_pattern(
                failed_attempts=failed_attempts if failed_attempts else [(a, o) for a, o, _ in trajectory[-5:]],
                task=task,
                llm_causal_insight=llm_why
            )

    # REMOVED FALLBACK: Always return a dict, never None
    if not compressed:
        raise ValueError(f"Compression failed for task '{task}' with success={success}. This should never happen - check compression logic!")

    return compressed


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]],
                 prompt_gen=None, env_under=None) -> List[Dict[str, Any]]:
    """Updates env_configs with reflections using BATCHED generation"""
    
    # LOG MEMORY UPDATE START
    if trial_log_path and os.path.exists(trial_log_path):
        with open(trial_log_path, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("[MEMORY UPDATE] Starting reflection generation\n")
            f.write(f"  Failed environments: {sum(1 for env in env_configs if not env.get('is_success', False))}\n")
            f.write("="*60 + "\n")

    print("\n" + "="*80)
    print("[REFLECTION GENERATION DEBUG]")
    print(f"  Trial log path: {trial_log_path}")
    print(f"  Number of environments: {len(env_configs)}")
    
    # Debug each environment's status
    for i, env in enumerate(env_configs):
        print(f"  Env {i}: success={env.get('is_success', 'MISSING')}, "
              f"skip={env.get('skip', False)}, "
              f"memory_size={len(env.get('memory', []))}")
    
    failed_count = sum(1 for env in env_configs if not env.get('is_success', False) and not env.get('skip', False))
    success_count = sum(1 for env in env_configs if env.get('is_success', False) and not env.get('skip', False))
    print(f"  Failed environments: {failed_count}")
    print(f"  Successful environments: {success_count}")
    print("="*80 + "\n")

        
    # CRITICAL FIX: Import and use the actual global instances
    global prompt_generator, env_understanding
    import alfworld_trial
    prompt_generator = alfworld_trial.prompt_generator
    env_understanding = alfworld_trial.env_understanding
    
    # ADD DEBUG OUTPUT
    print(f"\n{'='*80}")
    print(f"[MEMORY UPDATE] Starting reflection generation for trial")
    print(f"  Failed environments to process: {sum(1 for env in env_configs if not env['is_success'])}")
    print(f"  Total environments: {len(env_configs)}")
    print(f"{'='*80}\n")
    
    if not prompt_generator or not env_understanding:
        raise RuntimeError("CRITICAL: Components not initialized!")
    
    print(f"[MEMORY UPDATE] Using prompt_generator with {len(prompt_generator.prompt_components)} components")
    
    # Use global memory manager
    global global_memory_manager
    
    print("DEBUG MEMORY UPDATE:")
    for i, env in enumerate(env_configs):
        print(f"\nEnv {i}:")
        print(f"  Success: {env.get('is_success', False)}")
        print(f"  Current memory size: {len(env.get('memory', []))}")
        print(f"  Has task? {'task' in env}")
        if 'task' in env:
            print(f"  Task: {env['task']}")
    print(f"{'='*60}\n")
    
    # THIS WAS NOT INDENTED - NOW FIXED
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
    
    # Robust environment log parsing using STATUS markers
    def parse_environment_logs(full_log: str, expected_count: int) -> List[str]:
        """
        Parse individual environment logs from the combined trial log.
        Uses STATUS markers for accurate splitting.
        """
        env_logs = []
        current_env_log = []
        current_env_num = None
        
        lines = full_log.split('\n')
        
        for i, line in enumerate(lines):
            # Detect environment start
            if line.startswith('Environment #'):
                # Extract environment number
                import re
                match = re.match(r'Environment #(\d+):', line)
                if match:
                    # Save previous environment if exists
                    if current_env_log and current_env_num is not None:
                        # Join and save
                        env_logs.append({
                            'number': current_env_num,
                            'content': '\n'.join(current_env_log)
                        })
                    
                    # Start new environment
                    current_env_num = int(match.group(1))
                    current_env_log = [line]
            
            # Add lines to current environment
            elif current_env_num is not None:
                current_env_log.append(line)
                
                # Check if this is the end marker
                if line.strip() in ['STATUS: OK', 'STATUS: FAIL']:
                    # Save this environment
                    env_logs.append({
                        'number': current_env_num,
                        'content': '\n'.join(current_env_log)
                    })
                    # Reset for next environment
                    current_env_log = []
                    current_env_num = None
        
        # Handle case where last environment didn't have STATUS marker
        if current_env_log and current_env_num is not None:
            env_logs.append({
                'number': current_env_num,
                'content': '\n'.join(current_env_log)
            })
        
        # Sort by environment number to ensure correct order
        env_logs.sort(key=lambda x: x['number'])
        
        # Extract just the content in order
        ordered_logs = []
        for i in range(expected_count):
            # Find log for environment i
            matching_log = None
            for log_entry in env_logs:
                if log_entry['number'] == i:
                    matching_log = log_entry['content']
                    break
            
            if matching_log:
                ordered_logs.append(matching_log)
            else:
                # Environment missing from log - use empty
                print(f"[WARNING] Environment {i} not found in log, using empty log")
                ordered_logs.append(f"Environment #{i}:\nNo actions recorded\nSTATUS: FAIL")
        
        return ordered_logs
    
    # Use the robust parser
    env_logs = parse_environment_logs(full_log, len(env_configs))
    
    # Validate parsing
    if len(env_logs) != len(env_configs):
        print(f"[ERROR] Parsing mismatch: got {len(env_logs)} logs but expected {len(env_configs)}")
        print(f"[INFO] First few lines of full_log: {full_log[:500]}")
        
        # Emergency fallback - don't crash, just skip reflection generation
        print("[WARNING] Skipping reflection generation due to log parsing failure")
        return env_configs
    
    # Additional validation - check each log has content
    for i, log in enumerate(env_logs):
        if not log or len(log) < 10:
            print(f"[WARNING] Environment {i} has empty or invalid log")
            env_logs[i] = f"Environment #{i}:\nNo valid actions recorded\nSTATUS: FAIL"
    
    if DEBUG_REFLEXION:
        print_debug("MEMORY UPDATE", 
                f"Processing {len(env_logs)} environment logs\n"
                f"Failed environments: {sum(1 for env in env_configs if not env['is_success'] and not env.get('skip', False))}",
                "blue")
    
    # Collect all failed environments for batch processing
    failed_indices = []
    failed_logs = []
    failed_memories = []
    failed_configs = []
    
    for i, env in enumerate(env_configs):
        if not env.get('skip', False):  # Process ALL environments for learning
            failed_indices.append(i)
            failed_logs.append(env_logs[i])
            failed_memories.append(env.get('memory', []))
            failed_configs.append(env)
    
    if not failed_indices:
        print("No failed environments to process")
        return env_configs
    
    print(f"Processing {len(failed_indices)} failed environments in batch...")
    
    # BATCH GENERATE REFLECTIONS
    reflection_prompts = []
    
    for log_str, memory, env_config in zip(failed_logs, failed_memories, failed_configs):
        # Parse trajectory for Reflexion
        trajectory = _parse_trajectory(log_str)
        
        # Extract task
        task = env_config.get('task', None)
        if not task:
            task = _extract_task_from_log(log_str)
        if not task:
            task = "Complete the given task by selecting appropriate actions"
        
        # Evaluator scores the trajectory
        evaluation = _evaluate_trajectory(trajectory, task)
        
        # Build reflection prompt
        reflection_prompt = f"""You are a self-reflection module in a Reflexion agent.

TASK: {task}

TRAJECTORY EVALUATION:
- Performance Score: {evaluation['score']}/10
- Identified Issues: {', '.join(evaluation['issues'])}
- State Changes Detected: {evaluation['state_changes']}
- Cycles Detected: {evaluation['cycles']}
- EXACT Actions Repeated: {', '.join(evaluation['exact_repeated_actions']) if evaluation['exact_repeated_actions'] else 'None'}

RECENT TRAJECTORY (last 15 steps):
"""
        for action, obs, reasoning in trajectory[-15:]:
            reflection_prompt += f"\n> {action}"
            if reasoning:
                reflection_prompt += f" [Reasoning: {reasoning[:50]}]"
            reflection_prompt += f"\n{obs}..."
        
        # Add memory context
        if memory:
            reflection_prompt += "\n\nPREVIOUS REFLECTIONS (memory buffer):"
            # Filter for episodic reflections only (not step reflections)
            episodic_mems = [m for m in memory if 'BEHAVIORAL ADAPTATIONS' in m or 'MISTAKE DIAGNOSIS' in m]
            for i, mem in enumerate(episodic_mems[-2:]):
                # Extract key points from previous reflections
                key_points = []
                for line in mem.split('\n'):
                    if any(marker in line for marker in ['MISTAKE:', 'INSIGHT:', 'HYPOTHESIS:', '- ']):
                        key_points.append(line.strip())
                if key_points:
                    reflection_prompt += f"\n\nReflection {i+1} key points:\n" + '\n'.join(key_points[:10])

        # CRITICAL FIX: Differential framing based on success/failure status
        # This prevents memory contamination where successes get framed as failures
        is_success = env_config.get('is_success', False)

        if is_success:
            # SUCCESS FRAMING: Ask what worked (positive reinforcement)
            reflection_prompt += """

Generate a self-reflection on this SUCCESSFUL episode using these REQUIRED keywords:
- Use "worked" or "succeeded" when describing what went right
- Use "because" for explanations of success
- Use "key" or "critical" for important factors
- Use "should" for recommendations
- Use "maintain" or "repeat" for future strategies

Format your response as:

1. WHAT WORKED: Which actions succeeded and led to task completion?
2. WHY IT WORKED: Explain because of what reason these actions succeeded
3. KEY SUCCESS FACTORS: What were the critical elements that led to success?
4. WHAT TO MAINTAIN: What should be repeated in similar situations?
5. TRANSFERABLE STRATEGY: What strategy can be applied to other tasks?

Be specific about EXACT action strings when mentioning them.
Focus on the successful workflow and what made it effective.

Make the reflection specific, causal, and actionable for future attempts.
Keep it concise (under 500 words) and avoid thinking tags or metacommentary.
Use single quotes around EXACT action strings when mentioning them."""
        else:
            # FAILURE FRAMING: Ask what failed (learning from mistakes)
            reflection_prompt += """

Generate a self-reflection on this FAILED episode using these REQUIRED keywords:
- Use "failed" or "didn't" when describing what went wrong
- Use "must" or "need" for requirements
- Use "because" for explanations
- Use "should" for recommendations
- Use "try" or "attempt" for future strategies

Format your response as:

1. WHAT FAILED: Which actions failed or didn't work?
2. WHY IT FAILED: Explain because of what reason
3. WHAT YOU NEED: What must be done or need to happen?
4. WHAT YOU SHOULD DO: What should be the approach?
5. WHAT TO TRY: What to try or attempt next?

Be specific about EXACT action strings when mentioning them.
Keep it concise but use the keywords above.

Make the reflection specific, causal, and actionable for future attempts.
Keep it concise (under 500 words) and avoid thinking tags or metacommentary.
Use single quotes around EXACT action strings when mentioning them."""
        
        reflection_prompts.append(reflection_prompt)
    
    # BATCH GENERATE WITH vLLM
    sampling_params = SamplingParams(
        max_tokens=2000,
        temperature=0.7,
        stop=["---", "\n\n\n"],
        skip_special_tokens=True
    )
    
    # ABLATION: Check if Reflexion is enabled
    if not alfworld_trial.USE_REFLEXION:
        print(f"[ABLATION] Reflexion disabled - skipping reflection generation for {len(failed_indices)} failed environments")
        # Return empty reflections
        for env_idx in failed_indices:
            if 'memory' not in env_configs[env_idx]:
                env_configs[env_idx]['memory'] = []
        return env_configs

    print(f"Generating {len(reflection_prompts)} reflections in batch...")
    # Generate with quota protection
    try:
        outputs = model.generate(reflection_prompts, sampling_params)
    except Exception as e:
        error_str = str(e).lower()
        if any(word in error_str for word in ['quota', 'rate', 'limit', '429']):
            print(f"\n[API QUOTA ERROR] {e}")
            print("[INFO] State has been saved. Resume with --is_resume flag")
            # Save what we can
            for i, env in enumerate(env_configs):
                if 'memory' not in env:
                    env['memory'] = []
            return env_configs
        raise e
    
    # ADD DEBUG OUTPUT
    if DEBUG_REFLEXION:
        for i, output in enumerate(outputs):
            print(f"[REFLECTION {i}] Generated: {output.outputs[0].text[:200]}...")
    
    # Define log_debug function locally if needed
    def log_debug(msg):
        print(msg)
    
    # Process outputs and generate gradients
    trial_gradients = []
    
    for idx, (env_idx, output) in enumerate(zip(failed_indices, outputs)):
        reflection = clean_deepseek_response(output.outputs[0].text.strip())
        reflection = reflection.replace('</think>', '').replace('<think>', '').strip()

        # Keep full reflection for proper learning - only limit extreme cases
        if len(reflection) > 50000:  # Much higher limit
            break_point = reflection.rfind('.', 49000, 50000)
            if break_point > 0:
                reflection = reflection[:break_point + 1]
            else:
                reflection = reflection[:8000] + "..."
        
        if DEBUG_REFLEXION:
            print_debug(f"ENV {env_idx} REFLECTION", reflection + "...", "green")
        
        # Generate gradients for TextGrad (if enabled)
        task = failed_configs[idx].get('task', 'unknown')
        if alfworld_trial.USE_TEXTGRAD:
            gradients = prompt_generator.compute_prompt_gradient(
                trajectory=failed_logs[idx],
                success=env_configs[failed_indices[idx]].get('is_success', False),  # <-- ACTUAL STATUS
                task=task,
                reflection=reflection
            )
        else:
            # Ablation: TextGrad disabled - use empty gradients
            gradients = {}
        
        # Parse structured insights
        structured_insights = parse_reflection_to_structured_insights(reflection)
        
        # Create processed gradients with actionable insights
        processed_gradients = {}
        for component, gradient in gradients.items():
            if component == 'adaptive_strategy' and structured_insights['must_do']:
                processed_gradients[component] = structured_insights['must_do'][0]
            elif component == 'action_discovery' and structured_insights['must_avoid']:
                processed_gradients[component] = f"Avoid: {structured_insights['must_avoid'][0]}"
            else:
                # Clean the gradient text
                processed_gradients[component] = gradient.split('.')[0] + "." if '.' in gradient else gradient
        
        # Apply gradients (if TextGrad is enabled)
        if alfworld_trial.USE_TEXTGRAD and gradients:
            # CRITICAL FIX: Update the environment's specific generator, not global
            # The env_idx matches the failed_indices order
            actual_env_id = failed_indices[idx] if idx < len(failed_indices) else idx

            # Import from alfworld_trial to get the env generators
            try:
                from alfworld_trial import env_prompt_generators
                if actual_env_id in env_prompt_generators:
                    env_prompt_generators[actual_env_id].update_prompt_components(processed_gradients)
                    print(f"[FIX] Updated env {actual_env_id} prompt_generator")
                else:
                    # Fallback to global if env generators not available
                    prompt_generator.update_prompt_components(processed_gradients)
            except ImportError:
                prompt_generator.update_prompt_components(processed_gradients)
        elif not alfworld_trial.USE_TEXTGRAD:
            print(f"[ABLATION] TextGrad disabled - skipping gradient application for env {env_idx}")
        
        # Create enhanced reflection
        enhanced_reflection = f"""{reflection}

BEHAVIORAL ADAPTATIONS (via TextGrad optimization):
"""


        # Log the episodic reflection
        if trial_log_path:
            logging_dir = os.path.dirname(trial_log_path)
            try:
                from enhanced_logging import ComprehensiveLogger
                comprehensive_logger = ComprehensiveLogger(logging_dir)
                
                comprehensive_logger.log_episodic_reflection(
                    env_id=env_idx,
                    task=task,
                    reflection=enhanced_reflection,
                    gradients=gradients
                )
            except Exception as e:
                print(f"[WARNING] Failed to log episodic reflection: {e}")
                # Continue without failing the whole process
        
        for component, gradient in gradients.items():
            gradient_clean = gradient.strip('[]').strip()
            enhanced_reflection += f"\n- {component}: {gradient_clean}"
        
        # Add quantitative metrics
        trajectory = _parse_trajectory(failed_logs[idx])
        evaluation = _evaluate_trajectory(trajectory, task)
        enhanced_reflection += f"\n\nQUANTITATIVE ANALYSIS:"
        enhanced_reflection += f"\n- Unique EXACT actions tried: {len(set(a for a, _, _ in trajectory if a))}"
        enhanced_reflection += f"\n- State changes achieved: {evaluation['state_changes']}"
        enhanced_reflection += f"\n- Exploration efficiency: {evaluation['efficiency']:.2f}"
        
        # Add EXACT failed actions
        failed_actions = set()
        for action, obs, reasoning in trajectory:
            if _get_env_understanding()._is_likely_failure(obs):
                failed_actions.add(action)
        
        if failed_actions:
            enhanced_reflection += "\n\nEXACT ACTIONS THAT FAILED:"
            for action in list(failed_actions)[:5]:
                enhanced_reflection += f"\n- '{action}'"
        
        # Store structured insights
        gradients['structured_insights'] = structured_insights
        

        # Update discovered knowledge with EXACT actions (append, don't overwrite)
        if structured_insights['must_do']:
            if 'action_priorities' not in _get_prompt_generator().discovered_knowledge:
                _get_prompt_generator().discovered_knowledge['action_priorities'] = []
            _get_prompt_generator().discovered_knowledge['action_priorities'].extend(structured_insights['must_do'])
            # Keep only last 10 to prevent bloat
            _get_prompt_generator().discovered_knowledge['action_priorities'] = _get_prompt_generator().discovered_knowledge['action_priorities'][-10:]

        if structured_insights['exact_failed_actions']:
            if 'action_blacklist' not in _get_prompt_generator().discovered_knowledge:
                _get_prompt_generator().discovered_knowledge['action_blacklist'] = []
            _get_prompt_generator().discovered_knowledge['action_blacklist'].extend(structured_insights['exact_failed_actions'])
            # Keep only last 20 to prevent bloat
            _get_prompt_generator().discovered_knowledge['action_blacklist'] = _get_prompt_generator().discovered_knowledge['action_blacklist'][-20:]
        
        # CRITICAL FIX: Compress memory using CAUSAL PATTERNS (deduplication + LLM WHY)
        compressed_memory = _compress_memory_with_causal_patterns(
            trajectory=trajectory,  # Use parsed trajectory for deduplication
            task=task,
            success=env_configs[env_idx].get('is_success', False),
            reflection=enhanced_reflection  # Fallback if causal compression fails
        )

        # Add COMPRESSED memory to global memory manager (not verbose!)
        # This ensures cross-task transfer uses compressed, consistent format
        global_memory_manager.add_experience(
            reflection=compressed_memory,  # ← CHANGED: Use compressed instead of verbose
            gradients=gradients,
            success=env_configs[env_idx].get('is_success', False),
            task=task,
            env_id=env_idx
        )

        # Log memory update
        if trial_log_path:
            comprehensive_logger.log_memory_update(
                env_id=env_idx,
                memory_type='global_memory_added',
                content=compressed_memory  # ← CHANGED: Log compressed version
            )

        # Log compression effectiveness
        # REMOVED FALLBACK: compressed_memory must ALWAYS be a dict
        if not isinstance(compressed_memory, dict):
            raise TypeError(f"ENV {env_idx}: compressed_memory must be a dict, got {type(compressed_memory)}. This indicates a bug in _compress_memory_with_causal_patterns!")

        insight_text = compressed_memory.get('insight', '')
        compression_ratio = len(insight_text) / len(enhanced_reflection) if len(enhanced_reflection) > 0 else 0
        print(f"[MEMORY COMPRESSION] ENV {env_idx}: {len(enhanced_reflection)} → {len(insight_text)} chars ({compression_ratio*100:.1f}%)")

        # Store compressed memory for actual use
        if 'memory' not in env_configs[env_idx]:
            env_configs[env_idx]['memory'] = []
        env_configs[env_idx]['memory'].append(compressed_memory)

        # Archive full verbose memory for debugging/analysis
        if 'verbose_memory_archive' not in env_configs[env_idx]:
            env_configs[env_idx]['verbose_memory_archive'] = []
        env_configs[env_idx]['verbose_memory_archive'].append(enhanced_reflection)
        # Limit archive size to prevent bloat
        if len(env_configs[env_idx]['verbose_memory_archive']) > 10:
            env_configs[env_idx]['verbose_memory_archive'] = env_configs[env_idx]['verbose_memory_archive'][-10:]

        # Apply same intelligent cap
        if len(env_configs[env_idx]['memory']) > 25:
            # Keep 5 foundation + 20 recent
            foundation = env_configs[env_idx]['memory'][:5]
            recent = env_configs[env_idx]['memory'][-20:]
            env_configs[env_idx]['memory'] = (foundation + recent)[:25]
        
        # Limit env memory to 4
        if len(env_configs[env_idx]['memory']) > 25:
            env_configs[env_idx]['memory'] = env_configs[env_idx]['memory'][-25:]
        
        # DEBUG confirmation
        print(f"[MEMORY UPDATE] ENV {env_idx} now has {len(env_configs[env_idx]['memory'])} memory items")
        
        trial_gradients.append(gradients)
        
        if DEBUG_REFLEXION:
            print_debug(f"ENV {env_idx} COMPLETE", 
                    f"Reflection length: {len(enhanced_reflection)}\n"
                    f"Gradients generated: {len(gradients)}\n"
                    f"New memory size: {len(env_configs[env_idx]['memory'])}",
                    "green")
    
    # Apply momentum-based gradient updates
    if trial_gradients:
        merged_gradients = {}
        for gradients in trial_gradients:
            for component, gradient in gradients.items():
                if component not in merged_gradients:
                    merged_gradients[component] = []
                merged_gradients[component].append(gradient)
        
        # Update each component with best gradient
        components_updated = set()
        for component, gradient_list in merged_gradients.items():
            if gradient_list and component in prompt_generator.prompt_components and component not in components_updated:
                # Score gradients by quality
                def score_gradient(grad):
                    score = 0
                    grad_lower = grad.lower()
                    
                    if 'must' in grad_lower: score += 5
                    if 'should' in grad_lower: score += 4
                    if 'exact' in grad_lower: score += 4
                    if 'avoid' in grad_lower: score += 3
                    if 'never' in grad_lower: score += 3
                    if 'always' in grad_lower: score += 3
                    if 'failed because' in grad_lower: score += 4
                    if 'succeeded' in grad_lower: score += 3
                    
                    if any(char.isdigit() for char in grad): score += 2
                    if "'" in grad: score += 2
                    
                    length_score = min(len(grad) / 50, 3)
                    score += length_score
                    
                    # if 'continue exploring' in grad_lower: score -= 2
                    # if 'more exploration needed' in grad_lower: score -= 2
                    # if 'gather more data' in grad_lower: score -= 2
                    
                    return score
                
                # Score all gradients and pick best
                scored_gradients = [(score_gradient(g), g) for g in gradient_list]
                scored_gradients.sort(key=lambda x: x[0], reverse=True)
                best_gradient = scored_gradients[0][1]
                
                log_debug(f"[GRADIENT] Best score for {component}: {scored_gradients[0][0]:.1f}")
                prompt_generator.update_component_with_momentum(component, best_gradient)
                components_updated.add(component)
    
    # Log optimization state
    optimization_state = prompt_generator.get_prompt_optimization_state()
    
    if DEBUG_REFLEXION:
        print_debug("TEXTGRAD OPTIMIZATION STATE", 
                f"Total gradient updates: {optimization_state['num_updates']}\n"
                f"Available actions found: {optimization_state['available_actions_found']}\n"
                f"Completion actions: {optimization_state['completion_actions'][:5]}\n"
                f"Interaction count: {optimization_state['interaction_count']}",
                "purple")
    
    print(f"\n[TextGrad] Optimization State:")
    print(f"  - Total gradient updates: {optimization_state['num_updates']}")
    print(f"  - Current prompt components:")
    for component, text in optimization_state['components'].items():
        print(f"    {component}: {text[:80]}...")
    
    # Log memory system state
    print(f"\n[Memory System] State:")
    print(f"  - Working memory: {len(global_memory_manager.memory.working_memory)} items")
    print(f"  - Consolidated insights: {len(global_memory_manager.memory.consolidated_memory)} patterns")
    print(f"  - Episodic archive: {len(global_memory_manager.memory.episodic_archive)} total experiences")
    
    # SAVE PROMPT GENERATOR STATE AFTER UPDATES
    try:
        import alfworld_trial
        alfworld_trial.save_prompt_generator()
        print(f"[SAVE] Saved prompt_generator successfully")
    except Exception as e:
        print(f"[WARNING] Could not save prompt_generator: {e}")
    
    # LOG MEMORY UPDATE COMPLETE
    if trial_log_path and os.path.exists(trial_log_path):
        with open(trial_log_path, 'a') as f:
            f.write("\n[MEMORY UPDATE COMPLETE]\n")
            f.write(f"  Gradients generated: {len(trial_gradients) if 'trial_gradients' in locals() else 0}\n")
            f.write(f"  Memory items added: {sum(len(env.get('memory', [])) for env in env_configs)}\n")
            f.write("="*60 + "\n")

    return env_configs

