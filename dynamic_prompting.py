"""Universal Dynamic Prompt Generator with TextGrad Optimization"""

from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
# Dynamic model loading based on MODEL_PROVIDER env var
import os as _os
if _os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
    from shared_model_gemini import model
else:
    from shared_model import model
try:
    from vllm import SamplingParams
except ImportError:
    from shared_model import SamplingParams
import re
import json
import sys
import numpy as np

# Debug flags - define locally to avoid circular imports
DEBUG_ACTOR = False
DEBUG_CRITIC = False



def print_debug(category: str, content: str, color: str = "blue"):
    """Debug print function"""
    if DEBUG_ACTOR or DEBUG_CRITIC:
        print(f"\n[{category}]\n{content}\n")

def set_debug_flags(actor=False, critic=False):
    """Allow external setting of debug flags"""
    global DEBUG_ACTOR, DEBUG_CRITIC
    DEBUG_ACTOR = actor
    DEBUG_CRITIC = critic


def action_stats_factory():
    return {'success': 0, 'fail': 0}

def float_defaultdict_factory():
    return defaultdict(float)
    
class DynamicPromptGenerator:
    
    def __init__(self):
        """Initialize dynamic prompt generator"""
        self.task_embedding = None  # Store task representation
        self.task_signature = None  # Store task text for similarity
        self.prompt_components = {
            'environment_understanding': "observe and learn the environment's rules through interaction",
            'action_discovery': "try different action formats to discover what works",
            'pattern_recognition': "identify patterns in successes and failures",
            'hypothesis_testing': "form and test hypotheses about environment constraints",
            'adaptive_strategy': "adapt behavior based on discovered patterns",
            'task_decomposition': "analyze the task to identify required objects and locations"
        }
        
        self.discovered_knowledge = {
            'confirmed_rules': [],
            'hypotheses': [],
            'successful_strategies': [],
            'failed_approaches': [],
            'command_templates': {},
            'available_actions': set(),
            'numbered_items': set(),
            'completion_actions': set(),
            'successful_action_structures': [],
            'failed_action_structures': [],
            'max_actions_seen': 0,  # Track highest action count seen
        }
        
        self.action_outcome_history = []
        self.interaction_count = 0
        self.prompt_gradients = []
        self.action_structure_analysis = defaultdict(action_stats_factory)
        self.gradient_momentum = 0.9
        self.gradient_velocity = {}
        self._task = None
        self._last_observation = ""
        self.task_action_correlations = defaultdict(float_defaultdict_factory)
        self.successful_trajectories = []
        self.generator_pool = {}  # Pool of task-specific generators

    def format_initial_observation(self, observation: str, memory: List[str]) -> str:
        """Format initial observation for environment history"""
        prompt = f"""INITIAL STATE:
    {observation}

    TASK: {self._task if hasattr(self, '_task') and self._task else 'Unknown task'}
    """
        
        if memory:
            prompt += "\nLEARNINGS FROM PREVIOUS ATTEMPTS:\n"
            for i, mem in enumerate(memory[-5:], 1):
                # Handle dict memory - extract insight field
                if isinstance(mem, dict):
                    mem_text = mem.get('insight', str(mem))
                else:
                    mem_text = str(mem)

                # Extract key points
                key_points = []
                for line in mem_text.split('\n'):
                    if any(marker in line for marker in ['MISTAKE:', 'INSIGHT:', 'must', 'should']):
                        key_points.append(line.strip())
                if key_points:
                    prompt += f"Attempt {i}:\n" + '\n'.join(key_points[:3]) + "\n"
        
        return prompt

    def set_task(self, task: str):
        """Set the task for this episode - PERMANENT"""
        if not task:
            raise ValueError("Cannot set empty task!")
        
        self._task = task
        self._original_task = task  # Keep original for reference
        
        print(f"[TASK SET] Task for this episode: {task}")
        
        # Clear any step-level adjustments from previous episode
        if hasattr(self, '_step_level_adjustments'):
            self._step_level_adjustments = None

    def apply_universal_step_gradient(self, step_gradient: Dict[str, Any]):
        """Apply step-level gradient with semantic understanding"""
        
        if not hasattr(self, '_step_gradient_history'):
            self._step_gradient_history = []
        
        # CRITICAL FIX: Store reference to ensure updates persist
        if not hasattr(self, '_components_before_step'):
            self._components_before_step = self.prompt_components.copy()

        self._step_gradient_history.append({
            'gradient': step_gradient,
            'timestamp': self.interaction_count
        })
        
        # Keep only recent history
        if len(self._step_gradient_history) > 20:
            self._step_gradient_history.pop(0)
        
        # Update components based on LLM's understanding
        if 'prerequisites' in step_gradient:
            missing = step_gradient['prerequisites'].get('missing', [])
            if missing:
                self.prompt_components['adaptive_strategy'] = f"Achieve prerequisites: {', '.join(missing[:3])}"
        
        if 'task_progress' in step_gradient:
            remaining = step_gradient['task_progress'].get('remaining', [])
            if remaining:
                self.prompt_components['task_decomposition'] = f"Remaining: {', '.join(remaining[:3])}"
        
        if 'next_action_guidance' in step_gradient and step_gradient['next_action_guidance']:
            # CRITICAL: Direct dictionary update to ensure persistence
            self.prompt_components['action_discovery'] = step_gradient['next_action_guidance']
            
        # Force the update to stick
        self.prompt_components = dict(self.prompt_components)
        
        self._current_step_gradient = step_gradient
        

        # STRONG LOOP DETECTION SIGNAL
        if 'raw_reflection' in step_gradient:
            reflection_lower = step_gradient['raw_reflection'].lower()
            
            # Check for repetition indicators
            if 'same' in reflection_lower or 'again' in reflection_lower or 'nothing' in reflection_lower:
                # Strong signal to change strategy
                self.prompt_components['adaptive_strategy'] = "CRITICAL: Stuck in loop! Must try completely different action type."
                self.prompt_components['action_discovery'] = "Avoid examining same locations. Try new areas."
                
                # Track loop count
                if not hasattr(self, '_loop_count'):
                    self._loop_count = 0
                self._loop_count += 1
                
                if self._loop_count > 2:
                    self.prompt_components['environment_understanding'] = "Environment not responding to current approach. Need fundamental strategy change."

        if DEBUG_CRITIC:
            print(f"\n[UNIVERSAL STEP GRADIENT]")
            print(f"State: {step_gradient.get('semantic_state_before', 'None')}")
            print(f"Prerequisites missing: {step_gradient.get('prerequisites', {}).get('missing', [])}")
            print(f"Task remaining: {step_gradient.get('task_progress', {}).get('remaining', [])}")

    def _apply_meta_gradient(self, gradient_text: str):
        """Apply meta-level gradient to improve prompt components"""
        # Use the gradient text to update relevant prompt components
        gradient_lower = gradient_text.lower()
        
        # Universal component updates based on gradient insights
        if 'explore' in gradient_lower or 'try different' in gradient_lower:
            self.prompt_components['adaptive_strategy'] = gradient_text[:500]
        elif 'understand' in gradient_lower or 'observe' in gradient_lower:
            self.prompt_components['environment_understanding'] = gradient_text[:500]
        elif 'pattern' in gradient_lower or 'discover' in gradient_lower:
            self.prompt_components['pattern_recognition'] = gradient_text[:500]
        elif 'hypothesis' in gradient_lower or 'test' in gradient_lower:
            self.prompt_components['hypothesis_testing'] = gradient_text[:500]

    def _universal_failure_check(self, observation: str) -> bool:
        """
        Universal failure detection based on information theory
        NO domain knowledge - pure statistical approach
        """
        if not observation:
            return True
        
        obs_lower = observation.lower().strip()
        
        # 1. Check for universal failure phrases (action not understood)
        exact_failures = [
            "nothing happens.",
            "nothing happens",
            "you can't do that.",
            "you can't do that",
            "i don't understand that.",
            "i don't understand that",
            "that doesn't make sense.",
            "that doesn't make sense"
        ]
        
        if obs_lower in exact_failures:
            return True
        
        # 2. Check for action rejection patterns (universal)
        rejection_patterns = [
            "don't understand",
            "i don't understand",
            "not sure what you mean",
            "invalid command",
            "invalid action",
            "that's not a verb i recognise",
            "that's not something you can do",
            "can't do that",
            "cannot do that",
            "not possible"
        ]
        
        for pattern in rejection_patterns:
            if pattern in obs_lower:
                return True
        
        # 3. Check against learned failure patterns
        if hasattr(self, 'confirmed_error_messages'):
            if observation.strip() in self.confirmed_error_messages:
                return True
        
        # 4. Entropy check - very low entropy often means error
        unique_chars = len(set(observation.lower()))
        total_chars = len(observation)
        
        if total_chars > 0:
            char_diversity = unique_chars / total_chars
            # If text is extremely repetitive
            if char_diversity < 0.15 and total_chars > 10:
                return True
        
        # 5. Length check - very short responses often indicate failure
        if len(obs_lower) < 10 and obs_lower not in ['ok', 'done', 'success']:
            # Short responses that aren't success indicators
            if not any(word in obs_lower for word in ['you', 'the', 'on', 'in', 'at']):
                return True
        
        # Default: NOT a failure (important for exploration)
        return False

    def _update_failure_knowledge(self, action: str, observation: str, confirmed_failure: bool):
        """
        Learn what failures look like over time
        Only called when we KNOW something failed (e.g., task not completed)
        """
        if not hasattr(self, 'confirmed_error_messages'):
            self.confirmed_error_messages = set()
        
        if confirmed_failure and len(observation) < 100:
            # Short messages that led to failure are likely error messages
            self.confirmed_error_messages.add(observation.strip())



    def inject_discovered_knowledge(self, knowledge: Dict[str, Any]):
        """Inject discovered environment knowledge"""
        self.environment_knowledge = knowledge
        
        # IMPORTANT: Don't list discovered action templates in the prompt
        # The agent should ONLY choose from the current valid actions
        if knowledge and knowledge.get('action_space'):
            # Don't touch available_actions here - it should be set from env.get_current_valid_actions()
            # This function is only for storing discovered patterns, not runtime actions
            pass
        
        if knowledge.get('uses_numbered_items'):
            self.discovered_knowledge['uses_numbers'] = True
        
        if knowledge.get('completion_actions'):
            self.discovered_knowledge['completion_actions'].update(knowledge['completion_actions'])
        
        # Don't update action_discovery component - keep it generic
        # The agent must learn to select from provided valid actions only
    
    def generate_initial_prompt_with_actions(self, observation: str, memory: List[str], 
                                            valid_actions: List[str]) -> str:
        """Generate initial prompt WITHOUT ranking - just store actions for reference"""
        print(f"\n[PROMPT GEN] Called with {len(valid_actions)} actions")
        self.interaction_count += 1
        
        # Store the valid actions for later reference (no ranking)
        self._last_valid_actions = valid_actions
        
        # Simple prompt without action list (since reasoning will handle it)
        prompt = f"""You are an AI agent in a text-based environment.

    CURRENT STATE:
    {observation}

    TASK TO COMPLETE: {self._task if hasattr(self, '_task') and self._task else 'Unknown task'}

    Number of available actions: {len(valid_actions)}
    """
        
        # Add memory insights if available
        if memory:
            prompt += "\nKEY LEARNINGS FROM PREVIOUS ATTEMPTS:\n"
            for i, reflection in enumerate(memory[-2:]):
                lines = reflection.split('\n')
                actionable = [line.strip() for line in lines 
                            if any(word in line.lower() for word in ['must', 'should', 'need', 'requires', 'always', 'never'])]
                if actionable:
                    prompt += f"Memory {i+1}: " + "; ".join(actionable[:2]) + "\n"
        
        return prompt


    def _calculate_state_similarity(self, state1: str, state2: str) -> float:
        """
        Calculate similarity between two state signatures
        Universal - uses set similarity
        """
        if not state1 or not state2:
            return 0.0
        
        # Extract tokens from state signatures
        tokens1 = set(state1.split(':')[-1].split(',')) if ':' in state1 else set(state1.split())
        tokens2 = set(state2.split(':')[-1].split(',')) if ':' in state2 else set(state2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0

    def _pattern_matches(self, text: str, pattern: str) -> bool:
        """
        Universal pattern matching without language assumptions
        """
        text_lower = text.lower()
        pattern_lower = pattern.lower()
        
        # Direct substring match
        if pattern_lower in text_lower:
            return True
        
        # Token overlap match (at least 50% tokens match)
        pattern_tokens = set(pattern_lower.split())
        text_tokens = set(text_lower.split())
        
        if pattern_tokens and text_tokens:
            overlap = len(pattern_tokens & text_tokens)
            if overlap >= len(pattern_tokens) * 0.5:
                return True
        
        return False

    def _last_action_was_examine(self) -> bool:
        """Check if last action was examine"""
        if hasattr(self, 'action_outcome_history') and self.action_outcome_history:
            last_action = self.action_outcome_history[-1]['action']
            return last_action.startswith('examine')
        return False

    def _get_tried_actions(self) -> set:
        """Get set of all tried actions"""
        if hasattr(self, 'action_outcome_history'):
            return {h['action'] for h in self.action_outcome_history}
        return set()

    def _extract_task_from_observation(self, observation: str) -> Optional[str]:
        """
        Universal task extraction using information density and directive detection
        NO domain-specific knowledge, NO structural assumptions
        """
        if not observation:
            return None
        
        # Clean observation universally (remove Python artifacts)
        clean_obs = observation
        if clean_obs.startswith("('") or clean_obs.startswith('("'):
            clean_obs = clean_obs[2:]
            for suffix in ["',)", '",)', "')", '")']:
                if clean_obs.endswith(suffix):
                    clean_obs = clean_obs[:-len(suffix)]
                    break
        
        clean_obs = clean_obs.replace('\\n', '\n').replace('\\t', '\t')
        clean_obs = clean_obs.replace("\\'", "'").replace('\\"', '"')
        
        # PRIORITY 1: Check for standard "Your task is to:" format (most common in benchmarks)
        import re
        
        # Pattern 1: Standard format with period at end
        task_match = re.search(
            r'[Yy]our task is to:\s*([^.\n]+(?:\.[^.\n]+)*?)\.(?:\s|$)', 
            clean_obs
        )
        
        if task_match:
            task = task_match.group(1).strip()
            # Don't strip the content words, just clean formatting
            task = re.sub(r'\s+', ' ', task)  # Normalize whitespace
            if len(task) > 10:
                return task
        
        # Pattern 2: Task might not end with period
        task_match = re.search(
            r'[Yy]our task is to:\s*([^\n]+?)(?:\n\n|\n[A-Z>]|\n-=|$)', 
            clean_obs
        )
        
        if task_match:
            task = task_match.group(1).strip()
            task = re.sub(r'\s+', ' ', task)
            # Remove only trailing punctuation that's clearly not part of task
            task = task.rstrip(',;!:')  # Don't remove period - might be part of task
            if len(task) > 10:
                return task
        
        # PRIORITY 2: Try other common patterns
        patterns = [
            r'[Tt]ask:\s*([^\n]+)',
            r'[Gg]oal:\s*([^\n]+)',
            r'[Oo]bjective:\s*([^\n]+)',
            r'[Mm]ission:\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_obs)
            if match:
                task = match.group(1).strip()
                task = re.sub(r'\s+', ' ', task)
                if len(task) > 10:
                    return task
        
        # FALLBACK: Information density approach (original method)
        # This handles non-standard formats
        lines = clean_obs.split('\n')
        task_candidates = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            score = 0.0
            
            # Score based on directive indicators
            directive_words = ['task', 'goal', 'objective', 'mission', 'quest', 
                            'complete', 'achieve', 'your', 'must', 'need', 'should']
            for word in directive_words:
                if word in line_clean.lower():
                    score += 2.0
            
            # Has colon (often precedes task)
            if ':' in line_clean:
                score += 1.0
            
            # Information density
            tokens = line_clean.lower().split()
            if tokens:
                stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 
                            'been', 'being', 'have', 'has', 'had', 'do', 'does'}
                content_words = [t for t in tokens if t not in stop_words]
                if tokens:
                    density = len(content_words) / len(tokens)
                    score += density * 3.0
            
            # Action verbs (common in tasks)
            action_verbs = ['put', 'place', 'take', 'get', 'find', 'move', 'clean',
                        'heat', 'cool', 'examine', 'look', 'search', 'open',
                        'close', 'turn', 'pick', 'drop', 'use', 'make', 'create']
            for verb in action_verbs:
                if verb in line_clean.lower():
                    score += 1.5
            
            task_candidates.append((score, i, line_clean))
        
        # Select highest scoring candidate
        task_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if task_candidates and task_candidates[0][0] > 2.0:
            best_score, best_idx, best_line = task_candidates[0]
            
            # Extract task content
            if ':' in best_line:
                task_text = best_line.split(':', 1)[1].strip()
                
                # Check if next line continues the task
                if best_idx + 1 < len(lines):
                    next_line = lines[best_idx + 1].strip()
                    # Don't add lines that start a new section
                    if (next_line and 
                        not next_line.startswith('>') and 
                        not next_line.startswith('-') and
                        not any(word in next_line.lower() for word in ['you are', 'you see'])):
                        # Check if it's a continuation
                        if not next_line[0].isupper() or len(next_line.split()) < 10:
                            task_text += ' ' + next_line
                
                return task_text if len(task_text) > 10 else None
            else:
                return best_line if len(best_line) > 10 else None
        
        return None

    def _build_computation_graph(self, interactions: List[Tuple[str, str]]) -> Dict:
        """Build computation graph representation of the trajectory"""
        graph = {
            'nodes': [],
            'edges': [],
            'state_transformations': []
        }
        
        prev_state_hash = None
        for i, (action, observation) in enumerate(interactions):
            # Create node for this computation step
            node = {
                'id': i,
                'type': 'computation',
                'input': action,
                'output': observation,
                'state_hash': hash(observation)  # Simple state representation
            }
            graph['nodes'].append(node)
            
            # Track state transformations
            if prev_state_hash:
                transformation = {
                    'from_state': prev_state_hash,
                    'to_state': node['state_hash'],
                    'via_action': action,
                    'changed': prev_state_hash != node['state_hash']
                }
                graph['state_transformations'].append(transformation)
            
            prev_state_hash = node['state_hash']
        
        return graph

    def _format_computation_trace(self, graph: Dict) -> str:
        """Format computation graph as readable trace"""
        trace = []
        for transform in graph['state_transformations'][-10:]:  # Last 10 steps
            change_indicator = "â†’" if transform['changed'] else "â‰ˆ"
            trace.append(f"State[{transform['from_state']%1000}] --[{transform['via_action']}]--> State[{transform['to_state']%1000}] {change_indicator}")
        return '\n'.join(trace)

    def _clip_gradients(self, gradients: Dict[str, str], max_length: int = 2000) -> Dict[str, str]:
        """Clip gradients to prevent explosion (TextGrad technique)"""
        clipped = {}
        for component, gradient in gradients.items():
            if len(gradient) > max_length:
                # Clip but preserve key information
                clipped[component] = gradient[:max_length-20] + "... [clipped for stability]"
            else:
                clipped[component] = gradient
        return clipped

    def _apply_momentum(self, gradients: Dict[str, str], beta: float = 0.9) -> Dict[str, str]:
        """Apply momentum to gradients (TextGrad optimization technique)"""
        if not hasattr(self, 'gradient_velocity'):
            self.gradient_velocity = {}
        
        momentum_gradients = {}
        for component, gradient in gradients.items():
            if component in self.gradient_velocity:
                # Combine with previous gradient direction
                prev = self.gradient_velocity[component]
                # Simple text momentum: blend current and previous
                momentum_gradients[component] = f"{gradient} [Building on: {prev[:50]}...]"
            else:
                momentum_gradients[component] = gradient
            
            # Update velocity
            self.gradient_velocity[component] = gradient
        
        return momentum_gradients
    
    def compute_prompt_gradient(self, trajectory: str, success: bool, task: str = None,
                                reflection: str = None, progress_score: int = None, valid_actions: List[str] = None) -> Dict[str, str]:
        """Compute gradients with progress awareness"""
        
        # Parse interactions
        interactions = self._parse_trajectory(trajectory)
        if not interactions:
            return {}
        
        gradients = {}
        
        # NEW: Handle progress-based gradients for step reflexions
        if progress_score is not None:
            if progress_score >= 7:
                # High progress - reinforce
                if reflection:
                    # Handle dict reflection - extract insight field
                    refl_text = reflection.get('insight', str(reflection)) if isinstance(reflection, dict) else str(reflection)
                    if 'must' in refl_text.lower():
                        must_do = [line.strip() for line in refl_text.split('\n')
                                if 'must' in line.lower() or 'should' in line.lower()][:2]
                        gradients['adaptive_strategy'] = f"WORKING WELL (progress {progress_score}/10): {'; '.join(must_do)}"
                    else:
                        gradients['adaptive_strategy'] = f"Continue current approach (progress {progress_score}/10)"
                else:
                    gradients['adaptive_strategy'] = f"Continue current approach (progress {progress_score}/10)"

                gradients['action_discovery'] = "Current action pattern is effective"

            elif progress_score >= 4:
                # Moderate progress - refine
                if reflection:
                    # Handle dict reflection - extract insight field
                    refl_text = reflection.get('insight', str(reflection)) if isinstance(reflection, dict) else str(reflection)
                    # Extract what needs adjustment
                    need_lines = [line.strip() for line in refl_text.split('\n')
                                if 'need' in line.lower() or 'try' in line.lower()][:2]
                    gradients['adaptive_strategy'] = f"REFINE (progress {progress_score}/10): {'; '.join(need_lines)}"

            else:
                # Low progress - correct course
                if reflection:
                    # Handle dict reflection - extract insight field
                    refl_text = reflection.get('insight', str(reflection)) if isinstance(reflection, dict) else str(reflection)
                    avoid_lines = [line.strip() for line in refl_text.split('\n')
                                if 'failed' in line.lower() or 'avoid' in line.lower()][:2]
                    gradients['adaptive_strategy'] = f"CHANGE APPROACH (low progress {progress_score}/10): {'; '.join(avoid_lines)}"

                    # Universal LLM-based action extraction using fast_model (gpt-4o-mini)
                    if len(reflection) > 20:
                        # Build valid actions context if available
                        valid_actions_context = ""
                        if valid_actions and len(valid_actions) > 0:
                            actions_list = "\n".join(f"   - {act}" for act in valid_actions[:30])
                            if len(valid_actions) > 30:
                                actions_list += f"\n   ... and {len(valid_actions)-30} more"
                            valid_actions_context = f"""
Valid Actions (you MUST extract from this list ONLY):
{actions_list}

"""

                        extraction_prompt = f"""Analyze this Reflexion output to extract any recommended action.

Reflexion:
---
{reflection}
---

{valid_actions_context}Task: Does Reflexion recommend a SPECIFIC action to try next?

Output ONE of these:
1. The EXACT action from the Valid Actions list above (if Reflexion recommends one)
2. The word "none" (if no action recommended OR action not in valid list)

CRITICAL: Only output actions that appear in the Valid Actions list. Do not paraphrase or modify.

Output:"""

                        # Use existing fast_model (already configured as gpt-4o-mini or gemini-1.5-flash)
                        try:
                            from vllm import SamplingParams
                        except ImportError:
                            from shared_model import SamplingParams
                        # Dynamic model loading
                        if _os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
                            from shared_model_gemini import fast_model
                        else:
                            from shared_model import fast_model

                        extraction_output = fast_model.generate([extraction_prompt], SamplingParams(
                            max_tokens=80,
                            temperature=0.0,
                            stop=['\n']
                        ))[0].outputs[0].text.strip()

                        # Strict validation (no silent fallbacks)
                        if extraction_output and extraction_output.lower() != 'none':
                            if len(extraction_output) > 3:
                                # Update both fields for complete synergy
                                gradients['action_discovery'] = extraction_output
                                gradients['next_action_guidance'] = extraction_output  # CRITICAL: This gets prioritized!
                                print(f"✓ [TEXTGRAD] Extracted action: {extraction_output}")
                            else:
                                print(f"❌ [TEXTGRAD] Invalid (too short): '{extraction_output}'")
                        else:
                            if DEBUG_CRITIC:
                                print(f"[TEXTGRAD] No action recommended by Reflexion")
        
        # Handle full task success/failure (existing code but cleaned)
        elif success:
            # Task completed successfully
            action_sequence = [act for act, _ in interactions]
            key_actions = action_sequence[-3:] if len(action_sequence) > 3 else action_sequence
            
            gradients['adaptive_strategy'] = f'SUCCESS PATTERN: [{", ".join(key_actions)}] completes "{task}"'
            gradients['environment_understanding'] = f'Task completed in {len(action_sequence)} steps'
            gradients['action_discovery'] = f'Successful sequence: {" -> ".join(key_actions)}'
            gradients['pattern_recognition'] = 'This exact sequence works for this task type'
            gradients['hypothesis_testing'] = 'Hypothesis confirmed - reuse this pattern'
            gradients['task_decomposition'] = f'Complete sequence: {" -> ".join(action_sequence[:5])}'
             
        else:
            # Extract semantic insights from reflection WITHOUT assumptions
            if reflection:
                import re
                
                # Extract lines that indicate missing elements (universal patterns)
                missing_elements = []
                for line in reflection.split('\n'):
                    if any(indicator in line.lower() for indicator in ['never', 'failed to', 'missed', 'forgot', 'didn\'t']):
                        # Extract what comes after these indicators
                        missing_elements.append(line.strip().lstrip('-').strip())
                
                # Extract lines that indicate requirements (universal patterns)
                requirements = []
                for line in reflection.split('\n'):
                    if any(indicator in line.lower() for indicator in ['must', 'need', 'require', 'should have', 'necessary']):
                        requirements.append(line.strip())
                
                # Extract causal relationships (universal patterns)
                causal_insights = []
                for line in reflection.split('\n'):
                    if any(indicator in line.lower() for indicator in ['because', 'since', 'therefore', 'leads to', 'causes']):
                        causal_insights.append(line.strip())
                
                # Extract hypotheses (universal patterns)
                hypotheses = []
                for line in reflection.split('\n'):
                    if any(indicator in line.lower() for indicator in ['try', 'test', 'attempt', 'explore', 'check']):
                        hypotheses.append(line.strip())
                


                # Build gradients from extracted insights

                if missing_elements:
                    # Accumulate with previous knowledge
                    prev_strategy = self.prompt_components.get('adaptive_strategy', '')
                    if 'LEARNED:' in prev_strategy:
                        # Extract previous learned part
                        learned_part = prev_strategy.split('|')[0].replace('LEARNED:', '').strip()
                        cleaned_elements = [elem.lstrip('-').strip() for elem in missing_elements[:2]]
                        gradients['adaptive_strategy'] = f"AVOID: {'; '.join(cleaned_elements)}. REMEMBER: {learned_part}"
                    else:
                        # Clean up the text for better readability
                        cleaned_elements = [elem.lstrip('-').strip() for elem in missing_elements[:3]]
                        gradients['adaptive_strategy'] = f"AVOID: {'; '.join(cleaned_elements)}"
                else:
                    gradients['adaptive_strategy'] = 'Adjust approach based on observations'
                
                if causal_insights:
                    # Use full causal insight
                    gradients['environment_understanding'] = ' '.join(causal_insights[:3])
                else:
                    gradients['environment_understanding'] = 'Observe state changes more carefully'
                
                if requirements:
                    # Use full requirements text
                    gradients['action_discovery'] = ' '.join(requirements[:3])
                else:
                    gradients['action_discovery'] = 'Explore action effects systematically'
                
                if hypotheses:
                    gradients['hypothesis_testing'] = hypotheses[0]
                else:
                    gradients['hypothesis_testing'] = 'Form testable hypotheses'
                
                
                # Extract specific task requirements from reflection
                task_steps = []
                for line in reflection.split('\n'):
                    if any(word in line.lower() for word in ['first', 'then', 'next', 'finally', 'must']):
                        task_steps.append(line.strip())
                
                # Clean task_decomposition specifically
                if task_steps:
                    # Remove numbered list prefixes and keep only the content
                    import re
                    cleaned_steps = []
                    for step in task_steps[:3]:
                        cleaned = re.sub(r'^\d+\.\s*\*?\*?[A-Z\s]+\*?\*?:\s*', '', step)
                        cleaned = cleaned.split('.')[0] + '.' if '.' in cleaned else cleaned
                        if len(cleaned) > 20:
                            cleaned_steps.append(cleaned)
                    if cleaned_steps:
                        gradients['task_decomposition'] = ' -> '.join(cleaned_steps)
                else:
                    gradients['task_decomposition'] = f"Break down '{task}' into subtasks"


                # Track actual patterns from failures
                if len(interactions) >= 50:
                    gradients['pattern_recognition'] = f"TIMEOUT after {len(interactions)} steps. Need more direct approach for '{task[:30]}'"
                else:
                    gradients['pattern_recognition'] = f"Failed at step {len(interactions)}. Last action: {interactions[-1][0] if interactions else 'none'}"
                
            else:
                # No reflection available
                gradients['adaptive_strategy'] = 'Insufficient information for strategy update'
                gradients['environment_understanding'] = 'More exploration needed'
                gradients['action_discovery'] = 'Try different action types'
                gradients['pattern_recognition'] = 'No patterns identified yet'
                gradients['hypothesis_testing'] = 'Gather more data'
                gradients['task_decomposition'] = 'Task structure unclear'
        
        return gradients



    def create_copy_for_environment(self):
        """Create an isolated copy for a specific environment"""
        import copy
        new_pg = DynamicPromptGenerator()
        # CRITICAL: Copy the CURRENT learned components, not defaults
        new_pg.prompt_components = copy.deepcopy(self.prompt_components)
        new_pg.discovered_knowledge = copy.deepcopy(self.discovered_knowledge)
        new_pg.prompt_gradients = copy.deepcopy(self.prompt_gradients)  # Keep gradient history
        new_pg._task = self._task if hasattr(self, '_task') else None
        new_pg.interaction_count = self.interaction_count  # Keep count
        # Keep the momentum velocity for continued learning
        if hasattr(self, 'gradient_velocity'):
            new_pg.gradient_velocity = copy.deepcopy(self.gradient_velocity)
        return new_pg

    def update_prompt_components(self, gradients: Dict[str, str]):
        """Update components with clean, actionable insights only"""
        
        self.prompt_gradients.append(gradients)
        
        for component, gradient in gradients.items():
            if component not in self.prompt_components:
                continue
            
            # Skip non-component keys
            if component in ['structured_insights', 'raw_reflection']:
                continue
            
            # Extract clean insight
            cleaned = self._extract_clean_insight(gradient)
            
            if cleaned and len(cleaned) > 10:
                # Check if this is a duplicate of current value
                current = self.prompt_components[component]
                if cleaned != current and cleaned not in current:
                    # Apply with momentum to prevent overwrites
                    self.update_component_with_momentum(component, cleaned)
            
            if DEBUG_CRITIC:
                print(f"[UPDATE] {component}: {self.prompt_components[component]}")

    def _extract_clean_insight(self, text: str) -> str:
        """Extract a clean, actionable insight from gradient text - NO TRUNCATION, NO FALLBACKS"""
        import re
        
        if not text:
            return ""
        
        # Clean the text but preserve content
        cleaned = text
        
        # Remove STEPS: prefix
        cleaned = re.sub(r'^STEPS:\s*', '', cleaned)
        
        # Remove numbered lists with markdown
        cleaned = re.sub(r'\d+\.\s*\*\*[^*]+\*\*:?\s*', '', cleaned)
        
        # Remove numbered lists without markdown  
        cleaned = re.sub(r'\d+\.\s*[A-Z][^:]*:\s*', '', cleaned)
        
        # Remove any remaining numbers at start
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # Remove markdown formatting
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Replace arrows with periods
        cleaned = cleaned.replace(' -> ', '. ').replace(' → ', '. ')
        
        # Clean up whitespace
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.strip()
        
        # CRITICAL: Return the FULL cleaned text - NO TRUNCATION
        # Only limit if it's absolutely massive (indicates error)
        if len(cleaned) > 50000:  # Increased from 5000 to 50000
            # This is an error case - something generated way too much text
            return cleaned[:50000] + "... [ERROR: Excessive generation]"
        
        
        return cleaned

    def update_component_with_momentum(self, component: str, new_gradient: str, momentum: float = 0.7):
        if component not in self.prompt_components:
            return
        
        if len(new_gradient) < 10:
            return
        
        # Prevent accumulation of contradictory gradients
        if not hasattr(self, '_gradient_hash'):
            self._gradient_hash = set()
        
        # Create hash of this gradient
        import hashlib
        gradient_hash = hashlib.md5(new_gradient.encode()).hexdigest()[:8]
        
        # Skip if we've seen this exact gradient before
        if gradient_hash in self._gradient_hash:
            return
        self._gradient_hash.add(gradient_hash)
        
        new_gradient = self._extract_clean_insight(new_gradient)
            
        # Initialize history
        if not hasattr(self, '_component_history'):
            self._component_history = defaultdict(list)
        
        # Score new gradient
        new_importance = self._score_gradient_importance(new_gradient)
        

        # Store with metadata
        self._component_history[component].append({
            'text': new_gradient,
            'timestamp': self.interaction_count,
            'importance': new_importance
        })
        
        # Keep only recent history
        if len(self._component_history[component]) > 50:
            # Keep only high importance recent items
            history = self._component_history[component]
            history.sort(key=lambda x: (x['importance'], -x['timestamp']), reverse=True)
            self._component_history[component] = history[:30]
        
        # INTELLIGENT MERGING: Combine insights by category
        history = self._component_history[component]

        # Categorize insights
        must_do = []
        must_avoid = []
        hypotheses = []
        general = []
        success_patterns_by_task = {}  # NEW: Group success patterns by task type

        for item in history[-10:]:  # Last 10 items
            text_lower = item['text'].lower()

            # Extract task-specific SUCCESS PATTERNS
            if 'success pattern:' in text_lower:
                # Extract task from pattern like: "SUCCESS PATTERN: [actions] completes \"task name\""
                import re
                task_match = re.search(r'completes\s+["\']([^"\']+)["\']', item['text'], re.IGNORECASE)
                if task_match:
                    task_type = task_match.group(1).strip()
                    # Normalize task type (e.g., "cool some pan..." -> "cool pan")
                    task_key = ' '.join(task_type.split()[:3])  # First 3 words
                    if task_key not in success_patterns_by_task:
                        success_patterns_by_task[task_key] = []
                    success_patterns_by_task[task_key].append(item)
            elif 'must' in text_lower or 'should' in text_lower:
                must_do.append(item)
            elif 'avoid' in text_lower or 'never' in text_lower or "don't" in text_lower:
                must_avoid.append(item)
            elif 'hypothesis' in text_lower or 'might' in text_lower or 'try' in text_lower:
                hypotheses.append(item)
            else:
                general.append(item)

        # Build STRUCTURED insight instead of concatenation
        final_insight_parts = []

        # PRIORITY 1: Include ALL success patterns (one per task type) - NEVER lose these!
        if success_patterns_by_task:
            for task_key, patterns in success_patterns_by_task.items():
                # Use the most recent success pattern for each task
                best_pattern = max(patterns, key=lambda x: (x['timestamp'], x['importance']))
                final_insight_parts.append(best_pattern['text'])

        # PRIORITY 2: Add failure avoidance (learn from mistakes)
        if must_avoid:
            # Take TOP 2 most important failures to learn from
            must_avoid.sort(key=lambda x: x['importance'], reverse=True)
            for avoid_item in must_avoid[:2]:
                avoid_text = avoid_item['text']
                if not avoid_text.startswith('AVOID:'):
                    avoid_text = f"AVOID: {avoid_text}"
                final_insight_parts.append(avoid_text)

        # PRIORITY 3: Positive directives (what to do)
        if must_do:
            best_must = max(must_do, key=lambda x: x['importance'])
            final_insight_parts.append(best_must['text'])

        # PRIORITY 4: Hypothesis if no strong directives
        if not final_insight_parts and hypotheses:
            best_hyp = max(hypotheses, key=lambda x: x['importance'])
            final_insight_parts.append(best_hyp['text'])

        # PRIORITY 5: General insight as last resort
        if not final_insight_parts and general:
            best_general = max(general, key=lambda x: x['importance'])
            final_insight_parts.append(best_general['text'])
        
        # Update component with FULL structured insight
        if final_insight_parts:
            # Store the COMPLETE insight - NO TRUNCATION
            full_insight = ' '.join(final_insight_parts)  # Join ALL parts, not just first
    
            self.prompt_components[component] = full_insight
        if DEBUG_CRITIC:
            print(f"[UPDATE] {component}: {self.prompt_components[component]}...")
            # Also log the importance score
            print(f"  Importance: {new_importance}")
            print(f"  Momentum: {momentum}")

    def _score_gradient_importance(self, gradient: str) -> float:
        """Score gradient importance for prioritization"""
        score = 0.0
        grad_lower = gradient.lower()
        
        # High-value keywords
        importance_keywords = {
            'must': 5, 'should': 4, 'never': 5, 'always': 5,
            'failed because': 4, 'succeeded': 3, 'exact': 4,
            'avoid': 3, 'critical': 5, 'important': 3
        }
        
        for keyword, weight in importance_keywords.items():
            if keyword in grad_lower:
                score += weight
        
        # Specificity bonus
        if "'" in gradient:  # Contains quoted actions
            score += 3
        if any(char.isdigit() for char in gradient):  # Contains numbers
            score += 2
        
        # Length penalty for overly verbose
        if len(gradient) > 200:
            score -= 2
        
        return score

    def _parse_trajectory(self, trajectory: str) -> List[Tuple[str, str]]:
        """Parse trajectory into interaction pairs - SIMPLIFIED"""
        interactions = []
        lines = trajectory.split('\n')
        
        current_action = None
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                current_action = line[1:].strip()
            elif current_action and line:
                interactions.append((current_action, line))
                
                # Simple learning from interaction (no patterns)
                self._learn_from_interaction(current_action, line)
                
                current_action = None
        
        return interactions
    

    
    def _learn_from_interaction(self, action: str, observation: str):
        """Learn from individual interaction with state-action tracking - SIMPLIFIED VERSION"""
        
        # Track when new actions appear
        current_action_count = len(self.discovered_knowledge.get('available_actions', []))
        if current_action_count > self.discovered_knowledge.get('max_actions_seen', 0):
            recent_sequence = [h['action'] for h in self.action_outcome_history[-5:]]
            self.discovered_knowledge['max_actions_seen'] = current_action_count
        
        # Store interaction WITHOUT learning values or patterns
        self.action_outcome_history.append({
            'action': action,
            'observation': observation,
           
            'available_actions': current_action_count
        })
        
        # Debug output (keep for debugging)
        if DEBUG_ACTOR:
            print(f"\n{'='*60}")
            print(f"[LEARN] Action: {action}")
            print(f"[LEARN] Available actions: {current_action_count}")
            print(f"{'='*60}\n")

    
    def _parse_critiques(self, critique_text: str) -> Dict[str, str]:
        """Parse critique text into component gradients"""
        gradients = {}
        lines = critique_text.split('\n')
        
        current_component = None
        current_critique = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for component start
            found = False
            for component in self.prompt_components:
                if line.lower().startswith(component + ':'):
                    # Save previous
                    if current_component and current_critique:
                        gradients[current_component] = ' '.join(current_critique)
                    
                    # Start new
                    current_component = component
                    remainder = line[len(component)+1:].strip()
                    current_critique = [remainder] if remainder else []
                    found = True
                    break
            
            if not found and current_component:
                current_critique.append(line)
        
        # Save last component
        if current_component and current_critique:
            gradients[current_component] = ' '.join(current_critique)
        
        return gradients
    
  
  
    
    def _extract_actionable_insights(self, memory_text: str) -> List[str]:
        """Extract actionable insights from memory"""
        actionable = []
        lines = memory_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['should', 'must', 'try', 'avoid', 
                                                   'critical', 'important', 'need']):
                actionable.append(line.strip())
        
        return actionable
    
    def _extract_available_actions(self, observation: str) -> List[str]:
        """Extract available actions from observation text"""
        actions = []
        
        # Pattern 1: "Available actions: action1, action2, action3"
        pattern1 = r'[Aa]vailable actions?:\s*([^\n]+)'
        match = re.search(pattern1, observation)
        if match:
            action_str = match.group(1)
            raw_actions = re.split(r'[,;]|\sand\s', action_str)
            for action in raw_actions:
                action = action.strip().rstrip('.')
                if action and not action.startswith('('):
                    actions.append(action)
        
        # Pattern 2: "You can: action1, action2"
        pattern2 = r'(?:You can|Commands?|Actions?):\s*([^\n]+)'
        matches = re.findall(pattern2, observation, re.IGNORECASE)
        for match in matches:
            raw_actions = re.split(r'[,;]|\sand\s', match)
            for action in raw_actions:
                action = action.strip().rstrip('.')
                if action and action not in actions and not action.startswith('('):
                    actions.append(action)
        
        return actions
    
    def _compute_success_gradients(self, interactions: List[Tuple[str, str]], 
                                 task: str, reflection: str) -> Dict[str, str]:
        """Generate reinforcement gradients for successful attempts"""
        gradients = {}
        
        # Analyze what worked
        successful_structures = set()
        for action, obs in interactions:
            if "nothing happens" not in obs.lower():
                structure = self._get_action_structure(action)
                successful_structures.add(structure)
        
        if successful_structures:
            patterns_str = ', '.join(list(successful_structures)[:3])
            gradients['adaptive_strategy'] = (
                f"SUCCESS REINFORCEMENT: Patterns {patterns_str} led to task completion. "
                f"Continue using these structures."
            )
        
        return gradients
    
    def get_prompt_optimization_state(self) -> Dict[str, Any]:
        """Get current optimization state for logging - SIMPLIFIED"""
        
        # Build optimization state
        state = {
            'components': self.prompt_components.copy(),
            'discovered_knowledge': {
                'available_actions': list(self.discovered_knowledge.get('available_actions', []))

                # REMOVED: hypotheses and discovered_patterns
            },
            'gradient_history': len(self.prompt_gradients) if hasattr(self, 'prompt_gradients') else 0,
            'num_updates': len(self.prompt_gradients) if hasattr(self, 'prompt_gradients') else 0,
            'interaction_count': self.interaction_count,
            'has_environment_knowledge': self.environment_knowledge is not None if hasattr(self, 'environment_knowledge') else False,
            'available_actions_found': len(self.discovered_knowledge.get('available_actions', [])),
            'uses_numbered_items': bool(self.discovered_knowledge.get('uses_numbers', False)),
            'completion_actions': list(self.discovered_knowledge.get('completion_actions', [])),
            'optimization_algorithm': 'TGD (Textual Gradient Descent)'
        }
        
        return state

