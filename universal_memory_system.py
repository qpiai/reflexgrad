"""Truly Universal Memory System with Zero Environment Assumptions"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import hashlib

class UniversalMemorySystem:
    def __init__(self, memory_dir: str = "universal_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # LAYER 1: Environmental Knowledge Graph
        self.environment_graph = {
            'locations': defaultdict(lambda: {
                'objects_seen': set(),
                'information_content': [],
                'visit_count': 0,
                'avg_tokens': 0
            }),
            'object_locations': defaultdict(set),
            'location_transitions': defaultdict(set)
        }
        
        # LAYER 2: Context-Aware Action Tracking
        self.contextual_actions = defaultdict(lambda: defaultdict(lambda: {
            'outcomes': [],
            'success_rate': 0.0,
            'contexts': []
        }))
        
        # LAYER 3: Sequence Pattern Storage
        self.sequence_patterns = {
            'successful_sequences': [],
            'partial_patterns': {},
            'sequence_embeddings': {}
        }

        # Keep existing tracking
        self.state_action_outcomes = defaultdict(lambda: defaultdict(list))
        
        # ADD THESE MISSING ATTRIBUTES:
        self.transition_graph = defaultdict(lambda: defaultdict(list))
        self.action_effects = {}  # Initialize action_effects
        self.action_location_effects = defaultdict(lambda: defaultdict(list))  # From earlier suggestion
        
        self.total_interactions = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        
        # Initialize tracking variables
        self._last_location = None  
        self._last_state_hash = None 
        self.action_prerequisites = {}
        self.semantic_memory = []
        self.episode_contexts = {}
        self.load_memory()

    def extract_location_info(self, observation: str) -> Dict:
        """Universal information extraction using state fingerprinting"""
        info = {
            'location': None,
            'objects': set(),
            'tokens': set()
        }
        
        # State hash as location identifier
        info['location'] = hashlib.md5(observation.encode()).hexdigest()[:8]
        
        # Token extraction
        tokens = observation.lower().split()
        info['tokens'] = set(tokens)
        
        # Universal object pattern: word followed by digit
        for i in range(len(tokens) - 1):
            if tokens[i + 1].isdigit():
                info['objects'].add(f"{tokens[i]}_{tokens[i + 1]}")
        
        return info
    
    def update_environment_graph(self, observation: str, action: str = None):
        """Update environmental knowledge graph with state transitions"""
        info = self.extract_location_info(observation)
        
        # Store state information
        state_hash = info['location']
        loc_data = self.environment_graph['locations'][state_hash]
        loc_data['visit_count'] += 1
        loc_data['objects_seen'].update(info['objects'])
        loc_data['information_content'].append(len(info['tokens']))
        loc_data['avg_tokens'] = sum(loc_data['information_content']) / len(loc_data['information_content'])
        
        # Track object-state mappings
        for obj in info['objects']:
            self.environment_graph['object_locations'][obj].add(state_hash)
        
        # Track state transitions if action caused change
        if action and hasattr(self, '_last_state_hash') and self._last_state_hash:
            if state_hash != self._last_state_hash:
                # Successful transition
                self.environment_graph['location_transitions'][self._last_state_hash].add((action, state_hash))
        
        self._last_state_hash = state_hash
    
    def get_context_hash(self, observation: str, recent_actions: List[str]) -> str:
        """Generate context hash from state features and action history"""
        # Recent actions as context
        action_context = '|'.join(recent_actions[-3:]) if recent_actions else ''
        
        # State features
        tokens = observation.lower().split()
        
        # Key features that define context
        features = [
            f"tokens_{len(tokens)}",
            f"digits_{sum(1 for t in tokens if any(c.isdigit() for c in t))}",
            f"carrying_{'yes' if any(word in tokens for word in ['carrying', 'holding']) else 'no'}",
            action_context
        ]
        
        context_str = '|'.join(features)
        return hashlib.md5(context_str.encode()).hexdigest()[:16]
    
    def store_sequence_pattern(self, trajectory: List[Tuple[str, str]], success: bool, task: str):
        """Extract and store sequence patterns from trajectory"""
        if not trajectory or not success:
            return
        
        # Extract action sequence
        action_sequence = [action for action, _, _ in trajectory]  # Unpack 3 elements
        
        # Store complete successful sequence
        self.sequence_patterns['successful_sequences'].append({
            'sequence': action_sequence,
            'task_tokens': set(task.lower().split()),
            'length': len(action_sequence)
        })
        
        # Convert partial_patterns to regular dict if it's not
        if not isinstance(self.sequence_patterns['partial_patterns'], dict):
            self.sequence_patterns['partial_patterns'] = {}
        
        # Extract partial patterns (n-grams)
        for n in range(2, min(6, len(action_sequence) + 1)):
            for i in range(len(action_sequence) - n + 1):
                # Use string key instead of tuple
                pattern_key = '|'.join(action_sequence[i:i+n-1])
                next_action = action_sequence[i+n-1]
                
                if pattern_key not in self.sequence_patterns['partial_patterns']:
                    self.sequence_patterns['partial_patterns'][pattern_key] = []
                self.sequence_patterns['partial_patterns'][pattern_key].append(next_action)
    
    def match_sequence_pattern(self, recent_actions: List[str], task: str) -> Optional[str]:
        """Find next action based on sequence patterns"""
        if len(recent_actions) < 2:
            return None
        
        # Try different pattern lengths
        for n in range(min(5, len(recent_actions)), 1, -1):
            # Use string key format
            pattern_key = '|'.join(recent_actions[-n:])
            if pattern_key in self.sequence_patterns.get('partial_patterns', {}):
                candidates = self.sequence_patterns['partial_patterns'][pattern_key]
                # Return most common next action
                if candidates:
                    return Counter(candidates).most_common(1)[0][0]
        
        return None
    
    def record_semantic_interaction(self, semantic_state_before: str, action: str,
                                action_reasoning: str,  # ADD THIS PARAMETER
                                semantic_state_after: str, prerequisites: Dict,
                                task_progress: Dict, success: bool, task: str,
                                episode_id: str) -> None:
        """Record interaction with semantic understanding from LLM"""

        # PROFILING: Track call count and memory growth
        import sys
        if not hasattr(self, '_profiling_call_count'):
            self._profiling_call_count = 0
            self._profiling_log_interval = 10  # Log every 10 calls

        self._profiling_call_count += 1

        # Initialize storage if needed
        if not hasattr(self, 'semantic_memory'):
            self.semantic_memory = []

        # Detect if state actually changed meaningfully
        state_changed = semantic_state_before != semantic_state_after
        objects_before = set(semantic_state_before.lower().split())
        objects_after = set(semantic_state_after.lower().split())
        new_objects_appeared = len(objects_after - objects_before) > 0

        # PROFILING: Log sizes periodically
        if self._profiling_call_count % self._profiling_log_interval == 0:
            print(f"\n[MEMORY LEAK PROFILING] Call #{self._profiling_call_count}")
            print(f"  semantic_memory length: {len(self.semantic_memory)}")
            print(f"  action_effects keys: {len(self.action_effects)}")
            print(f"  action_prerequisites keys: {len(self.action_prerequisites)}")
            print(f"  action_location_effects keys: {len(self.action_location_effects)}")
            print(f"  Sample state_before size: {len(semantic_state_before)} chars")
            print(f"  Sample state_after size: {len(semantic_state_after)} chars")

        self.semantic_memory.append({
            'episode_id': episode_id,
            'timestamp': self.total_interactions,
            'action': action,
            'action_reasoning': action_reasoning,
            'state_before': semantic_state_before,
            'state_after': semantic_state_after,
            'state_changed': state_changed,
            'new_objects_appeared': new_objects_appeared,
            'prerequisites_met': prerequisites.get('present', []),
            'prerequisites_missing': prerequisites.get('missing', []),
            'task_addressed': task_progress.get('addressed', []),
            'task_remaining': task_progress.get('remaining', []),
            'success': success,
            'task': task
        })


        # Extract location from state (universal pattern: "at X" or "in X")
        import re
        location_match = re.search(r'(?:at|in|on)\s+(\w+\s*\d*)', semantic_state_before.lower())
        if location_match:
            location = location_match.group(1)
            effect_key = f"{action}@{location}"
            self.action_location_effects[effect_key]['outcomes'].append({
                'state_changed': state_changed,
                'task_progress': len(task_progress.get('addressed', [])) > 0
            })

        # Learn action effects universally
        if action not in self.action_effects:
            self.action_effects[action] = {
                'causes_state_change': [],
                'causes_new_objects': [],
                'no_effect_count': 0
            }

        if state_changed:
            self.action_effects[action]['causes_state_change'].append(semantic_state_after)
        if new_objects_appeared:
            self.action_effects[action]['causes_new_objects'].append(list(objects_after - objects_before))
        if not state_changed and not new_objects_appeared:
            self.action_effects[action]['no_effect_count'] += 1
        # Learn action prerequisites
        if action not in self.action_prerequisites:
            self.action_prerequisites[action] = {
                'success_conditions': [],
                'failure_conditions': []
            }

        if success:
            self.action_prerequisites[action]['success_conditions'].append({
                'state': semantic_state_before,
                'prerequisites': prerequisites.get('present', [])
            })
        else:
            self.action_prerequisites[action]['failure_conditions'].append({
                'state': semantic_state_before,
                'missing': prerequisites.get('missing', [])
            })

        # PROFILING: Show detailed list growth
        if self._profiling_call_count % self._profiling_log_interval == 0:
            print(f"\n[MEMORY DUPLICATION ANALYSIS]")
            print(f"  action_effects['{action}']['causes_state_change'] has {len(self.action_effects[action]['causes_state_change'])} items")
            print(f"  action_prerequisites['{action}'] success: {len(self.action_prerequisites[action]['success_conditions'])}, fail: {len(self.action_prerequisites[action]['failure_conditions'])}")
            if action in self.action_effects and self.action_effects[action]['causes_state_change']:
                sample_entry = self.action_effects[action]['causes_state_change'][0]
                print(f"  Sample action_effects entry size: {len(sample_entry)} chars (DUPLICATE of state_after!)")
            print(f"  Total memory in action_effects for '{action}': ~{sum(len(s) for s in self.action_effects[action]['causes_state_change'])} chars\n")
        
        self.total_interactions += 1
    
    def get_semantic_recommendations(self, current_state_description: str, 
                                    task_remaining: List[str], available_actions: List[str]) -> Dict:
        """Get recommendations based on semantic understanding"""
        
        recommendations = {
            'strongly_recommended': [],
            'previously_succeeded': [],
            'avoid': [],
            'unexplored': []
        }
        
        if not hasattr(self, 'semantic_memory'):
            return recommendations
        
        # Find similar situations in memory
        for action in available_actions:
            if action in self.action_prerequisites:
                prereqs = self.action_prerequisites[action]

                # Check if current state matches successful conditions
                for success_case in prereqs.get('success_conditions', [])[-5:]:
                    # Also check if similar reasoning succeeded before
                    if success_case.get('action_reasoning'):
                        if len(success_case['action_reasoning']) > 10:  # Valid reasoning
                            recommendations['previously_succeeded'].append({
                                'action': action,
                                'reason': f"Similar reasoning: {success_case['action_reasoning'][:30]}"
                            })
                            break
                
                # Check failure conditions
                for failure_case in prereqs.get('failure_conditions', [])[-5:]:
                    if any(miss in failure_case.get('missing', []) for miss in task_remaining):
                        recommendations['avoid'].append({
                            'action': action,
                            'reason': f"Failed when missing: {', '.join(failure_case['missing'][:2])}"
                        })
                        break
            else:
                recommendations['unexplored'].append(action)
        
        # Find actions that address remaining task requirements
        for action in available_actions:
            action_lower = action.lower()
            if any(req.lower() in action_lower for req in task_remaining):
                recommendations['strongly_recommended'].append({
                    'action': action,
                    'reason': f"Addresses task requirement"
                })
        
        return recommendations



    def record_episode(self, trajectory: List[Tuple[str, str]], task: str, final_success: bool):
        """
        Record full episode for learning - semantic version
        """
        if not trajectory:
            return
        
        # Update counters
        if final_success:
            self.successful_episodes += 1
        else:
            self.failed_episodes += 1
        
        # Store episode with semantic understanding
        if not hasattr(self, 'episode_memory'):
            self.episode_memory = []
        
        self.episode_memory.append({
            'task': task,
            'success': final_success,
            'length': len(trajectory),
            'exact_actions': [action for action, _, _ in trajectory],
            'final_state': trajectory[-1][1][:200] if trajectory else "empty",
            'timestamp': self.total_interactions
        })
        
        # Learn successful sequences
        if final_success:
            action_sequence = [action for action, _, _ in trajectory]  # Unpack 3 elements
            self.sequence_patterns['successful_sequences'].append({
                'sequence': action_sequence,
                'task_tokens': set(task.lower().split()),
                'length': len(action_sequence)
            })
    
    def get_recommendations(self, current_state: str, task: str,
                        tried_actions: Set[str], available_actions: List[str],
                        context: Dict = None, current_episode_id: str = None,
                        recent_actions: List[str] = None) -> Dict:
        """Enhanced recommendations using all three memory layers"""
        recommendations = {
            'delete': [],
            'avoid': [],
            'historically_successful': [],
            'explore': [],
            'sequence_suggested': None,
            'location_hints': {}
        }
        
        # LAYER 3: Check sequence patterns first
        if recent_actions:
            next_action = self.match_sequence_pattern(recent_actions, task)
            if next_action and next_action in available_actions:
                recommendations['sequence_suggested'] = {
                    'action': next_action,
                    'reason': f'Continues successful pattern from {len(recent_actions)} previous actions'
                }
        
        # LAYER 1: Add location-based hints
        current_info = self.extract_location_info(current_state)
        task_tokens = set(task.lower().split())
        
        # Find objects mentioned in task
        for obj in task_tokens:
            if obj in self.environment_graph['object_locations']:
                locations = self.environment_graph['object_locations'][obj]
                recommendations['location_hints'][obj] = list(locations)
        
        
        # Add untried actions
        for action in available_actions:
            if action not in tried_actions and \
            not any(r['action'] == action for r in recommendations['delete']):
                # Check if already in explore list
                if not any(r['action'] == action for r in recommendations['explore']):
                    recommendations['explore'].append({
                        'action': action,
                        'reason': 'Never tried'
                    })
        
        # Suggest actions based on task needs
        task_tokens = set(task.lower().split())
        current_tokens = set(current_state.lower().split())
        needed_tokens = task_tokens - current_tokens

        if needed_tokens and hasattr(self, 'transformations'):
            for action, transforms in self.transformations.items():
                for transform in transforms[-5:]:  # Check recent transformations
                    if any(token in transform.get('added_tokens', []) for token in needed_tokens):
                        if action in available_actions:
                            recommendations['explore'].insert(0, {
                                'action': action,
                                'reason': 'Previously added needed tokens'
                            })
                            break

        return recommendations
    
    def is_action_deterministic(self, state_hash: str, action_hash: str, task_hash: str = None) -> Tuple[bool, float]:
        """
        Check if action appears deterministic in this EXACT state FOR THIS EXACT TASK
        Returns: (is_deterministic, confidence)
        """
        if state_hash not in self.state_action_outcomes:
            return True, 0.0  # Assume deterministic if no data
        
        if action_hash not in self.state_action_outcomes[state_hash]:
            return True, 0.0  # Assume deterministic if no data
        
        all_outcomes = self.state_action_outcomes[state_hash][action_hash]
        
        # Filter by task if provided
        if task_hash:
            outcomes = [o for o in all_outcomes if o.get('task_hash') == task_hash]
        else:
            outcomes = all_outcomes
            
        if len(outcomes) < 2:
            # Need at least 2 attempts to check variance
            return True, 1.0  # Assume deterministic until proven otherwise
        
        # Check if result hashes are identical (strong determinism signal)
        result_hashes = [o.get('result_hash', '') for o in outcomes]
        all_same_result = len(set(result_hashes)) == 1
        
        if all_same_result:
            # Exact same result every time = definitely deterministic
            return True, 1.0
        
        # Check effectiveness variance
        effectiveness_values = [o.get('effectiveness', 0) for o in outcomes]
        variance = np.var(effectiveness_values)
        
        # If variance is very low, likely deterministic
        if variance < 0.05:
            return True, 1.0 - variance
        
        # High variance = stochastic
        return False, variance
    
    def should_explore(self, state_hash: str, available_actions: List[str], 
                      tried_actions: Set[str]) -> Tuple[bool, float]:
        """
        Simple exploration decision based on coverage in THIS EXACT state
        """
        # Calculate coverage
        coverage = len(tried_actions) / max(len(available_actions), 1)
        
        # Check state-specific coverage
        if state_hash in self.state_action_outcomes:
            tried_in_state = len(self.state_action_outcomes[state_hash])
            state_coverage = tried_in_state / max(len(available_actions), 1)
        else:
            state_coverage = 0.0
        
        # Simple threshold
        explore_threshold = 0.3
        
        should_explore = coverage < explore_threshold or state_coverage < explore_threshold
        confidence = 1.0 - min(coverage, state_coverage)
        
        return should_explore, confidence
    
    def get_statistics(self) -> Dict:
        """
        Get memory statistics for debugging
        """
        return {
            'total_interactions': self.total_interactions,
            'unique_states': len(self.state_action_outcomes),
            'successful_episodes': self.successful_episodes,
            'failed_episodes': self.failed_episodes,
            'total_episodes': self.successful_episodes + self.failed_episodes,
            'success_rate': self.successful_episodes / max(1, self.successful_episodes + self.failed_episodes) if (self.successful_episodes + self.failed_episodes) > 0 else 0.0
        }
    

    
    def save_memory(self):
        """Save memory to disk"""
        memory_file = self.memory_dir / "universal_memory.pkl"
        memory_data = {
            'state_action_outcomes': dict(self.state_action_outcomes),
            'sequence_patterns': dict(self.sequence_patterns),
            'transition_graph': dict(self.transition_graph),
            'total_interactions': self.total_interactions,
            'successful_episodes': self.successful_episodes,
            'failed_episodes': self.failed_episodes
        }
        
        try:
            with open(memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            print(f"[MEMORY] Saved {len(self.semantic_memory)} semantic interactions")
        except Exception as e:
            print(f"[MEMORY] Error saving memory: {e}")
    
    def load_memory(self):
        """Load memory from disk"""
        memory_file = self.memory_dir / "universal_memory.pkl"
        if memory_file.exists():
            try:
                with open(memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                
                self.state_action_outcomes = defaultdict(lambda: defaultdict(list), 
                                                        memory_data.get('state_action_outcomes', {}))
                self.sequence_patterns = defaultdict(list, memory_data.get('sequence_patterns', {}))
                self.transition_graph = defaultdict(lambda: defaultdict(list), 
                                                   memory_data.get('transition_graph', {}))
                self.total_interactions = memory_data.get('total_interactions', 0)
                self.successful_episodes = memory_data.get('successful_episodes', 0)
                self.failed_episodes = memory_data.get('failed_episodes', 0)
                
                # Clean up old token_correlations and other pattern data if it exists
                if 'token_correlations' in memory_data:
                    print("[MEMORY] Ignoring old token_correlations data")
                if 'information_changes' in memory_data:
                    print("[MEMORY] Ignoring old information_changes data")
                
                print(f"[MEMORY] Loaded memory with {len(self.state_action_outcomes)} states")
                print(f"[MEMORY] Episodes: {self.successful_episodes} successful, {self.failed_episodes} failed")
                
            except Exception as e:
                print(f"[MEMORY] Error loading memory: {e}")
                print("[MEMORY] Starting with fresh memory")

# Global instance
universal_memory = UniversalMemorySystem()