"""Universal Environment Understanding - MINIMAL VERSION - Only Failure Tracking"""

from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

class EnvironmentUnderstanding:
    def __init__(self):
        # Only track failures - that's all we need
        self.discovered_knowledge = {
            'failed_actions': set(),  # EXACT actions that failed
            'failure_phrases': set()  # Short failure messages
        }
        
        # Simple history tracking
        self.action_outcome_history = []  # List of (action, observation)
        self.current_task = None
        
        # Environment wrapper reference (if needed)
        self.wrapper = None
        
        # Discovered environment knowledge (from discovery phase)
        self.environment_knowledge = None
    
    def set_wrapper(self, wrapper):
        """Set reference to environment wrapper for domain-specific checks"""
        self.wrapper = wrapper
    
    def inject_discovered_knowledge(self, knowledge: Dict[str, Any]):
        """Store discovered knowledge from discovery phase"""
        self.environment_knowledge = knowledge
    
    def learn_from_interaction(self, action: str, observation: str, success: Optional[bool] = None):
        """Learn from EXACT action-observation pairs - ONLY TRACK FAILURES"""
        
        # Store interaction
        self.action_outcome_history.append((action, observation))
        
        # Only track failures
        if action:  # Only if there was an action
            if self._is_likely_failure(observation):
                # Track EXACT failed action
                self.discovered_knowledge['failed_actions'].add(action)
                # Track failure phrase if short (likely error message)
                if len(observation) < 100:
                    self.discovered_knowledge['failure_phrases'].add(observation.strip()[:50])
    
    def compute_state_transition_gradient(self, action: str, prev_obs: str, curr_obs: str) -> float:
        """
        Simple binary gradient - did state change or not?
        Returns: positive if changed, negative if no change or failure
        """
        # No change at all
        if prev_obs.strip() == curr_obs.strip():
            return -20.0
        
        # Check if it's a failure message
        if self._is_likely_failure(curr_obs):
            return -15.0
        
        # Some change occurred
        return 5.0
    
    def _is_likely_failure(self, observation: str, prev_observation: str = "") -> bool:
        """
        Universal failure detection - the ONLY thing we need to detect
        """
        if not observation:
            return True
        
        obs_lower = observation.lower().strip()
        
        # Universal failure phrases
        universal_failures = [
            "nothing happens",
            "don't understand", 
            "i don't understand",
            "invalid",
            "error",
            "can't do that",
            "not a verb i recognise",
            "not possible",
            "that's not",
            "you can't",
            "doesn't work",
            "not sure what you mean",
            "unrecognized",
            "unknown command",
            "that doesn't make sense",
            "i don't know how to",
            "you can't see any such thing",
            "there's no"
        ]
        
        # Check for failure phrases
        for failure in universal_failures:
            if failure in obs_lower:
                return True
        
        # No change detection
        if prev_observation and observation.strip() == prev_observation.strip():
            return True
        
        # Check learned failure phrases
        for phrase in self.discovered_knowledge.get('failure_phrases', []):
            if phrase.lower() in obs_lower:
                return True
        
        # Check wrapper-specific failures if available
        if self.wrapper and hasattr(self.wrapper, 'get_failure_indicators'):
            for indicator in self.wrapper.get_failure_indicators():
                if indicator.lower() in obs_lower:
                    return True
        
        return False
    
    def analyze_trajectory(self, trajectory: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Analyze trajectory for simple metrics - ONLY FAILURES AND REPETITIONS"""
        analysis = {
            'total_actions': len(trajectory),
            'unique_actions': len(set(act for act, _ in trajectory if act)),
            'repeated_actions': [],  # EXACT actions repeated
            'failure_count': 0,
            'no_change_count': 0
        }
        
        # Count EXACT action frequencies
        action_counts = Counter(act for act, _ in trajectory if act)
        
        # Find repeated EXACT actions
        for action, count in action_counts.items():
            if count > 1:
                analysis['repeated_actions'].append({
                    'exact_action': action,
                    'count': count
                })
        
        # Count failures
        prev_obs = ""
        for action, obs in trajectory:
            if self._is_likely_failure(obs, prev_obs):
                analysis['failure_count'] += 1
            elif obs.strip() == prev_obs.strip():
                analysis['no_change_count'] += 1
            prev_obs = obs
        
        return analysis
    
    def get_discovered_knowledge(self) -> Dict[str, Any]:
        """Return discovered knowledge for inspection"""
        return {
            'failed_actions': list(self.discovered_knowledge['failed_actions']),
            'failure_phrase_count': len(self.discovered_knowledge['failure_phrases']),
            'total_interactions': len(self.action_outcome_history)
        }