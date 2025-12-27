"""Meta-discovery for transfer learning across environments"""



import json

import pickle

from typing import Dict, List, Any, Optional

from pathlib import Path



class MetaEnvironmentKnowledge:

    def __init__(self, knowledge_dir: str = "discovered_environments"):

        self.knowledge_dir = Path(knowledge_dir)

        self.knowledge_dir.mkdir(exist_ok=True)

        self.meta_patterns = {

            'common_action_verbs': set(),

            'common_formats': [],

            'universal_patterns': [],

            'environment_signatures': {}

        }

    

    def save_environment_knowledge(self, env_name: str, knowledge: Dict):

        """Save discovered knowledge for an environment"""

        filepath = self.knowledge_dir / f"{env_name}_knowledge.json"

        

        # Convert sets to lists for JSON serialization

        serializable = self._make_serializable(knowledge)

        

        with open(filepath, 'w') as f:

            json.dump(serializable, f, indent=2)

        

        # Update meta patterns

        self._update_meta_patterns(knowledge)

        

        # Save meta patterns

        self._save_meta_patterns()

    

    def load_similar_environment(self, current_obs: str) -> Optional[Dict]:

        """Find and load knowledge from similar environments"""

        best_match = None

        best_score = 0

        

        for knowledge_file in self.knowledge_dir.glob("*_knowledge.json"):

            try:

                with open(knowledge_file, 'r') as f:

                    knowledge = json.load(f)

                

                # Calculate similarity score

                score = self._calculate_similarity(current_obs, knowledge)

                

                if score > best_score:

                    best_score = score

                    best_match = knowledge

            except Exception as e:

                print(f"Warning: Could not load {knowledge_file}: {e}")

                continue

        

        return best_match if best_score > 0.5 else None

    

    def _calculate_similarity(self, obs: str, knowledge: Dict) -> float:

        """Calculate similarity between current observation and saved knowledge"""

        score = 0.0

        obs_lower = obs.lower() if obs else ""

        

        # Check for similar vocabulary

        if 'state_space' in knowledge and knowledge['state_space']:

            locations = knowledge['state_space'].get('locations', [])

            if locations:

                # Count matching locations

                matches = 0

                for loc in locations:

                    if loc and str(loc).lower() in obs_lower:

                        matches += 1

                if locations:

                    score += min(matches / max(len(locations), 1), 0.5)

        

        # Check for similar task patterns

        task_indicators = ['your task is to:', 'goal:', 'objective:', 'mission:', 

                          'quest:', 'complete:', 'achieve:', 'must:']

        has_task = any(indicator in obs_lower for indicator in task_indicators)

        

        if has_task and knowledge.get('help_info'):

            score += 0.3

        

        # Check for similar object types

        if 'state_space' in knowledge and knowledge['state_space']:

            objects = knowledge['state_space'].get('objects', [])

            if objects:

                obj_matches = 0

                for obj in objects:

                    if obj and str(obj).lower() in obs_lower:

                        obj_matches += 1

                if objects:

                    score += min(obj_matches / max(len(objects), 1), 0.2)

        

        # Check for similar action space size (indicates similar complexity)

        if 'action_space' in knowledge and knowledge['action_space']:

            if isinstance(knowledge['action_space'], dict):

                num_actions = len(knowledge['action_space'])

                # Environments with similar numbers of actions might be similar

                if 10 <= num_actions <= 20:  # Medium complexity

                    score += 0.1

                elif num_actions > 20:  # High complexity

                    score += 0.05

        

        return score

    

    def get_universal_bootstrap_actions(self) -> List[str]:

        """Get universal actions that work across many environments"""

        # Load saved meta patterns

        self._load_meta_patterns()

        

        # Start with truly universal commands that work in most environments

        universal = [

            'help', '?', 'look', 'examine', 'inventory', 'status',

            'go', 'take', 'put', 'use', 'open', 'close', 'move',

            'get', 'drop', 'interact', 'inspect', 'check'

        ]

        

        # Add common action verbs from meta patterns

        if self.meta_patterns['common_action_verbs']:

            # Sort by frequency if we've been tracking it

            common_verbs = list(self.meta_patterns['common_action_verbs'])[:10]

            for verb in common_verbs:

                if verb not in universal:

                    universal.append(verb)

        

        return list(dict.fromkeys(universal))  # Remove duplicates while preserving order

    

    def _update_meta_patterns(self, knowledge: Dict):

        """Update meta patterns from discovered knowledge"""

        if not knowledge:

            return

            

        # Extract action verbs from action_space

        if 'action_space' in knowledge and knowledge['action_space']:

            if isinstance(knowledge['action_space'], dict):

                for action in knowledge['action_space']:

                    if action and isinstance(action, str):

                        verb = action.split()[0]

                        self.meta_patterns['common_action_verbs'].add(verb)

        

        # Extract action verbs from action_effects

        if 'action_effects' in knowledge and knowledge['action_effects']:

            if isinstance(knowledge['action_effects'], dict):

                for action in knowledge['action_effects']:

                    if action and isinstance(action, str):

                        verb = action.split()[0]

                        self.meta_patterns['common_action_verbs'].add(verb)

        

        # Extract common formats

        if 'command_format' in knowledge and knowledge['command_format']:

            if isinstance(knowledge['command_format'], dict):

                fmt = knowledge['command_format']

                

                # Safely check for common_structures

                if fmt.get('common_structures'):

                    for structure in fmt['common_structures']:

                        if structure and structure not in self.meta_patterns['common_formats']:

                            self.meta_patterns['common_formats'].append(structure)

                

                # Track word count patterns

                if fmt.get('word_counts'):

                    # This helps understand command complexity across environments

                    pass  # Could track this for meta-analysis

        

        # Extract universal patterns from constraints

        if 'constraints' in knowledge and knowledge['constraints']:

            for constraint in knowledge['constraints']:

                if constraint and isinstance(constraint, str):

                    # Look for patterns that might apply across environments

                    if 'format' in constraint.lower() or 'word' in constraint.lower():

                        if constraint not in self.meta_patterns['universal_patterns']:

                            self.meta_patterns['universal_patterns'].append(constraint)

    

    def _save_meta_patterns(self):

        """Save meta patterns to disk"""

        try:

            meta_file = self.knowledge_dir / "meta_patterns.json"

            serializable = self._make_serializable(self.meta_patterns)

            with open(meta_file, 'w') as f:

                json.dump(serializable, f, indent=2)

        except Exception as e:

            print(f"Warning: Could not save meta patterns: {e}")

    

    def _load_meta_patterns(self):

        """Load meta patterns from disk"""

        meta_file = self.knowledge_dir / "meta_patterns.json"

        if meta_file.exists():

            try:

                with open(meta_file, 'r') as f:

                    loaded = json.load(f)

                

                # Convert lists back to sets where appropriate

                if 'common_action_verbs' in loaded and isinstance(loaded['common_action_verbs'], list):

                    loaded['common_action_verbs'] = set(loaded['common_action_verbs'])

                

                # Update patterns

                self.meta_patterns.update(loaded)

            except Exception as e:

                print(f"Warning: Could not load meta patterns: {e}")

    

    def _make_serializable(self, obj):

        """Convert sets and other non-serializable objects for JSON"""

        if obj is None:

            return None

        elif isinstance(obj, set):

            return list(obj)

        elif isinstance(obj, dict):

            return {k: self._make_serializable(v) for k, v in obj.items()}

        elif isinstance(obj, list):

            return [self._make_serializable(item) for item in obj]

        elif isinstance(obj, (str, int, float, bool)):

            return obj

        else:

            # For any other type, convert to string

            return str(obj)

    

    def get_discovery_summary(self) -> str:

        """Get summary of all discovered environments"""

        self._load_meta_patterns()

        

        summary = "=== META DISCOVERY SUMMARY ===\n\n"

        

        # Count environments

        env_files = list(self.knowledge_dir.glob("*_knowledge.json"))

        env_count = len([f for f in env_files if f.name != "meta_patterns.json"])

        summary += f"Environments discovered: {env_count}\n"

        

        # Common action verbs

        if self.meta_patterns['common_action_verbs']:

            verbs = list(self.meta_patterns['common_action_verbs'])[:15]

            summary += f"\nMost common action verbs: {', '.join(verbs)}\n"

        

        # Common formats

        if self.meta_patterns['common_formats']:

            summary += f"\nCommon command formats:\n"

            for fmt in self.meta_patterns['common_formats'][:10]:

                summary += f"  - {fmt}\n"

        

        # Universal patterns

        if self.meta_patterns['universal_patterns']:

            summary += f"\nUniversal patterns discovered:\n"

            for pattern in self.meta_patterns['universal_patterns'][:5]:

                summary += f"  - {pattern}\n"

        

        return summary

    

    def suggest_exploration_strategy(self, failed_actions: List[str] = None) -> List[str]:

        """Suggest exploration strategies based on meta-knowledge"""

        suggestions = []

        

        # Load current knowledge

        self._load_meta_patterns()

        

        # Base suggestions

        suggestions.append("Try 'help' or '?' first - works in most environments")

        

        # If we have common formats

        if self.meta_patterns['common_formats']:

            suggestions.append(f"Common formats: {', '.join(self.meta_patterns['common_formats'][:3])}")

        

        # If certain actions have failed

        if failed_actions:

            failed_verbs = set(action.split()[0] for action in failed_actions if action)

            

            # Suggest alternatives

            alternatives = []

            for verb in self.meta_patterns['common_action_verbs']:

                if verb not in failed_verbs:

                    alternatives.append(verb)

            

            if alternatives:

                suggestions.append(f"Try these verbs: {', '.join(alternatives[:5])}")

        

        return suggestions