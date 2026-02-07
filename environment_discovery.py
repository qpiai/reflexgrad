"""Simplified Universal Environment Discovery - Just get action space"""



from typing import List, Dict, Any, Tuple

import re

from collections import defaultdict



class UniversalEnvironmentDiscovery:

    def __init__(self):

        self.discovery_results = {

            'action_space': {},

            'raw_actions': [],

            'uses_numbered_items': False,

            'action_patterns': defaultdict(list),

            'successful_formats': [],

            'failed_formats': []

        }

    

    def discover_environment(self, env, initial_observation: str, max_discovery_steps: int = 50):

        """Universal discovery with multiple strategies"""

        print("=== UNIVERSAL ENVIRONMENT DISCOVERY ===\n")

        

        discovered_actions = set()

        successful_patterns = []

        

        # Strategy 1: Direct action space from environment

        try:

            actions = env.get_action_space()

            if actions:

                discovered_actions.update(actions)

                print(f"âœ“ Found {len(actions)} actions from environment")

        except Exception as e:

            print(f"âš  Could not get action space directly: {e}")

        

        # Strategy 2: Help command exploration

        help_responses = self._explore_help_commands(env, discovered_actions)

        

        # Strategy 3: Systematic exploration through trial and error

        print("\n  Exploring environment to discover action patterns...")

        exploration_results = self._systematic_exploration(env, initial_observation, 

                                                         discovered_actions, max_discovery_steps)

        

        # Strategy 4: Learn from successful interactions

        successful_patterns.extend(exploration_results['successful_patterns'])

        

        # Parse initial observation for available actions

        if initial_observation:

            obs_actions = self._extract_available_actions_from_text(initial_observation)

            discovered_actions.update(obs_actions)

        

        # Analyze patterns to understand action format

        self._analyze_action_patterns(successful_patterns, exploration_results['failed_patterns'])

        

        # Convert discovered actions to results format

        self.discovery_results['raw_actions'] = list(discovered_actions)

        

        # Store in action_space format with learned patterns

        for action in discovered_actions:

            self.discovery_results['action_space'][action] = {

                'description': 'Discovered action',

                'format': self._get_best_format_for_action(action),

                'success_rate': self._calculate_success_rate(action, exploration_results)

            }

        

        # Check for numbered items pattern

        numbered_items = re.findall(r'(\w+)\s+(\d+)', initial_observation)

        if len(numbered_items) > 2:

            self.discovery_results['uses_numbered_items'] = True

            print(f"âœ“ Detected numbered item pattern in observation")

        

        # Set flag for continuous discovery

        self.discovery_results['supports_continuous_discovery'] = True

        

        print(f"\nâœ“ Total discovered actions: {len(discovered_actions)}")

        print(f"âœ“ Successful patterns found: {len(successful_patterns)}")

        

        return self.discovery_results

    

    def _explore_help_commands(self, env, discovered_actions: set) -> List[Dict]:

        """Explore various help commands"""

        help_responses = []

        help_variants = ['help', '?', 'h', 'commands', 'actions', 'verbs', 'info']

        

        for help_cmd in help_variants:

            if help_cmd in discovered_actions or not discovered_actions:

                try:

                    obs, _, _, _ = env.step([help_cmd])

                    if len(obs) > 20:  # Got meaningful response

                        help_responses.append({'command': help_cmd, 'response': obs})

                        

                        # Parse help output for action patterns

                        lines = obs.split('\n')

                        for line in lines:

                            # Look for command patterns

                            if ':' in line:

                                cmd_part = line.split(':')[0].strip()

                                if cmd_part and len(cmd_part) < 100:

                                    discovered_actions.add(cmd_part)

                            

                            # Look for action lists

                            action_words = re.findall(r'\b(go|take|put|open|close|use|examine|look|get|drop)\b', 

                                                    line.lower())

                            discovered_actions.update(action_words)

                        

                        print(f"âœ“ Found additional actions from '{help_cmd}' command")

                        env.reset()

                        break

                except:

                    pass

        

        return help_responses

    

    def _systematic_exploration(self, env, initial_obs: str, known_actions: set, 

                              max_steps: int) -> Dict[str, List]:

        """Systematically explore action combinations"""

        results = {

            'successful_patterns': [],

            'failed_patterns': [],

            'state_changes': []

        }

        

        # Extract entities from observation

        entities = self._extract_entities(initial_obs)

        locations = self._extract_locations(initial_obs)

        

        # Base action words to try

        action_verbs = ['look', 'go', 'take', 'put', 'open', 'close', 'examine', 

                       'use', 'get', 'drop', 'move', 'inventory']

        action_verbs.extend(list(known_actions)[:10])  # Add some known actions

        

        tested_combinations = set()

        steps = 0

        

        # Try different combination patterns

        for verb in action_verbs:

            if steps >= max_steps:

                break

                

            # Pattern 1: Verb only

            if verb not in tested_combinations:

                tested_combinations.add(verb)

                success, obs = self._test_action(env, verb)

                if success:

                    results['successful_patterns'].append({'action': verb, 'pattern': 'VERB'})

                else:

                    results['failed_patterns'].append({'action': verb, 'pattern': 'VERB'})

                steps += 1

            

            # Pattern 2: Verb + Entity

            for entity in entities[:5]:

                if steps >= max_steps:

                    break

                action = f"{verb} {entity}"

                if action not in tested_combinations:

                    tested_combinations.add(action)

                    success, obs = self._test_action(env, action)

                    if success:

                        results['successful_patterns'].append({'action': action, 'pattern': 'VERB OBJ'})

                    else:

                        results['failed_patterns'].append({'action': action, 'pattern': 'VERB OBJ'})

                    steps += 1

            

            # Pattern 3: Verb + Entity + Preposition + Location

            for entity in entities[:3]:

                for prep in ['from', 'to', 'in', 'on']:

                    for location in locations[:3]:

                        if steps >= max_steps:

                            break

                        action = f"{verb} {entity} {prep} {location}"

                        if action not in tested_combinations:

                            tested_combinations.add(action)

                            success, obs = self._test_action(env, action)

                            if success:

                                results['successful_patterns'].append({

                                    'action': action, 

                                    'pattern': f'VERB OBJ {prep.upper()} LOC'

                                })

                            else:

                                results['failed_patterns'].append({

                                    'action': action,

                                    'pattern': f'VERB OBJ {prep.upper()} LOC'

                                })

                            steps += 1

        

        # Reset environment after exploration

        env.reset()

        

        return results

    

    def _test_action(self, env, action: str) -> Tuple[bool, str]:

        """Test an action and determine if it was successful"""

        try:

            prev_obs = str(env) if hasattr(env, '__str__') else ""

            obs, _, _, _ = env.step([action])

            obs_str = obs[0] if isinstance(obs, list) else str(obs)

            

            # Action is successful if it doesn't produce common failure responses

            failure_indicators = [

                "nothing happens",

                "don't understand",

                "what?",

                "huh?",

                "pardon?",

                "invalid",

                "error",

                "can't"

            ]

            

            is_failure = any(indicator in obs_str.lower() for indicator in failure_indicators)

            is_success = not is_failure and obs_str != prev_obs

            

            return is_success, obs_str

        except:

            return False, ""

    

    def _extract_entities(self, text: str) -> List[str]:

        """Extract potential entities from observation"""

        entities = []

        

        # Pattern 1: Things with numbers (e.g., "apple 1", "cabinet 3")

        numbered = re.findall(r'(\w+)\s+(\d+)', text)

        entities.extend([f"{item} {num}" for item, num in numbered])

        

        # Pattern 2: Things after articles

        articles = re.findall(r'\b(?:a|an|the)\s+(\w+)', text.lower())

        entities.extend(articles)

        

        # Pattern 3: Quoted items

        quoted = re.findall(r'"([^"]+)"', text)

        entities.extend(quoted)

        

        # Deduplicate while preserving order

        seen = set()

        unique_entities = []

        for entity in entities:

            if entity not in seen:

                seen.add(entity)

                unique_entities.append(entity)

        

        return unique_entities

    

    def _extract_locations(self, text: str) -> List[str]:

        """Extract potential locations from observation"""

        locations = []

        

        # Common location indicators

        location_patterns = [

            r'(?:on|in|at|near)\s+(?:the\s+)?(\w+\s*\d*)',

            r'(countertop|cabinet|drawer|shelf|table|fridge|sink|stove)\s*\d*',

            r'go to (\w+\s*\d*)'

        ]

        

        for pattern in location_patterns:

            matches = re.findall(pattern, text.lower())

            locations.extend(matches)

        

        # Also use numbered items as potential locations

        numbered = re.findall(r'(\w+\s+\d+)', text)

        locations.extend(numbered)

        

        # Deduplicate

        return list(dict.fromkeys(locations))

    

    def _analyze_action_patterns(self, successful_patterns: List[Dict], 

                               failed_patterns: List[Dict]):

        """Analyze patterns to understand action format"""

        # Count pattern frequencies

        success_pattern_counts = defaultdict(int)

        fail_pattern_counts = defaultdict(int)

        

        for pattern_info in successful_patterns:

            pattern = pattern_info.get('pattern', 'UNKNOWN')

            success_pattern_counts[pattern] += 1

            self.discovery_results['action_patterns'][pattern].append(pattern_info['action'])

        

        for pattern_info in failed_patterns:

            pattern = pattern_info.get('pattern', 'UNKNOWN')

            fail_pattern_counts[pattern] += 1

        

        # Identify most successful patterns

        if success_pattern_counts:

            most_successful = sorted(success_pattern_counts.items(), 

                                   key=lambda x: x[1], reverse=True)

            print(f"\nâœ“ Most successful action patterns:")

            for pattern, count in most_successful[:3]:

                success_rate = count / (count + fail_pattern_counts.get(pattern, 0))

                print(f"  - {pattern}: {count} successes ({success_rate:.0%} success rate)")

                self.discovery_results['successful_formats'].append(pattern)

    

    def _get_best_format_for_action(self, action: str) -> str:

        """Get the best format for an action based on learned patterns"""

        # Check if this action appeared in successful patterns

        for pattern, actions in self.discovery_results['action_patterns'].items():

            if action in actions:

                return pattern

        

        # Default format based on word count

        word_count = len(action.split())

        if word_count == 1:

            return "VERB"

        elif word_count == 2:

            return "VERB OBJ"

        elif word_count >= 4:

            return "VERB OBJ PREP LOC"

        else:

            return "VERB OBJ OBJ"

    

    def _calculate_success_rate(self, action: str, exploration_results: Dict) -> float:

        """Calculate success rate for an action"""

        successes = sum(1 for p in exploration_results['successful_patterns'] 

                       if p['action'] == action)

        failures = sum(1 for p in exploration_results['failed_patterns'] 

                      if p['action'] == action)

        

        total = successes + failures

        if total == 0:

            return 0.5  # Unknown

        

        return successes / total

    

    def _extract_available_actions_from_text(self, text: str) -> set:

        """Extract actions from observation text - universal patterns"""

        actions = set()

        

        # Pattern 1: "Available actions: x, y, z"

        patterns = [

            r'[Aa]vailable actions?:\s*([^\n]+)',

            r'[Yy]ou can:\s*([^\n]+)',

            r'[Cc]ommands?:\s*([^\n]+)',

            r'[Vv]alid actions?:\s*([^\n]+)'

        ]

        

        for pattern in patterns:

            matches = re.findall(pattern, text)

            for match in matches:

                # Split by commas, semicolons, or 'and'

                action_list = re.split(r'[,;]\s*|\s+and\s+', match)

                for action in action_list:

                    action = action.strip().rstrip('.')

                    if action and len(action) < 50:

                        actions.add(action)

        

        return actions

    

    def generate_discovery_report(self) -> str:

        """Generate simple discovery report"""

        report = "=== ENVIRONMENT DISCOVERY REPORT ===\n\n"

        report += f"DISCOVERED ACTIONS: {len(self.discovery_results['action_space'])} found\n"

        

        # Show actions grouped by pattern

        if self.discovery_results['action_patterns']:

            report += "\nACTION PATTERNS:\n"

            for pattern, actions in self.discovery_results['action_patterns'].items():

                if actions:

                    report += f"\n{pattern}:\n"

                    for action in actions[:5]:

                        report += f"  - {action}\n"

                    if len(actions) > 5:

                        report += f"  ... and {len(actions) - 5} more\n"

        

        # Show most successful formats

        if self.discovery_results['successful_formats']:

            report += f"\nMOST SUCCESSFUL FORMATS: {', '.join(self.discovery_results['successful_formats'][:3])}\n"

        

        if self.discovery_results['uses_numbered_items']:

            report += "\nâœ“ Environment uses numbered items\n"

        

        return report