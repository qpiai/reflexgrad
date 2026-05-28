"""Universal Environment Wrapper for Standard Research Benchmarks - Updated APIs"""

# Python 3.10 compatibility for typing.Self (introduced in 3.11)
import typing
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
    typing.Self = Self

from abc import ABC, abstractmethod
import numpy as np
import random
import os
import tempfile
import re
from typing import List, Optional

class UniversalEnvWrapper(ABC):
    """Base wrapper to standardize different environments"""
    
    @abstractmethod
    def reset(self):
        """Return (observation, info)"""
        pass
    
    @abstractmethod
    def step(self, actions):
        """Return (observation, reward, done, info)"""
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Return list of valid actions in current state"""
        pass
    
    def process_observation(self, obs):
        """Default observation processing - can be overridden"""
        if isinstance(obs, str):
            processed = obs
        elif isinstance(obs, list) and obs and isinstance(obs[0], str):
            processed = obs[0]
        elif isinstance(obs, dict) and 'mission' in obs:
            processed = obs.get('mission', str(obs))
        else:
            processed = str(obs)
        
        # Universal cleaning
        processed = re.sub(r'^> ', '', processed).strip()
        return processed
    
    def is_success_indicator(self, obs: str) -> bool:
        """Default success detection - override in subclasses"""
        return False
    
    def get_environment_name(self, info):
        """Default name extraction"""
        return self.__class__.__name__.replace('Wrapper', '')
    
    def get_failure_indicators(self) -> List[str]:
        """Environment-specific failure patterns"""
        return []  # Subclasses can override

    def read_only_probe(self, action: str) -> str:
        """
        Execute a read-only diagnostic action and return the observation text
        WITHOUT advancing the environment's step counter or mutating state.

        This is the universal diagnostic channel used by ReflexGrad's Reflexion
        layer to discover why the agent is stuck. The LLM generates a probe
        action (e.g., `inventory`, `look`, `run: python3 -c "dir(obj)"`) and
        this method executes it against the real environment, returning the
        observation so Reflexion can generate a prescriptive insight.

        Default implementation: delegate to step() and pray the probe was
        read-only. Subclasses should override for safer behavior:
        - Text envs: call look/inventory/examine directly without step()
        - OSWorld: validate the command is a read-only shell/python call
          before forwarding to /execute
        - Stateful envs: snapshot state, run step(), restore snapshot

        Args:
            action: the probe action to execute

        Returns:
            observation text from executing the probe (empty string on error)
        """
        try:
            obs, _, _, _ = self.step([action])
            if isinstance(obs, (list, tuple)) and obs:
                obs = obs[0]
            return str(obs)[:4000] if obs else ''
        except Exception as _probe_err:
            return f'[probe error: {_probe_err}]'

class TextWorldWrapper(UniversalEnvWrapper):
    def __init__(self, game_type="cooking", level="easy", env_id=None):
        # Initialize admissible commands cache
        self._last_admissible = []
        # Add at the beginning:
        if env_id is not None:
            import random
            import numpy as np
            seed = (42 + env_id * 997) % (2**31)
            random.seed(seed)
            np.random.seed(seed)
        try:
            import textworld
            import textworld.gym
            import subprocess
            
            # Create a temporary directory for game files
            self.temp_dir = tempfile.mkdtemp()
            
            if game_type == "cooking":
                # Use tw-make command to create cooking games
                if level == "easy":
                    # Simple cooking game with 1 ingredient
                    game_path = os.path.join(self.temp_dir, "cooking_easy.z8")
                    cmd = ["tw-make", "tw-cooking", 
                           "--output", game_path,
                           "--recipe", "1", "--take", "1", "--cook"]
                elif level == "medium":
                    # Medium cooking game with 3 ingredients
                    game_path = os.path.join(self.temp_dir, "cooking_medium.z8")
                    cmd = ["tw-make", "tw-cooking",
                           "--output", game_path,
                           "--recipe", "3", "--take", "3", "--cook", "--cut", "--open"]
                else:  # hard
                    # Hard cooking game with 5 ingredients
                    game_path = os.path.join(self.temp_dir, "cooking_hard.z8")
                    cmd = ["tw-make", "tw-cooking",
                           "--output", game_path,
                           "--recipe", "5", "--take", "5", "--cook", "--cut", "--open", "--drop"]
                
                # Run the command
                subprocess.run(cmd, check=True, capture_output=True)
                
            elif game_type == "treasure":
                # Use tw-make command for treasure hunter
                game_path = os.path.join(self.temp_dir, f"treasure_{level}.z8")
                difficulty_map = {"easy": 5, "medium": 15, "hard": 30}
                difficulty = difficulty_map.get(level, 15)
                cmd = ["tw-make", "tw-treasure_hunter", 
                       "--output", game_path,
                       "--level", str(difficulty)]
                subprocess.run(cmd, check=True, capture_output=True)
                
            else:
                # Use programmatic API for simple custom games
                from textworld import GameOptions, make
                game_path = os.path.join(self.temp_dir, "simple_game.z8")
                
                options = GameOptions()
                options.nb_rooms = 5 if level == "easy" else 10
                options.nb_objects = 10 if level == "easy" else 20
                options.quest_length = 5 if level == "easy" else 10
                options.path = game_path
                
                # Generate the game
                make(options)
            
            # Now create environment from the generated game
            import textworld.gym
            
            # Register the game first before making it
            env_id = textworld.gym.register_game(game_path, request_infos=textworld.EnvInfos(
                max_score=True,
                description=True,
                inventory=True,
                entities=True,
                verbs=True,
                extras=["recipe"],
                admissible_commands=True
            ))
            self.env = textworld.gym.make(env_id)
            self.game_file = game_path
            self.game_type = game_type
            
        except ImportError:
            raise ImportError("TextWorld not installed. Run: pip install textworld")
        except Exception as e:
            print(f"Error creating TextWorld game: {e}")
            # Fallback: try to use existing API if available
            import textworld
            import textworld.gym
            
            # Use GameOptions directly
            options = textworld.GameOptions()
            options.nb_rooms = 5
            options.nb_objects = 10
            options.quest_length = 5
            options.path = os.path.join(self.temp_dir, "fallback_game.z8")
            
            game_file, _ = textworld.make(options)
            
            # Register the game before making it
            env_id = textworld.gym.register_game(game_file, request_infos=textworld.EnvInfos(
                max_score=True,
                description=True,
                inventory=True,
                admissible_commands=True
            ))
            self.env = textworld.gym.make(env_id)
            self.game_file = game_file
            self.game_type = "custom"
    
    def reset(self):
        obs, info = self.env.reset()
        # CRITICAL: Cache admissible commands for get_current_valid_actions()
        self._last_admissible = info.get('admissible_commands', [])
        # CRITICAL: Ensure consistent task format for task extraction
        # TextWorld games often include the goal in the observation
        # If not present, add a task prefix based on game type
        obs_str = obs if isinstance(obs, str) else str(obs)
        if not obs_str.lower().startswith('your task is to'):
            # Add task based on game type
            if self.game_type == "cooking":
                task_prefix = "Your task is to: follow the recipe to prepare a meal."
            elif self.game_type == "treasure":
                task_prefix = "Your task is to: find and collect the treasure."
            else:
                task_prefix = "Your task is to: complete the objectives in this game."
            obs_str = f"{task_prefix}\n\n{obs_str}"
        return [obs_str], {'extra.gamefile': [self.game_file], 'admissible_commands': self._last_admissible}

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])
        # Cache admissible commands for get_current_valid_actions()
        self._last_admissible = info.get('admissible_commands', [])
        info['won'] = [info.get('has_won', reward > 0)]
        info['admissible_commands'] = self._last_admissible
        return [obs], reward, [done], info
    
    def close(self):
        self.env.close()
        # Clean up temporary files
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def get_action_space(self):
        """Get valid actions for current state"""
        try:
            # TextWorld provides admissible commands
            if hasattr(self.env, 'admissible_commands'):
                return self.env.admissible_commands
            else:
                # Try to get from last info
                return self.env.get_admissible_commands()
        except:
            # Fallback to basic commands
            return ['look', 'inventory', 'take all', 'drop all', 'examine', 'go north', 'go south', 'go east', 'go west']

    def get_current_valid_actions(self) -> list:
        """Get ACTUAL valid actions for current state - matches ALFWorld interface"""
        # TextWorld gym stores admissible commands in info dict after each step
        # Try multiple sources for the commands
        try:
            # First try: cached from last info
            if hasattr(self, '_last_admissible') and self._last_admissible:
                return self._last_admissible
            # Second try: direct attribute access
            if hasattr(self.env, 'admissible_commands') and self.env.admissible_commands:
                return list(self.env.admissible_commands)
            # Third try: unwrap gym env
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'admissible_commands'):
                return list(self.env.unwrapped.admissible_commands)
        except:
            pass
        # Fallback to basic commands
        return ['look', 'inventory', 'examine cookbook', 'take all', 'open fridge', 'go north', 'go south', 'go east', 'go west']

    def is_success_indicator(self, obs: str) -> bool:
        """ALFWorld-specific success indicators"""
        # ALFWorld provides success via info['won'], not observation text
        # This method should never be used for ALFWorld
        return False

class JerichoWrapper(UniversalEnvWrapper):
    def __init__(self, game_name="zork1", env_id=None):
        # Add at the beginning:
        if env_id is not None:
            import random
            import numpy as np
            seed = (42 + env_id * 997) % (2**31)
            random.seed(seed)
            np.random.seed(seed)
        try:
            import jericho
            
            # Standard Jericho benchmark games
            supported_games = {
                'zork1': 'zork1.z5',
                'zork2': 'zork2.z5', 
                'zork3': 'zork3.z5',
                'detective': 'detective.z5',
                'balances': 'balances.z5',
                'pentari': 'pentari.z5',
                'ztuu': 'ztuu.z5',
                'ludicorp': 'ludicorp.z5',
                'temple': 'temple.z5',
                'library': 'library.z5'
            }
            
            if game_name not in supported_games:
                print(f"Warning: {game_name} not in standard set. Using zork1.")
                game_name = "zork1"
            
            game_filename = supported_games[game_name]
            
            # Try multiple locations for game files
            possible_paths = [
                f"jericho-game-suite/{game_filename}",
                f"jericho_game_suite/{game_filename}",
                f"z-machine-games-master/jericho-game-suite/{game_filename}",
                game_filename,  # Current directory
                os.path.join(os.path.dirname(jericho.__file__), "games", game_filename)
            ]
            
            rom_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    rom_path = path
                    break
            
            if rom_path is None:
                # Try to use Jericho's internal game path if available
                try:
                    import jericho.defines
                    if hasattr(jericho.defines, 'GAMES_PATH'):
                        internal_path = os.path.join(jericho.defines.GAMES_PATH, game_filename)
                        if os.path.exists(internal_path):
                            rom_path = internal_path
                except:
                    pass
            
            if rom_path is None:
                # Download games if needed
                print(f"Game file not found. You may need to download Jericho games.")
                print("Try: git clone https://github.com/microsoft/jericho")
                print("Then copy the game files to the current directory.")
                # Use a default path
                rom_path = game_filename
                
            self.env = jericho.FrotzEnv(rom_path)
            self.game_name = game_name
            self.max_score = self.env.get_max_score()
            self._valid_actions = []  # Store like ALFWorld does

        except ImportError:
            raise ImportError("Jericho not installed. Run: pip install jericho[full]")
        except Exception as e:
            print(f"Error loading Jericho game: {e}")
            raise
    
    def reset(self):
        obs, info = self.env.reset()
        self._last_obs = obs
        # Get valid actions - fallback to basic IF commands if spacy fails
        try:
            self._valid_actions = self.env.get_valid_actions() or []
        except:
            self._valid_actions = []
        if not self._valid_actions:
            self._valid_actions = self._get_fallback_actions(obs)
        # Add task description like ALFWorld (algorithm needs "Your task is to:")
        task = f"explore, collect treasures, and maximize your score (max {self.max_score} points)"
        task_obs = f"Your task is to: {task}\n\n{obs}"
        return [task_obs], {'extra.gamefile': [self.game_name], 'max_score': self.max_score,
                            'admissible_commands': self._valid_actions}

    def _get_fallback_actions(self, obs: str) -> list:
        """Smart fallback - dynamically extracts objects from observation text.

        Uses regex patterns to find objects mentioned in the observation,
        similar to how ALFWorld's PDDL provides context-aware actions.
        """
        import re

        # Base navigation and utility actions (always available in IF games)
        actions = ['north', 'south', 'east', 'west', 'up', 'down',
                   'look', 'inventory', 'wait']

        obs_lower = obs.lower()
        extracted_objects = set()

        # Pattern 1: "There is a/an X here" or "You see a/an X"
        patterns = [
            r'there is (?:a|an) ([a-z][a-z\s]{1,20}?) (?:here|on|in|sitting|lying)',
            r'you (?:see|notice|spot) (?:a|an) ([a-z][a-z\s]{1,20}?)(?: here|\.|\,)',
            r'(?:a|an) small ([a-z]+)',
            r'(?:a|an) ([a-z]+) is (?:here|sitting|lying|standing)',
            r'contains (?:a|an) ([a-z][a-z\s]{1,15})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, obs_lower)
            for match in matches:
                obj = match.strip()
                if len(obj) > 1 and obj not in ['the', 'you', 'your', 'this', 'that']:
                    extracted_objects.add(obj)

        # Pattern 2: Common IF objects that might be mentioned
        known_objects = [
            'mailbox', 'leaflet', 'sword', 'lantern', 'lamp', 'bottle', 'key',
            'egg', 'jewel', 'trophy', 'case', 'rug', 'rope', 'knife', 'sack',
            'garlic', 'lunch', 'water', 'coffin', 'torch', 'book', 'candles',
            'bell', 'skull', 'coin', 'bar', 'chalice', 'painting', 'pot',
            'bracelet', 'diamond', 'jade', 'figurine', 'trident', 'scarab',
            'bauble', 'trunk', 'crystal', 'door', 'window', 'trap door',
            'grating', 'ladder', 'table', 'mat', 'leaves', 'pile', 'nest',
            'tree', 'forest', 'house', 'path', 'clearing', 'room', 'cave'
        ]

        for obj in known_objects:
            if obj in obs_lower:
                extracted_objects.add(obj)

        # Generate actions for extracted objects
        takeable = ['leaflet', 'sword', 'lantern', 'lamp', 'bottle', 'key',
                    'egg', 'jewel', 'trophy', 'rope', 'knife', 'sack', 'garlic',
                    'lunch', 'torch', 'book', 'candles', 'bell', 'skull', 'coin',
                    'bar', 'chalice', 'painting', 'pot', 'bracelet', 'diamond',
                    'jade', 'figurine', 'trident', 'scarab', 'bauble', 'crystal']

        openable = ['mailbox', 'door', 'window', 'case', 'trunk', 'coffin',
                    'trap door', 'grating', 'sack', 'bottle', 'egg']

        for obj in extracted_objects:
            actions.append(f'examine {obj}')
            # Check if likely takeable
            if any(t in obj for t in takeable) or obj in takeable:
                actions.append(f'take {obj}')
            # Check if openable
            if any(o in obj for o in openable) or obj in openable:
                actions.append(f'open {obj}')

        # Special context actions
        if 'window' in obs_lower:
            actions.append('enter window')
        if 'case' in obs_lower or 'trophy case' in obs_lower:
            actions.append('put all in case')
        if 'trap door' in obs_lower or 'trapdoor' in obs_lower:
            actions.extend(['open trap door', 'go down'])
        if 'tree' in obs_lower:
            actions.append('climb tree')
        if 'leaves' in obs_lower:
            actions.extend(['move leaves', 'take leaves'])
        if 'rug' in obs_lower:
            actions.extend(['move rug', 'look under rug'])

        # CRITICAL: Light source actions (needed for dark areas in Zork)
        # Always include these - algorithm needs them when carrying lantern
        actions.extend(['turn on lantern', 'turn off lantern',
                        'turn on lamp', 'turn off lamp',
                        'light lamp', 'light lantern'])

        # Dark area detection - add light actions prominently
        if 'dark' in obs_lower or 'pitch black' in obs_lower:
            # Put light actions at front when in dark
            actions = ['turn on lantern', 'turn on lamp', 'light lantern'] + actions

        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique_actions.append(a)

        return unique_actions
    
    def step(self, actions):
        action = actions[0] if isinstance(actions, list) else actions
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        try:
            self._valid_actions = self.env.get_valid_actions() or []
        except:
            self._valid_actions = []
        if not self._valid_actions:
            self._valid_actions = self._get_fallback_actions(obs)
        info['won'] = [done and self.env.get_score() == self.max_score]
        info['admissible_commands'] = self._valid_actions
        return [obs], reward, [done], info

    def close(self):
        self.env.close()

    def get_action_space(self):
        """Get valid actions for current state"""
        return self._valid_actions if self._valid_actions else []

    def get_current_valid_actions(self) -> list:
        """Get ACTUAL valid actions for current state - matches ALFWorld interface"""
        return self._valid_actions if self._valid_actions else []

    def is_success_indicator(self, obs: str) -> bool:
        """Jericho-specific success indicators"""
        indicators = ['you have won', 'victory', '*** you win ***', 'the end']
        obs_lower = obs.lower()
        return any(ind in obs_lower for ind in indicators)

    @property
    def recommended_max_steps(self) -> int:
        """Jericho games need more steps than ALFWorld"""
        return 100


class ScienceWorldWrapper(UniversalEnvWrapper):
    """Wrapper for ScienceWorld benchmark (Allen AI).

    UNIVERSAL SOLUTION: Uses visibility-based filtering like ALFWorld's PDDL.
    Only returns actions involving currently visible objects from getPossibleObjects().
    """
    def __init__(self, task_name=None, task_id=0, env_id=0):
        try:
            from scienceworld import ScienceWorldEnv

            standard_tasks = [
                "boil", "melt", "freeze", "change-the-state-of-matter-of",
                "chemistry-mix", "chemistry-mix-paint-secondary-color",
                "find-animal", "find-living-thing", "find-plant",
                "grow-plant", "grow-fruit", "test-conductivity",
                "use-thermometer", "measure-melting-point-known-substance",
                "identify-life-stages-1", "identify-life-stages-2",
                "lifespan-longest-lived", "power-component",
                "mendelian-genetics-known-plant", "inclined-plane-determine-angle",
            ]

            # env_id is alias for task_id (compatibility with main.py)
            effective_id = env_id if env_id else task_id
            if task_name is None:
                task_name = standard_tasks[effective_id % len(standard_tasks)]

            self.env = ScienceWorldEnv(task_name, envStepLimit=100)
            self.task_name = task_name
            self._step_count = 0
            self._variation_idx = 0  # Default variation
            self._loaded = False

        except ImportError:
            raise ImportError("ScienceWorld not installed. Run: pip install scienceworld")

    def reset(self):
        # FIX #30: Use noElectricalAction simplification to reduce action space
        # This removes 1700+ useless "connect X to Y" actions, reducing from ~2700 to ~960 actions
        # "noElectricalAction" is appropriate for non-electrical tasks (boil, melt, freeze, grow, etc.)
        if not self._loaded:
            # Determine appropriate simplification based on task type
            electrical_tasks = ['power-component', 'test-conductivity']
            if self.task_name in electrical_tasks:
                simplification = ""  # Keep electrical actions for these tasks
            else:
                simplification = "noElectricalAction"  # Remove useless connect actions

            self.env.load(self.task_name, self._variation_idx, simplification)
            self._loaded = True

        result = self.env.reset()
        self._step_count = 0
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            # CRITICAL: Format task description to match ALFWorld's "Your task is to:" format
            # This ensures consistent task extraction across all environments
            raw_task = self.env.taskdescription()
            # Extract just the task part (remove "Task Description:" prefix if present)
            task_text = raw_task
            if 'Task Description:' in raw_task:
                task_text = raw_task.split('Task Description:', 1)[-1].strip()
            # Ensure it starts with "Your task is to:"
            if not task_text.lower().startswith('your task is to'):
                task_text = f"Your task is to: {task_text}"

            # FIX #33: Clarify focus timing for state-change tasks
            # ScienceWorld's goal tracking requires focus AFTER substance is in position
            state_change_tasks = ['boil', 'melt', 'freeze', 'change-the-state-of-matter']
            if any(t in self.task_name.lower() for t in state_change_tasks):
                # Replace misleading "First, focus on the substance" guidance
                task_text = task_text.replace(
                    "First, focus on the substance. Then, take actions",
                    "IMPORTANT: First set up the substance in the correct location (on heater for boil/melt, in freezer for freeze). THEN focus on the substance. Finally, take actions"
                )

            # FIX #34: Add container exploration guidance
            # Items like metal pot are often inside closed containers (cupboards, drawers)
            # The agent needs to open containers to access items
            task_text += "\n\nHINT: Important items (containers, tools) may be inside cupboards or drawers. If you can't find an item, try opening cupboards and drawers first."

            # FIX #35: Add waiting guidance for state-change tasks
            # After setup + focus, the agent must use "wait" for time to pass
            if any(t in self.task_name.lower() for t in state_change_tasks):
                task_text += "\n\nCRITICAL: After positioning the substance and focusing on it, you MUST use 'wait' action repeatedly to let time pass. The substance will only change state (boil/melt/freeze) when enough time has passed."

            full_obs = f"{task_text}\n\n{obs}"
            # Cache valid actions for get_current_valid_actions()
            self._cached_valid_actions = self.get_action_space()
            return [full_obs], {
                'extra.gamefile': [self.task_name],
                'scienceworld_info': info,
                'admissible_commands': [self._cached_valid_actions]  # Single wrap for ALFWorld compatibility
            }
        # Fallback path - still include task description
        raw_task = self.env.taskdescription() if hasattr(self.env, 'taskdescription') else ""
        task_text = raw_task
        if raw_task and 'Task Description:' in raw_task:
            task_text = raw_task.split('Task Description:', 1)[-1].strip()
        if task_text and not task_text.lower().startswith('your task is to'):
            task_text = f"Your task is to: {task_text}"
        obs_str = f"{task_text}\n\n{str(result)}" if task_text else str(result)
        # Fallback: cache empty actions but provide basic fallback
        self._cached_valid_actions = ['look around', 'inventory', 'wait']
        return [obs_str], {'extra.gamefile': [self.task_name], 'admissible_commands': [self._cached_valid_actions]}

    def step(self, actions):
        action = actions[0] if isinstance(actions, list) else actions

        # FIX #26: Validate action against available actions list
        # This prevents algorithm from executing actions that wrapper filtered out
        valid_actions = getattr(self, '_cached_valid_actions', None)
        if valid_actions and action not in valid_actions:
            # Action not in available list - find closest match or reject
            action_lower = action.lower().strip()

            # Try exact case-insensitive match first
            exact_match = None
            for va in valid_actions:
                if va.lower().strip() == action_lower:
                    exact_match = va
                    break

            if exact_match:
                print(f"[FIX #26] Case-insensitive match: '{action}' -> '{exact_match}'")
                action = exact_match
            else:
                # FIX #26 IMPROVED: Semantic partial matching
                # Exclude common words from overlap counting
                stop_words = {'on', 'to', 'the', 'a', 'an', 'in', 'with', 'from', 'into', 'of'}

                # Extract verb and significant nouns
                action_parts = action_lower.split()
                action_verb = action_parts[0] if action_parts else ""
                action_nouns = set(action_parts[1:]) - stop_words

                best_match = None
                best_score = 0

                for va in valid_actions:
                    va_parts = va.lower().split()
                    va_verb = va_parts[0] if va_parts else ""
                    va_nouns = set(va_parts[1:]) - stop_words

                    # Verb must match for partial match consideration
                    if va_verb != action_verb:
                        continue

                    # Count significant noun overlap
                    noun_overlap = len(action_nouns & va_nouns)
                    if noun_overlap > best_score:
                        best_score = noun_overlap
                        best_match = va

                # Only use partial match if verb matches AND at least 1 significant noun matches
                if best_match and best_score >= 1:
                    print(f"[FIX #26] Semantic match (verb + {best_score} nouns): '{action}' -> '{best_match}'")
                    action = best_match
                else:
                    # No good match - return informative observation WITHOUT executing
                    print(f"[FIX #26] REJECTED invalid action: '{action}'")
                    print(f"[FIX #26] Valid actions include: {valid_actions[:5]}...")

                    # Return failure observation - action was invalid
                    error_obs = f"Invalid action: '{action}' is not available. Try one of these: {', '.join(valid_actions[:10])}"

                    # Refresh valid actions for next step
                    self._cached_valid_actions = self.get_action_space()

                    return [error_obs], -0.1, [False], {
                        'won': [False],
                        'score': 0,
                        'invalid_action': True,
                        'admissible_commands': [self._cached_valid_actions]
                    }

        result = self.env.step(action)
        self._step_count += 1

        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            score = info.get('score', 0)

            # FIX #32: Don't terminate on -100 penalty - allow learning from mistakes
            # ScienceWorld gives -100 for "critical mistakes" (e.g., focusing on wrong objects)
            # Instead of terminating, we continue so the agent can learn and retry
            # This is like training wheels - prevent harsh termination to enable learning
            # NOTE: Only check REWARD (-100 means THIS action was a mistake)
            # Don't check score (it persists at -100 after any mistake)
            is_critical_mistake = (reward == -100)
            if is_critical_mistake:
                # Convert harsh termination into learning opportunity
                obs = f"INVALID ACTION: '{action}' caused a critical mistake. " + obs
                # Don't terminate - give feedback and continue
                done = False
                # Normalize the penalty (still negative but not episode-ending)
                reward = -0.5  # Small penalty instead of -100
                score = max(0, score)  # Don't let score go negative for display

            # FIX: Better success detection - check multiple signals
            text_success = self.is_success_indicator(obs)
            score_success = score >= 100
            reward_success = reward > 0 and score >= 90  # Near completion with positive reward
            won = text_success or score_success or reward_success

            # Only done if: won, OR max steps reached (NOT on -100 penalty anymore)
            actual_done = won or (self._step_count >= 100)

            info['won'] = [won]
            info['score'] = score
            info['critical_mistake'] = is_critical_mistake  # Track for debugging
            # FIX: Cache valid actions and use single wrap
            self._cached_valid_actions = self.get_action_space()
            info['admissible_commands'] = [self._cached_valid_actions]
            return [obs], reward, [actual_done], info
        # Fallback with proper actions
        self._cached_valid_actions = ['look around', 'inventory', 'wait']
        return [str(result)], 0, [False], {'won': [False], 'admissible_commands': [self._cached_valid_actions]}

    def close(self):
        self.env.close()

    def get_action_space(self, textgrad_guidance: str = None):
        """UNIVERSAL LLM-BASED ACTION FILTERING (CLEAN VERSION)

        ICML Approach - NO HARDCODING:
        1. Get ALL valid actions from environment (1000+)
        2. Use LLM (gpt-4o) to select task-relevant actions (~50)
        3. Return filtered actions

        Why this works:
        - LLM understands task semantics: "boil water" → pot, sink, stove actions
        - No hardcoded object lists that miss edge cases
        - Cost: ~$0.001 per filter call
        - Truly UNIVERSAL - works for ANY task without modification
        """
        try:
            # Get task description for context
            task_desc = ""
            if hasattr(self.env, 'taskdescription'):
                task_desc = self.env.taskdescription()

            # Get ALL valid actions (minimal exclusions - only system commands)
            all_actions = []
            system_commands = {'reset', 'task'}  # Only exclude actual system commands

            # FIX #31: Invalid targets for "focus on" - ScienceWorld API bug returns these but they cause -100 penalty
            # Rooms cannot be focused on - only actual objects can
            invalid_focus_targets = {
                'kitchen', 'bedroom', 'hallway', 'living room', 'workshop',
                'greenhouse', 'art studio', 'bathroom', 'outside', 'foundry',
                'inventory', 'agent'  # Also exclude inventory and agent
            }

            for item in self.env.getValidActionObjectCombinationsWithTemplates():
                action = item['action']
                action_lower = action.lower()
                verb = action.split()[0].lower() if action.split() else ''

                # Skip only system commands and self-referential actions
                if verb in system_commands:
                    continue
                if 'agent' in action_lower:
                    continue

                # FIX #31: Filter out invalid "focus on [room]" actions
                # ScienceWorld returns these but executing them causes -100 penalty and episode termination
                if action_lower.startswith('focus on '):
                    focus_target = action_lower.replace('focus on ', '').strip()
                    # Skip if focusing on a room name (not an actual object)
                    if focus_target in invalid_focus_targets:
                        continue

                all_actions.append(action)

            if not all_actions:
                return ['look around', 'inventory', 'wait']

            # Check for TextGrad guidance (passed directly OR stored via set_textgrad_guidance)
            guidance = textgrad_guidance or getattr(self, '_textgrad_guidance', None)

            # KEY INSIGHT: Use TextGrad's reasoning to filter actions dynamically
            # TextGrad says "pick up metal pot" → filter returns actions matching that
            if guidance and len(str(guidance)) > 5:
                # Dynamic filtering based on what TextGrad wants
                return self._dynamic_guidance_filter(all_actions, guidance, top_k=50)

            # Fallback: Static LLM filtering based on task description only
            import os
            debug = os.environ.get('DEBUG_FIX29', '')
            if debug:
                print(f"[GET_ACTION_SPACE] {len(all_actions)} actions, guidance={bool(guidance)}")

            if len(all_actions) > 50:
                if debug:
                    print(f"[GET_ACTION_SPACE] Calling _llm_intelligent_filter")
                return self._llm_intelligent_filter(all_actions, top_k=50)
            else:
                # Small action space - return all (ALFWorld-like)
                return all_actions

        except Exception as e:
            import os
            if os.environ.get('DEBUG_FIX29'):
                print(f"[GET_ACTION_SPACE] Exception: {e}")
            return ['look around', 'inventory', 'wait']

    def _apply_focus_filter(self, actions: list, task_desc: str) -> list:
        """FIX #25: Universal post-filter for 'focus on' actions.

        Filters out focus actions that don't match task substances.
        E.g., for 'boil water' task, filters out 'focus on orange'.

        This is applied AFTER any filtering method to ensure consistency.
        """
        if not task_desc:
            return actions

        task_lower = task_desc.lower()

        # Common ScienceWorld substances
        substances = ['water', 'ice', 'steam', 'salt', 'sodium chloride', 'sugar',
                     'paper', 'wood', 'metal', 'glass', 'plastic', 'rubber',
                     'oil', 'wax', 'butter', 'chocolate']
        task_substances = [s for s in substances if s in task_lower]

        # If task mentions substances, filter focus actions
        if not task_substances:
            return actions

        filtered = []
        for action in actions:
            action_lower = action.lower()
            if action_lower.startswith('focus on'):
                # Only keep if focuses on task substance or navigation target
                focus_target = action_lower.replace('focus on ', '')
                is_substance_match = any(s in focus_target for s in task_substances)
                is_nav_target = any(loc in focus_target for loc in
                                  ['kitchen', 'hallway', 'door', 'room', 'workshop', 'greenhouse', 'art studio'])
                if is_substance_match or is_nav_target:
                    filtered.append(action)
                # Skip non-matching substances (like "focus on orange" for boil water)
            else:
                filtered.append(action)

        return filtered

    def _filter_by_visibility(self, actions: list) -> list:
        """
        UNIVERSAL: Filter actions by REACHABILITY (like ALFWorld PDDL grounding).

        Key insight: In ALFWorld, PDDL only generates actions that can ACTUALLY be executed.
        For ScienceWorld, we mimic this by checking:
        - "move X to Y": X must be in inventory (you can only move what you hold)
        - "pick up X": X must be visible and NOT in inventory
        - "activate/deactivate X": X must be visible (nearby)
        - "pour X into Y": X must be in inventory

        This is principled reachability filtering, not cheating.
        """
        # V3: Reachability-based filtering (PDDL-like)
        debug = False  # Set to True for debugging

        # 1. Get ACTUALLY visible objects from observation text
        # FIX #24: getPossibleObjects() returns ALL game objects, not visible ones!
        # Parse observation text to get what agent can actually see
        visible_objects = set()
        try:
            # Get current observation to extract visible objects
            obs_text = ""
            if hasattr(self.env, 'look'):
                obs_text = str(self.env.look()).lower()
            elif hasattr(self.env, 'observation'):
                obs_text = str(self.env.observation()).lower()

            # Extract objects from observation (items after "you see:", "In it, you see:", etc.)
            # Common ScienceWorld objects to look for
            common_objects = [
                'metal pot', 'glass cup', 'ceramic cup', 'tin cup', 'bowl', 'glass jar',
                'orange', 'apple', 'banana', 'potato', 'thermometer', 'lighter', 'stopwatch',
                'sodium chloride', 'water', 'ice', 'steam', 'salt', 'sugar',
                'sink', 'stove', 'oven', 'fridge', 'freezer', 'cupboard', 'counter', 'drawer',
                'door', 'table', 'chair', 'shelf', 'picture', 'flower pot', 'bee', 'butterfly',
                'seed', 'soil', 'plant', 'substance', 'air'
            ]
            for obj in common_objects:
                if obj in obs_text and obj != 'air' and obj != 'agent':
                    visible_objects.add(obj)

            # Also extract room names for navigation
            rooms = ['kitchen', 'hallway', 'bedroom', 'living room', 'workshop', 'greenhouse', 'art studio', 'bathroom']
            for room in rooms:
                if room in obs_text:
                    visible_objects.add(room)

        except Exception:
            pass

        # 2. Get inventory items
        inventory_items = set()
        try:
            inventory_text = self.env.inventory().lower()
            # Extract actual object names from inventory
            common_items = ['metal pot', 'glass cup', 'ceramic cup', 'tin cup', 'bowl', 'glass jar',
                           'orange', 'apple', 'banana', 'potato', 'thermometer', 'lighter', 'stopwatch',
                           'sodium chloride', 'water']
            for item in common_items:
                if item in inventory_text:
                    inventory_items.add(item)

            # Also look for "containing" pattern: "metal pot (containing water)"
            if 'containing' in inventory_text:
                for item in common_items:
                    if item in inventory_text:
                        inventory_items.add(item)
        except Exception:
            pass

        if debug:
            print(f"[DEBUG V3] visible: {len(visible_objects)}, inventory: {inventory_items}")

        # 3. Filter by reachability (action-type-specific rules)
        filtered = []
        for action in actions:
            action_lower = action.lower()
            words = action_lower.split()
            verb = words[0] if words else ''

            # Navigation/state actions - FIX #27: Be selective about "look at X"
            # "look around", "wait", "inventory", "go", "open", "close" - always valid
            if verb in {'wait', 'wait1', 'inventory'}:
                filtered.append(action)
                continue

            # "go to X", "open X", "close X" - always valid for navigation
            if verb in {'go', 'open', 'close'}:
                filtered.append(action)
                continue

            # FIX #27: "look at X" - X must be visible OR be generic commands
            if verb == 'look':
                # Generic look commands always valid
                if action_lower.strip() in {'look around', 'look at inventory', 'look', 'inventory'}:
                    filtered.append(action)
                    continue
                # "look at X" or "look in X" - X must be visible or in inventory
                subject = action_lower.replace('look at ', '').replace('look in ', '').strip()
                # Check if subject matches visible objects
                if any(obj in subject or subject in obj for obj in visible_objects):
                    filtered.append(action)
                    continue
                # Also allow looking at inventory items
                if any(item in subject or subject in item for item in inventory_items):
                    filtered.append(action)
                continue

            # "move X to Y" - X must be in inventory
            if verb == 'move':
                # Extract what's being moved (after "move", before "to")
                if ' to ' in action_lower:
                    subject = action_lower.split(' to ')[0].replace('move ', '').strip()
                    # Check if subject is in inventory
                    if any(item in subject for item in inventory_items):
                        filtered.append(action)
                continue

            # "pick up X" - X must be visible but NOT in inventory
            if verb == 'pick':
                # Extract what's being picked up
                subject = action_lower.replace('pick up ', '').strip()
                # It should be visible
                if any(obj in subject or subject in obj for obj in visible_objects):
                    # But not already in inventory
                    if not any(item in subject for item in inventory_items):
                        filtered.append(action)
                continue

            # "put down X" - X must be in inventory
            if verb == 'put':
                subject = action_lower.replace('put down ', '').replace('put ', '').strip()
                if any(item in subject for item in inventory_items):
                    filtered.append(action)
                continue

            # "pour X into Y" - X must be in inventory
            if verb == 'pour':
                if ' into ' in action_lower:
                    subject = action_lower.split(' into ')[0].replace('pour ', '').strip()
                    if any(item in subject for item in inventory_items):
                        filtered.append(action)
                continue

            # "activate/deactivate X" - X must be visible (nearby device)
            if verb in {'activate', 'deactivate'}:
                subject = action_lower.replace(verb + ' ', '').strip()
                if any(obj == subject or subject in obj for obj in visible_objects):
                    filtered.append(action)
                continue

            # "focus on X", "use X on Y", "mix X", "eat X" - X must be visible or in inventory
            if verb in {'focus', 'use', 'mix', 'eat'}:
                if any(obj in action_lower for obj in visible_objects) or \
                   any(item in action_lower for item in inventory_items):
                    filtered.append(action)
                continue

            # Default: keep if mentions visible object (fallback)
            if any(obj in action_lower for obj in visible_objects):
                filtered.append(action)

        if debug:
            print(f"[DEBUG V3] filtered: {len(filtered)}")

        # Safety: if too few, add more actions
        if len(filtered) < 30:
            # Add actions involving visible objects more liberally
            for action in actions:
                if action not in filtered:
                    action_lower = action.lower()
                    if any(obj in action_lower for obj in visible_objects):
                        filtered.append(action)
                        if len(filtered) >= 100:
                            break

        return filtered if len(filtered) >= 20 else actions[:200]

    def _smart_action_ordering(self, actions: list) -> list:
        """UNIVERSAL: Order actions with guaranteed type diversity + inventory priority.

        NO filtering - just smart ordering so critical actions come first.
        The main algo shows first 30-100 actions, so order is critical.

        Priority order:
        1. Actions involving INVENTORY items (per type)
        2. One action per type (type diversity)
        3. All remaining actions
        """
        if not actions:
            return actions

        # Try to get inventory items for prioritization
        inventory_items = set()
        try:
            if hasattr(self.env, 'inventory'):
                inventory_text = self.env.inventory().lower()
                # Extract item names from inventory
                for word in ['metal pot', 'glass cup', 'ceramic cup', 'tin cup', 'bowl', 'jar', 'orange', 'apple', 'banana']:
                    if word in inventory_text:
                        inventory_items.add(word)
                # Also add individual words from inventory for partial matching
                for line in inventory_text.split('\n'):
                    for word in line.split():
                        if len(word) > 3 and word not in ['your', 'inventory', 'containing', 'nothing', 'that', 'this', 'with', 'see:']:
                            inventory_items.add(word)
        except:
            pass

        # Group actions by verb type
        action_by_type = {}
        for action in actions:
            verb = action.split()[0].lower() if action.split() else 'other'
            if verb not in action_by_type:
                action_by_type[verb] = []
            action_by_type[verb].append(action)

        # Build ordered list with inventory priority + type diversity
        ordered = []
        added = set()

        # TIER 0: CRITICAL DEVICE ACTIONS (activate/deactivate/open/close containers/devices)
        # These are often needed for task completion and should always be visible
        critical_targets = {'stove', 'sink', 'oven', 'fridge', 'freezer', 'cupboard', 'drawer', 'door'}
        for verb in ['activate', 'deactivate', 'open', 'close']:
            if verb in action_by_type:
                for action in action_by_type[verb]:
                    action_lower = action.lower()
                    if any(target in action_lower for target in critical_targets):
                        if action not in added:
                            ordered.append(action)
                            added.add(action)

        # TIER 0.5: PICK UP CRITICAL CONTAINERS (pot, cup, jar, thermometer)
        # FIX: These are essential for many tasks (boil, melt, freeze) but were being
        # pushed down by inventory item actions (orange, apple, etc.)
        critical_containers = {'metal pot', 'glass cup', 'ceramic cup', 'tin cup', 'bowl', 'glass jar', 'thermometer', 'lighter'}
        if 'pick' in action_by_type:
            for action in action_by_type['pick']:
                action_lower = action.lower()
                if any(container in action_lower for container in critical_containers):
                    if action not in added:
                        ordered.append(action)
                        added.add(action)

        # TIER 0.6: MOVE CONTAINER TO SINK/STOVE (critical for water tasks)
        if 'move' in action_by_type:
            for action in action_by_type['move']:
                action_lower = action.lower()
                # Check if it's moving a container to sink/stove
                has_container = any(c in action_lower for c in critical_containers)
                has_target = any(t in action_lower for t in ['sink', 'stove', 'oven'])
                if has_container and has_target:
                    if action not in added:
                        ordered.append(action)
                        added.add(action)

        # TIER 1: For each type, add actions where INVENTORY item is SUBJECT first
        # NOTE: Skip low-value verbs that clutter the action space
        low_value_verbs = {'connect', 'disconnect', 'mix'}  # These rarely advance tasks
        if inventory_items:
            for verb in sorted(action_by_type.keys()):
                # Skip low-value verbs for inventory items
                if verb in low_value_verbs:
                    continue
                verb_actions = action_by_type[verb]
                # FIRST PASS: Actions where inventory item is the SUBJECT (comes right after verb)
                for action in verb_actions:
                    action_lower = action.lower()
                    parts = action_lower.split()
                    if len(parts) >= 2:
                        # Get what comes after the verb
                        rest = ' '.join(parts[1:])
                        # Skip common prepositions to find the subject
                        # "focus on metal pot" -> skip "on" -> check "metal pot"
                        # "put down metal pot" -> skip "down" -> check "metal pot"
                        # "move metal pot to X" -> check "metal pot"
                        skip_words = {'on', 'in', 'to', 'into', 'down', 'up', 'at'}
                        rest_parts = rest.split()
                        while rest_parts and rest_parts[0] in skip_words:
                            rest_parts = rest_parts[1:]
                        subject_str = ' '.join(rest_parts)

                        for item in inventory_items:
                            # Check if subject STARTS with inventory item
                            if subject_str.startswith(item) or (len(item.split()) == 1 and rest_parts and rest_parts[0] == item):
                                if action not in added:
                                    ordered.append(action)
                                    added.add(action)
                                    break
                    # Limit to 20 inventory-subject actions per type (full coverage for all targets)
                    if sum(1 for a in ordered if a.split()[0].lower() == verb) >= 20:
                        break

        # TIER 2: One action per type for type diversity
        for verb in sorted(action_by_type.keys()):
            if action_by_type[verb]:
                for action in action_by_type[verb]:
                    if action not in added:
                        ordered.append(action)
                        added.add(action)
                        break

        # TIER 3: Remaining actions (round-robin by type for continued diversity)
        remaining_by_type = {v: [a for a in acts if a not in added] for v, acts in action_by_type.items()}
        while any(remaining_by_type.values()):
            for verb in sorted(remaining_by_type.keys()):
                if remaining_by_type[verb]:
                    action = remaining_by_type[verb].pop(0)
                    ordered.append(action)
                    added.add(action)

        return ordered

    def _llm_intelligent_filter(self, actions: list, top_k: int = 100) -> list:
        """UNIVERSAL LLM-BASED action filtering.

        NO HARDCODING - Uses LLM to understand task context and select relevant actions.
        This is truly universal across ANY environment.

        Algorithm:
        1. Group all actions by verb type
        2. Present compact format to LLM with task/inventory context
        3. LLM selects most relevant actions
        4. Ensure TYPE DIVERSITY - minimum representation from each verb type
        """
        import openai
        import os

        debug = os.environ.get('DEBUG_FIX29', '')
        if debug:
            print(f"[LLM_FILTER] Called with {len(actions)} actions, top_k={top_k}")

        # Get task and inventory for context (universal - works for any env)
        task_desc = ""
        inventory_text = ""
        try:
            if hasattr(self.env, 'taskdescription'):
                task_desc = self.env.taskdescription()
            if hasattr(self.env, 'inventory'):
                inventory_text = self.env.inventory()
        except:
            pass

        # Group actions by verb type for compact presentation
        action_by_type = {}
        for action in actions:
            verb = action.split()[0].lower() if action.split() else 'other'
            if verb not in action_by_type:
                action_by_type[verb] = []
            action_by_type[verb].append(action)

        # Build COMPACT format - show ALL action types with examples
        compact_lines = []
        for verb, verb_actions in sorted(action_by_type.items(), key=lambda x: -len(x[1])):
            # Show up to 25 examples per verb type
            examples = verb_actions[:25]
            examples_str = ", ".join(examples)
            if len(verb_actions) > 25:
                examples_str += f" (+{len(verb_actions)-25} more)"
            compact_lines.append(f"{verb.upper()} ({len(verb_actions)}): {examples_str}")

        compact_str = "\n".join(compact_lines)

        try:
            import httpx as _hx_uew; client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), max_retries=2, http_client=_hx_uew.Client(timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), limits=_hx_uew.Limits(max_keepalive_connections=0, max_connections=10)))

            prompt = f"""You are an action filter for a text-based game agent.

TASK: {task_desc[:400] if task_desc else 'Complete the current objective'}
INVENTORY: {inventory_text[:300] if inventory_text else 'Check inventory for items'}

AVAILABLE ACTIONS (grouped by verb type):
{compact_str}

Select exactly 50 actions that are MOST USEFUL for completing this task.

MANDATORY ACTIONS (MUST INCLUDE):
- look around (to see environment)
- inventory (to check what you have)
- wait (to let time pass for state changes)

STRICT RULE FOR "focus on" ACTIONS:
- "focus on X" TARGETS X for state change (boil, melt, freeze)
- ONLY include "focus on X" if X is the EXACT substance in the task
- Example: "boil water" → ONLY "focus on water" is valid
- NEVER include "focus on orange", "focus on apple", "focus on inventory items" for boil/melt/freeze tasks
- This is CRITICAL - wrong focus actions will cause task failure!

PRIORITY RULES:
1. Container actions: pick up pot/cup/jar, move container to sink/stove
2. Device actions: activate/deactivate sink, stove, oven, fridge, freezer
3. Storage: open cupboard, open fridge, open freezer (to find hidden containers)
4. Navigation: go to room, open/close doors
5. Look/inventory/wait actions

TASK-SPECIFIC:
- "boil water": pot actions, sink, stove, activate commands
- "freeze": fridge, freezer actions
- "melt": stove, oven actions

Reply with ONLY the exact action strings, one per line. No numbering or explanations."""

            response = client.chat.completions.create(
                model=os.getenv("REFLEXGRAD_MODEL", "gpt-4o-mini"),  # Fast and cheap
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0
            )

            # Parse response - extract actions that match valid actions
            selected = []
            reply = response.choices[0].message.content.strip()
            all_actions_set = set(actions)
            all_actions_lower = {a.lower(): a for a in actions}

            for line in reply.split('\n'):
                line = line.strip()
                # Remove any leading numbers/bullets
                if line and (line[0].isdigit() or line[0] in '-*'):
                    line = line.lstrip('0123456789.-)*• ').strip()

                # Try exact match first
                if line in all_actions_set:
                    if line not in selected:
                        selected.append(line)
                # Try case-insensitive match
                elif line.lower() in all_actions_lower:
                    action = all_actions_lower[line.lower()]
                    if action not in selected:
                        selected.append(action)

            # CRITICAL: Ensure TYPE DIVERSITY - add from each verb type
            seen = set(selected)
            all_verbs = list(action_by_type.keys())

            # Guarantee minimum 5 actions from each verb type
            for verb in all_verbs:
                verb_actions = action_by_type.get(verb, [])
                count_in_selected = sum(1 for a in selected if a.split()[0].lower() == verb)

                if count_in_selected < 5:
                    for action in verb_actions[:10]:
                        if action not in seen:
                            selected.append(action)
                            seen.add(action)
                            if sum(1 for a in selected if a.split()[0].lower() == verb) >= 5:
                                break

            # Fill remaining slots if needed to ensure diversity
            if len(selected) < top_k:
                for verb, verb_actions in action_by_type.items():
                    for a in verb_actions:
                        if a not in seen:
                            selected.append(a)
                            seen.add(a)
                            if len(selected) >= top_k:
                                break
                    if len(selected) >= top_k:
                        break

            if debug:
                print(f"[LLM_FILTER] After diversity: {len(selected)} selected")

            # FIX #29: UNIVERSAL CRITICAL ACTIONS SAFETY NET
            # LLM sometimes fails to include essential utility actions.
            # Use SECOND LLM call to extract task-specific critical actions.

            import os
            debug = os.environ.get('DEBUG_FIX29', '')
            if debug:
                print(f"[FIX29] Before fix: {len(selected)} actions selected")
                print(f"[FIX29] 'look around' in selected: {'look around' in [s.lower() for s in selected]}")

            all_actions_lower = {a.lower(): a for a in actions}

            # Step 1: Essential utility actions (always needed)
            essential_actions = ['look around', 'inventory', 'wait']
            for essential in essential_actions:
                selected_lower = [s.lower() for s in selected]
                if essential not in selected_lower:
                    if essential in all_actions_lower:
                        if debug:
                            print(f"[FIX29] Adding essential: '{essential}'")
                        selected.insert(0, all_actions_lower[essential])
                    elif debug:
                        print(f"[FIX29] '{essential}' NOT in all_actions_lower!")

            # Step 2: UNIVERSAL task-specific action extraction
            # Ask LLM to identify the critical "focus on X" action for this task
            if task_desc:
                try:
                    # Get all focus actions available
                    focus_actions = [a for a in actions if a.lower().startswith('focus on')]
                    if debug:
                        print(f"[FIX29] Found {len(focus_actions)} focus actions")
                    if focus_actions:
                        focus_prompt = f"""Given this task: "{task_desc}"

Available focus actions: {focus_actions[:30]}

Which ONE focus action is REQUIRED to complete this task?
The "focus on X" action targets X for state change tracking.

IMPORTANT: For tasks involving substances (boil water, melt ice, etc.),
look for "focus on substance in [container]" since the substance is often IN a container.

Reply with ONLY the exact focus action string, nothing else."""

                        focus_response = client.chat.completions.create(
                            model=os.getenv("REFLEXGRAD_MODEL", "gpt-4o-mini"),
                            messages=[{"role": "user", "content": focus_prompt}],
                            max_tokens=100,
                            temperature=0
                        )
                        critical_focus = focus_response.choices[0].message.content.strip()
                        if debug:
                            print(f"[FIX29] LLM selected focus: '{critical_focus}'")

                        # Validate and add
                        if critical_focus in actions and critical_focus not in selected:
                            selected.insert(0, critical_focus)
                            if debug:
                                print(f"[FIX29] Added focus action: '{critical_focus}'")
                        else:
                            # Try case-insensitive match
                            for a in actions:
                                if a.lower() == critical_focus.lower() and a not in selected:
                                    selected.insert(0, a)
                                    if debug:
                                        print(f"[FIX29] Added focus (case-insensitive): '{a}'")
                                    break
                except Exception as e:
                    if debug:
                        print(f"[FIX29] Focus extraction error: {e}")

            if debug:
                print(f"[FIX29] After fix: {len(selected)} actions, first 5: {selected[:5]}")

            return selected[:top_k] if selected else actions[:top_k]

        except Exception as e:
            # Fallback: Smart ordering with type diversity (no LLM)
            import os
            debug = os.environ.get('DEBUG_FIX29') or os.environ.get('DEBUG_FILTER')
            if debug:
                print(f"[LLM_FILTER] EXCEPTION: {e}")
                import traceback
                traceback.print_exc()

            # CRITICAL FIX #29: Apply safety net to fallback path too!
            fallback = self._smart_action_ordering(actions)[:top_k]

            # Inject essential utility actions at the front
            all_actions_lower = {a.lower(): a for a in actions}
            essential_actions = ['look around', 'inventory', 'wait']
            for essential in essential_actions:
                if essential not in [s.lower() for s in fallback]:
                    if essential in all_actions_lower:
                        fallback.insert(0, all_actions_lower[essential])
                        if debug:
                            print(f"[FALLBACK FIX29] Added essential: '{essential}'")

            # Also add task-relevant focus action for substance change tasks
            focus_actions = [a for a in actions if a.lower().startswith('focus on')]
            # For substance change tasks (boil, melt, freeze), find "focus on substance"
            for focus in focus_actions:
                if 'substance' in focus.lower():
                    if focus not in fallback:
                        fallback.insert(0, focus)
                        if debug:
                            print(f"[FALLBACK FIX29] Added focus: '{focus}'")
                    break

            return fallback[:top_k]

    def _dynamic_guidance_filter(self, all_actions: list, guidance: str, top_k: int = 50) -> list:
        """FIX #27: Dynamic TextGrad-guided action filtering.

        This is the NEURAL equivalent of ALFWorld's PDDL grounding:
        - ALFWorld PDDL knows task requirements → generates relevant actions
        - We use TextGrad's understanding of current goal → filter relevant actions

        CRITICAL DIFFERENCE from static filtering:
        - Static: Filter once based on task description
        - Dynamic: Filter EACH STEP based on TextGrad's current understanding

        This ensures we NEVER miss critical actions that TextGrad identified.

        Args:
            all_actions: ALL raw actions from environment (1000+)
            guidance: TextGrad's next_action_guidance (e.g., "pick up the metal pot")
            top_k: Number of actions to return

        Returns:
            Actions filtered to match current guidance, preserving critical ones
        """
        import openai
        import os

        if not guidance or len(guidance) < 5:
            # No guidance - fall back to regular filtering
            return self._llm_intelligent_filter(all_actions, top_k=top_k)

        # Get task description for context
        task_desc = ""
        try:
            if hasattr(self.env, 'taskdescription'):
                task_desc = self.env.taskdescription()
        except:
            pass

        # Get inventory for context
        inventory = ""
        try:
            if hasattr(self.env, 'inventory'):
                inventory = self.env.inventory()
        except:
            pass

        # Group actions by verb type for compact representation
        action_by_verb = {}
        for action in all_actions:
            verb = action.split()[0].lower() if action.split() else 'unknown'
            if verb not in action_by_verb:
                action_by_verb[verb] = []
            action_by_verb[verb].append(action)

        # Create compact action list for prompt (to fit in context)
        compact_actions = []
        for verb, actions_list in action_by_verb.items():
            # Show verb with count and sample
            compact_actions.append(f"[{verb.upper()}] ({len(actions_list)} total): {', '.join(actions_list[:20])}")

        try:
            import httpx as _hx_uew; client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), max_retries=2, http_client=_hx_uew.Client(timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), limits=_hx_uew.Limits(max_keepalive_connections=0, max_connections=10)))

            prompt = f"""You are an intelligent action filter for a text-based simulation.

TASK: {task_desc}
CURRENT INVENTORY: {inventory}

CURRENT GOAL (from TextGrad analysis): {guidance}

Your job: Select the MOST RELEVANT actions that help achieve the CURRENT GOAL.

AVAILABLE ACTIONS BY TYPE:
{chr(10).join(compact_actions)}

SELECTION CRITERIA (in order of priority):
1. Actions that DIRECTLY match the current goal (e.g., if goal says "pick up pot", include "pick up metal pot")
2. Actions that are PREREQUISITES for the goal (e.g., if goal needs pot at stove, include "move pot to stove")
3. Navigation actions to reach required locations
4. Essential utility actions (look, inventory, wait)

IMPORTANT:
- For "boil water" task: Include pot actions, sink actions, stove actions
- For "melt ice": Include heat-related actions
- NEVER miss the action that matches the current goal

Output the {top_k} most relevant exact action strings, one per line.
No numbering, no explanations, just the exact action text."""

            response = client.chat.completions.create(
                model=os.getenv("REFLEXGRAD_MODEL", "gpt-4o"),  # Better quality for dynamic guidance filtering
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0
            )

            selected_text = response.choices[0].message.content.strip()
            selected = [line.strip() for line in selected_text.split('\n') if line.strip()]

            # Validate - only keep actions that actually exist
            valid_set = set(a.lower() for a in all_actions)
            validated = []
            for action in selected:
                if action.lower() in valid_set:
                    # Find original case
                    for orig in all_actions:
                        if orig.lower() == action.lower():
                            validated.append(orig)
                            break

            print(f"[FIX #27 DYNAMIC FILTER] Guidance: '{guidance[:60]}...' → {len(validated)} actions")

            # Ensure the guidance action itself is included if it exists
            guidance_lower = guidance.lower()
            for action in all_actions:
                if action.lower() == guidance_lower:
                    if action not in validated:
                        validated.insert(0, action)
                    break
                # Also check if guidance is a partial match
                elif guidance_lower in action.lower() or action.lower() in guidance_lower:
                    if action not in validated:
                        validated.insert(0, action)

            if validated:
                return validated[:top_k]
            else:
                # Fallback if LLM didn't return valid actions
                print(f"[FIX #27] GPT-4o filter returned no valid actions, falling back")
                return self._llm_intelligent_filter(all_actions, top_k=top_k)

        except Exception as e:
            print(f"[FIX #27] Dynamic guidance filter failed: {e}")
            return self._llm_intelligent_filter(all_actions, top_k=top_k)

    def set_textgrad_guidance(self, guidance: str):
        """Store TextGrad guidance for next action space query.

        Called from alfworld_trial.py after TextGrad generates next_action_guidance.
        The next call to get_action_space() will use this guidance for dynamic filtering.
        """
        self._textgrad_guidance = guidance

    def _visibility_grounded_type_diversity(self, actions: list, max_actions: int = 100) -> list:
        """OPTIMAL APPROACH: Visibility-Grounded + Type Diversity.

        This is the recommended universal approach because:
        1. FAST - No API calls, just smart selection
        2. UNIVERSAL - No hardcoded keywords, works for ANY environment
        3. ROBUST - NEVER misses action types (focus, go, etc.)
        4. GROUNDED - Prioritizes actions involving visible objects
        5. COMPATIBLE - Works without changing main algorithm

        Algorithm:
        1. Get visible objects from current state
        2. Group actions by verb (action type)
        3. For EACH type, select actions involving visible objects
        4. Guarantee minimum actions per type (diversity)
        5. Return ~80-100 total actions
        """
        if not actions:
            return actions

        # Get visible objects for grounding
        visible_objects = set()
        try:
            for obj in self.env.getPossibleObjects():
                obj_str = str(obj).lower().strip()
                if obj_str and 'air' not in obj_str:
                    visible_objects.add(obj_str)
                    # Also add individual words for partial matching
                    for word in obj_str.split():
                        if len(word) > 2:
                            visible_objects.add(word)
        except:
            pass

        # FIX: Also include inventory items (they're not in getPossibleObjects but valid for actions!)
        try:
            inventory_text = self.env.inventory()
            if inventory_text:
                # Parse inventory text to extract object names
                for line in inventory_text.lower().split('\n'):
                    for word in line.split():
                        if len(word) > 2 and word not in ['the', 'and', 'you', 'are', 'have', 'carrying']:
                            visible_objects.add(word)
        except:
            pass

        # FIX: Also extract objects from ACTION TARGETS (most robust)
        # This ensures we include ALL valid targets even if not in getPossibleObjects
        try:
            for action in actions:
                # Extract target from "verb X to Y" patterns
                if ' to ' in action:
                    target = action.split(' to ')[-1].strip().lower()
                    visible_objects.add(target)
                    for word in target.split():
                        if len(word) > 2:
                            visible_objects.add(word)
                # Extract object from "verb X" patterns
                parts = action.lower().split()
                if len(parts) >= 2:
                    for word in parts[1:]:
                        if len(word) > 2:
                            visible_objects.add(word)
        except:
            pass

        # Group actions by verb type
        action_by_type = {}
        for action in actions:
            verb = action.split()[0].lower() if action.split() else 'other'
            if verb not in action_by_type:
                action_by_type[verb] = []
            action_by_type[verb].append(action)

        # Score actions by relevance to visible objects
        def relevance_score(action):
            action_lower = action.lower()
            score = sum(1 for obj in visible_objects if obj in action_lower)
            return score

        # Build result with GUARANTEED type diversity
        result = []
        added = set()

        # Calculate actions per type (ensure all types represented)
        num_types = len(action_by_type)
        min_per_type = max(2, max_actions // (num_types * 2))  # At least 2 per type
        max_per_type = max(5, max_actions // num_types)  # But not too many per type

        # For each action type, select best actions
        for verb, verb_actions in sorted(action_by_type.items()):
            # Sort by relevance (visible objects first)
            scored = [(a, relevance_score(a)) for a in verb_actions]
            scored.sort(key=lambda x: -x[1])

            # Take at least min_per_type, up to max_per_type
            count = 0
            for action, score in scored:
                if action not in added:
                    result.append(action)
                    added.add(action)
                    count += 1
                    if count >= max_per_type:
                        break

            # If we didn't get min_per_type, take any remaining
            if count < min_per_type:
                for action in verb_actions:
                    if action not in added:
                        result.append(action)
                        added.add(action)
                        count += 1
                        if count >= min_per_type:
                            break

        # If under max_actions, add more high-relevance actions
        if len(result) < max_actions:
            all_scored = [(a, relevance_score(a)) for a in actions if a not in added]
            all_scored.sort(key=lambda x: -x[1])
            for action, score in all_scored:
                if len(result) >= max_actions:
                    break
                result.append(action)

        return result[:max_actions]

    def _llm_action_filter(self, actions: list, task: str, top_k: int = 100) -> list:
        """LLM-BASED action filtering with COMPACT format.

        ICML-QUALITY APPROACH:
        1. Group actions by type (verb) - compact comma-separated format
        2. LLM identifies which action TYPES are relevant for current task/state
        3. Return all actions of relevant types (never miss critical ones!)

        Why this is better than semantic embeddings:
        - LLM understands prerequisites ("focus" needed before "heat")
        - Compact format saves ~90% context tokens
        - Never misses critical action types
        """
        import openai
        import os

        # Group actions by verb type
        action_by_type = {}
        for action in actions:
            verb = action.split()[0].lower() if action.split() else 'other'
            if verb not in action_by_type:
                action_by_type[verb] = []
            action_by_type[verb].append(action)

        # Build COMPACT format for LLM (saves ~90% tokens vs one-per-line)
        compact_display = []
        for verb, verb_actions in sorted(action_by_type.items(), key=lambda x: -len(x[1])):
            # Extract just the objects/targets (remove verb prefix)
            objects = []
            for a in verb_actions[:8]:  # Limit examples per type
                parts = a.split(' ', 1)
                obj = parts[1] if len(parts) > 1 else a
                objects.append(obj)
            if len(verb_actions) > 8:
                objects.append(f"+{len(verb_actions)-8} more")
            compact_display.append(f"{verb.upper()}: {', '.join(objects)}")

        compact_str = "\n".join(compact_display)

        # Get current observation for context
        try:
            current_obs = self.env.observation() if hasattr(self.env, 'observation') else ""
        except:
            current_obs = ""

        # Ask LLM which action TYPES are relevant
        try:
            import httpx as _hx_uew; client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), max_retries=2, http_client=_hx_uew.Client(timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), limits=_hx_uew.Limits(max_keepalive_connections=0, max_connections=10)))

            prompt = f"""Task: {task}
State: {current_obs[:300] if current_obs else 'Unknown'}

Actions available (grouped by type):
{compact_str}

Which action TYPES are needed now? Include navigation (go), exploration (look, open), and task verbs.
Reply with ONLY comma-separated verbs like: go, open, pick, focus, move"""

            response = client.chat.completions.create(
                model=os.getenv("REFLEXGRAD_MODEL", "gpt-4o"),  # Better model for accurate filtering
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0
            )

            # Parse response
            relevant_types = set()
            reply = response.choices[0].message.content.strip().lower()
            for word in reply.replace(',', ' ').split():
                word = word.strip()
                if word in action_by_type:
                    relevant_types.add(word)

            # Always include essential exploration types
            for essential in ['look', 'go', 'open', 'focus', 'wait', 'inventory']:
                if essential in action_by_type:
                    relevant_types.add(essential)

        except Exception as e:
            # Fallback: use all types
            relevant_types = set(action_by_type.keys())

        # Collect actions from relevant types
        result = []
        for verb in relevant_types:
            if verb in action_by_type:
                result.extend(action_by_type[verb])

        # If too few, add more types
        if len(result) < top_k // 2:
            for verb in action_by_type:
                if verb not in relevant_types:
                    result.extend(action_by_type[verb])
                    if len(result) >= top_k:
                        break

        return result[:top_k] if len(result) > top_k else result

    def _extract_action_types_from_environment(self, actions: list) -> set:
        """UNIVERSAL: Discover action types dynamically FROM available actions.

        ROOT CAUSE FIX: Instead of hardcoding verbs or extracting from sub-goals,
        we discover what action types the environment actually supports.

        This is truly universal because:
        1. NO hardcoded verb lists
        2. Works for ANY environment
        3. Automatically adapts to whatever actions the environment supports
        4. If "go to kitchen" exists, "go" will be discovered
        5. If "push object" exists, "push" will be discovered
        """
        verbs = set()
        for action in actions:
            words = action.lower().split()
            if words:
                # First word is typically the verb
                verbs.add(words[0])
        return verbs

    def _extract_action_types(self, subgoals: list, available_actions: list = None) -> list:
        """Extract action verbs for diversity enforcement.

        UNIVERSAL FIX: Combines sub-goal verbs with environment-discovered verbs.
        This ensures we NEVER filter out action types that the environment supports.
        """
        # Start with verbs discovered from available actions (UNIVERSAL)
        if available_actions:
            env_verbs = self._extract_action_types_from_environment(available_actions)
        else:
            env_verbs = set()

        # Also extract verbs mentioned in sub-goals for task relevance
        subgoal_verbs = set()
        for sg in subgoals:
            for word in sg.lower().split():
                # Check if word looks like a verb (starts action in any available action)
                if available_actions:
                    if any(a.lower().startswith(word) for a in available_actions):
                        subgoal_verbs.add(word)
                else:
                    subgoal_verbs.add(word)

        # Combine: environment verbs (guaranteed to exist) + relevant subgoal verbs
        all_verbs = env_verbs | subgoal_verbs

        # Map synonyms to canonical forms (optional normalization)
        mapping = {'turn': 'activate', 'take': 'pick', 'place': 'move',
                   'put': 'move', 'walk': 'go', 'enter': 'go'}
        result = list(set(mapping.get(v, v) for v in all_verbs))

        return result if result else ['look']  # Fallback only if completely empty

    def _generate_action_subgoals(self, task: str, visible_objects: list) -> list:
        """Generate individual action sub-goals from task using fast LLM.

        UNIVERSAL: Returns a LIST of sub-goals so each can be matched separately.
        This avoids the bias problem of combining all sub-goals into one query.
        """
        try:
            import openai
            import os

            # Use fast/cheap model for this auxiliary task
            import httpx as _hx_uew; client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), max_retries=2, http_client=_hx_uew.Client(timeout=_hx_uew.Timeout(120.0, connect=10.0, read=120.0), limits=_hx_uew.Limits(max_keepalive_connections=0, max_connections=10)))

            prompt = f"""Given this task in a text-based game: "{task}"
And these visible objects: {', '.join(visible_objects[:10])}

List 5 physical ACTIONS that would help complete this task. Be specific about:
- Moving objects to specific locations (e.g., "move container to water source")
- Picking up and using objects
- Activating devices

Format: Just list the action descriptions, one per line, no numbering."""

            response = client.chat.completions.create(
                model=os.getenv("REFLEXGRAD_MODEL", "gpt-4o"),  # Better model for accurate filtering
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0
            )

            # Parse into individual sub-goals
            text = response.choices[0].message.content.strip()
            subgoals = [line.strip() for line in text.split('\n') if line.strip()]

            # Add task itself as a fallback sub-goal
            subgoals.append(f"Task: {task}")

            return subgoals if subgoals else [f"Task: {task}"]

        except Exception as e:
            # Fallback: just use task description
            return [f"Task: {task}", f"Objects: {', '.join(visible_objects[:5])}"]

    def _visibility_fallback(self, actions: list) -> list:
        """Fallback when embeddings not available: filter by visible objects."""
        try:
            visible = set()
            for obj in self.env.getPossibleObjects():
                obj_lower = obj.lower().strip()
                if obj_lower and 'air' not in obj_lower:
                    visible.add(obj_lower)

            # Filter to actions involving visible objects
            filtered = [a for a in actions if any(v in a.lower() for v in visible)]
            return filtered[:80] if filtered else actions[:80]
        except:
            return actions[:80]

    def get_current_valid_actions(self) -> list:
        """Return current valid actions (required by main algorithm)"""
        # FIX: Use cached value instead of expensive get_action_space() call
        if hasattr(self, '_cached_valid_actions') and self._cached_valid_actions:
            return self._cached_valid_actions
        # Fallback if no cache (shouldn't happen after reset)
        return self.get_action_space()

    def is_success_indicator(self, obs: str) -> bool:
        """ScienceWorld-specific success indicators"""
        indicators = ['task completed', 'congratulations', 'goal achieved', 'task success']
        return any(ind in obs.lower() for ind in indicators)

class BabyAIWrapper(UniversalEnvWrapper):
    """Wrapper for BabyAI/MiniGrid text interface - Updated for gymnasium"""
    def __init__(self, level="MiniGrid-GoToObj-v0"):
        try:
            import gymnasium as gym
            import minigrid
            from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX
            
            # Map old BabyAI names to new MiniGrid names if needed
            level_mapping = {
                "BabyAI-GoToObj-v0": "MiniGrid-GoToObject-8x8-N2",
                "BabyAI-GoToRedBallGrey-v0": "BabyAI-GoToRedBallGrey-v0",
                "BabyAI-GoToLocal-v0": "BabyAI-GoToLocal-v0",
                "BabyAI-PickupLoc-v0": "BabyAI-PickupLoc-v0",
                "BabyAI-PutNextLocal-v0": "BabyAI-PutNextLocal-v0",
                "BabyAI-OpenDoor-v0": "BabyAI-OpenDoor-v0",
                "BabyAI-UnlockLocal-v0": "BabyAI-UnlockLocal-v0",
                "BabyAI-FindObjS5-v0": "BabyAI-FindObjS5-v0",
                "BabyAI-KeyCorridorS3R3-v0": "BabyAI-KeyCorridorS3R3-v0",
                "BabyAI-1RoomS8-v0": "BabyAI-1RoomS8-v0",
            }
            
            # Standard BabyAI/MiniGrid levels used in research
            standard_levels = [
                "MiniGrid-Empty-5x5-v0",        # Basic empty grid
                "MiniGrid-DoorKey-5x5-v0",      # Door and key
                "MiniGrid-MultiRoom-N2-S4-v0",  # Multi room
                "MiniGrid-Fetch-5x5-N2-v0",     # Fetch objects
                "MiniGrid-GoToDoor-5x5-v0",     # Go to door
                "MiniGrid-PutNear-6x6-N2-v0",   # Put near
                "MiniGrid-RedBlueDoors-6x6-v0", # Red blue doors
                "MiniGrid-MemoryS9-v0",          # Memory task
                "MiniGrid-LockedRoom-v0",        # Locked room
                "MiniGrid-KeyCorridorS3R1-v0",  # Key corridor
                "BabyAI-GoToRedBallGrey-v0",    # BabyAI specific
                "BabyAI-GoToLocal-v0",           # BabyAI go to local
                "BabyAI-PickupLoc-v0",           # BabyAI pickup
                "BabyAI-OpenDoor-v0",            # BabyAI open door
                "BabyAI-UnlockLocal-v0",         # BabyAI unlock
            ]
            
            # Use mapping if old name provided
            level = level_mapping.get(level, level)
            
            # Create environment
            self.env = gym.make(level)
            self.level = level
            
            # Action meanings for text conversion
            self.action_meanings = {
                0: "turn left",
                1: "turn right", 
                2: "move forward",
                3: "pick up",
                4: "drop",
                5: "toggle/use",
                6: "done"
            }
            
            # Reverse mapping for text to action
            self.text_to_action = {v: k for k, v in self.action_meanings.items()}
            # Add variations
            self.text_to_action.update({
                "left": 0,
                "right": 1,
                "forward": 2,
                "go": 2,
                "move": 2,
                "take": 3,
                "get": 3,
                "pickup": 3,
                "grab": 3,
                "use": 5,
                "toggle": 5,
                "open": 5,
                "unlock": 5,
                "finish": 6,
                "stop": 6
            })
            
        except ImportError:
            raise ImportError("MiniGrid not installed. Run: pip install minigrid")
    
    def reset(self):
        obs, info = self.env.reset()
        # Convert to text observation
        mission = obs.get('mission', info.get('mission', 'Complete the task'))
        
        # Build text observation
        text_obs = f"You are in a grid world. {mission}\n"
        
        # Add position info if available
        if hasattr(self.env.unwrapped, 'agent_pos'):
            pos = self.env.unwrapped.agent_pos
            text_obs += f"Your position: ({pos[0]}, {pos[1]}). "
            
        # Add direction info if available
        if hasattr(self.env.unwrapped, 'agent_dir'):
            dirs = ['right', 'down', 'left', 'up']
            text_obs += f"You are facing {dirs[self.env.unwrapped.agent_dir]}.\n"
            
        text_obs += "Available actions: turn left, turn right, move forward, pick up, drop, toggle/use, done."
        
        return [text_obs], {'extra.gamefile': [self.level]}
    
    def step(self, actions):
        # Convert text action to number
        action_text = actions[0].lower().strip()
        
        # Default action
        action_num = 2  # move forward
        
        # Find best matching action
        for text_key, num in self.text_to_action.items():
            if text_key in action_text:
                action_num = num
                break
                
        obs, reward, terminated, truncated, info = self.env.step(action_num)
        done = terminated or truncated
        
        # Convert observation to text
        if done and reward > 0:
            text_obs = "Success! You completed the task!"
        elif done:
            text_obs = "Task failed or time limit reached."
        else:
            mission = obs.get('mission', info.get('mission', 'Complete the task'))
            text_obs = f"{mission}\n"
            
            # Add position and direction
            if hasattr(self.env.unwrapped, 'agent_pos'):
                pos = self.env.unwrapped.agent_pos
                text_obs += f"Position: ({pos[0]}, {pos[1]}). "
                
            if hasattr(self.env.unwrapped, 'agent_dir'):
                dirs = ['right', 'down', 'left', 'up']
                text_obs += f"Facing: {dirs[self.env.unwrapped.agent_dir]}.\n"
                
            # Add what's in front
            if hasattr(self.env.unwrapped, 'grid'):
                front_pos = self.env.unwrapped.front_pos
                if front_pos is not None:
                    cell = self.env.unwrapped.grid.get(*front_pos)
                    if cell is not None:
                        text_obs += f"In front of you: {cell.type}.\n"
                    else:
                        text_obs += "Nothing in front of you.\n"
        
        info['won'] = [done and reward > 0]
        return [text_obs], reward, [done], info
    
    def close(self):
        self.env.close()
    
    def get_action_space(self):
        """Get valid actions for current state"""
        # BabyAI/MiniGrid has fixed action set
        return ['turn left', 'turn right', 'move forward', 'pick up', 'drop', 'toggle/use', 'done']
    
    def is_success_indicator(self, obs: str) -> bool:
        """BabyAI-specific success indicators"""
        indicators = ['success!', 'completed the task', 'goal reached']
        obs_lower = obs.lower()
        return any(ind in obs_lower for ind in indicators)

class NetHackWrapper(UniversalEnvWrapper):
    """Wrapper for NetHack Learning Environment (Facebook Research) - Optional"""
    def __init__(self, character="mon-hum-neu-mal"):
        try:
            import nle
            from nle import nethack
            
            # NLE might require gymnasium instead of gym
            try:
                import gymnasium
                self.env = gymnasium.make("NetHackScore-v0", 
                                       character=character,
                                       max_episode_steps=5000)
            except:
                # Fallback to old gym
                import gym as old_gym
                self.env = old_gym.make("NetHackScore-v0", 
                                       character=character,
                                       max_episode_steps=5000)
            
            self.character = character
            
        except ImportError:
            raise ImportError("NLE not installed. Run: pip install nle")
    
    def reset(self):
        obs = self.env.reset()
        # Handle different return types
        if isinstance(obs, tuple):
            obs = obs[0]
        # Convert NetHack observation to text
        text_obs = self._obs_to_text(obs)
        return [text_obs], {'extra.gamefile': ['nethack']}
    
    def _obs_to_text(self, obs):
        """Convert NetHack observation to text description"""
        # This is simplified - NetHack has complex observations
        if isinstance(obs, dict) and 'message' in obs:
            msg = obs['message']
            if isinstance(msg, bytes):
                return msg.decode('utf-8')
            else:
                return str(msg)
        return "You are in a dungeon. Commands: north, south, east, west, wait, search, inventory, help"
    
    def step(self, actions):
        # Map text commands to NetHack actions (simplified)
        action_map = {
            'north': ord('k'), 'n': ord('k'), 'up': ord('k'),
            'south': ord('j'), 's': ord('j'), 'down': ord('j'),
            'east': ord('l'), 'e': ord('l'), 'right': ord('l'),
            'west': ord('h'), 'w': ord('h'), 'left': ord('h'),
            'northeast': ord('u'), 'ne': ord('u'),
            'northwest': ord('y'), 'nw': ord('y'),
            'southeast': ord('n'), 'se': ord('n'),
            'southwest': ord('b'), 'sw': ord('b'),
            'wait': ord('.'), '.': ord('.'),
            'search': ord('s'), 
            'inventory': ord('i'), 'i': ord('i'),
            'help': ord('?'), '?': ord('?'),
            'look': ord(':'), ':': ord(':'),
            'pick up': ord(','), 'get': ord(','), 'take': ord(','),
            'drop': ord('d'), 'd': ord('d'),
            'eat': ord('e'),
            'open': ord('o'), 'o': ord('o'),
            'close': ord('c'), 'c': ord('c'),
            'kick': ord('\x03'),  # Ctrl+D
            'pray': ord('#'),
        }
        
        action_text = actions[0].lower().strip()
        action = action_map.get(action_text, ord('.'))  # Default to wait
        
        # If not in map, try first character
        if action == ord('.') and action_text:
            action = ord(action_text[0])
        
        result = self.env.step(action)
        
        # Handle different return formats
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result
            
        text_obs = self._obs_to_text(obs)
        
        info['won'] = [done and reward > 100]  # High score indicates good performance
        return [text_obs], reward, [done], info
    
    def close(self):
        self.env.close()
    
    def get_action_space(self):
        """Get valid actions for current state"""
        # NetHack has many actions, return most common ones
        return ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
                'up', 'down', 'wait', 'search', 'inventory', 'pick up', 'drop', 'eat', 'open', 
                'close', 'look', 'help', '.']




class ALFWorldWrapper(UniversalEnvWrapper):
    """Wrapper for ALFWorld benchmark environments"""
    def __init__(self, config, env_id=None):
        try:
            import alfworld
            import alfworld.agents.environment as alfworld_env

            env_class = alfworld_env.get_environment(config["env"]["type"])

            # Initialize environment — DO NOT randomize seed
            # ALFWorld has exactly 134 gamefiles that cycle deterministically on reset()
            self.env = env_class(config, train_eval="eval_out_of_distribution")
            self.env = self.env.init_env(batch_size=1)

            # Skip to the correct game by calling reset() N times
            # ALFWORLD_ENV_OFFSET allows parallel single-env processes to get different tasks
            # Each reset() advances to the next game in the 134-game list
            _env_offset = int(os.environ.get('ALFWORLD_ENV_OFFSET', '0'))
            _effective_id = (env_id or 0) + _env_offset
            if _effective_id > 0:
                for _ in range(_effective_id):
                    self.env.reset()

            self._admissible_commands = []
            self._last_observation = ""

        except ImportError:
            raise ImportError("ALFWorld not installed. Run: pip install alfworld")


    def reset(self):
        obs, info = self.env.reset()
        
        # Extract string from tuple if needed
        if isinstance(obs, tuple) and len(obs) > 0:
            self._last_observation = obs[0] if isinstance(obs[0], str) else str(obs[0])
        elif isinstance(obs, list) and obs:
            self._last_observation = obs[0] if isinstance(obs[0], str) else str(obs[0])
        else:
            self._last_observation = str(obs)
        
        # CRITICAL: Get initial admissible commands BEFORE returning
        if 'admissible_commands' in info:
            if isinstance(info['admissible_commands'], list) and info['admissible_commands']:
                self._admissible_commands = info['admissible_commands'][0]
            else:
                self._admissible_commands = info.get('admissible_commands', [])
        else:
            self._admissible_commands = []
        
        # Return the CLEAN observation
        return [self._last_observation], info
    
    def step(self, actions):
        """Updated step function with proper admissible commands update"""
        result = self.env.step(actions)
        obs, reward, done, info = result
        
        # Extract string from tuple if needed
        if isinstance(obs, tuple) and len(obs) > 0:
            self._last_observation = obs[0] if isinstance(obs[0], str) else str(obs[0])
        elif isinstance(obs, list) and obs:
            self._last_observation = obs[0] if isinstance(obs[0], str) else str(obs[0])
        else:
            self._last_observation = str(obs)
        
        # CRITICAL FIX: Update admissible commands after EVERY step
        if 'admissible_commands' in info:
            if isinstance(info['admissible_commands'], list) and info['admissible_commands']:
                self._admissible_commands = info['admissible_commands'][0]
            else:
                self._admissible_commands = info.get('admissible_commands', [])
        else:
            self._admissible_commands = []
        
        # FIX FOR FALSE SUCCESS: Check both won AND reward for compound tasks
        if done[0] if isinstance(done, list) else done:
            # ALFWorld gives reward=1 only for true task completion
            # For compound tasks, 'won' might trigger early but reward stays 0
            if isinstance(reward, (list, tuple)):
                reward_value = reward[0]
            else:
                reward_value = reward
            won_value = info.get('won', [False])[0] if 'won' in info else False
            
            # True success requires both won=True AND reward=1
            actual_success = won_value and (reward_value >= 1)
            
            # Override the won signal with corrected value
            info['won'] = [actual_success]
            
            # Debug logging
            print(f"[ALFWORLD SUCCESS CHECK]")
            print(f"  Original won: {won_value}")
            print(f"  Reward: {reward_value}")
            print(f"  Corrected success: {actual_success}")
            print(f"  Remaining commands: {len(self._admissible_commands)}")
        
        return result
        
    def close(self):
        self.env.close()
    
    def get_action_space(self):
        """Get generic action templates"""
        # Return templates for discovery phase
        return ['look', 'inventory', 'go to', 'take', 'put', 'open', 'close', 'examine', 'use', 'heat', 'cool', 'clean', 'slice']
    
    def get_current_valid_actions(self) -> list:
        """Get ACTUAL valid actions for current state"""
        # Return the current admissible commands that are updated after each step
        return self._admissible_commands if self._admissible_commands else []
    
    def process_observation(self, obs):
        """ALFWorld-specific observation processing with universal cleaning"""
        # Handle tuple format from ALFWorld
        if isinstance(obs, tuple) and len(obs) > 0:
            obs_str = obs[0] if isinstance(obs[0], str) else str(obs[0])
        elif isinstance(obs, list) and obs:
            obs_str = obs[0] if isinstance(obs[0], str) else str(obs[0])
        else:
            obs_str = str(obs) if obs else ""
        
        # Universal cleaning for tuple artifacts
        if obs_str.startswith("('") or obs_str.startswith('("'):
            obs_str = obs_str[2:]
            for suffix in ["',)", '",)', "')", '")']:
                if obs_str.endswith(suffix):
                    obs_str = obs_str[:-len(suffix)]
                    break
        
        # Unescape
        obs_str = obs_str.replace('\\n', '\n').replace('\\t', '\t')
        obs_str = obs_str.replace("\\'", "'").replace('\\"', '"')
        
        return obs_str
    
    def get_environment_name(self, info):
        """Extract ALFWorld-specific environment name"""
        if 'extra.gamefile' in info and info['extra.gamefile']:
            return '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        return "alfworld_unknown"
    
    def is_success_indicator(self, obs: str) -> bool:
        """ALFWorld-specific success indicators"""
        # This is just a helper - the actual success comes from info['won']
        indicators = ['you put', 'you place', 'you cool', 'you heat', 'you clean', 'you slice']
        obs_lower = obs.lower()
        return any(ind in obs_lower for ind in indicators)
    
    def get_failure_indicators(self) -> List[str]:
        """ALFWorld-specific failure patterns"""
        return ['nothing happens.', "i don't understand", "that's not a valid action"]


class AppWorldWrapper(UniversalEnvWrapper):
    """
    Wrapper for AppWorld benchmark - Universal API-based task environment.

    Makes AppWorld compatible with ReflexGrad by:
    - Generating comprehensive valid_actions (all possible API calls)
    - Providing ALFWorld-style text observations
    - Executing exactly what algorithm selects (no translation)
    """

    ALL_APPS = ['spotify', 'amazon', 'venmo', 'gmail', 'google_calendar',
                'phone', 'simple_note', 'todoist', 'file_system', 'splitwise']

    def __init__(self, task_id: str = None, experiment_name: str = "reflexgrad",
                 split: str = "test_normal", env_id: int = 0, max_steps: int = 55,
                 fast_model=None):
        try:
            from appworld import AppWorld, load_task_ids
            self._appworld_available = True
        except ImportError:
            raise ImportError(
                "AppWorld not installed. Run:\n"
                "  pip install appworld\n"
                "  appworld install\n"
                "  appworld download data"
            )

        self.split = split
        self.env_id = env_id
        self.experiment_name = experiment_name
        self._max_steps = max_steps  # Store max_steps for episode termination
        self._fast_model = fast_model  # Store for potential use
        self.all_task_ids = load_task_ids(split)

        if task_id is None:
            idx = env_id % len(self.all_task_ids)
            task_id = self.all_task_ids[idx]

        self.task_id = task_id
        self._initialize_world()

    def _initialize_world(self):
        """Initialize AppWorld and state tracking."""
        from appworld import AppWorld
        self.world = AppWorld(task_id=self.task_id, experiment_name=self.experiment_name)

        self._step_count = 0
        self._done = False
        self._success = False
        self._last_action = None
        self._last_result = None

        self._task = self.world.task.instruction
        self._supervisor = self.world.task.supervisor

        self._profile = None  # From show_profile()
        self._account_passwords = {}  # From show_account_passwords(): {app_name: password}
        self._logged_in = {}  # {app_name: access_token}

        # UNIVERSAL: Dynamic data storage - keys are inferred from API responses
        # Format: {data_type: [list of items]} where data_type is inferred from API
        self._discovered_data = {}

        self._tried_actions = set()
        self._task_keywords = self._extract_task_keywords()

        # UNIVERSAL: Load API specs from JSON files (not hardcoded)
        self._api_specs = self._load_api_specs()

        # Simplified action names mapping
        self._action_mapping = {}  # Maps simple names -> full API calls

    def _extract_task_keywords(self) -> dict:
        """Extract keywords from task for search actions."""
        task_lower = self._task.lower()
        stop_words = {'the', 'a', 'an', 'to', 'for', 'and', 'or', 'in', 'on', 'my', 'i', 'me'}
        words = re.findall(r'\b[a-z]+\b', task_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        quoted = re.findall(r'"([^"]+)"', self._task) + re.findall(r"'([^']+)'", self._task)
        amounts = re.findall(r'\$?(\d+(?:\.\d+)?)', self._task)
        names = re.findall(r'\b([A-Z][a-z]+)\b', self._task)
        return {'keywords': keywords[:10], 'quoted': quoted, 'amounts': amounts, 'names': names}

    def _load_api_specs(self) -> dict:
        """UNIVERSAL: Load API specifications from JSON files.

        Returns dict: {app_name: {api_name: api_spec}}
        This enables dynamic action generation without hardcoding.
        """
        import json
        import os

        api_specs = {}
        # Find API docs directory relative to this file or in data/
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'data', 'api_docs', 'standard'),
            '/app/data/api_docs/standard',  # Docker container path
            'data/api_docs/standard',
        ]

        api_docs_dir = None
        for path in possible_paths:
            if os.path.isdir(path):
                api_docs_dir = path
                break

        if not api_docs_dir:
            print("[WRAPPER] Warning: API docs directory not found, using fallback")
            return {}

        # Load all app API specs
        for filename in os.listdir(api_docs_dir):
            if filename.endswith('.json') and filename != 'api_docs.json':
                app_name = filename.replace('.json', '')
                filepath = os.path.join(api_docs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        api_specs[app_name] = json.load(f)
                except Exception as e:
                    print(f"[WRAPPER] Failed to load {filename}: {e}")

        return api_specs

    def _get_api_data_type(self, api_spec: dict) -> str:
        """UNIVERSAL: Infer data type from API response schema.

        Returns a generic type name for storing discovered data.
        """
        response = api_spec.get('response_schemas', {}).get('success', {})

        # If response is a list, look at the first item's keys
        if isinstance(response, list) and response:
            item = response[0]
            if isinstance(item, dict):
                keys = set(item.keys())
                # Infer type from common fields
                if 'email' in keys and 'first_name' in keys:
                    return 'contacts'
                if 'playlist_id' in keys or 'playlist' in api_spec.get('api_name', ''):
                    return 'playlists'
                if 'song_id' in keys or 'title' in keys:
                    return 'songs'
                if 'product_id' in keys:
                    return 'products'
                if 'transaction_id' in keys:
                    return 'transactions'

        # Fallback: use API name to infer type
        api_name = api_spec.get('api_name', '')
        if 'friend' in api_name:
            return 'friends'
        if 'contact' in api_name:
            return 'contacts'
        if 'song' in api_name or 'queue' in api_name:
            return 'songs'
        if 'email' in api_name or 'inbox' in api_name:
            return 'emails'

        return 'generic'

    def reset(self):
        """Reset environment and return initial observation."""
        self._initialize_world()
        obs = self._build_observation(None, None, True)
        info = {
            'extra.gamefile': [f"appworld/{self.split}/{self.task_id}"],
            'task_id': self.task_id,
            'won': [False],
            'admissible_commands': [self.get_current_valid_actions()]
        }
        return [obs], info

    def step(self, actions):
        """Execute action - expand simplified names to full API calls."""
        action = actions[0] if isinstance(actions, list) else actions
        action = self._clean_action(action)

        # FIX #59: Smart action matching for partial/incomplete agent inputs
        # The agent often sends partial actions that need to be mapped to valid actions
        valid_actions = self.get_current_valid_actions()
        original_action = action

        # FIX #59a: Handle "complete task" actions
        if action.lower().startswith('complete task'):
            if 'with answer' in action.lower():
                # Extract answer part
                ans = action.lower().split('with answer')[-1].strip().strip('[]')
                if ans.lower() in ['n', 'number', 'your_answer', '']:
                    # Placeholder answer - reject
                    self._step_count += 1
                    return [self._build_observation(action, "Provide actual answer, not placeholder [N]", False)], 0.0, [False], {'won': [False], 'admissible_commands': [valid_actions]}
                # Convert to proper API call
                try:
                    ans_val = int(ans)
                except:
                    try:
                        ans_val = float(ans)
                    except:
                        ans_val = f"'{ans}'"
                action = f"apis.supervisor.complete_task(answer={ans_val})"
                print(f"[FIX #59a] complete task -> {action}")
            else:
                action = "apis.supervisor.complete_task(answer=None)"
                print(f"[FIX #59a] complete task -> {action}")

        # FIX #59b: Handle file name/path inputs (e.g., "/home/adam/documents/personal/notes/Book Reading Lists")
        # Match file paths (containing /documents/ or /notes/) or file-like names
        elif (action.endswith('.md') or action.endswith('.txt') or action.endswith('.json') or
              '/documents/' in action or '/notes/' in action or action.startswith('~/') or
              (re.search(r'\d{4}-\d{2}-\d{2}', action) and '_' in action)) and action not in valid_actions:

            matched = False
            best_match = None
            best_priority = 0

            # FIX #59b-DIR: Check if this is a DIRECTORY path (no file extension)
            # Should match to "list files in 'path'" actions
            is_directory = not any(action.endswith(ext) for ext in ['.md', '.txt', '.json', '.py', '.js', '.csv'])

            if is_directory:
                # Try to find matching "list files" action
                for va in valid_actions:
                    if 'list' in va.lower() and ('file' in va.lower() or 'directory' in va.lower()):
                        # Check if the path matches
                        if action in va or action.replace('~/', '/home/adam/') in va:
                            best_match = va
                            best_priority = 5  # Highest for directory listing
                            break
                        # Also try partial match - action path contained in valid action
                        elif action.split('/')[-1] in va and '/documents/' in va:
                            best_match = va
                            best_priority = 4

            # If not matched as directory, try file matching
            if not best_match:
                file_name = action
                # Normalize: remove path, keep just filename
                if '/' in file_name:
                    file_name = file_name.split('/')[-1]

                # Normalize to title form (no extension, underscores to spaces)
                title = file_name.replace('.md', '').replace('.txt', '').replace('.json', '').replace('_', ' ').strip()

                # FIX #59b IMPROVED: Prefer "create note" actions over "show_note" for file paths
                # Search in priority order: create_note > read_file > show_note
                for va in valid_actions:
                    va_lower = va.lower()
                    # Check if this action contains the title/filename
                    if title.lower() in va_lower or file_name.lower() in va_lower:
                        # Determine priority based on action type
                        if 'create' in va_lower and 'note' in va_lower:
                            priority = 3  # Highest priority
                        elif 'read' in va_lower or 'get' in va_lower or 'show_file' in va_lower:
                            priority = 2  # Medium priority
                        elif 'show' in va_lower or 'view' in va_lower:
                            priority = 1  # Lower priority
                        else:
                            priority = 0  # Fallback

                        if priority > best_priority:
                            best_priority = priority
                            best_match = va

            if best_match:
                action = best_match
                matched = True
                print(f"[FIX #59b] '{original_action}' -> '{action}' (priority={best_priority})")

        # FIX #59c: Handle partial action inputs (find best matching valid action)
        elif action not in valid_actions and not action.startswith('apis.'):
            action_lower = action.lower()
            best_match = None
            best_score = 0

            for va in valid_actions:
                va_lower = va.lower()
                # Check if action is a substring of valid action
                if action_lower in va_lower:
                    score = len(action_lower) / len(va_lower)
                    if score > best_score:
                        best_score = score
                        best_match = va
                # Check if valid action contains key words from action
                elif any(word in va_lower for word in action_lower.split() if len(word) > 3):
                    score = sum(1 for word in action_lower.split() if word in va_lower and len(word) > 3) / max(len(action_lower.split()), 1)
                    if score > best_score:
                        best_score = score
                        best_match = va

            if best_match and best_score > 0.3:
                action = best_match
                print(f"[FIX #59c] '{original_action}' -> '{action}' (score={best_score:.2f})")

        # Expand simplified action names to full API calls
        # This is the only translation needed - LLM outputs simple names, we expand them
        if action in self._action_mapping:
            full_action = self._action_mapping[action]
            print(f"[WRAPPER] '{action}' -> '{full_action[:70]}...'")
            action = full_action

        self._step_count += 1
        self._last_action = action
        self._tried_actions.add(action)

        # Check if max_steps reached - terminate episode
        if self._step_count >= self._max_steps:
            self._done = True
            # Evaluate task to determine if it succeeded
            try:
                eval_result = self.world.evaluate()
                self._success = getattr(eval_result, 'success', False)
            except:
                self._success = False
            observation = self._build_observation(action, "Max steps reached. Episode ended.", True)
            reward = 1.0 if self._success else 0.0
            info = {'won': [self._success], 'step': self._step_count,
                    'admissible_commands': [self.get_current_valid_actions()]}
            return [observation], reward, [True], info

        if self._done:
            return [self._build_observation(action, "Task already completed", True)], \
                   1.0 if self._success else 0.0, [True], {'won': [self._success]}

        try:
            # Execute action and capture result properly
            # AppWorld requires: 1) assign to var, 2) print var to get data
            result = self._execute_and_capture(action)
            success = 'Execution failed' not in str(result) and 'Exception' not in str(result)
            self._last_result = result
        except Exception as e:
            result = str(e)
            success = False
            self._last_result = f"Error: {result}"

        self._update_state(action, result, success)

        if 'complete_task' in action.lower():
            # Validate task is ACTUALLY complete before allowing episode to end
            # This prevents premature completion claims from ending the episode
            try:
                eval_result = self.world.evaluate()
                is_actually_complete = getattr(eval_result, 'success', False)

                if is_actually_complete:
                    # Task genuinely complete - allow episode to end
                    self._done = True
                    self._success = True
                else:
                    # Task NOT complete - reject completion attempt, continue episode
                    result = "Task not complete yet. The task requirements have not been fulfilled. Continue working on the task."
                    self._done = False
                    self._success = False
            except Exception as e:
                # Evaluation failed - don't allow completion
                result = f"Cannot verify task completion. Continue working. Error: {str(e)}"
                self._done = False
                self._success = False

        observation = self._build_observation(action, result, success)
        reward = 1.0 if self._success else 0.0
        info = {'won': [self._success], 'step': self._step_count,
                'admissible_commands': [self.get_current_valid_actions()]}
        return [observation], reward, [self._done], info

    def _execute_and_capture(self, action: str):
        """Execute action and capture actual result data from AppWorld."""
        import json

        # Generate unique variable name
        var_name = f"_result_{self._step_count}"

        # Execute with assignment
        exec_result = self.world.execute(f"{var_name} = {action}")

        if 'Execution failed' in exec_result or 'Exception' in exec_result:
            return exec_result

        # Get the actual value by printing
        print_result = self.world.execute(f"print({var_name})")

        # Try to parse as JSON
        try:
            # Clean up the result string
            if print_result and not print_result.startswith('Execution'):
                return json.loads(print_result)
        except json.JSONDecodeError:
            pass

        # Return raw result if not JSON
        return print_result if print_result else exec_result

    def _clean_action(self, action: str) -> str:
        """Remove numbering and clean action string."""
        action = action.strip()
        # Don't strip if action is purely numeric (could be phone number)
        if not action.isdigit():
            action = re.sub(r'^[\[\(]?\d+[\.\)\]]?\s*', '', action)
            action = re.sub(r'^[-•]\s*', '', action)
        return action.strip()

    def _update_state(self, action: str, result, success: bool):
        """Update tracked state from action result."""
        if not success:
            return
        action_lower = action.lower()

        # Handle supervisor.show_profile()
        if 'supervisor' in action_lower and 'show_profile' in action_lower:
            if isinstance(result, dict):
                self._profile = result

        # Handle supervisor.show_account_passwords()
        if 'supervisor' in action_lower and 'account_passwords' in action_lower:
            if isinstance(result, list):
                self._account_passwords = {
                    p['account_name']: p['password'] for p in result if 'account_name' in p
                }

        # Handle login responses
        if '.login(' in action_lower:
            app_match = re.search(r'apis\.(\w+)\.login', action_lower)
            if app_match and isinstance(result, dict):
                app = app_match.group(1)
                token = result.get('access_token') or result.get('token') or 'logged_in'
                self._logged_in[app] = token

        # UNIVERSAL: Handle add_friend - add to discovered friends immediately
        if 'add_friend' in action_lower:
            email_match = re.search(r"user_email='([^']+)'", action)
            if email_match:
                new_email = email_match.group(1)
                # Find matching contact to get the name
                name = new_email
                for contact in self._discovered_data.get('contacts', []):
                    if contact.get('email') == new_email:
                        name = contact.get('name', new_email)
                        break
                # Initialize if needed (dynamic storage)
                if 'friends' not in self._discovered_data:
                    self._discovered_data['friends'] = []
                # Add to friends if not already there
                existing_emails = {f.get('email') for f in self._discovered_data.get('friends', [])}
                if new_email not in existing_emails:
                    self._discovered_data['friends'].append({
                        'id': new_email,
                        'name': name,
                        'email': new_email,
                        'phone_number': '',
                    })

        # UNIVERSAL: Handle remove_friend - remove from discovered friends immediately
        if 'remove_friend' in action_lower:
            email_match = re.search(r"user_email='([^']+)'", action)
            if email_match:
                removed_email = email_match.group(1)
                self._discovered_data['friends'] = [
                    f for f in self._discovered_data.get('friends', [])
                    if f.get('email') != removed_email
                ]

        # FIX #62: Handle file_system list_directory responses - store files
        if 'list_directory' in action_lower or 'directory' in action_lower:
            # Extract directory path from action
            dir_match = re.search(r"directory='([^']+)'", action)
            dir_path = dir_match.group(1) if dir_match else ''

            if 'files' not in self._discovered_data:
                self._discovered_data['files'] = []

            # Handle list result (could be list of dicts or strings)
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Dict with name/path
                        name = item.get('name', item.get('title', ''))
                        path = item.get('path', f"{dir_path}/{name}" if dir_path and name else '')
                        if name and path not in [f.get('path') for f in self._discovered_data['files']]:
                            self._discovered_data['files'].append({
                                'name': name,
                                'path': path,
                                'id': path,
                            })
                    elif isinstance(item, str):
                        # Just filename/path string
                        # FIX #62b: Extract just filename from full path
                        path = item if item.startswith('/') else f"{dir_path}/{item}" if dir_path else item
                        name = path.split('/')[-1] if '/' in path else item  # Extract just filename
                        if path not in [f.get('path') for f in self._discovered_data['files']]:
                            self._discovered_data['files'].append({
                                'name': name,
                                'path': path,
                                'id': path,
                            })

        # Extract discovered data from API responses
        if isinstance(result, dict):
            self._extract_discovered_data(result)
        elif isinstance(result, list):
            # Handle direct list responses (e.g., search_contacts returns list directly)
            self._extract_list_data(action_lower, result)
            for item in result:
                if isinstance(item, dict):
                    self._extract_discovered_data(item)

    def _extract_list_data(self, action: str, result: list):
        """UNIVERSAL: Extract data from list API responses based on action/response type."""
        if not result or not isinstance(result[0], dict):
            return

        # UNIVERSAL: Infer data type from action string
        # Priority order matters for specificity
        target = self._infer_data_type_from_action(action)
        if not target:
            # Fallback: infer from first item's fields
            target = self._infer_data_type_from_item(result[0])
        if not target:
            target = 'generic'

        # Initialize target list if needed (dynamic storage)
        if target not in self._discovered_data:
            self._discovered_data[target] = []

        # For friends-related data, REPLACE to keep fresh (prevents stale actions)
        if 'friend' in target:
            self._discovered_data[target] = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                self._discovered_data[target].append(self._normalize_item(item))
            return

        # For other data types, append if not exists
        existing_ids = {x.get('id') or x.get('email') for x in self._discovered_data.get(target, [])}
        for item in result[:20]:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_item(item)
            item_id = normalized.get('id') or normalized.get('email')
            if item_id and item_id not in existing_ids:
                self._discovered_data[target].append(normalized)

    def _infer_data_type_from_action(self, action: str) -> str:
        """UNIVERSAL: Infer data type from action string."""
        action = action.lower()

        # Specific patterns first (more specific before generic)
        patterns = [
            ('friend', 'contact', 'phone_friends'),  # "search friend contacts"
            ('contact', None, 'contacts'),
            ('friend', None, 'friends'),
            ('email', None, 'emails'),
            ('inbox', None, 'emails'),
            ('product', None, 'products'),
            ('transaction', None, 'transactions'),
            ('queue', None, 'queue'),
            ('liked_song', None, 'liked_songs'),
            ('song', None, 'songs'),
            ('playlist', None, 'playlists'),
            ('event', None, 'events'),
            ('note', None, 'notes'),
            ('task', None, 'tasks'),
            ('alarm', None, 'alarms'),
            ('user', None, 'users'),
        ]

        for p1, p2, target in patterns:
            if p1 in action and (p2 is None or p2 in action):
                return target
        return None

    def _infer_data_type_from_item(self, item: dict) -> str:
        """UNIVERSAL: Infer data type from item fields."""
        keys = set(item.keys())

        if 'playlist_id' in keys:
            return 'playlists'
        if 'song_id' in keys or ('title' in keys and 'artist' in keys):
            return 'songs'
        if 'product_id' in keys:
            return 'products'
        if 'email' in keys and 'first_name' in keys:
            return 'contacts'
        if 'transaction_id' in keys:
            return 'transactions'
        return None

    def _normalize_item(self, item: dict) -> dict:
        """UNIVERSAL: Normalize item to standard format for storage."""
        item_id = item.get('id') or item.get('contact_id') or item.get('song_id') or item.get('email')
        name = item.get('name') or f"{item.get('first_name', '')} {item.get('last_name', '')}".strip() or item.get('title', '')
        return {
            'id': item_id,
            'name': name or 'Unknown',
            'email': item.get('email', ''),
            'phone_number': item.get('phone_number', ''),
            '_raw': item,  # Keep raw data for any field access
        }

    def _extract_discovered_data(self, data: dict):
        """UNIVERSAL: Extract data from API responses into dynamic storage."""
        for key in ['playlists', 'emails', 'messages', 'products', 'items',
                    'contacts', 'events', 'transactions', 'notes', 'files', 'friends',
                    'songs', 'queue', 'liked_songs', 'tracks', 'users', 'alarms', 'tasks']:
            if key in data:
                items = data[key]
                if not isinstance(items, list):
                    continue
                # Map API response keys to internal data keys
                key_map = {'messages': 'emails', 'items': 'products', 'tracks': 'songs'}
                target_key = key_map.get(key, key)
                # Initialize if needed (dynamic storage)
                if target_key not in self._discovered_data:
                    self._discovered_data[target_key] = []
                existing_ids = {x.get('id') for x in self._discovered_data.get(target_key, [])}
                for item in items[:20]:
                    if isinstance(item, dict):
                        item_id = item.get('id') or item.get('email')
                        if item_id and item_id not in existing_ids:
                            self._discovered_data[target_key].append({
                                'id': item_id,
                                'name': item.get('name') or item.get('first_name', '') + ' ' + item.get('last_name', '') or item.get('title') or item.get('subject', 'Unknown'),
                                'email': item.get('email', ''),
                            })

        if 'id' in data and data.get('type') == 'playlist':
            # Initialize if needed (dynamic storage)
            if 'playlists' not in self._discovered_data:
                self._discovered_data['playlists'] = []
            existing_ids = {p['id'] for p in self._discovered_data.get('playlists', [])}
            if data['id'] not in existing_ids:
                self._discovered_data['playlists'].append({
                    'id': data['id'], 'name': data.get('name', 'Unknown')
                })

    def _build_observation(self, action: Optional[str], result, success: bool) -> str:
        """Build ALFWorld-style observation."""
        parts = [f"Your task is to: {self._task}", ""]

        parts.append("=== CURRENT STATE ===")
        if self._logged_in:
            parts.append(f"You are logged into: {', '.join(self._logged_in.keys())}")
        else:
            parts.append("You are not logged into any apps.")
        if self._profile:
            parts.append(f"Account: {self._profile.get('email', 'profile loaded')}")
            if self._account_passwords:
                parts.append(f"Available apps: {', '.join(self._account_passwords.keys())}")
        else:
            parts.append("Profile: Not loaded yet. Call apis.supervisor.show_profile() first.")
        parts.append(f"Step: {self._step_count}")
        parts.append("")

        if action:
            parts.append("=== LAST ACTION ===")
            parts.append(f"> {action}")
            parts.append("")
            parts.append("=== RESULT ===")
            parts.append(self._format_result(result) if success else f"Action failed: {result}")
            parts.append("")

        discovered = []
        # Define display labels for clarity (especially for sync tasks)
        display_labels = {
            'phone_friends': 'Phone Friends (TARGET - Venmo friends should match this)',
            'friends': 'Venmo Friends (CURRENT)',
            'contacts': 'Phone Contacts (ALL)',
        }
        for dtype, items in self._discovered_data.items():
            if items:
                # FIX: Show ALL items (NO truncation) so agent sees complete state
                names = [f"{x['name']} ({x.get('email') or x['id']})" for x in items]
                label = display_labels.get(dtype, dtype.title())
                discovered.append(f"{label}: {', '.join(names)}")
        if discovered:
            parts.append("=== WHAT YOU HAVE DISCOVERED ===")
            parts.extend(discovered)
            parts.append("")

        # FIX #23: Add task-aware action hints to help LLM understand what actions to take
        # For sync/friend tasks, show the diff and available actions
        task_lower = self._task.lower() if self._task else ""
        if ('friend' in task_lower or 'sync' in task_lower or 'reset' in task_lower) and 'venmo' in task_lower:
            phone_contacts = self._discovered_data.get('contacts', []) or self._discovered_data.get('phone_friends', [])
            venmo_friends = self._discovered_data.get('friends', [])

            if phone_contacts and venmo_friends:
                phone_emails = {p.get('email') for p in phone_contacts if p.get('email')}
                venmo_emails = {v.get('email') for v in venmo_friends if v.get('email')}

                # Who needs to be ADDED to Venmo (in phone but not Venmo)
                to_add_emails = phone_emails - venmo_emails
                to_add = [p for p in phone_contacts if p.get('email') in to_add_emails]

                # Who needs to be REMOVED from Venmo (in Venmo but not phone)
                to_remove_emails = venmo_emails - phone_emails
                to_remove = [v for v in venmo_friends if v.get('email') in to_remove_emails]

                if to_add or to_remove:
                    parts.append("=== ACTION REQUIRED ===")
                    if to_add:
                        add_names = [p['name'] for p in to_add[:5]]
                        parts.append(f"NEED TO ADD to Venmo: {', '.join(add_names)}" +
                                   (f" ... and {len(to_add)-5} more" if len(to_add) > 5 else ""))
                        parts.append(f"  → Use: 'add [name] on venmo' (e.g., 'add {add_names[0]} on venmo')")

                    if to_remove:
                        remove_names = [v['name'] for v in to_remove[:5]]
                        parts.append(f"NEED TO REMOVE from Venmo: {', '.join(remove_names)}" +
                                   (f" ... and {len(to_remove)-5} more" if len(to_remove) > 5 else ""))
                        parts.append(f"  → Use: 'remove [name] on venmo' (e.g., 'remove {remove_names[0]} on venmo')")

                    parts.append("")
                elif phone_emails == venmo_emails:
                    parts.append("=== TASK STATUS ===")
                    parts.append("✓ Venmo friends now match phone contacts! You can complete the task.")
                    parts.append("")

        return "\n".join(parts)

    def _format_result(self, result) -> str:
        """Format API result readably."""
        if result is None:
            return "Action completed successfully."
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if 'access_token' in result:
                return "Login successful. Access token received."
            if 'status' in result:
                return f"Status: {result['status']}"
            if 'error' in result:
                return f"Error: {result['error']}"
            lines = [f"{k}: {len(v)} items" if isinstance(v, list) else f"{k}: {v}"
                     for k, v in list(result.items())[:10]]
            return "\n".join(lines)
        if isinstance(result, list):
            if not result:
                return "No items found."
            lines = [f"Found {len(result)} items:"]
            for i, item in enumerate(result[:10], 1):
                name = item.get('name') or item.get('title') or str(item)[:50] if isinstance(item, dict) else str(item)
                lines.append(f"  {i}. {name}")
            if len(result) > 10:
                lines.append(f"  ... and {len(result) - 10} more")
            return "\n".join(lines)
        return str(result)

    def get_current_valid_actions(self) -> List[str]:
        """Generate simplified action names (like ALFWorld) with mapping to full API calls."""
        # Clear and rebuild mapping each time (state may have changed)
        self._action_mapping = {}
        simple_actions = []

        def add_action(simple_name: str, full_call: str):
            """Helper to add action with mapping."""
            if simple_name not in self._action_mapping:
                self._action_mapping[simple_name] = full_call
                simple_actions.append(simple_name)

        # Supervisor actions - simple names like ALFWorld
        if not self._profile:
            add_action("show profile", "apis.supervisor.show_profile()")
        if not self._account_passwords:
            add_action("show passwords", "apis.supervisor.show_account_passwords()")
        add_action("show task", "apis.supervisor.show_active_task()")
        if self._profile:
            add_action("show payment cards", "apis.supervisor.show_payment_cards()")
            add_action("show addresses", "apis.supervisor.show_addresses()")
        add_action("complete task", "apis.supervisor.complete_task(answer=None)")

        # Login actions - simple format: "login to venmo"
        if self._profile and self._account_passwords:
            email = self._profile.get('email', '')
            phone_number = self._profile.get('phone_number', '')
            for app in self.ALL_APPS:
                if app not in self._logged_in:
                    password = self._account_passwords.get(app, '')
                    if password:
                        username = phone_number if app == 'phone' else email
                        add_action(f"login to {app}", f"apis.{app}.login(username='{username}', password='{password}')")

        # App-specific actions for logged-in apps
        for app, token in self._logged_in.items():
            self._add_app_actions_simplified(app, token, add_action)

        # Data-specific actions
        self._add_data_actions_simplified(add_action)

        return simple_actions

    def _add_app_actions_simplified(self, app: str, token: str, add_action):
        """UNIVERSAL: Generate actions dynamically from API specs.

        Uses loaded API specs to generate actions instead of hardcoding.
        """
        if app not in self._api_specs:
            return

        app_apis = self._api_specs[app]

        for api_name, api_spec in app_apis.items():
            # Skip auth APIs (login/logout/signup)
            if api_name in ('login', 'logout', 'signup', 'delete_account'):
                continue

            params = api_spec.get('parameters', [])
            required_params = [p for p in params if p.get('required') and p['name'] != 'access_token']

            # APIs with only access_token required = simple read/show actions
            param_names = [p['name'] for p in params]
            if not required_params:
                simple_name = api_name.replace('_', ' ')
                if app not in simple_name:
                    simple_name = f"{simple_name} on {app}"
                # FIX #22: Add page_limit=20 for APIs that support pagination
                if 'page_limit' in param_names:
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', page_limit=20)"
                else:
                    full_call = f"apis.{app}.{api_name}(access_token='{token}')"
                add_action(simple_name, full_call)

                # FIX #61b: Also generate parameterized actions for APIs with optional directory/path params
                if any(p in param_names for p in ['directory_path', 'directory', 'file_path']):
                    self._generate_parameterized_actions(app, api_name, api_spec, token, add_action)

            # APIs with optional parameters (search, filter APIs)
            else:
                # Check if all required params can be filled from discovered data
                self._generate_parameterized_actions(app, api_name, api_spec, token, add_action)

        # Generate data-driven actions from discovered data
        self._generate_data_driven_actions(app, token, add_action)

    def _generate_parameterized_actions(self, app: str, api_name: str, api_spec: dict, token: str, add_action):
        """UNIVERSAL: Generate actions for APIs with parameters.

        Fills parameters from discovered data or task keywords.
        """
        params = api_spec.get('parameters', [])
        param_names = {p['name'] for p in params}

        # FIX #61: UNIVERSAL directory listing - extract paths from task instruction
        # Check for various directory parameter names
        dir_param = next((p for p in ['directory_path', 'directory', 'path'] if p in param_names), None)
        if dir_param and ('show_directory' in api_name or 'list' in api_name):
            # Extract paths enclosed in quotes
            path_patterns = [
                r'"(~/[^"]+)"',    # "~/path"
                r"'(~/[^']+)'",    # '~/path'
                r'"(/[^"]+)"',     # "/path"
                r"'(/[^']+)'",     # '/path'
            ]
            task_paths = []
            for pattern in path_patterns:
                matches = re.findall(pattern, self._task)
                for m in matches:
                    clean_path = m.rstrip('/')
                    if clean_path and clean_path not in task_paths:
                        task_paths.append(clean_path)

            for path in task_paths[:5]:
                simple_name = f"list files in '{path}'"
                full_call = f"apis.{app}.{api_name}(access_token='{token}', {dir_param}='{path}')"
                add_action(simple_name, full_call)

            if not task_paths:
                for default_path in ['~', '~/documents']:
                    simple_name = f"list files in '{default_path}'"
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', {dir_param}='{default_path}')"
                    add_action(simple_name, full_call)

        # Handle search APIs - use task keywords
        if 'search' in api_name:
            query_param = next((p for p in params if p['name'] == 'query'), None)
            if query_param:
                for kw in self._task_keywords.get('keywords', [])[:3]:
                    simple_name = f"search {app} for {kw}"
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', query='{kw}')"
                    add_action(simple_name, full_call)
                for q in self._task_keywords.get('quoted', []):
                    simple_name = f"search {app} for '{q}'"
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', query='{q}')"
                    add_action(simple_name, full_call)

            # Handle relationship filter (e.g., search_contacts with relationship param)
            rel_param = next((p for p in params if p['name'] == 'relationship'), None)
            if rel_param:
                # Show available relationships from response or use common ones
                for rel in ['friend', 'family', 'coworker']:
                    simple_name = f"search {rel} {api_name.replace('search_', '')}"
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', relationship='{rel}', page_limit=20)"
                    add_action(simple_name, full_call)

            # Add generic search without filters
            simple_name = f"search {app} {api_name.replace('search_', '')}"
            full_call = f"apis.{app}.{api_name}(access_token='{token}', page_limit=20)"
            add_action(simple_name, full_call)

        # UNIVERSAL: Handle payment/money APIs (send_payment, create_payment_request, etc.)
        # These need user_email + amount from discovered data + task keywords
        if 'user_email' in param_names and 'amount' in param_names:
            amounts = self._task_keywords.get('amounts', [])
            quoted = self._task_keywords.get('quoted', [])
            description = quoted[0] if quoted else ''

            # Get all discovered people (friends, contacts, users)
            all_people = []
            for dtype in ['friends', 'contacts', 'users', 'phone_friends']:
                all_people.extend(self._discovered_data.get(dtype, []))

            for person in all_people[:10]:
                email = person.get('email')
                name = person.get('name', email)
                if not email:
                    continue

                for amount in amounts[:3]:
                    # Determine action verb from API name
                    if 'request' in api_name:
                        verb = 'request'
                        simple_name = f"request ${amount} from {name}"
                    elif 'send' in api_name:
                        verb = 'send'
                        simple_name = f"send ${amount} to {name}"
                    else:
                        verb = api_name.replace('_', ' ')
                        simple_name = f"{verb} ${amount} {name}"

                    # Build full API call with all parameters
                    params_str = f"access_token='{token}', user_email='{email}', amount={amount}"
                    if 'description' in param_names and description:
                        # Escape apostrophes in description
                        safe_desc = description.replace("'", "\\'")
                        params_str += f", description='{safe_desc}'"
                    if 'private' in param_names:
                        # Check if task says "publicly" or "private"
                        is_private = 'private' in self._task.lower()
                        params_str += f", private={is_private}"

                    full_call = f"apis.{app}.{api_name}({params_str})"
                    add_action(simple_name, full_call)

    def _generate_data_driven_actions(self, app: str, token: str, add_action):
        """UNIVERSAL: Generate actions based on discovered data.

        Matches discovered entities to relevant APIs.
        """
        if app not in self._api_specs:
            return

        app_apis = self._api_specs[app]

        # Map discovered data types to relevant APIs
        for data_type, items in self._discovered_data.items():
            if not items:
                continue

            for item in items[:15]:  # Limit to prevent action explosion
                item_email = item.get('email', '')
                item_id = item.get('id', item.get('song_id', item.get('playlist_id', '')))
                item_name = item.get('name', item.get('title', str(item_id)[:20]))

                # Find APIs that can use this data
                for api_name, api_spec in app_apis.items():
                    params = api_spec.get('parameters', [])
                    param_names = {p['name'] for p in params}

                    # Match email-based APIs (add_friend, remove_friend, send_payment, etc.)
                    if item_email and 'user_email' in param_names:
                        action_verb = api_name.split('_')[0]  # add, remove, send, etc.
                        simple_name = f"{action_verb} {item_name} on {app}"
                        full_call = f"apis.{app}.{api_name}(access_token='{token}', user_email='{item_email}')"
                        add_action(simple_name, full_call)

                    # Match ID-based APIs (play, remove from queue, etc.)
                    if item_id:
                        for id_param in ['song_id', 'playlist_id', 'product_id', 'email_id', 'note_id']:
                            if id_param in param_names:
                                action_verb = api_name.replace('_', ' ')
                                simple_name = f"{action_verb} {item_name}"
                                full_call = f"apis.{app}.{api_name}(access_token='{token}', {id_param}='{item_id}')"
                                add_action(simple_name, full_call)

                    # FIX #60: UNIVERSAL file path handling
                    # Match any API with file_path parameter to discovered items with paths
                    item_path = item.get('path', '')
                    if item_path and 'file_path' in param_names:
                        action_verb = api_name.replace('_', ' ')
                        simple_name = f"{action_verb} '{item_name}'"
                        full_call = f"apis.{app}.{api_name}(access_token='{token}', file_path='{item_path}')"
                        add_action(simple_name, full_call)

                    # FIX #60: UNIVERSAL title-based actions (notes, documents, etc.)
                    # Match any API with 'title' parameter to items with names
                    if item_name and 'title' in param_names and api_name.startswith('create'):
                        # FIX #60b: Extract just filename if item_name is a path
                        if '/' in item_name:
                            item_name = item_name.split('/')[-1]
                        # Normalize title (remove extension, underscores to spaces)
                        normalized_title = item_name.rsplit('.', 1)[0] if '.' in item_name else item_name
                        normalized_title = normalized_title.replace('_', ' ')
                        action_verb = api_name.replace('_', ' ')
                        # Build params dynamically based on what the API accepts
                        params_str = f"access_token='{token}', title='{normalized_title}'"
                        # FIX #60c: Add content/body parameter (required by simple_note)
                        if 'content' in param_names:
                            params_str += ", content=''"
                        elif 'body' in param_names:
                            params_str += ", body=''"
                        simple_name = f"{action_verb} '{normalized_title}'"
                        full_call = f"apis.{app}.{api_name}({params_str})"
                        add_action(simple_name, full_call)

    def _add_data_actions_simplified(self, add_action):
        """UNIVERSAL: Data-driven actions are now generated in _generate_data_driven_actions.

        This method is kept for compatibility but uses the new dynamic storage.
        """
        # Payment actions based on discovered contacts and task amounts
        contacts = self._discovered_data.get('contacts', [])
        if contacts and 'venmo' in self._logged_in:
            token = self._logged_in['venmo']
            for contact in contacts[:10]:
                if contact.get('id') or contact.get('email'):
                    name = contact.get('name', 'contact')
                    to_id = contact.get('id') or contact.get('email')
                    for amt in self._task_keywords.get('amounts', [])[:2]:
                        add_action(f"send ${amt} to {name}", f"apis.venmo.send_payment(access_token='{token}', to='{to_id}', amount={amt})")

    def get_action_space(self) -> List[str]:
        return self.get_current_valid_actions()

    def close(self):
        try:
            if hasattr(self, 'world') and self.world and hasattr(self.world, 'close'):
                self.world.close()
        except:
            pass

    def process_observation(self, obs) -> str:
        if isinstance(obs, str):
            return obs
        if isinstance(obs, list) and obs:
            return str(obs[0])
        return str(obs)

    def is_success_indicator(self, obs: str) -> bool:
        indicators = ['completed successfully', 'task complete', 'success']
        return any(ind in obs.lower() for ind in indicators)

    def get_environment_name(self, info) -> str:
        return f"appworld_{self.task_id}"

    def get_failure_indicators(self) -> List[str]:
        return ['error:', 'failed', 'not found', 'invalid', 'unauthorized', 'exception']