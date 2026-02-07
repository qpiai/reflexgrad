"""Universal Environment Wrapper for Standard Research Benchmarks - Updated APIs"""

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

class TextWorldWrapper(UniversalEnvWrapper):
    def __init__(self, game_type="cooking", level="easy", env_id=None):
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
        return [obs], {'extra.gamefile': [self.game_file]}
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])
        info['won'] = [info.get('has_won', reward > 0)]
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
            
        except ImportError:
            raise ImportError("Jericho not installed. Run: pip install jericho[full]")
        except Exception as e:
            print(f"Error loading Jericho game: {e}")
            raise
    
    def reset(self):
        obs, info = self.env.reset()
        return [obs], {'extra.gamefile': [self.game_name], 'max_score': self.max_score}
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions[0])
        # Jericho provides normalized score
        info['won'] = [done and self.env.get_score() == self.max_score]
        return [obs], reward, [done], info
    
    def close(self):
        self.env.close()
    
    def get_action_space(self):
        """Get valid actions for current state"""
        try:
            # Jericho provides valid actions
            valid_actions = self.env.get_valid_actions()
            if valid_actions:
                return valid_actions
            else:
                # Sometimes returns empty, try templates
                return self.env.get_valid_action_templates()
        except:
            # Fallback
            return ['look', 'inventory', 'north', 'south', 'east', 'west', 'take all', 'drop all', 'examine']
    
    def is_success_indicator(self, obs: str) -> bool:
        """Jericho-specific success indicators"""
        indicators = ['you have won', 'victory', '*** you win ***', 'the end']
        obs_lower = obs.lower()
        return any(ind in obs_lower for ind in indicators)

class ScienceWorldWrapper(UniversalEnvWrapper):
    """Wrapper for ScienceWorld benchmark (Allen AI)"""
    def __init__(self, task_name=None, task_id=0):
        try:
            from scienceworld import ScienceWorldEnv
            
            # Standard ScienceWorld tasks used in papers
            standard_tasks = [
                "boil",                              # Boil water
                "melt",                              # Melt ice
                "freeze",                            # Freeze water
                "change-the-state-of-matter-of",     # Change state of matter
                "chemistry-mix",                     # Chemistry mixing
                "chemistry-mix-paint-secondary-color", # Mix paint colors
                "find-animal",                       # Find animal
                "find-living-thing",                 # Find living thing
                "find-plant",                        # Find plant
                "grow-plant",                        # Grow plant
                "grow-fruit",                        # Grow fruit
                "test-conductivity",                 # Test electrical conductivity
                "use-thermometer",                   # Use thermometer
                "measure-melting-point-known-substance",   # Measure melting point
                "identify-life-stages-1",            # Life stages
                "identify-life-stages-2",            # Life cycle
                "lifespan-longest-lived",            # Lifespan
                "power-component",                   # Power components
                "mendelian-genetics-known-plant",    # Genetics
                "inclined-plane-determine-angle",    # Physics
            ]
            
            if task_name is None:
                task_name = standard_tasks[task_id % len(standard_tasks)]
                
            self.env = ScienceWorldEnv(task_name)
            self.task_name = task_name
            
        except ImportError:
            raise ImportError("ScienceWorld not installed. Run: pip install scienceworld")
        except Exception as e:
            print(f"Error creating ScienceWorld environment: {e}")
            raise
    
    def reset(self):
        result = self.env.reset()
        # ScienceWorld returns (observation_string, info_dict)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            # obs is the string observation
            return [obs], {'extra.gamefile': [self.task_name], 'scienceworld_info': info}
        else:
            # Fallback
            return [str(result)], {'extra.gamefile': [self.task_name]}
    
    def step(self, actions):
        result = self.env.step(actions[0])
        # ScienceWorld returns (observation_string, reward, done, info_dict)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            info['won'] = [done and reward > 0]
            return [obs], reward, [done], info
        else:
            # Fallback for unexpected formats
            return [str(result)], 0, [False], {'won': [False]}
    
    def close(self):
        self.env.close()
    
    def get_action_space(self):
        """Get valid actions for current state"""
        try:
            # ScienceWorld has getPossibleActions()
            return self.env.getPossibleActions()
        except:
            # Fallback
            return ['look', 'inventory', 'take', 'drop', 'examine', 'use', 'open', 'close']
    
    def is_success_indicator(self, obs: str) -> bool:
        """ScienceWorld-specific success indicators"""
        indicators = ['task completed', 'congratulations', 'goal achieved']
        obs_lower = obs.lower()
        return any(ind in obs_lower for ind in indicators)

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
            import random
            
            env_class = alfworld_env.get_environment(config["env"]["type"])
            
            # CRITICAL: Set a different random seed for each environment
            # This ensures each gets a different game from the dataset
            if env_id is not None:
                # Use a deterministic seed based on env_id
                seed = 42 + env_id * 137  # Prime number spacing
                random.seed(seed)
                import numpy as np
                np.random.seed(seed)
            
            # Initialize environment
            self.env = env_class(config, train_eval="eval_out_of_distribution")
            self.env = self.env.init_env(batch_size=1)
            
            # Force selection of a specific game by resetting with seed
            if env_id is not None and env_id > 0:
                # The seed should cause ALFWorld to select a different game
                self.env.seed(seed)  # If this method exists
            
            self._admissible_commands = []
            self._last_observation = ""
            
        except ImportError:
            raise ImportError("ALFWorld not installed. Run: pip install alfworld")
        except AttributeError:
            # If seed method doesn't exist, try multiple resets
            if env_id is not None and env_id > 0:
                for _ in range(env_id):
                    self.env.reset()


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