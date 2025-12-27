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
                 split: str = "test_normal", env_id: int = 0, max_steps: int = 55):
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
            if not required_params:
                simple_name = api_name.replace('_', ' ')
                if app not in simple_name:
                    simple_name = f"{simple_name} on {app}"
                # FIX #22: Add page_limit=20 for APIs that support pagination
                param_names = [p['name'] for p in params]
                if 'page_limit' in param_names:
                    full_call = f"apis.{app}.{api_name}(access_token='{token}', page_limit=20)"
                else:
                    full_call = f"apis.{app}.{api_name}(access_token='{token}')"
                add_action(simple_name, full_call)

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