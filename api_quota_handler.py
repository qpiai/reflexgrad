import os
import pickle
import json
import sys
from datetime import datetime

class APIQuotaHandler:
    def __init__(self, checkpoint_manager, logging_dir):
        self.checkpoint_manager = checkpoint_manager
        self.logging_dir = logging_dir
        self.last_error = None
        
    def handle_api_error(self, error, trial_idx, env_configs, env_states):
        """Automatically handle API errors and save state"""
        error_str = str(error).lower()
        
        # Check if it's a quota error
        quota_indicators = [
            'rate limit', 'quota', 'insufficient_quota', 'exceeded',
            'too many requests', '429', 'billing', 'limit'
        ]
        
        is_quota_error = any(indicator in error_str for indicator in quota_indicators)
        
        if is_quota_error:
            print(f"\n[API QUOTA EXHAUSTED] {error}")
            print("[EMERGENCY SAVE] Saving all state for resume...")
            
            # Save everything
            self._emergency_save(trial_idx, env_configs, env_states)
            
            print("\n" + "="*80)
            print("API QUOTA EXHAUSTED - State saved successfully!")
            print("To resume, run with:")
            print(f"  --is_resume --resume_dir {self.logging_dir} --start_trial_num {trial_idx}")
            print("="*80 + "\n")
            
            # Exit gracefully
            sys.exit(0)
        
        return False
    
    def _emergency_save(self, trial_idx, env_configs, env_states):
        """Emergency save all state"""
        # Save master checkpoint
        completed_envs = []
        pending_envs = []
        
        if env_states:
            completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
            pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]
        
        self.checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
        
        # Save env configs
        env_config_path = os.path.join(self.logging_dir, f'env_results_trial_{trial_idx}.json')
        with open(env_config_path, 'w') as f:
            json.dump(env_configs, f, indent=4)
        
        # Save universal memory
        try:
            from alfworld_trial import universal_memory
            universal_memory.save_memory()
            print("[SAVED] Universal memory")
        except:
            pass
        
        # Save state embeddings cache
        try:
            from universal_state_embeddings import state_embeddings
            state_embeddings.save_cache()
            print("[SAVED] State embeddings cache")
        except Exception as e:
            print(f"[WARNING] Could not save state embeddings: {e}")

        # Save global memory manager
        try:
            from generate_reflections import global_memory_manager
            memory_path = os.path.join(self.logging_dir, 'global_memory_manager.pkl')
            with open(memory_path, 'wb') as f:
                pickle.dump(global_memory_manager, f)
            print("[SAVED] Global memory manager")
        except:
            pass
        
        # Save prompt generator
        try:
            from alfworld_trial import save_prompt_generator
            save_prompt_generator()
            print("[SAVED] Prompt generator")
        except:
            pass
        
        # Save timestamp
        with open(os.path.join(self.logging_dir, 'emergency_save.txt'), 'w') as f:
            f.write(f"Emergency save at: {datetime.now()}\n")
            f.write(f"Trial: {trial_idx}\n")
            f.write(f"Completed envs: {completed_envs}\n")
            f.write(f"API calls made: {self.checkpoint_manager.api_calls}\n")