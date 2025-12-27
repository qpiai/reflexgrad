import json
import time
from datetime import datetime
from pathlib import Path

class ComprehensiveLogger:
    def __init__(self, trial_dir):
        self.trial_dir = Path(trial_dir)
        self.trial_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.step_gradients_file = self.trial_dir / 'step_gradients.jsonl'
        self.step_reflexions_file = self.trial_dir / 'step_reflexions.jsonl'
        self.episodic_file = self.trial_dir / 'episodic_reflections.txt'
        self.memory_file = self.trial_dir / 'memory_evolution.jsonl'
        self.gradients_file = self.trial_dir / 'all_gradients.jsonl'
    
    def log_step_gradient(self, env_id, step, gradient):
        """Log step-level gradient with all fields (PURE TEXTUAL FEEDBACK - NO numeric scores)"""
        with open(self.step_gradients_file, 'a') as f:
            entry = {
                'timestamp': time.time(),
                'env_id': env_id,
                'step': step,
                'action': gradient.get('action', 'N/A'),  # CRITICAL FIX: Include action for analysis!
                'progress_status': gradient.get('progress_status', 'EXPLORING'),  # Pure textual feedback
                'state_change': gradient.get('state_change', ''),
                'hypothesis': gradient.get('hypothesis', ''),
                'next_action_guidance': gradient.get('next_action_guidance', ''),
                'semantic_state_before': gradient.get('semantic_state_before', '')[:200],
                'semantic_state_after': gradient.get('semantic_state_after', '')[:200],
                'prerequisites_missing': gradient.get('prerequisites', {}).get('missing', []),
                'task_remaining': gradient.get('task_progress', {}).get('remaining', []),
                'guidance_source': gradient.get('guidance_source', 'unknown')  # PHASE 2 FIX: Track TextGrad vs Reflexion
            }
            f.write(json.dumps(entry) + '\n')
    
    def log_step_reflexion(self, env_id, step, action, reflection, success):
        """Log working/step reflexion"""
        with open(self.step_reflexions_file, 'a') as f:
            entry = {
                'timestamp': time.time(),
                'env_id': env_id,
                'step': step,
                'action': action,
                'reflection': reflection,
                'success': success
            }
            f.write(json.dumps(entry) + '\n')
    
    def log_episodic_reflection(self, env_id, task, reflection, gradients):
        """Log episodic reflection with gradients"""
        with open(self.episodic_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ENV {env_id} - {datetime.now()}\n")
            f.write(f"Task: {task}\n")
            f.write(f"{'='*80}\n")
            f.write(reflection)
            f.write(f"\n\nGRADIENTS:\n")
            for component, gradient in gradients.items():
                if component != 'structured_insights':
                    f.write(f"  {component}: {gradient}\n")
            f.write(f"{'='*80}\n\n")
    
    def log_memory_update(self, env_id, memory_type, content):
        """Log memory system updates"""
        with open(self.memory_file, 'a') as f:
            entry = {
                'timestamp': time.time(),
                'env_id': env_id,
                'type': memory_type,
                'content': content[:500] if isinstance(content, str) else str(content)[:500]
            }
            f.write(json.dumps(entry) + '\n')