"""Real-time progress logging for monitoring training on-the-go"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List


class ProgressLogger:
    """Logs training progress incrementally for real-time monitoring"""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.progress_file = os.path.join(run_dir, 'progress.jsonl')
        self.summary_file = os.path.join(run_dir, 'progress_summary.json')

        # Initialize progress tracking
        self.trial_stats = {}

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a single event to JSONL file (append-only, crash-safe)"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            **data
        }

        # Append to JSONL (each line is valid JSON)
        with open(self.progress_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def log_trial_start(self, trial_idx: int, num_envs: int):
        """Log trial start"""
        self.log_event('trial_start', {
            'trial_idx': trial_idx,
            'num_envs': num_envs
        })

        # Initialize trial stats
        self.trial_stats[trial_idx] = {
            'started': datetime.now().isoformat(),
            'num_envs': num_envs,
            'completed_envs': 0,
            'success_count': 0,
            'failure_count': 0
        }

    def log_env_result(self, trial_idx: int, env_id: int, task: str,
                       success: bool, steps: int, trajectory_length: int):
        """Log individual environment result"""
        self.log_event('env_result', {
            'trial_idx': trial_idx,
            'env_id': env_id,
            'task': task[:80],  # Truncate long tasks
            'success': success,
            'steps': steps,
            'trajectory_length': trajectory_length
        })

        # Update trial stats
        if trial_idx in self.trial_stats:
            self.trial_stats[trial_idx]['completed_envs'] += 1
            if success:
                self.trial_stats[trial_idx]['success_count'] += 1
            else:
                self.trial_stats[trial_idx]['failure_count'] += 1

            # Update summary file in real-time
            self._update_summary()

    def log_textgrad_alignment(self, trial_idx: int, env_id: int,
                               recommendation: str, selected_action: str,
                               matched: bool, tier: str):
        """Log TextGrad recommendation alignment"""
        self.log_event('textgrad_alignment', {
            'trial_idx': trial_idx,
            'env_id': env_id,
            'recommendation': recommendation[:100],
            'selected': selected_action[:100],
            'matched': matched,
            'tier': tier
        })

    def log_memory_update(self, trial_idx: int, env_id: int,
                         memory_count: int, memory_type: str):
        """Log memory update"""
        self.log_event('memory_update', {
            'trial_idx': trial_idx,
            'env_id': env_id,
            'memory_count': memory_count,
            'memory_type': memory_type
        })

    def log_trial_end(self, trial_idx: int):
        """Log trial end and update summary"""
        if trial_idx in self.trial_stats:
            stats = self.trial_stats[trial_idx]
            stats['ended'] = datetime.now().isoformat()

            # Calculate success rate
            total = stats['completed_envs']
            success_rate = (stats['success_count'] / total * 100) if total > 0 else 0

            self.log_event('trial_end', {
                'trial_idx': trial_idx,
                'success_rate': success_rate,
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'total_envs': stats['num_envs']
            })

            self._update_summary()

    def _update_summary(self):
        """Update summary file with current progress (atomic write)"""
        summary = {
            'last_updated': datetime.now().isoformat(),
            'trials': {}
        }

        for trial_idx, stats in self.trial_stats.items():
            total = stats['completed_envs']
            success_rate = (stats['success_count'] / total * 100) if total > 0 else 0

            summary['trials'][f'trial_{trial_idx}'] = {
                'success_rate': f"{success_rate:.1f}%",
                'successes': stats['success_count'],
                'failures': stats['failure_count'],
                'completed': stats['completed_envs'],
                'total': stats['num_envs'],
                'started': stats.get('started', 'unknown'),
                'ended': stats.get('ended', 'in_progress')
            }

        # Atomic write
        temp_file = self.summary_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(summary, f, indent=2)
        os.replace(temp_file, self.summary_file)

    def get_current_stats(self, trial_idx: int) -> Dict[str, Any]:
        """Get current stats for a trial"""
        if trial_idx in self.trial_stats:
            stats = self.trial_stats[trial_idx].copy()
            total = stats['completed_envs']
            stats['success_rate'] = (stats['success_count'] / total * 100) if total > 0 else 0
            return stats
        return None


# Global instance
_progress_logger = None

def get_progress_logger(run_dir: str = None) -> ProgressLogger:
    """Get or create global progress logger"""
    global _progress_logger
    if _progress_logger is None and run_dir:
        _progress_logger = ProgressLogger(run_dir)
    return _progress_logger
