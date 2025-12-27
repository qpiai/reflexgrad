"""
Deep Memory Profiler + Learning Transfer Tracker

This will help us:
1. Find the TRUE root cause of the 219GB memory leak
2. Track what learning is being transferred between trials
3. Verify learning is working correctly
"""

import sys
import gc
import psutil
import os
import json
from collections import defaultdict

# Try to import pympler, but make it optional
try:
    from pympler import asizeof, tracker
    HAS_PYMPLER = True
except ImportError:
    HAS_PYMPLER = False
    print("[WARNING] pympler not installed - using sys.getsizeof instead")

class DetailedMemoryProfiler:
    """Profile memory usage of specific objects"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        if HAS_PYMPLER:
            self.tracker = tracker.SummaryTracker()
        else:
            self.tracker = None
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.checkpoints = []

        # Learning transfer tracking
        self.learning_log = []
        self.trial_learning_summary = defaultdict(lambda: {
            'env_memories': {},
            'working_reflexions': {},
            'global_memory_items': 0,
            'universal_memory_items': 0
        })

    def checkpoint(self, label: str, objects_to_track: dict = None):
        """Create a memory checkpoint with detailed object tracking"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        delta = current_memory - self.baseline_memory

        print(f"\n{'='*80}")
        print(f"[MEMORY CHECKPOINT] {label}")
        print(f"  Total RSS: {current_memory:.2f} MB")
        print(f"  Delta from baseline: {delta:.2f} MB")

        if objects_to_track:
            print(f"\n  Object Sizes:")
            for name, obj in objects_to_track.items():
                try:
                    # Use pympler if available, otherwise sys.getsizeof
                    if HAS_PYMPLER:
                        size_bytes = asizeof.asizeof(obj)
                    else:
                        size_bytes = sys.getsizeof(obj)
                    size_mb = size_bytes / 1024 / 1024

                    # Get object type info
                    obj_type = type(obj).__name__

                    # Get length if applicable
                    try:
                        length = len(obj)
                        print(f"    {name} ({obj_type}): {size_mb:.2f} MB (length={length})")

                        # For large objects, drill down
                        if size_mb > 10 and hasattr(obj, '__iter__') and not isinstance(obj, str):
                            if isinstance(obj, (list, tuple)):
                                if obj:
                                    if HAS_PYMPLER:
                                        sample_size = asizeof.asizeof(obj[0]) / 1024 / 1024
                                    else:
                                        sample_size = sys.getsizeof(obj[0]) / 1024 / 1024
                                    print(f"      → Sample item size: {sample_size:.4f} MB")
                                    print(f"      → Estimated total: {sample_size * length:.2f} MB")
                            elif isinstance(obj, dict):
                                if obj:
                                    key, value = next(iter(obj.items()))
                                    if HAS_PYMPLER:
                                        sample_size = asizeof.asizeof((key, value)) / 1024 / 1024
                                    else:
                                        sample_size = sys.getsizeof((key, value)) / 1024 / 1024
                                    print(f"      → Sample entry size: {sample_size:.4f} MB")
                                    print(f"      → Estimated total: {sample_size * length:.2f} MB")
                    except TypeError:
                        print(f"    {name} ({obj_type}): {size_mb:.2f} MB")
                except Exception as e:
                    print(f"    {name}: ERROR - {e}")

        self.checkpoints.append({
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': delta
        })

        print(f"{'='*80}\n")

        return current_memory

    def print_summary(self):
        """Print summary of all checkpoints"""
        print(f"\n{'='*80}")
        print("MEMORY PROFILING SUMMARY")
        print(f"{'='*80}")
        print(f"Baseline: {self.baseline_memory:.2f} MB\n")

        for i, cp in enumerate(self.checkpoints):
            print(f"{i+1}. {cp['label']}")
            print(f"   Memory: {cp['memory_mb']:.2f} MB (+{cp['delta_mb']:.2f} MB)")

            if i > 0:
                delta_from_prev = cp['memory_mb'] - self.checkpoints[i-1]['memory_mb']
                print(f"   Change from previous: {delta_from_prev:+.2f} MB")
            print()

        print(f"{'='*80}\n")

    def get_top_memory_consumers(self, limit=20):
        """Get top memory consuming objects"""
        if self.tracker:
            print(f"\n[TOP MEMORY CONSUMERS]")
            self.tracker.print_diff()
        else:
            print(f"\n[TOP MEMORY CONSUMERS] Not available without pympler")

    def log_learning_transfer(self, trial_idx: int, env_id: int, learning_type: str, content: any, source: str = "unknown"):
        """
        Log learning transfer events

        Args:
            trial_idx: Current trial number
            env_id: Environment ID
            learning_type: Type of learning (working_reflexion, env_memory, global_memory, etc.)
            content: The actual learning content
            source: Where this learning came from (trial number, file, etc.)
        """
        log_entry = {
            'trial': trial_idx,
            'env_id': env_id,
            'type': learning_type,
            'source': source,
            'content_summary': self._summarize_content(content),
            'content_size': len(str(content)) if content else 0
        }

        self.learning_log.append(log_entry)

        # Also update trial summary
        trial_key = f"trial_{trial_idx}"
        if learning_type == 'working_reflexion':
            if env_id not in self.trial_learning_summary[trial_key]['working_reflexions']:
                self.trial_learning_summary[trial_key]['working_reflexions'][env_id] = []
            self.trial_learning_summary[trial_key]['working_reflexions'][env_id].append(log_entry)
        elif learning_type == 'env_memory':
            if env_id not in self.trial_learning_summary[trial_key]['env_memories']:
                self.trial_learning_summary[trial_key]['env_memories'][env_id] = []
            self.trial_learning_summary[trial_key]['env_memories'][env_id].append(log_entry)
        elif learning_type == 'global_memory':
            self.trial_learning_summary[trial_key]['global_memory_items'] += 1
        elif learning_type == 'universal_memory':
            self.trial_learning_summary[trial_key]['universal_memory_items'] += 1

    def _summarize_content(self, content: any) -> str:
        """Create a brief summary of learning content"""
        if isinstance(content, dict):
            keys = list(content.keys())[:5]
            return f"Dict with keys: {keys}"
        elif isinstance(content, list):
            return f"List with {len(content)} items"
        elif isinstance(content, str):
            return content[:100] + "..." if len(content) > 100 else content
        else:
            return str(type(content))

    def print_learning_transfer_summary(self, trial_idx: int):
        """Print summary of learning transferred in this trial"""
        trial_key = f"trial_{trial_idx}"
        summary = self.trial_learning_summary[trial_key]

        print(f"\n{'='*80}")
        print(f"[LEARNING TRANSFER SUMMARY] Trial {trial_idx}")
        print(f"{'='*80}")

        print(f"\nWorking Reflexions (within-trial learning):")
        for env_id, reflexions in summary['working_reflexions'].items():
            print(f"  ENV {env_id}: {len(reflexions)} reflexions")
            for i, refl in enumerate(reflexions[:3]):  # Show first 3
                print(f"    {i+1}. From {refl['source']}: {refl['content_summary']}")

        print(f"\nEnvironment Memories (episodic):")
        for env_id, memories in summary['env_memories'].items():
            print(f"  ENV {env_id}: {len(memories)} memories")
            for i, mem in enumerate(memories[:3]):  # Show first 3
                print(f"    {i+1}. From {mem['source']}: {mem['content_summary']}")

        print(f"\nGlobal Memory Items: {summary['global_memory_items']}")
        print(f"Universal Memory Items: {summary['universal_memory_items']}")
        print(f"{'='*80}\n")

    def export_learning_log(self, filepath: str):
        """Export detailed learning log to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'learning_events': self.learning_log,
                'trial_summaries': dict(self.trial_learning_summary)
            }, f, indent=2)
        print(f"[PROFILER] Learning log exported to {filepath}")


# Global profiler instance
profiler = DetailedMemoryProfiler()
