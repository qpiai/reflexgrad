"""
Deep Memory Tracker - Find the REAL source of 219GB leak
Tracks object sizes with deep inspection to find exponential growth
"""

import sys
import gc
from collections import defaultdict

class DeepMemoryTracker:
    def __init__(self):
        self.checkpoints = []
        self.object_id_tracker = defaultdict(list)  # Track same object across checkpoints

    def deep_sizeof(self, obj, seen=None):
        """
        Recursively calculate size of an object and all its references
        This will help us find nested/circular references causing the leak
        """
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0  # Already counted (circular reference detected!)

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        # Recursively calculate for container types
        if isinstance(obj, dict):
            size += sum(self.deep_sizeof(k, seen) + self.deep_sizeof(v, seen)
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self.deep_sizeof(item, seen) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += self.deep_sizeof(obj.__dict__, seen)

        return size

    def checkpoint(self, label, objects_to_track):
        """Track specific objects and their deep sizes"""
        print(f"\n{'='*80}")
        print(f"[DEEP MEMORY TRACKER] {label}")
        print(f"{'='*80}")

        checkpoint_data = {
            'label': label,
            'objects': {}
        }

        for name, obj in objects_to_track.items():
            # Shallow size
            shallow_size = sys.getsizeof(obj)

            # Deep size (with circular reference detection)
            deep_size = self.deep_sizeof(obj)

            # Track object identity to see if same object is being reused
            obj_id = id(obj)
            self.object_id_tracker[name].append(obj_id)

            # Check if this is a NEW object or SAME object as before
            is_new_object = len(self.object_id_tracker[name]) == 1 or \
                           obj_id != self.object_id_tracker[name][-2]

            checkpoint_data['objects'][name] = {
                'shallow': shallow_size,
                'deep': deep_size,
                'type': type(obj).__name__,
                'id': obj_id,
                'is_new': is_new_object
            }

            print(f"\n{name}:")
            print(f"  Type: {type(obj).__name__}")
            print(f"  Shallow size: {shallow_size:,} bytes ({shallow_size/1024/1024:.2f} MB)")
            print(f"  Deep size: {deep_size:,} bytes ({deep_size/1024/1024:.2f} MB)")
            print(f"  Ratio (deep/shallow): {deep_size/shallow_size if shallow_size > 0 else 0:.1f}x")
            print(f"  Object ID: {obj_id}")
            print(f"  Is new object: {is_new_object}")

            # Additional details for specific types
            if isinstance(obj, (list, tuple)):
                print(f"  Length: {len(obj)}")
                if len(obj) > 0:
                    first_item_size = sys.getsizeof(obj[0]) if obj else 0
                    print(f"  First item size: {first_item_size:,} bytes")
                    print(f"  Estimated total: ~{first_item_size * len(obj):,} bytes")

                    # GRANULAR BREAKDOWN: If this is env_states, break down each dict field
                    if name == 'env_states' and isinstance(obj[0], dict):
                        print(f"\n  [GRANULAR BREAKDOWN of env_states[0] fields]:")
                        state = obj[0]
                        field_sizes = []
                        for field_name, field_value in state.items():
                            field_deep = self.deep_sizeof(field_value)
                            field_sizes.append((field_name, field_deep))

                        # Sort by size descending
                        field_sizes.sort(key=lambda x: x[1], reverse=True)

                        # Show top 10 largest fields
                        print(f"  Top 10 largest fields in env_states[0]:")
                        for i, (field_name, field_deep) in enumerate(field_sizes[:10]):
                            field_shallow = sys.getsizeof(state[field_name])
                            field_ratio = field_deep / field_shallow if field_shallow > 0 else 0
                            print(f"    {i+1}. {field_name}: {field_deep:,} bytes ({field_deep/1024:.1f} KB), ratio: {field_ratio:.1f}x")

            elif isinstance(obj, dict):
                print(f"  Keys count: {len(obj)}")
                if len(obj) > 0:
                    sample_key = list(obj.keys())[0]
                    sample_val = obj[sample_key]
                    sample_size = sys.getsizeof(sample_key) + sys.getsizeof(sample_val)
                    print(f"  Sample entry size: {sample_size:,} bytes")
                    print(f"  Estimated total: ~{sample_size * len(obj):,} bytes")

        self.checkpoints.append(checkpoint_data)

        # Compare with previous checkpoint
        if len(self.checkpoints) > 1:
            print(f"\n[GROWTH ANALYSIS]")
            prev = self.checkpoints[-2]
            curr = self.checkpoints[-1]

            for name in curr['objects']:
                if name in prev['objects']:
                    prev_deep = prev['objects'][name]['deep']
                    curr_deep = curr['objects'][name]['deep']
                    growth = curr_deep - prev_deep
                    growth_pct = (growth / prev_deep * 100) if prev_deep > 0 else 0

                    if growth > 1024 * 1024:  # Only show if grew by > 1MB
                        print(f"  {name}: +{growth:,} bytes (+{growth_pct:.1f}%)")

        print(f"{'='*80}\n")

        # Force garbage collection and report
        collected = gc.collect()
        if collected > 0:
            print(f"[GC] Collected {collected} objects\n")

    def get_growth_summary(self):
        """Return summary of which objects grew the most"""
        if len(self.checkpoints) < 2:
            return "Not enough checkpoints for comparison"

        first = self.checkpoints[0]
        last = self.checkpoints[-1]

        growth_summary = []
        for name in last['objects']:
            if name in first['objects']:
                growth = last['objects'][name]['deep'] - first['objects'][name]['deep']
                growth_summary.append((name, growth))

        growth_summary.sort(key=lambda x: x[1], reverse=True)
        return growth_summary

# Global tracker instance
tracker = DeepMemoryTracker()
