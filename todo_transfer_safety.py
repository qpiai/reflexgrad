"""
Safe TODO Transfer System with Cross-Contamination Prevention

Transfers successful TODO patterns across environments within the same trial
while preventing cross-contamination through:
1. Task similarity checking (70%+ required)
2. Object abstraction (replace source object with target object)
3. Location sanitization (remove specific location numbers)
4. Action-task compatibility validation
"""

import re
from typing import List, Dict, Optional, Tuple
from task_classifier import task_classifier
from knowledge_classifier import knowledge_classifier


class TodoTransferSafety:
    """Safely transfer TODO patterns across environments"""

    # Specific objects that need abstraction
    SPECIFIC_OBJECTS = [
        'pan', 'mug', 'pillow', 'apple', 'potato', 'tomato', 'cd', 'plate',
        'bowl', 'cup', 'fork', 'knife', 'spoon', 'pen', 'pencil', 'book',
        'newspaper', 'watch', 'keychain', 'creditcard', 'cellphone', 'laptop'
    ]

    # Specific locations that need removal
    SPECIFIC_LOCATIONS = [
        'cabinet', 'drawer', 'shelf', 'stoveburner', 'countertop',
        'desk', 'dresser', 'sidetable', 'coffeetable', 'garbagecan'
    ]

    def __init__(self):
        self.transfer_log = []  # Track all transfer attempts

    def can_transfer_todo(
        self,
        source_task: str,
        target_task: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if TODOs can be transferred from source to target task

        Returns:
            (can_transfer: bool, reason: Optional[str])
        """
        # Check 1: Task similarity (stricter than action transfer)
        similarity = task_classifier.calculate_task_similarity(source_task, target_task)

        if similarity < 0.7:  # Require 70%+ similarity for TODO transfer
            return False, f"Task similarity too low: {similarity:.2f} < 0.70"

        # Check 2: Task family must match
        source_class = task_classifier.classify_task(source_task)
        target_class = task_classifier.classify_task(target_task)

        if source_class['family'] != target_class['family']:
            return False, f"Task family mismatch: {source_class['family']} != {target_class['family']}"

        # Check 3: Temperature requirements must match
        if source_class['requires_temperature'] != target_class['requires_temperature']:
            return False, "Temperature requirement mismatch"

        # Check 4: Multi-object requirements must match
        if source_class['requires_multi_object'] != target_class['requires_multi_object']:
            return False, "Multi-object requirement mismatch"

        return True, None

    def extract_main_object(self, task: str) -> Optional[str]:
        """
        Extract the main object from a task string using pattern matching

        Universal approach: looks for nouns after common verbs/prepositions
        No hardcoded object lists needed!
        """
        task_lower = task.lower()

        # Pattern 1: "put [a/some/two] OBJECT in/on"
        match = re.search(r'put\s+(?:a|an|some|two|three)?\s*(\w+)\s+(?:in|on)', task_lower)
        if match:
            return match.group(1)

        # Pattern 2: "cool/heat/clean [a/some] OBJECT"
        match = re.search(r'(?:cool|heat|clean|wash)\s+(?:a|an|some)?\s*(\w+)', task_lower)
        if match:
            return match.group(1)

        # Pattern 3: "examine OBJECT with"
        match = re.search(r'examine\s+(?:a|an|the)?\s*(\w+)\s+with', task_lower)
        if match:
            return match.group(1)

        # Pattern 4: "find OBJECT"
        match = re.search(r'find\s+(?:a|an|the|some)?\s*(\w+)', task_lower)
        if match:
            return match.group(1)

        return None

    def extract_target_location(self, task: str) -> Optional[str]:
        """
        Extract target location from task using pattern matching

        Universal approach: looks for locations after in/on prepositions
        """
        task_lower = task.lower()

        # Pattern: "in/on LOCATION"
        match = re.search(r'(?:in|on)\s+(?:a|the)?\s*(\w+)', task_lower)
        if match:
            return match.group(1)

        return None

    def abstract_todo(
        self,
        todo_text: str,
        source_task: str,
        target_task: str
    ) -> str:
        """
        Abstract a TODO by replacing specific details

        Example:
            "Find a pan" + (source: "cool pan", target: "cool apple")
            → "Find an apple"
        """
        abstracted = todo_text

        # Step 1: Replace source object with target object
        source_obj = self.extract_main_object(source_task)
        target_obj = self.extract_main_object(target_task)

        if source_obj and target_obj and source_obj != target_obj:
            # Replace with proper article handling
            # "Find a pan" → "Find an apple"
            # "Pick up the pan" → "Pick up the apple"
            abstracted = re.sub(
                rf'\b(a |an |the |some ){source_obj}(s)?\b',
                lambda m: f"{m.group(1)}{target_obj}{m.group(2) or ''}",
                abstracted,
                flags=re.IGNORECASE
            )

            # Also replace standalone object
            abstracted = re.sub(
                rf'\b{source_obj}(s)?\b',
                f"{target_obj}\\1",
                abstracted,
                flags=re.IGNORECASE
            )

        # Step 1b: Replace source location with target location
        source_loc = self.extract_target_location(source_task)
        target_loc = self.extract_target_location(target_task)

        if source_loc and target_loc and source_loc != target_loc:
            # Replace location references
            # "in cabinet" → "in sofa"
            # "on countertop" → "on desk"
            abstracted = re.sub(
                rf'\b(in |on |at |to )?{source_loc}(s)?\b',
                lambda m: f"{m.group(1) or ''}{target_loc}{m.group(2) or ''}",
                abstracted,
                flags=re.IGNORECASE
            )

        # Step 2: Remove specific location numbers (keep location type)
        # "Go to cabinet 3" → "Go to a cabinet"
        for location in self.SPECIFIC_LOCATIONS:
            abstracted = re.sub(
                rf'\b{location}\s+\d+\b',
                f'a {location}',
                abstracted,
                flags=re.IGNORECASE
            )

        # Step 3: Remove numbered objects
        # "pan 1" → "pan"
        for obj in self.SPECIFIC_OBJECTS:
            abstracted = re.sub(
                rf'\b{obj}\s+\d+\b',
                obj,
                abstracted,
                flags=re.IGNORECASE
            )

        return abstracted

    def validate_todo_safety(
        self,
        abstracted_todo: str,
        target_task: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Final safety validation before using an abstracted TODO

        Returns:
            (is_safe: bool, reason: Optional[str])
        """
        todo_lower = abstracted_todo.lower()
        task_lower = target_task.lower()

        # Check 1: No wrong objects mentioned
        target_obj = self.extract_main_object(target_task)

        for obj in self.SPECIFIC_OBJECTS:
            if obj in todo_lower:
                if target_obj and obj != target_obj:
                    return False, f"TODO mentions '{obj}' but task needs '{target_obj}'"

        # Check 2: No specific location numbers (should be removed)
        if re.search(r'(cabinet|drawer|shelf|stoveburner|countertop)\s+\d+', todo_lower):
            return False, "TODO contains specific location numbers"

        # Check 3: Action-task compatibility
        # Cooling actions
        if any(word in todo_lower for word in ['cool', 'fridge', 'chill']):
            if not any(word in task_lower for word in ['cool', 'cooled']):
                return False, "TODO wants to cool but task doesn't require cooling"

        # Heating actions
        if any(word in todo_lower for word in ['heat', 'microwave', 'stove', 'warm']):
            if not any(word in task_lower for word in ['hot', 'heated', 'warm']):
                return False, "TODO wants to heat but task doesn't require heating"

        # Cleaning actions
        if any(word in todo_lower for word in ['clean', 'wash', 'sinkbasin']):
            if 'clean' not in task_lower:
                return False, "TODO wants to clean but task doesn't require cleaning"

        # Examination actions
        if 'examine' in todo_lower or 'desklamp' in todo_lower:
            if 'examine' not in task_lower:
                return False, "TODO wants to examine but task doesn't require examination"

        return True, None

    def transfer_todos(
        self,
        successful_patterns: List[Dict],
        target_task: str
    ) -> List[str]:
        """
        Transfer TODOs from successful patterns to target task

        Args:
            successful_patterns: List of {source_task, todos, task_family}
            target_task: Task to transfer TODOs to

        Returns:
            List of safely abstracted TODO suggestions
        """
        transferred_todos = []

        for pattern in successful_patterns:
            source_task = pattern['source_task']
            source_todos = pattern['todos']

            # Safety Check 1: Can we transfer?
            can_transfer, reason = self.can_transfer_todo(source_task, target_task)

            if not can_transfer:
                self.transfer_log.append({
                    'source_task': source_task,
                    'target_task': target_task,
                    'action': 'BLOCKED',
                    'reason': reason,
                    'stage': 'task_similarity'
                })
                continue

            # Safety Check 2: Abstract and validate each TODO
            abstracted_todos = []
            for todo in source_todos:
                # Abstract the TODO
                abstracted = self.abstract_todo(todo, source_task, target_task)

                # Safety Check 3: Final validation
                is_safe, safety_reason = self.validate_todo_safety(abstracted, target_task)

                if not is_safe:
                    self.transfer_log.append({
                        'source_task': source_task,
                        'target_task': target_task,
                        'original_todo': todo,
                        'abstracted_todo': abstracted,
                        'action': 'BLOCKED',
                        'reason': safety_reason,
                        'stage': 'todo_validation'
                    })
                    continue

                abstracted_todos.append(abstracted)
                self.transfer_log.append({
                    'source_task': source_task,
                    'target_task': target_task,
                    'original_todo': todo,
                    'abstracted_todo': abstracted,
                    'action': 'TRANSFERRED',
                    'reason': 'Passed all safety checks',
                    'stage': 'success'
                })

            if abstracted_todos:
                transferred_todos.extend(abstracted_todos)

        return transferred_todos

    def get_transfer_stats(self) -> Dict:
        """Get statistics about transfer attempts"""
        total = len(self.transfer_log)
        if total == 0:
            return {'total': 0, 'transferred': 0, 'blocked': 0}

        transferred = sum(1 for log in self.transfer_log if log['action'] == 'TRANSFERRED')
        blocked = sum(1 for log in self.transfer_log if log['action'] == 'BLOCKED')

        blocked_reasons = {}
        for log in self.transfer_log:
            if log['action'] == 'BLOCKED':
                reason = log['reason']
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

        return {
            'total': total,
            'transferred': transferred,
            'blocked': blocked,
            'block_rate': blocked / total if total > 0 else 0,
            'blocked_reasons': blocked_reasons
        }


# Global instance
todo_transfer_safety = TodoTransferSafety()
