"""
Knowledge Classification System
Classifies knowledge as universal, task-family, or task-specific
to prevent cross-contamination during knowledge transfer
"""

import re
from typing import Tuple, Optional
from task_classifier import task_classifier


class KnowledgeClassifier:
    """Classify knowledge items to determine transferability"""

    # Universal strategy patterns (always transferable)
    UNIVERSAL_PATTERNS = [
        r'avoid.*repeat(?:ing|ed)?',
        r'systematically.*(?:check|search|explore)',
        r'open.*before.*(?:examine|take|get)',
        r'(?:verify|check).*state',
        r'minimize.*navigation',
        r'(?:must|should|need to).*(?:be at|go to).*(?:location|place)',
        r'track.*(?:inventory|items|objects)',
        r'don\'t.*(?:waste|repeat)',
        r'first.*then',
        r'pattern:.*(?:navigation|exploration)',
    ]

    # Task-family specific patterns
    TEMPERATURE_PATTERNS = [
        r'\b(?:fridge|microwave|stove|oven)\b',
        r'\b(?:heat|cool|warm|cold|temperature)\b',
        r'use.*(?:for cooling|for heating|to cool|to heat)',
    ]

    PLACEMENT_PATTERNS = [
        r'navigate.*(?:before|first)',
        r'(?:take|pick up).*before.*(?:place|put)',
        r'must be.*(?:at|near).*(?:to place|to put)',
    ]

    MULTI_OBJECT_PATTERNS = [
        r'count.*(?:objects|items)',
        r'(?:first|second|third).*(?:object|item)',
        r'(?:both|all).*(?:objects|items)',
    ]

    # Specific object/location indicators (task-specific, don't transfer)
    SPECIFIC_INDICATORS = [
        r'\b(?:pan|mug|pillow|apple|potato|tomato|cd|plate|bowl)\s+\d+\b',
        r'\b(?:cabinet|drawer|shelf|stoveburner|countertop|fridge)\s+\d+\b',
        r'at\s+(?:cabinet|drawer|shelf|stoveburner)\s+\d+',
        r'(?:go to|found at|located at).*\d+',
    ]

    def classify_knowledge(self, knowledge_text: str, source_task: str) -> Tuple[str, Optional[str]]:
        """
        Classify knowledge as universal, task_family, or task_specific

        Args:
            knowledge_text: The knowledge item to classify
            source_task: The task this knowledge came from

        Returns:
            ('universal', None) or
            ('task_family', 'family_name') or
            ('task_specific', None)
        """
        knowledge_lower = knowledge_text.lower()

        # Check for task-specific indicators FIRST (most restrictive)
        if self._contains_specific_objects(knowledge_lower):
            return ('task_specific', None)

        # Check for universal strategies
        if self._is_universal_strategy(knowledge_lower):
            return ('universal', None)

        # Check for task family patterns
        family = self._detect_task_family_pattern(knowledge_lower, source_task)
        if family:
            return ('task_family', family)

        # Default to task-specific (conservative approach)
        return ('task_specific', None)

    def _contains_specific_objects(self, text: str) -> bool:
        """Check if text contains specific object or location references"""
        for pattern in self.SPECIFIC_INDICATORS:
            if re.search(pattern, text):
                return True
        return False

    def _is_universal_strategy(self, text: str) -> bool:
        """Check if text is a universal meta-strategy"""
        for pattern in self.UNIVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_task_family_pattern(self, text: str, source_task: str) -> Optional[str]:
        """Detect if knowledge is task-family specific"""

        # Temperature manipulation
        if any(re.search(p, text) for p in self.TEMPERATURE_PATTERNS):
            task_class = task_classifier.classify_task(source_task)
            if task_class['requires_temperature']:
                return task_class['family']

        # Placement patterns
        if any(re.search(p, text) for p in self.PLACEMENT_PATTERNS):
            return 'placement_tasks'  # Generic placement family

        # Multi-object patterns
        if any(re.search(p, text) for p in self.MULTI_OBJECT_PATTERNS):
            task_class = task_classifier.classify_task(source_task)
            if task_class['requires_multi_object']:
                return 'multi_object_tasks'

        return None

    def abstract_knowledge(self, knowledge_text: str, target_task: str) -> Optional[str]:
        """
        Abstract task-specific details from knowledge

        Args:
            knowledge_text: Knowledge to abstract
            target_task: Task it's being transferred to

        Returns:
            Abstracted knowledge or None if too specific
        """
        abstract = knowledge_text

        # Replace specific objects with generic placeholders
        abstract = re.sub(r'\b(pan|mug|pillow|apple|potato)\s+\d+', 'TARGET_OBJECT', abstract)
        abstract = re.sub(r'\b(cabinet|drawer|shelf|stoveburner)\s+\d+', 'CONTAINER', abstract)
        abstract = re.sub(r'\b(countertop|desk|fridge|microwave)\s+\d+', 'LOCATION', abstract)

        # If too many placeholders, knowledge is too specific
        placeholder_count = abstract.count('TARGET_OBJECT') + abstract.count('CONTAINER') + abstract.count('LOCATION')
        if placeholder_count > 3:
            return None  # Too specific to be useful

        # If nothing was abstracted, it's already general
        if abstract == knowledge_text:
            return abstract

        # If some abstraction happened, return it
        return abstract

    def should_transfer_knowledge(
        self,
        knowledge_text: str,
        source_task: str,
        target_task: str,
        classification: Tuple[str, Optional[str]]
    ) -> Tuple[bool, float]:
        """
        Determine if knowledge should be transferred and with what confidence

        Returns:
            (should_transfer: bool, confidence: float)
        """
        knowledge_type, family = classification

        # Always transfer universal knowledge
        if knowledge_type == 'universal':
            return (True, 1.0)

        # Never transfer task-specific knowledge
        if knowledge_type == 'task_specific':
            return (False, 0.0)

        # Task family knowledge: transfer if tasks are similar
        if knowledge_type == 'task_family':
            task_similarity = task_classifier.calculate_task_similarity(source_task, target_task)

            # Require moderate similarity for family knowledge
            if task_similarity >= 0.4:
                return (True, task_similarity)
            else:
                return (False, task_similarity)

        return (False, 0.0)

    def validate_no_contamination(self, knowledge_text: str, target_task: str) -> Tuple[bool, Optional[str]]:
        """
        Final validation to ensure no cross-contamination

        Returns:
            (is_safe: bool, reason: Optional[str])
        """
        knowledge_lower = knowledge_text.lower()
        target_lower = target_task.lower()

        # Extract objects from knowledge
        knowledge_objects = set(re.findall(r'\b(pan|mug|pillow|apple|potato|tomato|cd|plate|bowl)\b', knowledge_lower))

        # Extract objects from target task
        target_objects = set(re.findall(r'\b(pan|mug|pillow|apple|potato|tomato|cd|plate|bowl)\b', target_lower))

        # Check for object mismatch
        for obj in knowledge_objects:
            if obj not in target_objects:
                return (False, f"Knowledge mentions '{obj}' not in target task")

        # Check for specific location references
        if re.search(r'\b(?:cabinet|drawer|shelf|stoveburner|countertop)\s+\d+', knowledge_lower):
            # Specific locations should not transfer
            return (False, "Knowledge contains specific location references")

        return (True, None)


# Global instance
knowledge_classifier = KnowledgeClassifier()
