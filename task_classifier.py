"""
Task Classification System for ALFWorld
Classifies tasks into families to enable smart knowledge transfer
"""

import re
from typing import Dict, List, Set, Tuple


class TaskClassifier:
    """Classifies ALFWorld tasks into families and extracts properties"""

    # Task family patterns
    TASK_FAMILIES = {
        'cool_and_place': [
            r'cool.*and.*put.*in',
            r'cool.*and.*place',
        ],
        'heat_and_place': [
            r'heat.*and.*put.*in',
            r'heat.*and.*place',
        ],
        'clean_and_place': [
            r'clean.*and.*put.*in',
            r'clean.*and.*place',
        ],
        'multi_object_place': [
            r'put\s+two.*in',
            r'put\s+three.*in',
            r'place\s+two.*in',
        ],
        'examine_with_light': [
            r'look.*at.*under',
            r'examine.*under.*lamp',
        ],
        'simple_place': [
            r'put\s+(?:some|a)\s+\w+\s+(?:on|in)',
            r'place\s+(?:some|a)\s+\w+\s+(?:on|in)',
        ],
        'slice_and_place': [
            r'slice.*and.*put',
            r'slice.*and.*place',
        ],
    }

    # Object types
    TEMPERATURE_OBJECTS = {'pan', 'pot', 'mug', 'cup', 'plate', 'bowl'}
    FOOD_OBJECTS = {'apple', 'potato', 'tomato', 'lettuce', 'egg', 'bread'}
    SMALL_OBJECTS = {'pen', 'pencil', 'cd', 'creditcard', 'keychain', 'watch'}
    SOFT_OBJECTS = {'pillow', 'cloth', 'towel'}

    # Container types
    STORAGE_CONTAINERS = {'cabinet', 'drawer', 'shelf', 'safe', 'box'}
    TEMPERATURE_CONTAINERS = {'fridge', 'microwave'}
    FURNITURE = {'desk', 'sidetable', 'coffeetable', 'dresser'}
    SURFACES = {'countertop', 'sofa', 'bed', 'armchair'}

    def classify_task(self, task_string: str) -> Dict:
        """
        Classify a task and extract its properties

        Returns:
            {
                'family': str,  # Task family name
                'objects': List[str],  # Required objects
                'containers': List[str],  # Target containers
                'requires_temperature': bool,
                'requires_multi_object': bool,
                'action_types': Set[str]  # Required actions (heat, cool, clean, etc.)
            }
        """
        task_lower = task_string.lower()

        # Determine task family
        family = self._detect_family(task_lower)

        # Extract objects and containers
        objects = self._extract_objects(task_lower)
        containers = self._extract_containers(task_lower)

        # Determine requirements
        requires_temperature = self._requires_temperature(task_lower)
        requires_multi_object = self._requires_multi_objects(task_lower)
        action_types = self._extract_action_types(task_lower)

        return {
            'family': family,
            'objects': objects,
            'containers': containers,
            'requires_temperature': requires_temperature,
            'requires_multi_object': requires_multi_object,
            'action_types': action_types,
            'task_string': task_string
        }

    def _detect_family(self, task_lower: str) -> str:
        """Detect which task family this belongs to"""
        for family, patterns in self.TASK_FAMILIES.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    return family

        # Default family
        return 'simple_place'

    def _extract_objects(self, task_lower: str) -> List[str]:
        """Extract required objects from task"""
        objects = []

        # Common object patterns
        all_objects = (self.TEMPERATURE_OBJECTS | self.FOOD_OBJECTS |
                      self.SMALL_OBJECTS | self.SOFT_OBJECTS)

        for obj in all_objects:
            if obj in task_lower:
                objects.append(obj)

        # Extract other nouns that might be objects
        # Pattern: "put/place/cool/heat OBJECT"
        obj_matches = re.findall(r'(?:put|place|cool|heat|clean|slice)\s+(?:some|a|two|three)?\s+(\w+)', task_lower)
        for match in obj_matches:
            if match not in objects and match not in ['in', 'on', 'and', 'the']:
                objects.append(match)

        return list(set(objects))  # Remove duplicates

    def _extract_containers(self, task_lower: str) -> List[str]:
        """Extract target containers from task"""
        containers = []

        # All container types
        all_containers = (self.STORAGE_CONTAINERS | self.TEMPERATURE_CONTAINERS |
                         self.FURNITURE | self.SURFACES)

        for container in all_containers:
            if container in task_lower:
                containers.append(container)

        # Pattern: "in/on CONTAINER"
        container_matches = re.findall(r'(?:in|on)\s+(\w+)', task_lower)
        for match in container_matches:
            if match not in containers and len(match) > 3:
                containers.append(match)

        return list(set(containers))

    def _requires_temperature(self, task_lower: str) -> bool:
        """Check if task requires heating or cooling"""
        return bool(re.search(r'\b(heat|cool|warm|cold)\b', task_lower))

    def _requires_multi_objects(self, task_lower: str) -> bool:
        """Check if task requires multiple objects"""
        return bool(re.search(r'\b(two|three|multiple)\b', task_lower))

    def _extract_action_types(self, task_lower: str) -> Set[str]:
        """Extract required action types"""
        actions = set()

        action_keywords = {
            'heat': r'\bheat\b',
            'cool': r'\bcool\b',
            'clean': r'\bclean\b',
            'slice': r'\bslice\b',
            'examine': r'\b(look|examine)\b',
            'place': r'\b(put|place)\b',
        }

        for action, pattern in action_keywords.items():
            if re.search(pattern, task_lower):
                actions.add(action)

        return actions

    def calculate_task_similarity(self, task1: str, task2: str) -> float:
        """
        Calculate similarity between two tasks based on their classifications

        Returns: 0.0 to 1.0, where 1.0 is identical
        """
        class1 = self.classify_task(task1)
        class2 = self.classify_task(task2)

        similarity = 0.0

        # Family similarity (most important)
        if class1['family'] == class2['family']:
            similarity += 0.5  # Same family = 50% similarity
        elif self._are_compatible_families(class1['family'], class2['family']):
            similarity += 0.25  # Compatible families = 25%

        # Temperature requirement similarity
        if class1['requires_temperature'] == class2['requires_temperature']:
            similarity += 0.15

        # Multi-object requirement similarity
        if class1['requires_multi_object'] == class2['requires_multi_object']:
            similarity += 0.10

        # Action type overlap
        action_overlap = len(class1['action_types'] & class2['action_types'])
        action_union = len(class1['action_types'] | class2['action_types'])
        if action_union > 0:
            similarity += 0.15 * (action_overlap / action_union)

        # Container type overlap
        container_overlap = len(set(class1['containers']) & set(class2['containers']))
        if container_overlap > 0:
            similarity += 0.10

        return min(similarity, 1.0)

    def _are_compatible_families(self, family1: str, family2: str) -> bool:
        """Check if two task families are compatible for knowledge transfer"""
        # Temperature manipulation families are compatible
        temp_families = {'cool_and_place', 'heat_and_place'}
        if family1 in temp_families and family2 in temp_families:
            return True

        # Placement families are compatible
        place_families = {'simple_place', 'multi_object_place'}
        if family1 in place_families and family2 in place_families:
            return True

        # Manipulation families are compatible
        manip_families = {'clean_and_place', 'slice_and_place'}
        if family1 in manip_families and family2 in manip_families:
            return True

        return False


# Global instance
task_classifier = TaskClassifier()
