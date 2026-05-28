"""
Universal Knowledge Classification System

Classifies knowledge as universal, env-family, or task-specific
to prevent cross-contamination during knowledge transfer.

UPDATED: Now supports visual environments (OSWorld, etc.) without
hardcoded ALFWorld-specific patterns. Uses LLM-based classification
for ambiguous cases.
"""

import re
import os
from typing import Tuple, Optional, List

# Import task_classifier if available
try:
    from task_classifier import task_classifier
except ImportError:
    task_classifier = None


class UniversalKnowledgeClassifier:
    """
    Environment-agnostic knowledge classification.

    Classifies knowledge to determine transferability across tasks and environments.
    Uses a combination of pattern matching (for clear cases) and LLM-based
    classification (for ambiguous cases).
    """

    # Universal strategy patterns (transferable to ANY environment)
    # These are meta-cognitive strategies, not domain-specific rules
    UNIVERSAL_PATTERNS = [
        r'avoid.*repeat(?:ing|ed)?',
        r'systematically.*(?:check|search|explore)',
        r'(?:verify|check).*(?:state|result|output)',
        r'minimize.*(?:navigation|steps|actions)',
        r'don\'t.*(?:waste|repeat|retry)',
        r'first.*then',
        r'before.*(?:proceed|continue|move)',
        r'after.*(?:complete|finish|succeed)',
        r'if.*fail.*try',
        r'when.*stuck.*(?:try|consider|look)',
        r'pattern:.*(?:navigation|exploration|interaction)',
        r'strategy:.*',
        r'principle:.*',
        r'always.*(?:check|verify|confirm)',
        r'never.*(?:assume|skip|ignore)',
        r'ensure.*before',
        r'wait.*(?:for|until)',
        r'break.*into.*steps',
        r'track.*(?:progress|state|changes)',
    ]

    # Clearly task-specific indicators (never transfer)
    # These patterns indicate knowledge tied to specific instances
    SPECIFIC_INDICATORS = [
        r'at\s*\(\d+,\s*\d+\)',           # Pixel coordinates
        r'at position\s*\d+',             # Position indices
        r'in\s+(?:slot|cell|index)\s+\d+', # Specific slots
        r'/[\w/]+/[\w.]+',                # File paths
        r'id[=:]\s*\d+',                  # Specific IDs
        r'step\s+\d+',                    # Specific step numbers
        r'try\s+\d+',                     # Specific trial numbers
        r'(?:cabinet|drawer|shelf|burner|counter)\s+\d+',  # ALFWorld specific locations
        r'(?:pan|mug|pillow|apple|potato|tomato|cd|plate|bowl)\s+\d+',  # ALFWorld specific objects
        r'at\s+(?:the\s+)?[\w]+\s+\d+',   # Generic "at X N" patterns
    ]

    # Environment family patterns (transferable within similar environments)
    ENV_FAMILY_PATTERNS = {
        'gui_desktop': [
            r'\b(?:click|double[-\s]?click|right[-\s]?click)\b',
            r'\b(?:menu|toolbar|button|icon|window|dialog)\b',
            r'\bhotkey\b.*(?:ctrl|alt|shift|cmd)',
            r'\b(?:scroll|drag|drop)\b',
        ],
        'text_adventure': [
            r'\b(?:examine|look|inventory|take|drop|put)\b',
            r'\b(?:go|north|south|east|west|up|down)\b',
            r'\b(?:open|close|unlock|use)\b',
        ],
        'temperature_tasks': [
            r'\b(?:heat|cool|warm|cold|freeze|microwave|fridge|oven)\b',
        ],
        'file_operations': [
            r'\b(?:file|folder|directory|copy|move|delete|rename)\b',
            r'\b(?:save|load|open|close|create)\b.*(?:file|document)',
        ],
        'web_interaction': [
            r'\b(?:browser|tab|url|link|search|navigate)\b',
            r'\b(?:login|submit|form|input)\b',
        ],
    }

    def __init__(self, use_llm_fallback: bool = True):
        """
        Args:
            use_llm_fallback: Whether to use LLM for ambiguous classifications
        """
        self.use_llm_fallback = use_llm_fallback
        self._llm = None  # Lazy initialization

    def classify_knowledge(
        self,
        knowledge_text: str,
        source_task: str,
        source_env: str = None
    ) -> Tuple[str, Optional[str]]:
        """
        Classify knowledge as universal, env_family, or task_specific.

        Args:
            knowledge_text: The knowledge item to classify
            source_task: The task this knowledge came from
            source_env: The environment type (optional, helps classification)

        Returns:
            ('universal', None) or
            ('env_family', 'family_name') or
            ('task_specific', None)
        """
        knowledge_lower = knowledge_text.lower()

        # 1. Check for clearly task-specific indicators (fast path: don't transfer)
        if self._is_clearly_specific(knowledge_lower):
            return ('task_specific', None)

        # 2. Check for universal strategies (fast path: always transfer)
        if self._is_universal_strategy(knowledge_lower):
            return ('universal', None)

        # 3. Check for environment family patterns
        family = self._detect_env_family(knowledge_lower, source_env)
        if family:
            return ('env_family', family)

        # 4. LLM-based classification for ambiguous cases
        if self.use_llm_fallback:
            return self._llm_classify(knowledge_text, source_task, source_env)

        # 5. Default: conservative (task-specific)
        return ('task_specific', None)

    def _is_clearly_specific(self, text: str) -> bool:
        """Check if text contains clearly task-specific references."""
        for pattern in self.SPECIFIC_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_universal_strategy(self, text: str) -> bool:
        """Check if text is a universal meta-strategy."""
        for pattern in self.UNIVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_env_family(self, text: str, source_env: str = None) -> Optional[str]:
        """Detect if knowledge belongs to an environment family."""
        # If source_env is provided, prioritize its family
        if source_env:
            source_env_lower = source_env.lower()
            if 'osworld' in source_env_lower or 'gui' in source_env_lower or 'desktop' in source_env_lower:
                return 'gui_desktop'
            elif 'alfworld' in source_env_lower or 'textworld' in source_env_lower:
                return 'text_adventure'
            elif 'web' in source_env_lower or 'browser' in source_env_lower:
                return 'web_interaction'

        # Check pattern matches
        for family, patterns in self.ENV_FAMILY_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if matches >= 2:  # Require at least 2 pattern matches
                return family

        return None

    def _llm_classify(
        self,
        knowledge: str,
        task: str,
        env: str = None
    ) -> Tuple[str, Optional[str]]:
        """
        Use LLM to classify ambiguous knowledge.
        Falls back to 'task_specific' if LLM unavailable.
        """
        try:
            # Lazy load LLM
            if self._llm is None:
                if os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
                    from shared_model_gemini import fast_model
                    self._llm = fast_model
                elif os.getenv("MODEL_PROVIDER", "openai").lower() == "openrouter":
                    from shared_model_openrouter import fast_model
                    self._llm = fast_model
                elif os.getenv("MODEL_PROVIDER", "openai").lower() == "vllm":
                    from shared_model_vllm import fast_model
                    self._llm = fast_model
                else:
                    from shared_model import fast_model
                    self._llm = fast_model

            prompt = f"""Classify this learned knowledge for transferability:

KNOWLEDGE: "{knowledge}"
SOURCE TASK: "{task}"
SOURCE ENVIRONMENT: "{env or 'unknown'}"

Classification options:
A) UNIVERSAL - This is a general strategy that applies to ANY task in ANY environment
   Examples: "always verify state before proceeding", "break complex tasks into steps"

B) ENV_FAMILY - This applies to similar environments but not all
   Examples: "use Ctrl+S to save in GUI apps", "examine objects before taking them"

C) TASK_SPECIFIC - This only applies to this exact task or contains specific details
   Examples: "the file is in folder X", "click button at position (100, 200)"

Respond with ONLY the letter (A, B, or C):"""

            from vllm import SamplingParams
            response = self._llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0.0))
            result = response[0].outputs[0].text.strip().upper()

            if 'A' in result:
                return ('universal', None)
            elif 'B' in result:
                return ('env_family', 'llm_detected')
            else:
                return ('task_specific', None)

        except Exception as e:
            # Fallback to conservative classification
            print(f"[KnowledgeClassifier] LLM classification failed: {e}")
            return ('task_specific', None)

    def abstract_knowledge(self, knowledge_text: str, target_task: str = None) -> Optional[str]:
        """
        Abstract task-specific details from knowledge.

        Args:
            knowledge_text: Knowledge to abstract
            target_task: Task it's being transferred to (optional)

        Returns:
            Abstracted knowledge or None if too specific
        """
        abstract = knowledge_text

        # Replace specific coordinates with placeholders
        abstract = re.sub(r'\(\d+,\s*\d+\)', '(TARGET_POSITION)', abstract)
        abstract = re.sub(r'position\s+\d+', 'position TARGET_POSITION', abstract)

        # Replace file paths with placeholders
        abstract = re.sub(r'/[\w/]+/[\w.]+', 'TARGET_PATH', abstract)

        # Replace numbered objects/locations (ALFWorld style)
        abstract = re.sub(r'\b(\w+)\s+\d+\b', r'\1 TARGET_N', abstract)

        # Replace specific IDs
        abstract = re.sub(r'id[=:]\s*\d+', 'id=TARGET_ID', abstract)

        # Count placeholders
        placeholder_count = (
            abstract.count('TARGET_POSITION') +
            abstract.count('TARGET_PATH') +
            abstract.count('TARGET_N') +
            abstract.count('TARGET_ID')
        )

        # If too many placeholders, knowledge is too specific to be useful
        if placeholder_count > 4:
            return None

        return abstract

    def should_transfer_knowledge(
        self,
        knowledge_text: str,
        source_task: str,
        target_task: str,
        classification: Tuple[str, Optional[str]],
        source_env: str = None,
        target_env: str = None
    ) -> Tuple[bool, float]:
        """
        Determine if knowledge should be transferred and with what confidence.

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

        # Environment family: transfer if environments are similar
        if knowledge_type == 'env_family':
            # Check if source and target environments are in same family
            if source_env and target_env:
                source_family = self._detect_env_family("", source_env)
                target_family = self._detect_env_family("", target_env)
                if source_family == target_family:
                    return (True, 0.8)
                else:
                    return (False, 0.2)

            # Without environment info, use task similarity if available
            if task_classifier:
                similarity = task_classifier.calculate_task_similarity(source_task, target_task)
                if similarity >= 0.4:
                    return (True, similarity)
                else:
                    return (False, similarity)

            # Conservative default
            return (False, 0.3)

        return (False, 0.0)

    def validate_no_contamination(
        self,
        knowledge_text: str,
        target_task: str,
        target_env: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Final validation to ensure no cross-contamination.

        Returns:
            (is_safe: bool, reason: Optional[str])
        """
        knowledge_lower = knowledge_text.lower()

        # Check for specific coordinate references (never safe to transfer)
        if re.search(r'\(\d+,\s*\d+\)', knowledge_lower):
            return (False, "Knowledge contains specific coordinates")

        # Check for file path references
        if re.search(r'/[\w/]+/[\w.]+', knowledge_lower):
            return (False, "Knowledge contains specific file paths")

        # Check for numbered object/location references
        if re.search(r'\b(?:object|item|element|button|field)\s+\d+\b', knowledge_lower):
            return (False, "Knowledge contains numbered element references")

        # Check for ID references
        if re.search(r'\bid[=:]\s*\d+', knowledge_lower):
            return (False, "Knowledge contains specific ID references")

        return (True, None)


# Backward compatibility: create both old and new names
class KnowledgeClassifier(UniversalKnowledgeClassifier):
    """Alias for backward compatibility."""
    pass


# Global instance
knowledge_classifier = UniversalKnowledgeClassifier(use_llm_fallback=True)
