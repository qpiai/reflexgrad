"""
Learning Extraction System
Extracts transferable knowledge from completed environments
and classifies it for safe cross-environment transfer
"""

import re
from typing import Dict, List, Optional, Tuple
from knowledge_classifier import knowledge_classifier
from task_classifier import task_classifier


class LearningExtractor:
    """Extract and classify learnings from environment executions"""

    def extract_learnings(
        self,
        env_state: Dict,
        result: Dict,
        task: str
    ) -> List[Dict]:
        """
        Extract all learnings from a completed environment

        Args:
            env_state: State of the completed environment
            result: Execution result (success/failure, steps, etc.)
            task: The task string

        Returns:
            List of learning items with classifications
        """
        learnings = []

        # Extract from prompt generator components (TextGrad)
        if 'prompt_generator' in env_state:
            component_learnings = self._extract_from_components(
                env_state['prompt_generator'],
                task,
                result.get('success', False)
            )
            learnings.extend(component_learnings)

        # Extract from step reflexions
        if 'reflexions' in env_state:
            reflexion_learnings = self._extract_from_reflexions(
                env_state['reflexions'],
                task
            )
            learnings.extend(reflexion_learnings)

        # Extract successful action sequences (only if succeeded)
        if result.get('success', False) and 'trajectory' in result:
            sequence_learnings = self._extract_action_sequences(
                result['trajectory'],
                task
            )
            learnings.extend(sequence_learnings)

        # Extract failure patterns (only if failed)
        if not result.get('success', False):
            failure_learnings = self._extract_failure_patterns(
                env_state,
                result,
                task
            )
            learnings.extend(failure_learnings)

        # Classify all learnings
        classified_learnings = []
        for learning in learnings:
            classification = knowledge_classifier.classify_knowledge(
                learning['text'],
                task
            )

            # Validate no contamination potential
            is_safe, reason = knowledge_classifier.validate_no_contamination(
                learning['text'],
                task
            )

            classified_learnings.append({
                'text': learning['text'],
                'source': learning['source'],
                'source_task': task,
                'classification': classification,
                'is_safe': is_safe,
                'validation_reason': reason,
                'confidence': learning.get('confidence', 0.5)
            })

        return classified_learnings

    def _extract_from_components(
        self,
        prompt_generator,
        task: str,
        success: bool
    ) -> List[Dict]:
        """Extract knowledge from TextGrad prompt components"""
        learnings = []

        # Only extract from successful executions
        if not success:
            return learnings

        # Extract from adaptive strategy component
        if hasattr(prompt_generator, 'components'):
            components = prompt_generator.components

            if 'adaptive_strategy' in components:
                strategy = components['adaptive_strategy']
                if strategy and len(strategy.strip()) > 20:
                    learnings.append({
                        'text': strategy,
                        'source': 'textgrad_strategy',
                        'confidence': 0.8
                    })

            # Extract from error recovery component
            if 'error_recovery' in components:
                recovery = components['error_recovery']
                if recovery and len(recovery.strip()) > 20:
                    learnings.append({
                        'text': recovery,
                        'source': 'textgrad_recovery',
                        'confidence': 0.7
                    })

            # Extract from task decomposition
            if 'task_decomposition' in components:
                decomp = components['task_decomposition']
                if decomp and len(decomp.strip()) > 20:
                    # Check if it contains useful strategic patterns
                    if self._contains_strategic_pattern(decomp):
                        learnings.append({
                            'text': decomp,
                            'source': 'textgrad_decomposition',
                            'confidence': 0.75
                        })

        return learnings

    def _extract_from_reflexions(
        self,
        reflexions: List[str],
        task: str
    ) -> List[Dict]:
        """Extract strategic insights from step reflexions"""
        learnings = []

        for reflexion in reflexions:
            # Extract explicit strategy statements
            strategies = self._extract_strategies(reflexion)
            for strategy in strategies:
                learnings.append({
                    'text': strategy,
                    'source': 'step_reflexion',
                    'confidence': 0.6
                })

            # Extract pattern insights
            patterns = self._extract_patterns(reflexion)
            for pattern in patterns:
                learnings.append({
                    'text': pattern,
                    'source': 'reflexion_pattern',
                    'confidence': 0.65
                })

        return learnings

    def _extract_action_sequences(
        self,
        trajectory: List[Dict],
        task: str
    ) -> List[Dict]:
        """
        Extract successful action patterns from trajectory
        Only extract high-level patterns, not specific sequences
        """
        learnings = []

        # Extract navigation patterns (if systematic)
        nav_pattern = self._detect_navigation_pattern(trajectory)
        if nav_pattern:
            learnings.append({
                'text': nav_pattern,
                'source': 'action_sequence',
                'confidence': 0.7
            })

        # Extract task-specific workflow patterns
        workflow = self._detect_workflow_pattern(trajectory, task)
        if workflow:
            learnings.append({
                'text': workflow,
                'source': 'workflow_pattern',
                'confidence': 0.75
            })

        return learnings

    def _extract_failure_patterns(
        self,
        env_state: Dict,
        result: Dict,
        task: str
    ) -> List[Dict]:
        """Extract patterns to avoid from failures"""
        learnings = []

        # Extract repeated failed actions
        if 'trajectory' in result:
            repeated_failures = self._detect_repeated_failures(result['trajectory'])
            for failure in repeated_failures:
                learnings.append({
                    'text': f"Avoid: {failure}",
                    'source': 'failure_pattern',
                    'confidence': 0.6
                })

        return learnings

    def _contains_strategic_pattern(self, text: str) -> bool:
        """Check if text contains strategic patterns worth extracting"""
        strategic_indicators = [
            r'first.*then',
            r'must.*before',
            r'pattern:',
            r'strategy:',
            r'systematically',
            r'avoid.*repeat',
            r'check.*before',
        ]

        for pattern in strategic_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_strategies(self, text: str) -> List[str]:
        """Extract explicit strategy statements from reflexion text"""
        strategies = []

        # Pattern: "I must/should/need to X"
        must_patterns = re.findall(
            r'I (?:must|should|need to) ([^.!?]+)',
            text,
            re.IGNORECASE
        )
        for match in must_patterns:
            if len(match.strip()) > 15:  # Meaningful length
                strategies.append(f"Strategy: {match.strip()}")

        # Pattern: "Pattern: X"
        pattern_statements = re.findall(
            r'Pattern: ([^.!?]+)',
            text,
            re.IGNORECASE
        )
        for match in pattern_statements:
            if len(match.strip()) > 15:
                strategies.append(f"Pattern: {match.strip()}")

        return strategies

    def _extract_patterns(self, text: str) -> List[str]:
        """Extract behavioral patterns from reflexion"""
        patterns = []

        # Navigation patterns
        if re.search(r'(?:navigate|go to|check).*systematically', text, re.IGNORECASE):
            patterns.append("Pattern: Systematic navigation before action")

        # Verification patterns
        if re.search(r'(?:verify|check|confirm).*before.*(?:act|do|take)', text, re.IGNORECASE):
            patterns.append("Pattern: Verify state before acting")

        # Inventory patterns
        if re.search(r'(?:check|track|monitor).*inventory', text, re.IGNORECASE):
            patterns.append("Pattern: Track inventory state")

        return patterns

    def _detect_navigation_pattern(self, trajectory: List[Tuple]) -> Optional[str]:
        """Detect systematic navigation patterns"""
        # Count unique locations visited
        locations_visited = set()
        for step in trajectory:
            # Trajectory is list of tuples: (action, observation, reward)
            if isinstance(step, tuple) and len(step) >= 1:
                action = step[0]  # First element is action
            elif isinstance(step, dict):
                action = step.get('action', '')
            else:
                continue

            if action.startswith('go to'):
                location = action[6:].strip()
                # Remove numbers to detect pattern
                location_type = re.sub(r'\s+\d+$', '', location)
                locations_visited.add(location_type)

        # If systematically explored many locations
        if len(locations_visited) >= 3:
            return "Pattern: Systematically explore multiple locations"

        return None

    def _detect_workflow_pattern(
        self,
        trajectory: List[Tuple],
        task: str
    ) -> Optional[str]:
        """Detect high-level workflow patterns"""
        task_class = task_classifier.classify_task(task)

        # Extract actions from trajectory tuples
        actions = []
        for step in trajectory:
            if isinstance(step, tuple) and len(step) >= 1:
                actions.append(step[0])
            elif isinstance(step, dict):
                actions.append(step.get('action', ''))

        # Temperature workflow pattern
        if task_class['requires_temperature']:
            has_open = any('open' in a for a in actions)
            has_take = any('take' in a for a in actions)
            has_temp = any(a.startswith('heat') or a.startswith('cool') for a in actions)
            has_close = any('close' in a for a in actions)

            if has_open and has_take and has_temp:
                return "Workflow: Open container → Take object → Apply temperature → Place"

        # Multi-object workflow pattern
        if task_class['requires_multi_object']:
            take_count = sum(1 for a in actions if 'take' in a)
            if take_count >= 2:
                return "Workflow: Collect all required objects before placing"

        return None

    def _detect_repeated_failures(self, trajectory: List[Tuple]) -> List[str]:
        """Detect actions that were repeated and likely failed"""
        action_counts = {}

        for step in trajectory:
            # Extract action from tuple or dict
            if isinstance(step, tuple) and len(step) >= 1:
                action = step[0]
            elif isinstance(step, dict):
                action = step.get('action', '')
            else:
                continue

            # Generic action type (remove object IDs)
            generic_action = re.sub(r'\d+', 'X', action)
            action_counts[generic_action] = action_counts.get(generic_action, 0) + 1

        # Actions repeated >2 times likely indicate failure
        repeated = []
        for action, count in action_counts.items():
            if count > 2:
                repeated.append(action)

        return repeated

    def abstract_learning(
        self,
        learning: Dict,
        target_task: str
    ) -> Optional[Dict]:
        """
        Abstract a learning item for transfer to target task

        Args:
            learning: Learning item with classification
            target_task: Task it will be transferred to

        Returns:
            Abstracted learning or None if not transferable
        """
        classification = learning['classification']
        knowledge_type, family = classification

        # Universal knowledge: no abstraction needed
        if knowledge_type == 'universal':
            return learning

        # Task-specific: don't transfer
        if knowledge_type == 'task_specific':
            return None

        # Task-family: abstract and validate similarity
        if knowledge_type == 'task_family':
            source_task = learning['source_task']
            similarity = task_classifier.calculate_task_similarity(
                source_task,
                target_task
            )

            # Require minimum similarity
            if similarity < 0.4:
                return None

            # Abstract the knowledge
            abstracted_text = knowledge_classifier.abstract_knowledge(
                learning['text'],
                target_task
            )

            if abstracted_text is None:
                return None  # Too specific to abstract

            # Final contamination check
            is_safe, reason = knowledge_classifier.validate_no_contamination(
                abstracted_text,
                target_task
            )

            if not is_safe:
                return None

            # Return abstracted version
            return {
                **learning,
                'text': abstracted_text,
                'abstracted': True,
                'similarity': similarity,
                'confidence': learning['confidence'] * similarity
            }

        return None

    def filter_learnings_for_task(
        self,
        learnings: List[Dict],
        target_task: str,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Filter and abstract learnings for a specific target task

        Args:
            learnings: All available learnings
            target_task: Target task to filter for
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered and abstracted learnings safe for target task
        """
        filtered = []

        for learning in learnings:
            # Skip low confidence
            if learning.get('confidence', 0) < min_confidence:
                continue

            # Skip if validation failed
            if not learning.get('is_safe', True):
                continue

            # Abstract for target task
            abstracted = self.abstract_learning(learning, target_task)

            if abstracted is not None:
                filtered.append(abstracted)

        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return filtered


# Global instance
learning_extractor = LearningExtractor()
