"""
Task TODO Manager for Agent - Hierarchical Learning System

This module implements a task decomposition and tracking system that:
1. Breaks down high-level tasks into concrete subtasks
2. Tracks progress through pending/in_progress/completed states
3. Updates based on TextGrad progress scores (NO hardcoding)
4. Learns from reflexions across trials (strategic improvement)
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TodoItem:
    """A single TODO item with status tracking"""
    content: str  # What needs to be done
    status: TodoStatus
    active_form: str  # Present continuous form (e.g., "Finding the key")

    # Tracking fields
    attempts: int = 0
    last_action: Optional[str] = None
    failure_reasons: List[str] = field(default_factory=list)
    discovered_info: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "status": self.status.value,
            "active_form": self.active_form,
            "attempts": self.attempts,
            "last_action": self.last_action,
            "failure_reasons": self.failure_reasons,
            "discovered_info": self.discovered_info
        }


class TaskTodoManager:
    """
    Manages task decomposition and progress tracking.

    Key features:
    - Breaks tasks into actionable subtasks (universal, no hardcoding)
    - Learns from reflexion memory across trials
    - Updates based ONLY on TextGrad progress scores
    - Prevents loops by maintaining state awareness
    """

    def __init__(self, llm_model, fast_model=None):
        self.llm = llm_model  # Reasoning model for TODO generation
        self.fast_llm = fast_model if fast_model else llm_model  # Fast model for verification
        self.todos: List[TodoItem] = []
        self.visited_locations: Dict[str, List[str]] = {}  # location -> [observations]
        self.tried_actions: Set[str] = set()
        self.completed_subtasks: List[str] = []

    def initialize_from_task(
        self,
        task: str,
        initial_observation: str,
        valid_actions: List[str],
        trial_num: int = 0,
        reflexion_memory: Optional[List[str]] = None,
        similar_todo_suggestions: Optional[List[str]] = None
    ) -> List[TodoItem]:
        """
        Decompose a high-level task into concrete TODO items.

        Trial 0: Pure task decomposition (LLM common sense)
        Trial 1+: Informed by reflexion learnings (strategic improvement)

        NO FALLBACK - If decomposition fails, raise error to expose issue.
        """

        # CRITICAL: Clear all state from previous task to prevent cross-contamination
        self.todos = []
        self.visited_locations = {}
        self.tried_actions = set()
        self.completed_subtasks = []

        if trial_num == 0:
            # Trial 0: Universal task decomposition (action-oriented, no environment bias)
            prompt = f"""Decompose this task into sequential subgoals.

TASK: {task}
CURRENT STATE: {initial_observation}
"""

            # Add similar task suggestions if available (CROSS-ENV LEARNING)
            if similar_todo_suggestions:
                prompt += f"""
SUCCESSFUL APPROACHES FROM SIMILAR TASKS:
(These are suggestions from similar tasks that succeeded - adapt them to THIS task)
"""
                for i, suggestion in enumerate(similar_todo_suggestions[:5], 1):  # Max 5 examples
                    prompt += f"{i}. {suggestion}\n"

                prompt += "\n"

            prompt += """Requirements:
1. Use ACTION VERBS (what to DO, not what state to BE IN)
2. Keep goals HIGH-LEVEL (e.g., "Cool the object", not "Put object in fridge")
3. Goals should be UNIVERSAL (work in any environment)
4. Each subgoal must be achievable before moving to next
5. Final subgoal completes the entire task
6. Use ONLY objects/concepts mentioned in the task above

Format: Start each line with "TODO: " followed by the high-level goal

Generate 3-8 subgoals:"""

        else:
            # Trial 1+: Pure task decomposition WITHOUT memory injection
            # CRITICAL FIX: Memory injection was causing TODO degradation
            # Evidence: Trial 0 (no memory): 78% success, Trial 1 (with memory): 0% success
            # Root cause: Success memories passed to "learn from failures" prompt caused LLM confusion

            prompt = f"""Decompose this task into sequential subgoals.

TASK: {task}
CURRENT STATE: {initial_observation}

Requirements:
1. Use ACTION VERBS (what to DO, not what state to BE IN)
2. Keep goals HIGH-LEVEL (e.g., "Cool the object", not "Put object in fridge")
3. Goals should be UNIVERSAL (work in any environment)
4. Each subgoal must be achievable before moving to next
5. Final subgoal completes the entire task
6. Use ONLY objects/concepts mentioned in the task above

Format: Start each line with "TODO: " followed by the high-level goal

Generate 3-8 subgoals:"""

        # Call LLM with retry logic for truncated responses
        try:
            from vllm import SamplingParams
        except ImportError:
            from shared_model import SamplingParams

        max_retries = 3
        base_tokens = 7000  # Increased to 7000 tokens for GPT-5 reasoning='high' mode (prevents empty responses)

        for attempt in range(max_retries):
            # Increase tokens on retry (in case output was truncated)
            max_tokens = base_tokens + (attempt * 1000)
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.1)

            print(f"[TODO MANAGER] Attempt {attempt+1}/{max_retries}: Requesting {max_tokens} output tokens with reasoning='high'...")
            # Use reasoning='high' for TODO generation (called only once per env per trial = 36 calls total)
            # Testing with 5000+ tokens to see if this resolves empty response issue
            responses = self.llm.generate([prompt], sampling_params, reasoning_effort='high')
            response = responses[0].outputs[0].text if hasattr(responses[0], 'outputs') else responses[0].text
            print(f"[TODO MANAGER] Response length: {len(response)} chars")
            if response and len(response) > 10:
                print(f"[TODO MANAGER] Response preview: {response[:200]}...")
            else:
                print(f"[TODO MANAGER] WARNING: Empty or very short response!")

            try:
                # Parse simple TODO: format (more reliable than JSON)
                response_text = response.strip()

                # Extract TODO lines (handle both "TODO:" and bullet "-" formats)
                todo_lines = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('TODO:'):
                        todo_text = line[5:].strip()  # Remove "TODO:" prefix
                        if todo_text:
                            todo_lines.append(todo_text)
                    elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        # Handle bullet points
                        todo_text = line[1:].strip()
                        if todo_text and len(todo_text) > 10:  # Filter out short noise
                            todo_lines.append(todo_text)

                # Need at least 3 TODOs for action-level decomposition
                if len(todo_lines) < 3:
                    raise ValueError(f"Only found {len(todo_lines)} TODOs, need at least 3")

                # Create TODO items
                self.todos = []
                for todo_text in todo_lines[:8]:  # Max 8 TODOs for action sequences
                    # Generate active_form by adding "...ing" pattern
                    active_form = todo_text
                    if not any(word in active_form.lower() for word in ['ing ', 'ing,']):
                        # Simple transformation: "Find X" -> "Finding X"
                        first_word = todo_text.split()[0] if todo_text.split() else ""
                        if first_word:
                            active_form = first_word + "ing" + todo_text[len(first_word):]

                    self.todos.append(TodoItem(
                        content=todo_text,
                        status=TodoStatus.PENDING,
                        active_form=active_form
                    ))

                # Mark first TODO as in_progress
                if self.todos:
                    self.todos[0].status = TodoStatus.IN_PROGRESS

                return self.todos

            except ValueError as e:
                # Empty or insufficient response - retry with more tokens
                if attempt < max_retries - 1:
                    continue
                # Last attempt - fail loudly
                raise ValueError(f"[TODO MANAGER] Failed to decompose task '{task}' after {attempt + 1} attempts: {e}\nLLM Response: {response[:500]}")
            except Exception as e:
                # Unexpected error - fail immediately (no retry)
                raise ValueError(f"[TODO MANAGER] Failed to decompose task '{task}': {e}\nLLM Response: {response[:500]}")

    def _extract_critical_failures(self, reflexion_memory: List[str]) -> str:
        """Extract key failure patterns from reflexion memory (universal)"""
        if not reflexion_memory:
            return "No previous failures."

        # Take most recent reflexions (they contain failure summaries)
        recent = reflexion_memory[-3:] if len(reflexion_memory) > 3 else reflexion_memory

        # Handle dict memories - extract insight field
        failure_parts = []
        for r in recent:
            if isinstance(r, dict):
                # Extract insight from dict memory
                r_text = r.get('insight', str(r))[:200]
            else:
                # String memory (legacy)
                r_text = str(r)[:200]
            failure_parts.append(f"- {r_text}")

        failure_text = "\n".join(failure_parts)
        return failure_text if failure_text else "No specific failures identified."

    def update_from_action_feedback(
        self,
        action: str,
        prev_observation: str,
        curr_observation: str,
        progress_status: str
    ) -> None:
        """
        Update TODO status based on action results.

        Uses ONLY progress_status from TextGrad (PURE TEXTUAL FEEDBACK - NO numeric scores).
        """
        print(f"[TODO UPDATE] Called with progress_status={progress_status}, action={action[:50]}")

        # Extract location from action
        location = self._extract_location(action)
        if location:
            if location not in self.visited_locations:
                self.visited_locations[location] = []
            self.visited_locations[location].append(curr_observation[:200])

        # Track tried actions
        self.tried_actions.add(action)

        # Update current in_progress TODO
        current_todo = self._get_current_todo()
        if current_todo:
            current_todo.attempts += 1
            current_todo.last_action = action
            print(f"[TODO UPDATE] Current TODO: {current_todo.content}, attempts={current_todo.attempts}")

            # Check if this action completed the current TODO
            if self._check_todo_completion(current_todo, action, prev_observation, curr_observation, progress_status):
                print(f"[TODO COMPLETE] ✅ TODO completed: {current_todo.content}")
                current_todo.status = TodoStatus.COMPLETED
                self.completed_subtasks.append(current_todo.content)

                # Move to next TODO
                self._advance_to_next_todo()
            elif progress_status == 'NO_PROGRESS' and current_todo.attempts >= 3:
                # Stuck on this TODO - mark as struggling
                current_todo.failure_reasons.append(f"No progress after {current_todo.attempts} attempts")

    def _extract_location(self, action: str) -> Optional[str]:
        """Extract location from action string (universal)"""
        # Common patterns: "go to X", "examine X", "open X"
        if "go to" in action:
            return action.split("go to")[1].strip()
        elif "examine" in action:
            return action.split("examine")[1].strip()
        return None

    def _get_current_todo(self) -> Optional[TodoItem]:
        """Get the currently active TODO"""
        for todo in self.todos:
            if todo.status == TodoStatus.IN_PROGRESS:
                return todo
        return None

    def _check_todo_completion(
        self,
        todo: TodoItem,
        action: str,
        prev_observation: str,
        curr_observation: str,
        progress_status: str
    ) -> bool:
        """
        Check if a TODO subgoal was completed by this action.

        DUAL VERIFICATION: Use BOTH progress_status AND LLM verification
        - If progress_status == MAJOR_PROGRESS or TASK_COMPLETE: High confidence, auto-complete
        - Else: Ask LLM for semantic verification
        """
        print(f"[TODO CHECK] Verifying subgoal with dual check (progress_status={progress_status})")

        # SMART DUAL CHECK: TextGrad progress status + LLM verification
        # CRITICAL: Only auto-complete on high-confidence statuses (MAJOR_PROGRESS or TASK_COMPLETE)
        # PARTIAL_PROGRESS often means "making progress" not "completed"
        # FIXED: ALWAYS verify with LLM - don't blindly trust progress_status
        # TextGrad's progress_status shows good progress, but TODO needs actual completion
        is_achieved = self._verify_subgoal_with_llm(
            subgoal=todo.content,
            prev_obs=prev_observation,
            curr_obs=curr_observation,
            action=action
        )

        if is_achieved:
            print(f"[TODO CHECK] ✅ LLM confirmed: subgoal '{todo.content}' is achieved")
            return True
        else:
            print(f"[TODO CHECK] ❌ LLM says: subgoal '{todo.content}' not yet achieved (status={progress_status})")
            return False

    def _verify_subgoal_with_llm(
        self,
        subgoal: str,
        prev_obs: str,
        curr_obs: str,
        action: str
    ) -> bool:
        """
        Use LLM to verify if subgoal is observably achieved.

        INTELLIGENT: Pure semantic understanding, no keyword matching.
        NO FALLBACK: If LLM can't answer, raise error (fail loudly).
        """
        verification_prompt = f"""Determine if this subgoal is achieved based on observable state changes.

SUBGOAL: {subgoal}
PREVIOUS STATE: {prev_obs[:300]}
ACTION TAKEN: {action}
CURRENT STATE: {curr_obs[:300]}

Question: Based on the state change from PREVIOUS to CURRENT, is the SUBGOAL now observably achieved?

Answer ONLY with: YES or NO"""

        try:
            from vllm import SamplingParams
        except ImportError:
            from shared_model import SamplingParams
        # Use fast model for verification (simple YES/NO, no reasoning needed)
        sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

        try:
            # Use fast_llm for quick verification (extraction, not reasoning)
            responses = self.fast_llm.generate([verification_prompt], sampling_params)
            response = responses[0].outputs[0].text if hasattr(responses[0], 'outputs') else responses[0].text
            response_raw = response  # Keep raw response for diagnostics
            response = response.strip().upper()

            # Simple YES/NO parsing
            if 'YES' in response:
                return True
            elif 'NO' in response:
                return False
            else:
                # Fail loudly with full diagnostics - don't guess
                error_msg = f"""[TODO VERIFY] LLM returned invalid response (expected YES or NO)

VERIFICATION PROMPT:
{verification_prompt}

LLM RAW RESPONSE:
'{response_raw}'

LLM PROCESSED RESPONSE:
'{response}'
"""
                raise ValueError(error_msg)

        except Exception as e:
            # NO FALLBACK - expose the issue with full context
            print(f"[TODO VERIFY ERROR] Failed to verify subgoal '{subgoal}': {e}")
            raise

    def _advance_to_next_todo(self) -> None:
        """Move to the next pending TODO"""
        for todo in self.todos:
            if todo.status == TodoStatus.PENDING:
                todo.status = TodoStatus.IN_PROGRESS
                break

    def get_formatted_todos(self) -> str:
        """Get formatted TODO list for display in prompts"""
        if not self.todos:
            return ""  # Return empty if no TODOs (will be ignored in prompt)

        output = "\n=== TASK TODO LIST ===\n"
        for i, todo in enumerate(self.todos, 1):
            status_icon = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.IN_PROGRESS: "🔧",
                TodoStatus.COMPLETED: "✅",
                TodoStatus.FAILED: "❌"
            }[todo.status]

            output += f"{i}. {status_icon} {todo.content}"

            if todo.status == TodoStatus.IN_PROGRESS:
                output += f" (Attempts: {todo.attempts})"

            output += "\n"

            # Add failure reasons if any
            if todo.failure_reasons:
                output += f"   ⚠️  Issues: {'; '.join(todo.failure_reasons[-2:])}\n"

        # Add visited locations summary (for state awareness)
        if self.visited_locations:
            output += "\n=== EXPLORED LOCATIONS ===\n"
            for loc, visits in list(self.visited_locations.items())[:5]:
                output += f"- {loc}: visited {len(visits)} time(s)\n"

        return output

    def get_state_dict(self) -> Dict:
        """Export full state for persistence"""
        return {
            "todos": [t.to_dict() for t in self.todos],
            "visited_locations": self.visited_locations,
            "tried_actions": list(self.tried_actions),
            "completed_subtasks": self.completed_subtasks
        }

    def should_stop_task(self) -> bool:
        """Check if all TODOs are completed"""
        return all(t.status == TodoStatus.COMPLETED for t in self.todos)
