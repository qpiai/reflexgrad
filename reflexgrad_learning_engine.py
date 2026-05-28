"""ReflexGrad Universal Learning Engine — shared by ALL environments.

Provides model-agnostic, environment-agnostic learning components:
- FailureMemory: tracks failed approaches, prevents repetition
- PolicyGradientStore: accumulates learned policy improvements
- StepInsightsAccumulator: within-episode action→outcome tracking
- LearningContext: bundles all state for prompt injection

Used by both OSWorld (osworld_cua_agent_v12.py) and ALFWorld (reflexgrad_trial.py).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import hashlib


class FailureMemory:
    """Tracks failed approaches. Injected into every prompt so the agent
    never repeats a mistake. Model-agnostic — works for any LLM."""

    def __init__(self, max_entries: int = 20):
        self._memory: Dict[str, str] = {}
        self.max_entries = max_entries

    def record(self, approach_key: str, error_summary: str) -> None:
        """Record a failed approach. Key should be unique per approach."""
        self._memory[approach_key] = error_summary
        # FIFO eviction
        if len(self._memory) > self.max_entries:
            self._memory = dict(list(self._memory.items())[-self.max_entries:])

    def record_from_action(self, action: str, observation: str) -> None:
        """Auto-record when an action fails (e.g., 'Nothing happens')."""
        key = f"fail_{hashlib.md5(action.encode()).hexdigest()[:8]}"
        self.record(key, f"Action '{action}' failed: {observation[:200]}")

    def format_for_prompt(self) -> str:
        """Format as injectable prompt text."""
        if not self._memory:
            return ""
        lines = [f"  - {v}" for v in self._memory.values()]
        return (
            "\n⚠️ FAILED APPROACHES (do NOT repeat these — they already failed):\n"
            + "\n".join(lines[-10:])  # Show last 10
            + "\n"
        )

    def to_dict(self) -> Dict[str, str]:
        """For passing to Reflexion/Synthesizer."""
        return dict(self._memory)

    def clear(self) -> None:
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory)

    def __bool__(self) -> bool:
        return bool(self._memory)

    def __contains__(self, key: str) -> bool:
        return key in self._memory

    def __getitem__(self, key: str) -> str:
        return self._memory[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.record(key, value)


class PolicyGradientStore:
    """Accumulates learned policy improvements from TextGrad backward pass.
    Injected into prompts to guide future actions."""

    def __init__(self, max_gradients: int = 3):
        self._gradients: List[Dict] = []
        self.max_gradients = max_gradients

    def record(self, gradient: Dict) -> bool:
        """Record a policy gradient. Returns False if invalid."""
        if not gradient or not isinstance(gradient, dict):
            return False
        self._gradients.append(gradient)
        self._gradients = self._gradients[-self.max_gradients:]
        return True

    def format_for_prompt(self) -> str:
        """Format as injectable prompt text."""
        if not self._gradients:
            return ""
        lines = []
        for i, pg in enumerate(self._gradients, 1):
            rule = pg.get('general_rule', pg.get('rule', ''))
            pattern = pg.get('problem_pattern', '')
            approach = pg.get('approach', '')
            if rule:
                lines.append(f"  {i}. {rule}")
                if pattern:
                    lines.append(f"     Pattern: {pattern}")
                if approach:
                    lines.append(f"     Approach: {approach}")
        return (
            "\n📚 LEARNED POLICY (apply these lessons):\n"
            + "\n".join(lines)
            + "\n"
        )

    def get_all(self) -> List[Dict]:
        return list(self._gradients)

    def clear_on_success(self, score: int, threshold: int = 8) -> None:
        """Clear stale gradients when task is nearly complete."""
        if score >= threshold and self._gradients:
            self._gradients = []

    def clear(self) -> None:
        self._gradients.clear()

    def __len__(self) -> int:
        return len(self._gradients)

    def __bool__(self) -> bool:
        return bool(self._gradients)

    def append(self, gradient: Dict) -> None:
        """List-compatible append for backward compat."""
        self.record(gradient)


class StepInsightsAccumulator:
    """Within-episode action→outcome tracking. Builds the raw history
    that TextGrad and Reflexion use for diagnosis."""

    def __init__(self, max_insights: int = 20):
        self.insights: List[Dict] = []
        self.max_insights = max_insights

    def record(self, step: int, action: str, observation: str,
               backward_gradient: str = "", progress_score: int = 0,
               progress_status: str = "", next_guidance: str = "",
               metadata: Optional[Dict] = None) -> None:
        """Record one step's action and outcome."""
        entry = {
            'step': step,
            'action': action,
            'observation': observation[:500],
            'backward_gradient': backward_gradient,
            'progress_score': progress_score,
            'progress_status': progress_status,
            'next_action_guidance': next_guidance,
        }
        if metadata:
            entry.update(metadata)
        self.insights.append(entry)
        # Auto-truncate
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[-self.max_insights:]

    def format_for_prompt(self, is_visual: bool = False) -> str:
        """Format raw history for TextGrad prompt."""
        if not self.insights:
            return ""
        lines = ["🎯 RAW ACTION HISTORY (interpret fresh):"]
        for ins in self.insights:
            action = ins.get('action', '?')
            obs = ins.get('observation', '')
            step = ins.get('step', '?')
            if is_visual and obs:
                # For visual envs, trim to state change + command output
                parts = []
                sc_idx = obs.find('[State Change]')
                if sc_idx >= 0:
                    sc_end = obs.find('\n[', sc_idx + 1)
                    parts.append(obs[sc_idx:sc_end if sc_end > sc_idx else sc_idx + 200].strip())
                co_idx = obs.find('[Command Output')
                if co_idx >= 0:
                    parts.append(obs[co_idx:co_idx + 300].strip())
                obs = ' | '.join(parts) if parts else 'No visible change'
            lines.append(f"  Step {step}: ACTION: \"{action}\" → RESULT: \"{obs[:150]}\"")
            bg = ins.get('backward_gradient', '')
            if bg:
                lines.append(f"    📚 LESSON: {bg[:200]}")
        return "\n".join(lines) + "\n"

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.insights[-n:]

    def get_failed_actions(self) -> List[str]:
        """Return actions that got 'Nothing happens' (invalid actions only)."""
        failed = []
        for ins in self.insights:
            obs = ins.get('observation', '')
            if 'Nothing happens' in obs:
                failed.append(ins.get('action', ''))
        return list(set(failed))

    def __len__(self) -> int:
        return len(self.insights)

    def __bool__(self) -> bool:
        return bool(self.insights)


@dataclass
class LearningContext:
    """Bundles all learning state for prompt injection.
    Both OSWorld and ALFWorld construct one, then call format_for_prompt()."""

    failure_memory: FailureMemory = field(default_factory=FailureMemory)
    policy_gradients: PolicyGradientStore = field(default_factory=PolicyGradientStore)
    step_insights: StepInsightsAccumulator = field(default_factory=StepInsightsAccumulator)

    def format_for_prompt(self, is_visual: bool = False) -> str:
        """Combine all learning state into injectable prompt text."""
        parts = []
        fm = self.failure_memory.format_for_prompt()
        if fm:
            parts.append(fm)
        pg = self.policy_gradients.format_for_prompt()
        if pg:
            parts.append(pg)
        si = self.step_insights.format_for_prompt(is_visual=is_visual)
        if si:
            parts.append(si)
        failed = self.step_insights.get_failed_actions()
        if failed:
            parts.append(
                "\n🚫 ACTIONS THAT FAILED (returned 'Nothing happens' — do NOT repeat):\n"
                + ", ".join(f"'{a}'" for a in failed[-10:])
                + "\n"
            )
        return "\n".join(parts) if parts else ""

    def record_failure(self, action: str, observation: str) -> None:
        """Convenience: record a failure in both memory and insights."""
        self.failure_memory.record_from_action(action, observation)

    def record_step(self, step: int, action: str, observation: str, **kwargs) -> None:
        """Convenience: record a step and auto-detect failures."""
        self.step_insights.record(step=step, action=action, observation=observation, **kwargs)
        if 'Nothing happens' in str(observation):
            self.failure_memory.record_from_action(action, observation)
