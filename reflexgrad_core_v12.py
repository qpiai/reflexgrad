"""
ReflexGrad Core — Universal Dual-Process Controller (Inference-Time)

This module extracts the shared ReflexGrad routing logic used by both:
  - Text environments (ALFWorld, TextWorld, etc.) via reflexgrad_trial.py
  - Visual environments (OSWorld CUA) via cua_testing/

ReflexGrad is an inference-time technique that runs within a SINGLE episode:
  - TextGrad (System 1): Fast progress scoring → action hints
  - Reflexion (System 2): Deep causal diagnosis after sustained failure
  - Dual-process routing decides which system activates each check

No multi-trial learning needed — tasks solved in a single attempt.
"""

import os
import logging
from typing import List, Dict, Any, Optional

VERBOSE = os.environ.get("REFLEXGRAD_VERBOSE", "").lower() in ("1", "true")
logger = logging.getLogger("reflexgrad_cua")


class ReflexGradCore:
    """
    Universal ReflexGrad dual-process controller (inference-time).

    Runs within a SINGLE episode/trial:
    - TextGrad (System 1): Fast progress scoring every N steps
    - Reflexion (System 2): Deep causal diagnosis after sustained low scores
    - Dual-process routing decides which system activates

    Used by both text environments (via reflexgrad_trial.py) and
    visual environments (via cua_testing/).

    Parameters:
        reflection_interval: Check progress every N steps (default: 3)
        stall_threshold: Consecutive low scores before Reflexion triggers (default: 2)
        low_score_cutoff: Scores below this are "low" (default: 4)
    """

    def __init__(
        self,
        reflection_interval: int = 3,
        stall_threshold: int = 2,  # 2 consecutive low scores → Reflexion (data shows more Reflexion = more wins)
        low_score_cutoff: int = 5,  # Scores ≤5 count as low
    ):
        self.reflection_interval = reflection_interval
        self.stall_threshold = stall_threshold
        self.low_score_cutoff = low_score_cutoff

        # State (reset per episode)
        self.progress_scores: List[int] = []
        self.consecutive_low_progress: int = 0
        self.steps_since_reflexion: int = 0
        self.textgrad_steps: int = 0
        self.reflexion_activations: int = 0

        # Storyboard-visualizer plumbing: dashboard reads this dict from
        # current_state.json. Populated on every return branch of
        # choose_analysis_mode() so the UI can show WHICH signal fired and WHY.
        self.last_decision: Dict[str, Any] = {
            "signal": None, "reason": "(no decision yet)",
            "mode_prev": None, "mode_curr": None,
            "cooldown_remaining": 0, "score_window": [],
        }
        self._last_mode: Optional[str] = None

        # Router-rule ablation hook (T1.1 — NeurIPS 2027 paper).
        # Read once at construction so a single env var change propagates through
        # the trial pipeline without needing to thread an extra arg through every
        # call site. Modes: "progress_gated" (default), "random", "fixed_cadence".
        import os, random as _random
        self._router_mode = os.environ.get('REFLEXGRAD_ROUTER_MODE', 'progress_gated')
        self._router_step = 0
        self._router_rng = _random.Random(int(os.environ.get('REFLEXGRAD_ROUTER_SEED', '0')))
        if self._router_mode != 'progress_gated':
            print(f"[ROUTER] Ablation mode active: {self._router_mode}")

    def should_reflect(self, step_count: int) -> bool:
        """
        Whether to run a ReflexGrad check this step.

        Returns True every `reflection_interval` steps after the initial interval.
        Callers may add extra triggers (e.g., stuck detector) on top of this.
        """
        if step_count < self.reflection_interval:
            return False
        return step_count % self.reflection_interval == 0

    def _record_decision(self, signal: str, reason: str, mode: str) -> str:
        """Write the router's decision to self.last_decision for the storyboard dashboard.
        Returns `mode` so it can be used inline at `return self._record_decision(...)`."""
        cooldown_remaining = max(0, 2 - self.steps_since_reflexion)
        self.last_decision = {
            "signal": signal,
            "reason": reason,
            "mode_prev": self._last_mode,
            "mode_curr": mode,
            "cooldown_remaining": cooldown_remaining,
            "score_window": list(self.progress_scores[-self.reflection_interval - 2:]),
            "consecutive_low": self.consecutive_low_progress,
            "stall_threshold": self.stall_threshold,
            "low_score_cutoff": self.low_score_cutoff,
        }
        self._last_mode = mode
        # Storyboard: emit a real-time ROUTER substage event. Write directly to
        # .reflexgrad_live/substages.jsonl by resolving the path next to this
        # module so we don't create a circular import with reflexgrad_trial.
        try:
            import os as _os, json as _json, time as _time
            _dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.reflexgrad_live')
            if _os.path.isdir(_dir):
                _line = _json.dumps({
                    'ts': _time.time(),
                    'stage': 'ROUTER',
                    'payload': dict(self.last_decision),
                })
                with open(_os.path.join(_dir, 'substages.jsonl'), 'a') as _f:
                    _f.write(_line + '\n')
        except Exception:
            pass
        return mode

    def choose_analysis_mode(self, assessment: str = "", status: str = "", score: int = -1) -> str:
        """
        Pure textual gradient routing (inference-time). Model-agnostic.

        Routes on TEXTUAL assessment content, NOT numeric scores.
        This prevents self-deception where the actor scores itself 10
        for opening the wrong app.

        Args:
            assessment: Rich text describing what the LLM observed on screen
            status: Categorical label from TextGrad ("complete"|"progressing"|"stuck"|"regressed")
            score: DEPRECATED — kept as optional fallback only, not used for primary routing

        Returns "textgrad", "reflexion", or "verify".
        """
        # Track score history for backward compatibility (logging, stats)
        if score >= 0:
            self.progress_scores.append(score)
        self.steps_since_reflexion += 1
        self._router_step += 1

        # === T1.1 ROUTER ABLATIONS (NeurIPS 2027) ===
        # Bypass the progress-gated rule for ablation runs. "complete" status
        # always returns "verify" so termination still works correctly.
        status_lower_early = (status or "").lower()
        if status_lower_early == "complete":
            return self._record_decision("A_early", "status='complete' (early exit)", "verify")
        if self._router_mode == 'random':
            choice = self._router_rng.choice(['textgrad', 'reflexion'])
            if choice == 'reflexion':
                self.reflexion_activations += 1
                self.steps_since_reflexion = 0
            return self._record_decision("ABLATION_random", f"random ablation chose {choice}", choice)
        if self._router_mode == 'fixed_cadence':
            # Reflexion every k=10 steps (paper's fixed-cadence baseline);
            # all other steps go to textgrad.
            if self._router_step % 10 == 0:
                self.reflexion_activations += 1
                self.steps_since_reflexion = 0
                return self._record_decision("ABLATION_fixed", f"fixed cadence k=10 fired at step {self._router_step}", "reflexion")
            return self._record_decision("ABLATION_fixed", f"fixed cadence skipped at step {self._router_step}", "textgrad")

        # Handle backward compat: text path passes score as first positional arg
        if isinstance(assessment, (int, float)):
            score = int(assessment) if score < 0 else score
            assessment = ""
        assessment_lower = (assessment or "").lower()
        status_lower = (status or "").lower()

        # === SIGNAL A: STATUS says complete → "verify" ===
        if status_lower == "complete":
            if VERBOSE:
                logger.info(f"[G2:ROUTER_DECISION] → verify (Signal A: status='complete')")
            return self._record_decision("A", "status='complete'", "verify")

        # === SIGNAL B: STATUS says stuck or regressed → "reflexion" ===
        if status_lower in ("stuck", "regressed"):
            if self.steps_since_reflexion >= 2:  # cooldown
                self.steps_since_reflexion = 0
                self.reflexion_activations += 1
                if VERBOSE:
                    logger.info(f"[G2:ROUTER_DECISION] → reflexion (Signal B: status='{status_lower}', cooldown passed)")
                return self._record_decision("B", f"status='{status_lower}', cooldown passed", "reflexion")

        # === SIGNAL C: Assessment TEXT analysis (when status is ambiguous) ===
        _complete_kw = ["complete", "done", "finished", "accomplished", "successfully",
                        "playing", "saved", "applied", "created", "installed", "configured"]
        _negative_kw = ["not", "no progress", "hasn't", "failed", "wrong", "incorrect",
                        "stuck", "unable", "error", "no evidence", "unrelated"]
        _stuck_kw = ["no progress", "stuck", "loop", "repeating", "wrong app",
                     "wrong window", "not related", "off-task", "unrelated",
                     "hasn't changed", "no evidence", "no visual change", "no meaningful"]

        _complete_signals = sum(1 for kw in _complete_kw if kw in assessment_lower)
        _negative_signals = sum(1 for kw in _negative_kw if kw in assessment_lower)
        _stuck_signals = sum(1 for kw in _stuck_kw if kw in assessment_lower)

        if VERBOSE:
            logger.info(f"[G2:ROUTER_INTERNALS] consecutive_low_progress={self.consecutive_low_progress}, "
                        f"steps_since_reflexion={self.steps_since_reflexion}, "
                        f"stall_threshold={self.stall_threshold}, low_score_cutoff={self.low_score_cutoff}, "
                        f"complete_signals={_complete_signals}, negative_signals={_negative_signals}, "
                        f"stuck_signals={_stuck_signals}, status='{status_lower}', score={score}")

        # Text says complete with no negatives → verify
        if _complete_signals >= 1 and _negative_signals == 0:
            if VERBOSE:
                logger.info(f"[G2:ROUTER_DECISION] → verify (Signal C: complete_signals={_complete_signals}, no negatives)")
            return self._record_decision("C_complete", f"complete_signals={_complete_signals}, no negatives", "verify")

        # Text says stuck → reflexion (with cooldown)
        if _stuck_signals >= 1 and self.steps_since_reflexion >= 2:
            self.steps_since_reflexion = 0
            self.reflexion_activations += 1
            if VERBOSE:
                logger.info(f"[G2:ROUTER_DECISION] → reflexion (Signal C: stuck_signals={_stuck_signals}, cooldown passed)")
            return self._record_decision("C_stuck", f"stuck_signals={_stuck_signals}, cooldown passed", "reflexion")

        # FIX Gap#6: Verification failure forces Reflexion regardless of TextGrad score
        if getattr(self, '_force_reflexion', False):
            self._force_reflexion = False
            self.steps_since_reflexion = 0
            self.reflexion_activations += 1
            self.consecutive_low_progress = 0
            if VERBOSE:
                logger.info(f"[G2:ROUTER_DECISION] → reflexion (FORCED: verification failure)")
            return self._record_decision("FORCED", "verification failure forced reflexion", "reflexion")

        # === FALLBACK: Score-based routing (only when text is ambiguous) ===
        if score >= 0:
            cutoff = self.low_score_cutoff
            if score <= cutoff:
                self.consecutive_low_progress += 1
            else:
                self.consecutive_low_progress = 0

            if (self.consecutive_low_progress >= self.stall_threshold
                    and self.steps_since_reflexion >= 2):
                self.steps_since_reflexion = 0
                self.reflexion_activations += 1
                self.consecutive_low_progress = 0
                if VERBOSE:
                    logger.info(f"[G2:ROUTER_DECISION] → reflexion (Score fallback: consecutive_low={self.stall_threshold}+ reached, score={score}≤{cutoff})")
                return self._record_decision(
                    "SCORE_FALLBACK",
                    f"consecutive_low={self.stall_threshold}+ reached, score={score}≤{cutoff}",
                    "reflexion",
                )

        self.textgrad_steps += 1
        if VERBOSE:
            logger.info(f"[G2:ROUTER_DECISION] → textgrad (no trigger fired, default)")
        return self._record_decision("DEFAULT", "no trigger fired", "textgrad")

    def record_score(self, score: int):
        """Track a progress score without triggering routing (for external use)."""
        self.progress_scores.append(score)

    def reset(self):
        """Reset for new episode."""
        self.progress_scores = []
        self.consecutive_low_progress = 0
        self.steps_since_reflexion = 0
        self.textgrad_steps = 0
        self.reflexion_activations = 0
        self.last_decision = {
            "signal": None, "reason": "(reset)",
            "mode_prev": None, "mode_curr": None,
            "cooldown_remaining": 0, "score_window": [],
        }
        self._last_mode = None

    def get_stats(self) -> Dict[str, Any]:
        """Return inference-time routing statistics."""
        return {
            "textgrad_steps": self.textgrad_steps,
            "reflexion_activations": self.reflexion_activations,
            "progress_scores": self.progress_scores,
            "total_checks": self.textgrad_steps + self.reflexion_activations,
        }

    def __repr__(self) -> str:
        return (
            f"ReflexGradCore(interval={self.reflection_interval}, "
            f"stall={self.stall_threshold}, "
            f"scores={len(self.progress_scores)}, "
            f"textgrad={self.textgrad_steps}, "
            f"reflexion={self.reflexion_activations})"
        )
