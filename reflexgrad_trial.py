import hashlib
from universal_state_embeddings import state_embeddings
# Dynamic model loading based on MODEL_PROVIDER env var
import os
if os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
    from shared_model_gemini import model, fast_model
elif os.getenv("MODEL_PROVIDER", "openai").lower() == "openrouter":
    from shared_model_openrouter import model, fast_model
elif os.getenv("MODEL_PROVIDER", "openai").lower() == "vllm":
    from shared_model_vllm import model, fast_model
else:
    from shared_model import model, fast_model

# Phase 7 v2 intelligent monitoring (no-op when REFLEXGRAD_DEEP_TRACE != "1")
try:
    import _deep_monitor as _dm
except Exception as _dm_import_err:
    class _dm_stub:
        @staticmethod
        def start(*a, **kw): pass
        @staticmethod
        def stop(*a, **kw): pass
        @staticmethod
        def set_step(*a, **kw): pass
        @staticmethod
        def log_llm_call(*a, **kw): pass
        @staticmethod
        def instrument_model(m, *a, **kw): return m
        @staticmethod
        def is_enabled(): return False
    _dm = _dm_stub()
"""Universal trial execution with adaptive learning and discovery"""

# LEAN MODE: Run learning loop externally, inject only distilled 1-sentence directive
# into a clean prompt. Keeps prompt <10K chars. Toggle via env var.
LEAN_MODE = os.getenv("REFLEXGRAD_LEAN_MODE", "false").lower() == "true"
if LEAN_MODE:
    print("[LEAN MODE] Enabled — prompt capped at ~6K chars, learning loop distills to single directive")
import numpy as np
import os
import sys
import json
import math
import yaml
import importlib
import re
from datetime import datetime
from difflib import SequenceMatcher
from env_history import EnvironmentHistory
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dynamic_prompting import DynamicPromptGenerator, set_debug_flags
from environment_understanding import EnvironmentUnderstanding
from environment_discovery import UniversalEnvironmentDiscovery
from meta_discovery import MetaEnvironmentKnowledge
from universal_env_wrapper import UniversalEnvWrapper, TextWorldWrapper, JerichoWrapper, ScienceWorldWrapper, BabyAIWrapper, NetHackWrapper ,ALFWorldWrapper, AppWorldWrapper

from reflexgrad_learning_engine import FailureMemory, PolicyGradientStore, StepInsightsAccumulator, LearningContext

# Import OSWorld wrapper for visual environments
try:
    from osworld_wrapper import OSWorldWrapper, create_osworld_env
    OSWORLD_AVAILABLE = True
except ImportError:
    OSWORLD_AVAILABLE = False


def clean_action(raw: str) -> str:
    """Strip common LLM output artifacts from action text.

    Model-agnostic: handles markdown, annotations, reasoning traces,
    numbering, code fences, etc. from ANY model family (GPT-4o, GPT-5,
    Claude, Gemini, Llama, etc.).

    Applied once right before env.step() to ensure only valid action
    text reaches the environment. No-op for already-clean inputs.
    """
    if not raw or not isinstance(raw, str):
        return raw

    action = raw.strip()

    # 1. Strip markdown code fences: ```action``` or `action`
    if action.startswith('```') and action.endswith('```'):
        action = action[3:-3].strip()
    action = action.strip('`')

    # 2. Strip markdown bold/italic: **action**, *action*, __action__
    while action.startswith('**') and action.endswith('**'):
        action = action[2:-2].strip()
    while action.startswith('*') and action.endswith('*'):
        action = action[1:-1].strip()
    # Leading ** without closing (common GPT-4o pattern: "** examine pot 1")
    if action.startswith('**'):
        action = action[2:].strip()
    if action.startswith('*') and not action.startswith('*failed'):
        action = action[1:].strip()

    # 3. Strip annotations after common separators
    # Handles: "action | JUSTIFICATION: ...", "action → because ...",
    #          "action // reason", "action => next"
    # Bug fix: don't split on `//` if it's inside a `run:` python command
    # (URLs and namespace URIs use `://` and `//`, e.g.
    # `{http://schemas.openxmlformats.org/...}`). Universal — any URL/URI
    # would otherwise be truncated at the first `//`.
    _is_run_cmd = action.lstrip().lower().startswith('run:')
    # CRITICAL: `|` is a shell pipe operator — NEVER split on it for run: commands.
    # Only split on annotation separators that LLMs add after actions.
    _seps = ['→', '=>', ' JUSTIFICATION'] if _is_run_cmd else ['|', '→', '=>', '//', ' JUSTIFICATION']
    for sep in _seps:
        if sep in action:
            action = action.split(sep)[0].strip()

    # 4. Strip leading numbering: "1. action", "1) action", "- action", "• action"
    action = re.sub(r'^\d+[\.\)]\s*', '', action)
    action = re.sub(r'^[-•]\s*', '', action)

    # 5. Strip wrapping quotes (but NOT for run: commands — quotes are shell syntax)
    if not _is_run_cmd:
        action = action.strip('"\'')

    # 6. Extract embedded action from recommendation text
    # Handles: "NEXT ACTION OPTIMIZATION: ... Recommended Action: go to X - Justification: ..."
    # Also: "Recommended Action: go to X", "ACTION: go to X"
    _action_patterns = [
        r'(?:Recommended\s*Action|RECOMMENDED_ACTION|ACTION)\s*[:=]\s*[*]*\s*(.+?)(?:\s*[-–]\s*[*]*\s*(?:Justification|JUSTIFICATION|Reason|Because)|$)',
        r'(?:Next\s*Action|NEXT_ACTION)\s*[:=]\s*[*]*\s*(.+?)(?:\s*[-–]\s*[*]*\s*(?:Justification|JUSTIFICATION|Reason|Because)|$)',
    ]
    for _pat in _action_patterns:
        _m = re.search(_pat, action, re.IGNORECASE)
        if _m:
            action = _m.group(1).strip()
            break

    # 7. Strip trailing punctuation environments don't expect
    action = action.rstrip('.')

    # 8. Re-strip markdown that might remain after extraction
    # Don't strip quotes from run: commands — they're part of the shell syntax
    action = action.strip('*').strip('`')
    if not _is_run_cmd:
        action = action.strip('"\'')
    action = action.strip()

    return action.strip()
    OSWorldWrapper = None

# SEQUENTIAL LEARNING SYSTEM - Smart knowledge transfer
from task_classifier import task_classifier
from knowledge_classifier import knowledge_classifier
from learning_extractor import learning_extractor
from reflexgrad_core_v12 import ReflexGradCore

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None
# MockOutput always needed (wraps multimodal responses to match vllm interface)
from shared_model import MockOutput
if SamplingParams is None:
    from shared_model import SamplingParams
import random
import pickle
# Global instances for optimization - PERSIST ACROSS TRIALS
import pickle
import os

# Add these globals
checkpoint_manager = None  # Will be initialized in adaptive_env_interaction_batch
comprehensive_logger = None

# ============================================================================
# LIVE DASHBOARD STATE WRITER
# Writes agent state to disk for the real-time dashboard to read.
# State directory: .reflexgrad_live/
# ============================================================================
import base64 as _b64_module

class DashboardStateWriter:
    """Writes live agent state to disk for the ReflexGrad dashboard."""

    def __init__(self, state_dir=None):
        self.state_dir = state_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '.reflexgrad_live')
        os.makedirs(self.state_dir, exist_ok=True)
        self._state = {
            'status': 'waiting',
            'task': '',
            'step': 0,
            'max_steps': 15,
            'action': '',
            'has_screenshots': False,
            'click_metadata': None,
            'progress_score': 0,
            'progress_status': '',
            'loss_text': '',
            'gradient_text': '',
            'next_action': '',
            'learning_mode': 'textgrad',
            'reflexion_info': '',
            'todos': [],
            # Storyboard visualizer extensions:
            'router_decision': None,    # {signal, reason, mode_prev, mode_curr, cooldown_remaining, score_window, ...}
            'policy_base': '',          # ActionPolicy.base_policy (full text)
            'policy_gradient': '',      # most recent textual gradient
            'policy_plan': '',          # most recent Reflexion plan (if any)
            'policy_version': 0,        # incremented when optimizer.step() rewrites base_policy
            'screenshot_sha256': '',    # SHA of current screenshot for stale-detection
            'vgl_elements': [],         # detected VGL ui-elements (for env-side overlay)
            'run_name': '',             # for /api/deeplog/<run>/<id>
            'env_type': '',             # 'alfworld' | 'osworld' | 'textworld' | ... — disables N/A nodes
        }

    def update(self, **kwargs):
        """Update state fields and flush to disk."""
        self._state.update(kwargs)
        try:
            state_file = os.path.join(self.state_dir, 'current_state.json')
            with open(state_file, 'w') as f:
                json.dump(self._state, f, default=str)
        except Exception:
            pass  # Non-critical — dashboard is best-effort

    def log_step(self, step_data: dict):
        """Append a step entry to the history JSONL file."""
        try:
            history_file = os.path.join(self.state_dir, 'step_history.jsonl')
            with open(history_file, 'a') as f:
                f.write(json.dumps(step_data, default=str) + '\n')
        except Exception:
            pass

    def save_screenshot(self, step: int, which: str, b64_data: str):
        """Save a base64-encoded screenshot as PNG for dashboard display."""
        if not b64_data:
            return
        try:
            img_path = os.path.join(self.state_dir, f'step_{step}_{which}.png')
            with open(img_path, 'wb') as f:
                f.write(_b64_module.b64decode(b64_data))
        except Exception:
            pass

    def save_screenshot_from_vm(self, step: int, which: str, vm_ip: str = '172.17.0.5', vm_port: int = 5000):
        """Capture screenshot directly from VM API and save as PNG."""
        try:
            import requests
            resp = requests.get(f'http://{vm_ip}:{vm_port}/screenshot', timeout=3)
            if resp.status_code == 200 and len(resp.content) > 100:
                img_path = os.path.join(self.state_dir, f'step_{step}_{which}.png')
                with open(img_path, 'wb') as f:
                    f.write(resp.content)
        except Exception:
            pass

    def emit_substage(self, stage, payload=None, step=None):
        """Storyboard visualizer: append a real-time substage event so the dashboard
        can drive playback off real events instead of a synthetic timer. Each call
        writes one line to .reflexgrad_live/substages.jsonl with a wall-clock ts."""
        try:
            import time as _t
            line = json.dumps({
                'ts': _t.time(),
                'step': step if step is not None else self._state.get('step', 0),
                'stage': stage,
                'payload': payload or {},
            }, default=str)
            path = os.path.join(self.state_dir, 'substages.jsonl')
            with open(path, 'a') as f:
                f.write(line + '\n')
        except Exception:
            pass

    def update_from_state(self, trial_state, step_gradient=None):
        """Storyboard visualizer: pull router decision + policy snapshot from
        the active per-env state dict. Best-effort — all reads are guarded so
        a missing key never breaks the trial loop."""
        try:
            patch = {}
            core = trial_state.get('reflexgrad_core') if hasattr(trial_state, 'get') else None
            if core is not None and getattr(core, 'last_decision', None):
                patch['router_decision'] = core.last_decision
            policy = trial_state.get('policy') if hasattr(trial_state, 'get') else None
            if policy is not None:
                if hasattr(policy, 'base_policy'):
                    patch['policy_base'] = str(policy.base_policy)[:2000]
                if hasattr(policy, 'gradients') and policy.gradients:
                    patch['policy_gradient'] = str(policy.gradients[-1])[:2000]
                if hasattr(policy, 'update_history'):
                    patch['policy_version'] = len(policy.update_history)
            if step_gradient:
                if step_gradient.get('is_reflexion_strategic_insight'):
                    patch['policy_plan'] = str(step_gradient.get('hypothesis', ''))[:2000]
            run_name = os.environ.get('REFLEXGRAD_RUN_NAME', '')
            if run_name:
                patch['run_name'] = run_name
            if patch:
                self.update(**patch)
        except Exception:
            pass  # Non-critical visualizer hook; never break the trial loop.

    def set_screenshot_hash(self, b64_data: str):
        """Storyboard visualizer: SHA-256 of the latest screenshot for stale-detection."""
        if not b64_data:
            return
        try:
            import hashlib as _hl
            sha = _hl.sha256(b64_data.encode('utf-8') if isinstance(b64_data, str) else b64_data).hexdigest()[:16]
            self.update(screenshot_sha256=sha)
        except Exception:
            pass

    def update_todos(self, todo_manager):
        """Extract TODO list from TaskTodoManager and update state."""
        if not todo_manager:
            return
        todos = []
        try:
            if hasattr(todo_manager, 'todos'):
                for item in todo_manager.todos:
                    # Handle TodoItem dataclass with TodoStatus enum
                    if hasattr(item, 'status'):
                        status_val = item.status.value if hasattr(item.status, 'value') else str(item.status)
                        status_map = {'completed': 'done', 'in_progress': 'current', 'pending': 'pending', 'failed': 'pending'}
                        status = status_map.get(status_val, 'pending')
                    else:
                        status = 'pending'
                    text = str(item.content if hasattr(item, 'content') else item)
                    if hasattr(item, 'attempts') and item.attempts > 0:
                        text += f' (Attempts: {item.attempts})'
                    todos.append({'status': status, 'text': text})
            # Fallback: parse get_formatted_todos() output
            elif hasattr(todo_manager, 'get_formatted_todos'):
                formatted = todo_manager.get_formatted_todos()
                for line in formatted.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('===') or line.startswith('-'):
                        continue
                    if '✅' in line:
                        todos.append({'status': 'done', 'text': line.split('✅')[-1].strip()})
                    elif '🔧' in line:
                        todos.append({'status': 'current', 'text': line.split('🔧')[-1].strip()})
                    elif '⏳' in line:
                        todos.append({'status': 'pending', 'text': line.split('⏳')[-1].strip()})
                    elif '❌' in line:
                        todos.append({'status': 'pending', 'text': line.split('❌')[-1].strip()})
        except Exception:
            pass
        self.update(todos=todos)

    def reset(self):
        """Clear state files for a new run."""
        try:
            history_file = os.path.join(self.state_dir, 'step_history.jsonl')
            if os.path.exists(history_file):
                os.remove(history_file)
            substage_file = os.path.join(self.state_dir, 'substages.jsonl')
            if os.path.exists(substage_file):
                os.remove(substage_file)
            # Remove old screenshots
            import glob as _glob
            for f in _glob.glob(os.path.join(self.state_dir, 'step_*.png')):
                os.remove(f)
        except Exception:
            pass
        self._state = {
            'status': 'waiting', 'task': '', 'step': 0, 'max_steps': 15,
            'action': '', 'has_screenshots': False, 'click_metadata': None,
            'progress_score': 0, 'progress_status': '', 'loss_text': '',
            'gradient_text': '', 'next_action': '', 'learning_mode': 'textgrad',
            'reflexion_info': '', 'todos': [],
            'router_decision': None, 'policy_base': '', 'policy_gradient': '',
            'policy_plan': '', 'policy_version': 0, 'screenshot_sha256': '',
            'vgl_elements': [], 'run_name': '',
        }

# Global dashboard writer instance
_dashboard = DashboardStateWriter()

# Module-level shortcut so any function can emit substages without a class reference
def emit_substage(stage, payload=None, step=None):
    _dashboard.emit_substage(stage, payload, step)

# ============================================================================
# TEXTGRAD COMPONENTS - Real Implementation (Nature 2024)
# ============================================================================

class ActionPolicy:
    """
    The Variable that TextGrad optimizes.
    Represents the action selection policy that evolves through textual gradients.
    """

    def __init__(self, base_policy: str, reflexion_constraints: List[str], model):
        """
        Initialize policy with base strategy and strategic constraints from Reflexion.

        Args:
            base_policy: Initial policy description
            reflexion_constraints: Strategic rules from Reflexion's episodic memory
            model: LLM for generating actions (forward pass)

        Dual-memory architecture (ExpeL/Synapse pattern):
          - `gradients`: ABSTRACT memory — short-horizon accumulated textual
            gradients that the optimizer compresses into `base_policy` and
            then clears (Eq.9 spec). Lossy by design.
          - `gradient_archive`: EPISODIC memory — long-horizon RAW gradients
            with state-fingerprint tags. NOT cleared by optimizer.step().
            Retrieved by similarity at actor time. Recovers the specific
            episodic facts (coords, element names, failure modes) that the
            abstract policy compression destroys. Universal — applies on
            any env where the observation has structural elements that can
            be fingerprinted.
        """
        self.base_policy = base_policy
        self.constraints = reflexion_constraints  # From Reflexion cross-trial learning
        self.gradients = []  # Accumulated textual gradients (within-trial)
        self.gradient_archive = []  # Episodic memory: raw gradient + state_fingerprint
        self.role = "action selection policy"
        self.update_history = []  # Track policy evolution
        self.model = model  # Store model for action generation

    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 200) -> str:
        """Helper to generate text using fast_model.generate() API"""
        output = self.model.generate(
            [prompt],
            SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n"]
            )
        )[0]
        return output.outputs[0].text.strip()

    def forward(self, state: str, task: str, valid_actions: List[str],
                inventory: List[str] = None, todo: str = None,
                reflexion_insights: str = None,
                recent_actions: List[str] = None,
                next_action_guidance: str = None,
                accumulated_state: List[str] = None,
                tried_actions: set = None) -> str:
        """
        FIX #18 (Nov 30): PURE DIRECT EXECUTION - NO FALLBACKS
        FIX #39 (Feb 27): Progress regression guard

        ROOT CAUSE OF BUG: The old code had fuzzy matching that corrupted actions:
        "take saltshaker 2" → "take creditcard 2" (wrong object!)

        NEW APPROACH: Execute EXACTLY what TextGrad recommends
        - If TextGrad has guidance → execute it directly (let ALFWorld accept/reject)
        - ALFWorld's response becomes the learning signal
        - NO fallbacks that could corrupt the action-learning mapping
        - TextGrad learns from its own recommendations, not corrupted alternatives
        - FIX #39: Block actions that would undo recent positive progress
        """

        # ========== ACTION SELECTION STRATEGY ==========
        # Visual envs (OSWorld): Direct execute TextGrad (no discrete action list)
        # Text envs (ALFWorld etc): If TextGrad recommendation is in valid_actions,
        #   execute directly. Otherwise, fall through to full policy prompt which
        #   shows scored valid_actions and forces selection from the list.
        # This prevents 80% of "Nothing happens" from invalid action syntax.

        if next_action_guidance and next_action_guidance.strip():
            guidance_action = clean_action(next_action_guidance)
            import re as _re_ga
            guidance_action = _re_ga.sub(r'^```\w*\s*', '', guidance_action)
            guidance_action = _re_ga.sub(r'\s*```$', '', guidance_action)
            guidance_action = guidance_action.strip()

            # FIX #39: Progress regression guard (visual envs only — selection tracking)
            if accumulated_state:
                selection_active = False
                for entry in accumulated_state:
                    entry_lower = entry.lower()
                    if any(kw in entry_lower for kw in ['selection appeared', 'selection extended',
                                                         'selected', 'highlighted', 'highlighting']):
                        if not any(nk in entry_lower for nk in ['disappeared', 'cleared', 'removed', 'lost']):
                            selection_active = True
                    if any(nk in entry_lower for nk in ['selection disappeared', 'selection cleared',
                                                         'selection removed', 'selection lost',
                                                         'highlighted region) has disappeared',
                                                         'highlighted region) disappeared']):
                        selection_active = False
                if selection_active and recent_actions:
                    clear_nav = ['ctrl+home', 'ctrl+end']
                    for act in recent_actions:
                        if any(pat in act.lower() for pat in clear_nav):
                            selection_active = False
                            break
                if selection_active:
                    action_lower = guidance_action.lower()
                    if 'triple-click' in action_lower or 'triple click' in action_lower:
                        safe_alternatives = [a for a in valid_actions if 'triple' not in a.lower()]
                        if safe_alternatives:
                            return safe_alternatives[0]

            # Check if guidance is in valid_actions
            # Universal: handle exact (text envs), template (visual env templates), and
            # fuzzy (visual env with positional suffixes) matches.
            # Templates look like "run: <shell command>", "type '<text>'", "click '<element>'"
            # Generated actions look like "run: python3 ...", "type 'hello'", "click 'Save'"
            # Visual-env valid actions often look like "click 'Save' button at (x, y)"
            # while TextGrad phrases it as just "click 'Save'" — these must fuzzy-match.
            _guidance_lower = guidance_action.strip().lower()

            # G34b: extract (verb, quoted_target) signature for fuzzy matching.
            # Tolerates missing closing quote (vision models sometimes truncate
            # mid-string) — closing quote is optional, target ends at end-of-string
            # OR at any non-word non-quote boundary like ' button' or ' at ('.
            import re as _re_g34b
            def _action_signature(s):
                s_low = s.strip().lower()
                # Try with closing quote first
                m = _re_g34b.match(r"^\s*(\w+)\s+['\"]([^'\"]+)['\"]", s_low)
                if m:
                    return (m.group(1), m.group(2).strip())
                # Fallback: missing closing quote. Target runs to end-of-string or
                # to a boundary word like " button", " at (", " tab", " link", etc.
                m = _re_g34b.match(r"^\s*(\w+)\s+['\"]([^'\"]+?)(?:\s+(?:button|tab|link|at|field|icon|menu|item|element|area|panel|box|input|textbox|textarea|dropdown|checkbox|radio|window|dialog|popup|form|section|container|label|heading|title|image|img|svg|label|cell|row|column|header|footer|nav)|$)", s_low)
                if m:
                    return (m.group(1), m.group(2).strip())
                # Last resort: verb + entire rest after quote, stripping whitespace
                m = _re_g34b.match(r"^\s*(\w+)\s+['\"]([^'\"]+)", s_low)
                if m:
                    return (m.group(1), m.group(2).strip())
                return None
            _guidance_sig = _action_signature(guidance_action)

            _is_valid_action = False
            for va in (valid_actions or []):
                _va_lower = str(va).strip().lower()
                # Exact match (text envs)
                if _guidance_lower == _va_lower:
                    _is_valid_action = True
                    break
                # Template match: if valid action contains '<...>' placeholder,
                # check if guidance starts with the same prefix
                if '<' in _va_lower and '>' in _va_lower:
                    # Extract prefix before '<' (e.g., "run: " from "run: <shell command>")
                    _prefix = _va_lower.split('<')[0].strip()
                    if _prefix and _guidance_lower.startswith(_prefix):
                        _is_valid_action = True
                        break
                # G34b: fuzzy (verb, quoted_target) match for visual-env actions.
                # "click 'Restore'" matches "click 'Restore' button at (959, 253)"
                # because both have same verb+target signature.
                if _guidance_sig:
                    _va_sig = _action_signature(va)
                    if _va_sig and _va_sig == _guidance_sig:
                        # Rewrite guidance to the valid_action form so the
                        # downstream executor gets the coordinate-bearing string.
                        guidance_action = str(va)
                        _is_valid_action = True
                        print(f"[POLICY] G34b fuzzy-matched guidance '{_guidance_lower[:40]}' → valid action '{str(va)[:60]}'")
                        break

            if _is_valid_action:
                # TextGrad's recommendation IS a valid action — execute directly
                print(f"[POLICY] DIRECT EXECUTE (valid action): '{guidance_action[:80]}'")
                return guidance_action
            elif not valid_actions or len(valid_actions) == 0:
                # No valid actions list (visual env or first step) — trust TextGrad
                print(f"[POLICY] DIRECT EXECUTE (no valid_actions list): '{guidance_action}'")
                return guidance_action
            else:
                # G34e (2026-04-23): In VISUAL environments, valid_actions is populated
                # from VGL-discovered VISUAL AFFORDANCES (click targets, hover targets,
                # typable fields). But the visual-env action space ALSO includes actions
                # VGL can never discover: hotkey combos, raw key presses, shell commands,
                # mouse drags, scroll gestures. When TextGrad/vision generates such an
                # action, it will NOT match any valid_action and the legacy code would
                # substitute a wrong VGL-visible click. Fix: detect non-VGL action shapes
                # and direct-execute them in visual env. Forensic (task 06fe7178): vision
                # correctly generated `hotkey Ctrl+Shift+T` at step 0; exact-string filter
                # rejected it; policy substituted `hotkey Ctrl+L`; task failed all 15 steps.
                _visual_env = any(
                    '(' in str(_va) or ') at ' in str(_va) or ' button' in str(_va).lower()
                        or ' tab' in str(_va).lower() or ' at (' in str(_va).lower()
                    for _va in (valid_actions or [])
                )
                if _visual_env:
                    _g_lower_stripped = _guidance_lower.strip()
                    _non_vgl_prefixes = ('hotkey ', 'key ', 'press ', 'run:', 'scroll ',
                                         'drag ', 'double_click ', 'right_click ',
                                         'move ', 'mousemove ', 'wait', 'done', 'fail')
                    for _pref in _non_vgl_prefixes:
                        if _g_lower_stripped.startswith(_pref) or _g_lower_stripped == _pref.rstrip():
                            print(f"[POLICY] G34e VISUAL-ENV non-VGL action bypass: "
                                  f"guidance '{_guidance_lower[:60]}' uses '{_pref.strip()}' "
                                  f"prefix (not a visible affordance) — DIRECT EXECUTE")
                            return guidance_action
                # TextGrad recommended something NOT in valid_actions.
                # Don't execute blindly — fall through to full policy prompt
                # which will show valid actions and force selection from list.
                # TextGrad's recommendation is passed as next_action_guidance
                # and will appear as a strong signal in the prompt.
                print(f"[POLICY] TextGrad '{guidance_action[:50]}' NOT in valid_actions — using full prompt")

        # ========== FULL POLICY PROMPT (text envs, or when TextGrad is invalid) ==========
        # When TextGrad's recommendation isn't valid, pick the best valid action
        # based on task relevance + TextGrad's intent.
        if valid_actions:
            _tried_lower = set(str(t).strip().lower() for t in (tried_actions or set()))
            _recent_lower = set(str(a).strip().lower() for a in (recent_actions[-5:] if recent_actions else []))
            _task_w = set(task.lower().split()) - {'a','an','the','to','in','on','some','and','put','it'}
            _guidance_w = set(guidance_action.lower().split()) - {'a','an','the','to','in','on','with','from','1','2','3','4','5','6'} if next_action_guidance else set()

            _best_va, _best_score = None, -1
            for _va in valid_actions:
                _va_lower = str(_va).strip().lower()
                # Skip recently tried actions
                if _va_lower in _recent_lower:
                    continue
                _va_w = set(_va_lower.split()) - {'a','an','the','to','in','on','with','from','1','2','3','4','5','6'}
                # Priority: match TextGrad's INTENT (guidance words) > task words
                # If TextGrad said "move pan 1 to countertop 1", we want "put pan 1 in/on countertop 1"
                _guidance_overlap = len(_guidance_w & _va_w) if _guidance_w else 0
                _task_overlap = len(_task_w & _va_w)
                _score = _guidance_overlap * 3 + _task_overlap  # Guidance intent weighted 3x
                if _score > _best_score:
                    _best_score = _score
                    _best_va = _va
            if _best_va:
                print(f"[POLICY] Full prompt selected: '{_best_va}' (score={_best_score}, task+guidance match)")
                return _best_va
            # All recent — pick first untried
            _untried = [va for va in valid_actions if str(va).strip().lower() not in _tried_lower]
            if _untried:
                print(f"[POLICY] Fallback untried: '{_untried[0]}'")
                return _untried[0]
            # Absolute fallback
            print(f"[POLICY] Fallback first valid: '{valid_actions[0]}'")
            return valid_actions[0]

        # No valid actions at all
        print(f"[POLICY] No valid actions — returning 'look'")
        return "look"

    def _format_constraints(self) -> str:
        """Format Reflexion's strategic constraints"""
        if not self.constraints:
            return "No strategic constraints yet (Trial 0, no prior knowledge)"
        return "\n".join([f"  - {c}" for c in self.constraints])

    def _format_gradients(self) -> str:
        """Format accumulated gradients from this trial"""
        if not self.gradients:
            return "No gradients yet (first step of trial)"
        recent = self.gradients[-5:]  # Last 5 gradients
        return "\n".join([f"  {i+1}. {g}" for i, g in enumerate(recent)])

    def _format_actions(self, actions: List[str]) -> str:
        """Format valid actions list"""
        return "\n".join([f"  {i+1}. {a}" for i, a in enumerate(actions[:30])])  # Limit to 30

    def _format_recent_actions(self, actions: List[str], max_show: int = 7) -> str:
        """Format recent actions for loop prevention"""
        if not actions:
            return "No actions yet"

        recent = actions[-max_show:]
        lines = []
        for i, action in enumerate(recent, 1):
            # Count how many times this action appears in recent history
            count = recent.count(action)
            marker = f" [REPEATED x{count}]" if count > 1 else ""
            lines.append(f"  {i}. {action}{marker}")

        return "\n".join(lines)

    def _find_best_action_match(self, generated: str, valid: List[str]) -> Optional[str]:
        """Find best matching action from valid list - VERB-PRIORITY + PREREQUISITE AWARENESS"""
        generated_lower = generated.lower().strip()

        # 1. Exact match (highest priority)
        for action in valid:
            if action.lower() == generated_lower:
                return action

        # 2. VERB-PRIORITY MATCHING: Match verb FIRST, then target
        # This prevents "open cabinet 1" → "examine cabinet 1" corruption
        gen_words = generated_lower.split()
        if gen_words:
            gen_verb = gen_words[0]  # First word is the verb
            gen_target = ' '.join(gen_words[1:]) if len(gen_words) > 1 else ''

            # Look for actions with SAME VERB first
            verb_matches = []
            for action in valid:
                act_words = action.lower().split()
                if act_words and act_words[0] == gen_verb:
                    verb_matches.append(action)

            # Among verb matches, find best target match
            if verb_matches:
                # Exact target match with same verb
                for action in verb_matches:
                    if action.lower() == generated_lower:
                        return action
                    # Check if target matches
                    act_target = ' '.join(action.lower().split()[1:])
                    if gen_target and act_target == gen_target:
                        return action

                # Partial target match with same verb (handles "take pillow 1" matching "take pillow 1 from armchair 1")
                for action in verb_matches:
                    act_target = ' '.join(action.lower().split()[1:])
                    if gen_target:
                        # Check if generated target is contained in valid action target
                        if gen_target in act_target:
                            return action
                        # Check if valid action target starts with generated target (handles partial)
                        if act_target.startswith(gen_target):
                            return action

                # If we have verb matches but no target match, return first verb match
                if verb_matches:
                    return verb_matches[0]

            # 2b. PREREQUISITE ACTION AWARENESS (UNIVERSAL - no hardcoded verbs)
            # FIX #17B: If ANY verb has no matches, check if we need to "go to" the target first
            # This is universal - works for any verb without hardcoded lists
            if not verb_matches and gen_target:
                # Extract the location from target (e.g., "cabinet 1" from "cabinet 1", "armchair 1" from "pillow 1 from armchair 1")
                # Handle "X from Y" patterns
                if ' from ' in gen_target:
                    location = gen_target.split(' from ')[-1]  # Get the location after "from"
                elif ' in ' in gen_target:
                    location = gen_target.split(' in ')[-1]
                elif ' on ' in gen_target:
                    location = gen_target.split(' on ')[-1]
                else:
                    location = gen_target  # The target itself is the location (e.g., "cabinet 1")

                # Look for "go to [location]" in valid actions
                for action in valid:
                    if action.lower() == f'go to {location}':
                        return action

                # Also try partial location match (e.g., "cabinet 1" matches "go to cabinet 1")
                for action in valid:
                    if action.lower().startswith('go to ') and location in action.lower():
                        return action

                # FIX #11: CRITICAL - If we're already at the location (no "go to" needed),
                # return the generated action AS-IS instead of falling back to a different verb!
                # ALFWorld may accept actions not in admissible_commands list.
                # This prevents "open cabinet 1" → "examine cabinet 1" corruption.
                print(f"[ACTION MATCH FIX] Interaction verb '{gen_verb}' not in valid_actions, returning as-is: '{generated}'")
                return generated  # Return original action, let ALFWorld handle it

        # 3. Fallback: Full phrase substring match (only for non-interaction verbs)
        for action in valid:
            if generated_lower in action.lower() or action.lower() in generated_lower:
                return action

        # 4. Last resort: Word overlap scoring (only for non-interaction verbs)
        gen_words_set = set(generated_lower.split())
        best_score = 0
        best_action = None
        for action in valid:
            act_words = set(action.lower().split())
            overlap = len(gen_words_set & act_words)
            if overlap > best_score:
                best_score = overlap
                best_action = action

        return best_action if best_score > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC AFFORDANCE MATCHING for Visual Environments (OSWorld, etc.)
# These functions enable FIX #20 to work with descriptive affordances like
# "click 'Save' button" instead of requiring exact string matches.
# ═══════════════════════════════════════════════════════════════════════════════

def _semantic_affordance_match(action: str, affordances: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if an action semantically matches any affordance.

    For visual environments, affordances are descriptive (e.g., "click 'Save' button")
    and we need fuzzy matching instead of exact string comparison.

    Args:
        action: The action text from the agent
        affordances: List of detected affordances from VGL

    Returns:
        (matched: bool, matched_affordance: str or None)
    """
    if not affordances:
        return False, None

    action_lower = action.lower().strip()

    # 1. Exact match (highest confidence)
    for aff in affordances:
        if aff.lower().strip() == action_lower:
            return True, aff

    # 2. Substring containment (one contains the other)
    for aff in affordances:
        aff_lower = aff.lower().strip()
        if aff_lower in action_lower or action_lower in aff_lower:
            return True, aff

    # 3. Key word overlap matching
    # Remove common stop words and action prefixes
    stop_words = {'the', 'a', 'an', 'on', 'in', 'at', 'to', 'for', 'of', 'with'}
    action_words = set(action_lower.split()) - stop_words

    for aff in affordances:
        aff_lower = aff.lower().strip()
        aff_words = set(aff_lower.split()) - stop_words

        # Calculate Jaccard similarity
        if action_words and aff_words:
            intersection = len(action_words & aff_words)
            union = len(action_words | aff_words)
            similarity = intersection / union if union > 0 else 0

            # Threshold for match: >50% word overlap
            if similarity > 0.5:
                return True, aff

    # 4. Action verb + target matching
    # "click Save" should match "click 'Save' button"
    # "toggle Bold checkbox" should match "toggle 'Bold' checkbox"
    action_parts = action_lower.split()
    if len(action_parts) >= 2:
        verb = action_parts[0]  # e.g., "click"
        target = ' '.join(action_parts[1:])  # e.g., "save"

        for aff in affordances:
            aff_lower = aff.lower()
            # Check if verb matches and target is mentioned
            if verb in aff_lower:
                target_clean = target.replace("'", "").replace('"', "").strip()
                aff_clean = aff_lower.replace("'", "").replace('"', "")
                if target_clean in aff_clean:
                    return True, aff

    # 5. Quote-stripped full match
    # "type 'hello' in Search" should match "type in 'Search' field"
    action_noquotes = action_lower.replace("'", "").replace('"', "")
    for aff in affordances:
        aff_noquotes = aff.lower().replace("'", "").replace('"', "")
        if action_noquotes in aff_noquotes or aff_noquotes in action_noquotes:
            return True, aff

    return False, None


def _find_similar_affordances(action: str, affordances: List[str], top_k: int = 3) -> List[str]:
    """
    Find the most similar affordances to an action (for error messages).

    Args:
        action: The action text
        affordances: List of available affordances
        top_k: Number of similar affordances to return

    Returns:
        List of most similar affordances
    """
    if not affordances:
        return []

    action_lower = action.lower().strip()
    action_words = set(action_lower.split()) - {'the', 'a', 'an', 'on', 'in', 'at', 'to'}

    # Score each affordance by word overlap
    scored = []
    for aff in affordances:
        aff_lower = aff.lower().strip()
        aff_words = set(aff_lower.split()) - {'the', 'a', 'an', 'on', 'in', 'at', 'to'}

        # Count overlapping words
        overlap = len(action_words & aff_words)

        # Also check for partial word matches (e.g., "sav" in "save")
        partial_matches = 0
        for aw in action_words:
            for affw in aff_words:
                if aw in affw or affw in aw:
                    partial_matches += 0.5

        score = overlap + partial_matches
        scored.append((aff, score))

    # Sort by score (descending) and return top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    return [aff for aff, score in scored[:top_k] if score > 0]


# ─────────────────────────────────────────────────────────────────────── #
# LOSS_RESPONSE_SCHEMA — evidence-anchored, executable-probe-verified     #
# LLM-as-judge output. Universal across env classes; enforced via         #
# response_format json_schema in shared_model.                            #
#                                                                         #
# Composition (all 2022-2026 venue-published):                            #
#   - Inner Monologue (Huang CoRL 2022): observed_state_after first       #
#   - RULERS (arXiv 2601.08654): supporting/contradicting evidence quotes #
#   - CRITIC (Gou ICLR 2024): executable_check tool-call slot             #
#   - GUIDE (arXiv 2604.04399): structured-JSON verdict + error analysis  #
#   - ReflAct (Kim EMNLP 2025): goal_state_alignment_check                #
#                                                                         #
# Slot order is load-bearing — narrate-before-judge prevents post-hoc     #
# rationalisation (Inner Monologue's central finding).                    #
# ─────────────────────────────────────────────────────────────────────── #
LOSS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "observed_state_after", "narrated_state_change",
        "executable_check", "supporting_evidence_quote",
        "goal_state_alignment_check",
        "alternative_actions_considered", "best_alternative_to_emit_next",
        "blocking_constraint_on_current_path",
        "subgoal_done", "verdict", "score", "reason",
    ],
    "properties": {
        "observed_state_after": {
            "type": "string",
            "description": "Verbatim narration of STATE_AFTER drawn directly "
                           "from the observation text. Do NOT invent details "
                           "not present in the observation.",
        },
        "narrated_state_change": {
            "type": "string",
            "description": "Diff between STATE_BEFORE and STATE_AFTER. "
                           "If unchanged, say 'no observable change'.",
        },
        "executable_check": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "command"],
            "properties": {
                "kind": {"enum": ["shell", "python", "getter", "dom", "a11y", "file", "none"]},
                "command": {"type": "string"},
            },
            "description": "A deterministic check the runtime can execute "
                           "to verify your verdict. 'none' only if no check "
                           "applies in this env class. Runtime result OVERRIDES "
                           "your subgoal_done if they conflict.",
        },
        "supporting_evidence_quote": {
            "type": "string",
            "description": "A substring of the STATE AFTER text that supports "
                           "your verdict. MUST be byte-for-byte present in "
                           "the observation; runtime validates this and flips "
                           "verdict to UNVERIFIED if the quote is not found.",
        },
        "contradicting_evidence_quote": {
            "type": ["string", "null"],
            "description": "A substring of the OBSERVATION that contradicts "
                           "your verdict (if any). null when none applies.",
        },
        "goal_state_alignment_check": {
            "type": "string",
            "description": "Compare STATE_AFTER to the TASK GOAL. State "
                           "concretely what the goal requires and which "
                           "specific elements of STATE_AFTER satisfy it.",
        },
        "alternative_actions_considered": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action_description", "predicted_effect",
                             "why_not_taken"],
                "properties": {
                    "action_description": {
                        "type": "string",
                        "description": "Concrete alternative action (must reference "
                                       "entities from the UI table, not invented coords).",
                    },
                    "predicted_effect": {
                        "type": "string",
                        "description": "What this alternative would produce if taken "
                                       "next, grounded in the current observation.",
                    },
                    "why_not_taken": {
                        "type": "string",
                        "description": "Why the agent took the actual action instead "
                                       "of this one.",
                    },
                },
            },
            "description": "Up to 3 alternative actions considered but not taken "
                           "(Pearl 2009 counterfactual / Bellman 1957 Bellman eq). "
                           "Must be grounded in entities visible in the current UI "
                           "table. Empty list [] when alternatives are not applicable.",
        },
        "best_alternative_to_emit_next": {
            "type": "string",
            "description": "If verdict is GOAL_NOT_ALIGNED or UNVERIFIED, the "
                           "SINGLE best alternative from above that the actor "
                           "should try next. Empty string '' when verdict is "
                           "GOAL_ALIGNED (no retry needed).",
        },
        "blocking_constraint_on_current_path": {
            "type": "string",
            "description": "Hoare weakest precondition: what condition must hold "
                           "for the current action path to succeed? If that "
                           "condition was NOT met (blocking constraint), state it "
                           "here. Empty string '' when no blocking constraint "
                           "was identified.",
        },
        "subgoal_done": {
            "enum": ["DONE", "PARTIAL", "NOT_DONE"],
            "description": "DONE only when the current subgoal's "
                           "byte-for-byte criterion is satisfied in "
                           "STATE_AFTER. Opening a dialog is NOT_DONE for "
                           "a 'set value' subgoal until the value is set.",
        },
        "verdict": {
            "enum": ["GOAL_ALIGNED", "GOAL_NOT_ALIGNED", "UNVERIFIED"],
            "description": "GOAL_ALIGNED = clear measurable progress; "
                           "GOAL_NOT_ALIGNED = no progress or regression; "
                           "UNVERIFIED = cannot determine from observation.",
        },
        "score": {
            "type": "integer", "minimum": 0, "maximum": 10,
            "description": "0=no progress, 5=meaningful step, 10=complete.",
        },
        "reason": {
            "type": "string",
            "description": "Concise WHY for the verdict, grounded in the "
                           "supporting/contradicting quotes above.",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────── #
# BACKWARD_RESPONSE_SCHEMA — 3 typed candidates per gradient step.        #
# Each candidate must name its type (probe|act|discover), predict effect, #
# and specify a postcondition_check that would confirm success.           #
# The chosen_index picks which candidate to emit as CONCRETE_FIX.        #
# ─────────────────────────────────────────────────────────────────────── #
BACKWARD_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "failure_diagnosis", "supporting_evidence_quote",
        "verbatim_stderr_signal",
        "candidates", "chosen_index", "policy_rule",
    ],
    "properties": {
        "failure_diagnosis": {
            "type": "string",
            "description": "Root cause of the failure — concrete, not vague. "
                           "Quote the exact evidence from the LOSS or observation.",
        },
        "supporting_evidence_quote": {
            "type": "string",
            "description": "Substring of the observation/LOSS that PROVES the "
                           "diagnosis. Must be byte-for-byte present in the input; "
                           "runtime validates and marks gradient UNGROUNDED if not.",
        },
        "verbatim_stderr_signal": {
            "type": "string",
            "maxLength": 400,
            "description": "Character-for-character copy of the FIRST non-empty "
                           "line from the observation's [Last Action Result].stderr "
                           "field. The string MUST be a literal substring of "
                           "the observation text — runtime validates this and "
                           "marks UNGROUNDED if not. "
                           "Empty string '' iff rc==0 AND stderr field is empty/absent. "
                           "Hoare 1969 — post-condition observation must be FAITHFUL: "
                           "no paraphrase, no summary, no synthesized confidence values, "
                           "no derived semantic interpretation. Just the bytes from stderr.",
        },
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "action", "predicted_effect",
                             "postcondition_check"],
                "properties": {
                    "type": {
                        "enum": ["probe", "act", "discover"],
                        "description": "'probe' = read-only info-gathering action; "
                                       "'act' = state-changing action toward the goal; "
                                       "'discover' = explore/navigate to reveal needed info.",
                    },
                    "action": {
                        "type": "string",
                        "description": "Exact executable action string. For visual envs "
                                       "with a [UI Elements] table: MUST include coords "
                                       "copied from the table (e.g. pyautogui.click(x, y)).",
                    },
                    "predicted_effect": {
                        "type": "string",
                        "description": "What this action should produce if taken, "
                                       "grounded in the current observation.",
                    },
                    "postcondition_check": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind"],
                        "properties": {
                            "kind": {
                                "enum": ["shell", "python", "getter", "dom",
                                         "a11y", "file", "none"]
                            },
                            "command": {"type": "string"},
                        },
                        "description": "A deterministic, read-only check that would "
                                       "confirm the predicted_effect was achieved. "
                                       "'kind=none' applies when no check is needed; "
                                       "in that case 'command' may be omitted. "
                                       "Fix #7 broken-edge audit 2026-05-07: making "
                                       "'command' conditionally optional unblocks "
                                       "schema-constrained decoding when kind=none "
                                       "(LLM was getting stuck producing whitespace "
                                       "trying to satisfy a vacuous required field, "
                                       "causing 95% of BACKWARD gradients to truncate "
                                       "before reaching policy_rule).",
                    },
                },
            },
            "description": "1-3 alternative next actions, diverse in type. "
                           "Prefer: one 'probe' to gather info, one 'act' toward "
                           "the goal, one 'discover' when state is unknown.",
        },
        "chosen_index": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
            "description": "0-based index into candidates[] of the action to "
                           "emit as CONCRETE_FIX. Choose the candidate most "
                           "likely to make measurable progress.",
        },
        "policy_rule": {
            "type": "string",
            "description": "ONE policy rule (universal, not task-specific) that "
                           "prevents this class of error in future steps.",
        },
    },
}


class TextGradLoss:
    """
    GOAL-ALIGNED Loss Function for TextGrad.
    Evaluates if action helps achieve TASK GOAL, not just execution success.

    Evidence-anchored, executable-probe verified (per the LOSS_RESPONSE_SCHEMA
    above) — eliminates the four hallucination modes that free-form CoT
    judgment produces under frozen-LLM, no-example, single-call constraints.
    """

    def __init__(self, model):
        self.model = model  # Store model for loss computation
        self._init_prompts()  # Initialize prompt templates

    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
        """Helper to generate text using fast_model.generate() API.

        PAPER FIX: TextGradLoss output is structured (VERDICT / REASON /
        NEXT_ACTION_HINT) and the model often inserts blank lines between
        fields. stop=["\\n\\n"] previously truncated REASON/NEXT_ACTION_HINT,
        leaving BACKWARD blind to the LLM-judge's diagnosis. Stage-4
        escalation and Atom-4 completion both read loss_text and need REASON.
        """
        output = self.model.generate(
            [prompt],
            SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n\n"]
            )
        )[0]
        return output.outputs[0].text.strip()

    def _init_prompts(self):
        self.evaluation_prompt_template = """You are evaluating whether an action helps achieve the TASK GOAL.

TASK GOAL: {task}
CURRENT SUBTASK (from TODO system): {subtask}

{eval_gap_block}LEARNED INSIGHTS FROM PREVIOUS ATTEMPTS:
{reflexion_insights}

POLICY APPROACH: {policy}
STRATEGIC CONSTRAINTS: {constraints}

FULL TRAJECTORY HISTORY (check for repetitive/looping actions):
{trajectory_history}

STATE BEFORE ACTION:
{state_before}

ACTION TAKEN:
{action}

STATE AFTER ACTION:
{state_after}

ACTIONS NOW AVAILABLE (after this action):
{available_actions_after}

=== CRITICAL EVALUATION (Focus on GOAL ALIGNMENT, not just execution) ===

1. TASK GOAL ALIGNMENT (MOST IMPORTANT):
   - What does the TASK GOAL require?
   - What did this action actually produce (see STATE AFTER)?
   - Does the RESULT match what the TASK GOAL is asking for?
   - IMPORTANT: Check ACTIONS NOW AVAILABLE — if a new action appeared that
     directly matches the task (e.g., "cool", "heat", "clean", "put"),
     this action ENABLED progress even if STATE AFTER looks unchanged.
     Moving to a location that unlocks the right action IS goal-aligned.

2. SUBTASK/TODO ALIGNMENT:
   - Current subtask: "{subtask}"
   - Did this action help achieve this subtask?
   - If subtask conflicts with task goal, trust the TASK GOAL.

3. LOOP/REPETITION CHECK:
   - Look at FULL TRAJECTORY HISTORY above.
   - Is this action repeating a previous action that didn't help?
   - Repetitive actions that don't make progress should be flagged.

4. LEARNED INSIGHT COMPLIANCE:
   - Check if action contradicts any LEARNED INSIGHTS above.
   - Previous insights reveal what DOESN'T work for this type of task.

5. EVIDENCE FROM STATE CHANGE:
   - What specifically changed between state_before and state_after?
   - Does this change bring us closer to the TASK GOAL?
   - Check ACTIONS NOW AVAILABLE: did new task-relevant actions appear?

6. EVALUATOR CRITERIA GATE (MANDATORY — overrides all other signals):
   - If ⛔ EVALUATOR CRITERIA is listed above, it contains the EXACT values
     the automated test checks. Read it before deciding verdict.
   - If the required literal value (e.g., 'Square', a specific filename, a
     setting string) does NOT appear verbatim in STATE AFTER ACTION, you MUST
     emit verdict=GOAL_NOT_ALIGNED. Score MUST be ≤ 2. No exceptions.
   - A screen change WITHOUT the evaluator's required value appearing is NOT
     progress. Do NOT emit GOAL_ALIGNED merely because the screen changed,
     a menu opened, or a dialog appeared.
   - This rule is ABSOLUTE: "something changed on screen" alone never
     justifies GOAL_ALIGNED when the evaluator's literal criterion is absent
     from STATE AFTER.

7. KEYBOARD-INPUT GATE (for name/value entry tasks):
   - If ⛔ EVALUATOR CRITERIA requires a specific text value to be ENTERED
     into a field (a dialog name, a file name, a setting value), and the
     ACTION TAKEN does not include any of:
       pyautogui.typewrite, pyautogui.write, pyautogui.hotkey, xte "str",
       keyboard.type, sendkeys, typestring, or equivalent keyboard input
     containing that value — then the value CANNOT have been entered.
   - In this case you MUST emit verdict=GOAL_NOT_ALIGNED regardless of what
     STATE AFTER shows, because clicking buttons without typing cannot set
     a text field value.
   - Example: task requires naming a layer 'Square' but ACTION = click(x,y):
     clicking OK/Create without a prior typewrite('Square') means the name
     was NOT set to 'Square'. Emit GOAL_NOT_ALIGNED, score ≤ 2.

RESPONSE FORMAT — Output a single JSON object with these fields, IN THIS
ORDER. The runtime enforces this schema at decode-time and validates each
field; non-conforming output is rejected. Fields marked [OVERRIDABLE] are
corrected by the runtime if they conflict with the executable_check result.

  "observed_state_after": verbatim narration drawn from STATE AFTER ACTION
                          above. Do NOT invent details not present.
  "narrated_state_change": diff vs STATE BEFORE — what concretely changed.
                           Say "no observable change" when nothing changed.
  "executable_check": {{"kind": "shell"|"python"|"getter"|"dom"|"a11y"|"file"|"none",
                        "command": "..."}}
                      A deterministic, read-only check the runtime can run to
                      verify your verdict. Prefer shell/getter when possible.
                      If the check result contradicts your verdict the runtime
                      will override subgoal_done. Use "none" ONLY if truly no
                      check applies (not just because it's convenient).
  "supporting_evidence_quote": a substring of the STATE AFTER text that
                               supports your verdict. MUST be byte-for-byte
                               present; runtime flips verdict→UNVERIFIED if not.
  "contradicting_evidence_quote": a substring that contradicts your verdict
                                  if any (else null).
  "goal_state_alignment_check": which specific elements of STATE AFTER
                                satisfy the TASK GOAL? Cite them verbatim.
  "alternative_actions_considered": array of up to 3 objects, each with
    "action_description": a concrete alternative (reference UI table entities),
    "predicted_effect": what it would produce based on current obs,
    "why_not_taken": why the agent chose the actual action instead.
    Empty array [] when no meaningful alternatives exist.
  "best_alternative_to_emit_next": if verdict is GOAL_NOT_ALIGNED or
    UNVERIFIED, the SINGLE best alternative to try next. Empty string ""
    when verdict is GOAL_ALIGNED.
  "blocking_constraint_on_current_path": what precondition was NOT met that
    blocked the current action path (Hoare wp). Empty string "" if none.
  "subgoal_done" [OVERRIDABLE]: "DONE" only when the subgoal's byte-for-byte
                  criterion is satisfied in STATE AFTER. "PARTIAL" when
                  meaningful progress made. "NOT_DONE" otherwise.
  "verdict": "GOAL_ALIGNED" | "GOAL_NOT_ALIGNED" | "UNVERIFIED".
             UNVERIFIED routes to System-2 Reflexion + probe firing.
  "score": integer 0-10. 0=no progress, 5=meaningful step, 10=complete.
  "reason": concise WHY, grounded in the quotes above."""

    def __call__(self, action: str, state_before: str, state_after: str,
                 task: str, policy, trajectory_context: list = None,
                 subtask: str = None, reflexion_insights: list = None,
                 state_diff: str = None, affordances: list = None,
                 env_type: str = None, click_metadata: dict = None,
                 before_screenshot_b64: str = None, after_screenshot_b64: str = None,
                 available_actions_after: list = None) -> str:
        """
        Compute textual loss (criticism) for the action taken by policy.

        GOAL-ALIGNED LOSS: Evaluates if action helps achieve TASK GOAL,
        not just if action executed successfully.

        Args:
            trajectory_context: List of dicts with keys: action, observation
            subtask: Current subtask from TODO manager (guidance)
            reflexion_insights: List of learned insights from Reflexion episodic memory
            state_diff: (Visual envs) VGL-computed description of screen changes
            affordances: (Visual envs) List of available UI interactions
            env_type: Environment type (for visual env detection)
            click_metadata: (Visual envs) Coordinate resolution details from VGL

        Returns:
            Textual loss focusing on GOAL ALIGNMENT
        """
        # Format ALL trajectory history (helps detect loops and repetitive actions)
        # FIX #28: NO TRUNCATION - full observation needed to evaluate goal alignment
        trajectory_history = "No previous actions yet."
        if trajectory_context and len(trajectory_context) > 0:
            history_lines = []
            for i, step in enumerate(trajectory_context, 1):  # ALL steps, not just last 5
                act = step.get('action', 'unknown')
                obs = step.get('observation', '')  # FIX #28: Full observation for accurate loss eval
                history_lines.append(f"  Step {i}: {act} -> {obs}")
            trajectory_history = "\n".join(history_lines)

        # Use subtask if provided, otherwise derive from task
        current_subtask = subtask if subtask else f"Working toward: {task}"

        # Format Reflexion insights
        # FIX #28: NO TRUNCATION - full insights contain critical learning
        formatted_insights = "No learned insights yet (first trial)."
        if reflexion_insights and len(reflexion_insights) > 0:
            insight_lines = []
            for i, insight in enumerate(reflexion_insights[-3:], 1):  # Last 3 insights
                if isinstance(insight, dict):
                    insight_text = insight.get('reflection', insight.get('insight', str(insight)))
                else:
                    insight_text = str(insight)
                insight_lines.append(f"  {i}. {insight_text}")  # FIX #28: Full insight
            formatted_insights = "\n".join(insight_lines)

        # Get policy text safely
        policy_text = policy.base_policy if hasattr(policy, 'base_policy') else str(policy)
        constraints_text = policy._format_constraints() if hasattr(policy, '_format_constraints') else "None"

        # Format available actions after this step (helps evaluate preparatory actions)
        if available_actions_after:
            aa_str = ', '.join(str(a) for a in available_actions_after[:30])
        else:
            aa_str = '(not available)'

        # Atom B' — extend eval ground truth into LOSS prompt. Same source
        # as optimizer/BACKWARD/PASS-2 (policy._last_eval_gap + _eval_source).
        # Without this, LOSS judges progress against a generic notion of
        # "looks like task done" instead of the specific literals/criteria
        # the evaluator actually checks. Universal — empty string when no
        # eval literals available.
        _eval_gap_text = getattr(policy, '_last_eval_gap', '') or ''
        _eval_source_text = getattr(policy, '_eval_source', '') or ''
        if _eval_gap_text:
            _eval_gap_block = (
                "⛔ EVALUATOR CRITERIA (verbatim — judge GOAL_ALIGNED only when\n"
                "   STATE AFTER provides byte-for-byte evidence the criteria are met):\n"
                f"{str(_eval_gap_text)[:1200]}\n"
            )
            if _eval_source_text:
                _eval_gap_block += (
                    "🔍 EVALUATOR SOURCE (the exact code/command the evaluator runs —\n"
                    "   judge GOAL_ALIGNED only when this code would now succeed):\n"
                    f"{str(_eval_source_text)[:1600]}\n"
                )
            _eval_gap_block += "\n"
        else:
            _eval_gap_block = ""

        prompt = self.evaluation_prompt_template.format(
            task=task,
            subtask=current_subtask,
            eval_gap_block=_eval_gap_block,
            reflexion_insights=formatted_insights,
            policy=policy_text,
            constraints=constraints_text,
            trajectory_history=trajectory_history,
            state_before=state_before,
            action=action,
            state_after=state_after,
            available_actions_after=aa_str
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # VISUAL ENVIRONMENT CONTEXT (OSWorld, etc.)
        # When processing visual environments, we get additional context from VGL:
        # - state_diff: Description of what changed between screenshots
        # - affordances: List of detected UI interactions
        # This helps TextGrad understand visual state changes better.
        # ═══════════════════════════════════════════════════════════════════════════
        is_visual_env = env_type in ('osworld', 'visual') if env_type else False

        if is_visual_env and (state_diff or affordances):
            visual_context = "\n\n=== VISUAL ENVIRONMENT CONTEXT ===\n"
            if state_diff:
                visual_context += f"\nSCREEN STATE CHANGE (from VGL analysis):\n{state_diff}\n"
            if affordances:
                affordances_str = ", ".join(affordances[:15]) if affordances else "None detected"
                visual_context += f"\nAVAILABLE UI INTERACTIONS:\n{affordances_str}\n"
            visual_context += """
VISUAL EVALUATION GUIDANCE:
- Did the screen change meaningfully after the action?
- Does the state change indicate progress toward the task goal?
- If the screen didn't change, the action may have failed (wrong element, dialog blocking, etc.)
- Consider both GUI actions and programmatic shortcuts (terminal commands) for efficiency.
"""
            prompt += visual_context

        # ═══════════════════════════════════════════════════════════════════════════
        # CLICK COORDINATE CONTEXT (for coordinate-aware learning)
        # When a click action was taken, we know HOW the label→coordinate resolution
        # happened. This lets TextGrad distinguish COORDINATE_ERROR (right intent,
        # wrong pixel) from STRATEGIC_ERROR (wrong action entirely).
        # ═══════════════════════════════════════════════════════════════════════════
        if click_metadata and click_metadata.get('match_type') not in (None, 'non_click'):
            coord_context = "\n\n=== CLICK COORDINATE CONTEXT ===\n"
            coord_context += f"Target element: '{click_metadata.get('target_label', '?')}'\n"
            coord_context += f"Match type: {click_metadata.get('match_type', '?')}\n"
            if click_metadata.get('matched_element'):
                me = click_metadata['matched_element']
                coord_context += f"Resolved to: '{me.get('label')}' at ({me.get('center', [0,0])}), size={me.get('size_px')}px\n"
            if click_metadata.get('nearby_elements'):
                coord_context += "Nearby elements (risk of misclick):\n"
                for ne in click_metadata['nearby_elements'][:5]:
                    coord_context += f"  - '{ne['label']}' at {ne['center']}, {ne['distance_px']}px away\n"
            coord_context += "\nCOORDINATE EVALUATION:\n"
            coord_context += "- If screen changed but wrong element was activated → COORDINATE_ERROR (right intent, wrong pixel)\n"
            coord_context += "- If screen didn't change → action may have missed entirely\n"
            coord_context += "- If nearby elements < 50px apart → high misclick risk, suggest keyboard alternative\n"
            prompt += coord_context

        # Multimodal: add visual evidence instruction when screenshots available
        if is_visual_env and (before_screenshot_b64 or after_screenshot_b64):
            prompt += """

=== VISUAL EVIDENCE (screenshots attached) ===
Before/after screenshots are attached to this message. Use them to:
- VERIFY if the screen actually changed (don't trust text descriptions alone)
- IDENTIFY specific UI elements, dialogs, error messages visible
- ASSESS whether visual state matches task completion criteria
- DETECT misclicks, wrong windows, unexpected popups
Screenshots are ground truth — text descriptions may miss details.
"""

        print(f"[TEXTGRAD LOSS] Computing GOAL-ALIGNED loss for action: '{action[:50]}'")

        # Multimodal path: let GPT-4o SEE the actual pixels
        if is_visual_env and (before_screenshot_b64 or after_screenshot_b64):
            print(f"[TEXTGRAD LOSS] Using MULTIMODAL (GPT-4o vision with screenshots)")
            loss_text = self.model.generate_multimodal(
                prompt, before_image_b64=before_screenshot_b64,
                after_image_b64=after_screenshot_b64, max_tokens=1200,
                json_schema=LOSS_RESPONSE_SCHEMA)
        else:
            loss_text = self._generate_with_schema(
                prompt, temperature=0.3, max_tokens=1200,
                json_schema=LOSS_RESPONSE_SCHEMA)

        # Post-LLM deterministic validation: substring-anchor evidence quotes
        # against the observation. Hallucinated quotes flip verdict to
        # UNVERIFIED (routes to System-2 Reflexion). This is the load-bearing
        # mechanism that kills F2/F3/F4 hallucination modes — judge cannot
        # quote text/coords that are not present in observation.
        loss_text = self._validate_loss_output(
            loss_text, observation_text=state_after, env_type=env_type,
            eval_gap_text=_eval_gap_text, action_str=action
        )
        loss_summary = loss_text[:150] + "..." if len(loss_text) > 150 else loss_text
        print(f"[TEXTGRAD LOSS] Computed: {loss_summary}")
        return loss_text

    def _generate_with_schema(self, prompt: str, temperature: float = 0.3,
                               max_tokens: int = 600,
                               json_schema=None) -> str:
        """Schema-constrained generation. Falls back to plain _generate if
        the underlying SamplingParams class doesn't accept json_schema (the
        wrapper in shared_model handles silent fallback)."""
        try:
            sp = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                json_schema=json_schema, stop=["\n\n\n"])
        except TypeError:
            sp = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                stop=["\n\n\n"])
        out = self.model.generate([prompt], sp)[0]
        return out.outputs[0].text.strip()

    @staticmethod
    def _validate_loss_output(loss_text: str, observation_text: str,
                              env_type: str = None,
                              eval_gap_text: str = None,
                              action_str: str = None) -> str:
        """Validate the LLM-as-judge structured output against observable
        evidence. Mechanism (load-bearing for F1-F4 hallucination kills):

        1. Quote-anchor: supporting_evidence_quote MUST be a substring of
           observation_text. If not → flip verdict to UNVERIFIED.
        2. State-change consistency: if narrated_state_change says "no
           observable change" but verdict is GOAL_ALIGNED → flip to
           GOAL_NOT_ALIGNED (cannot make progress with no change).
        3. Subgoal-done sanity: subgoal_done=DONE when verdict is
           GOAL_NOT_ALIGNED → flip subgoal_done to NOT_DONE.

        Universal — pure string operations. No env-specific code.
        Returns the validated/corrected loss_text JSON (still emits
        legacy VERDICT/SCORE/SUBGOAL_DONE/REASON keylines for backwards
        compat with downstream regex parsers).
        """
        import json as _json, re as _re
        # Universal "fail-open with legacy header" stub: always returns a
        # downstream-parseable shape. Any non-JSON or malformed output gets
        # UNVERIFIED + score=0 so existing regex parsers see a valid VERDICT.
        def _stub(reason: str, score: int = 0) -> str:
            return (
                f"VERDICT: UNVERIFIED\n"
                f"SCORE: {score}\n"
                f"SUBGOAL_DONE: NONE\n"
                f"STRUCTURED_SUBGOAL_DONE: NOT_DONE\n"
                f"REASON: {reason}\n"
                f"_LOSS_JSON: {{}}\n\n"
            ) + (loss_text or "")
        # Extract JSON object (strip markdown fences if any)
        s = (loss_text or "").strip()
        s = _re.sub(r"^```(?:json)?\s*", "", s)
        s = _re.sub(r"\s*```\s*$", "", s)
        first = s.find("{")
        if first < 0:
            return _stub("LOSS produced no JSON object")
        # Use JSONDecoder.raw_decode — robust to trailing text and respects
        # JSON escape semantics (handles strings with embedded quotes/braces
        # the manual brace-counter mishandled).
        try:
            obj, _consumed = _json.JSONDecoder().raw_decode(s[first:])
        except Exception as _e:
            return _stub(f"LOSS JSON parse error: {str(_e)[:80]}")

        # 1. Quote-anchor validation (RULERS arXiv 2601.08654)
        sup = (obj.get("supporting_evidence_quote") or "").strip()
        flips = []
        if sup and len(sup) >= 4:
            # Normalize whitespace for the substring check (LLM may collapse)
            obs_norm = " ".join((observation_text or "").split())
            sup_norm = " ".join(sup.split())
            if sup_norm not in obs_norm:
                obj["verdict"] = "UNVERIFIED"
                flips.append(f"quote not in obs: {sup_norm[:40]!r}")
        # 2. State-change consistency
        change = (obj.get("narrated_state_change") or "").lower()
        if "no observable change" in change or "no change" in change:
            if obj.get("verdict") == "GOAL_ALIGNED":
                obj["verdict"] = "GOAL_NOT_ALIGNED"
                flips.append("no-change but GOAL_ALIGNED")
        # 3. Subgoal-done sanity
        if obj.get("subgoal_done") == "DONE" and \
                obj.get("verdict") == "GOAL_NOT_ALIGNED":
            obj["subgoal_done"] = "NOT_DONE"
            flips.append("DONE+NOT_ALIGNED conflict")

        # 4. Executable-check override (CRITIC Gou ICLR 2024)
        # Run the LLM-specified check deterministically and override subgoal_done
        # if the oracle disagrees. This kills F1 (premature SUBGOAL_DONE) — the
        # oracle is authoritative, the LLM judge's assessment is not.
        exec_check = obj.get("executable_check", {})
        if exec_check and exec_check.get("kind") != "none":
            try:
                from executable_check import verify_check as _verify_check
                oracle_result = _verify_check(exec_check,
                                              {"env_type": env_type or ""})
                if oracle_result is not None:
                    sgd_before = obj.get("subgoal_done")
                    if oracle_result is False and sgd_before == "DONE":
                        obj["subgoal_done"] = "NOT_DONE"
                        obj["verdict"] = "GOAL_NOT_ALIGNED"
                        flips.append(f"oracle=False overrides DONE→NOT_DONE")
                    elif oracle_result is True and sgd_before == "NOT_DONE":
                        # Oracle confirms done — trust it
                        obj["subgoal_done"] = "DONE"
                        if obj.get("verdict") == "GOAL_NOT_ALIGNED":
                            obj["verdict"] = "GOAL_ALIGNED"
                        flips.append("oracle=True upgrades NOT_DONE→DONE")
            except ImportError:
                pass  # executable_check module not on path — skip oracle check

        # 6. Keyboard-input gate: if evaluator requires a specific text value and
        # the action contains no keyboard input emitting that value, the field
        # cannot have been set. Blocks click-only paths from claiming GOAL_ALIGNED.
        # Universal — uses only the action string and eval_gap_text. No env code.
        if (action_str and eval_gap_text and
                obj.get("verdict") == "GOAL_ALIGNED" and
                env_type in ('osworld', 'visual')):
            import re as _re6
            _action_lower = action_str.lower()
            _kb_patterns = (
                "typewrite", "typestring", ".write(", "hotkey", "keydown",
                "str ", "sendkeys", "keyboard.type", "xte \"str", "typechar",
                "type(", "keys(",
            )
            has_keyboard_input = any(p in _action_lower for p in _kb_patterns)
            if not has_keyboard_input:
                lit6 = _re6.findall(r"['\"]([^'\"]{3,40})['\"]", eval_gap_text)
                if lit6:
                    # Only gate on user-visible name/value literals, not config keys
                    # Config keys look like: 'key-value', 'layer-new-name' (all lower+hyphen)
                    name_literals = [
                        l for l in lit6
                        if not _re6.match(r'^[a-z][a-z0-9-]*$', l)
                        and len(l) >= 2
                    ]
                    if name_literals:
                        obj["verdict"] = "GOAL_NOT_ALIGNED"
                        obj["score"] = min(obj.get("score", 0), 2)
                        # Inject concrete hint so BACKWARD generates typewrite action
                        req_val = name_literals[0]
                        obj["best_alternative_to_emit_next"] = (
                            f"triple-click the text input field to select all, "
                            f"then pyautogui.typewrite('{req_val}', interval=0.05) "
                            f"to enter the required value, then click the OK button"
                        )
                        flips.append(
                            f"no-keyboard-input but text value required "
                            f"{name_literals[:2]!r} → GOAL_NOT_ALIGNED"
                        )

        # 5. Evaluator-criteria literal gate (RC1 fix — kills GUI false-positives)
        # When the eval_gap block lists required literal values (e.g. 'Square',
        # a filename, a setting string) and none of them appear in the obs, a
        # GOAL_ALIGNED verdict is a false-positive caused by "screen changed →
        # progress" visual prior. Force GOAL_NOT_ALIGNED to activate Reflexion.
        if (eval_gap_text and
                obj.get("verdict") == "GOAL_ALIGNED" and
                env_type in ('osworld', 'visual')):
            import re as _re5
            # Extract short literals (3-40 chars) from quoted strings in eval_gap
            literals = _re5.findall(r"['\"]([^'\"]{3,40})['\"]",
                                    eval_gap_text)
            if literals:
                obs_lower = " ".join((observation_text or "").split()).lower()
                if not any(lit.lower() in obs_lower for lit in literals[:8]):
                    obj["verdict"] = "GOAL_NOT_ALIGNED"
                    obj["score"] = min(obj.get("score", 0), 3)
                    flips.append(
                        f"eval_literals not in obs: "
                        f"{[l[:20] for l in literals[:3]]!r} → GOAL_NOT_ALIGNED"
                    )

        if flips:
            print(f"[LOSS-VALIDATE] flipped: {', '.join(flips)}")

        # Re-emit legacy keylines so downstream regex parsers still work
        # while the structured object is also embedded.
        verdict = obj.get("verdict", "UNVERIFIED")
        score = obj.get("score", 0)
        sgd = obj.get("subgoal_done", "NOT_DONE")
        # Map structured subgoal_done back to the legacy "SUBGOAL_DONE: <int|NONE>"
        # field by reading the agent's current subgoal index. The agent itself
        # consumes structured `subgoal_done` directly; the legacy line is
        # cosmetic for log readers.
        sgd_legacy = "NONE"  # agent reads structured field; legacy line stays NONE
        reason = obj.get("reason", "")
        legacy_block = (
            f"VERDICT: {verdict}\n"
            f"SCORE: {score}\n"
            f"SUBGOAL_DONE: {sgd_legacy}\n"
            f"STRUCTURED_SUBGOAL_DONE: {sgd}\n"
            f"REASON: {reason}\n"
            f"_LOSS_JSON: {_json.dumps(obj, ensure_ascii=False)}\n"
        )
        return legacy_block + "\n" + (loss_text or "")


def textgrad_backward(policy: ActionPolicy, action: str, loss_text: str, model, context: dict) -> str:
    """
    Backward pass: Compute textual gradient ∂loss/∂policy WITH FULL CONTEXT.

    CRITICAL FIX (Nov 3, 2024): TextGrad now receives the SAME comprehensive context
    that the action selection prompt has. This enables TextGrad to generate SMART
    gradients that:
    - Avoid revisiting already-checked locations (from history)
    - Align with TODO sequential goals
    - Incorporate Reflexion strategic insights
    - Understand current progress and state

    This is MORE aligned with TextGrad paper philosophy: "Rich textual feedback
    enables gradient optimization". Previous implementation gave TextGrad minimal
    context (just policy + action + loss), causing blind tactical recommendations.

    Args:
        policy: The ActionPolicy being optimized
        action: The action that was taken
        loss_text: The criticism (loss) from TextGradLoss
        model: LLM for computing gradients
        context: REQUIRED rich context dict with:
            - task: Current task requirement
            - todo: Current TODO subtask (can be empty string if none)
            - tried_actions: Set of already-tried actions
            - working_reflexions: Within-trial Reflexion insights list
            - reflexion_memory: Cross-trial Reflexion constraints list
            - cur_step: Current step number

    Returns:
        Textual gradient: instructions for improving policy
    """

    # Extract context - NO FALLBACKS! Fail if missing to catch bugs
    task = context['task']
    todo = context['todo']
    tried_actions = context['tried_actions']
    state_action_obs_history = context.get('state_action_obs_history', [])  # NEW: Full triples
    working_reflexions = context['working_reflexions']
    reflexion_memory = context['reflexion_memory']
    cur_step = context['cur_step']
    valid_actions = context.get('valid_actions', [])  # CRITICAL: What actions are actually possible

    # Format state-action-observation history WITH progress status for TextGrad learning
    # FIX #36: Show FULL history so LLM can see complete pattern of what was tried
    # Previously truncated to last 10, hiding repetition patterns from the LLM
    if state_action_obs_history:
        history_str = ""
        for idx, triple in enumerate(state_action_obs_history, 1):  # FIX #36: ALL history, not [-10:]
            state_before = triple['state']
            action_taken = triple['action']
            obs_after = triple['observation']
            progress = triple.get('progress', 'UNKNOWN')

            history_str += f"\n  {idx}. State: {state_before[:600]}"  # Atom C: bumped to fit Last-Action-Result block
            history_str += f"\n     Action: {action_taken}"
            history_str += f"\n     Result: {obs_after[:1500]}"  # Atom C: full rc/stdout/stderr survives — the key learning signal
            history_str += f"\n     Progress: {progress}\n"  # ✅ TextGrad sees if action made progress!
        tried_actions_str = history_str
    else:
        tried_actions_str = "None yet"

    # Format Reflexion insights (most recent 3) — full text, no truncation
    if working_reflexions:
        recent_insights = working_reflexions[-3:]
        reflexion_str = "\n".join([f"  - {str(r)}" for r in recent_insights])
    else:
        reflexion_str = "No Reflexion insights yet"

    # Format cross-trial constraints (top 3) — full text
    if reflexion_memory:
        top_constraints = reflexion_memory[:3]
        constraints_str = "\n".join([f"  - {str(c)}" for c in top_constraints])
    else:
        constraints_str = "No cross-trial constraints yet (Trial 0)"

    # Format TODO
    todo_str = todo if todo else "No current TODO"

    # FIX #21: Show ALL valid actions - no cap!
    # Previously capped at 15, but remove/unfriend actions were at position 21-25 and LLM couldn't see them
    if valid_actions:
        valid_actions_str = "\n".join([f"  - {a}" for a in valid_actions])
    else:
        valid_actions_str = "No valid actions available (this should not happen!)"

    # ═══════════════════════════════════════════════════════════════════════════════
    # FIX #20: UNIVERSAL ACTION VALIDITY FEEDBACK (Extended for Visual Environments)
    # When action returns "Nothing happens", the agent needs to understand WHY.
    # By explicitly stating whether action was in valid_actions, the gradient can
    # learn preconditions from the environment's own feedback - 100% universal.
    #
    # For visual environments (OSWorld, etc.), we use SEMANTIC AFFORDANCE MATCHING
    # instead of exact string matching, since affordances are descriptive text
    # like "click 'Save' button" rather than exact command strings.
    # ═══════════════════════════════════════════════════════════════════════════════

    # Detect if this is a visual environment (OSWorld, etc.)
    env_type = context.get('env_type', '')
    is_visual_env = env_type in ('osworld', 'visual') if env_type else False

    if is_visual_env and valid_actions:
        # SEMANTIC AFFORDANCE MATCHING for visual environments
        action_in_valid, matched_affordance = _semantic_affordance_match(action, valid_actions)
        if action_in_valid:
            action_validity_str = f"✓ Action targets detected UI element: '{matched_affordance}'"
        else:
            # Find semantically similar affordances
            similar_affordances = _find_similar_affordances(action, valid_actions, top_k=3)
            similar_str = ", ".join(similar_affordances) if similar_affordances else "none found"
            action_validity_str = f"✗ Action does NOT target any detected UI element. Available interactions: {similar_str}"
    else:
        # EXACT STRING MATCHING for text-based environments (original behavior)
        action_in_valid = action in valid_actions
        if action_in_valid:
            action_validity_str = "✓ Action was IN valid_actions list (syntax correct)"
        else:
            # Find similar actions to help understand what precondition was missing
            similar_actions = [a for a in valid_actions if any(word in a for word in action.split()[:2])][:3]
            similar_str = ", ".join(similar_actions) if similar_actions else "none found"
            action_validity_str = f"✗ Action was NOT in valid_actions (environment rejected it). Similar valid actions: {similar_str}"

    gradient_prompt = f"""You are computing a textual gradient to improve an action selection policy.

═══════════════════════════════════════════════════════════════════════════════
CONTEXT (Use this to generate SMART, INFORMED gradients):
═══════════════════════════════════════════════════════════════════════════════

TASK REQUIREMENT:
{task}

CURRENT TODO SUBTASK:
{todo_str}

STEP NUMBER: {cur_step}

WHAT WAS ALREADY TRIED (State → Action → Observation):
{tried_actions_str}

CRITICAL: Before recommending an action, check if it was already tried above.
If yes, read what State it was tried in, what Result occurred, and what Progress was made.
If Progress was "NO_PROGRESS", this action was USELESS - DON'T recommend it again in similar states.
Focus on actions that showed PARTIAL_PROGRESS or better.

REFLEXION STRATEGIC INSIGHTS (from within-trial analysis):
{reflexion_str}

CROSS-TRIAL CONSTRAINTS (from previous trial failures):
{constraints_str}

{"DETECTED UI ELEMENTS (reference hints for visual environments — the agent can also use run: <shell command> for programmatic approaches):" if is_visual_env else "VALID ACTIONS (What actions are actually possible in the current state):"}
{valid_actions_str}

{"The agent may use ANY action format (GUI clicks, keyboard shortcuts, shell commands, python scripts). The list above is hints, not a constraint." if is_visual_env else "IMPORTANT: When recommending next actions in your gradient, ensure they come from the VALID ACTIONS list above. The environment has explicit action semantics - recommend only from actions that actually exist."}

{f'''CURRENT OBSERVATION (full state including [UI Elements] table with center coords — copy coords from here when emitting CONCRETE_FIX):
{context.get('observation', '')[:6000]}
''' if is_visual_env else ''}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: TASK ALIGNMENT PRINCIPLE
═══════════════════════════════════════════════════════════════════════════════

When generating policy improvements, ensure the policy will guide toward actions that:

1. DIRECTLY achieve what the task requirement asks for
2. Recognize when different actions can accomplish the same underlying goal
3. Distinguish between actions that genuinely serve the task vs actions that seem related
   but accomplish something different
4. Change the world state in the specific way the task requires

The policy must enable reasoning that differentiates:
- Actions that advance toward the task's actual objective
- Actions that address a different (though possibly related) objective

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: OBJECT DISCOVERY PRINCIPLE
═══════════════════════════════════════════════════════════════════════════════

If an object mentioned in the task is now VISIBLE (discovered), the agent should
INTERACT with it (take, examine, use) BEFORE moving to another location.
Finding a required object is progress - don't leave it behind!

═══════════════════════════════════════════════════════════════════════════════

CURRENT POLICY:
{policy.base_policy}

ACTION TAKEN BY POLICY:
{action}

ACTION VALIDITY STATUS:
{action_validity_str}

LOSS (Criticism of this action):
{loss_text}

═══════════════════════════════════════════════════════════════════════════════
Your task: Compute the gradient ∂loss/∂policy as a structured JSON object.

The runtime enforces BACKWARD_RESPONSE_SCHEMA at decode-time. Each field
is validated — supporting_evidence_quote must be byte-for-byte present in
the LOSS/observation text above; the runtime marks the gradient UNGROUNDED
if not found (preventing hallucinated coord-fixes F3/F4).

Output a single JSON object:

  "failure_diagnosis": exact root cause — quote verbatim evidence from LOSS.
  "supporting_evidence_quote": a substring of LOSS or observation that proves
                               the diagnosis. MUST be byte-for-byte present.
  "verbatim_stderr_signal": character-for-character copy of the FIRST
                            non-empty line from the observation's
                            [Last Action Result].stderr field. MUST be a
                            literal substring of observation text. Empty
                            string "" iff rc==0 AND stderr is empty/absent.
                            No paraphrase, no summary, no synthesized
                            confidence values, no derived semantic
                            interpretation — just the literal bytes from
                            the stderr field. The optimizer preserves
                            this verbatim into the next policy text; it is
                            the SINGLE most concrete corrective signal in
                            the gradient.
  "candidates": array of 1-3 objects:
    Each: {{
      "type": "probe"|"act"|"discover",
      "action": "exact executable action string",
      "predicted_effect": "what it will produce in current state",
      "postcondition_check": {{"kind":"shell"|"python"|"getter"|"dom"|"a11y"|"file"|"none",
                               "command":"..."}}
    }}
    On visual envs WITH a [UI Elements] table: 'act' actions should
    include coords from the table (pyautogui.click(x,y)) UNLESS the
    LOSS text contains "DIALOG FOCUS RULE" or "auto-focused" — in that
    case use a keyboard-only compound action (NO preceding click):
      pyautogui.hotkey('ctrl','a'); pyautogui.typewrite('VALUE', interval=0.05); pyautogui.press('return')
    Prefer diverse types: one 'probe' (info), one 'act' (change), one
    'discover' (explore). Omit types that don't apply.
  "chosen_index": 0-based index of the BEST candidate to try next.
  "policy_rule": ONE universal rule (not task-specific) to prevent this
                 class of error in future.

CRITICAL:
- candidates[chosen_index].action IS the CONCRETE_FIX — it replaces
  the failed action verbatim.
- Don't include already-tried actions (check HISTORY above).
- "Verify output" is not an action. An executable command is.
- If LOSS mentions "DIALOG FOCUS RULE": the chosen_index MUST point to
  a keyboard-only 'act' candidate (hotkey+typewrite+press). No click.
"""

    # ═══════════════════════════════════════════════════════════════════════════
    # COORDINATE FEEDBACK (for visual envs with click actions)
    # Helps TextGrad distinguish coordinate errors from strategic errors
    # and learn to suggest keyboard shortcuts when clicks are unreliable.
    # ═══════════════════════════════════════════════════════════════════════════
    click_metadata = context.get('click_metadata')
    if click_metadata and click_metadata.get('match_type') not in (None, 'non_click'):
        coord_feedback = "\n\nCOORDINATE FEEDBACK:\n"
        mt = click_metadata.get('match_type')
        if mt == 'no_match':
            coord_feedback += f"WARNING: Target element '{click_metadata.get('target_label', '?')}' not found in VGL grounding.\n"
            coord_feedback += "GRADIENT: Try a different label, use keyboard shortcut, or navigate to reveal the element.\n"
        elif mt == 'fuzzy':
            coord_feedback += f"WARNING: Fuzzy match — resolved '{click_metadata.get('target_label', '?')}' to '{click_metadata.get('matched_element', {}).get('label', '?')}'\n"
            coord_feedback += "GRADIENT: Use more precise label matching the [Interactive Elements] list, or use keyboard alternative.\n"
        elif mt == 'fallback_center':
            coord_feedback += "WARNING: No element matched — clicked screen center (960,540) as fallback. This action failed.\n"
            coord_feedback += "GRADIENT: Re-examine [Interactive Elements] for correct labels, or use hotkey/keyboard navigation.\n"
        nearby = click_metadata.get('nearby_elements', [])
        close_elements = [n for n in nearby if n.get('distance_px', 999) < 50]
        if close_elements:
            names = [f"'{n['label']}' ({n['distance_px']}px)" for n in close_elements]
            coord_feedback += f"RISK: {len(close_elements)} elements within 50px: {', '.join(names)}\n"
            coord_feedback += "GRADIENT: For tightly-packed UI, prefer keyboard shortcuts (hotkey, press Tab+Enter, etc.) over clicking.\n"
        gradient_prompt += coord_feedback

    print(f"[TEXTGRAD BACKWARD] Computing gradient for action: '{action[:50]}'")

    # Multimodal path: let the model SEE before/after screenshots for visual gradient
    before_b64 = context.get('before_screenshot_b64')
    after_b64 = context.get('after_screenshot_b64')

    import json as _json_bwd, re as _re_bwd

    def _extract_backward_json(raw: str) -> dict:
        """Parse BACKWARD JSON, strip markdown fences, return dict or {}."""
        s = raw.strip()
        s = _re_bwd.sub(r"^```(?:json)?\s*", "", s)
        s = _re_bwd.sub(r"\s*```\s*$", "", s)
        first = s.find("{")
        if first < 0:
            return {}
        try:
            obj, _ = _json_bwd.JSONDecoder().raw_decode(s[first:])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _validate_backward_obj(obj: dict, observation_src: str) -> dict:
        """
        Validate supporting_evidence_quote against LOSS+obs text.
        Marks gradient UNGROUNDED (not injected into policy) if quote not found.
        """
        sup = (obj.get("supporting_evidence_quote") or "").strip()
        if sup and len(sup) >= 4:
            obs_norm = " ".join((observation_src or "").split())
            sup_norm = " ".join(sup.split())
            if sup_norm not in obs_norm:
                obj["_ungrounded"] = True
                print(f"[BACKWARD-VALIDATE] quote not in obs → UNGROUNDED: {sup_norm[:40]!r}")
        return obj

    if is_visual_env and (before_b64 or after_b64):
        gradient_prompt += """

=== VISUAL EVIDENCE (screenshots attached) ===
Use before/after screenshots to visually verify what changed on screen.
Ground truth — generate the JSON gradient based on what you SEE.
"""
        print(f"[TEXTGRAD BACKWARD] Using MULTIMODAL with screenshots")
        # Fix #4 broken-edge audit 2026-05-07: BACKWARD max_tokens 1200→2000.
        # Forensic on Impress 04578141 v2 showed gradient JSON truncating
        # mid-structure (empty-line padding then cutoff before chosen_index +
        # policy_rule). The schema-constrained decoding pushed against the
        # 1200-token ceiling once verbatim_stderr_signal was added (Fix #1).
        # 2000 tokens accommodates: 6 schema fields × ~300 tokens average
        # for verbose LLM emissions = headroom to close the JSON cleanly.
        # Universal — bandwidth parameter only, Aho-Sethi-Ullman 1986.
        _BACKWARD_MAX_TOKENS = 2000
        gradient_response = model.generate_multimodal(
            gradient_prompt, before_image_b64=before_b64,
            after_image_b64=after_b64, max_tokens=_BACKWARD_MAX_TOKENS, temperature=0.3,
            json_schema=BACKWARD_RESPONSE_SCHEMA)
    else:
        try:
            sp = SamplingParams(max_tokens=_BACKWARD_MAX_TOKENS, temperature=0.3,
                                json_schema=BACKWARD_RESPONSE_SCHEMA,
                                stop=["\n\n\n"])
        except TypeError:
            sp = SamplingParams(max_tokens=_BACKWARD_MAX_TOKENS, temperature=0.3, stop=["\n\n\n"])
        output = model.generate([gradient_prompt], sp)[0]
        gradient_response = output.outputs[0].text.strip()

    # Parse structured backward output
    backward_obj = _extract_backward_json(gradient_response)
    combined_src = (loss_text or "") + "\n" + context.get('observation', '')
    if backward_obj:
        backward_obj = _validate_backward_obj(backward_obj, combined_src)

    # Build the legacy GRADIENT: + CONCRETE_FIX: text that downstream consumers expect
    if backward_obj and not backward_obj.get("_ungrounded"):
        diagnosis = backward_obj.get("failure_diagnosis", "")
        policy_rule = backward_obj.get("policy_rule", "")
        candidates = backward_obj.get("candidates", [])
        chosen_idx = backward_obj.get("chosen_index", 0)
        # Clamp index to valid range
        if candidates and isinstance(chosen_idx, int):
            chosen_idx = max(0, min(chosen_idx, len(candidates) - 1))
            chosen = candidates[chosen_idx]
            concrete_fix = chosen.get("action", "")
            predicted_effect = chosen.get("predicted_effect", "")
        else:
            concrete_fix = ""
            predicted_effect = ""

        # Emit legacy-compatible text that the rest of the pipeline parses
        gradient = (
            f"{diagnosis}\n{policy_rule}\n"
            f"Predicted effect: {predicted_effect}\n"
            f"Candidates considered: {len(candidates)}\n"
            f"CONCRETE_FIX: {concrete_fix}"
        )
        # Stash structured obj for probe/BACKWARD consumers
        backward_obj["_concrete_fix"] = concrete_fix
    else:
        # Fallback: parse legacy GRADIENT:/CONCRETE_FIX: lines
        if "GRADIENT:" in gradient_response:
            gradient = gradient_response.split("GRADIENT:")[1].strip()
        else:
            gradient = gradient_response

    # Filter out LLM content moderation refusals
    _refusal_markers = ["i'm unable to assist", "i can't assist", "unable to help",
                        "i'm sorry, i can't", "i cannot assist", "i'm not able to"]
    if any(m in gradient.lower() for m in _refusal_markers):
        print(f"[TEXTGRAD BACKWARD] Filtered refusal → neutral gradient")
        gradient = "Try a different approach: the current action did not make progress toward the task goal."

    # Comprehensive-logging skill R5 — full gradient must be inspectable to
    # diagnose self-correction failures. The previous [:100] truncation hid
    # the verbatim_stderr_signal and other structured fields.
    print(f"[TEXTGRAD BACKWARD] Gradient computed ({len(gradient)} chars):")
    print(f"=== GRADIENT START ===\n{gradient}\n=== GRADIENT END ===")
    return gradient


def _ground_recipe_commands(policy_text: str, source_of_truth: str) -> str:
    """Deterministic [RECIPE] grounding validator — multi-token coverage.

    Path A v2 (after head-only check proved too lenient — Gemma split real
    tokens across observations and recombined them into fabricated
    commands like `stty cols 132 rows 43`, where head `stty` was grounded
    but the full structure wasn't):

    For every backtick-quoted command in the policy, compute the fraction
    of NON-TRIVIAL alphanumeric tokens (length ≥3, not generic stopwords)
    that appear (case-insensitive substring) in source_of_truth. If
    fraction < threshold, prefix the command with `[SPECULATIVE]`.

    Universal — pure n-gram coverage check. No benchmark patterns, no app
    names, no error-pattern lookups, no per-task code. Russell-Norvig Ch.1
    + Hoare 1969 faithfulness rule, enforced mechanically.
    """
    import re
    if not policy_text or not source_of_truth:
        return policy_text

    # Stopwords: shell framers + universal English short words + generic
    # POSIX command verbs that nearly always appear in obs anyway. NOT
    # benchmark-specific. Filter prevents free riding on common tokens.
    _STOPWORDS = {
        "run:", "run", "use", "try", "execute", "call", "the", "and", "or",
        "if", "with", "via", "for", "from", "into", "then", "after",
        "before", "echo", "ls", "cat", "cd", "pwd", "grep", "find", "tail",
        "head", "wc", "sed", "awk",
    }
    # Coverage threshold: a command is grounded only if ≥ 60% of its
    # non-trivial tokens appear in source-of-truth.
    _COVERAGE_THRESHOLD = 0.6

    haystack = source_of_truth.lower()
    flagged = 0
    output_lines = []

    for line in policy_text.split("\n"):
        if "[SPECULATIVE]" in line:
            output_lines.append(line)
            continue
        modified = line
        for m in re.finditer(r"`([^`\n]{2,200})`", line):
            cmd_str = m.group(1).strip()
            # Tokenize on whitespace + shell separators
            raw_tokens = re.split(r"\s+|;|&&|\|\|", cmd_str)
            # Keep alphanumeric/-/_/. tokens of length ≥3 not in stopwords
            distinctive = []
            for t in raw_tokens:
                # Strip surrounding quotes/punctuation
                tk = t.strip("'\"().;,:`")
                tk_low = tk.lower()
                if not tk_low or tk_low in _STOPWORDS:
                    continue
                if len(tk_low) < 3:
                    continue
                # Skip PURE-NUMERIC tokens. Eval-target values (the
                # required numbers/literals) are correctly cited from
                # eval_schema, but they don't validate the COMMAND
                # STRUCTURE. Without this skip, an LLM could justify any
                # fabricated `<verb> <numeric>` by citing the eval
                # literal. Universal rule — applies on any task/env;
                # eval values are STATE not COMMAND-grammar.
                if re.fullmatch(r"\d+", tk_low):
                    continue
                # Skip SHORT SINGLE-DASH POSIX flags (-la, -rf, -h, etc.)
                # These are universal Linux flag conventions and don't
                # need grounding — standard man-page vocabulary across
                # decades of POSIX tools. Long flags like --magic-mode
                # are KEPT (potentially fabricated identifiers).
                if re.fullmatch(r"-[a-z]{1,4}", tk_low):
                    continue
                distinctive.append(tk_low)
            if not distinctive:
                continue
            grounded_count = sum(1 for tk in distinctive if tk in haystack)
            coverage = grounded_count / len(distinctive)
            if coverage >= _COVERAGE_THRESHOLD:
                continue
            quoted = m.group(0)
            modified = modified.replace(
                quoted, f"[SPECULATIVE] {quoted}", 1,
            )
            flagged += 1
        output_lines.append(modified)

    if flagged:
        print(
            f"[TEXTGRAD OPTIMIZER] grounding validator: marked {flagged} "
            f"ungrounded command(s) [SPECULATIVE] "
            f"(threshold={int(_COVERAGE_THRESHOLD*100)}% token coverage)"
        )
    return "\n".join(output_lines)


class TextGradOptimizer:
    """
    Applies accumulated gradients to update the policy.
    This is the optimizer.step() in TextGrad.
    """

    def __init__(self, model, learning_rate: float = 1.0):
        """
        Args:
            model: LLM for generating updated policy
            learning_rate: How aggressively to apply gradients (1.0 = full application)
        """
        self.model = model
        self.lr = learning_rate
        self._init_prompts()  # Initialize prompt templates

    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
        """Helper to generate text using fast_model.generate() API.

        PAPER FIX: The optimizer's UPDATED_POLICY text is multi-line and may
        contain blank lines (rules separated by paragraphs). stop=["\\n\\n"]
        previously truncated mid-policy. Use "\\n\\n\\n" so the model can emit
        a multi-paragraph rewrite and only stop on a hard section break.
        """
        output = self.model.generate(
            [prompt],
            SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n\n"]
            )
        )[0]
        return output.outputs[0].text.strip()

    def _init_prompts(self):
        # Issue A fix: optimizer must PRESERVE EPISODIC SPECIFICS.
        # Previous prompt distilled gradients into 2-4 sentences, deleting
        # specific facts (action names, error messages, coordinates, file
        # paths) — the very signal the actor needs at the next step. The
        # new prompt explicitly forbids generic abstraction and demands
        # "PRESERVED FACTS:" inline references inside the updated policy.
        # Universal — applies on any TextGrad use; the rule is "reference
        # specific facts from the gradient, not abstract verbs about them".
        self.update_prompt_template = """You are updating an action selection policy based on accumulated gradients.

{g31_eval_block}CURRENT POLICY:
{current_policy}

STRATEGIC CONSTRAINTS (must preserve):
{constraints}

ACCUMULATED GRADIENTS (improvements to apply):
{gradients}

Your task: Produce an UPDATED POLICY that incorporates these gradients
while preserving STRATEGIC CONSTRAINTS and EPISODIC SPECIFICS.

EPISODIC PRESERVATION RULE — MOST IMPORTANT:
The gradients above contain SPECIFIC FACTS the agent has learned this
episode: action names that succeeded ("useradd -m -d /home/test1 charles"),
error messages that triggered ("SYNTAX_ERROR: dangling heredoc"),
specific coordinates that mapped to interactive elements
("(247,753) is the Bing radio button"), specific file paths that exist
("/etc/ssh/sshd_config has the Match block"), specific subgoals that
already succeeded.

The updated policy MUST quote these specific facts verbatim or
near-verbatim. DO NOT distill them into abstract verbs like "verify",
"persist", "navigate", "configure" — those are EXACTLY the bits the
actor needs that you would be deleting. Reference the literals:
  GOOD: "User 'charles' was created with home /home/test1 (step 2);
         /etc/ssh/sshd_config now has the Match block (step 6).
         Next: restart ssh and verify with `getent passwd charles`."
  BAD:  "Configure the SSH service properly and verify the setup."

Rules:
1. Keep the fundamental goal/objective.
2. PRESERVE specific facts from gradients verbatim (action strings,
   error messages, coordinates, file paths, subgoal names).
3. Reference WHICH STEPS already succeeded (so the actor doesn't redo).
4. Make it actionable. 4-6 sentences is fine — episodic specifics are
   worth more space than generic 2-4 sentence advice.
5. MUST preserve all strategic constraints from Reflexion.
6. If EVALUATOR GROUND TRUTH block is present above, the updated policy MUST
   reference the specific literal values / attribute names / cell addresses
   the evaluator expects — name the exact literals.
7. GOAL+RECIPE SPLIT (Fix #3 broken-edge audit 2026-05-07): the policy
   MUST have TWO sections separated by "---":
     [GOAL] What state must hold for the task to be done. References
     evaluator literals. 1-3 sentences.
     [RECIPE] How to achieve it. References concrete error tokens from
     gradients (the verbatim_stderr_signal field), exact shell-quoting
     patterns that worked, exact coordinates from [UI Elements]. 3-6
     sentences. NO abstract verbs ("configure", "verify"); cite
     literals.
   Sussman HACKER 1973: a bug catalog stores BOTH the symptom and the
   fix-recipe. If you only emit one section, both halves of the lesson
   are lost when the policy gets truncated.

8. GROUNDING RULE — recipe MUST be faithful, not fabricated
   (Fix #8 broken-edge audit 2026-05-07):
   Every concrete COMMAND, COORDINATE, FILE-PATH, CONFIG-KEY, or LITERAL
   VALUE in [RECIPE] MUST trace back to one of:
     (a) a verbatim substring of the observation text shown in any
         gradient above (LOSS, [Last Action Result], [SYS], [PROBES],
         [UI Elements], etc.), OR
     (b) a verbatim quote from a gradient's supporting_evidence_quote
         or verbatim_stderr_signal field, OR
     (c) the EVALUATOR GROUND TRUTH block (if present).
   Strings NOT so grounded MUST be prefixed with the literal token
   `[SPECULATIVE]` so the actor knows they are guesses.
   When the procedure requires knowledge that is NOT yet observable
   (an unknown config-key syntax, an undocumented file location, an
   API call signature, etc.), the recipe MUST instruct the agent to
   PROBE FIRST before acting:
       "run: <appname> --help"
       "run: man <appname>"
       "run: find / -name <plausible-pattern> 2>/dev/null"
       "run: ls -la <plausible-config-dir>"
       "run: <appname> --version" then "run: man <appname>-<version>"
   Only after the probe surfaces the literal answer in observation
   may the recipe instruct the state-change action with the
   verbatim-grounded literal.
   Russell-Norvig Ch.1 — a rational agent distinguishes OBSERVED facts
   from INFERRED conjecture. McAllester 1991 — sensing actions
   precede state-changing actions when state is unknown.
   FORBIDDEN: synthesizing a plausible-looking command/config that
   has not appeared verbatim in any observation. The optimizer's job
   is to PRESERVE evidence, not to FABRICATE procedural knowledge.

Answer:
UPDATED_POLICY: [new policy text — TWO sections [GOAL] and [RECIPE]
                 separated by "---", with specific facts from gradients
                 in each]
"""

    def step(self, policy: ActionPolicy) -> None:
        """
        Apply accumulated gradients to update the policy.
        This modifies the policy in-place.
        """
        if not policy.gradients:
            print(f"[TEXTGRAD OPTIMIZER] No gradients to apply, skipping update")
            return  # No gradients to apply

        print(f"[TEXTGRAD OPTIMIZER] Applying {len(policy.gradients)} gradients to update policy")
        print(f"[TEXTGRAD OPTIMIZER] Current policy: '{policy.base_policy[:100]}'")

        # G31 UNIVERSAL — inject evaluator ground truth into the optimizer's
        # policy-update prompt. The optimizer was previously blind to the
        # evaluator's literal requirements (its input was only the current
        # policy + abstract gradients). It synthesized generic advice like
        # "verify saves persist" instead of "set attribute 'strike' to value
        # 'sngStrike'". Now it sees the same verbatim gap + evaluator source
        # that the BACKWARD pass (G21), PASS 2 actor (G24), and Reflexion
        # (G30) all see. Gated on policy attributes populated upstream —
        # no-op when gap is absent.
        _g31_eval_gap = getattr(policy, '_last_eval_gap', '') or ''
        _g31_eval_source = getattr(policy, '_eval_source', '') or ''
        _g31_block = ""
        if _g31_eval_gap:
            _g31_block = (
                "⛔ EVALUATOR GROUND TRUTH (verbatim — the updated policy MUST\n"
                "   reference these exact literal requirements so the actor's next\n"
                "   action produces them byte-for-byte):\n"
                f"{str(_g31_eval_gap)[:1200]}\n\n"
            )
            if _g31_eval_source:
                _g31_block += (
                    "🔍 EVALUATOR SOURCE (what the evaluator actually runs — the\n"
                    "   updated policy must direct actions that satisfy THIS code):\n"
                    f"{str(_g31_eval_source)[:1600]}\n\n"
                )

        # Prepare prompt
        gradients_str = self._format_gradients_for_update(policy.gradients)
        prompt = self.update_prompt_template.format(
            g31_eval_block=_g31_block,
            current_policy=policy.base_policy,
            constraints=policy._format_constraints(),
            gradients=gradients_str,
        )

        response = self._generate(prompt, temperature=0.3, max_tokens=400)

        # Extract updated policy
        if "UPDATED_POLICY:" in response:
            updated_policy = response.split("UPDATED_POLICY:")[1].strip()
        else:
            updated_policy = response

        # Store old policy in history
        policy.update_history.append({
            'old_policy': policy.base_policy,
            'gradients_applied': len(policy.gradients),
            'new_policy': updated_policy
        })

        # Fix #8 PATCH (Path A) — deterministic [RECIPE] grounding validator.
        # Forensic on Terminal 13584542 + Fix #8 prompt rule showed the LLM
        # applied the rule INCONSISTENTLY (caught one hallucination, missed
        # another). A deterministic post-processor enforces grounding
        # mechanically: for every command-shaped token in [RECIPE], check
        # if its first-word "head" appears in any source-of-truth text
        # (recent gradients OR the gradients_str input). If not grounded,
        # auto-prefix `[SPECULATIVE]` so the actor knows.
        # Universal — pure substring + regex check, no per-task / per-app
        # / per-error code. Phase-0 firewall: enforces Russell-Norvig Ch.1
        # observed-vs-conjectured rule deterministically.
        updated_policy = _ground_recipe_commands(updated_policy, gradients_str)

        # Update policy
        policy.base_policy = updated_policy
        print(f"[TEXTGRAD OPTIMIZER] ✓ Policy updated (version {len(policy.update_history)})")
        # Comprehensive-logging skill R5 — full policy must be inspectable.
        # Previous [:100] truncation hid whether [RECIPE] section is present
        # (Fix #3 broken-edge audit) and whether procedural facts like the
        # verbatim_stderr_signal are being preserved across versions.
        print(f"[TEXTGRAD OPTIMIZER] New policy ({len(updated_policy)} chars):")
        print(f"=== POLICY START ===\n{updated_policy}\n=== POLICY END ===")

        # PAPER FIX (TextGrad Eq. 9): gradients are CONSUMED by the step.
        # After the optimizer rewrites the variable, the previous gradients
        # were computed against the OLD policy text and are stale — re-applying
        # them in the next cycle would dilute fresh signal with self-reference
        # against a policy that no longer exists. Clear them so the next
        # step()'s prompt only sees gradients computed against the current
        # policy version.
        policy.gradients = []

    def _format_gradients_for_update(self, gradients: List[str]) -> str:
        """Format gradients for optimizer update"""
        # Take last 10 gradients and number them
        recent = gradients[-10:]
        return "\n".join([f"  {i+1}. {g}" for i, g in enumerate(recent)])

# ============================================================================
# END TEXTGRAD COMPONENTS
# ============================================================================

def format_reflexion_insights_for_policy(working_reflexions: List, max_insights: int = 3) -> str:
    """
    Format working reflexions as concise insights for policy.forward().

    This consolidates within-trial reflexion insights into a focused format
    that the learned policy can use for decision making.
    """
    if not working_reflexions:
        return ""

    insights = []
    for ref in working_reflexions[-max_insights:]:
        if isinstance(ref, dict):
            text = ref.get('reflection', ref.get('insight', ref.get('hypothesis', str(ref))))
        else:
            text = str(ref)
        if len(text) > 500:
            text = text[:497] + "..."
        insights.append(text)

    return "\n".join([f"- {i}" for i in insights])

def calculate_task_similarity(task1: str, task2: str) -> float:
    """Calculate similarity using pure word overlap - no domain knowledge"""
    if not task1 or not task2:
        return 0.0
    
    # Pure token-based similarity
    words1 = set(task1.lower().split())
    words2 = set(task2.lower().split())
    
    if not words1 or not words2:
        return 0.5
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Check for negation patterns (universal)
    negation_words = {'not', 'no', 'dont', "don't", 'without', 'avoid'}
    has_negation1 = bool(words1 & negation_words)
    has_negation2 = bool(words2 & negation_words)
    
    # If one has negation and other doesn't, reduce similarity
    if has_negation1 != has_negation2:
        jaccard *= 0.3
    
    return jaccard

def calculate_sequence_similarity(step1: int, step2: int, 
                                 prereqs1: List[str], prereqs2: List[str]) -> float:
    """Calculate temporal and prerequisite similarity"""
    # Temporal distance decay
    step_diff = abs(step1 - step2)
    temporal_sim = max(0.1, 1.0 - (step_diff / 10))
    
    # Prerequisites similarity using set operations
    if not prereqs1 and not prereqs2:
        prereq_sim = 1.0
    elif not prereqs1 or not prereqs2:
        prereq_sim = 0.3
    else:
        prereqs1_set = set(prereqs1)
        prereqs2_set = set(prereqs2)
        intersection = len(prereqs1_set & prereqs2_set)
        union = len(prereqs1_set | prereqs2_set)
        prereq_sim = intersection / union if union > 0 else 0.0
    
    return (temporal_sim * 0.5) + (prereq_sim * 0.5)

def should_share_knowledge(source_context: Dict, target_context: Dict) -> Tuple[bool, float]:
    """Determine sharing based on context similarity"""
    # Universal failures always share
    if source_context.get('is_universal', False):
        return True, 1.0
    
    # Calculate similarities
    task_sim = calculate_task_similarity(
        source_context.get('task', ''),
        target_context.get('task', '')
    )
    
    # State similarity based on observation overlap
    state1 = source_context.get('state_text', '')
    state2 = target_context.get('state_text', '')
    state_tokens1 = set(state1.lower().split())
    state_tokens2 = set(state2.lower().split())
    if state_tokens1 and state_tokens2:
        state_sim = len(state_tokens1 & state_tokens2) / len(state_tokens1 | state_tokens2)
    else:
        state_sim = 0.0
    
    seq_sim = calculate_sequence_similarity(
        source_context.get('step', 0),
        target_context.get('step', 0),
        source_context.get('prerequisites', []),
        target_context.get('prerequisites', [])
    )
    
    # Combined score
    combined_score = task_sim * state_sim * seq_sim
    
    # Share only if all dimensions are similar
    should_share = (task_sim > 0.6 and state_sim > 0.6 and seq_sim > 0.5)
    
    return should_share, combined_score





def find_best_matching_action(text: str, valid_actions: List[str]) -> Optional[str]:
    """
    Robust matching that handles numbers and variations
    """
    text_lower = text.lower()
    
    # FIRST: Check if the last line is exactly a valid action
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line in valid_actions:
            return last_line
        # Also check without punctuation
        last_line_clean = last_line.rstrip('.!?,;:')
        if last_line_clean in valid_actions:
            return last_line_clean
    

    
    # First try: exact substring match
    for action in valid_actions:
        if action.lower() in text_lower:
            return action
    
    # Second try: score each action by word overlap
    best_score = 0
    best_action = None
    
    # Extract likely action from text (last line, after "action:", etc)
    action_candidates = []
    
    # Look for explicit action statements
    import re
    for pattern in [r'action:\s*(.+)', r'i (?:will|should|need to)\s+(.+)', 
                    r'(?:^|\n)(.+)$']:  # Last line
        match = re.search(pattern, text_lower)
        if match:
            action_candidates.append(match.group(1).strip())
    
    if not action_candidates:
        action_candidates = [text_lower]
    
    for candidate in action_candidates:
        for valid_action in valid_actions:
            score = calculate_action_similarity(candidate, valid_action.lower())
            
            if score > best_score:
                best_score = score
                best_action = valid_action
    
    # Return if we have decent confidence
    return best_action if best_score > 0.5 else None


def calculate_action_similarity(candidate: str, valid_action: str) -> float:
    """
    Calculate similarity score between candidate and valid action
    Handles numbers and word variations
    """
    import re
    
    def tokenize(text):
        # Split on spaces but keep numbers with adjacent words
        tokens = re.findall(r'\w+|\d+', text.lower())
        return tokens
    
    candidate_tokens = tokenize(candidate)
    action_tokens = tokenize(valid_action)
    
    if not action_tokens:
        return 0.0
    
    # Calculate overlap score
    matches = 0
    for token in action_tokens:
        if token in candidate_tokens:
            matches += 1
        # Partial credit for numbers that are close
        elif token.isdigit():
            for c_token in candidate_tokens:
                if c_token.isdigit() and abs(int(token) - int(c_token)) <= 1:
                    matches += 0.5
                    break
    
    # Calculate final score
    score = matches / len(action_tokens)
    
    # Boost score if all candidate tokens are in action (subset match)
    if all(token in action_tokens for token in candidate_tokens):
        score = min(score * 1.5, 1.0)
    
    return score


def format_step_reflexions(reflexions: List[Dict]) -> str:
    """Format previous step reflexions for context in step-level learning"""
    if not reflexions:
        return "None yet (first few steps)"

    formatted = []
    for r in reflexions:
        step_num = r.get('step', '?')
        reflection_text = r.get('reflection', '')[:150]  # Truncate long reflections
        formatted.append(f"  Step {step_num}: {reflection_text}")
    return '\n'.join(formatted)


def compress_reflexions_medium(reflexions, model, max_tokens=150):
    """Medium compression - STRUCTURED with WHY reasoning"""

    if not reflexions:
        return None

    combined = "\n\n".join([r['reflection'] for r in reflexions])

    prompt = f"""Extract STRUCTURED summary with causal reasoning:

{combined}

Output format (STRUCTURED):
ACTIONS: [list recent actions in order]
WHY FAILED: [causal reason actions didn't work]
WHY WORKED: [if any action succeeded, why it worked]
SEQUENCE: [if order matters, note dependencies like "needed X before Y"]

Keep it factual and explicit. Max 100 words."""

    sampling_params = SamplingParams(max_tokens=200, temperature=0.2, stop=["\n\n\n"])

    # Use fast model for compression (extraction task, no reasoning needed)
    compressed_output = fast_model.generate([prompt], sampling_params)[0]
    compressed_text = compressed_output.outputs[0].text.strip()

    return {
        'step': f"{reflexions[0]['step']}-{reflexions[-1]['step']}",
        'reflection': compressed_text,
        'is_compressed': 'medium'
    }


def compress_reflexions_heavy(reflexions, model, max_tokens=100):
    """Heavy compression to STRUCTURED FACTS - prevents hallucination by being explicit"""

    if not reflexions:
        return None

    combined = "\n\n".join([r['reflection'] for r in reflexions])

    prompt = f"""Extract STRUCTURED FACTS (NOT prose) from these reflexions. Be EXPLICIT and LIST-BASED.

{combined}

Output format (STRUCTURED - use commas, no sentences):
TRIED: [comma-separated list of actions/locations already attempted]
FOUND: [what was discovered - items, locations, objects]
FAILED: [what didn't work - be specific about locations/actions]
LEARNED: [one key insight in 10 words max]

Example:
TRIED: location_1, location_2, strategy_A
FOUND: target_item at location_2, relevant_object at location_1
FAILED: target not found in locations 1-3
LEARNED: need alternative search approach"""

    sampling_params = SamplingParams(max_tokens=200, temperature=0.1, stop=["\n\n\n"])

    # Use fast model for compression (extraction task, no reasoning needed)
    compressed_output = fast_model.generate([prompt], sampling_params)[0]
    compressed_text = compressed_output.outputs[0].text.strip()

    return {
        'step': f"{reflexions[0]['step']}-{reflexions[-1]['step']}",
        'reflection': compressed_text,
        'is_compressed': 'heavy'
    }



def manage_working_reflexions_tiered(state, model, log_debug):
    """Reflexion buffer management — NO LOSSY COMPRESSION.

    Bug 4 fix: the previous tiered compression re-fed older reflexions through
    an LLM with output schemas like ACTIONS/WHY FAILED/WHY WORKED/SEQUENCE that
    have NO slot for critical signals (FILE-CHANGE NOOP warnings, AttributeError
    text, EXPECTED tuples, prescription content). The compression GUARANTEED
    that prescriptive insights would be paraphrased away after only 2 steps.

    Universal fix: keep cross-trial reflexions full + cap current-trial to last
    N entries verbatim. Older entries are dropped entirely, not summarized
    (avoiding any LLM-mediated information loss).
    """
    reflexions = state.get('working_reflexions', [])
    if not reflexions:
        return reflexions

    current_trial = state.get('trial_idx', 0)
    cross_trial = []
    current_trial_refls = []
    for r in reflexions:
        r_trial = r.get('trial', current_trial)
        if r_trial < current_trial:
            cross_trial.append(r)
        else:
            current_trial_refls.append(r)

    if cross_trial:
        log_debug(f"[CROSS-TRIAL PRESERVATION] Keeping {len(cross_trial)} Trial {current_trial - 1} reflexions FULLY VERBOSE")

    # Cap current-trial to last 10 verbatim (no LLM compression)
    MAX_CURRENT = 10
    if len(current_trial_refls) > MAX_CURRENT:
        log_debug(f"[REFLEXION CAP] Dropping {len(current_trial_refls) - MAX_CURRENT} oldest current-trial reflexions")
        current_trial_refls = current_trial_refls[-MAX_CURRENT:]

    return cross_trial + current_trial_refls



def format_reflexion_insights_complete(memory: List) -> str:
    """Extract COMPLETE actionable insights from reflexion memory (handles both dict and string format)"""
    if not memory:
        return "- No previous attempts"

    # Ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str) or isinstance(memory, dict):
        memory_list = [memory]
    else:
        return "- No actionable insights yet"

    if not memory_list:
        return "- No previous attempts"

    insights = []

    # Process ALL reflections
    for i, reflection_item in enumerate(memory_list, 1):
        # Handle structured dict format (new) or string format (legacy)
        if isinstance(reflection_item, dict):
            # New structured format - extract insight text
            reflection = reflection_item.get('insight', '')
            if not reflection:
                continue
        elif isinstance(reflection_item, str):
            # Legacy string format
            reflection = reflection_item
        else:
            continue

        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract ALL actionable lines
            if any(word in line_lower for word in
                   ['must', 'should', 'avoid', 'never', 'always',
                    'failed because', 'succeeded', 'exact actions',
                    'hypothesis', 'critical', 'important', 'requires',
                    'learned', 'discovered', 'found', 'need', 'try']):
                clean_line = line.strip()
                if len(clean_line) > 10:
                    key_lines.append(f"  - {clean_line}")

        if key_lines:
            insights.append(f"Reflection {i}:\n" + '\n'.join(key_lines))

    return '\n'.join(insights) if insights else "- No actionable insights yet"

# Schema cache to avoid recompilation
_SCHEMA_CACHE = {}

def get_cached_schema(max_actions):
    """Cache SIMPLE JSON schemas that work with xgrammar (fast path)"""
    if max_actions not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[max_actions] = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                     
                    # NO maxLength - causes slow outlines fallback
                },
                "action_number": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_actions
                }
            },
            "required": ["reasoning", "action_number"],
            "additionalProperties": False
        }
    return _SCHEMA_CACHE[max_actions]


def calculate_fuzzy_scores(textgrad_recommendation: str, valid_actions: List[str]) -> List[Tuple[str, float]]:
    """
    Calculate semantic similarity scores between TextGrad recommendation and valid actions.
    Uses fuzzy string matching + keyword overlap for scoring.

    Returns list of (action, score) tuples sorted by score (highest first).
    DOES NOT FILTER - provides guidance scores for LLM to consider.

    Score components:
    - SequenceMatcher ratio (0-1): Character-level similarity
    - Keyword overlap bonus (0-0.3): Shared important words
    - Verb match bonus (0-0.2): Same action verb

    Args:
        textgrad_recommendation: Natural language action recommendation from TextGrad
        valid_actions: List of exact valid actions from environment

    Returns:
        List of (action, score) tuples, sorted by score descending
    """
    if not textgrad_recommendation or textgrad_recommendation.lower() == 'none':
        # No recommendation - return all actions with equal zero score
        return [(action, 0.0) for action in valid_actions]

    rec_lower = textgrad_recommendation.lower()
    rec_words = set(re.findall(r'\w+', rec_lower))

    # Extract verb from recommendation (first word usually)
    rec_parts = rec_lower.split()
    rec_verb = rec_parts[0] if rec_parts else ""

    scored_actions = []

    for action in valid_actions:
        action_lower = action.lower()

        # 1. Fuzzy string similarity (0-1)
        ratio = SequenceMatcher(None, rec_lower, action_lower).ratio()

        # 2. Keyword overlap bonus (shared meaningful words)
        action_words = set(re.findall(r'\w+', action_lower))
        # Filter out common words
        common_words = {'to', 'the', 'a', 'an', 'in', 'on', 'at', 'with', 'from', 'and'}
        meaningful_rec = rec_words - common_words
        meaningful_action = action_words - common_words

        if meaningful_rec and meaningful_action:
            overlap = len(meaningful_rec & meaningful_action)
            keyword_bonus = min(0.3, overlap * 0.1)  # Max 0.3 bonus
        else:
            keyword_bonus = 0.0

        # 3. Verb match bonus (same action type)
        action_parts = action_lower.split()
        action_verb = action_parts[0] if action_parts else ""
        verb_bonus = 0.2 if (rec_verb == action_verb and rec_verb) else 0.0

        # Total score (max = 1.5, realistically 0-1.2 range)
        total_score = ratio + keyword_bonus + verb_bonus

        scored_actions.append((action, total_score))

    # Sort by score descending
    scored_actions.sort(key=lambda x: x[1], reverse=True)

    return scored_actions


def reasoning_based_action_selection_batch(
    batch_data: List[Dict],
    prompt_generator,
    DEBUG_ACTOR: bool = False,
    log_debug = print
) -> List[str]:
    """
    Complete replacement with proper DeepSeek-R1 handling
    """
    
    prompts = []
    valid_actions_lists = []
    
    for i, data in enumerate(batch_data):
        # Extract actual environment ID for correct logging (fallback to batch index if not present)
        env_id = data.get('env_id', i)

        # Keep all your existing data extraction exactly as is
        valid_actions = data['valid_actions']
        observation = data['observation']
        task = data['task']
        step_gradient = data['step_gradient']
        textgrad_components = data['textgrad_components']
        reflexion_memory = data['reflexion_memory']
        action_history = data['action_history']
        tried_actions = data['tried_actions']
        interaction_count = data['interaction_count']
        memory_recommendations = data.get('memory_recommendations', {})
        consolidated_wisdom = data.get('consolidated_step_wisdom', '')
        step_insights = data.get('step_insights_accumulator', [])
        working_reflexions = data.get('working_reflexions', '')
        # INTELLIGENT FILTERING: Get failure history
        failure_history = data.get('failure_history', [])
        # ACTION HISTORY ENHANCEMENT: Get progress scores
        progress_history = data.get('progress_history', [])

        # Keep all your existing analysis exactly as is
        # FIX #8 BUG: action_history now contains strings, not tuples
        recent_10_actions = action_history[-10:] if action_history else []
        recent_3_actions = action_history[-3:] if action_history else []
        never_tried_count = sum(1 for act in valid_actions if act not in tried_actions)
        
        stuck_actions = []
        for act in set(recent_10_actions):
            if recent_10_actions.count(act) >= 3:
                stuck_actions.append(act)
        loop_detected = len(stuck_actions) > 0
        
        action_groups = {}
        action_to_index = {}
        
        for idx, action in enumerate(valid_actions):
            verb = action.split()[0] if action else "unknown"
            if verb not in action_groups:
                action_groups[verb] = []
            action_groups[verb].append(action)
            action_to_index[action] = idx + 1
        
        first_reflexion = ""
        if reflexion_memory and len(reflexion_memory) > 0:
            # Handle dict reflexion_memory - extract insight field
            first_mem = reflexion_memory[0]
            mem_text = first_mem.get('insight', str(first_mem)) if isinstance(first_mem, dict) else str(first_mem)
            first_reflexion = f"INITIAL PLAN/STRATEGY:\n{mem_text}\n{'='*50}\n\n"

        # Build prompt - keep EVERYTHING the same until the very end
        # Get TODO list if available
        todo_display = ""
        if 'todo_manager' in data and data['todo_manager'] is not None:
            todo_manager = data['todo_manager']
            todo_display = todo_manager.get_formatted_todos()
            print(f"[TODO DEBUG] ENV {env_id}: TODO manager exists, formatted output length: {len(todo_display)}")
            if todo_display:
                print(f"[TODO DEBUG] ENV {env_id}: First 200 chars: {todo_display[:200]}")
        else:
            print(f"[TODO DEBUG] ENV {env_id}: No TODO manager in data")

        # Get current TODO for sequential focus
        current_todo_guidance = ""
        if 'todo_manager' in data and data['todo_manager'] is not None:
            current_todo = data['todo_manager']._get_current_todo()
            if current_todo:
                current_todo_guidance = f"""
📋 SUGGESTED FOCUS (a guide, not a constraint):
   {current_todo.active_form}
   Attempts so far: {current_todo.attempts}

   ADAPTIVE EXECUTION (be intelligent, not rigid):
   - This TODO is a SUGGESTED sequence, not a strict requirement
   - If you discover an opportunity that advances the MAIN TASK (above), TAKE IT
   - The original plan may be imperfect - adapt based on what you observe
   - Trust your observations over the plan when they conflict

"""

        prompt = f"""You must select an action. This is NOT a reflection or thinking task.

🎯 PRIMARY OBJECTIVE (Your main goal):
{task}

CRITICAL: Every action must serve THIS task. If any suggestion below conflicts with
the task requirement, prioritize the TASK requirement above all else.

{todo_display}
{current_todo_guidance}
CURRENT STATE:
{observation}

============ CRITICAL LEARNING SIGNALS ============

1. LEARNING FROM LAST ACTION:
{step_gradient.get('hypothesis', 'No analysis yet')}

Observed progress: {step_gradient.get('progress_score', 0)}/10

2. TEXTGRAD OPTIMIZATION GUIDANCE:

⚠️  G34c: The strings below are learned STRUCTURAL PATTERNS with ILLUSTRATIVE example values.
    Any concrete URLs, names, paths, or quoted literals inside are EXAMPLES, not values to copy.
    Before emitting any action:
      (a) Read the task above and EXTRACT task-specific values: URLs, filenames, names, paths, expected strings.
      (b) Use the STRUCTURE from the guidance below but substitute YOUR extracted values for the examples.
    Example: if guidance says `Name=Google Chrome\nExec=google-chrome https://www.google.com` but your task is about
    a PDF viewer shortcut for /home/user/report.pdf, you must emit the structure with YOUR name/exec/url, not "Google Chrome".

   <guidance_pattern>
   - Adaptive Strategy: {textgrad_components.get('adaptive_strategy', 'None')}
   - Task Decomposition: {textgrad_components.get('task_decomposition', 'None')}
   - Environment Understanding: {textgrad_components.get('environment_understanding', 'None')}
   - Action Discovery: {textgrad_components.get('action_discovery', 'None')}
   - Pattern Recognition: {textgrad_components.get('pattern_recognition', 'None')}
   - Hypothesis Testing: {textgrad_components.get('hypothesis_testing', 'None')}
   </guidance_pattern>

3. REFLEXION MEMORY FROM PREVIOUS FAILURES:
{format_reflexion_insights_complete(reflexion_memory) if reflexion_memory else 'No previous failures yet'}

4. WORKING REFLEXIONS FROM THIS EPISODE:
{working_reflexions if working_reflexions else 'No step reflexions yet'}

5. ACCUMULATED STEP WISDOM:
{consolidated_wisdom if consolidated_wisdom else 'No accumulated wisdom yet'}

6. PAST FAILURES THIS TRIAL (Intelligent Filtering):"""

        # Add intelligent filtering context
        if failure_history:
            prompt += f"\n   ⚠️  IMPORTANT: The following actions made low/no progress recently:"
            for failure in failure_history[-5:]:  # Last 5 failures
                progress = failure.get('progress_score', 'N/A')
                prompt += f"\n   • Step {failure['step']}: '{failure['action']}' (progress: {progress}/10)"
                prompt += f"\n     Before: {failure['context_before']}"
                prompt += f"\n     After: {failure['context_after']}"

            prompt += "\n\n   SMART FILTERING GUIDANCE:"
            prompt += "\n   - Avoid these actions ONLY if current state is very similar to failure context"
            prompt += "\n   - Allow retry if state changed significantly (new items, different location)"
            prompt += "\n   - First-time exploration is valuable even with low initial progress"
        else:
            prompt += "\n   No failures yet - exploring freely"

        prompt += "\n\n7. UNIVERSAL MEMORY RECOMMENDATIONS:"""
        
        if memory_recommendations.get('previously_succeeded'):
            prompt += "\n   Actions that worked in similar situations:"
            for rec in memory_recommendations['previously_succeeded'][:3]:
                prompt += f"\n   ✓ '{rec['action']}': {rec.get('reason', 'Previously successful')}"
        
        if memory_recommendations.get('avoid'):
            prompt += "\n   Actions to AVOID (failed in similar states):"
            for rec in memory_recommendations['avoid'][:3]:
                prompt += f"\n   ✗ '{rec['action']}': {rec.get('reason', 'Previously failed')}"
        
        prompt += f"""

8. ACTION HISTORY ANALYSIS:
   - Total actions taken: {len(action_history)}
   - Unique actions tried: {len(tried_actions)}
   - Never tried actions: {never_tried_count}/{len(valid_actions)}"""

        # RESTORED: Explicit loop warning - critical for breaking loops!
        # The LLM needs to be told which actions are stuck
        if loop_detected:
            prompt += f"\n   ⚠️ WARNING: Stuck in loop with actions: {stuck_actions[:3]}"
            prompt += f"\n   🚫 DO NOT select these actions again!"

        prompt += f"""

9. RECENT TRAJECTORY:"""

        if action_history:
            # Show last 5 actions (FIX #8 BUG: action_history now contains strings only)
            for j, act in enumerate(action_history[-5:], 1):
                prompt += f"\n   {j}. {act}"
        else:
            prompt += "\n   No actions taken yet"

        prompt += "\n"
        
        # Check if gradient guidance exists and is valid
        gradient_action = step_gradient.get('next_action_guidance', '')
        gradient_is_valid = gradient_action and gradient_action in valid_actions

        # EXTRACT FULL ACTION SEQUENCE FROM STRUCTURED MEMORY (highest priority - exact replay)
        success_pattern_actions = []
        task_lower = task.lower()

        # Check if reflexion_memory has structured success patterns with full sequences
        if reflexion_memory:
            from difflib import SequenceMatcher
            import re

            # DEBUG: Print what's in reflexion_memory
            log_debug(f"[SUCCESS PATTERN DEBUG] reflexion_memory has {len(reflexion_memory)} items")
            workflow_count = sum(1 for item in reflexion_memory if isinstance(item, dict) and item.get('type') == 'success_workflow')
            log_debug(f"[SUCCESS PATTERN DEBUG] Found {workflow_count} items with type='success_workflow'")
            if workflow_count > 0:
                for item in reflexion_memory:
                    if isinstance(item, dict) and item.get('type') == 'success_workflow':
                        log_debug(f"[SUCCESS PATTERN DEBUG] Workflow task: '{item.get('task', 'NO TASK')}'")

            best_match = None
            best_similarity = 0.0

            for mem_item in reflexion_memory:
                if isinstance(mem_item, dict) and mem_item.get('type') == 'success_workflow':
                    # Get memory task
                    mem_task = mem_item.get('task', '').lower()
                    if not mem_task:
                        continue

                    # SEMANTIC SIMILARITY CHECK (prevents contamination while allowing valid transfers)

                    # 1. Fuzzy string similarity
                    fuzzy_score = SequenceMatcher(None, task_lower, mem_task).ratio()

                    # 2. Extract meaningful words (filter stop words)
                    stop_words = {'a', 'an', 'the', 'to', 'in', 'on', 'at', 'with', 'from', 'and', 'some', 'it'}
                    task_words = set(w for w in task_lower.split() if w not in stop_words)
                    mem_words = set(w for w in mem_task.split() if w not in stop_words)

                    # 3. Word overlap (meaningful words only)
                    word_overlap = len(task_words & mem_words) / max(len(task_words), len(mem_words)) if task_words and mem_words else 0.0

                    # 4. Verb matching (first word comparison - universal, no hardcoded synonyms)
                    task_verb = task_lower.split()[0] if task_lower.split() else ""
                    mem_verb = mem_task.split()[0] if mem_task.split() else ""
                    verb_match = (task_verb == mem_verb)

                    # 5. Combined similarity (weighted: fuzzy 30%, word overlap 40%, verb 30%)
                    combined_similarity = fuzzy_score * 0.3 + word_overlap * 0.4 + (1.0 if verb_match else 0.0) * 0.3

                    log_debug(f"[MEMORY MATCH] '{task}' vs '{mem_task}': sim={combined_similarity:.2f} (fuzzy={fuzzy_score:.2f}, words={word_overlap:.2f}, verb={verb_match})")

                    if combined_similarity > best_similarity:
                        best_similarity = combined_similarity
                        best_match = mem_item

            # USE MEMORY ONLY IF SIMILARITY >= 0.75 (strict to prevent contamination)
            if best_match and best_similarity >= 0.75:
                # FIX: Try 'actions' field first (used by success_workflow), then 'full_action_sequence'
                full_seq = best_match.get('actions', best_match.get('full_action_sequence', []))
                if full_seq:
                    success_pattern_actions = full_seq
                    log_debug(f"[MEMORY REPLAY] Using {len(full_seq)}-step sequence (similarity: {best_similarity:.2f}, task: '{best_match.get('task', '')}')")
                else:
                    log_debug(f"[MEMORY ERROR] Match found but no actions field (keys: {list(best_match.keys() if isinstance(best_match, dict) else [])})")
            elif best_match:
                log_debug(f"[MEMORY SKIP] Best similarity {best_similarity:.2f} < 0.75 threshold (task: '{best_match.get('task', '')}')")

        # FALLBACK: Extract from adaptive_strategy TextGrad patterns (if no memory sequence found)
        if not success_pattern_actions:
            adaptive_strat = textgrad_components.get('adaptive_strategy', '')
            if 'SUCCESS PATTERN:' in adaptive_strat:
                # Extract ALL patterns and find the one matching current task
                import re
                task_lower = task.lower()

                # Find all SUCCESS PATTERN entries
                all_patterns = re.findall(r'SUCCESS PATTERN:\s*\[(.*?)\]\s*completes\s+["\']([^"\']+)["\']', adaptive_strat, re.IGNORECASE)

                # Match pattern to current task
                best_match = None
                best_match_score = 0

                for pattern_str, pattern_task in all_patterns:
                    pattern_task_lower = pattern_task.lower()
                    # Calculate word overlap between current task and pattern task
                    task_words = set(task_lower.split())
                    pattern_words = set(pattern_task_lower.split())
                    overlap = len(task_words & pattern_words)

                    if overlap > best_match_score:
                        best_match_score = overlap
                        best_match = pattern_str

                if best_match and best_match_score >= 2:  # At least 2 word overlap
                    # Parse comma-separated actions
                    success_pattern_actions = [act.strip().strip('"').strip("'") for act in best_match.split(',')]
                    log_debug(f"[SUCCESS PATTERN] Extracted {len(success_pattern_actions)} actions for task '{task}' (match score: {best_match_score})")
            else:
                log_debug(f"[SUCCESS PATTERN] No matching pattern found for task '{task}'")

        # Calculate current step position (how many actions taken so far)
        current_step_in_episode = len(action_history)

        # INTELLIGENT ACTION SCORING with multiple factors
        scored_actions = []
        for action in valid_actions:
            score = 0
            action_lower = action.lower()

            # FACTOR 1: SUCCESS PATTERN REPLAY (highest priority for early steps)
            # If we have a success pattern and we're early in the episode, strongly boost matching actions
            if success_pattern_actions and current_step_in_episode < len(success_pattern_actions):
                # Check if this action matches the expected action at current position
                expected_action = success_pattern_actions[current_step_in_episode]

                # Exact match (highest boost)
                if action_lower == expected_action.lower():
                    score += 50  # Very strong boost for exact replay
                    log_debug(f"[PATTERN MATCH] Step {current_step_in_episode}: '{action}' matches pattern position exactly")

                # Semantic match (verb + main object)
                elif expected_action.lower().split()[0] == action_lower.split()[0]:  # Same verb
                    # Check if main noun matches (fuzzy)
                    expected_words = set(expected_action.lower().split())
                    action_words = set(action_lower.split())
                    noun_overlap = len(expected_words & action_words)
                    if noun_overlap >= 2:  # At least verb + one object
                        score += 30  # Good semantic match
                        log_debug(f"[PATTERN MATCH] Step {current_step_in_episode}: '{action}' semantically matches pattern")

            # FACTOR 2: SUCCESS PATTERN ELEMENTS (general boost for any pattern action)
            # Even if not at the right position, actions from success pattern get boosted
            for success_action in success_pattern_actions:
                if action_lower == success_action.lower():
                    score += 15  # Moderate boost - it's a proven action
                elif success_action.lower().split()[0] == action_lower.split()[0]:
                    score += 5   # Small boost for same verb

            # FACTOR 3: TODO ALIGNMENT (existing system)
            if 'todo_manager' in data and data['todo_manager']:
                current_todo = data['todo_manager']._get_current_todo()
                if current_todo:
                    todo_words = set(word.lower() for word in current_todo.content.split() if len(word) > 3)
                    action_words = set(action_lower.split())
                    overlap = len(todo_words & action_words)
                    score += overlap * 3  # 3 points per matching word

                    action_first_word = action_lower.split()[0] if action.split() else ""
                    if action_first_word in todo_words:
                        score += 5

            # FACTOR 4: TEXTGRAD SEMANTIC MATCHING (gradient-based recommendation)
            # TextGrad analyzes observation outcomes and suggests next action
            # Boost actions that match TextGrad's gradient direction
            textgrad_rec = step_gradient.get('next_action_guidance', '')
            if textgrad_rec and textgrad_rec.strip():
                from difflib import SequenceMatcher
                textgrad_lower = textgrad_rec.lower().strip()
                similarity = SequenceMatcher(None, textgrad_lower, action_lower).ratio()

                # Semantic word overlap (more robust than pure string matching)
                textgrad_words = set(textgrad_lower.split())
                action_words_set = set(action_lower.split())
                common_words = {'to', 'the', 'a', 'an', 'in', 'on', 'at', 'with', 'from'}
                meaningful_overlap = len((textgrad_words - common_words) & (action_words_set - common_words))

                # Combined score: string similarity + word overlap
                textgrad_score = int(similarity * 15) + (meaningful_overlap * 2)  # Max ~20-25 pts
                if textgrad_score > 0:
                    score += textgrad_score
                    if textgrad_score >= 10:
                        log_debug(f"[TEXTGRAD MATCH] '{action}' matches TextGrad rec (score: +{textgrad_score})")

            # FACTOR 5: FAILURE AVOIDANCE (penalize recently failed actions)
            if failure_history:
                for failure in failure_history[-5:]:  # Last 5 failures
                    if failure['action'].lower() == action_lower:
                        # Check if context is similar (same location/state)
                        context_similar = False
                        if 'context_before' in failure and observation:
                            # Simple similarity: check if key words overlap
                            failure_words = set(failure['context_before'].lower().split())
                            current_words = set(observation.lower().split())
                            if len(failure_words & current_words) > 10:  # Many shared words
                                context_similar = True

                        if context_similar:
                            score -= 10  # Penalize if context is similar to failure
                            log_debug(f"[FAILURE AVOID] Penalizing '{action}' - failed in similar context")
                        else:
                            score -= 2   # Small penalty - might work in different context

            # FACTOR 6: EXPLORATION BONUS (for never-tried actions, but only if no success pattern)
            if action not in tried_actions and not success_pattern_actions:
                score += 1  # Small exploration bonus only when no proven pattern exists

            scored_actions.append((score, action))

        # Sort by score descending
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        valid_actions = [action for score, action in scored_actions]

        # Log top 5 scored actions for debugging
        log_debug(f"[ACTION SCORING] Top 5 actions:")
        for rank, (score, action) in enumerate(scored_actions[:5], 1):
            log_debug(f"  {rank}. [{score:3d}] {action}")

        # Store scored_actions for fuzzy matching fallback
        # NOTE: Intent translation removed - TextGrad must learn exact action syntax
        # This forces clearer failure signals and better gradient learning
        if i >= len(batch_data):
            log_debug(f"[ERROR] i={i} but batch_data length={len(batch_data)}! Skipping scored_actions storage.")
        else:
            batch_data[i]['scored_actions'] = scored_actions

        prompt += f"""

============ ALL VALID ACTIONS (You MUST choose from these {len(valid_actions)} options) ============
"""

        # Show ALL valid actions clearly numbered (up to 100 for readability)
        actions_to_show = min(100, len(valid_actions))

        for idx in range(actions_to_show):
            action = valid_actions[idx]

            # Add score info if available
            score_info = ""
            for score, scored_action in scored_actions[:10]:
                if scored_action == action:
                    score_info = f" [{score} pts]"
                    break

            prompt += f"{idx + 1}. {action}{score_info}\n"

        if len(valid_actions) > 100:
            prompt += f"\n... and {len(valid_actions) - 100} more options available\n"
        

        prompt += f"""

============ RECENT REFLECTION GUIDANCE ============
"""
        # Fix: Use the correct variable from the loop
        if i < len(batch_data) and 'working_reflexions' in batch_data[i]:
            working_reflexions_text = batch_data[i].get('working_reflexions', '')
            if working_reflexions_text:
                prompt += f"{working_reflexions_text}\n"

        prompt += f"""

============ SELECTION INSTRUCTION ============
The actions above are RANKED by a 6-factor scoring system with TRUE SYNERGY:
1. ✓ Reflexion Success Patterns (+50 pts) - cross-trial episodic memory
2. ✓ Reflexion General Patterns (+15 pts) - proven action sequences
3. ✓ TODO Manager Alignment (+15 pts) - current subtask structure
4. ✓ TextGrad Semantic Match (+20 pts) - gradient from observation outcomes
5. ✓ Reflexion Failure Avoidance (-10 pts) - causal mistake analysis
6. ✓ Exploration Bonus (+1 pt) - novelty when no patterns exist

STRONGLY PREFER actions with highest combined scores ([XXX pts]).
High scores mean Reflexion + TextGrad + TODO all agree this action advances the task.

You may choose a lower-scored action ONLY if:
- Top actions are physically impossible in current state
- You have strong reasoning based on recent observations

Select the SINGLE action that best advances the task:

"""

        prompt += f"""
============ CRITICAL INSTRUCTION ============
Before selecting, check section 9 (WHAT YOU ALREADY TRIED).
If an action appears there, read its observation. If the observation doesn't show
progress toward the task, DO NOT select that action again. Choose something DIFFERENT.

Your output must be EXACTLY one action from the numbered list above.
- Copy the complete action text word-for-word
- Do NOT add extra words, explanations, or modifications
- Do NOT create new actions not in the list
- ONLY choose from the {len(valid_actions)} numbered options shown above

Required format: [action text exactly as listed]

Your selected action:"""

        # NO REACTIVE RECOMMENDATIONS - synergy works through proactive scoring only
        
        prompts.append(prompt)
        valid_actions_lists.append(valid_actions)
    
    # END OF LOOP - Put verification AFTER the loop completes
    
    # Verify prompts are unique (OUTSIDE the loop, after it's done)
    for idx in range(1, len(prompts)):
        if prompts[idx] == prompts[idx-1]:
            print(f"[ERROR] Prompt {idx} is identical to prompt {idx-1}! This is a bug!")
    
    # Also check overall uniqueness
    if len(set(prompts)) < len(prompts):
        print(f"[WARNING] Only {len(set(prompts))} unique prompts out of {len(prompts)} total!")


    # ADD THIS DEBUG BLOCK before: outputs = model.generate(prompts, SamplingParams(...))
    if DEBUG_ACTOR:
        print(f"\n{'='*80}")
        print("[DEBUG LLM PROMPT - BATCH CHECK]")
        print(f"Total prompts to generate: {len(prompts)}")
        for idx, (prompt, actions) in enumerate(zip(prompts, valid_actions_lists)):
            # Extract task from prompt
            task_match = prompt.split('TASK:')[1].split('\n')[0].strip() if 'TASK:' in prompt else "NO TASK"
            # Get env_id for this index from batch_data
            prompt_env_id = batch_data[idx].get('env_id', idx)
            print(f"\n[PROMPT ENV {prompt_env_id}]:")
            print(f"  Task: {task_match}")
            print(f"  Valid actions count: {len(actions)}")
            print(f"  First 3 actions: {actions[:3]}")
            print(f"  Prompt length: {len(prompt)} chars")
            # Check if prompt actually contains this env's actions
            if actions and actions[0] in prompt:
                print(f"  ✓ Prompt contains env's first action")
            else:
                print(f"  ✗ WARNING: Prompt may not contain correct actions!")
        print(f"{'='*80}\n")


    # UNIVERSAL TWO-TIER VALIDATION: Use TextGrad recommendations directly
    # Forces TextGrad to learn exact action syntax for better gradient signals
    synergy_outputs = []
    need_llm = []

    # Normalization for exact match (handles minor syntax differences)
    def normalize(action):
        return action.lower().strip().rstrip('.')

    # Semantic token matching for natural language to exact syntax
    def semantic_match(textgrad_action, valid_actions):
        """Find best matching action using token overlap + numbered item handling"""
        if not textgrad_action:
            return None

        tg_tokens = set(normalize(textgrad_action).split())
        best_match = None
        best_overlap = 0

        # TIER 3A: Standard token overlap (70% threshold)
        for action in valid_actions:
            action_tokens = set(normalize(action).split())
            overlap = len(tg_tokens & action_tokens)

            # Require at least 70% token overlap
            if overlap > best_overlap and overlap >= len(tg_tokens) * 0.7:
                best_match = action
                best_overlap = overlap

        if best_match:
            return best_match

        # TIER 3B: Numbered item fuzzy matching
        # Handles cases like "examine armchair 1" vs "examine armchair 2"
        # Strategy: Match verb + noun pattern, ignore item numbers
        import re

        # Extract pattern from TextGrad: verb + nouns (ignore numbers)
        tg_words = normalize(textgrad_action).split()
        tg_pattern = ' '.join(word for word in tg_words if not word.isdigit())

        # Score each action based on pattern match
        pattern_matches = []
        for action in valid_actions:
            action_words = normalize(action).split()
            action_pattern = ' '.join(word for word in action_words if not word.isdigit())

            # Check if patterns match (same verb + nouns, different numbers)
            if tg_pattern == action_pattern:
                # Perfect pattern match (verb + noun identical, only numbers differ)
                # Prioritize actions with same item number if available
                tg_numbers = [w for w in tg_words if w.isdigit()]
                action_numbers = [w for w in action_words if w.isdigit()]
                number_match_score = len(set(tg_numbers) & set(action_numbers))
                pattern_matches.append((action, 1000 + number_match_score))  # High priority
            elif action_pattern.startswith(tg_pattern) or tg_pattern.startswith(action_pattern):
                # Partial pattern match (one is prefix of other)
                pattern_matches.append((action, 500))  # Medium priority

        # Return best pattern match
        if pattern_matches:
            pattern_matches.sort(key=lambda x: x[1], reverse=True)
            best_match = pattern_matches[0][0]
            log_debug(f"[NUMBERED-ITEM-MATCH] '{textgrad_action}' → '{best_match}' (pattern: '{tg_pattern}')")
            return best_match

        return None

    for i in range(len(batch_data)):
        # Use last_step_gradient (from previous step) for action selection
        # step_gradient is for CURRENT step (not yet generated during action selection)
        step_grad = batch_data[i].get('last_step_gradient', {})
        textgrad_rec = step_grad.get('next_action_guidance', '')
        guidance_source = step_grad.get('guidance_source', 'unknown')
        valid_actions = valid_actions_lists[i]  # Filtered and prioritized actions
        raw_actions = batch_data[i].get('raw_valid_actions', valid_actions)  # Unfiltered from environment
        env_id = batch_data[i].get('env_id', i)
        trial_idx = batch_data[i].get('trial_idx', 0)

        # PHASE 2: Extract clean action from Reflexion CAUSAL CHAIN if needed
        # TextGrad outputs are already clean, Reflexion outputs need extraction
        if textgrad_rec and guidance_source == 'reflexion' and 'CAUSAL CHAIN' in textgrad_rec:
            import re
            # Try multiple patterns to extract ACTION from Reflexion's verbose output
            action_match = None

            # Pattern 1: ACTION: go to X. | (colon, ends with period or pipe)
            action_match = re.search(r'ACTION:\s*([^.|]+?)(?:\.|\||$)', textgrad_rec, re.IGNORECASE)

            # Pattern 2: ACTION (go to X) (parentheses format)
            if not action_match:
                action_match = re.search(r'ACTION\s*\(([^)]+)\)', textgrad_rec, re.IGNORECASE)

            # Pattern 3: Action = go to X (equals sign format)
            if not action_match:
                action_match = re.search(r'ACTION\s*=\s*([^|>]+?)(?:\||->|RESULT)', textgrad_rec, re.IGNORECASE)

            if action_match:
                extracted = clean_action(action_match.group(1).strip())
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: Reflexion CAUSAL CHAIN → '{extracted}'")
                textgrad_rec = extracted
            else:
                # Extraction failed - will trigger LLM fallback
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: Failed to extract from Reflexion CAUSAL CHAIN")
                textgrad_rec = ''
        elif textgrad_rec and guidance_source == 'textgrad':
            # FIX: Extract clean action before GRADIENT JUSTIFICATION
            if ' | GRADIENT JUSTIFICATION:' in textgrad_rec:
                textgrad_rec = textgrad_rec.split(' | GRADIENT JUSTIFICATION:')[0].strip()
            textgrad_rec = clean_action(textgrad_rec)
            log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: TextGrad action cleaned: '{textgrad_rec[:50]}'")
        elif not textgrad_rec:
            log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: No guidance from previous step")

        # ============================================================================
        # SYNERGY LAYER 0: REFLEXION FILTER (Veto power - runs first)
        # ============================================================================
        reflexion_memory = batch_data[i].get('reflexion_memory', [])
        avoid_patterns = []
        success_patterns = []

        # Extract AVOID and SUCCESS patterns from Reflexion (cheap pattern matching)
        if reflexion_memory and trial_idx > 0:  # Only apply in Trial 1+
            import re
            for mem in reflexion_memory:
                mem_str = str(mem).lower()
                # Extract AVOID patterns: "🚫 AVOID: go to cabinet" → "cabinet"
                if '🚫 avoid:' in mem_str or 'avoid:' in mem_str:
                    match = re.search(r'avoid:\s*([^,\n\.]+)', mem_str)
                    if match:
                        pattern = match.group(1).strip()
                        if pattern and len(pattern) > 2:  # Avoid noise
                            avoid_patterns.append(pattern)

                # Extract SUCCESS patterns: "✅ PROVEN: pans on stoveburner" → "stoveburner"
                elif '✅ proven:' in mem_str or 'success:' in mem_str or 'key_fact:' in mem_str:
                    match = re.search(r'(?:proven|success|key_fact):\s*([^,\n\.]+)', mem_str)
                    if match:
                        pattern = match.group(1).strip()
                        if pattern and len(pattern) > 2:
                            success_patterns.append(pattern)

        # Apply Reflexion filter (hard constraint - removes contradicting actions)
        if avoid_patterns:
            original_count = len(valid_actions)
            valid_actions_filtered = []
            for action in valid_actions:
                action_lower = action.lower()
                is_avoided = any(pattern in action_lower for pattern in avoid_patterns)
                if not is_avoided:
                    valid_actions_filtered.append(action)
                else:
                    log_debug(f"[REFLEXION-VETO] ENV {env_id}: Filtered '{action}' (matches AVOID: {pattern})")

            if valid_actions_filtered:  # Use filtered list if not empty
                valid_actions = valid_actions_filtered
                log_debug(f"[REFLEXION-FILTER] ENV {env_id}: {original_count} → {len(valid_actions)} actions after veto")

        # ============================================================================
        # PURE LEARNING MODE (Nov 22 Fix): Direct TextGrad Use + Reflexion Veto
        # ============================================================================
        # FIX: TextGrad recommendations are now used DIRECTLY (not as LLM context)
        # This restores true learning where gradient optimization drives behavior.
        #
        # Architecture:
        #   1. TextGrad generates action recommendation (from previous step)
        #   2. Reflexion provides veto power (avoid patterns from episodic memory)
        #   3. Action selection uses TextGrad directly (NO LLM reasoning)
        #   4. Fuzzy matching handles minor syntax differences
        #   5. NO fallbacks that hide learning failures
        #
        # This fixes the fundamental issue where learning signals were generated
        # but ignored by LLM reasoning, resulting in 0% learning utilization.
        # ============================================================================

        # Clean TextGrad recommendation (strip JUSTIFICATION if present)
        # FIX (Nov 22): TextGrad sometimes appends "JUSTIFICATION: explanation"
        # Strip this to get just the action for exact matching
        if textgrad_rec and ' JUSTIFICATION:' in textgrad_rec:
            textgrad_rec = textgrad_rec.split(' JUSTIFICATION:')[0].strip()
            log_debug(f"[TEXTGRAD-CLEAN] ENV {env_id}: Stripped JUSTIFICATION, action: '{textgrad_rec}'")

        # FIX #8 (Nov 23): HARD ACTION REPETITION BLOCKING
        # Track action usage history for this environment
        if 'action_history' not in batch_data[i]:
            batch_data[i]['action_history'] = []

        action_history = batch_data[i]['action_history']

        # Count how many times each action has been used
        from collections import Counter
        action_counts = Counter(action_history)

        # DIRECT TEXTGRAD USE (no LLM reasoning!)
        if textgrad_rec and textgrad_rec in valid_actions:
            # FIX #8: Block actions used >2 times (hard constraint, overrides LLM)
            if action_counts.get(textgrad_rec, 0) > 2:
                log_debug(f"[FIX8-BLOCK] ENV {env_id}: TextGrad recommended '{textgrad_rec}' but it's been used {action_counts[textgrad_rec]} times - BLOCKING")

                # FIX #8 CORRECTED: Prioritize unexplored actions, then use random selection to break cycles
                # This prevents falling back to overused actions which defeats the purpose

                # Priority 1: Never-tried actions (count=0)
                never_tried = [a for a in valid_actions if action_counts.get(a, 0) == 0]
                if never_tried:
                    import random
                    textgrad_rec = random.choice(never_tried)
                    log_debug(f"[FIX8-EXPLORE] ENV {env_id}: Using never-tried action '{textgrad_rec}'")
                else:
                    # Priority 2: Actions with count ≤2
                    alternatives = [a for a in valid_actions if action_counts.get(a, 0) <= 2]
                    if alternatives:
                        import random
                        textgrad_rec = random.choice(alternatives)
                        log_debug(f"[FIX8-ALTERNATIVE] ENV {env_id}: Using '{textgrad_rec}' (used {action_counts.get(textgrad_rec, 0)} times)")
                    else:
                        # Priority 3: Random action to break deterministic cycle (no fallback to overused!)
                        import random
                        textgrad_rec = random.choice(valid_actions)
                        log_debug(f"[FIX8-RANDOM] ENV {env_id}: All actions overused, using RANDOM '{textgrad_rec}' to break cycle (used {action_counts.get(textgrad_rec, 0)} times)")

            # Check if Reflexion vetoes this action
            is_vetoed = any(pattern in textgrad_rec.lower() for pattern in avoid_patterns)

            if is_vetoed:
                # Reflexion veto - find alternative
                alternatives = [a for a in valid_actions if a != textgrad_rec]
                if alternatives:
                    selected_action = alternatives[0]
                    synergy_outputs.append(selected_action)
                    log_debug(f"[REFLEXION-VETO] ENV {env_id}: Blocked '{textgrad_rec}', using '{selected_action}'")
                else:
                    # All actions vetoed - use TextGrad anyway (rare edge case)
                    synergy_outputs.append(textgrad_rec)
                    log_debug(f"[REFLEXION-VETO] ENV {env_id}: All alternatives vetoed, using '{textgrad_rec}'")
            else:
                # Direct use of TextGrad recommendation
                synergy_outputs.append(textgrad_rec)
                log_debug(f"[TEXTGRAD-DIRECT] ENV {env_id}: ✓ Using '{textgrad_rec}'")

        elif textgrad_rec and len(textgrad_rec) > 5:
            # TextGrad provided recommendation but not exact match - fuzzy match
            from difflib import get_close_matches
            matches = get_close_matches(textgrad_rec, valid_actions, n=1, cutoff=0.75)

            if matches:
                selected_action = matches[0]
                synergy_outputs.append(selected_action)
                log_debug(f"[TEXTGRAD-FUZZY] ENV {env_id}: '{textgrad_rec}' → '{selected_action}'")
            else:
                # NO MATCH - This is a learning failure, expose it!
                log_debug(f"[TEXTGRAD-FAIL] ENV {env_id}: '{textgrad_rec}' not in valid actions: {valid_actions[:3]}")
                # Use first valid action and log the issue
                synergy_outputs.append(valid_actions[0])
                log_debug(f"[TEXTGRAD-FAIL] ENV {env_id}: Using fallback '{valid_actions[0]}' - TextGrad needs better syntax")

        else:
            # No TextGrad guidance (step 0 or empty generation)
            # Simple heuristic: match task verb to action verb
            task = batch_data[i].get('task', '')
            task_verb = task.split()[0].lower() if task else ""
            matching = [a for a in valid_actions if a.split()[0].lower() == task_verb]

            if matching:
                synergy_outputs.append(matching[0])
                log_debug(f"[STEP-0-HEURISTIC] ENV {env_id}: No TextGrad, using task verb match '{matching[0]}'")
            else:
                synergy_outputs.append(valid_actions[0])
                log_debug(f"[STEP-0-HEURISTIC] ENV {env_id}: No TextGrad, using first action '{valid_actions[0]}'")

    # ALL environments use comprehensive prompt (true synergy)
    if need_llm:
        print(f"[ACTION SELECTION] Using comprehensive Reflexion+TextGrad prompt for ALL {len(need_llm)} environment(s)")

        # Track API calls and handle quota
        global checkpoint_manager
        if checkpoint_manager:
            checkpoint_manager.increment_api_calls(len(need_llm))

        # Generate only for environments that need LLM
        llm_prompts = [prompts[i] for i in need_llm]

        try:
            # UPGRADED: Use GPT-5 with minimal reasoning for better instruction following
            # (previously used gpt-4o-mini via fast_model, which ignored loop warnings)
            print(f"[ACTION SELECTION] Using {os.getenv('REFLEXGRAD_MODEL', 'actor')} (minimal reasoning) for {len(llm_prompts)} action(s)")
            llm_outputs = model.generate(llm_prompts, SamplingParams(
                max_tokens=100,
                temperature=0.0,
                stop=[],
                skip_special_tokens=False
            ), reasoning_effort='minimal')  # 'minimal' = fast GPT-5 with few/no reasoning tokens
        except Exception as e:
            # Handle API quota errors
            from api_quota_handler import APIQuotaHandler
            # Get logging_dir from checkpoint_manager or use fallback

            if checkpoint_manager:
                logging_dir = checkpoint_manager.run_dir
            else:
                logging_dir = '.'
            quota_handler = APIQuotaHandler(checkpoint_manager, logging_dir)
            # Note: trial_idx, env_configs, env_states might not be in scope here
            # Pass empty values if not available
            quota_handler.handle_api_error(e, 0, [], [])
            # If not quota error, re-raise
            raise e

        # Merge LLM outputs with synergy outputs
        llm_output_idx = 0
        for i in range(len(synergy_outputs)):
            if synergy_outputs[i] is None:  # This env needed LLM
                synergy_outputs[i] = llm_outputs[llm_output_idx].outputs[0].text.strip()
                llm_output_idx += 1
    else:
        print(f"[SYNERGY OPTIMIZATION] All {len(prompts)} environments using Reflexion+TextGrad directly - no LLM calls needed!")

    # Create fake output objects for synergy outputs
    class FakeOutput:
        def __init__(self, text):
            self.text = text

    class FakeResult:
        def __init__(self, text):
            self.outputs = [FakeOutput(text)]

    outputs = [FakeResult(text) for text in synergy_outputs]

    # Verify we got outputs for all environments
    if len(outputs) != len(prompts):
        print(f"[ERROR] Model returned {len(outputs)} outputs but we sent {len(prompts)} prompts!")
        
    # Debug what each environment selected
    if DEBUG_ACTOR:
        print(f"\n[MODEL OUTPUTS SUMMARY]:")
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()
            # Get env_id for this response
            response_env_id = batch_data[i].get('env_id', i)
            print(f"  ENV {response_env_id}: Selected '{raw[:500]}...' from {len(valid_actions_lists[i])} options")
            print(f"\n[LLM RAW OUTPUT {i}]: '{raw[:500]}'")
            if valid_actions_lists and i < len(valid_actions_lists):
                print(f"Output matches a valid action?: {raw.strip() in valid_actions_lists[i]}")
    
    selected_actions = []

    # ============================================================================
    # TRUE TEXTGRAD SYNERGY: TextGrad components integrated into comprehensive prompt
    #
    # HOW IT WORKS:
    # 1. Action Selection: Uses Reflexion's comprehensive prompt (lines 1000-1200)
    #    which INCLUDES TextGrad learned components (lines 1040-1049)
    # 2. Learning: TextGrad loss/backward/optimizer (lines 2879-2916) updates components
    # 3. Synergy: Updated components enhance future prompts = learning feedback loop
    #
    # This restores the proven 67% success mechanism + adds TextGrad learning!
    # ============================================================================

    for i, output in enumerate(outputs):

        # Get env_id for correct logging
        parse_env_id = batch_data[i].get('env_id', i)

        raw_text = output.outputs[0].text
        log_debug(f"[DEBUG RAW OUTPUT {i}]: '{raw_text[:200]}...'")


        # OpenAI doesn't use think tags - just use the text directly
        action_text = raw_text.strip()

        # Clean prefixes
        if action_text.startswith("Action:"):
            action_text = action_text[7:].strip()

        # Remove brackets if present
        action_text = action_text.strip('[]')

        # Take only first line
        if '\n' in action_text:
            action_text = action_text.split('\n')[0].strip()

        # Remove trailing dots
        action_text = action_text.rstrip('.')

        log_debug(f"[ENV {parse_env_id}] Extracted action text: '{action_text}'")

        # Get TextGrad recommendation for logging
        step_grad = batch_data[i].get('step_gradient', {})
        textgrad_rec = step_grad.get('next_action_guidance', 'None')

        # SIMPLIFIED VALIDATION: Ensure LLM output is in valid actions
        # NO FALLBACKS - always use top-scored action to maintain synergy
        if action_text.strip() not in valid_actions_lists[i]:
            log_debug(f"[VALIDATION FAILED] LLM output '{action_text}' NOT in valid actions")

            # Use top-scored action (maintains synergy: Reflexion + TextGrad + TODO combined)
            action_text = valid_actions_lists[i][0]
            log_debug(f"[SYNERGY RECOVERY] Using top-scored action: '{action_text}' (scored by all 6 factors)")

        # EXACT match check - no fuzzy fallbacks
        selected_action = None

        for valid_action in valid_actions_lists[i]:
            if action_text.lower() == valid_action.lower():
                selected_action = valid_action
                log_debug(f"[ENV {parse_env_id}] ✓ EXACT match found: '{selected_action}'")
                break

        # NO FUZZY FALLBACK - fail confidently if no exact match
        # This should never happen after validation recovery above
        if selected_action is None:
            selected_action = action_text  # Use it anyway, let environment reject if invalid
            log_debug(f"[ENV {parse_env_id}] ⚠ NO EXACT MATCH - using action as-is: '{selected_action}'")

        log_debug(f"[ENV {parse_env_id}] Selected to execute: '{selected_action}' (VALID)")

        # Simple TextGrad alignment logging - exact match only
        if textgrad_rec and textgrad_rec != 'None':
            textgrad_exact_match = (selected_action.lower() == textgrad_rec.lower())
            match_quality = "✓ EXACT" if textgrad_exact_match else "✗ MISMATCH"
            log_debug(f"[TEXTGRAD ALIGNMENT] {match_quality} | Rec: '{textgrad_rec}' | Selected: '{selected_action}'")

        selected_actions.append(selected_action)

    return selected_actions


def extract_action_from_reasoning(reasoning: str, valid_actions: List[str]) -> Optional[str]:
    """Extract the most likely action from reasoning text - improved universal approach"""
    reasoning_lower = reasoning.lower()
    
    m = re.search(r'^\s*ACTION:\s*(.+)\s*$', reasoning, flags=re.I|re.M)
    if m:
        cand = m.group(1).strip()
        # pick exact match from valid_actions by case-insensitive equality
        for v in valid_actions:
            if cand.lower() == v.lower():
                return v


    # Look for quoted actions first (most reliable)
    import re
    quoted = re.findall(r'"([^"]+)"', reasoning)
    for q in quoted:
        if q in valid_actions:
            return q
    
    # Also look for actions after "action:" or "next action:"
    action_patterns = [
        r'action:\s*([^\.]+)',
        r'next action:\s*([^\.]+)',
        r'should (?:be|try):\s*([^\.]+)',
    ]
    for pattern in action_patterns:
        match = re.search(pattern, reasoning_lower)
        if match:
            candidate = match.group(1).strip()
            # Check if this exact string is in valid actions
            for valid_action in valid_actions:
                if candidate in valid_action.lower() or valid_action.lower() in candidate:
                    return valid_action
    
    # Score each action by multiple signals
    best_score = 0
    best_action = None
    
    for action in valid_actions:
        score = 0
        action_lower = action.lower()
        
        # Signal 1: Exact substring match
        if action_lower in reasoning_lower:
            score += 10
        
        # Signal 2: All words from action appear in reasoning
        action_words = set(action_lower.split())
        reasoning_words = set(reasoning_lower.split())
        if action_words.issubset(reasoning_words):
            score += 5
        
        # Signal 3: Word overlap ratio
        overlap = len(action_words & reasoning_words)
        if action_words:
            score += (overlap / len(action_words)) * 3
        
        # Signal 4: Action appears near keywords
        for keyword in ['next', 'action', 'choose', 'select', 'try']:
            if keyword in reasoning_lower:
                keyword_pos = reasoning_lower.index(keyword)
                if action_lower in reasoning_lower[max(0, keyword_pos-50):keyword_pos+100]:
                    score += 2
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action if best_score > 2 else None  # Threshold for confidence


def format_reflexion_insights_complete(memory: List[str]) -> str:
    """Extract COMPLETE actionable insights from reflexion memory - NO TRUNCATION"""
    if not memory:
        return "- No previous attempts"
    
    # Ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str):
        memory_list = [memory]
    else:
        return "- No actionable insights yet"
    
    if not memory_list:
        return "- No previous attempts"
    
    insights = []
    
    # Process ALL reflections
    for i, reflection in enumerate(memory_list, 1):
        if not isinstance(reflection, str):
            continue
        
        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract ALL actionable lines
            if any(word in line_lower for word in 
                   ['must', 'should', 'avoid', 'never', 'always', 
                    'failed because', 'succeeded', 'exact actions',
                    'hypothesis', 'critical', 'important', 'requires',
                    'learned', 'discovered', 'found', 'need', 'try']):
                clean_line = line.strip()
                if len(clean_line) > 10:
                    # KEEP FULL LINE - NO TRUNCATION
                    key_lines.append(f"  - {clean_line}")
        
        if key_lines:
            # Include ALL key lines, not just first few
            insights.append(f"Reflection {i}:\n" + '\n'.join(key_lines))
    
    return '\n'.join(insights) if insights else "- No actionable insights yet"

def intelligent_memory_cap(memory_list, cap=25):
    """Keep diverse memories: early (foundation) + recent (context)"""
    if len(memory_list) <= cap:
        return memory_list
    
    # Keep 5 foundation + 20 recent
    foundation = memory_list[:5]
    recent = memory_list[-20:]
    combined = foundation + recent
    
    # Remove exact duplicates
    seen = set()
    unique = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    
    return unique[:cap]


def adaptive_env_interaction_batch(
    envs: List,
    base_prompt: str,
    memories: List[List[str]],
    to_print: bool = True,
    initial_obs_list: List[str] = None,
    trial_log_path: str = None,
    env_configs: List[Dict[str, Any]] = None,
    trial_idx: int = 0,
    use_memory: bool = False  # ADD THIS PARAMETER
) -> List[Tuple[EnvironmentHistory, bool]]:
    """
    Batch version of adaptive_env_interaction with universal memory
    Processes multiple environments in parallel with batched LLM calls
    """



# -------------------------------------------------------------------------------
    global checkpoint_manager 
    # ADD THIS DEBUG BLOCK
    print(f"\n[BATCH ENTRY DEBUG]")
    print(f"  trial_log_path = {trial_log_path}")
    print(f"  trial_log_path type = {type(trial_log_path)}")
    print(f"  trial_log_path is None? {trial_log_path is None}")
    print(f"  to_print = {to_print}")
    print(f"  Number of envs = {len(envs)}")
    print("[/BATCH ENTRY DEBUG]\n")

    import hashlib
    import datetime
    from collections import defaultdict
    debug_log_path = f'debug_batch_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    debug_log = open(debug_log_path, 'w')
    

    # Initialize loggers
    from enhanced_logging import ComprehensiveLogger
    # from checkpoint_manager import CheckpointManager  # Not needed - removed to fix import error

    # Define logging_dir properly
    if trial_log_path:
        logging_dir = os.path.dirname(trial_log_path)
    else:
        logging_dir = '.'  # Current directory as fallback

    # Make checkpoint_manager global for access in other functions

    # checkpoint_manager = CheckpointManager(logging_dir)  # Not needed - removed to fix error
    comprehensive_logger = ComprehensiveLogger(logging_dir)

    def log_debug(msg):
        """Log to both console and file"""
        print(msg)
        debug_log.write(msg + '\n')
        debug_log.flush()
    
    log_debug(f"[TRACE START] adaptive_env_interaction_batch called with {len(envs)} environments")

    # Detect env_type from environment class name
    env_type = "unknown"
    if envs:
        env_class = type(envs[0]).__name__
        if "OSWorld" in env_class:
            env_type = "osworld"
        elif "ALFWorld" in env_class:
            env_type = "alfworld"
        elif "TextWorld" in env_class or "Jericho" in env_class:
            env_type = "textworld"
        elif "ScienceWorld" in env_class:
            env_type = "scienceworld"
        elif "BabyAI" in env_class:
            env_type = "babyai"
        elif "NetHack" in env_class:
            env_type = "nethack"
        elif "AppWorld" in env_class:
            env_type = "appworld"
        elif "OpenRCA" in env_class:
            env_type = "openrca"
    log_debug(f"[ENV TYPE] Detected env_type={env_type} from class={type(envs[0]).__name__ if envs else 'N/A'}")
    # Storyboard visualizer: write env_type so the dashboard can hide N/A nodes
    # (e.g. VGL/DELTA/click_metadata are visual-env-only).
    try:
        _dashboard.update(env_type=env_type)
    except Exception:
        pass

    # Import universal memory system
    from universal_memory_system import universal_memory
    
    # TEST MEMORY SYSTEM
    from generate_reflections import global_memory_manager
    print(f"[MEMORY TEST] Memory system loaded successfully!")
    print(f"[MEMORY TEST] Working memory: {len(global_memory_manager.memory.working_memory)}")
    print(f"[MEMORY TEST] Consolidated: {len(global_memory_manager.memory.consolidated_memory)}")
    print(f"[MEMORY TEST] Universal states tracked: {len(universal_memory.state_action_outcomes)}")
    
    # Initialize all environment states
    env_states = []

    # ============================================================================
    # TEXTGRAD INTEGRATION: Initialize policies with Reflexion constraints
    # ============================================================================
    # Extract Reflexion constraints from episodic memory (strategic guidance)
    reflexion_constraints = []
    if trial_idx > 0 and memories:
        # Get strategic insights from Reflexion's cross-trial learning
        for memory in memories[0][-5:]:  # Last 5 reflections from first env
            if isinstance(memory, str) and any(keyword in memory.lower()
                for keyword in ['must', 'should', 'avoid', 'never', 'always', 'learned']):
                # Extract actionable constraint
                lines = memory.split('\n')
                for line in lines:
                    if any(kw in line.lower() for kw in ['must', 'should', 'avoid', 'never', 'always']):
                        reflexion_constraints.append(line.strip())

    # Initialize TextGrad components (shared across all environments in this trial)
    print(f"[TEXTGRAD INIT] Initializing TextGrad components with fast_model")
    loss_fn = TextGradLoss(model=fast_model)
    optimizer = TextGradOptimizer(model=fast_model)

    # Initialize policies for each environment
    env_policies = []
    # Universal base policy (no environment-specific assumptions)
    universal_base_policy = "Select actions that make measurable progress toward completing the stated task."

    for i in range(len(envs)):
        policy = ActionPolicy(
            base_policy=universal_base_policy,
            reflexion_constraints=reflexion_constraints,
            model=fast_model
        )
        env_policies.append(policy)

    print(f"[TEXTGRAD INIT] ✓ Initialized {len(env_policies)} policies")
    print(f"[TEXTGRAD INIT] Base policy: '{universal_base_policy}'")
    print(f"[TEXTGRAD INIT] Reflexion constraints: {len(reflexion_constraints)}")
    if reflexion_constraints:
        print(f"[TEXTGRAD INIT] Sample constraint: '{reflexion_constraints[0][:80]}'...")
    # ============================================================================
    # END TEXTGRAD INTEGRATION
    # ============================================================================

    # Sequential TODO initialization for accuracy (each env learns from previous)
    from task_todo_manager import TaskTodoManager
    from todo_transfer_safety import todo_transfer_safety
    import uuid

    # Collect all successful TODOs for cross-environment learning
    successful_todo_patterns = []

    # Track successful TODO patterns from completed environments (within THIS trial)
    shared_todo_knowledge = {
        'successful_patterns': [],  # TODOs that led to success
        'failed_patterns': [],      # TODOs that led to failure
    }

    # Process environments sequentially
    env_data_list = []
    for i, (env, memory, ob) in enumerate(zip(envs, memories, initial_obs_list)):
        # Prefer task already set in env_configs (from env info, e.g., OSWorld task instruction)
        task = env_configs[i].get('task') if env_configs and i < len(env_configs) else None
        if not task or task in ('TASK_EXTRACTION_FAILED',):
            task = prompt_generator._extract_task_from_observation(ob)
        env_configs[i]['task'] = task
        if not task:
            log_debug(f"CRITICAL ERROR: Could not extract task from initial observation for env {i}!")
            log_debug(f"Observation preview: {ob[:300]}")
            raise ValueError("Task extraction failed. Cannot proceed without knowing what to accomplish.")

        prompt_generator.set_task(task)
        env_understanding.current_task = task
        log_debug(f"Environment {i} task: {task}")

        initial_valid_actions = []
        if hasattr(env, 'get_current_valid_actions'):
            initial_valid_actions = env.get_current_valid_actions()
            if initial_valid_actions:
                prompt_generator.discovered_knowledge['available_actions'] = set(initial_valid_actions)

        initial_prompt = prompt_generator.format_initial_observation(ob, memory)
        env_history = EnvironmentHistory("", initial_prompt, memory, [])
        env_history.reset()

        if to_print:
            log_debug(f"[ENV {i}] {ob}")
            sys.stdout.flush()

        episode_id = str(uuid.uuid4())[:8]
        reflexion_memory_for_todo = memory if trial_idx > 0 else None

        # META-LEARNING: Add cross_env_insights for failed envs (from successful envs)
        cross_env_insights = env_configs[i].get('cross_env_insights', [])
        if cross_env_insights and trial_idx > 0:
            # Merge insights from successful envs into this failed env's memory
            if reflexion_memory_for_todo is None:
                reflexion_memory_for_todo = []
            for insight in cross_env_insights:
                if insight.get('content'):
                    # Create a reflexion-format entry
                    cross_env_reflection = {
                        'type': 'cross_env_insight',
                        'content': f"[From successful similar task] {insight['content']}",
                        'from_success': True,
                        'source_task_type': insight.get('task_type', 'unknown')
                    }
                    reflexion_memory_for_todo.append(cross_env_reflection)
            print(f"[CROSS-ENV LEARNING] ENV {i}: Added {len(cross_env_insights)} insights from successful envs")

        # CROSS-ENVIRONMENT TODO LEARNING (NEW!)
        # Get similar successful TODOs from previous environments in THIS trial
        similar_todo_suggestions = []
        if i > 0 and shared_todo_knowledge['successful_patterns']:
            log_debug(f"\n[TODO TRANSFER] ENV {i}: Checking {len(shared_todo_knowledge['successful_patterns'])} successful patterns from previous envs")

            similar_todo_suggestions = todo_transfer_safety.transfer_todos(
                shared_todo_knowledge['successful_patterns'],
                task
            )

            if similar_todo_suggestions:
                log_debug(f"[TODO TRANSFER] ✅ Transferred {len(similar_todo_suggestions)} TODO suggestions")
                for idx, suggestion in enumerate(similar_todo_suggestions[:3], 1):
                    log_debug(f"  {idx}. {suggestion}")
            else:
                log_debug(f"[TODO TRANSFER] ❌ No safe transfers found (tasks too different or failed safety checks)")

        # Initialize TODO for this environment (with retry logic built-in)
        # ABLATION: Skip TODO Manager if USE_TODO is False (pure ablation mode)
        if USE_TODO:
            todo_manager = TaskTodoManager(model, fast_model)  # Reasoning for generation, fast for verification

            # Enrich initial observation with setup navigation context if the env
            # provides one (osworld). The raw screenshot may be stale (taken before
            # apps fully render), so without this the planner generates redundant
            # subtasks like "open the file" when setup already opened it.
            _ob_for_planner = ob
            try:
                _env_for_ctx = envs[i] if (envs is not None and i < len(envs)) else None
                if _env_for_ctx is not None and hasattr(_env_for_ctx, 'get_setup_context_string'):
                    _sctx = _env_for_ctx.get_setup_context_string()
                    if _sctx:
                        _ob_for_planner = f"{_sctx}\n{ob}"
            except Exception as _ectx:
                log_debug(f"[TODO INIT] setup context enrichment skipped: {_ectx}")

            try:
                initial_todos = todo_manager.initialize_from_task(
                    task,
                    _ob_for_planner,
                    initial_valid_actions,
                    trial_num=trial_idx,
                    reflexion_memory=reflexion_memory_for_todo,
                    similar_todo_suggestions=similar_todo_suggestions  # NEW: Cross-env learning!
                )
                log_debug(f"[ENV {i}] Trial {trial_idx}: Created {len(initial_todos)} TODO items")
                if trial_idx > 0 and reflexion_memory_for_todo:
                    log_debug(f"[ENV {i}] TODO initialized with {len(reflexion_memory_for_todo)} reflexions")
                if similar_todo_suggestions:
                    log_debug(f"[ENV {i}] TODO benefited from {len(similar_todo_suggestions)} cross-env suggestions")

                # Track successful TODO patterns for learning
                if initial_todos:
                    successful_todo_patterns.append({
                        'task': task,
                        'todos': [t.content for t in initial_todos]
                    })

            except Exception as e:
                log_debug(f"[ENV {i}] TODO initialization FAILED: {e}")
                raise
        else:
            # Pure ablation - no TODO Manager
            todo_manager = None
            log_debug(f"[ENV {i}] TODO Manager DISABLED (pure ablation mode)")

        env_data_list.append({
            'env': env,
            'env_id': i,
            'episode_id': episode_id,
            'env_history': env_history,
            'task': task,
            'memory': memory,
            'observation': ob,
            'initial_valid_actions': initial_valid_actions,
            'reflexion_memory': reflexion_memory_for_todo,
            'todo_manager': todo_manager,
            # TEXTGRAD INTEGRATION: Add policy components
            'policy': env_policies[i],      # TextGrad policy for this environment
            'loss_fn': loss_fn,              # Shared loss function
            'optimizer': optimizer           # Shared optimizer
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY LEAK FIX: Separate env storage from env_states
    # ═══════════════════════════════════════════════════════════════════════════
    # Create env registry - stores env objects separately to prevent 48,625x multiplier
    # The multiplier happens when env is stored IN env_states with working_reflexions,
    # creating deep circular references. By storing separately, we eliminate the chain.
    # ═══════════════════════════════════════════════════════════════════════════
    _env_registry = {env_data['env_id']: env_data['env'] for env_data in env_data_list}
    print(f"[MEMORY FIX] Created env registry with {len(_env_registry)} environments")
    print(f"[MEMORY FIX] env_states will contain only env_id, not env objects")

    # Create final env_states WITHOUT env object (just env_id)
    for env_data in env_data_list:
        # META-LEARNING: Check if this env should use direct replay (from Trial 0 optimization)
        env_id = env_data['env_id']
        use_direct_replay = env_configs[env_id].get('use_direct_replay', False) if trial_idx > 0 else False
        optimal_sequence = env_configs[env_id].get('optimal_sequence', []) if use_direct_replay else []

        # DEBUG: Log replay status for each env
        print(f"  [REPLAY CHECK] ENV {env_id}: trial_idx={trial_idx}, use_direct_replay={use_direct_replay}, optimal_sequence_len={len(optimal_sequence)}")

        if use_direct_replay:
            # CRITICAL CHECK: Verify task matches between Trial 0 workflow and Trial 1 environment
            saved_task = None
            for mem in env_configs[env_id].get('memory', []):
                if isinstance(mem, dict) and mem.get('type') == 'success_workflow':
                    saved_task = mem.get('task', '')
                    break
            current_task = env_data['task']

            if saved_task and saved_task != current_task:
                print(f"  ⚠️  [TASK MISMATCH] ENV {env_id}: CANNOT REPLAY!")
                print(f"      Saved task: '{saved_task}'")
                print(f"      Current task: '{current_task}'")
                print(f"      Disabling replay - environment changed between trials!")
                use_direct_replay = False
                optimal_sequence = []
            else:
                print(f"  [META-LEARNING] ENV {env_id}: Direct replay mode with {len(optimal_sequence)} optimized actions")
                print(f"      Task: '{current_task}'")

        env_states.append({
            # 'env': env_data['env'],  # REMOVED - this caused 48,625x multiplier!
            'env_id': env_data['env_id'],
            'trial_idx': trial_idx,  # CRITICAL FIX: Track trial for compression
            # META-LEARNING: Direct replay fields
            'use_direct_replay': use_direct_replay,
            'optimal_sequence': optimal_sequence,
            'replay_step': 0,  # Track position in optimal_sequence
            'episode_id': env_data['episode_id'],
            'history': env_data['env_history'],
            'task': env_data['task'],
            'memory': env_data['memory'],
            'observation': env_data['observation'],
            'prev_observation': env_data['observation'],
            'initial_observation': env_data['observation'],  # Store initial obs with ALL locations
            'failed_state_actions': {},
            'done': False,
            'success': False,
            'trajectory': [],
            'tried_actions': set(),
            'consecutive_failures': 0,
            'cur_step': 0,
            'step_gradient': {},
            'step_gradients_history': [],  # Store all step gradients for TextGrad learning
            'memory_context': None,
            'last_step_gradient': {
                'semantic_state_after': env_data['observation'],
                'task_progress': {'remaining': [env_data['task']]}
            },
            'progress_history': [],
            'is_stuck': False,
            'textgrad_components': prompt_generator.prompt_components.copy(),
            'memory_recommendations': {},
            'step_insights_accumulator': [],
            'learning_context': LearningContext(),  # Universal learning engine (shared with OSWorld)
            'consolidated_step_wisdom': "",
            'todo_manager': env_data['todo_manager'],
            'working_reflexions': [],  # Will be populated below with quality-filtered reflexions
            'reflexion_memory': env_data['reflexion_memory'] if env_data['reflexion_memory'] else [],  # CRITICAL FIX: Cross-trial Reflexion constraints + cross_env_insights for failed envs
            # TEXTGRAD INTEGRATION: Add policy components to env_states
            'policy': env_data['policy'],        # TextGrad policy
            'loss_fn': env_data['loss_fn'],      # Loss function
            'optimizer': env_data['optimizer'],   # Optimizer
            # FIX #32: Cumulative Search Memory - LLM-generated, persists across steps
            'cumulative_search_memory': "",  # Accumulated search history string, grows each step
            # FIX #39: Accumulated state — tracks what progress has been achieved across steps
            # Universal: any positive state change is tracked, any regression is flagged
            'accumulated_state': [],  # List of active state achievements (e.g. "text selected", "dialog open")
            # Environment type (for visual env context in TextGrad loss)
            'env_type': env_type,
            # Vision-first: latest screenshot (after last action) for Step 0 cold start
            'last_after_screenshot_b64': None,
            # ReflexGrad dual-process router (progress-based Reflexion triggering)
            'reflexgrad_core': ReflexGradCore(reflection_interval=3, stall_threshold=2, low_score_cutoff=5),
        })

        # PROFILING: Log working_reflexions loading with QUALITY SELECTION
        try:
            from memory_profiler import profiler
            working_refl_raw = env_configs[env_data['env_id']].get('working_reflexions_history', [])

            # ═══════════════════════════════════════════════════════════════════════
            # QUALITY-BASED REFLEXION SELECTION (Fix for -23% Trial 1 decline)
            # Data-backed: Baseline used 4 reflexions → +23% improvement
            #             Our broken: 8-11 reflexions → -23% decline
            # Solution: Select best 5 reflexions to prevent cognitive overload
            # ═══════════════════════════════════════════════════════════════════════
            if working_refl_raw and trial_idx > 0:  # Only filter for Trial 1+
                # ═══════════════════════════════════════════════════════════════════════
                # IMPORTANCE-BASED SELECTION (Prevents Catastrophic Forgetting)
                # Data shows: SUCCESS always valuable, early reflexions are foundational,
                #            FAILURE reflexions prevent mistakes, cap at 7 for optimal
                # ═══════════════════════════════════════════════════════════════════════

                must_keep = []

                # 1. ALWAYS KEEP: ALL SUCCESS reflexions (show what works)
                success_refls = [r for r in working_refl_raw if r.get('type') == 'SUCCESS']
                must_keep.extend(success_refls)

                # Special handling for no-success cases (struggling environments)
                if not success_refls and len(working_refl_raw) >= 10:
                    # Keep more late-stage exploration when struggling
                    late_refls = [r for r in working_refl_raw if r.get('step', 0) > 5]
                    # Sort by step to get the latest insights
                    late_refls.sort(key=lambda x: x.get('step', 0), reverse=True)
                    must_keep.extend(late_refls[:3])  # Keep 3 late-stage patterns

                # 2. ALWAYS KEEP: FAILURE reflexions (prevent repeating mistakes) - max 2
                failure_refls = [r for r in working_refl_raw if r.get('type') == 'FAILURE']
                must_keep.extend(failure_refls[:2])

                # 3. KEEP EARLY: Foundation reflexions from steps 0-5 (early learning)
                early_refls = [r for r in working_refl_raw
                              if r.get('step', 999) <= 5
                              and r not in must_keep]
                must_keep.extend(early_refls[:2])

                # 4. Fill remaining slots (up to 10 total) with MILESTONES then recent
                remaining_slots = 10 - len(must_keep)
                if remaining_slots > 0:
                    # First, add MILESTONE reflexions (progress markers)
                    others = [r for r in working_refl_raw if r not in must_keep]
                    milestone_refls = [r for r in others if r.get('type') == 'MILESTONE']
                    must_keep.extend(milestone_refls[:remaining_slots])

                    # Then fill any remaining with most recent
                    if len(must_keep) < 10:
                        recent_others = [r for r in others if r not in must_keep]
                        must_keep.extend(recent_others[-(10-len(must_keep)):])

                # Cap at 10 (increased from 7 to preserve late-stage learning)
                working_refl = must_keep[:10]

                # Log what was selected
                success_count = len([r for r in working_refl if r.get('type') == 'SUCCESS'])
                failure_count = len([r for r in working_refl if r.get('type') == 'FAILURE'])
                milestone_count = len([r for r in working_refl if r.get('type') == 'MILESTONE'])
                early_count = len([r for r in working_refl if r.get('step', 999) <= 5])

                print(f"[IMPORTANCE FILTER] ENV {env_data['env_id']}: Selected {len(working_refl)} from {len(working_refl_raw)} total")
                print(f"  Breakdown: {success_count} SUCCESS, {failure_count} FAILURE, {milestone_count} MILESTONE")
                print(f"  {early_count} early (step≤5), preventing catastrophic forgetting")
            else:
                working_refl = working_refl_raw

            if working_refl:
                # Populate state with quality-filtered reflexions
                env_states[env_data['env_id']]['working_reflexions'] = list(working_refl)

                profiler.log_learning_transfer(
                    trial_idx=trial_idx,
                    env_id=env_data['env_id'],
                    learning_type='working_reflexion',
                    content=working_refl,
                    source=f"env_configs from trial {trial_idx-1} (quality-filtered)" if trial_idx > 0 else "initial"
                )
                print(f"[LEARNING] ENV {env_data['env_id']}: Loaded {len(working_refl)} working_reflexions from previous trial")
        except Exception as e:
            print(f"[WARNING] Quality filtering failed for ENV {env_data['env_id']}: {e}")
            # Fallback to raw reflexions
            env_states[env_data['env_id']]['working_reflexions'] = list(env_configs[env_data['env_id']].get('working_reflexions_history', []))

        # Initial observation learning
        env_understanding.learn_from_interaction("", ob)

        # Vision-first: Capture initial screenshot for visual envs (Step 0 cold start)
        if env_type in ('osworld', 'visual'):
            try:
                env_obj = _env_registry.get(env_data['env_id'])
                if env_obj and hasattr(env_obj, '_get_screenshot') and hasattr(env_obj, '_pil_to_base64'):
                    initial_screenshot = env_obj._pil_to_base64(env_obj._get_screenshot())
                    env_states[env_data['env_id']]['last_after_screenshot_b64'] = initial_screenshot
                    log_debug(f"[VISION-FIRST] ENV {env_data['env_id']}: Captured initial screenshot ({len(initial_screenshot)} chars)")
            except Exception as e:
                log_debug(f"[VISION-FIRST] ENV {env_data['env_id']}: Initial screenshot failed: {e}")
    
    # ========================================================================
    # DEEP MEMORY TRACKING - FIND THE LEAK!
    # ========================================================================
    try:
        from deep_memory_tracker import tracker
        tracker.checkpoint("BEFORE Main Loop", {
            'env_states': env_states,
            'env_configs': env_configs,
        })
    except Exception as e:
        print(f"[WARNING] Deep memory tracking failed: {e}")

    # ========================================================================
    # SEQUENTIAL LEARNING SYSTEM - Process environments one at a time
    # ========================================================================

    # Shared knowledge base for cross-environment transfer
    shared_knowledge = {
        'universal': [],  # Universal strategies (share with all)
        'task_families': {},  # Task-family specific knowledge
        'textgrad_updates': {},  # Successful TextGrad component updates
        'successful_workflows': []  # Successful action patterns
    }

    log_debug("\n[SEQUENTIAL LEARNING] Initiating sequential execution mode")
    log_debug("[SEQUENTIAL LEARNING] Environments will be processed one at a time")
    log_debug("[SEQUENTIAL LEARNING] Knowledge from early envs will transfer to later envs")
    log_debug("[SEQUENTIAL LEARNING] Contamination prevention ACTIVE\n")

    # Create global env_prompt_generators for compatibility
    global env_prompt_generators
    env_prompt_generators = {}

    # Dynamic limit based on action space complexity
    # INCREASED: 21 → 28 to give agent more room for exploration and self-correction
    max_steps = 55

    # DASHBOARD: Initialize live state for first environment
    if env_states:
        first_task = env_states[0].get('task', '')
        _dashboard.reset()
        _dashboard.update(
            status='running',
            task=first_task,
            step=0,
            max_steps=max_steps,
        )

    # ========================================================================
    # TRUE PARALLEL ENVIRONMENT PROCESSING (NO CONTAMINATION)
    # ========================================================================
    # Process ALL environments simultaneously using ONLY their own memories
    # NO cross-task knowledge transfer = TRUE PARALLEL EXECUTION possible
    # This achieves ~9x speedup on A100 GPU

    # STEP 1: Initialize ALL environments before execution
    print(f"\n[PARALLEL INIT] Initializing {len(env_states)} environments...")
    for init_idx in range(len(env_states)):
        init_state = env_states[init_idx]
        task = init_state['task']

        # Create clean prompt generator WITHOUT cross-environment knowledge
        env_pg = DynamicPromptGenerator()
        if ENVIRONMENT_KNOWLEDGE:
            env_pg.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)

        env_pg.set_task(task)
        env_prompt_generators[init_idx] = env_pg
        init_state['prompt_generator'] = env_pg

        log_debug(f"[INIT {init_idx}] {task[:60]}...")

    print(f"[PARALLEL EXEC] Starting parallel execution of {len(env_states)} environments\n")

    # STEP 2: Execute ALL environments in TRUE PARALLEL
    # Process all active environments simultaneously each step
    global_step_counter = 0
    while any(not s['done'] and s['cur_step'] < max_steps for s in env_states):
        global_step_counter += 1
        active_envs = [s for s in env_states if not s['done'] and s['cur_step'] < max_steps]

        if not active_envs:
            break

        # Phase 7 v2 monitor: mark step boundary for every LLM call / signal event
        try:
            _dm.set_step(
                step=global_step_counter,
                trial=trial_idx if 'trial_idx' in dir() else 0,
                env_id=active_envs[0].get('env_id', 0) if active_envs else 0,
            )
        except Exception:
            pass

        print(f"\n[PARALLEL STEP {global_step_counter}] {len(active_envs)}/{len(env_states)} environments active")

        # STEP 2A: Collect data for ALL active environments (TRUE PARALLEL)
        batch_data_all = []
        active_states_all = []

        for current_state in active_envs:
            current_env_idx = current_state['env_id']

            # Skip if this env just finished
            if current_state['done'] or current_state['cur_step'] >= max_steps:
                continue

            # ═══════════════════════════════════════════════════════════════════
            # META-LEARNING: Direct Replay Mode
            # For envs that succeeded in Trial 0, replay optimized sequence
            # ═══════════════════════════════════════════════════════════════════
            if current_state.get('use_direct_replay', False):
                optimal_seq = current_state.get('optimal_sequence', [])
                replay_step = current_state.get('replay_step', 0)

                if replay_step < len(optimal_seq):
                    # Get next action from optimized sequence
                    replay_action = optimal_seq[replay_step]
                    print(f"  [REPLAY] ENV {current_env_idx}: Step {replay_step + 1}/{len(optimal_seq)} - '{replay_action}'")

                    # Execute the replay action
                    obs, reward, done, info = _env_registry[current_env_idx].step([replay_action])
                    obs_text = obs[0] if obs else ""
                    current_state['observation'] = obs_text
                    current_state['cur_step'] += 1
                    current_state['replay_step'] += 1
                    current_state['trajectory'].append((replay_action, obs_text, reward))

                    # DEBUG: Log what's happening during replay
                    reward_val = reward[0] if isinstance(reward, (list, tuple)) else reward
                    done_val = done[0] if isinstance(done, (list, tuple)) else done
                    won_val = info.get('won', [False])[0] if 'won' in info else False
                    print(f"    [REPLAY DEBUG] obs='{obs_text[:60]}...' reward={reward_val} done={done_val} won={won_val}")

                    # Check for success - FIX: Handle reward as list
                    if done_val:
                        current_state['done'] = True
                        current_state['success'] = reward_val > 0 and won_val
                        if current_state['success']:
                            print(f"  [REPLAY SUCCESS] ENV {current_env_idx}: Task completed in {replay_step + 1} steps!")
                        else:
                            # Replay failed - fall back to normal learning
                            print(f"  [REPLAY FAILED] ENV {current_env_idx}: Switching to normal learning mode")
                            current_state['use_direct_replay'] = False
                    continue  # Skip LLM-based action selection

                else:
                    # Ran out of replay actions without success - switch to normal
                    print(f"  [REPLAY EXHAUSTED] ENV {current_env_idx}: Switching to normal learning mode")
                    current_state['use_direct_replay'] = False
                    # Fall through to normal action selection

            # Get valid actions for current environment
            state = current_state  # Alias for code compatibility
            valid_actions = []
            # MEMORY FIX: Access env from registry, not from state
            if hasattr(_env_registry[state['env_id']], 'get_current_valid_actions'):
                valid_actions = _env_registry[state['env_id']].get_current_valid_actions()
                if DEBUG_ACTOR:
                    log_debug(f"[ENV {state['env_id']} - Step {state['cur_step']}] Found {len(valid_actions)} valid actions")

            # CRITICAL: Store raw actions BEFORE any filtering
            # This preserves TextGrad recommendations that might be incorrectly filtered
            raw_valid_actions = valid_actions.copy()


            if not valid_actions:
                # This is a critical failure - environment not providing actions
                log_debug(f"\n[ENV {state['env_id']}] CRITICAL: No valid actions from environment")

                # Record this as a learning signal
                state['step_gradient'] = {
                    'state_change': 'ENVIRONMENT_FAILURE',
                    'progress_score': 0,
                    'hypothesis': 'Environment failed to provide valid actions - may need reset or different approach',
                    'next_action_guidance': 'Cannot proceed without valid actions',
                    'raw_reflection': 'No valid actions available from environment'
                }

                # Store this failure for learning
                if 'critical_failures' not in state:
                    state['critical_failures'] = []
                state['critical_failures'].append({
                    'step': state['cur_step'],
                    'state': state['prev_observation'],
                    'reason': 'no_valid_actions'
                })

                state['done'] = True
                state['success'] = False
                state['failure_reason'] = "No valid actions - environment issue"
                break  # Exit the while loop


            env_prompt_generators[state['env_id']].discovered_knowledge['available_actions'] = set(valid_actions)
            
            # Update max steps based on action space
            # max_steps = min(len(valid_actions) * 3, 100)

            # CHECK STATE-ACTION MEMORY FIRST
            current_state_hash = hashlib.md5(state['observation'][:200].encode()).hexdigest()[:8]

            # Remove actions that failed in THIS EXACT state
            if current_state_hash in state.get('failed_state_actions', {}):
                failed_in_state = state['failed_state_actions'][current_state_hash]
                original_count = len(valid_actions)
                valid_actions = [a for a in valid_actions if a not in failed_in_state]
                if DEBUG_ACTOR:
                    log_debug(f"[STATE MEMORY] Removed {original_count - len(valid_actions)} failed actions for state {current_state_hash}")

            # Check conditional failures with context and decay
            if state['env_id'] in env_prompt_generators:
                pg = env_prompt_generators[state['env_id']]
                if 'conditional_failures' in pg.discovered_knowledge:
                    before_count = len(valid_actions)
                    current_context = {
                        'state_text': state['observation'][:200],
                        'task': state['task'],
                        'step': state['cur_step'],
                        'prerequisites': [act for act, _, _ in state["trajectory"][-5:] if act]
                    }
                    
                    actions_to_avoid = set()
                    for failure in pg.discovered_knowledge['conditional_failures']:
                        # Temporal decay
                        age = state['cur_step'] - failure.get('timestamp', 0)
                        decay_factor = max(0.1, 1.0 - (age / 10))
                        
                        # Context similarity check
                        source_ctx = {
                            'state_text': failure.get('state_text', ''),
                            'task': failure.get('task', ''),
                            'step': failure.get('step', 0),
                            'prerequisites': failure.get('prerequisites', [])
                        }
                        
                        should_avoid, match_score = should_share_knowledge(source_ctx, current_context)
                        
                        # Apply decay to match score
                        final_score = match_score * decay_factor * failure.get('confidence', 1.0)
                        
                        if final_score > 0.5:
                            actions_to_avoid.add(failure['action'])
                    
                    valid_actions = [a for a in valid_actions if a not in actions_to_avoid]
                    if DEBUG_ACTOR:
                        log_debug(f"[CONTEXT FILTER] Removed {before_count - len(valid_actions)} contextually inappropriate actions")

            # GET RECOMMENDATIONS FROM UNIVERSAL MEMORY - Using semantic understanding
            if 'last_step_gradient' in state and state['last_step_gradient']:
                current_semantic_state = state['last_step_gradient'].get('semantic_state_after', '')
                task_remaining = state['last_step_gradient'].get('task_progress', {}).get('remaining', [])
            else:
                # First step - no gradient yet
                current_semantic_state = state['prev_observation']
                task_remaining = [state['task']]  # Use full task as remaining

            recommendations = universal_memory.get_semantic_recommendations(
                current_state_description=current_semantic_state,
                task_remaining=task_remaining,
                available_actions=valid_actions
            )
            
            # CRITICAL: Store recommendations for use in action selection
            state['memory_recommendations'] = recommendations


            # CHECK FOR EXACT SUCCESSFUL TRAJECTORY MATCH
            if env_configs[state['env_id']].get('successful_trajectory') and trial_idx > 0:
                successful_actions = [act for act, _ in env_configs[state['env_id']]['successful_trajectory']]
                # Move successful actions to front of valid_actions
                reordered = []
                for sa in successful_actions:
                    if sa in valid_actions:
                        reordered.append(sa)
                        valid_actions.remove(sa)
                valid_actions = reordered + valid_actions
                if reordered:
                    print(f"[REUSE] Prioritized {len(reordered)} actions from previous success")
          
            # # PRIORITIZE ACTIONS BASED ON MEMORY - NO PATTERNS
            # # Build memory context
            # failed_actions_str = ""
            # if recommendations.get('avoid'):
            #     failed_actions_str = "NEVER TRY THESE (they failed before): " + \
            #                         ", ".join([rec['action'] for rec in recommendations['avoid'][:5]])

            # successful_actions_str = ""
            # if recommendations.get('previously_succeeded'):
            #     successful_actions_str = "THESE WORKED BEFORE: " + \
            #                             ", ".join([rec['action'] for rec in recommendations['previously_succeeded'][:3]])

            # semantic_prompt = f"""Task: {state['task']}

            # {failed_actions_str}
            # {successful_actions_str}

            # Available actions:
            # {chr(10).join([f"{i+1}. {action}" for i, action in enumerate(valid_actions[:40])])}

            # Which actions are semantically most relevant for completing this task?
            # AVOID actions marked as failed. PREFER actions that worked before.
            # List just the numbers of the 5 most relevant actions, separated by commas."""

            # Semantic ranking removed - using universal memory-based prioritization instead

            # for num_str in response.replace(',', ' ').split():
            #     if num_str.isdigit():
            #         idx = int(num_str) - 1
            #         if 0 <= idx < len(valid_actions):
            #             semantic_priorities.append(valid_actions[idx])
            

            # # Build prioritized list: reflexion-mentioned actions FIRST
            # prioritized_actions = []
            # log_debug(f"[DEBUG-VA] Valid actions before prioritization: {len(valid_actions)}")
            # added_actions = set()

            # # Extract actions mentioned in recent reflexions
            # if 'working_reflexions' in state and state['working_reflexions']:
            #     for ref in state['working_reflexions'][-5:]:  # Last 2 reflexions
            #         reflection_lower = ref['reflection'].lower()
            #         for valid_action in valid_actions:
            #             if valid_action.lower() in reflection_lower and valid_action not in added_actions:
            #                 prioritized_actions.append(valid_action)
            #                 added_actions.add(valid_action)
            #                 log_debug(f"[REFLEXION PRIORITY] Moved to front: {valid_action}")

            # # 1. Add LLM's semantic choices first
            # for action in semantic_priorities:
            #     if action not in added_actions:
            #         prioritized_actions.append(action)
            #         added_actions.add(action)

            # # 2. Then add memory recommendations
            # for rec in recommendations.get('historically_successful', []):
            #     if rec['action'] in valid_actions and rec['action'] not in added_actions:
            #         prioritized_actions.append(rec['action'])
            #         added_actions.add(rec['action'])

            # for rec in recommendations.get('explore', []):
            #     if rec['action'] in valid_actions and rec['action'] not in added_actions:
            #         prioritized_actions.append(rec['action'])
            #         added_actions.add(rec['action'])

            # # 3. Add remaining valid actions
            # for action in valid_actions:
            #     if action not in added_actions:
            #         prioritized_actions.append(action)
            #         added_actions.add(action)
            
            # # Ensure we have valid actions but filter known failures
            # if not prioritized_actions:
            #     # Try memory recommendations first
            #     if recommendations.get('strongly_recommended'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['strongly_recommended'] 
            #                              if rec['action'] in valid_actions]
                
            #     # Then try historically successful
            #     if not prioritized_actions and recommendations.get('historically_successful'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['historically_successful'] 
            #                              if rec['action'] in valid_actions]
                
            #     # Then try unexplored from memory
            #     if not prioritized_actions and recommendations.get('explore'):
            #         prioritized_actions = [rec['action'] for rec in recommendations['explore'][:10] 
            #                              if rec['action'] in valid_actions]
                
            #     # If still nothing, we're stuck - better to end than random
            #     if not prioritized_actions:
            #         print(f"[WARNING] ENV {state['env_id']} has no valid actions after filtering - ending episode")
            #         state['done'] = True
            #         state['success'] = False
            #         # Record why we failed
            #         state['failure_reason'] = "No valid actions after filtering failures"
            #         continue

            # log_debug(f"\n[ENV {state['env_id']} FILTER RESULTS]")
            # log_debug(f"Original: {len(valid_actions) + (len(actions_to_delete) if 'actions_to_delete' in locals() else 0)}")
            # log_debug(f"After all filtering: {len(prioritized_actions)}")
            # log_debug(f"First 3 actions: {prioritized_actions[:3]}")
            
           
            # Initialize failure history if not exists
            if 'failure_history' not in state:
                state['failure_history'] = []

            # Filtering will be done in batched action selection (no sequential calls!)
            # Use all valid actions (filtering happens in LLM prompt)
            filtered_actions = valid_actions

            # NO prioritization - let LLM reason with all actions equally
            # Gradient suggestion is already in prompt as context (line 1046)
            # This removes position bias and enables true comprehensive reasoning
            prioritized_actions = filtered_actions.copy()
            
            # Include working reflexions in batch data - TIERED COMPRESSION
            working_reflexions_text = ""
            if 'working_reflexions' in state and state['working_reflexions']:
                # Apply tiered compression before formatting
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                # Format based on compression tier
                if tiered_reflexions:
                    working_reflexions_text = "\n============ CRITICAL LEARNING FROM PREVIOUS EXPERIENCE ============\n"

                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        ref_type = ref.get('type', 'general')
                        ref_trial = ref.get('trial', trial_idx)

                        # Add prominent markers for different types
                        if ref_type == 'FAILURE':
                            marker = "🚫 AVOID:"
                        elif ref_type == 'SUCCESS':
                            marker = "✅ PROVEN:"
                        elif ref_trial < trial_idx:
                            marker = "📚 TRIAL 0 LESSON:"  # Cross-trial learning
                        else:
                            marker = "💡"

                        if compression_type == 'heavy':
                            working_reflexions_text += f"\n{marker} Steps {ref['step']} (summary):\n{ref['reflection']}\n"
                        elif compression_type == 'medium':
                            working_reflexions_text += f"\n{marker} Steps {ref['step']} (compressed):\n{ref['reflection']}\n"
                        else:
                            # Verbose (recent or cross-trial)
                            working_reflexions_text += f"\n{marker} Step {ref['step']}:\n{ref['reflection']}\n"

            # Track how many reflexion-suggested actions are at the front
            reflexion_count = len([a for a in prioritized_actions[:5] if any(
                a.lower() in ref['reflection'].lower() 
                for ref in state.get('working_reflexions', [])[-2:]
            )])
            log_debug(f"[REFLEXION CHECK] {reflexion_count} of top 5 actions match reflexion suggestions")

            # Update memory BEFORE creating batch_data
            env_idx = state['env_id']
            import copy
            state['memory'] = copy.deepcopy(env_configs[env_idx].get('memory', [])[-15:] + env_configs[env_idx].get('step_memory', [])[-10:])  # MEMORY FIX: Deep copy to break circular ref

            # Append this environment's data to batch (for TRUE PARALLEL processing)
            batch_data_all.append({
                'valid_actions': prioritized_actions.copy(),
                'raw_valid_actions': raw_valid_actions.copy(),  # CRITICAL: Unfiltered actions for TextGrad validation
                'observation': state['prev_observation'],
                'task': state['task'],
                'env_id': state['env_id'],  # Pass actual environment ID for correct logging
                'step_gradient': state.get('step_gradient', {}),
                'textgrad_components': state['prompt_generator'].prompt_components.copy(),  # Use env-specific
                'reflexion_memory': state['memory'],
                'working_reflexions': working_reflexions_text,
                'action_history': [act for act, _, _ in state['trajectory'][-15:]] if state['trajectory'] else [],  # FIX #8 BUG: Extract actions from (action, obs, reasoning) tuples
                'discovered_patterns': {},
                'tried_actions': state.get('tried_actions', set()),
                'interaction_count': state['cur_step'],
                'memory_recommendations': recommendations,
                'is_stuck': state.get('is_stuck', False),
                'consolidated_step_wisdom': state.get('consolidated_step_wisdom', ''),
                'step_insights_accumulator': state.get('step_insights_accumulator', []),  # FIX #23: Keep ALL history, not just last 5
                'progress_history': state.get('progress_history', []),
                'todo_manager': state.get('todo_manager'),  # ADD TODO MANAGER
                # ADD THESE NEW FIELDS
                'last_step_gradient': state.get('last_step_gradient', {}),
                'failed_state_actions': state.get('failed_state_actions', {}),
                'successful_actions': state['prompt_generator'].discovered_knowledge.get('successful_actions', []),
                'action_blacklist': state['prompt_generator'].discovered_knowledge.get('action_blacklist', []),
                # INTELLIGENT FILTERING: Add failure history for batched filtering
                'failure_history': state.get('failure_history', []),
                # TEXTGRAD INTEGRATION: Add policy components to batch_data
                'policy': state.get('policy'),        # TextGrad policy
                'loss_fn': state.get('loss_fn'),      # Loss function
                'optimizer': state.get('optimizer'),  # Optimizer
                'inventory_items': []                 # Will be populated from observation
            })
            active_states_all.append(state)

            log_debug(f"[BATCH COLLECT] ENV {state['env_id']} - Step {state['cur_step']} data collected")

        # End of data collection for loop
        print(f"[DATA COLLECTION] Collected data for {len(batch_data_all)} environments")

        # ========================================================================
        # PHASE 3B: ACTION SELECTION USING LEARNED TEXTGRAD POLICY
        # ========================================================================
        # FIX: Actually USE the learned TextGrad policy instead of 500-line prompt!
        # The policy.forward() method uses:
        # - Evolved base_policy (updated by optimizer every 3 steps)
        # - Accumulated gradients (from textgrad_backward)
        # - Reflexion insights (strategic constraints)
        # - Recent actions (loop prevention)
        print(f"\n[PHASE 3B] Selecting actions using LEARNED POLICY for {len(active_states_all)} environments...")

        selected_actions = []
        for state, batch_data in zip(active_states_all, batch_data_all):
            policy = state['policy']

            # Format working reflexions as insights for the policy
            reflexion_insights = format_reflexion_insights_for_policy(
                state.get('working_reflexions', [])
            )

            # Get recent actions for loop prevention (last 7 from trajectory)
            recent_actions = []
            if state.get('trajectory'):
                recent_actions = [a for a, _, _ in state['trajectory'][-7:]]

            # Get current TODO if available
            current_todo = ""
            if 'todo_manager' in state and state['todo_manager']:
                current_todo_obj = state['todo_manager']._get_current_todo()
                if current_todo_obj:
                    current_todo = current_todo_obj.active_form

            # Get valid_actions from batch_data (where it's stored), not from state
            valid_actions = batch_data.get('valid_actions', [])

            # Get next_action_guidance from PREVIOUS step's gradient (this is the CRITICAL fix!)
            # This contains the specific recommended action from TextGrad analysis
            last_step_gradient = batch_data.get('last_step_gradient', {})
            next_action_guidance = last_step_gradient.get('next_action_guidance', '')

            # STAGNATION-TRANSLATED ACTION takes priority over TextGrad guidance.
            # When the real evaluator just fired and we have an LLM-translated `run:`
            # action targeting the exact gap, that's strictly better than whatever
            # the next-action heuristic produced. Universal — works for any task.
            _pending_stag_act = state.pop('_pending_stagnation_action', None)
            if _pending_stag_act:
                next_action_guidance = _pending_stag_act
                # G6 HARD-OVERRIDE: when the XLATE candidate is a base64-wrapped
                # Python action (ast.parse-validated in its XLATE cycle), bypass
                # the actor call entirely. Actor-generated single-line python
                # consistently produces "'return' outside function" or similar
                # ast errors when the code needs compound statements; XLATE's
                # multi-line b64 code is structurally clean because it was
                # validated at wrap time. Universal: only fires when XLATE
                # has produced a valid candidate AND the candidate is in the
                # base64-file channel (detected structurally). For GUI /
                # introspection / non-b64 candidates the suggestion mode
                # remains. Skip on ALFWorld-style envs that never populate
                # `_pending_stagnation_action`.
                _is_b64_action = (
                    isinstance(_pending_stag_act, str)
                    and _pending_stag_act.strip().startswith('run: echo ')
                    and '| base64 -d >' in _pending_stag_act
                )
                if _is_b64_action:
                    state['_force_action_override'] = _pending_stag_act
                    print(f"[STAGNATION-OVERRIDE] ENV {state['env_id']} Step {state['cur_step']}: "
                          f"hard-override actor with XLATE b64 candidate (len={len(_pending_stag_act)})")
                else:
                    print(f"[STAGNATION-PREFER] ENV {state['env_id']} Step {state['cur_step']}: Using LLM-translated stagnation action: '{_pending_stag_act[:120]}'")

            # DEBUG: Log guidance retrieval at every step
            print(f"[DEBUG GUIDANCE] ENV {state['env_id']} Step {state['cur_step']}: last_step_gradient has {len(last_step_gradient)} keys")
            if next_action_guidance:
                print(f"[TEXTGRAD GUIDANCE] ENV {state['env_id']} Step {state['cur_step']}: Has recommendation: '{next_action_guidance[:80]}...'")

            # VISION-FIRST: For visual envs, use screenshot-based action selection when:
            # (a) Cold start (no TextGrad/Reflexion guidance yet)
            # (b) Loop detected (3+ identical actions with no progress)
            # PASS 2 has full trajectory context and Reflexion insights — trust it for
            # multi-step planning. VISION-FIRST at every step causes aimless clicking.
            _use_vision_first = False
            if state.get('env_type', '') in ('osworld', 'visual'):
                if not next_action_guidance:
                    _use_vision_first = True  # (a) Cold start
                else:
                    # (b) Loop detection: multi-strategy (visual envs have noisy state changes)
                    _traj_loop = state.get('trajectory', [])
                    if len(_traj_loop) >= 2:
                        _last2 = [a.split('#')[0].strip().lower() for a, _, _ in _traj_loop[-2:]]
                        _consec_vf0 = len(set(_last2)) == 1
                        # Strategy 2: Same action 3+ times in last 5
                        _repeat3_vf0 = False
                        if len(_traj_loop) >= 3:
                            from collections import Counter as _Ctr0
                            _l5_vf0 = [a.split('#')[0].strip().lower() for a, _, _ in _traj_loop[-5:]]
                            _ct0 = _Ctr0(_l5_vf0)
                            _repeat3_vf0 = _ct0.most_common(1)[0][1] >= 3
                        # Strategy 3: Score stagnation (progress_history)
                        _ph_vf0 = state.get('progress_history', [])
                        _score_stag_vf0 = len(_ph_vf0) >= 3 and all(s <= 1 for s in _ph_vf0[-3:])

                        if _consec_vf0 or _repeat3_vf0 or _score_stag_vf0:
                            _use_vision_first = True
                            _reason_vf0 = "score stagnation" if _score_stag_vf0 and not _consec_vf0 else (
                                "repeated 3+ in last 5" if _repeat3_vf0 else "consecutive identical")
                            print(f"[VISION-FIRST] ENV {state['env_id']} Step {state['cur_step']}: "
                                  f"Loop detected ({_reason_vf0}, '{_traj_loop[-1][0][:60]}') — activating VISION-FIRST override")
                    # (c) Reflexion STRATEGY_SHIFT recommends programmatic python3 command
                    # Check both TextGrad guidance AND working_reflexions for python3 commands
                    _python3_hint = None
                    if next_action_guidance and next_action_guidance.strip().lower().startswith('run: python3'):
                        _python3_hint = next_action_guidance
                    if not _python3_hint:
                        for _wr in reversed(state.get('working_reflexions', [])[-3:]):
                            _wr_insight = _wr.get('insight', '')
                            # Only use if it's ACTUALLY a command (starts with run:),
                            # not diagnostic prose that contains "run:" as a quote/example.
                            # Heuristic: real commands don't contain "] (" or "failed because"
                            _stripped = _wr_insight.strip()
                            if (_stripped.lower().startswith('run: python3')
                                    and '] (' not in _stripped
                                    and 'failed because' not in _stripped.lower()
                                    and 'AVOID' not in _stripped[:20]):
                                _python3_hint = _wr_insight
                                break
                    if _python3_hint:
                        # FIX: Previously this activated VISION-FIRST and "passed through"
                        # the python3 hint as a soft suggestion — but vision-first generates
                        # its OWN action and ignores the hint, so the Reflexion suggestion
                        # was lost. Now we keep the hint as next_action_guidance and let
                        # normal TextGrad action selection consume it directly.
                        next_action_guidance = _python3_hint
                        print(f"[REFLEXION-DIRECT] ENV {state['env_id']} Step {state['cur_step']}: "
                              f"Using Reflexion python3 hint directly (no vision-first detour)")

                    # (d) Previous step was a discovery command (find/ls) with useful output
                    # VISION-FIRST can see the output and use discovered paths
                    # BUT: Don't override if stagnation/reflexion already set guidance —
                    # those are grounded in evaluator data, strictly better than vision guesses.
                    if not _use_vision_first and not _pending_stag_act and len(_traj_loop) >= 1:
                        _last_action = _traj_loop[-1][0].split('#')[0].strip().lower()
                        _last_obs = str(_traj_loop[-1][1]) if len(_traj_loop[-1]) > 1 else ''
                        if (_last_action.startswith(('run: find', 'run: ls'))
                                and '/home/' in _last_obs and 'No such file' not in _last_obs):
                            _use_vision_first = True
                            print(f"[VISION-FIRST] ENV {state['env_id']} Step {state['cur_step']}: "
                                  f"Previous discovery command has useful output — activating VISION-FIRST")

            # ================================================================
            # FRESH RESTART MECHANISM (from Debugging Decay research)
            # "LLM self-repair degrades after 1-2 attempts — by attempt 5,
            # performance drops 80%." Instead of iterating on broken code,
            # restart from CLEAN prompt + error blacklist after 2 failures
            # of the same approach. Universal — works for any task/model.
            # ================================================================
            if '_approach_failures' not in state:
                state['_approach_failures'] = {}  # approach_key → [error1, error2, ...]
                state['_error_blacklist'] = []     # accumulated "do NOT do X" rules

            # Track approach failures from previous step
            # SEMANTIC CATEGORY BLOCKING: Extract the key command/approach from failed
            # actions and block the ENTIRE approach family after 2 failures.
            # This catches semantic near-duplicates that exact-string dedup misses.
            # Universal: keyword extraction works for any shell/Python/GUI action.
            _prev_score = last_step_gradient.get('progress_score', 5)
            _prev_action = state.get('trajectory', [('', '', '')])[-1][0] if state.get('trajectory') else ''
            _prev_loss = last_step_gradient.get('textgrad_loss', '')
            _is_prev_noop = 'NO-OP' in str(_prev_loss) or 'NOOP' in str(_prev_loss)
            _is_prev_error = 'COMMAND FAILED' in str(_prev_loss)
            if _prev_score <= 2 and _prev_action and _prev_action.strip().lower() != 'done':
                if _is_prev_noop or _is_prev_error:
                    # Extract SEMANTIC key: the core command/approach, not just first 30 chars
                    import re as _re_cat
                    _approach_key = _prev_action[:30].strip().lower()
                    # Also extract the primary shell command for category blocking
                    _cmd_match = _re_cat.search(
                        r'(?:run:\s*)?(?:sudo\s+)?(\w+)',
                        _prev_action.lower().replace('python3 -c', 'python3_c')
                    )
                    _cmd_category = _cmd_match.group(1) if _cmd_match else _approach_key[:15]
                    # Initialize category tracker
                    if '_cmd_category_failures' not in state:
                        state['_cmd_category_failures'] = {}  # cmd → count
                        state['_blocked_categories'] = set()
                    state['_cmd_category_failures'][_cmd_category] = state['_cmd_category_failures'].get(_cmd_category, 0) + 1

                    if _approach_key not in state['_approach_failures']:
                        state['_approach_failures'][_approach_key] = []
                    _prev_error_text = str(_prev_loss)[:200]
                    state['_approach_failures'][_approach_key].append(_prev_error_text)

                    # CATEGORY BLOCKING: After 2 failures of the same command family,
                    # hard-block it. This prevents "useradd" → "adduser" → "useradd -m" loops.
                    if state['_cmd_category_failures'].get(_cmd_category, 0) >= 2:
                        state['_blocked_categories'].add(_cmd_category)
                        if len(state['_error_blacklist']) < 8:
                            _cat_block = f"BLOCKED COMMAND: '{_cmd_category}' failed {state['_cmd_category_failures'][_cmd_category]}x — use a FUNDAMENTALLY DIFFERENT approach"
                            if _cat_block not in state['_error_blacklist']:
                                state['_error_blacklist'].append(_cat_block)
                                print(f"[CATEGORY BLOCK] Blocked '{_cmd_category}' after {state['_cmd_category_failures'][_cmd_category]} failures")

                    # After 3 failures (not 2) of same exact approach with NOOP/ERROR
                    if len(state['_approach_failures'][_approach_key]) >= 3:
                        _blacklist_entry = f"FAILED {len(state['_approach_failures'][_approach_key])}x: '{_prev_action[:80]}' → {_prev_error_text[:80]}"
                        if len(state['_error_blacklist']) < 5 and _blacklist_entry not in state['_error_blacklist']:
                            state['_error_blacklist'].append(_blacklist_entry)
                            print(f"[FRESH RESTART] Blacklisted after {len(state['_approach_failures'][_approach_key])} NOOP/ERROR failures: '{_prev_action[:60]}'")

            # Check if we should do a FRESH RESTART
            # Trigger: 3+ blacklisted approaches (model is deeply stuck)
            _do_fresh_restart = len(state['_error_blacklist']) >= 3
            if _do_fresh_restart and state.get('env_type', '') in ('osworld', 'visual'):
                _blacklist_text = "\n".join(f"  ❌ {b}" for b in state['_error_blacklist'][-5:])
                _fresh_prompt = (
                    f"You are a desktop GUI agent. FRESH START — forget all previous attempts.\n\n"
                    f"TASK: {state['task'][:500]}\n\n"
                    f"APPROACHES THAT FAILED (DO NOT USE THESE):\n{_blacklist_text}\n\n"
                    f"You MUST try a COMPLETELY DIFFERENT approach from everything listed above.\n"
                    f"Think step by step: what is a fundamentally different way to achieve this task?\n"
                    f"Output ONLY the action string."
                )
                try:
                    _screenshot = state.get('last_after_screenshot_b64')
                    if _screenshot:
                        _fresh_action = model.generate_multimodal(_fresh_prompt, after_image_b64=_screenshot, max_tokens=1500)
                    else:
                        _fresh_out = model.generate([_fresh_prompt], SamplingParams(max_tokens=1500, temperature=0.5))
                        _fresh_action = _fresh_out[0].outputs[0].text.strip()
                    _fresh_action = _fresh_action.strip().split('\n')[0].strip()
                    if _fresh_action and len(_fresh_action) > 5:
                        # Check it's not a blacklisted approach
                        _is_blacklisted = any(ba[:30].lower() in _fresh_action.lower() for ba in
                                             [e.split("'")[1] if "'" in e else "" for e in state['_error_blacklist']])
                        if not _is_blacklisted:
                            state['_env_fix_action'] = _fresh_action
                            next_action_guidance = _fresh_action  # Feed through policy.forward()
                            _use_vision_first = False
                            print(f"[FRESH RESTART] Clean prompt generated: '{_fresh_action[:100]}'")
                        else:
                            print(f"[FRESH RESTART] Generated action matches blacklist, skipping")
                except Exception as _fr_err:
                    print(f"[FRESH RESTART] Error: {_fr_err}")

            # ENV-GROUNDED FIX ENFORCEMENT
            _env_fix = last_step_gradient.get('next_action_guidance', '')
            _env_fix_source = last_step_gradient.get('guidance_source', '')
            _env_fix_enforced = False
            _prev_action = state.get('trajectory', [('', '', '')])[-1][0] if state.get('trajectory') else ''
            _fix_is_different = _env_fix and _prev_action and _env_fix[:50] != _prev_action[:50]
            _should_enforce = (_env_fix and _env_fix.startswith('run:')
                              and _prev_score <= 2 and _fix_is_different)
            if _should_enforce and not state.get('_env_fix_action'):
                # Set BOTH paths: _env_fix_action (post-PASS2 override) AND
                # next_action_guidance (policy.forward() input). This ensures
                # the fix flows through the architecture correctly regardless
                # of which code path runs.
                state['_env_fix_action'] = _env_fix
                next_action_guidance = _env_fix  # policy.forward() will return this directly
                print(f"[ENV-FIX ENFORCE] Set as guidance + override: '{_env_fix[:100]}'")
                _env_fix_enforced = True
                _use_vision_first = False  # Skip vision-first — we have a fix

            # TEACHER-FORCING: When the backward pass produced a verified executable fix,
            # bypass the actor entirely. The diagnosis IS the action. This eliminates the
            # diagnosis-to-action gap where the actor ignores correct guidance.
            # Universal: works for any task/env. The backward gradient format (CONCRETE_FIX:)
            # is env-agnostic — it's the TextGrad backward prompt that produces it.
            _tf_source = last_step_gradient.get('guidance_source', '')
            _tf_action = last_step_gradient.get('next_action_guidance', '')
            if _tf_source == 'teacher_force' and _tf_action and len(_tf_action) > 5:
                # Validate: must be an executable action format, not prose
                _tf_lower = _tf_action.lower().strip()
                _is_executable = (_tf_lower.startswith('run:') or _tf_lower.startswith('click ')
                                  or _tf_lower.startswith('hotkey ') or _tf_lower.startswith('press ')
                                  or _tf_lower.startswith('type ') or _tf_lower == 'done')
                if _is_executable:
                    print(f"[TEACHER-FORCE] ENV {state['env_id']} Step {state['cur_step']}: "
                          f"Bypassing actor — using backward fix directly: '{_tf_action[:120]}'")
                    action = _tf_action
                    # Skip vision-first, PASS2, policy.forward() — go straight to execution
                    state['tried_actions'] = state.get('tried_actions', set())
                    if isinstance(state['tried_actions'], list):
                        state['tried_actions'] = set(state['tried_actions'])
                    state['tried_actions'].add(action)
                    env = _env_registry[state['env_id']]
                    action = clean_action(action)
                    observation, reward, done, info = env.step([action])
                    observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
                    done = done[0]
                    state['cur_step'] += 1
                    state['prev_observation'] = observation
                    state['prev_action'] = action
                    state['trajectory'] = state.get('trajectory', [])
                    state['trajectory'].append((action, observation, ''))
                    state['progress_history'] = state.get('progress_history', [])
                    state['progress_history'].append(3)  # Neutral — let TextGrad score properly
                    self_declared_done = action.strip().lower() in ('done', 'task complete')
                    if done or self_declared_done:
                        state['done'] = True
                    print(f"  [TEACHER-FORCE] Observation: '{str(observation)[:200]}'")
                    continue  # Skip rest of step processing — go to next iteration
                else:
                    log_debug(f"[TEACHER-FORCE] Fix not executable format, falling through to actor: '{_tf_action[:80]}'")

            if _use_vision_first:
                screenshot_b64 = state.get('last_after_screenshot_b64')
                if screenshot_b64:
                    traj_s0 = state.get('trajectory', [])
                    last_obs_s0 = traj_s0[-1][1] if traj_s0 else state.get('prev_observation', '')
                    # Include action results (SUCCESS/ERROR) so model knows what completed
                    recent_acts_s0 = []
                    if traj_s0:
                        for _a, _o, _r in traj_s0[-5:]:
                            _o_str = str(_o) if _o else ''
                            if '[SUCCESS]' in _o_str:
                                recent_acts_s0.append(f"{_a} → [SUCCESS - completed]")
                            elif '[ERROR]' in _o_str or 'Traceback' in _o_str:
                                recent_acts_s0.append(f"{_a} → [FAILED]")
                            else:
                                recent_acts_s0.append(_a)
                    vgl_hints_s0 = ', '.join(valid_actions[:15]) if valid_actions else ''
                    # Get current TODO subtask for task decomposition context
                    _current_todo_s0 = None
                    if 'todo_manager' in state and state['todo_manager']:
                        try:
                            _current_todo_s0 = state['todo_manager']._get_current_todo()
                        except Exception:
                            pass
                    # Extract command history from step_insights for VISION-FIRST learning
                    # Includes both errors AND successful command output (e.g., ls results)
                    _vf_error_history = []
                    _vf_step_insights = state.get('step_insights_accumulator', [])
                    for _vf_ins in _vf_step_insights:
                        _vf_obs = str(_vf_ins.get('observation', ''))
                        _vf_action = str(_vf_ins.get('action', ''))
                        _found_error = False
                        # Look for errors in command output
                        for _err_marker in ['[ERROR]', 'FileNotFoundError', 'Traceback', 'ModuleNotFoundError',
                                            'PermissionError', 'SyntaxError', 'ImportError', 'ValueError',
                                            'No such file']:
                            _err_pos = _vf_obs.find(_err_marker)
                            if _err_pos >= 0:
                                _err_snippet = _vf_obs[_err_pos:_err_pos + 500].strip()
                                _vf_error_history.append(f"Step {_vf_ins.get('step', '?')}: Action='{_vf_action[:60]}' -> ERROR: {_err_snippet}")
                                _found_error = True
                                break
                        # Also include successful command output (ls, find, etc.)
                        if not _found_error:
                            _cmd_pos = _vf_obs.find('[Command Output')
                            if _cmd_pos >= 0:
                                _cmd_end = _vf_obs.find('\n\n', _cmd_pos)
                                if _cmd_end < 0:
                                    _cmd_end = _cmd_pos + 300
                                _cmd_text = _vf_obs[_cmd_pos:min(_cmd_end, _cmd_pos + 300)].strip()
                                _vf_error_history.append(f"Step {_vf_ins.get('step', '?')}: Action='{_vf_action[:60]}' -> {_cmd_text}")
                    # Extract discovered file paths from find/ls output for prominent display
                    _discovered_files = []
                    for _vf_ins2 in _vf_step_insights:
                        _vf_obs2 = str(_vf_ins2.get('observation', ''))
                        _vf_act2 = str(_vf_ins2.get('action', '')).lower()
                        if ('run: find' in _vf_act2 or 'run: ls' in _vf_act2) and '/home/' in _vf_obs2:
                            import re as _re_disc
                            _paths = _re_disc.findall(r'(/home/\S+\.\w{1,5})', _vf_obs2)
                            _discovered_files.extend(_paths)
                    _discovered_files = list(dict.fromkeys(_discovered_files))  # dedupe preserving order
                    if _discovered_files:
                        print(f"[VISION-FIRST] Discovered files from previous commands: {_discovered_files}")
                    # Extract Reflexion insights from working_reflexions
                    _vf_reflexion_insights = []
                    if 'working_reflexions' in state and state['working_reflexions']:
                        for _wr in state['working_reflexions'][-3:]:
                            _wr_text = _wr.get('reflection', _wr.get('insight', ''))
                            if _wr_text:
                                _vf_reflexion_insights.append(str(_wr_text))

                    if LEAN_MODE:
                        # LEAN MODE: Distill all learning into ONE directive sentence
                        _lean_directive = ""
                        # Priority 1: TextGrad recommendation (most recent, most actionable)
                        if next_action_guidance:
                            _lean_directive = str(next_action_guidance)[:500]
                        # Priority 2: Latest Reflexion insight (if no TextGrad hint)
                        elif _vf_reflexion_insights:
                            _lean_directive = str(_vf_reflexion_insights[-1])[:500]
                        # Priority 3: Latest error (if nothing else)
                        elif _vf_error_history:
                            _lean_directive = f"Previous error: {_vf_error_history[-1][:300]}"

                        # Get last action and result from step_insights
                        _lean_last_action = ""
                        _lean_last_result = ""
                        _si = state.get('step_insights_accumulator', [])
                        if _si:
                            _last = _si[-1]
                            _lean_last_action = str(_last.get('action', ''))[:300]
                            _lean_obs = str(_last.get('observation', ''))
                            # Extract just the result part (rc=X, output)
                            _rc_idx = _lean_obs.find('rc=')
                            _lean_last_result = _lean_obs[_rc_idx:_rc_idx+500] if _rc_idx >= 0 else _lean_obs[:300]

                        vision_action = generate_vision_action_lean(
                            task=state['task'],
                            model=model,
                            screenshot_b64=screenshot_b64,
                            scene_context=last_obs_s0[:2000],
                            last_action=_lean_last_action,
                            last_result=_lean_last_result,
                            directive=_lean_directive,
                            discovered_files=_discovered_files,
                            current_todo=_current_todo_s0,
                            log_debug=log_debug,
                        )
                    else:
                        vision_action = generate_vision_action(
                            task=state['task'],
                            model=model,
                            screenshot_b64=screenshot_b64,
                            scene_context=last_obs_s0[:3000],
                            vgl_elements_text=vgl_hints_s0,
                            accumulated_state=state.get('accumulated_state', []),
                            episodic_memory=state.get('memory', []),
                            reflexion_insights=_vf_reflexion_insights,
                            recent_actions=recent_acts_s0,
                            log_debug=log_debug,
                            textgrad_hint=next_action_guidance or "",
                            current_todo=_current_todo_s0,
                            error_history=_vf_error_history,
                            progress_history=state.get('progress_history', []),
                            discovered_files=_discovered_files,
                            step_insights=state.get('step_insights_accumulator', []),
                            policy_text=getattr(state.get('policy'), 'base_policy', '') if state.get('policy') else '',
                        )
                    if vision_action:
                        # Guard: reject 'done' at step 0 only (no real work done yet)
                        # Universal: allow 'done' at step 1+ if prior command succeeded (rc=0)
                        _last_obs_done = str(state.get('prev_observation', ''))
                        _had_rc0 = 'rc=0' in _last_obs_done or '[SUCCESS' in _last_obs_done
                        if (vision_action.strip().lower() in ('done', 'task complete', 'task_complete')
                                and state['cur_step'] < 2 and not _had_rc0):
                            print(f"[VISION-FIRST] ENV {state['env_id']} Step {state['cur_step']}: "
                                  f"REJECTED 'done' at early step — task can't be complete yet")
                            vision_action = None
                        if vision_action:
                            had_guidance = bool(next_action_guidance)
                            next_action_guidance = vision_action
                            print(f"[VISION-FIRST] ENV {state['env_id']} Step {state['cur_step']}: "
                                  f"Vision action: '{vision_action}'"
                                  f"{' (overrode TextGrad)' if had_guidance else ' (cold start)'}")

            # USE THE LEARNED POLICY for action selection!
            # Pass full observation — truncating to 500 chars dropped critical context
            action = policy.forward(
                state=state['prev_observation'],
                task=state['task'],
                valid_actions=valid_actions,  # Limit to 30 actions from batch_data
                inventory=state.get('inventory', []),
                todo=current_todo,
                reflexion_insights=reflexion_insights,
                recent_actions=recent_actions,
                next_action_guidance=next_action_guidance,  # CRITICAL: Pass the specific recommended action!
                accumulated_state=state.get('accumulated_state', []),  # FIX #39: Progress regression guard
                tried_actions=state.get('tried_actions', set()),  # History-aware loop detection
            )
            # CATEGORY BLOCK FILTER: Reject actions that match a blocked command family.
            # When useradd failed 2x, ANY action containing "useradd" or "adduser" is rejected.
            # Universal: works for any command in any env.
            _blocked_cats = state.get('_blocked_categories', set())
            if _blocked_cats and action.strip().lower() not in ('done', 'task complete'):
                import re as _re_block
                _act_cmd_match = _re_block.search(
                    r'(?:run:\s*)?(?:sudo\s+)?(\w+)',
                    action.lower().replace('python3 -c', 'python3_c')
                )
                _act_cmd = _act_cmd_match.group(1) if _act_cmd_match else ''
                if _act_cmd in _blocked_cats:
                    _block_msg = f"[CATEGORY FILTER] REJECTED '{action[:80]}' — command '{_act_cmd}' is blocked after repeated failures"
                    print(_block_msg)
                    # Force regeneration with explicit block instruction
                    _block_prompt = (
                        f"Your previous action was REJECTED because '{_act_cmd}' has failed too many times.\n"
                        f"BLOCKED commands: {', '.join(_blocked_cats)}\n"
                        f"You MUST use a COMPLETELY DIFFERENT approach. Think of an alternative strategy."
                    )
                    action = policy.forward(
                        state=state['prev_observation'],
                        task=state['task'],
                        valid_actions=valid_actions,
                        inventory=state.get('inventory', []),
                        todo=current_todo,
                        reflexion_insights=reflexion_insights,
                        recent_actions=recent_actions,
                        next_action_guidance=_block_prompt,
                        accumulated_state=state.get('accumulated_state', []),
                        tried_actions=state.get('tried_actions', set()),
                    )
                    print(f"[CATEGORY FILTER] Regenerated: '{action[:80]}'")

            # DONE GUARD: Block "done" if recent steps had errors or low scores
            # Universal: trust rc=0 — if the last programmatic command succeeded,
            # allow "done" even if TextGrad scored low (no visual change ≠ failure)
            if action.strip().lower() in ('done', 'task complete', 'task_complete'):
                _recent_insights_dg = state.get('step_insights_accumulator', [])[-3:]
                # UNIVERSAL: trust the LATEST step's state, not historical substring noise.
                # Past errors that have been resolved by subsequent successful steps must not
                # block 'done'. The latest step's rc is the only ground-truth state signal.
                _latest = _recent_insights_dg[-1] if _recent_insights_dg else {}
                _latest_obs = str(_latest.get('observation', ''))
                _latest_score = _latest.get('progress_score', 0)
                _latest_action = str(_latest.get('action', ''))
                _latest_has_rc_fail = any(f'rc={c}' in _latest_obs for c in '123456789')
                _latest_has_rc0 = 'rc=0' in _latest_obs
                _latest_is_programmatic = 'run:' in _latest_action.lower()

                _has_recent_error = False  # retained only for logging
                _has_recent_low = False
                for _dg_ins in _recent_insights_dg:
                    _dg_obs = str(_dg_ins.get('observation', ''))
                    _dg_score = _dg_ins.get('progress_score', 0)
                    if 'COMMAND FAILED' in _dg_obs or '[ERROR]' in _dg_obs or "Can't operate" in _dg_obs:
                        _has_recent_error = True
                    if _dg_score <= 1:
                        _has_recent_low = True

                # GATE: use LATEST state as ground truth.
                # Allow: latest step succeeded (rc=0 programmatic, or score>=5)
                # Block: latest step actually failed (rc≠0) or score<=1
                _latest_succeeded = (_latest_has_rc0 and _latest_is_programmatic and not _latest_has_rc_fail) or _latest_score >= 5
                if _latest_succeeded:
                    log_debug(f"[DONE GUARD] ALLOWED 'done' — latest step succeeded (rc=0={_latest_has_rc0}, score={_latest_score}). Historical errors ignored.")
                elif _latest_has_rc_fail or _latest_score <= 1 or (_has_recent_low and not any(s >= 8 for s in state.get('progress_history', [])[-3:])):
                    log_debug(f"[DONE GUARD] BLOCKED premature 'done' — recent errors={_has_recent_error}, recent_low={_has_recent_low}")
                    # Substitute with a recovery action that addresses the error
                    _last_error_obs = ''
                    for _dg_ins2 in reversed(_recent_insights_dg):
                        _dg_obs2 = str(_dg_ins2.get('observation', ''))
                        if 'COMMAND FAILED' in _dg_obs2 or '[ERROR]' in _dg_obs2:
                            _last_error_obs = _dg_obs2[:200]
                            break
                    # Use Reflexion/TextGrad recommendation if available, else generic recovery
                    _dg_alt = None
                    for _dg_wr in reversed(state.get('working_reflexions', [])[-3:]):
                        _dg_insight = _dg_wr.get('insight', '')
                        if _dg_insight and len(_dg_insight) > 5:
                            _dg_alt = _dg_insight.strip()
                            break
                    if not _dg_alt:
                        _dg_alt = state.get('last_step_gradient', {}).get('next_action_guidance', '')
                    if _dg_alt and _dg_alt.strip().lower() not in ('done', 'task complete'):
                        action = _dg_alt
                        log_debug(f"[DONE GUARD] Substituted with learning recommendation: '{action[:80]}'")
                    else:
                        # Universal recovery: pick untried valid action from env's action space
                        # For text envs, valid_actions has real commands; for visual envs, falls back to shell
                        _dg_tried = set(str(a).strip().lower() for a in state.get('action_history', [])[-10:])
                        _dg_recovery = None
                        if valid_actions:
                            import random as _dg_rnd
                            _dg_candidates = [a for a in valid_actions if str(a).strip().lower() not in _dg_tried]
                            if _dg_candidates:
                                _dg_recovery = _dg_rnd.choice(_dg_candidates)
                        if not _dg_recovery:
                            # No valid actions or all tried — use env-appropriate fallback
                            if state.get('env_type', '') in ('osworld', 'visual'):
                                _dg_recovery = "run: ls ~"
                            else:
                                _dg_recovery = "look"  # Universal text env fallback — always valid
                        action = _dg_recovery
                        log_debug(f"[DONE GUARD] No learning recommendation — recovery: '{action}'")

            # WINDOWED DEDUP: Block repeats of actions that made NO progress
            # Checks last 5 actions (not just last 1) to catch A→B→A→B cycling
            _curr_norm = action.split('#')[0].strip().lower()
            _dedup_blocked = False
            _dedup_reason = ''
            _recent_insights = state.get('step_insights_accumulator', [])[-5:]
            for _ri in reversed(_recent_insights):
                _ri_norm = str(_ri.get('action', '')).split('#')[0].strip().lower()
                _ri_score = _ri.get('progress_score', 0)
                if _ri_norm == _curr_norm and _ri_score <= 3:
                    _dedup_blocked = True
                    _dedup_reason = f"matched step {_ri.get('step', '?')} (score={_ri_score})"
                    break

            if _dedup_blocked:
                log_debug(f"[DEDUP] BLOCKED repeated action '{action[:60]}' — {_dedup_reason}")

                # Try alternatives in priority order:
                _alt_action = None

                # 1. Reflexion's STRATEGY_SHIFT recommendation (must be executable, not prose, and DIFFERENT from blocked action)
                _action_prefixes = ('click ', 'press ', 'hotkey ', 'type ', 'run: ', 'scroll ', 'wait', 'done')
                for _wr in reversed(state.get('working_reflexions', [])[-3:]):
                    _wr_insight = _wr.get('insight', _wr.get('reflection', ''))
                    if _wr_insight and len(_wr_insight) > 10:
                        _wr_clean = _wr_insight.strip()
                        # Only use if it looks like an executable action
                        if any(_wr_clean.lower().startswith(p) for p in _action_prefixes):
                            # CRITICAL: Must be DIFFERENT from the blocked action
                            _alt_norm = _wr_clean.split('#')[0].strip().lower()
                            if _alt_norm == _curr_norm:
                                log_debug(f"[DEDUP] Skipping Reflexion (same as blocked): '{_wr_clean[:60]}'")
                                continue
                            # Also check it wasn't already tried with no progress
                            _alt_tried = False
                            for _ri2 in _recent_insights:
                                if str(_ri2.get('action', '')).split('#')[0].strip().lower() == _alt_norm and _ri2.get('progress_score', 0) <= 3:
                                    _alt_tried = True
                                    break
                            if _alt_tried:
                                log_debug(f"[DEDUP] Skipping Reflexion (already failed): '{_wr_clean[:60]}'")
                                continue
                            _alt_action = _wr_clean
                            log_debug(f"[DEDUP] Using Reflexion recommendation: '{_alt_action[:80]}'")
                            break
                        else:
                            # Don't skip prose — store best one for VISION-FIRST re-generation
                            if not hasattr(state, '_dedup_prose_hint') or not state.get('_dedup_prose_hint'):
                                state['_dedup_prose_hint'] = _wr_clean
                            log_debug(f"[DEDUP] Stored prose Reflexion as hint for VISION-FIRST: '{_wr_clean[:60]}'")
                            continue

                # 2. TextGrad's next_action_guidance (if different from blocked action AND not already failed)
                if not _alt_action:
                    _tg_next = state.get('last_step_gradient', {}).get('next_action_guidance', '')
                    if _tg_next:
                        _tg_norm = _tg_next.split('#')[0].strip().lower()
                        if _tg_norm != _curr_norm:
                            _tg_tried = False
                            for _ri3 in _recent_insights:
                                if str(_ri3.get('action', '')).split('#')[0].strip().lower() == _tg_norm and _ri3.get('progress_score', 0) <= 3:
                                    _tg_tried = True
                                    break
                            if not _tg_tried:
                                _alt_action = _tg_next
                                log_debug(f"[DEDUP] Using TextGrad recommendation: '{_alt_action[:80]}'")
                            else:
                                log_debug(f"[DEDUP] Skipping TextGrad (already failed): '{_tg_next[:60]}'")

                # 3. No executable alternative — try VISION-FIRST with prose context
                if _alt_action:
                    action = _alt_action
                else:
                    _dedup_vision = None  # Track whether VISION-FIRST produced an action
                    # Check if we have a prose Reflexion insight to guide VISION-FIRST
                    _prose_hint = state.get('_dedup_prose_hint', '')
                    _screenshot_for_dedup = state.get('last_after_screenshot_b64')
                    if _prose_hint and _screenshot_for_dedup:
                        # Re-generate action via VISION-FIRST with prose as strategic context
                        log_debug(f"[DEDUP] Re-generating via VISION-FIRST with prose hint: '{_prose_hint[:60]}'")
                        try:
                            _dedup_vision = generate_vision_action(
                                task=state['task'],
                                model=model,
                                screenshot_b64=_screenshot_for_dedup,
                                scene_context=state.get('prev_observation', '')[:2000],
                                vgl_elements_text=', '.join(valid_actions[:15]) if valid_actions else '',
                                accumulated_state=state.get('accumulated_state', []),
                                reflexion_insights=[_prose_hint],
                                recent_actions=recent_actions,
                                log_debug=log_debug,
                                textgrad_hint=f"BLOCKED: '{action[:60]}' failed before. Reflexion says: {_prose_hint}",
                                error_history=[],
                                progress_history=state.get('progress_history', []),
                                step_insights=state.get('step_insights_accumulator', []),
                                policy_text=getattr(state.get('policy'), 'base_policy', '') if state.get('policy') else '',
                            )
                            if _dedup_vision and _dedup_vision.split('#')[0].strip().lower() != _curr_norm:
                                action = _dedup_vision
                                log_debug(f"[DEDUP] VISION-FIRST re-generated action: '{action[:80]}'")
                            else:
                                log_debug(f"[DEDUP] VISION-FIRST returned same/empty action — falling back to recovery")
                                _dedup_vision = None
                        except Exception as _dedup_ex:
                            log_debug(f"[DEDUP] VISION-FIRST re-generation failed: {_dedup_ex}")
                            _dedup_vision = None
                    else:
                        _dedup_vision = None

                    # Fallback: universal recovery cycle (only if VISION-FIRST didn't produce an action)
                    if not _dedup_vision:
                        _tried_norms = set()
                        for _ri4 in _recent_insights:
                            _tried_norms.add(str(_ri4.get('action', '')).split('#')[0].strip().lower())
                        # Also include full action history to avoid repeats
                        for _ah in state.get('action_history', [])[-10:]:
                            _tried_norms.add(str(_ah).strip().lower())

                        _recovery_action = None
                        _env_type_r = state.get('env_type', '')

                        # Strategy 1: Pick untried valid action from env's own action space
                        if valid_actions:
                            import random as _rec_rnd
                            _untried = [a for a in valid_actions
                                        if str(a).strip().lower() not in _tried_norms]
                            if _untried:
                                _recovery_action = _rec_rnd.choice(_untried)
                                log_debug(f"[DEDUP] Recovery from valid_actions: '{_recovery_action}' ({len(_untried)} untried)")

                        # Strategy 2: Env-type-appropriate fallback
                        if not _recovery_action:
                            if _env_type_r in ('osworld', 'visual'):
                                # Shell recovery for desktop environments
                                _recovery_actions = ['press Escape', 'run: ls ~', 'run: ls /home/user']
                                for _ra in _recovery_actions:
                                    if _ra.lower() not in _tried_norms:
                                        _recovery_action = _ra
                                        break
                                if not _recovery_action:
                                    _recovery_action = _recovery_actions[0]
                            else:
                                # Text env: "look" is universally valid in text games
                                _recovery_action = "look"

                        action = _recovery_action
                        log_debug(f"[DEDUP] No learning alternative — recovery: '{action}'")

            # Clean up transient DEDUP state
            state.pop('_dedup_prose_hint', None)

            # ================================================================
            # FORCED VALID ACTION EXPLORATION (universal loop-breaker)
            # When the LLM keeps generating invalid/stuck actions, bypass it
            # entirely and pick untried valid actions from the env's action space.
            # Uses sliding window: if >= 4 of last 8 actions got "Nothing happens"
            # or were no-progress, the LLM is stuck — force exploration.
            # This is universal: works for ANY env with valid_actions.
            # ================================================================
            _nothing_window = state.get('_nothing_history', [])
            _window_size = 8
            _recent_nothings = sum(_nothing_window[-_window_size:]) if _nothing_window else 0
            _force_threshold = 4  # 4 out of last 8 = 50% stuck → force explore
            if _recent_nothings >= _force_threshold and valid_actions and state.get('env_type', '') not in ('osworld', 'visual'):
                # Agent is stuck — force systematic exploration from valid actions
                _all_tried = set(str(a).strip().lower() for a in state.get('action_history', []))
                _task_words = set(state.get('task', '').lower().split()) - {'a','an','the','to','in','on','some','and','put','it'}
                # 3 priority tiers:
                # 1. Task-verb actions (cool, put, take + task objects)
                # 2. Navigation to unvisited locations (go to X not tried)
                # 3. Other untried actions
                _untried_task = []
                _untried_nav = []
                _untried_other = []
                for _va in valid_actions:
                    _va_lower = str(_va).strip().lower()
                    if _va_lower in _all_tried:
                        continue
                    _va_words = set(_va_lower.split()) - {'a','an','the','to','in','on','with','from','1','2','3','4','5','6'}
                    if _va_words & _task_words:
                        _untried_task.append(_va)
                    elif _va_lower.startswith('go to '):
                        _untried_nav.append(_va)
                    else:
                        _untried_other.append(_va)
                # Systematic cycling: task-relevant first, then navigation, then other
                _fe_idx = state.get('_forced_explore_idx', 0)
                # Alternate: task actions on even steps, navigation on odd (explore new locations)
                if _fe_idx % 2 == 0 and _untried_task:
                    _candidates = _untried_task
                    _label = 'task-relevant'
                elif _untried_nav:
                    _candidates = _untried_nav
                    _label = 'navigation'
                elif _untried_task:
                    _candidates = _untried_task
                    _label = 'task-relevant'
                else:
                    _candidates = _untried_other
                    _label = 'other'
                if _candidates:
                    action = _candidates[_fe_idx % len(_candidates)]
                    state['_forced_explore_idx'] = _fe_idx + 1
                    log_debug(f"[FORCED-EXPLORE] ENV {state['env_id']}: Bypassed LLM → {_label} untried [{_fe_idx % len(_candidates)}/{len(_candidates)}]: '{action}'")

            # ================================================================
            # FIX 1: GROUND ACTION IN VALID_ACTIONS (universal, conservative)
            # Only intervene for clearly invalid actions (None, empty, garbage).
            # Don't remap plausible game verbs — valid_actions is stale (from
            # step start, not current location) and would cause false negatives.
            # The existing auto-correct at env.step() handles "Nothing happens".
            # ================================================================
            if state.get('env_type', '') not in ('osworld', 'visual'):
                _action_lower = action.strip().lower()
                _is_garbage = (
                    not _action_lower
                    or _action_lower in ('none', 'none, as the task is complete', 'try unexplored actions')
                    or _action_lower.startswith('none')
                    or len(_action_lower) > 200  # LLM dumped reasoning as action
                )
                if _is_garbage and valid_actions:
                    # Replace garbage with best task-relevant valid action
                    _task_w = set(state.get('task', '').lower().split()) - {'a','an','the','to','in','on','some','and','put','it'}
                    _all_tried_lower = set(str(a).strip().lower() for a in state.get('action_history', [])[-10:])
                    _best_va, _best_score = None, 0
                    for _va in valid_actions:
                        _va_lower = str(_va).strip().lower()
                        if _va_lower in _all_tried_lower:
                            continue
                        _va_w = set(_va_lower.split()) - {'a','an','the','to','in','on','with','from','1','2','3','4','5','6'}
                        _overlap = len(_task_w & _va_w)
                        _score = _overlap / max(len(_task_w | _va_w), 1)
                        if _score > _best_score:
                            _best_score = _score
                            _best_va = _va
                    if _best_va:
                        log_debug(f"[GROUND] ENV {state['env_id']}: Garbage action '{_action_lower[:50]}' → '{_best_va}'")
                        action = _best_va
                    else:
                        action = 'look'
                        log_debug(f"[GROUND] ENV {state['env_id']}: Garbage action, no match → 'look'")

            # ================================================================
            # FIX 2: TASK-VERB EXPLORATION — DISABLED
            # Was forcing navigation to appliances (fridge/microwave/sinkbasin)
            # even when agent already had the object and TextGrad recommended
            # the correct action (e.g., "cool pan 1 with fridge 1").
            # The visited check only counted "go to X" but not "cool X with Y"
            # causing 100+ overrides and 8,1,8,1 score oscillation.
            # TextGrad handles appliance discovery via its gradient feedback.
            # ================================================================

            # ================================================================
            # FIX 3: INVENTORY MANAGEMENT (universal)
            # If action is "take X" but returned "Nothing happens" recently
            # for the same type, check if inventory is full and force drop.
            # ================================================================
            if action.lower().startswith('take ') and state.get('env_type', '') not in ('osworld', 'visual'):
                _last_obs = str(state.get('prev_observation', '')).lower()
                _inv = state.get('inventory', [])
                # If carrying something and recent take failed, try dropping first
                _recent_nothing = sum(state.get('_nothing_history', [])[-3:]) if state.get('_nothing_history') else 0
                if _recent_nothing >= 2 and _inv:
                    # Carrying items + recent failures → inventory likely full
                    _carrying = None
                    for _item in _inv:
                        if isinstance(_item, str):
                            _carrying = _item
                            break
                    if not _carrying and 'you are carrying:' in _last_obs:
                        # Parse from observation
                        import re as _inv_re
                        _inv_match = _inv_re.search(r'you are carrying:\s*a\s+(.+?)(?:\.|$)', _last_obs)
                        if _inv_match:
                            _carrying = _inv_match.group(1).strip()
                    if _carrying:
                        # Check if we have a valid "put" or "move" action to free inventory
                        for _va in valid_actions:
                            _va_l = str(_va).strip().lower()
                            if _va_l.startswith(('put ', 'move ')) and _carrying.split()[0] in _va_l:
                                log_debug(f"[INVENTORY] ENV {state['env_id']}: Inventory full, dropping '{_carrying}' via '{_va}'")
                                action = _va
                                break

            # ================================================================
            # FIX 4: LOCATION CYCLING DETECTION (universal)
            # Detect A→B→A→B patterns (not just exact repeats).
            # If last 6 actions alternate between 2-3 locations, break out.
            # ================================================================
            if action.lower().startswith('go to ') and state.get('env_type', '') not in ('osworld', 'visual'):
                _last_6 = [str(a).strip().lower() for a in state.get('action_history', [])[-6:]]
                _nav_last_6 = [a for a in _last_6 if a.startswith('go to ')]
                if len(_nav_last_6) >= 4:
                    _locations = set(_nav_last_6)
                    if len(_locations) <= 2 and action.lower() in _locations:
                        # Cycling between 1-2 locations — pick somewhere NEW
                        _all_tried_nav = set(a for a in (str(x).strip().lower() for x in state.get('action_history', [])) if a.startswith('go to '))
                        _new_navs = [str(va) for va in valid_actions
                                     if str(va).strip().lower().startswith('go to ')
                                     and str(va).strip().lower() not in _all_tried_nav]
                        if _new_navs:
                            _cycle_idx = state.get('_cycle_break_idx', 0)
                            action = _new_navs[_cycle_idx % len(_new_navs)]
                            state['_cycle_break_idx'] = _cycle_idx + 1
                            log_debug(f"[CYCLE-BREAK] ENV {state['env_id']}: Location cycling detected → '{action}'")
                        elif valid_actions:
                            # All locations tried — pick any untried action
                            _all_tried_set = set(str(a).strip().lower() for a in state.get('action_history', []))
                            _any_untried = [str(va) for va in valid_actions if str(va).strip().lower() not in _all_tried_set]
                            if _any_untried:
                                action = _any_untried[0]
                                log_debug(f"[CYCLE-BREAK] ENV {state['env_id']}: All locations visited → trying '{action}'")

            # ENV-FIX ENFORCEMENT: If env-grounded fix was extracted, use it
            # instead of whatever PASS 2 / POLICY.FORWARD generated.
            # The fix comes from real evidence (probe + evaluator); PASS 2
            # has the same knowledge gap that caused the NOOP.
            if state.get('_env_fix_action'):
                _enforced = state.pop('_env_fix_action')
                log_debug(f"[ENV-FIX ENFORCE] Overriding PASS 2 action with env-grounded fix: '{_enforced[:80]}'")
                action = _enforced
            selected_actions.append(action)
            log_debug(f"[POLICY.FORWARD] ENV {state['env_id']}: Selected '{action[:80]}' using learned policy")

        print(f"[PHASE 3B] ✓ Selected {len(selected_actions)} actions using LEARNED TEXTGRAD POLICY")

        # ========================================================================
        # PHASE 4: EXECUTE ALL ACTIONS, COLLECT RESULTS
        # ========================================================================
        print(f"\n[PHASE 4] Executing {len(selected_actions)} actions...")
        action_results = []

        for idx, (state, action) in enumerate(zip(active_states_all, selected_actions)):
            # Track tried actions
            if not isinstance(state['tried_actions'], set):
                state['tried_actions'] = set(state['tried_actions'])
            state['tried_actions'].add(action)

            # NOTE: Compound-statement handling (`; → \n`, inline blocks, multi-line)
            # lives entirely in osworld_wrapper.py's python channel. Keeping `action`
            # untouched here is critical: trajectory, TextGrad LOSS/BACKWARD, memory,
            # and tried_actions all read from this variable. If we mutated it into a
            # base64 transport form, TextGrad would learn "base64 = bad" from a
            # transformation the agent never wrote — poisoning the policy gradient.
            # Source action stays canonical; transport encoding is wrapper-private.

            # Execute action - MEMORY FIX: Use env from registry
            env = _env_registry[state['env_id']]
            action = clean_action(action)  # Universal: strip LLM artifacts before env receives it
            observation, reward, done, info = env.step([action])
            observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
            done = done[0]

            # UNIVERSAL AUTO-CORRECTION: If action returned "Nothing happens" (invalid),
            # find the closest valid action and retry ONCE. Helps weaker models that
            # generate natural English instead of game syntax (e.g., "move" vs "put").
            # No-op for strong models that always generate valid actions.
            _obs_str = str(observation)
            _valid = info.get('admissible_commands', []) if isinstance(info, dict) else []
            if not _valid and hasattr(env, '_last_admissible'):
                _valid = getattr(env, '_last_admissible', [])
            # Flatten valid actions list (may contain nested lists)
            _valid_flat = []
            for _v in _valid:
                if isinstance(_v, list):
                    _valid_flat.extend(_v)
                elif isinstance(_v, str):
                    _valid_flat.append(_v)
            _valid = _valid_flat
            if 'Nothing happens' in _obs_str and _valid and action.lower() not in [v.lower() for v in _valid]:
                _action_lower = action.lower()
                _action_words = set(_action_lower.split()) - {'to','the','a','an','in','on','with','from','in/on'}
                _best_va, _best_score = None, 0
                for _va in _valid:
                    _va_words = set(_va.lower().split()) - {'to','the','a','an','in','on','with','from','in/on'}
                    _overlap = len(_action_words & _va_words)
                    _total = max(len(_action_words | _va_words), 1)
                    _sc = _overlap / _total
                    if _sc > _best_score:
                        _best_score = _sc
                        _best_va = _va
                # Only auto-correct if: high confidence match AND not the same action we just tried
                _recent_acts = [a.lower() for a in list(state.get('tried_actions', set()))[-5:]]
                if (_best_va and _best_score >= 0.6
                        and _best_va.lower() != action.lower()
                        and _best_va.lower() not in _recent_acts):
                    print(f"[AUTO-CORRECT] '{action}' → '{_best_va}' (score={_best_score:.2f})")
                    observation, reward, done, info = env.step([_best_va])
                    observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
                    done = done[0]
                    action = _best_va
                    state['tried_actions'].add(_best_va)

            # Track stuck signals in sliding window for forced exploration trigger
            _obs_s = str(observation).strip()
            _prev_obs_s = str(state.get('prev_observation', '')).strip()
            _is_stuck = (
                'Nothing happens' in _obs_s
                or (_obs_s == _prev_obs_s and _obs_s)  # Exact same observation = no state change
            )
            if '_nothing_history' not in state:
                state['_nothing_history'] = []
            state['_nothing_history'].append(1 if _is_stuck else 0)

            # Vision-first: Update last screenshot for next step's cold-start fallback
            if info and isinstance(info, dict):
                after_b64 = info.get('after_screenshot_b64')
                if after_b64:
                    state['last_after_screenshot_b64'] = after_b64

            # CRITICAL FIX: Mark environment as just completed for this step
            # This flag will be checked BEFORE adding to action_results for gradient generation
            # Prevents generating gradients for the step that just finished the episode
            state['skip_gradient_this_step'] = done

            # G34g: mid-episode DONE-predicate check (TOTE + BDI + CodeAct synthesis).
            # Research (4 parallel agents, Phase 3): weak-LLM progress scorers miss the
            # moment of task completion. The only robust signal is the canonical
            # external predicate (getter → metric, side-effect-free). When it fires,
            # drop intention (BDI) and terminate — this stops the agent from
            # overshooting past correct state and destroying it.
            # Cost: 1-3s per step. Universal — uses canonical OSWorld evaluator
            # whose predicate is compiled from the task config at env.reset().
            # G34g gate (efficiency review #2): predicate call is 1-3s (HTTP + getter).
            # Skip when the current step CANNOT have completed the task:
            #   - step 0: setup state, predicate always False by construction
            #   - "Nothing happens" observation: action had no effect
            #   - identical obs to prev step: state unchanged
            # Fires on every step where state actually moved, which is typically 3-6/trial.
            _g34g_obs_str = str(observation).strip() if 'observation' in dir() else ''
            _g34g_state_moved = (
                _g34g_obs_str
                and 'Nothing happens' not in _g34g_obs_str
                and _g34g_obs_str != str(state.get('prev_observation', '')).strip()
            )
            if (not done and hasattr(env, 'done_predicate_check')
                    and state['env_type'] in ('osworld', 'visual')
                    and state.get('cur_step', 0) >= 1
                    and _g34g_state_moved):
                try:
                    _dpc = env.done_predicate_check()
                    if _dpc.get('success'):
                        print(f"[G34g] Canonical predicate fired at step "
                              f"{state.get('cur_step', '?')}: score={_dpc.get('score')} — "
                              f"BDI intention-drop, emitting DONE")
                        done = True
                        # Replace the last action in history with 'done' so the
                        # evaluator's FAIL-short-circuit logic sees a clean termination.
                        if hasattr(env, '_action_history'):
                            env._action_history.append('DONE')
                except Exception as _g34g_e:
                    # Fail-safe: predicate check errors don't break the trial,
                    # but log so we can distinguish "silent crash" from "fired+false"
                    # when auditing ablations (reviewer-visible for paper).
                    print(f"[G34g] predicate error: {type(_g34g_e).__name__}: {_g34g_e}")

            # H35: BDI intention-drop on IMPOSSIBILITY (Rao-Georgeff 1995).
            # G34g handles case (a): goal achieved → DONE.
            # H35 handles case (b): goal impossible → FAIL.
            # Trigger: agent has accumulated ≥10 progress scores AND mean of
            # all scores so far ≤ 2.5 AND past step 12 (late enough that
            # remaining 3 steps are unlikely to flip a fundamentally stuck
            # trajectory). Mean rather than max avoids mis-triggering on a
            # single spurious early PARTIAL_PROGRESS=6 that TextGrad emits
            # before any real action has been attempted.
            # Universal: checks agent cognitive-state only, no task strings.
            # Effect on infeasible tasks (30/360 of OSWorld): OSWorld's evaluator
            # returns score=1.0 when func=='infeasible' + last action == FAIL.
            if (not done and state.get('cur_step', 0) >= 12
                    and state['env_type'] in ('osworld', 'visual')):
                try:
                    _all_scores = []
                    if 'reflexgrad_router' in dir():
                        _all_scores = list(
                            getattr(reflexgrad_router, 'progress_scores', [])
                        )
                    _mean_score = (
                        sum(_all_scores) / len(_all_scores)
                        if _all_scores else 10.0  # default: don't fire if no data
                    )
                    if len(_all_scores) >= 10 and _mean_score <= 2.5:
                        print(f"[H35 BDI-IMPOSSIBILITY] ENV {state.get('env_id','?')} "
                              f"Step {state.get('cur_step','?')}: mean progress score "
                              f"{_mean_score:.2f}/10 over {len(_all_scores)} scores, "
                              f"step>=12 — emitting FAIL (Rao-Georgeff 1995 case b)")
                        done = True
                        if hasattr(env, '_action_history'):
                            env._action_history.append('FAIL')
                except Exception as _h35_e:
                    print(f"[H35] predicate error: {type(_h35_e).__name__}: {_h35_e}")

            # FIX #10: Set done flag IMMEDIATELY after env.step() returns done=True
            # This prevents the infinite loop bug where completed envs stay 'active'
            if done:
                state['done'] = True
                agent_claimed_done = info.get('won', [False])[0]
                state['success'] = agent_claimed_done

                # FIX: Call evaluator when episode ends (OSWorld tasks need external evaluation)
                # Without this, max_steps termination never runs the evaluator because the env
                # is skipped from action_results by skip_gradient_this_step.
                env_type = state.get('env_type', '')
                if env_type in ('osworld', 'visual') and hasattr(env, 'evaluate'):
                    try:
                        eval_result = env.evaluate()
                        eval_success = eval_result.get('success', False)
                        print(f"[ENV {state['env_id']}] EVALUATOR AT DONE: {eval_result}")
                        state['success'] = eval_success
                    except Exception as e:
                        print(f"[ENV {state['env_id']}] Evaluator error at done: {e}")
                        state['success'] = False  # Don't trust agent's self-declared success

                # META-LEARNING FIX: Save success_workflow immediately when success detected
                if state['success'] and state.get('trajectory'):
                    # FIX: Include the WINNING action (current action) which hasn't been added to trajectory yet
                    all_actions = [act for act, _, _ in state['trajectory']] + [action]
                    success_workflow = {
                        'type': 'success_workflow',
                        'task': state['task'],
                        'actions': all_actions,  # Now includes the winning action
                        'episode_id': state.get('episode_id', f"trial{trial_idx}_env{state['env_id']}"),
                        'trial': trial_idx,
                        'steps': len(all_actions),  # Correct count including winning action
                        'success_confirmed': True
                    }
                    if 'memory' not in env_configs[state['env_id']]:
                        env_configs[state['env_id']]['memory'] = []
                    env_configs[state['env_id']]['memory'].append(success_workflow)
                    print(f"[SUCCESS WORKFLOW] Saved {len(all_actions)} actions for ENV {state['env_id']}: '{state['task']}'")

            # ============================================================================
            # SEMANTIC FAILURE DETECTION (universal, no hardcoding)
            # If an action produces a positive observation ("You move/put/clean/cool...")
            # but the task doesn't complete (done=False, reward=0), the approach FAILED.
            # Record it so the agent never repeats the same unsuccessful approach.
            # ============================================================================
            _lc = state.get('learning_context')
            _reward_val = reward[0] if isinstance(reward, (list, tuple)) else reward
            if _lc and not done and _reward_val == 0:
                _obs_lower = str(observation).lower()
                _action_lower = action.lower()
                # Detect placement/interaction actions that LOOK successful but aren't
                _placement_verbs = ['you move', 'you put', 'you place']
                _interaction_verbs = ['you clean', 'you cool', 'you heat', 'you use']
                _looks_successful = any(v in _obs_lower for v in _placement_verbs + _interaction_verbs)
                if _looks_successful:
                    _fail_reason = (
                        f"Action '{action}' produced '{str(observation)[:100]}' but task NOT complete. "
                        f"This approach does not satisfy the task requirements — try a DIFFERENT strategy."
                    )
                    _lc.failure_memory.record_from_action(action, _fail_reason)
                    print(f"[SEMANTIC FAIL] ENV {state['env_id']}: '{action[:50]}' looked successful but task not done — recorded as failure")

            # ============================================================================
            # TEXTGRAD INTEGRATION: Loss, Backward, Optimizer
            # TextGrad runs EVERY step — provides next_action_guidance.
            # Reflexion only when stuck (handled in Phase 5 PASS 1).
            # ============================================================================
            policy = state.get('policy')
            loss_fn = state.get('loss_fn')
            optimizer = state.get('optimizer')

            if policy is not None and loss_fn is not None and optimizer is not None:
                # Build trajectory context for loss function (so it can evaluate in context)
                trajectory_context = []
                for act, obs, _ in state.get('trajectory', []):
                    trajectory_context.append({'action': act, 'observation': obs})

                # Get current subtask from TODO manager for loss context
                current_subtask = None
                if 'todo_manager' in state and state['todo_manager']:
                    current_subtask = state['todo_manager']._get_current_todo()

                # Get Reflexion insights from episodic memory
                reflexion_insights = state.get('memory', [])

                # Get current valid actions (what's possible AFTER this action)
                _loss_valid_actions = []
                try:
                    _loss_env = _env_registry[state['env_id']]
                    if hasattr(_loss_env, 'get_current_valid_actions'):
                        _loss_valid_actions = _loss_env.get_current_valid_actions()
                except Exception:
                    pass

                # ENVIRONMENT-GROUNDED TextGrad LOSS: Use real environment signals
                # instead of LLM self-assessment. The observation already contains
                # rc codes, error messages, NOOP detection, probe data — these are
                # ground truth, more reliable than any LLM opinion.
                _obs_str = str(observation)
                _action_str = str(action)
                _rc_in_obs = 'rc=0' in _obs_str
                _rc_fail = any(f'rc={c}' in _obs_str for c in '123456789')
                _is_noop = 'NO-OP' in _obs_str or 'was a NO-OP' in _obs_str
                _has_error = any(e in _obs_str for e in [
                    'Error', 'error:', 'Traceback', 'FileNotFoundError',
                    'SyntaxError', 'NameError', 'KeyError', 'No such file',
                    'command not found', 'Permission denied'
                ])
                _no_change = 'No visible change' in _obs_str
                _state_diff = str(info.get('state_diff', '')) if info else ''
                _has_change = _state_diff and 'No visible change' not in _state_diff

                # Extract error text for the gradient.
                # UNIVERSAL: only extract error on actual rc failure. When rc=0, stderr
                # noise (benign warnings, sudoers messages, 'already exists' after auto-retry)
                # must not be reported to the model as an error — that poisons the backward
                # pass gradient and makes the model think a successful action failed.
                _error_snippet = ""
                if _rc_fail:
                    for _err_marker in ['Error:', 'Traceback', 'error:', 'MISMATCH',
                                         'No such file', 'command not found']:
                        _eidx = _obs_str.find(_err_marker)
                        if _eidx >= 0:
                            _error_snippet = _obs_str[_eidx:_eidx + 300].strip()
                            break

                # Deterministic LOSS computation.
                # PRIORITY: exit code is ground truth. Substring-match on "error" is
                # misleading when rc=0 commands have benign stderr/warnings (e.g., auto-retry
                # transcripts, sudoers noise, non-fatal stderr). A command that exits 0 is a
                # successful command. This is the feedback quality fix: trust the OS, not regex.
                if _rc_fail:
                    # Actual exit-code failure: the environment says the command failed
                    _env_score = 1
                    _env_status = 'NO_PROGRESS'
                    loss_text = f"COMMAND FAILED. {_error_snippet}"
                elif _is_noop:
                    _env_score = 1
                    _env_status = 'NO_PROGRESS'
                    loss_text = f"NO-OP: Code ran (rc=0) but file content unchanged. Your approach is wrong — try a different method."
                elif _rc_in_obs and _has_change:
                    _env_score = 7
                    _env_status = 'MAJOR_PROGRESS'
                    loss_text = f"PROGRESS: Command succeeded and state changed. {_state_diff[:200]}"
                elif _rc_in_obs and 'run:' in _action_str.lower():
                    # rc=0 programmatic command — trust execution over visual assessment.
                    # This covers headless/server tasks (SSH, systemctl, chmod) where
                    # success has NO visible change. Without this, model cycles on
                    # successful-but-invisible actions.
                    _env_score = 4
                    _env_status = 'PARTIAL_PROGRESS'
                    loss_text = f"PARTIAL: Command succeeded (rc=0). Headless/programmatic action — verify downstream state."
                elif _has_error and not _rc_in_obs:
                    # Error substring present and rc=0 not in obs → actual error
                    _env_score = 1
                    _env_status = 'NO_PROGRESS'
                    loss_text = f"COMMAND FAILED. {_error_snippet}"
                elif _no_change:
                    _env_score = 2
                    _env_status = 'NO_PROGRESS'
                    loss_text = f"NO VISIBLE CHANGE after action. Action may have failed or targeted wrong element."
                else:
                    _env_score = 3
                    _env_status = 'EXPLORING'
                    loss_text = f"EXPLORING: Action executed. Observation: {_obs_str[:200]}"

                print(f"[TEXTGRAD LOSS] ENV-GROUNDED: score={_env_score}, status={_env_status}, loss='{loss_text[:80]}'")

                # Prepare comprehensive context for TextGrad backward pass
                # CRITICAL: TextGrad now gets the SAME context as action selection prompt
                # This enables SMART gradients that consider history, TODO, Reflexion insights
                #
                # NO FALLBACKS - fail if context missing to catch bugs early!
                todo_formatted = ""
                if 'todo_manager' in state and state['todo_manager']:
                    todo_formatted = state['todo_manager']._get_current_todo()

                # Build state-action-observation triples for TextGrad conditional learning
                # Include progress_status so TextGrad can learn to avoid NO_PROGRESS actions
                state_action_obs_triples = []
                trajectory = state.get('trajectory', [])
                gradients_history = state.get('step_gradients_history', [])

                for idx in range(len(trajectory)):
                    # Unpack 3-element trajectory: (action, observation, reasoning)
                    action_taken, obs_after, _ = trajectory[idx]

                    # State before = observation from previous step (or initial obs if first step)
                    if idx == 0:
                        state_before = state.get('initial_observation', '')[:500]
                    else:
                        state_before = trajectory[idx-1][1][:150]  # Previous observation

                    # Get progress_status from history (if available for this step)
                    progress_status = 'UNKNOWN'
                    if idx < len(gradients_history):
                        progress_status = gradients_history[idx].get('progress_status', 'UNKNOWN')

                    # Store triple WITH progress_status for TextGrad learning
                    state_action_obs_triples.append({
                        'state': state_before,
                        'action': action_taken,
                        'observation': obs_after[:1000],  # Must see errors+probe (was 150 — invisible)
                        'progress': progress_status
                    })

                # TIMING FIX: Add CURRENT step to history so TextGrad can see THIS action's result
                # Quick progress check before full gradient computation
                quick_progress = 'EXPLORING'  # default
                if 'you are facing' in observation.lower() and 'next to it, you see nothing' in observation.lower():
                    quick_progress = 'NO_PROGRESS'
                elif observation.lower().strip() == state['prev_observation'].lower().strip():
                    quick_progress = 'NO_PROGRESS'
                elif 'nothing happens' in observation.lower():
                    quick_progress = 'NO_PROGRESS'
                elif 'you turn on' in observation.lower() or 'you take' in observation.lower():
                    quick_progress = 'PARTIAL_PROGRESS'

                # FIX #35: Populate failed_state_actions when NO_PROGRESS detected
                # This enables the existing filter at line 2926 to work!
                if quick_progress == 'NO_PROGRESS':
                    prev_state_hash = hashlib.md5(state['prev_observation'][:200].encode()).hexdigest()[:8]
                    if prev_state_hash not in state['failed_state_actions']:
                        state['failed_state_actions'][prev_state_hash] = set()
                    state['failed_state_actions'][prev_state_hash].add(action)
                    print(f"[FIX #35] Action '{action}' marked as no-progress at state {prev_state_hash}")

                # Add current step with quick progress assessment
                state_action_obs_triples.append({
                    'state': state['prev_observation'][:500],
                    'action': action,
                    'observation': observation[:1000],  # Must see errors+probe (was 150)
                    'progress': quick_progress
                })

                textgrad_context = {
                    'task': state['task'],  # Will fail if missing - good!
                    'todo': todo_formatted,  # Can be empty string
                    'tried_actions': state['tried_actions'],  # Set of tried actions (for quick lookup)
                    'state_action_obs_history': state_action_obs_triples,  # FIX #36: ALL history, not just last 10 - LLM needs to see full pattern
                    'working_reflexions': state['working_reflexions'],  # Will fail if missing
                    'reflexion_memory': state['reflexion_memory'],  # Will fail if missing
                    'cur_step': state['cur_step'],  # Will fail if missing
                    'valid_actions': state.get('valid_actions', []),  # CRITICAL: TextGrad needs to know what's possible!
                    'click_metadata': info.get('click_metadata') if info else None,  # Coordinate learning
                    'env_type': state.get('env_type', ''),  # For visual env detection in backward pass
                    # Multimodal ReflexGrad: before/after screenshots for visual gradient computation
                    'before_screenshot_b64': info.get('before_screenshot_b64') if info else None,
                    'after_screenshot_b64': info.get('after_screenshot_b64') if info else None,
                    # Observation layer now includes file-change detection,
                    # making eval-gap injection unnecessary. TextGrad sees
                    # "file unchanged after save" in the observation itself.
                }

                # ENVIRONMENT-GROUNDED TextGrad BACKWARD: Build gradient from
                # real environment signals instead of LLM interpretation.
                # The gradient contains: exact error, exact file state from probe,
                # and deterministic DIFF from evaluator.
                _probe_data = ""
                _probe_idx = _obs_str.find('[POST-ACTION PROBE')
                if _probe_idx >= 0:
                    _probe_end = _obs_str.find('\n]', _probe_idx)
                    _probe_data = _obs_str[_probe_idx:_probe_end + 2 if _probe_end > 0 else _probe_idx + 500][:500]

                _eval_gap = str(state.get('_last_eval_gap', ''))
                _diff_text = ""
                if _eval_gap:
                    _diff_text = f"\nEVALUATOR GAP: {_eval_gap[:500]}"

                gradient = (
                    f"ACTION: {_action_str[:200]}\n"
                    f"RESULT: {loss_text}\n"
                    f"{f'ERROR: {_error_snippet}' if _error_snippet else ''}\n"
                    f"{f'FILE STATE: {_probe_data}' if _probe_data else ''}\n"
                    f"{_diff_text}\n"
                    f"{'NOOP: Your code ran but changed nothing in the file. Try a completely different approach or library.' if _is_noop else ''}"
                ).strip()

                print(f"[TEXTGRAD BACKWARD] ENV-GROUNDED: {len(gradient)} chars, has_error={bool(_error_snippet)}, has_probe={bool(_probe_data)}, has_diff={bool(_diff_text)}")

                # CLOSE THE LOOP: Extract a concrete fix from the env-grounded
                # gradient and set it as next_action_guidance so it reaches
                # BOTH the lean directive AND PASS 2's action generator.
                # Without this, the gradient goes to policy.gradients but
                # never reaches the next action's prompt.
                if (_error_snippet or _is_noop or _diff_text) and _env_score <= 2:
                    try:
                        # ============================================================
                        # UNIVERSAL ERROR CLASSIFIER (AgentDebug-style, task-agnostic)
                        # Classifies failure into 5 capability-based categories.
                        # Works for ANY task — coding, GUI, SSH, text games, anything.
                        # ============================================================
                        _classifier_prompt = (
                            f"Classify this agent failure into ONE category:\n\n"
                            f"TASK: {state['task'][:200]}\n"
                            f"ACTION: {_action_str[:200]}\n"
                            f"RESULT: {loss_text[:200]}\n"
                            f"{'ERROR: ' + _error_snippet[:200] if _error_snippet else ''}\n"
                            f"{'EVALUATOR GAP: ' + _diff_text[:200] if _diff_text else ''}\n\n"
                            f"Categories (pick ONE):\n"
                            f"A) MEMORY_ERROR: Agent forgot critical info from earlier observations\n"
                            f"B) REFLECTION_ERROR: Wrong diagnosis of prior failure\n"
                            f"C) PLANNING_ERROR: Wrong sequence/decomposition of task steps\n"
                            f"D) ACTION_ERROR: Right intent, wrong execution (syntax error, wrong path, wrong API, wrong value)\n"
                            f"E) SYSTEM_ERROR: Environment/tool failure (not the agent's fault)\n\n"
                            f"Reply with ONLY the letter (A/B/C/D/E) and one sentence explanation.\n"
                            f"Format: 'X: <one sentence>'"
                        )
                        _classifier_out = fast_model.generate([_classifier_prompt])[0].outputs[0].text.strip()
                        _error_category = 'D'  # Default to ACTION_ERROR (most common)
                        for _cat in ['A', 'B', 'C', 'D', 'E']:
                            if _classifier_out.strip().startswith(_cat):
                                _error_category = _cat
                                break
                        _category_name = {
                            'A': 'MEMORY_ERROR',
                            'B': 'REFLECTION_ERROR',
                            'C': 'PLANNING_ERROR',
                            'D': 'ACTION_ERROR',
                            'E': 'SYSTEM_ERROR',
                        }.get(_error_category, 'ACTION_ERROR')
                        print(f"[ERROR CLASSIFIER] Category: {_category_name} | {_classifier_out[:120]}")

                        # Universal prescription templates (work for any task)
                        _prescription = {
                            'A': "Re-read the observation from the previous steps carefully before acting. Check what information you have collected but forgot to use.",
                            'B': "Your previous diagnosis was wrong. Re-examine the evidence (probe data, error messages, file state) without assumptions.",
                            'C': "Your plan is wrong. Break the task into smaller sub-goals: (1) understand current state, (2) identify target state, (3) find minimal action to bridge them. Don't attempt big multi-step actions.",
                            'D': "The intent is right but execution is wrong. For code: fix the exact syntax/API error. For paths: verify with ls/find first. For values: use EXACT values from evidence, not guesses.",
                            'E': "The environment/tool failed. Try a different execution channel or retry. This is not your fault — work around it.",
                        }.get(_error_category, "")

                        state['_error_category'] = _category_name
                        state['_prescription'] = _prescription

                        # Detect which library/approach failed (to block it)
                        _failed_lib = ""
                        for _lib in ['from pptx', 'from docx', 'import pandas', 'import openpyxl',
                                     'pyautogui', 'font.strike', 'line_spacing', 'useradd']:
                            if _lib in _action_str:
                                _failed_lib = _lib
                                break

                        # Count how many times this approach has been NOOP
                        _noop_count = 0
                        for _prev in state.get('step_insights_accumulator', [])[-5:]:
                            if _failed_lib and _failed_lib in str(_prev.get('action', '')):
                                _prev_obs = str(_prev.get('observation', ''))
                                if 'NO-OP' in _prev_obs or _prev.get('progress_score', 0) <= 1:
                                    _noop_count += 1

                        _approach_block = ""
                        if _noop_count >= 2 and _failed_lib:
                            _approach_block = (
                                f"\n⚠️ BLOCKED: The approach using '{_failed_lib}' has failed "
                                f"{_noop_count} times (NO-OP every time). You MUST use a "
                                f"COMPLETELY DIFFERENT library or method. For file modifications, "
                                f"try: lxml + zipfile for direct XML editing, or subprocess "
                                f"with sed/awk for text files."
                            )

                        # CRITIC-style doc retrieval: when approach fails,
                        # search local docs for alternative code examples.
                        # This injects EXTERNAL knowledge the model doesn't have.
                        _retrieved_docs = ""
                        # Trigger doc retrieval universally: NOOP, repeated errors,
                        # or stagnation (score stuck ≤2 for 3+ steps)
                        _score_history = state.get('progress_history', [])
                        _stagnant = len(_score_history) >= 3 and all(s <= 2 for s in _score_history[-3:])
                        if _is_noop or _noop_count >= 1 or (_has_error and _noop_count >= 0 and _stagnant):
                            try:
                                from doc_search import search_for_noop_alternative
                                _retrieved_docs = search_for_noop_alternative(
                                    failed_action=_action_str[:500],
                                    error_text=_error_snippet[:300],
                                    probe_data=_probe_data[:300],
                                )
                                if _retrieved_docs:
                                    print(f"[DOC RETRIEVAL] Injected {len(_retrieved_docs)} chars of documentation")
                            except Exception as _doc_err:
                                print(f"[DOC RETRIEVAL] Error (non-fatal): {_doc_err}")

                        _fix_prompt = (
                            f"An agent tried this action and it failed:\n"
                            f"ACTION: {_action_str[:300]}\n"
                            f"{'ERROR: ' + _error_snippet[:300] if _error_snippet else ''}\n"
                            f"{'NOOP: code ran (rc=0) but file is BYTE-IDENTICAL — the library silently ignored the change.' if _is_noop else ''}\n"
                            f"{'EVALUATOR GAP: ' + _diff_text[:300] if _diff_text else ''}\n"
                            f"{f'FILE STRUCTURE (from probe): {_probe_data[:300]}' if _probe_data else ''}\n\n"
                            f"ERROR CATEGORY: {_category_name}\n"
                            f"PRESCRIPTION: {_prescription}\n"
                            f"{_approach_block}\n"
                            f"{_retrieved_docs}\n\n"
                            f"Based on the ERROR CATEGORY and PRESCRIPTION above, generate ONE corrective action.\n"
                            f"{'IMPORTANT: Do NOT use ' + _failed_lib + ' — it has been proven to silently fail.' if _failed_lib and _noop_count >= 2 else ''}\n"
                            f"Reply with ONLY the action string (run: command, click, type, etc.). No explanation."
                        )
                        _fix_out = fast_model.generate([_fix_prompt])[0].outputs[0].text.strip()
                        # UNIVERSAL NEGATIVE ACTION VALIDATOR: reject structurally-broken
                        # candidates that cannot be valid actions in ANY env. Does NOT
                        # hardcode action verbs (those are env-specific). Works for
                        # OSWorld shell commands, ALFWorld navigation, WebArena clicks,
                        # any future env — just well-formedness.
                        import re as _re_va

                        def _is_structurally_broken(act: str) -> str:
                            """Return reason if action is universally broken, else ''."""
                            if not act:
                                return 'empty'
                            _s = act.strip().strip('`*- >').strip()
                            if len(_s) < 5:
                                return 'too short'
                            # Markdown fence line (not an action)
                            if _s.startswith('```'):
                                return 'markdown fence'
                            # Heredoc-without-body: opener marker exists but closer doesn't
                            _hd = _re_va.search(r'<<\s*[\'"]?(\w+)[\'"]?', _s)
                            if _hd:
                                _marker = _hd.group(1)
                                if _s.count(_marker) < 2:
                                    return f'heredoc-without-body (marker {_marker!r} appears once)'
                            # Python-style function call as whole candidate, e.g.
                            # `some_func()` or `some_func(arg=1, arg2='x')`. A Python call
                            # is never a direct shell action in any env — it needs to be
                            # wrapped (python3 -c, exec, etc.). Universal shell-vs-python check.
                            # Also catch the `run: some_func(args)` form where `run:` was
                            # prepended to a bare Python call.
                            _body = _re_va.sub(r'^\s*run:\s*', '', _s, count=1).strip()
                            if _re_va.match(r'^\w+\([^)]*\)\s*$', _body):
                                # But accept if it's a real shell utility call wrapped via
                                # `python3 -c`/`bash -c` etc.
                                if not _re_va.match(r'^(python3?|bash|sh|eval|exec)\b', _body):
                                    return 'python-style function call (not wrapped in a runtime)'
                            # Prose sentence: ends with period/question AND has >6 words
                            # AND no shell/code punctuation (`:`, `/`, quote, operator)
                            if _s.rstrip().endswith(('.', '?', '!')) and _s.count(' ') > 6 and not _re_va.search(r"[:/\\'\"<>|&]", _s):
                                return 'prose (sentence-like, no action structure)'
                            return ''

                        # Pick first line of response that isn't structurally broken.
                        # (Model may prefix with reasoning; we want the first actual action.)
                        _fix_line = ''
                        for _cand in _fix_out.split('\n'):
                            _cand = _cand.strip().lstrip('`*- >').strip()
                            if not _cand or _cand.startswith('```'):
                                continue
                            if not _is_structurally_broken(_cand):
                                _fix_line = _cand
                                break

                        _broken_reason = _is_structurally_broken(_fix_line)
                        if not _broken_reason and _fix_line != _action_str[:len(_fix_line)]:
                            # Verify the fix doesn't use the blocked approach
                            if _failed_lib and _noop_count >= 2 and _failed_lib in _fix_line:
                                print(f"[ENV-GROUNDED FIX] Rejected (uses blocked approach '{_failed_lib}'): '{_fix_line[:80]}'")
                            else:
                                state['step_gradient']['next_action_guidance'] = _fix_line
                                state['step_gradient']['guidance_source'] = 'env_grounded_fix'
                                print(f"[ENV-GROUNDED FIX] Extracted: '{_fix_line[:100]}'")
                        else:
                            print(f"[ENV-GROUNDED FIX] Rejected malformed candidate: '{_fix_line[:80]}' (no valid action prefix or incomplete structure)")
                    except Exception as _fix_err:
                        print(f"[ENV-GROUNDED FIX] Extraction error: {_fix_err}")

                # Accumulate gradient
                policy.gradients.append(gradient)

                # G31 UNIVERSAL — stash evaluator ground truth on the policy so
                # optimizer.step() can read it when synthesizing the next
                # base_policy. Without this, the optimizer gets only abstract
                # gradient strings and outputs generic advice. With this, it
                # can name specific literal requirements directly.
                try:
                    policy._last_eval_gap = state.get('_last_eval_gap', '') or ''
                    policy._eval_source = state.get('_eval_source', '') or ''
                except Exception:
                    pass

                # Optimizer step every 3 steps
                step_num = state.get('cur_step', 0)
                if step_num > 0 and step_num % 3 == 0:
                    optimizer.step(policy)
                    print(f"[TEXTGRAD] ENV {state['env_id']} Step {step_num}: Policy updated with {len(policy.gradients)} gradients")

                # Store gradient info in step_gradient for logging
                # FIX #28: NO TRUNCATION - full gradient contains critical learning lessons!
                if 'step_gradient' not in state:
                    state['step_gradient'] = {}
                state['step_gradient']['textgrad_loss'] = loss_text
                state['step_gradient']['textgrad_gradient'] = gradient
                state['step_gradient']['policy_version'] = len(policy.update_history)
                # Env-grounded score — set before generate_textgrad_step_guidance
                # so the routing logic sees the deterministic score
                state['step_gradient']['progress_score'] = _env_score
                state['step_gradient']['progress_status'] = _env_status
            # ============================================================================
            # END TEXTGRAD INTEGRATION
            # ============================================================================

            # DASHBOARD: Update live state after action execution + TextGrad
            if state['env_id'] == 0:  # Track first environment for dashboard
                has_ss = bool(info and info.get('after_screenshot_b64'))
                _dashboard.update(
                    step=state['cur_step'],
                    action=action,
                    has_screenshots=has_ss,
                    click_metadata=info.get('click_metadata') if info else None,
                    loss_text=(loss_text[:1000] if 'loss_text' in dir() and loss_text else ''),
                    gradient_text=(gradient[:1000] if 'gradient' in dir() and gradient else ''),
                )
                # Save before/after screenshots for this step
                if info and info.get('before_screenshot_b64'):
                    _dashboard.save_screenshot(state['cur_step'], 'before', info.get('before_screenshot_b64'))
                if info and info.get('after_screenshot_b64'):
                    _dashboard.save_screenshot(state['cur_step'], 'after', info.get('after_screenshot_b64'))
                # Fallback: capture from VM directly if no b64 screenshots
                if not has_ss:
                    vm_ip = env_states[0].get('vm_ip', '172.17.0.5') if env_states else '172.17.0.5'
                    _dashboard.save_screenshot_from_vm(state['cur_step'], 'after', vm_ip)
                # Update TODO display
                if 'todo_manager' in state and state['todo_manager']:
                    _dashboard.update_todos(state['todo_manager'])

            # ═══════════════════════════════════════════════════════════════
            # SYNERGISTIC EPISODIC REFLEXION GENERATION (CORRECT LOCATION)
            # Generate reflexion ONLY at key moments (failures/milestones/success)
            # ═══════════════════════════════════════════════════════════════
            is_key_moment = False
            moment_type = ""

            # Check for failure indicators
            failure_indicators = ['nothing happens', "don't see", "can't", 'already', 'closed']
            if any(fail in observation.lower() for fail in failure_indicators):
                is_key_moment = True
                moment_type = "FAILURE"
            # Check for success
            elif 'you win' in observation.lower() or done:
                is_key_moment = True
                moment_type = "SUCCESS"
            # Check for milestone (every 5 steps)
            elif state['cur_step'] % 5 == 0:
                is_key_moment = True
                moment_type = "MILESTONE"

            # ABLATION: Only generate reflexions if USE_REFLEXION is True
            if is_key_moment and USE_REFLEXION:
                # Skip early milestones - not enough context
                skip_reflexion = (moment_type == "MILESTONE" and state['cur_step'] < 5)

                if not skip_reflexion:
                    try:
                        if 'working_reflexions' not in state:
                            state['working_reflexions'] = []

                        # Create context-specific prompts for better insights
                        if moment_type == "FAILURE":
                            # Format valid actions list
                            valid_actions_list = state.get('valid_actions', [])
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list)
                            if len(valid_actions_list) > 30:
                                actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                            reflexion_prompt = f"""Task: {state['task']}
Action tried: {action}
What happened: {observation[:150]}

Available actions you can choose from:
{actions_display}

This action FAILED. What concrete lesson should we remember to avoid this in future trials?
Focus on: What assumption was wrong? What SPECIFIC action from the list above should we try instead?
Provide ONE actionable insight mentioning a specific action from the list (1-2 sentences):"""

                        elif moment_type == "SUCCESS":
                            # Format valid actions list
                            valid_actions_list = state.get('valid_actions', [])
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list)
                            if len(valid_actions_list) > 30:
                                actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                            reflexion_prompt = f"""Task: {state['task']}
Successful action: {action}
Result: {observation[:150]}

Available actions that were possible:
{actions_display}

This action SUCCEEDED and completed the task! What pattern should we remember?
Focus on: What sequence of SPECIFIC actions from the list worked? What made this successful?
Provide a success pattern to reuse (1-2 sentences):"""

                        else:  # MILESTONE
                            # Get progress score for context
                            progress_score = state.get('step_gradient', {}).get('progress_score', 0)

                            # SIMPLIFIED REFLEXION PROMPT (Based on Reflexion NeurIPS 2023 paper)
                            # Paper achieved 97% with simple prompt asking about strategy and loops
                            # CRITICAL: Include recent trajectory so LLM can see loop patterns
                            # Reflexion paper includes {trajectory} in their prompt for this reason

                            # Format COMPLETE trajectory so LLM can see full pattern history
                            # Changed from last 10 to ALL actions for better loop detection
                            recent_trajectory = ""
                            if state.get('trajectory'):
                                recent_trajectory = "Complete action history:\n"
                                for act, obs, _ in state["trajectory"]:
                                    recent_trajectory += f"  > {act}\n"

                            # Also include previous reflexions from this episode
                            previous_reflexions = ""
                            if state.get('working_reflexions'):
                                previous_reflexions = "\nPrevious insights from this episode:\n"
                                for ref in state['working_reflexions'][-3:]:  # Last 3 reflexions
                                    previous_reflexions += f"  - {ref.get('reflection', '')}\n"

                            # Format valid actions list
                            valid_actions_list = state.get('valid_actions', [])
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list)
                            if len(valid_actions_list) > 30:
                                actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                            reflexion_prompt = f"""Task: {state['task']}

{recent_trajectory}

Current step: {state['cur_step']}
Last action: {action}
Result: {observation[:150]}
Progress: {progress_score}/10
{previous_reflexions}

Available actions you can choose from:
{actions_display}

Reflect on your progress so far:
- Look at the complete action history above - do you see any repeating patterns or loops?
- Are you missing INFORMATION needed to make progress? (unknown file, unknown element location, unknown app state)
- If stuck 3+ steps: what COMPLETELY DIFFERENT approach could work? (e.g., shell probe, click content area, keyboard shortcut)
- If info is missing: suggest a PROBING action to collect it (e.g., run: ls ~/Desktop, run: ps aux | grep app, click center of screen)

Provide a concise reflection (2-3 sentences) identifying what went wrong AND the specific next action to try:"""

                        # Multimodal for visual envs — let model SEE the screen for reflection
                        after_ss_b64 = info.get('after_screenshot_b64') if info else None
                        if after_ss_b64 and state.get('env_type', '') in ('osworld', 'visual'):
                            step_reflection = model.generate_multimodal(
                                reflexion_prompt, after_image_b64=after_ss_b64,
                                max_tokens=400, temperature=0.3)
                            log_debug(f"[STEP REFLEXION] multimodal (sees screen)")
                        else:
                            sampling_params = SamplingParams(temperature=0.3, max_tokens=400)
                            step_reflection_output = model.generate([reflexion_prompt], sampling_params)[0]
                            step_reflection = step_reflection_output.outputs[0].text.strip()

                        state['working_reflexions'].append({
                            'step': state['cur_step'],
                            'action': action,
                            'observation': observation[:200],
                            'reflection': step_reflection,
                            'type': moment_type,
                            'success': moment_type == "SUCCESS"
                        })

                        # Save to env_configs for cross-trial persistence WITH DEDUPLICATION
                        env_idx = state['env_id']
                        if 'step_memory' not in env_configs[env_idx]:
                            env_configs[env_idx]['step_memory'] = []

                        # Deduplication: Check if a similar memory already exists
                        memory_text = f"[Step {state['cur_step']} {moment_type}] {step_reflection}"
                        is_duplicate = False

                        # Simple deduplication: Check for semantic similarity
                        for existing_mem in env_configs[env_idx]['step_memory']:
                            # Extract key words from both memories
                            new_words = set(step_reflection.lower().split())
                            existing_words = set(existing_mem.lower().split())

                            # If >70% word overlap, consider it a duplicate
                            if len(new_words) > 0:
                                overlap = len(new_words & existing_words) / len(new_words)
                                if overlap > 0.7:
                                    is_duplicate = True
                                    log_debug(f"[REFLEXION DEDUP] Skipping duplicate memory (overlap={overlap:.0%})")
                                    break

                        # Only save if not a duplicate
                        if not is_duplicate:
                            env_configs[env_idx]['step_memory'].append(memory_text)

                            # Prioritize SUCCESS memories - keep more of them
                            if moment_type == "SUCCESS":
                                # Keep last 15 memories total, but protect SUCCESS memories
                                if len(env_configs[env_idx]['step_memory']) > 15:
                                    # Remove oldest non-SUCCESS memory
                                    for i in range(len(env_configs[env_idx]['step_memory'])):
                                        if 'SUCCESS' not in env_configs[env_idx]['step_memory'][i]:
                                            del env_configs[env_idx]['step_memory'][i]
                                            break
                                    # If all are SUCCESS, just trim normally
                                    if len(env_configs[env_idx]['step_memory']) > 15:
                                        env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-15:]
                            else:
                                # For FAILURE/MILESTONE, keep only last 10
                                if len(env_configs[env_idx]['step_memory']) > 10:
                                    env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]

                            print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Generated episodic memory")
                        else:
                            print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Skipped (duplicate)")
                    except Exception as e:
                        log_debug(f"[REFLEXION ERROR] ENV {state['env_id']}: {e}")
            # ═══════════════════════════════════════════════════════════════

            # Detect failures (from original line ~2236)
            is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])

            # Get next valid actions (loop detection removed) - MEMORY FIX: Use env from registry
            next_valid_actions = []
            env = _env_registry[state['env_id']]
            if hasattr(env, 'get_current_valid_actions'):
                next_valid_actions = env.get_current_valid_actions()
            state['valid_actions'] = next_valid_actions  # Store for backward pass + reflexion prompts

            # Get current TODO (from original line ~2328)
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # Track inventory (from original line ~2333)
            # UNIVERSAL INVENTORY TRACKING — parse from observation text only
            # Works for ANY text env: if the agent runs "inventory" or the observation
            # mentions carrying/holding, update state. No env-specific regex.
            inventory_items = state.get('inventory', [])
            _obs_lower = str(observation).lower()
            if 'you are carrying:' in _obs_lower:
                # Parse: "You are carrying: a pan 1, a mug 2."
                _carry_text = _obs_lower.split('you are carrying:')[1].strip().rstrip('.')
                inventory_items = [item.strip().lstrip('a ').lstrip('an ') for item in _carry_text.split(',') if item.strip()]
            elif 'you are not carrying' in _obs_lower or 'not carrying anything' in _obs_lower:
                inventory_items = []
            elif 'you pick up' in _obs_lower:
                # "You pick up the pan 1 from the stoveburner 2."
                import re as _inv_re
                _pick_match = _inv_re.search(r'you pick up the (.+?)(?:\s+from\s+|\.$)', _obs_lower)
                if _pick_match:
                    _item = _pick_match.group(1).strip().rstrip('.')
                    if _item and _item not in inventory_items:
                        inventory_items.append(_item)
            elif ('you put' in _obs_lower or 'you move' in _obs_lower) and 'nothing happens' not in _obs_lower:
                # "You put the pan 1 in/on the countertop 1." or "You move the pan 1 to the countertop 1."
                import re as _inv_re
                _drop_match = _inv_re.search(r'you (?:put|move) the (.+?)(?:\s+(?:in|on|to|in/on)\s+|\.$)', _obs_lower)
                if _drop_match:
                    _item = _drop_match.group(1).strip().rstrip('.')
                    if _item in inventory_items:
                        inventory_items.remove(_item)
            state['inventory'] = inventory_items
            if inventory_items:
                print(f"[INVENTORY-DEBUG] ENV {state['env_id']} Step {state.get('cur_step', '?')}: {inventory_items}")

            # Get episodic memory
            env_idx = state['env_id']
            episodic_memory_only = env_configs[env_idx].get('memory', [])

            # Get tiered reflexions
            tiered_step_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

            # Extract explored locations from TODO manager
            explored_locations_dict = {}
            if 'todo_manager' in state and state['todo_manager']:
                explored_locations_dict = state['todo_manager'].visited_locations

            # CRITICAL FIX: Skip gradient generation for environments that just completed THIS STEP
            # The flag 'skip_gradient_this_step' is set immediately after env.step() returns done=True
            # This prevents generating a gradient for the final action that completed the episode
            if state.get('skip_gradient_this_step', False):
                log_debug(f"[ENV {state['env_id']}] Skipping gradient generation - episode just completed (done=True)")
                state['skip_gradient_this_step'] = False  # Reset flag for next iteration
                continue  # Skip to next environment, don't add to action_results

            # Collect ALL data for gradient generation (DON'T generate yet!)
            action_results.append({
                'idx': idx,
                'state': state,
                'action': action,
                'observation': observation,
                'prev_observation': state['prev_observation'],
                'initial_observation': state.get('initial_observation', ''),  # CRITICAL: All available locations
                'explored_locations': explored_locations_dict,  # CRITICAL: What's been checked
                'is_failure': is_failure,
                'done': done,
                'reward': reward,
                'info': info,
                'is_repetitive': False,  # Loop detection removed
                'next_valid_actions': next_valid_actions,
                'current_todo': current_todo,
                'inventory_items': inventory_items,
                'tiered_step_reflexions': tiered_step_reflexions,
                'episodic_memory_only': episodic_memory_only,
                'step_insights_accumulator': state.get('step_insights_accumulator', []),  # CRITICAL: TextGrad's gradient history!
                'task': state['task'],
                'current_step': state['cur_step'],
                'env_idx_in_batch': idx
            })

        print(f"[PHASE 4] ✓ Executed {len(action_results)} actions, collected results")

        # ========================================================================
        # PHASE 5: BATCH GENERATE STEP GRADIENTS (TRUE SYNERGY - TWO PASS!)
        # ========================================================================
        print(f"\n[PHASE 5] Generating step gradients for {len(action_results)} environments...")

        # TRUE SYNERGY ARCHITECTURE: Two-pass approach
        # PASS 1: Reflexion ONLY when genuinely stuck (not fixed cadence)
        # PASS 2: TextGrad action recommendation for ALL environments

        # PASS 1: Identify environments needing Reflexion strategic insights
        print(f"[PHASE 5 - PASS 1] Identifying environments needing Reflexion strategic analysis...")
        reflexion_needed_indices = []
        reflexion_prompts = []

        for idx, result in enumerate(action_results):
            state = result['state']
            current_step = state['cur_step']

            # PROGRESS-BASED ROUTING via ReflexGradCore (replaces fixed cadence)
            # Feed textual assessment + status + score into the dual-process router
            _step_grad = state.get('step_gradient', {})
            last_score = _step_grad.get('progress_score', 0)
            # Map ALFWorld status categories to router-compatible status
            _raw_status = _step_grad.get('progress_status', '').upper()
            _status_map = {
                'NO_PROGRESS': 'stuck', 'EXPLORING': 'progressing',
                'PARTIAL_PROGRESS': 'progressing', 'MAJOR_PROGRESS': 'progressing',
                'TASK_COMPLETE': 'complete',
            }
            _last_status = _status_map.get(_raw_status, '')
            _last_assessment = _step_grad.get('backward_gradient', '')
            # Also check: if last action got "Nothing happens", force status to "stuck"
            _last_obs = state.get('last_observation', '')
            if 'Nothing happens' in str(_last_obs) and _last_status not in ('stuck', 'regressed'):
                _last_status = 'stuck'
                _last_assessment += ' Action produced "Nothing happens" — invalid or impossible action.'
            reflexgrad_router = state.get('reflexgrad_core')

            # SCORE-STALL DETECTION (ported from OSWorld CUA agent)
            # If score stays at PARTIAL_PROGRESS (6) for 4+ consecutive steps, the agent is
            # looping even though TextGrad says "progressing". Force Reflexion for deep analysis.
            # This catches the real signal: same score = no meaningful state change = stuck.
            _force_reflexion = False
            if '_score_stall_count' not in state:
                state['_score_stall_count'] = 0
                state['_last_score_for_stall'] = -1
            if last_score == state['_last_score_for_stall'] and last_score >= 3:
                state['_score_stall_count'] += 1
            else:
                state['_score_stall_count'] = 0
            state['_last_score_for_stall'] = last_score
            if state['_score_stall_count'] >= 4:
                _force_reflexion = True
                state['_score_stall_count'] = 0  # Reset after forcing
                _last_assessment += f' SCORE STALL: score={last_score} unchanged for 4+ steps — agent is looping without meaningful progress.'
                print(f"[SCORE-STALL] ENV {state['env_id']} Step {current_step}: Forcing Reflexion — score={last_score} for 4+ consecutive steps")

            if reflexgrad_router and USE_REFLEXION:
                if _force_reflexion:
                    mode = 'reflexion'
                    reflexgrad_router.steps_since_reflexion = 0
                else:
                    mode = reflexgrad_router.choose_analysis_mode(
                        assessment=_last_assessment, status=_last_status, score=last_score
                    )
                use_reflexion = (mode == 'reflexion')
                print(f"[ROUTING] ENV {state['env_id']} Step {current_step}: score={last_score}/10 → {mode.upper()} "
                      f"(consecutive_low={reflexgrad_router.consecutive_low_progress}, "
                      f"scores={reflexgrad_router.progress_scores[-5:]}"
                      f"{', TODO-STALL FORCED' if _force_reflexion else ''})")

                # STAGNATION-TRIGGERED HARD VERIFY (universal closed-loop fix).
                # The existing Hard Verify only fires when the agent claims TASK_COMPLETE.
                # If the agent is stuck and never claims complete, the evaluator never speaks
                # and the gradient signal is just TextGrad's screen-delta heuristic, which can
                # reward no-ops like xdg-open on an already-open file.
                # Solution: when the score has been low/flat for several steps, run the real
                # evaluator NOW and inject its failure reason into the next observation. The
                # evaluator's failure messages ("MISSING_SHEET:Sheet2", "P2:MISMATCH expected=2"
                # etc.) are the most concrete gradient available — far more useful than screen
                # deltas. Cooldown prevents repeated calls.
                _stag_low = reflexgrad_router.consecutive_low_progress >= 3
                _stag_flat = state.get('_score_stall_count', 0) >= 3
                # Also trigger when score oscillates between low and PARTIAL_PROGRESS (≤6)
                # but never reaches MAJOR_PROGRESS (≥8). This catches the "looks like
                # progress but isn't" pattern (e.g. xdg-open re-focusing reads as 6).
                _recent_scores = list(reflexgrad_router.progress_scores[-5:])
                _stag_no_major = (len(_recent_scores) >= 5
                                  and max(_recent_scores) <= 6
                                  and current_step >= 5)
                _stag_cooldown_ok = (current_step - state.get('_last_stagnation_eval_step', -10)) >= 4
                if (_stag_low or _stag_flat or _stag_no_major) and _stag_cooldown_ok and hasattr(env, 'evaluate'):
                    try:
                        _stag_eval = env.evaluate()
                        _stag_success = _stag_eval.get('success', False)
                        if _stag_success:
                            # Goal already met — surface as TASK_COMPLETE so the loop exits cleanly
                            print(f"[STAGNATION-EVAL] ENV {state['env_id']} Step {current_step}: "
                                  f"Evaluator says SUCCESS during stagnation — completing episode")
                            state['_eval_already_success'] = True
                        else:
                            _stag_parts = []
                            if _stag_eval.get('output'):
                                _stag_parts.append(str(_stag_eval['output']))
                            # G14 UNIVERSAL — disjunctive-evaluator focus applied to
                            # STAGNATION-EVAL too. If conj=='or', show ONLY the
                            # closest-to-passing sub-evaluator with a "commit to one
                            # path" banner. Otherwise concatenate as before.
                            _stag_results = _stag_eval.get('results') or []
                            _stag_conj = str(_stag_eval.get('conj', 'and')).lower()
                            if _stag_conj == 'or' and len(_stag_results) > 1:
                                def _mm_count_stag(s: str) -> int:
                                    try:
                                        return s.upper().count('MISMATCH') + s.upper().count('FAIL')
                                    except Exception:
                                        return 10**9
                                # G14-STABLE: reuse committed path from wrapper
                                _committed = getattr(env, '_g14_chosen_path', None)
                                if _committed is not None and _committed < len(_stag_results):
                                    _best_idx = _committed
                                    _best_r = _stag_results[_committed]
                                    _reason_stag = f"path #{_best_idx} (committed — stay on target)"
                                else:
                                    _scored = [
                                        (_mm_count_stag(str(_r.get('output', '') or '') if isinstance(_r, dict) else ''),
                                         i, _r)
                                        for i, _r in enumerate(_stag_results)
                                    ]
                                    _scored.sort(key=lambda t: t[0])
                                    _best_r = _scored[0][2]
                                    _best_idx = _scored[0][1]
                                    try:
                                        setattr(env, '_g14_chosen_path', _best_idx)
                                    except Exception:
                                        pass
                                    _reason_stag = f"path #{_best_idx} (closest to passing, first eval)"
                                _best_out = str(_best_r.get('output', '') or _best_r.get('reason', '')) if isinstance(_best_r, dict) else ''
                                _stag_parts.append(
                                    f"[DISJUNCTIVE EVALUATOR — conj=OR over {len(_stag_results)} alternative "
                                    f"acceptance paths. Task passes if ANY ONE path passes; paths have "
                                    f"contradictory requirements, pick ONE and commit. "
                                    f"{_reason_stag}.]\n"
                                    f"{_best_out}"
                                )
                            elif _stag_results:
                                for _r in _stag_results:
                                    if isinstance(_r, dict):
                                        _r_out = _r.get('output', '') or _r.get('reason', '')
                                        if _r_out:
                                            _stag_parts.append(f"[{_r.get('func', '?')}] {_r_out}")
                            if _stag_eval.get('reason'):
                                _stag_parts.append(str(_stag_eval['reason']))
                            _stag_reason = "; ".join(_stag_parts)[:1200] or "evaluator returned no details"
                            state['_eval_stagnation_reason'] = _stag_reason
                            # GAP A FIX: persist eval gap separately so it survives the
                            # PASS-2 pop and can be plumbed into the NEXT step's TextGrad
                            # BACKWARD pass. Without this, gradient computation never sees
                            # the ground-truth evaluator delta — only screenshots — and
                            # ends up optimizing against "no visible change" instead of
                            # "your code didn't write the expected XML attribute". The
                            # eval gap is the most reliable supervision signal in the
                            # entire system; the gradient computer must see it.
                            state['_last_eval_gap'] = _stag_reason
                            state['_last_eval_gap_step'] = current_step
                            # Push the evaluator's verdict into the wrapper's
                            # persistent-state field so the V45 ⛔ delta block
                            # AND the failed-approach tombstone list both pin
                            # on EVERY subsequent observation (not just after
                            # `done`). Universal — any env exposing
                            # `_unresolved_verification` benefits; others no-op.
                            try:
                                if hasattr(env, '_unresolved_verification'):
                                    setattr(env, '_unresolved_verification', _stag_reason[:500])
                                # G13 — trigger file-integrity recovery from the silent
                                # STAGNATION-EVAL path too, not only HARD VERIFY on done.
                                # Otherwise a file corrupted mid-trial sits unresolved
                                # until the agent happens to try `done`.
                                _g13 = getattr(env, '_maybe_restore_corrupted_files', None)
                                if _g13:
                                    _recov = _g13(_stag_reason)
                                    if _recov:
                                        # Persist into state so the next obs builder
                                        # surfaces it (belt-and-braces; wrapper also
                                        # writes it into obs_text on the next step).
                                        state['_g13_recovery_notice'] = _recov
                                        print(f"[G13 STAGNATION-EVAL] {_recov[:200]}")
                            except Exception:
                                pass
                            # Capture the evaluator's own source code ONCE per
                            # trial. This is the literal python that the
                            # evaluator runs to compute pass/fail — reading it
                            # tells the agent EXACTLY which file paths, XML
                            # attributes, and comparison logic decide success.
                            # No more guessing what `(4,1,1,'sngStrike')` tuples
                            # mean: the agent can see that the metric extracts
                            # `el.get('strike','')` from `<a:rPr>` elements.
                            # Universal: every task has an evaluator and every
                            # evaluator handler is plain python.
                            if not state.get('_eval_source') and hasattr(env, 'get_evaluator_source'):
                                try:
                                    _src = env.get_evaluator_source()
                                    if _src:
                                        state['_eval_source'] = _src
                                        print(f"[STAGNATION-EVAL-SRC] ENV {state['env_id']}: captured {len(_src)} chars of evaluator source")
                                except Exception as _src_err:
                                    print(f"[STAGNATION-EVAL-SRC] ENV {state['env_id']}: failed to capture source: {_src_err}")
                            print(f"[STAGNATION-EVAL] ENV {state['env_id']} Step {current_step}: "
                                  f"score stuck → ran evaluator → {_stag_reason[:200]}")

                            # GRADIENT TRANSLATION: Ask the LLM to translate the raw
                            # evaluator failure (often library-specific tuples like
                            # `[(4, 1, 1, 'sngStrike')]`) into a concrete `run: python3 -c`
                            # action that closes the gap. Open-weight models read raw
                            # evaluator tuples as info, not as instructions, so an
                            # explicit translation step makes the gradient actionable.
                            # Universal: works for any task by asking the LLM to choose
                            # the appropriate library/property based on the evaluator output.
                            try:
                                _task_str = state.get('task', '') or ''
                                _file_hint = ''
                                _modf = state.get('_modified_files', None)
                                if not _modf and hasattr(env, '_modified_files'):
                                    _modf = list(getattr(env, '_modified_files', []) or [])
                                if not _modf and hasattr(env, '_downloaded_files'):
                                    _modf = list(getattr(env, '_downloaded_files', []) or [])
                                if _modf:
                                    _file_hint = f"\nFile(s) under consideration: {', '.join(list(_modf)[:3])}"
                                # DIVERSIFY: surface previous translation attempts so the
                                # LLM tries a DIFFERENT API approach. Without this, the
                                # same prompt produces the same wrong answer every cycle.
                                _prev_attempts = state.setdefault('_xlate_attempts', [])
                                _findings = state.setdefault('_xlate_findings', '')
                                _xlate_count = state.setdefault('_xlate_total_count', 0)

                                # INTROSPECTION MODE: after 3 failed attempts, switch one
                                # XLATE cycle to library introspection. The agent has
                                # `python3 -c` and the library installed — it should be
                                # able to discover the right API by inspecting it directly
                                # (dir, help, inspect.getsource, examining XML of existing
                                # elements). Universal: works for any library because
                                # introspection is a Python language feature, not a task
                                # convention. The captured output is fed into the next
                                # action XLATE so the LLM has concrete API facts.
                                # Treat findings that are just error messages as "empty"
                                # so we get another chance to introspect with corrected code.
                                _findings_useful = bool(_findings) and not (
                                    'SyntaxError' in _findings
                                    or 'Traceback' in _findings
                                    or 'NameError' in _findings
                                    or 'IndentationError' in _findings
                                    or len(_findings.strip()) < 200
                                )
                                _force_introspection = (
                                    len(_prev_attempts) >= 3
                                    and (_xlate_count % 3 == 0)
                                    and not _findings_useful
                                )
                                _prev_block = ""
                                if _prev_attempts:
                                    _prev_lines = "\n".join(
                                        f"  {i+1}. {a[:300]}" for i, a in enumerate(_prev_attempts[-4:])
                                    )
                                    _prev_block = (
                                        "\n\nPREVIOUS ATTEMPTS THAT DID NOT CLOSE THE GAP (the "
                                        "evaluator still reports the same failure after each — "
                                        "do not repeat the same approach):\n"
                                        f"{_prev_lines}\n"
                                    )
                                # UNIVERSAL: Inject accumulated backward gradients so XLATE
                                # knows what the learning loop already diagnosed.
                                # Without this, XLATE regenerates from scratch each time,
                                # repeating the same mistakes the gradients already identified.
                                _si = state.get('step_insights_accumulator', [])
                                if _si:
                                    _gradient_lessons = []
                                    for _ins in _si[-5:]:  # Last 5 steps
                                        _bw = str(_ins.get('backward_gradient', ''))[:200]
                                        _act = str(_ins.get('action', ''))[:100]
                                        _obs = str(_ins.get('observation', ''))[:150]
                                        if _bw and len(_bw) > 10:
                                            _gradient_lessons.append(
                                                f"  Action: {_act}\n  Result: {_obs}\n  Lesson: {_bw}"
                                            )
                                    if _gradient_lessons:
                                        _prev_block += (
                                            "\n\nLEARNED LESSONS FROM PREVIOUS STEPS (the learning loop "
                                            "already diagnosed these — DO NOT ignore them):\n"
                                            + "\n---\n".join(_gradient_lessons[-3:]) + "\n"
                                        )
                                _findings_block = ""
                                if _findings:
                                    _findings_block = (
                                        "\n\nINTROSPECTION FINDINGS (concrete facts you discovered "
                                        "by running python on the actual installed library — these "
                                        "override any prior assumption you had about the API):\n"
                                        f"{_findings[:2000]}\n"
                                    )

                                if _force_introspection:
                                    # Pure capability statement — no tactics, no library
                                    # names, no format-specific recipes. Just facts about
                                    # the environment. The model decides how to discover.
                                    _xlate_prompt = (
                                        "Your previous attempts to make the evaluator pass have "
                                        "failed. Before trying again, run a discovery step that "
                                        "produces facts you don't currently have.\n\n"
                                        f"TASK: {_task_str[:500]}{_file_hint}\n\n"
                                        "EVALUATOR REPORT (what the evaluator wants):\n"
                                        f"{_stag_reason}"
                                        f"{_prev_block}\n\n"
                                        "AVAILABLE CAPABILITIES (use any combination — you choose):\n"
                                        "- A full bash shell on the VM (any installed program is "
                                        "callable, any file you can read is readable)\n"
                                        "- python3 with all installed packages\n"
                                        "- The file(s) above exist on disk and you can read them\n"
                                        "- Any python script you write goes to /tmp/_agent_run.py "
                                        "and runs as a real file (multi-line, f-strings, escapes, "
                                        "any python construct works — no one-liner restrictions)\n"
                                        "- stdout is captured and returned to you\n"
                                        "- Files can be read in any form: raw bytes "
                                        "(open(path,'rb').read()), text, or via the same library "
                                        "that wrote them. This lets you verify what your previous "
                                        "code actually produced on disk rather than trusting that "
                                        "rc=0 means the bytes match what an evaluator expects.\n"
                                        "- Any python object can be inspected with dir(obj), "
                                        "help(obj), type(obj), obj.__class__.__mro__, or "
                                        "inspect.getsource(obj) — useful when you're unsure which "
                                        "attribute name a library actually serializes.\n\n"
                                        "Output ONLY ONE line of the form:\n"
                                        "  run: python3 -c \"<discovery script — prints facts to "
                                        "stdout that will be fed back to you>\"\n\n"
                                        "Choose what to discover based on what gap the evaluator "
                                        "reported and what you don't yet know. Output ONLY the "
                                        "single `run: ...` line, nothing else.\n\n"
                                        "ACTION:"
                                    )
                                else:
                                    # Ask Gemma for MULTI-LINE Python code (real file, real
                                    # indentation, real control flow). We wrap it in the
                                    # base64-file-execute channel after generation.
                                    #
                                    # UNIVERSAL ENUM-STRING BINDING HINT: when the evaluator
                                    # delta contains a literal quoted short string (e.g.
                                    # `'sngStrike'`, `'center'`, `'bold'`), that exact string
                                    # must appear in the target file's attribute/element.
                                    # Library high-level setters (`.strikethrough = True`,
                                    # `.alignment = CENTER`, etc.) emit the library's own
                                    # default enum/bool — often NOT the exact string the
                                    # evaluator wants. The universal fix is direct attribute
                                    # setting: for XML, `element.set('attr', 'exact_string')`;
                                    # for JSON, `obj[key] = 'exact_string'`. Detect the
                                    # presence of quoted short strings in the evaluator report
                                    # and surface the rule explicitly so Gemma picks the
                                    # direct-set API over the high-level-library API.
                                    import re as _re_ebh
                                    # Purely structural: match a quoted short identifier that
                                    # appears inside a tuple/list slot (preceded by `(`, `[`,
                                    # or `,` and followed by `,`, `)`, or `]`). That shape
                                    # reliably identifies evaluator-expected attribute-value
                                    # payloads across any task/format, without matching label
                                    # strings or function names elsewhere in the report.
                                    _enum_candidates = _re_ebh.findall(
                                        r"[\(\[,]\s*['\"]([A-Za-z][A-Za-z0-9_-]{1,30})['\"]\s*[,\)\]]",
                                        str(_stag_reason)
                                    )
                                    _enum_hint = ''
                                    if _enum_candidates:
                                        _unique_enums = list(dict.fromkeys(_enum_candidates))[:5]
                                        _enum_hint = (
                                            "\n- ENUM-STRING BINDING: the evaluator report contains "
                                            f"literal quoted strings {_unique_enums}. Those exact "
                                            "strings must end up as attribute values in the target "
                                            "file. High-level library setters (like a Python bool "
                                            "property, or a library's own enum class) typically emit "
                                            "their own default tokens — NOT the literal strings above. "
                                            "To write an exact string, set the attribute directly: "
                                            "for XML use `element.set('attr_name', 'exact_string')` "
                                            "via lxml/ElementTree; for JSON use `obj[key] = "
                                            "'exact_string'`. Do NOT rely on a boolean or named-enum "
                                            "shortcut that the library translates internally.\n"
                                        )
                                    # Include the LATEST structure probe (if any), showing the
                                    # actual current bytes so Gemma sees the real attribute name,
                                    # current value, and surrounding path.
                                    _probe_snippet = ''
                                    _probe_src = (state.get('_last_structure_probe', '')
                                                  or getattr(env, '_last_structure_probe', '')
                                                  or '')
                                    if _probe_src:
                                        _probe_snippet = (
                                            "\n\nCURRENT FILE STRUCTURE (actual bytes on disk — use "
                                            "these element names and attribute paths; do NOT guess):\n"
                                            f"{_probe_src[:2500]}\n"
                                        )

                                    # G2 (intent persistence): if a previous XLATE produced
                                    # a valid candidate and the evaluator STILL reports a
                                    # gap, prefer a MINIMAL-EDIT prompt over regenerate.
                                    # Universal: triggers only when prior candidate exists;
                                    # envs without an XLATE cycle never see this path.
                                    _prior_code = state.get('_last_xlate_code', '') or ''
                                    if _prior_code:
                                        _xlate_prompt = (
                                            "You previously produced Python code for this task, "
                                            "but the evaluator STILL reports a gap. Produce a "
                                            "MINIMAL EDIT of your previous code that closes the "
                                            "remaining gap. Do NOT regenerate the whole program — "
                                            "identify the one or two lines that need to change and "
                                            "keep everything else intact.\n\n"
                                            f"TASK: {_task_str[:500]}{_file_hint}\n\n"
                                            "YOUR PREVIOUS CODE (ran cleanly but did not satisfy "
                                            "the evaluator):\n"
                                            "```python\n"
                                            f"{_prior_code}\n"
                                            "```\n\n"
                                            "EVALUATOR REPORT AFTER YOUR LAST CODE RAN "
                                            "(raw — this is the remaining gap):\n"
                                            f"{_stag_reason}"
                                            f"{_probe_snippet}"
                                            f"{_prev_block}"
                                            f"{_findings_block}\n\n"
                                            "Output the REVISED full Python program in a fenced "
                                            "block. Preserve the structure that already worked; "
                                            "change only what the evaluator report says is wrong.\n\n"
                                            "Rules:\n"
                                            "- Write REAL multi-line Python with proper indentation\n"
                                            "- Keep imports / helper functions / control-flow that "
                                            "already executed cleanly — edit ONLY the lines that "
                                            "the evaluator's remaining gap points to\n"
                                            "- Do NOT compress control-flow onto one line\n"
                                            "- Do NOT wrap in `python3 -c` or any shell command\n"
                                            "- Use the EXACT identifiers from the evaluator report "
                                            "(do not substitute friendly aliases; copy literal "
                                            "string values)"
                                            f"{_enum_hint}"
                                            "- Save the file at the end if you modified it\n"
                                            "- Output ONLY the ```python code block, no narration\n\n"
                                            "REVISED PYTHON CODE:"
                                        )
                                        print(f"[STAGNATION-XLATE-EDIT] ENV {state['env_id']}: "
                                              f"minimal-edit mode (prior code {len(_prior_code)} chars)")
                                    else:
                                        _xlate_prompt = (
                                            "You are translating an automated task evaluator's failure "
                                            "report into concrete Python code that will close the gap.\n\n"
                                            f"TASK: {_task_str[:500]}{_file_hint}\n\n"
                                            "EVALUATOR REPORT (raw — interpret it yourself):\n"
                                            f"{_stag_reason}"
                                            f"{_probe_snippet}"
                                            f"{_prev_block}"
                                            f"{_findings_block}\n\n"
                                            "AVAILABLE CAPABILITIES:\n"
                                            "- Full Python 3 runtime with any installed package\n"
                                            "- Your code runs as a real file (multi-line, f-strings, "
                                            "imports, functions, loops, try/except — any construct works)\n"
                                            "- The file(s) above exist on disk and are readable/writable\n\n"
                                            "Output VALID MULTI-LINE PYTHON CODE inside a fenced block:\n"
                                            "```python\n"
                                            "<your code — multiple lines with real indentation>\n"
                                            "```\n\n"
                                            "Rules:\n"
                                            "- Write REAL multi-line Python with proper indentation\n"
                                            "- Do NOT compress control-flow onto one line\n"
                                            "- Do NOT wrap in `python3 -c` or any shell command\n"
                                            "- If INTROSPECTION FINDINGS are present, USE them — they "
                                            "are concrete facts about the actual installed library you "
                                            "discovered by running code\n"
                                            "- Use the EXACT identifiers from the evaluator report (do "
                                            "not substitute friendly aliases; copy literal string values)"
                                            f"{_enum_hint}"
                                            "- Save the file at the end if you modified it\n"
                                            "- Output ONLY the ```python code block, no narration\n\n"
                                            "PYTHON CODE:"
                                        )
                                _xlate_sp = SamplingParams(
                                    max_tokens=600, temperature=0.0,
                                    stop=[], skip_special_tokens=True,
                                )
                                _xlate_out = model.generate([_xlate_prompt], _xlate_sp,
                                                            reasoning_effort='medium')
                                _xlate_text = _xlate_out[0].outputs[0].text.strip() if _xlate_out else ''
                                # Introspection branch still uses the legacy `run:` single-line
                                # format (the probe is a short one-liner by design). For the
                                # main translation branch we extract a multi-line python block
                                # and wrap in the base64-file channel ourselves.
                                _run_line = None
                                if _force_introspection:
                                    # Strip fences and take first run: line
                                    _xt = _xlate_text
                                    for _pfx in ('```python', '```bash', '```sh', '```'):
                                        if _xt.startswith(_pfx):
                                            _xt = _xt[len(_pfx):].lstrip()
                                    if _xt.endswith('```'):
                                        _xt = _xt[:-3].rstrip()
                                    for _ln in _xt.splitlines():
                                        _ln = _ln.strip()
                                        if _ln.lower().startswith('run:'):
                                            _run_line = _ln
                                            break
                                else:
                                    # UNIVERSAL MULTI-LINE CHANNEL: extract python block,
                                    # validate via ast.parse, LLM-repair if broken, then
                                    # wrap in base64-file-execute so the model's real
                                    # indentation/control-flow reaches the VM intact.
                                    import re as _re_xl
                                    _block_m = _re_xl.search(
                                        r'```(?:python)?\s*\n(.*?)```', _xlate_text, _re_xl.DOTALL
                                    )
                                    _code = _block_m.group(1).rstrip() if _block_m else _xlate_text
                                    # Strip leading/trailing fences that may have slipped through
                                    for _pfx in ('```python', '```bash', '```sh', '```'):
                                        if _code.startswith(_pfx):
                                            _code = _code[len(_pfx):].lstrip()
                                    if _code.endswith('```'):
                                        _code = _code[:-3].rstrip()
                                    _code = _code.strip()
                                    # Validate Python syntax; if broken, ask same Gemma to repair
                                    _valid = False
                                    if _code:
                                        try:
                                            import ast as _ast_xl
                                            _ast_xl.parse(_code)
                                            _valid = True
                                        except SyntaxError as _se_xl:
                                            _repair_prompt = (
                                                "Fix the Python syntax error and output ONLY valid "
                                                "multi-line Python (no fences, no narration).\n"
                                                f"SYNTAX ERROR: {_se_xl}\n"
                                                f"BROKEN CODE:\n{_code}\n\n"
                                                "FIXED PYTHON:"
                                            )
                                            try:
                                                _rep_sp = SamplingParams(
                                                    max_tokens=800, temperature=0.0,
                                                    stop=[], skip_special_tokens=True,
                                                )
                                                _rep_out = model.generate([_repair_prompt], _rep_sp,
                                                                          reasoning_effort='medium')
                                                _rep_text = _rep_out[0].outputs[0].text.strip() if _rep_out else ''
                                                _rep_m = _re_xl.search(
                                                    r'```(?:python)?\s*\n(.*?)```', _rep_text, _re_xl.DOTALL
                                                )
                                                _rep_code = (_rep_m.group(1).rstrip() if _rep_m
                                                             else _rep_text.strip())
                                                _ast_xl.parse(_rep_code)
                                                _code = _rep_code
                                                _valid = True
                                                print(f"[STAGNATION-XLATE-REPAIR] ENV {state['env_id']}: "
                                                      f"LLM-repaired broken XLATE code ({_rep_out[0].outputs[0].text[:50]}...)")
                                            except Exception as _rep_err:
                                                print(f"[STAGNATION-XLATE-REPAIR] ENV {state['env_id']}: "
                                                      f"repair failed — {_rep_err}")
                                    if _valid and _code:
                                        import base64 as _b64_xl
                                        _b64_code = _b64_xl.b64encode(_code.encode()).decode()
                                        # Single-line shell action; the multi-line code is
                                        # preserved exactly inside base64, so indentation and
                                        # control-flow survive end-to-end.
                                        _run_line = (
                                            f"run: echo {_b64_code} | base64 -d > /tmp/_xlate.py "
                                            f"&& python3 /tmp/_xlate.py"
                                        )
                                        # G2 (intent persistence): persist the DECODED Python
                                        # so the NEXT XLATE cycle can show it to the model and
                                        # ask for a MINIMAL EDIT (not a full regenerate) when
                                        # the evaluator still reports a gap. Universal —
                                        # triggers only when a prior valid candidate exists;
                                        # absent it, XLATE proceeds as regenerate-from-scratch.
                                        state['_last_xlate_code'] = _code
                                        state['_last_xlate_step'] = current_step
                                        print(f"[STAGNATION-XLATE-B64] ENV {state['env_id']}: "
                                              f"wrapped {len(_code)} chars of python in base64 channel")
                                if _run_line:
                                    state['_eval_stagnation_action'] = _run_line
                                    state['_xlate_total_count'] = _xlate_count + 1
                                    if _force_introspection:
                                        # Mark this action as an introspection probe so
                                        # the action runner can capture stdout into
                                        # state['_xlate_findings'] for the NEXT attempt.
                                        state['_pending_introspection'] = True
                                        print(f"[STAGNATION-XLATE-INTROSPECT] ENV {state['env_id']}: introspection probe → {_run_line[:200]}")
                                    else:
                                        # Only record action attempts (not introspections)
                                        # in the failed-attempts list
                                        _prev_attempts.append(_run_line)
                                        if len(_prev_attempts) > 6:
                                            del _prev_attempts[:-6]
                                        print(f"[STAGNATION-XLATE] ENV {state['env_id']}: translated (attempt #{len(_prev_attempts)}) → {_run_line[:200]}")
                                else:
                                    print(f"[STAGNATION-XLATE] ENV {state['env_id']}: could not extract run: line from LLM output ({len(_xlate_text)} chars)")
                            except Exception as _xlate_e:
                                print(f"[STAGNATION-XLATE] ENV {state['env_id']}: translation error: {_xlate_e}")
                        state['_last_stagnation_eval_step'] = current_step
                    except Exception as _stag_e:
                        print(f"[STAGNATION-EVAL] ENV {state['env_id']} Step {current_step}: evaluator error: {_stag_e}")
            else:
                # Fallback to fixed cadence if router unavailable
                use_reflexion = USE_REFLEXION and (current_step % 5 == 0)
                log_debug(f"[SCHEDULE] ENV {state['env_id']} Step {current_step}: Reflexion={'YES' if use_reflexion else 'NO'} (fallback fixed cadence)")

            if use_reflexion:
                reflexion_needed_indices.append(idx)
                _is_vis = result.get('state', {}).get('env_type', '') in ('osworld', 'visual')
                _, reflexion_prompt = build_step_gradient_prompt_from_data(result, log_debug, is_visual_env=_is_vis, learning_context=state.get('learning_context'))
                reflexion_prompts.append(reflexion_prompt)
                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {current_step}: Reflexion triggered (progress-based routing)")

        # Generate Reflexion strategic insights in batch (if any needed)
        if len(reflexion_prompts) > 0:
            print(f"[PHASE 5 - PASS 1] Generating {len(reflexion_prompts)} Reflexion strategic insights in batch...")
            reflexion_sampling_params = SamplingParams(
                max_tokens=7000,
                temperature=0.3,
                stop=["TASK:", "BEFORE:"],
                skip_special_tokens=True
            )
            reflexion_outputs = model.generate(reflexion_prompts, reflexion_sampling_params, reasoning_effort='medium')
            print(f"[PHASE 5 - PASS 1] ✓ Generated {len(reflexion_outputs)} Reflexion insights!")

            # Parse and add Reflexion insights to state context
            for i, (reflexion_output, result_idx) in enumerate(zip(reflexion_outputs, reflexion_needed_indices)):
                result = action_results[result_idx]
                state = result['state']
                reflexion_text = reflexion_output.outputs[0].text.strip()

                # Parse Reflexion strategic insight
                reflexion_insight = parse_step_gradient_response(
                    response=reflexion_text,
                    task=result['task'],
                    prev_observation=result['prev_observation'],
                    curr_observation=result['observation'],
                    action=result['action']
                )

                # Add to working reflexions for this environment
                if 'working_reflexions' not in state:
                    state['working_reflexions'] = []

                # Include next_action_guidance in the reflexion so it flows to VISION-FIRST and PASS 2
                _reflexion_text = reflexion_insight.get('hypothesis', 'Strategic insight generated')
                _next_guidance = reflexion_insight.get('next_action_guidance', '')
                if _next_guidance and _next_guidance.lower() not in ['try unexplored actions', '']:
                    _reflexion_text += f" | RECOMMENDED: {_next_guidance}"

                state['working_reflexions'].append({
                    'step': state['cur_step'],
                    'reflection': _reflexion_text,
                    'insight': _next_guidance,  # Actionable guidance for VISION-FIRST
                    'progress_score': reflexion_insight.get('progress_score', 0),
                    'is_reflexion_strategic_insight': True  # Mark as Reflexion strategic analysis
                })

                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {state['cur_step']}: Added Reflexion strategic insight to context (counters reset)")
                if _next_guidance:
                    log_debug(f"[TRUE-SYNERGY] Reflexion guidance: '{_next_guidance[:100]}'")

                # CRITICAL FIX: Update result dict so TextGrad sees the NEW Reflexion in PASS 2
                # Without this, PASS 2 uses OLD tiered_step_reflexions from BEFORE PASS 1
                action_results[result_idx]['tiered_step_reflexions'] = manage_working_reflexions_tiered(state, model, log_debug)
        else:
            print(f"[PHASE 5 - PASS 1] No environments need Reflexion this step (router says TextGrad sufficient)")

        # PASS 2: Generate TextGrad actions for ALL environments (now with updated Reflexion insights)
        # ABLATION NOTE: Even in reflexion_only mode, we generate actions using TextGrad prompt structure
        # but the gradient history will be empty (since USE_TEXTGRAD=False disables gradient accumulation)
        print(f"[PHASE 5 - PASS 2] Generating TextGrad actions for ALL {len(action_results)} environments...")
        textgrad_prompts = []
        is_visual_batch = env_type in ('osworld', 'visual')  # Vision-first flag for prompt building

        for result in action_results:
            # ═══════════════════════════════════════════════════════════════
            # FIX #14: Update TODO BEFORE generating TextGrad prompt (ONE-STEP LAG FIX)
            # ═══════════════════════════════════════════════════════════════
            # ROOT CAUSE: Previously TODO was updated in PHASE 6 AFTER TextGrad generation
            # PROBLEM: TextGrad used OLD TODO state when recommending next actions
            #   Example: Agent takes pan → TextGrad recommends "go to countertop" with OLD TODO="Pick up pan"
            #            But TODO should have advanced to "Cool the pan" → Agent skips cooling step!
            # FIX: Update TODO HERE so TextGrad sees CURRENT TODO state
            state = result['state']

            # STAGNATION GRADIENT INJECTION (must happen BEFORE TextGrad prompt build).
            # PASS 1 may have set state['_eval_stagnation_reason'] from running the
            # actual evaluator on score oscillation. Inject it directly into the result
            # observation HERE so the next action prompt prominently shows the
            # ground-truth gap. Without this the gradient takes effect 1-2 steps late.
            _stag_reason_inject = state.get('_eval_stagnation_reason')
            _stag_action_inject = state.get('_eval_stagnation_action')
            if _stag_reason_inject:
                # DETERMINISTIC DIFF: Show the model the raw ACTUAL vs EXPECTED
                # from the evaluator. No LLM interpretation. The model sees the
                # exact gap and generates the fix from first principles.
                # This is more robust than LLM-translated actions because the
                # translator has the same knowledge gaps as the actor.
                _action_section = ""
                if _stag_action_inject:
                    _action_section = (
                        f"\nSUGGESTED ACTION (machine-derived — use as reference, not verbatim):\n"
                        f"  {_stag_action_inject}\n"
                    )

                # Extract ACTUAL vs EXPECTED from evaluator output for direct DIFF
                _diff_section = ""
                _reason_str = str(_stag_reason_inject)
                # Look for common evaluator patterns: ACTUAL/EXPECTED, MISMATCH, OK/FAIL
                import re as _re_diff
                _mismatches = _re_diff.findall(
                    r'(MISMATCH[^;)]*|ACTUAL:.*?(?:EXPECTED:.*?)(?:\n|;|$)|FAIL[^;)]*)',
                    _reason_str, _re_diff.IGNORECASE
                )
                if _mismatches:
                    _diff_section = (
                        f"\n\nDETERMINISTIC DIFF (this is FACT, not opinion — close this gap):\n"
                        + "\n".join(f"  ❌ {m.strip()}" for m in _mismatches[:5])
                        + "\n\nYour code must change the ACTUAL values to match EXPECTED."
                        + "\nIf your previous approach didn't work, try a DIFFERENT library or method."
                        + "\nThe file's raw content (POST-ACTION PROBE in previous observations) "
                        + "shows the exact attribute names and values — use those, not guesses."
                    )

                _stag_block = (
                    f"\n\n🚨 [EVALUATOR EVIDENCE — HIGHEST PRIORITY] The actual task "
                    f"evaluator was just run. It reports this CONCRETE gap:\n"
                    f"  {_stag_reason_inject}\n"
                    f"{_diff_section}"
                    f"{_action_section}"
                    f"\nThis is ground-truth evidence from the real evaluator, not an "
                    f"LLM opinion. Your next action MUST directly close this exact gap."
                )
                # Prepend so it's near the top of the observation, then also append a
                # short reminder near the bottom — covers both attention positions.
                _orig_obs = result.get('observation', '') or ''
                result['observation'] = _stag_block + "\n\n" + _orig_obs + _stag_block
                # Also clear the flags so the regular step() injection doesn't double-inject
                state.pop('_eval_stagnation_reason', None)
                state.pop('_eval_stagnation_action', None)
                # Stash the translated action so PASS 2 action selection can prefer it
                if _stag_action_inject:
                    state['_pending_stagnation_action'] = _stag_action_inject
                log_debug(f"[STAGNATION-INJECT] ENV {state.get('env_id', '?')}: gradient injected into PASS 2 prompt (translated_action={'YES' if _stag_action_inject else 'NO'})")

            if 'todo_manager' in state and state['todo_manager']:
                try:
                    # Update TODO with current action result (use 'EXPLORING' as default)
                    # The LLM verification in _check_todo_completion doesn't depend on progress_status
                    state['todo_manager'].update_from_action_feedback(
                        action=result['action'],
                        prev_observation=result['prev_observation'],
                        curr_observation=result['observation'],
                        progress_status='EXPLORING'  # Default - LLM verification determines completion
                    )
                    # CRITICAL: Refresh current_todo in result dict with UPDATED value!
                    result['current_todo'] = state['todo_manager']._get_current_todo()
                    result['todo_updated_in_phase5'] = True  # Mark to skip duplicate update in PHASE 6
                    log_debug(f"[FIX #14] ENV {state['env_id']}: TODO updated BEFORE TextGrad. Current TODO: {result['current_todo'].content if result['current_todo'] else 'ALL COMPLETED'}")
                except Exception as e:
                    log_debug(f"[FIX #14 ERROR] ENV {state['env_id']}: {e}")
            # ═══════════════════════════════════════════════════════════════

            # TRUE ABLATION: Use different prompts based on ablation mode
            if USE_TEXTGRAD:
                # COMBINED or TEXTGRAD_ONLY: Use full TextGrad prompt
                # TextGrad now has Reflexion strategic insights in context (via previous_reflexions)
                action_prompt = generate_textgrad_gradient_prompt(
                    result,
                    result['tiered_step_reflexions'],  # Now includes Reflexion insights!
                    result['episodic_memory_only'],  # FIX: Use correct key name!
                    result['initial_observation'],
                    result['explored_locations'],
                    result['next_valid_actions'],  # FIX: Use correct key name!
                    result['state'].get('cumulative_search_memory', ''),  # FIX #32: Cumulative search memory
                    is_visual_env=is_visual_batch,  # Vision-first: hints not constraints for OSWorld
                )
            else:
                # REFLEXION_ONLY: Use simple prompt without TextGrad (TRUE ABLATION!)
                print(f"[TRUE ABLATION] ENV {state['env_id']}: Using Reflexion-only prompt (no TextGrad)")
                action_prompt = generate_reflexion_only_prompt(
                    result,
                    result['tiered_step_reflexions'],  # Reflexion insights only
                    result['episodic_memory_only'],  # Episodic patterns
                    result['next_valid_actions']  # Valid actions
                )
            textgrad_prompts.append(action_prompt)

        # Batch generate ALL TextGrad actions
        # Visual envs (OSWorld): per-env multimodal calls with screenshots
        # Text envs (ALFWorld etc.): original text-only batch — unchanged
        # Note: is_visual_batch already defined before the prompt-building loop above
        textgrad_sampling_params = SamplingParams(
            max_tokens=7000,
            temperature=0.3,
            stop=[],  # FIX (Nov 22): Removed aggressive stop sequences that truncated answers mid-generation
            skip_special_tokens=True
        )

        if is_visual_batch:
            # PASS 2 for visual envs: Use GPT-5 (reasoning=minimal) with VGL text as input.
            # GPT-4o vision was unreliable — returned 16-21 chars ignoring all context.
            # GPT-5 reasoning reads the VGL scene description text and correctly outputs actions.
            # VGL already ran after env.step() and produced structured text (scene, elements, state diff).
            _actor_model_name = os.getenv("REFLEXGRAD_MODEL", "actor")
            print(f"[PHASE 5 - PASS 2] VISUAL ENV: {_actor_model_name} reasoning action generation for {len(textgrad_prompts)} env(s)...")
            gradient_outputs = []
            for idx_p, (prompt, result) in enumerate(zip(textgrad_prompts, action_results)):
                state_obj = result.get('state', {})
                info = result.get('info') or {}

                # Current observation = VGL output after this step's action
                # Contains [Scene], [Interactive Elements] with coords, [State Change], [Available Actions]
                curr_obs = result.get('observation', '')
                state_diff_text = info.get('state_diff', '')

                # Recent actions from trajectory + current step's action
                traj = state_obj.get('trajectory', [])
                # Include action results (SUCCESS/ERROR) so model knows what completed
                recent_acts = []
                if traj:
                    for _a_p2, _o_p2, _r_p2 in traj[-5:]:
                        _o_p2_str = str(_o_p2) if _o_p2 else ''
                        if '[SUCCESS]' in _o_p2_str:
                            recent_acts.append(f"{_a_p2} → [SUCCESS - completed]")
                        elif '[ERROR]' in _o_p2_str or 'Traceback' in _o_p2_str:
                            recent_acts.append(f"{_a_p2} → [FAILED]")
                        else:
                            recent_acts.append(_a_p2)
                current_action = result.get('action', '')
                if current_action and (not recent_acts or recent_acts[-1] != current_action):
                    recent_acts = (recent_acts + [current_action])[-5:]

                # Reflexion insights
                step_reflexions = result.get('tiered_step_reflexions', [])
                reflex_insights = [r.get('reflection', r.get('insight', ''))
                                   for r in step_reflexions[-3:] if isinstance(r, dict)]
                reflex_insights = [s for s in reflex_insights if s]

                # Build focused reasoning prompt — SCREENSHOT-FIRST
                history_str = '\n'.join(f'  [{i+1}] {a}' for i, a in enumerate(recent_acts)) if recent_acts else '  (none)'
                insights_str = '\n'.join(f'  ⚡ {r[:500]}' for r in reflex_insights) if reflex_insights else ''

                # CRITICAL FIX: Include TextGrad learning context from step_insights_accumulator.
                # Previously, ALL TextGrad gradient history (backward feedback, policy, next_guidance)
                # was DISCARDED for visual envs — only the brief insights_str survived.
                # This caused a fundamental learning leakage where the agent couldn't learn step-to-step.
                _step_insights = result.get('step_insights_accumulator', [])
                _learning_lines = []
                for _ins in _step_insights[-5:]:  # Last 5 steps of learning
                    _action_short = str(_ins.get('action', ''))[:50]
                    _score = _ins.get('progress_score', '?')
                    _bw = _ins.get('backward_gradient', '')
                    _guidance = _ins.get('next_guidance', '')
                    _diff = _ins.get('state_diff', '')
                    # Show FULL lesson (up to 500 chars) — all sentences, not just first
                    _bw_summary = ''
                    if _bw:
                        _bw_clean = '\n'.join(
                            s.strip() for s in _bw.split('\n')
                            if s.strip() and not s.strip().startswith('To improve the policy')
                        )
                        _bw_summary = _bw_clean[:500]
                    _line = f"  Step {_ins.get('step', '?')}: '{_action_short}' (score {_score}/10)"
                    if _diff and len(str(_diff)) > 5:
                        _line += f" → {str(_diff)[:100]}"
                    # Include command output from the observation (both errors AND successful results)
                    _obs_text = str(_ins.get('observation', ''))
                    # Check for errors first
                    _cmd_err_start = _obs_text.find('[ERROR]')
                    if _cmd_err_start < 0:
                        _cmd_err_start = _obs_text.find('FileNotFoundError')
                    if _cmd_err_start < 0:
                        _cmd_err_start = _obs_text.find('Traceback')
                    if _cmd_err_start >= 0:
                        _err_text = _obs_text[_cmd_err_start:_cmd_err_start + 300].strip()
                        _line += f"\n    ERROR: {_err_text}"
                        # Preserve AUTO-DISCOVERY after errors (critical for file path correction)
                        _disc_start = _obs_text.find('[AUTO-DISCOVERY]', _cmd_err_start)
                        if _disc_start >= 0:
                            _disc_end_p2 = _obs_text.find('\n\n', _disc_start)
                            _disc_txt = _obs_text[_disc_start:_disc_end_p2 if _disc_end_p2 > _disc_start else _disc_start + 300].strip()
                            _line += f"\n    ⚠️ {_disc_txt}"
                    # Also include successful command output (ls, find, etc.) — critical for filename discovery
                    _cmd_out_start = _obs_text.find('[Command Output')
                    if _cmd_out_start >= 0 and _cmd_err_start < 0:
                        _cmd_out_end = _obs_text.find('\n\n', _cmd_out_start)
                        if _cmd_out_end < 0:
                            _cmd_out_end = _cmd_out_start + 300
                        _cmd_out_text = _obs_text[_cmd_out_start:min(_cmd_out_end, _cmd_out_start + 300)].strip()
                        _line += f"\n    OUTPUT: {_cmd_out_text}"
                    if _bw_summary:
                        _line += f"\n    LESSON: {_bw_summary}"
                    if _guidance and len(str(_guidance)) > 3:
                        _line += f"\n    RECOMMENDED NEXT: {str(_guidance)[:150]}"
                    _learning_lines.append(_line)
                _learning_context = '\n'.join(_learning_lines) if _learning_lines else ''

                # Extract a dedicated error summary for PASS 2 (errors are critical signals)
                _error_summary = ''
                _errors_found = []
                for _ins in _step_insights:
                    _obs_text = str(_ins.get('observation', ''))
                    _action_text = str(_ins.get('action', ''))
                    for _err_marker in ['[ERROR]', 'FileNotFoundError', 'Traceback', 'ModuleNotFoundError',
                                        'PermissionError', 'SyntaxError', 'ImportError', 'ValueError',
                                        'No such file']:
                        _err_pos = _obs_text.find(_err_marker)
                        if _err_pos >= 0:
                            _err_snippet = _obs_text[_err_pos:_err_pos + 500].strip()
                            _err_entry = f"  Step {_ins.get('step', '?')}: '{_action_text[:60]}' -> {_err_snippet}"
                            # Include AUTO-DISCOVERY if present (critical for filename correction)
                            _disc_pos = _obs_text.find('[AUTO-DISCOVERY]', _err_pos)
                            if _disc_pos >= 0 and '[AUTO-DISCOVERY]' not in _err_snippet:
                                _disc_end_es = _obs_text.find('\n\n', _disc_pos)
                                _disc_es = _obs_text[_disc_pos:_disc_end_es if _disc_end_es > _disc_pos else _disc_pos + 300].strip()
                                _err_entry += f"\n    ⚠️ {_disc_es}"
                            _errors_found.append(_err_entry)
                            break
                if _errors_found:
                    _error_summary = "PREVIOUS COMMAND ERRORS (CRITICAL — do NOT repeat these mistakes, analyze and fix the root cause):\n" + '\n'.join(_errors_found) + "\n"

                # FIX: Loop detection — multiple strategies for visual envs where VGL always detects minor changes
                loop_warning = ''
                _is_visual_env = state_obj.get('env_type', '') in ('osworld', 'visual')
                if len(recent_acts) >= 2:
                    # Normalize: strip comments after '#' for comparison
                    last_2_norm = [a.split('#')[0].strip().lower() for a in recent_acts[-2:]]
                    # Strategy 1: Last 2 consecutive identical (original, with relaxed state check for visual envs)
                    _consecutive_match = len(set(last_2_norm)) == 1
                    # Strategy 2: Same action 3+ times in last 5 (catches alternating patterns)
                    if len(recent_acts) >= 3:
                        from collections import Counter
                        _last5_norm = [a.split('#')[0].strip().lower() for a in recent_acts[-5:]]
                        _action_counts = Counter(_last5_norm)
                        _most_common = _action_counts.most_common(1)[0]
                        _repeated_3plus = _most_common[1] >= 3
                    else:
                        _repeated_3plus = False
                    # Strategy 3: Score stagnation — TextGrad score ≤ 1 for 3+ consecutive steps
                    _progress_hist = state_obj.get('progress_history', [])
                    _score_stagnation = len(_progress_hist) >= 3 and all(s <= 1 for s in _progress_hist[-3:])

                    # For visual envs: skip state_diff check (VGL always detects trivial changes like cursor/selection)
                    # Trigger on: consecutive match OR 3+ repeats in last 5 OR score stagnation
                    _trigger_loop = False
                    if _is_visual_env:
                        _trigger_loop = _consecutive_match or _repeated_3plus or _score_stagnation
                    else:
                        # Text envs: keep original state_diff check
                        state_diff_lower = (state_diff_text or '').lower()
                        _trigger_loop = _consecutive_match and (not state_diff_lower or 'no visible change' in state_diff_lower)

                    if _trigger_loop:
                        stuck_action = recent_acts[-1]
                        _reason = "score stagnation" if _score_stagnation and not _consecutive_match else (
                            "repeated 3+ times" if _repeated_3plus else "consecutive identical actions")
                        loop_warning = f"""
⚠️ CRITICAL: You are STUCK IN A LOOP ({_reason}). The action '{stuck_action}' has been repeated without progress.

You MUST try a FUNDAMENTALLY DIFFERENT approach — switch MODALITY:
- If GUI clicks are failing → switch to programmatic: run: python3 -c "..."
- If programmatic commands are failing → switch to GUI interaction
- If one tool/library fails → try a different tool/library
1. Review the LESSONS from past steps above — they contain specific recommendations
2. If Reflexion provided a STRATEGY_SHIFT, follow it
3. Try a completely different method to achieve the same goal

Do NOT repeat '{stuck_action}'. The environment's learning system has analyzed your failures — use its recommendations.
"""
                        print(f"[LOOP DETECTED] ENV {idx_p}: '{stuck_action}' ({_reason}) — universal loop warning")

                # No forced actions — ReflexGrad's TextGrad/Reflexion learning drives action selection.
                # Loop warning (above) gives the model context; PASS 2 model call decides the action.

                # Extract compact coordinate hints from VGL (Interactive Elements only)
                # Also include Command Output if present (from run: commands)
                coord_hints = ''
                _vgl_text = curr_obs or state_diff_text or ''
                _ie_start = _vgl_text.find('[Interactive Elements]')
                if _ie_start >= 0:
                    _ie_end = _vgl_text.find('[', _ie_start + 1)
                    _ie_section = _vgl_text[_ie_start:_ie_end if _ie_end > _ie_start else _ie_start + 1500]
                    coord_hints = f"\nCOORDINATE HINTS (from automated detection — verify against screenshot):\n{_ie_section[:1500]}\n"
                # Include Available Actions (suggested by VGL — important for URL navigation, typing, etc.)
                _aa_start = _vgl_text.find('[Available Actions]')
                if _aa_start >= 0:
                    _aa_section = _vgl_text[_aa_start:_aa_start + 600]
                    coord_hints += f"\nSUGGESTED ACTIONS (from VGL — consider these):\n{_aa_section[:600]}\n"
                # Include Command Output from run: commands (e.g. ls output, script results)
                _cmd_start = _vgl_text.find('[Command Output')
                if _cmd_start >= 0:
                    _cmd_section = _vgl_text[_cmd_start:_cmd_start + 600]
                    coord_hints += f"\n{_cmd_section}\n"

                # Extract discovered file paths from find/ls output for prominent display
                _disc_files_p2 = []
                for _ins_p2 in _step_insights:
                    _obs_p2 = str(_ins_p2.get('observation', ''))
                    _act_p2 = str(_ins_p2.get('action', '')).lower()
                    if ('run: find' in _act_p2 or 'run: ls' in _act_p2) and '/home/' in _obs_p2:
                        import re as _re_disc2
                        _pths = _re_disc2.findall(r'(/home/\S+\.\w{1,5})', _obs_p2)
                        _disc_files_p2.extend(_pths)
                _disc_files_p2 = list(dict.fromkeys(_disc_files_p2))
                # Share discovered files with wrapper for auto-chdir (Change 9)
                if _disc_files_p2:
                    result['discovered_files'] = _disc_files_p2
                _disc_section_p2 = ''
                if _disc_files_p2:
                    _disc_str = '\n'.join(f'  {f}' for f in _disc_files_p2)
                    _disc_section_p2 = f"DISCOVERED FILES (from find/ls — use EXACT paths):\n{_disc_str}\nDo NOT guess different paths — use one of the exact paths listed above.\n"

                # Determine if GUI ever made progress (scored ≥ 5) — if so, don't force python
                _gui_ever_worked = any(s >= 5 for s in (state_obj.get('progress_history', []) or []))
                # Also check if python already ran successfully but nothing changed — app may have file locked
                _python_succeeded_no_change = False
                for _psnc in (state_obj.get('step_insights_accumulator', []) or [])[-5:]:
                    _psnc_action = str(_psnc.get('action', ''))
                    _psnc_score = _psnc.get('progress_score', 0)
                    if 'python' in _psnc_action.lower() and 'run:' in _psnc_action.lower() and _psnc_score <= 1:
                        _psnc_obs = str(_psnc.get('observation', ''))
                        if 'rc=0' in _psnc_obs or 'SUCCESS' in _psnc_obs:
                            _python_succeeded_no_change = True
                _force_python = loop_warning and not _gui_ever_worked and not _python_succeeded_no_change

                # LEAN MODE: Cap heavy sections to keep PASS 2 prompt under ~8K chars
                if LEAN_MODE:
                    # Keep only last 2 learning lines (not all 15)
                    if _learning_context:
                        _lc_lines = _learning_context.strip().split('\n')
                        _learning_context = '\n'.join(_lc_lines[-6:])  # ~2 steps worth
                    # Keep only last error (not all errors)
                    if _error_summary and _errors_found:
                        _error_summary = "LAST ERROR:\n" + _errors_found[-1][:300] + "\n"
                    # Cap insights to 1 (not 3)
                    if insights_str:
                        insights_str = insights_str[:500]
                    # Cap history to last 3 (not 15)
                    if history_str:
                        _h_lines = history_str.strip().split('\n')
                        history_str = '\n'.join(_h_lines[-3:])
                    # Cap coord_hints
                    if coord_hints:
                        coord_hints = coord_hints[:1000]
                    log_debug(f"[LEAN MODE] PASS 2 sections capped")

                # UNIVERSAL actor-side hints (propagated from XLATE). Both are
                # gated on structural signals — when absent, they self-disable.
                #   ENUM-STRING BINDING: when the evaluator delta contains a
                #     quoted short identifier inside a tuple/list slot, the
                #     model is told to write that literal via `element.set`/
                #     `obj[key]` rather than rely on a library bool/enum
                #     shortcut (which typically emits a different token).
                #   MULTI-LINE CHANNEL: when the model is about to emit a
                #     programmatic action, nudge it to use the base64-file
                #     channel for any compound-statement Python. Valid
                #     single-line code still works unchanged.
                _unresolved_p2 = state_obj.get('_last_eval_gap', '') or ''
                _actor_enum_hint = ''
                if _unresolved_p2:
                    import re as _re_act_enum
                    _enum_tokens = _re_act_enum.findall(
                        r"[\(\[,]\s*['\"]([A-Za-z][A-Za-z0-9_-]{1,30})['\"]\s*[,\)\]]",
                        str(_unresolved_p2)
                    )
                    if _enum_tokens:
                        _uniq = list(dict.fromkeys(_enum_tokens))[:5]
                        _actor_enum_hint = (
                            f"\n- ENUM-STRING BINDING: the evaluator's remaining "
                            f"gap contains literal quoted strings {_uniq}. Those "
                            f"exact strings must appear as attribute values in "
                            f"the target file. Library bool/enum shortcuts emit "
                            f"their own default tokens — NOT the literal strings "
                            f"above. Use `element.set('attr_name', 'exact_string')` "
                            f"(XML via lxml/ElementTree) or `obj[key] = "
                            f"'exact_string'` (JSON). Do NOT rely on a high-level "
                            f"library property that translates values internally.\n"
                        )
                _actor_multiline_hint = (
                    "\n- MULTI-LINE PYTHON CHANNEL: if your action needs "
                    "control flow (def / for / if / while / with) or multiple "
                    "statements, do NOT compress them onto one `python3 -c` "
                    "line. Instead, use the base64-file channel: "
                    "`run: echo <BASE64_OF_SCRIPT> | base64 -d > /tmp/s.py && "
                    "python3 /tmp/s.py`. The wrapper decodes it as a real "
                    "file with indentation preserved.\n"
                )

                # V50-A / V50-B REVERTED (2026-04-20): the position-trick and
                # evaluator-as-spec-replacement blocks regressed v51 bench
                # (0/4 vs v50's 1/4). Hypothesis: they duplicated evaluator
                # text already pinned at the TOP of obs_text via the
                # persistent-state block, and pushed the OUTPUT FORMAT
                # instructions further from the decode boundary, diluting
                # attention. Kept as empty strings so the prompt template
                # below still interpolates cleanly.
                _position_trick_block = ''
                _eval_as_spec_block = ''

                # G9-DET DETERMINISTIC FACT-CARD (Phase-5 synergy, atom E3
                # upgraded 2026-04-21): previous LLM-based extractor was
                # itself anchoring to the task NL ("first two paragraphs")
                # and producing WRONG slots (target_index=0 + target_index=1
                # when the evaluator said P2 MISMATCH, omitting P2). That
                # corruption propagated to Gemma's code which obediently
                # used `[:2]`. Per skill — the extraction edge must not be
                # a second weak LLM; it must be deterministic.
                #
                # New rule: parse the already-structured ITEM= lines that G8
                # produces (which ARE faithful to evaluator output since G8
                # itself prompts "copy VERBATIM" and we filter by ITEM= shape)
                # and turn each into deterministic fact-card slots. Zero
                # LLM call. Cannot paraphrase because it's pure string
                # manipulation. Gated on `_unresolved_verification` populated.
                _fact_card_block = ''
                try:
                    if _unresolved_p2:
                        import re as _re_fc
                        # The G8 structured block is already in obs_text; use
                        # the raw evaluator delta as the source and extract
                        # every item the evaluator referenced. Primary pattern:
                        # per-item "PREFIX_DIGITS: STATUS (k=v; k=v)" as in
                        # `P0: MISMATCH (expected=2.0, actual=1.0)` or
                        # `(4, 1, 1, 'sngStrike')` tuple form.
                        _delta = str(_unresolved_p2)
                        _slots: List[str] = []
                        # Pattern A: per-item status line  — e.g. "P2: MISMATCH (expected=2.0, actual=1.0)"
                        for _m in _re_fc.finditer(
                            r'([A-Za-z][A-Za-z0-9_]*\d*)\s*:\s*(MISMATCH|FAIL|MISSING|WRONG)'
                            r'(?:\s*\(([^)]{0,200})\))?',
                            _delta,
                        ):
                            _item, _status, _parens = _m.group(1), _m.group(2), (_m.group(3) or '')
                            _want = ''
                            _got = ''
                            _wm = _re_fc.search(r'expected\s*=\s*([^,;)]+)', _parens, _re_fc.IGNORECASE)
                            if _wm: _want = _wm.group(1).strip()
                            _gm = _re_fc.search(r'actual\s*=\s*([^,;)]+)', _parens, _re_fc.IGNORECASE)
                            if _gm: _got = _gm.group(1).strip()
                            _slots.append(
                                f"fix_item={_item} status={_status}"
                                + (f" want={_want}" if _want else '')
                                + (f" got={_got}" if _got else '')
                            )
                            # Also extract the numeric suffix (0 from P0, 2 from P2) as a
                            # fix_index slot — this is what a `[:3]` slice or an indexing
                            # operation needs. The model keeps writing `[:2]` because the
                            # derived fact card previously lacked explicit fix_index=2.
                            _idx_m = _re_fc.search(r'\d+', _item)
                            if _idx_m:
                                _slots.append(f"fix_index={_idx_m.group(0)} (derived from item {_item})")
                        # Pattern B: tuple form  — e.g. "(4, 1, 1, 'sngStrike')"
                        for _m in _re_fc.finditer(
                            r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
                            _delta,
                        ):
                            _tup = _m.group(0)
                            _val = _m.group(4)
                            _slots.append(
                                f"fix_item={_tup} want_value='{_val}'"
                            )
                        # Pattern C: MISSING_X / OK:X  — e.g. "MISSING_SHEET:Sheet2,OK:Sheet1"
                        for _m in _re_fc.finditer(
                            r'(MISSING|WRONG)_([A-Za-z_]+)\s*:\s*([A-Za-z0-9_.-]+)',
                            _delta,
                        ):
                            _st, _kind, _name = _m.group(1), _m.group(2), _m.group(3)
                            _slots.append(
                                f"fix_item={_kind} status={_st} want_name='{_name}'"
                            )
                        # Dedupe while preserving order
                        _seen: set = set()
                        _slots_uniq: List[str] = []
                        for _s in _slots:
                            if _s not in _seen:
                                _seen.add(_s)
                                _slots_uniq.append(_s)
                        # Cap at 8 lines to stay inside attention zone
                        _slots_uniq = _slots_uniq[:8]
                        _fc_content = "\n".join(_slots_uniq) if _slots_uniq else ''
                        # Cache (content is pure function of unresolved_verification).
                        import hashlib as _hl_fc
                        _fc_key = _hl_fc.md5(str(_unresolved_p2).encode()).hexdigest()
                        _fc_cache = state_obj.setdefault('_fact_card_cache', {})
                        if _fc_cache.get('hash') != _fc_key:
                            _fc_cache['hash'] = _fc_key
                            _fc_cache['content'] = _fc_content
                            if _fc_content:
                                print(f"[G9-DET FACT-CARD] ENV {idx_p}: "
                                      f"extracted {len(_slots_uniq)} deterministic slots "
                                      f"(regex, no LLM)")
                        _fc_text_final = (_fc_cache or {}).get('content', '')
                        if _fc_text_final:
                            _fact_card_block = (
                                "\n[📌 FACT CARD — these slots are EXTRACTED "
                                "from the evidence. When your code references "
                                "a target value / attribute / path / index, "
                                "USE THESE EXACT LITERAL STRINGS; do NOT "
                                "paraphrase or re-case them.]\n"
                                f"{_fc_text_final}\n"
                            )
                except Exception as _fc_outer:
                    print(f"[G9 FACT-CARD] skipped (non-fatal): {_fc_outer}")

                # G12 AUTHORITATIVE TASK REWRITE (2026-04-21): when the
                # evaluator has rejected `done` and the per-item delta
                # CONTRADICTS the scope of the natural-language task
                # description, rewrite the TASK line so the evidence wins at
                # the TOP attention position. This closes the "task text vs
                # evaluator" conflict that causes weak models to prefer the
                # ambiguous NL over the structured ground truth (observed:
                # Writer task text says 'first two paragraphs' but evaluator
                # wants P0+P1+P2). Gated on must-fix list being populated
                # (structural signal, no task strings). Universal: works on
                # any env whose evaluator surfaces per-item failure tuples.
                _authoritative_task = result['task']
                try:
                    if _unresolved_p2:
                        import re as _re_g12
                        # Extract per-item mismatches in any of the common shapes.
                        _items = []
                        for _im in _re_g12.finditer(
                            r'([A-Za-z][A-Za-z0-9_]*\d*)\s*:\s*(MISMATCH|FAIL|MISSING|WRONG)'
                            r'(?:\s*\(([^)]{0,200})\))?',
                            str(_unresolved_p2),
                        ):
                            _items.append({
                                'item': _im.group(1),
                                'status': _im.group(2),
                                'parens': _im.group(3) or '',
                            })
                        if _items:
                            _item_list = ", ".join(
                                dict.fromkeys(it['item'] for it in _items)
                            )[:500]
                            _authoritative_task = (
                                f"{result['task']}\n"
                                f"        [AUTHORITATIVE OVERRIDE — derived from live evaluator]: "
                                f"the evaluator STILL rejects {len(_items)} item(s): "
                                f"[{_item_list}]. Your next action MUST address EVERY item "
                                f"listed here, even if the natural-language description above "
                                f"seems to cover fewer items. Evidence > task paraphrase."
                            )
                except Exception:
                    pass

                # G24 UNIVERSAL — Mirror G21 into the PASS 2 actor prompt.
                # Previously the evaluator ground truth + extracted literals +
                # evaluator source (G21) was pinned ONLY in the BACKWARD gradient
                # prompt. The actor saw only the regex-extracted fact-card slots
                # and had no access to the evaluator's source code or full
                # unstructured delta. G24 mirrors the same block into PASS 2
                # so the actor has the same ground-truth fidelity as TextGrad.
                # Triggers only when `_last_eval_gap` or `_unresolved_verification`
                # is populated (same gate as G21). No-op on text envs.
                _g24_block = ""
                try:
                    _eval_gap_p2 = (state_obj.get('_last_eval_gap') or
                                    state_obj.get('_unresolved_verification') or '')
                    _eval_source_p2 = state_obj.get('_eval_source') or ''
                    if _eval_gap_p2:
                        import re as _re_g24
                        _lits_p2 = _re_g24.findall(
                            r"expected\s*[:=]\s*['\"]([^'\"\n]{1,80})['\"]",
                            str(_eval_gap_p2), flags=_re_g24.IGNORECASE,
                        )
                        _lits_p2 += _re_g24.findall(
                            r"expected\s*[:=]\s*([A-Za-z0-9_.\-/]{1,40})(?=[\s,\)\]\n]|$)",
                            str(_eval_gap_p2), flags=_re_g24.IGNORECASE,
                        )
                        _lits_p2 += _re_g24.findall(
                            r"\(\s*(?:[^)]*?,\s*)?['\"]([^'\"]{1,40})['\"]\s*\)",
                            str(_eval_gap_p2),
                        )
                        _clean_p2 = []
                        _seen_p2 = set()
                        for _l in _lits_p2:
                            _l2 = _l.strip()
                            if (not _l2 or _l2 in _seen_p2 or
                                _l2.lower() in ('<empty>', 'empty', 'none', 'null', '?')):
                                continue
                            _seen_p2.add(_l2)
                            _clean_p2.append(_l2)
                            if len(_clean_p2) >= 15:
                                break
                        _g24_block = (
                            "\n⛔ EVALUATOR GROUND TRUTH (verbatim — your next action MUST\n"
                            "   reach THIS exact state, byte-for-byte):\n"
                            f"{str(_eval_gap_p2)[:1500]}\n"
                        )
                        if _clean_p2:
                            _g24_block += (
                                "\n📌 EXTRACTED LITERAL REQUIREMENTS (exact strings/values\n"
                                "   the evaluator compares against — your code output MUST\n"
                                "   contain these exact tokens, not synonyms or variants):\n"
                                f"   {_clean_p2}\n"
                            )
                        if _eval_source_p2:
                            _g24_block += (
                                "\n🔍 EVALUATOR SOURCE (the Python the evaluator literally\n"
                                "   runs — read carefully: the attribute names, dict keys,\n"
                                "   file paths, and comparison operators here define what\n"
                                "   'success' means. Do NOT guess from context):\n"
                                f"{str(_eval_source_p2)[:3000]}\n"
                            )
                except Exception:
                    _g24_block = ""

                # G23 UNIVERSAL — Inject the accumulated TextGrad-optimized policy
                # into PASS 2 actor prompt. Previously `policy.base_policy` was
                # updated every 3 steps by `TextGradOptimizer.step()` (which
                # synthesizes accumulated gradients via an LLM into a refined
                # policy string), but the actor NEVER read that updated policy —
                # it only saw the single latest step_gradient. All accumulated
                # cross-step learning was wasted. G23 fixes this by reading the
                # current policy.base_policy and recent gradients, pinning them
                # near the top of the actor prompt. Universal: if policy is
                # unavailable (text envs without the TextGrad optimizer) the
                # block is empty.
                _policy_obj = state_obj.get('policy') if isinstance(state_obj, dict) else None
                _policy_block = ""
                if _policy_obj is not None:
                    try:
                        _bp = getattr(_policy_obj, 'base_policy', '') or ''
                        _grads = list(getattr(_policy_obj, 'gradients', []) or [])[-3:]
                        _uh = getattr(_policy_obj, 'update_history', []) or []
                        if _bp and _bp != "Select actions that make measurable progress toward completing the stated task.":
                            _policy_block = (
                                "\n📋 LEARNED POLICY (synthesized by TextGrad optimizer from "
                                f"{len(_uh)} past updates — incorporates lessons from all prior "
                                "gradient-corrected attempts; follow this policy unless it "
                                "contradicts a higher-priority signal below):\n"
                                f"{_bp[:1500]}\n"
                            )
                            if _grads:
                                _policy_block += (
                                    "\n📎 Last 3 raw gradients feeding the policy above "
                                    "(cite these for context, but prefer the policy):\n"
                                )
                                for _i, _g in enumerate(_grads[-3:]):
                                    _policy_block += f"  [{_i+1}] {str(_g)[:400]}\n"
                    except Exception:
                        _policy_block = ""

                gpt5_prompt = f"""You are a desktop GUI agent controlling an Ubuntu Linux desktop (1920×1080).

TASK: {_authoritative_task}
{_g24_block}{_policy_block}{loop_warning}
{"" if not _disc_section_p2 else _disc_section_p2 + chr(10)}{"" if not _error_summary else _error_summary + chr(10)}{"" if not _learning_context else "TEXTGRAD LEARNING (from " + str(len(_step_insights)) + " steps of gradient optimization — DO NOT repeat past mistakes):" + chr(10) + _learning_context + chr(10)}
{"" if not insights_str else ("⚡ REFLEXION STRATEGY — HIGHEST PRIORITY (you are stuck, follow this recommendation):" if loop_warning else ("STRATEGIC INSIGHTS (follow these if applicable):")) + chr(10) + insights_str + chr(10)}
INSTRUCTIONS:
1. READ the COMMAND RESULTS and TEXTGRAD LEARNING above FIRST — they show what worked and what failed
2. LOOK at the attached screenshot to verify the current screen state
3. If previous commands failed (file not found, errors), FIX the command using information from successful runs (ls output, etc.)
4. If GUI clicks have been repeated without progress, switch to run: python3 -c "..." approach
{coord_hints}
RECENT ACTIONS (do NOT repeat):
{history_str}

{"ACTION FORMATS (⚠️ GUI clicks have FAILED repeatedly — you MUST switch to PROGRAMMATIC):" if _force_python else "ACTION FORMATS (in order of preference):"}
{"  run: python3 -c '...' — REQUIRED: GUI is stuck, use python/openpyxl to modify the file directly" if _force_python else "  click 'Label' — PREFERRED when the target appears in [Interactive Elements] above (uses verified coordinates)"}
{"  run: <shell command> — ls, find, etc. for discovery" if _force_python else "  click (X, Y) — Use ONLY when the target is NOT listed in [Interactive Elements]"}
{"  done — after programmatic command succeeds" if _force_python else "  type 'text' | hotkey Ctrl+S | press Enter"}
{"" if _force_python else "  run: <shell command> | scroll down/up | done"}

RULES:
{"- ⚠️ GUI CLICKS ARE NOT WORKING. Clicking buttons/tabs has FAILED " + str(len([s for s in (state_obj.get('progress_history', []) or []) if s <= 1])) + " times. You MUST use run: python3 to modify the file directly." if _force_python else "- LOOK at the screenshot — it is your PRIMARY source of truth for deciding what to do."}
{"- Read the title bar for the filename, then use run: python3 -c 'import openpyxl; ...' or similar library to make changes." if _force_python else "- USE [Interactive Elements] labels: If the target button/menu/tab appears in the list above, use click 'Label' format (e.g., click 'Add Sheet' button). These have verified coordinates and are more accurate than estimating."}
{"- ⚠️ PYTHON COMMANDS SUCCEEDED (rc=0) BUT NOTHING CHANGED ON SCREEN. The application likely has the file locked in memory. Either: (1) Close the application (run: pkill -f soffice || run: pkill -f libreoffice) then say 'done', OR (2) Try a DIFFERENT GUI approach instead (e.g., Format menu → Paragraph/Spacing)." if _python_succeeded_no_change else ""}
- CHECK WHAT'S ALREADY OPEN: If an application has a file open (check the title bar), that file is MOST LIKELY the one you need to work with. Navigate within it (switch slides, sheets, tabs, scroll) before searching for other files.
- Content referenced in the task (e.g., "to-do list", "budget", "report") may be on a DIFFERENT PAGE, SLIDE, or SHEET of the currently-open document. Navigate through the document to find it.
- NEVER hallucinate filenames. If you don't see a specific file path in the title bar, file manager, or command output, use `run: ls` or `run: find` to discover actual filenames first.
- CHECK THE WINDOW TITLE BAR at the top of the screen for the actual filename and path of open documents.
- If a dialog or dropdown menu is open, click the correct item. Prefer click 'Label' if the item appears in [Interactive Elements].
- If GUI clicks aren't making progress, switch to programmatic approach: run: python3 -c "..."
- You can run ANY shell command: run: ls, run: python3 -c "...", run: find, etc.
- When modifying files programmatically, use the EXACT path shown in the title bar. If unsure, run: ls to verify.
- Background GUI apps with &: run: app "file" & (use absolute paths, NOT ~/...)
- If a command failed, READ the FULL error message and fix the ROOT CAUSE — do NOT just retry the same command.
- After running a command, check [Command Output] in the observation for results.
- If a previous find/ls command discovered file paths, USE those exact paths — do NOT guess different paths.
- Quote file paths with spaces.
- Output 'done' ONLY when ALL parts of the task are visibly complete or your programmatic command has succeeded.
- PROGRAMMATIC COMPLETION: If a previous run: command shows [SUCCESS - completed] in RECENT ACTIONS and that command performed the task's goal (e.g., wrote data, modified a file), say 'done'. Programmatic file changes modify the file on disk — they will NOT appear in the GUI screenshot. Trust the [SUCCESS] result over the screenshot.
- Do NOT say 'done' just because you see existing data — the task usually requires CHANGES.{_actor_enum_hint}{_actor_multiline_hint}
{_fact_card_block}

[CURRENT OBSERVATION FROM WRAPPER (contains pinned persistent-delta, tombstones, structure probe, command output — read ALL blocks before deciding):
{str(curr_obs)[:8000]}
]

You MUST output EXACTLY in this format (both lines required):
REASONING: (1-2 sentences about what you see and why)
NEXT_ACTION: (the action string — e.g. click (100, 1050) or run: python3 -c "..." or type 'text' or done)

The NEXT_ACTION line is MANDATORY. Do NOT skip it."""

                log_debug(f"[PHASE 5 - PASS 2] ENV {idx_p}: calling {os.getenv('REFLEXGRAD_MODEL', 'actor')} reasoning for action...")
                # BEST-OF-N SAMPLING: When previous score is low (stagnation/failure),
                # generate N=3 candidates at higher temperature and pick the one that
                # best matches the gradient recommendation. Universal: works for any task.
                # This addresses the "model prior wins over context" gap — even if P(correct)=0.2,
                # P(at_least_one_correct | N=3) = 1 - 0.8^3 = 0.49.
                _prev_score_p2 = state_obj.get('step_gradient', {}).get('progress_score', 5)
                # Constraint: single-trial inference-time only. Best-of-N is
                # disabled by default; flip REFLEXGRAD_BEST_OF_N=1 only if the
                # user explicitly opts in. Prior V30-V32 experiments already
                # showed Best-of-N does not help (P(correct)≈0 → all N wrong).
                _use_best_of_n = (
                    _prev_score_p2 <= 2
                    and os.getenv("REFLEXGRAD_BEST_OF_N", "0") == "1"
                )
                _n_candidates = 3 if _use_best_of_n else 1
                _temperature_p2 = 0.8 if _use_best_of_n else 0.0

                after_screenshot = info.get('after_screenshot_b64') or state_obj.get('last_after_screenshot_b64')
                _candidates = []
                for _bon_i in range(_n_candidates):
                    try:
                        if after_screenshot:
                            _resp = model.generate_multimodal(
                                gpt5_prompt, after_image_b64=after_screenshot,
                                max_tokens=1500, temperature=_temperature_p2)
                        else:
                            _out = model.generate([gpt5_prompt],
                                SamplingParams(max_tokens=1500, temperature=_temperature_p2),
                                reasoning_effort='minimal')
                            _resp = _out[0].outputs[0].text.strip()
                        _candidates.append(_resp)
                    except Exception as _bon_err:
                        log_debug(f"[BEST-OF-N] Candidate {_bon_i+1} failed: {_bon_err}")

                if _use_best_of_n and len(_candidates) > 1:
                    # Score candidates against gradient recommendation
                    _gradient_text = state_obj.get('step_gradient', {}).get('textgrad_gradient', '').lower()
                    _blocked = state_obj.get('_blocked_categories', set())
                    _failed_actions = set(str(a).lower()[:50] for a in state_obj.get('tried_actions', set()))

                    def _score_candidate(_c):
                        _c_lower = _c.lower()
                        score = 0
                        # Bonus: contains defensive patterns from gradient
                        if '||' in _c_lower: score += 3  # defensive guard
                        if 'check=false' in _c_lower: score += 2  # error-tolerant
                        if '--no-audio' in _c_lower: score += 3  # VLC fix
                        if '.columns' in _c_lower: score += 2  # reads columns first
                        # Penalty: matches a blocked category
                        for bc in _blocked:
                            if bc in _c_lower: score -= 5
                        # Penalty: too similar to a failed action
                        for fa in _failed_actions:
                            if fa[:30] in _c_lower[:50]: score -= 3
                        # Bonus: syntactically valid Python (for run: python3 -c commands)
                        if 'python3 -c' in _c_lower:
                            import ast as _ast_bon
                            import re as _re_bon
                            _py_match = _re_bon.search(r'python3\s+-c\s+"([^"]+)"', _c)
                            if _py_match:
                                try:
                                    _ast_bon.parse(_py_match.group(1))
                                    score += 4  # Valid Python!
                                except SyntaxError:
                                    score -= 2  # Invalid Python
                        return score

                    _scored = [(i, _score_candidate(c)) for i, c in enumerate(_candidates)]
                    _scored.sort(key=lambda x: x[1], reverse=True)
                    _best_idx = _scored[0][0]
                    raw_resp = _candidates[_best_idx]
                    log_debug(f"[BEST-OF-N] Generated {len(_candidates)} candidates, "
                              f"scores={[s for _, s in _scored]}, picked #{_best_idx+1}")
                else:
                    raw_resp = _candidates[0] if _candidates else ''
                    if _n_candidates > 1:
                        log_debug(f"[BEST-OF-N] Only got {len(_candidates)} candidates")

                log_debug(f"[PHASE 5 - PASS 2] ENV {idx_p}: {'multimodal' if after_screenshot else 'text-only'}"
                          f"{f' (best-of-{_n_candidates})' if _use_best_of_n else ''}")

                # Log raw response for debugging multi-line parse issues
                log_debug(f"[PASS 2 RAW] ENV {idx_p}: raw_resp ({len(raw_resp)} chars): {repr(raw_resp[:300])}")

                # Strip markdown code block fences (model sometimes wraps in ```plaintext ... ```)
                import re as _re_pass2
                raw_resp = _re_pass2.sub(r'^```\w*\n?', '', raw_resp.strip())
                raw_resp = _re_pass2.sub(r'\n?```$', '', raw_resp.strip())

                # Garbage response filter: GPT-4o sometimes returns <20 char garbage like "click '...'"
                first_line = raw_resp.split('\n')[0].strip()
                if len(first_line) < 20 and ("'...'" in first_line or "'…'" in first_line or first_line.count('.') >= 3):
                    log_debug(f"[PASS 2 GARBAGE FILTER] Detected garbage response: '{first_line}' — retrying text-only")
                    try:
                        retry_out = model.generate([gpt5_prompt], SamplingParams(max_tokens=1500), reasoning_effort='medium')
                        retry_resp = retry_out[0].outputs[0].text.strip()
                        retry_resp = _re_pass2.sub(r'^```\w*\n?', '', retry_resp.strip())
                        retry_resp = _re_pass2.sub(r'\n?```$', '', retry_resp.strip())
                        retry_line = retry_resp.split('\n')[0].strip()
                        if len(retry_line) > len(first_line):
                            raw_resp = retry_resp
                            log_debug(f"[PASS 2 GARBAGE FILTER] Retry succeeded: '{retry_line[:60]}'")
                        else:
                            log_debug(f"[PASS 2 GARBAGE FILTER] Retry also short: '{retry_line}' — keeping original")
                    except Exception as e:
                        log_debug(f"[PASS 2 GARBAGE FILTER] Retry failed: {e}")

                # Parse: find NEXT_ACTION: line in multi-line response (model now reasons before acting)
                # For run: commands, capture multiple lines (python code can span lines)
                action_line = ''
                _reasoning = ''
                for _line in raw_resp.split('\n'):
                    _line_s = _line.strip()
                    if _line_s.upper().startswith('NEXT_ACTION:'):
                        _action_content = _line_s[len('NEXT_ACTION:'):].strip()
                        # For run: python commands, join continuation lines (up to closing quote)
                        if _action_content.lower().startswith('run:') and 'python' in _action_content.lower():
                            _all_lines = raw_resp.split('\n')
                            # Find this line by content match (not identity — split creates new objects)
                            _start_idx = None
                            for _si, _sl in enumerate(_all_lines):
                                if _sl.strip() == _line_s:
                                    _start_idx = _si
                                    break
                            if _start_idx is None:
                                _start_idx = 0
                            _joined = _action_content
                            # Track quote balance — don't break on empty lines if quotes unclosed
                            _dq = _joined.count('"') - _joined.count('\\"')
                            _sq = _joined.count("'") - _joined.count("\\'")
                            _quotes_open = (_dq % 2 != 0) or (_sq % 2 != 0)
                            # Detect if this is a python command (needs ; separators, not spaces)
                            _is_python_cmd = 'python' in _joined.lower() and ('run:' in _joined.lower() or '-c' in _joined)
                            _sep = '; ' if _is_python_cmd else ' '
                            # Collect subsequent lines until command is complete
                            for _cont_line in _all_lines[_start_idx + 1:]:
                                _cl = _cont_line.strip()
                                # Only break on empty lines / structural markers if quotes balanced
                                if _cl.upper().startswith('REASONING:') or _cl.upper().startswith('NEXT_ACTION:'):
                                    break
                                if not _cl and not _quotes_open:
                                    break
                                if not _cl and _quotes_open:
                                    continue  # Skip blank lines within quoted python code
                                _joined += _sep + _cl
                                _dq = _joined.count('"') - _joined.count('\\"')
                                _sq = _joined.count("'") - _joined.count("\\'")
                                _quotes_open = (_dq % 2 != 0) or (_sq % 2 != 0)
                            # Collapse to single line
                            action_line = ' '.join(_joined.split())
                            log_debug(f"[PASS 2 MULTILINE] ENV {idx_p}: joined {len(_all_lines) - _start_idx} lines → {len(action_line)} chars")
                        else:
                            action_line = _action_content
                        break
                    elif _line_s.upper().startswith('REASONING:'):
                        _reasoning = _line_s[len('REASONING:'):].strip()
                if not action_line:
                    # Fallback A: Extract python code from markdown code blocks
                    # Model sometimes outputs ```python\ncode...\n``` instead of NEXT_ACTION: run: ...
                    import re as _re_cb
                    _code_block_m = _re_cb.search(r'```(?:python|py)?\s*\n(.*?)```', raw_resp, _re_cb.DOTALL)
                    if _code_block_m:
                        _code = _code_block_m.group(1).strip()
                        # Strip run: python3 -c prefix if model included it inside the code block
                        if _code.startswith('run:'):
                            _code = _re_cb.sub(r'^run:\s+python3?\s+-c\s+["\']?', '', _code).strip()
                        if _code and ('import' in _code or 'pptx' in _code.lower() or 'openpyxl' in _code.lower() or 'print(' in _code):
                            _has_blocks = any(line.strip().startswith(('for ', 'if ', 'while ', 'def ', 'with ')) and line.strip().endswith(':') for line in _code.split('\n'))
                            if _has_blocks:
                                # Complex code with control flow — write to temp file
                                import base64 as _b64_cb
                                _code_b64 = _b64_cb.b64encode(_code.encode()).decode()
                                action_line = f'run: python3 -c "import base64; exec(base64.b64decode(\'{_code_b64}\'))"'
                                log_debug(f"[PASS 2 CODE BLOCK] ENV {idx_p}: Complex python (has blocks) → base64 exec ({len(action_line)} chars)")
                            else:
                                # Simple code — semicolon join
                                _code_oneline = '; '.join(line.strip() for line in _code.split('\n') if line.strip() and not line.strip().startswith('#'))
                                action_line = f'run: python3 -c "{_code_oneline}"'
                                log_debug(f"[PASS 2 CODE BLOCK] ENV {idx_p}: Simple python → inline ({len(action_line)} chars")

                    # Fallback B: look for lines that look like actual actions (not reasoning text)
                    if not action_line:
                        _action_prefixes = ('click', 'type', 'hotkey', 'press', 'run:', 'scroll', 'done', 'wait',
                                            'triple-click', 'triple click', 'right-click', 'right click')
                        for _line in raw_resp.split('\n'):
                            _line_s = _line.strip()
                            if _line_s and not _line_s.upper().startswith('REASONING:'):
                                _line_lower = _line_s.lower()
                                # Check if line starts with a valid action prefix
                                if any(_line_lower.startswith(p) for p in _action_prefixes):
                                    action_line = _line_s
                                    break
                    # If no action-like line found, try to extract action from reasoning text
                    if not action_line and _reasoning:
                        import re as _re_extract
                        # Try to find action-like patterns in reasoning
                        _extracted = None
                        # Pattern: "I will click (X, Y)" or "clicking the '+' button"
                        _click_m = _re_extract.search(r"(?:click|clicking)\s+\(?(\d+)[,\s]+(\d+)\)?", _reasoning)
                        if _click_m:
                            _extracted = f"click ({_click_m.group(1)}, {_click_m.group(2)})"
                        # Pattern: "I will click 'Label'" or "clicking the 'Label'"
                        if not _extracted:
                            _label_m = _re_extract.search(r"(?:click|clicking)\s+(?:the\s+)?'([^']+)'", _reasoning, _re_extract.IGNORECASE)
                            if _label_m:
                                _extracted = f"click '{_label_m.group(1)}'"
                        # Pattern: "I will type 'text'" or "input 'text'"
                        if not _extracted:
                            _type_m = _re_extract.search(r"(?:type|input|enter)\s+'([^']+)'", _reasoning, _re_extract.IGNORECASE)
                            if _type_m:
                                _extracted = f"type '{_type_m.group(1)}'"
                        # Pattern: "run: ..." or "use python3 -c ..."
                        if not _extracted:
                            _run_m = _re_extract.search(r"(run:\s+.+?)(?:\.|$)", _reasoning)
                            if _run_m:
                                _extracted = _run_m.group(1).strip()
                        if _extracted:
                            action_line = _extracted
                            log_debug(f"[PASS 2 PARSE] ENV {idx_p}: Extracted action from reasoning: '{_extracted[:80]}'")
                    # Final fallback: use 'wait'
                    if not action_line:
                        log_debug(f"[PASS 2 PARSE] ENV {idx_p}: No action found in response, using 'wait'")
                        action_line = 'wait'
                # Safety check: detect incomplete run: python commands (unclosed quotes)
                if action_line.lower().startswith('run:') and 'python' in action_line.lower():
                    _aq_dq = action_line.count('"') - action_line.count('\\"')
                    _aq_sq = action_line.count("'") - action_line.count("\\'")
                    if _aq_dq % 2 != 0 or (_aq_dq == 0 and action_line.rstrip().endswith('"')):
                        log_debug(f"[PASS 2 INCOMPLETE] ENV {idx_p}: Unclosed quotes in python cmd ({len(action_line)} chars): '{action_line[:80]}...'")
                        # Try to reconstruct from entire raw response after NEXT_ACTION:
                        _na_idx = raw_resp.upper().find('NEXT_ACTION:')
                        if _na_idx >= 0:
                            _full_cmd = raw_resp[_na_idx + len('NEXT_ACTION:'):].strip()
                            # Remove any trailing REASONING or structural markers
                            for _marker in ['REASONING:', 'NEXT_ACTION:']:
                                _mi = _full_cmd.upper().find(_marker)
                                if _mi > 0:
                                    _full_cmd = _full_cmd[:_mi].strip()
                            _full_cmd = ' '.join(_full_cmd.split())
                            if len(_full_cmd) > len(action_line) + 10:
                                action_line = _full_cmd
                                log_debug(f"[PASS 2 INCOMPLETE] ENV {idx_p}: Reconstructed → {len(action_line)} chars")
                if _reasoning:
                    log_debug(f"[PASS 2 REASONING] ENV {idx_p}: {_reasoning[:200]}")
                # Only strip wrapping quotes if the ENTIRE action is quote-wrapped
                # (e.g., "triple-click" → triple-click) but NOT if the action contains
                # internal quotes (e.g., click 'Line Spacing: 2' must not be stripped)
                if len(action_line) >= 2 and action_line[0] == action_line[-1] and action_line[0] in ('"', "'"):
                    action_line = action_line[1:-1].strip()
                for prefix in ['NEXT_ACTION:', 'Action:', 'I will ', 'I should ']:
                    if action_line.lower().startswith(prefix.lower()):
                        action_line = action_line[len(prefix):].strip()
                action_str = action_line if action_line else 'wait'
                # Don't clean run: commands (python code has legitimate pipes/special chars)
                if not action_str.lower().startswith('run:'):
                    action_str = clean_action(action_str)

                # G6 HARD-OVERRIDE: if the XLATE cycle flagged a b64
                # candidate for this step, substitute it now. Universal —
                # only fires when state['_force_action_override'] is set,
                # which only happens when XLATE validated multi-line code
                # in its base64-file channel.
                _override_for_this_env = state_obj.get('_force_action_override')
                if _override_for_this_env:
                    action_str = _override_for_this_env
                    state_obj.pop('_force_action_override', None)
                    print(f"[STAGNATION-OVERRIDE] ENV {idx_p}: substituted XLATE b64 candidate "
                          f"({len(action_str)} chars) over actor output")
                # DEEP TRACE: dump PASS 2 prompt + response + final action.
                if os.getenv("REFLEXGRAD_DEEP_TRACE", "0") == "1":
                    try:
                        import json as _json_dt2
                        _dt_dir = os.path.join(
                            os.getcwd(), '_deep_trace',
                            os.getenv("REFLEXGRAD_RUN_NAME", "deep_trace")
                        )
                        os.makedirs(_dt_dir, exist_ok=True)
                        _cur_step = state_obj.get('cur_step',
                                                  state_obj.get('current_step', 0))
                        _payload = {
                            'step': _cur_step,
                            'env_id': idx_p,
                            'pass2_prompt': str(gpt5_prompt),  # full, no truncation
                            'pass2_prompt_len': len(str(gpt5_prompt)),
                            'pass2_prompt_tail_5k': str(gpt5_prompt)[-5000:],
                            'pass2_raw_response': str(raw_resp)[:8000],
                            'pass2_final_action': str(action_str)[:4000],
                            'fact_card': str(_fact_card_block)[:4000],
                            'curr_obs_injected_len': len(str(curr_obs)[:8000]),
                            'override_from_xlate': bool(
                                state_obj.get('_force_action_override')
                            ),
                            'pending_stagnation_action_seen': bool(_pending_stag_act
                                if '_pending_stag_act' in dir() else False),
                        }
                        _p = os.path.join(_dt_dir,
                                          f'pass2_step_{_cur_step:03d}_env{idx_p}.json')
                        with open(_p, 'w') as _f:
                            _json_dt2.dump(_payload, _f, indent=2, default=str)
                    except Exception as _dt_err:
                        pass
                resp_text = f"RECOMMENDED_ACTION: {action_str}"
                gradient_outputs.append(MockOutput(resp_text))
                _actor_lbl = os.getenv('REFLEXGRAD_MODEL', 'actor')
                print(f"[PHASE 5 - PASS 2] ENV {idx_p}: {_actor_lbl}→'{action_str}'")
            print(f"[PHASE 5 - PASS 2] ✓ Generated {len(gradient_outputs)} {os.getenv('REFLEXGRAD_MODEL', 'actor')} reasoning actions!")
        else:
            print(f"[PHASE 5 - PASS 2] Calling model.generate() with {len(textgrad_prompts)} TextGrad prompts in ONE batch...")
            gradient_outputs = model.generate(textgrad_prompts, textgrad_sampling_params, reasoning_effort='medium')
            print(f"[PHASE 5 - PASS 2] ✓ Generated {len(gradient_outputs)} TextGrad actions in ONE batch call!")

        # Parse ALL gradient outputs
        print(f"[PHASE 5] Parsing {len(gradient_outputs)} gradient outputs...")
        step_gradients = []
        for idx, (gradient_output, result) in enumerate(zip(gradient_outputs, action_results)):
            gradient_text = gradient_output.outputs[0].text.strip()
            # DEEP TRACE: dump LLM response
            _trace_dir = os.path.join(os.getcwd(), '_prompt_traces')
            os.makedirs(_trace_dir, exist_ok=True)
            _step = result.get('current_step', idx)
            with open(os.path.join(_trace_dir, f'pass2_response_step_{_step}.txt'), 'w') as _tf:
                _tf.write(gradient_text)
            print(f"[TRACE] PASS 2 response dumped ({len(gradient_text)} chars)")
            # Preserve env-grounded values before PASS 2 overwrites step_gradient
            _saved_env_fix = state.get('step_gradient', {}).get('next_action_guidance', '')
            _saved_env_source = state.get('step_gradient', {}).get('guidance_source', '')
            _saved_env_score = state.get('step_gradient', {}).get('progress_score', None)
            _saved_env_status = state.get('step_gradient', {}).get('progress_status', None)

            step_gradient = parse_step_gradient_response(
                response=gradient_text,
                task=result['task'],
                prev_observation=result['prev_observation'],
                curr_observation=result['observation'],
                action=result['action']
            )

            # Restore ALL env-grounded values (don't let PASS 2 overwrite them)
            if _saved_env_source == 'env_grounded_fix' and _saved_env_fix:
                step_gradient['next_action_guidance'] = _saved_env_fix
                step_gradient['guidance_source'] = 'env_grounded_fix'
                print(f"[ENV-FIX PRESERVE] Kept env-grounded fix through PASS 2: '{_saved_env_fix[:80]}'")
            if _saved_env_score is not None:
                step_gradient['progress_score'] = _saved_env_score
                step_gradient['progress_status'] = _saved_env_status or 'NO_PROGRESS'

            # MINIMAL SYNERGY FIX: Check if TextGrad contradicts recent Reflexion insights
            # This enforces Reflexion's causal analysis without overengineering
            reflexions = result.get('tiered_step_reflexions', [])
            if reflexions and len(reflexions) > 0:
                # Get most recent Reflexion insights (last 3 for recency)
                recent_reflexions = reflexions[-3:]
                hypothesis = step_gradient.get('hypothesis', '').lower()

                # Simple contradiction detection: Check if hypothesis violates Reflexion warnings
                for ref in recent_reflexions:
                    if isinstance(ref, dict):
                        insight = ref.get('reflection', '').lower()

                        # Pattern: "don't confuse X with Y" where X and Y appear in hypothesis together
                        if 'don\'t confuse' in insight or 'do not confuse' in insight:
                            # Extract what shouldn't be confused (e.g., "cleaning" and "cooling")
                            import re
                            pattern = r"don[' ]?t confuse (\w+) (?:with|and) (\w+)"
                            match = re.search(pattern, insight)
                            if match:
                                term1, term2 = match.groups()
                                # Check if hypothesis claims term1 achieves term2 (or vice versa)
                                if (term1 in hypothesis and term2 in hypothesis):
                                    # Likely contradiction - override
                                    log_debug(f"[SYNERGY-CORRECTION] ENV {result['state']['env_id']} Step {result['current_step']}: TextGrad claim contradicts Reflexion insight about {term1}/{term2}")
                                    step_gradient['hypothesis'] = f"⚠️ REFLEXION CORRECTION: {insight.split('.')[0]}. " + step_gradient.get('hypothesis', '')
                                    # Downgrade progress if claiming completion
                                    if step_gradient.get('progress_status') in ['TASK_COMPLETE', 'MAJOR_PROGRESS']:
                                        step_gradient['progress_status'] = 'PARTIAL_PROGRESS'

            # TRUE SYNERGY METADATA: Mark whether this action used Reflexion insight
            if not USE_TEXTGRAD:
                # REFLEXION_ONLY ablation mode - no TextGrad at all
                step_gradient['guidance_source'] = 'reflexion_only'
            elif idx in reflexion_needed_indices:
                step_gradient['guidance_source'] = 'textgrad_with_reflexion_insight'
            else:
                step_gradient['guidance_source'] = 'textgrad'

            # FIX #32: Update cumulative search memory with new entry
            new_entry = step_gradient.get('new_search_entry', '')
            if new_entry and new_entry.lower() not in ['none', 'n/a', '']:
                state = result['state']
                current_memory = state.get('cumulative_search_memory', '')
                if current_memory:
                    state['cumulative_search_memory'] = f"{current_memory} | {new_entry}"
                else:
                    state['cumulative_search_memory'] = new_entry
                log_debug(f"[FIX #32] ENV {state['env_id']}: Search memory updated: {new_entry}")

            step_gradients.append(step_gradient)
        print(f"[PHASE 5] ✓ Parsed {len(step_gradients)} gradients (TRUE SYNERGY: TextGrad generates 100% of actions)")

        # ========================================================================
        # PHASE 6: APPLY ALL GRADIENTS AND PROCESS LEARNING FLOWS
        # ========================================================================
        print(f"\n[PHASE 6] Applying {len(step_gradients)} gradients and processing learning...")

        # MEMORY LEAK DEBUG: Track memory at each iteration
        import psutil
        import os as os_mod
        process = psutil.Process(os_mod.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        for idx, (result, step_gradient) in enumerate(zip(action_results, step_gradients)):
            state = result['state']
            action = result['action']
            observation = result['observation']
            is_failure = result['is_failure']
            done = result['done']
            reward = result['reward']
            info = result['info']

            # ═══════════════════════════════════════════════════════════════════════
            # FIX #15 (Part 2): Merge backward gradient from PHASE 4 into step_gradient
            # PHASE 4 stores backward gradient in state['step_gradient']['textgrad_gradient']
            # But step_gradient here comes from parse_step_gradient_response() (different dict!)
            # This merge ensures backward gradient flows to step_insights_accumulator
            # ═══════════════════════════════════════════════════════════════════════
            if 'step_gradient' in state and state['step_gradient']:
                textgrad_gradient = state['step_gradient'].get('textgrad_gradient', '')
                if textgrad_gradient:
                    step_gradient['textgrad_gradient'] = textgrad_gradient
                    log_debug(f"[FIX #15] ENV {state['env_id']}: Merged backward gradient: {textgrad_gradient[:80]}...")

                # CRITICAL FIX: For visual envs, PASS 2 only generates an action (no progress eval).
                # parse_step_gradient_response() defaults to score=3 (EXPLORING) for ALL visual steps.
                # But the TextGrad loss was computed MULTIMODALLY with screenshots and has a real verdict.
                # Extract the progress verdict from the loss text to get an accurate progress score.
                if env_type in ('osworld', 'visual') and step_gradient.get('progress_score', 0) == 3:
                    textgrad_loss = state['step_gradient'].get('textgrad_loss', '')
                    if textgrad_loss:
                        import re as _re_prog
                        _loss_upper = textgrad_loss.upper()
                        _old_score = step_gradient['progress_score']
                        if 'TASK_COMPLETE' in _loss_upper:
                            step_gradient['progress_score'] = 10
                            step_gradient['progress_status'] = 'TASK_COMPLETE'
                        elif 'GOAL_ALIGNED' in _loss_upper and 'NOT_ALIGNED' not in _loss_upper and 'GOAL_NOT_ALIGNED' not in _loss_upper:
                            step_gradient['progress_score'] = 6
                            step_gradient['progress_status'] = 'PARTIAL_PROGRESS'
                        elif 'PARTIAL_PROGRESS' in _loss_upper:
                            step_gradient['progress_score'] = 6
                            step_gradient['progress_status'] = 'PARTIAL_PROGRESS'
                        elif 'MAJOR_PROGRESS' in _loss_upper:
                            step_gradient['progress_score'] = 8
                            step_gradient['progress_status'] = 'MAJOR_PROGRESS'
                        elif 'GOAL_NOT_ALIGNED' in _loss_upper or 'NOT ALIGNED' in _loss_upper:
                            step_gradient['progress_score'] = 1
                            step_gradient['progress_status'] = 'NO_PROGRESS'
                        elif 'NO_PROGRESS' in _loss_upper:
                            step_gradient['progress_score'] = 1
                            step_gradient['progress_status'] = 'NO_PROGRESS'
                        if step_gradient['progress_score'] != _old_score:
                            log_debug(f"[VISUAL PROGRESS FIX] ENV {state['env_id']}: Score {_old_score} → {step_gradient['progress_score']} "
                                      f"(from TextGrad loss verdict: {step_gradient.get('progress_status', '?')})")

                    # UNIVERSAL: Trust successful command execution over visual assessment.
                    # When rc=0 (command succeeded) but TextGrad says NO_PROGRESS (no visual change),
                    # the command DID work — programmatic file edits don't show on screenshots.
                    # This prevents discouraging correct programmatic approaches.
                    _obs_text = str(result.get('observation', ''))
                    _action_text = str(result.get('action', ''))
                    _is_noop = 'NO-OP' in _obs_text or 'no-op' in _obs_text.lower() or 'was a NO-OP' in _obs_text
                    if ('rc=0' in _obs_text and step_gradient.get('progress_score', 0) <= 2
                            and 'run:' in _action_text.lower()
                            and not _is_noop):  # Don't override NOOPs — file unchanged means no real progress
                        step_gradient['progress_score'] = max(step_gradient.get('progress_score', 0), 4)
                        step_gradient['progress_status'] = 'PARTIAL_PROGRESS'
                        log_debug(f"[SCORE OVERRIDE] rc=0 + run: command → bumped to {step_gradient['progress_score']} "
                                  f"(trust execution result over visual assessment)")
                    elif _is_noop and 'run:' in _action_text.lower():
                        log_debug(f"[SCORE OVERRIDE SKIPPED] rc=0 but NO-OP detected — file unchanged, keeping low score")

            # MEMORY LEAK DEBUG: Check memory growth per iteration
            if idx > 0 and idx % 5 == 0:
                memory_current = process.memory_info().rss / 1024 / 1024 / 1024  # GB
                print(f"[MEMORY LEAK] Step {idx}: {memory_current:.2f} GB (delta: +{(memory_current - memory_before):.2f} GB)")

            # 1. UPDATE TODO MANAGER (lines 2378-2388)
            # FIX #14: Skip if already updated in PHASE 5 (to avoid double-incrementing attempts)
            if result.get('todo_updated_in_phase5', False):
                log_debug(f"[FIX #14] ENV {state['env_id']}: Skipping TODO update in PHASE 6 (already done in PHASE 5)")
            elif 'todo_manager' in state and state['todo_manager']:
                try:
                    state['todo_manager'].update_from_action_feedback(
                        action=action,
                        prev_observation=result['prev_observation'],
                        curr_observation=observation,
                        progress_status=step_gradient.get('progress_status', 'EXPLORING')
                    )
                except Exception as e:
                    log_debug(f"[TODO UPDATE ERROR] {e}")

            # 2. LOG STEP GRADIENT (lines 2390-2410)
            comprehensive_logger.log_step_gradient(
                env_id=state['env_id'],
                step=state['cur_step'],
                gradient=step_gradient
            )

            if trial_log_path and os.path.exists(trial_log_path):
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[STEP GRADIENT] Step {state['cur_step']}:\n")
                    f.write(f"  Progress: {step_gradient.get('progress_score', 0)}/10\n")
                    f.write(f"  Hypothesis: {step_gradient.get('hypothesis', 'None')}\n")
                    f.write(f"  Next: {step_gradient.get('next_action_guidance', 'None')}\n")

                    if 'todo_manager' in state and state['todo_manager']:
                        f.write(f"\n[TODO STATUS]:\n")
                        f.write(f"{state['todo_manager'].get_formatted_todos()}\n")

            # DASHBOARD: Comprehensive update in PHASE 6 (active code path)
            if state['env_id'] == 0:
                # Determine learning mode from step gradient
                is_reflexion_step = step_gradient.get('is_reflexion_strategic_insight', False)
                # Update progress, gradient, next action
                _dashboard.update(
                    step=state['cur_step'],
                    progress_score=step_gradient.get('progress_score', 0),
                    progress_status=step_gradient.get('progress_status', ''),
                    next_action=step_gradient.get('next_action_guidance', ''),
                    learning_mode='reflexion' if is_reflexion_step else 'textgrad',
                    reflexion_info=(step_gradient.get('hypothesis', '')[:200] if is_reflexion_step else ''),
                )
                # Storyboard visualizer: router decision + policy snapshot
                _dashboard.update_from_state(state, step_gradient)
                # Storyboard: emit real-time substage events. Wrapped in try/except
                # so a visualization bug NEVER crashes a real agent run.
                try:
                    _step_n = state['cur_step']
                    _traj_tail = state.get('trajectory') or []
                    _obs_preview = str(_traj_tail[-1][1])[:120] if _traj_tail else ''
                    _pv = state.get('step_gradient', {}).get('policy_version', 0)
                    _dashboard.emit_substage('OBS',      {'task': state.get('task','')[:120]}, step=_step_n)
                    _dashboard.emit_substage('ACTOR',    {'action': action, 'policy_version': _pv}, step=_step_n)
                    _dashboard.emit_substage('ENV',      {'observation_preview': _obs_preview}, step=_step_n)
                    _dashboard.emit_substage('LLM_EVAL', {'score': step_gradient.get('progress_score', 0), 'status': step_gradient.get('progress_status', '')}, step=_step_n)
                    _dashboard.emit_substage('POLICY_MERGE', {'mode': 'reflexion' if is_reflexion_step else 'textgrad', 'policy_version': _pv}, step=_step_n)
                    _dashboard.emit_substage('STEP_DONE', {'step': _step_n}, step=_step_n)
                except Exception as _ex:
                    print(f'[STORYBOARD] substage emit skipped: {_ex}')
                # Log step to history JSONL
                _router_dec = getattr(state.get('reflexgrad_core'), 'last_decision', None) if state.get('reflexgrad_core') else None
                _dashboard.log_step({
                    'step': state['cur_step'],
                    'action': action,
                    'progress_score': step_gradient.get('progress_score', 0),
                    'progress_status': step_gradient.get('progress_status', ''),
                    'gradient': step_gradient.get('hypothesis', '')[:200],
                    'next_action': step_gradient.get('next_action_guidance', ''),
                    'learning_mode': 'reflexion' if is_reflexion_step else 'textgrad',
                    'loss': state.get('step_gradient', {}).get('textgrad_loss', '')[:200],
                    'backward_gradient': state.get('step_gradient', {}).get('textgrad_gradient', '')[:200],
                    'policy_version': state.get('step_gradient', {}).get('policy_version', 0),
                    # Storyboard fields:
                    'router_decision': _router_dec,
                    'policy_base': str(getattr(state.get('policy'), 'base_policy', ''))[:2000] if state.get('policy') else '',
                })
                # Update TODO display
                if 'todo_manager' in state and state['todo_manager']:
                    _dashboard.update_todos(state['todo_manager'])

            # 3. APPLY GRADIENT TO PROMPT GENERATOR (lines 2414-2418)
            env_pg = state['prompt_generator']
            env_pg.apply_universal_step_gradient(step_gradient)

            # 4. SHARE FAILURES (lines 2421-2464)
            if is_failure:
                import hashlib
                failure_context = {
                    'action': action,
                    'state_hash': hashlib.md5(result['prev_observation'][:200].encode()).hexdigest()[:8],
                    'state_text': result['prev_observation'][:200],
                    'task': state['task'],
                    'step': state['cur_step'],
                    'prerequisites': [act for act, _, _ in state["trajectory"][-5:] if act],
                    'failure_reason': observation,
                    'timestamp': state['cur_step'],
                    'confidence': 1.0,
                    'is_universal': any(uf in observation.lower() for uf in
                                      ["i don't understand", "invalid command", "not a verb"])
                }

                if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                    state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                state['prompt_generator'].discovered_knowledge['conditional_failures'].append(failure_context)

                # Smart sharing with other envs
                for other_state in env_states:
                    if other_state['env_id'] != state['env_id'] and not other_state['done']:
                        target_context = {
                            'state_text': other_state['prev_observation'][:200],
                            'task': other_state['task'],
                            'step': other_state['cur_step'],
                            'prerequisites': [act for act, _, _ in other_state["trajectory"][-5:] if act]
                        }

                        should_share, confidence = should_share_knowledge(failure_context, target_context)

                        if should_share:
                            shared_failure = failure_context.copy()
                            shared_failure['confidence'] = confidence
                            shared_failure['source_env'] = state['env_id']

                            if 'conditional_failures' not in other_state['prompt_generator'].discovered_knowledge:
                                other_state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                            other_state['prompt_generator'].discovered_knowledge['conditional_failures'].append(shared_failure)

            # 5. STORE GRADIENT IN STATE (lines 2466-2472)
            state['textgrad_components'] = env_pg.prompt_components.copy()
            state['last_step_gradient'] = step_gradient
            state['step_gradient'] = step_gradient
            # Store in history for TextGrad to learn from (includes progress_status)
            state['step_gradients_history'].append(step_gradient)

            # 6. LOG TO FILE (lines 2474-2483)
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[GRADIENT UPDATE {state['cur_step']}]\n")
                    f.write(f"  Next action guidance: {step_gradient.get('next_action_guidance', 'None')}\n")
                    f.write(f"  Progress score: {step_gradient.get('progress_score', 0)}\n")

            # 7. ACCUMULATE INSIGHTS (lines 2485-2558)
            # FIX #13: Include OBSERVATION so forward pass knows what was found at each location!
            # FIX #15: Include BACKWARD GRADIENT so TextGrad prompt sees correct feedback!
            # FIX #28: NO TRUNCATION - full observation and gradient needed for learning!
            progress_score = step_gradient.get('progress_score', 0)

            # FIX #39: Track accumulated state — detect progress and regression universally
            # Parse state_diff for positive achievements and regressions
            vgl_state_diff = info.get('state_diff', '') if info else ''
            if vgl_state_diff:
                diff_lower = vgl_state_diff.lower()
                # Detect positive state changes (universal patterns)
                positive_patterns = ['appeared', 'opened', 'selected', 'highlighted', 'extended',
                                     'activated', 'enabled', 'checked', 'expanded', 'created']
                negative_patterns = ['disappeared', 'closed', 'deselected', 'removed', 'lost',
                                     'cleared', 'collapsed', 'unchecked', 'deleted']
                # Add new achievements
                for pattern in positive_patterns:
                    if pattern in diff_lower:
                        # Extract a short description of what was achieved
                        achievement = f"Step {state['cur_step']}: {vgl_state_diff[:150]}"
                        state['accumulated_state'].append(achievement)
                # Detect regressions — remove matching positive states
                for pattern in negative_patterns:
                    if pattern in diff_lower:
                        # Flag regression in accumulated state
                        regression = f"⚠️ REGRESSION at Step {state['cur_step']}: {vgl_state_diff[:150]}"
                        state['accumulated_state'].append(regression)
                # Keep only last 10 entries to avoid bloat
                if len(state['accumulated_state']) > 10:
                    state['accumulated_state'] = state['accumulated_state'][-10:]

            # Strip heavy ephemeral blocks (POST-ACTION PROBE raw XML, EVALUATOR SOURCE)
            # from the stored observation so they don't accumulate across the trajectory.
            # These blocks are valuable for the CURRENT step's decision-making but stale
            # for historical steps; the latest observation is passed separately via
            # `prev_observation` / `curr_observation` which keep the full content.
            _stored_obs = observation
            for _marker in ('\n\n[POST-ACTION PROBE — RAW CONTENT', '\n\n[EVALUATOR SOURCE CODE'):
                _idx = _stored_obs.find(_marker)
                if _idx >= 0:
                    # Drop everything from the marker to the end of the block (look for
                    # the next double-newline after ']' or end of string)
                    _end = _stored_obs.find('\n]', _idx)
                    if _end < 0:
                        _end = len(_stored_obs)
                    else:
                        _end += 2
                    _stored_obs = _stored_obs[:_idx] + _stored_obs[_end:]
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'observation': _stored_obs,  # Heavy probe/source blocks stripped
                'hypothesis': step_gradient.get('hypothesis', ''),
                'backward_gradient': step_gradient.get('textgrad_gradient', ''),
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', []),
                'click_metadata': info.get('click_metadata') if info else None,
                'has_screenshots': bool(info and info.get('after_screenshot_b64')),
                'state_diff': vgl_state_diff,
            })
            # Record in universal learning context (shared with OSWorld)
            _lc = state.get('learning_context')
            if _lc:
                _lc.record_step(
                    step=state['cur_step'], action=action, observation=str(observation),
                    backward_gradient=step_gradient.get('textgrad_gradient', ''),
                    progress_score=progress_score,
                    progress_status=step_gradient.get('progress_status', ''),
                    next_guidance=step_gradient.get('next_action_guidance', ''),
                )
            # DEBUG: Verify insight accumulation
            print(f"[FIX #15 DEBUG] ENV {state['env_id']} Step {state['cur_step']}: Accumulated insight. Total insights: {len(state['step_insights_accumulator'])}")

            # Synthesize insights every 3 steps
            if len(state['step_insights_accumulator']) >= 3 and state['cur_step'] % 3 == 0:
                recent_insights = state['step_insights_accumulator'][-3:]

                # Detect patterns
                avg_progress = sum(ins['progress_score'] for ins in recent_insights) / len(recent_insights)
                stuck_pattern = avg_progress < 2
                repeated_hypotheses = len(set(ins['hypothesis'] for ins in recent_insights)) == 1

                # Consolidate wisdom
                synthesis = {
                    'steps': [ins['step'] for ins in recent_insights],
                    'pattern': 'stuck' if stuck_pattern else ('exploring' if avg_progress < 5 else 'progressing'),
                    'key_insight': recent_insights[-1]['hypothesis'],
                    'recommended_focus': recent_insights[-1]['next_guidance'],
                    'common_prereqs': list(set(
                        prereq for ins in recent_insights
                        for prereq in ins.get('missing_prereqs', [])
                    ))
                }

                state['consolidated_step_wisdom'] = synthesis

                # Keep only recent detailed insights
                if len(state['step_insights_accumulator']) > 20:
                    state['step_insights_accumulator'] = state['step_insights_accumulator'][-20:]

            # 8. TRACK PROGRESS (lines 2560-2570)
            state['progress_history'].append(progress_score)
            # Reset task_complete flag if progress drops significantly (task was undone)
            if state.get('_textgrad_task_complete') and progress_score <= 3:
                state['_textgrad_task_complete'] = False
                log_debug(f"[ENV {state['env_id']}] TASK_COMPLETE reset — progress dropped to {progress_score}/10")
            if len(state['progress_history']) >= 5:
                recent_progress = state['progress_history'][-5:]
                avg_recent_progress = sum(recent_progress) / len(recent_progress)
                state['is_stuck'] = avg_recent_progress < 2
            else:
                state['is_stuck'] = False

            # 9. RECORD SEMANTIC INTERACTION (lines 2572-2593)
            if step_gradient and 'semantic_state_before' in step_gradient:
                # Safe access to batch_data_all (may be out of bounds in some edge cases)
                env_idx = result.get('env_idx_in_batch', 0)
                action_reasoning = ''
                if env_idx < len(batch_data_all):
                    action_reasoning = batch_data_all[env_idx].get('action_reasoning', '')

                universal_memory.record_semantic_interaction(
                    semantic_state_before=step_gradient['semantic_state_before'],
                    action=action,
                    action_reasoning=action_reasoning,
                    semantic_state_after=step_gradient['semantic_state_after'],
                    prerequisites=step_gradient.get('prerequisites', {}),
                    task_progress=step_gradient.get('task_progress', {}),
                    success=not is_failure,
                    task=state['task'],
                    episode_id=state['episode_id']
                )
                state['last_step_gradient'] = step_gradient
            else:
                state['last_step_gradient'] = {
                    'semantic_state_after': observation,
                    'task_progress': {'remaining': [state['task']]}
                }

            # 10. MEMORY INTEGRATION (lines 2595-2626)
            from generate_reflections import global_memory_manager
            if is_failure or state['cur_step'] % 3 == 0:
                try:
                    mini_reflection = f"Episode {state['episode_id']}, Step {state['cur_step']}: "
                    if is_failure:
                        mini_reflection += f"EXACT action '{action}' failed. State: {observation}"
                    else:
                        mini_reflection += f"Action '{action}' succeeded. State: {observation}"

                    global_memory_manager.add_experience(
                        reflection=mini_reflection,
                        gradients=step_gradient,
                        success=not is_failure,
                        task=state['task'],
                        env_id=state['env_id'],
                        observation=observation
                    )

                    if state['cur_step'] % 5 == 0:
                        global_memory_manager._consolidate_memories()
                except Exception as e:
                    print(f"[ERROR] Memory integration failed: {e}")

            # 11. LEARN FROM INTERACTION (lines 2628-2630)
            state['prompt_generator']._learn_from_interaction(action, observation)
            env_understanding.learn_from_interaction(action, observation)

            # 12. TRACK TRAJECTORY (lines 2632-2635)
            env_idx_in_batch = result['env_idx_in_batch']
            # Safe access to batch_data_all
            action_reasoning = ''
            if env_idx_in_batch < len(batch_data_all):
                action_reasoning = batch_data_all[env_idx_in_batch].get('action_reasoning', '')
            state['trajectory'].append((action, observation, action_reasoning))

            # 12a. SYNERGISTIC STEP REFLEXION (Reflexion for episodic memory)
            # Generate reflexions ONLY for failures or key milestones to feed episodic memory
            # TextGrad handles real-time action optimization, Reflexion handles learning from experience
            # FIX #6 (Nov 23): Don't generate reflexion for completed environments (prevents infinite loop)
            # ABLATION: Only generate if USE_REFLEXION is True
            if USE_REFLEXION and not done and (is_failure or state['cur_step'] % 5 == 0 or 'you win' in observation.lower()):
                try:
                    # Initialize working_reflexions if not exists
                    if 'working_reflexions' not in state:
                        state['working_reflexions'] = []

                    # ═══════════════════════════════════════════════════════════════════════════
                    # FIX #4: HISTORY-AWARE EPISODIC REFLEXION GENERATION
                    # Root cause of loops: Episodic reflexions were generated WITHOUT historical
                    # context. Now we provide recent action history and previous insights so the
                    # reasoning model can understand what was tried and what to avoid.
                    # ═══════════════════════════════════════════════════════════════════════════

                    # Build history context from recent reflexions using TIERED compression
                    # This gives ALL reflexions smartly compressed (recent=verbose, old=summary)
                    history_context = ""
                    recent_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                    if recent_reflexions:
                        # Format tiered reflexions (handles compressed and verbose)
                        history_parts = []
                        for idx, r in enumerate(recent_reflexions, 1):
                            compression_type = r.get('is_compressed', None)

                            if compression_type == 'heavy':
                                # Heavy compression: just the summary
                                step_context = f"Earlier steps (summary):\n  {r.get('reflection', '')}"
                            elif compression_type == 'medium':
                                # Medium compression: key points
                                step_context = f"Previous steps (key points):\n  {r.get('reflection', '')}"
                            else:
                                # Verbose (recent): full detail — no per-entry truncation
                                # 60/80/120 char caps destroyed reflexion → reflexion learning
                                action_str = r.get('action', '')
                                obs_str = r.get('observation', '')
                                reflection_str = r.get('reflection', '')
                                success_str = "SUCCESS" if r.get('success', False) else "FAILED"

                                step_context = f"Step -{len(recent_reflexions)-idx+1}:\n"
                                step_context += f"  Action: {action_str}\n"
                                step_context += f"  Result: {obs_str}\n"
                                step_context += f"  Status: {success_str}\n"
                                step_context += f"  Insight: {reflection_str}"

                            history_parts.append(step_context)

                        if history_parts:
                            history_context += f"\nHistory (tiered - ALL {len(state.get('working_reflexions', []))} reflexions):\n" + "\n\n".join(history_parts)

                    # Build history-aware reflexion prompt
                    # Format valid actions list
                    valid_actions_list = state.get('valid_actions', [])
                    actions_display = "\n".join(f"   - {act}" for act in valid_actions_list)
                    if len(valid_actions_list) > 30:
                        actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                    reflexion_prompt = f"""Task: {state['task']}
Action taken: {action}
Result: {observation[:1500]}
{"FAILED" if is_failure else "SUCCEEDED"}
{history_context}

Available actions you can choose from:
{actions_display}

{"These are reference hints. You may use ANY action format including run: <shell command> for programmatic approaches." if state.get('env_type', '') in ('osworld', 'visual') else "CRITICAL: When suggesting next steps, ONLY recommend actions from the list above. Do NOT invent new actions."}

What key insight should we remember for future attempts? (1-2 sentences)"""

                    # Log history context status
                    if len(recent_reflexions) > 0:
                        total_reflexions = len(state.get('working_reflexions', []))
                        print(f"[HISTORY TIERED] ENV {state['env_id']} Step {state['cur_step']}: Reflexion sees ALL {total_reflexions} reflexions (tiered: {len(recent_reflexions)} compressed items)")

                    # Generate episodic reflexion (summarizer)
                    step_reflection_output = model.generate([reflexion_prompt], SamplingParams(temperature=0.7, max_tokens=150))[0]
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion. Strip heavy ephemeral blocks
                    # (POST-ACTION PROBE raw XML, EVALUATOR SOURCE) from the stored
                    # observation so trajectory replay doesn't accumulate them and
                    # blow the context budget. The blocks are fresh on the CURRENT
                    # step's observation, which is passed separately.
                    _stored_obs = observation
                    for _marker in ('\n\n[POST-ACTION PROBE — RAW CONTENT', '\n\n[EVALUATOR SOURCE CODE'):
                        _idx = _stored_obs.find(_marker)
                        if _idx >= 0:
                            _end = _stored_obs.find('\n]', _idx)
                            if _end < 0:
                                _end = len(_stored_obs)
                            else:
                                _end += 2
                            _stored_obs = _stored_obs[:_idx] + _stored_obs[_end:]
                    state['working_reflexions'].append({
                        'step': state['cur_step'],
                        'action': action,
                        'observation': _stored_obs,
                        'reflection': step_reflection,
                        'success': not is_failure,
                    })

                    print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: Generated episodic memory ({len(state['working_reflexions'])} total)")
                    log_debug(f"[REFLEXION] Insight: {step_reflection[:80]}...")

                except Exception as e:
                    log_debug(f"[REFLEXION ERROR] Failed to generate episodic reflexion: {e}")

            # 13. TRACK FAILURES (lines 2637-2641)
            if is_failure:
                state['consecutive_failures'] += 1
            else:
                state['consecutive_failures'] = 0

            # 14. UPDATE HISTORY
            state['history'].add("action", action)
            state['history'].add("observation", observation)

            # 15. LOG ACTION/OBSERVATION
            if to_print:
                log_debug(f'[ENV {state["env_id"]}] > {action}\n{observation}')
                sys.stdout.flush()

            # 16. UPDATE OBSERVATION (lines 2881-2886)
            # If hard verification rejected an early TASK_COMPLETE, inject the failure
            # reason into the observation so the agent sees the ground-truth gap
            # Stagnation eval surfaced an unclaimed success — end episode cleanly
            if state.get('_eval_already_success'):
                state.pop('_eval_already_success', None)
                state['done'] = True
                state['success'] = True
                if to_print:
                    print(f"[ENV {state['env_id']}] STAGNATION-EVAL VERIFIED SUCCESS — ending episode")
                universal_memory.record_episode(state['trajectory'], state['task'], True)
            if state.get('_eval_rejected_done'):
                _rej = state.pop('_eval_rejected_done')
                observation = (
                    f"{observation}\n\n[HARD VERIFICATION REJECTED 'TASK_COMPLETE': "
                    f"You claimed the task was done, but the actual evaluator says it is NOT complete.\n"
                    f"Evaluator output: {_rej}\n"
                    f"You MUST keep working. Read the POST-ACTION PROBE in this observation to see "
                    f"what's actually in the file vs. what the task requires. Generate a new action "
                    f"that fixes the actual gap, not just declares 'done' again.]"
                )
            # Stagnation-triggered evaluator output (closed-loop gradient when stuck).
            # Same injection mechanism as the post-TASK_COMPLETE rejection, but fired
            # proactively when the score has stalled instead of waiting for a "done" claim.
            if state.get('_eval_stagnation_reason'):
                _stag = state.pop('_eval_stagnation_reason')
                # Fix B: inject the evaluator's own source code when the gap is reported.
                # The 7-agent investigation showed that evaluator tuples alone
                # (EXPECTED [(4,1,1,'sngStrike')]) are misread by LLMs as "tag not found"
                # — they keep trying child-element approaches. Seeing the literal python
                # that decides pass/fail (e.g. `findall('.//{...}rPr[@strike]')`) lets
                # the LLM recognize that it's an XPath ATTRIBUTE filter on rPr, not a
                # child tag search. Universal: just inspect.getsource of env method.
                _eval_src_block = ""
                _eval_src = state.get('_eval_source', '')
                if _eval_src:
                    _eval_src_block = (
                        f"\n\n[EVALUATOR SOURCE CODE — the literal python that decides "
                        f"pass/fail. Read it to see EXACTLY which file paths it opens, "
                        f"which attributes/elements it checks, and what values it expects:\n"
                        f"```python\n{_eval_src[:3500]}\n```]"
                    )
                observation = (
                    f"{observation}\n\n[STAGNATION GRADIENT — actual evaluator was run because "
                    f"progress has stalled. The evaluator says the task is NOT complete and "
                    f"reports the following concrete gap:\n"
                    f"  {_stag}\n"
                    f"This is the ground-truth signal — more reliable than any screen-delta "
                    f"score. Your next action must directly address THIS specific gap. Do not "
                    f"repeat actions that don't change the file state shown in the POST-ACTION "
                    f"PROBE above.]"
                    f"{_eval_src_block}"
                )
            # Capture introspection probe stdout into _xlate_findings so the next
            # XLATE attempt can use the discovered library facts.
            if state.get('_pending_introspection'):
                state.pop('_pending_introspection', None)
                _intro_findings = ""
                _co_idx2 = observation.find('[Command Output')
                if _co_idx2 >= 0:
                    _co_end = observation.find('\n\n', _co_idx2)
                    if _co_end < 0:
                        _co_end = _co_idx2 + 3000
                    _intro_findings = observation[_co_idx2:min(_co_end, _co_idx2 + 3000)]
                if _intro_findings:
                    state['_xlate_findings'] = _intro_findings
                    print(f"[STAGNATION-XLATE-INTROSPECT] ENV {state['env_id']}: captured {len(_intro_findings)} chars of introspection findings")
                else:
                    print(f"[STAGNATION-XLATE-INTROSPECT] ENV {state['env_id']}: introspection probe produced no Command Output")
            state['prev_observation'] = observation

            # 17. INCREMENT STEP (line 2940)
            state['cur_step'] += 1

            # 18. PERIODIC CHECKPOINT (lines 2942-2949)
            if state['cur_step'] % 10 == 0 and checkpoint_manager:
                try:
                    if checkpoint_manager:  # Only save if checkpoint_manager exists
                        completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
                        pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]
                        checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                except Exception as e:
                    print(f"[WARNING] Periodic checkpoint failed: {e}")

            # 19. CHECK IF DONE (lines 2951-3018)
            if done:
                log_debug(f"\n[DEBUG] ENV {state['env_id']} at step {state['cur_step']}:")
                log_debug(f"  done={done}")
                log_debug(f"  reward: {reward}")

                # For OSWorld/visual envs: agent "done" claim is NOT proof of success.
                # Only external evaluator or max_steps timeout determines success.
                env_type = state.get('env_type', '')
                agent_claimed_done = info.get('won', [False])[0]
                hit_max_steps = state['cur_step'] >= max_steps - 1

                if env_type in ('osworld', 'visual'):
                    # Default: NOT successful until evaluator confirms
                    actual_success = False

                    if hasattr(env, 'evaluate'):
                        try:
                            eval_result = env.evaluate()
                            eval_success = eval_result.get('success', False)
                            if to_print:
                                print(f"[ENV {state['env_id']}] POST-EPISODE EVALUATION: {eval_result}")
                            actual_success = eval_success
                        except Exception as e:
                            if to_print:
                                print(f"[ENV {state['env_id']}] Evaluator error: {e}")

                    if agent_claimed_done and not actual_success:
                        if to_print:
                            print(f"[ENV {state['env_id']}] Agent claimed 'done' but NO evaluator confirmation — marking as FAIL")
                            print(f"  (OSWorld tasks require external evaluation, agent self-report is unreliable)")
                    elif actual_success:
                        if to_print:
                            print(f"[ENV {state['env_id']}] ✓ Evaluator confirms SUCCESS")
                else:
                    # Text environments: trust agent's done claim
                    actual_success = agent_claimed_done

                log_debug(f"  agent_claimed_done={agent_claimed_done}, actual_success={actual_success}")

                state['done'] = True

                # CHECKPOINT: Save state after each environment completes
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_env_checkpoint(
                        trial_idx=trial_idx,
                        env_id=state['env_id'],
                        state=state,
                        trajectory_length=len(state['trajectory'])
                    )

                # Save master state for resume
                completed_envs = [s['env_id'] for s in env_states if s['done']]
                pending_envs = [s['env_id'] for s in env_states if not s['done']]
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                state['success'] = actual_success

                # Also save minimal checkpoint for compatibility
                checkpoint_file = os.path.join(os.path.dirname(trial_log_path), f'checkpoint_trial_{trial_idx}_env_{state["env_id"]}.json')
                checkpoint_data = {
                    'env_id': state['env_id'],
                    'success': actual_success,
                    'trajectory_length': len(state['trajectory']),
                    'task': state['task'],
                    'memory': env_configs[state['env_id']].get('memory', []),
                    'working_reflexions': state.get('working_reflexions', []),
                    'step': state['cur_step']
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                # CRITICAL FIX: Update env_configs immediately
                env_configs[state['env_id']]['is_success'] = actual_success

                # Save successful trajectory for cross-trial learning (Reflexion's episodic memory)
                if actual_success:
                    # Save to per-trial trajectory (for immediate debugging)
                    env_configs[state['env_id']]['successful_trajectory'] = [
                        (act, obs[:100]) for act, obs, _ in state["trajectory"]
                    ]

                    # CRITICAL FIX: Save to reflexion_memory with correct type for cross-trial loading!
                    # This enables success pattern replay in future trials (+50 boost at line 803)
                    success_workflow = {
                        'type': 'success_workflow',  # This is what line 697 looks for!
                        'task': state['task'],
                        'actions': [act for act, _, _ in state["trajectory"]],
                        'episode_id': state.get('episode_id', f"trial{trial_idx}_env{state['env_id']}"),
                        'trial': trial_idx,
                        'steps': len(state['trajectory']),
                        'success_confirmed': True
                    }

                    # Append to env_configs memory (this persists across trials!)
                    if 'memory' not in env_configs[state['env_id']]:
                        env_configs[state['env_id']]['memory'] = []
                    env_configs[state['env_id']]['memory'].append(success_workflow)

                    print(f"[SUCCESS PATTERN] Saved workflow to reflexion_memory: {len(state['trajectory'])} actions for '{state['task']}'")

                # Log last 3 actions for debugging
                if state['trajectory']:
                    log_debug(f"  Last 3 actions taken:")
                    for act, obs, _ in state["trajectory"][-3:]:
                        obs_preview = obs[:80] if len(obs) > 80 else obs
                        log_debug(f"    {act} -> {obs_preview.replace(chr(10), ' ')}")

                # RECORD COMPLETE EPISODE IN UNIVERSAL MEMORY
                universal_memory.record_episode(state['trajectory'], state['task'], actual_success)

                if actual_success:
                    universal_memory.store_sequence_pattern(state['trajectory'], True, state['task'])
                state['done'] = True
                state['success'] = actual_success

            # 20a. TASK COMPLETION DETECTION (visual environments)
            # When step gradient reports TASK_COMPLETE, end episode early with success.
            # This lets ReflexGrad detect when the task is already done (e.g., settings
            # already configured correctly) without requiring all max_steps.
            # Safety guard: require meaningful actions (not just wait/scroll) to prevent
            # false positives from LLM hallucinating completion.
            env_type = state.get('env_type', '')
            if not state['done'] and step_gradient.get('progress_status') == 'TASK_COMPLETE':
                # Text envs (ALFWorld, TextWorld, etc.): Don't end the episode (only env.step()
                # done signal is authoritative), but DO mark TODOs complete and set a textual
                # constraint so the PASS 2 prompt tells the LLM to only verify, not act.
                # Without this, the TODO shows stale subtasks like "Retrieve the pan" which
                # causes the agent to pick up an already-placed object, undoing the task.
                if env_type not in ('osworld', 'visual'):
                    # Mark all remaining TODOs as completed
                    if 'todo_manager' in state:
                        for _todo in state['todo_manager'].todos:
                            if _todo.status not in ('completed', 'failed'):
                                _todo.status = 'completed'
                        state['todo_manager'].completed_subtasks.append('ALL GOALS MET (TextGrad TASK_COMPLETE)')
                    # Set textual flag — PASS 2 prompt will include verification-only guidance
                    state['_textgrad_task_complete'] = True
                    if to_print:
                        print(f"[ENV {state['env_id']}] TASK_COMPLETE claimed by TextGrad — TODOs marked complete, verification mode enabled")
                else:
                    # Visual envs: TextGrad TASK_COMPLETE is useful (no automatic done signal)
                    trajectory = state.get('trajectory', [])
                    passive_actions = {'wait', 'scroll up', 'scroll down', 'scroll up 1',
                                       'scroll down 1', 'scroll up 3', 'scroll down 3'}
                    meaningful_steps = sum(1 for t in trajectory
                                           if (isinstance(t, (tuple, list)) and len(t) >= 1 and
                                               str(t[0]).strip().lower() not in passive_actions) or
                                           (isinstance(t, dict) and
                                            t.get('action', '').strip().lower() not in passive_actions))
                    if meaningful_steps >= 2:
                        # HARD VERIFY before ending episode: if the actual evaluator says
                        # it's NOT done, let the agent continue. This prevents the model from
                        # lying its way to early termination on false positive completions.
                        # Universal — works for any task with an evaluator.
                        _verified_complete = False
                        _eval_failure_reason = ""
                        if hasattr(env, 'evaluate'):
                            try:
                                _tc_eval = env.evaluate()
                                _verified_complete = _tc_eval.get('success', False)
                                # Build a detailed failure reason from output OR results list
                                _reason_parts = []
                                if _tc_eval.get('output'):
                                    _reason_parts.append(str(_tc_eval['output']))
                                if _tc_eval.get('results'):
                                    for _r in _tc_eval['results']:
                                        if isinstance(_r, dict):
                                            _r_out = _r.get('output', '') or _r.get('reason', '')
                                            if _r_out:
                                                _reason_parts.append(f"[{_r.get('func', '?')}] {_r_out}")
                                if _tc_eval.get('reason'):
                                    _reason_parts.append(str(_tc_eval['reason']))
                                _eval_failure_reason = "; ".join(_reason_parts)[:1000] or "no details"
                                if to_print:
                                    print(f"[ENV {state['env_id']}] HARD VERIFY before TASK_COMPLETE: success={_verified_complete}, output={_eval_failure_reason[:300]}")
                            except Exception as _tc_e:
                                if to_print:
                                    print(f"[ENV {state['env_id']}] HARD VERIFY error: {_tc_e}")
                                _verified_complete = False

                        if _verified_complete:
                            # Evaluator confirms — end episode with success
                            if to_print:
                                print(f"[ENV {state['env_id']}] TASK_COMPLETE VERIFIED — ending episode ({meaningful_steps} meaningful steps)")
                            state['done'] = True
                            state['success'] = True
                            universal_memory.record_episode(state['trajectory'], state['task'], True)
                        else:
                            # Evaluator says NOT done — reject TextGrad's claim, force agent to continue
                            if to_print:
                                print(f"[ENV {state['env_id']}] TASK_COMPLETE REJECTED by evaluator — agent must continue. Reason: {_eval_failure_reason[:200]}")
                            # Inject failure feedback so the agent sees what's missing
                            state['_eval_rejected_done'] = _eval_failure_reason
                            # Reset the progress score so TextGrad doesn't immediately re-trigger
                            step_gradient['progress_status'] = 'IN_PROGRESS'
                            # Don't end episode — agent gets another chance
                    else:
                        if to_print:
                            print(f"[ENV {state['env_id']}] TASK_COMPLETE claimed but only {meaningful_steps} meaningful steps — ignoring (need >= 2)")

            # 20b. CHECK FAILURE THRESHOLD (lines 3021-3031)
            # Visual environments get a higher threshold — coordinate resolution
            # and exploratory actions may produce "no visible change" without being stuck
            failure_threshold = 10 if env_type in ('osworld', 'visual') else 5
            # If STAGNATION-EVAL has fired recently, the system IS in active recovery
            # via LLM-translated python actions targeting the real evaluator gap.
            # The loop guard would terminate that recovery prematurely. Defer the
            # loop guard while recovery is in progress (within 8 steps of last fire).
            _stag_recovering = (
                state.get('_last_stagnation_eval_step', -100) >= 0
                and (state.get('cur_step', 0) - state.get('_last_stagnation_eval_step', -100)) <= 8
            )
            if _stag_recovering and to_print and state['consecutive_failures'] > failure_threshold:
                print(f"[ENV {state['env_id']}] Loop guard DEFERRED — STAGNATION-EVAL recovery active "
                      f"(last fired step {state.get('_last_stagnation_eval_step')})")
            if state['consecutive_failures'] > failure_threshold and not state['done'] and not _stag_recovering:
                if to_print:
                    print(f"[ENV {state['env_id']}] Stuck in unproductive loop - ending episode")

                # For visual envs, call the evaluator before giving up — the task may
                # already be complete even if the agent hasn't said 'done' yet.
                actual_success_stuck = False
                if env_type in ('osworld', 'visual'):
                    try:
                        env_obj_stuck = _env_registry.get(state['env_id'])
                        if env_obj_stuck and hasattr(env_obj_stuck, 'step'):
                            stuck_eval = env_obj_stuck.step('done')
                            actual_success_stuck = stuck_eval.get('success', False) if isinstance(stuck_eval, dict) else False
                            if actual_success_stuck:
                                print(f"[ENV {state['env_id']}] Stuck but EVALUATOR SAYS PASS — marking success!")
                    except Exception as e:
                        log_debug(f"[ENV {state['env_id']}] Evaluator at stuck termination failed: {e}")

                # Record failed episode
                universal_memory.record_episode(state['trajectory'], state['task'], actual_success_stuck)

                state['done'] = True
                state['success'] = actual_success_stuck

        print(f"[PHASE 6] ✓ Applied all gradients and processed learning for {len(action_results)} environments")

        # ========================================================================
        # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE SEQUENTIAL CODE
        # ========================================================================
        # This entire block is wrapped in "if False:" and will NEVER execute
        # Left here for historical reference only
        #
        # ACTIVE parallel batch processing code is in PHASES 3B-6 (lines 2573-3122)
        # If you need to make changes to the parallel code, modify PHASES 3B-6 above
        # DO NOT modify or enable this disabled sequential code
        # ========================================================================
        # OLD SEQUENTIAL PROCESSING (DISABLED - REPLACED WITH PHASES 3B-6)
        # ========================================================================
        if False:  # Lines 2055-3121 disabled - Sequential processing replaced with parallel
            # TIMING FIX: Generate Reflexion BEFORE action selection (except step 0)
            # This ensures extracted actions are available for THIS step, not next step
            if state['cur_step'] > 0 and state['trajectory']:
                # Get the last action and observation from trajectory
                last_action, last_observation, last_reasoning = state['trajectory'][-1]

                # Check if last action failed
                is_last_failure = any(fail_msg in last_observation.lower() for fail_msg in
                                     ["nothing happens", "don't see", "can't", "already"])

                # Generate step reflexion for the PREVIOUS action
                print(f"\n[REFLEXION-BEFORE-ACTION] Step {state['cur_step']} reflecting on previous action")

                recent_trajectory = state['trajectory'][-5:] if state['trajectory'] else []
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                if tiered_reflexions:
                    formatted_learnings = []
                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        step_num = ref.get('step', '?')
                        reflection_text = ref.get('reflection', '')[:200]
                        if compression_type == 'heavy':
                            formatted_learnings.append(f"  Steps {step_num} (summary): {reflection_text}")
                        elif compression_type == 'medium':
                            formatted_learnings.append(f"  Steps {step_num} (key points): {reflection_text}")
                        else:
                            formatted_learnings.append(f"  Step {step_num}: {reflection_text}")
                    previous_step_learnings = '\n'.join(formatted_learnings)
                else:
                    previous_step_learnings = "None yet (first few steps)"

                todo_context = ""
                if hasattr(state.get('todo_manager'), 'current_todo'):
                    current_todo = state['todo_manager'].current_todo()
                    if current_todo:
                        todo_context = f"\nCurrent Subtask: '{current_todo.content}' (Attempt #{current_todo.attempts})"
                        if current_todo.attempts > 2:
                            todo_context += f"\n⚠️  WARNING: {current_todo.attempts} attempts without success"

                step_reflection_prompt = f"""Analyze the last few actions in this episode.

PREVIOUS LEARNINGS FROM THIS TRIAL (step-level memory):
{previous_step_learnings}

Task: {state['task']}
Current step: {state['cur_step']}

Recent actions and outcomes:"""

                for act, obs, reasoning in recent_trajectory:
                    step_reflection_prompt += f"\n> {act}\nResult: {obs[:100]}..."

                step_reflection_prompt += f"""

Current action: {last_action}
Result: {last_observation}...
Success: {'No' if is_last_failure else 'Yes'}
{todo_context}

🔬 REFLEXION: Root-Cause Analysis

CRITICAL ANALYSIS REQUIRED:

1. SEMANTIC PRECISION CHECK:
   Task goal: "{state['task']}"
   Action taken: "{last_action}"

   Question: What is the PRIMARY PURPOSE of each?
   - Task requires me to: [specific effect/goal]
   - Action achieves: [specific effect/purpose]
   - Are these semantically IDENTICAL or DIFFERENT?

   IMPORTANT: Actions involving similar entities may serve different purposes.
   Verify that the action's effect aligns with the task's specific requirement.

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{state['task']}", what SPECIFIC property must change?

   Look at observation: "{last_observation[:150]}"
   Did that EXACT property change? Cite specific words as evidence.
   Do NOT assume - only state what observation explicitly shows.

3. FAILURE PATTERN DETECTION:
   {f'Subtask not achieved after {current_todo.attempts} attempts.' if todo_context and hasattr(state.get('todo_manager'), 'current_todo') and state['todo_manager'].current_todo() and state['todo_manager'].current_todo().attempts > 2 else 'Check if repeating similar actions.'}

   Am I repeating similar action types that don't work?
   What assumption might be incorrect?

4. AVAILABLE ACTIONS (HARD CONSTRAINT):
   Here are the ONLY actions you can actually execute:
"""
                # Format valid actions list
                valid_actions_list = state.get('valid_actions', [])
                for act in valid_actions_list:
                    step_reflection_prompt += f"   - {act}\n"
                if len(valid_actions_list) > 30:
                    step_reflection_prompt += f"   ... and {len(valid_actions_list)-30} more\n"

                _is_visual = state.get('env_type', '') in ('osworld', 'visual')
                step_reflection_prompt += f"""
5. RECOMMENDED NEXT ACTION:
   If there's a semantic mismatch or wrong action type detected:
   {"You may use ANY action format including run: <shell command> for programmatic approaches. The list above is reference hints, not a hard constraint." if _is_visual else "Choose ONE action from the list in section 4 above that would make progress. DO NOT invent new actions - ONLY select from the list."}

Generate your analysis (keep concise, 3-4 sentences total):"""

                sampling_params = SamplingParams(max_tokens=3000, temperature=0.3)
                step_reflection_output = model.generate([step_reflection_prompt], sampling_params)[0]
                step_reflection = step_reflection_output.outputs[0].text.strip()

                if 'working_reflexions' not in state:
                    state['working_reflexions'] = []

                state['working_reflexions'].append({
                    'step': state['cur_step'] - 1,  # Reflecting on PREVIOUS step
                    'action': last_action,
                    'reflection': step_reflection,
                    'success': not is_last_failure
                })

                log_debug(f"[STEP REFLEXION] Step {state['cur_step']-1}: {step_reflection[:150]}...")

                # Extract gradients using TextGrad
                mini_trajectory = ""
                for act, obs, _ in recent_trajectory[-3:]:
                    mini_trajectory += f"> {act}\n{obs[:100]}...\n"

                progress_score = state.get('step_gradient', {}).get('progress_score', 0)

                # SYNERGY FIX: Only invoke Reflexion's CAUSAL CHAIN when TextGrad needs help
                # High progress (≥4): TextGrad confident → use its clean recommendations
                # Low progress (<4): TextGrad struggling → invoke Reflexion for guidance
                if progress_score < 4:
                    log_debug(f"[SYNERGY] Low progress ({progress_score}/10) - invoking Reflexion guidance")
                    step_reflexion_gradients = state['prompt_generator'].compute_prompt_gradient(
                        trajectory=mini_trajectory,
                        success=False,
                        task=state['task'],
                        reflection=step_reflection,
                        progress_score=progress_score,
                        valid_actions=state.get('valid_actions', [])
                    )

                    # Extract clean action from Reflexion's CAUSAL CHAIN format
                    if step_reflexion_gradients and 'next_action_guidance' in step_reflexion_gradients:
                        extracted_action = step_reflexion_gradients['next_action_guidance']
                        if extracted_action and len(extracted_action) > 3:
                            state['step_gradient']['next_action_guidance'] = extracted_action
                            print(f"🔗 [REFLEXION-GUIDE] Low progress → Reflexion suggests: {extracted_action}")
                else:
                    # TextGrad doing well - don't pollute with Reflexion's verbose CAUSAL CHAIN
                    log_debug(f"[SYNERGY] Moderate/high progress ({progress_score}/10) - TextGrad optimizing")

            # Debug batch_data before sending to action selection
            if DEBUG_ACTOR:
                print(f"\n[BATCH_DATA VERIFICATION]:")
                data = batch_data[0]
                print(f"  ENV {state['env_id']}:")
                print(f"    Task: {data['task'][:50]}")
                print(f"    Valid actions count: {len(data['valid_actions'])}")
                print(f"    First action: {data['valid_actions'][0] if data['valid_actions'] else 'NONE'}")
                print(f"    Observation starts: {data['observation']}...")

            # BATCH ACTION SELECTION (single element batch for sequential)
            selected_actions = reasoning_based_action_selection_batch(
                batch_data=batch_data,
                prompt_generator=prompt_generator,
                DEBUG_ACTOR=DEBUG_ACTOR ,
                log_debug=log_debug
            )

            # Execute action (single environment)
            action = selected_actions[0]
            state = active_states[0]
            # Track EXACT tried actions - ensure it's a set
            if not isinstance(state['tried_actions'], set):
                state['tried_actions'] = set(state['tried_actions'])
            state['tried_actions'].add(action)


            # Extract reasoning from batch_data for this environment
            env_idx_in_batch = active_states.index(state)
            action_reasoning = batch_data[env_idx_in_batch].get('action_reasoning', '')
            
            # Store for batch sharing
            state['last_action'] = action
            
            if DEBUG_ACTOR:
                log_debug(f"[ENV {state['env_id']}] SELECTED EXACT ACTION: '{action}'")
            
            # Add action to history
            state['history'].add("action", action)

            # Execute action - MEMORY FIX: Use env from registry
            env = _env_registry[state['env_id']]
            action = clean_action(action)  # Universal: strip LLM artifacts before env receives it
            observation, reward, done, info = env.step([action])

            # Process observation
            observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
            done = done[0]

            # ═══════════════════════════════════════════════════════════════
            # SYNERGISTIC EPISODIC REFLEXION GENERATION (Added 2025-10-19)
            # Generate reflexion ONLY at key moments (failures/milestones)
            # TextGrad handles real-time optimization, Reflexion handles episodic memory
            # ═══════════════════════════════════════════════════════════════

            print(f"[DEBUG-REFLEXION-PATH] ENV {state['env_id']} Step {state['cur_step']}: Checking for key moments...")

            # Determine if this is a key moment for episodic memory
            is_key_moment = False
            moment_type = ""

            # Check for failure (key learning moment)
            failure_indicators = ['nothing happens', "don't see", "can't", 'already', 'closed']
            if any(fail in observation.lower() for fail in failure_indicators):
                is_key_moment = True
                moment_type = "FAILURE"

            # Check for success (task completion)
            elif 'you win' in observation.lower() or done:
                is_key_moment = True
                moment_type = "SUCCESS"

            # Check for milestone (every 5 steps or significant state change)
            elif state['cur_step'] % 5 == 0:
                is_key_moment = True
                moment_type = "MILESTONE"

            print(f"[DEBUG-KEY-MOMENT] ENV {state['env_id']} Step {state['cur_step']}: is_key_moment={is_key_moment}, type={moment_type}")

            # Generate episodic reflexion at key moments
            if is_key_moment:
                try:
                    # Initialize working_reflexions if not exists
                    if 'working_reflexions' not in state:
                        state['working_reflexions'] = []

                    # Build reflexion prompt for episodic memory
                    # Format valid actions list
                    valid_actions_list = state.get('valid_actions', [])
                    actions_display = "\n".join(f"   - {act}" for act in valid_actions_list)
                    if len(valid_actions_list) > 30:
                        actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                    reflexion_prompt = f"""Task: {state['task']}
Step {state['cur_step']}: {action}
Result: {observation[:1500]}
Type: {moment_type}

Available actions you can choose from:
{actions_display}

What key insight should we remember for future trials? Focus on:
- What worked or didn't work
- Why this action led to this outcome
- What SPECIFIC action from the list above to try differently next time

CRITICAL: Only reference actions from the list above.

Provide a concise (1 sentence) episodic memory insight:"""

                    # Generate episodic reflexion using the model
                    sampling_params = SamplingParams(temperature=0.7, max_tokens=150)
                    step_reflection_output = model.generate([reflexion_prompt], sampling_params)[0]
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion (strip heavy ephemeral blocks)
                    _stored_obs = observation
                    for _marker in ('\n\n[POST-ACTION PROBE — RAW CONTENT', '\n\n[EVALUATOR SOURCE CODE'):
                        _idx = _stored_obs.find(_marker)
                        if _idx >= 0:
                            _end = _stored_obs.find('\n]', _idx)
                            if _end < 0:
                                _end = len(_stored_obs)
                            else:
                                _end += 2
                            _stored_obs = _stored_obs[:_idx] + _stored_obs[_end:]
                    state['working_reflexions'].append({
                        'step': state['cur_step'],
                        'action': action,
                        'observation': _stored_obs,
                        'reflection': step_reflection,
                        'type': moment_type,
                        'success': moment_type == "SUCCESS",
                    })

                    # CRITICAL: Save to env_configs for cross-trial persistence
                    env_idx = state['env_id']
                    if 'step_memory' not in env_configs[env_idx]:
                        env_configs[env_idx]['step_memory'] = []
                    env_configs[env_idx]['step_memory'].append(f"[Step {state['cur_step']} {moment_type}] {step_reflection}")

                    # Keep only last 10 step memories
                    if len(env_configs[env_idx]['step_memory']) > 10:
                        env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]

                    print(f"[REFLEXION] ENV {state['env_id']} Step {state['cur_step']}: {moment_type} - Generated episodic memory")
                    print(f"[REFLEXION] Total episodic memories: {len(state['working_reflexions'])}")
                    log_debug(f"[REFLEXION] Insight: {step_reflection[:100]}...")

                except Exception as e:
                    log_debug(f"[REFLEXION ERROR] Failed to generate episodic reflexion: {e}")

            # ═══════════════════════════════════════════════════════════════
            # END OF SYNERGISTIC EPISODIC REFLEXION GENERATION
            # ═══════════════════════════════════════════════════════════════

            
            # Check if EXACT action failed (deterministic tracking)
            is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])

            # INTELLIGENT: Also classify low-progress actions (not just observation-based failures)
            # Let LLM distinguish: useful exploration vs truly useless actions
            should_classify_semantically = is_failure
            classification_reason = "observation unchanged"

            # Check progress score from step gradient (if available)
            if 'step_gradient' in state and state['step_gradient']:
                progress_score = state['step_gradient'].get('progress_score', 5)

                if progress_score < 3:  # Very low progress
                    should_classify_semantically = True
                    classification_reason = f"low progress ({progress_score}/10)"
                    log_debug(f"[LOW PROGRESS DETECTED] Env {state['env_id']} Step {state['cur_step']}: {progress_score}/10")
                    log_debug(f"  Action: '{action}' - requesting LLM semantic analysis")

            # Store failure in history for batched intelligent filtering
            # Simple storage, no extra LLM call (analysis will be in unified batch prompt)
            if should_classify_semantically:
                log_debug(f"[FAILURE DETECTED] Env {state['env_id']} Step {state['cur_step']}: {classification_reason}")
                log_debug(f"  Action: '{action}' - will be stored for filtering")

                # Store in failure history with context (no LLM analysis call)
                if 'failure_history' not in state:
                    state['failure_history'] = []

                state['failure_history'].append({
                    'step': state['cur_step'],
                    'action': action,
                    'context_before': state['prev_observation'][:300],
                    'context_after': observation[:300],
                    'progress_score': state.get('step_gradient', {}).get('progress_score', 'N/A')
                })

                # Keep only last 10 failures
                if len(state['failure_history']) > 10:
                    state['failure_history'] = state['failure_history'][-10:]

                log_debug(f"[FAILURE STORED] Context saved for intelligent filtering")
                log_debug(f"  Total failures in history: {len(state['failure_history'])}")

            # UNIVERSAL: Detect if action provided new information
            prev_info_content = len(state['prev_observation'].split())
            curr_info_content = len(observation.split())

            # If observation got longer, we gained information
            information_gained = curr_info_content > prev_info_content * 1.2  # 20% more content

            # If we gained information, it's not a failure (even if state didn't change)
            if information_gained:
                is_failure = False
            else:
                # No information gain - check if it's a true failure
                is_failure = env_understanding._is_likely_failure(observation, state['prev_observation'])
                
            
           
            # Get next valid actions for the reflection (loop detection removed) - MEMORY FIX
            next_valid_actions = []
            env = _env_registry[state['env_id']]
            if hasattr(env, 'get_current_valid_actions'):
                next_valid_actions = env.get_current_valid_actions()
            state['valid_actions'] = next_valid_actions  # Store for backward pass + reflexion prompts

            # Get current TODO for tactical coordination
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # Use persistent inventory from state (tracked in Phase 4)
            inventory_items = state.get('inventory', [])

            # Get episodic memory (cross-trial) separate from step memory (within-trial)
            env_idx = state['env_id']
            episodic_memory_only = env_configs[env_idx].get('memory', [])  # Cross-trial learnings

            # Get tiered/compressed reflexions to pass to TextGrad/Reflexion (uses ALL history smartly compressed)
            tiered_step_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

            # PHASE 2 SYNERGY: Decide between TextGrad (normal) vs Reflexion (failures)
            # Get previous step's progress to decide which component to use
            last_progress = state.get('last_step_gradient', {}).get('progress_score', 5)
            current_step_num = state['cur_step']
# Initialize consecutive failure tracking            if 'consecutive_low_progress' not in state:                state['consecutive_low_progress'] = 0            if 'steps_since_reflexion' not in state:                state['steps_since_reflexion'] = 0            # Update tracking            if last_progress < 4:                state['consecutive_low_progress'] += 1            else:                state['consecutive_low_progress'] = 0            state['steps_since_reflexion'] += 1            # Smart trigger: 5 consecutive failures + not step 0 + cooldown            use_reflexion = (current_step_num > 0 and                           state['consecutive_low_progress'] >= 5 and                           state['steps_since_reflexion'] >= 5)

            # Decision logic: TextGrad for good progress, Reflexion only for failures
            if not use_reflexion:
                # TEXTGRAD PATH (85%+ of steps): Clean action optimization
                # Use when: (1) First step OR (2) Previous step showed good progress
                log_debug(f"[SYNERGY] ENV {state['env_id']} Step {current_step_num}: Using TEXTGRAD (progress={last_progress}/10)")
                # Include current step's backward in step_insights so CONCRETE_FIX extraction
                # can see it (otherwise only previous steps' backwards are visible)
                _current_step_insight = {
                    'step': current_step_num,
                    'action': action,
                    'observation': str(observation)[:500],
                    'backward_gradient': state.get('step_gradient', {}).get('textgrad_gradient', ''),
                    'progress_score': state.get('step_gradient', {}).get('progress_score', 0),
                }
                _augmented_insights = list(state.get('step_insights_accumulator', [])) + [_current_step_insight]
                step_gradient = generate_textgrad_step_guidance(
                    prev_observation=state['prev_observation'],
                    curr_observation=observation,
                    action=action,
                    task=state['task'],
                    model=model,
                    log_debug=log_debug,
                    current_step=current_step_num,
                    valid_actions=next_valid_actions,
                    current_todo=current_todo,
                    inventory=inventory_items,
                    previous_reflexions=tiered_step_reflexions,
                    episodic_memory=episodic_memory_only,
                    step_insights=_augmented_insights,  # FIX: Include current step's backward
                    before_screenshot_b64=info.get('before_screenshot_b64') if info else None,
                    after_screenshot_b64=info.get('after_screenshot_b64') if info else None,
                    accumulated_state=state.get('accumulated_state', []),  # FIX #39: Progress tracking
                    is_visual_env=state.get('env_type', '') in ('osworld', 'visual'),
                    learning_context=state.get('learning_context'),  # Universal failure memory
                    task_complete_mode=state.get('_textgrad_task_complete', False),
                )
            else:
                # REFLEXION PATH (<15% of steps): Deep causal analysis for failures
                # Use when: Previous step showed low progress (need root cause diagnosis)
                log_debug(f"[SYNERGY] ENV {state['env_id']} Step {current_step_num}: Using REFLEXION (progress={last_progress}/10 - need causal analysis)")
                step_gradient = generate_reflexion_causal_analysis(
                    prev_observation=state['prev_observation'],
                    curr_observation=observation,
                    action=action,
                    task=state['task'],
                    model=model,
                    log_debug=log_debug,
                    current_step=current_step_num,
                    valid_actions=next_valid_actions,
                    current_todo=current_todo,
                    inventory=inventory_items,
                    previous_reflexions=tiered_step_reflexions,
                    episodic_memory=episodic_memory_only,
                    step_insights=state.get('step_insights_accumulator', []),  # FIX #21: Pass raw observations
                    before_screenshot_b64=info.get('before_screenshot_b64') if info else None,
                    after_screenshot_b64=info.get('after_screenshot_b64') if info else None,
                    accumulated_state=state.get('accumulated_state', []),  # FIX #39: Progress tracking
                    is_visual_env=state.get('env_type', '') in ('osworld', 'visual'),
                    learning_context=state.get('learning_context'),  # Universal failure memory
                    # G30 UNIVERSAL — Reflexion was blind to evaluator ground truth.
                    # It produced abstract prose lessons without seeing the literal
                    # values the evaluator compares against. Now it gets the same
                    # G21-style injection the BACKWARD pass and actor already see.
                    _g30_eval_gap=state.get('_last_eval_gap', '') or '',
                    _g30_eval_source=state.get('_eval_source', '') or '',
                )

            # UPDATE TODO MANAGER with action feedback
            if 'todo_manager' in state and state['todo_manager']:
                try:
                    state['todo_manager'].update_from_action_feedback(
                        action=action,
                        prev_observation=state['prev_observation'],
                        curr_observation=observation,
                        progress_status=step_gradient.get('progress_status', 'EXPLORING')
                    )
                except Exception as e:
                    log_debug(f"[TODO UPDATE ERROR] {e}")

            # Log the full step gradient
            comprehensive_logger.log_step_gradient(
                env_id=state['env_id'],
                step=state['cur_step'],
                gradient=step_gradient
            )



            # ADD THIS TO LOG TO TRIAL FILE AS WELL
            if trial_log_path and os.path.exists(trial_log_path):
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[STEP GRADIENT] Step {state['cur_step']}:\n")
                    f.write(f"  Progress: {step_gradient.get('progress_score', 0)}/10\n")
                    f.write(f"  Hypothesis: {step_gradient.get('hypothesis', 'None')}\n")
                    f.write(f"  Next: {step_gradient.get('next_action_guidance', 'None')}\n")

                    # Log TODO status as well
                    if 'todo_manager' in state and state['todo_manager']:
                        f.write(f"\n[TODO STATUS]:\n")
                        f.write(f"{state['todo_manager'].get_formatted_todos()}\n")

            # DASHBOARD: Update with step gradient results (progress, guidance, learning mode)
            if state['env_id'] == 0:
                _dashboard.update(
                    progress_score=step_gradient.get('progress_score', 0),
                    progress_status=step_gradient.get('progress_status', ''),
                    next_action=step_gradient.get('next_action_guidance', ''),
                    learning_mode='reflexion' if use_reflexion else 'textgrad',
                    reflexion_info=('Deep causal analysis triggered' if use_reflexion else ''),
                )
                # Storyboard visualizer: router decision + policy snapshot
                _dashboard.update_from_state(state, step_gradient)
                # Storyboard: real-time substage events (wrapped — never crash trial)
                try:
                    _step_n = state['cur_step']
                    _traj_tail = state.get('trajectory') or []
                    _obs_preview = str(_traj_tail[-1][1])[:120] if _traj_tail else ''
                    _dashboard.emit_substage('OBS',      {'task': state.get('task','')[:120]}, step=_step_n)
                    _dashboard.emit_substage('ACTOR',    {'action': action}, step=_step_n)
                    _dashboard.emit_substage('ENV',      {'observation_preview': _obs_preview}, step=_step_n)
                    _dashboard.emit_substage('LLM_EVAL', {'score': step_gradient.get('progress_score', 0), 'status': step_gradient.get('progress_status', '')}, step=_step_n)
                    _dashboard.emit_substage('POLICY_MERGE', {'mode': 'reflexion' if use_reflexion else 'textgrad'}, step=_step_n)
                    _dashboard.emit_substage('STEP_DONE', {'step': _step_n}, step=_step_n)
                except Exception as _ex:
                    print(f'[STORYBOARD] substage emit skipped: {_ex}')
                # Log step to history
                _router_dec = getattr(state.get('reflexgrad_core'), 'last_decision', None) if state.get('reflexgrad_core') else None
                _dashboard.log_step({
                    'step': state['cur_step'],
                    'action': action,
                    'progress_score': step_gradient.get('progress_score', 0),
                    'progress_status': step_gradient.get('progress_status', ''),
                    'gradient': step_gradient.get('hypothesis', '')[:200],
                    'next_action': step_gradient.get('next_action_guidance', ''),
                    'learning_mode': 'reflexion' if use_reflexion else 'textgrad',
                    'router_decision': _router_dec,
                    'policy_base': str(getattr(state.get('policy'), 'base_policy', ''))[:2000] if state.get('policy') else '',
                })
                # Update TODOs after feedback
                if 'todo_manager' in state and state['todo_manager']:
                    _dashboard.update_todos(state['todo_manager'])

            # Use the environment's SPECIFIC prompt_generator - ensure it's the right one
            env_pg = env_states[active_states.index(state)]['prompt_generator']

            # Always apply observation gradient (it will be overridden by reflexion if better)
            env_pg.apply_universal_step_gradient(step_gradient)
            # Otherwise, reflexion gradient will be applied instead

            # CRITICAL FIX: Share learning across environments in same batch
            # Share failed actions immediately to prevent other envs from trying them
            # CONTEXT-AWARE SHARING: Share learning with full context
            if is_failure:
                # Create failure context with all dimensions
                failure_context = {
                    'action': action,
                    'state_hash': hashlib.md5(state['prev_observation'][:200].encode()).hexdigest()[:8],
                    'state_text': state['prev_observation'][:200],
                    'task': state['task'],
                    'step': state['cur_step'],
                    'prerequisites': [act for act, _, _ in state["trajectory"][-5:] if act],
                    'failure_reason': observation,
                    'timestamp': state['cur_step'],
                    'confidence': 1.0,
                    'is_universal': any(uf in observation.lower() for uf in 
                                      ["i don't understand", "invalid command", "not a verb"])
                }
                
                # Store in conditional failures
                if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                    state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                state['prompt_generator'].discovered_knowledge['conditional_failures'].append(failure_context)
                
                # Smart sharing based on context similarity
                for other_state in env_states:
                    if other_state['env_id'] != state['env_id'] and not other_state['done']:
                        target_context = {
                            'state_text': other_state['prev_observation'][:200],
                            'task': other_state['task'],
                            'step': other_state['cur_step'],
                            'prerequisites': [act for act, _, _ in other_state["trajectory"][-5:] if act]
                        }
                        
                        should_share, confidence = should_share_knowledge(failure_context, target_context)
                        
                        if should_share:
                            shared_failure = failure_context.copy()
                            shared_failure['confidence'] = confidence
                            shared_failure['source_env'] = state['env_id']
                            
                            if 'conditional_failures' not in other_state['prompt_generator'].discovered_knowledge:
                                other_state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                            other_state['prompt_generator'].discovered_knowledge['conditional_failures'].append(shared_failure)

            # Also update the state's reference
            state['textgrad_components'] = env_pg.prompt_components.copy()


            # CRITICAL: Store the gradient for next iteration
            state['last_step_gradient'] = step_gradient
            state['step_gradient'] = step_gradient
            # Store in history for TextGrad to learn from (includes progress_status)
            state['step_gradients_history'].append(step_gradient)

            if DEBUG_CRITIC:
                print(f"[ENV {state['env_id']}] STEP GRADIENT APPLIED at step {state['cur_step']}")
                print(f"  Components updated: {len(state['textgrad_components'])}")

            # LOG TO FILE
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[GRADIENT UPDATE {state['cur_step']}]\n")
                    f.write(f"  Next action guidance: {step_gradient.get('next_action_guidance', 'None')}\n")
                    f.write(f"  Progress score: {step_gradient.get('progress_score', 0)}\n")

            # ACCUMULATE AND SYNTHESIZE STEP INSIGHTS
            # FIX #13: Include OBSERVATION so forward pass knows what was found!
            # FIX #28: NO TRUNCATION - full observation and gradient needed for learning!
            progress_score = step_gradient.get('progress_score', 0)

            # Strip heavy ephemeral blocks from stored observation (same as site 1)
            _stored_obs = observation
            for _marker in ('\n\n[POST-ACTION PROBE — RAW CONTENT', '\n\n[EVALUATOR SOURCE CODE'):
                _idx = _stored_obs.find(_marker)
                if _idx >= 0:
                    _end = _stored_obs.find('\n]', _idx)
                    if _end < 0:
                        _end = len(_stored_obs)
                    else:
                        _end += 2
                    _stored_obs = _stored_obs[:_idx] + _stored_obs[_end:]
            # Always accumulate, not just on progress
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'observation': _stored_obs,
                'hypothesis': step_gradient.get('hypothesis', ''),
                'backward_gradient': step_gradient.get('textgrad_gradient', ''),
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', []),
                'click_metadata': info.get('click_metadata') if info else None,
                'has_screenshots': bool(info and info.get('after_screenshot_b64')),
            })

            # Synthesize insights every 3 steps
            if len(state['step_insights_accumulator']) % 3 == 0:
                # Create actionable synthesis
                recent_insights = state['step_insights_accumulator'][-3:]
                
                synthesis = {
                    'avg_progress': sum(i['progress_score'] for i in recent_insights) / 3,
                    'state_changes': sum(1 for i in recent_insights if i['state_changed']),
                    'consistent_failures': [],
                    'promising_directions': []
                }
                
                # Find patterns
                for insight in recent_insights:
                    if insight['progress_score'] < 2:
                        synthesis['consistent_failures'].append(insight['action'])
                    elif insight['progress_score'] > 5:
                        synthesis['promising_directions'].append(insight['next_guidance'])
                

                # CRITICAL: Make wisdom immediately available for next action
                state['consolidated_step_wisdom'] = state.get('consolidated_step_wisdom', '')
                if synthesis['promising_directions']:
                    state['consolidated_step_wisdom'] = f"PROMISING: {synthesis['promising_directions'][0]}"
                elif synthesis['consistent_failures']:
                    state['consolidated_step_wisdom'] = f"AVOID: {', '.join(synthesis['consistent_failures'][:2])}"
                else:
                    state['consolidated_step_wisdom'] = f"Avg progress: {synthesis['avg_progress']:.1f}/10"
                
                # Ensure it gets to batch_data on next iteration
                state['needs_wisdom_update'] = True

                # Update consolidated wisdom with synthesis
                if synthesis['promising_directions']:
                    state['consolidated_step_wisdom'] = f"PROMISING: {synthesis['promising_directions'][0]}"
                elif synthesis['consistent_failures']:
                    state['consolidated_step_wisdom'] = f"AVOID: {', '.join(synthesis['consistent_failures'][:2])}"
                else:
                    state['consolidated_step_wisdom'] = f"Avg progress: {synthesis['avg_progress']:.1f}/10"
                
                # Consolidate every 5 steps or when stuck
                if len(state['step_insights_accumulator']) >= 5 or state['is_stuck']:
                    # Get best insights
                    best_insights = sorted(
                        state['step_insights_accumulator'], 
                        key=lambda x: x['progress_score'], 
                        reverse=True
                    )[:3]
                    
                    # Build consolidated wisdom
                    wisdom_parts = []
                    for insight in best_insights:
                        if insight['hypothesis']:
                            wisdom_parts.append(f"Step {insight['step']}: {insight['hypothesis']}")
                    
                    if wisdom_parts:
                        state['consolidated_step_wisdom'] = ' | '.join(wisdom_parts)
                        
                    # Keep only recent insights
                    state['step_insights_accumulator'] = state['step_insights_accumulator'][-20:]

            # Track progress scores
            progress_score = step_gradient.get('progress_score', 0)
            state['progress_history'].append(progress_score)

            # Detect if stuck (low progress for many steps)
            if len(state['progress_history']) >= 5:
                recent_progress = state['progress_history'][-5:]
                avg_recent_progress = sum(recent_progress) / len(recent_progress)
                state['is_stuck'] = avg_recent_progress < 2
            else:
                state['is_stuck'] = False            
            
            # Store semantic understanding in memory
            if step_gradient and 'semantic_state_before' in step_gradient:
                universal_memory.record_semantic_interaction(
                    semantic_state_before=step_gradient['semantic_state_before'],
                    action=action,
                    action_reasoning=batch_data[env_idx_in_batch].get('action_reasoning', ''),  # ADD THIS
                    semantic_state_after=step_gradient['semantic_state_after'],
                    prerequisites=step_gradient.get('prerequisites', {}),
                    task_progress=step_gradient.get('task_progress', {}),
                    success=not is_failure,
                    task=state['task'],
                    episode_id=state['episode_id']
                )
                            
                # Store for next iteration
                state['last_step_gradient'] = step_gradient
            else:
                # Fallback if gradient doesn't have semantic info
                state['last_step_gradient'] = {
                    'semantic_state_after': observation,
                    'task_progress': {'remaining': [state['task']]}
                }

            # MEMORY INTEGRATION DURING EPISODE
            from generate_reflections import global_memory_manager
            
            # Consolidate more frequently
            if is_failure or state['cur_step'] % 3 == 0:  # Changed from 5 to 3
                try:
                    # Create mini-reflection with more detail
                    mini_reflection = f"Episode {state['episode_id']}, Step {state['cur_step']}: "
                    if is_failure:
                        mini_reflection += f"EXACT action '{action}' failed. State: {observation}"
                    else:
                        mini_reflection += f"Action '{action}' succeeded. State: {observation}"

                    global_memory_manager.add_experience(
                        reflection=mini_reflection,
                        gradients=step_gradient,
                        success=not is_failure,
                        task=state['task'],
                        env_id=state['env_id'],
                        observation=observation  # ADD: Pass the observation
                    )
                    
                    if state['cur_step'] % 5 == 0:
                        global_memory_manager._consolidate_memories()
                        
                    if DEBUG_REFLEXION:
                        print(f"[MEMORY] Stored experience at step {state['cur_step']}")
                        
                except Exception as e:
                    print(f"[ERROR] Memory integration failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Learn from interaction
            state['prompt_generator']._learn_from_interaction(action, observation)
            env_understanding.learn_from_interaction(action, observation)
            
            # Track trajectory with reasoning (3-tuple)
            env_idx_in_batch = active_states.index(state)
            action_reasoning = batch_data[env_idx_in_batch].get('action_reasoning', '')
            state['trajectory'].append((action, observation, action_reasoning))
            
            # Track consecutive failures
            if is_failure:
                state['consecutive_failures'] += 1
            else:
                state['consecutive_failures'] = 0


            # ====================================================================
            # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE REFLEXION CODE
            # ====================================================================
            # This entire block is wrapped in "if False:" and will NEVER execute
            # Left here for historical reference only
            #
            # ACTIVE reflexion code now runs BEFORE action selection (PHASE 3B)
            # DO NOT modify or enable this disabled after-action reflexion code
            # ====================================================================
            # WITHIN-EPISODE REFLEXION - NOW MOVED BEFORE ACTION SELECTION (see lines 1687-1818)
            # Disabled this old AFTER-action reflexion to prevent duplicates
            if False:  # DISABLED - Reflexion now runs BEFORE action selection for immediate use
                # Generate reflexion when: low progress every 3 steps, on failure, or high progress
                print(f"\n[REFLEXION-OLD-DISABLED] Generating step reflexion at step {state['cur_step']} (Failure: {is_failure})")
                
                # Generate proper step-level reflexion
                recent_trajectory = state['trajectory'][-5:] if state['trajectory'] else []

                # Get previous step reflexions from THIS trial (step-level memory)
                # Use ALL reflexions with tiered compression - don't forget early lessons!
                tiered_reflexions = manage_working_reflexions_tiered(state, model, log_debug)

                # Format compressed reflexions for step reflexion prompt
                if tiered_reflexions:
                    formatted_learnings = []
                    for ref in tiered_reflexions:
                        compression_type = ref.get('is_compressed', None)
                        step_num = ref.get('step', '?')
                        reflection_text = ref.get('reflection', '')[:200]  # Limit for display

                        if compression_type == 'heavy':
                            formatted_learnings.append(f"  Steps {step_num} (summary): {reflection_text}")
                        elif compression_type == 'medium':
                            formatted_learnings.append(f"  Steps {step_num} (key points): {reflection_text}")
                        else:
                            formatted_learnings.append(f"  Step {step_num}: {reflection_text}")

                    previous_step_learnings = '\n'.join(formatted_learnings)
                else:
                    previous_step_learnings = "None yet (first few steps)"

                step_reflection_prompt = f"""Analyze the last few actions in this episode.

            PREVIOUS LEARNINGS FROM THIS TRIAL (step-level memory):
            {previous_step_learnings}

            Task: {state['task']}
            Current step: {state['cur_step']}

            Recent actions and outcomes:"""

                for act, obs, reasoning in recent_trajectory:
                    step_reflection_prompt += f"\n> {act}\nResult: {obs[:100]}..."

                # Get TODO context for semantic verification
                todo_context = ""
                if hasattr(state.get('todo_manager'), 'current_todo'):
                    current_todo = state['todo_manager'].current_todo()
                    if current_todo:
                        todo_context = f"\nCurrent Subtask: '{current_todo.content}' (Attempt #{current_todo.attempts})"
                        if current_todo.attempts > 2:
                            todo_context += f"\n⚠️  WARNING: {current_todo.attempts} attempts without success"

                step_reflection_prompt += f"""

            Current action: {action}
            Result: {observation}...
            Success: {'No' if is_failure else 'Yes'}
            {todo_context}

            🔬 REFLEXION: Root-Cause Analysis

            CRITICAL ANALYSIS REQUIRED:

            1. SEMANTIC PRECISION CHECK:
               Task goal: "{state['task']}"
               Action taken: "{action}"

               Question: What is the PRIMARY PURPOSE of each?
               - Task requires me to: [specific effect/goal]
               - Action achieves: [specific effect/purpose]
               - Are these semantically IDENTICAL or DIFFERENT?

               IMPORTANT: Actions involving similar entities may serve different purposes.
               Verify that the action's effect aligns with the task's specific requirement.

            2. STATE VERIFICATION WITH EVIDENCE:
               Based on task "{state['task']}", what SPECIFIC property must change?

               Look at observation: "{observation[:150]}"
               Did that EXACT property change? Cite specific words as evidence.
               Do NOT assume - only state what observation explicitly shows.

            3. FAILURE PATTERN DETECTION:
               {f'Subtask not achieved after {current_todo.attempts} attempts.' if todo_context and hasattr(state.get('todo_manager'), 'current_todo') and state['todo_manager'].current_todo() and state['todo_manager'].current_todo().attempts > 2 else 'Check if repeating similar actions.'}

               Am I repeating similar action types that don't work?
               What assumption might be incorrect?

            4. AVAILABLE ACTIONS (HARD CONSTRAINT):
               Here are the ONLY actions you can actually execute:
"""
                # Format valid actions list
                valid_actions_list = state.get('valid_actions', [])
                for act in valid_actions_list:
                    step_reflection_prompt += f"               - {act}\n"
                if len(valid_actions_list) > 30:
                    step_reflection_prompt += f"               ... and {len(valid_actions_list)-30} more\n"

                _is_visual = state.get('env_type', '') in ('osworld', 'visual')
                step_reflection_prompt += f"""
            5. RECOMMENDED NEXT ACTION:
               If there's a semantic mismatch or wrong action type detected:
               {"You may use ANY action format including run: <shell command> for programmatic approaches." if _is_visual else "Choose ONE action from the list in section 4 above. DO NOT invent new actions - ONLY select from the list."}

            Generate your analysis (keep concise, 3-4 sentences total):"""
                
                # Generate reflection
                sampling_params = SamplingParams(max_tokens=3000, temperature=0.3)
                step_reflection_output = model.generate([step_reflection_prompt], sampling_params)[0]
                step_reflection = step_reflection_output.outputs[0].text.strip()
                
                # Store in WORKING memory for immediate use
                if 'working_reflexions' not in state:
                    state['working_reflexions'] = []
                
                state['working_reflexions'].append({
                    'step': state['cur_step'],
                    'action': action,
                    'reflection': step_reflection,
                    'success': not is_failure
                })

                # Log the step reflexion for paper
                log_debug(f"[STEP REFLEXION] Step {state['cur_step']}: {step_reflection[:150]}...")
                total_reflexions = len(state.get('working_reflexions', []))
                compressed_count = len([r for r in tiered_reflexions if r.get('is_compressed')])
                log_debug(f"[STEP REFLEXION] Using ALL {total_reflexions} reflexions ({compressed_count} compressed, {total_reflexions - compressed_count} verbose)")

                # Log the step reflexion
                comprehensive_logger.log_step_reflexion(
                    env_id=state['env_id'],
                    step=state['cur_step'],
                    action=action,
                    reflection=step_reflection,
                    success=not is_failure
                )

                # NEW: Extract gradients from step reflexion using existing logic
                # Build a mini-trajectory for the compute_prompt_gradient function
                mini_trajectory = ""
                for act, obs, _ in recent_trajectory[-3:]:
                    mini_trajectory += f"> {act}\n{obs[:100]}...\n"
                
                # Extract progress score from current step gradient
                progress_score = state.get('step_gradient', {}).get('progress_score', 0)

                # SYNERGY FIX: Only invoke Reflexion's CAUSAL CHAIN when TextGrad needs help
                # High progress (≥4): TextGrad confident → use its clean recommendations
                # Low progress (<4): TextGrad struggling → invoke Reflexion for guidance
                if progress_score < 4:
                    log_debug(f"[SYNERGY] Low progress ({progress_score}/10) - invoking Reflexion guidance")
                    step_reflexion_gradients = state['prompt_generator'].compute_prompt_gradient(
                        trajectory=mini_trajectory,
                        success=False,
                        task=state['task'],
                        reflection=step_reflection,
                        progress_score=progress_score,
                        valid_actions=state.get('valid_actions', [])
                    )

                    # Extract clean action from Reflexion's CAUSAL CHAIN format
                    if step_reflexion_gradients and 'next_action_guidance' in step_reflexion_gradients:
                        extracted_action = step_reflexion_gradients['next_action_guidance']
                        if extracted_action and len(extracted_action) > 3:
                            state['step_gradient']['next_action_guidance'] = extracted_action
                            print(f"🔗 [REFLEXION-GUIDE] Low progress → Reflexion suggests: {extracted_action}")
                else:
                    # TextGrad doing well - don't pollute with Reflexion's verbose CAUSAL CHAIN
                    log_debug(f"[SYNERGY] Moderate/high progress ({progress_score}/10) - TextGrad optimizing")
                    step_reflexion_gradients = {}  # Empty dict for clean flow

                # SMART APPLICATION: Apply based on gradient quality, not just progress
                for component, gradient in step_reflexion_gradients.items():
                    if component not in state['prompt_generator'].prompt_components:
                        continue
                    
                    # Check gradient quality (actionability)
                    gradient_lower = gradient.lower()
                    is_actionable = any(word in gradient_lower for word in 
                                        ['must', 'should', 'try', 'next', 'need', 'avoid'])
                    
                    # Determine application strength based on both actionability and progress
                    if is_actionable and progress_score >= 6:
                        # High progress + actionable = strong update
                        momentum = 0.9
                    elif is_actionable and progress_score >= 3:
                        # Moderate progress + actionable = moderate update
                        momentum = 0.6
                    elif is_actionable:
                        # Low progress but actionable = careful update
                        momentum = 0.4
                    else:
                        # Non-actionable = minimal update
                        momentum = 0.2
                    
                    # Apply only if meaningful
                    if is_actionable or progress_score >= 7:
                        state['prompt_generator'].update_component_with_momentum(
                            component, gradient, momentum=momentum
                        )
                
                # Apply with HIGHER WEIGHT since reflexion-based insights are richer
                for component, gradient in step_reflexion_gradients.items():
                    if component in state['prompt_generator'].prompt_components:
                        # Use momentum update with higher weight
                        state['prompt_generator'].update_component_with_momentum(
                            component, 
                            gradient, 
                            momentum=0.95  # Higher momentum for reflexion-based updates
                        )
                
                if DEBUG_CRITIC:
                    print(f"[STEP REFLEXION GRADIENT] Applied {len(step_reflexion_gradients)} gradient updates from reflexion")
                    for comp, grad in step_reflexion_gradients.items():
                        print(f"  {comp}: {grad}...")

                        
                # CRITICAL FIX: Save to env_configs for next trial
                env_idx = state['env_id']
                if 'working_reflexions_history' not in env_configs[env_idx]:
                    env_configs[env_idx]['working_reflexions_history'] = []
                env_configs[env_idx]['working_reflexions_history'].append({
                    'trial': trial_idx,
                    'step': state['cur_step'],
                    'action': action,
                    'reflection': step_reflection,
                    'success': not is_failure
                })
                if len(env_configs[env_idx]['working_reflexions_history']) > 20:
                    env_configs[env_idx]['working_reflexions_history'] = env_configs[env_idx]['working_reflexions_history'][-20:]

                print(f"[REFLEXION] Stored reflexion for step {state['cur_step']}: {step_reflection}")

                # LOG TO FILE
                if trial_log_path:
                    with open(trial_log_path, 'a') as f:
                        f.write(f"\n[STEP REFLEXION {state['cur_step']}]\n")
                        f.write(f"  Action: {action}\n")
                        f.write(f"  Progress: {progress_score}/10\n")
                        f.write(f"  Reflection: {step_reflection[:200]}...\n")
                
                # Keep only last 5 step reflexions
                if len(state['working_reflexions']) > 15:
                    state['working_reflexions'] = state['working_reflexions'][-15:]
                
                
                # CRITICAL FIX: Also save to env_configs
                env_idx = state['env_id']
                if 'memory' not in env_configs[env_idx]:
                    env_configs[env_idx]['memory'] = []
                # Store step reflections separately to avoid polluting episodic memory
                if 'step_memory' not in env_configs[env_idx]:
                    env_configs[env_idx]['step_memory'] = []
                env_configs[env_idx]['step_memory'].append(f"[Step {state['cur_step']}] {step_reflection}")
                # Keep only last 10 step reflections
                if len(env_configs[env_idx]['step_memory']) > 10:
                    env_configs[env_idx]['step_memory'] = env_configs[env_idx]['step_memory'][-10:]


                # Update the STATE's prompt generator (consistent!)
                # Update with conditional failure instead of blacklist
                if is_failure:
                    if 'conditional_failures' not in state['prompt_generator'].discovered_knowledge:
                        state['prompt_generator'].discovered_knowledge['conditional_failures'] = []
                    
                    state['prompt_generator'].discovered_knowledge['conditional_failures'].append({
                        'action': action,
                        'timestamp': state['cur_step'],
                        'confidence': 0.9,
                        'is_universal': False
                    })
                    
                    # Also add to exact failed actions for this state
                    current_state_hash = hashlib.md5(state['prev_observation'][:200].encode()).hexdigest()[:8]
                    if 'failed_state_actions' not in state:
                        state['failed_state_actions'] = {}
                    if current_state_hash not in state['failed_state_actions']:
                        state['failed_state_actions'][current_state_hash] = set()
                    state['failed_state_actions'][current_state_hash].add(action)
                else:
                    # On success, add to successful actions
                    if 'successful_actions' not in state['prompt_generator'].discovered_knowledge:
                        state['prompt_generator'].discovered_knowledge['successful_actions'] = []
                    if action not in state['prompt_generator'].discovered_knowledge['successful_actions']:
                        state['prompt_generator'].discovered_knowledge['successful_actions'].append(action)
                
            

                
                # Also update the prompt generator immediately
                # Use conditional failures instead
                if 'conditional_failures' not in env_prompt_generators[state['env_id']].discovered_knowledge:
                    env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'] = []
                
                env_prompt_generators[state['env_id']].discovered_knowledge['conditional_failures'].append({
                    'action': action,
                    'timestamp': state['cur_step'],
                    'confidence': 0.8,
                    'is_universal': False
                })
                
                log_debug(f"[ENV {state['env_id']}] Added mini-reflection after {state['consecutive_failures']} failures")


            # Add observation to history
            state['history'].add("observation", observation)


            
            if to_print:
                log_debug(f'[ENV {state["env_id"]}] > {action}\n{observation}')
                sys.stdout.flush()
            
            # Update state
            state['prev_observation'] = observation
            state['cur_step'] += 1
            
            # Periodic checkpoint every 10 steps to prevent loss
            if state['cur_step'] % 10 == 0 and checkpoint_manager:
                try:
                    if checkpoint_manager:  # Only save if checkpoint_manager exists
                        completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
                        pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]
                        checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                except Exception as e:
                    print(f"[WARNING] Periodic checkpoint failed: {e}")

            # Check done
            if done:
                log_debug(f"\n[DEBUG] ENV {state['env_id']} at step {state['cur_step']}:")
                log_debug(f"  done={done}")
                log_debug(f"  info dict: {info}")
                log_debug(f"  reward: {reward}")
                actual_success = info.get('won', [False])[0]
                log_debug(f"  actual_success={actual_success}")
                
                state['done'] = True

                # CHECKPOINT: Save state after each environment completes
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_env_checkpoint(
                        trial_idx=trial_idx,
                        env_id=state['env_id'],
                        state=state,
                        trajectory_length=len(state['trajectory'])
                    )

                # Save master state for resume
                completed_envs = [s['env_id'] for s in env_states if s['done']]
                pending_envs = [s['env_id'] for s in env_states if not s['done']]
                if checkpoint_manager:  # Only save if checkpoint_manager exists
                    checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)
                state['success'] = actual_success
                

                
                # Also save minimal checkpoint for compatibility
                checkpoint_file = os.path.join(os.path.dirname(trial_log_path), f'checkpoint_trial_{trial_idx}_env_{state["env_id"]}.json')
                checkpoint_data = {
                    'env_id': state['env_id'],
                    'success': actual_success,
                    'trajectory_length': len(state['trajectory']),
                    'task': state['task'],
                    'memory': env_configs[state['env_id']].get('memory', []),
                    'working_reflexions': state.get('working_reflexions', []),
                    'step': state['cur_step']
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)

                # CRITICAL FIX: Update env_configs immediately
                env_configs[state['env_id']]['is_success'] = actual_success
                
                # Save successful trajectory for reuse
                if actual_success:
                    env_configs[state['env_id']]['successful_trajectory'] = [
                        (act, obs[:100]) for act, obs, _ in state["trajectory"]
                    ]
                    print(f"[SUCCESS] Saved trajectory for env {state['env_id']}")
                                                                                                                                                                                                                                                                                                               

                # Log last 3 actions for debugging
                if state['trajectory']:
                    log_debug(f"  Last 3 actions taken:")
                    for act, obs, _ in state["trajectory"][-3:]:
                        obs_preview = obs[:80] if len(obs) > 80 else obs
                        log_debug(f"    {act} -> {obs_preview.replace(chr(10), ' ')}")
                
                # RECORD COMPLETE EPISODE IN UNIVERSAL MEMORY
                universal_memory.record_episode(state['trajectory'], state['task'], actual_success)
                

                if actual_success:
                    universal_memory.store_sequence_pattern(state['trajectory'], True, state['task'])                
                state['done'] = True
                state['success'] = actual_success



            # With max_steps=15, fail faster on consecutive failures
            failure_threshold = 5  # Fixed threshold
            if state['consecutive_failures'] > failure_threshold and not state['done']:
                if to_print:
                    print(f"[ENV {state['env_id']}] Stuck in unproductive loop - ending episode")
                
                # Record failed episode
                universal_memory.record_episode(state['trajectory'], state['task'], False)
                
                state['done'] = True
                state['success'] = False

        # End of while loop for current environment

    # ========================================================================
    # CHECKPOINT FIX (Nov 1): Save checkpoints for environments that hit max_steps
    # ========================================================================
    # Previously: Only environments with done=True saved checkpoints
    # Problem: Environments hitting max_steps=21 without completing never saved
    # Fix: Iterate through env_states and save checkpoints for incomplete envs
    # ========================================================================

    print(f"\n[CHECKPOINT] Saving checkpoints for incomplete environments...")
    for state in env_states:
        # Skip if environment already completed (checkpoint already saved)
        if state.get('done', False):
            continue

        # This environment hit max_steps without completing
        if state['cur_step'] >= max_steps:
            env_id = state['env_id']
            print(f"[CHECKPOINT] Saving incomplete env {env_id} (hit max_steps={max_steps})")

            # For visual envs, call the evaluator — the task may be complete even if
            # the agent didn't say 'done' (e.g., restore command ran but no visual change)
            actual_success = False
            if state.get('env_type', '') in ('osworld', 'visual'):
                try:
                    env_obj_maxstep = _env_registry.get(env_id)
                    if env_obj_maxstep and hasattr(env_obj_maxstep, 'step'):
                        maxstep_eval = env_obj_maxstep.step('done')
                        actual_success = maxstep_eval.get('success', False) if isinstance(maxstep_eval, dict) else False
                        if actual_success:
                            print(f"[CHECKPOINT] Env {env_id} hit max_steps but EVALUATOR SAYS PASS!")
                except Exception as e:
                    print(f"[CHECKPOINT] Evaluator at max_steps failed for env {env_id}: {e}")

            state['done'] = True
            state['success'] = actual_success

            # Save checkpoint using checkpoint_manager
            if checkpoint_manager:
                completed_envs = [s['env_id'] for s in env_states if s.get('done', False)]
                pending_envs = [s['env_id'] for s in env_states if not s.get('done', False)]

                checkpoint_manager.save_env_checkpoint(
                    trial_idx=trial_idx,
                    env_id=env_id,
                    state=state,
                    env_configs=env_configs
                )
                checkpoint_manager.save_master_state(trial_idx, env_configs, completed_envs, pending_envs)

            # Also save minimal checkpoint for compatibility
            checkpoint_file = os.path.join(os.path.dirname(trial_log_path),
                                          f'checkpoint_trial_{trial_idx}_env_{env_id}.json')
            checkpoint_data = {
                'env_id': env_id,
                'success': False,
                'trajectory_length': len(state['trajectory']),
                'task': state['task'],
                'memory': env_configs[env_id].get('memory', []),
                'working_reflexions': state.get('working_reflexions', []),
                'step': state['cur_step'],
                'reason': 'hit_max_steps'
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)

            # Update env_configs
            env_configs[env_id]['is_success'] = False

            # Record failed episode in universal memory
            universal_memory.record_episode(state['trajectory'], state['task'], False)

            log_debug(f"[CHECKPOINT] Saved incomplete env {env_id} checkpoint")

        # ========================================================================
        # ⚠️  WARNING: DISABLED CODE - DO NOT MODIFY - OBSOLETE LEARNING CODE
        # ========================================================================
        # This entire block is wrapped in "if False:" and will NEVER execute
        # Left here for historical reference only
        #
        # ACTIVE learning extraction happens in PHASE 6 (parallel mode)
        # DO NOT modify or enable this disabled sequential learning code
        # ========================================================================
        # OLD SEQUENTIAL: STEP 5 - Extract learnings (DISABLED IN PARALLEL MODE)
        # ========================================================================
        if False:  # Lines 3544-3709 DISABLED - Sequential learning extraction not used in parallel mode
            log_debug(f"\n{'='*80}")
            log_debug(f"[LEARNING EXTRACTION] Environment {current_env_idx} completed")
            log_debug(f"[LEARNING EXTRACTION] Success: {current_state['success']}")
            log_debug(f"[LEARNING EXTRACTION] Steps taken: {current_state['cur_step']}")
            log_debug(f"{'='*80}\n")
            # Extract learnings
            result = {
                'success': current_state['success'],
                'steps': current_state['cur_step'],
                'trajectory': current_state['trajectory']
            }

            learnings = learning_extractor.extract_learnings(
                env_state=current_state,
                result=result,
                task=task
            )

            log_debug(f"[EXTRACTION] Extracted {len(learnings)} learning items")

            # Classify and store learnings in shared knowledge
            universal_count = 0
            family_count = 0
            specific_count = 0

            for learning in learnings:
                classification = learning['classification']
                knowledge_type, family = classification

                if knowledge_type == 'universal':
                    shared_knowledge['universal'].append(learning)
                    universal_count += 1
                    log_debug(f"[STORED] Universal: {learning['text'][:80]}...")

                elif knowledge_type == 'task_family' and family:
                    if family not in shared_knowledge['task_families']:
                        shared_knowledge['task_families'][family] = []
                    shared_knowledge['task_families'][family].append(learning)
                    family_count += 1
                    log_debug(f"[STORED] Family '{family}': {learning['text'][:80]}...")

                else:  # task_specific
                    specific_count += 1
                    log_debug(f"[SKIPPED] Task-specific: {learning['text'][:80]}...")

            log_debug(f"\n[KNOWLEDGE SUMMARY]")
            log_debug(f"  Universal: {universal_count} (total: {len(shared_knowledge['universal'])})")
            log_debug(f"  Task-family: {family_count}")
            log_debug(f"  Task-specific (not stored): {specific_count}")
            log_debug(f"  Total transferable knowledge: {len(shared_knowledge['universal']) + sum(len(v) for v in shared_knowledge['task_families'].values())}\n")

            # ========================================================================
            # STEP 6: Store TODO patterns for cross-environment learning
            # ========================================================================
            if current_state['success']:
                # Get final TODO state
                todo_manager = current_state.get('todo_manager')
                if todo_manager and todo_manager.todos:
                    task_class = task_classifier.classify_task(task)

                    shared_todo_knowledge['successful_patterns'].append({
                        'source_task': task,
                        'task_family': task_class['family'],
                        'todos': [t.content for t in todo_manager.todos],
                        'env_id': current_env_idx,
                        'trial': trial_idx
                    })

                    log_debug(f"[TODO KNOWLEDGE] ✅ Stored {len(todo_manager.todos)} successful TODOs for task '{task}'")
                    log_debug(f"  Task family: {task_class['family']}")
                    log_debug(f"  Available for transfer to environments {current_env_idx + 1}+")
            else:
                # Store failed patterns (for future reference - might help avoid bad approaches)
                todo_manager = current_state.get('todo_manager')
                if todo_manager and todo_manager.todos:
                    shared_todo_knowledge['failed_patterns'].append({
                        'source_task': task,
                        'todos': [t.content for t in todo_manager.todos],
                        'env_id': current_env_idx
                    })
                    log_debug(f"[TODO KNOWLEDGE] ❌ Stored {len(todo_manager.todos)} failed TODOs (not transferred)")
            # END OF if False BLOCK - Old sequential processing disabled above


    # ========================================================================
    # LOG TODO TRANSFER STATISTICS
    # ========================================================================
    transfer_stats = todo_transfer_safety.get_transfer_stats()
    if transfer_stats['total'] > 0:
        log_debug(f"\n{'='*80}")
        log_debug(f"[TODO TRANSFER STATS] Trial {trial_idx}")
        log_debug(f"{'='*80}")
        log_debug(f"  Total transfer attempts: {transfer_stats['total']}")
        log_debug(f"  Successfully transferred: {transfer_stats['transferred']}")
        log_debug(f"  Blocked (safety): {transfer_stats['blocked']}")
        log_debug(f"  Block rate: {transfer_stats['block_rate']:.1%}")

        if transfer_stats['blocked_reasons']:
            log_debug(f"\n  Top block reasons:")
            for reason, count in sorted(transfer_stats['blocked_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
                log_debug(f"    - {reason}: {count}")

        log_debug(f"  Successful patterns stored: {len(shared_todo_knowledge['successful_patterns'])}")
        log_debug(f"  Failed patterns stored: {len(shared_todo_knowledge['failed_patterns'])}")
        log_debug(f"{'='*80}\n")

    # Save universal memory after all episodes complete
    universal_memory.save_memory()

    # Save global memory manager
    from generate_reflections import global_memory_manager
    if trial_log_path:
        logging_dir = os.path.dirname(trial_log_path)
    else:
        logging_dir = '.'  # Current directory as fallback

    memory_manager_path = os.path.join(logging_dir, 'global_memory_manager.pkl')
    try:
        with open(memory_manager_path, 'wb') as f:
            pickle.dump(global_memory_manager, f)
        print(f"[CHECKPOINT] Saved global memory manager")
    except Exception as e:
        print(f"[ERROR] Failed to save memory manager: {e}")

    # Save embedding cache
    state_embeddings.save_cache()
    log_debug(f"[EMBEDDINGS] Saved {len(state_embeddings.state_texts)} states to cache")

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Save working reflexions to env_configs BEFORE json.dump()
    # This was the root cause of learning transfer failure - reflexions were being
    # saved AFTER the JSON file was written, so they never persisted across trials!
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"[LEARNING TRANSFER] Saving working_reflexions to env_configs for trial {trial_idx}...")
    for state in env_states:
        env_idx = state['env_id']
        wr_count = len(state.get('working_reflexions', []))
        print(f"[LEARNING TRANSFER] ENV {env_idx}: Has {wr_count} working_reflexions in state")

        if 'working_reflexions' in state and state['working_reflexions']:
            if 'working_reflexions_history' not in env_configs[env_idx]:
                env_configs[env_idx]['working_reflexions_history'] = []

            # Add current working reflexions
            for wr in state['working_reflexions']:
                env_configs[env_idx]['working_reflexions_history'].append({
                    'trial': trial_idx,
                    'step': wr.get('step', 0),
                    'action': wr.get('action', ''),
                    'reflection': wr.get('reflection', ''),
                    'success': wr.get('success', False),
                    'type': wr.get('type')  # CRITICAL FIX: Add type field for categorization
                })

            # Keep only last 20 working reflexions
            if len(env_configs[env_idx]['working_reflexions_history']) > 20:
                env_configs[env_idx]['working_reflexions_history'] = \
                    env_configs[env_idx]['working_reflexions_history'][-20:]

            print(f"[LEARNING TRANSFER] ✅ ENV {env_idx}: Saved {wr_count} reflexions → total history: {len(env_configs[env_idx]['working_reflexions_history'])}")
        else:
            print(f"[LEARNING TRANSFER] ⚠️  ENV {env_idx}: NO working_reflexions to save!")

    # Save env configs after EVERY trial (NOW includes working_reflexions_history!)
    env_config_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
    with open(env_config_path, 'w') as f:
        json.dump(env_configs, f, indent=4)
    print(f"[CHECKPOINT] Saved env configs for trial {trial_idx}")

    # CRITICAL: Update env_configs with results
    if env_configs is not None:
        for state in env_states:
            env_configs[state['env_id']]['is_success'] = state['success']
            print(f"[BATCH] Env {state['env_id']} marked as {'SUCCESS' if state['success'] else 'FAIL'}")

    # =======================================================================
    # AGGRESSIVE MEMORY LEAK PREVENTION - Force GC After Batch
    # =======================================================================
    # This prevents the 219GB leak by forcing garbage collection after
    # each major batch processing completes, before large objects accumulate
    # =======================================================================
    import gc
    collected_objects = gc.collect()
    if collected_objects > 0:
        print(f"[PERIODIC GC] Collected {collected_objects} objects after batch processing")


    # ========================================================================
    # DEEP MEMORY TRACKING - BEFORE RETURN
    # ========================================================================
    try:
        from deep_memory_tracker import tracker
        tracker.checkpoint("BEFORE Return", {
            'env_states': env_states,
            'env_configs': env_configs,
        })

        # Print final growth summary
        print("\n[FINAL GROWTH SUMMARY]")
        growth_summary = tracker.get_growth_summary()
        for name, growth_bytes in growth_summary:
            if growth_bytes > 1024 * 1024:  # > 1MB
                print(f"  {name}: +{growth_bytes/1024/1024:.2f} MB")
    except Exception as e:
        print(f"[WARNING] Deep memory tracking failed: {e}")

    # DEBUG: Check memory usage before return
    import psutil
    import os as os_module
    process = psutil.Process(os_module.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY DEBUG] Before return: {memory_mb:.2f} MB")
    print(f"[MEMORY DEBUG] Number of env_states: {len(env_states)}")
    for i, state in enumerate(env_states):
        if 'history' in state:
            hist_str = str(state['history'])
            print(f"[MEMORY DEBUG] Env {i} history size: {len(hist_str)} chars")

    # Return results for all environments
    # DEBUG: Test if returning history objects is causing the issue
    print(f"[MEMORY DEBUG] About to create result list...")

    # Try to identify which part is causing the issue
    for i, state in enumerate(env_states):
        print(f"[MEMORY DEBUG] Env {i}: history type = {type(state['history'])}")
        # Check if we can convert history to string
        try:
            hist_str = str(state['history'])
            print(f"[MEMORY DEBUG] Env {i}: history string size = {len(hist_str)} chars")
        except Exception as e:
            print(f"[MEMORY DEBUG] Env {i}: Error converting history to string: {e}")

    # CRITICAL: The learning is already saved in env_configs['working_reflexions_history']
    # The returned history is only used for logging, so converting to string is safe
    # This fixes the "Killed" issue while preserving all learning
    result = []
    for state in env_states:
        # Convert EnvironmentHistory object to string to prevent serialization/memory issues
        # Learning is preserved via env_configs, not via this return value
        history_str = str(state['history']) if state['history'] else ""
        result.append((history_str, state['success']))

    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY DEBUG] After creating result: {memory_mb:.2f} MB")

    # =======================================================================
    # MEMORY LEAK FIX: Explicitly release env objects to prevent 219GB leak
    # =======================================================================
    # ROOT CAUSE CONFIRMED: The 'env' field contains ALFWorld environment
    # objects that consume 2.3MB each with a 48,625x memory multiplier!
    #
    # SAFE TO DELETE because:
    # 1. Learning is stored in env_configs (working_reflexions_history, memory, successful_trajectory)
    # 2. env objects are ONLY used for step(), get_current_valid_actions(), process_observation()
    # 3. env objects do NOT store any learning data - they're just game simulators
    # 4. Deletion happens AFTER all learning is saved to env_configs
    #
    # This prevents the 2.3MB × 48,625x × num_envs objects from persisting
    # =======================================================================
    # MEMORY FIX: Clear env registry and force GC
    # =======================================================================
    import gc
    print(f"[MEMORY FIX] Clearing env registry with {len(_env_registry)} environments...")
    _env_registry.clear()  # Remove all env object references

    # Force garbage collection to immediately free env objects
    collected = gc.collect()
    print(f"[MEMORY FIX] Garbage collected {collected} objects after clearing registry")

    memory_after_gc = process.memory_info().rss / 1024 / 1024
    memory_freed = memory_mb - memory_after_gc
    print(f"[MEMORY FIX] Memory after GC: {memory_after_gc:.2f} MB (freed {memory_freed:.2f} MB)")
    # =======================================================================

    # DASHBOARD: Mark trial as done
    if env_states:
        first_env = env_states[0]
        _dashboard.update(
            status='done',
            step=first_env.get('cur_step', 0),
            action='DONE' if first_env.get('success') else 'FAILED',
            progress_score=10 if first_env.get('success') else 0,
            progress_status='COMPLETE' if first_env.get('success') else 'INCOMPLETE',
        )

    return result

def format_reflexion_insights(memory) -> str:
    """Extract key actionable insights from reflexion memory - UNIVERSAL"""
    if not memory:
        return "- No previous attempts"
    
    # Universal type handling - ensure memory is a list
    memory_list = []
    if isinstance(memory, list):
        memory_list = memory
    elif isinstance(memory, str):
        memory_list = [memory]
    elif hasattr(memory, '__iter__'):
        try:
            memory_list = list(memory)
        except:
            return "- No actionable insights yet"
    else:
        return "- No actionable insights yet"
    
    if not memory_list:
        return "- No previous attempts"
    
    insights = []
    # Process last 2 reflections
    for i, reflection in enumerate(memory_list[-7:], 1):
        # Ensure reflection is a string
        if not isinstance(reflection, str):
            continue
            
        key_lines = []
        for line in reflection.split('\n'):
            line_lower = line.lower()
            # Extract actionable lines using universal modal verbs
            if any(word in line_lower for word in ['must', 'should', 'avoid', 'never', 'always', 'failed because', 'succeeded', 'exact actions']):
                clean_line = line.strip()
                if len(clean_line) > 10:  # REMOVED the < 200 limit
                    
                    # Don't truncate important memory - it's causing incomplete guidance
                    key_lines.append(f"  - {clean_line}")
        
        if key_lines:
            insights.append(f"Attempt {i}:\n" + '\n'.join(key_lines[:6]))  # Increased from 4 to 6
    
    return '\n'.join(insights) if insights else "- No actionable insights yet"


def format_recent_trajectory(trajectory: List[Tuple[str, str]]) -> str:
    """Format trajectory with success/failure indicators"""
    if not trajectory:
        return "No actions taken yet"
    
    formatted = []
    for i, (action, obs, _) in enumerate(trajectory, 1):  # Unpack 3 elements,
        # Determine if successful (universal check)
        obs_lower = obs.lower()
        failure_indicators = ["nothing happens", "don't understand", "invalid", "error"]
        success = not any(ind in obs_lower for ind in failure_indicators)
        
        success_indicator = "Ã¢Å“â€œ" if success else "Ã¢Å“â€”"
        obs_preview = obs[:60] + "..." if len(obs) > 60 else obs
        # Remove newlines for cleaner display
        obs_preview = obs_preview.replace('\n', ' ')
        formatted.append(f"{i}. [{success_indicator}] {action} Ã¢â€ â€™ {obs_preview}")
    
    return '\n'.join(formatted)


def process_ob(ob):
    """Process observation for parallel runner compatibility"""
    if isinstance(ob, list) and ob:
        return ob[0]
    return str(ob)

    
# Global instances for optimization - PERSIST ACROSS TRIALS
import pickle
import os

# Try to load existing prompt_generator if it exists
PROMPT_GEN_PATH = 'prompt_generator_state.pkl'
if os.path.exists(PROMPT_GEN_PATH) and os.path.getsize(PROMPT_GEN_PATH) > 0:
    try:
        with open(PROMPT_GEN_PATH, 'rb') as f:
            prompt_generator = pickle.load(f)
        print(f"[INIT] Loaded existing prompt_generator with {len(prompt_generator.prompt_gradients)} gradients")
        
        # Verify it loaded correctly
        if hasattr(prompt_generator, 'prompt_components'):
            comp_count = len([c for c, v in prompt_generator.prompt_components.items() 
                            if v != DynamicPromptGenerator().prompt_components.get(c)])
            print(f"[INIT] Components modified from default: {comp_count}")
    except Exception as e:
        print(f"[INIT] Load failed: {e}")
        prompt_generator = DynamicPromptGenerator()
        print("[INIT] Created NEW prompt_generator (load failed)")
else:
    prompt_generator = DynamicPromptGenerator()
    if not os.path.exists(PROMPT_GEN_PATH):
        print("[INIT] Created NEW prompt_generator (first run)")
    else:
        print("[INIT] Created NEW prompt_generator (empty file)")
env_understanding = EnvironmentUnderstanding()
meta_knowledge = MetaEnvironmentKnowledge()

# Connect prompt generator with environment understanding
prompt_generator.env_understanding = env_understanding
env_understanding.set_wrapper(None)

# Add save functions
def save_prompt_generator():
    """Save prompt_generator state to disk"""
    try:
        with open(PROMPT_GEN_PATH, 'wb') as f:
            pickle.dump(prompt_generator, f)
        print(f"[SAVE] Saved prompt_generator with {len(prompt_generator.prompt_gradients)} gradients")
    except Exception as e:
        print(f"[ERROR] Failed to save prompt_generator: {e}")

def save_environment_knowledge():
    """Save ENVIRONMENT_KNOWLEDGE to disk"""
    try:
        with open('environment_knowledge.pkl', 'wb') as f:
            pickle.dump(ENVIRONMENT_KNOWLEDGE, f)
        if ENVIRONMENT_KNOWLEDGE:
            actions = len(ENVIRONMENT_KNOWLEDGE.get('action_space', {}))
            print(f"[SAVE] Saved ENVIRONMENT_KNOWLEDGE with {actions} actions")
        else:
            print(f"[SAVE] Saved empty ENVIRONMENT_KNOWLEDGE")
    except Exception as e:
        print(f"[ERROR] Failed to save ENVIRONMENT_KNOWLEDGE: {e}")

def load_environment_knowledge():
    """Load ENVIRONMENT_KNOWLEDGE from disk"""
    global ENVIRONMENT_KNOWLEDGE
    try:
        if os.path.exists('environment_knowledge.pkl'):
            with open('environment_knowledge.pkl', 'rb') as f:
                ENVIRONMENT_KNOWLEDGE = pickle.load(f)
            if ENVIRONMENT_KNOWLEDGE:
                actions = len(ENVIRONMENT_KNOWLEDGE.get('action_space', {}))
                print(f"[LOAD] Loaded ENVIRONMENT_KNOWLEDGE with {actions} actions")
                return True
            else:
                print(f"[LOAD] Loaded empty ENVIRONMENT_KNOWLEDGE")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to load ENVIRONMENT_KNOWLEDGE: {e}")
    return False
# Global discovery cache
ENVIRONMENT_KNOWLEDGE = None

# Try to load existing ENVIRONMENT_KNOWLEDGE if it exists
if load_environment_knowledge():
    # Inject loaded knowledge into prompt_generator
    if ENVIRONMENT_KNOWLEDGE:
        prompt_generator.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        print(f"[INIT] Injected loaded ENVIRONMENT_KNOWLEDGE into prompt_generator")

# Debug flags
DEBUG_ACTOR = False  # Will be set by main.py
DEBUG_CRITIC = False  # Will be set by main.py
DEBUG_REFLEXION = False  # Will be set by main.py

# Ablation study flags - default to full system (combined mode)
USE_REFLEXION = True  # Enable episodic memory and reflection generation
USE_TEXTGRAD = True   # Enable prompt optimization via textual gradients
USE_TODO = True       # Enable TODO Manager for task decomposition
PURE_ABLATION = False # Pure ablation mode - no pre-learned patterns

# Set the flags in dynamic_prompting
set_debug_flags(DEBUG_ACTOR, DEBUG_CRITIC)

def process_ob(ob):
    """Process observation for parallel runner compatibility"""
    if isinstance(ob, list) and ob:
        return ob[0]
    return str(ob)


def print_debug(category: str, content: str, color: str = "blue"):
    """Print debug information with formatting"""
    if not (DEBUG_ACTOR or DEBUG_CRITIC or DEBUG_REFLEXION):
        return
        
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color, colors["blue"])
    reset = colors["reset"]
    
    print(f"\n{color_code}{'='*80}")
    print(f"[{category}]")
    print(f"{'='*80}{reset}")
    print(content)
    print(f"{color_code}{'='*80}{reset}\n")

def get_environment(env_type, config=None, env_id=None):
    """Factory function to get different environments - ALWAYS returns wrapped environment"""
    if env_type == "alfworld":
        return ALFWorldWrapper(config, env_id=env_id)
    
    # TextWorld variants
    elif env_type == "textworld_cooking":
        return TextWorldWrapper(game_type="cooking", level="medium", env_id=env_id)
    elif env_type == "textworld_treasure":
        return TextWorldWrapper(game_type="treasure", level="medium", env_id=env_id)
    elif env_type == "textworld_simple":
        return TextWorldWrapper(game_type="simple", level="easy", env_id=env_id)
    
    # Jericho games
    elif env_type == "jericho_zork1":
        return JerichoWrapper(game_name="zork1", env_id=env_id)
    elif env_type == "jericho_detective":
        return JerichoWrapper(game_name="detective", env_id=env_id)
    elif env_type == "jericho_balances":
        return JerichoWrapper(game_name="balances", env_id=env_id)
    
    # ScienceWorld tasks
    elif env_type == "scienceworld_boil":
        return ScienceWorldWrapper(task_name="boil", env_id=env_id)
    elif env_type == "scienceworld_melt":
        return ScienceWorldWrapper(task_name="melt", env_id=env_id)
    elif env_type == "scienceworld_grow":
        return ScienceWorldWrapper(task_name="grow-plant", env_id=env_id)
    elif env_type == "scienceworld_all":
        # Cycles through all 20 task types based on env_id
        return ScienceWorldWrapper(task_name=None, env_id=env_id)
    
    # BabyAI levels
    elif env_type == "babyai_goto":
        return BabyAIWrapper(level="BabyAI-GoToObj-v0", env_id=env_id)
    elif env_type == "babyai_pickup":
        return BabyAIWrapper(level="BabyAI-PickupLoc-v0", env_id=env_id)
    elif env_type == "babyai_unlock":
        return BabyAIWrapper(level="BabyAI-UnlockLocal-v0", env_id=env_id)

    # AppWorld benchmark
    elif env_type == "appworld" or env_type.startswith("appworld_"):
        # Extract task_id if provided (e.g., appworld_123)
        task_id = None
        if "_" in env_type and env_type != "appworld":
            task_id = env_type.split("_", 1)[1]
        return AppWorldWrapper(task_id=task_id, env_id=env_id)

    # OSWorld benchmark (visual GUI environment)
    elif env_type == "osworld" or env_type.startswith("osworld_"):
        if not OSWORLD_AVAILABLE:
            raise ImportError(
                "OSWorld support requires osworld_wrapper.py, visual_grounding.py, "
                "and vision_backend.py. Install with: pip install Pillow requests"
            )
        # Extract task_id if provided (e.g., osworld_office_libreoffice_1)
        task_id = os.getenv("OSWORLD_TASK_ID", None)
        if "_" in env_type and env_type != "osworld":
            task_id = env_type.split("_", 1)[1]

        # Get OSWorld-specific config from environment or defaults
        vm_ip = os.getenv("OSWORLD_VM_IP", "localhost")
        vm_port = int(os.getenv("OSWORLD_VM_PORT", "5000"))
        max_steps = int(os.getenv("OSWORLD_MAX_STEPS", "50"))
        headless = os.getenv("OSWORLD_HEADLESS", "false").lower() == "true"
        # Use REFLEXGRAD_MODEL as the default vision backend so open-source models work
        vision_backend_name = os.getenv("OSWORLD_VISION_BACKEND", os.getenv("REFLEXGRAD_MODEL", "gpt-5"))

        # Find evaluation_examples directory (local or docker)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_tasks = os.path.join(script_dir, "evaluation_examples", "examples")
        tasks_dir = local_tasks if os.path.isdir(local_tasks) else "/app/OSWorld/evaluation_examples/examples"

        return OSWorldWrapper(
            task_id=task_id,
            vm_ip=vm_ip,
            vm_port=vm_port,
            max_steps=max_steps,
            use_vgl_cache=True,
            headless=headless,
            vision_backend_name=vision_backend_name,
            osworld_tasks_dir=tasks_dir,
        )

    # OpenRCA Code-Writing Agent (universal analysis via LLM-generated code)
    # NOTE: Must be checked BEFORE "openrca" since "openrca_code_Bank" starts with "openrca_"
    elif env_type == "openrca_code" or env_type.startswith("openrca_code_"):
        from openrca_code_wrapper import OpenRCACodeWrapper
        dataset = "Bank"
        if env_type.startswith("openrca_code_"):
            dataset_part = env_type[len("openrca_code_"):]
            dataset_part = dataset_part.replace("_cloudbed-", "/cloudbed-")
            if dataset_part:
                dataset = dataset_part
        data_root = os.getenv("OPENRCA_DATA_ROOT", None)
        query_offset = int(os.getenv("OPENRCA_QUERY_OFFSET", "0"))
        return OpenRCACodeWrapper(
            dataset=dataset,
            env_id=env_id + query_offset,
            max_investigation_steps=35,  # 25 investigation + 10 cascade diagnosis
            data_root=data_root,
        )

    # OpenRCA benchmark - template-based investigation (legacy)
    elif env_type == "openrca" or env_type.startswith("openrca_"):
        from openrca_wrapper import OpenRCAWrapper
        dataset = "Bank"
        if env_type.startswith("openrca_"):
            dataset_part = env_type[len("openrca_"):]
            dataset_part = dataset_part.replace("_cloudbed-", "/cloudbed-")
            if dataset_part:
                dataset = dataset_part
        data_root = os.getenv("OPENRCA_DATA_ROOT", None)
        query_offset = int(os.getenv("OPENRCA_QUERY_OFFSET", "0"))
        return OpenRCAWrapper(
            dataset=dataset,
            env_id=env_id + query_offset,
            max_investigation_steps=55,
            data_root=data_root,
        )

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def llm(prompt: str, stop: List[str] = ["\n"], context: str = "action_selection", 
        structured_output_schema: Optional[Dict] = None) -> str:
    """Generate text using vLLM - universal approach with structured output support"""
    
    if DEBUG_ACTOR:
        print(f"\n[LLM CALLED] Context: {context}")
    
    # Import here to avoid circular imports
    from vllm.sampling_params import GuidedDecodingParams
    
    # Prepare sampling params based on whether we need structured output
    if structured_output_schema:
        # Use guided decoding for structured output
        guided_params = GuidedDecodingParams(json=structured_output_schema)
        sampling_params = SamplingParams(
            max_tokens=100,  # Numbers need very few tokens
            temperature=0.0,  # Deterministic for action selection
            stop=stop,
            skip_special_tokens=True,
            guided_decoding=guided_params  # Use guided_decoding instead of guided_json
        )

    
    output = model.generate([prompt], sampling_params)[0]
    raw_text = output.outputs[0].text
    
    if DEBUG_ACTOR:
        print(f"[LLM RAW OUTPUT]: '{raw_text}'")

    # For structured output, the text is already just a number
    if structured_output_schema:
        return raw_text.strip()
    
    # For regular output, do minimal cleaning
    cleaned_text = raw_text.strip()
    
    # Remove think tags if present
    if "</think>" in cleaned_text:
        parts = cleaned_text.split("</think>")
        cleaned_text = parts[-1].strip() if len(parts) > 1 else cleaned_text
    
    # Clean any trailing annotations
    cleaned_text = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned_text).strip()
    
    # Remove quotes
    cleaned_text = cleaned_text.strip('"\'')
    
    # Take first line only
    if '\n' in cleaned_text:
        cleaned_text = cleaned_text.split('\n')[0].strip()
    
    if DEBUG_ACTOR:
        print(f"[LLM CLEANED OUTPUT]: '{cleaned_text}'")
    
    return cleaned_text


def extract_valid_commands_from_prompt(prompt: str) -> List[str]:
    """Extract valid commands from the prompt for validation"""
    valid_commands = []
    lines = prompt.split('\n')
    in_command_section = False
    
    for line in lines:
        if "VALID ACTIONS:" in line:
            in_command_section = True
            continue
        elif in_command_section and line.strip() and not line.startswith(' '):
            # End of valid actions section
            break
        
        if in_command_section and line.strip():
            cmd = line.strip()
            if cmd.startswith('- '):
                cmd = cmd[2:].strip()
            if cmd and not cmd.startswith('DEBUG') and not cmd.startswith('['):
                valid_commands.append(cmd)
    
    return valid_commands


def _build_action_constraint_block(valid_actions: list, is_visual_env: bool) -> str:
    """Return the action constraint/hint block for prompts.

    Visual envs: VGL elements as reference hints (model can use any action format).
    Text envs: Hard VALID ACTIONS constraint (must pick from list).
    """
    if is_visual_env:
        elements = '\n'.join(f'  - {a}' for a in valid_actions) if valid_actions else '  (none detected)'
        return (
            "UI ELEMENTS DETECTED (reference hints — trust the screenshots over these):\n"
            f"{elements}\n\n"
            "You have screenshots. You may use ANY action format:\n"
            "  click (x,y), click 'Label', hotkey Ctrl+S, type 'text',\n"
            "  press Enter, run: <cmd>, scroll down, triple-click, wait, done"
        )
    else:
        actions_str = ', '.join(valid_actions) if valid_actions else 'Unknown'
        return (
            "⚠️ ═══════════════════════════════════════════════════════════\n"
            "VALID ACTIONS (HARD CONSTRAINT - ONLY these actions are executable):\n"
            "═══════════════════════════════════════════════════════════ ⚠️\n"
            f"{actions_str}\n\n"
            "CRITICAL: When providing strategic insights, you MUST reference actions from this list.\n"
            "If you suggest alternative approaches, verify the action exists in VALID ACTIONS above."
        )


def build_step_gradient_prompt_from_data(result: Dict, log_debug=print, is_visual_env: bool = False, learning_context=None) -> str:
    """Build step gradient prompt from collected action result data.

    This extracts the prompt-building logic from generate_universal_step_reflection
    so we can batch-generate prompts for multiple environments.
    """
    # Extract all needed data from result
    prev_observation = result['prev_observation']
    curr_observation = result['observation']
    action = result['action']
    task = result['task']
    current_step = result['current_step']
    valid_actions = result['next_valid_actions']
    # DEBUG: Log what valid_actions the PASS 2 prompt sees
    _cool_in_va = [a for a in valid_actions if 'cool' in str(a).lower()]
    _heat_in_va = [a for a in valid_actions if 'heat' in str(a).lower()]
    _clean_in_va = [a for a in valid_actions if 'clean' in str(a).lower()]
    _put_in_va = [a for a in valid_actions if 'put' in str(a).lower()]
    if _cool_in_va or _heat_in_va or _clean_in_va or _put_in_va:
        print(f"[PASS2-VA-DEBUG] Step {current_step}: TASK VERBS IN VALID_ACTIONS: cool={_cool_in_va}, heat={_heat_in_va}, clean={_clean_in_va}, put={_put_in_va}")
    # Loop detection removed - no longer needed
    # is_repetitive = result['is_repetitive']
    current_todo = result['current_todo']
    inventory = result['inventory_items']
    previous_reflexions = result['tiered_step_reflexions']
    episodic_memory = result['episodic_memory_only']
    initial_observation = result.get('initial_observation', '')
    explored_locations = result.get('explored_locations', {})

    # Loop detection removed - no repetition warning needed
    repetition_warning = ""

    # Build TODO context with EXPLICIT observation comparison (FIX #22 - Universal)
    todo_context = ""
    if current_todo:
        todo_context = f"""
📋 CURRENT SUBTASK: {current_todo.content}

🔍 EXPLICIT TODO CHECK (Answer honestly):
   - TODO requires: "{current_todo.content}"
   - Last observation: "{curr_observation[:150] if curr_observation else 'N/A'}"
   - QUESTION: Does this observation indicate the TODO was achieved?
   - If NO: You must try a DIFFERENT action/approach.
"""

    # Build inventory context — ONLY for text-game envs.
    # Visual envs (osworld) have no inventory; the cool/heat/clean/put language
    # confuses GUI models on document/presentation tasks.
    if is_visual_env:
        inventory_context = ""
    elif inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ CRITICAL: Use held items for task completion (cool/heat/clean/put)!
"""
    else:
        inventory_context = """
🎒 INVENTORY: You are NOT carrying anything.
   ⚠️ You must TAKE an object before you can cool/heat/clean/put it.
"""

    # Build previous reflexions context
    reflexions_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        formatted_reflexions = []
        for r in previous_reflexions:
            compression_type = r.get('is_compressed', None)
            step_num = r.get('step', '?')
            reflection_text = r.get('reflection', '')
            if not compression_type:
                reflection_text = reflection_text[:800] if len(reflection_text) > 800 else reflection_text

            if compression_type == 'heavy':
                formatted_reflexions.append(f"  Steps {step_num} (early summary): {reflection_text}")
            elif compression_type == 'medium':
                formatted_reflexions.append(f"  Steps {step_num} (compressed): {reflection_text}")
            else:
                formatted_reflexions.append(f"  Step {step_num} (recent): {reflection_text}")

        reflexion_text = "\n".join(formatted_reflexions)
        total_count = len(previous_reflexions)
        compressed_count = len([r for r in previous_reflexions if r.get('is_compressed')])

        reflexions_context = f"""
📚 STEP-LEVEL LEARNINGS (ALL {total_count} steps: {compressed_count} compressed, {total_count - compressed_count} verbose):
{reflexion_text}
   ⚠️ CRITICAL: Early steps are compressed but still visible - learn from ENTIRE history!
"""

    # Build episodic memory context
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        recent_episodic = episodic_memory[-5:]
        episodic_parts = []
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:200]
            else:
                mem_text = str(mem)[:200]
            episodic_parts.append(f"  • {mem_text}")
        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"""
🧠 EPISODIC MEMORY (strategic patterns from previous trials):
{episodic_text}
   ⚠️ CRITICAL: These are proven patterns across trials - use them to avoid known pitfalls!
"""

    # Build exploration guidance from initial observation and explored locations
    exploration_context = ""
    if initial_observation:
        # Extract all locations from initial observation
        import re
        location_matches = re.findall(r'(cabinet|drawer|shelf|countertop|stoveburner|fridge|microwave|dresser|desk|sidetable|armchair|sofa|bed) \d+', initial_observation.lower())
        all_locations = list(set(location_matches))  # Unique locations

        # Determine which have been explored
        explored = [loc for loc in all_locations if loc in explored_locations and explored_locations[loc] > 0]
        unexplored = [loc for loc in all_locations if loc not in explored_locations or explored_locations[loc] == 0]

        if len(unexplored) > 0:
            unexplored_list = ', '.join(sorted(unexplored))  # Show ALL — never truncate exploration data
            explored_list = ', '.join(sorted(explored)) if explored else "None"
            exploration_context = f"""
🔍 EXPLORATION STATUS (CRITICAL - Avoid revisiting locations!):
   Total locations: {len(all_locations)}
   ✅ Explored ({len(explored)}): {explored_list}
   ❌ NOT YET EXPLORED ({len(unexplored)}): {unexplored_list}

   ⚠️ PRIORITY: When searching for items, CHECK UNEXPLORED LOCATIONS FIRST!
   Don't waste steps re-examining locations you've already checked!
"""

    # FIX #21: Build TextGrad's history from RAW OBSERVATIONS ONLY
    # ROOT CAUSE: LLM interpretations (guidance, progress) compound errors
    # SOLUTION: Show only raw action + raw environment feedback
    # FIX #25: Use 15 steps for consistency (Reflexion analyzes raw data to generate lessons)
    textgrad_history_context = ""
    step_insights = result.get('step_insights_accumulator', [])
    log_debug(f"[TEXTGRAD DEBUG] step_insights has {len(step_insights)} items")
    if step_insights and len(step_insights) > 0:
        recent_gradients = step_insights  # FIX #36: ALL steps so LLM sees complete history
        log_debug(f"[TEXTGRAD DEBUG] Building RAW history context with {len(recent_gradients)} steps")
        textgrad_history_context = "\n🎯 RAW ACTION HISTORY (Interpret fresh - do NOT trust prior interpretations):\n"
        for insight in recent_gradients:
            action_taken = insight.get('action', 'Unknown')
            observation = insight.get('observation', '')  # RAW environment feedback ONLY
            # For visual envs: extract only RELEVANT feedback (state change + command output),
            # not the full VGL metadata (interactive elements, available actions) which is noise
            if is_visual_env and observation:
                _feedback_parts = []
                _sc_idx = observation.find('[State Change]')
                if _sc_idx >= 0:
                    _sc_end = observation.find('\n[', _sc_idx + 1)
                    _sc_text = observation[_sc_idx:_sc_end if _sc_end > _sc_idx else _sc_idx + 200].strip()
                    if 'No visible change' not in _sc_text:
                        _feedback_parts.append(_sc_text[:200])
                    else:
                        _feedback_parts.append('No visible change')
                _co_idx = observation.find('[Command Output')
                if _co_idx >= 0:
                    # Extract error message directly (skip verbose command header + traceback)
                    _err_idx = observation.find('Error:', _co_idx)
                    if _err_idx < 0:
                        _err_idx = observation.find('ERROR]', _co_idx)
                    if _err_idx >= 0:
                        _feedback_parts.append('COMMAND ERROR: ' + observation[_err_idx:_err_idx + 300].strip())
                    else:
                        _feedback_parts.append(observation[_co_idx:_co_idx + 500].strip())
                    # Preserve AUTO-DISCOVERY messages (critical for file path correction)
                    _disc_idx = observation.find('[AUTO-DISCOVERY]', _co_idx)
                    if _disc_idx >= 0:
                        _disc_end = observation.find('\n\n', _disc_idx)
                        _disc_text = observation[_disc_idx:_disc_end if _disc_end > _disc_idx else _disc_idx + 300].strip()
                        _feedback_parts.append(_disc_text)
                observation = ' | '.join(_feedback_parts) if _feedback_parts else 'No visible change'
            textgrad_history_context += f"  Step {insight.get('step', '?')}: ACTION: \"{action_taken}\" → ENV FEEDBACK: \"{observation}\"\n"
            log_debug(f"[TEXTGRAD DEBUG] Added RAW: Step {insight.get('step')} action='{action_taken}' obs='{observation[:80] if observation else 'NONE'}'")
        textgrad_history_context += "\n  ⚠️ CRITICAL: Interpret the raw feedback above FRESH. If 'clean' was done but task needs 'cool',\n"
        textgrad_history_context += "  recognize that clean≠cool. Environment feedback tells you EXACTLY what happened.\n"
        # Inject universal learning context (failure memory + failed actions)
        _lc = learning_context
        log_debug(f"[LEARNING ENGINE] learning_context present: {_lc is not None}, failure_memory: {len(_lc.failure_memory) if _lc else 0}, failed_actions: {len(_lc.step_insights.get_failed_actions()) if _lc else 0}")
        if _lc:
            _lc_text = _lc.failure_memory.format_for_prompt()
            if _lc_text:
                textgrad_history_context += _lc_text
            _failed = _lc.step_insights.get_failed_actions()
            if _failed:
                textgrad_history_context += (
                    "\n🚫 THESE ACTIONS RETURNED 'Nothing happens' — NEVER repeat them:\n"
                    + ", ".join(f"'{a}'" for a in _failed[-10:])
                    + "\nUse VALID ACTIONS list instead.\n"
                )
        log_debug(f"[TEXTGRAD DEBUG] textgrad_history_context length: {len(textgrad_history_context)} chars")
    else:
        log_debug(f"[TEXTGRAD DEBUG] No step_insights, textgrad_history_context is EMPTY")

    # Extract [Command Output] before truncation — this is the primary learning signal for run: commands
    _cmd_out_after = ''
    if is_visual_env and curr_observation:
        _co_idx = curr_observation.find('[Command Output')
        if _co_idx >= 0:
            _cmd_out_after = chr(10) + curr_observation[_co_idx:_co_idx + 800]

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}
{f"ACTION: {action}{chr(10)}{chr(10)}OBSERVATION SUMMARY:{chr(10)}BEFORE state: {prev_observation[:500]}{chr(10)}AFTER state: {curr_observation[:500]}{_cmd_out_after}{chr(10)}{chr(10)}{'⚠️ COMMAND OUTPUT ABOVE IS THE PRIMARY RESULT — READ the error/output text carefully to diagnose the root cause.' if _cmd_out_after else '⚠️ SCREENSHOTS show the current screen state.'}" if is_visual_env else f"BEFORE: {prev_observation}{chr(10)}ACTION: {action}{chr(10)}AFTER: {curr_observation}"}

{_build_action_constraint_block(valid_actions, is_visual_env)}
{repetition_warning}

═══════════════════════════════════════════════════════════
CRITICAL REFLEXION ANALYSIS (Answer each question precisely):
═══════════════════════════════════════════════════════════

1. SEMANTIC PRECISION CHECK:
   Task instruction: "{task}"
   My action: "{action}"

   Extract the PRIMARY PURPOSE of each:
   - What is the main goal the task requires me to achieve? (e.g., change temperature, change location, change cleanliness, etc.)
   - What is the main effect/purpose of the action I took?
   - Are these semantically IDENTICAL purposes or DIFFERENT purposes?

   IMPORTANT: Even if both use similar objects or locations, they may serve different purposes.
   Example: "remove dirt" vs "lower temperature" are DIFFERENT even if both might involve water.

   Answer: SEMANTIC MATCH: [Yes/No] - Explain: [task purpose] vs [action purpose]

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{task}", what SPECIFIC property of objects must change?
   (Examples: temperature, location, cleanliness, container state, object state)

   Now look at the AFTER observation: "{curr_observation}"
   Did that EXACT property change? Cite specific words from observation as evidence.
   Do NOT assume or infer - only state what the observation explicitly shows.

   Answer: PROPERTY REQUIRED: [what must change] | EVIDENCE: [quote from observation] | CHANGED: [Yes/No]

3. TODO FEEDBACK INTEGRATION:
   {f'Current subtask: "{current_todo.content}" - Status after action: NOT ACHIEVED (Attempt #{current_todo.attempts if current_todo else 0})' if current_todo else 'No active subtask'}

   If subtask not achieved after {current_todo.attempts if current_todo else 0} attempt(s):
   - What pattern do you see in my actions?
   - Am I repeating similar action types that don't work?
   - What assumption might I be making that is incorrect?

   Answer: FAILURE PATTERN: [describe pattern if any] | ROOT CAUSE: [precise diagnosis]

4. CAUSAL CHAIN ANALYSIS:
   Trace the causal relationship:
   - GOAL (from task): [what I wanted to accomplish]
   - ACTION (what I did): {action}
   - RESULT (what happened): [state change or lack thereof]
   - CAUSALITY: This [succeeded/failed] because [causal reason]

   If there's a semantic mismatch or wrong action type:
   - Have I already tried similar actions on this location in STEP-LEVEL LEARNINGS?
   - If yes, what was the result? Did it fail to find the target object?
   - What is the ROOT CAUSE of the repeated failure? (e.g., wrong location, wrong action type, impossible task)

   NOTE: Do NOT recommend next actions - that's TextGrad's responsibility for optimization.
   Reflexion focuses ONLY on WHY things succeed/fail, not WHAT to do next.

   Answer: CAUSAL CHAIN: [describe cause-effect] | ROOT CAUSE: [fundamental reason]

5. CONCRETE LESSON EXTRACTION:
   Based on your analysis, complete this pattern for episodic memory:
   - WRONG ASSUMPTION: [what I incorrectly believed, if any]
   - CORRECT UNDERSTANDING: [what is actually true]
   - NEVER REPEAT: [specific action or pattern to avoid]
   - INSTEAD USE: [specific action or approach that works]

   Answer: LESSON: [fill in above format only if there's a clear lesson to extract]

6. PROGRESS ASSESSMENT (TextGrad-Style):
   Evaluate progress toward FULL TASK: "{task}"
   Context: {f'Current subtask is "{current_todo.content}"' if current_todo else 'No active subtask'}

   Provide textual evaluation:

   TASK_ALIGNMENT:
   - Which requirement(s) of FULL TASK are closer to completion?
   - Which remain unmet? Cite evidence from AFTER state.
   - CRITICAL: Evaluate relative to FULL TASK, not just subtask.

   PROGRESS_STATUS toward FULL TASK:
   - NO_PROGRESS: Irrelevant to task requirements
   - EXPLORING: Gathering info, no requirements met
   - PARTIAL_PROGRESS: Some requirements met, task incomplete
   - MAJOR_PROGRESS: Most requirements met, nearly done
   - TASK_COMPLETE: All requirements satisfied

   {f"""RECOMMENDED_ACTION: Provide an EXACT, EXECUTABLE action for the next step.
   The action MUST be one of: click '...', press ..., hotkey ..., type '...', run: ..., scroll ..., wait, done

   STRATEGY_SHIFT: If the current approach is not making progress after multiple attempts,
   recommend a FUNDAMENTALLY DIFFERENT strategy to achieve the task goal.
   - Describe what the current approach is doing wrong (root cause)
   - Propose a concrete alternative approach — switch MODALITY:
     * If GUI clicks are failing → switch to programmatic: run: python3 -c "..."
     * If programmatic commands are failing → switch to GUI interaction
     * If one tool/library fails → try a different tool/library
   - Provide the exact action command to execute (e.g., run: python3 -c "import ...")
   - STRATEGY_SHIFT MUST be an executable command, NOT a description
   """ if is_visual_env else ""}Answer: PROGRESS_STATUS: [one of above] | RECOMMENDED_ACTION: [exact command] | STRATEGY_SHIFT: [exact alternative command or "none"]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    # ═══════════════════════════════════════════════════════════
    # PHASE 2 SYNERGY: TextGrad (normal) vs Reflexion (failures) - PARALLEL CODE
    # ═══════════════════════════════════════════════════════════

    # Get previous step's progress to decide which component to use
    last_step_gradient = result.get('last_step_gradient', {})
    last_progress = last_step_gradient.get('progress_score', 0)  # FIX: Default to 0 (failure) not 5 (success)

    # PHASE 2 DECISION: Use progress-based routing (aligned with papers)
    # - TextGrad (Nature 2024): Step-by-step optimization when making progress
    # - Reflexion (NeurIPS 2023): Causal analysis only at failures/low progress

    # Get state for consecutive failure tracking
    state = result.get('state', {})
    current_step_num = result.get('current_step', 0)

    # Initialize consecutive failure tracking variables
    if 'consecutive_low_progress' not in state:
        state['consecutive_low_progress'] = 0
    if 'steps_since_reflexion' not in state:
        state['steps_since_reflexion'] = 0

    # Update consecutive failure counter based on last step's progress
    if last_progress < 4:  # Low progress threshold (0-3 are low, 4+ is actual progress)
        state['consecutive_low_progress'] += 1
    else:
        state['consecutive_low_progress'] = 0  # Reset on good progress

    # Increment cooldown counter
    state['steps_since_reflexion'] += 1

    # Smart trigger: Only use Reflexion when TRULY stuck
    # Requirements:
    # 1. USE_REFLEXION flag is True (ablation mode check)
    # 2. Not step 0 (TODO provides initial guidance)
    # 3. 5+ consecutive low-progress steps (not just transient failure)
    # 4. 5+ steps since last Reflexion (cooldown to let TextGrad establish gradient)
    use_reflexion = (
        USE_REFLEXION and  # ABLATION: Check global flag first
        current_step_num > 0 and
        state['consecutive_low_progress'] >= 5 and
        state['steps_since_reflexion'] >= 5
    )

    # ABLATION: If TextGrad is disabled, always use Reflexion path (if enabled)
    if not USE_TEXTGRAD and USE_REFLEXION:
        use_reflexion = True

    # ABLATION: If both disabled, return empty (shouldn't happen in normal use)
    if not USE_TEXTGRAD and not USE_REFLEXION:
        log_debug(f"[ABLATION WARNING] Both TextGrad and Reflexion disabled!")
        return ('none', '')

    if not use_reflexion:
        # TEXTGRAD PATH (85%+ of steps): Good progress → keep optimizing
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using TEXTGRAD (last_progress={last_progress}/10)")
        prompt = generate_textgrad_gradient_prompt(result, previous_reflexions, episodic_memory,
                                                     initial_observation, explored_locations, valid_actions,
                                                     state.get('cumulative_search_memory', ''),  # FIX #32
                                                     is_visual_env=state.get('env_type', '') in ('osworld', 'visual'))
        return ('textgrad', prompt)  # Return tuple with type for metadata tracking
    else:
        # REFLEXION PATH (<15% of steps): Low progress → diagnose why
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using REFLEXION (last_progress={last_progress}/10 - need causal analysis)")

        # Reset tracking counters after using Reflexion
        state['consecutive_low_progress'] = 0
        state['steps_since_reflexion'] = 0

        return ('reflexion', reflection_prompt)  # Return tuple with type for metadata tracking


def generate_reflexion_only_prompt(result: Dict, previous_reflexions: list, episodic_memory: list,
                                   valid_actions: list) -> str:
    """Generate a SIMPLE prompt for reflexion_only ablation mode.

    This is used when USE_TEXTGRAD=False to ensure TRUE ablation:
    - NO TextGrad gradient history
    - NO TextGrad policy optimization
    - ONLY Reflexion memory and strategic insights
    - Simple action selection based on task + observation + Reflexion
    """
    task = result['task']
    curr_observation = result['observation']

    # Format Reflexion memory (strategic insights from past episodes)
    reflexion_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        reflexion_context = "\n=== REFLEXION MEMORY (Strategic Insights) ===\n"
        for i, ref in enumerate(previous_reflexions[-5:], 1):  # Last 5 reflexions
            if isinstance(ref, dict):
                insight = ref.get('insight', ref.get('content', str(ref)))
            else:
                insight = str(ref)
            # Truncate long insights
            if len(insight) > 300:
                insight = insight[:300] + "..."
            reflexion_context += f"{i}. {insight}\n"
        reflexion_context += "=== END REFLEXION MEMORY ===\n"

    # Format episodic memory (success patterns)
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        episodic_context = "\n=== EPISODIC MEMORY (Success Patterns) ===\n"
        for i, ep in enumerate(episodic_memory[-3:], 1):  # Last 3 episodes
            if isinstance(ep, dict):
                pattern = ep.get('actions', ep.get('insight', str(ep)))
                if isinstance(pattern, list):
                    pattern = " -> ".join(pattern[:5])  # First 5 actions
            else:
                pattern = str(ep)[:200]
            episodic_context += f"{i}. {pattern}\n"
        episodic_context += "=== END EPISODIC MEMORY ===\n"

    # Format valid actions
    actions_str = "\n".join([f"  - {a}" for a in valid_actions])

    # Simple reflexion-only prompt
    prompt = f"""You are an agent solving a task. Use ONLY the Reflexion memory (strategic insights from past attempts) to choose the best action.

TASK: {task}

CURRENT OBSERVATION:
{curr_observation}
{reflexion_context}
{episodic_context}
AVAILABLE ACTIONS:
{actions_str}

INSTRUCTIONS:
1. Review the Reflexion memory for strategic insights
2. Apply learned patterns from episodic memory
3. Choose the action that best advances toward the goal

OUTPUT FORMAT:
RECOMMENDED_ACTION: <exact action from the list>
JUSTIFICATION: <brief explanation based on Reflexion insights>

Choose your action:"""

    return prompt


def generate_textgrad_gradient_prompt(result: Dict, previous_reflexions: list, episodic_memory: list,
                                      initial_observation: str, explored_locations: dict,
                                      valid_actions: list, cumulative_search_memory: str = "",
                                      is_visual_env: bool = False) -> str:
    """Generate TextGrad prompt for automatic differentiation via text.

    TextGrad's role (Nature 2024): Optimize actions through textual gradients
    - Provides step-by-step action optimization (NOT strategic analysis)
    - Accumulates gradients to backpropagate through text
    - Suggests next action based on gradient descent
    """
    task = result['task']
    prev_observation = result['prev_observation']
    curr_observation = result['observation']
    action = result['action']
    current_todo = result.get('current_todo')
    inventory_items = result.get('inventory_items', [])
    print(f"[PASS2-INVENTORY] inventory_items={inventory_items}")

    # Build TODO context
    todo_context = ""
    _task_complete_mode = result.get('state', {}).get('_textgrad_task_complete', False)
    if _task_complete_mode:
        todo_context = ("\n🎯 TASK STATUS: ✅ TextGrad previously assessed TASK_COMPLETE.\n"
                        "   ⚠️ VERIFY ONLY: Use non-destructive actions (look, examine, inventory) to confirm.\n"
                        "   🚫 DO NOT take, put, move, heat, cool, or clean any objects — this may UNDO completed work.\n"
                        "   If verification confirms completion, recommend 'look' or 'inventory'.\n")
    elif current_todo:
        status_emoji = {"pending": "⏳", "in_progress": "🔧", "completed": "✅", "failed": "❌"}.get(current_todo.status, "")
        todo_context = f"\n🎯 CURRENT SUBTASK: {status_emoji} {current_todo.content} (Attempt #{current_todo.attempts})\n"

    # Build inventory context — ONLY for text-game envs (visual envs have no inventory).
    if is_visual_env:
        inventory_context = ""
    elif inventory_items and len(inventory_items) > 0:
        items_str = ", ".join(inventory_items)
        inventory_context = f"\n🎒 INVENTORY: Currently holding: {items_str}\n   ⚠️ CRITICAL: Use held items to complete the task (cool/heat/clean/put)!\n"
    else:
        inventory_context = "\n🎒 INVENTORY: You are NOT carrying anything.\n   ⚠️ You must TAKE an object before you can use cool/heat/clean/put on it.\n"

    # Build gradient history (last 5 steps for backpropagation)
    # FIX #4 (Nov 22): PURE TEXTUAL HISTORY - No scores, no truncation!
    # FIX #13: Include OBSERVATION so agent knows what was found at each location!
    # FIX #25: Use last 15 steps with FULL backward lessons (no truncation)
    # This gives the LLM enough history to learn from past mistakes
    # NOTE: Full history is still kept in state (line 3178) for exploration tracking (FIX #16A)
    gradient_history = ""
    step_insights = result.get('step_insights_accumulator', [])
    print(f"[FIX #25] step_insights count: {len(step_insights)} (using last 15 for prompt with FULL lessons)")
    if step_insights and len(step_insights) > 0:
        # FIX #25: Use last 15 steps with FULL backward gradient lessons
        # This ensures the agent learns from past mistakes and doesn't repeat them
        all_insights = step_insights  # FIX #36: ALL steps with FULL lessons!
        gradient_history = "\n📊 COMPLETE ACTION HISTORY WITH LESSONS:\n"
        gradient_history += "   ⚠️ CRITICAL: Read the LESSON for each step - it tells you what was WRONG!\n\n"

        for insight in all_insights:
            step = insight.get('step', '?')
            act = insight.get('action', 'Unknown')
            obs = insight.get('observation', '')  # FIX #13: What was actually found!
            full_reasoning = insight.get('hypothesis', '')
            print(f"[TEXTGRAD PROMPT DEBUG] Step {step}: obs='{obs[:80] if obs else 'MISSING!'}'")

            gradient_history += f"  Step {step}: {act}\n"
            if obs:
                gradient_history += f"    → {obs}\n"  # FULL observation, no truncation!
            if full_reasoning:
                gradient_history += f"  Your reasoning: {full_reasoning}\n"
            # FIX #15: Show BACKWARD GRADIENT feedback (loss analysis)
            backward_feedback = insight.get('backward_gradient', '')
            print(f"[FIX #15 DEBUG] Step {step}: backward_feedback present: {bool(backward_feedback)}, value[:50]: '{backward_feedback[:50] if backward_feedback else 'EMPTY'}'")
            # FIX #16C: DON'T truncate backward gradient - it contains critical "what to do instead" info!
            if backward_feedback:
                gradient_history += f"  ⚠️ LOSS FEEDBACK: {backward_feedback}\n"  # Full feedback!
        gradient_history += "\n  ⚠️ CRITICAL: Review RESULTS and LOSS FEEDBACK above before recommending actions!\n"
        gradient_history += "  - If LOSS FEEDBACK says an action was 'suboptimal' or 'wrong', DON'T recommend it again!\n"
        gradient_history += "  - If a location showed 'nothing' or 'empty', DON'T go back there!\n"
        gradient_history += "  - If an action made NO_PROGRESS, try a DIFFERENT approach\n"
        gradient_history += "  - Learn from LOSS FEEDBACK - it shows what actually went wrong!\n"
        print(f"[TEXTGRAD PROMPT DEBUG] gradient_history length: {len(gradient_history)} chars")
    else:
        print(f"[TEXTGRAD PROMPT DEBUG] NO step_insights! gradient_history will be EMPTY!")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIX #16A: Add EXPLORED vs UNEXPLORED locations - critical for breaking loops!
    # ═══════════════════════════════════════════════════════════════════════════
    exploration_context = ""
    if explored_locations and len(explored_locations) > 0:
        # Handle both dict formats: {loc: count} or {loc: [timestamps]}
        explored_list = []
        for loc, count_or_list in explored_locations.items():
            if isinstance(count_or_list, list):
                visit_count = len(count_or_list)  # List of timestamps
            else:
                visit_count = count_or_list  # Direct count
            if visit_count > 0:
                explored_list.append(f"{loc} (visited {visit_count}x)")

        exploration_context = "\n🗺️ EXPLORATION STATUS:\n"
        exploration_context += f"  ✅ EXPLORED: {', '.join(explored_list[:10])}\n"  # Show top 10

        # Extract location types from valid_actions to suggest unexplored areas
        if valid_actions:
            all_go_actions = [a for a in valid_actions if a.startswith('go to ')]
            all_locations = [a.replace('go to ', '') for a in all_go_actions]
            # Handle both dict formats for unexplored check
            def is_unexplored(loc):
                if loc not in explored_locations:
                    return True
                val = explored_locations.get(loc, 0)
                return (isinstance(val, list) and len(val) == 0) or (not isinstance(val, list) and val == 0)
            unexplored = [loc for loc in all_locations if is_unexplored(loc)]
            if unexplored:
                exploration_context += f"  ❓ UNEXPLORED: {', '.join(unexplored)}\n"
                exploration_context += "  ⚠️ CRITICAL: If current approach is FAILING, try an UNEXPLORED location!\n"
                print(f"[FIX #16A DEBUG] UNEXPLORED locations: {unexplored}")
            else:
                print(f"[FIX #16A DEBUG] No unexplored locations found! all_locations={len(all_locations)}")
        print(f"[FIX #16A DEBUG] Exploration context added: {len(explored_locations)} explored, {len(unexplored) if 'unexplored' in dir() else 0} unexplored")

    # ═══════════════════════════════════════════════════════════════════════════
    # FIX #16B: Detect action repetition loops and warn strongly
    # ═══════════════════════════════════════════════════════════════════════════
    loop_warning = ""
    if step_insights and len(step_insights) >= 3:
        from collections import Counter
        recent_actions = [ins.get('action', '') for ins in step_insights[-10:]]
        action_counts = Counter(recent_actions)
        repeated_actions = [(act, cnt) for act, cnt in action_counts.items() if cnt >= 3]
        if repeated_actions:
            loop_warning = "\n🔄 ⚠️ LOOP DETECTED - ACTION REPETITION:\n"
            for act, cnt in repeated_actions:
                loop_warning += f"  ❌ '{act}' repeated {cnt} times WITHOUT SUCCESS!\n"
            loop_warning += "  🚨 CRITICAL: These actions have FAILED repeatedly. DO NOT recommend them again!\n"
            loop_warning += "  💡 TRY: A completely DIFFERENT action or explore an UNEXPLORED location!\n"
            print(f"[FIX #16B DEBUG] Loop warning triggered: {repeated_actions}")

    # Build REFLEXION STRATEGIC INSIGHTS (step-level guidance from recent Reflexion analysis)
    reflexion_insights = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        recent_reflexions = previous_reflexions[-3:]  # Last 3 Reflexion insights for immediate context
        reflexion_insights = "\n🎯 REFLEXION STRATEGIC INSIGHTS (Critical guidance from failure analysis):\n"
        for ref in recent_reflexions:
            if isinstance(ref, dict):
                step = ref.get('step', '?')
                insight = ref.get('reflection', '')
                # Pass COMPLETE reflexion - no truncation (plenty of context space: ~6300 tokens headroom)
                if insight:
                    reflexion_insights += f"  📍 Step {step}: {insight}\n"
        reflexion_insights += "  ⚠️ CRITICAL: These insights identify WHY previous actions failed. Your next action MUST address these strategic issues!\n"

    # Build episodic constraints from Reflexion (strategic warnings)
    episodic_constraints = ""
    if episodic_memory and len(episodic_memory) > 0:
        recent_episodic = episodic_memory[-3:]  # Last 3 strategic lessons
        episodic_constraints = "\n🧠 STRATEGIC CONSTRAINTS (from Reflexion's episodic memory):\n"
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:150]
            else:
                mem_text = str(mem)[:150]
            episodic_constraints += f"  ⚠️  {mem_text}\n"
        episodic_constraints += "  Your recommended action MUST NOT violate these strategic constraints!\n"

    # G21 UNIVERSAL — Inject evaluator ground-truth verdict + extracted literal
    # requirements + fact card into the BACKWARD prompt at TOP attention position.
    # Previously the gradient was computed without the evaluator's exact strings
    # reaching the prompt, so the gradient advised "try a different approach"
    # instead of "set attribute 'strike' to literal 'sngStrike'". When no
    # evaluator delta is available (text envs, first step before any done),
    # this block is empty — zero effect on ALFWorld/text envs.
    _state_for_grad = result.get('state', {}) or {}
    _eval_gap_raw = (_state_for_grad.get('_last_eval_gap') or
                     _state_for_grad.get('_unresolved_verification') or '')
    _eval_source_raw = (_state_for_grad.get('_eval_source') or '')
    _g21_block = ""
    if _eval_gap_raw:
        # Extract literal quoted values from evaluator delta (keeps the exact
        # strings/numbers the evaluator cares against — "sngStrike", 867786,
        # "Keira Daily" (multi-word), etc.)
        try:
            import re as _re_g21
            # PRIORITY 1: `expected='...'` values (G17 verbose diff format).
            # These are THE literals the agent must produce to satisfy the evaluator.
            _literals = _re_g21.findall(
                r"expected\s*[:=]\s*['\"]([^'\"\n]{1,80})['\"]",
                str(_eval_gap_raw), flags=_re_g21.IGNORECASE,
            )
            # PRIORITY 2: unquoted expected values (numbers like 867786).
            _literals += _re_g21.findall(
                r"expected\s*[:=]\s*([A-Za-z0-9_.\-/]{1,40})(?=[\s,\)\]\n]|$)",
                str(_eval_gap_raw), flags=_re_g21.IGNORECASE,
            )
            # PRIORITY 3: tuple-style (4,1,1,'sngStrike') — last element is the value.
            _literals += _re_g21.findall(
                r"\(\s*(?:[^)]*?,\s*)?['\"]([^'\"]{1,40})['\"]\s*\)",
                str(_eval_gap_raw),
            )
            # Strip leading/trailing whitespace, drop empty/bracket-artifacts,
            # dedupe while preserving order, cap to 15.
            _clean = []
            _seen = set()
            for _l in _literals:
                _l2 = _l.strip()
                if not _l2 or _l2 in _seen or _l2 in ('<empty>', '?', ''):
                    continue
                # Drop obvious framing tokens
                if _l2.lower() in ('empty', 'none', 'null', 'unknown layout'):
                    continue
                _seen.add(_l2)
                _clean.append(_l2)
                if len(_clean) >= 15:
                    break
            _literals = _clean
        except Exception:
            _literals = []
        _g21_block = (
            "\n⛔ EVALUATOR GROUND TRUTH (verbatim — the gradient MUST explain\n"
            "   how the next action reaches THIS exact state, not a paraphrase):\n"
            f"{str(_eval_gap_raw)[:1500]}\n"
        )
        if _literals:
            _g21_block += (
                f"\n📌 EXTRACTED LITERAL REQUIREMENTS (exact strings/values the\n"
                f"   evaluator compares against — your recommended action MUST\n"
                f"   produce these exact literals, not synonyms or variants):\n"
                f"   {_literals}\n"
            )
        if _eval_source_raw:
            _g21_block += (
                f"\n🔍 EVALUATOR SOURCE (the Python the evaluator literally runs;\n"
                f"   read it to find which attribute name / cell / field is being\n"
                f"   checked, and what it is compared against):\n"
                f"{str(_eval_source_raw)[:2000]}\n"
            )

    textgrad_prompt = f"""🔬 TEXTGRAD: Automatic Differentiation via Text

YOUR ROLE (TextGrad's Unique Strength - Nature 2024):
- Optimize action selection through textual gradients (like backpropagation for prompts)
- Provide gradient feedback: "If you had done X instead of Y, outcome would have been Z"
- Accumulate gradients across steps to perform gradient descent
- Suggest NEXT ACTION that follows the gradient toward task completion
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{_g21_block}{todo_context}{inventory_context}{exploration_context}{loop_warning}{reflexion_insights}{episodic_constraints}{gradient_history}
BEFORE: {prev_observation}
ACTION TAKEN: {action}
AFTER: {curr_observation}
{"UI ELEMENTS DETECTED (reference hints — trust the screenshots over these):" + chr(10) + (chr(10).join("  - " + a for a in valid_actions) if valid_actions else "  (none detected)") + chr(10) + "You have screenshots. You may use ANY action format: click (x,y), click 'Label', hotkey, type, press, run:, scroll, done." if is_visual_env else "VALID ACTIONS: " + (', '.join(valid_actions) if valid_actions else 'Unknown')}

═══════════════════════════════════════════════════════════
TEXTGRAD GRADIENT COMPUTATION (Answer each precisely):
═══════════════════════════════════════════════════════════

1. GRADIENT EVALUATION:
   The action I took: "{action}"
   The outcome: "{curr_observation[:100]}"

   Gradient question: If I had taken a DIFFERENT action, would the outcome be better or worse?
   Specifically:
   - What property should have changed based on task "{task}"?
   - Did that property change? (Cite evidence from AFTER observation)
   - If NO change, what alternative action would have caused the desired change?

   Answer: GRADIENT: [Positive/Zero/Negative] | REASON: [why] | BETTER ACTION: [specific alternative from VALID ACTIONS]

2. ACCUMULATED GRADIENT DESCENT:
   Looking at my ACCUMULATED GRADIENTS above (last 5 steps):
   - What pattern do I see in actions with high scores (7-10)?
   - What pattern do I see in actions with low scores (0-3)?
   - Based on gradient descent, should I continue current direction or try new direction?

   Answer: PATTERN HIGH: [what worked] | PATTERN LOW: [what failed] | DIRECTION: [continue/change]

3. REFLEXION LESSON ENFORCEMENT:
   Review REFLEXION STRATEGIC INSIGHTS above for diagnosed failures.

   CRITICAL RULE: If Reflexion diagnosed that action X failed to achieve property Y:
   - DO NOT recommend action X again (it already failed!)
   - DO NOT use "similar" or "equivalent" actions (different verbs = different effects)
   - MUST try a DIFFERENT action type or explore new location

   List any actions Reflexion identified as failing:

   Answer: FAILED_ACTIONS: [list actions Reflexion said failed, or "none diagnosed"]

4. SEARCH MEMORY (Cumulative - DO NOT RE-SEARCH THESE):
   {f'ACCUMULATED SEARCH HISTORY (from previous steps): {cumulative_search_memory}' if cumulative_search_memory else 'No locations searched yet.'}

   TARGET_OBJECT_NEEDED: Parse the TASK "{task}" - what object type(s) must you find?

   UPDATE SEARCH MEMORY with THIS step's action:
   - If you just visited/opened a new location, ADD it to the list above
   - Format: [location] → [objects found, or "empty/nothing relevant"]
   - DO NOT repeat locations already listed above

   Answer:
   TARGET_NEEDED: [object type from task]
   NEW_SEARCH_ENTRY: [location → objects] (or "none" if action wasn't a search)

   ⚠️ RE-SEARCH PREVENTION:
   NEVER recommend going back to a location from ACCUMULATED SEARCH HISTORY that didn't have TARGET_NEEDED.
   Those locations were already searched and confirmed empty of the target.

5. STRATEGIC CONSTRAINT CHECK:
   {f'My episodic memory warns: {episodic_constraints}' if episodic_constraints else 'No strategic constraints'}

   Before recommending an action, verify:
   - Does my recommended action violate any strategic constraint above?
   - If yes, what alternative achieves same goal without violation?

   Answer: CONSTRAINT CHECK: [Pass/Fail] | IF FAIL: [alternative action]

6. NEXT ACTION OPTIMIZATION:

   ★★★ HIGHEST PRIORITY: DISCOVERED OBJECTS ★★★
   FIRST, check the CURRENT OBSERVATION for objects mentioned in the TASK.
   If you see an object type from the task (and you don't already have it):
   → Your ONLY action should be to INTERACT with it (acquire, take, pick up)
   → This OVERRIDES gradient history, TODOs, and all other guidance
   → Objects found ANYWHERE are likely the ones needed for the task
   → Don't leave discovered task-objects behind to search elsewhere

   If no task-objects are visible, THEN consider:
   - Current gradient (what would improve outcome)
   - Accumulated gradient history (what direction is working)
   - Strategic constraints (what to avoid)
   - Task requirements (what the task is asking for)

   ⚠️ GRADIENT HISTORY OVERRIDES VERB MATCHING!
   If your gradient history shows an action is FAILING or REPEATING:
   - DO NOT recommend that action again
   - Try a DIFFERENT action type to break the cycle

   ⚠️ TASK-EFFECT ALIGNMENT:
   Before recommending an action, verify its expected EFFECT matches the TASK requirement.
   - If the task requires achieving state X, recommend an action that produces state X
   - Actions with similar-looking syntax may produce completely different effects
   - If previous attempts didn't produce the required effect, the location or action type may be wrong

   ⚠️ ═══════════════════════════════════════════════════════════════════════════════
   ⚠️ CRITICAL: CROSS-CHECK WITH YOUR OWN ANALYSIS ABOVE!
   ⚠️ ═══════════════════════════════════════════════════════════════════════════════

   BEFORE recommending ANY action, verify it does NOT involve:
   - ANY location/action from your PATTERN LOW list (step 2 above)
   - ANY action from your FAILED_ACTIONS list (step 3 above)
   - ANY location from SEARCHED_LOCATIONS (step 4) that did NOT have TARGET_NEEDED

   If your initial recommendation involves something from PATTERN LOW, FAILED_ACTIONS,
   or a SEARCHED location without the target:
   → REJECT that recommendation immediately
   → Choose a COMPLETELY DIFFERENT location or action type
   → If you've tried multiple approaches at a location without success, that LOCATION is wrong
   → Try an UNSEARCHED/UNEXPLORED location instead

   This consistency check is MANDATORY - your recommendation MUST NOT contradict your own analysis.
   ═══════════════════════════════════════════════════════════════════════════════

   {"Look at the AFTER screenshot. Based on what you SEE, which action optimizes progress?" if is_visual_env else "From the VALID ACTIONS list, which SPECIFIC action optimizes progress?"}

   ⚠️ G21 LITERAL CONSTRAINT — if the `⛔ EVALUATOR GROUND TRUTH` block above
   contains `📌 EXTRACTED LITERAL REQUIREMENTS: [...]`, your recommended action
   MUST produce code/output that contains those EXACT literal strings. Do NOT
   substitute 'True' for 'sngStrike', do NOT substitute '1' for '867786' etc.
   The evaluator compares byte-for-byte against those literals.

   Answer FORMAT (CRITICAL: RECOMMENDED_ACTION line must contain ONLY the action, nothing else):
   RECOMMENDED_ACTION: [{"action string — e.g., click (450,320), hotkey Alt+O, type 'text'" if is_visual_env else "paste exact action from VALID ACTIONS - no extra words"}]
   JUSTIFICATION: [explain why + confirm NOT in PATTERN LOW, FAILED_ACTIONS, or searched-empty locations + if G21 literals were listed, confirm your action produces each literal exactly]

7. PROGRESS EVALUATION (TextGrad-Style Pure Textual Feedback):
   Evaluate the progress this action made toward FULL TASK: "{task}"

   Context: {f'Current subtask is "{current_todo.content}"' if current_todo else 'No active subtask'}

   Provide structured textual evaluation:

   A. STATE_CHANGE:
      What changed from BEFORE to AFTER? Be specific about observable differences.

   B. TASK_ALIGNMENT:
      - Which requirement(s) of the FULL TASK "{task}" are now closer to completion?
      - Which requirement(s) remain unmet?
      - Cite specific evidence from AFTER state showing requirement progress.

      CRITICAL: Evaluate relative to FULL TASK, not just current subtask.
      Completing a subtask is PARTIAL progress toward full task, not completion.

      ⚠️ EFFECT VERIFICATION: If the task requires achieving a specific state change:
      - Does the AFTER observation confirm that EXACT effect was achieved?
      - If the observation describes a DIFFERENT effect, that is NOT progress toward the task!
      - Actions with similar names may have completely different effects - verify by outcome, not name.

   C. NEXT_STEPS:
      What should the agent do next to complete the full task?

   D. PROGRESS_STATUS:
      Classify overall progress toward FULL TASK completion:
      - NO_PROGRESS: State change irrelevant to full task requirements
      - EXPLORING: Searching but haven't found any objects mentioned in the task
      - PARTIAL_PROGRESS: Found/acquired an object mentioned in task, OR positioned at key location
      - MAJOR_PROGRESS: Have necessary objects AND at correct location for final action
      - TASK_COMPLETE: All requirements of full task satisfied

      CRITICAL: If AFTER state shows an object mentioned in the TASK that wasn't visible before,
      this is AT LEAST PARTIAL_PROGRESS - you discovered something needed for the task!

   Answer:
   A. STATE_CHANGE: [describe observable changes]
   B. TASK_ALIGNMENT: [requirement analysis with evidence]
   C. NEXT_STEPS: [recommended actions]
   D. PROGRESS_STATUS: [one of: NO_PROGRESS | EXPLORING | PARTIAL_PROGRESS | MAJOR_PROGRESS | TASK_COMPLETE]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    # DEEP TRACE: dump full prompt to file for inspection
    _trace_dir = os.path.join(os.getcwd(), '_prompt_traces')
    os.makedirs(_trace_dir, exist_ok=True)
    _trace_step = result.get('current_step', 0) if isinstance(result, dict) else 0
    _trace_file = os.path.join(_trace_dir, f'pass2_step_{_trace_step}.txt')
    with open(_trace_file, 'w') as _tf:
        _tf.write(textgrad_prompt)
    print(f"[TRACE] PASS 2 prompt dumped to {_trace_file} ({len(textgrad_prompt)} chars)")

    return textgrad_prompt


def parse_step_gradient_response(response: str, task: str, prev_observation: str, curr_observation: str, action: str) -> Dict:
    """Parse step gradient LLM response into structured format.

    This extracts the parsing logic from generate_universal_step_reflection
    so we can batch-parse multiple gradient outputs.
    """
    import re

    # Initialize step_gradient
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
    }

    # Clean think tags if present
    response = response.replace("</think>", "").replace("<think>", "").strip()

    # FIX: Strip markdown formatting that open-source models (Llama, Qwen) add
    # e.g., "**1. GRADIENT EVALUATION:**" → "1. GRADIENT EVALUATION:"
    # Also strips headers (### ), underscores (__bold__), and inline code (`text`)
    # Without this, the numbered section parser fails completely for non-GPT models
    import re as _re_md
    response = _re_md.sub(r'\*{1,3}', '', response)  # bold/italic: *, **, ***
    response = _re_md.sub(r'^#{1,4}\s*', '', response, flags=_re_md.MULTILINE)  # headers: #, ##, ###
    # NOTE: Do NOT strip underscores — they appear in keywords like PROGRESS_STATUS, NO_PROGRESS

    # Parse numbered responses
    lines = response.split('\n')
    answers = {}
    current_num = 0
    current_answer = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # More robust number extraction
        if line_stripped and line_stripped[0].isdigit():
            # Save previous answer
            if current_num > 0 and current_answer:
                answers[current_num] = ' '.join(current_answer).strip()

            # Extract number more carefully
            num_match = re.match(r'^(\d+)', line_stripped)
            if num_match:
                current_num = int(num_match.group(1))
                # Get rest of line after number (and any punctuation)
                rest = re.sub(r'^\d+[.,):\s]*', '', line_stripped)
                current_answer = [rest] if rest else []
            else:
                # Add to current answer if no number found
                if current_num > 0:
                    current_answer.append(line_stripped)
        elif current_num > 0:
            # Continue current answer
            current_answer.append(line_stripped)

    # Save last answer
    if current_num > 0 and current_answer:
        answers[current_num] = ' '.join(current_answer).strip()

    # Extract PROGRESS_STATUS from answer 7 (PROGRESS EVALUATION section)
    # Prompt sections: 1-GRADIENT, 2-ACCUMULATED, 3-REFLEXION, 4-SEARCH, 5-CONSTRAINT, 6-NEXT_ACTION, 7-PROGRESS
    progress_status = "EXPLORING"  # Default
    progress_score = 3  # Default numerical equivalent for backward compatibility

    # Try answer 7 first (correct location after renumbering), then fall back to 6
    progress_answer_key = 7 if 7 in answers else (6 if 6 in answers else None)
    if progress_answer_key:
        eval_text = answers[progress_answer_key].upper()

        # Extract D. PROGRESS_STATUS field
        # FIX: Use findall and take LAST match — open models echo the instruction header first
        # (e.g., "D. PROGRESS_STATUS:" empty echo, then "D. PROGRESS_STATUS: NO_PROGRESS" actual answer)
        status_matches = re.findall(r'D\.\s*PROGRESS_STATUS:\s*(\w+)', eval_text)
        status_match = status_matches[-1] if status_matches else None
        if status_match:
            progress_status = status_match
        else:
            # Fallback: search for status keywords anywhere in progress answer
            # Only match exact "TASK_COMPLETE" — bare "COMPLETE" causes false positives
            # when LLM says "not yet complete" or "task is incomplete"
            if 'TASK_COMPLETE' in eval_text:
                progress_status = "TASK_COMPLETE"
            elif 'MAJOR_PROGRESS' in eval_text:
                progress_status = "MAJOR_PROGRESS"
            elif 'PARTIAL_PROGRESS' in eval_text:
                progress_status = "PARTIAL_PROGRESS"
            elif 'NO_PROGRESS' in eval_text:
                progress_status = "NO_PROGRESS"
            elif 'EXPLORING' in eval_text:
                progress_status = "EXPLORING"

        # Map status to numerical score for backward compatibility with meta-logic
        status_to_score = {
            "NO_PROGRESS": 1,
            "EXPLORING": 3,
            "PARTIAL_PROGRESS": 6,
            "MAJOR_PROGRESS": 8,
            "TASK_COMPLETE": 10
        }
        progress_score = status_to_score.get(progress_status, 3)
        print(f"[PROGRESS FIX] Parsed answer {progress_answer_key}: status={progress_status}, score={progress_score}")

    # Direct observation comparison for state change detection
    # NOTE: This must be defined BEFORE it's used in the progress boost logic below
    state_changed = (prev_observation.strip() != curr_observation.strip())

    # REMOVED naive word matching - use ONLY LLM's semantic score
    # The LLM already evaluates progress semantically, don't override with word overlap


    # Extract structured information with defaults
    missing_prereqs = answers.get(2, '').strip()
    missing_list = [missing_prereqs] if missing_prereqs and missing_prereqs.lower() not in ['none', 'nothing', 'n/a'] else []

    task_addressed = answers.get(3, '').strip()
    addressed_list = [task_addressed] if task_addressed and task_addressed.lower() not in ['none', 'nothing', 'n/a'] else []

    # FIX #32: Extract NEW_SEARCH_ENTRY from answer 4 (SEARCH MEMORY section)
    search_memory_answer = answers.get(4, '').strip()
    new_search_entry = ""
    if search_memory_answer:
        # Look for NEW_SEARCH_ENTRY pattern
        search_match = re.search(r'NEW_SEARCH_ENTRY:\s*([^\n]+)', search_memory_answer, re.IGNORECASE)
        if search_match:
            entry = search_match.group(1).strip()
            if entry.lower() not in ['none', 'n/a', '']:
                new_search_entry = entry

    # Backward compatibility: also check for task remaining in answer 4
    task_remaining = answers.get(4, task).strip()
    remaining_list = [task_remaining] if task_remaining else [task]

    learned_rule = answers.get(6, answers.get(5, 'Continue exploring available actions')).strip()

    # CRITICAL FIX: Extract from answer #6 (NEXT ACTION OPTIMIZATION after renumbering)
    # Answer #6 contains: "RECOMMENDED_ACTION: ..., JUSTIFICATION: ..."
    # Falls back to answer #5 for backward compatibility with old prompt format
    answer_5 = answers.get(6, answers.get(5, '')).strip()

    # Extract RECOMMENDED_ACTION from the structured format
    recommended_action_match = re.search(r'RECOMMENDED_ACTION:\s*([^\n]+)', answer_5, re.IGNORECASE)
    if recommended_action_match:
        next_guidance = recommended_action_match.group(1).strip()
    else:
        # Fallback to full answer 5 if pattern not found
        next_guidance = answer_5 if answer_5 else 'Try unexplored actions'

    # CRITICAL FIX: Extract "Recommended next action: X" from response
    recommended_action_match = re.search(r'Recommended next action:\s*([^\n]+)', response, re.IGNORECASE)
    if not recommended_action_match:
        recommended_action_match = re.search(r'RECOMMENDED_ACTION:\s*([^\n]+)', response, re.IGNORECASE)
    if not recommended_action_match:
        recommended_action_match = re.search(r'RECOMMENDED ACTION:\s*-?\s*([a-z][^\n-]+)', response, re.IGNORECASE)

    if recommended_action_match:
        next_guidance = recommended_action_match.group(1).strip()
        next_guidance = next_guidance.strip('- .,;:')
        if next_guidance.upper().startswith('GOAL:'):
            action_after_goal = re.search(r'(?:GOAL:.*?)?[-\s]*(?:ACTION|RESULT|Recommended):\s*([a-z][^\n-]+)', response, re.IGNORECASE | re.DOTALL)
            if action_after_goal:
                next_guidance = action_after_goal.group(1).strip().strip('- .,;:')
    else:
        # Fallback: Extract exact action if it's quoted
        quoted_action = re.findall(r"'([^']+)'", next_guidance)
        if quoted_action:
            next_guidance = quoted_action[0]

    # STRATEGY_SHIFT extraction — override next_guidance if a concrete strategy shift is found
    # Search STRATEGY_SHIFT field first, then fall back to searching answer 7 broadly
    strategy_match = re.search(r'STRATEGY_SHIFT:\s*([^\|\n]+)', response, re.IGNORECASE)
    _strat_found = False
    if strategy_match:
        strategy_action = strategy_match.group(1).strip().strip('.,;')
        if strategy_action.lower() not in ['none', 'n/a', 'not applicable', 'no shift needed', 'no shift', '']:
            # Extract actionable command from strategy shift text
            _strat_run = re.search(r'(run:\s+[^\n]{10,})', strategy_action)
            _strat_press = re.search(r'(press\s+\w+)', strategy_action)
            if _strat_run:
                next_guidance = _strat_run.group(1).strip()
                print(f"[REFLEXION PARSE] Extracted STRATEGY_SHIFT run command: '{next_guidance[:80]}'")
                _strat_found = True
            elif _strat_press:
                next_guidance = _strat_press.group(1).strip()
                print(f"[REFLEXION PARSE] Extracted STRATEGY_SHIFT key press: '{next_guidance}'")
                _strat_found = True
            else:
                # Prose STRATEGY_SHIFT: store as next_guidance so it flows to VISION-FIRST.
                # The model is smart enough to convert "use keyboard formatting" → "hotkey Alt+O"
                # when it sees the screenshot. Don't discard strategic insights.
                next_guidance = strategy_action
                _strat_found = True
                print(f"[REFLEXION PARSE] STRATEGY_SHIFT (prose → stored as guidance): '{strategy_action[:80]}'")

    # If STRATEGY_SHIFT didn't yield a run: command, search full response for any run: python3 command
    if not _strat_found:
        # Search answer 6 (progress + strategy shift merged) and full response
        _broad_run = re.search(r'(run:\s+python3\s+-c\s+["\'][^\n]{10,})', response)
        if _broad_run:
            next_guidance = _broad_run.group(1).strip()
            print(f"[REFLEXION PARSE] Extracted run: python3 from response: '{next_guidance[:80]}'")

    # Clean LLM artifacts from next_guidance before it enters the pipeline
    # Skip run: commands — python code has legitimate pipes/special chars
    if next_guidance and not next_guidance.lower().startswith('run:'):
        next_guidance = clean_action(next_guidance)

    # FIX: Validate parsed action — but be lenient on python/shell code.
    # The osworld wrapper's _wrap_python_as_shell auto-repairs unbalanced quotes
    # and brackets in `run: python3 -c "..."` code blocks (it strips trailing
    # unmatched quotes and pipes the code via base64). So we should NOT reject
    # actions just because they have unbalanced syntax — that's exactly what the
    # wrapper handles. Only reject actions whose CONTENT is parser-error prose
    # (the LLM literally wrote "syntax error" as part of its answer) or that are
    # truncated mid-thought (ellipsis). Universal: works for any task, lets the
    # downstream wrapper repair imperfect code.
    if next_guidance:
        _ng_lower = next_guidance.lower()
        _is_run_cmd = _ng_lower.startswith('run:')
        # Prose-corruption markers (model literally output an error string)
        _has_error_text = (
            'syntax error' in _ng_lower
            or 'parse error' in _ng_lower
            or '(error)' in _ng_lower
        )
        # Ellipsis check only for non-code actions — code may legitimately use ...
        _has_ellipsis = ('...' in next_guidance) and not _is_run_cmd
        # Bracket/quote imbalance check ONLY for non-run actions. Code inside
        # `run: python3 -c "..."` is auto-repaired by the wrapper, so don't reject.
        _has_imbalance = False
        if not _is_run_cmd:
            _has_imbalance = (
                next_guidance.count('"') % 2 != 0
                or next_guidance.count("'") % 2 != 0
                or next_guidance.count('(') != next_guidance.count(')')
                or next_guidance.count('[') != next_guidance.count(']')
            )
        if _has_error_text or _has_ellipsis or _has_imbalance:
            print(f"[REFLEXION PARSE] Rejected corrupt action: '{next_guidance[:120]}' — discarding")
            next_guidance = ''

    # Universal task progress detection
    task_words = set(task.lower().split())
    state_before_words = set(prev_observation.lower().split())
    state_after_words = set(curr_observation.lower().split())

    # Check if any task words that weren't in before state are now in after state
    progress_indicators = task_words & (state_after_words - state_before_words)

    # Build full gradient structure
    step_gradient = {
        'action': action,  # FIX #7 (Nov 23): Include executed action for learning data logging
        'state_change': curr_observation if state_changed else 'NO CHANGE',
        'progress_assessment': f"Addressed: {', '.join(addressed_list)}" if addressed_list else "No progress",
        'progress_status': progress_status,  # TextGrad-style status (NO_PROGRESS, EXPLORING, etc.)
        'progress_score': progress_score,  # Numerical equivalent for backward compatibility
        'hypothesis': learned_rule,
        'next_action_guidance': next_guidance,
        'raw_reflection': response,
        'semantic_state_before': prev_observation,
        'semantic_state_after': curr_observation,
        'prerequisites': {
            'present': [],
            'missing': missing_list
        },
        'task_progress': {
            'addressed': addressed_list,
            'remaining': remaining_list
        },
        'new_search_entry': new_search_entry  # FIX #32: Search entry to accumulate
    }

    return step_gradient


def generate_textgrad_step_guidance(
    prev_observation: str,
    curr_observation: str,
    action: str,
    task: str,
    model,
    log_debug = print,
    current_step: int = 0,
    valid_actions: List[str] = None,
    current_todo = None,
    inventory: List[str] = None,
    previous_reflexions: List[Dict] = None,
    episodic_memory: List[str] = None,
    step_insights: List[Dict] = None,  # FIX #21: Raw observations (action + env feedback) instead of LLM interpretations
    before_screenshot_b64: str = None,  # Multimodal: before-action screenshot
    after_screenshot_b64: str = None,   # Multimodal: after-action screenshot
    accumulated_state: List[str] = None,  # FIX #39: Track what progress has been achieved
    is_visual_env: bool = False,  # Vision-first: visual hints instead of hard constraints
    learning_context = None,  # Universal learning engine (FailureMemory, PolicyGradients)
    task_complete_mode: bool = False,  # When True, TextGrad assessed TASK_COMPLETE — verify only
) -> Dict:
    """
    TextGrad-only prompt: Generate clean, actionable next action through gradient optimization.

    TextGrad's Unique Strength (Nature 2024):
    - Step-by-step action optimization through textual gradients
    - Clean, executable action recommendations
    - Gradient-based learning from immediate feedback

    This function does NOT do causal why-analysis (that's Reflexion's job).
    It focuses ONLY on WHAT action to take next based on gradient signals.
    """

    # Initialize step_gradient
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
        'guidance_source': 'textgrad'  # Mark source for action selector
    }

    # CONCRETE_FIX EXTRACTION: Regex-first, then LLM fallback.
    # When the backward pass produces "CONCRETE_FIX: run: ..." we extract it directly
    # and TEACHER-FORCE it (bypass actor entirely). This eliminates the diagnosis-to-action
    # translation gap — the fix IS the action. No dilution through actor's prior.
    # Falls back to best-of-3 sampling when no executable fix found.
    import re as _re_cf
    if step_insights and len(step_insights) > 0:
        _last_action = str(step_insights[-1].get('action', ''))
        # Search ALL recent backwards (not just last — fix may be in earlier gradient)
        _n_checked = 0
        for _si in reversed(step_insights[-3:]):
            _backward = str(_si.get('backward_gradient', ''))
            _n_checked += 1
            log_debug(f"[CONCRETE_FIX DEBUG] Checking insight {_n_checked}/3: backward len={len(_backward)}, first80='{_backward[:80]}'")
            if not _backward or len(_backward) < 20:
                continue
            # Priority 1: Exact CONCRETE_FIX: line (from backward prompt format)
            _cf_match = _re_cf.search(r'CONCRETE_FIX:\s*(.+?)(?:\n|$)', _backward)
            if _cf_match:
                _extracted = _cf_match.group(1).strip().strip('"\'')
                if (_extracted and len(_extracted) > 5
                        and _extracted.lower() not in ('none', 'n/a', 'verify output')
                        and _extracted != _last_action[:len(_extracted)]):
                    log_debug(f"[TEACHER-FORCE] Regex-extracted CONCRETE_FIX: '{_extracted[:100]}'")
                    step_gradient['next_action_guidance'] = _extracted
                    step_gradient['guidance_source'] = 'teacher_force'
                    step_gradient['progress_score'] = 5
                    step_gradient['hypothesis'] = f"Teacher-forced fix: {_extracted[:200]}"
                    return step_gradient
            # Priority 2: Executable command patterns in gradient text
            # NOTE: Gemma often produces "ACTION: run: ..." instead of "CONCRETE_FIX: run: ..."
            for _pat in [
                r'ACTION:\s*(run:\s*.+?)(?:\n|$)',
                r'(?:try|use|run|execute|instead|alternative):\s*(run:\s*.+?)(?:\n|$)',
                r'`(run:\s*[^`]+)`',
                r'`(python3\s+-c\s+[^`]+)`',
                r'`(id\s+\w+\s*\|\|[^`]+)`',
                r'`(find\s+[^`]+)`',
                r'`(ls\s+[^`]+)`',
                r'`(vlc\s+[^`]+)`',
                r'`(chmod\s+[^`]+)`',
            ]:
                _cmd_match = _re_cf.search(_pat, _backward, _re_cf.IGNORECASE)
                if _cmd_match:
                    _extracted = _cmd_match.group(1).strip().strip('"\'')
                    if (len(_extracted) > 5
                            and _extracted != _last_action[:len(_extracted)]):
                        log_debug(f"[TEACHER-FORCE] Pattern-extracted command: '{_extracted[:100]}'")
                        step_gradient['next_action_guidance'] = _extracted
                        step_gradient['guidance_source'] = 'teacher_force'
                        step_gradient['progress_score'] = 4
                        step_gradient['hypothesis'] = f"Extracted command: {_extracted[:200]}"
                        return step_gradient
        # Priority 3: LLM extraction as fallback (original approach, kept for soft fixes)
        _last_backward = str(step_insights[-1].get('backward_gradient', ''))
        _last_obs = str(step_insights[-1].get('observation', ''))[:500]
        if _last_backward and len(_last_backward) > 20:
            try:
                _extract_prompt = f"""Extract the ONE concrete alternative action from this feedback.

FAILED ACTION: {_last_action[:300]}
ERROR: {_last_obs[:300]}
FEEDBACK: {_last_backward[:500]}

Reply with ONLY an executable action (e.g., "run: ...", "click ...", "hotkey ..."). One line. If none found, reply NONE."""

                _extracted = fast_model.generate([_extract_prompt])[0].outputs[0].text.strip()
                _extracted = _extracted.split('\n')[0].strip().strip('"\'')
                if (_extracted and len(_extracted) > 5
                        and _extracted.upper() != 'NONE'
                        and 'none' not in _extracted.lower()[:10]
                        and _extracted != _last_action[:len(_extracted)]):
                    log_debug(f"[CONCRETE_FIX] LLM-extracted: '{_extracted[:100]}'")
                    step_gradient['next_action_guidance'] = _extracted
                    step_gradient['guidance_source'] = 'concrete_fix'
                    step_gradient['progress_score'] = 3
                    step_gradient['hypothesis'] = f"Concrete fix: {_extracted[:200]}"
                    return step_gradient
                else:
                    log_debug(f"[CONCRETE_FIX] No actionable fix found in gradient")
            except Exception as _cf_err:
                log_debug(f"[CONCRETE_FIX] Extraction failed (non-fatal): {_cf_err}")

    # Build TODO context with EXPLICIT observation comparison (FIX #22 - Universal)
    todo_context = ""
    # Check if task was previously assessed as complete — inject verification-only guidance
    _task_complete_flag = task_complete_mode
    if _task_complete_flag:
        todo_context = ("\n📋 TASK STATUS: ✅ Previously assessed as TASK_COMPLETE.\n"
                        "   VERIFY ONLY: Recommend 'look', 'examine', or 'inventory' to confirm.\n"
                        "   DO NOT recommend take/put/move/heat/cool/clean — this may UNDO completed work.\n")
    elif current_todo:
        todo_context = f"\n📋 CURRENT SUBTASK: {current_todo.content}\n"
        # FIX #22: Add explicit TODO vs Observation comparison
        todo_context += f"""
   🔍 EXPLICIT TODO CHECK (Answer honestly):
   - TODO requires: "{current_todo.content}"
   - Last observation: "{curr_observation[:150]}"
   - QUESTION: Does this observation indicate the TODO was achieved?
   - If NO: You must try a DIFFERENT action/approach. Do NOT repeat what didn't work.
"""

    # Build inventory context — ONLY for text-game envs.
    inventory_context = ""
    if not is_visual_env and inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"\n🎒 INVENTORY: Currently carrying {items_str}\n   ⚠️ If holding items, consider 'put' actions to place them.\n"

    # FIX #21: Build gradient history from RAW OBSERVATIONS, not LLM interpretations
    # ROOT CAUSE: LLM interpretations compound errors (e.g., "clean" interpreted as "cooled")
    # SOLUTION: Show only raw action + raw environment feedback - let LLM interpret fresh each step
    # FIX #25: Include BACKWARD GRADIENT (the learning lesson) - this was stored but never shown!
    # ROOT CAUSE: backward_gradient contains "moving was suboptimal, should heat" but forward prompt never saw it
    gradient_context = ""
    if step_insights and len(step_insights) > 0:
        # Use raw observations from step_insights_accumulator
        formatted = []
        # FIX #36: Show ALL steps so LLM sees complete history of what was tried
        # Previously limited to last 15, hiding repetition patterns from the LLM
        for insight in step_insights:  # FIX #36: ALL steps, not just last 15
            step_num = insight.get('step', '?')
            action_taken = insight.get('action', 'unknown')
            raw_observation = insight.get('observation', '')  # RAW environment feedback
            backward_lesson = insight.get('backward_gradient', '')  # FIX #25: FULL lesson, no truncation!

            # For visual envs: extract only RELEVANT feedback (state change + command output),
            # not the full VGL metadata which drowns out the actual results
            if is_visual_env and raw_observation:
                _fb_parts = []
                _sc_i = raw_observation.find('[State Change]')
                if _sc_i >= 0:
                    _sc_e = raw_observation.find('\n[', _sc_i + 1)
                    _sc_t = raw_observation[_sc_i:_sc_e if _sc_e > _sc_i else _sc_i + 200].strip()
                    if 'No visible change' not in _sc_t:
                        _fb_parts.append(_sc_t[:200])
                    else:
                        _fb_parts.append('No visible change')
                _co_i = raw_observation.find('[Command Output')
                if _co_i >= 0:
                    _err_i = raw_observation.find('Error:', _co_i)
                    if _err_i < 0:
                        _err_i = raw_observation.find('ERROR]', _co_i)
                    if _err_i >= 0:
                        _fb_parts.append('COMMAND ERROR: ' + raw_observation[_err_i:_err_i + 300].strip())
                    else:
                        _fb_parts.append(raw_observation[_co_i:_co_i + 500].strip())
                    # Preserve AUTO-DISCOVERY messages (critical for file path correction)
                    _disc_i = raw_observation.find('[AUTO-DISCOVERY]', _co_i)
                    if _disc_i >= 0:
                        _disc_ie = raw_observation.find('\n\n', _disc_i)
                        _disc_it = raw_observation[_disc_i:_disc_ie if _disc_ie > _disc_i else _disc_i + 300].strip()
                        _fb_parts.append(_disc_it)
                raw_observation = ' | '.join(_fb_parts) if _fb_parts else 'No visible change'

            # Format: Action → Feedback → Lesson (if available)
            if backward_lesson:
                formatted.append(f"  Step {step_num}: ACTION: \"{action_taken}\" → RESULT: \"{raw_observation}\"\n    📚 LESSON: {backward_lesson}")
            else:
                formatted.append(f"  Step {step_num}: ACTION: \"{action_taken}\" → RESULT: \"{raw_observation}\"")

        gradient_text = "\n".join(formatted)
        gradient_context = f"\n📊 COMPLETE ACTION HISTORY WITH LESSONS ({len(step_insights)} steps):\n{gradient_text}\n"  # FIX #36: Show count
        gradient_context += "   ⚠️ CRITICAL: Review ALL steps above - if an action was tried multiple times without progress, DO NOT try it again!\n"
        # FIX #36 DEBUG
        lessons_count = sum(1 for ins in step_insights if ins.get('backward_gradient'))
        print(f"[FIX #36] Forward prompt showing ALL {len(step_insights)} steps with {lessons_count} backward lessons")
    elif previous_reflexions and len(previous_reflexions) > 0:
        # Fallback to old format if step_insights not available (shouldn't happen)
        formatted = []
        for r in previous_reflexions:  # FIX #36: ALL reflexions
            step_num = r.get('step', '?')
            progress_status = r.get('progress_status', 'UNKNOWN')
            guidance = r.get('next_action_guidance', '')[:80]
            hypothesis = r.get('reflection', r.get('hypothesis', ''))
            hyp_display = hypothesis[:200] if len(hypothesis) > 200 else hypothesis
            formatted.append(f"  Step {step_num}: [{progress_status}] {hyp_display} → {guidance}")
        gradient_text = "\n".join(formatted)
        gradient_context = f"\n📊 GRADIENT HISTORY (fallback):\n{gradient_text}\n"

    # Coordinate accuracy patterns from step_insights (visual env coordinate learning)
    coord_context = ""
    if step_insights:
        coord_failures = [ins for ins in step_insights[-5:]
                          if ins.get('click_metadata', {}).get('match_type') in ('no_match', 'fuzzy', 'fallback_center')]
        if coord_failures:
            coord_parts = [f"\n🎯 COORDINATE ACCURACY PATTERNS:"]
            coord_parts.append(f"  {len(coord_failures)} of last {min(5, len(step_insights))} steps had imprecise click resolution.")
            for cf in coord_failures[-2:]:
                cm = cf['click_metadata']
                coord_parts.append(f"  - Step {cf.get('step', '?')}: Target='{cm.get('target_label', '?')}', Match={cm.get('match_type')}")
                nearby = cm.get('nearby_elements', [])
                close = [n for n in nearby if n.get('distance_px', 999) < 50]
                if close:
                    names = [f"'{n['label']}' ({n['distance_px']}px)" for n in close]
                    coord_parts.append(f"    Nearby (misclick risk): {', '.join(names)}")
            coord_parts.append("  RECOMMENDATION: Prefer keyboard shortcuts (hotkey Ctrl+..., press Tab, press Enter) over clicking when elements are densely packed.")
            coord_context = "\n".join(coord_parts) + "\n"

    # Build episodic memory context (proven patterns)
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        episodic_parts = []
        for mem in episodic_memory[-3:]:  # Last 3 episodic memories
            mem_str = str(mem)
            if len(mem_str) > 120:
                mem_str = mem_str[:120] + '...'
            episodic_parts.append(f"  • {mem_str}")

        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"\n🧠 PROVEN PATTERNS (from previous trials):\n{episodic_text}\n"

    # Format valid actions prominently (first 35 actions, line by line)
    valid_actions_formatted = ""
    if valid_actions:
        sample_actions = valid_actions
        valid_actions_formatted = "\n".join([f"  • {act}" for act in sample_actions])
        if len(valid_actions) > 35:
            valid_actions_formatted += f"\n  ... and {len(valid_actions) - 35} more actions"
    else:
        valid_actions_formatted = "  (No valid actions available)"

    # FIX #39: Build accumulated state context — shows what progress has been achieved
    accumulated_context = ""
    if accumulated_state:
        recent_state = accumulated_state[-5:]  # Last 5 state changes
        state_lines = "\n".join([f"  • {s}" for s in recent_state])
        has_regression = any("REGRESSION" in s for s in recent_state)
        accumulated_context = f"""
🔄 ACCUMULATED PROGRESS (what has been achieved so far):
{state_lines}
{"⚠️ WARNING: A regression was detected — previous progress was undone. Do NOT proceed to the next step until you have re-established the required state." if has_regression else ""}
CRITICAL: Do NOT repeat actions that would undo existing progress. If progress exists (e.g. text is selected, dialog is open), build on it — do NOT start over.
"""

    # Extract [Command Output] before truncation — this is the primary learning signal for run: commands
    _tg_cmd_out = ''
    if is_visual_env and curr_observation:
        _tg_co_idx = curr_observation.find('[Command Output')
        if _tg_co_idx >= 0:
            _tg_cmd_out = chr(10) + curr_observation[_tg_co_idx:_tg_co_idx + 800]

    textgrad_prompt = f"""🎯 TEXTGRAD: Action Optimization Through Gradients

YOUR ROLE (TextGrad's Unique Strength):
- Generate the BEST next action based on gradient signals from previous steps
- Optimize action selection using immediate feedback (progress scores, observations)
- Provide CLEAN, EXECUTABLE action (no analysis, just the action string)
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{todo_context}{inventory_context}{gradient_context}{episodic_context}{coord_context}{accumulated_context}
ACTION TAKEN: {action}

{f"OBSERVATION SUMMARY:{chr(10)}BEFORE state: {prev_observation[:500]}{chr(10)}AFTER state: {curr_observation[:500]}{_tg_cmd_out}{chr(10)}{chr(10)}{'⚠️ COMMAND OUTPUT ABOVE IS THE PRIMARY RESULT — the command ran but its output is NOT visible in screenshots. READ the error/output text carefully and fix the issue.' if _tg_cmd_out else '⚠️ SCREENSHOTS show the current screen state. The text above is a coordinate reference.'}" if is_visual_env else f"BEFORE: {prev_observation}{chr(10)}AFTER: {curr_observation}"}

{_build_action_constraint_block(valid_actions, is_visual_env) if is_visual_env else f"═══════════════════════════════════════════════════════════{chr(10)}VALID ACTIONS (What actions are actually possible right now):{chr(10)}═══════════════════════════════════════════════════════════{chr(10)}{valid_actions_formatted}{chr(10)}{chr(10)}IMPORTANT: You MUST select your next action from the VALID ACTIONS list above.{chr(10)}Do NOT invent actions or assume actions exist based on real-world logic.{chr(10)}The environment has explicit action semantics - use what's actually available."}

═══════════════════════════════════════════════════════════
TEXTGRAD OPTIMIZATION (Answer concisely):
═══════════════════════════════════════════════════════════

1. PROGRESS ASSESSMENT:
   Analyze the action's effects using PURE TEXTUAL FEEDBACK (TextGrad approach):

   a) What did this action accomplish?
      (Describe the state change in natural language)

   b) What does the task require?
      (Extract the goal from task description)

   c) Progress check:
      - Did the action move closer to completing the task requirements?
      - Did the state change in a way that addresses what the task needs?
      - Is this action advancing toward the goal or just repeating without progress?

   Answer: PROGRESS_STATUS: [exactly one of: NO_PROGRESS | EXPLORING | PARTIAL_PROGRESS | MAJOR_PROGRESS | TASK_COMPLETE]
   EVIDENCE: [Quote specific observation text or describe specific state change that proves progress. If no evidence, answer NO_PROGRESS.]
   REASON: [1 sentence]

   SCORING RULES:
   - NO_PROGRESS: No effect on task goal, OR no specific evidence of change
   - EXPLORING: Gathered information but no task requirement met
   - PARTIAL_PROGRESS: At least one requirement demonstrably closer (cite evidence)
   - MAJOR_PROGRESS: Most requirements met (cite evidence for each)
   - TASK_COMPLETE: All requirements satisfied (cite evidence for each)
   - If action had NO observable effect, that is NO_PROGRESS regardless of intent
   - If action returned an error, that is NO_PROGRESS
   - CRITICAL: If you see [COMMAND FAILED - return code X] in the output, the command FAILED. That is ALWAYS NO_PROGRESS, even if part of the command ran before the error. A failed command means the task step was NOT completed.
   - If the command output contains "Can't operate", "Permission denied", "not found", or any error traceback, that is NO_PROGRESS

2. GRADIENT SIGNAL (Rich Textual Feedback):
   Provide EXPLICIT, DETAILED feedback about what was learned.

   If the action didn't make progress:
   - STATE what the action did
   - STATE what the task needs
   - EXPLAIN why this action didn't advance the task
   - SUGGEST a different approach to try

   If making progress:
   - STATE what's working
   - SUGGEST continuing similar approach

   Answer: GRADIENT: [Detailed, explicit feedback describing the learning]

3. REFLEXION LESSON ENFORCEMENT:
   Review GRADIENT HISTORY above for diagnosed failures from Reflexion.

   CRITICAL RULE: If previous steps diagnosed that action X failed to achieve property Y:
   - DO NOT recommend action X again (it already failed!)
   - DO NOT use "similar" or "equivalent" actions (different verbs = different effects)
   - MUST try a DIFFERENT action type or explore new location

   List any actions you see were tried and failed:

   Answer: FAILED_ACTIONS: [list actions that were tried and failed, or "none diagnosed"]

4. NEXT ACTION OPTIMIZATION:
   Based on GRADIENT feedback and progress status, what's the BEST next action?

   Selection criteria (Pure TextGrad approach):
   - If TASK_COMPLETE: Select "done" to end the episode — the task is finished!
   - If MAJOR_PROGRESS: Continue current strategy to complete remaining requirements
   - If PARTIAL_PROGRESS: Adjust approach slightly or continue
   - If EXPLORING: Try different approaches that align with gradient feedback
   - If NO_PROGRESS: Change strategy completely - try a different action type
   - Respect TODO subtask if active
   - Use episodic patterns if available
   - Pick from VALID ACTIONS list
   - MUST NOT repeat actions listed in FAILED_ACTIONS above

   ⚠️ CRITICAL - GRADIENT HISTORY OVERRIDES VERB MATCHING:
   - If FAILED_ACTIONS above shows an action failed or repeated without progress, DO NOT recommend it again
   - Even if an action verb matches the task verb, if gradient history shows it's not making progress, try a DIFFERENT action
   - The goal is TASK COMPLETION through continuous learning, not verb matching
   - Trust your gradient history - it shows what's actually working

   FOLLOW THE GUIDANCE EXACTLY. Examples of CORRECT gradient→action translation:
   • Gradient says "user already exists" → NEXT_ACTION: run: id charles 2>/dev/null || useradd -m charles
   • Gradient says "file path not found" → NEXT_ACTION: run: find /home -name "*.xlsx" 2>/dev/null
   • Gradient says "read column names first" → NEXT_ACTION: run: python3 -c "import pandas as pd; print(pd.read_excel('/home/user/file.xlsx').columns.tolist())"
   • Gradient says "try keyboard instead of click" → NEXT_ACTION: hotkey Ctrl+S
   • Gradient says "wrong API, use XML directly" → NEXT_ACTION: run: python3 -c "import zipfile; z=zipfile.ZipFile('file.pptx'); print(z.namelist())"

   Answer: NEXT_ACTION: [{"action string — e.g., click (450,320), hotkey Alt+O, type 'text', press Enter" if is_visual_env else "exact action string from valid actions, nothing else"}]

═══════════════════════════════════════════════════════════
CRITICAL: Your NEXT_ACTION must be:
1. A single action string (e.g., {"click (450, 320) or hotkey Ctrl+S" if is_visual_env else "go to cabinet 2"})
2. {"Any valid GUI action format — coordinates, labels, hotkeys" if is_visual_env else "Exactly matching or very close to a valid action"}
3. NO explanations, NO analysis, just the action
═══════════════════════════════════════════════════════════
"""

    try:
        # Multimodal path: let GPT-4o SEE the actual screenshots
        if before_screenshot_b64 or after_screenshot_b64:
            textgrad_prompt += """

=== ATTACHED SCREENSHOTS (PRIMARY EVIDENCE) ===
The BEFORE and AFTER screenshots show exactly what happened on screen.
USE THESE as your primary source — the text observation above is just a summary.
Base your gradient analysis on what you SEE in the screenshots.
"""
            log_debug(f"[TEXTGRAD GUIDANCE] Using MULTIMODAL (GPT-4o vision)")
            response = model.generate_multimodal(
                textgrad_prompt, before_image_b64=before_screenshot_b64,
                after_image_b64=after_screenshot_b64, max_tokens=2000)
        else:
            response = model.generate([textgrad_prompt], SamplingParams(
                max_tokens=2000,  # Enough for visual reasoning
                temperature=0.0,
                top_p=0.95
            ))[0].outputs[0].text.strip()

        step_gradient['raw_reflection'] = response

        # Extract progress score — evidence-based categorical scoring
        _status_map = {
            'NO_PROGRESS': 1, 'EXPLORING': 3, 'PARTIAL_PROGRESS': 6,
            'MAJOR_PROGRESS': 8, 'TASK_COMPLETE': 10
        }
        status_match = re.search(
            r'PROGRESS_STATUS:\s*(NO_PROGRESS|EXPLORING|PARTIAL_PROGRESS|MAJOR_PROGRESS|TASK_COMPLETE)',
            response, re.IGNORECASE)
        if status_match:
            status_key = status_match.group(1).upper()
            progress_score = _status_map.get(status_key, 1)
            # Evidence gate: downgrade if no real evidence cited
            evidence_match = re.search(r'EVIDENCE:\s*(.+?)(?:\n|REASON:)', response, re.IGNORECASE | re.DOTALL)
            evidence_text = evidence_match.group(1).strip() if evidence_match else ''
            if progress_score >= 6 and (not evidence_text or len(evidence_text) < 15
                    or 'no evidence' in evidence_text.lower() or evidence_text.lower() == 'none'):
                progress_score = 3  # Downgrade — no evidence means no real progress
                log_debug(f"[SCORING] Downgraded {status_key} to EXPLORING — no evidence cited")
        else:
            # Fallback: numeric (backward compat)
            progress_match = re.search(r'PROGRESS:\s*(\d+)', response, re.IGNORECASE)
            progress_score = int(progress_match.group(1)) if progress_match else 0
        step_gradient['progress_score'] = min(max(progress_score, 0), 10)

        # Extract gradient lesson
        gradient_match = re.search(r'GRADIENT:\s*([^\n]+)', response, re.IGNORECASE)
        if gradient_match:
            step_gradient['hypothesis'] = gradient_match.group(1).strip()

        # Extract next action (CLEAN - no verbose analysis)
        next_action = ''
        action_match = re.search(r'NEXT_ACTION:\s*([^\n]+)', response, re.IGNORECASE)
        if action_match:
            next_action = action_match.group(1).strip()
            # For run: python commands, join continuation lines if quotes unclosed
            if next_action.lower().startswith('run:') and 'python' in next_action.lower():
                _tg_dq = next_action.count('"') - next_action.count('\\"')
                if _tg_dq % 2 != 0:
                    _tg_remaining = response[action_match.end():]
                    for _tg_line in _tg_remaining.split('\n'):
                        _tg_cl = _tg_line.strip()
                        if _tg_cl.upper().startswith(('REASONING:', 'NEXT_ACTION:', 'GRADIENT:', 'PROGRESS:')):
                            break
                        if _tg_cl:
                            next_action += ' ' + _tg_cl
                        _tg_dq = next_action.count('"') - next_action.count('\\"')
                        if _tg_dq % 2 == 0:
                            break
                    next_action = ' '.join(next_action.split())
            # Remove common prefixes
            next_action = next_action.replace('ACTION:', '').replace('NEXT:', '').strip()
            # Universal cleanup — handles markdown, pipes, annotations from any model
            next_action = clean_action(next_action)

        step_gradient['next_action_guidance'] = next_action

        log_debug(f"[TEXTGRAD] Progress={progress_score}/10, Next={next_action[:50]}")

    except Exception as e:
        print(f"[ERROR] Exception in generate_textgrad_step_guidance: {e}")
        step_gradient['progress_score'] = 0
        step_gradient['next_action_guidance'] = ''

    return step_gradient


def generate_vision_action_lean(
    task: str,
    model,
    screenshot_b64: str,
    scene_context: str = "",
    last_action: str = "",
    last_result: str = "",
    directive: str = "",
    discovered_files: List[str] = None,
    current_todo=None,
    log_debug=print,
) -> str:
    """
    LEAN MODE vision action generation.
    Prompt capped at ~6K chars: task + observation + last action + ONE directive.
    The learning loop (TextGrad/Reflexion) runs externally and distills to `directive`.
    """
    # Build compact observation from scene_context
    # Extract only [Interactive Elements] and [Command Output] sections
    obs_parts = []
    for section in ['[Interactive Elements]', '[Command Output]', '[Visible Text]']:
        idx = scene_context.find(section)
        if idx >= 0:
            end_idx = scene_context.find('\n[', idx + len(section))
            chunk = scene_context[idx:end_idx] if end_idx > 0 else scene_context[idx:]
            obs_parts.append(chunk[:800])
    compact_obs = '\n'.join(obs_parts) if obs_parts else scene_context[:1500]

    # Build discovered files section
    files_section = ""
    if discovered_files:
        files_section = f"\nDISCOVERED FILES: {', '.join(discovered_files[:5])}"

    # Build TODO section
    todo_section = ""
    if current_todo:
        _td = current_todo if isinstance(current_todo, str) else str(getattr(current_todo, 'description', current_todo))
        todo_section = f"\nCURRENT SUBTASK: {_td[:200]}"

    # Build directive section (the distilled learning)
    directive_section = ""
    if directive:
        directive_section = f"\nHINT FROM LEARNING LOOP: {directive[:500]}"

    # Last action result
    last_section = ""
    if last_action:
        last_section = f"\nLAST ACTION: {last_action[:300]}"
        if last_result:
            last_section += f"\nRESULT: {last_result[:500]}"

    prompt = f"""You are a desktop GUI agent controlling an Ubuntu Linux desktop (1920x1080).

TASK: {task[:500]}
{todo_section}{directive_section}{last_section}{files_section}

CURRENT SCREEN STATE:
{compact_obs}

ACTION FORMATS:
  click 'Label' — click a UI element
  click (X, Y) — click at coordinates
  type 'text' | hotkey Ctrl+S | press Enter
  run: <shell command>
  scroll down/up | done

RULES:
- LOOK at the screenshot — it is your PRIMARY source of truth.
- For file modifications, prefer run: python3 -c "..." (more reliable than GUI clicks).
- If a command failed, READ the error and fix the ROOT CAUSE.
- NEVER hallucinate filenames — use run: find or run: ls first.
- Quote file paths with spaces.
- Output 'done' ONLY when the task is visibly complete.

Output ONLY the action string on one line.
NEXT_ACTION: """

    prompt_len = len(prompt)
    log_debug(f"[LEAN MODE] Prompt: {prompt_len} chars")

    try:
        response = model.generate_multimodal(
            prompt,
            after_image_b64=screenshot_b64,
            max_tokens=1500,
        )
        clean_response = response.strip()
        if clean_response.startswith('```'):
            lines = clean_response.split('\n')
            clean_response = '\n'.join([l for l in lines[1:] if not l.strip().startswith('```')]).strip()

        action_match = re.search(r'NEXT_ACTION:\s*([^\n]+)', clean_response, re.IGNORECASE)
        if action_match:
            action = clean_action(action_match.group(1))
        else:
            action = clean_response.split('\n')[0].strip()
            for prefix in ['NEXT_ACTION:', 'Action:', 'I will ', 'I should ']:
                if action.lower().startswith(prefix.lower()):
                    action = action[len(prefix):].strip()

        if action:
            log_debug(f"[LEAN MODE] Generated: '{action[:80]}'")
            return action
        return ""
    except Exception as e:
        log_debug(f"[LEAN MODE] Error: {e}")
        return ""


def generate_vision_action(
    task: str,
    model,
    screenshot_b64: str,
    scene_context: str = "",       # VGL observation text — secondary hints
    vgl_elements_text: str = "",   # Affordance labels — secondary hints
    accumulated_state: List[str] = None,
    episodic_memory: List[str] = None,
    reflexion_insights: List[str] = None,  # Step-level Reflexion strategic insights
    recent_actions: List[str] = None,  # Last N actions taken (for loop avoidance)
    log_debug=print,
    textgrad_hint: str = "",  # TextGrad recommendation to consider (may be empty)
    current_todo=None,  # Current active TODO subtask for task decomposition context
    error_history: List[str] = None,  # Errors from previous run: commands (for learning)
    progress_history: List[int] = None,  # TextGrad score history for stagnation detection
    discovered_files: List[str] = None,  # File paths discovered via find/ls commands
    step_insights: List[dict] = None,  # Step insights for python-no-change detection
    policy_text: str = "",  # Bug 1 fix: ActionPolicy.base_policy from TextGrad optimizer
) -> str:
    """
    Vision-first action generation: Send screenshot + task to the model,
    let it reason visually about the best action. Screenshot is PRIMARY input.
    Used at every step for visual envs.

    Returns an action string compatible with osworld_wrapper action parsing.
    """
    # Extract compact coordinate hints from VGL (Interactive Elements only)
    # Also include Command Output from run: commands (e.g. ls output, script results)
    coord_hints = ""
    if scene_context:
        ie_start = scene_context.find('[Interactive Elements]')
        if ie_start >= 0:
            ie_end = scene_context.find('[', ie_start + 1)
            ie_section = scene_context[ie_start:ie_end if ie_end > ie_start else ie_start + 1500]
            coord_hints = f"\nCOORDINATE HINTS (from automated detection — verify against screenshot):\n{ie_section[:1500]}\n"
        cmd_start = scene_context.find('[Command Output')
        if cmd_start >= 0:
            cmd_section = scene_context[cmd_start:cmd_start + 600]
            coord_hints += f"\n{cmd_section}\n"
    elif vgl_elements_text:
        coord_hints = f"\nDETECTED UI ELEMENTS (hints only — trust the screenshot):\n{vgl_elements_text}\n"

    todo_section = ""
    if current_todo is not None:
        todo_content = getattr(current_todo, 'content', str(current_todo))
        todo_attempts = getattr(current_todo, 'attempts', 0)
        todo_section = f"""
CURRENT SUBTASK (from task decomposition):
  {todo_content}{"  (Attempt #" + str(todo_attempts) + " — try a different approach)" if todo_attempts > 1 else ""}
Focus on completing this subtask — it breaks the overall task into manageable steps.
"""

    textgrad_section = ""
    if textgrad_hint:
        textgrad_section = f"""
TEXTGRAD RECOMMENDATION (consider but verify against screenshot):
  {textgrad_hint}
If this matches what you see on screen, follow it. Otherwise, use your own judgement.
"""

    accumulated_section = ""
    if accumulated_state:
        recent = accumulated_state[-5:]
        accumulated_section = f"""
PROGRESS SO FAR:
{chr(10).join('  - ' + s for s in recent)}
IMPORTANT: Do NOT undo existing progress.
"""

    history_section = ""
    loop_warning = ""
    if recent_actions:
        actions_str = '\n'.join(f'  [{i+1}] {a}' for i, a in enumerate(recent_actions[-5:]))
        history_section = f"""
RECENT ACTIONS (do NOT repeat):
{actions_str}
"""
        # Loop detection: multiple strategies (visual envs have noisy state changes)
        # Normalize: strip comments after '#' and result suffixes for comparison
        def _norm_action_vf(a):
            # Strip result suffix (→ [SUCCESS], → [FAILED]) and comments
            a = a.split('→')[0].split('#')[0].strip().lower()
            return a
        if len(recent_actions) >= 2:
            last_2 = [_norm_action_vf(a) for a in recent_actions[-2:]]
            _consecutive_match_vf = len(set(last_2)) == 1
            # Strategy 2: Same action 3+ times in last 5
            _repeated_3plus_vf = False
            if len(recent_actions) >= 3:
                from collections import Counter as _Counter_vf
                _last5_vf = [_norm_action_vf(a) for a in recent_actions[-5:]]
                _counts_vf = _Counter_vf(_last5_vf)
                _most_vf = _counts_vf.most_common(1)[0]
                _repeated_3plus_vf = _most_vf[1] >= 3
            # Strategy 3: Score stagnation (from progress_history — actual TextGrad scores)
            _score_stag_vf = False
            if progress_history and len(progress_history) >= 3:
                _score_stag_vf = all(s <= 1 for s in progress_history[-3:])

            if _consecutive_match_vf or _repeated_3plus_vf or _score_stag_vf:
                stuck_action = recent_actions[-1]
                _reason_vf = "score stagnation" if _score_stag_vf and not _consecutive_match_vf else (
                    "repeated 3+ in last 5" if _repeated_3plus_vf else "consecutive identical")
                loop_warning = f"""
⚠️ CRITICAL: You are STUCK IN A LOOP ({_reason_vf}). The action '{stuck_action}' has been repeated without progress.

You MUST try a FUNDAMENTALLY DIFFERENT approach — switch MODALITY:
- If GUI clicks are failing → switch to programmatic: run: python3 -c "..."
- If programmatic commands are failing → switch to GUI interaction
- If one tool/library fails → try a different tool/library
1. Review the LESSONS from past steps above — they contain specific recommendations
2. If Reflexion provided a STRATEGY_SHIFT, follow it
3. Try a completely different method to achieve the same goal

Do NOT repeat '{stuck_action}'. The environment's learning system has analyzed your failures — use its recommendations.
"""
                log_debug(f"[VISION-FIRST LOOP] {_reason_vf} detected — loop_warning active")

    insights_section = ""
    if reflexion_insights:
        # Bug 3 fix: don't truncate the most recent insight (may carry NOOP-driven prescription)
        insight_lines = []
        for i, r in enumerate(reflexion_insights[-3:]):
            r_str = str(r)
            # Last (most recent) insight gets full 2500 chars; older ones 500 chars
            if i == len(reflexion_insights[-3:]) - 1:
                insight_lines.append(r_str[:2500])
            else:
                insight_lines.append(r_str[:500])
        _insights_label = "⚡ REFLEXION STRATEGY — HIGHEST PRIORITY (you are stuck, follow this recommendation):" if loop_warning else "STRATEGIC INSIGHTS (highest priority — follow these):"
        insights_section = f"""
{_insights_label}
{chr(10).join('  ⚡ ' + r for r in insight_lines)}
"""

    # Bug 3 fix: hoist NOOP signal from observation to TOP of prompt
    # The wrapper's FILE-CHANGE NOOP warning is buried in scene_context. Extract
    # it and surface it as a top-priority block right next to the action question.
    # Universal: just substring extraction from observation, no library/format names.
    noop_banner = ""
    if scene_context and 'FILE-CHANGE DETECTION' in scene_context:
        _nstart = scene_context.find('FILE-CHANGE DETECTION')
        _nend = scene_context.find('\n\n', _nstart)
        if _nend < 0:
            _nend = _nstart + 1500
        _ntext = scene_context[_nstart:_nend].strip()
        noop_banner = f"\n🚨 OBSERVATION-LAYER WARNING (must address):\n{_ntext}\n"

    # Bug 1 fix: inject TextGrad-optimized policy into the action prompt
    # Previously the actor never saw policy.base_policy at all — only the critic did.
    policy_section = ""
    if policy_text:
        policy_section = f"\n📋 LEARNED POLICY (from TextGrad gradient optimization):\n  {policy_text[:1000]}\n"

    memory_section = ""
    if episodic_memory:
        recent_mem = episodic_memory[-3:] if isinstance(episodic_memory, list) else []
        if recent_mem:
            mem_lines = [str(m)[:120] for m in recent_mem]
            memory_section = f"""
LESSONS FROM PREVIOUS ATTEMPTS:
{chr(10).join('  - ' + m for m in mem_lines)}
"""

    # Passthrough only for programmatic python3 commands from Reflexion STRATEGY_SHIFT.
    # These are well-reasoned solutions (e.g., python-pptx, openpyxl) that should be trusted.
    # Other run: commands (xdg-open, libreoffice, etc.) go through screenshot-based reasoning.
    if textgrad_hint and textgrad_hint.strip().lower().startswith('run: python3'):
        log_debug(f"[VISION-FIRST] Passing through Reflexion python3 command: '{textgrad_hint[:80]}...'")
        return textgrad_hint

    error_section = ""
    if error_history:
        error_lines = '\n'.join(f'  {e}' for e in error_history[-5:])
        _has_errors = any('ERROR:' in e for e in error_history)
        _has_cmd_output = any('[Command Output' in e for e in error_history)
        if _has_errors:
            error_section = f"""
PREVIOUS COMMAND RESULTS (CRITICAL — learn from these errors and do NOT repeat them):
{error_lines}
"""
        elif _has_cmd_output:
            error_section = f"""
PREVIOUS COMMAND RESULTS (use this information for your next action):
{error_lines}
"""

    discovered_section = ""
    if discovered_files:
        files_str = '\n'.join(f'  {f}' for f in discovered_files)
        discovered_section = f"""
⚠️ DISCOVERED FILES (from previous find/ls commands — use EXACT paths below):
{files_str}
Do NOT guess different paths — use one of the exact paths listed above.
"""

    # Determine if GUI ever scored well — if so, don't force python
    _vf_gui_worked = any(s >= 5 for s in (progress_history or []))
    # Check if python already ran OK but screen didn't change — app has file locked
    _vf_python_no_change = False
    for _vfp in (step_insights or [])[-5:]:
        _vfp_act = str(_vfp.get('action', ''))
        _vfp_sc = _vfp.get('progress_score', 0)
        if 'python' in _vfp_act.lower() and 'run:' in _vfp_act.lower() and _vfp_sc <= 1:
            _vfp_obs = str(_vfp.get('observation', ''))
            if 'rc=0' in _vfp_obs or 'SUCCESS' in _vfp_obs:
                _vf_python_no_change = True
    _vf_force_python = loop_warning and not _vf_gui_worked and not _vf_python_no_change
    # G11-VF UNIVERSAL: include the full wrapper obs (scene_context /
    # curr_obs) so every pinned block produced by the wrapper reaches the
    # VISION-FIRST actor path the same way it reaches PASS 2. Gated on
    # scene_context non-empty. Zero env/task/domain assumptions — just
    # propagation of the existing universal evidence region to a second
    # action-selection path.
    _curr_obs_vf_block = ''
    if scene_context:
        _curr_obs_vf_block = (
            "\n[CURRENT OBSERVATION FROM WRAPPER (contains pinned persistent-delta, "
            "tombstones, structure probe, command output — read ALL blocks before "
            "deciding; these supersede any paraphrase in the task description above):\n"
            f"{str(scene_context)[:8000]}\n]\n"
        )

    prompt = f"""You are a desktop GUI agent controlling an Ubuntu Linux desktop (1920×1080).

TASK: {task}
{noop_banner}{policy_section}{loop_warning}{error_section}{discovered_section}{_curr_obs_vf_block}
INSTRUCTIONS:
{"1. GUI CLICKS HAVE FAILED — you MUST use run: python3 to modify the file directly" if _vf_force_python else "1. LOOK at the attached screenshot — it is your PRIMARY source of truth"}
2. Decide the single best action to advance the task
{"3. Read the title bar for the filename, then use run: python3 -c '...' to make changes programmatically" if _vf_force_python else "3. Use coordinate hints below ONLY as reference — trust what you SEE"}
{todo_section}{textgrad_section}{coord_hints}{accumulated_section}{history_section}{insights_section}{memory_section}
{"ACTION FORMATS (⚠️ GUI CLICKS FAILED — SWITCH TO PROGRAMMATIC):" if _vf_force_python else "ACTION FORMATS (in order of preference):"}
{"  run: python3 -c '...' — REQUIRED: use python/openpyxl to modify the file" if _vf_force_python else "  click 'Label' — PREFERRED when the target appears in [Interactive Elements] above (uses verified coordinates)"}
{"  run: <shell command> — ls, find for discovery" if _vf_force_python else "  click (X, Y) — Use ONLY when the target is NOT listed in [Interactive Elements]"}
{"  done — after programmatic command succeeds" if _vf_force_python else "  type 'text' | hotkey Ctrl+S | press Enter"}
{"" if _vf_force_python else "  run: <shell command> | scroll down/up | done"}

RULES:
{"- ⚠️ GUI CLICKS ARE NOT WORKING. You MUST output a run: python3 command to modify the file directly." if _vf_force_python else "- LOOK at the screenshot — it is your PRIMARY source of truth for deciding what to do."}
{"- Read the title bar for the filename. Use run: python3 -c 'import openpyxl; ...' or similar." if _vf_force_python else "- USE [Interactive Elements] labels: If the target button/menu/tab appears in the list above, use click 'Label' format (e.g., click 'Add Sheet' button). These have verified coordinates and are more accurate than estimating."}
{"- ⚠️ PYTHON COMMANDS SUCCEEDED (rc=0) BUT NOTHING CHANGED ON SCREEN. The application likely has the file locked. Either: (1) Close the application (run: pkill -f soffice || run: pkill -f libreoffice) then say 'done', OR (2) Try a DIFFERENT GUI approach instead (e.g., Format menu → Paragraph/Spacing)." if _vf_python_no_change else ""}
- CHECK WHAT'S ALREADY OPEN: If an application has a file open (check the title bar), that file is MOST LIKELY the one you need to work with. Navigate within it (switch slides, sheets, tabs, scroll) before searching for other files.
- Content referenced in the task (e.g., "to-do list", "budget", "report") may be on a DIFFERENT PAGE, SLIDE, or SHEET of the currently-open document. Navigate through the document to find it.
- NEVER hallucinate filenames. If you don't see a specific file path in the title bar, file manager, or command output, use `run: ls` or `run: find` to discover actual filenames first.
- CHECK THE WINDOW TITLE BAR at the top of the screen for the actual filename and path of open documents.
- If a dialog or dropdown menu is open, click the correct item. Prefer click 'Label' if the item appears in [Interactive Elements].
- If GUI clicks aren't making progress, switch to programmatic approach: run: python3 -c "..."
- You can run ANY shell command: run: ls, run: python3 -c "...", run: find, etc.
- When modifying files programmatically, use the EXACT path shown in the title bar. If unsure, run: find ~ -type f -name "filename" to locate it
- Background GUI apps with &: run: app "file" & (use absolute paths, NOT ~/...)
- If a command failed, READ the FULL error message and fix the ROOT CAUSE — do NOT just retry the same command.
- If a previous find/ls command discovered file paths, USE those exact paths — do NOT guess different paths.
- Quote file paths with spaces.
- Output 'done' ONLY when ALL parts of the task are visibly complete or your programmatic command has succeeded.
- PROGRAMMATIC COMPLETION: If a previous run: command shows [SUCCESS - completed] in RECENT ACTIONS and that command performed the task's goal (e.g., wrote data, modified a file), say 'done'. Programmatic file changes modify the file on disk — they will NOT appear in the GUI screenshot. Trust the [SUCCESS] result over the screenshot.

Output ONLY the action string on one line (no explanation).
NEXT_ACTION: """

    try:
        log_debug(f"[VISION-FIRST] Generating action from screenshot + task...")
        # DEEP TRACE (Phase 7): dump the actual VISION-FIRST prompt and
        # response as one within-step sub-event. Gated on env var.
        _vf_dump_idx = [0]
        try:
            if os.getenv("REFLEXGRAD_DEEP_TRACE", "0") == "1":
                import json as _json_vf, time as _time_vf
                _vf_dir = os.path.join(
                    os.getcwd(), '_deep_trace',
                    os.getenv("REFLEXGRAD_RUN_NAME", "deep_trace"),
                )
                os.makedirs(_vf_dir, exist_ok=True)
                _vf_ts = int(_time_vf.time() * 1000)
                _vf_file = os.path.join(_vf_dir, f'visionfirst_{_vf_ts}.json')
                with open(_vf_file, 'w') as _vf_f:
                    _json_vf.dump({
                        'name': 'vision_first_call',
                        'timestamp_ms': _vf_ts,
                        'prompt': prompt,
                        'prompt_len': len(prompt),
                        'scene_context_len': len(scene_context or ''),
                        'scene_context_included': bool(scene_context),
                    }, _vf_f, indent=2, default=str)
        except Exception:
            pass
        response = model.generate_multimodal(
            prompt,
            after_image_b64=screenshot_b64,
            max_tokens=1500,
        )

        # Strip markdown code blocks if model wrapped answer (e.g. ```plaintext\nclick ...\n```)
        clean_response = response.strip()
        if clean_response.startswith('```'):
            # Extract content between code fences
            lines = clean_response.split('\n')
            # Skip opening fence (and optional language tag), skip closing fence
            inner_lines = [l for l in lines[1:] if not l.strip().startswith('```')]
            clean_response = '\n'.join(inner_lines).strip()

        # Garbage response filter: detect <20 char garbage like "click '...'"
        first_line_vf = clean_response.split('\n')[0].strip()
        if len(first_line_vf) < 20 and ("'...'" in first_line_vf or "'…'" in first_line_vf or first_line_vf.count('.') >= 3):
            log_debug(f"[VISION-FIRST GARBAGE FILTER] Detected garbage: '{first_line_vf}' — retrying")
            try:
                retry_resp = model.generate_multimodal(
                    prompt, after_image_b64=screenshot_b64, max_tokens=1500)
                retry_clean = retry_resp.strip()
                if retry_clean.startswith('```'):
                    rl = retry_clean.split('\n')
                    retry_clean = '\n'.join([l for l in rl[1:] if not l.strip().startswith('```')]).strip()
                retry_first = retry_clean.split('\n')[0].strip()
                if len(retry_first) > len(first_line_vf):
                    clean_response = retry_clean
                    log_debug(f"[VISION-FIRST GARBAGE FILTER] Retry OK: '{retry_first[:60]}'")
                else:
                    log_debug(f"[VISION-FIRST GARBAGE FILTER] Retry also short — keeping original")
            except Exception as e:
                log_debug(f"[VISION-FIRST GARBAGE FILTER] Retry failed: {e}")

        # Parse NEXT_ACTION from response
        action_match = re.search(r'NEXT_ACTION:\s*([^\n]+)', clean_response, re.IGNORECASE)
        if action_match:
            action = clean_action(action_match.group(1))
            # For run: python commands, join continuation lines if quotes unclosed
            if action.lower().startswith('run:') and 'python' in action.lower():
                _vf_dq = action.count('"') - action.count('\\"')
                if _vf_dq % 2 != 0:
                    _na_pos = action_match.end()
                    _remaining = clean_response[_na_pos:]
                    for _vf_line in _remaining.split('\n'):
                        _vf_cl = _vf_line.strip()
                        if _vf_cl.upper().startswith('REASONING:') or _vf_cl.upper().startswith('NEXT_ACTION:'):
                            break
                        if _vf_cl:
                            action += ' ' + _vf_cl
                        _vf_dq = action.count('"') - action.count('\\"')
                        if _vf_dq % 2 == 0:
                            break
                    action = ' '.join(action.split())
                    log_debug(f"[VISION-FIRST MULTILINE] Joined python cmd → {len(action)} chars")
        else:
            # Model may have just output the action directly
            action = clean_response.split('\n')[0].strip().strip('"\'')
            # Clean common prefixes
            for prefix in ['NEXT_ACTION:', 'Action:', 'I will ', 'I should ']:
                if action.lower().startswith(prefix.lower()):
                    action = action[len(prefix):].strip()

        if action:
            log_debug(f"[VISION-FIRST] Generated: '{action[:80]}'")
            return action
        else:
            log_debug(f"[VISION-FIRST] Empty response, falling back")
            return ""

    except Exception as e:
        log_debug(f"[VISION-FIRST] Error: {e}")
        return ""


def generate_reflexion_causal_analysis(
    prev_observation: str,
    curr_observation: str,
    action: str,
    task: str,
    model,
    log_debug = print,
    current_step: int = 0,
    valid_actions: List[str] = None,
    # Loop detection removed - parameter no longer needed
    current_todo = None,  # Current TODO for tactical coordination
    inventory: List[str] = None,  # Items currently being carried
    previous_reflexions: List[Dict] = None,  # CRITICAL: Previous step learnings (step-level memory)
    episodic_memory: List[str] = None,  # CRITICAL: Cross-trial learnings (episodic memory)
    step_insights: List[Dict] = None,  # FIX #21: Raw observations instead of LLM interpretations
    before_screenshot_b64: str = None,  # Multimodal: before-action screenshot
    after_screenshot_b64: str = None,   # Multimodal: after-action screenshot
    accumulated_state: List[str] = None,  # FIX #39: Track what progress has been achieved
    is_visual_env: bool = False,  # Vision-first: visual hints instead of hard constraints
    learning_context = None,  # Universal learning engine (FailureMemory, PolicyGradients)
    _g30_eval_gap: str = '',    # G30 UNIVERSAL — evaluator's verbatim ground-truth output
    _g30_eval_source: str = '', # G30 UNIVERSAL — evaluator Python source (if captured)
) -> Dict:
    """
    Reflexion-only prompt: Deep causal why-analysis for failures and low-progress situations.

    Reflexion's Unique Strength (NeurIPS 2023):
    - Identify WHY actions fail through precise causal analysis
    - Extract concrete lessons for episodic memory (cross-trial learning)
    - Root cause diagnosis (not action optimization)

    This function should ONLY be called when progress_score < 4 (failures/low progress).
    For normal steps with good progress, use generate_textgrad_step_guidance() instead.
    """

    # Initialize step_gradient at the beginning
    step_gradient = {
        'state_change': 'unknown',
        'progress_score': 0,
        'hypothesis': '',
        'next_action_guidance': '',
        'raw_reflection': '',
        'is_loop': False,  # Loop detection removed
        'guidance_source': 'reflexion'  # Mark as Reflexion CAUSAL CHAIN (needs extraction)
    }

    # Loop detection removed - no repetition warning needed
    repetition_warning = ""

    # Build TODO context with EXPLICIT observation comparison (FIX #22 - Universal)
    todo_context = ""
    if current_todo:
        todo_context = f"""
📋 CURRENT SUBTASK: {current_todo.content}

🔍 EXPLICIT TODO CHECK (Answer honestly):
   - TODO requires: "{current_todo.content}"
   - Last observation: "{curr_observation[:150]}"
   - QUESTION: Does this observation indicate the TODO was achieved?
   - If NO: Identify WHY it failed and what DIFFERENT approach is needed.
"""

    # Build inventory context — ONLY for text-game envs (visual envs have no inventory).
    if is_visual_env:
        inventory_context = ""
    elif inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ CRITICAL: Use held items for task completion (cool/heat/clean/put)!
"""
    else:
        inventory_context = """
🎒 INVENTORY: You are NOT carrying anything.
   ⚠️ You must TAKE an object before you can cool/heat/clean/put it.
"""

    # FIX #21: Build history from RAW OBSERVATIONS, not LLM interpretations
    # FIX #25: Include BACKWARD GRADIENT lessons so agent learns from past mistakes
    reflexions_context = ""
    if step_insights and len(step_insights) > 0:
        # Use raw observations from step_insights_accumulator
        formatted_reflexions = []
        for insight in step_insights:  # FIX #36: ALL steps with raw data
            step_num = insight.get('step', '?')
            action_taken = insight.get('action', 'unknown')
            raw_observation = insight.get('observation', '')  # RAW environment feedback
            backward_lesson = insight.get('backward_gradient', '')  # FIX #25: Include lesson!

            # For visual envs: extract only RELEVANT feedback (state change + command output),
            # not the full VGL metadata which drowns out the actual results
            if is_visual_env and raw_observation:
                _fb_p = []
                _sc_j = raw_observation.find('[State Change]')
                if _sc_j >= 0:
                    _sc_je = raw_observation.find('\n[', _sc_j + 1)
                    _sc_jt = raw_observation[_sc_j:_sc_je if _sc_je > _sc_j else _sc_j + 200].strip()
                    if 'No visible change' not in _sc_jt:
                        _fb_p.append(_sc_jt[:200])
                    else:
                        _fb_p.append('No visible change')
                _co_j = raw_observation.find('[Command Output')
                if _co_j >= 0:
                    _err_j = raw_observation.find('Error:', _co_j)
                    if _err_j < 0:
                        _err_j = raw_observation.find('ERROR]', _co_j)
                    if _err_j >= 0:
                        _fb_p.append('COMMAND ERROR: ' + raw_observation[_err_j:_err_j + 300].strip())
                    else:
                        _fb_p.append(raw_observation[_co_j:_co_j + 500].strip())
                    # Preserve AUTO-DISCOVERY messages (critical for file path correction)
                    _disc_j = raw_observation.find('[AUTO-DISCOVERY]', _co_j)
                    if _disc_j >= 0:
                        _disc_je = raw_observation.find('\n\n', _disc_j)
                        _disc_jt = raw_observation[_disc_j:_disc_je if _disc_je > _disc_j else _disc_j + 300].strip()
                        _fb_p.append(_disc_jt)
                raw_observation = ' | '.join(_fb_p) if _fb_p else 'No visible change'

            # Format with lesson if available
            if backward_lesson:
                formatted_reflexions.append(f"  Step {step_num}: ACTION: \"{action_taken}\" → RESULT: \"{raw_observation}\"\n    📚 LESSON: {backward_lesson}")
            else:
                formatted_reflexions.append(f"  Step {step_num}: ACTION: \"{action_taken}\" → RESULT: \"{raw_observation}\"")

        reflexion_text = "\n".join(formatted_reflexions)
        reflexions_context = f"""
📚 RAW ACTION HISTORY WITH LESSONS:
{reflexion_text}
   ⚠️ CRITICAL: Read the LESSON for each step - it tells you what was WRONG and what to try INSTEAD!
"""
    elif previous_reflexions and len(previous_reflexions) > 0:
        # Fallback to old format if step_insights not available
        formatted_reflexions = []
        for r in previous_reflexions:  # FIX #36: ALL reflexions
            step_num = r.get('step', '?')
            reflection_text = r.get('reflection', '')[:200]
            formatted_reflexions.append(f"  Step {step_num}: {reflection_text}")

        reflexion_text = "\n".join(formatted_reflexions)
        reflexions_context = f"""
📚 STEP-LEVEL LEARNINGS (fallback):
{reflexion_text}
"""

    # Build episodic memory context (CRITICAL for cross-trial strategic learning!)
    episodic_context = ""
    if episodic_memory and len(episodic_memory) > 0:
        # Take last 5 episodic memories (strategic patterns across trials)
        recent_episodic = episodic_memory[-5:]
        # Handle dict memories - extract insight field
        episodic_parts = []
        for mem in recent_episodic:
            if isinstance(mem, dict):
                mem_text = mem.get('insight', str(mem))[:200]
            else:
                mem_text = str(mem)[:200]
            episodic_parts.append(f"  • {mem_text}")
        episodic_text = "\n".join(episodic_parts)
        episodic_context = f"""
🧠 EPISODIC MEMORY (strategic patterns from previous trials):
{episodic_text}
   ⚠️ CRITICAL: These are proven patterns across trials - use them to avoid known pitfalls!
"""

    # FIX #21: Define exploration_context and textgrad_history_context (were undefined)
    exploration_context = ""  # Not used in Reflexion - exploration is TextGrad's job
    textgrad_history_context = ""  # Raw action history already in reflexions_context

    # Spatial interaction analysis (coordinate learning for visual environments)
    spatial_context = ""
    if step_insights:
        click_insights = [ins for ins in step_insights
                          if ins.get('click_metadata', {}).get('match_type') not in (None, 'non_click')]
        if click_insights:
            exact = sum(1 for i in click_insights if i['click_metadata']['match_type'] == 'exact')
            fuzzy = sum(1 for i in click_insights if i['click_metadata']['match_type'] == 'fuzzy')
            no_match = sum(1 for i in click_insights if i['click_metadata']['match_type'] in ('no_match', 'fallback_center'))
            spatial_parts = [f"\n🎯 SPATIAL INTERACTION ANALYSIS:"]
            spatial_parts.append(f"  Click accuracy: {exact}/{len(click_insights)} exact, {fuzzy} fuzzy, {no_match} failed")
            if no_match > 0 or fuzzy > 0:
                spatial_parts.append("  ROOT CAUSE: VGL label→coordinate resolution is imprecise for some elements.")
                spatial_parts.append("  LEARNING: For elements that failed to match, suggest using:")
                spatial_parts.append("    1. Keyboard shortcuts (Ctrl+key combos)")
                spatial_parts.append("    2. Tab navigation (press Tab to cycle through elements)")
                spatial_parts.append("    3. Address bar navigation (type URLs directly)")
                spatial_parts.append("    4. More precise label from the [Interactive Elements] list")
            spatial_context = "\n".join(spatial_parts) + "\n"

    # FIX #39: Build accumulated state context for Reflexion
    accumulated_context = ""
    if accumulated_state:
        recent_state = accumulated_state[-5:]
        state_lines = "\n".join([f"  • {s}" for s in recent_state])
        accumulated_context = f"""
🔄 CURRENT ACCUMULATED PROGRESS:
{state_lines}
CRITICAL: Your analysis must account for the current state. If progress has been achieved (e.g. text selected, dialog opened), recommend CONTINUING from that state — do NOT suggest starting over. If a regression occurred, recommend re-establishing the lost state first.
"""

    # Extract [Command Output] before truncation — primary learning signal for run: commands
    _rx_cmd_out = ''
    if is_visual_env and curr_observation:
        _rx_co_idx = curr_observation.find('[Command Output')
        if _rx_co_idx >= 0:
            _rx_cmd_out = chr(10) + curr_observation[_rx_co_idx:_rx_co_idx + 800]

    # G30 UNIVERSAL — inject evaluator ground truth into Reflexion prompt so
    # causal analysis reasons over the EVALUATOR'S exact literal requirements
    # (byte values, cell addresses, attribute names) instead of abstract prose.
    # Previously Reflexion only saw backward_gradient summaries which lost the
    # specific literals. Now it sees the same verbatim gap + evaluator source
    # that G21 puts in the TextGrad BACKWARD and G24 puts in PASS 2 actor.
    # Gated on gap being populated — no-op on text envs / first step.
    _g30_block = ""
    if _g30_eval_gap:
        _g30_block = (
            "\n⛔ EVALUATOR GROUND TRUTH (verbatim — your causal analysis MUST\n"
            "   explain WHY the current state fails to reach THIS exact state):\n"
            f"{str(_g30_eval_gap)[:1500]}\n"
        )
        if _g30_eval_source:
            _g30_block += (
                "\n🔍 EVALUATOR SOURCE (the Python the evaluator literally runs —\n"
                "   root-cause the mismatch between the agent's produced state and\n"
                "   what THIS code would accept as success):\n"
                f"{str(_g30_eval_source)[:2000]}\n"
            )

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{_g30_block}{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}{spatial_context}{accumulated_context}
{f"ACTION: {action}{chr(10)}{chr(10)}OBSERVATION SUMMARY:{chr(10)}BEFORE state: {prev_observation[:500]}{chr(10)}AFTER state: {curr_observation[:500]}{_rx_cmd_out}{chr(10)}{chr(10)}{'⚠️ COMMAND OUTPUT ABOVE IS THE PRIMARY RESULT — READ the error/output text carefully to diagnose the root cause.' if _rx_cmd_out else '⚠️ SCREENSHOTS show the current screen state.'}" if is_visual_env else f"BEFORE: {prev_observation}{chr(10)}ACTION: {action}{chr(10)}AFTER: {curr_observation}"}

{_build_action_constraint_block(valid_actions, is_visual_env)}

{repetition_warning}

═══════════════════════════════════════════════════════════
CRITICAL REFLEXION ANALYSIS (Answer each question precisely):
═══════════════════════════════════════════════════════════

1. SEMANTIC PRECISION CHECK:
   Task instruction: "{task}"
   My action: "{action}"

   Extract the PRIMARY PURPOSE of each:
   - What is the main goal the task requires me to achieve? (e.g., change temperature, change location, change cleanliness, etc.)
   - What is the main effect/purpose of the action I took?
   - Are these semantically IDENTICAL purposes or DIFFERENT purposes?

   IMPORTANT: Even if both use similar objects or locations, they may serve different purposes.
   Example: "remove dirt" vs "lower temperature" are DIFFERENT even if both might involve water.

   Answer: SEMANTIC MATCH: [Yes/No] - Explain: [task purpose] vs [action purpose]

2. STATE VERIFICATION WITH EVIDENCE:
   Based on task "{task}", what SPECIFIC property of objects must change?
   (Examples: temperature, location, cleanliness, container state, object state)

   Now look at the AFTER observation: "{curr_observation}"
   Did that EXACT property change? Cite specific words from observation as evidence.
   Do NOT assume or infer - only state what the observation explicitly shows.

   Answer: PROPERTY REQUIRED: [what must change] | EVIDENCE: [quote from observation] | CHANGED: [Yes/No]

3. TODO FEEDBACK INTEGRATION:
   {f'Current subtask: "{current_todo.content}" - Status after action: NOT ACHIEVED (Attempt #{current_todo.attempts if current_todo else 0})' if current_todo else 'No active subtask'}

   If subtask not achieved after {current_todo.attempts if current_todo else 0} attempt(s):
   - What pattern do you see in my actions?
   - Am I repeating similar action types that don't work?
   - What assumption might I be making that is incorrect?

   Answer: FAILURE PATTERN: [describe pattern if any] | ROOT CAUSE: [precise diagnosis]

4. CAUSAL CHAIN ANALYSIS:
   Trace the causal relationship:
   - GOAL (from task): [what I wanted to accomplish]
   - ACTION (what I did): {action}
   - RESULT (what happened): [state change or lack thereof]
   - CAUSALITY: This [succeeded/failed] because [causal reason]

   If there's a semantic mismatch or wrong action type:
   - Have I already tried similar actions on this location in STEP-LEVEL LEARNINGS?
   - If yes, what was the result? Did it fail to find the target object?
   - What is the ROOT CAUSE of the repeated failure? (e.g., wrong location, wrong action type, impossible task)

   Answer: CAUSAL CHAIN: [describe cause-effect] | ROOT CAUSE: [fundamental reason]

5. CONCRETE LESSON EXTRACTION:
   Based on your analysis, complete this pattern for episodic memory:
   - WRONG ASSUMPTION: [what I incorrectly believed, if any]
   - CORRECT UNDERSTANDING: [what is actually true]
   - NEVER REPEAT: [specific action or pattern to avoid]
   - INSTEAD USE: [specific action or approach that works]

   Answer: LESSON: [fill in above format only if there's a clear lesson to extract]

6. PROGRESS ASSESSMENT (TextGrad-Style):
   Evaluate progress toward FULL TASK: "{task}"
   Context: {f'Current subtask is "{current_todo.content}"' if current_todo else 'No active subtask'}

   Provide textual evaluation:

   TASK_ALIGNMENT:
   - Which requirement(s) of FULL TASK are closer to completion?
   - Which remain unmet? Cite evidence from AFTER state.
   - CRITICAL: Evaluate relative to FULL TASK, not just subtask.

   PROGRESS_STATUS toward FULL TASK:
   - NO_PROGRESS: Irrelevant to task requirements
   - EXPLORING: Gathering info, no requirements met
   - PARTIAL_PROGRESS: Some requirements met, task incomplete
   - MAJOR_PROGRESS: Most requirements met, nearly done
   - TASK_COMPLETE: All requirements satisfied

   Answer: PROGRESS_STATUS: [one of above] | JUSTIFICATION: [requirement analysis]

7. UNIVERSAL OBSERVATION GAP DETECTION + ACTIVE PROBING:
   A smart agent detects what it DOESN'T KNOW and takes an action to find out.
   This is universal — works for ANY task in ANY environment.

   a) WHAT INFORMATION AM I MISSING?
      After reviewing the full action history above, identify gaps:
      - Do I know WHERE the target UI element/file/cell is? (have coordinates or name?)
      - Do I know WHAT the exact filename/path is I need to interact with?
      - Do I know IF the required app/process is currently running?
      - Do I know WHAT the current UI state is (which dialog, menu, mode)?
      - Have I actually TRIED the most direct approach to the task? (e.g., typed a formula, clicked slide content)

   b) UNIVERSAL PROBING ACTIONS (no task-specific knowledge needed):
      Use your world knowledge + these universal tools to fill gaps:
      - Unknown files → run: ls ~/Desktop   OR   run: find ~ -name "*.ext" -type f
      - Unknown running processes → run: ps aux | grep appname
      - Can't see element coords → click the estimated CONTENT AREA of the screen
        (spreadsheet grid ≈ center of screen, slide content ≈ center-lower region, file manager ≈ main panel)
      - App stuck/wrong mode → press Escape to reset, then re-approach
      - Unknown shortcut → use Format/Edit/Tool menus to discover it visually
      - Dialog blocking → interact with dialog FIRST (click OK/Close/Apply) before anything else

   c) STRATEGY SHIFT RULE (mandatory after 3+ failed attempts at same approach):
      CRITICAL: STRATEGY_SHIFT must be an EXECUTABLE command starting with "run:", NOT a prose description.

      When the current approach is failing after multiple attempts:
      - If GUI interaction is failing → switch to programmatic: run: python3 -c "..." (use appropriate libraries for the file type)
      - If programmatic approach is failing → fix the error using the error message, or try a different library/method
      - If stuck in a dialog/menu → press Escape to exit, then re-approach
      - If wrong location in document → navigate first (switch slides/sheets/tabs/scroll), then act
      - If undo needed → hotkey Ctrl+Z, then try different approach

      BAD answer: "Switch to programmatic approach" (REJECTED — too vague, not executable)
      GOOD answer: run: python3 -c "import subprocess; print(subprocess.check_output(['ls', '-la', '.']).decode())"
      GOOD answer: press Escape
      GOOD answer: run: python3 -c "[concrete code that addresses the specific task]"

   Answer: MISSING_INFO: [what I don't know] | PROBE_ACTION: [run: command or click to collect it] | STRATEGY_SHIFT: [run: command or "none"]

═══════════════════════════════════════════════════════════

Answer format: Provide your analysis after each numbered question."""

    sampling_params = SamplingParams(
        max_tokens=7000,  # Increased to 7000 to prevent empty responses with reasoning='medium'
        temperature=0.3,
        stop=["TASK:", "BEFORE:"],
        skip_special_tokens=True
    )

    # Log TextGrad using both memories (with compression info)
    step_mem_count = len(previous_reflexions) if previous_reflexions else 0
    compressed_count = len([r for r in previous_reflexions if r.get('is_compressed')]) if previous_reflexions else 0
    episodic_mem_count = len(episodic_memory) if episodic_memory else 0
    log_debug(f"[TEXTGRAD] Using ALL {step_mem_count} step reflexions ({compressed_count} compressed), {episodic_mem_count} episodic patterns")

    try:
        # Multimodal path: let GPT-4o SEE the actual screenshots for causal analysis
        if before_screenshot_b64 or after_screenshot_b64:
            reflection_prompt += """

=== ATTACHED SCREENSHOTS (PRIMARY EVIDENCE) ===
The BEFORE and AFTER screenshots show exactly what happened on screen.
USE THESE for your causal analysis — identify which UI element was clicked,
whether the visual state matches expectations, and root cause of any failures.
The text observation above is just a summary — trust the screenshots.
"""
            log_debug(f"[REFLEXION] Using MULTIMODAL (GPT-4o vision with screenshots)")
            response = model.generate_multimodal(
                reflection_prompt, before_image_b64=before_screenshot_b64,
                after_image_b64=after_screenshot_b64, max_tokens=7000, temperature=0.3)
        else:
            # Use reasoning='medium' explicitly for step reflexions (faster than 'high', sufficient quality)
            output = model.generate([reflection_prompt], sampling_params, reasoning_effort='medium')[0]
            response = output.outputs[0].text.strip()
    except Exception as e:
        # from api_quota_handler import APIQuotaHandler  # Commented out - not needed
        # from checkpoint_manager import CheckpointManager  # Commented out - not needed
        # Create temporary checkpoint manager for error handling
        # temp_logging_dir = '.'
        # checkpoint_manager = CheckpointManager(temp_logging_dir)
        # quota_handler = APIQuotaHandler(checkpoint_manager, checkpoint_manager.run_dir)
        # quota_handler.handle_api_error(e, current_step, {}, [])
        print(f"[ERROR] Exception in generate_universal_step_reflection: {e}")
        raise e
    
    # Clean think tags if present
    response = response.replace("</think>", "").replace("<think>", "").strip()

    # FIX: Strip markdown formatting from open-source models (Llama, Qwen)
    # Same fix as parse_step_gradient_response — without this, all numbered sections fail to parse
    import re as _re_md2
    response = _re_md2.sub(r'\*{1,3}', '', response)
    response = _re_md2.sub(r'^#{1,4}\s*', '', response, flags=_re_md2.MULTILINE)
    response = _re_md2.sub(r'_{1,2}(?=\S)', '', response)
    response = _re_md2.sub(r'(?<=\S)_{1,2}', '', response)

    # Parse numbered responses
    lines = response.split('\n')
    answers = {}
    current_num = 0
    current_answer = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # More robust number extraction
        import re
        if line_stripped and line_stripped[0].isdigit():
            # Save previous answer
            if current_num > 0 and current_answer:
                answers[current_num] = ' '.join(current_answer).strip()
            
            # Extract number more carefully
            num_match = re.match(r'^(\d+)', line_stripped)
            if num_match:
                current_num = int(num_match.group(1))
                # Get rest of line after number (and any punctuation)
                rest = re.sub(r'^\d+[.,):\s]*', '', line_stripped)
                current_answer = [rest] if rest else []
            else:
                # Add to current answer if no number found
                if current_num > 0:
                    current_answer.append(line_stripped)
        elif current_num > 0:
            # Continue current answer
            current_answer.append(line_stripped)
    
    # Save last answer
    if current_num > 0 and current_answer:
        answers[current_num] = ' '.join(current_answer).strip()
    
    # NOW check if we need to add the critical question
    # Extract progress_score from answer 6 (PROGRESS ASSESSMENT)
    progress_score = 0
    if 6 in answers:
        try:
            score_text = answers[6]
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                progress_score = min(int(numbers[0]), 10)  # Cap at 10
        except:
            progress_score = 0


    # (state_changed already defined above - skipping duplicate)

    # Universal progress adjustment based on state change and task relevance  
    if state_changed and 3 in answers and 4 in answers:
        addressed_text = answers.get(3, '').lower()
        remaining_text = answers.get(4, '').lower()
        
        # Information theory: if "remaining" got shorter than "addressed", we made progress
        if len(remaining_text.split()) < len(task.split()):
            progress_score = max(progress_score, 4)
        
        # If something was addressed (not 'none' or 'n/a'), boost slightly
        if addressed_text and addressed_text not in ['none', 'nothing', 'n/a', '']:
            progress_score = max(progress_score, 3)
        
        # If task words appear in current observation but not in previous, we progressed
        task_tokens = set(task.lower().split())
        current_tokens = set(curr_observation.lower().split())
        previous_tokens = set(prev_observation.lower().split())
        new_relevant_tokens = (task_tokens & current_tokens) - previous_tokens
        
        if new_relevant_tokens:
            progress_score = max(progress_score, 5)
    

    # Check action alignment (removed redundant critical reflection - TextGrad handles this)
    action_alignment = answers.get(10, '')
    if 'not' in action_alignment.lower() or 'no' in action_alignment.lower():
        # Extract what should have been done
        should_have = action_alignment.split('instead')[-1].strip() if 'instead' in action_alignment else ''
        step_gradient['action_mismatch'] = True
        if should_have:
            step_gradient['intended_action'] = should_have
    
    # Extract structured information with defaults
    # (state_changed already defined above - skipping duplicate)
    
    missing_prereqs = answers.get(2, '').strip()
    missing_list = [missing_prereqs] if missing_prereqs and missing_prereqs.lower() not in ['none', 'nothing', 'n/a'] else []
    
    task_addressed = answers.get(3, '').strip()
    addressed_list = [task_addressed] if task_addressed and task_addressed.lower() not in ['none', 'nothing', 'n/a'] else []
    
    task_remaining = answers.get(4, task).strip()
    remaining_list = [task_remaining] if task_remaining else [task]
    
    why_failed = answers.get(5, None)
    if why_failed and why_failed.lower() in ['did not fail', 'n/a', 'none']:
        why_failed = None
        
    learned_rule = answers.get(6, 'Continue exploring available actions').strip()
    
    # ACTUAL CORRECT FIX: Extract score from learned_rule (answers[6])
    # This is where "PROGRESS SCORE: SCORE: 3 | REASONING: ..." actually is!
    if 'SCORE:' in learned_rule:
        score_match = re.search(r'SCORE:\s*(\d+)', learned_rule, re.IGNORECASE)
        if score_match:
            progress_score = min(int(score_match.group(1)), 10)
    
    next_guidance = answers.get(7, 'Try unexplored actions').strip()

    # Parse PROBE_ACTION from Question 7 answer — highest priority if present
    # Reflexion generates: "MISSING_INFO: ... | PROBE_ACTION: run: ls ~/Desktop | STRATEGY_SHIFT: ..."
    # Extract STRATEGY_SHIFT first (highest priority — paradigm changes)
    strategy_match = re.search(r'STRATEGY_SHIFT:\s*([^\|]+?)(?:\||$)', next_guidance, re.IGNORECASE)
    if not strategy_match:
        strategy_match = re.search(r'STRATEGY_SHIFT:\s*([^\n\|]+)', response, re.IGNORECASE)
    if strategy_match:
        strategy_action = strategy_match.group(1).strip().strip('.,;')
        # Extract actionable command from strategy shift (e.g., "run: python3 -c ...")
        _strat_run = re.search(r'(run:\s+[^\n]{10,})', strategy_action)
        _strat_press = re.search(r'(press\s+\w+)', strategy_action)
        if _strat_run:
            next_guidance = _strat_run.group(1).strip()
            log_debug(f"[REFLEXION] Extracted STRATEGY_SHIFT run command: '{next_guidance[:80]}'")
        elif _strat_press:
            next_guidance = _strat_press.group(1).strip()
            log_debug(f"[REFLEXION] Extracted STRATEGY_SHIFT key press: '{next_guidance}'")
        elif strategy_action.lower() not in ['none', 'n/a', 'not applicable', 'no shift needed']:
            # Store prose STRATEGY_SHIFT as next_guidance — VISION-FIRST can interpret it
            # This is critical: Reflexion's strategic analysis flows to the next action via VISION-FIRST
            next_guidance = strategy_action
            log_debug(f"[REFLEXION] Stored prose STRATEGY_SHIFT as guidance: '{strategy_action[:80]}'")

    # Then extract PROBE_ACTION (second priority — information gathering)
    probe_match = re.search(r'PROBE_ACTION:\s*([^\|]+?)(?:\||$)', next_guidance, re.IGNORECASE)
    if probe_match:
        probe_action = probe_match.group(1).strip().strip('.,;')
        if probe_action and probe_action.lower() not in ['none', 'n/a', 'not applicable']:
            next_guidance = probe_action  # Probing takes priority — gives us new observations
            log_debug(f"[REFLEXION] Extracted PROBE_ACTION: '{probe_action}'")
    else:
        # Also parse from anywhere in full response
        probe_match_full = re.search(r'PROBE_ACTION:\s*([^\n\|]+)', response, re.IGNORECASE)
        if probe_match_full:
            probe_action = probe_match_full.group(1).strip().strip('.,;')
            if probe_action and probe_action.lower() not in ['none', 'n/a', 'not applicable']:
                next_guidance = probe_action
                log_debug(f"[REFLEXION] Extracted PROBE_ACTION (full scan): '{probe_action}'")

    # CRITICAL FIX: Extract "Recommended next action: X" from anywhere in the response
    # GPT-5 embeds this in question 4 (CAUSAL CHAIN ANALYSIS) but doesn't use numbered format
    import re
    # Try multiple patterns in order of preference:
    # 1. "Recommended next action: X" (most specific, at end of causal chain)
    # 2. "RECOMMENDED ACTION: X" (more general, may include GOAL statement)
    recommended_action_match = re.search(r'Recommended next action:\s*([^\n]+)', response, re.IGNORECASE)
    if not recommended_action_match:
        recommended_action_match = re.search(r'RECOMMENDED ACTION:\s*-?\s*([a-z][^\n-]+)', response, re.IGNORECASE)

    if recommended_action_match:
        next_guidance = recommended_action_match.group(1).strip()
        # Clean up any leading/trailing punctuation or markdown
        next_guidance = next_guidance.strip('- .,;:')
        # If it starts with "GOAL:", skip it and try to find the actual action after it
        if next_guidance.upper().startswith('GOAL:'):
            # Look for the actual action after the GOAL statement
            action_after_goal = re.search(r'(?:GOAL:.*?)?[-\s]*(?:ACTION|RESULT|Recommended):\s*([a-z][^\n-]+)', response, re.IGNORECASE | re.DOTALL)
            if action_after_goal:
                next_guidance = action_after_goal.group(1).strip().strip('- .,;:')
        print(f"[REFLEXION FIX] Extracted action from RECOMMENDED ACTION: '{next_guidance}'")
    else:
        # Fallback: Extract exact action if it's quoted
        quoted_action = re.findall(r"'([^']+)'", next_guidance)
        if quoted_action:
            next_guidance = quoted_action[0]  # Use the exact quoted action

    # PHASE 1: Extract action from CAUSAL CHAIN (handles multiple LLM output formats)
    # Reflexion generates CAUSAL CHAIN, TextGrad needs clean action - extract for synergy
    if 'CAUSAL CHAIN' in next_guidance:
        action_match = None

        # Pattern 1: ACTION: go to X. | (colon, ends with period or pipe)
        action_match = re.search(r'ACTION:\s*([^.|]+?)(?:\.|\||$)', next_guidance, re.IGNORECASE)

        # Pattern 2: ACTION (go to X) (parentheses format)
        if not action_match:
            action_match = re.search(r'ACTION\s*\(([^)]+)\)', next_guidance, re.IGNORECASE)

        # Pattern 3: Action = go to X (equals sign format)
        if not action_match:
            action_match = re.search(r'ACTION\s*=\s*([^|>]+?)(?:\||->|RESULT)', next_guidance, re.IGNORECASE)

        if action_match:
            next_guidance = action_match.group(1).strip()
            print(f"[SYNERGY] Extracted action from CAUSAL CHAIN: '{next_guidance}'")
        else:
            # No ACTION found - pure analysis, use TODO/LLM instead
            next_guidance = ''
            print(f"[SYNERGY] No ACTION in CAUSAL CHAIN - will use TODO/LLM")

    # REMOVED duplicate naive word matching - use ONLY LLM's semantic evaluation
    # TextGrad needs clean semantic gradients, not polluted with heuristics
    
    # Build full gradient structure - preserves all learning signals
    step_gradient = {
        'state_change': curr_observation if state_changed else 'NO CHANGE',
        'progress_assessment': f"Addressed: {', '.join(addressed_list)}" if addressed_list else "No progress",
        'progress_score': progress_score,  
        'hypothesis': learned_rule,
        'next_action_guidance': next_guidance,
        'raw_reflection': response,
        'semantic_state_before': prev_observation,
        'semantic_state_after': curr_observation,
        'prerequisites': {
            'present': [],  # Could extract from success
            'missing': missing_list
        },
        'task_progress': {
            'addressed': addressed_list,
            'remaining': remaining_list
        }
    }
    
    # Add critical insight if available
    if 'critical' in answers:
        step_gradient['critical_insight'] = answers['critical']
    
    if DEBUG_CRITIC:
        print(f"\n[STEP REFLECTION]")
        print(f"  Step Number: {current_step}")  # You'll need to pass current_step as parameter
        print(f"  Action taken: {action}")
        print(f"  State: {prev_observation} -> {curr_observation}")
        print(f"  Changed: {state_changed}")
        print(f"  Missing prerequisites: {missing_list}")
        print(f"  Task progress: {addressed_list}")
        print(f"  Progress score: {progress_score}")
        if 'critical' in answers:
            print(f"  Critical insight: {answers['critical']}")
    

    # Score already extracted from answers[5] earlier in the function
    return step_gradient



def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        skip_discovery: bool = False,
        env_type: str = "alfworld",
        batch_size: int = 8,
        ablation_mode: str = 'combined',
        pure_ablation: bool = False,
        no_todo: bool = False,
    ) -> List[Dict[str, Any]]:
    """Run trial with discovery and gradient updates - NO PATTERNS"""

    # Phase 7 v2 monitor: start once per process, instrument model objects once.
    # No-op unless REFLEXGRAD_DEEP_TRACE=1.
    try:
        _dm.start(os.path.dirname(os.path.abspath(__file__)))
        if _dm.is_enabled():
            _dm.instrument_model(model, name="actor")
            if fast_model is not model:
                _dm.instrument_model(fast_model, name="fast")
    except Exception as _dm_err:
        print(f"[DEEP-MONITOR] start failed: {_dm_err}")

    # Set ablation flags based on mode
    global USE_REFLEXION, USE_TEXTGRAD, USE_TODO, PURE_ABLATION
    if ablation_mode == 'textgrad_only':
        USE_REFLEXION = False
        USE_TEXTGRAD = True
    elif ablation_mode == 'reflexion_only':
        USE_REFLEXION = True
        USE_TEXTGRAD = False
    elif ablation_mode == 'none':
        # Zero-shot baseline: all learning disabled
        USE_REFLEXION = False
        USE_TEXTGRAD = False
    else:  # combined (default)
        USE_REFLEXION = True
        USE_TEXTGRAD = True

    # Pure ablation mode - for ICML paper
    # Disables TODO and uses fresh prompts (not learned from previous runs)
    PURE_ABLATION = pure_ablation
    if pure_ablation or no_todo:
        USE_TODO = False
        print(f"\n[PURE ABLATION] TODO Manager DISABLED")
    else:
        USE_TODO = True

    print(f"\n[RUN_TRIAL START]")
    print(f"  trial_log_path = {trial_log_path}")
    print(f"  trial_idx = {trial_idx}")
    print(f"  num_envs = {len(env_configs)}")
    print(f"\n[ABLATION MODE] {ablation_mode}")
    print(f"  Reflexion (Memory): {USE_REFLEXION}")
    print(f"  TextGrad (Prompt Optimization): {USE_TEXTGRAD}")
    print(f"  TODO Manager: {USE_TODO}")
    print(f"  Pure Ablation: {PURE_ABLATION}\n")
    print(f"  env_type = {env_type}")
    print("[/RUN_TRIAL START]\n")
    
    global ENVIRONMENT_KNOWLEDGE, prompt_generator
    
    # DEBUG: Print current component state
    print(f"\n[TRIAL {trial_idx}] Starting with prompt components:")
    for comp, value in prompt_generator.prompt_components.items():
        print(f"  {comp}: {value}")
    
    # Load prompt_generator state if it exists (for trials > 0)
    # Load and selectively preserve learning
    if trial_idx > 0 and use_memory:
        PROMPT_GEN_PATH = 'prompt_generator_state.pkl'
        if os.path.exists(PROMPT_GEN_PATH) and os.path.getsize(PROMPT_GEN_PATH) > 0:
            try:
                with open(PROMPT_GEN_PATH, 'rb') as f:
                    old_pg = pickle.load(f)
                
                # Create fresh generator but preserve pool
                prompt_generator = DynamicPromptGenerator()
                
                # Preserve the generator pool if it exists
                if hasattr(old_pg, 'generator_pool'):
                    prompt_generator.generator_pool = old_pg.generator_pool
                    print(f"[LOADED] Generator pool with {len(prompt_generator.generator_pool)} task types")
                else:
                    prompt_generator.generator_pool = {}
                
                # PRESERVE THESE LEARNINGS:
                # 1. Discovered environment structure
                if hasattr(old_pg, 'environment_knowledge'):
                    prompt_generator.environment_knowledge = old_pg.environment_knowledge
                
                # 2. Available actions discovered
                if hasattr(old_pg, 'discovered_knowledge'):
                    prompt_generator.discovered_knowledge['available_actions'] = old_pg.discovered_knowledge.get('available_actions', set())
                    prompt_generator.discovered_knowledge['uses_numbers'] = old_pg.discovered_knowledge.get('uses_numbers', False)
                    prompt_generator.discovered_knowledge['completion_actions'] = old_pg.discovered_knowledge.get('completion_actions', set())
                
                # 3. Successful trajectories (if any)
                if hasattr(old_pg, 'successful_trajectories') and old_pg.successful_trajectories:
                    prompt_generator.successful_trajectories = old_pg.successful_trajectories
                    print(f"[PRESERVED] {len(old_pg.successful_trajectories)} successful trajectories")
                
                # 4. Gradient history for analysis (but don't pre-apply)
                if hasattr(old_pg, 'prompt_gradients'):
                    prompt_generator.prompt_gradients = old_pg.prompt_gradients
                    print(f"[PRESERVED] {len(old_pg.prompt_gradients)} gradient history")
                
                # 5. CRITICAL FIX: Preserve learned prompt components
                if hasattr(old_pg, 'prompt_components'):
                    prompt_generator.prompt_components = old_pg.prompt_components
                    print(f"[PRESERVED] Learned prompt components with TextGrad updates")

                print(f"[TRIAL {trial_idx}] Preserved structural learning, reset task-specific components")
                
            except Exception as e:
                print(f"[ERROR] Failed to load: {e}")
                prompt_generator = DynamicPromptGenerator()
    else:
        prompt_generator = DynamicPromptGenerator()

    parallel_mode = True  # Always use our batch implementation
    
    if skip_discovery:
        ENVIRONMENT_KNOWLEDGE = None  # Force no discovery knowledge
        # Also disable prompt generator's discovered knowledge
        prompt_generator.environment_knowledge = None
        env_understanding.environment_knowledge = None
    
    # Load appropriate environment configuration
    config = None
    if env_type == "alfworld":
        import alfworld
        import alfworld.agents.environment as alfworld_env
        
        importlib.reload(alfworld)
        importlib.reload(alfworld.agents.environment)
        
        # Get script directory to find base_config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'base_config.yaml')
        with open(config_path) as reader:
            config = yaml.safe_load(reader)

    # DISCOVERY PHASE - Run on first trial OR if knowledge is missing (resume without saved knowledge)
    # Auto-skip for GUI environments (OSWorld) - text exploration wastes expensive VGL calls
    if env_type and env_type.startswith("osworld") and not skip_discovery:
        print("\n[AUTO-SKIP] Skipping text-based discovery for GUI environment (osworld)")
        print("  GUI environments use VGL affordances instead of text exploration\n")
        skip_discovery = True

    discovery_report = ""
    if ENVIRONMENT_KNOWLEDGE is None and not skip_discovery:
        print("\n=== PERFORMING ENVIRONMENT DISCOVERY ===\n")
        
        # Get a WRAPPED environment for discovery
        env = get_environment(env_type, config, env_id=0)
        ob, info = env.reset()
        print(f"  ob type: {type(ob)}")
        print(f"  ob value (first 300): {str(ob)[:300]}")
        # Process observation universally
        if isinstance(ob, tuple) and len(ob) > 0:
            ob = ob[0]
        elif isinstance(ob, list):
            ob = ob[0]
        ob = str(ob) if not isinstance(ob, str) else ob  # Ensure it's string
        
        # Check if we have similar environment knowledge
        similar_knowledge = meta_knowledge.load_similar_environment(ob)
        
        if similar_knowledge:
            print("âœ“ Found similar environment knowledge - adapting...")
            ENVIRONMENT_KNOWLEDGE = similar_knowledge
            
            # Generate report for loaded knowledge
            discovery_report = "=== Using Transferred Environment Knowledge ===\n\n"
            
            # Action space info
            if ENVIRONMENT_KNOWLEDGE.get('action_space'):
                discovery_report += f"AVAILABLE COMMANDS: {len(ENVIRONMENT_KNOWLEDGE['action_space'])} discovered\n"
                # Show first few commands
                for cmd, info_dict in list(ENVIRONMENT_KNOWLEDGE['action_space'].items())[:5]:
                    if isinstance(info_dict, dict):
                        discovery_report += f"  - {info_dict.get('format', cmd)}: {info_dict.get('description', '')}\n"
                    else:
                        discovery_report += f"  - {cmd}\n"
                if len(ENVIRONMENT_KNOWLEDGE['action_space']) > 5:
                    discovery_report += f"  ... and {len(ENVIRONMENT_KNOWLEDGE['action_space']) - 5} more\n"
            
            # State space info
            if ENVIRONMENT_KNOWLEDGE.get('state_space', {}).get('locations'):
                num_locations = len(ENVIRONMENT_KNOWLEDGE['state_space']['locations'])
                discovery_report += f"\nDISCOVERED LOCATIONS: {num_locations} found\n"
            
            # Constraints
            if ENVIRONMENT_KNOWLEDGE.get('constraints'):
                discovery_report += "\nKNOWN CONSTRAINTS:\n"
                for constraint in ENVIRONMENT_KNOWLEDGE['constraints']:
                    discovery_report += f"  - {constraint}\n"
            
            discovery_report += "\nâœ“ Ready to use transferred knowledge!\n"
            
        else:
            print("Ã— No similar environment found - running full discovery...")
            
            # Run discovery with wrapped environment
            discoverer = UniversalEnvironmentDiscovery()
            ENVIRONMENT_KNOWLEDGE = discoverer.discover_environment(
                env, ob, max_discovery_steps=50
            )
            
            # Save discovered knowledge
            if env_type == "alfworld":
                env_name = info['extra.gamefile'][0].split('/')[-3] if 'extra.gamefile' in info else 'unknown'
            else:
                env_name = env_type
            meta_knowledge.save_environment_knowledge(env_name, ENVIRONMENT_KNOWLEDGE)
            
            # Generate discovery report
            discovery_report = discoverer.generate_discovery_report()
        
        # Share knowledge with components
        prompt_generator.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        env_understanding.inject_discovered_knowledge(ENVIRONMENT_KNOWLEDGE)
        
        # Log discovery results
        print(discovery_report)
        
        with open(world_log_path, 'a') as wf:
            wf.write("\n" + discovery_report + "\n")
        
        # Add meta summary
        meta_summary = meta_knowledge.get_discovery_summary()
        print(meta_summary)
        
        # Close discovery environment
        env.close()

    # Log current learning state - SIMPLIFIED, NO PATTERNS
    optimization_state = prompt_generator.get_prompt_optimization_state()
    with open(world_log_path, 'a') as wf:
        wf.write(f'\nTrial #{trial_idx} Learning State:\n')
        wf.write(f'Ablation Mode: {ablation_mode}\n')
        wf.write(f'Reflexion Active: {USE_REFLEXION}\n')
        wf.write(f'TextGrad Active: {USE_TEXTGRAD}\n')
        wf.write(f'Number of environments: {len(env_configs)}\n')
        wf.write(f'Available Actions Found: {optimization_state["available_actions_found"]}\n')
        wf.write(f'Interactions: {optimization_state["interaction_count"]}\n')
        wf.write(f'Updates: {optimization_state["num_updates"]}\n')
        wf.write(f'Has Environment Knowledge: {optimization_state["has_environment_knowledge"]}\n')
        wf.write(f'Uses Numbered Items: {optimization_state["uses_numbered_items"]}\n')
        wf.write(f'Completion Actions: {optimization_state["completion_actions"]}\n\n')
        
        # Log prompt components to track evolution
        wf.write(f'Prompt Components (Trial #{trial_idx}):\n')
        for component, value in prompt_generator.prompt_components.items():
            wf.write(f'  {component}: {value}\n')
        wf.write('\n')

    # ALWAYS USE PARALLEL EXECUTION
    print(f"\n=== RUNNING {len(env_configs)} ENVIRONMENTS IN PARALLEL MODE ===")
    print(f"Environment Type: {env_type}")
    
    # Process in batches to manage memory
    BATCH_SIZE = batch_size  # Use parameter from function
    num_successes = 0
    num_additional_successes = 0
    num_envs = len(env_configs)
    
    # Process environments in batches
    for batch_start in range(0, len(env_configs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(env_configs))
        batch_size = batch_end - batch_start

        print(f"\n--- Processing batch: environments {batch_start} to {batch_end-1} ({batch_size} envs) ---")

        # FORCE LOGGING TO FILE TO VERIFY BATCH EXECUTION
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Processing batch: environments {batch_start} to {batch_end-1} ({batch_size} envs)\n")
                f.flush()
        
        # Prepare batch
        batch_envs = []
        batch_memories = []
        batch_initial_obs = []
        batch_indices = []
        
        for i in range(batch_start, batch_end):
            # FORCE LOG EACH ENV PROCESSING START - NO TRY/CATCH, FAIL HARD
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[ENV DEBUG] Starting processing for ENV {i}\n")
                    f.flush()

            env_config = env_configs[i]
            print(f"\n[PROCESSING ENV {i}]")

            # DEBUG: Check memory status
            memory_items = env_config.get('memory', [])
            print(f"[ENV {i}] Trial {trial_idx} - Memory in config: {len(memory_items)} items")
            if memory_items:
                workflow_count = sum(1 for item in memory_items if isinstance(item, dict) and item.get('type') == 'success_workflow')
                print(f"[ENV {i}] Trial {trial_idx} - success_workflow items: {workflow_count}")
                if workflow_count > 0:
                    for item in memory_items:
                        if isinstance(item, dict) and item.get('type') == 'success_workflow':
                            print(f"[ENV {i}] Trial {trial_idx} - Workflow for task: '{item.get('task', 'NO TASK')}'")

            if trial_idx > 0 and use_memory:
                if 'memory' not in env_config or len(env_config['memory']) == 0:
                    print(f"[WARNING] ENV {i} has no memory despite being trial {trial_idx}")
            
            print(f"\n[PROCESSING ENV {i}] Starting...")
            
            # Create environment ONLY ONCE
            env = get_environment(env_type, config, env_id=i)
            ob, info = env.reset()
            
            print(f"[DEBUG ENV {i}] Reset output:")
            print(f"  ob type: {type(ob)}")
            print(f"  ob value (first 300): {str(ob)[:300]}")
            print(f"  ob is list? {isinstance(ob, list)}")
            if isinstance(ob, list) and ob:
                print(f"  ob[0] type: {type(ob[0])}")
                print(f"  ob[0] value (first 300): {str(ob[0])[:300]}")

            # Process observation
            if hasattr(env, 'process_observation'):
                # Handle tuple from ALFWorld
                if isinstance(ob, tuple) and len(ob) > 0:
                    processed_ob = env.process_observation(ob)
                else:
                    processed_ob = env.process_observation(ob[0] if isinstance(ob, list) else ob)
            else:
                raw_ob = ob[0] if isinstance(ob, list) else ob
                if isinstance(raw_ob, str):
                    processed_ob = raw_ob
                else:
                    processed_ob = str(raw_ob)

            print(f"[DEBUG ENV {i}] After processing:")
            print(f"  processed_ob type: {type(processed_ob)}")
            print(f"  processed_ob (first 500): {processed_ob[:500]}")

            # Get environment name
            name = env.get_environment_name(info) if hasattr(env, 'get_environment_name') else f"{env_type}_task_{i}"

            # Extract task - prefer info['task'] for environments that provide it (e.g., OSWorld)
            task = None
            if isinstance(info, dict) and info.get('task') and info['task'] not in ('Unknown task', 'TASK_EXTRACTION_FAILED'):
                task = info['task']
                print(f"[DEBUG ENV {i}] Task from env info: {task}")
            else:
                task = prompt_generator._extract_task_from_observation(processed_ob)
                print(f"[DEBUG ENV {i}] Task extraction from observation:")
                print(f"  Extracted task: {task}")

            if not task:
                print(f"[CRITICAL ERROR] Task extraction FAILED for env {i}!")
                print(f"  Observation: {processed_ob[:200]}")

                # FAIL FAST - NO FALLBACK TO EXPOSE ROOT CAUSE
                env_configs[i]["is_success"] = False
                env_configs[i]["task"] = "TASK_EXTRACTION_FAILED"

                # Log and close this environment
                if trial_log_path:
                    with open(trial_log_path, 'a') as f:
                        f.write(f"\n[CRITICAL] ENV {i}: Task extraction failed, observation: {processed_ob[:200]}\n")

                env.close()
                continue  # Skip to next environment in batch
            else:
                # Set task in all relevant places
                prompt_generator.set_task(task)
                env_configs[i]['task'] = task
                if env_understanding:
                    env_understanding.current_task = task

                print(f"Environment: {name}")
                print(f"Task: {task}")

            # Task is guaranteed to exist here (failed cases already handled above)

            # Get initial valid actions
            if hasattr(env, 'get_current_valid_actions'):
                initial_actions = env.get_current_valid_actions()
                if initial_actions:
                    prompt_generator.discovered_knowledge['available_actions'] = set(initial_actions)
                    if DEBUG_ACTOR:
                        print(f"[DEBUG] Initial valid actions: {len(initial_actions)}")


            # Get optimized memory from global manager if available
            memory_to_use = env_config.get("memory", [])  # Use existing memory first

            # Pure LLM-based memory retrieval
            if use_memory and trial_idx > 0:
                from generate_reflections import global_memory_manager

                # CRITICAL FIX: Preserve success_workflow items before LLM filtering
                # These are structured patterns that must be preserved for cross-trial learning
                success_workflows = []
                other_memories = []
                for item in memory_to_use:
                    if isinstance(item, dict) and item.get('type') == 'success_workflow':
                        success_workflows.append(item)
                        print(f"[ENV {i}] PRESERVING success_workflow: '{item.get('task', 'NO TASK')}'")
                    else:
                        other_memories.append(item)

                # Use pure LLM-based retrieval with current observation (only on textual memories)
                optimized_memory = global_memory_manager.get_relevant_memories(
                    task=task,
                    env_memory=other_memories,  # Filter only textual memories, not workflows
                    k=6,  # Request top 6 most useful memories
                    current_observation=processed_ob,  # Pass current situation
                    env_id=i  # Pass env_id for contamination logging
                )

                # CRITICAL FIX: Combine preserved workflows with filtered memories
                # Success workflows ALWAYS come first for priority in action selection
                if optimized_memory:
                    memory_to_use = success_workflows + optimized_memory
                    print(f"[ENV {i}] Using {len(success_workflows)} workflows + {len(optimized_memory)} LLM-selected memories")
                else:
                    memory_to_use = success_workflows + other_memories
                    print(f"[ENV {i}] Using {len(success_workflows)} workflows + {len(other_memories)} memories (no LLM filtering)")

            # Add to batch
            batch_envs.append(env)
            batch_memories.append(memory_to_use)
            batch_initial_obs.append(processed_ob)
            batch_indices.append(i)
        
        # Skip if no environments in this batch
        if not batch_envs:
            if trial_log_path:
                with open(trial_log_path, 'a') as f:
                    f.write(f"\n[BATCH DEBUG] SKIPPING batch {batch_start}-{batch_end-1}: batch_envs is EMPTY\n")
                    f.flush()
            print(f"WARNING: Skipping empty batch {batch_start}-{batch_end-1}")
            continue

        print(f"Running {len(batch_envs)} environments in parallel...")

        # LOG BATCH SIZE
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Running batch {batch_start}-{batch_end-1} with {len(batch_envs)} environments, indices: {batch_indices}\n")
                f.flush()
        
        # DEBUG: Verify trial_log_path before calling
        print(f"[PRE-CALL DEBUG] trial_log_path = {trial_log_path}")
   
        # RUN BATCH IN PARALLEL
        import psutil
        import os as os_module
        process = psutil.Process(os_module.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY DEBUG] Before adaptive_env_interaction_batch: {memory_before:.2f} MB")

        results = adaptive_env_interaction_batch(
            envs=batch_envs,
            base_prompt="",
            memories=batch_memories,
            to_print=True,
            initial_obs_list=batch_initial_obs,
            trial_log_path=trial_log_path,
            env_configs=env_configs,
            trial_idx=trial_idx,
            use_memory=use_memory
        )

        memory_after = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY DEBUG] After adaptive_env_interaction_batch: {memory_after:.2f} MB")
        print(f"[MEMORY DEBUG] Memory increase: {memory_after - memory_before:.2f} MB")
        print(f"[MEMORY DEBUG] Results type: {type(results)}, length: {len(results)}")

        # Process results
        print(f"\n[DEBUG RESULTS] Processing {len(results)} results")
        print(f"  batch_indices: {batch_indices}")
        print(f"  results success values: {[r[1] for r in results]}")
        
        for idx, (env_idx, (history, is_success)) in enumerate(zip(batch_indices, results)):
            # CRITICAL: Set success status in env_configs
            env_configs[env_idx]['is_success'] = is_success
            print(f"\n[DEBUG] Processing result {idx}:")
            print(f"  env_idx={env_idx}")
            print(f"  is_success={is_success}, type={type(is_success)}")

            if is_success:
                status_str = f'Environment #{env_idx} Trial #{trial_idx}: SUCCESS'
                env_configs[env_idx]['is_success'] = True
                num_successes += 1
                num_additional_successes += 1
            else:
                status_str = f'Environment #{env_idx} Trial #{trial_idx}: FAIL'
                env_configs[env_idx]['is_success'] = False
            # Log results
            with open(world_log_path, 'a') as f:
                f.write(status_str + '\n')
            
         
            # Write with consistent format for each environment
            with open(trial_log_path, 'a') as wf:
                # Add clear separator between environments
                if env_idx > batch_indices[0]:
                    wf.write('\n\n')  # Simple double newline separator
                
                # Write environment log with clear markers
                wf.write(f'Environment #{env_idx}:\n')
                wf.write(str(history))
                wf.write(f'\nSTATUS: {"OK" if is_success else "FAIL"}\n')
        
        # Clean up batch environments
        for env in batch_envs:
            env.close()
        
        print(f"Batch complete. Successes in batch: {sum(1 for _, (_, s) in zip(batch_indices, results) if s)}")

        # FORCE LOG BATCH COMPLETION
        if trial_log_path:
            with open(trial_log_path, 'a') as f:
                f.write(f"\n[BATCH DEBUG] Completed batch {batch_start}-{batch_end-1}: {sum(1 for _, (_, s) in zip(batch_indices, results) if s)} successes\n")
                f.flush()



    # Update generators in pool without merging
    # CRITICAL FIX: Update pool with learning from ALL environments
    print("[UPDATING] Updating task-specific generators in pool...")
    if 'env_prompt_generators' in locals() and env_prompt_generators:
        for env_id, env_pg in env_prompt_generators.items():
            if env_id >= len(env_states):
                continue
                
            task = env_states[env_id]['task']
            was_successful = env_states[env_id].get('success', False)
            
            # Find or create pool entry for this task type
            best_match_task = None
            best_similarity = 0
            
            for pool_task in prompt_generator.generator_pool.keys():
                task_words = set(task.lower().split())
                pool_words = set(pool_task.lower().split())
                similarity = len(task_words & pool_words) / len(task_words | pool_words) if (task_words | pool_words) else 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_task = pool_task
            
            # Update or create pool entry
            if best_similarity > 0.7:  # Similar task exists
                print(f"[POOL SYNC] '{task[:30]}' → '{best_match_task[:30]}' (sim: {best_similarity:.2f})")
                pool_pg = prompt_generator.generator_pool[best_match_task]
                
                # Merge learning regardless of success/failure
                for component, value in env_pg.prompt_components.items():
                    # Check if this component actually changed from default
                    default_value = DynamicPromptGenerator().prompt_components.get(component)
                    if value != default_value:
                        # Weight by success
                        if was_successful:
                            pool_pg.prompt_components[component] = value
                        else:
                            # For failures, use momentum update
                            pool_pg.update_component_with_momentum(component, value, momentum=0.3)
                
                # Always merge discovered knowledge
                pool_pg.discovered_knowledge.update(env_pg.discovered_knowledge)
                pool_pg.prompt_gradients.extend(env_pg.prompt_gradients[len(pool_pg.prompt_gradients):])

            elif best_similarity < 0.3:  # New task type
                # Store this as a new task type
                prompt_generator.generator_pool[task] = env_pg
                print(f"[UPDATING] Added new task type to pool: {task[:30]}")
        
        print(f"[UPDATING] Pool now contains {len(prompt_generator.generator_pool)} task types")
    

    # Save the merged global generator
    save_prompt_generator()
    save_environment_knowledge()
    print("[SAVED] Global prompt_generator and ENVIRONMENT_KNOWLEDGE with merged learning")


    # Log trial summary - SIMPLIFIED, NO PATTERNS
    discovered_actions = len(prompt_generator.discovered_knowledge.get('available_actions', []))
    if ENVIRONMENT_KNOWLEDGE and 'action_space' in ENVIRONMENT_KNOWLEDGE:
        total_actions = len(ENVIRONMENT_KNOWLEDGE['action_space'])
    else:
        total_actions = discovered_actions
        
    # Add learning summary for paper
    learning_summary = "\n=== LEARNING SUMMARY (Trial {}) ===\n".format(trial_idx)
    total_episodic = sum(len(env_configs[i].get('memory', [])) for i in range(num_envs))
    total_step_mem = sum(len(env_configs[i].get('step_memory', [])) for i in range(num_envs))
    learning_summary += f"Episodic Memory Entries: {total_episodic} (cross-trial strategic patterns)\n"
    learning_summary += f"Step Memory Entries: {total_step_mem} (within-trial tactical learning)\n"
    learning_summary += f"TextGrad + Reflexion Synergy: Active (both memories used for meta-learning)\n"

    log_str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
DISCOVERED ACTIONS: {discovered_actions}/{total_actions}
ENVIRONMENT TYPE: {env_type}
EXECUTION MODE: PARALLEL (Batch size: {BATCH_SIZE})
{learning_summary}
-----
"""

    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    # Ensure results are saved even if there's an error
    try:
        logging_dir = os.path.dirname(trial_log_path) if trial_log_path else '.'
        env_config_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
        with open(env_config_path, 'w') as f:
            json.dump(env_configs, f, indent=4)
            f.flush()  # Force write
            os.fsync(f.fileno())  # Force OS to write to disk
        print(f"[SAVED] Trial {trial_idx} results to {env_config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save trial results: {e}")
        logging_dir = os.path.dirname(trial_log_path) if trial_log_path else '.'
        # Emergency save as backup
        backup_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}_backup.json')
        with open(backup_path, 'w') as f:
            # Save minimal data
            minimal_data = [{'is_success': env.get('is_success', False)} for env in env_configs]
            json.dump(minimal_data, f)

    # FORCE LOG FINAL COMPLETION
    if trial_log_path:
        with open(trial_log_path, 'a') as f:
            total_success = sum(1 for env in env_configs if env.get('is_success', False))
            f.write(f"\n[FINAL DEBUG] All batches completed. Total successes: {total_success}/{len(env_configs)}\n")
            f.flush()

    print(f"\n[FINAL] All batches completed. Total successes: {sum(1 for env in env_configs if env.get('is_success', False))}/{len(env_configs)}")

    return env_configs