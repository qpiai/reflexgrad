import hashlib
from universal_state_embeddings import state_embeddings
# Dynamic model loading based on MODEL_PROVIDER env var
import os
if os.getenv("MODEL_PROVIDER", "openai").lower() == "gemini":
    from shared_model_gemini import model, fast_model
else:
    from shared_model import model, fast_model
"""Universal trial execution with adaptive learning and discovery"""
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

# SEQUENTIAL LEARNING SYSTEM - Smart knowledge transfer
from task_classifier import task_classifier
from knowledge_classifier import knowledge_classifier
from learning_extractor import learning_extractor

try:
    from vllm import LLM, SamplingParams
except ImportError:
    from shared_model import SamplingParams
    LLM = None
import random
import pickle
# Global instances for optimization - PERSIST ACROSS TRIALS
import pickle
import os

# Add these globals
checkpoint_manager = None  # Will be initialized in adaptive_env_interaction_batch
comprehensive_logger = None

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
        """
        self.base_policy = base_policy
        self.constraints = reflexion_constraints  # From Reflexion cross-trial learning
        self.gradients = []  # Accumulated textual gradients (within-trial)
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
                next_action_guidance: str = None) -> str:
        """
        FIX #18 (Nov 30): PURE DIRECT EXECUTION - NO FALLBACKS

        ROOT CAUSE OF BUG: The old code had fuzzy matching that corrupted actions:
        "take saltshaker 2" → "take creditcard 2" (wrong object!)

        NEW APPROACH: Execute EXACTLY what TextGrad recommends
        - If TextGrad has guidance → execute it directly (let ALFWorld accept/reject)
        - ALFWorld's response becomes the learning signal
        - NO fallbacks that could corrupt the action-learning mapping
        - TextGrad learns from its own recommendations, not corrupted alternatives
        """

        # ========== DIRECT EXECUTION OF TEXTGRAD GUIDANCE ==========
        if next_action_guidance and next_action_guidance.strip():
            # Extract just the action (before JUSTIFICATION)
            guidance_action = next_action_guidance.split(' JUSTIFICATION')[0].strip()

            # Check for loop (same action repeated 3+ times)
            is_loop = recent_actions and recent_actions[-5:].count(guidance_action) >= 3

            if not is_loop:
                # Execute TextGrad's recommendation DIRECTLY
                # Let ALFWorld accept or reject - that becomes the learning signal
                print(f"[FIX #18] DIRECT EXECUTE TextGrad recommendation: '{guidance_action}'")
                return guidance_action
            else:
                print(f"[FIX #18] TextGrad recommendation in LOOP - using 'look' to break")
                return "look"  # Safe action to break loops and get new info

        # ========== NO GUIDANCE - USE FIRST VALID ACTION ==========
        # If no TextGrad guidance, use first valid action (usually 'examine' or 'look')
        print(f"[FIX #18] No TextGrad guidance - using first valid action: '{valid_actions[0]}'")
        return valid_actions[0]

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


class TextGradLoss:
    """
    GOAL-ALIGNED Loss Function for TextGrad.
    Evaluates if action helps achieve TASK GOAL, not just execution success.
    """

    def __init__(self, model):
        self.model = model  # Store model for loss computation
        self._init_prompts()  # Initialize prompt templates

    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
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

    def _init_prompts(self):
        self.evaluation_prompt_template = """You are evaluating whether an action helps achieve the TASK GOAL.

TASK GOAL: {task}
CURRENT SUBTASK (from TODO system): {subtask}

LEARNED INSIGHTS FROM PREVIOUS ATTEMPTS:
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

=== CRITICAL EVALUATION (Focus on GOAL ALIGNMENT, not just execution) ===

1. TASK GOAL ALIGNMENT (MOST IMPORTANT):
   - What does the TASK GOAL require?
   - What did this action actually produce (see STATE AFTER)?
   - Does the RESULT match what the TASK GOAL is asking for?
   - If the task requires operation X but action performed operation Y, this is NOT goal-aligned.

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

VERDICT: [GOAL_ALIGNED / GOAL_NOT_ALIGNED / PARTIAL_PROGRESS]
REASON: [Explain specifically WHY the action does or doesn't help achieve the TASK GOAL]
NEXT_ACTION_HINT: [What should be done to actually achieve the TASK GOAL?]"""

    def __call__(self, action: str, state_before: str, state_after: str,
                 task: str, policy, trajectory_context: list = None,
                 subtask: str = None, reflexion_insights: list = None) -> str:
        """
        Compute textual loss (criticism) for the action taken by policy.

        GOAL-ALIGNED LOSS: Evaluates if action helps achieve TASK GOAL,
        not just if action executed successfully.

        Args:
            trajectory_context: List of dicts with keys: action, observation
            subtask: Current subtask from TODO manager (guidance)
            reflexion_insights: List of learned insights from Reflexion episodic memory

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

        prompt = self.evaluation_prompt_template.format(
            task=task,
            subtask=current_subtask,
            reflexion_insights=formatted_insights,
            policy=policy_text,
            constraints=constraints_text,
            trajectory_history=trajectory_history,
            state_before=state_before,
            action=action,
            state_after=state_after
        )

        print(f"[TEXTGRAD LOSS] Computing GOAL-ALIGNED loss for action: '{action[:50]}'")
        loss_text = self._generate(prompt, temperature=0.3, max_tokens=400)
        loss_summary = loss_text[:150] + "..." if len(loss_text) > 150 else loss_text
        print(f"[TEXTGRAD LOSS] Computed: {loss_summary}")
        return loss_text


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

            history_str += f"\n  {idx}. State: {state_before[:200]}"  # FIX #36: Increased from 100 to 200
            history_str += f"\n     Action: {action_taken}"
            history_str += f"\n     Result: {obs_after[:200]}"  # FIX #36: Increased from 100 to 200
            history_str += f"\n     Progress: {progress}\n"  # ✅ TextGrad sees if action made progress!
        tried_actions_str = history_str
    else:
        tried_actions_str = "None yet"

    # Format Reflexion insights (most recent 3)
    if working_reflexions:
        recent_insights = working_reflexions[-3:]
        reflexion_str = "\n".join([f"  - {str(r)[:150]}" for r in recent_insights])
    else:
        reflexion_str = "No Reflexion insights yet"

    # Format cross-trial constraints (top 3)
    if reflexion_memory:
        top_constraints = reflexion_memory[:3]
        constraints_str = "\n".join([f"  - {str(c)[:150]}" for c in top_constraints])
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
    # FIX #20: UNIVERSAL ACTION VALIDITY FEEDBACK
    # When action returns "Nothing happens", the agent needs to understand WHY.
    # By explicitly stating whether action was in valid_actions, the gradient can
    # learn preconditions from the environment's own feedback - 100% universal.
    # ═══════════════════════════════════════════════════════════════════════════════
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

VALID ACTIONS (What actions are actually possible in the current state):
{valid_actions_str}

IMPORTANT: When recommending next actions in your gradient, ensure they come from the
VALID ACTIONS list above. The environment has explicit action semantics - recommend only
from actions that actually exist.

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
Your task: Compute the gradient ∂loss/∂policy.

Using ALL the context above, describe HOW the policy should be modified to avoid
this loss in the future. Be smart:
- Don't recommend actions that were already tried (check HISTORY)
- Consider TODO as a guide, but prioritize TASK REQUIREMENT when they conflict
- Consider Reflexion insights (check STRATEGIC INSIGHTS)
- TASK REQUIREMENT is the ultimate goal - adapt plan if observations suggest better path
- Recommend only from VALID ACTIONS (check what's actually possible)
- If ACTION VALIDITY STATUS shows "NOT in valid_actions", the action was REJECTED by environment.
  Examine the similar valid actions to understand what precondition was missing.

Provide a concise, actionable policy improvement:
- What aspect of the policy led to this suboptimal action?
- How should that aspect be changed considering the context?
- What should the policy prioritize instead?

Be specific and actionable.

Answer:
GRADIENT: [specific improvement instruction for the policy]
"""

    print(f"[TEXTGRAD BACKWARD] Computing gradient for action: '{action[:50]}'")
    # Use fast_model.generate() API
    output = model.generate(
        [gradient_prompt],
        SamplingParams(max_tokens=200, temperature=0.3, stop=["\n\n"])
    )[0]
    gradient_response = output.outputs[0].text.strip()

    # Extract gradient
    if "GRADIENT:" in gradient_response:
        gradient = gradient_response.split("GRADIENT:")[1].strip()
    else:
        gradient = gradient_response

    print(f"[TEXTGRAD BACKWARD] Gradient computed: '{gradient[:100]}'")
    return gradient


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

    def _init_prompts(self):
        self.update_prompt_template = """You are updating an action selection policy based on accumulated gradients.

CURRENT POLICY:
{current_policy}

STRATEGIC CONSTRAINTS (must preserve):
{constraints}

ACCUMULATED GRADIENTS (improvements to apply):
{gradients}

Your task: Produce an UPDATED POLICY that incorporates these gradients while preserving the strategic constraints.

Rules:
1. Keep the fundamental goal/objective
2. Add or strengthen rules suggested by gradients
3. Make it concise and actionable (2-4 sentences)
4. MUST preserve all strategic constraints from Reflexion
5. Focus on the most important/frequent gradient patterns

Answer:
UPDATED_POLICY: [new policy text incorporating gradients]
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

        # Prepare prompt
        prompt = self.update_prompt_template.format(
            current_policy=policy.base_policy,
            constraints=policy._format_constraints(),
            gradients=self._format_gradients_for_update(policy.gradients)
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

        # Update policy
        policy.base_policy = updated_policy
        print(f"[TEXTGRAD OPTIMIZER] ✓ Policy updated (version {len(policy.update_history)})")
        print(f"[TEXTGRAD OPTIMIZER] New policy: '{updated_policy[:100]}'")

        # Keep gradients for history but could clear if memory is an issue
        # policy.gradients = []

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
            # Handle dict format - try multiple possible keys
            text = ref.get('reflection', ref.get('insight', ref.get('hypothesis', str(ref))))
        else:
            text = str(ref)
        # Truncate to 100 chars for focus
        if len(text) > 100:
            text = text[:97] + "..."
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
    """Tiered compression: recent=verbose, medium=compressed, old=summary

    CRITICAL FIX: NEVER compress cross-trial reflexions (Trial 0 lessons)
    These are the most important for learning and must be preserved in full detail.

    Tier 1: ALL cross-trial reflexions - keep FULL VERBOSE (preserve learning)
    Tier 2: Last 2 current-trial reflexions - keep FULL VERBOSE (immediate context)
    Tier 3: Steps 3-5 back (current trial) - MEDIUM compression
    Tier 4: Older than 5 (current trial) - HEAVY compression
    """

    reflexions = state.get('working_reflexions', [])

    if len(reflexions) <= 2:
        # Not enough history, keep all verbose
        return reflexions

    # Get current trial index (may not be set for Trial 0)
    current_trial = state.get('trial_idx', 0)

    # CRITICAL: Separate cross-trial vs current-trial reflexions
    cross_trial = []
    current_trial_refls = []

    for r in reflexions:
        r_trial = r.get('trial', current_trial)  # Default to current if not set
        if r_trial < current_trial:
            # This is from a PREVIOUS trial - NEVER compress!
            cross_trial.append(r)
        else:
            # This is from CURRENT trial - can compress if too many
            current_trial_refls.append(r)

    # Log what we found
    if cross_trial:
        log_debug(f"[CROSS-TRIAL PRESERVATION] Keeping {len(cross_trial)} Trial {current_trial - 1} reflexions FULLY VERBOSE (no compression)")

    # Build result: cross-trial first (never compressed), then current trial (tiered)
    result = cross_trial.copy()  # Keep all cross-trial reflexions verbose

    # Only compress CURRENT trial reflexions if too many
    if len(current_trial_refls) <= 2:
        result.extend(current_trial_refls)
    elif len(current_trial_refls) > 5:
        # Tier 1: Last 2 - VERBOSE
        recent = current_trial_refls[-2:]

        # Tier 2: Steps 3-5 back - MEDIUM compression
        medium_reflexions = current_trial_refls[-5:-2]
        medium_compressed = compress_reflexions_medium(
            reflexions=medium_reflexions,
            model=model,
            max_tokens=150
        )

        # Tier 3: Older than 5 - HEAVY compression
        old_reflexions = current_trial_refls[:-5]
        old_summary = compress_reflexions_heavy(
            reflexions=old_reflexions,
            model=model,
            max_tokens=100
        )

        log_debug(f"[CURRENT-TRIAL COMPRESSION] Steps {old_reflexions[0]['step']}-{old_reflexions[-1]['step']}: heavy | Steps {medium_reflexions[0]['step']}-{medium_reflexions[-1]['step']}: medium | Steps {recent[0]['step']}-{recent[1]['step']}: verbose")

        result.extend([old_summary, medium_compressed] + recent)
    else:
        # 3-5 reflexions: compress older, keep last 2 verbose
        older = current_trial_refls[:-2]
        compressed = compress_reflexions_medium(older, model, max_tokens=150)
        recent = current_trial_refls[-2:]

        log_debug(f"[CURRENT-TRIAL COMPRESSION] Steps {older[0]['step']}-{older[-1]['step']}: medium | Steps {recent[0]['step']}-{recent[1]['step']}: verbose")

        result.extend([compressed] + recent)

    return result



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
   - Adaptive Strategy: {textgrad_components.get('adaptive_strategy', 'None')}
   - Task Decomposition: {textgrad_components.get('task_decomposition', 'None')}
   - Environment Understanding: {textgrad_components.get('environment_understanding', 'None')}
   - Action Discovery: {textgrad_components.get('action_discovery', 'None')}
   - Pattern Recognition: {textgrad_components.get('pattern_recognition', 'None')}
   - Hypothesis Testing: {textgrad_components.get('hypothesis_testing', 'None')}

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
                prompt += f"\n     Before: {failure['context_before'][:80]}..."
                prompt += f"\n     After: {failure['context_after'][:80]}..."

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
                extracted = action_match.group(1).strip()
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
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: TextGrad action extracted: '{textgrad_rec}'")
            else:
                log_debug(f"[PHASE2-EXTRACT] ENV {env_id}: TextGrad clean action: '{textgrad_rec[:50]}'")
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
            print(f"[ACTION SELECTION] Using GPT-5 (minimal reasoning) for {len(llm_prompts)} action(s)")
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
        todo_manager = TaskTodoManager(model, fast_model)  # Reasoning for generation, fast for verification

        try:
            initial_todos = todo_manager.initialize_from_task(
                task,
                ob,
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
            'consolidated_step_wisdom': "",
            'todo_manager': env_data['todo_manager'],
            'working_reflexions': [],  # Will be populated below with quality-filtered reflexions
            'reflexion_memory': env_data['reflexion_memory'] if env_data['reflexion_memory'] else [],  # CRITICAL FIX: Cross-trial Reflexion constraints + cross_env_insights for failed envs
            # TEXTGRAD INTEGRATION: Add policy components to env_states
            'policy': env_data['policy'],        # TextGrad policy
            'loss_fn': env_data['loss_fn'],      # Loss function
            'optimizer': env_data['optimizer'],   # Optimizer
            # FIX #32: Cumulative Search Memory - LLM-generated, persists across steps
            'cumulative_search_memory': ""  # Accumulated search history string, grows each step
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
            valid_actions = batch_data.get('valid_actions', [])[:30]

            # Get next_action_guidance from PREVIOUS step's gradient (this is the CRITICAL fix!)
            # This contains the specific recommended action from TextGrad analysis
            last_step_gradient = batch_data.get('last_step_gradient', {})
            next_action_guidance = last_step_gradient.get('next_action_guidance', '')

            # DEBUG: Log guidance retrieval at every step
            print(f"[DEBUG GUIDANCE] ENV {state['env_id']} Step {state['cur_step']}: last_step_gradient has {len(last_step_gradient)} keys")
            if next_action_guidance:
                print(f"[TEXTGRAD GUIDANCE] ENV {state['env_id']} Step {state['cur_step']}: Has recommendation: '{next_action_guidance[:80]}...'")

            # USE THE LEARNED POLICY for action selection!
            action = policy.forward(
                state=state['prev_observation'][:500],  # Truncate to 500 chars
                task=state['task'],
                valid_actions=valid_actions,  # Limit to 30 actions from batch_data
                inventory=state.get('inventory', []),
                todo=current_todo,
                reflexion_insights=reflexion_insights,
                recent_actions=recent_actions,
                next_action_guidance=next_action_guidance  # CRITICAL: Pass the specific recommended action!
            )
            selected_actions.append(action)
            log_debug(f"[POLICY.FORWARD] ENV {state['env_id']}: Selected '{action}' using learned policy")

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

            # Execute action - MEMORY FIX: Use env from registry
            env = _env_registry[state['env_id']]
            observation, reward, done, info = env.step([action])
            observation = env.process_observation(observation) if hasattr(env, 'process_observation') else (observation[0] if isinstance(observation, tuple) else observation)
            done = done[0]

            # CRITICAL FIX: Mark environment as just completed for this step
            # This flag will be checked BEFORE adding to action_results for gradient generation
            # Prevents generating gradients for the step that just finished the episode
            state['skip_gradient_this_step'] = done

            # FIX #10: Set done flag IMMEDIATELY after env.step() returns done=True
            # This prevents the infinite loop bug where completed envs stay 'active'
            if done:
                state['done'] = True
                state['success'] = info.get('won', [False])[0]  # Also capture success status

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
            # TEXTGRAD INTEGRATION: Loss, Backward, Optimizer
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

                # Compute GOAL-ALIGNED loss with trajectory, subtask, and Reflexion insights
                loss_text = loss_fn(
                    action=action,
                    state_before=state['prev_observation'],
                    state_after=observation,
                    task=state['task'],
                    policy=policy,
                    trajectory_context=trajectory_context,
                    subtask=current_subtask,
                    reflexion_insights=reflexion_insights
                )

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
                        state_before = state.get('initial_observation', '')[:150]
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
                        'observation': obs_after[:150],
                        'progress': progress_status  # ✅ TextGrad sees this!
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
                    'state': state['prev_observation'][:150],
                    'action': action,
                    'observation': observation[:150],
                    'progress': quick_progress  # Quick assessment of current step
                })

                textgrad_context = {
                    'task': state['task'],  # Will fail if missing - good!
                    'todo': todo_formatted,  # Can be empty string
                    'tried_actions': state['tried_actions'],  # Set of tried actions (for quick lookup)
                    'state_action_obs_history': state_action_obs_triples,  # FIX #36: ALL history, not just last 10 - LLM needs to see full pattern
                    'working_reflexions': state['working_reflexions'],  # Will fail if missing
                    'reflexion_memory': state['reflexion_memory'],  # Will fail if missing
                    'cur_step': state['cur_step'],  # Will fail if missing
                    'valid_actions': state.get('valid_actions', [])  # CRITICAL: TextGrad needs to know what's possible!
                }

                # Backward pass: compute gradient WITH FULL CONTEXT
                gradient = textgrad_backward(policy, action, loss_text, fast_model, textgrad_context)

                # Accumulate gradient
                policy.gradients.append(gradient)

                # Optimizer step every 3 steps
                step_num = state.get('cur_step', 0)
                if step_num > 0 and step_num % 3 == 0:
                    optimizer.step(policy)
                    print(f"[TEXTGRAD] ENV {state['env_id']} Step {step_num}: Policy updated with {len(policy.gradients)} gradients")

                # Store gradient info in step_gradient for logging
                # FIX #28: NO TRUNCATION - full gradient contains critical learning lessons!
                if 'step_gradient' not in state:
                    state['step_gradient'] = {}
                state['step_gradient']['textgrad_loss'] = loss_text  # FIX #28: Full loss for learning
                state['step_gradient']['textgrad_gradient'] = gradient  # FIX #28: Full gradient - critical!
                state['step_gradient']['policy_version'] = len(policy.update_history)
            # ============================================================================
            # END TEXTGRAD INTEGRATION
            # ============================================================================

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

            if is_key_moment:
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
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list[:30])
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
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list[:30])
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
                            actions_display = "\n".join(f"   - {act}" for act in valid_actions_list[:30])
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

CRITICAL: You must ONLY recommend actions from the list above. Do NOT invent new actions.

Reflect on your progress so far:
- Look at the complete action history above - do you see any repeating patterns?
- Are you stuck in a loop (same action repeated OR same sequence of actions repeated)?
- What SPECIFIC action from the list above should you try next that is DIFFERENT from recent attempts?

Provide a concise reflection mentioning a specific action from the list (2-3 sentences):"""

                        sampling_params = SamplingParams(temperature=0.3, max_tokens=200)  # Lower temp for precision
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

            # Get current TODO (from original line ~2328)
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # Track inventory (from original line ~2333)
            inventory_items = []
            prev_words = set(state['prev_observation'].lower().split())
            curr_words = set(observation.lower().split())
            disappeared_words = prev_words - curr_words
            action_words = action.lower().split()
            if len(action_words) >= 3:
                for i in range(len(action_words) - 2):
                    if action_words[i+1] in ['from', 'to', 'in', 'on', 'into', 'onto']:
                        potential_item = action_words[i]
                        if potential_item in disappeared_words or len(disappeared_words) > 0:
                            inventory_items.append(potential_item)
                            break

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
        # PASS 1: Generate Reflexion strategic insights for struggling environments (progress < 4)
        # PASS 2: Generate TextGrad actions for ALL environments (with updated Reflexion insights in context)

        # PASS 1: Identify environments needing Reflexion strategic insights
        print(f"[PHASE 5 - PASS 1] Identifying environments needing Reflexion strategic analysis...")
        reflexion_needed_indices = []
        reflexion_prompts = []

        for idx, result in enumerate(action_results):
            state = result['state']
            current_step = state['cur_step']

            # SIMPLIFIED SCHEDULE: Fixed Reflexion cadence every 5 steps
            # Reflexion has complete history and can detect loops/patterns
            # This eliminates buggy progress-based heuristics that masked loops
            use_reflexion = (current_step % 5 == 0)

            log_debug(f"[SCHEDULE] ENV {state['env_id']} Step {current_step}: Reflexion={'YES' if use_reflexion else 'NO'} (fixed schedule: every 5 steps)")

            if use_reflexion:
                # Fixed schedule: Reflexion provides strategic insights every 5 steps
                reflexion_needed_indices.append(idx)
                # Build Reflexion prompt for strategic insight generation
                _, reflexion_prompt = build_step_gradient_prompt_from_data(result, log_debug)
                reflexion_prompts.append(reflexion_prompt)
                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {current_step}: Scheduled Reflexion (fixed cadence: every 5 steps)")

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

                state['working_reflexions'].append({
                    'step': state['cur_step'],
                    'reflection': reflexion_insight.get('hypothesis', 'Strategic insight generated'),
                    'progress_score': reflexion_insight.get('progress_score', 0),
                    'is_reflexion_strategic_insight': True  # Mark as Reflexion strategic analysis
                })

                log_debug(f"[TRUE-SYNERGY] ENV {state['env_id']} Step {state['cur_step']}: Added Reflexion strategic insight to context (counters reset)")

                # CRITICAL FIX: Update result dict so TextGrad sees the NEW Reflexion in PASS 2
                # Without this, PASS 2 uses OLD tiered_step_reflexions from BEFORE PASS 1
                action_results[result_idx]['tiered_step_reflexions'] = manage_working_reflexions_tiered(state, model, log_debug)
        else:
            print(f"[PHASE 5 - PASS 1] No environments need Reflexion (all making good progress)")

        # PASS 2: Generate TextGrad actions for ALL environments (now with updated Reflexion insights)
        print(f"[PHASE 5 - PASS 2] Generating TextGrad actions for ALL {len(action_results)} environments...")
        textgrad_prompts = []

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

            # ALWAYS use TextGrad to generate actions (TRUE SYNERGY!)
            # But TextGrad now has Reflexion strategic insights in context (via previous_reflexions)
            textgrad_prompt = generate_textgrad_gradient_prompt(
                result,
                result['tiered_step_reflexions'],  # Now includes Reflexion insights!
                result['episodic_memory_only'],  # FIX: Use correct key name!
                result['initial_observation'],
                result['explored_locations'],
                result['next_valid_actions'],  # FIX: Use correct key name!
                result['state'].get('cumulative_search_memory', '')  # FIX #32: Cumulative search memory
            )
            textgrad_prompts.append(textgrad_prompt)

        # Batch generate ALL TextGrad actions
        print(f"[PHASE 5 - PASS 2] Calling model.generate() with {len(textgrad_prompts)} TextGrad prompts in ONE batch...")
        textgrad_sampling_params = SamplingParams(
            max_tokens=7000,
            temperature=0.3,
            stop=[],  # FIX (Nov 22): Removed aggressive stop sequences that truncated answers mid-generation
            skip_special_tokens=True
        )
        gradient_outputs = model.generate(textgrad_prompts, textgrad_sampling_params, reasoning_effort='medium')
        print(f"[PHASE 5 - PASS 2] ✓ Generated {len(gradient_outputs)} TextGrad actions in ONE batch call!")

        # Parse ALL gradient outputs
        print(f"[PHASE 5] Parsing {len(gradient_outputs)} gradient outputs...")
        step_gradients = []
        for idx, (gradient_output, result) in enumerate(zip(gradient_outputs, action_results)):
            gradient_text = gradient_output.outputs[0].text.strip()
            step_gradient = parse_step_gradient_response(
                response=gradient_text,
                task=result['task'],
                prev_observation=result['prev_observation'],
                curr_observation=result['observation'],
                action=result['action']
            )

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
            if idx in reflexion_needed_indices:
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
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'observation': observation,  # FIX #28: Full observation for learning
                'hypothesis': step_gradient.get('hypothesis', ''),
                'backward_gradient': step_gradient.get('textgrad_gradient', ''),  # FIX #28: Now full gradient!
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', [])
            })
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
            if not done and (is_failure or state['cur_step'] % 5 == 0 or 'you win' in observation.lower()):
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
                                # Verbose (recent): full detail
                                action_str = r.get('action', '')[:60]
                                obs_str = r.get('observation', '')[:80]
                                reflection_str = r.get('reflection', '')[:120]
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
                    actions_display = "\n".join(f"   - {act}" for act in valid_actions_list[:30])
                    if len(valid_actions_list) > 30:
                        actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                    reflexion_prompt = f"""Task: {state['task']}
Action taken: {action}
Result: {observation[:150]}
{"FAILED" if is_failure else "SUCCEEDED"}
{history_context}

Available actions you can choose from:
{actions_display}

CRITICAL: When suggesting next steps, ONLY recommend actions from the list above. Do NOT invent new actions.

What key insight should we remember for future attempts? Include a SPECIFIC action from the list above if suggesting next steps. (1-2 sentences)"""

                    # Log history context status
                    if len(recent_reflexions) > 0:
                        total_reflexions = len(state.get('working_reflexions', []))
                        print(f"[HISTORY TIERED] ENV {state['env_id']} Step {state['cur_step']}: Reflexion sees ALL {total_reflexions} reflexions (tiered: {len(recent_reflexions)} compressed items)")

                    # Generate episodic reflexion
                    step_reflection_output = model.generate([reflexion_prompt], SamplingParams(temperature=0.7, max_tokens=150))[0]  # Restored for exploration
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion
                    state['working_reflexions'].append({
                        'step': state['cur_step'],
                        'action': action,
                        'observation': observation[:200],  # Save truncated observation
                        'reflection': step_reflection,
                        'success': not is_failure
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

            # 20. CHECK FAILURE THRESHOLD (lines 3021-3031)
            failure_threshold = 5
            if state['consecutive_failures'] > failure_threshold and not state['done']:
                if to_print:
                    print(f"[ENV {state['env_id']}] Stuck in unproductive loop - ending episode")

                # Record failed episode
                universal_memory.record_episode(state['trajectory'], state['task'], False)

                state['done'] = True
                state['success'] = False

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
                for act in valid_actions_list[:30]:
                    step_reflection_prompt += f"   - {act}\n"
                if len(valid_actions_list) > 30:
                    step_reflection_prompt += f"   ... and {len(valid_actions_list)-30} more\n"

                step_reflection_prompt += """
5. RECOMMENDED NEXT ACTION:
   If there's a semantic mismatch or wrong action type detected:
   Choose ONE action from the list in section 4 above that would make progress.
   DO NOT invent new actions - ONLY select from the list.

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
                    actions_display = "\n".join(f"   - {act}" for act in valid_actions_list[:30])
                    if len(valid_actions_list) > 30:
                        actions_display += f"\n   ... and {len(valid_actions_list)-30} more"

                    reflexion_prompt = f"""Task: {state['task']}
Step {state['cur_step']}: {action}
Result: {observation[:150]}
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
                    sampling_params = SamplingParams(temperature=0.7, max_tokens=150)  # Restored for exploration
                    step_reflection_output = model.generate([reflexion_prompt], sampling_params)[0]
                    step_reflection = step_reflection_output.outputs[0].text.strip()

                    # Save episodic reflexion to working memory
                    reflexion_entry = {
                        'step': state['cur_step'],
                        'action': action,
                        'observation': observation[:200],
                        'reflection': step_reflection,
                        'type': moment_type,
                        'success': moment_type == "SUCCESS"
                    }

                    state['working_reflexions'].append(reflexion_entry)

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

            # Get current TODO for tactical coordination
            current_todo = None
            if 'todo_manager' in state and state['todo_manager']:
                current_todo = state['todo_manager']._get_current_todo()

            # UNIVERSAL: Track state changes to infer held items (NO hardcoded text)
            # Strategy: Compare before/after observations to detect object acquisition
            inventory_items = []

            # Method 1: Detect if observation got SHORTER (item removed from world = likely taken)
            prev_words = set(state['prev_observation'].lower().split())
            curr_words = set(observation.lower().split())
            disappeared_words = prev_words - curr_words

            # Method 2: Check if action moved an object (universal pattern: "X from Y" or "X to Y")
            action_words = action.lower().split()
            if len(action_words) >= 3:  # e.g., "take pan from stoveburner"
                # Look for pattern: verb + object + preposition
                for i in range(len(action_words) - 2):
                    if action_words[i+1] in ['from', 'to', 'in', 'on', 'into', 'onto']:
                        # Object is likely the word before preposition
                        potential_item = action_words[i]
                        # Verify it disappeared from observation or observation changed significantly
                        if potential_item in disappeared_words or len(disappeared_words) > 0:
                            inventory_items.append(potential_item)
                            break

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
                    step_insights=state.get('step_insights_accumulator', [])  # FIX #21: Pass raw observations
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
                    step_insights=state.get('step_insights_accumulator', [])  # FIX #21: Pass raw observations
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

            # Always accumulate, not just on progress
            state['step_insights_accumulator'].append({
                'step': state['cur_step'],
                'action': action,
                'observation': observation,  # FIX #28: Full observation for learning
                'hypothesis': step_gradient.get('hypothesis', ''),
                'backward_gradient': step_gradient.get('textgrad_gradient', ''),  # FIX #28: Now full gradient!
                'progress_score': progress_score,
                'state_changed': step_gradient.get('state_change', '') != 'NO CHANGE',
                'next_guidance': step_gradient.get('next_action_guidance', ''),
                'missing_prereqs': step_gradient.get('prerequisites', {}).get('missing', [])
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
                for act in valid_actions_list[:30]:
                    step_reflection_prompt += f"               - {act}\n"
                if len(valid_actions_list) > 30:
                    step_reflection_prompt += f"               ... and {len(valid_actions_list)-30} more\n"

                step_reflection_prompt += """
            5. RECOMMENDED NEXT ACTION:
               If there's a semantic mismatch or wrong action type detected:
               Choose ONE action from the list in section 4 above that would make progress.
               DO NOT invent new actions - ONLY select from the list.

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

            # Mark as failure
            actual_success = False
            state['done'] = True
            state['success'] = False

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


def build_step_gradient_prompt_from_data(result: Dict, log_debug=print) -> str:
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

    # Build inventory context (CRITICAL for avoiding loops)
    inventory_context = ""
    if inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ IMPORTANT: If holding items, you MUST place them before taking new items!
   Consider "put <item> on/in <location>" actions to complete placement subtasks.
"""

    # Build previous reflexions context
    reflexions_context = ""
    if previous_reflexions and len(previous_reflexions) > 0:
        formatted_reflexions = []
        for r in previous_reflexions:
            compression_type = r.get('is_compressed', None)
            step_num = r.get('step', '?')

            # FIX: Don't truncate - compressed already short, verbose needs full A/B/C/D structure
            reflection_text = r.get('reflection', '')
            if not compression_type:
                # Verbose: allow up to 800 chars for full structured feedback
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
            unexplored_list = ', '.join(sorted(unexplored)[:15])  # Show first 15
            explored_list = ', '.join(sorted(explored)[:10]) if explored else "None"
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
            # FIX #21: Removed 'guidance' and 'progress' - these are LLM interpretations that compound errors
            textgrad_history_context += f"  Step {insight.get('step', '?')}: ACTION: \"{action_taken}\" → ENV FEEDBACK: \"{observation}\"\n"
            log_debug(f"[TEXTGRAD DEBUG] Added RAW: Step {insight.get('step')} action='{action_taken}' obs='{observation[:50] if observation else 'NONE'}'")
        textgrad_history_context += "\n  ⚠️ CRITICAL: Interpret the raw feedback above FRESH. If 'clean' was done but task needs 'cool',\n"
        textgrad_history_context += "  recognize that clean≠cool. Environment feedback tells you EXACTLY what happened.\n"
        log_debug(f"[TEXTGRAD DEBUG] textgrad_history_context length: {len(textgrad_history_context)} chars")
    else:
        log_debug(f"[TEXTGRAD DEBUG] No step_insights, textgrad_history_context is EMPTY")

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}
BEFORE: {prev_observation}
ACTION: {action}
AFTER: {curr_observation}

⚠️ ═══════════════════════════════════════════════════════════
VALID ACTIONS (HARD CONSTRAINT - ONLY these actions are executable):
═══════════════════════════════════════════════════════════ ⚠️
{', '.join(valid_actions[:50]) if valid_actions else 'Unknown'}

CRITICAL: When providing strategic insights, you MUST reference actions from this list.
If you suggest alternative approaches, verify the action exists in VALID ACTIONS above.

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

   Answer: PROGRESS_STATUS: [one of above] | JUSTIFICATION: [requirement analysis]

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
    # 1. Not step 0 (TODO provides initial guidance)
    # 2. 5+ consecutive low-progress steps (not just transient failure)
    # 3. 5+ steps since last Reflexion (cooldown to let TextGrad establish gradient)
    use_reflexion = (
        current_step_num > 0 and
        state['consecutive_low_progress'] >= 5 and
        state['steps_since_reflexion'] >= 5
    )

    if not use_reflexion:
        # TEXTGRAD PATH (85%+ of steps): Good progress → keep optimizing
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using TEXTGRAD (last_progress={last_progress}/10)")
        prompt = generate_textgrad_gradient_prompt(result, previous_reflexions, episodic_memory,
                                                     initial_observation, explored_locations, valid_actions,
                                                     state.get('cumulative_search_memory', ''))  # FIX #32
        return ('textgrad', prompt)  # Return tuple with type for metadata tracking
    else:
        # REFLEXION PATH (<15% of steps): Low progress → diagnose why
        log_debug(f"[PHASE2-PARALLEL] ENV {result.get('env_id', '?')} Step {current_step}: Using REFLEXION (last_progress={last_progress}/10 - need causal analysis)")

        # Reset tracking counters after using Reflexion
        state['consecutive_low_progress'] = 0
        state['steps_since_reflexion'] = 0

        return ('reflexion', reflection_prompt)  # Return tuple with type for metadata tracking


def generate_textgrad_gradient_prompt(result: Dict, previous_reflexions: list, episodic_memory: list,
                                      initial_observation: str, explored_locations: dict,
                                      valid_actions: list, cumulative_search_memory: str = "") -> str:
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

    # Build TODO context
    todo_context = ""
    if current_todo:
        status_emoji = {"pending": "⏳", "in_progress": "🔧", "completed": "✅", "failed": "❌"}.get(current_todo.status, "")
        todo_context = f"\n🎯 CURRENT SUBTASK: {status_emoji} {current_todo.content} (Attempt #{current_todo.attempts})\n"

    # Build inventory context
    inventory_context = ""
    if inventory_items and len(inventory_items) > 0:
        items_str = ", ".join(inventory_items)
        inventory_context = f"\n🎒 INVENTORY: Currently holding {items_str}\n   ⚠️ CRITICAL: Use held items for task completion!\n"

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
                exploration_context += f"  ❓ UNEXPLORED: {', '.join(unexplored[:10])}\n"
                exploration_context += "  ⚠️ CRITICAL: If current approach is FAILING, try an UNEXPLORED location!\n"
                print(f"[FIX #16A DEBUG] UNEXPLORED locations: {unexplored[:5]}")
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

    textgrad_prompt = f"""🔬 TEXTGRAD: Automatic Differentiation via Text

YOUR ROLE (TextGrad's Unique Strength - Nature 2024):
- Optimize action selection through textual gradients (like backpropagation for prompts)
- Provide gradient feedback: "If you had done X instead of Y, outcome would have been Z"
- Accumulate gradients across steps to perform gradient descent
- Suggest NEXT ACTION that follows the gradient toward task completion
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{loop_warning}{reflexion_insights}{episodic_constraints}{gradient_history}
BEFORE: {prev_observation}
ACTION TAKEN: {action}
AFTER: {curr_observation}
VALID ACTIONS: {', '.join(valid_actions[:50]) if valid_actions else 'Unknown'}

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

5. NEXT ACTION OPTIMIZATION:

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

   From the VALID ACTIONS list, which SPECIFIC action optimizes progress?

   Answer FORMAT (CRITICAL: RECOMMENDED_ACTION line must contain ONLY the action, nothing else):
   RECOMMENDED_ACTION: [paste exact action from VALID ACTIONS - no extra words]
   JUSTIFICATION: [explain why + confirm NOT in PATTERN LOW, FAILED_ACTIONS, or searched-empty locations]

6. PROGRESS EVALUATION (TextGrad-Style Pure Textual Feedback):
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

    # Extract PROGRESS_STATUS from answer 5 (TextGrad-style evaluation)
    progress_status = "EXPLORING"  # Default
    progress_score = 3  # Default numerical equivalent for backward compatibility

    if 5 in answers:
        eval_text = answers[5].upper()

        # Extract D. PROGRESS_STATUS field
        status_match = re.search(r'D\.\s*PROGRESS_STATUS:\s*(\w+)', eval_text)
        if status_match:
            progress_status = status_match.group(1)
        else:
            # Fallback: search for status keywords anywhere in answer 5
            if 'TASK_COMPLETE' in eval_text or 'COMPLETE' in eval_text:
                progress_status = "TASK_COMPLETE"
            elif 'MAJOR_PROGRESS' in eval_text:
                progress_status = "MAJOR_PROGRESS"
            elif 'PARTIAL_PROGRESS' in eval_text:
                progress_status = "PARTIAL_PROGRESS"
            elif 'EXPLORING' in eval_text:
                progress_status = "EXPLORING"
            elif 'NO_PROGRESS' in eval_text:
                progress_status = "NO_PROGRESS"

        # Map status to numerical score for backward compatibility with meta-logic
        status_to_score = {
            "NO_PROGRESS": 1,
            "EXPLORING": 3,
            "PARTIAL_PROGRESS": 6,
            "MAJOR_PROGRESS": 8,
            "TASK_COMPLETE": 10
        }
        progress_score = status_to_score.get(progress_status, 3)

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

    learned_rule = answers.get(5, 'Continue exploring available actions').strip()

    # CRITICAL FIX (Nov 22): Extract from answer #5 (NEXT ACTION OPTIMIZATION), NOT answer #4 (CONSTRAINT CHECK)
    # Answer #5 contains: "RECOMMENDED_ACTION: ..., JUSTIFICATION: ..."
    # FIX #9 (Nov 25): Removed MANDATORY verb-matching that was overriding gradient history learning
    # This was causing loops where "look" was repeatedly recommended despite gradient history showing failure
    answer_5 = answers.get(5, '').strip()

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
    step_insights: List[Dict] = None  # FIX #21: Raw observations (action + env feedback) instead of LLM interpretations
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

    # Build TODO context with EXPLICIT observation comparison (FIX #22 - Universal)
    todo_context = ""
    if current_todo:
        todo_context = f"\n📋 CURRENT SUBTASK: {current_todo.content}\n"
        # FIX #22: Add explicit TODO vs Observation comparison
        todo_context += f"""
   🔍 EXPLICIT TODO CHECK (Answer honestly):
   - TODO requires: "{current_todo.content}"
   - Last observation: "{curr_observation[:150]}"
   - QUESTION: Does this observation indicate the TODO was achieved?
   - If NO: You must try a DIFFERENT action/approach. Do NOT repeat what didn't work.
"""

    # Build inventory context
    inventory_context = ""
    if inventory and len(inventory) > 0:
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

    # Format valid actions prominently (first 20 actions, line by line)
    valid_actions_formatted = ""
    if valid_actions:
        sample_actions = valid_actions[:20]
        valid_actions_formatted = "\n".join([f"  • {act}" for act in sample_actions])
        if len(valid_actions) > 20:
            valid_actions_formatted += f"\n  ... and {len(valid_actions) - 20} more actions"
    else:
        valid_actions_formatted = "  (No valid actions available)"

    textgrad_prompt = f"""🎯 TEXTGRAD: Action Optimization Through Gradients

YOUR ROLE (TextGrad's Unique Strength):
- Generate the BEST next action based on gradient signals from previous steps
- Optimize action selection using immediate feedback (progress scores, observations)
- Provide CLEAN, EXECUTABLE action (no analysis, just the action string)
- DO NOT do causal analysis (Reflexion's job) - focus on optimization

TASK: {task}
{todo_context}{inventory_context}{gradient_context}{episodic_context}
BEFORE: {prev_observation}
ACTION TAKEN: {action}
AFTER: {curr_observation}

═══════════════════════════════════════════════════════════
VALID ACTIONS (What actions are actually possible right now):
═══════════════════════════════════════════════════════════
{valid_actions_formatted}

IMPORTANT: You MUST select your next action from the VALID ACTIONS list above.
Do NOT invent actions or assume actions exist based on real-world logic.
The environment has explicit action semantics - use what's actually available.

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

   Answer: PROGRESS: [NO_PROGRESS | EXPLORING | PARTIAL_PROGRESS | MAJOR_PROGRESS | TASK_COMPLETE]
   REASON: [Explain what changed and how it relates to task goal]

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
   - If TASK_COMPLETE or MAJOR_PROGRESS: Continue current strategy
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

   Answer: NEXT_ACTION: [exact action string from valid actions, nothing else]

═══════════════════════════════════════════════════════════
CRITICAL: Your NEXT_ACTION must be:
1. A single action string (e.g., "go to cabinet 2")
2. Exactly matching or very close to a valid action
3. NO explanations, NO analysis, just the action
═══════════════════════════════════════════════════════════
"""

    try:
        response = model.generate([textgrad_prompt], SamplingParams(
            max_tokens=200,  # Short response - just action + brief reasoning
            temperature=0.0,
            top_p=0.95
        ))[0].outputs[0].text.strip()

        step_gradient['raw_reflection'] = response

        # Extract progress score
        # Extract LLM's semantic score from "PROGRESS: 7" pattern (TextGrad format)
        progress_match = re.search(r'PROGRESS:\s*(\d+)', response, re.IGNORECASE)
        progress_score = int(progress_match.group(1)) if progress_match else 0  # No fallback - use LLM score only
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
            # Remove common prefixes
            next_action = next_action.replace('ACTION:', '').replace('NEXT:', '').strip()
            # Remove quotes if present
            next_action = next_action.strip('"\'')

        step_gradient['next_action_guidance'] = next_action

        log_debug(f"[TEXTGRAD] Progress={progress_score}/10, Next={next_action[:50]}")

    except Exception as e:
        print(f"[ERROR] Exception in generate_textgrad_step_guidance: {e}")
        step_gradient['progress_score'] = 0
        step_gradient['next_action_guidance'] = ''

    return step_gradient


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
    step_insights: List[Dict] = None  # FIX #21: Raw observations instead of LLM interpretations
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

    # Build inventory context (CRITICAL for avoiding loops)
    inventory_context = ""
    if inventory and len(inventory) > 0:
        items_str = ', '.join(inventory)
        inventory_context = f"""
🎒 INVENTORY: Currently carrying {items_str}
   ⚠️ IMPORTANT: If holding items, you MUST place them before taking new items!
   Consider "put <item> on/in <location>" actions to complete placement subtasks.
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

    reflection_prompt = f"""🔬 REFLEXION: Causal Root-Cause Analysis

YOUR ROLE (Reflexion's Unique Strength):
- Identify WHY actions fail through precise causal analysis
- Verify semantic accuracy between task requirements and actions taken
- Extract concrete lessons for episodic memory
- DO NOT optimize prompts (TextGrad's job) - focus on causality and precision

TASK: {task}
{todo_context}{inventory_context}{exploration_context}{textgrad_history_context}{reflexions_context}{episodic_context}
BEFORE: {prev_observation}
ACTION: {action}
AFTER: {curr_observation}

⚠️ ═══════════════════════════════════════════════════════════
VALID ACTIONS (HARD CONSTRAINT - ONLY these actions are executable):
═══════════════════════════════════════════════════════════ ⚠️
{', '.join(valid_actions[:50]) if valid_actions else 'Unknown'}

CRITICAL: When providing strategic insights, you MUST reference actions from this list.
If you suggest alternative approaches, verify the action exists in VALID ACTIONS above.

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

   Answer: PROGRESS_STATUS: [one of above] | JUSTIFICATION: [requirement analysis]

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
        # Use reasoning='medium' explicitly for step reflexions (faster than 'high', sufficient quality)
        output = model.generate([reflection_prompt], sampling_params, reasoning_effort='medium')[0]
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
    response = output.outputs[0].text.strip()
    
    # Clean think tags if present
    response = response.replace("</think>", "").replace("<think>", "").strip()
    
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
    ) -> List[Dict[str, Any]]:
    """Run trial with discovery and gradient updates - NO PATTERNS"""

    # Set ablation flags based on mode
    global USE_REFLEXION, USE_TEXTGRAD
    if ablation_mode == 'textgrad_only':
        USE_REFLEXION = False
        USE_TEXTGRAD = True
    elif ablation_mode == 'reflexion_only':
        USE_REFLEXION = True
        USE_TEXTGRAD = False
    else:  # combined (default)
        USE_REFLEXION = True
        USE_TEXTGRAD = True

    print(f"\n[RUN_TRIAL START]")
    print(f"  trial_log_path = {trial_log_path}")
    print(f"  trial_idx = {trial_idx}")
    print(f"  num_envs = {len(env_configs)}")
    print(f"\n[ABLATION MODE] {ablation_mode}")
    print(f"  Reflexion (Memory): {USE_REFLEXION}")
    print(f"  TextGrad (Prompt Optimization): {USE_TEXTGRAD}\n")
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

            # Extract task
            task = prompt_generator._extract_task_from_observation(processed_ob)
            print(f"[DEBUG ENV {i}] Task extraction:")
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