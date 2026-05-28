"""OpenRouter model wrapper — OpenAI-compatible API with OpenRouter base URL

Optimized for DeepSeek V3.2 Speciale via OpenRouter:
- Rate-limit-aware parallelism (max 3 concurrent, staggered starts)
- Graceful empty response fallback (never crash on empty)
- Exponential backoff with jitter for 429s
"""
import os
import random
import time
from typing import List, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# CRITICAL FIX: Load .env BEFORE initializing client
load_dotenv()

# OpenRouter uses OpenAI-compatible API
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", "dummy_key"),
    base_url="https://openrouter.ai/api/v1",
    timeout=180.0,
)

# -------------------- Compatibility shims --------------------
class MockCompletion:
    def __init__(self, text: str):
        self.text = text

class MockOutput:
    def __init__(self, text: str):
        self.outputs = [MockCompletion(text)]

class SamplingParams:
    """Minimal compatibility layer."""
    def __init__(self, max_tokens: int = 3000, **kwargs):
        self.max_tokens = max_tokens

# -------------------- Config --------------------
_default_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
_fast_model_name = os.getenv("OPENROUTER_FAST_MODEL", "meta-llama/llama-3.1-70b-instruct")
# Note: Llama 3.1 70B has no hidden reasoning tokens — works with any max_tokens

# Rate limit tuning — model-specific based on testing
# DeepSeek Speciale: needs heavy throttling (hidden reasoning tokens, rate limits)
# Llama/Qwen: can handle full parallelism (no hidden tokens, better rate limits)
def _get_rate_config(model_name: str):
    """Return (max_parallel, stagger_delay) based on model."""
    if "speciale" in model_name.lower():
        return 3, 1.0    # Speciale: strict — hidden reasoning tokens + rate limits
    elif "deepseek" in model_name.lower():
        return 5, 0.5    # DeepSeek base: moderate
    else:
        return 10, 0.2   # Llama, Qwen, others: full speed

_MAX_PARALLEL_MAIN, _STAGGER_DELAY = _get_rate_config(_default_model)
_MAX_PARALLEL_FAST, _ = _get_rate_config(_fast_model_name)


# -------------------- Main Model Wrapper --------------------
class OpenRouterModelWrapper:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or _default_model
        print(f"Using OpenRouter model: {self.model_name}")
        if not client.api_key or client.api_key == "dummy_key":
            raise RuntimeError("OPENROUTER_API_KEY not set.")

    def _truncate_prompt(self, prompt: str, max_chars: int = 60000) -> str:
        """Truncate long prompts to prevent empty responses from DeepSeek Speciale.
        Speciale's internal reasoning consumes tokens proportional to prompt size.
        Prompts over ~20K chars reliably cause empty responses at lower max_tokens."""
        if len(prompt) <= max_chars:
            return prompt
        # Keep start (task/instructions) and end (recent context) — drop middle (old gradient history)
        keep_start = max_chars // 3
        keep_end = max_chars * 2 // 3
        truncation_msg = f"\n\n[... TRUNCATED {len(prompt) - max_chars:,} chars — old gradient history removed ...]\n\n"
        truncated = prompt[:keep_start] + truncation_msg + prompt[-keep_end:]
        print(f"[OPENROUTER TRUNCATION] {len(prompt):,} → {len(truncated):,} chars")
        return truncated

    def _one_call(self, prompt: str, params: Optional[SamplingParams] = None, reasoning_effort: str = "medium") -> str:
        max_retries = 8  # More retries for resilience

        # Truncate long prompts to prevent empty responses
        prompt = self._truncate_prompt(prompt)

        # Model-specific token handling:
        # - DeepSeek Speciale: hidden reasoning tokens eat max_tokens budget → need min 2000
        # - All other models (Llama, Qwen, DeepSeek base): no hidden tokens → respect caller's max_tokens
        is_speciale = "speciale" in self.model_name.lower()
        min_tokens = 2000 if is_speciale else 200  # Only Speciale needs high minimum
        default_tokens = 4000 if is_speciale else 3000
        max_cap = 16000 if is_speciale else 8000

        max_output_tokens = default_tokens
        if params is not None:
            requested = int(getattr(params, "max_tokens", max_output_tokens))
            max_output_tokens = min(max(min_tokens, requested), max_cap)

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_output_tokens,
                    temperature=0.2,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/qpiai/reflexgrad",
                        "X-Title": "ReflexGrad",
                    },
                )
                result = resp.choices[0].message.content if resp.choices else ""

                if not result or len(result.strip()) == 0:
                    if attempt < max_retries - 1:
                        # Exponential backoff + jitter for empty responses
                        wait_time = min((2 ** attempt) + random.uniform(0, 2), 60)
                        print(f"[WARNING] Empty response from OpenRouter (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # GRACEFUL FALLBACK: Return a minimal valid response instead of crashing
                        print(f"[WARNING] OpenRouter returned empty response after {max_retries} attempts — returning fallback")
                        return "[No response from model — please retry or continue with available information]"

                return result

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()

                if attempt < max_retries - 1:
                    # Longer backoff for rate limits (exponential), shorter for other errors
                    if is_rate_limit:
                        wait_time = min((2 ** (attempt + 1)) * 3 + random.uniform(0, 5), 120)
                        print(f"[RATE LIMIT] OpenRouter 429 (attempt {attempt + 1}/{max_retries}), backing off {wait_time:.1f}s...")
                    else:
                        wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                        print(f"[WARNING] OpenRouter call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # GRACEFUL FALLBACK: Don't crash the entire run
                    print(f"[ERROR] OpenRouter API failed after {max_retries} attempts: {e}")
                    print(f"[FALLBACK] Returning empty-safe response to prevent crash")
                    return "[API error — model unavailable. Continue with available information.]"

        return "[No response]"

    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, reasoning_effort: str = "medium"):
        """Rate-limit-aware parallel execution with staggered starts."""
        start = time.time()

        if len(prompts) == 0:
            print(f"[{self.model_name}] No prompts to process")
            return []

        # For single prompt, just call directly (no threading overhead)
        if len(prompts) == 1:
            text = self._one_call(prompts[0], sampling_params, reasoning_effort)
            elapsed = time.time() - start
            print(f"[{self.model_name}] Processed 1 prompt in {elapsed:.2f}s [reasoning_effort={reasoning_effort}]")
            return [MockOutput(text)]

        # Staggered parallel execution to avoid rate limit storms
        results = [None] * len(prompts)

        def process_with_stagger(idx_prompt):
            idx, prompt = idx_prompt
            # Stagger starts to avoid simultaneous rate limit hits
            stagger = idx * _STAGGER_DELAY + random.uniform(0, 0.5)
            if stagger > 0:
                time.sleep(stagger)
            return idx, self._one_call(prompt, sampling_params, reasoning_effort)

        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_MAIN) as executor:
            futures = {executor.submit(process_with_stagger, (i, p)): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    idx = futures[future]
                    print(f"[ERROR] Prompt {idx} failed: {e} — using fallback")
                    results[idx] = "[API error — continue with available information]"

        outs = [MockOutput(text or "[empty]") for text in results]

        elapsed = time.time() - start
        print(f"[{self.model_name}] Processed {len(prompts)} prompts in {elapsed:.2f}s ({elapsed/len(prompts):.3f}s per prompt avg) [reasoning_effort={reasoning_effort}]")
        return outs

    def generate_multimodal(self, prompt: str, before_image_b64: str = None,
                            after_image_b64: str = None,
                            max_tokens: int = 8000, temperature: float = 0.0) -> str:
        """Multimodal generation — DeepSeek V3.2 is text-only, falls back to text."""
        # DeepSeek V3.2 Speciale doesn't support vision — always text-only
        if before_image_b64 or after_image_b64:
            print(f"[MULTIMODAL] DeepSeek V3.2 Speciale is text-only, ignoring images")
        result = self.generate([prompt])[0]
        return result.outputs[0].text


# -------------------- Fast Model for Extraction --------------------
class FastModelWrapper:
    """Fast model for extraction, matching, compression"""

    def __init__(self):
        self.model_name = _fast_model_name
        self.max_context_chars = 450000
        print(f"Using fast extraction model (OpenRouter): {self.model_name}")

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self.max_context_chars:
            return prompt
        keep_chars = self.max_context_chars - 500
        keep_start = keep_chars // 3
        keep_end = keep_chars * 2 // 3
        truncation_msg = f"\n\n[... TRUNCATED {len(prompt) - keep_chars:,} chars to fit context limit ...]\n\n"
        truncated = prompt[:keep_start] + truncation_msg + prompt[-keep_end:]
        print(f"[TRUNCATION] Truncated prompt from {len(prompt):,} to {len(truncated):,} chars")
        return truncated

    def generate(self, prompts, sampling_params=None):
        start = time.time()

        if len(prompts) == 0:
            return []

        def process_with_stagger(idx_prompt):
            idx, prompt = idx_prompt
            # Stagger fast model calls too (lighter than main model)
            stagger = idx * 0.3 + random.uniform(0, 0.2)
            if stagger > 0:
                time.sleep(stagger)

            prompt = self._truncate_prompt(prompt)
            max_retries = 5  # Increased from 3
            for attempt in range(max_retries):
                try:
                    # DeepSeek V3.2 (base) is NOT a reasoning model, but enforce minimum just in case
                    requested_tokens = getattr(sampling_params, 'max_tokens', 3000) if sampling_params else 3000
                    safe_tokens = max(1000, requested_tokens)  # Floor at 1000 for safety
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(sampling_params, 'temperature', 0.0) if sampling_params else 0.0,
                        max_tokens=safe_tokens,
                        extra_headers={
                            "HTTP-Referer": "https://github.com/qpiai/reflexgrad",
                            "X-Title": "ReflexGrad",
                        },
                    )
                    result = resp.choices[0].message.content
                    if not result or len(result.strip()) == 0:
                        if attempt < max_retries - 1:
                            wait_time = min((2 ** attempt) + random.uniform(0, 1), 30)
                            time.sleep(wait_time)
                            continue
                        else:
                            return idx, "[Empty response from fast model]"
                    return idx, result
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = min((2 ** attempt) * 2 + random.uniform(0, 2), 60)
                        print(f"[WARNING] Fast model failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] Fast model failed after {max_retries} attempts: {e}")
                        return idx, "[Fast model API error — continue with available information]"
            return idx, "[No response]"

        # Parallel with stagger
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_FAST) as executor:
            futures = {executor.submit(process_with_stagger, (i, p)): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    idx = futures[future]
                    print(f"[ERROR] Fast prompt {idx} failed: {e}")
                    results[idx] = "[API error]"

        outs = [MockOutput(text or "[empty]") for text in results]
        elapsed = time.time() - start
        print(f"[FAST MODEL OpenRouter] {len(prompts)} prompts in {elapsed:.2f}s")
        return outs

    def generate_multimodal(self, prompt: str, before_image_b64: str = None,
                            after_image_b64: str = None,
                            max_tokens: int = 3000, temperature: float = 0.0) -> str:
        """Text-only fallback for DeepSeek."""
        result = self.generate([prompt])[0]
        return result.outputs[0].text


# -------------------- Exports --------------------
fast_model = FastModelWrapper()
model = OpenRouterModelWrapper(model_name=_default_model)
LLM = OpenRouterModelWrapper
print(f"OpenRouter {_default_model} model loaded successfully!")
print(f"Fast extraction model ({fast_model.model_name}) loaded successfully!")
