"""OpenAI GPT-5 (Reasoning) model wrapper — single-model (no fallback)"""
import os
from typing import List, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor  # ADD THIS IMPORT
from dotenv import load_dotenv

# CRITICAL FIX: Load .env BEFORE initializing client
load_dotenv()

# Init client — expects OPENAI_API_KEY in env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))

# -------------------- Compatibility shims --------------------
class MockCompletion:
    def __init__(self, text: str):
        self.text = text

class MockOutput:
    def __init__(self, text: str):
        # Many Reflexion loops expect `.outputs[0].text`
        self.outputs = [MockCompletion(text)]

class SamplingParams:
    """
    Minimal compatibility layer for your pipeline.
    Note: For GPT-5 reasoning via Responses API, use `max_output_tokens`.
    Any fields like temperature/top_p/stop are accepted here for compatibility
    but are IGNORED when calling the API (they're unsupported for reasoning models).
    """
    def __init__(self, max_tokens: int = 3000, **kwargs):
        self.max_tokens = max_tokens
        # keep any extra kwargs for caller compatibility, but we won't send them

# -------------------- Wrapper --------------------
class OpenAIModelWrapper:
    def __init__(self, model_name: str = "gpt-5"):
        self.model_name = model_name
        print(f"Using OpenAI model (Responses API, reasoning=configurable): {self.model_name}")
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
    
    def _extract_text(self, resp) -> str:
        # Prefer convenience property if present
        if hasattr(resp, "output_text") and resp.output_text is not None and len(str(resp.output_text).strip()) > 0:
            return resp.output_text
        # Fallback parse of Responses API structure
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                    chunks.append(part.text)
        return "".join(chunks)
    
    def _one_call(self, prompt: str, params: Optional[SamplingParams], reasoning_effort: str = "medium") -> str:
        import time
        max_retries = 3

        # GPT-5 Responses API - supports multiple reasoning effort levels
        # reasoning='minimal': Few or no reasoning tokens (FAST - for action selection)
        # reasoning='low': Low reasoning (faster)
        # reasoning='medium': Medium reasoning (balanced)
        # reasoning='high': High reasoning (slower, more thorough)

        # Set minimum tokens based on reasoning effort
        effort_minimums = {"minimal": 1, "low": 1, "medium": 1000, "high": 5000}
        min_tokens = effort_minimums.get(reasoning_effort, 1000)

        max_output_tokens = 8000  # Increased to 8000 to accommodate 7000 token requests
        if params is not None:
            max_output_tokens = max(min_tokens, int(getattr(params, "max_tokens", max_output_tokens)))

        # Retry logic for empty responses and API failures
        for attempt in range(max_retries):
            try:
                # No temperature/top_p/stop/penalties (unsupported for reasoning models)
                resp = client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": prompt}],
                    reasoning={"effort": reasoning_effort},  # Configurable: 'low', 'medium', or 'high'
                    max_output_tokens=max_output_tokens,
                )

                # Extract text
                result = self._extract_text(resp)

                # CRITICAL: Retry if response is empty (API glitch/timeout)
                if not result or len(result.strip()) == 0:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"[WARNING] Empty response from API (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"API returned empty response after {max_retries} attempts")

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[WARNING] API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"OpenAI Responses API call failed for '{self.model_name}' after {max_retries} attempts: {e}") from e

        return ""  # Should never reach here
    
    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, reasoning_effort: str = "medium"):
        """PARALLEL VERSION - supports all reasoning effort levels (minimal, low, medium, high)"""
        import time
        start = time.time()

        # Define function for parallel execution
        def process_prompt(prompt):
            return self._one_call(prompt, sampling_params, reasoning_effort)

        # Execute all prompts in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            texts = list(executor.map(process_prompt, prompts))

        # Convert to expected format
        outs = [MockOutput(text) for text in texts]

        # Log performance improvement with mode indicator
        elapsed = time.time() - start
        mode = "MINIMAL (fast, few reasoning tokens)" if reasoning_effort == 'minimal' else f"reasoning={reasoning_effort}"
        # FIX: Handle edge case when all environments are skipped (len(prompts) == 0)
        if len(prompts) > 0:
            print(f"[GPT-5] Processed {len(prompts)} prompts in {elapsed:.2f}s ({elapsed/len(prompts):.3f}s per prompt avg) [{mode}]")
        else:
            print(f"[GPT-5] No prompts to process (all environments completed) [{mode}]")

        return outs

# -------------------- Fast Model for Extraction --------------------
class FastModelWrapper:
    """Fast model for extraction, matching, compression (no reasoning needed)"""

    def __init__(self):
        self.model_name = "gpt-4o-mini"
        print(f"Using fast extraction model: {self.model_name}")

    def generate(self, prompts, sampling_params=None):
        import time
        start = time.time()

        def process_prompt(prompt):
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(sampling_params, 'temperature', 0.0) if sampling_params else 0.0,
                        max_tokens=getattr(sampling_params, 'max_tokens', 3000) if sampling_params else 3000,  # FIX #5 (Nov 22): Increased from 150 to 3000 to prevent TextGrad truncation
                    )
                    result = resp.choices[0].message.content

                    # Retry if empty response
                    if not result or len(result.strip()) == 0:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 1  # Shorter backoff for fast model: 1s, 2s, 3s
                            print(f"[WARNING] Fast model empty response (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(f"Fast model returned empty response after {max_retries} attempts")

                    return result

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1
                        print(f"[WARNING] Fast model call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Fast model call failed after {max_retries} attempts: {e}")

            return ""  # Should never reach here

        # Parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            texts = list(executor.map(process_prompt, prompts))

        outs = [MockOutput(text) for text in texts]
        elapsed = time.time() - start
        print(f"[FAST MODEL] {len(prompts)} prompts in {elapsed:.2f}s")
        return outs

# -------------------- Exports --------------------
fast_model = FastModelWrapper()
model = OpenAIModelWrapper(model_name="gpt-5")
LLM = OpenAIModelWrapper
print("OpenAI GPT-5 (reasoning=medium) model loaded successfully!")
print("Fast extraction model (gpt-4o-mini) loaded successfully!")