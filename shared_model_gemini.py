"""Google Gemini model wrapper — compatible with OpenAI structure"""
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import time

try:
    import google.generativeai as genai
except ImportError:
    raise RuntimeError("google-generativeai package not installed. Run: pip install google-generativeai")

# Init Gemini client — expects GEMINI_API_KEY in env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------- Compatibility shims (matching OpenAI structure) --------------------
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
    Maps to Gemini's generation config parameters.
    """
    def __init__(self, max_tokens: int = 3000, temperature: float = 0.7,
                 top_p: float = 0.95, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # keep any extra kwargs for caller compatibility

# -------------------- Gemini Reasoning Model Wrapper --------------------
class GeminiModelWrapper:
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """
        Available Gemini models (2025 - Latest):

        REASONING/THINKING MODELS (Recommended):
        - gemini-2.5-pro (BEST: State-of-the-art reasoning, #1 on LMArena, 1M context)
        - gemini-2.5-flash (Fast reasoning model, price-performance leader)
        - gemini-2.0-flash (Good reasoning, fastest)

        OLDER MODELS:
        - gemini-2.0-flash-thinking-exp-01-21 (older experimental)
        - gemini-1.5-pro (previous generation)
        - gemini-1.5-flash (previous generation)

        Note: Gemini 2.5 Pro is the latest model with enhanced thinking capabilities,
        comparable to GPT-5 for complex reasoning, math, and coding tasks.
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        print(f"Using Google Gemini model: {self.model_name}")
        print(f"  (Latest 2025 reasoning model with thinking capabilities)")

    def _one_call(self, prompt: str, params: Optional[SamplingParams], reasoning_effort: str = "medium") -> str:
        """
        Single Gemini API call with retry logic
        reasoning_effort: 'low', 'medium', 'high' (compatible with OpenAI parameter)
        For Gemini, we map this to temperature:
        - low: temp=0.3
        - medium: temp=0.7
        - high: temp=1.0
        """

        # Set generation config
        max_tokens = 6000
        temperature = 0.7
        top_p = 0.95

        # Map reasoning effort to temperature
        effort_to_temp = {"low": 0.3, "medium": 0.7, "high": 1.0}
        temperature = effort_to_temp.get(reasoning_effort, 0.7)

        if params is not None:
            max_tokens = int(getattr(params, "max_tokens", max_tokens))
            # Allow override via params if provided
            if hasattr(params, 'temperature'):
                temperature = float(getattr(params, "temperature", temperature))
            top_p = float(getattr(params, "top_p", top_p))

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Retry logic for empty responses and API failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Configure safety settings to be less restrictive
                safety_settings = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }

                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # Extract text from response (handle blocked responses)
                result = ""
                try:
                    # Try to get text directly (will fail if blocked by safety)
                    if hasattr(response, 'text'):
                        result = response.text
                except (ValueError, AttributeError):
                    # Response was blocked or has no text, try alternatives
                    pass

                # If still empty, try other extraction methods
                if not result:
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            result = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])

                # If still empty after all attempts, log the response structure
                if not result:
                    print(f"[DEBUG] Empty response. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'unknown') if hasattr(response, 'candidates') and response.candidates else 'no candidates'}")

                # CRITICAL: Retry if response is empty (API glitch/timeout)
                if not result or len(result.strip()) == 0:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"[WARNING] Empty response from Gemini API (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Gemini API returned empty response after {max_retries} attempts")

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[WARNING] Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Gemini API call failed for '{self.model_name}' after {max_retries} attempts: {e}") from e

        return ""  # Should never reach here

    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, reasoning_effort: str = "medium"):
        """PARALLEL VERSION - Process multiple prompts concurrently (matching OpenAI interface)"""
        start = time.time()

        # Define function for parallel execution
        def process_prompt(prompt):
            return self._one_call(prompt, sampling_params, reasoning_effort)

        # Execute all prompts in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            texts = list(executor.map(process_prompt, prompts))

        # Convert to expected format
        outs = [MockOutput(text) for text in texts]

        # Log performance improvement
        elapsed = time.time() - start
        print(f"[PARALLEL GEMINI] Processed {len(prompts)} prompts in {elapsed:.2f}s ({elapsed/len(prompts):.3f}s per prompt avg) [reasoning={reasoning_effort}]")

        return outs

# -------------------- Fast Model for Extraction --------------------
class FastGeminiWrapper:
    """Fast model for extraction, matching, compression - also has thinking capabilities!"""

    def __init__(self):
        # Use 2.0 Flash for better stability (less aggressive safety filters)
        self.model_name = "gemini-2.0-flash-exp"  # More stable for action selection
        self.model = genai.GenerativeModel(self.model_name)
        print(f"Using fast extraction model: {self.model_name}")
        print(f"  (Optimized for reliable action extraction)")

    def generate(self, prompts, sampling_params=None):
        start = time.time()

        def process_prompt(prompt):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=getattr(sampling_params, 'max_tokens', 150) if sampling_params else 150,
                        temperature=getattr(sampling_params, 'temperature', 0.0) if sampling_params else 0.0,
                    )

                    # Configure safety settings to be less restrictive
                    safety_settings = {
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    }

                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                    )

                    # Extract text from response (handle blocked responses)
                    result = ""
                    try:
                        # Try to get text directly (will fail if blocked by safety)
                        if hasattr(response, 'text'):
                            result = response.text
                    except (ValueError, AttributeError):
                        # Response was blocked or has no text, try alternatives
                        pass

                    # If still empty, try other extraction methods
                    if not result:
                        if hasattr(response, 'candidates') and response.candidates:
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                result = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])

                    # If still empty, return fallback to prevent crash
                    if not result:
                        result = "none"  # Fallback for blocked responses

                    # Retry if empty response
                    if not result or len(result.strip()) == 0:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 1  # Shorter backoff for fast model: 1s, 2s, 3s
                            print(f"[WARNING] Fast Gemini model empty response (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(f"Fast Gemini model returned empty response after {max_retries} attempts")

                    return result

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1
                        print(f"[WARNING] Fast Gemini model call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Fast Gemini model call failed after {max_retries} attempts: {e}")

            return ""  # Should never reach here

        # Parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            texts = list(executor.map(process_prompt, prompts))

        outs = [MockOutput(text) for text in texts]
        elapsed = time.time() - start
        print(f"[FAST GEMINI MODEL] {len(prompts)} prompts in {elapsed:.2f}s")
        return outs

# -------------------- Exports (matching OpenAI structure) --------------------
fast_model = FastGeminiWrapper()
model = GeminiModelWrapper(model_name="gemini-2.5-pro")
LLM = GeminiModelWrapper
print("\n" + "="*70)
print("🚀 GOOGLE GEMINI 2.5 MODELS LOADED (2025 - LATEST)")
print("="*70)
print("Main Model: gemini-2.5-pro (State-of-the-art reasoning, #1 on LMArena)")
print("Fast Model: gemini-2.5-flash (Fast reasoning for simple tasks)")
print("="*70 + "\n")
