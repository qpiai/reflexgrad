"""OpenAI GPT-5 (Reasoning) model wrapper — single-model (no fallback)"""
import os
from typing import List, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor  # ADD THIS IMPORT
from dotenv import load_dotenv

# CRITICAL FIX: Load .env BEFORE initializing client
load_dotenv()

# Init client — expects OPENAI_API_KEY in env
# Universal timeout: 120s connect+read, with explicit retries=0 to prevent stale connection hangs
# Local vLLM servers may close idle connections — without retries, the SDK retries on the dead
# socket causing infinite hangs in CLOSE_WAIT state
import httpx as _httpx_init
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "dummy_key"),
    timeout=_httpx_init.Timeout(120.0, connect=10.0, read=120.0),
    max_retries=2,  # Bounded retries — prevents hung CLOSE-WAIT sockets
    http_client=_httpx_init.Client(
        timeout=_httpx_init.Timeout(120.0, connect=10.0, read=120.0),
        limits=_httpx_init.Limits(max_keepalive_connections=0, max_connections=10),
    ),
)

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

    `json_schema`: optional JSON-Schema object for constrained decoding. When
    set, vLLM/OpenAI Chat Completions enforces it via `response_format` so the
    LLM emits only schema-conformant output. Universal — schema is a type
    spec, NOT a few-shot example.
    """
    def __init__(self, max_tokens: int = 3000, json_schema=None, **kwargs):
        self.max_tokens = max_tokens
        self.json_schema = json_schema
        # keep any extra kwargs for caller compatibility, but we won't send them

# -------------------- Wrapper --------------------
class OpenAIModelWrapper:
    def __init__(self, model_name: str = "gpt-5"):
        self.model_name = model_name
        # gpt-4o, gpt-4o-mini etc use chat.completions API (no reasoning)
        # gpt-5, gpt-5.4 etc use responses API (with reasoning)
        self._use_responses_api = model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3")
        api_type = "Responses API (reasoning)" if self._use_responses_api else "Chat Completions API"
        # Adaptive context cap for local models
        _is_local = any(x in model_name.lower() for x in ["gemma", "opencua", "qwen", "llama", "intern", "phi", "/model"])
        self.max_context_chars = int(os.getenv("REFLEXGRAD_MAX_CONTEXT_CHARS", "50000")) if _is_local else 450000
        print(f"Using OpenAI model ({api_type}): {self.model_name} (max_context={self.max_context_chars})")
        if not client.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")

    def _truncate_prompt(self, prompt: str) -> str:
        """Adaptive smart truncation: keep beginning (task/instructions) and end (recent context).
        Preserves the most recent 2/3 since recent context is most important for action selection."""
        if len(prompt) <= self.max_context_chars:
            return prompt
        keep_chars = self.max_context_chars - 500
        keep_start = keep_chars // 3
        keep_end = keep_chars * 2 // 3
        truncation_msg = f"\n\n[... TRUNCATED {len(prompt) - keep_chars:,} chars to fit context limit ...]\n\n"
        return prompt[:keep_start] + truncation_msg + prompt[-keep_end:]
    
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
        max_retries = 5  # FIX #32: Increased from 3 to 5 for better resilience

        # Adaptive truncation for local models with limited context
        original_len = len(prompt)
        prompt = self._truncate_prompt(prompt)
        if len(prompt) < original_len:
            print(f"[shared_model] Truncated prompt: {original_len:,} → {len(prompt):,} chars (model={self.model_name})")

        # GPT-5 Responses API - supports multiple reasoning effort levels
        # reasoning='minimal': Few or no reasoning tokens (FAST - for action selection)
        # reasoning='low': Low reasoning (faster)
        # reasoning='medium': Medium reasoning (balanced)
        # reasoning='high': High reasoning (slower, more thorough)

        # Set minimum tokens based on reasoning effort
        effort_minimums = {"minimal": 1, "low": 1, "medium": 1000, "high": 5000}
        min_tokens = effort_minimums.get(reasoning_effort, 1000)

        # Adaptive output budget — local models have less context
        _is_local = any(x in self.model_name.lower() for x in ["gemma", "opencua", "qwen", "llama", "intern", "phi", "/model"])
        max_output_tokens = 3000 if _is_local else 8000
        if params is not None:
            max_output_tokens = max(min_tokens, int(getattr(params, "max_tokens", max_output_tokens)))
            if _is_local:
                max_output_tokens = min(max_output_tokens, 3000)  # Cap for local models

        # Retry logic for empty responses and API failures
        for attempt in range(max_retries):
            try:
                if self._use_responses_api:
                    # GPT-5/o1/o3: Responses API with reasoning
                    resp = client.responses.create(
                        model=self.model_name,
                        input=[{"role": "user", "content": prompt}],
                        reasoning={"effort": reasoning_effort},
                        max_output_tokens=max_output_tokens,
                    )
                    result = self._extract_text(resp)
                else:
                    # GPT-4o/4o-mini: Chat Completions API (no reasoning).
                    # Constrained decoding: pass JSON schema via response_format
                    # when the caller supplies one via SamplingParams. vLLM
                    # accepts OpenAI's structured-outputs format natively.
                    cc_kwargs = dict(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=min(max_output_tokens, 3000),
                        temperature=0.2,
                    )
                    _schema = getattr(params, "json_schema", None) if params else None
                    if _schema:
                        cc_kwargs["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "action",
                                "schema": _schema,
                                "strict": True,
                            },
                        }
                    resp = client.chat.completions.create(**cc_kwargs)
                    result = resp.choices[0].message.content if resp.choices else ""

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
                    # FIX #32: Longer backoff for GPT-5: 5s, 10s, 15s, 20s
                    wait_time = (attempt + 1) * 5
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
            print(f"[{self.model_name}] Processed {len(prompts)} prompts in {elapsed:.2f}s ({elapsed/len(prompts):.3f}s per prompt avg) [{mode}]")
        else:
            print(f"[{self.model_name}] No prompts to process (all environments completed) [{mode}]")

        return outs

    def generate_multimodal(self, prompt: str, before_image_b64: str = None,
                            after_image_b64: str = None,
                            max_tokens: int = 3000, temperature: float = 0.0,
                            json_schema: Optional[dict] = None) -> str:
        """Multimodal generation with before/after screenshots.

        Routes to GPT-4o via Chat Completions (Responses API doesn't support images).
        Falls back to text-only GPT-5 reasoning when no images provided.

        json_schema: optional JSON Schema for constrained decoding via
        OpenAI/vLLM response_format. Universal — same schema regardless
        of whether images are present.
        """
        # Truncate prompt to fit context budget — critical for Gemma (65K token window).
        # The regular generate() path already applies _truncate_prompt; multimodal
        # needs it too or it will blow the context with long trajectory histories.
        prompt = self._truncate_prompt(prompt)
        if not before_image_b64 and not after_image_b64:
            # Pass json_schema through SamplingParams for constrained decoding
            sp = SamplingParams(max_tokens=max_tokens, json_schema=json_schema) if json_schema else None
            result = self.generate([prompt], sampling_params=sp)[0]
            return result.outputs[0].text

        content = []
        content.append({"type": "text", "text": prompt})
        if before_image_b64:
            content.append({"type": "text", "text": "BEFORE screenshot:"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{before_image_b64}", "detail": "high"
            }})
        if after_image_b64:
            content.append({"type": "text", "text": "AFTER screenshot (current):"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{after_image_b64}", "detail": "high"
            }})

        import time
        max_retries = 3
        _vision_model = os.getenv("REFLEXGRAD_MODEL", "gpt-4o")
        for attempt in range(max_retries):
            try:
                cc_kwargs = dict(
                    model=_vision_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if json_schema:
                    cc_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "action",
                            "schema": json_schema,
                            "strict": True,
                        },
                    }
                resp = client.chat.completions.create(**cc_kwargs)
                result = resp.choices[0].message.content
                if result and len(result.strip()) > 0:
                    print(f"[MULTIMODAL] {_vision_model} vision response ({len(result)} chars)")
                    return result
                if attempt < max_retries - 1:
                    time.sleep(attempt + 1)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[MULTIMODAL] {_vision_model} call failed (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(attempt + 1)
                else:
                    print(f"[MULTIMODAL] {_vision_model} failed after {max_retries} attempts, falling back to text-only")
                    sp = SamplingParams(max_tokens=max_tokens, json_schema=json_schema) if json_schema else None
                    result = self.generate([prompt], sampling_params=sp)[0]
                    return result.outputs[0].text
        return ""

# -------------------- Fast Model for Extraction --------------------
class FastModelWrapper:
    """Fast model for extraction, matching, compression (no reasoning needed)"""

    def __init__(self):
        self.model_name = os.getenv("REFLEXGRAD_MODEL", "gpt-4o")
        # Adaptive context cap: small for local models, large for cloud reasoning models
        # Local models (gemma, opencua, qwen, llama) typically have 8K-32K context
        # 1 token ≈ 4 chars; reserve 3000 tokens for output → input limit
        _is_local = any(x in self.model_name.lower() for x in ["gemma", "opencua", "qwen", "llama", "intern", "phi", "/model"])
        if _is_local:
            # 16K context model: leave 3000 tokens for output → 13K tokens input → 50K chars
            self.max_context_chars = int(os.getenv("REFLEXGRAD_MAX_CONTEXT_CHARS", "50000"))
        else:
            self.max_context_chars = 450000  # ~112K tokens for cloud models
        print(f"Using fast extraction model: {self.model_name} (max_context_chars={self.max_context_chars}, local={_is_local})")

    def _truncate_prompt(self, prompt: str) -> str:
        """Smart truncation: keep beginning (task/instructions) and end (recent context)"""
        if len(prompt) <= self.max_context_chars:
            return prompt

        # Calculate how much to keep
        keep_chars = self.max_context_chars - 500  # Buffer for truncation message
        keep_start = keep_chars // 3  # 1/3 from beginning (task, instructions)
        keep_end = keep_chars * 2 // 3  # 2/3 from end (recent context, most relevant)

        truncation_msg = f"\n\n[... TRUNCATED {len(prompt) - keep_chars:,} chars to fit context limit ...]\n\n"

        truncated = prompt[:keep_start] + truncation_msg + prompt[-keep_end:]
        print(f"[FIX #31] Truncated prompt from {len(prompt):,} to {len(truncated):,} chars (saved recent {keep_end:,} chars)")
        return truncated

    def generate(self, prompts, sampling_params=None):
        import time
        start = time.time()

        def process_prompt(prompt):
            # FIX #31: Smart truncation for context overflow
            prompt = self._truncate_prompt(prompt)
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

    def generate_multimodal(self, prompt: str, before_image_b64: str = None,
                            after_image_b64: str = None,
                            max_tokens: int = 3000, temperature: float = 0.0) -> str:
        """Multimodal generation with before/after screenshots via GPT-4o.

        Falls back to text-only gpt-4o-mini if no images provided.
        """
        # Truncate to fit context budget (critical for Gemma's 65K window).
        prompt = self._truncate_prompt(prompt)
        if not before_image_b64 and not after_image_b64:
            result = self.generate([prompt])[0]
            return result.outputs[0].text

        content = []
        content.append({"type": "text", "text": prompt})
        if before_image_b64:
            content.append({"type": "text", "text": "BEFORE screenshot:"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{before_image_b64}", "detail": "high"
            }})
        if after_image_b64:
            content.append({"type": "text", "text": "AFTER screenshot (current):"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{after_image_b64}", "detail": "high"
            }})

        import time
        max_retries = 3
        _vision_model = os.getenv("REFLEXGRAD_MODEL", "gpt-4o")
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=_vision_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = resp.choices[0].message.content
                if result and len(result.strip()) > 0:
                    print(f"[MULTIMODAL] {_vision_model} vision response ({len(result)} chars)")
                    return result
                if attempt < max_retries - 1:
                    time.sleep(attempt + 1)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[MULTIMODAL] {_vision_model} call failed (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(attempt + 1)
                else:
                    print(f"[MULTIMODAL] {_vision_model} failed after {max_retries} attempts, falling back to text-only")
                    result = self.generate([prompt])[0]
                    return result.outputs[0].text
        return ""

# -------------------- Exports --------------------
fast_model = FastModelWrapper()
_default_model = os.getenv("REFLEXGRAD_MODEL", "gpt-5")
model = OpenAIModelWrapper(model_name=_default_model)
LLM = OpenAIModelWrapper
print(f"OpenAI {_default_model} (reasoning=medium) model loaded successfully!")
print(f"Fast extraction model ({fast_model.model_name}, 128K context with smart truncation) loaded successfully!")