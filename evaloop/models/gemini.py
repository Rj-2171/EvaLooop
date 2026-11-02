"""Gemini models through the google-genai SDK (Gemini API key or Vertex AI).

Adds batch_generate, so BatchEvaluationCycle sends the prompts of one cycle concurrently.
Requests are independent and greedy, so concurrency does not change the outputs.
"""

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .base import BaseLLM

for _name in ("google_genai", "httpx"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class GeminiLLM(BaseLLM):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        from google import genai

        self.logger = logging.getLogger(__name__)
        self.model_id = model_config["model_id"]
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.max_output_tokens = model_config.get("max_new_tokens", 1024)
        self.system_prompt = model_config.get("system_prompt") or None
        self.thinking_budget = model_config.get("thinking_budget")
        self.thinking_level = model_config.get("thinking_level")
        self.concurrency = max(1, int(model_config.get("concurrency", 8)))
        self.max_retries = int(model_config.get("max_retries", 8))

        if model_config.get("vertex"):
            self.client = genai.Client(
                vertexai=True,
                project=model_config.get("gcp_project") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=model_config.get("gcp_location") or "global",
            )
            self.backend = "vertex"
        else:
            key = model_config.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY, or pass --vertex True with a GCP project")
            self.client = genai.Client(api_key=key)
            self.backend = "gemini_api"

        self._lock = threading.Lock()
        self.stats = {
            "calls": 0,
            "failed_calls": 0,
            "empty_responses": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
        }

    def _config(self):
        from google.genai import types

        kwargs = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            system_instruction=self.system_prompt,
        )
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
        elif self.thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(**kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id, contents=prompt, config=self._config()
                )
                self._record(response)
                return response.text or ""
            except Exception as e:  # rate limits, 5xx, transient network errors
                code = getattr(e, "code", None)
                if isinstance(code, int) and 400 <= code < 500 and code not in (408, 429):
                    # Bad key, unknown model, invalid config: retrying cannot help, and silently
                    # returning "" would be scored as a model failure. Stop the run instead
                    # (SystemExit passes through the loop runner's `except Exception`).
                    raise SystemExit(f"Gemini request rejected ({code}): {e}")
                last_error = e
                wait = min(60, 2 ** attempt) + random.random()
                self.logger.warning(f"Gemini call failed ({e}); retry {attempt + 1} in {wait:.0f}s")
                time.sleep(wait)
        with self._lock:
            self.stats["failed_calls"] += 1
        self.logger.error(f"Gemini call gave up after {self.max_retries} attempts: {last_error}")
        return ""

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.generate, prompts))

    def _record(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        with self._lock:
            self.stats["calls"] += 1
            if not response.text:
                self.stats["empty_responses"] += 1
            if usage is not None:
                self.stats["prompt_tokens"] += usage.prompt_token_count or 0
                self.stats["output_tokens"] += usage.candidates_token_count or 0
                self.stats["thinking_tokens"] += getattr(usage, "thoughts_token_count", 0) or 0

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "system_prompt": self.system_prompt,
            "thinking_budget": self.thinking_budget,
            "thinking_level": self.thinking_level,
            "usage": dict(self.stats),
        }
