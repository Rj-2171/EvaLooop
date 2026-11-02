"""Local open-weight chat models (free, no API) with greedy batched decoding.

Prompts go through the model's own chat template. The upstream VllmLLM formats chats as
plain "System:/User:/Assistant:" text, which instruct models are not trained on.

engine="vllm": fast batched inference (needs a vLLM build that supports the GPU).
engine="hf":   transformers generate with left padding; slower but runs anywhere torch does.
"""

import logging
from typing import Any, Dict, List

from .base import BaseLLM


class LocalChatLLM(BaseLLM):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        import torch
        from transformers import AutoTokenizer

        self.logger = logging.getLogger(__name__)
        self.path = model_config["path"]
        self.engine = model_config.get("engine", "vllm")
        self.max_new_tokens = int(model_config.get("max_new_tokens", 1024))
        self.max_model_len = int(model_config.get("max_model_len", 4096))
        self.system_prompt = model_config.get("system_prompt") or None
        self.hf_batch_size = int(model_config.get("hf_batch_size", 16))
        self.stats = {"calls": 0, "hit_max_tokens": 0, "output_tokens": 0}

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Native bf16 needs compute capability 8.0+. torch.cuda.is_bf16_supported() also counts
        # emulation and returns True on a T4 (7.5), which vLLM rejects and which is slow in HF.
        bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.dtype = "bfloat16" if bf16 else "float16"

        if self.engine == "vllm":
            from vllm import LLM, SamplingParams

            self.llm = LLM(model=self.path, dtype=self.dtype, max_model_len=self.max_model_len,
                           gpu_memory_utilization=float(model_config.get("gpu_memory_utilization", 0.85)),
                           seed=0)
            self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=self.max_new_tokens)
        elif self.engine == "hf":
            from transformers import AutoModelForCausalLM

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                self.dtype = "float32"
            dtype = getattr(torch, self.dtype)
            self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=dtype).to(device)
            self.model.eval()
        else:
            raise ValueError(f"Unknown engine '{self.engine}' (use vllm or hf)")

    def _chat(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        texts = [self._chat(p) for p in prompts]
        outs = self._vllm(texts) if self.engine == "vllm" else self._hf(texts)
        self.stats["calls"] += len(outs)
        return outs

    def _vllm(self, texts: List[str]) -> List[str]:
        results = self.llm.generate(texts, self.sampling, use_tqdm=False)
        outs = []
        for r in results:
            o = r.outputs[0]
            self.stats["output_tokens"] += len(o.token_ids)
            self.stats["hit_max_tokens"] += o.finish_reason == "length"
            outs.append(o.text)
        return outs

    def _hf(self, texts: List[str]) -> List[str]:
        import torch

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outs = [""] * len(texts)
        for start in range(0, len(order), self.hf_batch_size):
            idx = order[start:start + self.hf_batch_size]
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self.tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row, i in zip(new, idx):
                n = int((row != self.tokenizer.pad_token_id).sum())
                self.stats["output_tokens"] += n
                self.stats["hit_max_tokens"] += n >= self.max_new_tokens
                outs[i] = self.tokenizer.decode(row, skip_special_tokens=True)
        return outs

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": f"local_{self.engine}",
            "model_id": self.path.rstrip("/").split("/")[-1],
            "path": self.path,
            "dtype": self.dtype,
            "temperature": 0.0,
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": self.system_prompt,
            "usage": dict(self.stats),
        }
