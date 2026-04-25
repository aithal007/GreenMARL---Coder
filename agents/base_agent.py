"""
Base agent for GreenMARL-Coder.

Provides:
  - Lazy-loaded HuggingFace model and tokenizer (shared via module-level cache).
  - HF token login (reads HF_TOKEN env var set by --hf-token CLI arg or Space secret).
  - Raw logprob extraction from generate() scores.
  - Shannon entropy computation over next-token distribution (ETD gating).
  - Abstract interface: act(), observe(), sleep_or_act().
"""

from __future__ import annotations

import abc
import logging
import os
from collections import deque
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def _hf_login() -> None:
    """Login to HuggingFace Hub if HF_TOKEN env var is set."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        try:
            from huggingface_hub import login  # type: ignore
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace login successful.")
        except Exception as exc:
            logger.warning("HuggingFace login failed: %s", exc)


def _load_model(model_name: str, device: str) -> tuple[Any, Any]:
    """Return (model, tokenizer), loading from cache or HuggingFace Hub."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    _hf_login()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    hf_token = os.environ.get("HF_TOKEN") or None

    logger.info("Loading model %s on %s …", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    # Ensure pad_token is set to avoid generation warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # float16 on GPU (2x memory, same precision for inference); float32 on CPU
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
        token=hf_token,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    _MODEL_CACHE[model_name] = (model, tokenizer)
    logger.info("Model %s ready.", model_name)
    return model, tokenizer


class BaseAgent(abc.ABC):
    """
    Abstract agent wrapping a HuggingFace causal LM.

    Subclasses implement:
        act(observation) -> str
        observe(obs_dict, reward) -> None   (optional)

    ETD uncertainty interface:
        calculate_entropy(prompt) -> float
        sleep_or_act(obs, past_rewards, prompt_fn, etd_enabled) -> str
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    ENTROPY_PROBE_TOKENS = 8
    ENTROPY_THRESHOLD = 1.5
    REWARD_STABILITY_WINDOW = 3

    def __init__(
        self,
        name: str,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

        self.obs_history: list[dict] = []
        self.reward_history: deque[float] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if not self._loaded:
            self._model, self._tokenizer = _load_model(self.model_name, self.device)
            self._loaded = True

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        return_scores: bool = False,
    ) -> tuple[str, list[torch.Tensor] | None]:
        """
        Generate a completion for *prompt*.

        Returns:
            (text, scores) — scores is a list of per-step vocab logit tensors
            only when return_scores=True.
        """
        self._ensure_model()
        tok = self._tokenizer
        model = self._model

        # truncation=True prevents OOM on long prompts
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        # Move inputs to the actual device of the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        max_tok = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                output_scores=return_scores,
                return_dict_in_generate=return_scores,
                pad_token_id=tok.pad_token_id,
            )

        if return_scores:
            new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            scores = list(out.scores)
        else:
            new_ids = out[0, inputs["input_ids"].shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            scores = None

        return text, scores

    # ------------------------------------------------------------------
    # ETD uncertainty measurement
    # ------------------------------------------------------------------

    def calculate_entropy(self, prompt: str) -> float:
        """
        Mean Shannon entropy (bits) over the first ENTROPY_PROBE_TOKENS steps.
        Low entropy = model is confident = ETD gate may suppress generation.
        """
        _, scores = self.generate(
            prompt,
            max_new_tokens=self.ENTROPY_PROBE_TOKENS,
            return_scores=True,
        )
        if not scores:
            return float("inf")

        entropies: list[float] = []
        for logit_vec in scores[: self.ENTROPY_PROBE_TOKENS]:
            probs = torch.softmax(logit_vec.squeeze(0), dim=-1).clamp(min=1e-9)
            h = -torch.sum(probs * torch.log2(probs)).item()
            entropies.append(h)

        return float(sum(entropies) / len(entropies)) if entropies else float("inf")

    # ------------------------------------------------------------------
    # ETD gating
    # ------------------------------------------------------------------

    def _rewards_are_stable(self, past_rewards: list[float]) -> bool:
        if len(past_rewards) < self.REWARD_STABILITY_WINDOW:
            return False
        recent = past_rewards[-self.REWARD_STABILITY_WINDOW:]
        mean = sum(recent) / len(recent)
        variance = sum((r - mean) ** 2 for r in recent) / len(recent)
        return mean > 0.5 and variance < 0.05

    def sleep_or_act(
        self,
        observation: dict[str, Any],
        past_rewards: list[float],
        prompt_fn: "callable[[dict], str]",
        etd_enabled: bool = True,
    ) -> str:
        prompt = prompt_fn(observation)

        if etd_enabled:
            entropy = self.calculate_entropy(prompt)
            stable = self._rewards_are_stable(past_rewards)
            logger.debug("[%s] ETD entropy=%.3f stable=%s", self.name, entropy, stable)
            if entropy < self.ENTROPY_THRESHOLD and stable:
                logger.info("[%s] ETD gate: sleeping (entropy=%.3f)", self.name, entropy)
                return "sleep_token"

        text, _ = self.generate(prompt)
        return text

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def act(self, observation: dict[str, Any]) -> str:
        """Generate an action given the current observation."""

    def observe(self, obs: dict[str, Any], reward: float) -> None:
        self.obs_history.append(obs)
        self.reward_history.append(reward)

    # ------------------------------------------------------------------
    # Chat prompt builder
    # ------------------------------------------------------------------

    def build_chat_prompt(self, system: str, user: str) -> str:
        """
        Format a chat-style prompt. Uses the tokenizer's template when loaded;
        falls back to a plain-text format so tests and offline runs work
        without loading model weights.
        """
        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template") and tok.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_name!r})"
