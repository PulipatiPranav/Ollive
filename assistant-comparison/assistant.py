import os
import re
from functools import lru_cache
from typing import List, Tuple

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline

OSS_MODEL = os.getenv("OSS_MODEL", "Qwen/Qwen-2.5-0.5B-Instruct")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HARMFUL_PATTERNS = [
    r"\bkill\b",
    r"\bterroris",
    r"\bmethod to make a bomb\b",
    r"\bsexual act\b",
    r"\bchild sexual\b",
    r"\bslur\b",
    r"\bhate speech\b",
    r"\bdiscriminate\b",
    r"\bnsfw\b",
    r"\bdrug\b",
    r"\bself[- ]harm\b",
    r"\bcommit suicide\b",
    r"\billegal\b",
]

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant with safe behavior. Keep answers concise, factual, and respectful. "
    "If a prompt is unsafe, refuse politely and do not generate dangerous or discriminatory content."
)


def is_harmful_text(text: str) -> bool:
    normalized = text.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def flatten_history(history: List[Tuple[str, str]]) -> str:
    lines = []
    for role, text in history:
        if role == "user":
            lines.append(f"User: {text}")
        else:
            lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def build_prompt(history: List[Tuple[str, str]], user_message: str) -> str:
    history_text = flatten_history(history)
    context = f"{SYSTEM_INSTRUCTIONS}\n\n{history_text}\nUser: {user_message}\nAssistant:"
    return context


@lru_cache(maxsize=1)
def load_oss_pipeline() -> TextGenerationPipeline:
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForCausalLM.from_pretrained(
        OSS_MODEL,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(OSS_MODEL, trust_remote_code=True)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


class OpenSourceAssistant:
    def __init__(self):
        self.pipeline = None

    def _ensure_ready(self):
        if self.pipeline is None:
            self.pipeline = load_oss_pipeline()

    def generate(self, history: List[Tuple[str, str]], user_message: str) -> str:
        if is_harmful_text(user_message):
            return (
                "I'm sorry, but I cannot assist with that request. "
                "Please ask me something safe and appropriate."
            )
        self._ensure_ready()
        prompt = build_prompt(history, user_message)
        output = self.pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        text = output[0]["generated_text"]
        # Strip prompt prefix if needed
        return text[len(prompt) :].strip()


class FrontierAssistant:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for the frontier assistant.")
        openai.api_key = OPENAI_API_KEY

    def generate(self, history: List[Tuple[str, str]], user_message: str) -> str:
        if is_harmful_text(user_message):
            return (
                "I'm sorry, but I cannot assist with that request. "
                "Please ask me something safe and appropriate."
            )
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        ]
        for role, text in history:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_message})

        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=320,
        )
        return response.choices[0].message.content.strip()


def summary_message() -> str:
    return (
        "This demo runs two assistant backends: an open-source Hugging Face model and a hosted frontier model. "
        "Use the OSS tab to test locally and the Frontier tab to compare against the hosted API. "
        "Keep the conversation going to observe how short-term memory works."
    )
