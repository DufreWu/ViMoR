import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_PATH = "decode_dataset.jsonl"
NUM_SAMPLES = 3000   # total samples
MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Prompt templates (ONLY what matters)
# =========================
NAV_TEMPLATES = [
    "A robot detects an obstacle ahead. What should it do next? Answer in 1 sentence.",
    "A robot is moving in a corridor. What is the next action? Answer briefly.",
]

REASON_TEMPLATES = [
    "A robot is navigating a cluttered environment. What should it do next? Explain step by step.",
    "A robot must avoid obstacles and continue moving. What should it do next? Provide detailed reasoning.",
]

PLAN_TEMPLATES = [
    "A robot needs to reach a target in a room with obstacles. Provide a short plan using 3 steps.",
    "A robot must navigate safely. Provide a 4-step plan.",
]

ALL_TEMPLATES = [
    ("navigation", NAV_TEMPLATES),
    ("reasoning", REASON_TEMPLATES),
    ("planning", PLAN_TEMPLATES),
]


# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    type=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
)

model.eval()


# =========================
# Generate response
# =========================
@torch.no_grad()
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =========================
# Token length
# =========================
def token_len(text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# =========================
# Dataset generation
# =========================
def main():
    out = Path(OUTPUT_PATH).open("w", encoding="utf-8")

    for i in range(NUM_SAMPLES):
        category, templates = random.choice(ALL_TEMPLATES)
        prompt = random.choice(templates)

        response = generate(prompt)

        sample = {
            "category": category,
            "prompt": prompt,
            "response": response,
            "prefill_length": token_len(prompt),
            "decode_length": token_len(response),
        }

        out.write(json.dumps(sample) + "\n")

        if i % 100 == 0:
            print(f"{i}/{NUM_SAMPLES}")

    out.close()
    print("Done.")


if __name__ == "__main__":
    main()