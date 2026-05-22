import json
import os
from datetime import datetime
from typing import Any, Dict, List

from assistant import FrontierAssistant, OpenSourceAssistant, is_harmful_text

OUTPUT_FILE = "evaluation_results.json"
PROMPTS_FILE = "prompts.json"


def load_prompts() -> List[Dict[str, Any]]:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_eval():
    prompts = load_prompts()
    oss = OpenSourceAssistant()
    frontier = None
    if os.getenv("OPENAI_API_KEY"):
        frontier = FrontierAssistant()
    results = []
    for item in prompts:
        question = item["prompt"]
        metadata = {
            "id": item["id"],
            "category": item["category"],
            "intent": item.get("intent", ""),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        history = []
        oss_answer = oss.generate(history, question)
        frontier_answer = frontier.generate(history, question) if frontier else "OPENAI_API_KEY not configured"
        results.append(
            {
                **metadata,
                "prompt": question,
                "oss_answer": oss_answer,
                "frontier_answer": frontier_answer,
                "harmful_prompt": bool(is_harmful_text(question)),
            }
        )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_eval()
