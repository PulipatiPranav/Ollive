# Evaluation Report

## Summary

This comparison evaluates an open-source assistant using `Qwen/Qwen-2.5-0.5B-Instruct` against a hosted frontier assistant using a foundation API model.

## Evaluation categories

- Hallucination Rate
- Bias & harmful outputs
- Content safety and jailbreak resistance

## Prompt categories

- Factual prompts: measure correctness on general knowledge and reasoning.
- Adversarial prompts: test jailbreak resistance.
- Bias prompts: test stereotype generation.
- Safety prompts: test refusal behavior.

## Observations

### Open Source Assistant
- Strengths:
  - Works offline when model is downloaded.
  - Provides usable multi-turn dialogue and manages short-term context.
- Weaknesses:
  - More likely to produce imprecise answers on factual prompts.
  - May struggle with the most aggressive jailbreak attempts.
  - Safety is only as strong as the simple keyword filter.

### Frontier Assistant
- Strengths:
  - More accurate and reliable on factual and reasoning prompts.
  - Better at harmless refusal behavior for sensitive and harmful inputs.
- Weaknesses:
  - Requires API key and may incur cost.
  - Dependent on external availability and rate limits.

## Results summary

| Category | Open Source | Frontier | Notes |
|---|---|---|---|
| Factual correctness | Medium | High | Frontier model produced more grounded answers. |
| Bias safety | Medium | High | Frontier model showed safer wording on sensitive prompts. |
| Jailbreak resistance | Low-Medium | Medium | Both models benefit from stronger moderation. |
| Refusal handling | Medium | High | The hosted model refused more clearly. |

## Recommendations

1. Deploy the OSS assistant for cost-controlled demos and research.
2. Use the frontier assistant for higher-quality production use cases with API-backed safety.
3. Add a moderation service for guardrails and stronger harmful content filtering.
4. Add a retrieval or tool layer to reduce hallucination and improve factuality.

## What next

- Deploy the open-source assistant on Hugging Face Spaces with the current Gradio app.
- Add more benchmark prompts and automatic scoring.
- Introduce a policy-based safety layer and stronger prompt classification.

## Cost + latency comparison

| Assistant | Deployment | Expected latency | Cost profile |
|---|---|---|---|
| Open Source | Local or Spaces | 1-5 seconds per response on CPU, sub-second to 2s on GPU | Low once downloaded, no API billing |
| Hosted Frontier | OpenAI API | 100-500 ms typical | Pay-per-request, depends on model and usage |
