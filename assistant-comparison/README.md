# Assistant Comparison

A lightweight Gradio demo comparing an open-source assistant built on Hugging Face and a hosted frontier assistant using a foundation model API.

## Project structure

- `app.py` - Gradio interface for dual assistant comparison
- `assistant.py` - Open-source and frontier assistant wrappers
- `evaluation.py` - Sample evaluation framework for prompt-based comparison
- `prompts.json` - Factual, adversarial, and bias-sensitive prompt set
- `evaluation.md` - One-page evaluation report summary
- `requirements.txt` - Python dependencies

## Setup

1. Create a Python environment.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Configure the frontier assistant (optional):
   ```bash
   set OPENAI_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   python app.py
   ```

## Usage

- Open the Open Source Assistant tab to interact with the local Hugging Face model.
- Open the Frontier Assistant tab to compare against the hosted API model.
- Use repeated turns to test short-term context retention and conversational memory.

## Architecture decisions

- Gradio UI for fast prototyping and Hugging Face Spaces compatibility.
- Open-source model: `Qwen/Qwen-2.5-0.5B-Instruct` as a deployable small model.
- Frontier model: OpenAI `gpt-4.1-mini` via API for direct performance comparison.
- Memory: multi-turn history is preserved within each session state.
- Safety: a lightweight keyword-based guardrail layer is applied before generation.

## Tradeoffs

- Local OSS model may be slow on CPU and requires a compatible environment.
- The frontier assistant depends on API access and may incur cost.
- Safety is intentionally lightweight; a production system should use a stronger moderation service.

## Improvements with more time

- Add tool use and external knowledge retrieval.
- Deploy the OSS assistant on Hugging Face Spaces.
- Implement richer safety/abuse detection with a moderation classifier.
- Add automatic evaluation scoring and visual dashboards.

## Deployment notes

This app is ready for Hugging Face Spaces deployment using the `app.py` Gradio interface and `requirements.txt`. Set `OSS_MODEL=Qwen/Qwen-2.5-0.5B-Instruct` and deploy to a CPU or GPU-backed space.

## Cost and latency expectations

| Deployment | Cost | Latency | Notes |
|---|---|---|---|
| Local OSS (`Qwen-2.5-0.5B-Instruct`) | Low / one-time model download | Medium to high on CPU, lower on GPU | Best for low-cost demos and research. |
| Hosted frontier (`gpt-4.1-mini`) | API billing per request | Low | Better quality and reliability, but with variable cost. |
