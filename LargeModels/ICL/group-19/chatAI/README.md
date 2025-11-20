# Prompt Strategy Playground

A lightweight browser client for probing how different prompting strategies affect large language model responses on a subset of the MMLU benchmark. The page runs entirely on the client and calls your preferred model provider over HTTPS (OpenAI and Qwen-compatible endpoints are supported out of the box).

## Features

- 🔌 Bring-your-own API key (stored locally if you opt in)
- 📚 Curated MMLU sample covering multiple subjects
- 🎯 Five prompt strategies: Base + 2 shots, Instruction-following, Chain-of-Thought, Self-reflection, and CoT + committee vote (N=5)
- 📈 Rich telemetry per run: latency, token usage, accuracy flag, cost estimate
- 🔍 Run log history slider for quick side-by-side comparisons of previous runs
- 🧠 Full reasoning trace for each call, with self-reflection and voting breakdowns where applicable
- 📝 Notepad for qualitative observations alongside the quantitative metrics

## Getting started

1. Serve the folder with any static web server, for example:
   ```pwsh
   cd c:\Users\zhang\projects\chatAI
   npx serve .
   ```
2. Open the reported URL in your browser.
3. Paste your API key, adjust the base URL if needed (defaults to Qwen compatible mode), choose a question and strategy, then run the experiment.

## Customisation tips

- **Model endpoint**: Use the "API Base URL" field in the UI (persisted locally) or change the fetch URL in `main.js` to point at Anthropic, Azure OpenAI, or your own gateway.
- **Pricing table**: Update `state.priceTable` for accurate cost estimation when using alternate models.
- **Dataset**: Extend `data/mmlu_samples.json` with additional MMLU items or subjects.
- **Strategies**: Add new prompt generators under the `runStrategy` switch—each can leverage the shared telemetry/rendering pipeline.

## Notes

- Token usage and pricing estimates rely on the provider responding with `usage` data.
- The majority voting strategy runs the model multiple times; watch your rate limits.
- The UI stores your API key and notes in `localStorage` only when you opt in.
