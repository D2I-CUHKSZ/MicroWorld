# WHU Baike Event Example

This example demonstrates how to use LightWorld for an end-to-end social simulation
and analysis of the Wuhan University Baidu Baike event.

## Quick Start

```bash
# 1. Install dependencies
cd /path/to/LightWorld
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env, set LLM_API_KEY and ZEP_API_KEY

# 3. Run full pipeline
uv run lightworld-full-run --config configs/full_run/full_run.template.json
```

## Input Data

Example input data is located in `data/examples/whu_baike_event/`, including:

- Event overview document
- Event timeline
- Related images and videos
- Source manifest

## Output

After the run completes, artifacts are written to `runs/run_<timestamp>/`,
with `run_manifest.json` as the single entry point for all artifacts.
