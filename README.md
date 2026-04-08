---
title: DisputeDesk
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: OpenEnv dispute-resolution environment.
tags:
  - openenv
  - fastapi
  - agents
  - reinforcement-learning
---

# DisputeDesk

`DisputeDesk` is a real-world environment for marketplace dispute resolution. The agent acts as an operations specialist who must inspect case evidence, classify disputes, and issue policy-compliant outcomes.

The package follows the OpenEnv 3-component pattern:

- `dispute_desk/models.py` for typed actions, observations, and state
- `dispute_desk/client.py` for the typed Python client
- `dispute_desk/server/` for environment logic and FastAPI serving

## Why this environment

- Real task: support and trust teams resolve these disputes every day.
- Deterministic grading: success can be measured against policy, evidence review, refund math, and escalation correctness.
- Reward shaping: the agent receives credit for useful intermediate work, not only the final answer.
- Seeded hard-case variants: the hardest tasks include seeded variants under the same task id, so agents must read evidence content instead of only memorizing a fixed action sequence.

## Environment shape

### Observation space

Each observation includes:

- case summary and task objective
- available evidence artifacts
- revealed evidence content
- revealed policy and operations context
- current classification state
- steps remaining
- provisional score

### Action space

The agent can:

- `review_artifact`
- `classify_case`
- `request_more_context`
- `resolve_case`

## Tasks

### `late_delivery_refund`

- Difficulty: easy
- Goal: identify a valid item-not-received claim and issue the correct refund.

### `partial_damage_partial_refund`

- Difficulty: medium
- Goal: refund only the damaged portion of a multi-item order.

### `wrong_item_premium_exchange`

- Difficulty: medium
- Goal: confirm a fulfillment error and replace the item while enforcing the required return policy.

### `safety_risk_damage_replacement`

- Difficulty: hard
- Goal: determine the correct replacement workflow for safety-critical damage, including whether return is required or must be waived for safe disposal.

### `suspicious_refund_abuse`

- Difficulty: hard
- Goal: detect abuse signals and choose the correct trust action, including deny-vs-escalate branches under seeded variants.

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /health`
- `GET /metadata`
- `GET /tasks`
- `GET /grader`
- `GET /baseline`

## Environment variables

For competition-compatible inference, set:

- `dispute_desk_env/.env`

Format:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-5-mini-2025-08-07
HF_TOKEN=your_key_here
```

The `.env` file is gitignored.
The Docker image does not copy `.env`; pass it at runtime with `--env-file .env`.
The runtime also accepts `OPENAI_API_KEY` and `OPENAI_MODEL` as local compatibility aliases.

## Local run

```bash
cd dispute_desk_env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
server
```

Or:

```bash
cd dispute_desk_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Python client

```python
from dispute_desk import DisputeDeskEnv
from dispute_desk.models import CaseAction

with DisputeDeskEnv(base_url="http://127.0.0.1:8000").sync() as env:
    env.health()
    env.tasks()
    result = env.reset(task_id="late_delivery_refund")
    result = env.step(CaseAction(action_type="review_artifact", artifact_id="order_summary"))
    state = env.state()
```

The typed client supports the standard OpenEnv `reset()`, `step()`, and `state()` flow over WebSocket, plus convenience helpers for `health()`, `metadata()`, `schema()`, `tasks()`, and `grader()` over HTTP.

## Baseline

The baseline runner uses the OpenAI Python client and supports the competition variables:

- `API_BASE_URL`
- `HF_TOKEN`
- `MODEL_NAME`

It also accepts local compatibility aliases:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

```bash
cd dispute_desk_env
dispute-desk-baseline
```

Submission-compatible root entrypoint:

```bash
cd dispute_desk_env
python inference.py
```

The root inference runner prints the exact validator-facing stdout structure:

```text
[START] task=late_delivery_refund env=dispute_desk model=gpt-5-mini-2025-08-07
[STEP] step=1 action=review_artifact(order_summary) reward=0.05 done=false error=null
[END] success=true steps=7 score=0.987 rewards=0.05,0.05,0.01,0.01,0.05,0.07,0.72
```

Use `python inference.py --json` if you also want the final `BaselineResponse` JSON on stdout after the structured blocks.
The run writes the latest score artifact to `outputs/evals/baseline_latest.json`.

Pinned default baseline model:

- `gpt-5-mini-2025-08-07`

This snapshot choice is for reproducibility. It can be overridden with `MODEL_NAME` or `OPENAI_MODEL`.

## Docker

```bash
cd dispute_desk_env
docker build -t dispute-desk-env .
docker run --rm --env-file .env -p 8000:8000 dispute-desk-env
```

The repository root now contains the Dockerfile expected by Hugging Face Docker Spaces.

## Hugging Face Space Deployment

This repository is configured for a Hugging Face Docker Space:

- `README.md` contains the required YAML frontmatter with `sdk: docker`
- the Space is configured to expose port `8000`
- the root `Dockerfile` builds and runs the FastAPI environment directly

In your Space settings, add:

- Secret: `HF_TOKEN`
- Variable: `API_BASE_URL` with `https://api.openai.com/v1`
- Variable or secret: `MODEL_NAME` if you want to override the pinned baseline model

If you already use local OpenAI-prefixed variables, the runtime still accepts them as aliases.

Once the Space repo exists, you can push this repo to Hugging Face with:

```bash
git remote add hf https://huggingface.co/spaces/<your-hf-username>/DisputeDesk
git push hf main
```

## Validation flow

Run the full local submission check with:

```bash
cd dispute_desk_env
python3.11 -m compileall dispute_desk tests server
pytest
openenv validate . --json
```

Once `HF_TOKEN` is present in `.env`, run:

```bash
cd dispute_desk_env
dispute-desk-baseline
```

## Baseline scores

Validated local run on `2026-03-26` with `gpt-5-mini-2025-08-07`:

- Average score: `0.9847`
- `late_delivery_refund`: `0.9867` in `7` steps
- `partial_damage_partial_refund`: `0.99` in `7` steps
- `wrong_item_premium_exchange`: `0.9867` in `9` steps
- `safety_risk_damage_replacement`: `0.98` in `10` steps
- `suspicious_refund_abuse`: `0.98` in `9` steps
