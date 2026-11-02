# self-steer

**Qwen3.5-4B switches its own neuron and weight edits mid-conversation, and can ask Claude to build new ones.**

- **Qwen picks its own edit.** It gets a `set_mode` tool. Calling it turns on an edit to its own MLP layers for the next answer: for example the Lazy Genius edit (344 neurons over the first 6 tokens), or a direction removed from a few layers' weights.
- **Qwen can ask for a new edit.** If no mode fits, it calls `request_mode("…")`. Claude, connected over MCP, writes a handful of contrastive examples, turns them into an edit in seconds, checks it with `try_mode`, and hands it back. Qwen switches it on and answers.
- **Nothing is retrained or reloaded.** Every edit is a bias patch or a rank-1 patch on `down_proj`. It sits next to the frozen 8-bit weights and can be switched between two tokens.

The results and the paper are in [`paper/`](paper/), and the pre-registered experiments are in [`experiments/`](experiments/).

## Run it
You need an Apple-silicon Mac with [uv](https://docs.astral.sh/uv/), plus the model:

```bash
huggingface-cli download mlx-community/Qwen3.5-4B-8bit \
  --revision 1704583a86670ba531b7f2c92530a2f725cfd45b --local-dir ~/models/Qwen3.5-4B-8bit
uv sync
uv run selfsteer serve        # web chat at http://127.0.0.1:8765, MCP at /mcp
```

Make Claude Code the controller from inside this repo. `.mcp.json` points it at the server, or add it by hand:

```bash
claude mcp add --transport http self-steer http://127.0.0.1:8765/mcp
```

Then ask Claude to "serve Qwen's mode requests". It loops on `wait_for_request` → `create_mode` → `try_mode` → `fulfill_request`.

For a terminal chat with no server, run `uv run selfsteer chat`. Inside the chat, `/mode lazy` arms a mode yourself.

## How it works
| Piece | File |
|---|---|
| Edits: `add` (a bias patch, e.g. Lazy Genius neuron doses or CAA vectors) and `ablate` (out − (out·u)u, i.e. W_down ← (I − uuᵀ)W_down) | `src/selfsteer/edits.py` |
| Edit stack: absolute token positions and windows; edits change between tokens | `src/selfsteer/stack.py` |
| Decode loop on a persistent cache. It reproduces mlx_lm's sampling exactly | `src/selfsteer/decode.py` |
| Session: an append-only token ledger, tool calls, `keep` or `reprefill` cache policy, forks | `src/selfsteer/session.py` |
| Qwen's tools and system prompt | `src/selfsteer/tools.py` |
| CAA mode builder | `src/selfsteer/probe.py` |
| MCP server, web chat, model thread | `src/selfsteer/server.py`, `worker.py`, `web/index.html` |

## Tests
`uv run python -m unittest discover tests` runs on CPU with a tiny random Qwen3.5 and never loads the 4B. It checks:
- with no edits, logits are bit-identical;
- edits match the Lazy Genius arithmetic exactly;
- `ablate` equals an explicit rank-1 weight edit;
- windows hit the right absolute positions;
- snapshots and re-prefill behave correctly;
- the tool-call loop works, including Qwen → Claude requests.

`scripts/parity_lazy.py` replays published Lazy Genius generations. It matches 16 of 16 token for token.

## Prior work
See [`docs/PRIOR_WORK.md`](docs/PRIOR_WORK.md). The closest work is Weight-Space Programming (MIT, Nov 2026), WeAct (Oct 2026), Transformer², SEAL, Text-to-LoRA and CAST.
