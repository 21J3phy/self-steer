# Pre-registration: self-steer experiments (written 2026-11-02, before any E3–E5 generation)

The model is Qwen3.5-4B, MLX 8-bit (`mlx-community/Qwen3.5-4B-8bit`, rev `1704583a…`), in non-thinking mode. Sampling uses Qwen's settings: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, at most 1024 new tokens per turn. Seeds come from `matched_seed(base_seed, question_id)`, as in lazy-genius-qwen.

**E1, fidelity, is already done before this file.** `scripts/parity_lazy.py` reproduces 16 of 16 published lazy-genius-qwen test-2 cells token for token, 8 baseline and 8 Lazy Genius. That is the evidence the new engine applies the same edit.

## E2. Overhead (descriptive)
`experiments/bench.py` measures:
- decode speed with no edit, with `lazy` (6 positions), with `lazy` over the whole answer, with a 9-layer rank-1 ablation, and with both;
- CAA mode-build time;
- re-prefill time at ledger lengths of 512, 1024, 2048 and 4096 tokens;
- the time to snapshot a cache.

No hypotheses are tested.

## E3. Does switching mid-conversation break the model's memory?
The test uses 24 programmatic 3-turn conversations, all with seed 1000+i:
1. **Turn 1:** four facts, then "Just reply 'Noted.'".
2. **Turn 2:** an unrelated everyday question, with the edit armed by the controller for this answer only.
3. **Turn 3:** a recall question about two of the facts, with no edit.

**Conditions:**
- `none`: no edit.
- `lazy+keep` and `lazy+reprefill`: `lazy`, window 6.
- `full+keep` and `full+reprefill`: the `lazy` vector over the whole turn-2 answer.

**Outcomes:**
- **Recall:** both facts appear verbatim in the turn-3 reply.
- **KL divergence** (nats) between the next-token distributions of the keep cache and a cache re-encoded with no edits. It is measured over the turn-3 prompt's last position plus 32 teacher-forced answer positions, taken from the reprefill branch.

**Hypotheses:**
- H3a: `lazy+keep` recall is within 2 of 24 conversations of `none`.
- H3b: mean KL under `full+keep` is larger than under `lazy+keep`.

## E4. Smart laziness: can Qwen decide when to be lazy?
**Data:** the lazy-genius-qwen test-1 sealed set `data/fair_final96.json`. Its 96 questions are retired, and Lazy Genius results on this set already exist. Seeds are 17 and 29, giving 192 (question, seed) cells per condition.

**Conditions:**
- `baseline` and `lazy`: the existing published test-1 generations and grades for seeds 17 and 29. E1 shows they are identical to what this engine produces.
- `self` (new): Qwen sees one tool, `set_mode`, whose library contains only `lazy`, plus the system prompt from `tools.system_prompt`. It decides per question whether to call it.
  - Disclosure: the mode description warns against planning and scheduling puzzles. That warning comes from lazy-genius-qwen **test 2**, a different question set.
- `prompt` (new): no edit and no tools. The system prompt is "Be concise. Give the answer directly, with no preamble, no restating the question, and no extra explanation unless asked."
- `routed` (no new generation): a Claude subagent sees only the 96 question texts and the `lazy` description, never answers or grades, and labels each question lazy or normal. The cell's answer is the existing `lazy` or `baseline` generation for that seed. E1 shows the engine output is identical in both cases.
- `oracle` (hindsight upper bound, not a policy): per cell, take the `lazy` answer if its core grade is at least the baseline's, otherwise the baseline answer.

**Grading:**
- The new answers (`self`, `prompt`; 384 in total) are shuffled together with 48 previously graded test-1 answers, chosen at random with `random.Random(0)` from the baseline and lazy rows at seeds 17 and 29. The old answers serve as calibration.
- The set is split into 9 chunks of 48, each blind-graded by one Claude Opus 5.5 grader using lazy-genius-qwen's `GRADING_SAMPLED.md` and the per-item rubrics.
- Old grades are used for `baseline`, `lazy`, `routed` and `oracle`. Grader drift is reported as agreement on the 48 calibration items.

**Outcomes:** each is computed for every condition against `baseline`, using a paired bootstrap over questions (mean over the 2 seeds; 10,000 resamples; `numpy default_rng(0)`; 95% percentile CI).
- **Total-word reduction:** 1 − Σwords(cond) / Σwords(baseline).
- **Median per-answer reduction.**
- **Core accuracy difference.**
- **Whole-answer accuracy difference.**
- **Lazy-choice rate** by category, for `self` and `routed`.

**Hypotheses:**
- H4a: `self` keeps at least half of `lazy`'s total-word reduction.
- H4b: on planning questions, `self` core accuracy is at least `lazy`'s.
- H4c: `self` core accuracy difference vs. baseline has a 95% CI lower bound above −5 points.

There is no directional hypothesis for `prompt` vs. `lazy`; it is reported as is.

## E5. Claude builds new modes over MCP
- Claude (Opus 5.5, the Claude Code session) uses the live MCP endpoint (`create_mode`, `try_mode`, `fulfill_request`) to build three modes from examples it writes.
  - `eli5`, an add edit.
  - `bullets`, an add edit.
  - `plain-math`, a rank-1 ablation (weight edit) that removes the LaTeX direction from MLP layers.
- Each mode is evaluated on 20 held-out prompts against normal and against a system-prompt instruction asking for the same style. The automatic metrics are:
  - Flesch–Kincaid grade, for `eli5`;
  - the share of answers that are mostly bullet lines, for `bullets`;
  - the share of answers containing LaTeX markup, plus numeric correctness on arithmetic word problems with known answers, for `plain-math`.
- One end-to-end request is also recorded and timed: Qwen calls `request_mode`, Claude builds the mode, and Qwen uses it.

E5 is a case study, not a hypothesis test. Strengths are chosen with `try_mode` on prompts that are not in the 20 held-out ones.

## Reporting
Every outcome is reported whatever its direction. Deviations from this plan are listed in the paper.

---

# Addendum (2026-11-02, before any E6 generation): E6, Claude reads the neurons

**Why.** In E5, Claude built modes by writing contrastive examples. The harness averaged them into a dense residual vector, and Claude tuned the strength by reading outputs. Claude never inspected a neuron or a weight. That is not what Weight-Space Programming did. There, Astra worked from a kernel over the raw weights plus a budget of forward passes that returned MLP activations, and it edited the specific W_down columns that write the problem direction. E6 gives Claude those instruments and retries the E5 task that failed.

**Task.** As in E5's `plain-math`: Qwen should write math in plain text instead of LaTeX, with the numbers intact.

**Controller rules.**
- **Who:** Claude Opus 5.5 in the Claude Code session, over MCP.
- **Instruments, all free:**
  - `neuron_info`: a neuron's W_down column norm and the tokens it promotes or suppresses.
  - `token_direction` and `top_writers`: which columns write a direction.
  - `export_down`: raw dequantized W_down, for analysis in Python.
- **Forward passes, a budget of 10:**
  - `read_activations`: teacher-forced, returning every MLP neuron's activation and the residual stream.
  - `generate_pass`: paired original vs. edited.
- **Prompts:** passes may use only practice prompts (`experiments/practice_math.py`, seed 99) and texts Claude writes, never E5's 20 held-out prompts.
- **Forbidden:** gradients, any fitting, and CAA dense vectors.
- **The final edit:** must be a `columns` edit, changing specific W_down columns (`scale` or `project`), registered as a mode Qwen can switch on.

**Evaluation.**
- **Task set:** E5's 20 held-out arithmetic prompts with the same seeds, `normal` vs. the edit, at 500 max tokens. Metrics are answers containing LaTeX and answers containing the correct number.
- **Side-effect check:** the same edit on E5's 20 held-out how-to prompts (`bullets` set). Metrics are mean words, and outputs that become degenerate (any 4-gram repeated 5 or more times).
- **Comparisons:** reported against E5's `plain-math` (20/20 LaTeX), `plain-math-add` (7/20 LaTeX, 11/20 correct) and the instruction (13/20 LaTeX, 20/20 correct).

**Hypothesis.** H6: the neuron-read edit leaves LaTeX in fewer held-out answers than the instruction does (fewer than 13 of 20), while 18 or more of 20 keep the correct number.

---

# Addendum 2 (2026-11-02 ~22:00, before any E7/E8 pass): runs move to Kaggle; E7 and E8

**Platform.** At Nirav's request, everything from E6 on runs on Kaggle: 2× T4, Qwen3.5-4B in the original Hugging Face weights (revision `851bf6e8…`) at FP32. The harness was ported to PyTorch (`remote/worker.py`, with CPU tests in `remote/test_worker.py`). Because the numbers differ from the local MLX 8-bit runs, every comparison inside E6–E8 is regenerated on Kaggle.

**E6 deviations, recorded as they happened.**
- The first attempt (passes 1–3) used mean activation contrasts plus weight lenses.
- The worker was then extended with a direct-logit-attribution read and restarted. The pass budget was carried over: 3 used, 7 left.

**E7: fix the self-steering failure from neuron values.**
- **Problem:** in E4, Qwen in the tool context with `lazy` answered arithmetic questions answer-first and wrong, while the placebo was fine.
- **Budget:** 10 passes. Reads may use the tool context with the edit live.
- **Practice:** arithmetic and time questions from the test-2 set (`fair2_final96`, round7).
- **Allowed fix:** a neuron-level change to the `lazy` edit (dropping or rescaling specific neurons or columns), chosen from reads and weights.
- **Evaluation:** the 24 arithmetic and time questions of the test-1 set (`fair_final96`) × seeds 17 and 29, blind-graded as in E4.
  - Conditions: `self` (tool + `lazy`), `self-fixed` (tool + the fixed edit) and `placebo`, all generated on Kaggle.
  - H7: `self-fixed` core accuracy is above `self`, and within 2 cells of `placebo`, while its answers stay shorter than placebo's.

**E8: `eli5` from neuron values.**
- **Budget:** 10 passes.
- **Practice:** 16 science questions not in E5's held-out 20.
- **Allowed edit:** scaling chosen neurons (a `columns` edit), chosen from reads of kid-style vs. normal answers plus weight lenses. No CAA vectors.
- **Evaluation:** E5's 20 held-out `eli5` prompts, comparing `normal`, the edit, and the instruction (all on Kaggle), using the Flesch–Kincaid grade.
  - H8: the edit lowers the grade by at least 3 points with no degenerate outputs.

---

# Addendum 3 (2026-11-03 ~01:45, written while the E8 held-out generations were running, before any were read, and before any E7 pass)

**E6, outcome and an exploratory metric.**
- H6 fails on its pre-registered metric: the 512-neuron edit left LaTeX in 20/20 held-out answers, against 14/20 for the instruction on Kaggle. All 20 kept the correct number.
- LaTeX density (markup tokens per 100 words) was added after the result and is reported as exploratory.

**E8, what the 8 passes did.**
- Pass 5 deviates from the plan. It read the *same* answer texts with and without the eli5 instruction, a matched context, instead of only kid-style vs. normal texts. The reason: pass 3's kid-vs-normal text contrast was dominated by the text itself, and an edit built from it (pass 4) did nothing.
- The final edit `eli5-cos29` is still a pure scaling edit: in layers 16–29, the 256 neurons per layer whose plain-context writes best align with the instruction effect (by cosine, sign-stable in ≥90% of texts) are scaled ×4. That is 3584 neurons.
- It was chosen at pass 8 of 10, from 16 practice prompts: FK 10.0 → 6.5, 0/16 degenerate. Pass 7 (the same rule with layers 30–31 included) had 2/16 degenerate.
- The held-out evaluation is as pre-registered: 20 prompts, seeds 2000–2019, 500 tokens, normal / edit / instruction.

**E7, the planned reads (budget 10).**
- Passes 1–2: `selfsteer_run` placebo and self on the 12 practice questions, seed 17.
- Passes 3–6: teacher-forced reads of the placebo answers (first 6 answer tokens, the `lazy` window) in four contexts: bare, bare + `lazy`, tool, tool + `lazy`.
- From these, the kernel compares the edit's cost to the good answers' log-probability in the bare vs. the tool context, and how the tool context's own shift aligns with each of the 344 `lazy` neuron vectors.
- The fix is a neuron-level change to `lazy`: dropping or rescaling some of its 344 neurons, or scaling downstream columns while it is on. It is registered under the name `lazy`, so the tool Qwen sees is unchanged.
