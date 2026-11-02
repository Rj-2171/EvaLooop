# Assignment 4: Hindi-summary loop for EvaLoop

Richa Jha (rjha3@ncsu.edu), CSC 591/791 Trustworthy AI for SE

**Public fork:** FORK_URL_HERE (based on upstream commit `ce22fe2`)

## What this adds

A new loop type (Option B): code to a one-sentence **Hindi** summary to code. Everything except the summary language is the same as the standard loop: same generation prompt, same one-sentence summary instruction (with the opener translated), same docstring wrapper, and the same retained assert line from the original MBPP+ prompt, which keeps the function name and signature fixed. Cycle 1 uses the original English prompt, so the model's own output is the only perturbation.

Summarization prompt:

```
Use one sentence to summarize the following code in Hindi and start with 'एक पायथन फ़ंक्शन लिखें जो':

```
{code}
```

```
एक पायथन फ़ंक्शन लिखें जो
```
```

## Changes to the framework

New files:

| File | Purpose |
|---|---|
| `evaloop/tasks/multilingual_summarization.py` | Hindi (also Chinese, Spanish) summarization task. Subclass of `CodeSummarizationTask`, so the assert-line logic is reused unchanged. Also `script_ratio()` for language compliance. |
| `evaloop/models/local_chat.py` | Local open models with the model's own chat template and greedy batched decoding (vLLM or transformers). |
| `evaloop/models/gemini.py` | Optional Gemini backend (API key or Vertex AI). Not used for the reported results. |
| `scripts/assignment4/analyze.py` | `summary.csv`, extended metrics, paired Wilcoxon test, pass@k curve, language compliance, failure sample for manual inspection. |
| `scripts/assignment4/failure_stats.py` | Automatic failure statistics used in the report (degenerate summaries, visible-assert check, base vs plus, fixed points). |
| `tests/test_multilingual.py` | 8 unit tests for the new task and config wiring. |

Modified files (why):

- `evaloop/core/config.py`: `loop` (`standard` or `multilingual`), `summary_language`, task subset options, local/Gemini backend options, and the task config for the new loop. This is the one place a new loop must touch per CONTRIBUTING.md.
- `evaloop/core/evaluator.py`: builds tasks from the configured loop, optional seeded task subset, checkpoint after every cycle (Colab can disconnect), and stores the run config and model usage in the results JSON.
- `evaloop/cli.py`: flags for the above.
- `evaloop/models/factory.py`: backends are imported lazily. Upstream imports every backend at import time, and `open_source_vllm.py` raises ImportError when vLLM is missing, so the framework could not run at all without vLLM.
- `evaloop/tasks/__init__.py`: registers the new task.

The loop runner (`evaloop/evaluation/cycle.py`) and the tester are unchanged. `git diff ce22fe2` gives the full patch (`evaloop_a4.patch` in the submission).

## Dependencies

Python 3.10+, `pip install -e . scipy`, plus `pip install vllm` for fast GPU inference (optional; `--engine hf` uses transformers). Results were produced on a free Google Colab Tesla T4 (fp16) with vLLM installed from pip on Nov 1, 2026.

## Models and tools

- Evaluated models: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct` (Hugging Face, fp16).
- Settings: temperature 0 (greedy), max 1024 new tokens, max 10 cycles, single run, all 378 MBPP+ tasks, system prompt "You are a helpful programming assistant.".
- Tests: EvalPlus MBPP+ base and plus tests through the framework's `EvalPlusCodeTester` (EvalPlus sandboxed execution with time limits).
- No LLM judge or classifier was used. ASL is reported without semantic weighting (`ASL_pow`, the leaderboard's `aslScore`).
- Cost: $0. GPU time 85 minutes in total.

## How to run

Easiest: open `A4_run_experiments.ipynb` in Colab with a T4 GPU and run all cells (it clones the fork or applies the patch, runs all four configurations, and runs the analysis).

From the command line:

```bash
pip install -e . scipy vllm
for M in Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct; do
  for LOOP in standard multilingual; do
    evaloop evaluate --model $M --backend local --engine vllm --loop $LOOP \
      --summary_language hindi --max_cycles 10 --temperature 0.0 \
      --output_dir results --experiment_name "$(basename $M)_$LOOP"
  done
done
python scripts/assignment4/analyze.py --results_dir results --out_dir analysis
python scripts/assignment4/failure_stats.py --results_dir results --out analysis/failure_stats.json
python -m pytest -q
```

Generated code is untrusted. EvalPlus runs it in a restricted subprocess, and `failure_stats.py` runs the visible-assert check in a subprocess with a timeout; run both on a disposable VM such as Colab, not on a machine with personal files.

## Where each reported number comes from

| Report item | Source |
|---|---|
| Table 1 (success rate, ASL) and `summary.csv` | `analyze.py` from `results/*_results.json` |
| pass@10, paired counts, mean sustained cycles, Wilcoxon p | `analysis/summary_extended.csv`, `analysis/report.json` (`paired`) |
| Figure 1 | `report.json` (`pass_at_k`) |
| Devanagari share, summary length | `report.json` (`summaries`) |
| Output tokens | `summary_extended.csv` (`output_tokens`) |
| Fixed points, degenerate summaries, failure cycle, base vs plus, visible-assert check, triangle words, survivors with degenerate summaries | `analysis/failure_stats.json` |
| Pass rate at cycle 2 after a degenerate vs normal summary | the cycle-2 snippet at the end of this README |
| Table 2 (manual failure modes) and the cited cases | `analysis/failure_annotations.csv` (34 loops, sampled with seed 7) |
| Harness check against the data package (2843/2843 verdicts, pass@1 0.772, ASL 7.179 for gemini-2.5-flash-lite) | replaying package code through `EvalPlusCodeTester`; records referenced by model_id and task_id |

Cycle-2 pass rate by summary type:

```python
import json, glob, sys
sys.path.insert(0, "scripts/assignment4")
from failure_stats import degenerate, summary_body
for f in glob.glob("results/*multilingual*_results.json"):
    if "partial" in f: continue
    d = json.load(open(f))[0]; c = {True: [0, 0], False: [0, 0]}
    for p in d["prompt_results"]:
        if len(p["cycles"]) >= 2:
            k = degenerate(summary_body(p["cycles"][0]["task_2_output"]))
            c[k][0] += p["cycles"][1]["success"]; c[k][1] += 1
    print(d["model_info"]["model_id"], c)
```

Records are referenced as model, configuration, task_id and cycle (for example `Qwen2.5-Coder-3B-Instruct, hindi, Mbpp/90, cycle 2`).

## Submission contents

- `A4_report_Richa_Jha.pdf`: the report (3 pages).
- `README.md`: this file.
- `evaloop_a4.patch`: diff against upstream `ce22fe2`.
- `summary.csv`: one row per run.
- `results/`: the framework's results JSON for all four runs.
- `analysis/`: extended metrics, paired statistics, failure statistics, manual annotations, figure.
- `A4_run_experiments.ipynb`: the executed Colab notebook.
