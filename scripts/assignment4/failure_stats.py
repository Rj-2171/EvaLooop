"""Automatic failure statistics used in the A4 report.

For every failed loop: syntax/entry-point checks, base vs plus failure, and whether the failing
code still passes the one assert line visible in the prompt. For every summary: degeneration
(repetition loops), length, and words for triangles in tasks that are not about triangles.

The visible-assert check executes model code; run it only in a sandbox (Colab VM, container).

Usage: python scripts/assignment4/failure_stats.py --results_dir results --out analysis/failure_stats.json
"""

import argparse
import ast
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

M = 10
TRIANGLE = re.compile(r"त्रिभुज|त्रिकोण|त्रिविम")


def summary_body(wrapped):
    lines = wrapped.strip().strip('"').strip().splitlines()
    return "\n".join(lines[:-1]).strip()


def degenerate(text):
    words = text.split()
    return not words or collections.Counter(words).most_common(1)[0][1] >= 10 or len(words) > 120


def entry_point(prompt):
    m = re.search(r"assert\s+(?:set\()?\(?\s*(?:math\.isclose\()?\s*([A-Za-z_]\w*)\(", prompt.splitlines()[-2])
    return m.group(1) if m else None


def visible_assert(code, assert_line):
    src = "import math, re, heapq, collections, itertools, functools\nfrom typing import *\n" + code + "\n" + assert_line + "\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        Path(path).write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            return "timeout"
    if r.returncode == 0:
        return "pass"
    return "assert_fail" if "AssertionError" in r.stderr else "error"


def run_name(data):
    loop = data.get("config", {}).get("loop", "standard")
    return f"{data['model_info']['model_id']} | {'standard' if loop == 'standard' else 'hindi'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="analysis/failure_stats.json")
    args = ap.parse_args()
    stats = {}
    runs = {}
    for f in sorted(Path(args.results_dir).glob("*_results.json")):
        if "_partial_" in f.name:
            continue
        data = json.loads(f.read_text())[0]
        runs[run_name(data)] = data

    for name, data in runs.items():
        s = collections.Counter()
        for pr in data["prompt_results"]:
            cycles = pr["cycles"]
            for c in cycles:
                if c.get("task_2_output"):
                    text = summary_body(c["task_2_output"])
                    s["summaries"] += 1
                    s["summary_words"] += len(text.split())
                    s["degenerate_summaries"] += degenerate(text)
                    if TRIANGLE.search(text) and "triang" not in pr["initial_prompt"].lower():
                        s["triangle_word_unrelated_task"] += 1
            sc = pr["successful_cycles"]
            if sc == M:
                s["survivors"] += 1
                s["survivors_fixed_point"] += cycles[-2].get("task_2_output") == cycles[-3].get("task_2_output")
                s["survivors_with_degenerate_summary"] += any(
                    degenerate(summary_body(c["task_2_output"])) for c in cycles if c.get("task_2_output"))
            elif sc >= 1:
                fail = cycles[-1]
                code = fail.get("task_1_output", "")
                tr = fail.get("test_results", {})
                s["failed_loops"] += 1
                s[f"failed_at_cycle_{fail['cycle']}"] += 1
                s["failed_after_degenerate_summary"] += degenerate(summary_body(cycles[-2].get("task_2_output", "")))
                try:
                    ast.parse(code)
                    ep = entry_point(pr["initial_prompt"])
                    if ep and not re.search(rf"def\s+{re.escape(ep)}\s*\(", code):
                        s["fail_entry_point_renamed"] += 1
                    elif tr.get("base_success") is False:
                        s["fail_base_tests"] += 1
                    else:
                        s["fail_plus_tests_only"] += 1
                except SyntaxError:
                    s["fail_syntax_or_extraction"] += 1
                s["visible_assert_" + visible_assert(code, pr["initial_prompt"].splitlines()[-2])] += 1
        s["mean_summary_words"] = round(s["summary_words"] / max(1, s["summaries"]), 1)
        stats[name] = dict(sorted(s.items()))

    models = sorted({n.split(" | ")[0] for n in runs})
    stats["hindi_survived_longer_than_standard"] = {}
    for m in models:
        std = {p["task_id"]: p["successful_cycles"] for p in runs[f"{m} | standard"]["prompt_results"]}
        hin = {p["task_id"]: p["successful_cycles"] for p in runs[f"{m} | hindi"]["prompt_results"]}
        stats["hindi_survived_longer_than_standard"][m] = [t for t in std if hin[t] > std[t] >= 1]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
