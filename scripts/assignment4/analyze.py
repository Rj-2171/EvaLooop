"""Assignment 4 analysis: summary.csv, paired comparison, curves, compliance, failure dump.

Usage:
    python scripts/assignment4/analyze.py --results_dir results --out_dir analysis \
        [--package_dir path/to/assignment-data]
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from evaloop.analysis.metrics import asl_bias, asl_pow, pass_at_k
from evaloop.tasks.multilingual_summarization import script_ratio

M = 10


def load_runs(results_dir):
    runs = []
    for path in sorted(Path(results_dir).glob("*_results.json")):
        if "_partial_" in path.name:
            continue
        data = json.loads(path.read_text())[0]
        cfg = data.get("config", {})
        info = data.get("model_info", {})
        model = info.get("model_id") or data["model"]
        conf = "standard-mbpp-plus" if cfg.get("loop", "standard") == "standard" else f"{cfg['summary_language']}-summary-mbpp-plus"
        if cfg.get("strip_summary_code"):
            conf += "-codestripped"
        runs.append({"path": path.name, "model": model, "configuration": conf, "data": data,
                     "cycles": {r["task_id"]: r["successful_cycles"] for r in data["prompt_results"]}})
    return runs


def metrics(cycles):
    sc = list(cycles.values())
    n = len(sc)
    return {
        "n_tasks": n,
        "success_rate": pass_at_k(sc, 1) / n,
        "asl_score": asl_pow(sc, M),
        "asl_bias": asl_bias(sc, M),
        "pass_at_10": pass_at_k(sc, M) / n,
        "pass_drop_pts": 100 * (pass_at_k(sc, M) - pass_at_k(sc, 1)) / n,
        "mean_loops": sum(sc) / n,
    }


def summary_text(wrapped):
    lines = wrapped.strip().strip('"').strip().splitlines()
    return "\n".join(lines[:-1]).strip() if len(lines) > 1 else wrapped


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        diffs = [x - y for x, y in zip(a, b)]
        if not any(diffs):
            return None
        return w(a, b).pvalue
    except ImportError:
        return None


def last_cycle_detail(cycle):
    tr = cycle.get("test_results", {})
    return {k: tr.get(k) for k in ("base_success", "plus_success", "error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--package_dir", default=None)
    ap.add_argument("--n_failures", type=int, default=15, help="failed loops sampled per run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir)
    report = {}

    # 1. summary.csv (+ extended metrics)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "configuration", "n_tasks", "success_rate", "asl_score"])
        for r in runs:
            m = metrics(r["cycles"])
            r["metrics"] = m
            w.writerow([r["model"], r["configuration"], m["n_tasks"], f"{m['success_rate']:.3f}", f"{m['asl_score']:.3f}"])
    with open(out / "summary_extended.csv", "w", newline="") as f:
        keys = ["n_tasks", "success_rate", "asl_score", "asl_bias", "pass_at_10", "pass_drop_pts", "mean_loops"]
        w = csv.writer(f)
        w.writerow(["model", "configuration", *keys, "calls", "failed_calls", "empty_responses", "prompt_tokens", "output_tokens", "thinking_tokens"])
        for r in runs:
            u = r["data"].get("model_info", {}).get("usage", {})
            w.writerow([r["model"], r["configuration"], *[round(r["metrics"][k], 4) for k in keys],
                        *[u.get(k, "") for k in ("calls", "failed_calls", "empty_responses", "prompt_tokens", "output_tokens", "thinking_tokens")]])

    # 2. paired comparison per model
    by_model = defaultdict(dict)
    for r in runs:
        by_model[r["model"]][r["configuration"]] = r
    paired = {}
    for model, confs in by_model.items():
        base = confs.get("standard-mbpp-plus")
        for name, ext in confs.items():
            if ext is base or base is None:
                continue
            ids = sorted(set(base["cycles"]) & set(ext["cycles"]))
            both = [t for t in ids if base["cycles"][t] >= 1 and ext["cycles"][t] >= 1]
            paired[f"{model} | {name}"] = {
                "n_common": len(ids),
                "cycle1_agree": sum((base["cycles"][t] >= 1) == (ext["cycles"][t] >= 1) for t in ids),
                "n_pass_cycle1_both": len(both),
                "ext_shorter": sum(ext["cycles"][t] < base["cycles"][t] for t in both),
                "ext_longer": sum(ext["cycles"][t] > base["cycles"][t] for t in both),
                "equal": sum(ext["cycles"][t] == base["cycles"][t] for t in both),
                "mean_loops_base_given_pass1": sum(base["cycles"][t] for t in both) / max(1, len(both)),
                "mean_loops_ext_given_pass1": sum(ext["cycles"][t] for t in both) / max(1, len(both)),
                "wilcoxon_p": wilcoxon([base["cycles"][t] for t in both], [ext["cycles"][t] for t in both]),
            }
    report["paired"] = paired

    # 3. rankings
    report["ranking"] = {
        conf: [r["model"] for r in sorted((r for r in runs if r["configuration"] == conf), key=lambda r: -r["metrics"]["asl_score"])]
        for conf in sorted({r["configuration"] for r in runs})
    }

    # 4. survival curves
    curves = {f"{r['model']} | {r['configuration']}": [pass_at_k(list(r["cycles"].values()), k) / len(r["cycles"]) for k in range(1, M + 1)] for r in runs}
    report["pass_at_k"] = curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for label, ys in curves.items():
            ax.plot(range(1, M + 1), ys, marker="o", ms=3, ls="--" if "standard" in label else "-", label=label.replace("-mbpp-plus", ""))
        ax.set_xlabel("cycle k")
        ax.set_ylabel("tasks still passing at cycle k")
        ax.set_xticks(range(1, M + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "pass_at_k.png", dpi=200)
    except ImportError:
        pass

    # 5. summary language compliance and length
    lang = {}
    for r in runs:
        language = r["data"].get("config", {}).get("summary_language") or "hindi"
        ratios, lengths, by_cycle = [], [], defaultdict(list)
        for pr in r["data"]["prompt_results"]:
            for c in pr["cycles"]:
                s = c.get("task_2_output")
                if not s:
                    continue
                text = summary_text(s)
                ratio = script_ratio(text, language)
                ratios.append(ratio)
                by_cycle[c["cycle"]].append(ratio)
                lengths.append(len(text.split()))
        if ratios:
            lang[f"{r['model']} | {r['configuration']}"] = {
                "summaries": len(ratios),
                f"share_mostly_{language}_script": sum(x >= 0.5 for x in ratios) / len(ratios),
                "mean_script_ratio": sum(ratios) / len(ratios),
                "mean_ratio_by_cycle": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cycle.items())},
                "mean_words": sum(lengths) / len(lengths),
            }
    report["summaries"] = lang

    # 6. agreement with the provided data package (standard loop, same model)
    if args.package_dir:
        pkg = Path(args.package_dir) / "loops"
        rep = {}
        for r in runs:
            if r["configuration"] != "standard-mbpp-plus":
                continue
            f = pkg / f"{r['model'].lower()}.json"
            if not f.exists():
                continue
            ref = {t["task_id"]: t["successful_cycles"] for t in json.loads(f.read_text())["tasks"]}
            ids = sorted(set(ref) & set(r["cycles"]))
            sub = {t: ref[t] for t in ids}
            rep[r["model"]] = {
                "n_common": len(ids),
                "package_metrics_on_same_tasks": metrics(sub),
                "cycle1_verdict_agreement": sum((ref[t] >= 1) == (r["cycles"][t] >= 1) for t in ids) / len(ids),
                "exact_loop_count_agreement": sum(ref[t] == r["cycles"][t] for t in ids) / len(ids),
            }
        report["vs_package"] = rep

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 7. failure dump for manual inspection: loops that failed after at least one summary
    rng = random.Random(args.seed)
    lines = ["# Sampled failed loops\n"]
    rows = []
    for r in runs:
        failed = [pr for pr in r["data"]["prompt_results"] if 1 <= pr["successful_cycles"] < M]
        for pr in sorted(rng.sample(failed, min(args.n_failures, len(failed))), key=lambda p: p["task_id"]):
            fail = pr["cycles"][-1]
            prev = pr["cycles"][-2]
            lines += [
                f"\n## {r['model']} | {r['configuration']} | {pr['task_id']} | failed at cycle {fail['cycle']}\n",
                "**Original prompt**\n```\n" + pr["initial_prompt"].strip() + "\n```",
                "**Cycle-1 summary**\n```\n" + pr["cycles"][0].get("task_2_output", "").strip() + "\n```",
                "**Prompt of failing cycle (last summary)**\n```\n" + prev.get("task_2_output", "").strip() + "\n```",
                "**Last passing code**\n```python\n" + prev.get("task_1_output", "").strip() + "\n```",
                "**Failing code**\n```python\n" + fail.get("task_1_output", "").strip() + "\n```",
                f"Test result: {last_cycle_detail(fail)}\n",
            ]
            rows.append([r["model"], r["configuration"], pr["task_id"], fail["cycle"], "", ""])
    (out / "failed_loops_sample.md").write_text("\n".join(lines))
    with open(out / "failure_annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "configuration", "task_id", "failed_cycle", "failure_mode", "note"])
        w.writerows(rows)

    print(json.dumps({k: report[k] for k in ("paired", "ranking", "summaries")}, indent=2, ensure_ascii=False))
    print(open(out / "summary.csv").read())


if __name__ == "__main__":
    main()
