"""Reproducible, model-independent result aggregation for the audited runner.

Reads only result.json and (when present) data_audit.json. Does not touch
checkpoint files, model weights, splits or candidate rosters. Python stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

EXCLUDED_RUN_FIELDS = frozenset({"seed", "output", "no_auto_summary", "audit_only"})
PREFERRED_METRICS = ("ROC-AUC", "AP", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "F1", "precision", "recall")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


class AggregationError(ValueError):
    """Results cannot be compared safely under the requested protocol."""


def group_name(config: dict) -> str:
    """One folder for each *effective* experimental configuration."""
    model = config["model"]
    if model == "WBTBGRL":
        label = f"WBTBGRL_{config['variant']}_{config.get('ssl_mode', 'original')}"
        flags = (config.get('propagation'), config.get('augmentation'), config.get('loss_weights'))
        if any(v is not None for v in flags):
            # A distinct PAS ablation must never be mixed with its default WP setting.
            default = config['variant'].startswith('WP_')
            values = [default if v is None else bool(v) for v in flags]
            label += '__P{}A{}S{}'.format(*(int(v) for v in values))
    else:
        label = model  # SSL mode / edge-weight toggles are not used by baselines.
    if config.get("dataset") == "synthetic":
        label += f"__{config.get('condition', 'informative')}__graph{config.get('graph_seed', 42)}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise AggregationError(f"Unsafe group label: {label!r}")
    return label


def _effective_config(config: dict) -> dict:
    copy = {k: v for k, v in config.items() if k not in EXCLUDED_RUN_FIELDS}
    if copy.get("model") != "WBTBGRL":
        # WBT-only flags have no effect in the baseline branches of revised/run.py.
        for k in ("variant", "ssl_mode", "propagation", "augmentation", "loss_weights", "lam", "edge_drop", "feature_drop"):
            copy.pop(k, None)
    return copy


def _atomic_text(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf8", newline="") as f:
            f.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_json(path: Path, data: Any):
    _atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: list[dict], headers: list[str]):
    import io
    stream = io.StringIO()
    w = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _read_run(path: Path) -> dict:
    try:
        raw = path.read_bytes()
        content = json.loads(raw)
        cfg = content["config"]
        assert isinstance(cfg["seed"], int) and isinstance(content["metrics"], dict)
        assert cfg["model"] and cfg["dataset"]
        assert content["candidate_sha256"] and content["protocol"]
    except (ValueError, KeyError, TypeError, AssertionError) as ex:
        raise AggregationError(f"Invalid result at {path}: {ex}") from ex
    audit_path = path.with_name("data_audit.json")
    audit = None
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf8"))
        except (ValueError, OSError) as ex:
            raise AggregationError(f"Unreadable data audit {audit_path}: {ex}") from ex
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "data": content, "audit": audit}


def _check_group(group: list[dict], expected_seeds: tuple[int, ...]) -> tuple[list[dict], list[int]]:
    expected = set(expected_seeds)
    seed_map = {}
    baseline = group[0]
    audit_reference = next((r for r in group if r["audit"] is not None), None)
    baseline_cfg = _effective_config(baseline["data"]["config"])
    compare_keys = ("protocol", "candidate_sha256")
    base_sampling = baseline["data"].get("negative_sampling", {})
    for entry in group:
        d = entry["data"]
        seed = d["config"]["seed"]
        if seed not in expected:
            raise AggregationError(f"Unexpected seed {seed} in {entry['path']}; requested {sorted(expected)}")
        if seed in seed_map:
            raise AggregationError(f"Duplicate seed {seed}: {seed_map[seed]['path']} and {entry['path']}")
        seed_map[seed] = entry
        if _effective_config(d["config"]) != baseline_cfg:
            raise AggregationError(f"Configuration differs from {baseline['path']}: {entry['path']}")
        for key in compare_keys:
            if d.get(key) != baseline["data"].get(key):
                raise AggregationError(f"{key} mismatch between {baseline['path']} and {entry['path']}")
        sampling = d.get("negative_sampling", {})
        for key in ("policy", "seed", "strata"):
            if sampling.get(key) != base_sampling.get(key):
                raise AggregationError(f"negative_sampling.{key} mismatch at {entry['path']}")
        if audit_reference is not None and entry["audit"] is not None:
            if audit_reference["audit"] != entry["audit"]:
                raise AggregationError(f"Different historical graph audit at {entry['path']}")
        # Require every run to report the same optional stratum identities.
        if set(d["metrics"].get("by_stratum", {})) != set(baseline["data"]["metrics"].get("by_stratum", {})):
            raise AggregationError(f"Mismatched test strata at {entry['path']}")
        if set(d["metrics"]) != set(baseline["data"]["metrics"]):
            raise AggregationError(f"Mismatched global metrics at {entry['path']}")
    return [seed_map[i] for i in expected_seeds if i in seed_map], [i for i in expected_seeds if i not in seed_map]


def _numeric_stats(values: list) -> dict | None:
    nums = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    if not nums:
        return None
    return {"mean": statistics.mean(nums), "std": statistics.stdev(nums) if len(nums) > 1 else None,
            "n": len(nums), "by_seed": nums}


def _sort_metrics(keys) -> list[str]:
    return sorted(keys, key=lambda x: (PREFERRED_METRICS.index(x) if x in PREFERRED_METRICS else len(PREFERRED_METRICS), x))


def _summary_target(root: Path, name: str, runs: list[dict]) -> Path:
    # New layout: <root>/<group>/seed42/result.json (group parent common).
    if all(re.fullmatch(r"seed\d+", r["path"].parent.name) for r in runs):
        group_parents = {r["path"].parent.parent.resolve() for r in runs}
        if len(group_parents) == 1 and next(iter(group_parents)).name == name:
            return next(iter(group_parents))
    # Legacy layout: <root>/<group>_seed42/result.json.
    return root / "_summaries" / runs[0]["data"]["config"]["dataset"] / name


def summarize(root: Path | str, expected_seeds=DEFAULT_SEEDS, strict=True) -> dict:
    """Create summaries for all groups; never silently combine incompatible runs.

    On incomplete sets, a status=incomplete manifest is written; never marked final.
    strict=True rejects conflicts (CLI); strict=False records errors (auto-update).
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise AggregationError(f"Results directory not found: {root}")
    expected_seeds = tuple(int(x) for x in expected_seeds)
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise AggregationError("Expected seeds must be nonempty and unique")
    paths = sorted(p for p in root.rglob("result.json") if "_summaries" not in p.parts)
    groups = defaultdict(list)
    errors = []
    for path in paths:
        try:
            item = _read_run(path)
            data = item["data"]
            cfg = data["config"]
            # Separate datasets, never combine synthetic, FAA, etc.
            groups[(cfg["dataset"], group_name(cfg))].append(item)
        except AggregationError as exc:
            if strict:
                raise
            errors.append({"dataset": None, "group": "invalid_result", "error": str(exc)})
    summaries = []
    for (dataset, name), members in sorted(groups.items()):
        try:
            runs, missing = _check_group(members, expected_seeds)
            is_complete = not missing
            cfg = _effective_config(runs[0]["data"]["config"])
            metric_keys = set.intersection(*(set(r["data"]["metrics"].keys()) for r in runs)) - {"by_stratum", "cold_start_note"}
            mstats = {metric: _numeric_stats([r["data"]["metrics"].get(metric) for r in runs]) for metric in _sort_metrics(metric_keys)}
            strata = {}
            strata_names = sorted(runs[0]["data"]["metrics"].get("by_stratum", {}))
            for s in strata_names:
                subruns = [r["data"]["metrics"]["by_stratum"][s] for r in runs]
                counts = {(x.get("positives"), x.get("negatives")) for x in subruns}
                if len(counts) != 1:
                    raise AggregationError(f"Stratum {s} has different test label counts across seeds in {name}")
                metrics_list = [x.get("metrics") for x in subruns]
                if any(x is None for x in metrics_list):
                    stratum_stats = {}
                else:
                    keys = set.intersection(*(set(m) for m in metrics_list))
                    stratum_stats = {k: _numeric_stats([m.get(k) for m in metrics_list]) for k in _sort_metrics(keys)}
                strata[s] = {"positives": subruns[0]["positives"], "negatives": subruns[0]["negatives"], "metrics": stratum_stats}
            output = _summary_target(root, name, runs)
            payload = {"schema": "wbtbgrl-results-v1", "dataset": dataset, "group": name,
                       "status": "complete" if is_complete else "incomplete", "expected_seeds": list(expected_seeds),
                       "present_seeds": [r["data"]["config"]["seed"] for r in runs], "missing_seeds": missing,
                       "configuration": cfg, "protocol": runs[0]["data"]["protocol"],
                       "candidate_sha256": runs[0]["data"]["candidate_sha256"],
                       "negative_sampling": runs[0]["data"].get("negative_sampling"),
                       "metrics": mstats, "by_stratum": strata,
                       "audit_coverage": {"present": sum(r["audit"] is not None for r in runs), "expected": len(expected_seeds)},
                       "runs": [{"seed": r["data"]["config"]["seed"], "path": str(r["path"].relative_to(root)),
                                 "result_sha256": r["sha256"], "has_audit": r["audit"] is not None} for r in runs]}
            _atomic_json(output / "summary.json", payload)
            rows = [{"metric": metric, "mean": item["mean"] if item else "",
                     "std": item["std"] if item and item["std"] is not None else "",
                     "n": item["n"] if item else 0, "complete": is_complete}
                    for metric, item in mstats.items()]
            _atomic_csv(output / "summary.csv", rows, ["metric", "mean", "std", "n", "complete"])
            srows = [{"stratum": s, "positives": details["positives"], "negatives": details["negatives"],
                      "metric": k, "mean": stat["mean"] if stat else "", "std": stat["std"] if stat and stat["std"] is not None else "",
                      "n": stat["n"] if stat else 0, "complete": is_complete}
                     for s, details in strata.items() for k, stat in details["metrics"].items()]
            _atomic_csv(output / "summary_by_stratum.csv", srows,
                        ["stratum", "positives", "negatives", "metric", "mean", "std", "n", "complete"])
            summaries.append({"dataset": dataset, "group": name, "status": payload["status"],
                              "present_seeds": payload["present_seeds"], "missing_seeds": missing,
                              "path": str(output.relative_to(root)),
                              "roc_auc_mean": mstats.get("ROC-AUC", {}).get("mean") if mstats.get("ROC-AUC") else None,
                              "ap_mean": mstats.get("AP", {}).get("mean") if mstats.get("AP") else None})
        except AggregationError as ex:
            errors.append({"dataset": dataset, "group": name, "error": str(ex)})
    _atomic_csv(root / "summary_index.csv", summaries,
                ["dataset", "group", "status", "present_seeds", "missing_seeds", "path", "roc_auc_mean", "ap_mean"])
    _atomic_json(root / "summary_index.json", {"expected_seeds": expected_seeds, "groups": summaries, "errors": errors})
    if strict and errors:
        raise AggregationError("Incompatible results:\n" + "\n".join(f"{x['group']}: {x['error']}" for x in errors))
    return {"groups": summaries, "errors": errors}


def export_compact(root: Path, destination: Path, include_scores=False):
    """Opt-in sharing export; no checkpoints, raw data, or training candidate lists."""
    entries = []
    for pattern in ("result.json", "data_audit.json", "summary.json", "summary.csv", "summary_by_stratum.csv", "summary_index.json", "summary_index.csv"):
        entries.extend(root.rglob(pattern))
    if include_scores:
        entries.extend(root.rglob("test_scores.csv"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for file in sorted(set(entries)):
            if file.resolve() != destination.resolve() and file.is_file():
                z.write(file, file.relative_to(root).as_posix())


def main(argv=None):
    p = argparse.ArgumentParser(description="Consolidate experimental results with verified shared candidates and protocol")
    p.add_argument("--results", required=True, type=Path, help="Root containing run folders")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--export", type=Path, help="Optional small archive; excludes checkpoints and data")
    p.add_argument("--with-scores", action="store_true", help="Include prediction CSVs in optional export (may contain sensitive IDs)")
    a = p.parse_args(argv)
    outcome = summarize(a.results, a.seeds)
    for s in outcome["groups"]:
        print(f"{s['group']}: {s['status']} ({len(s['present_seeds'])}/{len(a.seeds)} seeds), {s['path']}")
    if a.export:
        export_compact(a.results.resolve(), a.export.resolve(), a.with_scores)
        print(f"Exported: {a.export}")
    if not outcome["groups"]:
        p.error("No result.json found")
    return outcome


if __name__ == "__main__":
    main()
