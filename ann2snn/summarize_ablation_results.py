import argparse
import json
import math
from pathlib import Path


PREFERRED_ORDER = {
    "no_spike_sanity": 0,
    "block0": 10,
    "block01": 20,
    "block0123": 30,
    "patch_block01": 40,
}


def parse_args():
    parser = argparse.ArgumentParser("Summarize VideoMamba ANN2SNN ablation results")
    parser.add_argument("--root", default="./outputs/ann2snn_videomamba")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def fmt(value, digits=4):
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def fmt_indices(value):
    if value is None:
        return "-"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "none"
    text = str(value)
    return text if text else "none"


def load_results(root):
    rows = []
    for summary_path in sorted(root.glob("*/conversion_summary.json")):
        run_name = summary_path.parent.name
        try:
            metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append({"run": run_name, "error": f"json error: {exc}"})
            continue

        ann_acc1 = metrics.get("ann_acc1")
        snn_acc1 = metrics.get("snn_acc1")
        drop = None
        if ann_acc1 is not None and snn_acc1 is not None:
            drop = float(snn_acc1) - float(ann_acc1)

        rows.append(
            {
                "run": run_name,
                "spike_patch": metrics.get("spike_patch"),
                "blocks": fmt_indices(metrics.get("spike_block_indices")),
                "threshold_scale": metrics.get("threshold_scale"),
                "timesteps": metrics.get("timesteps"),
                "delay": metrics.get("delay"),
                "calibration_mse": metrics.get("calibration_mse"),
                "calibration_delta": metrics.get("calibration_delta"),
                "ann_acc1": ann_acc1,
                "snn_acc1": snn_acc1,
                "drop": drop,
                "test_snn_acc1": metrics.get("test_snn_acc1"),
                "test_snn_acc5": metrics.get("test_snn_acc5"),
            }
        )
    return sorted(rows, key=lambda row: (PREFERRED_ORDER.get(row["run"], 999), row["run"]))


def render_table(rows):
    lines = [
        "# VideoMamba ANN2SNN Ablation Summary",
        "",
        "| run | patch | blocks | scale | T | delay | calib_mse | ann_acc1 | snn_acc1 | drop | test_snn_acc1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['run']} | error | {row['error']} | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {run} | {patch} | {blocks} | {scale} | {timesteps} | {delay} | {mse} | {ann} | {snn} | {drop} | {test} |".format(
                run=row["run"],
                patch=row["spike_patch"],
                blocks=row["blocks"],
                scale=fmt(row["threshold_scale"], 2),
                timesteps=row["timesteps"],
                delay=row["delay"],
                mse=fmt(row["calibration_mse"], 6),
                ann=fmt(row["ann_acc1"]),
                snn=fmt(row["snn_acc1"]),
                drop=fmt(row["drop"]),
                test=fmt(row["test_snn_acc1"]),
            )
        )
    if not rows:
        lines.append("| no results | - | - | - | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = Path(args.root)
    rows = load_results(root)
    table = render_table(rows)
    print(table)

    output = Path(args.output) if args.output else root / "ablation_summary.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(f"Saved summary to {output}")


if __name__ == "__main__":
    main()
