#!/usr/bin/env python
"""Merge validation and test CSV files for the held-out view."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge held-out validation and test CSV files into one CSV."
    )
    parser.add_argument(
        "--data-path",
        default="/data/users/ouyangys/data/multiview_action_videos/",
        help="Directory containing the CSV files.",
    )
    parser.add_argument("--val-csv", default="v03_val_set.csv")
    parser.add_argument("--test-csv", default="v03_test_set.csv")
    parser.add_argument("--output-csv", default="v03_val_test_set.csv")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument(
        "--dedupe",
        choices=("none", "sample", "sample_label"),
        default="none",
        help="Optional duplicate removal mode. Default keeps all rows.",
    )
    return parser.parse_args()


def resolve_csv(data_path, csv_path):
    path = Path(csv_path)
    if path.is_absolute():
        return path
    return Path(data_path) / path


def read_rows(path, delimiter):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row or len(row) < 2:
                continue
            sample = row[0].strip()
            label = row[1].strip()
            if not sample:
                continue
            rows.append([sample, label, *row[2:]])
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def dedupe_rows(rows, mode):
    if mode == "none":
        return rows

    seen = set()
    merged = []
    for row in rows:
        key = row[0] if mode == "sample" else (row[0], row[1])
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def write_rows(path, rows, delimiter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerows(rows)


def main():
    args = parse_args()
    val_path = resolve_csv(args.data_path, args.val_csv)
    test_path = resolve_csv(args.data_path, args.test_csv)
    output_path = resolve_csv(args.data_path, args.output_csv)

    val_rows = read_rows(val_path, args.delimiter)
    test_rows = read_rows(test_path, args.delimiter)
    merged_rows = dedupe_rows(val_rows + test_rows, args.dedupe)
    write_rows(output_path, merged_rows, args.delimiter)

    print(f"Validation rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Merged rows: {len(merged_rows)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
