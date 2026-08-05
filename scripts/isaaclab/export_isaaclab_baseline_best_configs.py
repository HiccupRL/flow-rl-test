#!/usr/bin/env python3
"""Export the selected IsaacLab baseline configs into a self-contained bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"manifest has no header: {manifest}")
    required = {"slug", "seed", "config_path", "config_sha256", "selection_status"}
    missing = required.difference(fieldnames)
    if missing:
        raise SystemExit(f"manifest is missing fields: {sorted(missing)}")
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} rows, found {len(rows)}")

    exported_rows: list[dict[str, str]] = []
    destination_names: set[str] = set()
    for row in rows:
        source = (repo_root / row["config_path"]).resolve()
        if not source.is_file():
            raise SystemExit(f"missing config: {source}")
        source_digest = sha256(source)
        if source_digest != row["config_sha256"]:
            raise SystemExit(
                f"config digest mismatch for {source}: "
                f"{source_digest} != {row['config_sha256']}"
            )

        destination_name = f"{row['slug']}_seed{row['seed']}.yaml"
        if destination_name in destination_names:
            raise SystemExit(f"duplicate export name: {destination_name}")
        destination_names.add(destination_name)
        destination = configs_dir / destination_name
        shutil.copyfile(source, destination)
        if sha256(destination) != source_digest:
            raise SystemExit(f"copied config digest mismatch: {destination}")

        exported_row = dict(row)
        exported_row["exported_config"] = str(
            destination.relative_to(output_dir)
        )
        exported_rows.append(exported_row)

    exported_manifest = output_dir / "manifest.tsv"
    with exported_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[*fieldnames, "exported_config"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(exported_rows)

    status_counts = Counter(row["selection_status"] for row in rows)
    statuses = "\n".join(
        f"- `{status}`: {count}" for status, count in sorted(status_counts.items())
    )
    readme = output_dir / "README.md"
    readme.write_text(
        "# IsaacLab baseline best configs (200M)\n\n"
        f"This bundle contains {len(rows)} byte-for-byte copies of the configs "
        "selected by the source manifest. Each SHA-256 digest was verified "
        "before and after copying.\n\n"
        "Selection status:\n\n"
        f"{statuses}\n\n"
        "`manifest.tsv` records the original provenance, reference return, "
        "source config path, digest, and exported config path. Entries marked "
        "`historical_incomplete_fallback` had no completed historical 200M "
        "candidate; they are the best available fallback, not a terminal-best "
        "selection.\n",
        encoding="utf-8",
    )
    print(f"exported {len(rows)} verified configs to {output_dir}")


if __name__ == "__main__":
    main()
