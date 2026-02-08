"""Cleanup plots under opencode/runs.

Default: delete all PNG files under runs/*/plots (keep PDFs). This is useful for
keeping repo size small for GitHub.

Default behavior is dry-run (no deletion). Use --apply to actually delete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def cleanup_runs_plots(root_runs: Path, apply: bool, verbose: bool, *,
                       delete_png: bool, delete_pdf: bool) -> int:
    if not root_runs.exists():
        raise FileNotFoundError(f"runs folder not found: {root_runs}")

    n_deleted = 0
    for run_dir in sorted([p for p in root_runs.iterdir() if p.is_dir()]):
        plots_dir = run_dir / "plots"
        if not plots_dir.exists():
            continue

        for fp in sorted(plots_dir.iterdir()):
            if not fp.is_file():
                continue

            suffix = fp.suffix.lower()
            if suffix not in (".png", ".pdf"):
                continue

            if suffix == ".png" and not delete_png:
                continue
            if suffix == ".pdf" and not delete_pdf:
                continue

            if verbose:
                print(f"DELETE: {fp}")

            if apply:
                fp.unlink()

            n_deleted += 1

    return n_deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete non-essential plot files under runs/*/plots")
    parser.add_argument(
        "--runs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "runs"),
        help="Path to runs folder (default: <repo>/runs)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default: dry-run)",
    )
    parser.add_argument(
        "--delete-png",
        action="store_true",
        help="Delete .png files (default: enabled)",
    )
    parser.add_argument(
        "--delete-pdf",
        action="store_true",
        help="Delete .pdf files (default: disabled)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every file that would be deleted",
    )

    args = parser.parse_args()
    root_runs = Path(args.runs)

    delete_png = True if args.delete_png is False else bool(args.delete_png)
    n_deleted = cleanup_runs_plots(
        root_runs=root_runs,
        apply=bool(args.apply),
        verbose=bool(args.verbose),
        delete_png=delete_png,
        delete_pdf=bool(args.delete_pdf),
    )

    if args.apply:
        print(f"Done. Deleted {n_deleted} files.")
    else:
        print(f"Dry-run done. Would delete {n_deleted} files. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
