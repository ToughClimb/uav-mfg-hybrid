"""Select accepted independent runs without replacing any failed trial."""

import argparse
import json
import shutil
from pathlib import Path

from uav_mfg.paper_plots import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=Path("results/paper_figures_2d"))
    parser.add_argument("--accepted", type=Path, default=Path("results/paper_figures_verified"))
    args = parser.parse_args()
    records = []
    for name, _ in CASES:
        source = args.trials / name if name.startswith("homing") else args.accepted / name
        original = args.trials / name
        metadata = json.loads((source / "metadata.json").read_text())
        diagnostics = json.loads((source / "diagnostics.json").read_text())
        if metadata["status"] != "converged" or diagnostics["total_projection_mass_removed"] != 0:
            raise RuntimeError(f"Cannot select unverified {name}: {metadata['status']}")
        destination = args.accepted / name
        if source != destination:
            if destination.exists():
                raise FileExistsError(f"Refusing to replace an existing run: {destination}")
            shutil.copytree(source, destination)
        first_metadata = json.loads((original / "metadata.json").read_text())
        records.append({"case": name, "selected_source": str(source), "destination": str(destination),
                        "selected_source_rate": metadata["config"]["source"]["rate"],
                        "original_trial": str(original), "original_status": first_metadata["status"],
                        "original_source_rate": first_metadata["config"]["source"]["rate"],
                        "original_diagnostics": first_metadata["diagnostics"]})
    (args.accepted / "selection.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"accepted_cases": len(records), "selection": str(args.accepted / "selection.json")}))


if __name__ == "__main__":
    main()
