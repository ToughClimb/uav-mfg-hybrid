"""Select verified high-demand experiments and preserve their original trials."""

import json
from pathlib import Path
import shutil

from uav_mfg.paper_plots import CASES


def main():
    for dimension, names in ((2, [name for name, _ in CASES]),
                             (3, ["p2p_none", "p2p_obstacle", "p2p_height_obstacle"])):
        initial_root = Path(f"results/high_density_{dimension}d")
        output = Path(f"results/high_density_verified_{dimension}d")
        selections = []
        for name in names:
            original = initial_root / name
            source = original
            if dimension == 2 and name == "homing_uniform":
                source = Path("results/high_density_continuation_2d") / name
            if dimension == 2 and name == "homing_uniform_obstacle":
                source = Path("results/high_density_homing_obstacle_continued") / name
            metadata = json.loads((source / "metadata.json").read_text())
            diagnostics = json.loads((source / "diagnostics.json").read_text())
            if metadata["status"] != "converged" or diagnostics["total_projection_mass_removed"] != 0:
                raise RuntimeError(f"Cannot select {source}: {metadata['status']}")
            destination = output / name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace an existing run: {destination}")
            selections.append({"case": name, "selected_source": str(source), "destination": str(destination),
                               "source_rate": metadata["config"]["source"]["rate"],
                               "initial_trial": str(original),
                               "initial_metadata": json.loads((original / "metadata.json").read_text())})
        # Validate the whole group before copying any case.
        output.mkdir(parents=True, exist_ok=True)
        for item in selections:
            shutil.copytree(item["selected_source"], item["destination"])
        (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
        print(f"Selected {len(selections)} verified {dimension}D cases: {output}")


if __name__ == "__main__":
    main()
