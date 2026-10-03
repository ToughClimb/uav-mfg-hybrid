"""Paper-inspired layouts; plotted fields are generated solely by the new solver."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml


def style():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.titlesize": 12,
                         "figure.dpi": 120, "savefig.dpi": 180, "axes.labelsize": 11})


def load(root, name):
    directory = Path(root) / name
    with (directory / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with np.load(directory / "fields.npz", allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    return config, fields


def markers(ax, config, z, projected=False):
    target = config["target"]
    if target.get("type", "sphere") in ("sphere", "disk"):
        square = target["radius"] ** 2 if target.get("type") == "disk" else target["radius"] ** 2 - (z - target["center"][2]) ** 2
        if projected or square > 0:
            radius = target["radius"] if projected else np.sqrt(square)
            ax.add_patch(Circle(target["center"][:2], radius, fill=False, edgecolor="red", linestyle="--", linewidth=1.6))
    else:
        lo, hi = target["lower"], target["upper"]
        if projected or lo[2] <= z <= hi[2]:
            ax.add_patch(Rectangle(lo[:2], hi[0]-lo[0], hi[1]-lo[1], fill=False, edgecolor="red", linestyle="--"))
    source = config["source"]
    if source["type"] in ("sphere", "disk"):
        square = source["radius"] ** 2 if source["type"] == "disk" else source["radius"] ** 2 - (z - source["center"][2]) ** 2
        if projected or square > 0:
            ax.add_patch(Circle(source["center"][:2], source["radius"] if projected else np.sqrt(square), fill=False, edgecolor="green", linewidth=1.6))
    for box in config["obstacles"]:
        lo, hi = box["lower"], box["upper"]
        if lo[2] <= z <= hi[2]:
            ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], facecolor="#606060", edgecolor="#252525", zorder=5))
    ax.set(xlim=config["domain"][0], ylim=config["domain"][1], aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.grid(alpha=0.2, color="white", linewidth=0.5)


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.png", bbox_inches="tight")
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def density_scale_comparison(root, destination, upper):
    """Show identical computed density with two explicitly labeled mappings."""
    name=next((name for name in ("p2p_vortex_obstacle","p2p_obstacle","p2p_none")
               if (root/name/"fields.npz").exists()),None)
    if name is None:
        return
    config,data=load(root,name)
    k=int(np.abs(data["z"]-15).argmin())
    background=np.ma.array(data["density"][:,:,k].T,mask=data["obstacle"][:,:,k].T)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for ax,norm,title in zip(axes,(Normalize(0,upper),PowerNorm(.5,vmin=0,vmax=upper)),
                              ("Linear color scale","Square-root color scale (gamma=0.5)")):
        image=ax.pcolormesh(data["x"],data["y"],background,cmap="viridis",norm=norm,shading="nearest")
        markers(ax,config,float(data["z"][k]))
        fig.colorbar(image,ax=ax,label=r"Density $\rho$ (UAVs/m$^3$)",shrink=.8)
        ax.set_title(title)
    fig.suptitle(f"Same computed density; {name.replace('_',' ')}; z={data['z'][k]:.2f} m")
    save(fig,destination,"density_scale_comparison")


def make_figures(root):
    root = Path(root)
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    style()
    names = ["homing_none", "homing_uniform", "homing_uniform_obstacle", "p2p_none", "p2p_vortex", "p2p_vortex_obstacle"]
    existing = [name for name in names if (root / name / "fields.npz").exists()]
    if not existing:
        existing = [p.name for p in sorted(root.iterdir()) if p.is_dir() and (p/"fields.npz").exists()][:6]
    if existing:
        rows = 2 if len(existing) > 3 else 1
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False, layout="constrained")
        datasets = [load(root, name) for name in existing]
        upper = max(float(np.max(data["phi"])) for _, data in datasets)
        norm = Normalize(0, upper)
        for ax, name, (config, data) in zip(axes.ravel(), existing, datasets):
            height = 5 if name.startswith("homing") else 15
            k = int(np.abs(data["z"] - height).argmin())
            z = float(data["z"][k])
            background = np.ma.array(data["phi"][:, :, k].T, mask=data["obstacle"][:, :, k].T)
            image = ax.pcolormesh(data["x"], data["y"], background, cmap="viridis", norm=norm, shading="nearest")
            ax.contour(data["x"], data["y"], background, levels=np.linspace(0, upper, 18), colors="white", linewidths=0.35, alpha=0.55)
            markers(ax, config, z)
            ax.set_title(name.replace("_", " ") + f"\nz={z:.2f} m")
        for ax in axes.ravel()[len(existing):]:
            ax.set_visible(False)
        fig.colorbar(image, ax=list(axes.ravel()[:len(existing)]), label=r"Value $\phi$ (s)", shrink=0.85)
        fig.suptitle("Independent ESWA reconstruction — value fields", fontsize=15)
        save(fig, destination, "value_comparison")

    height_cases = [name for name in ("p2p_none", "p2p_obstacle", "p2p_height_obstacle") if (root/name/"fields.npz").exists()]
    if height_cases:
        datasets = [load(root, name) for name in height_cases]
        upper = max(float(np.max(data["density"])) for _, data in datasets)
        for mapping,norm,suffix in (("linear",Normalize(0,upper),""),
                                   ("square-root",PowerNorm(.5,vmin=0,vmax=upper),"_sqrt")):
            fig, axes = plt.subplots(len(height_cases), 4, figsize=(13, 3.1 * len(height_cases)), squeeze=False, layout="constrained")
            for row, (name, (config, data)) in enumerate(zip(height_cases, datasets)):
                # Include the central transport corridor as well as distinct heights.
                lo,hi=config["domain"][2]
                heights=lo+(hi-lo)*np.array([1/6,1/3,1/2,2/3])
                levels=np.asarray([int(np.abs(data["z"]-height).argmin()) for height in heights])
                for col, k in enumerate(levels):
                    ax = axes[row, col]
                    background = np.ma.array(data["density"][:, :, k].T, mask=data["obstacle"][:, :, k].T)
                    image = ax.pcolormesh(data["x"], data["y"], background, cmap="viridis", norm=norm, shading="nearest")
                    markers(ax, config, float(data["z"][k]))
                    ax.set_title(f"z={data['z'][k]:.2f} m")
                    if col == 0:
                        ax.set_ylabel(name.replace("p2p_", "").replace("_", " ") + "\ny (m)")
            fig.colorbar(image, ax=list(axes.ravel()), label=r"Density $\rho$ (UAVs/m$^3$)", shrink=0.7)
            fig.suptitle(f"3D density slices — shared {mapping} color scale", fontsize=15)
            save(fig, destination, "density_height_slices"+suffix)
        density_scale_comparison(root,destination,upper)

    featured = next((name for name in ("p2p_vortex_obstacle", "p2p_height_obstacle", "p2p_obstacle", "p2p_none") if (root/name/"fields.npz").exists()), None)
    if featured:
        config, data = load(root, featured)
        k = int(np.abs(data["z"] - 15).argmin())
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        for ax, field, label in zip(axes, ("phi", "density"), (r"$\phi$ (s)", r"$\rho$ (UAVs/m$^3$)")):
            image = ax.pcolormesh(data["x"], data["y"], np.ma.array(data[field][:,:,k].T, mask=data["obstacle"][:,:,k].T), cmap="viridis", shading="nearest")
            fig.colorbar(image, ax=ax, label=label, shrink=0.75)
            markers(ax, config, float(data["z"][k]))
        # These are in-plane instantaneous streamlines, not 3D particle paths.
        u = np.ma.array(data["velocity"][:,:,k,0].T, mask=(~data["active"][:,:,k]).T)
        v = np.ma.array(data["velocity"][:,:,k,1].T, mask=(~data["active"][:,:,k]).T)
        axes[0].streamplot(data["x"], data["y"], u, v, color="white", density=1.3, linewidth=0.55, arrowsize=0.7)
        axes[0].set_title("Induced in-plane streamlines")
        axes[1].set_title("Computed transport density")
        fig.suptitle(featured.replace("_", " ") + f"; z={data['z'][k]:.2f} m")
        save(fig, destination, "motion_and_density")

    diagnostics = []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and (directory / "diagnostics.json").exists():
            diagnostics.append({"case": directory.name, **json.loads((directory / "diagnostics.json").read_text())})
    (destination / "summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if diagnostics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        labels = [d["case"].replace("p2p_", "").replace("homing_", "H: ") for d in diagnostics]
        axes[0].bar(np.arange(len(labels)), [d["absorption_over_source"] for d in diagnostics], color="#21918c")
        axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Target absorption / source")
        axes[1].bar(np.arange(len(labels)), [d["relative_steady_balance"] for d in diagnostics], color="#440154")
        axes[1].set_yscale("log"); axes[1].set_ylabel("Relative steady mass imbalance")
        axes[2].bar(np.arange(len(labels)), [d["eikonal_mean"] for d in diagnostics], color="#3b528b")
        axes[2].set_ylabel("Mean absolute Eikonal residual")
        for ax in axes:
            ax.set_xticks(np.arange(len(labels)), labels, rotation=70, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
        save(fig, destination, "conservation_diagnostics")
    return destination
