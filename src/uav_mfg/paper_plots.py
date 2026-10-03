"""Manuscript figure conventions applied exclusively to independently solved fields.

Contours interpolate cell-center values for display. No density scaling,
filtering, painting, or reference-image digitization is performed.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from .plots import load, save
from .problem import Problem


CASES = (
    ("homing_none", "Homing, no wind"),
    ("homing_uniform", "Homing, x-wind"),
    ("homing_uniform_obstacle", "Homing, x-wind + obstacle"),
    ("p2p_none", "P2P, no wind"),
    ("p2p_vortex", "P2P, vortex"),
    ("p2p_vortex_obstacle", "P2P, vortex + obstacle"),
)


def figure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 13,
        "figure.dpi": 120, "savefig.dpi": 220,
        "pdf.fonttype": 42, "axes.linewidth": .85,
    })


def at_height(data, field, height):
    """Linear interpolation between saved heights; refuse extrapolation."""
    z = data["z"]
    if len(z) == 1:
        if not np.isclose(height, z[0]):
            raise ValueError("A planar result can only be plotted at its recorded height")
        return data[field][:, :, 0]
    if height < z[0] or height > z[-1]:
        raise ValueError("Requested figure height is outside saved cell centers")
    right = int(np.searchsorted(z, height, side="right"))
    right = min(max(right, 1), len(z) - 1)
    left = right - 1
    fraction = (height - z[left]) / (z[right] - z[left])
    return (1 - fraction) * data[field][:, :, left] + fraction * data[field][:, :, right]


def decorate(ax, config, height, *, projected=False, legend=True, compact=False):
    handles, labels = [], []
    for region, color, dashed, label in (
        (config["target"], "#ff2020", True, "Target (sink)"),
        (config["source"], "#008000", False, "Source"),
    ):
        if region["type"] == "homing":
            continue
        kind = region.get("type", "sphere")
        if kind in ("sphere", "disk"):
            radius = region["radius"]
            if kind == "sphere" and not projected:
                square = radius ** 2 - (height - region["center"][2]) ** 2
                if square <= 0:
                    continue
                radius = np.sqrt(square)
            patch = Circle(region["center"][:2], radius, fill=False,
                           edgecolor=color, linestyle="--" if dashed else "-",
                           linewidth=1.8, zorder=8)
        elif kind == "box":
            lo, hi = region["lower"], region["upper"]
            if not projected and not lo[2] <= height <= hi[2]:
                continue
            patch = Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fill=False,
                              edgecolor=color, linestyle="--" if dashed else "-", zorder=8)
        else:
            raise ValueError(f"Unsupported figure marker: {kind}")
        ax.add_patch(patch)
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=color,
                                 linestyle="--" if dashed else "-", linewidth=1.7))
        labels.append(label)
    obstacle_added = False
    for box in config["obstacles"]:
        lo, hi = box["lower"], box["upper"]
        if lo[2] <= height <= hi[2]:
            ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1],
                                   facecolor="#606060", edgecolor="#282828",
                                   linewidth=1, alpha=.85, zorder=7))
            obstacle_added = True
    if obstacle_added:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="#606060", edgecolor="#282828"))
        labels.append("Obstacle")
    ax.set(xlim=config["domain"][0], ylim=config["domain"][1], aspect="equal",
           xlabel="x (m)", ylabel="y (m)")
    ticks = np.arange(0, 101, 20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color="white", alpha=.2, linewidth=.45)
    ax.tick_params(labelsize=10 if compact else 12)
    if legend and handles:
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.24, 1.02),
                  borderaxespad=0, fontsize=9 if compact else 11,
                  handlelength=2.2, labelspacing=.25, borderpad=.35)


def contours(ax, config, data, field, height, upper, *, mapping="linear", alpha=1, allow_saturation=False):
    values = at_height(data, field, height).T
    obstacle = at_height(data, "obstacle", height).T > .5
    values = np.ma.array(values, mask=obstacle)
    # Extend the nearest cell center to its outer wall solely for rendering.
    x = np.r_[config["domain"][0][0], data["x"], config["domain"][0][1]]
    y = np.r_[config["domain"][1][0], data["y"], config["domain"][1][1]]
    values = np.ma.array(np.pad(values.filled(0), 1, mode="edge"),
                         mask=np.pad(np.ma.getmaskarray(values), 1, mode="edge"))
    norm = PowerNorm(.5, vmin=0, vmax=upper) if mapping == "sqrt" else Normalize(0, upper)
    # Uniform color intervals preserve the discrete viridis bars in the paper.
    levels = upper * np.linspace(0, 1, 25) ** (2 if mapping == "sqrt" else 1)
    maximum = float(values.max())
    exceeds = maximum > upper * (1 + 1e-9)
    if exceeds and not allow_saturation:
        raise ValueError(f"Color range {upper} would hide the actual {field} maximum {maximum}")
    image = ax.contourf(x, y, values, levels=levels, norm=norm, cmap="viridis",
                        alpha=alpha, zorder=1, extend="max" if exceeds else "neither")
    if maximum > levels[1]:
        ax.contour(x, y, values, levels=levels[1:-1], colors="white",
                   linewidths=.3, alpha=.32, zorder=2)
    return image


def colorbar(fig, image, ax, field, upper, *, compact=False):
    ticks = np.arange(0, 7.21, 1.2) if field == "phi" and upper == 8 else np.linspace(0, upper, 6)
    if field == "density" and upper == .05:
        ticks = np.arange(0, .04501, .0075)
    bar = fig.colorbar(image, ax=ax, pad=.022, fraction=.048, shrink=.96, ticks=ticks)
    bar.set_label(r"$\phi$ (s)" if field == "phi" else r"$\rho$ (UAVs/m$^3$)",
                  fontsize=10 if compact else 13)
    bar.ax.tick_params(labelsize=8 if compact else 11)
    if field == "density":
        bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    return bar


def value_panels(root, destination):
    fig, axes = plt.subplots(2, 3, figsize=(13.7, 9.5))
    fig.subplots_adjust(left=.05, right=.975, bottom=.09, top=.91, wspace=.38, hspace=.62)
    for index, (ax, (name, label)) in enumerate(zip(axes.ravel(), CASES)):
        config, data = load(root, name)
        height = float(data["z"][0])
        image = contours(ax, config, data, "phi", height, 8)
        decorate(ax, config, height, compact=True)
        ax.set_title(rf"$\phi$ at z={height:g}m", pad=7)
        colorbar(fig, image, ax, "phi", 8, compact=True)
        ax.text(.5, -.27, f"({chr(97 + index)}) {label}", transform=ax.transAxes,
                ha="center", va="top", fontsize=12, family="DejaVu Serif")
        ax.text(.5, -.36, f"q = {config['source']['rate']:g}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="#555555")
    save(fig, destination, "fig2_value")


def source_streamlines(ax, config, data, height):
    """Integrate actual in-plane velocities from a few source positions."""
    xy = (data["x"], data["y"])
    velocity = at_height(data, "velocity", height)[..., :2]
    interpolator = RegularGridInterpolator(xy, velocity, bounds_error=False, fill_value=None)
    target = config["target"]
    center, radius = np.asarray(config["source"]["center"][:2]), config["source"]["radius"]
    # A perpendicular segment across the source samples both sides of the plume.
    transverse = np.array([-1., 1.]) / np.sqrt(2)
    start_points = center + np.linspace(-.8, .8, 8)[:, None] * radius * transverse
    bounds = np.asarray(config["domain"][:2])
    target_xy = np.asarray(target["center"][:2])

    def rhs(_, point):
        vector = np.asarray(interpolator(point[None]))[0]
        speed = np.linalg.norm(vector)
        return vector / max(speed, 1e-12)

    def hit_target(_, point):
        return np.linalg.norm(point - target_xy) - target["radius"]

    def hit_wall(_, point):
        return min((point - bounds[:, 0]).min(), (bounds[:, 1] - point).min())

    hit_target.terminal = hit_wall.terminal = True
    hit_target.direction = hit_wall.direction = -1
    events = [hit_target, hit_wall]
    for box in config["obstacles"]:
        lo, hi = np.asarray(box["lower"][:2]), np.asarray(box["upper"][:2])
        def hit_obstacle(_, point, lo=lo, hi=hi):
            offset = np.abs(point - (lo + hi) / 2) - (hi - lo) / 2
            return np.linalg.norm(np.maximum(offset, 0)) + min(offset.max(), 0)
        hit_obstacle.terminal, hit_obstacle.direction = True, -1
        events.append(hit_obstacle)
    paths = []
    for start in start_points:
        trajectory = solve_ivp(rhs, (0, 230), start, max_step=.35,
                               rtol=1e-6, atol=1e-7, events=events)
        if not trajectory.success:
            raise RuntimeError(trajectory.message)
        path = trajectory.y.T
        line, = ax.plot(path[:, 0], path[:, 1], color="white", linewidth=1.1, zorder=6)
        line.set_path_effects([effects.Stroke(linewidth=1.9, foreground="#4b4b4b"), effects.Normal()])
        paths.append({"start": start.tolist(), "end": path[-1].tolist(),
                      "reached_target": bool(len(trajectory.t_events[0])),
                      "stopped_at_obstacle": any(len(t) for t in trajectory.t_events[2:])})
    return paths


def motion_and_density(root, destination):
    config, data = load(root, "p2p_vortex_obstacle")
    height = float(data["z"][0])
    fig, ax = plt.subplots(figsize=(7.9, 8.1))
    fig.subplots_adjust(left=.11, right=.87, bottom=.095, top=.84)
    image = contours(ax, config, data, "phi", height, 8, alpha=.55)
    paths = source_streamlines(ax, config, data, height)
    decorate(ax, config, height)
    ax.set_title(rf"$\mathbf{{u}}$ at z={height:g}m", pad=9)
    colorbar(fig, image, ax, "phi", 8)
    save(fig, destination, "fig3_streamlines")

    peak = float(data["density"].max())
    adaptive = np.ceil(peak / .005) * .005
    for name, upper, mapping in (("fig4_density", .05, "linear"),
                                  ("fig4_density_relative", adaptive, "linear"),
                                  ("fig4_density_sqrt", adaptive, "sqrt")):
        fig, ax = plt.subplots(figsize=(7.9, 8.1))
        fig.subplots_adjust(left=.11, right=.87, bottom=.095, top=.84)
        image = contours(ax, config, data, "density", height, upper, mapping=mapping,
                         allow_saturation=(name == "fig4_density"))
        decorate(ax, config, height)
        title = rf"$\rho$ at z={height:g}m"
        if mapping == "sqrt":
            title += " (sqrt color scale)"
        ax.set_title(title, pad=9)
        colorbar(fig, image, ax, "density", upper)
        save(fig, destination, name)
    return {"source_seeded_streamlines": paths, "density_peak": peak,
            "paper_density_range": [0, .05], "adaptive_density_range": [0, float(adaptive)],
            "density_source_region_peak": float(data["density"][data["source"] > 0].max()),
            "source_rate": config["source"]["rate"], "height": height,
            "density_upper_bound": config["physics"]["rho_max"],
            "paper_color_saturation": bool(peak > .05)}


def heights(root, destination):
    cases = (("p2p_none", "No obstacle"), ("p2p_obstacle", "Obstacle, no wind"),
             ("p2p_height_obstacle", "Obstacle, height wind"))
    datasets = [(name, label, *load(root, name)) for name, label in cases]
    altitudes = (1.875, 9.375, 20.625, 28.125)
    recorded = []
    for name, _, _, data in datasets:
        for height in altitudes:
            recorded.append({"case": name, "height": height,
                             "maximum_density": float(at_height(data, "density", height).max())})
    peak = max(item["maximum_density"] for item in recorded)
    magnitude = 10 ** np.floor(np.log10(peak))
    adaptive = np.ceil(peak / magnitude) * magnitude
    for filename, upper in (("height_slices", .05), ("height_slices_relative", adaptive)):
        fig, axes = plt.subplots(3, 4, figsize=(13.3, 9.4))
        fig.subplots_adjust(left=.095, right=.895, bottom=.1, top=.94, wspace=.11, hspace=.26)
        for row, (name, label, config, data) in enumerate(datasets):
            for col, height in enumerate(altitudes):
                ax = axes[row, col]
                image = contours(ax, config, data, "density", height, upper,
                                 allow_saturation=(filename == "height_slices"))
                decorate(ax, config, height, projected=True, legend=False, compact=True)
                ax.set_xticks([0, 50, 100])
                ax.set_yticks([0, 50, 100])
                ax.set_title(f"z={height:.1f}m", fontsize=12, family="DejaVu Serif")
                if col:
                    ax.set_ylabel("")
                if row < 2:
                    ax.set_xlabel("")
                if col == 0:
                    ax.set_ylabel(label + "\ny (m)", fontsize=10)
        bar_ax = fig.add_axes((.925, .13, .016, .72))
        ticks = np.arange(0, .04501, .0075) if upper == .05 else np.linspace(0, upper, 6)
        bar = fig.colorbar(image, cax=bar_ax, ticks=ticks)
        bar.set_label(r"$\rho$ (UAVs/m$^3$)")
        bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
        save(fig, destination, filename)
    return {"geometry_markers": "XY projections, as in manuscript height gallery",
            "height_interpolation": "linear between independently saved cell centers", "slices": recorded,
            "adaptive_shared_range": [0, float(adaptive)], "paper_shared_range": [0, .05],
            "source_rate": datasets[0][2]["source"]["rate"], "grid": list(datasets[0][3]["density"].shape)}


def methods(root, destination):
    locations = (Path(root) / "p2p_none", Path(root) / "baselines/monolithic", Path(root) / "baselines/fsm")
    datasets = [load(path.parent, path.name) for path in locations]
    labels = ("Hybrid PINN–FVM", "Monolithic PINN (2000 epochs)", "Independent FSM–FVM")
    density_upper = np.ceil(max(float(data["density"].max()) for _, data in datasets) / .005) * .005
    fig, axes = plt.subplots(3, 2, figsize=(9.7, 14.9))
    fig.subplots_adjust(left=.065, right=.96, bottom=.075, top=.95, wspace=.38, hspace=.7)
    levels = []
    for row, (label, (config, data)) in enumerate(zip(labels, datasets)):
        height = float(data["z"][int(np.abs(data["z"] - 15).argmin())])
        levels.append(height)
        for col, field in enumerate(("phi", "density")):
            ax = axes[row, col]
            upper = 8 if field == "phi" else density_upper
            image = contours(ax, config, data, field, height, upper)
            decorate(ax, config, height, compact=True)
            symbol = r"$\phi$" if field == "phi" else r"$\rho$"
            ax.set_title(symbol + f" at z={height:.2f}m", fontsize=12)
            colorbar(fig, image, ax, field, upper, compact=True)
            ax.text(.5, -.27, f"({chr(97 + 2 * row + col)}) {label}: " + symbol,
                    transform=ax.transAxes, ha="center", va="top", family="DejaVu Serif", fontsize=10)
    save(fig, destination, "fig5_methods")
    return {"dimensions": 3, "heights": levels, "density_shared_range": [0, float(density_upper)],
            "density_mapping": "linear", "source_directories": [str(path) for path in locations],
            "monolithic_epochs": 2000, "source_rate": datasets[0][0]["source"]["rate"],
            "note": "Previously solved 3D comparison; separate from new 2D figures"}


def training(root, destination):
    config, data = load(root, "p2p_vortex_obstacle")
    problem = Problem(config)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6))
    fig.subplots_adjust(left=.065, right=.98, bottom=.2, top=.9, wspace=.33)
    density_upper = max(.05, float(np.ceil(data["density"].max() / .05) * .05))
    density = np.linspace(0, density_upper, 350)
    speed = problem.speed(torch.from_numpy(density)).numpy()
    axes[0].plot(density, speed, "r-", label="SmoothMax–Greenshields")
    sample = data["density"][data["active"]].ravel()[::13]
    axes[0].scatter(sample, problem.speed(torch.from_numpy(sample)).numpy(),
                    s=7, alpha=.55, color="#71b9ff", label="Computed field samples")
    axes[0].axhline(config["physics"]["v_min"], color="#b852d7", linestyle="--", linewidth=1,
                   label=f"v_min = {config['physics']['v_min']:g}")
    axes[0].set(xlabel=r"Density $\rho$ (UAVs/m$^3$)", ylabel=r"Airspeed $v(\rho)$ (m/s)",
                title="Macroscopic Fundamental Diagram", xlim=(0, density_upper))
    axes[0].legend(fontsize=9)
    with (Path(root) / "p2p_vortex_obstacle/training.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Expected recorded training samples for convergence figure")
    epochs, losses, residuals = [], [], []
    offset, previous_outer, previous_epoch = 0, None, None
    for row in rows:
        outer, epoch = int(row["outer"]), int(row["epoch"])
        if previous_outer is not None and outer != previous_outer:
            offset += previous_epoch + 1
        epochs.append(offset + epoch)
        losses.append(max(float(row["loss"]) - config["training"].get("causality_weight", 0)
                          * float(row["causality_loss"]), 1e-15))
        residuals.append(float(row["mean_residual"]))
        previous_outer, previous_epoch = outer, epoch
    axes[1].semilogy(epochs, losses, "b-", label="Eikonal loss")
    axes[1].semilogy(epochs, residuals, "r--", label="Mean absolute Eikonal residual")
    axes[1].set(xlabel="PINN epoch (excluding graph pretraining)", ylabel="Loss / Residual",
                title="Training Loss & Residual")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.3, linewidth=.5)
    save(fig, destination, "fig6_diagnostics")


def density_load_comparison(low_root, high_root, destination, *, low_three_d_root=None, high_three_d_root=None):
    """Compare separately solved demands in exactly the same spatial cells."""
    comparisons = []
    pairs = [(low_root, high_root, name) for name, _ in CASES]
    if low_three_d_root is not None and high_three_d_root is not None:
        pairs += [(low_three_d_root, high_three_d_root, name)
                  for name in ("p2p_none", "p2p_obstacle", "p2p_height_obstacle")]
    for low_path, high_path, name in pairs:
        low_config, low_data = load(low_path, name)
        high_config, high_data = load(high_path, name)
        if (low_data["density"].shape != high_data["density"].shape
                or not np.array_equal(low_data["active"], high_data["active"])
                or any(not np.array_equal(low_data[a], high_data[a]) for a in ("x", "y", "z"))):
            raise ValueError("Demand comparison requires identical geometry and evaluation cells")
        active, source = low_data["active"], low_data["source"] > 0
        low_peak, high_peak = float(low_data["density"].max()), float(high_data["density"].max())
        low_mean, high_mean = float(low_data["density"][active].mean()), float(high_data["density"][active].mean())
        comparisons.append({"case": name, "dimension": high_config.get("spatial_dimension", 3),
                            "low_directory": str(Path(low_path) / name), "high_directory": str(Path(high_path) / name),
                            "low_source_rate": low_config["source"]["rate"], "high_source_rate": high_config["source"]["rate"],
                            "source_ratio": high_config["source"]["rate"] / low_config["source"]["rate"],
                            "low_density_peak": low_peak, "high_density_peak": high_peak,
                            "density_peak_ratio": high_peak / low_peak,
                            "low_mean_density": low_mean, "high_mean_density": high_mean,
                            "mean_density_ratio": high_mean / low_mean,
                            "low_source_density_peak": float(low_data["density"][source].max()),
                            "high_source_density_peak": float(high_data["density"][source].max())})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5))
    fig.subplots_adjust(left=.065, right=.96, top=.84, bottom=.12, wspace=.35)
    for ax, root, label in zip(axes, (low_root, high_root), ("Previous demand", "Increased demand")):
        config, data = load(root, "p2p_vortex_obstacle")
        height = float(data["z"][0])
        image = contours(ax, config, data, "density", height, .05, allow_saturation=True)
        decorate(ax, config, height, compact=True)
        ax.set_title(label + f"; q={config['source']['rate']:g}", fontsize=12)
        colorbar(fig, image, ax, "density", .05, compact=True)
    save(fig, Path(destination), "density_load_comparison")
    return {"same_cells_and_geometry": True, "comparison_color_scale": [0, .05],
            "display_above_0_05": "saturated yellow with an explicit upper colorbar extension; no density projection",
            "cases": comparisons}


def make_paper_figures(planar_root, three_d_root, destination, *, comparison_root=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figure_style()
    value_panels(planar_root, destination)
    details = motion_and_density(planar_root, destination)
    details["height_gallery"] = heights(three_d_root, destination)
    details["method_comparison"] = methods(comparison_root or three_d_root, destination)
    training(planar_root, destination)
    details["display_interpolation"] = "Contour rendering between raw cell centers; no numerical field changed"
    return details
