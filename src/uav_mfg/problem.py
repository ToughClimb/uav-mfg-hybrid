"""Geometry, coefficients, and source terms defined by the revised manuscript."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


def sphere_sdf(x, center, radius):
    center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
    return torch.linalg.vector_norm(x - center, dim=-1) - radius


def disk_sdf(x, center, radius):
    """XY disk in the explicitly declared two-dimensional reduction."""
    center = torch.as_tensor(center[:2], dtype=x.dtype, device=x.device)
    return torch.linalg.vector_norm(x[..., :2] - center, dim=-1) - radius


def box_sdf(x, lower, upper):
    lower = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    upper = torch.as_tensor(upper, dtype=x.dtype, device=x.device)
    offset = (x - (lower + upper) / 2).abs() - (upper - lower) / 2
    return torch.linalg.vector_norm(offset.clamp_min(0), dim=-1) + offset.amax(dim=-1).clamp_max(0)


@dataclass
class Problem:
    config: dict

    def __post_init__(self):
        self.dimension = self.config.get("spatial_dimension", 3)
        if self.dimension not in (2, 3):
            raise ValueError("spatial_dimension must be 2 or 3")
        self.bounds = np.asarray(self.config["domain"], dtype=float)
        if self.bounds.shape != (3, 2) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("domain must contain three increasing [lower, upper] pairs")
        self.target = self.config["target"]
        regions = (self.target, self.config["source"])
        if any(region.get("type") == "disk" for region in regions) and self.dimension != 2:
            raise ValueError("Disk geometry requires an explicit two-dimensional problem")
        if self.dimension == 2 and any(region.get("type") == "sphere" for region in regions):
            raise ValueError("Use disk geometry for a two-dimensional problem")
        self.obstacles = self.config.get("obstacles", [])
        self.physics = self.config["physics"]
        if not 0 < self.physics["v_min"] <= self.physics["v_free"]:
            raise ValueError("Require 0 < v_min <= v_free")
        if min(self.physics["beta"], self.physics["rho_jam"], self.physics["rho_max"]) <= 0:
            raise ValueError("beta and density scales must be positive")
        self.wind_config = self.config.get("wind", {"type": "none"})
        if self.wind_config["type"] == "gridded":
            with np.load(self.wind_config["path"], allow_pickle=False) as data:
                self.wind_axes = [data[axis].copy() for axis in ("x", "y", "z")]
                self.wind_values = data["velocity"].copy()
            expected = tuple(len(a) for a in self.wind_axes) + (3,)
            if self.wind_values.shape != expected or any(np.any(np.diff(a) <= 0) for a in self.wind_axes):
                raise ValueError("Gridded wind requires increasing x/y/z and velocity[nx,ny,nz,3]")

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            return cls(yaml.safe_load(stream))

    def target_distance(self, x):
        if self.target.get("type", "sphere") == "sphere":
            return sphere_sdf(x, self.target["center"], self.target["radius"])
        if self.target["type"] == "box":
            return box_sdf(x, self.target["lower"], self.target["upper"])
        if self.target["type"] == "disk":
            return disk_sdf(x, self.target["center"], self.target["radius"])
        raise ValueError(f"Unsupported target: {self.target['type']}")

    def obstacle_distance(self, x):
        if not self.obstacles:
            return torch.full(x.shape[:-1], float("inf"), dtype=x.dtype, device=x.device)
        return torch.stack([box_sdf(x, box["lower"], box["upper"]) for box in self.obstacles]).amin(dim=0)

    def free_mask(self, x, buffer=0.0):
        return (self.target_distance(x) > 0) & (self.obstacle_distance(x) > buffer)

    def speed(self, rho, law=None):
        p = self.physics
        greenshields = p["v_free"] * (1 - rho / p["rho_jam"])
        minimum = torch.full_like(rho, p["v_min"])
        law = law or p.get("mobility_law", "smoothmax")
        if law == "smoothmax":
            speed = torch.logaddexp(p["beta"] * minimum, p["beta"] * greenshields) / p["beta"]
        elif law == "linear":
            speed = greenshields.clamp_min(p["v_min"])
        elif law == "underwood":
            speed = (p["v_free"] * torch.exp(-rho / p["rho_jam"])).clamp_min(p["v_min"])
        elif law == "drake":
            speed = (p["v_free"] * torch.exp(-0.5 * (rho / p["rho_jam"]) ** 2)).clamp_min(p["v_min"])
        elif law == "triangular":
            critical = p.get("rho_critical", p["rho_jam"] / 2)
            congested = p["v_free"] * critical * (p["rho_jam"] - rho) / (rho.clamp_min(1e-12) * (p["rho_jam"] - critical))
            speed = torch.where(rho <= critical, torch.full_like(rho, p["v_free"]), congested).clamp_min(p["v_min"])
        else:
            raise ValueError(f"Unknown mobility law: {law}")
        return speed.clamp(p["v_min"], p["v_free"]) if p.get("clip_speed", False) else speed

    def wind(self, x):
        w = self.wind_config
        kind = w["type"]
        if kind == "none":
            return torch.zeros_like(x)
        if kind == "uniform":
            return torch.as_tensor(w["velocity"], dtype=x.dtype, device=x.device).expand_as(x)
        if kind == "vortex":
            center = torch.as_tensor(w["center"], dtype=x.dtype, device=x.device)
            offset = x - center
            radius = w["radius"]
            envelope = torch.exp(-0.5 * (offset / radius).square().sum(dim=-1, keepdim=True))
            rotation = torch.stack((-offset[..., 1], offset[..., 0], torch.zeros_like(offset[..., 0])), dim=-1)
            return (w["strength"] / radius) * rotation * envelope
        if kind == "height_dependent":
            h = (x[..., 2:3] - self.bounds[2, 0]) / (self.bounds[2, 1] - self.bounds[2, 0])
            base = torch.as_tensor(w["velocity"], dtype=x.dtype, device=x.device)
            return torch.exp(-w["decay"] * h) * base
        if kind == "gridded":
            axes = [torch.as_tensor(a, dtype=x.dtype, device=x.device) for a in self.wind_axes]
            field = torch.as_tensor(self.wind_values, dtype=x.dtype, device=x.device)
            indices, weights = [], []
            for d, axis in enumerate(axes):
                if ((x[..., d] < axis[0]) | (x[..., d] > axis[-1])).any():
                    raise ValueError("Wind query outside provided gridded data")
                index = torch.searchsorted(axis, x[..., d].contiguous(), right=True).clamp(1, len(axis) - 1) - 1
                indices.append(index)
                weights.append((x[..., d] - axis[index]) / (axis[index + 1] - axis[index]))
            result = torch.zeros_like(x)
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        corners = (a, b, c)
                        weight = torch.ones_like(x[..., 0])
                        for d, corner in enumerate(corners):
                            weight = weight * (weights[d] if corner else 1 - weights[d])
                        result = result + weight[..., None] * field[indices[0] + a, indices[1] + b, indices[2] + c]
            return result
        raise ValueError(f"Unknown wind type: {kind}")

    def source(self, x):
        s = self.config["source"]
        if s["type"] == "homing":
            return torch.full(x.shape[:-1], s["rate"], dtype=x.dtype, device=x.device) * self.free_mask(x)
        if s["type"] == "sphere":
            inside = sphere_sdf(x, s["center"], s["radius"]) <= 0
            return (inside & self.free_mask(x)).to(x.dtype) * s["rate"]
        if s["type"] == "disk":
            inside = disk_sdf(x, s["center"], s["radius"]) <= 0
            return (inside & self.free_mask(x)).to(x.dtype) * s["rate"]
        raise ValueError(f"Unknown source type: {s['type']}")
