"""Hard-target PINN, obstacle barrier, Eikonal residual, and geometry-aware sampling."""

import math

import numpy as np
import torch
from torch import nn


class Potential(nn.Module):
    def __init__(self, problem, widths=(64, 128, 128, 64)):
        super().__init__()
        self.problem = problem
        layers, previous = [], problem.dimension
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        # Analytic distance/v_free initialization, independently derived for zero wind.
        nn.init.normal_(layers[-1].weight, std=1e-4)
        nn.init.constant_(layers[-1].bias, math.log(math.expm1(1 / problem.physics["v_free"])))
        self.register_buffer("lower", torch.tensor(problem.bounds[:, 0], dtype=torch.float32))
        self.register_buffer("extent", torch.tensor(np.diff(problem.bounds, axis=1).ravel(), dtype=torch.float32))

    def forward(self, x):
        distance = self.problem.target_distance(x).clamp_min(0)
        coordinates = (2 * (x - self.lower) / self.extent - 1)[..., :self.problem.dimension]
        raw = self.network(coordinates).squeeze(-1)
        positive = nn.functional.softplus(raw)
        barrier = self.problem.config["barrier"]
        if self.problem.obstacles:
            positive = positive + barrier["height"] * nn.functional.softplus(-barrier["steepness"] * self.problem.obstacle_distance(x))
        return distance ** barrier.get("power", 1) * positive


def sample_points(problem, count, *, device, buffer=0.0, boundary_count=0):
    lower = torch.tensor(problem.bounds[:, 0], dtype=torch.float32, device=device)
    extent = torch.tensor(np.diff(problem.bounds, axis=1).ravel(), dtype=torch.float32, device=device)
    batches, remaining = [], count
    for _ in range(100):
        if remaining <= 0:
            break
        candidates = lower + torch.rand(max(remaining * 2, 64), 3, device=device) * extent
        # Target-shell samples remain part of the PDE objective; no soft target loss.
        nshell = len(candidates) // 5
        if problem.target.get("type", "sphere") == "sphere":
            direction = torch.randn(nshell, 3, device=device)
            direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)
            center = torch.tensor(problem.target["center"], dtype=torch.float32, device=device)
            radius = problem.target["radius"] * (1 + 0.02 + 0.6 * torch.rand(nshell, 1, device=device))
            candidates[:nshell] = center + radius * direction
        elif problem.target["type"] == "disk":
            angle = 2 * torch.pi * torch.rand(nshell, device=device)
            direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
            center = torch.tensor(problem.target["center"][:2], dtype=torch.float32, device=device)
            radius = problem.target["radius"] * (1.02 + 0.6 * torch.rand(nshell, 1, device=device))
            candidates[:nshell, :2] = center + radius * direction
        # Oversample just outside AABB obstacle shells, honoring the residual buffer.
        if problem.obstacles:
            box = problem.obstacles[torch.randint(len(problem.obstacles), ()).item()]
            lo = torch.tensor(box["lower"], dtype=torch.float32, device=device)
            hi = torch.tensor(box["upper"], dtype=torch.float32, device=device)
            nobs = len(candidates) // 5
            points = lo + torch.rand(nobs, 3, device=device) * (hi - lo)
            faces = torch.randint(6, (nobs,), device=device)
            for d in range(3):
                offset = buffer + 0.1 + 3 * torch.rand(nobs, device=device)
                points[:, d] = torch.where(faces == 2 * d, lo[d] - offset, points[:, d])
                points[:, d] = torch.where(faces == 2 * d + 1, hi[d] + offset, points[:, d])
            candidates[nshell:nshell + nobs] = points
        valid = problem.free_mask(candidates, buffer=buffer) & ((candidates >= lower) & (candidates <= lower + extent)).all(dim=-1)
        accepted = candidates[valid][:remaining]
        batches.append(accepted)
        remaining -= len(accepted)
    if remaining:
        raise RuntimeError("Geometry-aware rejection sampling exhausted its budget")
    samples = torch.cat(batches)
    if boundary_count:
        # Near-wall interior collocation samples supplement the PDE residual.
        walls = lower + torch.rand(boundary_count * 2, 3, device=device) * extent
        faces = torch.randint(6, (len(walls),), device=device)
        for d in range(3):
            walls[:, d] = torch.where(faces == 2 * d, lower[d] + 1e-4 * extent[d], walls[:, d])
            walls[:, d] = torch.where(faces == 2 * d + 1, lower[d] + (1 - 1e-4) * extent[d], walls[:, d])
        walls = walls[problem.free_mask(walls, buffer=buffer)][:boundary_count]
        samples = torch.cat((samples, walls))
    return samples


def gradient(phi, x, *, create_graph=True):
    return torch.autograd.grad(phi.sum(), x, create_graph=create_graph)[0]


def eikonal(model, x, density, epsilon):
    x = x.detach().requires_grad_(True)
    potential = model(x)
    grad = gradient(potential, x)
    norm = (grad.square().sum(dim=-1) + epsilon ** 2).sqrt()
    residual = model.problem.speed(density) * norm - (model.problem.wind(x) * grad).sum(dim=-1) - 1
    return residual


def causal_penalty(model, grid, density, settings):
    """Optional minimum-arrival-time branch selection, beyond the paper residual.

    A short feasible motion must lead to a lower remaining travel time. This
    one-sided Bellman inequality suppresses the spurious positive local minima
    that the unsigned Eikonal residual alone can admit. It is explicitly logged
    as an implementation enhancement, not attributed to the manuscript.
    """
    device = next(model.parameters()).device
    x = sample_points(model.problem, settings.get("causality_points", 256), device=device)
    if hasattr(model,"causality_anchors") and len(model.causality_anchors):
        anchors=model.causality_anchors
        jitter=0.5*torch.randn_like(anchors)
        extra=anchors+jitter
        lower,upper=model.lower,model.lower+model.extent
        valid=model.problem.free_mask(extra) & ((extra>=lower)&(extra<=upper)).all(dim=-1)
        x=torch.cat((x,anchors,extra[valid]),dim=0)
    directions = torch.tensor([[a,b,c] for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)
                               if (a,b,c) != (0,0,0) and (model.problem.dimension == 3 or c == 0)],dtype=x.dtype,device=device)
    directions = directions / torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
    step = torch.minimum(torch.full((len(x),), settings.get("causality_step",2.5),device=device),
                         0.5*model.problem.target_distance(x)).clamp_min(1e-4)
    if model.problem.target.get("type","sphere") in ("sphere", "disk"):
        coordinates = list(model.problem.target["center"])
        if len(coordinates) == 2:
            coordinates.append(float(model.problem.bounds[2].mean()))
        center = torch.tensor(coordinates,device=device,dtype=x.dtype)
    else:
        center = torch.tensor((np.asarray(model.problem.target["lower"])+np.asarray(model.problem.target["upper"]))/2,device=device,dtype=x.dtype)
    to_target = center - x
    if model.problem.dimension == 2:
        to_target[:, 2] = 0
    to_target = to_target / torch.linalg.vector_norm(to_target,dim=-1,keepdim=True).clamp_min(1e-12)
    directions = torch.cat((directions[None,:,:].expand(len(x),-1,-1),to_target[:,None,:]),dim=1)
    # Include the actual induced motion direction. A fixed angular stencil can
    # otherwise impose a positive Bellman error even on a correct smooth field.
    x=x.detach().requires_grad_(True)
    current_phi=model(x)
    current_grad=torch.autograd.grad(current_phi.sum(),x,retain_graph=True)[0].detach()
    with torch.no_grad():
        speed=model.problem.speed(grid.interpolate_torch(density,x))
        wind=model.problem.wind(x)
        ground=wind-speed[:,None]*current_grad/(current_grad.square().sum(dim=-1,keepdim=True)+settings["epsilon"]**2).sqrt()
        ground_direction=ground/torch.linalg.vector_norm(ground,dim=-1,keepdim=True).clamp_min(1e-12)
    directions=torch.cat((directions,ground_direction[:,None,:]),dim=1)
    next_points = x[:,None,:] + step[:,None,None] * directions
    midpoint = (x[:,None,:] + next_points) / 2
    lower, upper = model.lower, model.lower + model.extent
    valid = ((next_points >= lower) & (next_points <= upper)).all(dim=-1)
    valid = valid & (model.problem.obstacle_distance(next_points) > 0) & (model.problem.obstacle_distance(midpoint) > 0)
    with torch.no_grad():
        speed = model.problem.speed(grid.interpolate_torch(density,x))
        wind = model.problem.wind(x)
        drift_projection = torch.einsum("nd,nkd->nk",wind,directions)
        ground_speed = drift_projection + (speed[:,None].square() - wind.square().sum(dim=-1,keepdim=True) + drift_projection.square()).clamp_min(1e-12).sqrt()
        cost = step[:,None] / ground_speed
    proposal = model(next_points.reshape(-1,3)).reshape(next_points.shape[:2]) + cost
    proposal = proposal.masked_fill(~valid,float("inf")).amin(dim=-1)
    if not torch.isfinite(proposal).all():
        raise RuntimeError("Causality sampling found no feasible neighboring state")
    return torch.relu((proposal-current_phi) / (step / model.problem.physics["v_free"])).square().mean()


def train_value(model, optimizer, grid, density, settings, epochs, outer, history):
    device = next(model.parameters()).device
    buffer = model.problem.config["barrier"]["buffer"]
    for epoch in range(epochs):
        if settings.get("causality_weight",0) and epoch % 100 == 0:
            with torch.no_grad():
                locations=torch.tensor(grid.points.reshape(-1,3),dtype=torch.float32,device=device)
                field=model(locations).cpu().numpy().reshape(grid.shape)
            values=field.copy();values[grid.obstacle]=float("inf")
            minimum=np.full(grid.shape,float("inf"))
            for d in range(3):
                for shift in (-1,1):
                    neighbor=np.roll(values,shift,axis=d)
                    edge=[slice(None)]*3;edge[d]=0 if shift==1 else -1
                    neighbor[tuple(edge)]=float("inf")
                    minimum=np.minimum(minimum,neighbor)
            local_min=grid.active & (values<=minimum)
            model.causality_anchors=torch.tensor(grid.points[local_min][:128],dtype=torch.float32,device=device)
        x = sample_points(model.problem, settings["pde_points"], device=device, buffer=buffer,
                          boundary_count=settings.get("boundary_points", 0))
        with torch.no_grad():
            rho = grid.interpolate_torch(density, x)
        residual = eikonal(model, x, rho, settings["epsilon"])
        loss = residual.square().mean()
        causal = causal_penalty(model,grid,density,settings) if settings.get("causality_weight",0) else torch.zeros((),device=device)
        loss = loss + settings.get("causality_weight",0) * causal
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite PINN loss at outer={outer}, epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append({"outer": outer, "epoch": epoch, "loss": float(loss.detach()),
                            "mean_residual": float(residual.detach().abs().mean()), "causality_loss": float(causal.detach())})


def evaluate(model, grid, rho, epsilon, batch_size=16384):
    device = next(model.parameters()).device
    phi, velocity, residual = [], [], []
    for points, values in zip(np.array_split(grid.points.reshape(-1, 3), max(1, math.ceil(np.prod(grid.shape) / batch_size))),
                              np.array_split(rho.ravel(), max(1, math.ceil(np.prod(grid.shape) / batch_size)))):
        x = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
        potential = model(x)
        grad = gradient(potential, x, create_graph=False)
        with torch.no_grad():
            speed = model.problem.speed(torch.tensor(values, dtype=x.dtype, device=device))
            wind = model.problem.wind(x)
            norm = (grad.square().sum(dim=-1) + epsilon ** 2).sqrt()
            flow = wind - speed[:, None] * grad / norm[:, None]
            res = speed * norm - (wind * grad).sum(dim=-1) - 1
            phi.append(potential.detach().cpu().numpy())
            velocity.append(flow.cpu().numpy())
            residual.append(res.cpu().numpy())
    return (np.concatenate(phi).reshape(grid.shape), np.concatenate(velocity).reshape(grid.shape + (3,)),
            np.concatenate(residual).reshape(grid.shape))
