"""Conservative shared-face upwind operator and independently audited transport."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


@dataclass
class Grid:
    problem: object
    shape: tuple

    def __post_init__(self):
        self.shape = tuple(int(n) for n in self.shape)  # x,y,z throughout
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Grid requires three positive dimensions")
        if self.problem.dimension == 2 and self.shape[2] != 1:
            raise ValueError("A two-dimensional reduction requires exactly one z cell")
        self.spacing = np.diff(self.problem.bounds, axis=1).ravel() / self.shape
        self.axes = [a + (np.arange(n) + 0.5) * h for (a, _), n, h in zip(self.problem.bounds, self.shape, self.spacing)]
        self.points = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1)
        x = torch.from_numpy(self.points)
        self.obstacle = (self.problem.obstacle_distance(x) <= 0).numpy()
        self.target = (self.problem.target_distance(x) <= 0).numpy()
        if np.any(self.target & self.obstacle):
            raise ValueError("Target and obstacle occupy common grid cells")
        if not self.target.any():
            raise ValueError("No grid cell center resolves the absorbing target")
        self.active = ~(self.obstacle | self.target)
        self.cell_volume = float(np.prod(self.spacing))
        self.source = self.problem.source(x).numpy()
        if not np.any(self.source > 0):
            raise ValueError("No active grid cell resolves the source")

    def interpolate_torch(self, density, x):
        field = torch.as_tensor(density, dtype=x.dtype, device=x.device)
        lower = torch.as_tensor(self.problem.bounds[:, 0], dtype=x.dtype, device=x.device)
        spacing = torch.as_tensor(self.spacing, dtype=x.dtype, device=x.device)
        coord = (x - lower) / spacing - 0.5
        indices, weights = [], []
        for d, n in enumerate(self.shape):
            position = coord[:, d].clamp(0, n - 1)
            index = position.floor().long().clamp(max=max(n - 2, 0))
            indices.append(index)
            weights.append(position - index if n > 1 else torch.zeros_like(position))
        values = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    corner = (a, b, c)
                    weight = torch.ones_like(values)
                    location = []
                    for d, bit in enumerate(corner):
                        weight = weight * (weights[d] if bit else 1 - weights[d])
                        location.append((indices[d] + bit).clamp(max=self.shape[d] - 1))
                    values = values + weight * field[tuple(location)]
        return values


class Transport:
    """Each internal face has one velocity and one flux used by both cells."""

    def __init__(self, grid, velocity):
        self.grid = grid
        self.velocity = np.asarray(velocity, dtype=np.float64)
        if self.velocity.shape != grid.shape + (3,) or not np.isfinite(self.velocity).all():
            raise ValueError("Velocity must be finite and match grid[nx,ny,nz,3]")
        if grid.problem.dimension == 2 and np.any(np.abs(self.velocity[..., 2]) > 1e-12):
            raise ValueError("A two-dimensional transport field must have zero vertical velocity")
        full_ids = np.arange(np.prod(grid.shape)).reshape(grid.shape)
        self.ids = full_ids[grid.active]
        mapping = np.full(grid.shape, -1, dtype=int)
        mapping[grid.active] = np.arange(len(self.ids))
        rows, cols, data = [], [], []
        self.faces, self.obstacle_faces = [], []
        self.sink_rates = np.zeros(len(self.ids), dtype=float)
        for d, h in enumerate(grid.spacing):
            face_shape = list(grid.shape)
            face_shape[d] += 1
            face = np.zeros(face_shape, dtype=float)
            interior = [slice(None)] * 3
            interior[d] = slice(1, -1)
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[d], right[d] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            face_velocity = 0.5 * (self.velocity[left + (d,)] + self.velocity[right + (d,)])
            blocked = grid.obstacle[left] | grid.obstacle[right]
            face_velocity = np.where(blocked, 0.0, face_velocity)
            face[tuple(interior)] = face_velocity
            self.faces.append(face)
            self.obstacle_faces.append(blocked)
            donor = np.where(face_velocity >= 0, mapping[left], mapping[right]).ravel()
            receiver = np.where(face_velocity >= 0, mapping[right], mapping[left]).ravel()
            coefficient = (np.abs(face_velocity) / h).ravel()
            valid = (donor >= 0) & (coefficient > 0)
            donor, receiver, coefficient = donor[valid], receiver[valid], coefficient[valid]
            rows.append(donor); cols.append(donor); data.append(coefficient)
            internal = receiver >= 0
            rows.append(receiver[internal]); cols.append(donor[internal]); data.append(-coefficient[internal])
            np.add.at(self.sink_rates, donor[~internal], coefficient[~internal])
        # Preserve native arrays instead of allocating millions of Python scalars.
        self.operator = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(len(self.ids), len(self.ids))).tocsr()
        self.q = grid.source[grid.active]
        self.injection = float(self.q.sum() * grid.cell_volume)

    def fluxes(self, rho):
        fluxes = []
        for d, face in enumerate(self.faces):
            left, right, interior = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            left[d], right[d], interior[d] = slice(None, -1), slice(1, None), slice(1, -1)
            velocity = face[tuple(interior)]
            flux = np.zeros_like(face)
            flux[tuple(interior)] = velocity * np.where(velocity >= 0, rho[tuple(left)], rho[tuple(right)])
            fluxes.append(flux)
        return fluxes

    def audit(self, rho):
        r = np.asarray(rho)[self.grid.active]
        absorption = float(self.sink_rates @ r * self.grid.cell_volume)
        residual = self.operator @ r - self.q
        outer, obstacle = 0.0, 0.0
        for d, flux in enumerate(self.fluxes(rho)):
            area = self.grid.cell_volume / self.grid.spacing[d]
            outer += float((np.abs(np.take(flux, 0, axis=d)).sum() + np.abs(np.take(flux, -1, axis=d)).sum()) * area)
            interior = [slice(None)] * 3
            interior[d] = slice(1, -1)
            obstacle += float(np.abs(flux[tuple(interior)][self.obstacle_faces[d]]).sum() * area)
        return {"source_rate": self.injection, "target_absorption": absorption,
                "absorption_over_source": absorption / self.injection if self.injection else None,
                "relative_steady_balance": abs(self.injection - absorption) / max(self.injection, 1e-30),
                "outer_leakage": outer, "obstacle_leakage": obstacle,
                "pde_relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(self.q), 1e-30)),
                "total_mass": float(r.sum() * self.grid.cell_volume),
                "rho_max": float(r.max()), "rho_min": float(r.min())}

    def solve(self, initial, *, max_steps=10000, change_tolerance=1e-6,
              residual_tolerance=1e-4, balance_tolerance=1e-3, cfl=0.8,
              method="explicit", dt=None, backend="cpu", graph_steps=100):
        if backend == "cuda":
            if method != "explicit":
                raise ValueError("CUDA backend requires explicit pseudo-time transport")
            return self.solve_cuda(initial,max_steps=max_steps,change_tolerance=change_tolerance,
                                   residual_tolerance=residual_tolerance,balance_tolerance=balance_tolerance,
                                   cfl=cfl,dt=dt,graph_steps=graph_steps)
        if backend != "cpu":
            raise ValueError(f"Unknown transport backend: {backend}")
        if not 0 < cfl < 1 or max_steps < 1:
            raise ValueError("Require 0 < CFL < 1 and max_steps >= 1")
        r = np.asarray(initial, dtype=float)[self.grid.active].copy()
        if not np.isfinite(r).all() or np.any(r < 0):
            raise ValueError("Initial density must be finite and nonnegative")
        rate = float(self.operator.diagonal().max())
        dt_eff = min(dt or float("inf"), cfl / rate) if rate else (dt or 0.1)
        if method == "sparse":
            r = spsolve(self.operator, self.q)
            if not np.isfinite(r).all() or r.min() < -1e-10:
                raise RuntimeError("Sparse steady solve is singular or violates positivity; check reachable cells")
            steps, change, max_budget_error = 1, None, None
        elif method == "explicit":
            max_budget_error = 0.0
            converged = False
            for steps in range(1, max_steps + 1):
                absorption = float(self.sink_rates @ r * self.grid.cell_volume)
                previous = r
                r = previous + dt_eff * (self.q - self.operator @ previous)
                if not np.isfinite(r).all() or np.any(r < -1e-12):
                    raise FloatingPointError("Upwind density lost positivity or finiteness")
                mass_change = float((r - previous).sum() * self.grid.cell_volume)
                expected = dt_eff * (self.injection - absorption)
                max_budget_error = max(max_budget_error, abs(mass_change - expected))
                change = float(np.linalg.norm(r - previous) / max(np.linalg.norm(previous), 1e-30))
                if steps % 25 == 0 and change < change_tolerance:
                    stationary = np.linalg.norm(self.operator @ r - self.q) / max(np.linalg.norm(self.q), 1e-30)
                    balance = abs(self.injection - self.sink_rates @ r * self.grid.cell_volume) / max(self.injection, 1e-30)
                    if stationary < residual_tolerance and balance < balance_tolerance:
                        converged = True
                        break
        else:
            raise ValueError(f"Unknown transport method: {method}")
        result = np.zeros(self.grid.shape)
        result[self.grid.active] = r
        audit = self.audit(result)
        if method == "sparse":
            converged = audit["pde_relative_residual"] < residual_tolerance and audit["relative_steady_balance"] < balance_tolerance
        audit.update({"converged": bool(converged), "steps": steps, "dt": dt_eff,
                      "relative_time_update": change, "max_step_mass_budget_error": max_budget_error,
                      "method": method,"backend":"cpu"})
        return result, audit

    def solve_cuda(self,initial,*,max_steps,change_tolerance,residual_tolerance,balance_tolerance,cfl,dt,graph_steps):
        """Float64 CSR transport captured in CUDA graphs; audit every actual step."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA transport requested without an available CUDA device")
        if not 0<cfl<1 or min(max_steps,graph_steps)<1:
            raise ValueError("Require valid CFL and positive step budgets")
        import warnings
        a=self.operator
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Sparse CSR tensor support is in beta state.*")
            matrix=torch.sparse_csr_tensor(torch.tensor(a.indptr,device="cuda"),torch.tensor(a.indices,device="cuda"),
                                           torch.tensor(a.data,device="cuda"),size=a.shape,device="cuda",check_invariants=True)
        start=np.asarray(initial,dtype=float)[self.grid.active]
        if not np.isfinite(start).all() or np.any(start<0):
            raise ValueError("Initial density must be finite and nonnegative")
        initial_tensor=torch.tensor(start,device="cuda",dtype=torch.float64)
        r=initial_tensor.clone();q=torch.tensor(self.q,device="cuda",dtype=torch.float64)
        sink=torch.tensor(self.sink_rates,device="cuda",dtype=torch.float64)
        rate=float(a.diagonal().max());dt_eff=min(dt or float("inf"),cfl/rate) if rate else (dt or .1)
        volume=self.grid.cell_volume
        maximum_error=torch.zeros((),device="cuda",dtype=torch.float64)
        last_update=torch.zeros_like(r)

        def step():
            before=r.sum()*volume
            absorbed=torch.dot(sink,r)*volume
            last_update.copy_(q-torch.mv(matrix,r))
            r.add_(last_update,alpha=dt_eff)
            error=torch.abs(r.sum()*volume-before-dt_eff*(self.injection-absorbed))
            maximum_error.copy_(torch.maximum(maximum_error,error))

        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_steps):step()
        r.copy_(initial_tensor);maximum_error.zero_()
        steps=0;converged=False
        qnorm=torch.linalg.vector_norm(q).clamp_min(1e-30)
        while steps<max_steps:
            if max_steps-steps>=graph_steps:
                graph.replay();steps+=graph_steps
            else:
                step();steps+=1
            if not torch.isfinite(r).all().item() or r.min().item() < -1e-12:
                raise FloatingPointError("CUDA upwind transport lost positivity or finiteness")
            change=(dt_eff*torch.linalg.vector_norm(last_update)/torch.linalg.vector_norm(r-dt_eff*last_update).clamp_min(1e-30)).item()
            if change<change_tolerance:
                stationary=(torch.linalg.vector_norm(torch.mv(matrix,r)-q)/qnorm).item()
                balance=abs(self.injection-(torch.dot(sink,r)*volume).item())/max(self.injection,1e-30)
                if stationary<residual_tolerance and balance<balance_tolerance:
                    converged=True;break
        result=np.zeros(self.grid.shape);result[self.grid.active]=r.cpu().numpy()
        audit=self.audit(result)
        audit.update(converged=converged,steps=steps,dt=dt_eff,relative_time_update=change,
                     max_step_mass_budget_error=maximum_error.item(),method="explicit",backend="cuda_graph",
                     graph_steps=graph_steps,mass_audit_scope="every_step")
        return result,audit
