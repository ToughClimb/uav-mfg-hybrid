"""Independent isotropic fast-sweeping reference for the paper's benign baseline."""

import itertools

import numpy as np


def local_eikonal_update(neighbors, spacing, speed):
    """Solve sum(max(T-a_d,0)^2/h_d^2)=1/speed^2 causally."""
    order = np.argsort(neighbors)
    neighbors, spacing = np.asarray(neighbors)[order], np.asarray(spacing)[order]
    finite = np.isfinite(neighbors)
    neighbors, spacing = neighbors[finite], spacing[finite]
    if not len(neighbors):
        return float("inf")
    for count in range(1, len(neighbors) + 1):
        a, weight = neighbors[:count], 1 / spacing[:count] ** 2
        aa = weight.sum()
        bb = -2 * (a * weight).sum()
        cc = (a ** 2 * weight).sum() - 1 / speed ** 2
        candidate = (-bb + np.sqrt(max(0, bb ** 2 - 4 * aa * cc))) / (2 * aa)
        if count == len(neighbors) or candidate <= neighbors[count]:
            return float(candidate)
    raise AssertionError("Unreachable causal update branch")


def fast_sweeping(grid, speed, max_cycles=40, tolerance=1e-8, *, backend="python", threads=8):
    """Eight directional Gauss-Seidel sweeps, target cells fixed at zero."""
    if grid.problem.wind_config["type"] != "none":
        raise ValueError("This reference implements the zero-wind comparison only")
    if backend == "openmp":
        return parallel_fast_sweeping(grid, speed, max_cycles, tolerance, threads)
    if backend != "python":
        raise ValueError(f"Unknown fast-sweeping backend: {backend}")
    phi = np.full(grid.shape, float("inf"))
    phi[grid.target] = 0
    finite_previous = np.zeros(grid.shape, dtype=bool)
    for cycle in range(max_cycles):
        previous = phi.copy()
        for signs in itertools.product((1,-1),repeat=3):
            ranges = [range(n) if sign==1 else range(n-1,-1,-1) for n,sign in zip(grid.shape,signs)]
            for index in itertools.product(*ranges):
                if not grid.active[index]:
                    continue
                neighbors=[]
                for d,n in enumerate(grid.shape):
                    values=[]
                    for shift in (-1,1):
                        adjacent=list(index);adjacent[d]+=shift
                        values.append(phi[tuple(adjacent)] if 0<=adjacent[d]<n else float("inf"))
                    neighbors.append(min(values))
                phi[index]=min(phi[index],local_eikonal_update(neighbors,grid.spacing,speed[index]))
        finite=np.isfinite(phi) & grid.active
        if finite[grid.active].all() and finite_previous[grid.active].all():
            change=float(np.max(np.abs(phi[grid.active]-previous[grid.active])))
            if change<tolerance:
                phi[grid.obstacle]=0
                return phi,{"converged":True,"cycles":cycle+1,"maximum_update":change}
        finite_previous=finite
    phi[grid.obstacle]=0
    if not np.isfinite(phi).all():
        raise RuntimeError("Fast sweeping has unreachable cells")
    return phi,{"converged":False,"cycles":max_cycles}


def parallel_fast_sweeping(grid, speed, max_cycles, tolerance, threads):
    import ctypes
    from pathlib import Path

    library_path = Path(__file__).resolve().parents[2] / "build/fsm_kernel.so"
    if not library_path.exists():
        raise RuntimeError("Build the optional OpenMP kernel: .venv/bin/python scripts/build_native.py")
    speed = np.ascontiguousarray(speed, dtype=np.float64)
    if speed.shape != grid.shape or np.any(speed <= 0) or not np.isfinite(speed).all():
        raise ValueError("FSM requires finite positive speed on the grid")
    if min(threads, max_cycles) < 1:
        raise ValueError("Require positive thread and cycle budgets")
    library = ctypes.CDLL(str(library_path))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    mask = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    library.fsm_solve.argtypes = [array, array, mask, mask, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 array, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    library.fsm_solve.restype = ctypes.c_int
    phi = np.empty(grid.shape)
    change = ctypes.c_double()
    cycles = library.fsm_solve(phi, speed, np.ascontiguousarray(grid.active, dtype=np.uint8),
                               np.ascontiguousarray(grid.target, dtype=np.uint8), *grid.shape,
                               np.ascontiguousarray(grid.spacing), max_cycles, tolerance, threads, ctypes.byref(change))
    if cycles < 0:
        raise RuntimeError("Parallel fast sweeping found unreachable active cells")
    phi[grid.obstacle] = 0
    return phi, {"converged": cycles > 0, "cycles": cycles if cycles > 0 else max_cycles,
                 "maximum_update": change.value, "backend": "openmp_wavefront", "threads": threads}


def upwind_gradient(grid, phi):
    values=phi.copy();values[grid.obstacle]=float("inf")
    grad=np.zeros(grid.shape+(3,))
    for d,h in enumerate(grid.spacing):
        left=np.roll(values,1,axis=d);right=np.roll(values,-1,axis=d)
        boundary=[slice(None)]*3;boundary[d]=0;left[tuple(boundary)]=float("inf")
        boundary[d]=-1;right[tuple(boundary)]=float("inf")
        chosen=np.minimum(left,right)
        downhill=(chosen<values) & grid.active
        component=np.zeros(grid.shape)
        component[downhill]=np.where(left[downhill]<=right[downhill],1,-1)*(values[downhill]-chosen[downhill])/h
        grad[...,d]=component
    return grad
