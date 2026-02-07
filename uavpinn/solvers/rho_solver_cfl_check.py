"""
CFL condition check utility for RhoSolver.
Add this to improve numerical stability guarantee.
"""

import torch
import warnings


def check_cfl_condition(u_field: torch.Tensor, dt: float, 
                        dx: float, dy: float, dz: float,
                        warn_threshold: float = 1.0) -> dict:
    """
    Check CFL (Courant-Friedrichs-Lewy) condition for stability.
    
    CFL = max|u| * dt / h_min
    
    For explicit schemes, CFL ≤ 1 is typically required for stability.
    
    Args:
        u_field: (nz, ny, nx, 3) velocity field
        dt: time step
        dx, dy, dz: grid spacings
        warn_threshold: CFL threshold for warning (default 1.0)
        
    Returns:
        dict with CFL info and stability status
    """
    # Compute maximum velocity magnitude
    u_mag = torch.sqrt(torch.sum(u_field**2, dim=-1))
    u_max = torch.max(u_mag).item()
    
    # Minimum grid spacing
    h_min = min(dx, dy, dz)
    
    # CFL number
    cfl = u_max * dt / h_min
    
    # Stability check
    is_stable = cfl <= warn_threshold
    
    if not is_stable:
        # Suggest safe dt
        dt_safe = warn_threshold * h_min / (u_max + 1e-10)
        
        warnings.warn(
            f"\n"
            f"  ⚠️  CFL Condition Warning\n"
            f"  ─────────────────────────\n"
            f"  CFL number    = {cfl:.3f}\n"
            f"  Threshold     = {warn_threshold:.3f}\n"
            f"  Status        : {'✅ STABLE' if is_stable else '❌ POTENTIALLY UNSTABLE'}\n"
            f"\n"
            f"  Current dt    = {dt:.6f}\n"
            f"  Max |u|       = {u_max:.4f} m/s\n"
            f"  Min spacing   = {h_min:.4f} m\n"
            f"\n"
            f"  Suggested dt  ≤ {dt_safe:.6f}\n"
            f"  ─────────────────────────\n",
            UserWarning
        )
    
    return {
        'cfl': cfl,
        'u_max': u_max,
        'h_min': h_min,
        'dt': dt,
        'dt_safe': warn_threshold * h_min / (u_max + 1e-10),
        'is_stable': is_stable
    }
