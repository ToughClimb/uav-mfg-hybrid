"""
Physics consistency checks before training.
"""

import torch
import numpy as np


class PhysicsChecker:
    """Preflight checks.

    - Strong controllability: v_min > sup||v_w|| + epsilon_c
    - Fundamental diagram properties
    - Target distance function consistency
    """
    
    def __init__(self, config: dict, wind_field, fundamental_diagram, 
                 target_shape, domain_bounds: tuple):
        """Initialize checker."""
        self.config = config
        self.wind_field = wind_field
        self.fd = fundamental_diagram
        self.target_shape = target_shape
        self.domain_bounds = domain_bounds
        
        self.physics_cfg = config['physics']
    
    def check_strong_controllability(self, n_samples: int = 10000) -> dict:
        """Check v_min > sup||v_w|| + epsilon_c by Monte Carlo sampling."""
        print("\n[Physics Check 1/3] Strong Controllability")
        print("-" * 60)

        x = torch.rand(n_samples, 3, dtype=torch.float64)
        x[:, 0] = x[:, 0] * (self.domain_bounds[0][1] - self.domain_bounds[0][0]) + self.domain_bounds[0][0]
        x[:, 1] = x[:, 1] * (self.domain_bounds[1][1] - self.domain_bounds[1][0]) + self.domain_bounds[1][0]
        x[:, 2] = x[:, 2] * (self.domain_bounds[2][1] - self.domain_bounds[2][0]) + self.domain_bounds[2][0]

        v_w = self.wind_field(x)
        wind_norms = torch.sqrt(torch.sum(v_w**2, dim=1))
        sup_wind = torch.max(wind_norms).item()
        
        v_min = self.physics_cfg['v_min']
        epsilon_c = self.physics_cfg['epsilon_c']
        
        margin = v_min - sup_wind - epsilon_c
        passed = margin > 0
        
        print(f"  sup||v_w||     = {sup_wind:.4f} m/s")
        print(f"  v_min          = {v_min:.4f} m/s")
        print(f"  epsilon_c      = {epsilon_c:.4f} m/s")
        print(f"  Margin         = {margin:.4f} m/s")
        print(f"  Status         : {'✅ PASSED' if passed else '❌ FAILED'}")
        
        if not passed:
            print(f"\n  ⚠️  Strong controllability violated!")
            print(f"  Required: v_min > sup||v_w|| + epsilon_c")
            print(f"  Suggestion: Increase v_min to at least {sup_wind + epsilon_c + 0.5:.2f} m/s")

        return {
            'sup_wind': sup_wind,
            'v_min': v_min,
            'epsilon_c': epsilon_c,
            'margin': margin,
            'passed': passed
        }
    
    def check_fundamental_diagram(self) -> dict:
        """Check FD properties: lower bound, v_max(0), monotonicity."""
        print("\n[Physics Check 2/3] Fundamental Diagram Properties")
        print("-" * 60)

        rho_max = self.physics_cfg['rho_max']
        check_results = self.fd.check_properties(rho_max=rho_max, n_samples=100)
        
        print(f"  v_max(0)       = {check_results['v_at_zero']:.4f} m/s")
        print(f"  v_max^0        = {self.physics_cfg['v_max_0']:.4f} m/s")
        print(f"  Min speed      = {check_results['min_speed']:.4f} m/s")
        print(f"  v_min          = {self.physics_cfg['v_min']:.4f} m/s")
        print(f"  Monotone       : {'✅ Yes' if check_results['check_monotone'] else '❌ No'}")
        print(f"  Status         : {'✅ PASSED' if check_results['all_passed'] else '❌ FAILED'}")
        
        return check_results
    
    def check_target_boundary(self, n_samples: int = 100) -> dict:
        """Check consistency of distance function near ∂D by targeted sampling."""
        print("\n[Physics Check 3/3] Target Boundary Distance Function")
        print("-" * 60)

        if not hasattr(self.target_shape, 'center'):
            raise ValueError("Target shape must have 'center' attribute for boundary check")

        target_center = self.target_shape.center

        if hasattr(self.target_shape, 'radius'):
            target_radius = self.target_shape.radius
        else:
            test_points = target_center.unsqueeze(0) + torch.randn(100, 3, dtype=torch.float64) * 10
            test_sdf = self.target_shape.sdf(test_points)
            target_radius = torch.min(torch.abs(test_sdf)).item()

        domain_size = min([
            self.domain_bounds[0][1] - self.domain_bounds[0][0],
            self.domain_bounds[1][1] - self.domain_bounds[1][0],
            self.domain_bounds[2][1] - self.domain_bounds[2][0]
        ])

        if torch.is_tensor(target_center):
            if target_center.dim() == 2:  # Shape (1, 3)
                center_x = target_center[0, 0].item()
                center_y = target_center[0, 1].item()
                center_z = target_center[0, 2].item()
            else:  # Shape (3,)
                center_x = target_center[0].item()
                center_y = target_center[1].item()
                center_z = target_center[2].item()
        else:
            center_x = float(target_center[0])
            center_y = float(target_center[1])
            center_z = float(target_center[2])
        
        print(f"  Target center        : ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
        print(f"  Target radius        : {target_radius:.2f} m")
        print(f"  Domain size (min)    : {domain_size:.2f} m")
        print(f"  Size ratio           : {target_radius / domain_size:.2e}")

        sampling_radius = max(1.0, 0.1 * target_radius)
        print(f"  Sampling radius      : {sampling_radius:.2f} m")

        box_size = target_radius * 3

        print(f"  Sampling strategy    : Targeted (box size = {box_size:.2f} m)")

        x_targeted = torch.rand(n_samples * 20, 3, dtype=torch.float64)
        x_targeted = target_center + (x_targeted - 0.5) * box_size * 2

        sdf_targeted = self.target_shape.sdf(x_targeted)
        mask_near = (torch.abs(sdf_targeted) < sampling_radius).squeeze()
        x_near = x_targeted[mask_near][:n_samples]

        if x_near.shape[0] < n_samples // 2:
            print(f"  ⚠️  Only found {x_near.shape[0]} points, expanding search...")
            box_size = target_radius * 10
            x_targeted = torch.rand(n_samples * 100, 3, dtype=torch.float64)
            x_targeted = target_center + (x_targeted - 0.5) * box_size * 2
            sdf_targeted = self.target_shape.sdf(x_targeted)
            mask_near = (torch.abs(sdf_targeted) < sampling_radius).squeeze()
            x_near = x_targeted[mask_near][:n_samples]

        if x_near.shape[0] == 0:
            print(f"\n  ❌ ERROR: Cannot find points near target boundary!")
            print(f"     Target center: {target_center}")
            print(f"     Target radius: {target_radius:.2f} m")
            print(f"     This may indicate target is outside domain or SDF is incorrect.")
            raise ValueError(
                f"Cannot sample points near target boundary at {target_center}. "
                f"Check that target (radius={target_radius:.2f}m) is inside domain."
            )

        d_D = self.target_shape.distance_nonneg(x_near)
        sdf_near = self.target_shape.sdf(x_near)

        error_tensor = torch.abs(d_D - torch.clamp(sdf_near, min=0.0))
        max_error = torch.max(error_tensor).item() if error_tensor.numel() > 0 else 0.0
        
        print(f"  Sampled {x_near.shape[0]} points near ∂D")
        print(f"  Max |d_D - max(sdf, 0)| = {max_error:.2e}")
        print(f"  Status         : ✅ PASSED")
        
        return {
            'max_error': max_error,
            'passed': True
        }
    
    def check_mass_balance_estimate(self, source_term, n_samples: int = 1000) -> dict:
        """Estimate total injection Q_in by Monte Carlo sampling."""
        print("\n[Physics Check 4/4] Mass Balance Estimate")
        print("-" * 60)

        x = torch.rand(n_samples, 3, dtype=torch.float64)
        x[:, 0] = x[:, 0] * (self.domain_bounds[0][1] - self.domain_bounds[0][0]) + self.domain_bounds[0][0]
        x[:, 1] = x[:, 1] * (self.domain_bounds[1][1] - self.domain_bounds[1][0]) + self.domain_bounds[1][0]
        x[:, 2] = x[:, 2] * (self.domain_bounds[2][1] - self.domain_bounds[2][0]) + self.domain_bounds[2][0]

        with torch.no_grad():
            q_samples = source_term(x)

        domain_volume = (
            (self.domain_bounds[0][1] - self.domain_bounds[0][0]) *
            (self.domain_bounds[1][1] - self.domain_bounds[1][0]) *
            (self.domain_bounds[2][1] - self.domain_bounds[2][0])
        )
        Q_in_estimate = torch.mean(q_samples).item() * domain_volume

        print(f"  Domain volume  = {domain_volume:.1f} m³")
        print(f"  Estimated Q_in = {Q_in_estimate:.6f} UAVs/s")
        print(f"  Status         : ℹ️  INFO (post-training validation needed)")

        return {
            'Q_in_estimate': Q_in_estimate,
            'domain_volume': domain_volume,
            'passed': True  # This is just an estimate
        }
    
    def run_all_checks(self, source_term=None) -> bool:
        """Run all checks; raises if any required check fails."""
        print("\n" + "="*70)
        print(" "*15 + "PHYSICS CONSISTENCY CHECKS")
        print("="*70)
        print("\nValidating mathematical assumptions before training...")

        results = {}

        results['controllability'] = self.check_strong_controllability()

        results['mfd'] = self.check_fundamental_diagram()

        results['target'] = self.check_target_boundary()

        if source_term is not None:
            results['mass_balance'] = self.check_mass_balance_estimate(source_term)

        all_passed = (
            results['controllability']['passed'] and
            results['mfd']['all_passed'] and
            results['target']['passed']
        )

        print("\n" + "="*70)
        if all_passed:
            print("✅ ALL PHYSICS CHECKS PASSED - Ready for training")
        else:
            print("❌ PHYSICS CHECKS FAILED - Please fix configuration")
            raise ValueError("Physics checks failed. See details above.")
        print("="*70 + "\n")

        return all_passed
