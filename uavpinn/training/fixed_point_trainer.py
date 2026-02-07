"""
Fixed-point iteration (Picard + relaxation).

Steps:
1) Fix ρ^k, train φ^k (PINN)
2) Construct u^k from φ^k
3) Solve ρ̃^{k+1} from conservation
4) ρ^{k+1} = (1-α)ρ^k + α·clip(ρ̃^{k+1}, 0, ρ_max)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import csv

from ..models import PhiModel
from ..losses import EikonalLoss
from ..sampling import Sampler
from ..solvers import RhoSolver


class FixedPointTrainer:
    """Trains φ; solves ρ with a conservation solver."""
    
    def __init__(self, config: dict, objects: dict, output_dir: Path,
                 device: str = 'cpu'):
        """Initialize trainer."""
        self.config = config
        self.objects = objects
        self.output_dir = output_dir
        self.device = device

        self.physics_cfg = config['physics']
        self.network_cfg = config['network']
        self.training_cfg = config['training']
        self.fixed_point_cfg = config['fixed_point']
        self.sampling_cfg = config['sampling']
        self.numerical_cfg = config['numerical']

        barrier_params = config.get('barrier', None)
        p_value = config.get('barrier', {}).get('p') or config.get('phi_bc', {}).get('power', 1)
        self.phi_model = PhiModel(
            hidden_layers=self.network_cfg['hidden_layers'],
            target_shape=objects['target_shape'],
            domain_bounds=objects['domain_bounds'],
            p=p_value,
            obstacle_shape=objects['obstacle_shape'],
            barrier_params=barrier_params
        ).to(device)

        self.sampler = Sampler(
            domain_bounds=objects['domain_bounds'],
            target_shape=objects['target_shape'],
            obstacle_shape=objects['obstacle_shape'],
            n_pde=self.sampling_cfg['n_pde'],
            n_boundary=self.sampling_cfg['n_boundary'],
            n_depot=self.sampling_cfg.get('n_depot', 200),
            n_obstacle=self.sampling_cfg.get('n_obstacle', 0),
            device=device,
            dtype=torch.float64
        )

        rho_solver_cfg = config['rho_solver']
        self.rho_solver = RhoSolver(
            domain_bounds=objects['domain_bounds'],
            nx=rho_solver_cfg['grid']['nx'],
            ny=rho_solver_cfg['grid']['ny'],
            nz=rho_solver_cfg['grid']['nz'],
            target_shape=objects['target_shape'],
            obstacle_shape=objects['obstacle_shape'],
            dt=rho_solver_cfg['pseudo_time']['dt'],
            max_iters=rho_solver_cfg['max_iters'],
            tol=rho_solver_cfg['tol'],
            device=device,
            dtype=torch.float64
        )
        
        scenario_type = config.get('scenario', {}).get('type', 'homing').lower()
        if scenario_type == 'p2p':
            self.rho_grid = torch.zeros(
                (self.rho_solver.nz, self.rho_solver.ny, self.rho_solver.nx),
                dtype=torch.float64, device=device
            )
        else:
            self.rho_grid = torch.full(
                (self.rho_solver.nz, self.rho_solver.ny, self.rho_solver.nx),
                1e-6, dtype=torch.float64, device=device
            )

        self.eikonal_loss = EikonalLoss(
            epsilon_reg=self.numerical_cfg['epsilon_reg'],
            nu=self.numerical_cfg.get('nu', 0.0)
        ).to(device)

        self.optimizer = optim.Adam(
            self.phi_model.parameters(),
            lr=self.training_cfg['learning_rate']
        )

        scheduler_type = self.training_cfg.get('lr_scheduler', 'step')

        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_cfg['step_size'],
                gamma=self.training_cfg['gamma']
            )
            self.scheduler_type = 'step'

        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.training_cfg.get('gamma', 0.5),
                patience=self.training_cfg.get('patience', 500),
                verbose=True,
                min_lr=self.training_cfg.get('min_lr', 1e-6)
            )
            self.scheduler_type = 'plateau'

        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_cfg.get('T_max', 5000),
                eta_min=self.training_cfg.get('min_lr', 1e-6)
            )
            self.scheduler_type = 'cosine'

        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.training_cfg.get('gamma', 0.9999)
            )
            self.scheduler_type = 'exponential'

        else:
            raise ValueError(f"Unknown lr_scheduler: {scheduler_type}")

        self.metrics_file = output_dir / 'metrics.csv'
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'outer_iter', 'phi_epoch', 'loss_eik', 'mean_residual',
                'lr', 'rho_solver_iters', 'rho_residual', 'rho_change'
            ])
    
    def train_phi_subproblem(self, outer_iter: int, rho_field_func) -> dict:
        """Train φ for one outer iteration with fixed ρ."""
        phi_epochs = self.training_cfg['phi_epochs']

        pbar = tqdm(range(phi_epochs), desc=f"Outer iter {outer_iter+1} - Training φ")

        for epoch in pbar:
            samples = self.sampler.sample_all()
            x_pde = samples['pde_points'].to(self.device).requires_grad_(True)

            with torch.no_grad():
                rho_pde = rho_field_func(x_pde)

            v_max = self.objects['fundamental_diagram'].v_max(rho_pde)

            v_w = self.objects['wind_field'](x_pde)

            phi = self.phi_model.phi_total(x_pde)

            if self.objects['obstacle_shape'] is not None:
                delta = self.config.get('barrier', {}).get('delta', 2.0)
                sdf_obs = self.objects['obstacle_shape'].sdf(x_pde)
                mask = (sdf_obs > delta).float()
            else:
                mask = None

            loss_dict = self.eikonal_loss(phi, x_pde, v_max, v_w, mask=mask)
            loss = loss_dict['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler_type == 'plateau':
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4e}",
                'res': f"{loss_dict['mean_residual'].item():.4e}",
                'lr': f"{current_lr:.2e}"
            })

            if epoch % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        outer_iter, epoch, loss.item(),
                        loss_dict['mean_residual'].item(),
                        current_lr, '', '', ''
                    ])

        return {
            'final_loss': loss.item(),
            'final_residual': loss_dict['mean_residual'].item()
        }
    
    def construct_velocity_field(self) -> torch.Tensor:
        """u = v_w - v_max(ρ) * ∇φ / ||∇φ||_ε on the ρ grid."""
        n_total = self.rho_solver.nz * self.rho_solver.ny * self.rho_solver.nx
        batch_size = 10000
        u_list = []

        for i in range(0, n_total, batch_size):
            batch_end = min(i + batch_size, n_total)
            grid_coords_batch = self.rho_solver.grid_coords_flat[i:batch_end].to(self.device)

            with torch.no_grad():
                rho_batch = self.rho_solver.interpolate_to_points(
                    self.rho_grid, grid_coords_batch
                )

                v_max_batch = self.objects['fundamental_diagram'].v_max(rho_batch)

                v_w_batch = self.objects['wind_field'](grid_coords_batch)

            grid_coords_grad = grid_coords_batch.clone().detach().requires_grad_(True)
            phi_batch = self.phi_model.phi_total(grid_coords_grad)

            grad_phi = torch.autograd.grad(
                outputs=phi_batch,
                inputs=grid_coords_grad,
                grad_outputs=torch.ones_like(phi_batch),
                create_graph=False
            )[0]

            with torch.no_grad():
                eps_reg = self.numerical_cfg['epsilon_reg']
                grad_norm_reg = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + eps_reg**2)
                u_batch = v_w_batch - v_max_batch * (grad_phi / grad_norm_reg)
                u_list.append(u_batch.cpu())

        u_flat = torch.cat(u_list, dim=0)
        u_grid = u_flat.reshape(self.rho_solver.nz, self.rho_solver.ny, 
                                self.rho_solver.nx, 3)

        return u_grid
    
    def train(self) -> dict:
        """Run fixed-point iteration with relaxation."""
        outer_iters = self.fixed_point_cfg['outer_iters']
        alpha = self.fixed_point_cfg['relaxation_alpha']

        print(f"\nStarting Fixed-Point Iteration Training (Route B)")
        print(f"Outer iterations: {outer_iters}, Relaxation α: {alpha}")
        print("="*70)

        for k in range(outer_iters):
            print(f"\n{'='*70}")
            print(f"OUTER ITERATION {k+1}/{outer_iters}")
            print(f"{'='*70}")

            print(f"\n[Step 1/{4}] Training φ network with fixed ρ^{k}...")

            def rho_func(points):
                return self.rho_solver.interpolate_to_points(self.rho_grid, points)

            phi_metrics = self.train_phi_subproblem(k, rho_func)

            print(f"\n[Step 2/4] Constructing velocity field u^{k}...")
            u_grid = self.construct_velocity_field()

            print(f"\n[Step 3/4] Solving ρ conservation equation...")

            with torch.no_grad():
                q_flat = self.objects['source_term'](self.rho_solver.grid_coords_flat)
                q_grid = q_flat.reshape(self.rho_solver.nz, self.rho_solver.ny, 
                                       self.rho_solver.nx)

            u_grid = u_grid.to(self.rho_grid.device)
            q_grid = q_grid.to(self.rho_grid.device)

            rho_result = self.rho_solver.solve(u_grid, q_grid, rho_init=self.rho_grid)
            rho_tilde = rho_result['rho']

            print(f"  ρ solver: {rho_result['iterations']} iters, "
                  f"converged={rho_result['converged']}")

            print(f"\n[Step 4/4] Relaxation update...")
            rho_new = (1 - alpha) * self.rho_grid + alpha * torch.clamp(
                rho_tilde, min=0.0, max=self.physics_cfg['rho_max']
            )

            rho_change = torch.norm(rho_new - self.rho_grid) / (torch.norm(self.rho_grid) + 1e-10)
            print(f"  ||ρ^{k+1} - ρ^{k}|| / ||ρ^{k}|| = {rho_change.item():.4e}")

            self.rho_grid = rho_new

            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    k, 'outer_summary', phi_metrics['final_loss'],
                    phi_metrics['final_residual'], '',
                    rho_result['iterations'],
                    rho_result['residual_history'][-1] if rho_result['residual_history'] else 0,
                    rho_change.item()
                ])

            if (k + 1) % 5 == 0:
                self.save_checkpoint(k + 1)

        self.save_checkpoint('final')

        return {
            'phi_model': self.phi_model,
            'rho_grid': self.rho_grid
        }
    
    def save_checkpoint(self, iteration):
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'phi_model_state': self.phi_model.state_dict(),
            'rho_grid': self.rho_grid,
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_dir / f'checkpoint_{iteration}.pt')
        print(f"  Saved checkpoint: checkpoint_{iteration}.pt")
