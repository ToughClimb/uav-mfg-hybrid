"""
End-to-end trainer (Route A).
"""

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from ..models import PhiModel, RhoModel
from ..losses import EikonalLoss, ContinuityLoss, FluxLoss


class End2EndTrainer:
    """Trains φ and ρ networks jointly."""
    
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

        self.rho_model = RhoModel(
            hidden_layers=self.network_cfg['hidden_layers'],
            rho_max=self.physics_cfg['rho_max'],
            domain_bounds=objects['domain_bounds']
        ).to(device)

        from ..sampling import Sampler
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
        
        self.eikonal_loss = EikonalLoss(
            epsilon_reg=self.numerical_cfg['epsilon_reg'],
            nu=self.numerical_cfg.get('nu', 0.0)
        ).to(device)

        self.continuity_loss = ContinuityLoss(
            epsilon_reg=self.numerical_cfg['epsilon_reg'],
            kappa=self.numerical_cfg.get('kappa', 0.0)
        ).to(device)

        self.flux_loss = FluxLoss(
            epsilon_reg=self.numerical_cfg['epsilon_reg'],
            kappa=self.numerical_cfg.get('kappa', 0.0)
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.phi_model.parameters()) + list(self.rho_model.parameters()),
            lr=self.training_cfg['learning_rate']
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.training_cfg['step_size'],
            gamma=self.training_cfg['gamma']
        )

        self.lambda_eik = self.training_cfg.get('lambda_eik', 1.0)
        self.lambda_cont = self.training_cfg.get('lambda_cont', 1.0)
        self.lambda_flux = self.training_cfg.get('lambda_flux', 0.1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / 'metrics.csv'
        
        import csv
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'total_loss', 'loss_eik', 'loss_cont', 'loss_flux', 'lr'])
    
    def train(self) -> dict:
        """End-to-end optimization loop."""
        epochs = self.training_cfg.get('end2end_epochs', 20000)

        print(f"\nStarting End-to-End Training (Route A)")
        print(f"Total epochs: {epochs}")
        print("="*70)

        import csv

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            samples = self.sampler.sample_all()
            x_pde = samples['pde_points'].to(self.device).requires_grad_(True)
            x_boundary = samples['boundary_points'].to(self.device).requires_grad_(True)
            n_boundary = samples['boundary_normals'].to(self.device)

            phi = self.phi_model.phi_total(x_pde)
            rho = self.rho_model(x_pde)

            v_max = self.objects['fundamental_diagram'].v_max(rho)
            v_w = self.objects['wind_field'](x_pde)
            q = self.objects['source_term'](x_pde)

            if self.objects['obstacle_shape'] is not None:
                delta = self.config.get('barrier', {}).get('delta', 2.0)
                sdf_obs = self.objects['obstacle_shape'].sdf(x_pde)
                mask = (sdf_obs > delta).float()
            else:
                mask = None

            eik_dict = self.eikonal_loss(phi, x_pde, v_max, v_w, mask=mask)
            loss_eik = eik_dict['loss']

            cont_dict = self.continuity_loss(rho, phi, x_pde, v_max, v_w, q, mask=mask)
            loss_cont = cont_dict['loss']

            phi_boundary = self.phi_model.phi_total(x_boundary)
            rho_boundary = self.rho_model(x_boundary)
            v_max_boundary = self.objects['fundamental_diagram'].v_max(rho_boundary)
            v_w_boundary = self.objects['wind_field'](x_boundary)

            flux_dict = self.flux_loss(rho_boundary, phi_boundary, x_boundary,
                                      v_max_boundary, v_w_boundary, n_boundary)
            loss_flux = flux_dict['loss']

            total_loss = (self.lambda_eik * loss_eik + 
                          self.lambda_cont * loss_cont + 
                          self.lambda_flux * loss_flux)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'total': f"{total_loss.item():.4e}",
                'eik': f"{loss_eik.item():.4e}",
                'cont': f"{loss_cont.item():.4e}",
                'lr': f"{current_lr:.2e}"
            })

            if epoch % 100 == 0:
                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, total_loss.item(), loss_eik.item(),
                        loss_cont.item(), loss_flux.item(), current_lr
                    ])

        from ..solvers import RhoSolver
        rho_solver_cfg = self.config['rho_solver']
        rho_solver = RhoSolver(
            domain_bounds=self.objects['domain_bounds'],
            nx=rho_solver_cfg['grid']['nx'],
            ny=rho_solver_cfg['grid']['ny'],
            nz=rho_solver_cfg['grid']['nz'],
            target_shape=self.objects['target_shape'],
            obstacle_shape=self.objects['obstacle_shape'],
            dt=0.01,
            device=self.device
        )
        
        with torch.no_grad():
            grid_coords_flat = rho_solver.grid_coords_flat.to(self.device)
            rho_flat = self.rho_model(grid_coords_flat)
            rho_grid = rho_flat.reshape(rho_solver.nz, rho_solver.ny, rho_solver.nx)

        checkpoint = {
            'config': self.config,
            'phi_model_state': self.phi_model.state_dict(),
            'rho_model_state': self.rho_model.state_dict(),
            'rho_grid': rho_grid.cpu(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epochs
        }

        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'
        torch.save(checkpoint, checkpoint_path)

        print(f"\n✅ Training completed")
        print(f"Checkpoint saved: {checkpoint_path}")

        return {
            'final_loss': total_loss.item(),
            'checkpoint_path': str(checkpoint_path)
        }
