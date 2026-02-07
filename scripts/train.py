"""Training CLI (Route A end-to-end, Route B fixed-point)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

from uavpinn.utils import load_config, build_from_config, PhysicsChecker


class Tee:
    """Duplicate writes to multiple streams (e.g., console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except TypeError:
                stream.write(data.decode('utf-8', errors='replace'))
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')


def main():
    parser = argparse.ArgumentParser(description='Train UAVPINN++ model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--route', type=str, default='B', choices=['A', 'B'],
                       help='Training route: A (end-to-end) or B (fixed-point)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                       help='Device to use (auto=prefer GPU if available)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable automatic visualization after training')
    parser.add_argument('--paraview', action='store_true',
                       help='Enable ParaView VTK export (default: disabled)')
    parser.add_argument('--paraview-resolution', type=int, nargs=3, default=[100, 100, 50],
                       metavar=('NX', 'NY', 'NZ'),
                       help='Grid resolution for ParaView export (default: 100 100 50)')
    parser.add_argument('--paraview-batch-size', type=int, default=50000,
                       help='Batch size for ParaView field evaluation (default: 50000, reduce if OOM)')
    parser.add_argument('--skip-physics-checks', action='store_true',
                       help='Skip physics consistency checks (use for uncontrollable cases)')
    
    args = parser.parse_args()
    start_time = datetime.now()
    run_meta = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'route': args.route,
        'device_arg': args.device,
        'seed': args.seed,
        'config_path': args.config,
        'argv': sys.argv,
        'status': 'running'
    }
    output_dir = None
    log_fp = None
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    exit_code = 0
    
    try:
        if args.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   Using GPU for training\n")
            else:
                device = 'cpu'
                print("\n⚠️  No GPU detected, using CPU\n")
        else:
            device = args.device
            if device == 'cuda' and not torch.cuda.is_available():
                print("\n❌ CUDA requested but not available, falling back to CPU\n")
                device = 'cpu'
        
        torch.manual_seed(args.seed)
        if device == 'cuda':
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
        exp_name = config['run']['exp_name']

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        route_suffix = '_end2end' if args.route == 'A' else ''
        output_dir = Path('runs') / f"{timestamp}_{exp_name}{route_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / 'training.log'
        log_fp = open(log_file, 'a', encoding='utf-8')
        sys.stdout = Tee(stdout_orig, log_fp)
        sys.stderr = Tee(stderr_orig, log_fp)
        
        print(f"Output directory: {output_dir}")
        print(f"Logging to: {log_file}")
        
        run_meta.update({
            'exp_name': exp_name,
            'output_dir': str(output_dir),
            'log_file': str(log_file)
        })
        
        import shutil
        config_copy_path = output_dir / 'config.yaml'
        shutil.copy(args.config, config_copy_path)
        print(f"Config saved to: {config_copy_path}")

        print("\nBuilding geometry, wind field, and physics objects...")
        objects = build_from_config(config)

        if args.skip_physics_checks:
            print("\n⚠️  Skipping physics consistency checks (user override)")
            run_meta['physics_checks'] = 'skipped'
        else:
            checker = PhysicsChecker(
                config=config,
                wind_field=objects['wind_field'],
                fundamental_diagram=objects['fundamental_diagram'],
                target_shape=objects['target_shape'],
                domain_bounds=objects['domain_bounds']
            )
            
            try:
                checker.run_all_checks()
                run_meta['physics_checks'] = 'passed'
            except ValueError as e:
                print(f"\n❌ Physics checks failed: {e}")
                print("Please fix configuration and try again.\n")
                run_meta['status'] = 'physics_check_failed'
                run_meta['error'] = str(e)
                exit_code = 1
                raise SystemExit(1)
        
        if args.route == 'B':
            print(f"\n{'='*70}")
            print(f" "*15 + "ROUTE B: FIXED-POINT ITERATION")
            print(f"{'='*70}")
            
            from uavpinn.training import FixedPointTrainer
            trainer = FixedPointTrainer(
                config=config,
                objects=objects,
                output_dir=output_dir,
                device=device
            )
            
            results = trainer.train()
            
        else:  # Route A
            print(f"\n{'='*70}")
            print(f" "*15 + "ROUTE A: END-TO-END PINN")
            print(f"{'='*70}")
            print("\nℹ️  Route A: End-to-end dual-network PINN (comparison method)")
            
            from uavpinn.training import End2EndTrainer
            trainer = End2EndTrainer(
                config=config,
                objects=objects,
                output_dir=output_dir,
                device=device
            )
            
            results = trainer.train()
        
        run_meta['status'] = 'success'
        
        print(f"\n{'='*70}")
        print(f"✅ Training completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        if args.paraview:
            print(f"\n{'='*70}")
            print(f" "*15 + "📦 EXPORTING TO PARAVIEW (VTK)")
            print(f"{'='*70}\n")
            
            checkpoint_path = output_dir / 'checkpoints' / 'checkpoint_final.pt'
            
            if checkpoint_path.exists():
                try:
                    from uavpinn.viz import ParaViewExporter
                    
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    paraview_dir = output_dir / 'paraview'
                    exporter = ParaViewExporter(
                        output_dir=str(paraview_dir),
                        domain_bounds=objects['domain_bounds'],
                        grid_resolution=tuple(args.paraview_resolution)
                    )
                    
                    from uavpinn.models import PhiModel
                    barrier_params = config.get('barrier', None)
                    p_value = config.get('barrier', {}).get('p') or config.get('phi_bc', {}).get('power', 1)
                    phi_model = PhiModel(
                        hidden_layers=config['network']['hidden_layers'],
                        target_shape=objects['target_shape'],
                        domain_bounds=objects['domain_bounds'],
                        p=p_value,
                        obstacle_shape=objects['obstacle_shape'],
                        barrier_params=barrier_params
                    ).to(device)
                    phi_state = checkpoint.get('phi_model_state', checkpoint.get('phi_state_dict'))
                    if phi_state is None:
                        raise KeyError("phi_model_state not found in checkpoint")
                    phi_model.load_state_dict(phi_state)
                    phi_model.eval()
                    
                    rho_field = checkpoint.get('rho_grid', checkpoint.get('rho_field'))
                    if rho_field is None:
                        print("⚠️  Warning: No rho field in checkpoint, using zeros")
                        nx, ny, nz = args.paraview_resolution
                        rho_field = torch.zeros(nx, ny, nz)
                    else:
                        grid_cfg = config.get('rho_solver', {}).get('grid', {})
                        if grid_cfg:
                            nz = grid_cfg.get('nz')
                            ny = grid_cfg.get('ny')
                            nx = grid_cfg.get('nx')
                            if nz is not None and ny is not None and nx is not None:
                                if tuple(rho_field.shape) == (nz, ny, nx):
                                    rho_field = rho_field.permute(2, 1, 0).contiguous()
                    
                    obstacle_shape = objects.get('obstacle_shape')
                    obstacle_shapes = [obstacle_shape] if obstacle_shape is not None else []
                    iteration = checkpoint.get('iteration', checkpoint.get('outer_iter', 0))
                    exporter.export_fields(
                        phi_model=phi_model,
                        rho_field=rho_field,
                        wind_field=objects['wind_field'],
                        fundamental_diagram=objects['fundamental_diagram'],
                        target_shape=objects['target_shape'],
                        obstacle_shapes=obstacle_shapes,
                        iteration=iteration,
                        prefix='final',
                        batch_size=args.paraview_batch_size
                    )
                    
                    print(f"\n✅ ParaView files exported to: {paraview_dir}")
                    print(f"   Open 'final_iter_*.vtk' in ParaView to visualize")
                    run_meta['paraview_export'] = 'success'
                    
                except Exception as e:
                    print(f"\n⚠️  ParaView export failed: {e}")
                    run_meta['paraview_export'] = f"failed: {e}"
            else:
                print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
                run_meta['paraview_export'] = 'checkpoint_not_found'
        else:
            run_meta['paraview_export'] = 'disabled'
        
        if not args.no_viz:
            print(f"\n{'='*70}")
            print(f" "*18 + "📊 GENERATING VISUALIZATIONS")
            print(f"{'='*70}\n")
            
            checkpoint_path = output_dir / 'checkpoints' / 'checkpoint_final.pt'
            run_meta['checkpoint_path'] = str(checkpoint_path)
            
            if checkpoint_path.exists():
                import subprocess
                viz_cmd = [
                    sys.executable,
                    'scripts/generate_all_plots.py',
                    '--checkpoint', str(checkpoint_path)
                ]
                
                try:
                    subprocess.run(viz_cmd, check=True, timeout=600)
                    print(f"\n✅ All plots generated in: {output_dir / 'plots'}")
                    run_meta['visualization'] = 'success'
                except Exception as e:
                    print(f"\n⚠️  Visualization failed: {e}")
                    run_meta['visualization'] = f"failed: {e}"
            else:
                print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
                run_meta['visualization'] = 'checkpoint_not_found'
        else:
            run_meta['visualization'] = 'disabled'
    except SystemExit as e:
        if run_meta.get('status') == 'running':
            run_meta['status'] = 'exit'
        if isinstance(e.code, int):
            exit_code = e.code
        raise
    except Exception as e:
        run_meta['status'] = 'error'
        run_meta['error'] = str(e)
        exit_code = 1
        raise
    finally:
        end_time = datetime.now()
        run_meta['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
        run_meta['duration_seconds'] = (end_time - start_time).total_seconds()
        run_meta['exit_code'] = exit_code
        if output_dir is not None:
            run_meta_path = output_dir / 'run_meta.json'
            with open(run_meta_path, 'w', encoding='utf-8') as f:
                json.dump(run_meta, f, indent=2)
        
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        if log_fp is not None:
            log_fp.close()


if __name__ == "__main__":
    main()
