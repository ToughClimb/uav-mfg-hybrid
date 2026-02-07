"""Generate visualization plots from a checkpoint."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from pathlib import Path

from uavpinn.viz import Visualizer, plot_training_curves


def generate_all_plots(
    checkpoint_path: str,
    export_paraview: bool = False,
    paraview_resolution=None,
    paraview_batch_size: int = 50000,
    paraview_prefix: str = 'final',
    paper_only: bool = False
):
    """Generate plots for a checkpoint."""
    print("\n" + "="*70)
    print(" "*15 + "GENERATING ALL VISUALIZATION PLOTS")
    print("="*70)
    print(f"\nCheckpoint: {checkpoint_path}\n")
    
    z_slices_full = [5, 7, 10, 15, 25, 35]

    from uavpinn.utils import build_from_config

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    objects = build_from_config(config)
    output_dir = Path(checkpoint_path).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    print("\n[1/2] Generating 2D slices...")
    viz = Visualizer(checkpoint_path=checkpoint_path, config=config, objects=objects)
    viz.plot_phi_slices(z_slices=z_slices_full)
    viz.plot_rho_slices(z_slices=z_slices_full, use_interpolation=True)

    print("\n[2/2] Generating training curves...")
    run_dir = Path(checkpoint_path).parent.parent
    try:
        plot_training_curves(run_dir, output_dir)
    except Exception as e:
        print(f"  Warning: Training curves failed: {e}")
    
    print("\n" + "="*70)
    print(f"✅ ALL PLOTS GENERATED (Each as separate file)")
    print(f"   Location: {output_dir}")
    print(f"   2D Field Slices (each height separate):")
    print(f"     - phi_z5/7/10/15/25/35.pdf/png (6 files)")
    print(f"     - rho_z5/7/10/15/25/35_interp.pdf/png (6 files)")
    print(f"   Training Curves:")
    print(f"     - training_loss.pdf/png")
    print(f"     - training_residual.pdf/png")
    print(f"     - training_loss_residual.pdf/png")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate all visualization plots')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--paraview', action='store_true',
                       help='Export ParaView VTK files (default: disabled)')
    parser.add_argument('--paraview-resolution', type=int, nargs=3, default=None,
                       help='ParaView grid resolution: nx ny nz (default: use rho grid resolution)')
    parser.add_argument('--paraview-batch-size', type=int, default=50000,
                       help='Batch size for ParaView export (default: 50000)')
    parser.add_argument('--paraview-prefix', type=str, default='final',
                       help='Filename prefix for ParaView export (default: final)')
    parser.add_argument('--paper-only', action='store_true',
                       help='Generate only paper-ready plots (skip diagnostics)')
    
    args = parser.parse_args()
    
    import torch
    generate_all_plots(
        args.checkpoint,
        export_paraview=args.paraview,
        paraview_resolution=args.paraview_resolution,
        paraview_batch_size=args.paraview_batch_size,
        paraview_prefix=args.paraview_prefix,
        paper_only=args.paper_only
    )


if __name__ == "__main__":
    main()
