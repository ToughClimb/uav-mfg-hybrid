"""Visualization CLI for a checkpoint."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from pathlib import Path

from uavpinn.utils import load_config, build_from_config
from uavpinn.viz import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Visualize UAVPINN++ results')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: checkpoint_dir/plots)')
    parser.add_argument('--paraview', action='store_true',
                       help='Export ParaView VTK files (default: disabled)')
    parser.add_argument('--paraview-resolution', type=int, nargs=3, default=None,
                       help='ParaView grid resolution: nx ny nz (default: use rho grid resolution)')
    parser.add_argument('--paraview-batch-size', type=int, default=50000,
                       help='Batch size for ParaView export (default: 50000)')
    parser.add_argument('--paraview-prefix', type=str, default='final',
                       help='Filename prefix for ParaView export (default: final)')
    
    args = parser.parse_args()
    
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    print("Building geometry and physics objects...")
    objects = build_from_config(config)

    output_dir = args.output if args.output else None
    viz = Visualizer(
        checkpoint_path=args.checkpoint,
        config=config,
        objects=objects,
        output_dir=output_dir
    )
    
    viz.generate_all_plots()

    if args.paraview:
        grid_resolution = tuple(args.paraview_resolution) if args.paraview_resolution else None
        paraview_dir = viz.export_paraview(
            grid_resolution=grid_resolution,
            batch_size=args.paraview_batch_size,
            prefix=args.paraview_prefix
        )
        print(f"\n✅ ParaView exported to: {paraview_dir}")
    
    print(f"\n✅ Visualization completed!")


if __name__ == "__main__":
    main()
