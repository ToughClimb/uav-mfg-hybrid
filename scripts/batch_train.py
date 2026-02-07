"""One-click reproduction for Route B (fixed-point hybrid)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import subprocess
from pathlib import Path
from datetime import datetime
import json
import csv
from typing import Optional



def _safe_get(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _collect_batch_metrics(experiments):
    rows = []
    for exp in experiments:
        row = {
            'config': exp.get('config'),
            'status': exp.get('status'),
            'run_dir': exp.get('run_dir'),
            'duration_seconds': exp.get('duration_seconds'),
            'exit_code': exp.get('exit_code'),
            'visualization': exp.get('visualization')
        }
        rows.append(row)
    return rows


def _write_batch_metrics_csv(rows, output_path: Path):
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(output_path)


def _resolve_configs(configs, config_dir: Optional[str], glob_pattern: Optional[str]):
    paths = []
    if configs:
        for item in configs:
            paths.append(Path(item))
    if config_dir:
        base = Path(config_dir)
        if base.is_dir():
            if glob_pattern:
                paths.extend(sorted(base.glob(glob_pattern)))
            else:
                paths.extend(sorted(base.glob('*.yaml')))
    if glob_pattern and not config_dir:
        paths.extend(sorted(Path('.').glob(glob_pattern)))
    uniq = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def reproduce_route_b(
    configs,
    config_dir: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    device: str = 'auto',
    auto_visualize: bool = True,
    skip_physics_checks: bool = False
):
    config_paths = _resolve_configs(configs, config_dir, glob_pattern)
    
    results = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'route': 'B',
        'device': device,
        'experiments': []
    }
    
    print("\n" + "="*70)
    print(" "*15 + "BATCH TRAINING ALL CONFIGURATIONS")
    print("="*70)
    print(f"\nRoute: B")
    print(f"Device: {device}")
    print(f"Total configs: {len(config_paths)}\n")
    
    for idx, config_path in enumerate(config_paths, 1):
        
        if not config_path.exists():
            print(f"\n[{idx}/{len(config_paths)}] ⚠️  Config not found: {config_path}")
            results['experiments'].append({
                'config': str(config_path),
                'status': 'skipped',
                'reason': 'file not found'
            })
            continue
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(config_paths)}] Training: {config_path}")
        print(f"{'='*70}")
        
        exp_start = datetime.now()
        
        try:
            cmd = [
                sys.executable,
                'scripts/train.py',
                '--config', str(config_path),
                '--route', 'B',
                '--device', device,
                '--no-viz',
            ]
            if skip_physics_checks:
                cmd.append('--skip-physics-checks')
            
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
            
            exp_end = datetime.now()
            duration = (exp_end - exp_start).total_seconds()
            
            if result.returncode == 0:
                print(f"\n✅ Success: {config_path} (Duration: {duration:.1f}s)")
                
                exp_result = {
                    'config': str(config_path),
                    'status': 'success',
                    'duration_seconds': duration
                }

                exp_name = config_path.stem
                runs_dir = Path('runs')
                run_glob = f"*_{exp_name}"
                matching_dirs = sorted(runs_dir.glob(run_glob))
                latest_run = matching_dirs[-1] if matching_dirs else None
                checkpoint_path = latest_run / 'checkpoints' / 'checkpoint_final.pt' if latest_run else None
                if latest_run is not None:
                    exp_result['run_dir'] = str(latest_run)
                
                if auto_visualize:
                    print(f"\n📊 Generating visualizations...")
                    try:
                        if latest_run:
                            if checkpoint_path and checkpoint_path.exists():
                                viz_cmd = [
                                    sys.executable,
                                    'scripts/generate_all_plots.py',
                                    '--checkpoint', str(checkpoint_path)
                                ]
                                
                                viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=600)
                                
                                if viz_result.returncode == 0:
                                    print(f"   ✅ Visualizations saved to: {latest_run / 'plots'}")
                                    exp_result['visualization'] = 'success'
                                else:
                                    print(f"   ⚠️  Visualization failed (non-critical)")
                                    exp_result['visualization'] = 'failed'
                                    exp_result['visualization_error'] = (
                                        viz_result.stderr.strip() or viz_result.stdout.strip()
                                    )
                            else:
                                print(f"   ⚠️  Checkpoint not found")
                                exp_result['visualization'] = 'checkpoint_not_found'
                        else:
                            print(f"   ⚠️  Run directory not found")
                            exp_result['visualization'] = 'run_dir_not_found'
                    
                    except Exception as e:
                        print(f"   ⚠️  Visualization error: {e}")
                        exp_result['visualization'] = f'error: {str(e)}'
                
                results['experiments'].append(exp_result)
            else:
                print(f"\n❌ Failed: {config_path} (Exit code: {result.returncode})")
                results['experiments'].append({
                    'config': str(config_path),
                    'status': 'failed',
                    'exit_code': result.returncode
                })
        
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  Timeout: {config_path} (>2 hours)")
            results['experiments'].append({
                'config': str(config_path),
                'status': 'timeout'
            })
        
        except Exception as e:
            print(f"\n❌ Error: {config_path}")
            print(f"   {str(e)}")
            results['experiments'].append({
                'config': str(config_path),
                'status': 'error',
                'error': str(e)
            })
    
    results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    summary_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = Path('runs') / f"batch_summary_{summary_stamp}.json"
    batch_metrics = _collect_batch_metrics(results['experiments'])
    metrics_csv_path = summary_file.parent / f"batch_metrics_{summary_stamp}.csv"
    metrics_csv = _write_batch_metrics_csv(batch_metrics, metrics_csv_path)
    if metrics_csv:
        print(f"\n📄 Batch metrics CSV saved: {metrics_csv}")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(" "*20 + "BATCH TRAINING SUMMARY")
    print("="*70)
    
    success_count = sum(1 for exp in results['experiments'] if exp['status'] == 'success')
    failed_count = sum(1 for exp in results['experiments'] if exp['status'] in ['failed', 'error', 'timeout'])
    
    print(f"\nTotal experiments: {len(config_paths)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    print(f"\nDetailed results:")
    for exp in results['experiments']:
        status_icon = {
            'success': '✅',
            'failed': '❌',
            'error': '❌',
            'timeout': '⏱️',
            'skipped': '⚠️'
        }.get(exp['status'], '?')
        
        duration_str = f" ({exp.get('duration_seconds', 0):.0f}s)" if 'duration_seconds' in exp else ""
        print(f"  {status_icon} {exp['config']}{duration_str}")
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*70 + "\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='One-click reproduction (Route B)')
    parser.add_argument('--config', action='append', default=[], help='Config YAML path (repeatable)')
    parser.add_argument('--config-dir', type=str, default=None, help='Directory containing YAML configs')
    parser.add_argument('--glob', type=str, default=None, help='Glob pattern for configs (with --config-dir or relative to cwd)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--no-viz', action='store_true', help='Disable 2D slices + loss plots')
    parser.add_argument('--skip-physics-checks', action='store_true', help='Skip physics consistency checks')
    
    args = parser.parse_args()

    auto_viz = not args.no_viz
    results = reproduce_route_b(
        configs=args.config,
        config_dir=args.config_dir,
        glob_pattern=args.glob,
        device=args.device,
        auto_visualize=auto_viz,
        skip_physics_checks=args.skip_physics_checks
    )
    
    failed = sum(1 for exp in results['experiments'] if exp['status'] != 'success')
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
