"""Load YAML configs and construct geometry/wind/physics objects."""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from ..geometry import Sphere, AABB, Union
from ..wind import (ZeroWind, UniformWind, VortexWind, 
                    HeightDependentWind, CompositeWind, RegionConstantWind)
from ..physics import FundamentalDiagram, SourceTerm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_target_shape(config: dict):
    """Build target region shape from config."""
    target_cfg = config['target']
    
    if target_cfg['type'] == 'sphere':
        params = target_cfg['params']
        return Sphere(
            center=tuple(params['center']),
            radius=params['radius']
        )
    elif target_cfg['type'] == 'aabb':
        params = target_cfg['params']
        return AABB(
            min_corner=tuple(params['min']),
            max_corner=tuple(params['max'])
        )
    else:
        raise ValueError(f"Unknown target type: {target_cfg['type']}")


def build_obstacle_shape(config: dict):
    """Build obstacle shape from config."""
    obstacles_cfg = config.get('obstacles', [])
    
    if len(obstacles_cfg) == 0:
        return None
    elif len(obstacles_cfg) == 1:
        obs = obstacles_cfg[0]
        if obs['type'] == 'aabb':
            params = obs['params']
            return AABB(
                min_corner=tuple(params['min']),
                max_corner=tuple(params['max'])
            )
    else:
        # Union for multiple obstacles
        shapes = []
        for obs in obstacles_cfg:
            if obs['type'] == 'aabb':
                params = obs['params']
                shapes.append(AABB(
                    min_corner=tuple(params['min']),
                    max_corner=tuple(params['max'])
                ))
        return Union(shapes)


def build_wind_field(config: dict):
    """Build wind field from config."""
    wind_cfg = config['wind']
    base_cfg = wind_cfg['base']
    
    if base_cfg['type'] == 'none':
        base_wind = ZeroWind()
    elif base_cfg['type'] == 'uniform':
        params = base_cfg['params']
        base_wind = UniformWind(velocity=tuple(params['v']))
    elif base_cfg['type'] == 'vortex':
        params = base_cfg['params']
        base_wind = VortexWind(
            center=tuple(params['center']),
            strength=params['strength']
        )
    elif base_cfg['type'] == 'height_dependent':
        params = base_cfg['params']
        domain = config['domain']
        base_wind = HeightDependentWind(
            v0=tuple(params['v0']),
            alpha=params['alpha'],
            z_min=domain['z'][0],
            z_max=domain['z'][1]
        )
    else:
        raise ValueError(f"Unknown wind type: {base_cfg['type']}")
    
    patches = []
    if 'patches' in wind_cfg:
        for patch_cfg in wind_cfg['patches']:
            region_cfg = patch_cfg['region']
            if region_cfg['type'] == 'aabb':
                region_shape = AABB(
                    min_corner=tuple(region_cfg['params']['min']),
                    max_corner=tuple(region_cfg['params']['max'])
                )
            else:
                raise ValueError(f"Unknown region type: {region_cfg['type']}")
            
            transition_cfg = patch_cfg.get('transition', 'hard')
            if isinstance(transition_cfg, str):
                transition_type = transition_cfg
                sharpness = patch_cfg.get('sharpness', 50.0)
            else:
                transition_type = transition_cfg.get('type', 'hard')
                sharpness = transition_cfg.get('sharpness', 50.0)
            
            patch = RegionConstantWind(
                region_shape=region_shape,
                value=tuple(patch_cfg['value']),
                combine=patch_cfg.get('combine', 'override'),
                transition=transition_type,
                sharpness=sharpness
            )
            patches.append(patch)
    
    if len(patches) > 0:
        return CompositeWind(base=base_wind, patches=patches)
    else:
        return base_wind


def build_fundamental_diagram(config: dict) -> FundamentalDiagram:
    """Build fundamental diagram from config."""
    physics_cfg = config['physics']
    
    return FundamentalDiagram(
        v_max_0=physics_cfg['v_max_0'],
        v_min=physics_cfg['v_min'],
        rho_jam=physics_cfg['rho_jam'],
        beta=physics_cfg['beta'],
        clip_to_bounds=config.get('vmax', {}).get('clip_to_bounds', False)
    )


def build_source_term(config: dict, target_shape) -> SourceTerm:
    """Build source term from config."""
    scenario = config['scenario']['type']
    physics_cfg = config['physics']
    
    if scenario == 'homing':
        return SourceTerm(
            scenario='homing',
            target_shape=target_shape,
            q0=physics_cfg['source']['homing_q0']
        )
    elif scenario == 'p2p':
        source_cfg = physics_cfg['source']['p2p_source']
        return SourceTerm(
            scenario='p2p',
            target_shape=target_shape,
            q0=0.0,
            source_sphere_center=tuple(source_cfg['center']),
            source_sphere_radius=source_cfg['radius'],
            q_source=source_cfg['q_source']
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def build_from_config(config: dict) -> dict:
    """Build all objects from a config dict."""
    domain_bounds = (
        tuple(config['domain']['x']),
        tuple(config['domain']['y']),
        tuple(config['domain']['z'])
    )
    
    target_shape = build_target_shape(config)
    
    return {
        'config': config,
        'domain_bounds': domain_bounds,
        'target_shape': target_shape,
        'obstacle_shape': build_obstacle_shape(config),
        'wind_field': build_wind_field(config),
        'fundamental_diagram': build_fundamental_diagram(config),
        'source_term': build_source_term(config, target_shape)
    }
