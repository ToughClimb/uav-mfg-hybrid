from .style import apply_pub_style

apply_pub_style()

from .visualizer import Visualizer
from .mfd_plots import plot_mfd_validation
from .streamlines import plot_streamlines_2d
from .diagnostics import (
    plot_diagonal_profile,
    plot_diagonal_plane_slice,
    plot_residual_heatmap
)
from .viz_3d import (
    plot_3d_streamlines, 
    plot_3d_isosurface,
    plot_3d_vectors,
    plot_3d_density_volume
)
from .training_curves import (
    plot_training_curves,
    plot_fixed_point_convergence,
    plot_rho_solver_convergence
)
from .paraview_export import ParaViewExporter

__all__ = [
    'Visualizer',
    'plot_mfd_validation',
    'plot_streamlines_2d',
    'plot_diagonal_profile',
    'plot_diagonal_plane_slice',
    'plot_residual_heatmap',
    'plot_3d_streamlines',
    'plot_3d_isosurface',
    'plot_3d_vectors',
    'plot_3d_density_volume',
    'plot_training_curves',
    'plot_fixed_point_convergence',
    'plot_rho_solver_convergence',
    'ParaViewExporter',
]
