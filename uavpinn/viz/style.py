"""Matplotlib style helpers for publication figures."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager as fm

DEFAULT_FONT_FAMILY = "DejaVu Sans"


def apply_pub_style(font_family: str = DEFAULT_FONT_FAMILY) -> None:
    """Apply consistent publication style settings for matplotlib."""
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    resolved_family = font_family if font_family in available_fonts else DEFAULT_FONT_FAMILY
    sans_serif = list(
        dict.fromkeys(
            [resolved_family, "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
        )
    )
    mpl.rcParams.update(
        {
            "font.family": resolved_family,
            "font.sans-serif": sans_serif,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "image.cmap": "viridis",
        }
    )
