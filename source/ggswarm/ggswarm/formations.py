"""Formation geometry presets for drone swarms.

Each function returns [N, 3] offsets from the group centroid.
Train in polygon mode (policy learns "go to assigned slot"), then
swap formation preset at play time for any shape without retraining.
"""

from __future__ import annotations

import math

import torch


def polygon(n: int, radius: float = 0.5) -> torch.Tensor:
    """Regular N-gon with given circumradius.

    Args:
        n: number of agents
        radius: circumradius (m) — distance from center to each vertex

    Returns:
        [N, 3] XYZ offsets from centroid
    """
    offsets = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        offsets.append([radius * math.cos(angle), radius * math.sin(angle), 0.0])
    return torch.tensor(offsets)


def grid(n: int, spacing: float = 0.5) -> torch.Tensor:
    """Rectangular grid, rows x cols closest to sqrt(N).

    Args:
        n: number of agents
        spacing: distance between adjacent drones (m)

    Returns:
        [N, 3] XYZ offsets from centroid
    """
    cols = max(1, round(math.sqrt(n)))
    rows = math.ceil(n / cols)
    offsets = []
    for i in range(n):
        r = i // cols
        c = i % cols
        offsets.append([c * spacing, r * spacing, 0.0])
    t = torch.tensor(offsets)
    # Center around origin
    t[:, :2] -= t[:, :2].mean(dim=0)
    return t


def triangle_mesh(n: int, spacing: float = 0.5) -> torch.Tensor:
    """Equilateral triangle lattice (hex packing).

    Rows are offset by half-spacing for hex packing. Fills row by row
    until N agents are placed.

    Args:
        n: number of agents
        spacing: distance between adjacent drones (m)

    Returns:
        [N, 3] XYZ offsets from centroid
    """
    row_height = spacing * math.sqrt(3) / 2
    cols = max(1, round(math.sqrt(n)))
    offsets = []
    row = 0
    placed = 0
    while placed < n:
        x_offset = (row % 2) * spacing * 0.5  # stagger odd rows
        for c in range(cols):
            if placed >= n:
                break
            offsets.append([c * spacing + x_offset, row * row_height, 0.0])
            placed += 1
        row += 1
    t = torch.tensor(offsets)
    # Center around origin
    t[:, :2] -= t[:, :2].mean(dim=0)
    return t


def letter(char: str, n: int, size: float = 2.0) -> torch.Tensor:
    """Sample N points along a letter outline.

    Uses simple polyline definitions for uppercase letters.
    Points are evenly spaced along the path.

    Args:
        char: single uppercase letter (e.g. 'G')
        n: number of agents to place along the outline
        size: total height of the letter (m)

    Returns:
        [N, 3] XYZ offsets from centroid
    """
    # Define letter outlines as normalized polyline vertices [0,1] x [0,1]
    letters = _get_letter_paths()
    char = char.upper()
    if char not in letters:
        # Fallback: circle
        return polygon(n, radius=size / 2)

    path = letters[char]  # list of (x, y) normalized vertices

    # Compute cumulative arc length along path
    segments = []
    total_length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        segments.append(seg_len)
        total_length += seg_len

    if total_length < 1e-6:
        return polygon(n, radius=size / 2)

    # Sample N points evenly along the path
    points = []
    cumulative = [0.0]
    for s in segments:
        cumulative.append(cumulative[-1] + s)

    for i in range(n):
        target_dist = (i / max(n - 1, 1)) * total_length
        # Find which segment this falls on
        for seg_idx in range(len(segments)):
            if cumulative[seg_idx + 1] >= target_dist or seg_idx == len(segments) - 1:
                seg_start = cumulative[seg_idx]
                seg_len = segments[seg_idx]
                t = (target_dist - seg_start) / max(seg_len, 1e-6)
                t = max(0.0, min(1.0, t))
                x = path[seg_idx][0] + t * (path[seg_idx + 1][0] - path[seg_idx][0])
                y = path[seg_idx][1] + t * (path[seg_idx + 1][1] - path[seg_idx][1])
                points.append([x * size, y * size, 0.0])
                break

    result = torch.tensor(points)
    # Center around origin
    result[:, :2] -= result[:, :2].mean(dim=0)
    return result


def _get_letter_paths() -> dict[str, list[tuple[float, float]]]:
    """Return polyline paths for uppercase letters in [0,1] x [0,1] space."""
    return {
        "G": [
            (0.2, 0.1), (0.7, 0.1), (0.9, 0.3), (0.9, 0.7),
            (0.7, 0.9), (0.2, 0.9), (0.2, 0.5), (0.5, 0.5),
        ],
        "A": [
            (0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (0.75, 0.4), (0.25, 0.4),
        ],
        "V": [
            (0.0, 1.0), (0.5, 0.0), (1.0, 1.0),
        ],
        "S": [
            (0.8, 0.9), (0.2, 0.9), (0.1, 0.7), (0.2, 0.55),
            (0.8, 0.45), (0.9, 0.3), (0.8, 0.1), (0.2, 0.1),
        ],
        "O": [
            (0.5, 1.0), (0.15, 0.75), (0.1, 0.5), (0.15, 0.25),
            (0.5, 0.0), (0.85, 0.25), (0.9, 0.5), (0.85, 0.75), (0.5, 1.0),
        ],
        "L": [
            (0.2, 1.0), (0.2, 0.0), (0.8, 0.0),
        ],
        "T": [
            (0.0, 1.0), (1.0, 1.0), (0.5, 1.0), (0.5, 0.0),
        ],
        "H": [
            (0.1, 0.0), (0.1, 1.0), (0.1, 0.5), (0.9, 0.5),
            (0.9, 1.0), (0.9, 0.0),
        ],
    }


# Registry for easy lookup by name
FORMATIONS = {
    "polygon": polygon,
    "grid": grid,
    "triangle_mesh": triangle_mesh,
    "triangle": triangle_mesh,
}


def get_formation(name: str, n: int, **kwargs) -> torch.Tensor:
    """Look up a formation by name and return [N, 3] offsets.

    Args:
        name: formation name ('polygon', 'grid', 'triangle_mesh', 'letter_X')
        n: number of agents
        **kwargs: common params — each function picks what it needs:
            radius: for polygon (default 0.5)
            spacing: for grid/triangle_mesh (default 0.5)
            size: for letter (default 2.0)

    Returns:
        [N, 3] XYZ offsets from centroid
    """
    import inspect  # noqa: PLC0415

    if name.startswith("letter_") and len(name) >= 8:
        char = name.split("_", 1)[1]
        fn = letter
        # letter(char, n, size) — inject char as first arg
        sig_params = set(inspect.signature(fn).parameters.keys()) - {"char", "n"}
        filtered = {k: v for k, v in kwargs.items() if k in sig_params}
        return fn(char, n, **filtered)
    if name in FORMATIONS:
        fn = FORMATIONS[name]
        sig_params = set(inspect.signature(fn).parameters.keys()) - {"n"}
        filtered = {k: v for k, v in kwargs.items() if k in sig_params}
        return fn(n, **filtered)
    raise ValueError(f"Unknown formation: {name}. Available: {list(FORMATIONS.keys())} + letter_X")
