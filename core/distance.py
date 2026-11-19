from __future__ import annotations

import numpy as np

from .models import Customer, Depot


def build_distance_matrix(nodes: list[Customer | Depot]) -> np.ndarray:
    coords = np.array([(node.x, node.y) for node in nodes], dtype=float)
    return np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
