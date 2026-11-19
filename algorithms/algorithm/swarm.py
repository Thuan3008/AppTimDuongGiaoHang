from __future__ import annotations

from core.models import SwarmState


def swarm_profiles(k: int) -> list[tuple[str, float, float]]:
    if k == 1: return [("Balanced RAPO", 0.5, 0.5)]
    profiles = []
    for i in range(k):
        frac = i / (k - 1)
        p_aro = 0.8 - 0.6 * frac
        role = "Exploration" if frac < 0.34 else "Exploitation" if frac > 0.66 else "Balanced"
        profiles.append((role, p_aro, 1 - p_aro))
    return profiles


def adaptive_probabilities(base_aro: float, iteration: int, max_iterations: int, enabled: bool) -> tuple[float, float]:
    if not enabled: return base_aro, 1 - base_aro
    alpha = 1 - iteration / max_iterations
    minimum = max(0.05, base_aro - 0.2)
    p_aro = minimum + alpha * (base_aro - minimum)
    return p_aro, 1 - p_aro
