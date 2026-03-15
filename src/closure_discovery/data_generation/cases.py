from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ReactionDiffusionCase:
    """Defines a synthetic closure pair for the target PDE."""

    name: str
    description: str
    diffusion: Callable[[Array], Array]
    reaction: Callable[[Array], Array]
    value_range: tuple[float, float]


def case_a(d0: float = 0.01, d1: float = 0.05, r: float = 1.0) -> ReactionDiffusionCase:
    """Linear diffusion and logistic growth."""

    def diffusion(u: Array) -> Array:
        return d0 + d1 * u

    def reaction(u: Array) -> Array:
        return r * u * (1.0 - u)

    return ReactionDiffusionCase(
        name="case_a",
        description="D(u)=d0+d1*u, R(u)=r*u*(1-u)",
        diffusion=diffusion,
        reaction=reaction,
        value_range=(0.0, 1.2),
    )


def case_b(
    d0: float = 0.01,
    d1: float = 0.03,
    a: float = 1.0,
    b: float = 1.5,
    c: float = 0.5,
) -> ReactionDiffusionCase:
    """Quadratic diffusion and cubic reaction with a stable unit-scale regime."""

    def diffusion(u: Array) -> Array:
        return d0 + d1 * u**2

    def reaction(u: Array) -> Array:
        return a * u - b * u**2 + c * u**3

    return ReactionDiffusionCase(
        name="case_b",
        description="D(u)=d0+d1*u^2, R(u)=a*u-b*u^2+c*u^3",
        diffusion=diffusion,
        reaction=reaction,
        value_range=(0.0, 1.4),
    )


def case_c(
    d0: float = 0.01,
    d1: float = 0.08,
    d2: float = 4.0,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.2,
) -> ReactionDiffusionCase:
    """Saturating diffusion and rational reaction."""

    def diffusion(u: Array) -> Array:
        return d0 + (d1 * u**2) / (1.0 + d2 * u**2)

    def reaction(u: Array) -> Array:
        return (a * u) / (1.0 + b * u) - c * u

    return ReactionDiffusionCase(
        name="case_c",
        description="D(u)=d0+d1*u^2/(1+d2*u^2), R(u)=a*u/(1+b*u)-c*u",
        diffusion=diffusion,
        reaction=reaction,
        value_range=(0.0, 1.5),
    )


def case_exp(
    d0: float = 0.01,
    d1: float = 0.035,
    k: float = 2.5,
    a: float = 1.1,
    b: float = 1.4,
    c: float = 0.22,
) -> ReactionDiffusionCase:
    """Smooth non-polynomial closures for restricted-library stress tests."""

    def diffusion(u: Array) -> Array:
        return d0 + d1 * (1.0 - np.exp(-k * u))

    def reaction(u: Array) -> Array:
        return u * (a * np.exp(-b * u) - c)

    return ReactionDiffusionCase(
        name="case_exp",
        description="D(u)=d0+d1*(1-exp(-k*u)), R(u)=u*(a*exp(-b*u)-c)",
        diffusion=diffusion,
        reaction=reaction,
        value_range=(0.0, 1.8),
    )


CASE_BUILDERS: dict[str, Callable[[], ReactionDiffusionCase]] = {
    "case_a": case_a,
    "case_b": case_b,
    "case_c": case_c,
    "case_exp": case_exp,
}
