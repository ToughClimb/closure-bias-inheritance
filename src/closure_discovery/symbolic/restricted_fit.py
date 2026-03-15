from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from closure_discovery.data_generation.cases import ReactionDiffusionCase
from closure_discovery.evaluation.metrics import mean_squared_error


Array = np.ndarray


@dataclass(frozen=True)
class SymbolicExpression:
    family: str
    parameters: tuple[float, ...]
    expression: str
    complexity: int
    fit_mse: float

    def evaluate(self, u: Array) -> Array:
        values = np.asarray(u, dtype=np.float64)

        if self.family == "polynomial":
            output = np.zeros_like(values, dtype=np.float64)
            for power, coefficient in enumerate(self.parameters):
                output += coefficient * values**power
            return output

        if self.family == "rational":
            split_index = (len(self.parameters) + 1) // 2
            numerator_coefficients = self.parameters[:split_index]
            denominator_coefficients = self.parameters[split_index:]
            numerator = np.zeros_like(values, dtype=np.float64)
            denominator = np.ones_like(values, dtype=np.float64)
            for power, coefficient in enumerate(numerator_coefficients):
                numerator += coefficient * values**power
            for power, coefficient in enumerate(denominator_coefficients, start=1):
                denominator += coefficient * values**power
            return numerator / denominator

        if self.family == "exp_decay":
            offset, scale, rate = self.parameters
            return offset + scale * np.exp(rate * values)

        if self.family == "saturation":
            offset, scale, rate = self.parameters
            return offset + scale / (1.0 + rate * values)

        if self.family == "u_exp_decay":
            linear_scale, exp_scale, rate = self.parameters
            return linear_scale * values + exp_scale * values * np.exp(rate * values)

        if self.family == "u_saturation":
            linear_scale, sat_scale, rate = self.parameters
            return linear_scale * values + sat_scale * values / (1.0 + rate * values)

        raise ValueError(f"Unsupported symbolic family: {self.family}")


@dataclass(frozen=True)
class SymbolicClosurePair:
    diffusion_expression: SymbolicExpression
    reaction_expression: SymbolicExpression
    value_range: tuple[float, float]
    name: str = "symbolic_closure"

    def to_case(self, description: str | None = None) -> ReactionDiffusionCase:
        return ReactionDiffusionCase(
            name=self.name,
            description=description or "Restricted symbolic closure",
            diffusion=lambda u: self.diffusion_expression.evaluate(u),
            reaction=lambda u: self.reaction_expression.evaluate(u),
            value_range=self.value_range,
        )


def _solve_least_squares(design: Array, target: Array, ridge: float = 1.0e-10) -> tuple[Array, float]:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if ridge > 0.0:
        augmented_design = np.concatenate(
            [design, np.sqrt(ridge) * np.eye(design.shape[1], dtype=np.float64)],
            axis=0,
        )
        augmented_target = np.concatenate([target, np.zeros(design.shape[1], dtype=np.float64)], axis=0)
    else:
        augmented_design = design
        augmented_target = target

    coefficients, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    prediction = design @ coefficients
    return coefficients, mean_squared_error(target, prediction)


def _format_signed_terms(terms: list[tuple[float, str]]) -> str:
    pieces: list[str] = []
    for index, (coefficient, symbol) in enumerate(terms):
        if abs(coefficient) < 1.0e-12:
            continue
        magnitude = f"{abs(coefficient):.6g}"
        signed_piece = magnitude if symbol == "1" else f"{magnitude}*{symbol}"
        if index == 0 and not pieces:
            pieces.append(signed_piece if coefficient >= 0.0 else f"-{signed_piece}")
        else:
            sign = "+" if coefficient >= 0.0 else "-"
            pieces.append(f" {sign} {signed_piece}")
    return "".join(pieces) if pieces else "0"


def _polynomial_expression(coefficients: Array) -> str:
    terms = []
    for power, coefficient in enumerate(coefficients):
        if power == 0:
            symbol = "1"
        elif power == 1:
            symbol = "u"
        else:
            symbol = f"u^{power}"
        terms.append((float(coefficient), symbol))
    return _format_signed_terms(terms)


def _rational_expression(numerator: Array, denominator: Array) -> str:
    numerator_terms = []
    for power, coefficient in enumerate(numerator):
        if power == 0:
            symbol = "1"
        elif power == 1:
            symbol = "u"
        else:
            symbol = f"u^{power}"
        numerator_terms.append((float(coefficient), symbol))

    denominator_terms = [(1.0, "1")]
    for power, coefficient in enumerate(denominator, start=1):
        if power == 1:
            symbol = "u"
        else:
            symbol = f"u^{power}"
        denominator_terms.append((float(coefficient), symbol))

    return f"({_format_signed_terms(numerator_terms)}) / ({_format_signed_terms(denominator_terms)})"


def fit_polynomial_family(
    u: Array,
    values: Array,
    *,
    max_degree: int,
    ridge: float = 1.0e-10,
) -> list[SymbolicExpression]:
    candidates = []
    for degree in range(max_degree + 1):
        design = np.stack([u**power for power in range(degree + 1)], axis=-1)
        coefficients, fit_mse = _solve_least_squares(design, values, ridge=ridge)
        candidates.append(
            SymbolicExpression(
                family="polynomial",
                parameters=tuple(float(value) for value in coefficients),
                expression=_polynomial_expression(coefficients),
                complexity=degree + 1,
                fit_mse=fit_mse,
            )
        )
    return candidates


def fit_rational_family(
    u: Array,
    values: Array,
    *,
    numerator_degree: int = 2,
    denominator_degree: int = 2,
    ridge: float = 1.0e-10,
    denominator_tol: float = 1.0e-3,
) -> list[SymbolicExpression]:
    design_columns = [u**power for power in range(numerator_degree + 1)]
    design_columns.extend([-(values * (u**power)) for power in range(1, denominator_degree + 1)])
    design = np.stack(design_columns, axis=-1)
    coefficients, _ = _solve_least_squares(design, values, ridge=ridge)

    numerator = coefficients[: numerator_degree + 1]
    denominator = coefficients[numerator_degree + 1 :]
    denominator_values = np.ones_like(u, dtype=np.float64)
    for power, coefficient in enumerate(denominator, start=1):
        denominator_values += coefficient * u**power

    if np.min(np.abs(denominator_values)) < denominator_tol:
        return []

    numerator_values = np.zeros_like(u, dtype=np.float64)
    for power, coefficient in enumerate(numerator):
        numerator_values += coefficient * u**power
    prediction = numerator_values / denominator_values

    return [
        SymbolicExpression(
            family="rational",
            parameters=tuple(float(value) for value in np.concatenate([numerator, denominator])),
            expression=_rational_expression(numerator, denominator),
            complexity=numerator_degree + denominator_degree + 1,
            fit_mse=mean_squared_error(values, prediction),
        )
    ]


def _grid_search_linear_family(
    u: Array,
    values: Array,
    *,
    family: str,
    rate_grid: Array,
    feature_builder,
    complexity: int,
    ridge: float = 1.0e-10,
) -> list[SymbolicExpression]:
    best_candidate: SymbolicExpression | None = None
    for rate in rate_grid:
        design = feature_builder(u, float(rate))
        coefficients, fit_mse = _solve_least_squares(design, values, ridge=ridge)

        if family == "exp_decay":
            expression = f"{coefficients[0]:.6g} + {coefficients[1]:.6g}*exp({rate:.6g}*u)"
            parameters = (float(coefficients[0]), float(coefficients[1]), float(rate))
        elif family == "saturation":
            expression = f"{coefficients[0]:.6g} + {coefficients[1]:.6g}/(1 + {rate:.6g}*u)"
            parameters = (float(coefficients[0]), float(coefficients[1]), float(rate))
        elif family == "u_exp_decay":
            expression = f"{coefficients[0]:.6g}*u + {coefficients[1]:.6g}*u*exp({rate:.6g}*u)"
            parameters = (float(coefficients[0]), float(coefficients[1]), float(rate))
        elif family == "u_saturation":
            expression = f"{coefficients[0]:.6g}*u + {coefficients[1]:.6g}*u/(1 + {rate:.6g}*u)"
            parameters = (float(coefficients[0]), float(coefficients[1]), float(rate))
        else:
            raise ValueError(f"Unsupported family in grid search: {family}")

        candidate = SymbolicExpression(
            family=family,
            parameters=parameters,
            expression=expression,
            complexity=complexity,
            fit_mse=fit_mse,
        )
        if best_candidate is None or candidate.fit_mse < best_candidate.fit_mse:
            best_candidate = candidate

    return [best_candidate] if best_candidate is not None else []


def fit_diffusion_candidates(
    u: Array,
    values: Array,
    *,
    max_polynomial_degree: int = 3,
    ridge: float = 1.0e-10,
) -> list[SymbolicExpression]:
    u = np.asarray(u, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    candidates = fit_polynomial_family(u, values, max_degree=max_polynomial_degree, ridge=ridge)
    candidates.extend(fit_rational_family(u, values, numerator_degree=2, denominator_degree=2, ridge=ridge))

    exp_rates = np.concatenate([np.linspace(-8.0, -0.1, 80), np.linspace(0.1, 8.0, 80)])
    candidates.extend(
        _grid_search_linear_family(
            u,
            values,
            family="exp_decay",
            rate_grid=exp_rates,
            feature_builder=lambda support, rate: np.stack(
                [np.ones_like(support), np.exp(rate * support)],
                axis=-1,
            ),
            complexity=3,
            ridge=ridge,
        )
    )

    sat_rates = np.linspace(0.05, 12.0, 120)
    candidates.extend(
        _grid_search_linear_family(
            u,
            values,
            family="saturation",
            rate_grid=sat_rates,
            feature_builder=lambda support, rate: np.stack(
                [np.ones_like(support), 1.0 / (1.0 + rate * support)],
                axis=-1,
            ),
            complexity=3,
            ridge=ridge,
        )
    )
    return candidates


def fit_reaction_candidates(
    u: Array,
    values: Array,
    *,
    max_polynomial_degree: int = 4,
    ridge: float = 1.0e-10,
) -> list[SymbolicExpression]:
    u = np.asarray(u, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    candidates = fit_polynomial_family(u, values, max_degree=max_polynomial_degree, ridge=ridge)
    candidates.extend(fit_rational_family(u, values, numerator_degree=2, denominator_degree=2, ridge=ridge))

    exp_rates = np.concatenate([np.linspace(-8.0, -0.1, 80), np.linspace(0.1, 8.0, 80)])
    candidates.extend(
        _grid_search_linear_family(
            u,
            values,
            family="u_exp_decay",
            rate_grid=exp_rates,
            feature_builder=lambda support, rate: np.stack(
                [support, support * np.exp(rate * support)],
                axis=-1,
            ),
            complexity=3,
            ridge=ridge,
        )
    )

    sat_rates = np.linspace(0.05, 12.0, 120)
    candidates.extend(
        _grid_search_linear_family(
            u,
            values,
            family="u_saturation",
            rate_grid=sat_rates,
            feature_builder=lambda support, rate: np.stack(
                [support, support / (1.0 + rate * support)],
                axis=-1,
            ),
            complexity=3,
            ridge=ridge,
        )
    )
    return candidates


def select_symbolic_expression(
    candidates: list[SymbolicExpression],
    *,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1.0e-8,
) -> SymbolicExpression:
    if not candidates:
        raise ValueError("No symbolic candidates were produced.")

    best_mse = min(candidate.fit_mse for candidate in candidates)
    threshold = best_mse * (1.0 + relative_tolerance) + absolute_tolerance
    eligible = [candidate for candidate in candidates if candidate.fit_mse <= threshold]
    return min(eligible, key=lambda candidate: (candidate.complexity, candidate.fit_mse))
