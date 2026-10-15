"""Fixed-family coefficient slots, distinct from PySR literal sharing."""
from __future__ import annotations

import dataclasses
import numpy as np
import sympy as sp
from closure_discovery.symbolic.restricted_fit import SymbolicExpression


class FixedFamilyTemplate:
    def __init__(self, expression, starting=None):
        self.original = expression
        self.constants = expression.parameters
        self.parameters = np.array(expression.parameters if starting is None else starting, dtype=float)
        if len(self.parameters) != len(self.constants):
            raise ValueError('Fixed-family starting coefficients do not match structure')

    def evaluate(self, parameters, values):
        return dataclasses.replace(self.original, parameters=tuple(parameters)).evaluate(values)

    def expression(self, parameters):
        x=sp.Symbol('x0');p=[sp.Float(float(value),17) for value in parameters]
        family=self.original.family
        if family=='polynomial':
            expression=sum(coefficient*x**power for power,coefficient in enumerate(p))
        elif family=='rational':
            degree=self.original.numerator_degree
            degree=2 if degree is None else degree
            expression=sum(coefficient*x**power for power,coefficient in enumerate(p[:degree+1]))
            expression/=1+sum(coefficient*x**power for power,coefficient in enumerate(p[degree+1:],1))
        elif family=='exp_decay':expression=p[0]+p[1]*sp.exp(p[2]*x)
        elif family=='saturation':expression=p[0]+p[1]/(1+p[2]*x)
        elif family=='u_exp_decay':expression=p[0]*x+p[1]*x*sp.exp(p[2]*x)
        elif family=='u_saturation':expression=p[0]*x+p[1]*x/(1+p[2]*x)
        else:raise ValueError(f'Unsupported selected family: {family}')
        return str(expression)


def symbolic_pair(values):
    return tuple(SymbolicExpression(**{**value,'parameters':tuple(value['parameters'])}) for value in values)


def denominator_roots(expression, interval):
    if expression.family=='rational':
        degree=expression.numerator_degree
        degree=2 if degree is None else degree
        coefficients=[1.,*expression.parameters[degree+1:]]
    elif expression.family in ('saturation','u_saturation'):
        coefficients=[1.,expression.parameters[2]]
    else:return []
    roots=np.polynomial.polynomial.polyroots(coefficients)
    return [float(root.real) for root in roots if abs(root.imag)<=1e-10*max(1.,abs(root.real))
            and interval[0]-1e-12<=root.real<=interval[1]+1e-12]
