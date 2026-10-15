"""Independent scalar DOP853 check of completed uniform-state RK4 validations.

Uniform fields have zero diffusion flux. Compare signed terminal states and
absolute errors with a separate scalar ODE solver, preserving numerical failures.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'revision'))
import numpy as np
from scipy.integrate import solve_ivp, quad
from artifact_io import ROOT, json_write
from revision_core import extra_case
from run_reliability_supplement import load_manifest, digest
from evaluate_reliability_supplement import functions, evaluation_identity


def scalar_solution(reaction, initial, final_time):
    solved = solve_ivp(lambda t, a: reaction(a), (0., final_time), [initial],
        method='DOP853', rtol=1e-11, atol=1e-13, max_step=.001)
    return dict(success=bool(solved.success and np.isfinite(solved.y).all()),
        message=solved.message, reached_time=float(solved.t[-1]), nfev=int(solved.nfev),
        state=float(solved.y[0,-1]) if np.isfinite(solved.y[0,-1]) else None)


def pole_diagnostic(descriptor, initial, solved):
    """Diagnose uncancelled polynomial-rational poles approached by a failed ODE."""
    if descriptor['kind'] != 'restricted': return None
    expression = descriptor['symbols'][1]
    if expression['family'] != 'rational' or solved['state'] is None: return None
    degree = expression['numerator_degree']
    split = (2 if degree is None else degree)+1
    numerator = expression['parameters'][:split]; denominator = [1., *expression['parameters'][split:]]
    roots = np.polynomial.polynomial.polyroots(denominator)
    candidates = [float(r.real) for r in roots if abs(r.imag)<1e-10 and abs(r.real-solved['state'])<1e-5]
    if not candidates: return None
    root = min(candidates, key=lambda r: abs(r-solved['state']))
    numerator_at_root = float(np.polynomial.polynomial.polyval(root, numerator))
    if abs(numerator_at_root)<1e-12: return None
    lo, hi = sorted((initial, root))
    zeros = np.polynomial.polynomial.polyroots(numerator)
    if any(abs(z.imag)<1e-10 and lo<=z.real<=hi for z in zeros):
        return dict(pole_state=root, numerator_at_pole=numerator_at_root,
                    travel_time=None, reason='Reaction numerator has a zero on the proposed path')
    travel, error = quad(lambda a: np.polynomial.polynomial.polyval(a, denominator)/
                         np.polynomial.polynomial.polyval(a, numerator), initial, root,
                         epsabs=1e-13, epsrel=1e-13)
    return dict(pole_state=root, numerator_at_pole=numerator_at_root,
        travel_time=float(travel), quadrature_error_estimate=float(error),
        ode_time_difference=abs(float(travel)-solved['reached_time']),
        rule='Integrate 1/R along the uniform solution path to a noncancelled rational pole')


def main():
    load_manifest()
    result = dict(status='running', protocol=evaluation_identity(),
        verifier_sha256=digest(__file__), method='Scalar DOP853 rtol1e-11 atol1e-13 max_step.001',
        endpoint_absolute_tolerance=2e-9, checks=[], failed_forward_patterns_retained=[],
        independent_model_failures=[],
        failure_policy='Continue auditing all models; retain independent solver failures and mismatches; never certify an endpoint beyond an uncompleted scalar solve')
    path = ROOT/'evidence/uniform_ode_checks.json'
    try:
        for cohort, folder, expected in [('fresh', 'matched_validation', 10), ('archived', 'archive_validation', 30)]:
            files = sorted((ROOT/'evidence'/folder).glob('*.json'))
            if len(files) != expected: raise ValueError('Uniform verification requires the completed full cohort')
            for source in files:
                row = json.loads(source.read_text())
                if row['status'] != 'complete': raise ValueError('Uniform verification input incomplete')
                case = extra_case(row['config']['case'])
                for label, model in row['models'].items():
                    _, reaction = functions(model['descriptor'])
                    for pattern, record in model['validation'].items():
                        if not pattern.startswith('uniform_'): continue
                        identity = dict(cohort=cohort, id=row['id'], method=label, pattern=pattern,
                            source_record_sha256=digest(source))
                        info = record['info']
                        if not info['numerical_completion']:
                            result['failed_forward_patterns_retained'].append(dict(**identity, reason=info['reason']))
                            continue
                        if record['initial_min'] != record['initial_max']:
                            raise ValueError('Uniform pattern has nonuniform initial values')
                        if abs(info['state_max']-info['state_min']) > 2e-12:
                            raise AssertionError('Uniform forward state lost spatial uniformity')
                        initial, final_time = record['initial_min'], info['requested_final_time']
                        solved = scalar_solution(reaction, initial, final_time)
                        if not solved['success']:
                            result['independent_model_failures'].append(dict(**identity,
                                classification='independent_solver_failed_despite_rk4_guard_completion',
                                scalar_solver=solved, rk4_signed_endpoint=info['state_min'],
                                accepted_state_range_valid=info['accepted_state_range_valid'],
                                rational_pole=pole_diagnostic(model['descriptor'], initial, solved)))
                            continue
                        truth = scalar_solution(case.reaction, initial, final_time)
                        if not truth['success']: raise FloatingPointError('True scalar reference failed')
                        predicted, reference = solved['state'], truth['state']
                        difference = abs(predicted-info['state_min'])
                        error_difference = abs(abs(predicted-reference)-record['final_absolute_rms'])
                        if max(difference, error_difference) > result['endpoint_absolute_tolerance']:
                            result['independent_model_failures'].append(dict(**identity,
                                classification='completed_solvers_disagree_beyond_tolerance',
                                scalar_solver=solved, rk4_signed_endpoint=info['state_min'],
                                accepted_state_range_valid=info['accepted_state_range_valid'],
                                rk4_endpoint_difference=difference, absolute_rms_difference=error_difference))
                            continue
                        result['checks'].append(dict(**identity, signed_scalar_prediction=predicted,
                            scalar_reference=reference, rk4_endpoint_difference=difference,
                            absolute_rms_difference=error_difference))
        total = len(result['checks']) + len(result['failed_forward_patterns_retained']) + len(result['independent_model_failures'])
        if total != 640: raise ValueError(f'Expected 640 uniform model-pattern records, found {total}')
        result.update(status='complete', compared=len(result['checks']),
            numerical_failures=len(result['failed_forward_patterns_retained']), total=total,
            independent_failure_count=len(result['independent_model_failures']),
            all_completed_rk4_uniform_endpoints_independently_verified=not result['independent_model_failures'],
            maximum_endpoint_difference=max(c['rk4_endpoint_difference'] for c in result['checks']),
            maximum_absolute_rms_difference=max(c['absolute_rms_difference'] for c in result['checks']))
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
    json_write(path, result)
    shutil.copyfile(__file__, ROOT/'provenance/source/experiments'/Path(__file__).name)
    print(json.dumps({key:value for key,value in result.items() if key not in ('checks','failed_forward_patterns_retained','independent_model_failures','protocol')}), flush=True)
    if result['status'] != 'complete': raise RuntimeError('Independent uniform-state audit did not complete; inspect raw result')


if __name__ == '__main__': main()
