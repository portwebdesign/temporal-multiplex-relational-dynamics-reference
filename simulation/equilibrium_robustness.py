"""Local equilibrium continuation of the unchanged seven-state mechanism.

The protocol is recorded separately before new outcomes. This module tracks
smooth physical equilibrium branches; it neither fits observations nor claims
that its seed-connected curves exhaust all equilibria.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from model import STATE_NAMES, derivative
from network import aggregate_network
from regimes import SOLVED_COEFFICIENTS, analytic_jacobian, finite_difference_jacobian

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newton_correct(fun, jac, initial, *, tangent=None, predictor=None,
                   tolerance=1e-11, max_iterations=30, physical_dimension=None):
    """Damped Newton with optional arclength constraint; no state clipping."""
    y = np.array(initial, dtype=float)
    def valid(value):
        return (np.isfinite(value).all() and
                (physical_dimension is None or
                 ((value[:physical_dimension] >= 0).all() and
                  (value[:physical_dimension] <= 1).all() and value[-1] > 0)))
    def evaluate(value):
        if not valid(value):
            raise ValueError('Newton iterate outside the physical domain')
        residual = np.asarray(fun(value), dtype=float)
        matrix = np.asarray(jac(value), dtype=float)
        if tangent is not None:
            residual = np.r_[residual, np.dot(value-predictor, tangent)]
            matrix = np.vstack([matrix, tangent])
        return residual, matrix
    try:
        f, matrix = evaluate(y)
        for iteration in range(max_iterations + 1):
            norm = float(np.max(np.abs(f)))
            if norm <= tolerance:
                return y, {'success': True, 'iterations': iteration, 'residual': norm}
            if iteration == max_iterations:
                break
            try:
                step = np.linalg.solve(matrix, -f)
            except np.linalg.LinAlgError:
                return None, {'success': False, 'reason': 'singular_corrector_matrix', 'residual': norm}
            accepted = False
            for exponent in range(22):
                trial = y + step * 2.0**(-exponent)
                try:
                    trial_f, trial_matrix = evaluate(trial)
                except ValueError:
                    continue
                if np.max(np.abs(trial_f)) < norm:
                    y, f, matrix = trial, trial_f, trial_matrix
                    accepted = True
                    break
            if not accepted:
                return None, {'success': False, 'reason': 'backtracking_exhausted', 'residual': norm}
        return None, {'success': False, 'reason': 'iteration_budget', 'residual': float(np.max(np.abs(f)))}
    except ValueError as exc:
        return None, {'success': False, 'reason': str(exc)}


def null_tangent(matrix, previous=None, orientation=1):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular[-1] <= 1e-13:
        raise ValueError('equilibrium derivative loses row rank')
    tangent = vh[-1]
    if previous is None:
        if tangent[-1] * orientation < 0:
            tangent *= -1
    elif np.dot(tangent, previous) < 0:
        tangent *= -1
    return tangent


def continue_curve(fun, jac, initial, settings, bounds, orientation, physical_dimension=None):
    """Trace one oriented seed-connected curve, including passage through folds."""
    y = np.asarray(initial, dtype=float)
    tangent = null_tangent(jac(y), orientation=orientation)
    points = [(y.copy(), tangent.copy())]
    failures = []
    step = settings['initial_arclength_step']
    reason = 'maximum_steps'
    for index in range(settings['maximum_steps_per_sweep']):
        predictor = y + step*tangent
        corrected, record = newton_correct(fun, jac, predictor, tangent=tangent,
                                           predictor=predictor, tolerance=settings['residual_tolerance'],
                                           max_iterations=settings['newton_max_iterations'],
                                           physical_dimension=physical_dimension)
        if corrected is None:
            failures.append({'step_index': index, 'arclength_step': step, **record})
            step /= 2
            if step < settings['minimum_arclength_step']:
                reason = 'minimum_step_exhausted'
                break
            continue
        new_tangent = null_tangent(jac(corrected), previous=tangent)
        points.append((corrected.copy(), new_tangent.copy()))
        if corrected[-1] < bounds[0] or corrected[-1] > bounds[1]:
            reason = 'factor_boundary_crossed'
            break
        if len(points) > 20 and any(np.linalg.norm(corrected-old[0]) < step*0.1 for old in points[:-15]):
            reason = 'visited_curve_return'
            break
        y, tangent = corrected, new_tangent
        if record['iterations'] <= 4:
            step = min(step*1.2, settings['maximum_arclength_step'])
    return points, {'termination': reason, 'accepted_points': len(points),
                    'rejected_steps': failures, 'orientation': orientation}


class ModelFamily:
    def __init__(self, configuration, parameter):
        self.configuration = configuration
        self.parameter = parameter
        self.nominal = configuration['parameters'][parameter]
        self.inputs = configuration['inputs']['baseline']
        self.signals = aggregate_network(configuration['network']['baseline'])
        self.target = configuration['permeability_target']['baseline']

    def parameters(self, factor):
        return {**self.configuration['parameters'], self.parameter: self.nominal*float(factor)}

    def fun(self, value):
        return np.asarray(derivative(value[:-1], self.inputs, self.parameters(value[-1]), self.signals, self.target))

    def state_jacobian(self, value):
        return np.asarray(analytic_jacobian(value[:-1], self.inputs, self.parameters(value[-1]), self.signals, self.target))

    def parameter_derivative(self, value):
        s,e,r,c,p,w1,w2 = value[:-1]
        u = self.inputs
        par = self.parameters(value[-1])
        from model import diagnostics
        psi = diagnostics(value[:-1], u, par, self.signals).pressure_proxy
        result = np.zeros(7)
        index, expression = {
            'a_s': (0, p*(1-s)),
            'a_e': (1, -u['review']*e),
            'eta_r': (2, -psi*r),
            'rho_p': (4, s*(par['a_p']+par['b_p']*w1*u['review'])*(1-p)),
            'd_w1': (5, -e*(1-s)*w1),
            'h_w2': (6, -u['delay']*w2),
        }[self.parameter]
        result[index] = self.nominal*expression
        return result

    def jac(self, value):
        return np.column_stack([self.state_jacobian(value), self.parameter_derivative(value)])


def fixed_root(family, initial, factor):
    """Correct one branch crossing at a fixed factor, retaining root failures."""
    guess = np.r_[initial, factor]
    fun = lambda y: np.r_[family.fun(y), y[-1]-factor]
    jac = lambda y: np.vstack([family.jac(y), np.r_[np.zeros(7), 1.]])
    return newton_correct(fun, jac, guess, physical_dimension=7)


def point_record(family, value):
    eig = np.linalg.eigvals(family.state_jacobian(value))
    maximum = float(eig.real.max())
    stability = 'stable' if maximum < -1e-8 else 'unstable' if maximum > 1e-8 else 'marginal'
    return {'factor': float(value[-1]), 'parameter_value': float(value[-1]*family.nominal),
            **{name: float(number) for name, number in zip(STATE_NAMES, value[:-1])},
            'residual': float(np.max(np.abs(family.fun(value)))),
            'max_real_eigenvalue': maximum, 'unstable_eigenvalues': int(np.sum(eig.real > 1e-8)),
            'stability': stability}


def fold_diagnostic(family, left, right, settings, factor_bounds, maximum_step):
    start = (left + right)/2
    _, _, vh = np.linalg.svd(family.state_jacobian(start))
    initial = np.r_[start, vh[-1]]
    def residual(value):
        y, v = value[:8], value[8:]
        return np.r_[family.fun(y), family.state_jacobian(y)@v, v@v-1]
    lower = np.r_[np.full(7, 1e-9), .85, np.full(7, -np.inf)]
    upper = np.r_[np.full(7, 1-1e-9), 1.15, np.full(7, np.inf)]
    fit = least_squares(residual, initial, bounds=(lower, upper),
                        ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=3000)
    y = fit.x[:8]
    matrix = family.state_jacobian(y)
    u, singular, vh = np.linalg.svd(matrix)
    v, w = vh[-1], u[:, -1]
    eig = np.linalg.eigvals(matrix)
    quadratic = []
    for h in [1e-4, 5e-5]:
        high, low = y.copy(), y.copy()
        high[:-1] += h*v
        low[:-1] -= h*v
        second = (family.fun(high)-2*family.fun(y)+family.fun(low))/(h*h)
        quadratic.append(float(w@second))
    projection = float(w@family.parameter_derivative(y))
    error = float(np.max(np.abs(residual(fit.x))))
    candidate_distance = float(np.linalg.norm(y-start))
    local_candidate = bool(factor_bounds[0] <= y[-1] <= factor_bounds[1] and
                           candidate_distance <= 4*maximum_step)
    qualifies = bool(fit.success and error < settings['residual_tolerance'] and
                     local_candidate and
                     np.sum(np.abs(eig) < settings['null_eigenvalue_tolerance']) == 1 and
                     abs(projection) > settings['nondegeneracy_minimum_absolute'] and
                     min(abs(q) for q in quadratic) > settings['nondegeneracy_minimum_absolute'] and
                     quadratic[0]*quadratic[1] > 0)
    return {**point_record(family,y), 'status': 'nondegenerate_local_fold_numerical' if qualifies else 'unresolved_fold_candidate',
            'augmented_residual': error, 'solver_success': bool(fit.success),
            'solver_message': str(fit.message), 'minimum_singular_value': float(singular[-1]),
            'within_declared_factor_interval_and_local_candidate': local_candidate,
            'candidate_midpoint_distance': candidate_distance,
            'next_singular_value': float(singular[-2]), 'minimum_absolute_eigenvalue': float(np.min(np.abs(eig))),
            'left_null_parameter_projection': projection,
            'left_null_quadratic_projection_h1_h2': quadratic,
            'state': y[:-1].tolist()}


def initial_configuration(protocol):
    from referee_experiments import construction_record
    reference = json.loads((HERE/protocol['source_protocol']).read_text())
    cfg = json.loads((HERE/'presentation-results/aligned.json').read_text())
    cfg['parameters'] = construction_record(reference)['parameters']
    cfg['network'] = {'baseline': reference['network']['baseline'], 'schedule': []}
    cfg['inputs']['schedule'] = []
    cfg['permeability_target']['schedule'] = []
    lower = json.loads((HERE/'presentation-results/lower_aligned.json').read_text())['initial_state']
    return cfg, {'upper': np.array(reference['baseline_construction']['target']), 'lower': np.array(lower)}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(protocol, refine_factor=1):
    cfg, seeds = initial_configuration(protocol)
    settings = dict(protocol['continuation'])
    settings['initial_arclength_step'] /= refine_factor
    settings['maximum_arclength_step'] /= refine_factor
    all_points, roots, folds, diagnostics = [], [], [], []
    derivative_errors, state_errors, duplicate_crossing_errors = [], [], []
    for parameter in protocol['parameters']:
        family = ModelFamily(cfg, parameter)
        derivative_step = 1e-5
        y = np.r_[seeds['upper'], 1.]
        high, low = y.copy(), y.copy()
        high[-1] += derivative_step
        low[-1] -= derivative_step
        derivative_errors.append(float(np.max(np.abs((family.fun(high)-family.fun(low))/(2*derivative_step)-family.parameter_derivative(y)))))
        state_fd = finite_difference_jacobian(y[:-1],family.inputs,family.parameters(1),family.signals,family.target)
        state_errors.append(float(np.max(np.abs(family.state_jacobian(y)-np.array(state_fd)))))
        parameter_roots, parameter_folds, root_connections = [], [], []
        for branch, seed in seeds.items():
            start, corrected = fixed_root(family, seed, 1.)
            if start is None:
                raise RuntimeError(f'Nominal {branch} seed correction failed: {corrected}')
            for orientation in [-1,1]:
                points, diagnostic = continue_curve(family.fun,family.jac,start,settings,
                                                    protocol['factor_interval'],orientation,7)
                sweep=f'{parameter}_{branch}_{orientation:+d}'
                diagnostics.append({'sweep':sweep,**diagnostic})
                for index,(value,tangent) in enumerate(points):
                    all_points.append({'parameter':parameter,'sweep':sweep,'index':index,
                                       **point_record(family,value),'tangent_factor':float(tangent[-1])})
                nominal_segment = True
                for (left,lt),(right,rt) in zip(points[:-1],points[1:]):
                    if lt[-1]*rt[-1] < 0:
                        fold=fold_diagnostic(family,left,right,protocol['fold_diagnostic'],
                                             protocol['factor_interval'],settings['maximum_arclength_step'])
                        connections={branch} if nominal_segment else set()
                        duplicate=next((p for p in parameter_folds if abs(fold['factor']-p['factor'])<1e-7 and np.max(np.abs(np.array(fold['state'])-p['state']))<1e-6),None)
                        if duplicate is None:
                            fold['nominal_reference_connection']='|'.join(sorted(connections))
                            parameter_folds.append(fold)
                        elif connections:
                            previous=set(filter(None,duplicate['nominal_reference_connection'].split('|')))
                            duplicate['nominal_reference_connection']='|'.join(sorted(previous|connections))
                        nominal_segment=False
                    for factor in protocol['fixed_factor_checks']:
                        if (left[-1]-factor)*(right[-1]-factor) > 0:
                            continue
                        fraction = (factor-left[-1])/(right[-1]-left[-1]) if right[-1]!=left[-1] else .5
                        initial=left[:-1]+fraction*(right[:-1]-left[:-1])
                        root,record=fixed_root(family,initial,factor)
                        if root is None:
                            diagnostics.append({'sweep':sweep,'factor':factor,'fixed_factor_failure':record})
                        else:
                            duplicate=next((i for i,r in enumerate(parameter_roots) if abs(root[-1]-r[-1])<1e-10 and np.max(np.abs(root[:-1]-r[:-1]))<1e-6),None)
                            connections={branch} if nominal_segment else set()
                            if duplicate is None:
                                parameter_roots.append(root)
                                root_connections.append(connections)
                            else:
                                duplicate_crossing_errors.append(float(np.max(np.abs(root-parameter_roots[duplicate]))))
                                root_connections[duplicate].update(connections)
        roots += [{'parameter':parameter,**point_record(family,y),'state':y[:-1].tolist(),
                   'nominal_reference_connection':'|'.join(sorted(connection))}
                  for y,connection in sorted(zip(parameter_roots,root_connections),key=lambda item:(item[0][-1],item[0][2]))]
        folds += [{'parameter':parameter,**fold} for fold in sorted(parameter_folds,key=lambda f:f['factor'])]
        print(json.dumps({'parameter':parameter,'refinement_factor':refine_factor,'root_crossings':len(parameter_roots),'folds':[(f['factor'],f['status']) for f in parameter_folds]}),flush=True)
    return {'points':all_points,'roots':roots,'folds':folds,'diagnostics':diagnostics,
            'parameter_derivative_max_error':max(derivative_errors),'analytic_jacobian_max_error':max(state_errors),
            'duplicate_sweep_crossing_max_difference':max(duplicate_crossing_errors+[0])}


def compare_refinement(coarse, fine):
    root_errors, fold_errors = [], []
    failures=[]
    for label in ['roots','folds']:
        first, second=coarse[label],fine[label]
        if len(first)!=len(second):
            failures.append(f'{label} inventory differs: {len(first)} vs {len(second)}')
        for row in first:
            matches=[r for r in second if r['parameter']==row['parameter'] and (label=='folds' or abs(r['factor']-row['factor'])<1e-10)]
            if not matches:
                failures.append(f'No refined {label} match for {row["parameter"]} {row["factor"]}')
                continue
            match=min(matches,key=lambda r:np.max(np.abs(np.array(r['state'])-row['state'])))
            error=max(abs(match['factor']-row['factor']),float(np.max(np.abs(np.array(match['state'])-row['state']))))
            (root_errors if label=='roots' else fold_errors).append(error)
    return {'status':'PASS' if not failures and max(root_errors+fold_errors+[0])<1e-7 else 'FAIL',
            'max_root_difference':max(root_errors+[0]),'max_fold_difference':max(fold_errors+[0]),'failures':failures}


def previous_comparisons(result):
    rows=list(csv.DictReader((HERE/'presentation-results/neighborhood_sensitivity.csv').open()))
    comparisons=[]
    for row in rows:
        factor=float(row['factor']);parameter=row['parameter'];original=np.array(json.loads(row['baseline_state']))
        candidates=[r for r in result['roots'] if r['parameter']==parameter and abs(r['factor']-factor)<1e-10]
        stable=[r for r in candidates if r['stability']=='stable']
        closest=min(candidates,key=lambda r:np.max(np.abs(np.array(r['state'])-original))) if candidates else None
        difference=float(np.max(np.abs(np.array(closest['state'])-original))) if closest else None
        qualified=[f for f in result['folds'] if f['parameter']==parameter and f['status']=='nondegenerate_local_fold_numerical']
        upper_present=any('upper' in r['nominal_reference_connection'].split('|') for r in stable)
        reached_connections=(closest['nominal_reference_connection'].split('|')
                             if closest is not None and difference < 1e-7 else [])
        lower_reached='lower' in reached_connections
        interpretation=('upper_branch_persists_basin_change_compatible' if lower_reached and upper_present else
                        'upper_branch_loss_supported_by_local_fold' if lower_reached and any(factor < f['factor'] < 1 and 'upper' in f['nominal_reference_connection'].split('|') for f in qualified) else
                        'upper_reference_reached' if 'upper' in reached_connections else 'unresolved')
        comparisons.append({'parameter':parameter,'factor':factor,'previous_r':float(row['r_baseline']),
                            'matched_root_r':None if closest is None else closest['r'],
                            'matched_root_max_state_difference':difference,'stable_roots_found':len(stable),
                            'upper_root_present':upper_present,'interpretation':interpretation})
    return comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=HERE/'configs/equilibrium_robustness_protocol.json')
    parser.add_argument('--output',type=Path,default=HERE/'recovery-research-results')
    args=parser.parse_args()
    if not args.output.is_dir():
        raise ValueError('Output directory must already exist')
    protocol=json.loads(args.protocol.read_text())
    coarse=run(protocol)
    fine=run(protocol,2)
    refinement=compare_refinement(coarse,fine)
    comparisons=previous_comparisons(fine)
    unresolved=[r for r in fine['diagnostics'] if r.get('termination') not in {None,'factor_boundary_crossed'} or 'fixed_factor_failure' in r]
    report={'status':'PASS' if refinement['status']=='PASS' and not unresolved and
            fine['parameter_derivative_max_error']<1e-8 and fine['analytic_jacobian_max_error']<1e-8 and
            fine['duplicate_sweep_crossing_max_difference']<1e-7 and
            all(f['status']=='nondegenerate_local_fold_numerical' for f in fine['folds']) else 'FAIL',
            'date':protocol['date'],'protocol_sha256':digest(args.protocol),'source_sha256':digest(Path(__file__)),
            'unchanged_model_sha256':digest(HERE/'model.py'),'unchanged_analytic_jacobian_sha256':digest(HERE/'regimes.py'),
            'input_sha256':{'simulation/'+name:digest(HERE/name) for name in
                            ['model.py','regimes.py','network.py','referee_experiments.py',
                             protocol['source_protocol'],'configs/illustrative_baseline.json',
                             'presentation-results/aligned.json','presentation-results/lower_aligned.json',
                             'presentation-results/neighborhood_sensitivity.csv']},
            'refinement':refinement,'primary':coarse,'refined':fine,'previous_relaxation_comparisons':comparisons,
            'scope':'Numerical local smooth-branch analysis of equilibria connected to the two specified nominal references; not a global proof of completeness, empirical uncertainty analysis, or clinical threshold.'}
    for name,rows in [('branches',coarse['points']),('refined-branches',fine['points']),
                      ('roots',fine['roots']),('folds',fine['folds']),('previous-relaxation',comparisons)]:
        export=[{k:v for k,v in row.items() if k!='state'} for row in rows]
        if export:
            write_csv(args.output/f'equilibrium-robustness-{name}.csv',export)
    (args.output/'equilibrium-robustness-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'refinement':refinement,'folds':fine['folds'],'previous':comparisons},indent=2),flush=True)
    if report['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    main()
