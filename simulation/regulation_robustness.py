"""Finite-cost/time robustness map for the unchanged seven-state equations.

The protocol was written after the original factorial, before this grid. Every
program and every grid point is retained. New files use the regulation-robustness- prefix;
no previous output, core equation or existing protocol is overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from model import STATE_NAMES, derivative, simulate
from network import aggregate_network
from referee_experiments import Experiment, raw_scenarios
from regimes import analytic_jacobian
from regulation_research import factorial_subsets, point, read_json, subset_label, write_csv

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / 'configs/regulation_robustness_protocol.json'
DEFAULT_OUTPUT = ROOT / 'regulation-research-results'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rank_groups(values, tolerance=1e-8, descending=True):
    """Tied groups each have full spread <= tolerance; no transitive chaining."""
    if tolerance < 0 or not values or not all(np.isfinite(v) for v in values.values()):
        raise ValueError('finite values and nonnegative tolerance required')
    ordered = sorted(values, key=lambda label: ((-1 if descending else 1)*values[label], label))
    groups=[]
    for label in ordered:
        if not groups or abs(values[label] - values[groups[-1][0]]) > tolerance:
            groups.append([label])
        else:
            groups[-1].append(label)
    return groups


def sign_class(value, tolerance):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def bracketed_crossings(times, values, evaluate, tolerance=1e-8):
    """Report all observed strict sign reversals, with numerical ties separate.

    This is not an exhaustive search for tangencies or multiple crossings inside
    one grid interval. The initial time10 endpoint is never called a crossing.
    """
    times=np.asarray(times, float)
    values=np.asarray(values, float)
    if times.shape != values.shape or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('strictly increasing equally sized time/value arrays required')
    records=[]
    ties=[]
    previous=None
    for index,(time,value) in enumerate(zip(times,values)):
        sign=sign_class(float(value),tolerance)
        if not sign:
            ties.append(float(time))
            continue
        if previous is not None and sign != previous[2]:
            left,right=float(times[previous[0]]),float(time)
            root=brentq(evaluate,left,right,xtol=1e-8,rtol=1e-12)
            records.append({'bracket_left':left,'bracket_right':right,'time':float(root),
                            'contrast_at_root':float(evaluate(root)),
                            'direction':'D+R becomes higher' if sign>0 else 'all-three becomes higher'})
        previous=(index,float(value),sign)
    return records,ties


def program_configuration(prepared, subset, protocol):
    cfg=deepcopy(prepared)
    cfg['solver']['time_horizon']=protocol['horizon']
    cfg['inputs']['schedule']=[]
    if subset:
        start,end=protocol['pulse_interval']
        cfg['inputs']['schedule']=[{'start':start,'end':end,
                                  'values':{name:protocol['strong_inputs'][name] for name in subset}}]
    return cfg


def event_solutions(configuration, protocol, tolerances):
    """Use fixed interval forcings and carry exact endpoint states between them."""
    start,end=protocol['pulse_interval']
    boundaries=[0.,float(start),float(end),float(protocol['horizon'])]
    baseline_signals=aggregate_network(configuration['network']['baseline'])
    pulse_signals=aggregate_network(configuration['network']['schedule'][0]['network'])
    baseline_input=configuration['inputs']['baseline']
    pulse_input=deepcopy(baseline_input)
    if configuration['inputs']['schedule']:
        pulse_input.update(configuration['inputs']['schedule'][0]['values'])
    state=np.asarray(configuration['initial_state'],float)
    records=[]
    for index,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        inputs=pulse_input if index==1 else baseline_input
        signals=pulse_signals if index==1 else baseline_signals
        def rhs(time,value):
            return derivative(value,inputs,configuration['parameters'],signals,
                              configuration['permeability_target']['baseline'])
        result=solve_ivp(rhs,(left,right),state,method='DOP853',dense_output=True,
                         rtol=tolerances['rtol'],atol=tolerances['atol'],
                         max_step=protocol['integration']['max_step'])
        if not result.success:
            raise RuntimeError(result.message)
        if not np.isfinite(result.y).all() or np.min(result.y)<0 or np.max(result.y)>1:
            raise RuntimeError('Integrator state left the declared domain')
        records.append((left,right,result))
        state=result.y[:,-1]
    def evaluate(time):
        if time<boundaries[0] or time>boundaries[-1]:
            raise ValueError('time outside computed horizon')
        for left,right,result in records:
            if left<=time<=right:
                return result.sol(time)
        raise ValueError('uncovered time')
    return evaluate,{'function_evaluations':sum(item[2].nfev for item in records),
                     'event_intervals':[[item[0],item[1]] for item in records]}


def reference_audit(all_rows, original_path):
    import csv
    with Path(original_path).open() as stream:
        original=list(csv.DictReader(stream))
    checks=[]
    for subset in [subset_label(s) for s in factorial_subsets(['delay','boundary','review'])]:
        prior=next(row for row in original if row['family']=='input_factorial'
                   and float(row['cost'])==.2 and row['subset']==subset)
        for time in [10.,20.]:
            now=next(row for row in all_rows if row['regime']=='cost_0.2'
                     and row['program']==subset and row['time']==time)
            for coordinate in ['r','w1','w2']:
                error=abs(now[coordinate]-float(prior[f'{coordinate}_{int(time)}']))
                checks.append({'program':subset,'time':time,'coordinate':coordinate,
                               'absolute_error':error})
    maximum=max(row['absolute_error'] for row in checks)
    if maximum>1e-7:
        raise RuntimeError('New solver failed immutable original factorial endpoints')
    return {'comparisons':len(checks),'maximum_absolute_difference':maximum,'checks':checks}


def run(protocol_path=DEFAULT_PROTOCOL, output=DEFAULT_OUTPUT):
    protocol_path,output=Path(protocol_path).resolve(),Path(output).resolve()
    if not output.is_dir():
        raise ValueError('Output directory must already exist')
    if output!=DEFAULT_OUTPUT.resolve():
        # Clean verification may use an existing temporary extraction directory.
        if output.name in {'presentation-results','referee-results','dynamics-results','revision-results'}:
            raise ValueError('Historical output directory is protected')
    protocol=read_json(protocol_path)
    previous={p.name:digest(p) for p in output.iterdir() if p.is_file() and not p.name.startswith('regulation-robustness-')}
    source=read_json(ROOT/protocol['source_protocol'])
    raw=raw_scenarios(source)['opposed_tradeoffs']
    experiment=Experiment(source)
    specs=[{'label':f'cost_{cost:g}','family':'cost_grid','cost':cost,'parameter':'none','multiplier':1.}
           for cost in protocol['input_costs']]
    neighborhood=protocol['parameter_neighborhood']
    specs += [{'label':f"{parameter}_{factor:g}",'family':'one_at_a_time',
               'cost':neighborhood['cost'],'parameter':parameter,'multiplier':factor}
              for parameter in neighborhood['parameters'] for factor in neighborhood['multipliers']]
    settings=protocol['observation_times']
    count=round((settings['end']-settings['start'])/settings['step'])
    times=np.linspace(settings['start'],settings['end'],count+1)
    subsets=factorial_subsets(protocol['input_order'])
    rows=[]; ranks=[]; crossings=[]; refinement=[]; references=[]; failures=[]
    configs_for_rk4={}
    max_error=0.
    tolerance=protocol['ranking']['absolute_tie_tolerance']
    for index,spec in enumerate(specs,1):
        print(f"Initializing robustness regime {index}/{len(specs)}: {spec['label']}",flush=True)
        cfg=deepcopy(raw)
        cfg['parameters'].update(protocol['counter_parameters'])
        cfg['parameters']['k_input']=spec['cost']
        if spec['parameter']!='none':
            cfg['parameters'][spec['parameter']]*=spec['multiplier']
        try:
            prepared=experiment.initialize(cfg)
            baseline=experiment.baselines[prepared['metadata']['baseline_id']]
            eig=np.linalg.eigvals(np.asarray(analytic_jacobian(
                prepared['initial_state'],prepared['inputs']['baseline'],prepared['parameters'],
                aggregate_network(prepared['network']['baseline']),
                prepared['permeability_target']['baseline'])))
            references.append({**spec,'state':prepared['initial_state'],'residual':baseline['residual'],
                               'original_seed':baseline['branch_seed'],
                               'relaxation_time':baseline['time'],
                               'parameters':prepared['parameters'],
                               'eigenvalues':[[float(z.real),float(z.imag)] for z in eig],
                               'spectral_abscissa':float(max(z.real for z in eig))})
        except Exception as error:
            failures.append({'regime':spec['label'],'stage':'initialization','error':str(error)})
            continue
        evaluators={}; coarse_evaluators={}; regime_rows=[]
        for subset in subsets:
            label=subset_label(subset)
            program=program_configuration(prepared,subset,protocol)
            try:
                coarse,c_record=event_solutions(program,protocol,protocol['integration']['tolerances'][0])
                fine,f_record=event_solutions(program,protocol,protocol['integration']['tolerances'][1])
                fine_values=np.asarray([fine(time) for time in times])
                coarse_values=np.asarray([coarse(time) for time in times])
                error=float(np.max(np.abs(fine_values-coarse_values)))
                max_error=max(max_error,error)
                refinement.append({'regime':spec['label'],'program':label,'maximum_sampled_state_difference':error,
                                   'coarse_function_evaluations':c_record['function_evaluations'],
                                   'fine_function_evaluations':f_record['function_evaluations']})
                evaluators[label]=fine; coarse_evaluators[label]=coarse
                for time,value in zip(times,fine_values):
                    regime_rows.append({'regime':spec['label'],'family':spec['family'],'cost':spec['cost'],
                                        'parameter':spec['parameter'],'multiplier':spec['multiplier'],
                                        'program':label,'time':float(time),
                                        **{name:float(value[i]) for i,name in enumerate(STATE_NAMES)}})
                if spec['label']=='cost_0.2' and subset in [('delay','review'),('delay','boundary','review')]:
                    configs_for_rk4[label]=program
            except Exception as error:
                failures.append({'regime':spec['label'],'program':label,'stage':'integration','error':str(error)})
        rows.extend(regime_rows)
        if len(evaluators)!=8:
            continue
        dr='delay_review'; combined='delay_boundary_review'
        def contrast(time):
            return float(evaluators[dr](time)[2]-evaluators[combined](time)[2])
        def coarse_contrast(time):
            return float(coarse_evaluators[dr](time)[2]-coarse_evaluators[combined](time)[2])
        roots,ties=bracketed_crossings(times,[contrast(t) for t in times],contrast,tolerance)
        coarse_roots,coarse_ties=bracketed_crossings(times,[coarse_contrast(t) for t in times],coarse_contrast,tolerance)
        if len(roots)!=len(coarse_roots):
            failures.append({'regime':spec['label'],'stage':'crossing_refinement','error':'Crossing counts differ between tolerances'})
        for a,b in zip(roots,coarse_roots):
            a.update({'coarse_time':b['time'],'two_tolerance_time_difference':abs(a['time']-b['time'])})
        crossings.append({**spec,'strict_crossings':roots,'grid_tie_times':ties,
                          'contrast_time10':contrast(10.),'contrast_time20':contrast(20.),
                          'contrast_time60':contrast(60.),
                          'search_scope':'Strict sign changes on a half-unit grid; no completeness claim for tangencies or multiple crossings within one interval.'})
        for time in times:
            current={row['program']:row for row in regime_rows if row['time']==time}
            resources={name:row['r'] for name,row in current.items()}
            loads={name:row['w2'] for name,row in current.items()}
            resource_groups=rank_groups(resources,tolerance,True)
            load_groups=rank_groups(loads,tolerance,False)
            for row in current.values():
                row['resource_delta_from_baseline']=row['r']-current['none']['r']
                row['load_delta_from_baseline']=row['w2']-current['none']['w2']
                row['resource_rank']=next(i+1 for i,group in enumerate(resource_groups) if row['program'] in group)
                row['load_rank']=next(i+1 for i,group in enumerate(load_groups) if row['program'] in group)
            ranks.append({'regime':spec['label'],'family':spec['family'],'cost':spec['cost'],
                          'time':float(time),'resource_groups':';'.join('+'.join(group) for group in resource_groups),
                          'load_groups':';'.join('+'.join(group) for group in load_groups),
                          'D_R_minus_all3_resource':contrast(float(time)),
                          'D_R_vs_all3_sign':sign_class(contrast(float(time)),tolerance)})
        print(f"Completed {spec['label']}: eight programs, max tolerance difference {max_error:.3g}",flush=True)
    rk4=[]
    for label,cfg in configs_for_rk4.items():
        print(f'Independent long-horizon RK4 refinement: {label}',flush=True)
        traces=[]
        for step in protocol['integration']['reference_rk4_steps']:
            comparison=deepcopy(cfg);comparison['solver']['dt']=step
            trace=simulate(comparison)
            traces.append(np.asarray([[point(trace,float(time))[name] for name in STATE_NAMES] for time in times]))
        observed=np.asarray([[next(row for row in rows if row['regime']=='cost_0.2' and row['program']==label
                                  and row['time']==time)[name] for name in STATE_NAMES] for time in times])
        rk4.append({'program':label,'steps':protocol['integration']['reference_rk4_steps'],
                    'coarse_medium_maximum_state_difference':float(np.max(np.abs(traces[0]-traces[1]))),
                    'medium_fine_maximum_state_difference':float(np.max(np.abs(traces[1]-traces[2]))),
                    'dop853_fine_rk4_maximum_state_difference':float(np.max(np.abs(observed-traces[2])))})
    original=reference_audit(rows,ROOT/'regulation-research-results/summary.csv')
    if len(rows)!=len(specs)*8*len(times):
        failures.append({'stage':'inventory','error':f'Expected {len(specs)*8*len(times)} rows, found {len(rows)}'})
    if max_error>protocol['integration']['maximum_state_disagreement']:
        failures.append({'stage':'tolerance_refinement','error':'State disagreement exceeds prespecified bound'})
    if max_error*20>tolerance:
        failures.append({'stage':'tie_band','error':'Prespecified tie band is less than twenty times the observed integration disagreement'})
    if any(item['dop853_fine_rk4_maximum_state_difference']>protocol['integration']['maximum_state_disagreement'] for item in rk4):
        failures.append({'stage':'rk4_refinement','error':'Independent RK4 disagreement exceeds bound'})
    if previous!={p.name:digest(p) for p in output.iterdir() if p.is_file() and not p.name.startswith('regulation-robustness-')}:
        raise RuntimeError('A previous output file changed')
    report={'status':'FAIL' if failures else 'PASS','version':protocol['version'],'protocol_sha256':digest(protocol_path),
            'sources_sha256':{str(path.relative_to(ROOT)):digest(path) for path in [Path(__file__).resolve(),protocol_path,
                  ROOT/protocol['source_protocol'],ROOT/'model.py',ROOT/'network.py',ROOT/'regimes.py',
                  ROOT/'analysis.py',ROOT/'referee_experiments.py',ROOT/'regulation_research.py']},
            'regimes':len(specs),'programs':len(specs)*8,'observation_times':len(times),'sample_rows':len(rows),
            'initialization_rule':protocol['initialization'],'references':references,
            'original_factorial_audit':original,'maximum_two_tolerance_state_difference':max_error,
            'tie_tolerance':tolerance,'tie_to_integration_error_ratio':tolerance/max_error if max_error else None,
            'rk4_audit':rk4,'failures':failures,'previous_outputs_preserved':len(previous),
            'crossing_scope':'Observed strict sign changes only; no exhaustive tangency or all-real-time crossing claim.',
            'interpretation':protocol['interpretation']}
    write_csv(output/'regulation-robustness-trajectories.csv',rows)
    write_csv(output/'regulation-robustness-rankings.csv',ranks)
    write_csv(output/'regulation-robustness-refinement.csv',refinement)
    write_record(output/'regulation-robustness-crossings.json',crossings)
    write_record(output/'regulation-robustness-report.json',report)
    print(json.dumps({k:report[k] for k in ['status','regimes','programs','observation_times','sample_rows',
          'maximum_two_tolerance_state_difference','tie_to_integration_error_ratio','failures']},indent=2))
    if failures:
        raise RuntimeError('Robustness checks failed; all computed outcomes retained')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.protocol,args.output)


if __name__=='__main__':
    main()
