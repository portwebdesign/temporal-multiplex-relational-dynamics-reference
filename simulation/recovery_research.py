"""Equal-dose demand histories with unchanged seven-coordinate equations.

This is a separate, synthetic experiment.  Existing article records and nominal
coefficients are read, never rewritten.  The protocol precedes its new effects.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
from math import isclose, exp, isfinite
from pathlib import Path

from analysis import find_equilibrium
from model import STATE_NAMES, simulate
from network import aggregate_network
from referee_experiments import raw_scenarios, write_csv, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "configs/recovery_research_protocol.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def maximum_distance(left, right):
    return max(abs(a-b) for a, b in zip(left, right))


def state_from(row):
    return [row[name] for name in STATE_NAMES]


def validate_histories(protocol):
    """Require common high duration and last exposure end, not just row counts."""
    horizon = protocol["solver"]["time_horizon"]
    durations, ends = [], []
    for name, intervals in protocol["histories"].items():
        if not intervals:
            raise ValueError(f"empty history: {name}")
        previous = 0.0
        for start, end in intervals:
            if not 0 <= previous <= start < end <= horizon:
                raise ValueError(f"invalid history intervals: {name}")
            previous = end
        durations.append(sum(end-start for start,end in intervals))
        ends.append(intervals[-1][1])
    if any(not isclose(value,durations[0],abs_tol=1e-12) for value in durations):
        raise ValueError("histories must have identical total high-demand duration")
    if any(not isclose(value,ends[0],abs_tol=1e-12) for value in ends):
        raise ValueError("histories must share the final high-demand end")
    return durations[0]


def base_configuration(protocol, existing_protocol, direction):
    if direction not in ("aligned","opposed"):
        raise ValueError("unknown demand direction")
    cfg = raw_scenarios(existing_protocol)["aligned"]
    cfg["network"]["schedule"] = []
    cfg["inputs"]["schedule"] = []
    cfg["permeability_target"]["schedule"] = []
    cfg.pop("active_coordinates",None)
    for edge in cfg["network"]["baseline"]["demand_edges"]:
        edge["magnitude"] = protocol["baseline_mark"]
    cfg["network"]["baseline"]["demand_edges"][1]["direction"] = [-1.0 if direction == "opposed" else 1.0]
    cfg["solver"] = {"method":"classical_runge_kutta_4", "dt":protocol["solver"]["steps"][0],
                     "time_horizon":protocol["solver"]["time_horizon"]}
    cfg["metadata"] = {"version":protocol["version"],"direction":direction,
                       "coefficient_source":"unchanged existing nominal stationary construction"}
    return cfg


def history_configuration(base, protocol, history):
    cfg = deepcopy(base)
    high = deepcopy(cfg["network"]["baseline"])
    for edge in high["demand_edges"]:
        edge["magnitude"] = protocol["peak_mark"]
    cfg["network"]["schedule"] = [{"start":start,"end":end,"network":deepcopy(high)}
                                    for start,end in protocol["histories"][history]]
    cfg["metadata"]["history"] = history
    return cfg


def exposure_audit(configuration):
    """Exact integral for a two-level, piecewise-constant external program."""
    horizon = configuration["solver"]["time_horizon"]
    baseline = aggregate_network(configuration["network"]["baseline"])
    levels = [(item["end"]-item["start"],aggregate_network(item["network"]))
              for item in configuration["network"]["schedule"]]
    high_duration = sum(duration for duration,_ in levels)
    low_duration = horizon-high_duration
    result = {"high_duration":high_duration,"observation_duration":horizon,
              "last_high_end":max(item["end"] for item in configuration["network"]["schedule"])}
    # D equals total_demand here because both source edges lack within-source cancellation.
    for key in ("total_demand","external_demand","raw_conflict","conflict"):
        result[f"integral_{key}"] = low_duration*getattr(baseline,key)+sum(duration*getattr(level,key) for duration,level in levels)
        result[f"peak_{key}"] = max([getattr(baseline,key)]+[getattr(level,key) for _,level in levels])
    return result



def scalar_value(configuration, tau, time):
    """Exact stable scalar lag, a comparator for generic recency effects."""
    if not isfinite(tau) or tau <= 0 or not isfinite(time) or time < 0:
        raise ValueError("scalar time constant must be positive and time nonnegative")
    network = configuration["network"]
    basal = aggregate_network(network["baseline"]).external_demand
    value = basal
    boundaries = sorted({0.0,time,*[float(item[key]) for item in network["schedule"]
                                    for key in ("start","end") if 0 < item[key] < time]})
    for left,right in zip(boundaries,boundaries[1:]):
        midpoint = (left+right)/2
        demand = basal
        for item in network["schedule"]:
            if item["start"] <= midpoint < item["end"]:
                demand = aggregate_network(item["network"]).external_demand
                break
        value = demand+(value-demand)*exp(-(right-left)/tau)
    return value


def prepare_references(base, protocol, existing):
    settings = protocol["reference_preparation"]
    references = {}
    seeds = {}
    for name in settings["seed_names"]:
        result = find_equilibrium(base,existing["branch_seeds"][name],**{key:settings[key] for key in ("dt","tolerance","max_time","check_interval")})
        seeds[name] = asdict(result)
        same = next((key for key,value in references.items() if maximum_distance(result.state,value["state"])<1e-7),None)
        if same is None:
            direction = base["metadata"]["direction"]
            label = (f"aligned_{'upper' if name == 'interior_upper' else 'lower'}_reference"
                     if direction == "aligned" else f"opposed_reference_{len(references)+1}")
            references[label] = {**asdict(result),"source_seed_name":name}
    base["initial_state"] = list(seeds[settings["seed_names"][0]]["state"])
    return references,seeds


def relax(configuration, endpoint, protocol, dt):
    settings = protocol["post_history_relaxation"]
    try:
        result = find_equilibrium(configuration,endpoint,dt=dt,**{key:settings[key] for key in ("tolerance","max_time","check_interval")})
        return {"status":"converged",**asdict(result)}
    except (RuntimeError,ValueError) as error:
        return {"status":"unresolved","requested_dt":dt,"reason":str(error),"initial_state":list(endpoint)}


def classify(record, references, tolerance):
    if record["status"] != "converged":
        return record
    distances = {name:maximum_distance(record["state"],value["state"]) for name,value in references.items()}
    nearest = min(distances,key=distances.get)
    return {**record,"distances_to_same_field_references":distances,
            "branch":nearest if distances[nearest] <= tolerance else "other_stationary_state"}


def endpoint_metrics(rows, protocol):
    dt = rows[1]["time"]-rows[0]["time"]
    result = {}
    for time in protocol["outcome_times"]:
        row = rows[round(time/dt)]
        if not isclose(row["time"],time,abs_tol=1e-9):
            raise ValueError("outcome time not present on grid")
        for name in STATE_NAMES:
            result[f"{name}_t{time:g}"] = row[name]
    result["minimum_r"] = min(row["r"] for row in rows)
    result["maximum_w2"] = max(row["w2"] for row in rows)
    return result


def compare_terminals(records,tolerance):
    complete = {name:value for name,value in records.items() if value["status"] == "converged"}
    comparisons = []
    for left,right in itertools.combinations(complete,2):
        distance = maximum_distance(complete[left]["state"],complete[right]["state"])
        comparisons.append({"left":left,"right":right,"terminal_max_coordinate_distance":distance,
                            "distinct_at_fixed_tolerance":distance>tolerance})
    return {"all_converged":len(complete)==len(records),"pairwise":comparisons,
            "has_distinct_stationary_outcomes":any(row["distinct_at_fixed_tolerance"] for row in comparisons)}



def evaluate(configuration, protocol, references):
    """Run one full-state history on all declared grids and relaxation steps."""
    cfg = deepcopy(configuration)
    grids = []
    for dt in protocol["solver"]["steps"]:
        cfg["solver"]["dt"] = dt
        grids.append(simulate(cfg))
    errors = []
    for coarse,fine in zip(grids,grids[1:]):
        ratio = round((coarse[1]["time"]-coarse[0]["time"])/(fine[1]["time"]-fine[0]["time"]))
        errors.append(max(maximum_distance(state_from(a),state_from(b)) for a,b in zip(coarse,fine[::ratio])))
    convergence = {"coarse_medium_max_state_error":errors[0],"medium_fine_max_state_error":errors[1],
                   "passes_fixed_tolerance":max(errors)<=protocol["solver"]["state_convergence_tolerance"]}
    continuation = {str(dt):classify(relax(cfg,state_from(grids[-1][-1]),protocol,dt),references,
                                    protocol["post_history_relaxation"]["branch_distance_tolerance"])
                    for dt in protocol["post_history_relaxation"]["steps"]}
    completed = all(value["status"]=="converged" for value in continuation.values())
    step_distance = maximum_distance(*[value["state"] for value in continuation.values()]) if completed else None
    record = {"configuration":cfg,"exposure":exposure_audit(cfg),"metrics":endpoint_metrics(grids[-1],protocol),
              "grid_convergence":convergence,"continuations":continuation,"terminal_step_distance":step_distance}
    return record,grids[-1]


def duration_extension(report,protocol,output):
    amendment = protocol["exploratory_duration_extension"]
    summary = []
    origin = amendment["schedule_scaling_origin"]
    report["exploratory_duration_extension"] = {}
    for factor in amendment["scale_factors"]:
        scaled = deepcopy(protocol)
        scaled["histories"] = {name:[[origin+(start-origin)*factor,origin+(end-origin)*factor] for start,end in blocks]
                               for name,blocks in protocol["histories"].items()}
        last_end = max(blocks[-1][1] for blocks in scaled["histories"].values())
        horizon = last_end+amendment["recovery_window_after_last_high"]
        scaled["solver"]["time_horizon"] = horizon
        scaled["outcome_times"] = [last_end,horizon]
        validate_histories(scaled)
        for direction in protocol["directions"]:
            base = deepcopy(report["directions"][direction]["configuration"])
            base["solver"]["time_horizon"] = horizon
            references = report["directions"][direction]["references"]
            doses = []
            terminals = {}
            for history in scaled["histories"]:
                name = f"exploratory_scale_{factor:g}_{direction}_{history}"
                print(f"Exploratory duration check {name}",flush=True)
                cfg = history_configuration(base,scaled,history)
                record,trace = evaluate(cfg,scaled,references)
                record["design_status"] = amendment["status"]
                record["scale_factor"] = factor
                record["outcome_times"] = scaled["outcome_times"]
                chosen = list(record["continuations"].values())[-1]
                doses.append(record["exposure"])
                terminals[history] = chosen
                report["exploratory_duration_extension"][name] = record
                write_json(output/f"{name}.json",record)
                stride = round(scaled["solver"]["trace_output_step"]/scaled["solver"]["steps"][-1])
                write_csv(output/f"{name}.csv",trace[::stride])
                summary.append({"scenario":name,"status":"exploratory","scale_factor":factor,"direction":direction,"history":history,
                                "exposure_end":last_end,"followup_time":horizon,"high_duration":record["exposure"]["high_duration"],
                                "r_exposure_end":record["metrics"][f"r_t{last_end:g}"],"r_followup":record["metrics"][f"r_t{horizon:g}"],
                                "w2_followup":record["metrics"][f"w2_t{horizon:g}"],"terminal_status":chosen["status"],
                                "terminal_r":chosen.get("state",[None]*7)[2],"terminal_branch":chosen.get("branch"),
                                "terminal_residual":chosen.get("residual"),"additional_relaxation_time":chosen.get("time"),
                                "terminal_step_distance":record["terminal_step_distance"],**record["grid_convergence"]})
                write_json(output/"report.json",report)
            if not all(isclose(row[key],doses[0][key],rel_tol=1e-12,abs_tol=1e-10) for row in doses for key in doses[0]):
                raise ValueError("exploratory histories failed within-duration dose matching")
            report["directions"][direction][f"exploratory_scale_{factor:g}_terminal_comparison"] = compare_terminals(
                terminals,protocol["post_history_relaxation"]["branch_distance_tolerance"])
    write_csv(output/"exploratory_duration_summary.csv",summary)
    return summary


def render_results(report,output):
    """Render synthetic trajectories and a scalar recency comparator."""
    import os
    os.environ.setdefault("MPLCONFIGDIR","/tmp/math-life-matplotlib-cache")
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"one_block":"#9B392B","two_blocks":"#246CA5","four_blocks":"#23856D"}
    labels = {"one_block":"One block","two_blocks":"Two blocks","four_blocks":"Four blocks"}
    figure,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for history in report["protocol"]["histories"]:
        color = colors[history]
        for direction,ax in [("aligned",axes[0,1]),("opposed",axes[1,0])]:
            cfg = report["primary"][f"{direction}_{history}"]["configuration"]
            with (output/f"{direction}_{history}.csv").open() as handle:
                rows = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(handle)]
            ax.plot([row["time"] for row in rows],[row["r"] for row in rows],label=labels[history],color=color)
            if direction == "aligned":
                event_times = sorted({0.0,50.0,*[point for block in report["protocol"]["histories"][history] for point in block]})
                event_values = []
                for time in event_times:
                    network = cfg["network"]["baseline"]
                    for item in cfg["network"]["schedule"]:
                        if item["start"]<=time<item["end"]:
                            network = item["network"]
                            break
                    event_values.append(aggregate_network(network).external_demand)
                axes[0,0].step(event_times,event_values,where="post",label=labels[history],color=color,alpha=.8)
                times = [row["time"] for row in rows]
                axes[1,1].plot(times,[scalar_value(cfg,5,time) for time in times],label=labels[history],color=color)
    titles = ["A  Matched demand: 20 high units in every program","B  Resource, aligned sources: all return to 0.60",
              "C  Resource, opposed sources: all return to 0.357","D  Stable scalar lag (tau = 5): transient history effects"]
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,loc="left",fontsize=10)
        ax.set_xlabel("Synthetic model time")
        ax.grid(alpha=.2)
        ax.axvline(45,color="#777777",ls=":",lw=.8)
        ax.legend(frameon=False,fontsize=8)
    axes[0,0].set_ylabel("External demand")
    axes[0,1].set_ylabel("Resource coordinate r")
    axes[1,0].set_ylabel("Resource coordinate r")
    axes[1,1].set_ylabel("Scalar load coordinate z")
    figure.savefig(output/"matched_history_comparison.png",dpi=180)
    plt.close(figure)


def run(protocol_path, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_histories(protocol)
    existing = json.loads((ROOT/protocol["existing_protocol"]).read_text())
    output = Path(output).resolve()
    protected = [ROOT/name for name in ("presentation-results","referee-results","dynamics-results","revision-results")]
    if any(path == output or path in output.parents for path in protected):
        raise ValueError("existing numerical output directories are protected")
    output.mkdir(parents=True,exist_ok=True)
    report = {"version":protocol["version"],"protocol_sha256":sha256(protocol_path),"protocol":protocol,
              "fingerprints":{path.name:sha256(path) for path in [Path(__file__),ROOT/"model.py",ROOT/"network.py",ROOT/"analysis.py",ROOT/"regimes.py",ROOT/"referee_experiments.py",ROOT/protocol["existing_protocol"]]},
              "directions":{},"primary":{},"feedback_subsets":{},"scalar_comparator":{},"interpretation":protocol["interpretation"]}
    primary_rows,convergence_rows,feedback_rows,scalar_rows = [],[],[],[]
    for direction in protocol["directions"]:
        print(f"Preparing unchanged-coefficient {direction} references",flush=True)
        base = base_configuration(protocol,existing,direction)
        references,seeds = prepare_references(base,protocol,existing)
        report["directions"][direction] = {"configuration":deepcopy(base),"references":references,"seed_audit":seeds}
        direction_dose = []
        terminal_records = {}
        for history in protocol["histories"]:
            name = f"{direction}_{history}"
            print(f"Computing three grids and stationary continuation: {name}",flush=True)
            cfg = history_configuration(base,protocol,history)
            dose = exposure_audit(cfg)
            direction_dose.append(dose)
            scalar = []
            for tau in protocol["scalar_comparator"]["time_constants"]:
                scalar.append({"scenario":name,"direction":direction,"history":history,"tau":tau,
                               **{f"z_t{time:g}":scalar_value(cfg,tau,time) for time in protocol["outcome_times"]},
                               "terminal_z":aggregate_network(cfg["network"]["baseline"]).external_demand})
            scalar_rows.extend(scalar)
            report["scalar_comparator"][name] = scalar
            grids = []
            for dt in protocol["solver"]["steps"]:
                cfg["solver"]["dt"] = dt
                grids.append(simulate(cfg))
            finest = grids[-1]
            grid_errors = []
            for coarse,fine in zip(grids,grids[1:]):
                ratio = round((coarse[1]["time"]-coarse[0]["time"])/(fine[1]["time"]-fine[0]["time"]))
                error = max(maximum_distance(state_from(a),state_from(b)) for a,b in zip(coarse,fine[::ratio]))
                grid_errors.append(error)
            convergence = {"scenario":name,"coarse_medium_max_state_error":grid_errors[0],"medium_fine_max_state_error":grid_errors[1],
                           "passes_fixed_tolerance":max(grid_errors)<=protocol["solver"]["state_convergence_tolerance"]}
            convergence_rows.append(convergence)
            continuation = {}
            for dt in protocol["post_history_relaxation"]["steps"]:
                continuation[str(dt)] = classify(relax(cfg,state_from(finest[-1]),protocol,dt),references,
                                                   protocol["post_history_relaxation"]["branch_distance_tolerance"])
            chosen = continuation[str(protocol["post_history_relaxation"]["steps"][-1])]
            terminal_records[history] = chosen
            completed = all(value["status"]=="converged" for value in continuation.values())
            step_distance = maximum_distance(*[value["state"] for value in continuation.values()]) if completed else None
            record = {"configuration":deepcopy(cfg),"exposure":dose,"metrics":endpoint_metrics(finest,protocol),
                      "grid_convergence":convergence,"continuations":continuation,"terminal_step_distance":step_distance}
            report["primary"][name] = record
            write_json(output/f"{name}.json",record)
            stride = round(protocol["solver"]["trace_output_step"]/protocol["solver"]["steps"][-1])
            write_csv(output/f"{name}.csv",finest[::stride])
            primary_rows.append({"scenario":name,"direction":direction,"history":history,**record["metrics"],
                                 **dose,"terminal_status":chosen["status"],"terminal_r":chosen.get("state",[None]*7)[2],
                                 "terminal_residual":chosen.get("residual"),"terminal_full_residual":chosen.get("full_residual"),
                                 "additional_relaxation_time":chosen.get("time"),"terminal_branch":chosen.get("branch"),
                                 "terminal_step_distance":step_distance})
            write_json(output/"report.json",report)
        fields = list(direction_dose[0])
        dose_match = all(isclose(record[field],direction_dose[0][field],abs_tol=1e-11,rel_tol=1e-12)
                         for record in direction_dose[1:] for field in fields)
        if not dose_match:
            raise ValueError("declared histories failed exact piecewise dose matching")
        report["directions"][direction]["exposure_match"] = dose_match
        report["directions"][direction]["primary_terminal_comparison"] = compare_terminals(terminal_records,protocol["post_history_relaxation"]["branch_distance_tolerance"])
        optional = protocol["feedback_subsets"]["optional_active"]
        for mask in range(2**len(optional)):
            selected = [name for index,name in enumerate(optional) if mask & (1<<index)]
            active = [name for name in STATE_NAMES if name in selected+protocol["feedback_subsets"]["always_active"]]
            subset_name = "+".join(selected) or "none"
            key = f"{direction}:{subset_name}"
            print(f"Checking frozen-background subset {key}",flush=True)
            terminal_records = {}
            subset_records = {}
            for history in protocol["histories"]:
                cfg = history_configuration(base,protocol,history)
                cfg["active_coordinates"] = active
                cfg["solver"]["dt"] = protocol["feedback_subsets"]["step"]
                rows = simulate(cfg)
                terminal = relax(cfg,state_from(rows[-1]),protocol,protocol["feedback_subsets"]["relaxation_step"])
                terminal_records[history] = terminal
                metrics = endpoint_metrics(rows,protocol)
                subset_records[history] = {"metrics":metrics,"endpoint_state":state_from(rows[-1]),"continuation":terminal}
                feedback_rows.append({"direction":direction,"active_background":subset_name,"background_count":len(selected),
                                      "active_coordinates":"+".join(active),"history":history,**metrics,
                                      "terminal_status":terminal["status"],"terminal_r":terminal.get("state",[None]*7)[2],
                                      "terminal_active_residual":terminal.get("residual"),"terminal_full_residual":terminal.get("full_residual"),
                                      "additional_relaxation_time":terminal.get("time")})
            summary = compare_terminals(terminal_records,protocol["post_history_relaxation"]["branch_distance_tolerance"])
            report["feedback_subsets"][key] = {"direction":direction,"optional_active":selected,"active_coordinates":active,
                                               "frozen_reference":list(base["initial_state"]),"histories":subset_records,
                                               "terminal_comparison":summary}
            write_json(output/"report.json",report)
    exploratory_rows = duration_extension(report,protocol,output)
    report["verification"] = {"all_primary_grid_checks_pass":all(row["passes_fixed_tolerance"] for row in convergence_rows),
                              "all_primary_terminal_step_checks_pass":all(row["terminal_step_distance"] is not None and row["terminal_step_distance"]<=protocol["post_history_relaxation"]["branch_distance_tolerance"] for row in primary_rows),
                              "all_exploratory_grid_checks_pass":all(row["passes_fixed_tolerance"] for row in exploratory_rows),
                              "all_exploratory_terminal_step_checks_pass":all(row["terminal_step_distance"] is not None and row["terminal_step_distance"]<=protocol["post_history_relaxation"]["branch_distance_tolerance"] for row in exploratory_rows),
                              "exploratory_scenarios":len(exploratory_rows),
                              "primary_scenarios":len(primary_rows),"feedback_scenarios":len(feedback_rows),
                              "feedback_subsets":len(report["feedback_subsets"])}
    write_csv(output/"scalar_comparator.csv",scalar_rows)
    write_csv(output/"primary_summary.csv",primary_rows)
    write_csv(output/"grid_convergence.csv",convergence_rows)
    write_csv(output/"feedback_summary.csv",feedback_rows)
    write_json(output/"report.json",report)
    manifest = {str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name!="manifest.json"}
    write_json(output/"manifest.json",manifest)
    print(json.dumps(report["verification"],indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument("--output",type=Path,default=ROOT/"recovery-research-results")
    parser.add_argument("--render",action="store_true")
    args = parser.parse_args()
    report = run(args.protocol,args.output)
    if args.render:
        render_results(report,args.output.resolve())
        write_json(args.output/"manifest.json",{str(path.relative_to(args.output)):sha256(path) for path in sorted(args.output.rglob("*")) if path.is_file() and path.name!="manifest.json"})


if __name__ == "__main__":
    main()
