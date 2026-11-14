"""Reproduce the declared branch-aware mechanism comparisons.

Every coefficient, forcing schedule, initialization rule and output is declared
in configs/referee_protocol.json. Numerical values are synthetic and
uncalibrated. No parameter is optimized to produce a preferred trajectory.
"""
from __future__ import annotations

import argparse
import ast
import csv
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from math import isclose, ceil, floor, log10, exp
import os
from pathlib import Path
import shutil
import subprocess

from analysis import cycle_statistics, deviation_index, find_equilibrium
from model import STATE_NAMES, simulate
from network import aggregate_network, network_at_time
from regimes import construct_interior_parameters, local_stability_record

ROOT = Path(__file__).resolve().parent
VERSION = "0.4.0"
FIGURE_SOURCE_DATE_EPOCH = 1794614400


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot emit a table without rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def state_from(row):
    return tuple(row[name] for name in STATE_NAMES)


def at_time(rows, time):
    step = rows[1]["time"] - rows[0]["time"]
    index = round(time / step)
    if index >= len(rows) or not isclose(rows[index]["time"], time, abs_tol=1e-10):
        raise ValueError("declared output time is outside the trace grid")
    return rows[index]


def baseline_config(configuration):
    base = deepcopy(configuration)
    base["network"]["schedule"] = []
    base["inputs"]["schedule"] = []
    base["permeability_target"]["schedule"] = []
    base.pop("active_coordinates", None)
    base.pop("metadata", None)
    return base


def construction_record(protocol):
    baseline = json.loads((ROOT / protocol["base_configuration"]).read_text())
    baseline["parameters"].update(protocol["bridge_parameters"])
    return construct_interior_parameters(baseline["parameters"], baseline["inputs"]["baseline"],
                                         protocol["network"]["baseline"], protocol["baseline_construction"]["target"])


def raw_scenarios(protocol):
    base = json.loads((ROOT / protocol["base_configuration"]).read_text())
    base["solver"] = deepcopy(protocol["solver"])
    base["parameters"].update(protocol["bridge_parameters"])
    original_parameters = deepcopy(base["parameters"])
    constructed = construction_record(protocol)
    base["parameters"] = deepcopy(constructed["parameters"])
    base["network"] = deepcopy(protocol["network"])
    base["permeability_target"] = deepcopy(protocol["permeability_target"])
    scenarios = {}
    for name, options in protocol["comparisons"].items():
        cfg = deepcopy(base)
        regime = options.get("regime", "interior_upper")
        cfg["metadata"] = {"name": name, "version": VERSION, "regime": regime,
                           "baseline_seed": list(protocol["branch_seeds"][regime]),
                           "status": "Synthetic branch-aware mechanism comparison."}
        if regime == "saturated":
            cfg["parameters"] = deepcopy(original_parameters)
        cfg["parameters"].update(options.get("parameters", {}))
        if options.get("opposed", True):
            cfg["network"]["schedule"][0]["network"]["demand_edges"][1]["direction"] = [-1.0]
        if options.get("inputs"):
            cfg["inputs"]["schedule"] = deepcopy(protocol["input_pulse"])
        if options.get("moderation"):
            cfg["network"]["schedule"][0]["network"]["moderation"] = list(options["moderation"])
        if options.get("target_pulse", False):
            cfg["permeability_target"]["schedule"] = deepcopy(protocol["target_intervention"])
        if options.get("reduced"):
            cfg["active_coordinates"] = ["r", "w2"]
        scenarios[name] = cfg
    return scenarios


class Experiment:
    def __init__(self, protocol):
        self.protocol = protocol
        self.baselines = {}
        self.trace_cache = {}
        self.convergence_cache = {}

    def initialize(self, configuration):
        cfg = deepcopy(configuration)
        baseline = baseline_config(cfg)
        seed = list(cfg.get("metadata", {}).get("baseline_seed", self.protocol["initialization"]["seed"]))
        baseline["initial_state"] = seed
        key = signature({**{name: baseline[name] for name in ("parameters", "inputs", "network", "permeability_target")},
                         "branch_seed": seed})
        if key not in self.baselines:
            settings = self.protocol["initialization"]
            result = find_equilibrium(baseline, initial_state=seed, dt=settings["dt"],
                                      tolerance=settings["tolerance"], max_time=settings["max_time"],
                                      check_interval=settings["check_interval"])
            record = asdict(result)
            record["configuration"] = baseline
            record["branch_seed"] = seed
            record["local_stability"] = local_stability_record(record["state"], baseline["inputs"]["baseline"],
                                                               baseline["parameters"], aggregate_network(baseline["network"]["baseline"]),
                                                               baseline["permeability_target"]["baseline"])
            self.baselines[key] = record
        cfg["initial_state"] = list(self.baselines[key]["state"])
        cfg["metadata"]["baseline_id"] = key
        return cfg

    def trace(self, configuration, factor=1):
        cfg = deepcopy(configuration)
        cfg["solver"]["dt"] /= factor
        key = signature({name: value for name, value in cfg.items() if name != "metadata"})
        if key not in self.trace_cache:
            self.trace_cache[key] = simulate(cfg)
        return self.trace_cache[key]

    def convergence(self, configuration):
        key = signature({name: value for name, value in configuration.items() if name != "metadata"})
        if key not in self.convergence_cache:
            coarse = self.trace(configuration)
            medium = self.trace(configuration, 2)[::2]
            fine = self.trace(configuration, 4)[::4]
            first = max(abs(a[name] - b[name]) for a, b in zip(coarse, medium) for name in STATE_NAMES)
            second = max(abs(a[name] - b[name]) for a, b in zip(medium, fine) for name in STATE_NAMES)
            self.convergence_cache[key] = {"max_state_dt_vs_dt2": first, "max_state_dt2_vs_dt4": second,
                                           "ratio": first / second if second else None}
        return self.convergence_cache[key]


def matched_controls(configuration):
    context = deepcopy(configuration)
    baseline_edges = context["network"]["baseline"]["demand_edges"]
    for interval in context["network"]["schedule"]:
        for edge, original in zip(interval["network"]["demand_edges"], baseline_edges):
            edge["magnitude"] = original["magnitude"]
            edge["direction"] = list(original["direction"])
        interval.pop("end_network", None)
    context["metadata"]["name"] += "_context_control"
    context["metadata"]["control"] = "Declared input/target/filter pulse retained; demand vectors stay at baseline."
    no_pulse = deepcopy(configuration)
    no_pulse["network"]["schedule"] = []
    no_pulse["inputs"]["schedule"] = []
    no_pulse["permeability_target"]["schedule"] = []
    no_pulse["metadata"]["name"] += "_control"
    no_pulse["metadata"]["control"] = "All pulse schedules disabled at the same stationary initial state."
    return no_pulse, context


def summarize(name, rows, reference, common_reference, control=None):
    endpoint = at_time(rows, 10.0)
    final = at_time(rows, 20.0)
    values = [deviation_index(state_from(row), reference) for row in rows]
    result = {"scenario": name, "r_t10": endpoint["r"], "r_t20": final["r"],
              "w1_t20": final["w1"], "w2_t20": final["w2"], "min_r": min(row["r"] for row in rows),
              "V_t10": deviation_index(state_from(endpoint), reference), "V_t20": values[-1],
              "deltaV_1_t10": deviation_index(state_from(at_time(rows, 11)), reference) - deviation_index(state_from(endpoint), reference),
              "V_common_t10": deviation_index(state_from(endpoint), common_reference),
              "V_common_t20": deviation_index(state_from(final), common_reference),
              "pre_pulse_drift": max(abs(row[coordinate] - reference[index]) for row in rows if row["time"] < 5.0 for index, coordinate in enumerate(STATE_NAMES)),
              "r_contrast_t10": 0.0, "r_contrast_t20": 0.0, "max_abs_r_contrast": 0.0,
              "max_abs_r_contrast_time": 0.0, "signed_r_contrast_integral": 0.0, "absolute_r_contrast_integral": 0.0}
    result.update({"r_baseline": reference[2], "w2_baseline": reference[6],
                   "load_headroom": 1.0-reference[6], "min_r_contrast": 0.0,
                   "max_r_contrast": 0.0, "max_w2_contrast": 0.0, "max_abs_w2_contrast": 0.0})
    if control is not None:
        contrast = [row["r"] - baseline["r"] for row, baseline in zip(rows, control)]
        peak = max(range(len(rows)), key=lambda i: abs(contrast[i]))
        result.update({"r_contrast_t10": endpoint["r"] - at_time(control, 10)["r"],
                       "r_contrast_t20": final["r"] - at_time(control, 20)["r"],
                       "max_abs_r_contrast": abs(contrast[peak]), "max_abs_r_contrast_time": rows[peak]["time"],
                       "min_r_contrast": min(contrast), "max_r_contrast": max(contrast),
                       "max_w2_contrast": max(row["w2"]-base["w2"] for row,base in zip(rows,control)),
                       "max_abs_w2_contrast": max(abs(row["w2"]-base["w2"]) for row,base in zip(rows,control))})
        for left, right, dl, dr in zip(rows[:-1], rows[1:], contrast[:-1], contrast[1:]):
            width = right["time"] - left["time"]
            result["signed_r_contrast_integral"] += width * (dl + dr) / 2
            if dl * dr < 0:
                result["absolute_r_contrast_integral"] += width * (dl * dl + dr * dr) / (2 * (abs(dl) + abs(dr)))
            else:
                result["absolute_r_contrast_integral"] += width * (abs(dl) + abs(dr)) / 2
    return result


def add_outcomes(rows, reference, context):
    return [{**row, "V": deviation_index(state_from(row), reference),
             "r_demand_contrast": row["r"] - base["r"],
             "w2_demand_contrast": row["w2"] - base["w2"]} for row, base in zip(rows, context)]


def cycle_scenarios(protocol, raw_base):
    settings = protocol["cycle_experiment"]
    for direction in settings["directions"]:
        for duration in settings["phase_durations"]:
            cfg = deepcopy(raw_base)
            cfg["inputs"]["schedule"] = []
            cfg["permeability_target"] = {"baseline": settings["permeability_target"], "schedule": []}
            low = deepcopy(protocol["network"]["baseline"])
            if direction == "opposed":
                low["demand_edges"][1]["direction"] = [-1.0]
            high = deepcopy(low)
            for edge in low["demand_edges"]:
                edge["magnitude"] = settings["baseline_mark"]
            for edge in high["demand_edges"]:
                edge["magnitude"] = settings["peak_mark"]
            start = settings["start"]
            peak, end = start + duration, start + 2 * duration
            cfg["network"] = {"baseline": low, "schedule": [
                {"start": start, "end": peak, "network": low, "end_network": high},
                {"start": peak, "end": end, "network": high, "end_network": low}]}
            cfg["solver"]["time_horizon"] = end + settings["recovery_after_cycle"]
            name = f"cycle_{direction}_{duration:g}"
            cfg["metadata"] = {"name": name, "version": VERSION, "cycle_start": start,
                               "cycle_peak": peak, "cycle_end": end, "phase_duration": duration,
                               "regime": "cycle_"+direction,
                               "baseline_seed": list(protocol["baseline_construction"]["target"]),
                               "status": "Continuous triangular demand cycle; descriptive finite-rate path comparison."}
            yield name, cfg


def response_difference(full, core):
    difference = [a["r_demand_contrast"]-b["r_demand_contrast"] for a,b in zip(full,core)]
    peak = max(range(len(difference)),key=lambda i:abs(difference[i]))
    return {"t10":at_time(full,10)["r_demand_contrast"]-at_time(core,10)["r_demand_contrast"],
            "t20":difference[-1], "max_absolute":abs(difference[peak]),
            "time_of_maximum":full[peak]["time"],
            "interpretation":"Difference of within-model pulse-minus-context responses; evolving pathways remain bundled."}


def core_load_benchmark(configuration, rows):
    """Compare RK4 with exact load evolution in the frozen-background core."""
    if set(configuration.get("active_coordinates",[])) != {"r","w2"}:
        raise ValueError("exact core benchmark requires active r,w2 only")
    if configuration["inputs"]["schedule"] or configuration["permeability_target"]["schedule"]:
        raise ValueError("exact core benchmark requires fixed inputs and target")
    if any("end_network" in interval for interval in configuration["network"]["schedule"]):
        raise ValueError("exact core benchmark requires piecewise constant demand")
    s,e,_,c,p,_,w = configuration["initial_state"]
    par,u = configuration["parameters"],configuration["inputs"]["baseline"]
    loss = par["d_w2"]*s*p*u["boundary"]+par["h_w2"]*u["delay"]
    largest = abs(rows[0]["w2"]-w)
    boundaries = {float(interval[key]) for interval in configuration["network"]["schedule"] for key in ("start","end")}
    for previous,current in zip(rows,rows[1:]):
        points = [previous["time"],*sorted(t for t in boundaries if previous["time"]<t<current["time"]),current["time"]]
        for left,right in zip(points,points[1:]):
            signal = network_at_time(configuration["network"],.5*(left+right))
            gain = par["lambda_d"]*signal.external_demand+par["a_w2"]*c*(1-par["q_h"]*u["boundary"])+par["b_w2"]*e
            total = gain+loss
            if total:
                equilibrium = gain/total
                w = equilibrium+(w-equilibrium)*exp(-total*(right-left))
        largest = max(largest,abs(current["w2"]-w))
    return {"max_absolute_w2_error":largest, "grid_points":len(rows),
            "method":"piecewise exact scalar gain-loss solution", "frozen_coordinates":["s","e","c","p","w1"]}


def run_numerical_experiment(protocol_path, protocol, out):
    experiment = Experiment(protocol)
    raw = raw_scenarios(protocol)
    print(f"Preparing {len(raw)} branch-aware comparison baselines",flush=True)
    primary = {name:experiment.initialize(cfg) for name,cfg in raw.items()}
    common_reference = primary["opposed"]["initial_state"]
    scenarios = dict(primary)
    for name,cfg in primary.items():
        no_pulse,context = matched_controls(cfg)
        scenarios[name+"_control"] = no_pulse
        scenarios[name+"_context_control"] = context
    bare = {name:experiment.trace(cfg) for name,cfg in scenarios.items()}
    report = {"status":"Synthetic branch-aware mechanism comparison", "version":VERSION,
              "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "construction":construction_record(protocol), "scenarios":{},
              "primary_comparisons":list(primary), "convergence":{}, "baselines":{},
              "sensitivity":[], "sensitivity_failures":[], "neighborhood_sensitivity":[],
              "neighborhood_failures":[], "cycles":{}, "cycle_convergence":{},
              "scope":"Uncalibrated branch and parameter-regime comparison; all declared outcomes retained."}
    traces = {}
    print(f"Integrating and refining {len(scenarios)} comparison/control traces",flush=True)
    for name,cfg in scenarios.items():
        parent_name = name.removesuffix("_context_control").removesuffix("_control")
        context_rows = bare[parent_name+"_context_control"]
        reference = cfg["initial_state"]
        traces[name] = add_outcomes(bare[name],reference,context_rows)
        report["scenarios"][name] = summarize(name,bare[name],reference,common_reference,context_rows)
        report["convergence"][name] = experiment.convergence(cfg)
        report["scenarios"][name]["coordinate_spans"] = {coordinate:max(row[coordinate] for row in bare[name])-min(row[coordinate] for row in bare[name]) for coordinate in STATE_NAMES}
        report["scenarios"][name]["regime"] = cfg["metadata"]["regime"]
        report["scenarios"][name]["baseline_id"] = cfg["metadata"]["baseline_id"]
        write_json(out/f"{name}.json",cfg)
        write_csv(out/f"{name}.csv",traces[name])
    report["matched_ablation_max_state_difference"] = max(abs(a[key]-b[key]) for a,b in zip(bare["aligned_no_conflict"],bare["opposed_no_conflict"]) for key in STATE_NAMES)
    direction = [b["r"]-a["r"] for a,b in zip(bare["aligned"],bare["opposed"])]
    peak = max(range(len(direction)),key=lambda i:abs(direction[i]))
    report["direction_contrast"] = {"opposed_minus_aligned_t10":at_time(bare["opposed"],10)["r"]-at_time(bare["aligned"],10)["r"],
                                    "opposed_minus_aligned_t20":direction[-1], "max_absolute":abs(direction[peak]),
                                    "time_of_maximum":bare["opposed"][peak]["time"],
                                    "measurement_note":"Differences in normalized model coordinates."}
    report["core_full_demand_response"] = {}
    report["regime_comparisons"] = {}
    for regime,names in protocol["regime_comparisons"].items():
        cfg = primary[names["opposed"]]
        reference = cfg["initial_state"]
        item = {"regime":regime, "baseline_id":cfg["metadata"]["baseline_id"], "state":reference,
                "load_headroom":1-reference[6], "sp_product":reference[0]*reference[4],
                "local_load_demand_derivative":cfg["parameters"]["lambda_d"]*(1-reference[6])}
        for orientation in ("aligned","opposed"):
            row = report["scenarios"][names[orientation]]
            item[orientation] = {key:row[key] for key in ("r_t10","r_t20","max_abs_r_contrast","max_w2_contrast","V_t10")}
            key = regime+"_"+orientation
            contrast = response_difference(traces[names[orientation]],traces[names[orientation+"_core"]])
            report["core_full_demand_response"][key] = contrast
            item[orientation+"_full_minus_core"] = contrast
        report["regime_comparisons"][regime] = item
    report["core_load_benchmarks"] = {name:core_load_benchmark(cfg,bare[name]) for name,cfg in scenarios.items()
                                      if set(cfg.get("active_coordinates",[]))=={"r","w2"}
                                      and not cfg["inputs"]["schedule"] and not cfg["permeability_target"]["schedule"]}
    sensitivity_scenarios = {}
    neighborhood_scenarios = {}

    def sensitivity_condition(name,cfg,parameter,value,collection,failures):
        cfg["metadata"]["name"] = name
        saved = deepcopy(cfg)
        try:
            cfg = experiment.initialize(cfg)
            _,context = matched_controls(cfg)
            rows,context_rows = experiment.trace(cfg),experiment.trace(context)
            result = summarize(name,rows,cfg["initial_state"],common_reference,context_rows)
            result.update({"parameter":parameter,"value":value,"baseline_id":cfg["metadata"]["baseline_id"],
                           "status":"completed","baseline_state":cfg["initial_state"]})
            report["convergence"][name] = experiment.convergence(cfg)
            collection.append(result)
            return cfg
        except (RuntimeError,ValueError) as error:
            result = {"scenario":name,"parameter":parameter,"value":value,"status":"not_completed",
                      "reason":str(error),"stage":"stationarity or declared integration","seed":saved["metadata"]["baseline_seed"]}
            failures.append(result)
            collection.append(result)
            saved["metadata"]["failure"] = result
            return saved

    print("Evaluating declared pathway and normalization conditions",flush=True)
    for parameter,values in protocol["sensitivity_ranges"].items():
        for index,value in enumerate(values):
            parent = "opposed_tradeoffs_inputs" if parameter.startswith("k_") else "opposed"
            cfg = deepcopy(raw[parent])
            cfg["parameters"][parameter] = value
            name = f"sensitivity_{parameter}_{index}"
            sensitivity_scenarios[name] = sensitivity_condition(name,cfg,parameter,value,report["sensitivity"],report["sensitivity_failures"])
    for index,value in enumerate(protocol["normalization_sensitivity"]["values"]):
        cfg = deepcopy(raw["opposed"])
        cfg["network"]["baseline"]["psi_scale"] = value
        cfg["network"]["schedule"][0]["network"]["psi_scale"] = value
        name = f"sensitivity_psi_scale_{index}"
        sensitivity_scenarios[name] = sensitivity_condition(name,cfg,"psi_scale",value,report["sensitivity"],report["sensitivity_failures"])
    print("Evaluating twelve coefficient-neighborhood conditions",flush=True)
    for parameter in protocol["coefficient_neighborhood"]["parameters"]:
        for index,factor in enumerate(protocol["coefficient_neighborhood"]["factors"]):
            cfg = deepcopy(raw["opposed"])
            nominal_value = cfg["parameters"][parameter]
            cfg["parameters"][parameter] = value = nominal_value*factor
            name = f"neighborhood_{parameter}_{index}"
            record = sensitivity_condition(name,cfg,parameter,value,report["neighborhood_sensitivity"],report["neighborhood_failures"])
            report["neighborhood_sensitivity"][-1].update({"nominal_value":nominal_value,"factor":factor})
            neighborhood_scenarios[name] = record
    write_json(out/"sensitivity_scenarios.json",sensitivity_scenarios)
    write_json(out/"neighborhood_scenarios.json",neighborhood_scenarios)
    cycle_traces = {}
    cycle_configurations = {}
    print("Integrating and refining continuous cycles",flush=True)
    for name,cfg in cycle_scenarios(protocol,raw["opposed"]):
        cfg = experiment.initialize(cfg)
        cycle_configurations[name] = cfg
        rows = experiment.trace(cfg)
        start,end = cfg["metadata"]["cycle_start"],cfg["metadata"]["cycle_end"]
        branch_rows = [row for row in rows if start-1e-10<=row["time"]<=end+1e-10]
        metrics = cycle_statistics([row["external_demand"] for row in branch_rows],[row["r"] for row in branch_rows])
        base_signal = aggregate_network(cfg["network"]["baseline"])
        metrics.update({"scenario":name,"phase_duration":cfg["metadata"]["phase_duration"],
                        "direction":"opposed" if "opposed" in name else "aligned", "baseline_id":cfg["metadata"]["baseline_id"],
                        "baseline_r":cfg["initial_state"][2],"baseline_conflict":base_signal.conflict,
                        "baseline_state":cfg["initial_state"],
                        "V_cycle_end":deviation_index(state_from(at_time(rows,end)),cfg["initial_state"]),
                        "pre_cycle_drift":max(abs(row[key]-cfg["initial_state"][i]) for row in rows if row["time"]<start for i,key in enumerate(STATE_NAMES))})
        report["cycles"][name] = metrics
        report["cycle_convergence"][name] = cycle_refinement(experiment,cfg,rows)
        cycle_traces[name] = [{**row,"V":deviation_index(state_from(row),cfg["initial_state"])} for row in rows]
        write_json(out/f"{name}.json",cfg)
        write_csv(out/f"{name}.csv",cycle_traces[name])
    baseline = baseline_config(primary["opposed"])
    seed_audit = []
    print("Checking stationary seeds, relaxation steps and post-pulse branches",flush=True)
    for seed in protocol["equilibrium_audit_seeds"]:
        settings = protocol["initialization"]
        result = find_equilibrium(baseline,initial_state=seed,dt=settings["dt"],tolerance=settings["tolerance"],
                                  max_time=settings["max_time"],check_interval=settings["check_interval"])
        record = asdict(result)
        record["max_distance_from_upper_reference"] = max(abs(a-b) for a,b in zip(record["state"],common_reference))
        record["local_stability"] = local_stability_record(record["state"],baseline["inputs"]["baseline"],baseline["parameters"],
                                                          aggregate_network(baseline["network"]["baseline"]),baseline["permeability_target"]["baseline"])
        seed_audit.append(record)
    report["equilibrium_seed_audit"] = seed_audit
    step_audit = []
    settings = protocol["initialization"]
    for seed in protocol["relaxation_step_audit"]["seeds"]:
        results = [find_equilibrium(baseline, initial_state=seed, dt=step,
                                    tolerance=settings["tolerance"], max_time=settings["max_time"],
                                    check_interval=settings["check_interval"])
                   for step in protocol["relaxation_step_audit"]["steps"]]
        distance = max(abs(a-b) for a,b in zip(results[0].state,results[1].state))
        if distance > protocol["relaxation_step_audit"]["same_branch_distance_tolerance"]:
            raise ValueError("baseline relaxation steps identify inconsistent stationary branches")
        step_audit.append({"seed":seed,"steps":protocol["relaxation_step_audit"]["steps"],
                           "records":[asdict(result) for result in results],"max_state_difference":distance})
    report["relaxation_step_audit"] = step_audit
    upper_cfg,lower_cfg = primary["opposed"],primary["lower_opposed"]
    if upper_cfg["parameters"] != lower_cfg["parameters"]:
        raise ValueError("constructed branch comparison must hold every coefficient fixed")
    if upper_cfg["metadata"]["baseline_id"] == lower_cfg["metadata"]["baseline_id"]:
        raise ValueError("branch-aware baseline identifiers were merged")
    if max(abs(a-b) for a,b in zip(upper_cfg["initial_state"],lower_cfg["initial_state"])) < 1e-6:
        raise ValueError("declared stationary branches were not distinct")
    post_records = {}
    for name in protocol["post_pulse_relaxation"]["comparisons"]:
        cfg = primary[name]
        base = baseline_config(cfg)
        endpoint_state = list(state_from(at_time(bare[name],protocol["post_pulse_relaxation"]["start_at"])))
        try:
            result = find_equilibrium(base,initial_state=endpoint_state,dt=settings["dt"],
                                      tolerance=settings["tolerance"],max_time=settings["max_time"],
                                      check_interval=settings["check_interval"])
            record = asdict(result)
            references = ({"saturation":primary["saturated_opposed"]["initial_state"]}
                          if cfg["metadata"]["regime"] == "saturated" else
                          {"upper":upper_cfg["initial_state"],"lower":lower_cfg["initial_state"]})
            distances = {label:max(abs(a-b) for a,b in zip(result.state,reference)) for label,reference in references.items()}
            closest = min(distances,key=distances.get)
            record.update({"status":"completed","observation_end_time":20,
                           "additional_relaxation_time":result.time,"distances_to_references":distances,
                           "classified_branch":closest if distances[closest] <= protocol["post_pulse_relaxation"]["branch_distance_tolerance"] else "other_stationary_state"})
        except (RuntimeError,ValueError) as error:
            record = {"status":"not_completed","reason":str(error),"initial_state":endpoint_state,
                      "observation_end_time":20,"classified_branch":"unresolved"}
        post_records[name] = record
    report["post_pulse_relaxation"] = post_records
    write_csv(out/"post_pulse_relaxation.csv",[{"scenario":name,"status":record["status"],
                                               "classified_branch":record["classified_branch"],
                                               "additional_relaxation_time":record.get("additional_relaxation_time"),
                                               "residual":record.get("full_residual"),
                                               "final_state":record.get("state"),
                                               "distances_to_references":record.get("distances_to_references"),
                                               "reason":record.get("reason","")}
                                              for name,record in post_records.items()])
    post_cycle = {}
    for name,cfg in cycle_configurations.items():
        start_time = cfg["metadata"]["cycle_end"]
        endpoint_state = list(state_from(at_time(cycle_traces[name],start_time)))
        base = baseline_config(cfg)
        try:
            result = find_equilibrium(base,initial_state=endpoint_state,dt=settings["dt"],
                                      tolerance=settings["tolerance"],max_time=settings["max_time"],
                                      check_interval=settings["check_interval"])
            references = ({"upper":upper_cfg["initial_state"],"lower":lower_cfg["initial_state"]}
                          if report["cycles"][name]["direction"] == "aligned" else
                          {"cycle_reference":cfg["initial_state"]})
            distances = {label:max(abs(a-b) for a,b in zip(result.state,reference)) for label,reference in references.items()}
            closest = min(distances,key=distances.get)
            record = {**asdict(result),"status":"completed","cycle_end_time":start_time,
                      "additional_relaxation_time":result.time,"distances_to_references":distances,
                      "classified_branch":closest if distances[closest]<=protocol["post_cycle_relaxation"]["branch_distance_tolerance"] else "other_stationary_state"}
        except (RuntimeError,ValueError) as error:
            record = {"status":"not_completed","reason":str(error),"initial_state":endpoint_state,
                      "cycle_end_time":start_time,"classified_branch":"unresolved"}
        post_cycle[name] = record
    report["post_cycle_relaxation"] = post_cycle
    write_csv(out/"post_cycle_relaxation.csv",[{"scenario":name,"status":record["status"],
                                               "classified_branch":record["classified_branch"],
                                               "cycle_end_time":record["cycle_end_time"],
                                               "additional_relaxation_time":record.get("additional_relaxation_time"),
                                               "residual":record.get("full_residual"),"final_state":record.get("state"),
                                               "distances_to_references":record.get("distances_to_references"),
                                               "reason":record.get("reason","")}
                                              for name,record in post_cycle.items()])
    local_records = []
    audit = protocol["local_return_audit"]
    branch_configs = {"interior_upper":upper_cfg,"interior_lower":lower_cfg}
    for branch in audit["branches"]:
        cfg = branch_configs[branch]
        base = baseline_config(cfg)
        reference = cfg["initial_state"]
        for direction_index,direction in enumerate(audit["directions"]):
            for sign in audit["signs"]:
                seed = [value+sign*audit["amplitude"]*component for value,component in zip(reference,direction)]
                record = {"branch":branch,"direction_index":direction_index,"direction":direction,
                          "sign":sign,"amplitude":audit["amplitude"],"initial_state":seed}
                try:
                    result = find_equilibrium(base,initial_state=seed,dt=settings["dt"],
                                              tolerance=settings["tolerance"],max_time=settings["max_time"],
                                              check_interval=settings["check_interval"])
                    distance = max(abs(a-b) for a,b in zip(result.state,reference))
                    record.update({"status":"completed","state":list(result.state),"residual":result.full_residual,
                                   "additional_relaxation_time":result.time,"steps":result.steps,
                                   "step_reductions":result.step_reductions,"distance_to_own_reference":distance,
                                   "return_status":"returned" if distance<=audit["return_distance_tolerance"] else "other_stationary_state"})
                except (RuntimeError,ValueError) as error:
                    record.update({"status":"not_completed","return_status":"unresolved","reason":str(error)})
                local_records.append(record)
    report["local_return_audit"] = local_records
    write_csv(out/"local_return_audit.csv",local_records)
    report["baselines"] = experiment.baselines
    report["primary_coordinate_spans"] = {name:report["scenarios"][name]["coordinate_spans"] for name in ("aligned","opposed","opposed_target_shift")}
    report["network_examples"] = {"baseline":asdict(aggregate_network(primary["opposed"]["network"]["baseline"])),
                                   "opposed_pulse":asdict(aggregate_network(primary["opposed"]["network"]["schedule"][0]["network"])),
                                   "aligned_pulse":asdict(aggregate_network(primary["aligned"]["network"]["schedule"][0]["network"]))}
    summaries = [{key:value for key,value in report["scenarios"][name].items() if key!="coordinate_spans"} for name in scenarios]
    write_csv(out/"summary.csv",summaries)
    write_csv(out/"sensitivity.csv",report["sensitivity"])
    write_csv(out/"neighborhood_sensitivity.csv",report["neighborhood_sensitivity"])
    write_csv(out/"regime_comparisons.csv",[{"regime":name,"r_baseline":row["state"][2],"w2_baseline":row["state"][6],
                                            "sp_product":row["sp_product"],"load_headroom":row["load_headroom"],
                                            "local_load_demand_derivative":row["local_load_demand_derivative"],
                                            "aligned_max_abs_r_contrast":row["aligned"]["max_abs_r_contrast"],
                                            "opposed_max_abs_r_contrast":row["opposed"]["max_abs_r_contrast"],
                                            "aligned_max_w2_contrast":row["aligned"]["max_w2_contrast"],
                                            "opposed_full_minus_core_t10":row["opposed_full_minus_core"]["t10"]}
                                           for name,row in report["regime_comparisons"].items()])
    write_csv(out/"cycle_statistics.csv",list(report["cycles"].values()))
    report["directional_sensitivity"] = matched_direction_sensitivity(report,sensitivity_scenarios,primary["aligned"])
    write_csv(out/"direction_sensitivity.csv",report["directional_sensitivity"])
    report["implementation_sha256"] = implementation_hashes()
    report["numerical_files"] = sorted([f"{name}{suffix}" for name in scenarios for suffix in (".json",".csv")]
                                       +[f"{name}{suffix}" for name in cycle_traces for suffix in (".json",".csv")]
                                       +["summary.csv","sensitivity.csv","sensitivity_scenarios.json","cycle_statistics.csv","direction_sensitivity.csv",
                                         "neighborhood_sensitivity.csv","neighborhood_scenarios.json","regime_comparisons.csv",
                                         "post_pulse_relaxation.csv","post_cycle_relaxation.csv","local_return_audit.csv","report.json"])
    write_json(out/"report.json",report)
    return report,scenarios,traces,cycle_traces


def nice_max(value):
    magnitude=10.0**floor(log10(value))
    return next(scale*magnitude for scale in (1,2,5,10) if scale*magnitude>=value)


def cycle_refinement(experiment, configuration, coarse=None):
    traces=[experiment.trace(configuration) if coarse is None else coarse,
            experiment.trace(configuration,2),experiment.trace(configuration,4)]
    metadata=configuration["metadata"]
    summaries=[]
    for rows in traces:
        branch=[row for row in rows if metadata["cycle_start"]-1e-10<=row["time"]<=metadata["cycle_end"]+1e-10]
        summaries.append(cycle_statistics([row["external_demand"] for row in branch],[row["r"] for row in branch]))
    result=experiment.convergence(configuration).copy()
    result["path_statistics_by_step"]={str(configuration["solver"]["dt"]/factor):summary for factor,summary in zip((1,2,4),summaries)}
    for key in ("signed","absolute","output_gap"):
        first=abs(summaries[0][key]-summaries[1][key])
        second=abs(summaries[1][key]-summaries[2][key])
        result[key+"_dt_vs_dt2"]=first
        result[key+"_dt2_vs_dt4"]=second
        result[key+"_refinement_ratio"]=first/second if second else None
    return result


def draw_figure(path, panels, height=18.5):
    """Draw declared scientific series with explicit linear axes and no overlays."""
    lines = [r"\documentclass[11pt]{article}",
             rf"\usepackage[paperwidth=18cm,paperheight={height:g}cm,margin=0cm]{{geometry}}",
             r"\usepackage{tikz,lmodern,amsmath}", r"\pagestyle{empty}", r"\begin{document}",
             r"\noindent\begin{tikzpicture}[x=1cm,y=1cm,font=\rmfamily\scriptsize]",
             rf"\path[use as bounding box] (0,0) rectangle (18,{height:g});"]
    rows = (len(panels) + 1) // 2
    for index, panel in enumerate(panels):
        x0, y0 = 1.5 + (index % 2) * 8.7, height - 5.2 - (index // 2) * 6.05
        width, plot_height = 7.2, 3.55
        xmin, xmax = panel["xlim"]
        ymin, ymax = panel["ylim"]
        lines += [rf"\begin{{scope}}[shift={{({x0},{y0})}}]",
                  rf"\node[anchor=west,font=\rmfamily\bfseries\small] at (-.55,{plot_height+1.1}) {{{panel['title']}}};"]
        for value in panel.get("yticks", [ymin + (ymax-ymin)*j/4 for j in range(5)]):
            fraction = (value-ymin)/(ymax-ymin)
            tick = f"{value:g}"
            lines += [rf"\draw[gray!20] (0,{fraction*plot_height}) -- ({width},{fraction*plot_height});",
                      rf"\node[anchor=east] at (-.1,{fraction*plot_height}) {{{tick}}};"]
        for value in panel.get("xticks", [xmin + (xmax-xmin)*j/4 for j in range(5)]):
            fraction = (value-xmin)/(xmax-xmin)
            lines += [rf"\draw[gray!20] ({fraction*width},0) -- ({fraction*width},{plot_height});",
                      rf"\node[anchor=north] at ({fraction*width},-.08) {{{value:g}}};"]
        lines += [rf"\draw[gray!60] (0,0) rectangle ({width},{plot_height});",
                  rf"\node[rotate=90] at (-1.3,{plot_height/2}) {{{panel['ylabel']}}};",
                  rf"\node at ({width/2},-.55) {{{panel['xlabel']}}};"]
        if ymin < 0 < ymax:
            zero = (0-ymin)/(ymax-ymin)*plot_height
            lines.append(rf"\draw[gray!65,dotted] (0,{zero}) -- ({width},{zero});")
        legend = 0
        for j, series in enumerate(panel["series"]):
            points, color, label, style = series
            coordinates = " ".join(f"({(x-xmin)/(xmax-xmin)*width:.6f},{(y-ymin)/(ymax-ymin)*plot_height:.6f})" for x,y in points)
            lines.append(rf"\draw[{color},line width=.8pt{style}] plot coordinates {{{coordinates}}};")
            if label is None:
                continue
            ly = plot_height + .72 - .27*legend
            legend += 1
            lines += [rf"\draw[{color},line width=.8pt{style}] (0,{ly}) -- (.4,{ly});",
                      rf"\node[anchor=west] at (.48,{ly}) {{{label}}};"]
        lines.append(r"\end{scope}")
    lines += [r"\end{tikzpicture}", r"\end{document}"]
    path.write_text("\n".join(lines)+"\n")


def sampled_points(rows, key, *, jump=False, stride=5):
    selected = [row for index,row in enumerate(rows) if index%stride==0 or any(abs(row["time"]-value)<.025 for value in (5,10,15))]
    points=[]
    previous=None
    for row in selected:
        if jump and previous is not None and row[key] != previous[key]:
            points.append((row["time"],previous[key]))
        points.append((row["time"],row[key]))
        previous=row
    return points


def figure_tex(traces, report, path):
    blue,orange,green = "blue!75!black","orange!85!black","green!50!black"
    series = lambda name,key,color,label,style="":(sampled_points(traces[name],key),color,label,style)
    contrast = [(b["time"],b["r"]-a["r"]) for a,b in zip(traces["aligned"],traces["opposed"])][::5]
    extent = nice_max(max(abs(value) for _,value in contrast)*1.2 or .01)
    response_extent = nice_max(max(abs(row["r_demand_contrast"]) for name in ("opposed","opposed_reduced") for row in traces[name])*1.2 or .01)
    panels = [
        {"title":"A. Demand pulse", "xlabel":"Time (dimensionless)","ylabel":"Signal", "xlim":(0,20),"ylim":(0,1),"series":[
            (sampled_points(traces["opposed"],"external_demand",jump=True),blue,"External demand",""),
            (sampled_points(traces["opposed"],"conflict",jump=True),orange,"Opposed conflict","")]},
        {"title":"B. Resource response", "xlabel":"Time (dimensionless)","ylabel":r"Resource reserve $r$","xlim":(0,20),"ylim":(0,1),"series":[
            series("aligned","r",blue,"Aligned"),series("opposed","r",orange,"Opposed"),series("opposed_control","r",green,"Baseline control",",dashed")]},
        {"title":"C. Directional contrast", "xlabel":"Time (dimensionless)","ylabel":r"$r_{\mathrm{opp}}-r_{\mathrm{aligned}}$","xlim":(0,20),"ylim":(-extent,extent),"series":[
            (contrast,orange,"Opposed minus aligned","")]},
        {"title":"D. Experienced-load response", "xlabel":"Time (dimensionless)","ylabel":r"Experienced load $w_2$","xlim":(0,20),"ylim":(0,1),"series":[
            series("aligned","w2",blue,"Aligned"),series("opposed","w2",orange,"Opposed"),series("opposed_control","w2",green,"Baseline control",",dashed")]},
        {"title":"E. Full and core responses", "xlabel":"Time (dimensionless)","ylabel":r"Demand response $\delta r$","xlim":(0,20),"ylim":(-response_extent,response_extent),"series":[
            series("opposed","r_demand_contrast",orange,"Full model"),series("opposed_reduced","r_demand_contrast",blue,"Two-state core")]},
        {"title":"F. Separate target intervention", "xlabel":"Time (dimensionless)","ylabel":r"Permeability $c$","xlim":(0,20),"ylim":(0,1),"series":[
            series("opposed_target_shift","c",orange,r"Intervention $c$"),
            (sampled_points(traces["opposed_target_shift"],"c_target",jump=True),blue,r"Target $c_{\mathrm{tar}}$",",dashed"),
            series("opposed","c",green,"Primary fixed target",",dotted")]},
    ]
    draw_figure(path,panels)


def cycle_figure_tex(cycles, traces, report, path):
    colors = ["blue!75!black","orange!85!black","green!50!black"]
    selected = sorted((name for name in cycles if "opposed" in name),key=lambda name:report["cycles"][name]["phase_duration"])
    maximum_time = max(cycles[name][-1]["time"] for name in selected)
    path_series,signals = [],[]
    for name,color in zip(selected,colors):
        rows = cycles[name]
        phase = report["cycles"][name]["phase_duration"]
        start,peak,end = 5,5+phase,5+2*phase
        signals.append((sampled_points(rows,"external_demand",stride=10),color,rf"Half-cycle ${phase:g}$",""))
        up = [(row["external_demand"],row["r"]) for row in rows if start-1e-10<=row["time"]<=peak+1e-10][::5]
        down = [(row["external_demand"],row["r"]) for row in rows if peak-1e-10<=row["time"]<=end+1e-10][::5]
        path_series += [(up,color,rf"${phase:g}$ rising",""),(down,color,None,",dashed")]
    values = [value for points,_,_,_ in path_series for _,value in points]
    ylow = max(0,floor(min(values)*100)/100-.01)
    yhigh = min(1,ceil(max(values)*100)/100+.01)
    vmax = nice_max(max(row["V"] for name in ("aligned","opposed") for row in traces[name])*1.15 or .001)
    area_series = []
    for direction,color in zip(("aligned","opposed"),colors):
        points = sorted((row["phase_duration"],row["absolute"]) for row in report["cycles"].values() if row["direction"]==direction)
        area_series.append((points,color,direction.capitalize(),""))
    amax = nice_max(max(value for points,_,_,_ in area_series for _,value in points)*1.15 or .001)
    panels = [
        {"title":"A. Continuous demand cycles","xlabel":"Time (dimensionless)","ylabel":r"External demand $d_{\mathrm{ext}}$","xlim":(0,maximum_time),"ylim":(0,1),"series":signals},
        {"title":"B. Opposed input--output paths","xlabel":r"External demand $d_{\mathrm{ext}}$","ylabel":r"Resource reserve $r$","xlim":(.25,.72),"ylim":(ylow,yhigh),"xticks":[.3,.4,.5,.6,.7],"yticks":[.25,.30,.35],"series":path_series},
        {"title":"C. Pulse deviation from baseline","xlabel":"Time (dimensionless)","ylabel":r"Equal-weight $V$","xlim":(0,20),"ylim":(0,vmax),"yticks":[0,.01,.02,.03,.04,.05],"series":[
            (sampled_points(traces["aligned"],"V"),colors[0],"Aligned",""),(sampled_points(traces["opposed"],"V"),colors[1],"Opposed","")]},
        {"title":"D. Absolute branch separation","xlabel":"Half-cycle duration","ylabel":r"$\mathcal H_{\mathrm{abs}}$ (model units)","xlim":(0,22),"ylim":(0,amax),"xticks":[5,10,20],"yticks":[0,.01,.02,.03,.04,.05],"series":area_series},
    ]
    draw_figure(path,panels,height=12.45)


def vector_tex(values, precision=4):
    return "["+",".join(f"{float(value):.{precision}f}" for value in values)+r"]^{\top}"


def exact_example(network):
    rational=lambda value:Fraction(str(value))
    q=[Fraction(0)]*3;z=[Fraction(0)]*3
    for edge in network["demand_edges"]:
        k=edge["layer"]
        amount=rational(edge["weight"])*rational(edge["magnitude"])/rational(network["layer_scales"][k])
        q[k]+=amount;z[k]+=amount*rational(edge["direction"][0])
    sigma=network.get("moderation",[0]*3)
    h=[1+sum((1-rational(sigma[l]))*rational(network["coupling"][l][k]) for l in range(3)) for k in range(3)]
    demand=sum(h[k]*q[k] for k in range(3));forces=[h[k]*z[k] for k in range(3)]
    raw=sum(abs(value) for value in forces)**2-sum(forces)**2
    return demand/(1+demand),raw/(rational(network["psi_scale"])+raw)


def fraction_tex(value):
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}\approx {float(value):.2f}"


def results_tex(report, path, scenarios):
    records = report["scenarios"]
    nominal = scenarios["opposed"]
    reference = nominal["initial_state"]
    lower,_ = exact_example(nominal["network"]["baseline"])
    higher,conflict = exact_example(nominal["network"]["schedule"][0]["network"])
    direction = report["direction_contrast"]
    regime_labels = {"interior_upper":"Constructed upper branch","interior_lower":"Constructed lower branch","saturated":"Saturation regime"}
    lines = [r"% Generated from the declared branch-aware protocol and numerical report.",
             r"The primary experiment starts from the specified interior stationary reference",
             rf"\[\bm x^\star_{{\mathrm{{upper}}}}={vector_tex(reference,2)}.\]",
             r"Six gain--loss coefficients are obtained from the stationary equations at this reference. The remaining coefficients retain their declared values. The same constructed parameter set also has a lower attracting stationary branch reached from the lower starting seeds. A separately specified coefficient set supplies a saturation regime. Table~\ref{tab:regime-baselines} records their distinct initial states; permeability is $0.45$ in all three.",
             r"\begin{table}[H]\centering\footnotesize",
             r"\begin{tabularx}{\textwidth}{@{}Yrrrrrr@{}}\toprule",
             r"Stationary regime & $s$ & $e$ & $r$ & $p$ & $w_1$ & $w_2$ \\\midrule"]
    for regime,row in report["regime_comparisons"].items():
        values = [row["state"][i] for i in (0,1,2,4,5,6)]
        lines.append(" & ".join([regime_labels[regime]]+[f"{value:.4f}" for value in values])+r" \\")
    lines += [r"\bottomrule\end{tabularx}",
              r"\caption{Stationary references used for the declared parameter and branch comparisons. The two constructed branches share every coefficient; the saturation regime uses a different coefficient set.}\label{tab:regime-baselines}\end{table}",
              rf"During $5\leq t<10$, demand marks in layers 2 and 3 rise from $0.15$ to $1$. Source-preserving transmission gives $d_{{\mathrm{{ext}}}}={fraction_tex(lower)}$ before the pulse and ${fraction_tex(higher)}$ during it. Opposed demands give $\chi={fraction_tex(conflict)}$, while aligned demands give zero directional conflict. The baseline inputs are $\bm u_0={vector_tex([nominal['inputs']['baseline'][key] for key in ('delay','boundary','review')],2)}$.",
              r"The primary direction, pathway-removal, and full--core comparisons keep $c_{\mathrm{tar}}=0.45$ and these inputs fixed. Permeability therefore remains at its stationary target. A separate intervention shifts the target to $0.35$ during the opposed pulse and then restores it. Its context control retains that target schedule while demand stays at baseline. The two-state core fixes all background coordinates at the same stationary state as its paired full model.",
              r"We report the resource response $\delta r(t)=r_{\mathrm{pulse}}(t)-r_{\mathrm{context}}(t)$ and the corresponding load response $\delta w_2(t)$. For the fixed-context comparisons, the context control is the stationary baseline. Each deviation index $V$ uses its own full-system reference and equal weights.",
              r"\begin{table}[H]\centering\footnotesize",
              r"\begin{tabularx}{\textwidth}{@{}Yrrrrrr@{}}\toprule",
              r"Declared comparison & $r^\star$ & $r(10)$ & $r(20)$ & $\max_t|\delta r(t)|$ & $\max_t\delta w_2(t)$ & $10^3 V(10)$ \\\midrule"]
    labels = {"aligned":"Aligned","opposed":"Opposed","opposed_no_conflict":"Conflict path off",
              "opposed_no_external":"Direct demand gain off","opposed_no_support":"Support gain off",
              "opposed_reduced":"Two-state core","opposed_filtered":"Upstream filtering",
              "opposed_tradeoffs":"Counter-mechanism","opposed_inputs":"Stronger inputs",
              "opposed_tradeoffs_inputs":"Counter, stronger inputs","opposed_target_shift":"Boundary target shift"}
    for name,label in labels.items():
        row = records[name]
        values = [row["r_baseline"],row["r_t10"],row["r_t20"],row["max_abs_r_contrast"],row["max_w2_contrast"],1000*row["V_t10"]]
        lines.append(" & ".join([label]+[f"{value:.4f}" for value in values])+r" \\")
    lines += [r"\bottomrule\end{tabularx}",
              r"\caption{Synthetic pulse responses. $r^\star$ is each condition's stationary pre-pulse reference; maxima cover $0\leq t\leq20$. The response $\delta r=r_{\mathrm{pulse}}-r_{\mathrm{context}}$ uses a stationary context in the fixed-context arms and a moving matched context when inputs, filtering, or the target are scheduled. Structural parameter changes have separately equilibrated backgrounds.}\label{tab:synthetic-results}\end{table}"]
    direct_off,support_off = records["opposed_no_external"],records["opposed_no_support"]
    input_row,input_context = records["opposed_inputs"],records["opposed_inputs_context_control"]
    lines.append(rf"Removing the direct demand gain sets $\lambda_d=0$ while retaining the directional-conflict channel, and gives $r^\star={direct_off['r_baseline']:.4f}$; removing the support gain gives $r^\star={support_off['r_baseline']:.4f}$. At a matched state with unsaturated pressure, the conflict-slope difference contains $-\eta_r r a_{{\mathrm{{conflict}}}}\Delta\chi$ and therefore scales with the current resource level. Finite-window peaks additionally depend on the other reference coordinates and their subsequent evolution.")
    lines.append(rf"The stronger-input comparison illustrates the moving context: it starts at $r^\star={input_row['r_baseline']:.4f}$ and reaches $r(10)={input_row['r_t10']:.4f}$, while its matched input-only context reaches $r_{{\mathrm{{context}}}}(10)={input_context['r_t10']:.4f}$. Their difference is $\delta r(10)={input_row['r_contrast_t10']:.4f}$.")
    aligned,opposed = records["aligned"],records["opposed"]
    full_core = report["core_full_demand_response"]["interior_upper_opposed"]
    lines.append(rf"In the upper-branch comparison, aligned demand produces a peak load response of ${aligned['max_w2_contrast']:.4f}$ and a maximum resource departure of ${aligned['max_abs_r_contrast']:.4f}$. Opposed demand produces a maximum resource departure of ${opposed['max_abs_r_contrast']:.4f}$. The opposed-minus-aligned resource contrast is ${direction['opposed_minus_aligned_t10']:.4f}$ at $t=10$ and ${direction['opposed_minus_aligned_t20']:.4f}$ at $t=20$. Its largest magnitude is ${direction['max_absolute']:.4f}$ at $t={direction['time_of_maximum']:.2f}$. Removing the conflict pathway makes the matched directional trajectories coincide.")
    lines.append(rf"The full-minus-core difference in opposed demand responses is ${full_core['t10']:.4f}$ at $t=10$ and ${full_core['t20']:.4f}$ at $t=20$. The comparison permits several coordinates to evolve together; it measures their combined consequence. For the main opposed pulse, $V(10)={opposed['V_t10']:.4f}$ and $V(20)={opposed['V_t20']:.4f}$. Complete coordinate spans, response timing, integrals, and one-unit changes in $V$ are supplied with the experiment.")
    lines += [r"\begin{table}[H]\centering\footnotesize",
              r"\begin{tabularx}{\textwidth}{@{}Yrrrrr@{}}\toprule",
              r"Stationary regime & $1-w_2^\star$ & $\lambda_d(1-w_2^\star)$ & \shortstack{Aligned\\$\max_t|\delta r(t)|$} & \shortstack{Opposed\\$\max_t|\delta r(t)|$} & Full--core $\Delta_{10}$ \\\midrule"]
    for regime,row in report["regime_comparisons"].items():
        values = [row["load_headroom"],row["local_load_demand_derivative"],row["aligned"]["max_abs_r_contrast"],row["opposed"]["max_abs_r_contrast"],row["opposed_full_minus_core"]["t10"]]
        lines.append(" & ".join([regime_labels[regime]]+[f"{value:.4f}" for value in values])+r" \\")
    lines += [r"\bottomrule\end{tabularx}",
              r"\caption{Branch and parameter-regime dependence. $\Delta_{10}$ is the full-minus-core difference of matched opposed-demand responses at $t=10$. The local derivative isolates the load-headroom factor; differences between parameter regimes also reflect their other coefficients.}\label{tab:regime-responses}\end{table}"]
    completed = [row for row in report["neighborhood_sensitivity"] if row["status"]=="completed"]
    failed = len(report["neighborhood_failures"])
    if completed:
        neighborhood_min = min(row["baseline_state"][2] for row in completed)
        neighborhood_max = max(row["baseline_state"][2] for row in completed)
        lower_cases = [row for row in completed if row["factor"]<1 and row["parameter"] in ("a_s","a_e","rho_p","h_w2")]
        paragraph = rf"The pathway analysis retains the declared coefficient and normalization ranges. In a separate neighborhood analysis, each of the six constructed coefficients is perturbed by $\pm5\%$ without rebalancing the others. Stationary resource references range from ${neighborhood_min:.4f}$ to ${neighborhood_max:.4f}$."
        if len(lower_cases)==4:
            lower_min = min(row["baseline_state"][2] for row in lower_cases)
            lower_max = max(row["baseline_state"][2] for row in lower_cases)
            paragraph += rf" The four $5\%$ reductions in $a_s$, $a_e$, $\rho_p$, and $h_{{w2}}$ settle at lower resource reference states, with $r^\star$ between ${lower_min:.4f}$ and ${lower_max:.4f}$."
        paragraph += " The pulse responses therefore depend on these altered backgrounds as well as on the changed coefficient."
        lines.append(paragraph)
    else:
        lines.append(r"The coefficient-neighborhood analysis retains every declared condition and its stationarity failure; pulse outcomes are supplied only after a stationary reference has been obtained.")
    if report["sensitivity_failures"]:
        parameter_names = {"a_conflict":r"a_{\mathrm{conflict}}","lambda_d":r"\lambda_d","lambda_h":r"\lambda_h",
                           "k_input":r"k_{\mathrm{input}}","k_isolation":r"k_{\mathrm{isolation}}",
                           "k_visibility":r"k_{\mathrm{visibility}}","k_access":r"k_{\mathrm{access}}","psi_scale":r"\Psi_0"}
        conditions = ", ".join("$"+parameter_names[row["parameter"]]+f"={row['value']:g}$" for row in report["sensitivity_failures"])
        lines.append("The pathway conditions "+conditions+" remain unresolved within the declared stationary-relaxation budget; their settings and residual failures are retained with the condition table.")
    main_returns = [report["post_pulse_relaxation"][name] for name in ("aligned","opposed","opposed_target_shift")]
    if all(row["status"]=="completed" and row["classified_branch"]=="upper" for row in main_returns):
        lines.append(r"Subsequent basal-forcing relaxation returns all three main pulse outcomes to the upper stationary reference. Their remaining departures at $t=20$ occur during the finite observation window; the separate return probes record the eventual stationary outcomes.")
    cycle_initial = {}
    for row in report["cycles"].values():
        cycle_initial.setdefault(row["direction"],row)
    ca,co = cycle_initial["aligned"],cycle_initial["opposed"]
    lines.append(rf"Continuous triangular demand marks supply matched rising and falling demand branches. For each fixed direction, marks rise from $0.15$ to $1$ and return over half-cycle durations of $5$, $10$, or $20$. Aligned cycles start at $r^\star={ca['baseline_r']:.4f}$ with $\chi^\star={ca['baseline_conflict']:.4f}$; opposed cycles start at their own stationary state, $r^\star={co['baseline_r']:.4f}$ with $\chi^\star={co['baseline_conflict']:.4f}$, under opposition already present in the basal forcing. Signed and absolute areas summarize the complete rising and falling paths over the common external-demand range, with output endpoint gaps retained separately.")
    lines += [r"\begin{table}[H]\centering\small",r"\begin{tabular}{@{}lrrrr@{}}\toprule",
              r"Direction & Half-cycle duration & $\mathcal H_{r,d_{\mathrm{ext}}}$ & $\mathcal H_{\mathrm{abs}}$ & Endpoint $\Delta r$ \\\midrule"]
    for row in report["cycles"].values():
        lines.append(" & ".join([row["direction"].capitalize(),f"{row['phase_duration']:g}",f"{row['signed']:.5f}",f"{row['absolute']:.5f}",f"{row['output_gap']:.5f}"])+r" \\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Finite-rate branch separation for both directional conditions. A vertical closure at the common low input contributes zero to the signed input--output integral; the reported output gap is retained.}\label{tab:cycle-statistics}\end{table}"]
    if all(abs(row["absolute"]-abs(row["signed"]))<1e-10 for row in report["cycles"].values()):
        lines.append(r"The interpolated branch difference has one orientation in these cycles, so absolute area equals the magnitude of signed area. Figure~\ref{fig:cycles} shows one absolute-area curve for each direction condition. The duration dependence describes the response at the specified rates.")
    else:
        lines.append(r"The signed and absolute areas in Table~\ref{tab:cycle-statistics} distinguish net orientation from total branch separation. Figure~\ref{fig:cycles} compares absolute areas across direction conditions and cycle durations.")
    long_aligned = report["cycles"]["cycle_aligned_20"]
    long_opposed = report["cycles"]["cycle_opposed_20"]
    aligned_terminal = report["post_cycle_relaxation"]["cycle_aligned_20"]
    opposed_terminal = report["post_cycle_relaxation"]["cycle_opposed_20"]
    aligned_end = long_aligned["baseline_r"]+long_aligned["output_gap"]
    aligned_stationary_gap = aligned_terminal["state"][2]-long_aligned["baseline_r"]
    lines.append(rf"At half-cycle duration $20$, the aligned absolute area, ${long_aligned['absolute']:.5f}$, exceeds the opposed area, ${long_opposed['absolute']:.5f}$. The larger aligned area accompanies a change of terminal stationary reference: this path starts at $r^\star={long_aligned['baseline_r']:.4f}$, ends the forced cycle at $r(45)={aligned_end:.4f}$, and under continued basal forcing relaxes to the lower aligned-forcing reference, $r^\star={aligned_terminal['state'][2]:.4f}$. Its cycle-end gap is ${long_aligned['output_gap']:.5f}$, while the eventual stationary-reference difference is ${aligned_stationary_gap:.5f}$. The opposed cycle begins at its different opposed-forcing reference, $r^\star={long_opposed['baseline_r']:.4f}$ with $\chi^\star={long_opposed['baseline_conflict']:.4f}$, and its continuation returns to that reference. These areas compare whole paths with different starting states and basal forcing; the common-start pulse experiment supplies the matched direction contrast.")
    lines.append("The shorter aligned cycles return to the upper stationary reference. Full-state return checks and endpoint gaps are recorded separately from the branch-area integral.")
    path.write_text("\n\n".join(lines)+"\n")


def parameter_tex(name):
    """Render code coefficient names using the manuscript's mathematical notation."""
    special = {"P0":r"P_0","c_target":r"c_{\mathrm{tar}}","c_star":r"c_{\mathrm{thr}}",
               "psi_max":r"\psi_{\mathrm{max}}"}
    if name in special:
        return special[name]
    if name in {"a1","a2","a3","b1","b2","b3","q1"}:
        return name[0]+"_"+name[1:]
    prefix,suffix = name.split("_",1)
    greek = {"rho","nu","xi","mu","omega","kappa","alpha","gamma","theta","eta","tau","delta","zeta","lambda"}
    base = "\\"+prefix if prefix in greek else prefix
    index = r"\mathrm{"+suffix+"}" if suffix in {"conflict","visibility","access","input","isolation"} else suffix
    return base+"_{"+index+"}"


def parameters_tex(report, path, scenarios):
    """Generate the complete nominal coefficient table and scenario constants."""
    constructed = scenarios["opposed"]["parameters"]
    lower = scenarios["lower_opposed"]["parameters"]
    saturation = scenarios["saturated_opposed"]["parameters"]
    solved = list(report["construction"]["solved_coefficients"])
    if constructed != lower or set(constructed)!=set(saturation) or len(constructed)!=49:
        raise ValueError("parameter table requires 49 coefficients and identical constructed-branch coefficients")
    differences = [key for key in constructed if constructed[key]!=saturation[key]]
    if set(differences)!=set(solved) or len(solved)!=6:
        raise ValueError("parameter table differences do not match the six solved balances")
    common = [key for key in constructed if key not in solved]
    if len(common)!=43 or len(set(parameter_tex(key) for key in constructed))!=49:
        raise ValueError("parameter table coverage or symbol uniqueness failed")
    compact = lambda value:format(float(value),".12g")
    lines = [r"% Generated from the complete archived coefficient dictionaries.",
             r"\begin{table}[H]\centering\footnotesize",
             r"\begin{tabular}{@{}lrlrlr@{}}\toprule",
             r"\multicolumn{6}{@{}l}{Common coefficients: constructed and saturation regimes}\\",
             r"Coefficient & Value & Coefficient & Value & Coefficient & Value \\\midrule"]
    height = ceil(len(common)/3)
    for row in range(height):
        cells = []
        for column in range(3):
            index = column*height+row
            if index<len(common):
                key = common[index]
                cells += ["$"+parameter_tex(key)+"$",compact(constructed[key])]
            else:
                cells += ["",""]
        lines.append(" & ".join(cells)+r" \\")
    lines += [r"\midrule",
              r"\multicolumn{6}{@{}l}{Regime-specific coefficients}\\",
              r"\multicolumn{2}{@{}l}{Coefficient} & \multicolumn{2}{r}{Constructed upper/lower} & \multicolumn{2}{r@{}}{Saturation} \\\midrule"]
    for key in solved:
        lines.append(r"\multicolumn{2}{@{}l}{$"+parameter_tex(key)+r"$} & \multicolumn{2}{r}{"+f"{constructed[key]:.12f}"+r"} & \multicolumn{2}{r@{}}{"+compact(saturation[key])+r"} \\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Complete nominal ODE coefficients: 43 common values and six constructed/saturation pairs. The upper and lower constructed branches use identical coefficients. Constructed values in the lower panel are rounded to twelve decimal places; integrations use the full-precision values in the accompanying scenarios. The four $k$ coefficients are zero here and take the stated overrides in the counter-mechanism.}\label{tab:model-parameters}\end{table}"]
    nominal = scenarios["opposed"]
    network = nominal["network"]["baseline"]
    pulse = nominal["network"]["schedule"][0]
    high = pulse["network"]
    support = sum(float(edge["weight"])*float(edge["support"]) for edge in network["support_edges"])/float(network["support_scale"])
    counter = scenarios["opposed_tradeoffs"]["parameters"]
    cost_keys = ("k_visibility","k_access","k_input","k_isolation")
    baseline_inputs = [nominal["inputs"]["baseline"][key] for key in ("delay","boundary","review")]
    stronger = [scenarios["opposed_inputs"]["inputs"]["schedule"][0]["values"][key] for key in ("delay","boundary","review")]
    moderation = scenarios["opposed_filtered"]["network"]["schedule"][0]["network"]["moderation"]
    target = scenarios["opposed_target_shift"]["permeability_target"]["schedule"][0]
    if any(float(network["coupling"][i][j])!=0 for i in range(3) for j in range(3) if (i,j) not in {(1,2),(2,1)}):
        raise ValueError("the declared network note requires only the two displayed nonzero couplings")
    lines.append(rf"The network uses $\Gamma_{{23}}=\Gamma_{{32}}={network['coupling'][1][2]:g}$ with all other entries zero, $K_\ell={network['layer_scales'][0]:g}$ for all layers, $K_H={network['support_scale']:g}$, and $\Psi_0={network['psi_scale']:g}$. Two unit-weight demand edges have marks ${network['demand_edges'][0]['magnitude']:g}$ before the pulse and ${high['demand_edges'][0]['magnitude']:g}$ during ${pulse['start']:g}\leq t<{pulse['end']:g}$; their one-dimensional directions are $[1,1]$ or $[1,-1]$. Unit support gives $H={support:g}$ and $h_{{\mathrm{{net}}}}={support/(1+support):g}$. The filter comparison uses $\bm\sigma={vector_tex(moderation,1)}$ during the pulse; the other comparisons use zero filtering.")
    lines.append(rf"The baseline inputs are $\bm u_0={vector_tex(baseline_inputs,2)}$; stronger inputs use ${vector_tex(stronger,2)}$ during the pulse.")
    cost_values = ",".join(f"{counter[key]:.2f}" for key in cost_keys)
    lines.append(r"The counter-mechanism uses"+"\n"+rf"\[(k_{{\mathrm{{visibility}}}},k_{{\mathrm{{access}}}},k_{{\mathrm{{input}}}},k_{{\mathrm{{isolation}}}})=({cost_values}).\]")
    lines.append(rf"Pathway-removal comparisons set the named $a_{{\mathrm{{conflict}}}}$, $\lambda_d$, or $\lambda_h$ to zero. The primary target remains $c_{{\mathrm{{tar}}}}={nominal['permeability_target']['baseline']:g}$; the separate target comparison sets it to ${target['value']:g}$ during the pulse and restores it afterwards.")
    path.write_text("\n\n".join(lines)+"\n")
    return {"all_parameter_keys":list(constructed),"common_keys":common,"solved_keys":solved,
            "parameter_count":len(constructed),"common_count":len(common),"solved_count":len(solved),
            "constructed_branch_parameters_equal":constructed==lower,
            "constructed_parameters":constructed,"saturation_parameters":saturation,
            "constructed_display_decimal_places":12}


def presentation_manifest(report, parameter_coverage, out, rendered):
    """Record deterministic identities for the unchanged data and new presentation."""
    numerical = {name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in report["numerical_files"]}
    assets = ["revision-results.tex","revision-numerics.tex","revision-parameters.tex",
              "mechanism_figure.tex","cycle_statistics.tex"]
    if rendered:
        assets += ["mechanism_figure.pdf","mechanism_figure.png","cycle_statistics.pdf","cycle_statistics.png"]
    record = {"presentation_revision":1,"presentation_label":"self-contained-presentation", "scientific_version":report["version"],
              "protocol_sha256":report["protocol_sha256"],"implementation_sha256":report["implementation_sha256"],
              "numerical_file_count":len(numerical),"numerical_files_sha256":numerical,
              "parameter_table":parameter_coverage,
              "presentation_assets_sha256":{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in assets},
              "rendered":bool(rendered)}
    write_json(out/"presentation-manifest.json",record)


def scientific_number(value):
    mantissa,exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def numerics_tex(report, path, protocol):
    max_residual = max(row["full_residual"] for row in report["baselines"].values())
    max_drift = max(row["pre_pulse_drift"] for row in report["scenarios"].values())
    max_state_error = max(row["max_state_dt_vs_dt2"] for row in report["convergence"].values())
    max_cycle_error = max(row["max_state_dt_vs_dt2"] for row in report["cycle_convergence"].values())
    max_area_error = max(row["absolute_dt_vs_dt2"] for row in report["cycle_convergence"].values())
    max_area_fine_error = max(row["absolute_dt2_vs_dt4"] for row in report["cycle_convergence"].values())
    max_core_error = max(row["max_absolute_w2_error"] for row in report["core_load_benchmarks"].values())
    max_jacobian_error = max(row["local_stability"]["finite_difference_max_error"] for row in report["baselines"].values())
    local_completed = [row for row in report["local_return_audit"] if row["status"]=="completed"]
    local_returned = sum(row["return_status"]=="returned" for row in local_completed)
    local_distance = max((row["distance_to_own_reference"] for row in local_completed),default=0.0)
    settings = protocol["initialization"]
    lines = [r"% Generated numerical verification details for the declared protocol.",
             rf"The stationary construction solves six balance equations at the specified reference. Other baselines use bounded relaxation with step ${settings['dt']:g}$, residual tolerance ${scientific_number(settings['tolerance'])}$, and maximum duration ${settings['max_time']:g}$. The branch-aware cache includes the starting seed, so distinct equilibria of the same parameter set remain separate. The largest saved full-field residual is ${scientific_number(max_residual)}$ and the maximum pre-pulse drift is ${scientific_number(max_drift)}$. All four declared seed probes and their outcomes are retained.",
             rf"Analytical Jacobians are checked against centered finite differences at steps $10^{{-5}}$ and $5\times10^{{-6}}$; the largest analytical-versus-finer-difference discrepancy among saved baselines is ${scientific_number(max_jacobian_error)}$. The Routh table supplies a floating-point local linear diagnostic, with its full first column and numerical conditions recorded for every baseline. These local calculations distinguish the reported branches without establishing that all stationary branches have been found.",
             rf"Classical RK4 uses step $0.02$, splits at schedule boundaries, and evaluates continuous forcing at each stage. Refinements at $0.01$ and $0.005$ share the same unrounded initial states. The largest common-grid difference between the first two steps is ${scientific_number(max_state_error)}$ over pulse, control, and sensitivity conditions, and ${scientific_number(max_cycle_error)}$ over cycles. Across the ${len(report['core_load_benchmarks'])}$ eligible core traces, the maximum load error against the piecewise exact gain--loss solution is ${scientific_number(max_core_error)}$.",
             rf"Branch interpolation and quadrature are checked separately. The largest change in absolute area is ${scientific_number(max_area_error)}$ between steps $0.02$ and $0.01$, and ${scientific_number(max_area_fine_error)}$ between $0.01$ and $0.005$. Signed-area and endpoint-gap refinements are also retained. The complete supplement contains ${len(report['scenarios'])}$ pulse/control traces, ${len(report['sensitivity'])}$ pathway/normalization conditions, ${len(report['neighborhood_sensitivity'])}$ coefficient-neighborhood conditions, and ${len(report['cycles'])}$ continuous cycles.",
             rf"The local nonlinear audit perturbs both constructed stationary branches by $0.005$ in two fixed directions and both signs. Of the ${len(report['local_return_audit'])}$ declared probes, ${local_returned}$ return to their own reference within the specified distance tolerance; the largest completed-probe distance is ${scientific_number(local_distance)}$. Separate full-state relaxations continue the ${len(report['post_pulse_relaxation'])}$ pulse outcomes from $t=20$ and the ${len(report['post_cycle_relaxation'])}$ cycle outcomes from their low-input endpoints under the corresponding basal forcing. Every residual, terminal distance, additional duration, and unresolved outcome is retained."]
    path.write_text("\n\n".join(lines)+"\n")


def matched_direction_sensitivity(report, sensitivity_scenarios, aligned_configuration):
    rows=[]
    baseline=report["scenarios"]["aligned"]
    expected={key:value for key,value in deepcopy(aligned_configuration).items() if key!="metadata"}
    for result in report["sensitivity"]:
        parameter=result["parameter"]
        if parameter not in ("a_conflict","psi_scale"):
            continue
        if result.get("status","completed") != "completed":
            rows.append({"scenario":result.get("scenario"),"parameter":parameter,"value":result["value"],
                         "status":result["status"],"reason":result.get("reason","condition did not complete")})
            continue
        cfg=deepcopy(sensitivity_scenarios[result["scenario"]])
        cfg.pop("metadata",None)
        cfg["parameters"]["a_conflict"]=expected["parameters"]["a_conflict"]
        for network in [cfg["network"]["baseline"]]+[interval["network"] for interval in cfg["network"]["schedule"]]:
            network["psi_scale"]=expected["network"]["baseline"]["psi_scale"]
            for edge in network["demand_edges"]:
                edge["direction"]=[1.0]
            if aggregate_network(network).conflict!=0:
                raise ValueError("aligned reference reuse requires identically zero conflict")
        if cfg!=expected:
            raise ValueError("directional sensitivity does not share the declared aligned reference")
        rows.append({"parameter":parameter,"value":result["value"],
                     "r_aligned_t10":baseline["r_t10"],"r_opposed_t10":result["r_t10"],
                     "r_aligned_t20":baseline["r_t20"],"r_opposed_t20":result["r_t20"],
                     "opposed_minus_aligned_t10":result["r_t10"]-baseline["r_t10"],
                     "opposed_minus_aligned_t20":result["r_t20"]-baseline["r_t20"]})
    if len(rows)!=7:
        raise ValueError("the declared matched-direction endpoint study requires seven conditions")
    return rows


def implementation_hashes():
    names=("model.py","network.py","analysis.py","regimes.py","configs/illustrative_baseline.json")
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    selected={"Experiment","construction_record","raw_scenarios","matched_controls","summarize","add_outcomes","cycle_scenarios","run_numerical_experiment","cycle_refinement","matched_direction_sensitivity","response_difference","core_load_benchmark","baseline_config","at_time","state_from","canonical","signature"}
    tree=ast.parse(Path(__file__).read_text())
    numerical=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in selected],type_ignores=[])
    hashes["runner_numerical_ast"]=hashlib.sha256(ast.dump(numerical,include_attributes=False).encode()).hexdigest()
    return hashes


def load_saved_results(source, out, protocol_path, protocol):
    report=json.loads((source/"report.json").read_text())
    if report["version"] != VERSION or report["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("saved numerical results differ from the current protocol/version")
    if report["implementation_sha256"] != implementation_hashes():
        raise ValueError("saved numerical results differ from the current model or analysis implementation")
    experiment=Experiment(protocol)
    experiment.baselines=deepcopy(report["baselines"])
    primary={name:experiment.initialize(cfg) for name,cfg in raw_scenarios(protocol).items()}
    scenarios=dict(primary)
    for name,cfg in primary.items():
        no_pulse,context=matched_controls(cfg)
        scenarios[name+"_control"]=no_pulse
        scenarios[name+"_context_control"]=context
    if set(scenarios) != set(report["scenarios"]):
        raise ValueError("saved comparison inventory differs from declared scenarios")
    traces={}
    cycle_traces={}
    def read_trace(name,cfg):
        actual=json.loads((source/f"{name}.json").read_text())
        if actual!=cfg:
            raise ValueError(f"saved configuration differs: {name}")
        with (source/f"{name}.csv").open(newline="",encoding="utf-8") as stream:
            rows=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
        expected=round(cfg["solver"]["time_horizon"]/cfg["solver"]["dt"])+1
        if len(rows)!=expected or not isclose(rows[-1]["time"],cfg["solver"]["time_horizon"]):
            raise ValueError(f"saved trace grid differs: {name}")
        return rows
    for name,cfg in scenarios.items():
        traces[name]=read_trace(name,cfg)
        summary=report["scenarios"][name]
        for key,value in (("r_t10",at_time(traces[name],10)["r"]),("r_t20",at_time(traces[name],20)["r"]),
                          ("w1_t20",at_time(traces[name],20)["w1"]),("w2_t20",at_time(traces[name],20)["w2"])):
            if summary[key]!=value:
                raise ValueError(f"saved trace/report mismatch for {name}:{key}")
    for name,cfg in cycle_scenarios(protocol,raw_scenarios(protocol)["opposed"]):
        cfg=experiment.initialize(cfg)
        cycle_traces[name]=read_trace(name,cfg)
    for name in report["numerical_files"]:
        if Path(name).name!=name or not (source/name).is_file():
            raise ValueError("invalid or missing saved numerical artifact")
        data=(source/name).read_bytes()
        if name.endswith(".csv") and b"\r" in data:
            raise ValueError("saved CSVs must use LF line endings")
        if source/name != out/name:
            (out/name).write_bytes(data)
    return report,scenarios,traces,cycle_traces


def render_figure(out,stem):
    environment={**os.environ,"SOURCE_DATE_EPOCH":str(FIGURE_SOURCE_DATE_EPOCH)}
    subprocess.run(["tectonic","-Zdeterministic-mode","--outdir",str(out),str(out/f"{stem}.tex")],check=True,env=environment)
    subprocess.run(["pdftoppm","-singlefile","-r","150","-png",str(out/f"{stem}.pdf"),str(out/stem)],check=True)


def same_pdf_content(left,right):
    def inspect(path):
        text=subprocess.run(["pdftotext","-layout",str(path),"-"],check=True,stdout=subprocess.PIPE).stdout
        pixels=subprocess.run(["pdftoppm","-singlefile","-r","150","-png",str(path)],check=True,stdout=subprocess.PIPE).stdout
        return text,pixels
    return inspect(left)==inspect(right)


def sync_manuscript(out,manuscript):
    (manuscript/"figures").mkdir(parents=True,exist_ok=True)
    for source_name,target_name in (("mechanism_figure.pdf","figures/revision-mechanism.pdf"),
                                    ("cycle_statistics.pdf","figures/cycle-statistics.pdf"),
                                    ("revision-results.tex","revision-results.tex"),("revision-numerics.tex","revision-numerics.tex"),
                                    ("revision-parameters.tex","revision-parameters.tex")):
        source,target=out/source_name,manuscript/target_name
        if target.exists() and source.read_bytes()==target.read_bytes():
            continue
        if target.exists() and source.suffix==".pdf" and same_pdf_content(source,target):
            continue
        shutil.copy2(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/referee_protocol.json")
    parser.add_argument("--output",type=Path,default=ROOT/"presentation-results")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--from-results",type=Path,help="regenerate presentation from matching verified numerical files")
    parser.add_argument("--manuscript-dir",type=Path,help="explicit manuscript-asset destination; requires --render")
    args=parser.parse_args()
    if args.manuscript_dir and not args.render:
        parser.error("--manuscript-dir requires --render")
    out=args.output.resolve()
    protected=(ROOT/"referee-results").resolve()
    if out==protected or protected in out.parents:
        parser.error("referee-results is a preserved numerical snapshot; select another output directory")
    out.mkdir(parents=True,exist_ok=True)
    protocol_path=args.protocol.resolve()
    protocol=json.loads(protocol_path.read_text())
    if protocol["version"]!=VERSION or protocol["solver"]!={"method":"classical_runge_kutta_4","dt":.02,"time_horizon":20}:
        raise ValueError("the declared pulse experiment requires version 0.4.0, RK4 step 0.02 and horizon 20")
    if args.from_results:
        report,scenarios,traces,cycles=load_saved_results(args.from_results.resolve(),out,protocol_path,protocol)
    else:
        report,scenarios,traces,cycles=run_numerical_experiment(protocol_path,protocol,out)
    figure_tex(traces,report,out/"mechanism_figure.tex")
    cycle_figure_tex(cycles,traces,report,out/"cycle_statistics.tex")
    results_tex(report,out/"revision-results.tex",scenarios)
    numerics_tex(report,out/"revision-numerics.tex",protocol)
    parameter_coverage=parameters_tex(report,out/"revision-parameters.tex",scenarios)
    if args.render:
        for stem in ("mechanism_figure","cycle_statistics"):
            render_figure(out,stem)
    presentation_manifest(report,parameter_coverage,out,args.render)
    if args.manuscript_dir:
        sync_manuscript(out,args.manuscript_dir.resolve())
    print(json.dumps({"output":str(out),"version":VERSION,"reused_results":bool(args.from_results),
                      "primary_comparisons":len(report["primary_comparisons"]),"expanded_comparisons":len(report["scenarios"]),
                      "sensitivity_conditions":len(report["sensitivity"]),"cycles":len(report["cycles"]),
                      "maximum_state_refinement_difference":max(row["max_state_dt_vs_dt2"] for row in report["convergence"].values()),
                      "maximum_cycle_refinement_difference":max(row["max_state_dt_vs_dt2"] for row in report["cycle_convergence"].values()),
                      "direction_contrast":report["direction_contrast"],
                      "primary_coordinate_spans":report["primary_coordinate_spans"],
                      "neighborhood_failures":len(report["neighborhood_failures"])},indent=2))


if __name__=="__main__":
    main()
