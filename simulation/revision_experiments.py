"""Reproduce the declared stationary-baseline mechanism comparisons.

Every coefficient, forcing schedule, initialization rule and output is declared
in configs/dynamics_protocol.json. Numerical values are synthetic and
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
from math import isclose, ceil, floor, log10
import os
from pathlib import Path
import shutil
import subprocess

from analysis import cycle_statistics, deviation_index, find_equilibrium
from model import STATE_NAMES, simulate
from network import aggregate_network

ROOT = Path(__file__).resolve().parent
VERSION = "0.3.0"
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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


def raw_scenarios(protocol):
    base = json.loads((ROOT / "configs/illustrative_baseline.json").read_text())
    base["solver"] = deepcopy(protocol["solver"])
    base["parameters"].update(protocol["bridge_parameters"])
    base["network"] = deepcopy(protocol["network"])
    base["permeability_target"] = deepcopy(protocol["permeability_target"])
    scenarios = {}
    for name, options in protocol["comparisons"].items():
        cfg = deepcopy(base)
        cfg["metadata"] = {"name": name, "version": VERSION, "status": "Synthetic stationary-baseline mechanism comparison."}
        cfg["parameters"].update(options.get("parameters", {}))
        if options.get("opposed", True):
            cfg["network"]["schedule"][0]["network"]["demand_edges"][1]["direction"] = [-1.0]
        if options.get("inputs"):
            cfg["inputs"]["schedule"] = deepcopy(protocol["input_pulse"])
        if options.get("moderation"):
            cfg["network"]["schedule"][0]["network"]["moderation"] = list(options["moderation"])
        if not options.get("target_pulse", True):
            cfg["permeability_target"]["schedule"] = []
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
        baseline["initial_state"] = list(self.protocol["initialization"]["seed"])
        key = signature({name: baseline[name] for name in ("parameters", "inputs", "network", "permeability_target")})
        if key not in self.baselines:
            settings = self.protocol["initialization"]
            result = find_equilibrium(baseline, initial_state=settings["seed"], dt=settings["dt"],
                                      tolerance=settings["tolerance"], max_time=settings["max_time"],
                                      check_interval=settings["check_interval"])
            record = asdict(result)
            record["configuration"] = baseline
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
    if control is not None:
        contrast = [row["r"] - baseline["r"] for row, baseline in zip(rows, control)]
        peak = max(range(len(rows)), key=lambda i: abs(contrast[i]))
        result.update({"r_contrast_t10": endpoint["r"] - at_time(control, 10)["r"],
                       "r_contrast_t20": final["r"] - at_time(control, 20)["r"],
                       "max_abs_r_contrast": abs(contrast[peak]), "max_abs_r_contrast_time": rows[peak]["time"]})
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
             "r_demand_contrast": row["r"] - base["r"]} for row, base in zip(rows, context)]


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
                               "status": "Continuous triangular demand cycle; descriptive finite-rate path comparison."}
            yield name, cfg


def run_numerical_experiment(protocol_path, protocol, out):
    experiment = Experiment(protocol)
    raw = raw_scenarios(protocol)
    primary = {name: experiment.initialize(cfg) for name, cfg in raw.items()}
    common_reference = primary["opposed"]["initial_state"]
    scenarios = dict(primary)
    for name, cfg in primary.items():
        no_pulse, context = matched_controls(cfg)
        scenarios[name + "_control"] = no_pulse
        scenarios[name + "_context_control"] = context
    bare = {name: experiment.trace(cfg) for name, cfg in scenarios.items()}
    report = {"status": "Synthetic stationary-baseline mechanism verification", "version": VERSION,
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "scenarios": {}, "primary_comparisons": list(primary), "convergence": {}, "baselines": {},
              "sensitivity": [], "cycles": {}, "cycle_convergence": {}, "scope": "Uncalibrated model-coordinate contrasts; no empirical detectability or power claim."}
    traces = {}
    for name, cfg in scenarios.items():
        parent_name = name.removesuffix("_context_control").removesuffix("_control")
        context_rows = bare[parent_name + "_context_control"]
        reference = cfg["initial_state"]
        traces[name] = add_outcomes(bare[name], reference, context_rows)
        report["scenarios"][name] = summarize(name, bare[name], reference, common_reference, context_rows)
        report["convergence"][name] = experiment.convergence(cfg)
        report["scenarios"][name]["coordinate_spans"] = {coordinate: max(row[coordinate] for row in bare[name]) - min(row[coordinate] for row in bare[name]) for coordinate in STATE_NAMES}
        write_json(out / f"{name}.json", cfg)
        write_csv(out / f"{name}.csv", traces[name])
    report["matched_ablation_max_state_difference"] = max(abs(a[key] - b[key]) for a, b in zip(bare["aligned_no_conflict"], bare["opposed_no_conflict"]) for key in STATE_NAMES)
    direction = [b["r"] - a["r"] for a, b in zip(bare["aligned"], bare["opposed"])]
    peak = max(range(len(direction)), key=lambda i: abs(direction[i]))
    report["direction_contrast"] = {"opposed_minus_aligned_t10": direction[500], "opposed_minus_aligned_t20": direction[-1],
                                    "max_absolute": abs(direction[peak]), "time_of_maximum": bare["opposed"][peak]["time"],
                                    "measurement_note": "Normalized model units; neither percent human functioning nor an empirically calibrated effect size."}
    report["core_full_demand_response"] = {}
    for tag, full_name, core_name in [("with_target_pulse", "opposed", "opposed_reduced"),
                                      ("constant_target", "opposed_constant_target", "opposed_reduced_constant_target")]:
        difference = [a["r_demand_contrast"] - b["r_demand_contrast"] for a, b in zip(traces[full_name], traces[core_name])]
        index = max(range(len(difference)), key=lambda i: abs(difference[i]))
        report["core_full_demand_response"][tag] = {"t10": difference[500], "t20": difference[-1], "max_absolute": abs(difference[index]),
                                                   "time_of_maximum": traces[full_name][index]["time"],
                                                   "interpretation": "Difference of within-model pulse-minus-context-control responses; remaining state interactions remain bundled."}
    sensitivity_scenarios = {}
    for parameter, values in protocol["sensitivity_ranges"].items():
        for index, value in enumerate(values):
            parent = "opposed_tradeoffs_inputs" if parameter.startswith("k_") else "opposed"
            cfg = deepcopy(raw[parent])
            cfg["parameters"][parameter] = value
            name = f"sensitivity_{parameter}_{index}"
            cfg["metadata"]["name"] = name
            cfg = experiment.initialize(cfg)
            _, context = matched_controls(cfg)
            rows, context_rows = experiment.trace(cfg), experiment.trace(context)
            result = summarize(name, rows, cfg["initial_state"], common_reference, context_rows)
            result.update({"parameter": parameter, "value": value, "baseline_id": cfg["metadata"]["baseline_id"]})
            report["sensitivity"].append(result)
            report["convergence"][name] = experiment.convergence(cfg)
            sensitivity_scenarios[name] = cfg
    for index, value in enumerate(protocol["normalization_sensitivity"]["values"]):
        cfg = deepcopy(raw["opposed"])
        cfg["network"]["baseline"]["psi_scale"] = value
        cfg["network"]["schedule"][0]["network"]["psi_scale"] = value
        name = f"sensitivity_psi_scale_{index}"
        cfg["metadata"]["name"] = name
        cfg = experiment.initialize(cfg)
        _, context = matched_controls(cfg)
        rows = experiment.trace(cfg)
        result = summarize(name, rows, cfg["initial_state"], common_reference, experiment.trace(context))
        result.update({"parameter": "psi_scale", "value": value, "baseline_id": cfg["metadata"]["baseline_id"]})
        report["sensitivity"].append(result)
        report["convergence"][name] = experiment.convergence(cfg)
        sensitivity_scenarios[name] = cfg
    write_json(out / "sensitivity_scenarios.json", sensitivity_scenarios)
    cycle_traces = {}
    for name, cfg in cycle_scenarios(protocol, raw["opposed"]):
        cfg = experiment.initialize(cfg)
        rows = experiment.trace(cfg)
        start, end = cfg["metadata"]["cycle_start"], cfg["metadata"]["cycle_end"]
        branch_rows = [row for row in rows if start - 1e-10 <= row["time"] <= end + 1e-10]
        metrics = cycle_statistics([row["external_demand"] for row in branch_rows], [row["r"] for row in branch_rows])
        metrics.update({"scenario": name, "phase_duration": cfg["metadata"]["phase_duration"],
                        "direction": "opposed" if "opposed" in name else "aligned", "baseline_id": cfg["metadata"]["baseline_id"],
                        "V_cycle_end": deviation_index(state_from(at_time(rows, end)), cfg["initial_state"]),
                        "pre_cycle_drift": max(abs(row[key] - cfg["initial_state"][i]) for row in rows if row["time"] < start for i, key in enumerate(STATE_NAMES))})
        report["cycles"][name] = metrics
        report["cycle_convergence"][name] = cycle_refinement(experiment, cfg, rows)
        cycle_traces[name] = [{**row, "V": deviation_index(state_from(row), cfg["initial_state"])} for row in rows]
        write_json(out / f"{name}.json", cfg)
        write_csv(out / f"{name}.csv", cycle_traces[name])
    nominal = primary["opposed"]
    baseline = baseline_config(nominal)
    seed_audit = []
    for seed in protocol["equilibrium_audit_seeds"]:
        settings = protocol["initialization"]
        result = find_equilibrium(baseline, initial_state=seed, dt=settings["dt"], tolerance=settings["tolerance"],
                                  max_time=settings["max_time"], check_interval=settings["check_interval"])
        record = asdict(result)
        record["max_distance_from_nominal"] = max(abs(a - b) for a, b in zip(record["state"], common_reference))
        seed_audit.append(record)
    report["equilibrium_seed_audit"] = seed_audit
    report["baselines"] = experiment.baselines
    report["all_primary_coordinates_active"] = {name: all(value > 1e-8 for value in report["scenarios"][name]["coordinate_spans"].values()) for name in ("aligned", "opposed")}
    report["network_examples"] = {"baseline": asdict(aggregate_network(nominal["network"]["baseline"])),
                                   "opposed_pulse": asdict(aggregate_network(nominal["network"]["schedule"][0]["network"])),
                                   "aligned_pulse": asdict(aggregate_network(primary["aligned"]["network"]["schedule"][0]["network"]))}
    summaries = [{key: value for key, value in report["scenarios"][name].items() if key != "coordinate_spans"} for name in scenarios]
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "sensitivity.csv", report["sensitivity"])
    write_csv(out / "cycle_statistics.csv", list(report["cycles"].values()))
    report["implementation_sha256"] = implementation_hashes()
    report["numerical_files"] = sorted([f"{name}{suffix}" for name in scenarios for suffix in (".json", ".csv")] + [f"{name}{suffix}" for name in cycle_traces for suffix in (".json", ".csv")] + ["summary.csv", "sensitivity.csv", "sensitivity_scenarios.json", "cycle_statistics.csv", "direction_sensitivity.csv", "report.json"])
    report["directional_reference_reuse"] = "The aligned conflict signal is identically zero. The a_conflict and psi_scale changes therefore leave its full vector field unchanged; the stored stationary initial states are identical as well."
    report["directional_sensitivity"] = matched_direction_sensitivity(report, sensitivity_scenarios, primary["aligned"])
    write_csv(out / "direction_sensitivity.csv", report["directional_sensitivity"])
    write_json(out / "report.json", report)
    return report, scenarios, traces, cycle_traces


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
            tick = f"{value:g}" if "yticks" in panel else (f"{value:.3g}" if abs(ymax-ymin) < .1 else f"{value:.2f}")
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
    blue, orange, green = "blue!75!black", "orange!85!black", "green!50!black"
    series = lambda name,key,color,label,style="": (sampled_points(traces[name],key),color,label,style)
    contrast = [(b["time"], b["r"]-a["r"]) for a,b in zip(traces["aligned"],traces["opposed"])][::5]
    extent = nice_max(max(abs(value) for _,value in contrast)*1.2 or .01)
    response_extent = nice_max(max(abs(row["r_demand_contrast"]) for name in ("opposed_constant_target","opposed_reduced_constant_target") for row in traces[name])*1.2 or .01)
    panels = [
        {"title":"A. Declared forcing", "xlabel":"Time (dimensionless)", "ylabel":"Signal", "xlim":(0,20),"ylim":(0,1),"series":[
            (sampled_points(traces["opposed"],"external_demand",jump=True),blue,"External demand",""),
            (sampled_points(traces["opposed"],"conflict",jump=True),orange,"Directional conflict","")]},
        {"title":"B. Resource response", "xlabel":"Time (dimensionless)", "ylabel":r"Resource reserve $r$", "xlim":(0,20),"ylim":(0,1),"series":[
            series("aligned","r",blue,"Aligned"),series("opposed","r",orange,"Opposed"),series("opposed_control","r",green,"No-pulse control",",dashed")]},
        {"title":"C. Directional contrast", "xlabel":"Time (dimensionless)", "ylabel":r"$r_{\mathrm{opp}}-r_{\mathrm{aligned}}$", "xlim":(0,20),"ylim":(-extent,extent),"yticks":[-.04,-.02,0,.02,.04],"series":[
            (contrast,orange,"Opposed minus aligned","")]},
        {"title":"D. Experienced load", "xlabel":"Time (dimensionless)", "ylabel":r"Experienced load $w_2$", "xlim":(0,20),"ylim":(0,1),"series":[
            series("opposed","w2",orange,"Full model"),series("opposed_no_external","w2",blue,"External path off"),series("opposed_filtered","w2",green,"Upstream filtering",",dashed")]},
        {"title":"E. Controlled core/full responses", "xlabel":"Time (dimensionless)", "ylabel":r"Demand contrast $\delta r$", "xlim":(0,20),"ylim":(-response_extent,response_extent),"yticks":[-.04,-.02,0,.02,.04],"series":[
            series("opposed_constant_target","r_demand_contrast",orange,"Full, fixed target"),series("opposed_reduced_constant_target","r_demand_contrast",blue,"Core, fixed target")]},
        {"title":"F. Permeability adjustment", "xlabel":"Time (dimensionless)", "ylabel":r"Permeability $c$", "xlim":(0,20),"ylim":(0,1),"series":[
            series("opposed","c",orange,r"State $c$"),(sampled_points(traces["opposed"],"c_target",jump=True),blue,r"Target $c_{\mathrm{tar}}$",",dashed"),series("opposed_reduced","c",green,"Frozen core value",",dotted")]},
    ]
    draw_figure(path,panels)


def cycle_figure_tex(cycles, traces, report, path):
    colors = ["blue!75!black","orange!85!black","green!50!black"]
    selected = [name for name in cycles if "opposed" in name]
    selected.sort(key=lambda name:report["cycles"][name]["phase_duration"])
    maximum_time = max(cycles[name][-1]["time"] for name in selected)
    path_series=[]
    signals=[]
    for name,color in zip(selected,colors):
        rows=cycles[name]
        phase=report["cycles"][name]["phase_duration"]
        start,peak,end=5,5+phase,5+2*phase
        signals.append((sampled_points(rows,"external_demand",stride=10),color,rf"Half-cycle ${phase:g}$", ""))
        up=[(row["external_demand"],row["r"]) for row in rows if start-1e-10<=row["time"]<=peak+1e-10][::5]
        down=[(row["external_demand"],row["r"]) for row in rows if peak-1e-10<=row["time"]<=end+1e-10][::5]
        path_series += [(up,color,rf"${phase:g}$ rising",""),(down,color,None,",dashed")]
    # Compact line-key uses one series per rate; dashed return is identified in caption.
    path_series_for_plot=[]
    for index,(points,color,label,style) in enumerate(path_series):
        path_series_for_plot.append((points,color,label,style))
    values=[value for points,_,_,_ in path_series for _,value in points]
    ylow=max(0,floor(min(values)*100)/100-.01);yhigh=min(1,ceil(max(values)*100)/100+.01)
    vmax=nice_max(max(row["V"] for name in ("aligned","opposed") for row in traces[name])*1.15 or .001)
    areas=[(report["cycles"][name]["phase_duration"],report["cycles"][name]["absolute"]) for name in selected]
    signed=[(report["cycles"][name]["phase_duration"],report["cycles"][name]["signed"]) for name in selected]
    amax=nice_max(max(abs(value) for _,value in areas+signed)*1.15 or .001)
    panels=[
        {"title":"A. Continuous cycle schedules","xlabel":"Time (dimensionless)","ylabel":r"External demand $d_{\mathrm{ext}}$","xlim":(0,maximum_time),"ylim":(0,1),"series":signals},
        {"title":"B. Input--output paths","xlabel":r"External demand $d_{\mathrm{ext}}$","ylabel":r"Resource reserve $r$","xlim":(.25,.72),"ylim":(ylow,yhigh),"xticks":[.3,.4,.5,.6,.7],"yticks":[.28,.30,.32,.34],"series":path_series_for_plot},
        {"title":"C. Deviation from baseline","xlabel":"Time (dimensionless)","ylabel":r"Equal-weight $V$","xlim":(0,20),"ylim":(0,vmax),"yticks":[0,.001,.002,.003,.004,.005],"series":[
            (sampled_points(traces["aligned"],"V"),colors[0],"Aligned",""),(sampled_points(traces["opposed"],"V"),colors[1],"Opposed","")]},
        {"title":"D. Finite-rate path separation","xlabel":"Half-cycle duration","ylabel":"Area (model units)","xlim":(0,max(x for x,_ in areas)*1.1),"ylim":(min(0,min(value for _,value in signed)*1.15),amax),"xticks":[5,10,20],"series":[
            (areas,colors[1],r"Absolute $\mathcal H_{\mathrm{abs}}$",""),(signed,colors[0],r"Signed $\mathcal H$",",dashed")]},
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
    records=report["scenarios"]
    nominal=scenarios["opposed"]
    reference=nominal["initial_state"]
    baseline=report["baselines"][nominal["metadata"]["baseline_id"]]
    lower,_=exact_example(nominal["network"]["baseline"])
    higher,conflict=exact_example(nominal["network"]["schedule"][0]["network"])
    max_drift=max(records[name]["pre_pulse_drift"] for name in report["primary_comparisons"])
    max_error=max(row["max_state_dt_vs_dt2"] for row in report["convergence"].values())
    direction=report["direction_contrast"]
    lines=[r"% Generated from the declared stationary-baseline protocol and numerical report.",
           rf"Each structural variant is initialized by baseline relaxation until the infinity norm of its full vector field is at most $10^{{-11}}$. The nominal stationary state is approximately \begin{{gather*}}\bm x^\star_{{\mathrm{{base}}}}\approx {vector_tex(reference)}.\end{{gather*}} Its full-field residual is ${baseline['residual']:.2e}$, and the baseline input is $\bm u_0={vector_tex([nominal['inputs']['baseline'][key] for key in ('delay','boundary','review')],2)}$. The unrounded stationary values are used in integration and supplied with their seed, step, residual and relaxation history in the supplement. The largest pre-pulse drift among the declared primary comparisons is ${max_drift:.2e}$. Additional starting seeds are reported separately; these checks identify the selected stationary branch.",
           rf"During $5\leq t<10$, the two unit-weight demand edges in layers 2 and 3 increase their marks from $0.15$ to $1$, with aligned or opposed directions. The symmetric transmission coefficients are $\Gamma_{{23}}=\Gamma_{{32}}=0.2$. Source-preserving amplification gives $d_{{\mathrm{{ext}}}}={fraction_tex(lower)}$ before the pulse and ${fraction_tex(higher)}$ during it. The opposed pulse gives $\chi={fraction_tex(conflict)}$; the aligned signal has zero directional conflict. The nominal conflict coefficient remains $a_{{\mathrm{{conflict}}}}=0.30$.",
           r"The primary comparisons shift the declared permeability target from $0.45$ to $0.35$ during the pulse and back afterwards. Both direction conditions receive the same target schedule. Separate constant-target comparisons and context controls distinguish this exogenous drive from demand and cross-coordinate responses. The two-state core starts at the same full-system baseline equilibrium and fixes the remaining coordinates there. The counter-mechanism uses its own stationary baseline; its resource-benefit and cost coefficients remain $[0.10,0.08,0.20,0.20]^{\top}$. Stronger inputs use $[0.60,0.70,0.60]^{\top}$ during the pulse.",
           r"For every comparison, a no-pulse control holds all baseline conditions fixed. A context control retains the input, permeability-target and upstream-filter schedules while keeping demand vectors at baseline. We define the demand-response contrast as $\delta r(t)=r_{\mathrm{pulse}}(t)-r_{\mathrm{context}}(t)$. The deviation index uses the corresponding full-system baseline as reference and equal weights; common nominal-reference values are also supplied in the data.",
           r"\begin{table}[H]\centering\footnotesize",
           r"\begin{tabularx}{\textwidth}{@{}Yrrrrr@{}}\toprule",
           r"Declared comparison & $r(10)$ & $r(20)$ & $\max|\delta r|$ & $10^3V(10)$ & $10^3V(20)$ \\\midrule"]
    labels={"aligned":"Aligned", "opposed":"Opposed", "opposed_no_conflict":"Conflict path off", "opposed_no_external":"External path off", "opposed_no_support":"Support path off", "opposed_reduced":"Two-state core", "opposed_constant_target":"Full, constant target", "opposed_reduced_constant_target":"Core, constant target", "opposed_filtered":"Upstream filtering", "opposed_tradeoffs":"Counter-mechanism", "opposed_inputs":"Stronger inputs", "opposed_tradeoffs_inputs":"Counter, stronger inputs"}
    for name,label in labels.items():
        row=records[name]
        lines.append(f"{label} & {row['r_t10']:.4f} & {row['r_t20']:.4f} & {row['max_abs_r_contrast']:.4f} & {1000*row['V_t10']:.4f} & {1000*row['V_t20']:.4f} \\\\")
    lines += [r"\bottomrule\end{tabularx}",
              r"\caption{Stationary-baseline synthetic comparisons. $\delta r$ is the pulse-minus-context-control resource response; each $V$ uses its stated baseline reference, and the displayed deviation columns are multiplied by $10^3$. Full expanded tables include all controls, close-bond and load outputs, one-unit changes in $V$, coordinate spans, and contrast timing.}\label{tab:synthetic-results}\end{table}"]
    full_core=report["core_full_demand_response"]["constant_target"]
    active=report["all_primary_coordinates_active"]
    lines.append(rf"The opposed-minus-aligned resource contrast is ${direction['opposed_minus_aligned_t10']:.4f}$ at $t=10$ and ${direction['opposed_minus_aligned_t20']:.4f}$ at $t=20$; its maximum absolute value is ${direction['max_absolute']:.4f}$ at $t={direction['time_of_maximum']:.2f}$. These are uncalibrated normalized-coordinate differences. Their empirical detectability depends on a justified observation and measurement-error model. Removing the directional pathway makes the matched trajectories coincide (maximum state difference ${report['matched_ablation_max_state_difference']:g}$). The figure displays both full-range states and an explicitly scaled contrast axis, preserving the magnitude of the computed response.")
    lines.append(rf"With a constant permeability target, the difference between full and core demand-response contrasts is ${full_core['t10']:.4f}$ at $t=10$ and ${full_core['t20']:.4f}$ at $t=20$. This difference-of-differences removes pre-pulse drift and their matched context response. Remaining cross-coordinate changes are bundled, so assigning the difference to one feedback path would require further targeted removals. All seven coordinates change during the nominal target-shift comparisons, as verified by their recorded spans. The additional constant-target arms retain the separate interpretation of the prescribed permeability drive.")
    direction_rows=report["directional_sensitivity"]
    coefficient_rows=[row for row in direction_rows if row["parameter"]=="a_conflict"]
    scale_rows=[row for row in direction_rows if row["parameter"]=="psi_scale"]
    coefficient_values=[row["opposed_minus_aligned_t10"] for row in coefficient_rows]
    scale_values=[row["opposed_minus_aligned_t10"] for row in scale_rows]
    lines.append(rf"Seven matched-direction endpoint comparisons examine four values of the conflict-path coefficient and three conflict-normalization scales $\Psi_0$. At $t=10$, opposed-minus-aligned contrasts range from ${min(coefficient_values):.4f}$ to ${max(coefficient_values):.4f}$ across the coefficient range, and from ${min(scale_values):.4f}$ to ${max(scale_values):.4f}$ across the normalization scales. The aligned reference can be reused exactly for these settings because its conflict signal remains zero and its stationary initial state is unchanged. Other parameter sweeps retain the separately defined context-control comparison.")
    lines.append(r"Upstream filtering uses $\bm\sigma=[0,0.5,0.5]^{\top}$ during the pulse. It attenuates the transmitted component while preserving direct source demands. The supplement reports both resource and experienced-load responses for this comparison. Each cost/benefit variant and sensitivity condition is initialized on its own declared stationary branch, so comparisons represent responses around their respective baselines.")
    lines.append(r"A continuous triangular demand cycle provides the matched rising and falling branches required by the path statistics. Demand marks rise from $0.15$ to $1$ and return to $0.15$ over half-cycle durations of $5$, $10$, or $20$ model-time units. Direction is held fixed within a cycle and each cycle starts at its own low-demand equilibrium. Signed and absolute areas are evaluated on a common external-demand grid by piecewise-linear branch interpolation; output endpoint gaps are retained explicitly.")
    lines += [r"\begin{table}[H]\centering\small",r"\begin{tabular}{@{}rrrr@{}}\toprule",r"Half-cycle duration & $\mathcal H_{r,d}$ & $\mathcal H_{\mathrm{abs}}$ & Endpoint $\Delta r$ \\\midrule"]
    for name,row in report["cycles"].items():
        if row["direction"]=="opposed":
            lines.append(f"{row['phase_duration']:g} & {row['signed']:.5f} & {row['absolute']:.5f} & {row['output_gap']:.5f} \\\\")
    lines += [r"\bottomrule\end{tabular}",r"\caption{Descriptive path separation for opposed triangular demand cycles. A vertical closure at the common low input contributes zero to $\int r\,\mathrm d d_{\mathrm{ext}}$; the output gap remains reported. These finite-rate comparisons do not identify rate-independent hysteresis.}\label{tab:cycle-statistics}\end{table}"]
    mantissa,exponent=f"{max_error:.2e}".split("e")
    max_area_error=max(row["absolute_dt_vs_dt2"] for row in report["cycle_convergence"].values())
    max_area_fine_error=max(row["absolute_dt2_vs_dt4"] for row in report["cycle_convergence"].values())
    area_mantissa,area_exponent=f"{max_area_error:.2e}".split("e")
    fine_mantissa,fine_exponent=f"{max_area_fine_error:.2e}".split("e")
    lines.append(rf"The branch-area estimator is checked separately from state integration. Across all six cycles, the maximum change in $\mathcal H_{{\mathrm{{abs}}}}$ is ${area_mantissa}\times10^{{{int(area_exponent)}}}$ between steps $0.02$ and $0.01$, and ${fine_mantissa}\times10^{{{int(fine_exponent)}}}$ between $0.01$ and $0.005$. The supplement also reports signed-area and endpoint-gap refinements. These interpolation/quadrature checks are distinct from the RK4 state-error comparison.")
    lines.append(rf"Classical RK4 uses step $0.02$, with steps split at all schedule boundaries and continuous forcing evaluated at each RK stage. Repeating at $0.01$ and $0.005$ from the same unrounded initial states gives a maximum common-grid state difference of ${mantissa}\times10^{{{int(exponent)}}}$ between the first two steps across pulse, control and sensitivity runs. Cycle refinements are reported separately. The seven pathway sweeps retain zero, half, nominal and twice-nominal values; three conflict-normalization scales $\Psi_0$ are also evaluated. Every declared condition is supplied, including adverse and timing-dependent outcomes.")
    text="\n\n".join(lines)+"\n"
    # Scientific notation is typeset explicitly in the prose, retaining raw values in JSON.
    import re
    text=re.sub(r"\$(\d(?:\.\d+)?)e([+-]?\d+)\$",lambda match:rf"${match[1]}\times10^{{{int(match[2])}}}$",text)
    path.write_text(text)


def matched_direction_sensitivity(report, sensitivity_scenarios, aligned_configuration):
    rows=[]
    baseline=report["scenarios"]["aligned"]
    expected={key:value for key,value in deepcopy(aligned_configuration).items() if key!="metadata"}
    for result in report["sensitivity"]:
        parameter=result["parameter"]
        if parameter not in ("a_conflict","psi_scale"):
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
    names=("model.py","network.py","analysis.py","configs/illustrative_baseline.json")
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    selected={"Experiment","raw_scenarios","matched_controls","summarize","add_outcomes","cycle_scenarios","run_numerical_experiment","cycle_refinement","matched_direction_sensitivity"}
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
                                    ("revision-results.tex","revision-results.tex")):
        source,target=out/source_name,manuscript/target_name
        if target.exists() and source.read_bytes()==target.read_bytes():
            continue
        if target.exists() and source.suffix==".pdf" and same_pdf_content(source,target):
            continue
        shutil.copy2(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/dynamics_protocol.json")
    parser.add_argument("--output",type=Path,default=ROOT/"dynamics-results")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--from-results",type=Path,help="regenerate presentation from matching verified numerical files")
    parser.add_argument("--manuscript-dir",type=Path,help="explicit manuscript-asset destination; requires --render")
    args=parser.parse_args()
    if args.manuscript_dir and not args.render:
        parser.error("--manuscript-dir requires --render")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol_path=args.protocol.resolve()
    protocol=json.loads(protocol_path.read_text())
    if protocol["version"]!=VERSION or protocol["solver"]!={"method":"classical_runge_kutta_4","dt":.02,"time_horizon":20}:
        raise ValueError("the declared pulse experiment requires version 0.3.0, RK4 step 0.02 and horizon 20")
    if args.from_results:
        report,scenarios,traces,cycles=load_saved_results(args.from_results.resolve(),out,protocol_path,protocol)
    else:
        report,scenarios,traces,cycles=run_numerical_experiment(protocol_path,protocol,out)
    figure_tex(traces,report,out/"mechanism_figure.tex")
    cycle_figure_tex(cycles,traces,report,out/"cycle_statistics.tex")
    results_tex(report,out/"revision-results.tex",scenarios)
    if args.render:
        for stem in ("mechanism_figure","cycle_statistics"):
            render_figure(out,stem)
    if args.manuscript_dir:
        sync_manuscript(out,args.manuscript_dir.resolve())
    print(json.dumps({"output":str(out),"version":VERSION,"reused_results":bool(args.from_results),
                      "primary_comparisons":len(report["primary_comparisons"]),"expanded_comparisons":len(report["scenarios"]),
                      "sensitivity_conditions":len(report["sensitivity"]),"cycles":len(report["cycles"]),
                      "maximum_state_refinement_difference":max(row["max_state_dt_vs_dt2"] for row in report["convergence"].values()),
                      "maximum_cycle_refinement_difference":max(row["max_state_dt_vs_dt2"] for row in report["cycle_convergence"].values()),
                      "direction_contrast":report["direction_contrast"],
                      "primary_coordinates_active":report["all_primary_coordinates_active"]},indent=2))


if __name__=="__main__":
    main()
