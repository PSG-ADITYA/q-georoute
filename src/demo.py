"""
Q-GeoRoute: Geometry-Aware Quantum Routing Simulator & Benchmark
Powered by Qiskit SDK + Qiskit Aer.

Two Distinct Modes:
1. CUSTOM SIMULATION:
   - Interactive source/target qubit routing across 3 processor topologies
   - Real Qiskit circuit construction, native SWAP shuttling, and Aer simulation
   - Live topology coupling graph with highlighted routing path & circuit diagram
2. OFFICIAL 9-CASE BENCHMARK:
   - "Live Qiskit Aer Benchmark — 9 Cases"
   - Executes the 3 Topologies × 3 Conditions benchmark matrix using src.benchmark
   - Clean 9-case results table (Fidelity, XX, YY, ZZ, SWAPs, 2Q Gates, Depth, Yield)
   - Key comparison charts (Fidelity, SWAPs/Gates, Depth, Protection Impact)

Modes of Execution:
- Desktop GUI (default): python src/demo.py
- Browser Web App:       python src/demo.py --web
- Interactive Terminal:  python src/demo.py --cli
- Official Benchmark:    python src/demo.py --benchmark
- Custom CLI Run:        python src/demo.py --topology "Hyperbolic Inspired" --source 6 --target 11 --condition Noisy
"""

import sys
import os
import argparse
import io
import json
import base64
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator

from src.topologies import Topology, get_all_topologies
from src.routing import build_routed_bell_circuit, RoutingMetrics
from src.noise import NoiseConfig, build_noise_model
from src.benchmark import run_single_condition, run_full_benchmark
from src.visualization import (
    generate_all_plots,
    plot_fidelity_comparison,
    plot_swap_and_gates_comparison,
    plot_circuit_depth_comparison,
    plot_protection_yield_and_fidelity,
)
from src.metrics import GeometryMetrics, BenchmarkComparison


# ==============================================================================
# MODE 1: CUSTOM SIMULATION DATA STRUCTURES & EXECUTION
# ==============================================================================

@dataclass
class DemoResult:
    """Result container for an interactive Q-GeoRoute simulation run."""
    topology_name: str
    source: int
    target: int
    condition: str
    shots: int
    path: List[int]
    hops: int
    swaps: int
    cx_count: int
    two_qubit_gates: int
    circuit_depth: int
    fidelity: float
    xx: float
    yy: float
    zz: float
    survival_yield: Optional[float]
    circuit_diagram: str
    routing_metrics: RoutingMetrics


def get_topology_by_name(topology_name: str) -> Topology:
    """Retrieves a baseline Topology object by name."""
    topos = get_all_topologies()
    if topology_name not in topos:
        available = list(topos.keys())
        raise ValueError(f"Unknown topology '{topology_name}'. Available: {available}")
    return topos[topology_name]


def get_default_endpoints(topology_name: str) -> Tuple[int, int]:
    """Returns official benchmark default (source, destination) for a topology."""
    topo = get_topology_by_name(topology_name)
    return topo.source, topo.destination


def validate_demo_inputs(
    topology_name: str,
    source_val: Any,
    target_val: Any,
    condition: str,
    shots_val: Any,
) -> Tuple[bool, Optional[str], Optional[Topology], Optional[int]]:
    """
    Validates user inputs for demo simulation without raising unhandled exceptions.

    Validation rules:
    1. Topology name must be valid.
    2. Condition must be one of: Ideal, Noisy, Noisy + Protection.
    3. Shots must be a positive integer.
    4. Source and target must be valid integer qubit IDs within processor range.
    5. Source and target cannot be identical.
    6. Endpoints cannot be directly adjacent (Q-GeoRoute requires hops >= 2 for SWAP routing).
    7. A connected path must exist between source and target.

    Returns:
        (is_valid, error_message, custom_topology_instance, shots_int)
    """
    # 1. Validate topology
    topos = get_all_topologies()
    if topology_name not in topos:
        return False, f"Invalid topology '{topology_name}'. Choose from: {list(topos.keys())}.", None, None
    base_topo = topos[topology_name]

    # 2. Validate condition
    valid_conditions = ["Ideal", "Noisy", "Noisy + Protection"]
    if condition not in valid_conditions:
        return False, f"Invalid condition '{condition}'. Choose from: {valid_conditions}.", None, None

    # 3. Validate shots
    try:
        shots = int(shots_val)
        if shots <= 0:
            return False, "Shots must be a positive integer (e.g. 20000).", None, None
    except (ValueError, TypeError):
        return False, "Shots must be a valid integer (e.g. 20000).", None, None

    # 4. Validate source & target integers
    try:
        source = int(source_val)
        target = int(target_val)
    except (ValueError, TypeError):
        return False, "Source and Target qubit IDs must be valid integers.", None, None

    # Range check
    num_q = base_topo.num_qubits
    if source < 0 or source >= num_q:
        return False, f"Source qubit Q{source} is out of range for '{topology_name}' (valid qubits: 0 to {num_q - 1}).", None, None
    if target < 0 or target >= num_q:
        return False, f"Target qubit Q{target} is out of range for '{topology_name}' (valid qubits: 0 to {num_q - 1}).", None, None

    # 5. Distinct check
    if source == target:
        return False, f"Source and target qubits cannot be identical (Q{source} == Q{target}). Please select distinct qubits.", None, None

    # 6. Non-adjacent check
    if base_topo.graph.has_edge(source, target):
        return (
            False,
            f"Endpoints Q{source} and Q{target} are directly adjacent (1 hop). "
            f"Q-GeoRoute is designed for remote quantum routing across non-adjacent endpoints (hops >= 2). "
            f"Please select non-adjacent qubits to demonstrate SWAP routing and syndrome protection.",
            None,
            None,
        )

    # 7. Connectivity check
    if not nx.has_path(base_topo.graph, source, target):
        return False, f"No connected routing path exists between Q{source} and Q{target} in '{topology_name}'.", None, None

    # Construct and validate custom Topology instance
    custom_topo = Topology(
        name=base_topo.name,
        graph=base_topo.graph,
        num_qubits=base_topo.num_qubits,
        edge_list=base_topo.edge_list,
        source=source,
        destination=target,
        pos=base_topo.pos,
        description=base_topo.description,
    )

    try:
        custom_topo.validate()
    except ValueError as ve:
        return False, str(ve), None, None

    return True, None, custom_topo, shots


def simulate_demo(
    topology_name: str,
    source: int,
    target: int,
    condition: str,
    shots: int = 20000,
    seed: int = 42,
) -> DemoResult:
    """
    Executes the genuine Q-GeoRoute simulation pipeline using Qiskit SDK and Qiskit Aer.

    Flow:
    User input
    -> Q-GeoRoute topology & shortest-path routing
    -> Qiskit QuantumCircuit construction & SWAP insertion
    -> Qiskit Aer AerSimulator execution (with noise model when selected)
    -> Tomographic measurement (X, Y, Z bases)
    -> Pauli expectation values (<XX>, <YY>, <ZZ>) & Bell fidelity
    -> Intermediate qubit stabilizer post-selection & survival yield (if protected)
    """
    is_valid, err_msg, custom_topo, valid_shots = validate_demo_inputs(
        topology_name=topology_name,
        source_val=source,
        target_val=target,
        condition=condition,
        shots_val=shots,
    )
    if not is_valid or custom_topo is None or valid_shots is None:
        raise ValueError(err_msg or "Invalid inputs provided.")

    # Select simulator backend and noise model
    if condition == "Ideal":
        sim = AerSimulator()
    else:
        noise_cfg = NoiseConfig(seed=seed)
        noise_model = build_noise_model(noise_cfg)
        sim = AerSimulator(noise_model=noise_model)

    # Run genuine tomographic evaluation via benchmark engine
    fid, xx, yy, zz, mean_yield, rm = run_single_condition(
        topology=custom_topo,
        condition=condition,
        sim=sim,
        shots=valid_shots,
        seed=seed,
    )

    # Generate representative routed Qiskit circuit diagram
    with_protection = (condition == "Noisy + Protection")
    qc, _ = build_routed_bell_circuit(
        custom_topo,
        measurement_basis="Z",
        with_protection_checks=with_protection,
    )
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            circuit_diagram = str(qc.draw(output="text"))
    except Exception:
        circuit_diagram = f"Qiskit QuantumCircuit with {qc.num_qubits} qubits and {len(qc.data)} operations."

    return DemoResult(
        topology_name=topology_name,
        source=source,
        target=target,
        condition=condition,
        shots=valid_shots,
        path=rm.path,
        hops=rm.path_length,
        swaps=rm.num_swaps,
        cx_count=rm.num_cx,
        two_qubit_gates=rm.total_two_qubit_gates,
        circuit_depth=rm.circuit_depth,
        fidelity=float(fid),
        xx=float(xx),
        yy=float(yy),
        zz=float(zz),
        survival_yield=float(mean_yield) if with_protection else None,
        circuit_diagram=circuit_diagram,
        routing_metrics=rm,
    )


def create_topology_figure(
    topology: Topology,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.5, 5.0),
) -> plt.Figure:
    """
    Generates a Matplotlib figure visualizing the processor coupling graph,
    highlighting the source qubit, target qubit, and the actual routing path.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    g = topology.graph
    pos = topology.pos
    path = topology.shortest_path()
    path_edges = list(zip(path[:-1], path[1:]))

    # Base coupling edges
    nx.draw_networkx_edges(
        g, pos,
        ax=ax,
        edge_color="#94a3b8",
        width=1.8,
        alpha=0.6,
    )

    # Highlighted shortest routing path edges
    nx.draw_networkx_edges(
        g, pos,
        edgelist=path_edges,
        ax=ax,
        edge_color="#e41a1c",
        width=3.5,
        alpha=0.9,
    )

    # Node coloring
    node_colors = []
    for node in g.nodes():
        if node == topology.source:
            node_colors.append("#2ca02c")      # Source: Green
        elif node == topology.destination:
            node_colors.append("#e7298a")      # Destination: Magenta
        elif node in path:
            node_colors.append("#fd8d3c")      # Intermediate route: Orange
        else:
            node_colors.append("#cbd5e1")      # Idle qubits: Light slate

    nx.draw_networkx_nodes(
        g, pos,
        ax=ax,
        node_color=node_colors,
        node_size=550,
        edgecolors="#1e293b",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        g, pos,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="#0f172a",
    )

    path_str = " -> ".join(f"Q{n}" for n in path)
    plot_title = title or f"{topology.name} Coupling Architecture"
    ax.set_title(
        f"{plot_title}\nSource: Q{topology.source} -> Target: Q{topology.destination} | Route: {path_str}",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax.axis("off")

    custom_lines = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label=f"Source (Q{topology.source})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e7298a", markersize=9, label=f"Target (Q{topology.destination})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fd8d3c", markersize=9, label="Route Qubits"),
        plt.Line2D([0], [0], color="#e41a1c", lw=2.5, label=f"Shortest Path ({len(path)-1} hops)"),
    ]
    ax.legend(handles=custom_lines, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def render_topology_base64(topology: Topology) -> str:
    """Renders the topology graph to a Base64-encoded PNG string."""
    fig = create_topology_figure(topology)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# MODE 2: OFFICIAL 9-CASE BENCHMARK EXECUTION & PLOTTING
# ==============================================================================

def run_official_benchmark_demo(
    shots: int = 20000,
    seed: int = 42,
    output_csv_path: str = "results/benchmark_results.csv",
    plots_dir: str = "results/plots/",
) -> Tuple[pd.DataFrame, Dict[str, GeometryMetrics], Dict[str, BenchmarkComparison]]:
    """
    Executes the genuine official 9-case benchmark matrix using Qiskit Aer.
    Reuses existing src.benchmark.run_full_benchmark and src.visualization.generate_all_plots.

    Returns:
        (df_results, geom_metrics, comparisons)
    """
    noise_cfg = NoiseConfig(seed=seed)
    df_results, geom_metrics, comparisons = run_full_benchmark(
        noise_config=noise_cfg,
        shots=shots,
        seed=seed,
        output_csv_path=output_csv_path,
    )
    generate_all_plots(df_results, output_dir=plots_dir)
    return df_results, geom_metrics, comparisons


def get_benchmark_plot_base64(plot_filename: str, plots_dir: str = "results/plots/") -> Optional[str]:
    """Reads a generated benchmark comparison plot and encodes it to base64."""
    path = os.path.join(plots_dir, plot_filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==============================================================================
# DESKTOP GUI IMPLEMENTATION (Tkinter + Matplotlib Canvas)
# ==============================================================================

def run_gui() -> None:
    """Launches the native Python Tkinter desktop GUI simulator with both modes."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("Q-GeoRoute — Quantum Routing Simulator & Benchmark")
    root.geometry("1200x840")
    root.minsize(1020, 740)
    root.configure(bg="#f8fafc")

    topos = get_all_topologies()
    topo_names = list(topos.keys())

    # Header frame
    header_frame = tk.Frame(root, bg="#1e293b", padx=20, pady=12)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(
        header_frame,
        text="Q-GeoRoute — Quantum Routing Simulator & Benchmark",
        font=("Helvetica", 16, "bold"),
        fg="#ffffff",
        bg="#1e293b",
    )
    title_label.pack(anchor="w")

    subtitle_label = tk.Label(
        header_frame,
        text="Powered by Qiskit SDK + Qiskit Aer | Geometry-Aware Remote Entanglement & Noise Benchmarking",
        font=("Helvetica", 9),
        fg="#94a3b8",
        bg="#1e293b",
    )
    subtitle_label.pack(anchor="w")

    # Main Notebook (Two clearly separated modes)
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    # --------------------------------------------------------------------------
    # TAB 1: CUSTOM SIMULATION
    # --------------------------------------------------------------------------
    tab_custom = tk.Frame(notebook, bg="#f8fafc")
    notebook.add(tab_custom, text="  ⚡ Custom Simulation  ")

    # State variables for custom mode
    current_topo_var = tk.StringVar(value=topo_names[2])  # Default to Hyperbolic
    src_var = tk.StringVar()
    dst_var = tk.StringVar()
    cond_var = tk.StringVar(value="Noisy")
    shots_var = tk.StringVar(value="20000")
    status_var = tk.StringVar(value="Ready. Select parameters and click 'RUN SIMULATION'.")

    custom_paned = tk.PanedWindow(tab_custom, orient=tk.HORIZONTAL, bg="#e2e8f0", sashwidth=4)
    custom_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Left Column: Controls & Metrics
    left_frame = tk.Frame(custom_paned, bg="#ffffff", padx=14, pady=12, relief=tk.RIDGE, bd=1)
    custom_paned.add(left_frame, minsize=420, width=450)

    ctrl_group = tk.LabelFrame(left_frame, text=" Simulation Parameters ", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b", padx=10, pady=10)
    ctrl_group.pack(fill=tk.X, pady=(0, 10))

    tk.Label(ctrl_group, text="Processor Topology:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=4)
    topo_combo = ttk.Combobox(ctrl_group, textvariable=current_topo_var, values=topo_names, state="readonly", width=22)
    topo_combo.grid(row=0, column=1, sticky="ew", pady=4, padx=5)

    tk.Label(ctrl_group, text="Source Qubit:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=1, column=0, sticky="w", pady=4)
    src_combo = ttk.Combobox(ctrl_group, textvariable=src_var, state="readonly", width=22)
    src_combo.grid(row=1, column=1, sticky="ew", pady=4, padx=5)

    tk.Label(ctrl_group, text="Target Qubit:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=2, column=0, sticky="w", pady=4)
    dst_combo = ttk.Combobox(ctrl_group, textvariable=dst_var, state="readonly", width=22)
    dst_combo.grid(row=2, column=1, sticky="ew", pady=4, padx=5)

    tk.Label(ctrl_group, text="Condition:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=3, column=0, sticky="w", pady=4)
    cond_combo = ttk.Combobox(ctrl_group, textvariable=cond_var, values=["Ideal", "Noisy", "Noisy + Protection"], state="readonly", width=22)
    cond_combo.grid(row=3, column=1, sticky="ew", pady=4, padx=5)

    tk.Label(ctrl_group, text="Shots:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=4, column=0, sticky="w", pady=4)
    shots_entry = ttk.Entry(ctrl_group, textvariable=shots_var, width=24)
    shots_entry.grid(row=4, column=1, sticky="ew", pady=4, padx=5)

    def update_qubit_choices(*args):
        t_name = current_topo_var.get()
        if t_name in topos:
            t = topos[t_name]
            q_choices = [f"Q{i}" for i in range(t.num_qubits)]
            src_combo["values"] = q_choices
            dst_combo["values"] = q_choices
            src_var.set(f"Q{t.source}")
            dst_var.set(f"Q{t.destination}")
            render_custom_canvas(t)

    current_topo_var.trace_add("write", update_qubit_choices)

    btn_frame = tk.Frame(ctrl_group, bg="#ffffff", pady=6)
    btn_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 2))

    run_custom_btn = tk.Button(
        btn_frame,
        text="▶ RUN SIMULATION",
        font=("Helvetica", 10, "bold"),
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        cursor="hand2",
        padx=12,
        pady=5,
    )
    run_custom_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    reset_btn = tk.Button(
        btn_frame,
        text="Reset Defaults",
        font=("Helvetica", 9),
        bg="#e2e8f0",
        fg="#334155",
        cursor="hand2",
        padx=10,
        pady=5,
        command=lambda: update_qubit_choices(),
    )
    reset_btn.pack(side=tk.RIGHT, padx=(4, 0))

    status_label = tk.Label(
        left_frame,
        textvariable=status_var,
        font=("Helvetica", 8, "italic"),
        fg="#475569",
        bg="#f1f5f9",
        padx=8,
        pady=6,
        wraplength=410,
        justify=tk.LEFT,
    )
    status_label.pack(fill=tk.X, pady=(0, 10))

    results_group = tk.LabelFrame(left_frame, text=" Live Quantum Simulation Metrics ", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b", padx=10, pady=10)
    results_group.pack(fill=tk.BOTH, expand=True)

    results_text = ScrolledText(results_group, font=("Consolas", 10), bg="#f8fafc", fg="#0f172a", relief=tk.FLAT, height=18)
    results_text.pack(fill=tk.BOTH, expand=True)
    results_text.insert(tk.END, "Configure parameters and click 'RUN SIMULATION' to evaluate real Bell-state fidelity.")
    results_text.config(state=tk.DISABLED)

    # Right Column: Graph & Circuit
    right_paned = tk.PanedWindow(custom_paned, orient=tk.VERTICAL, bg="#e2e8f0", sashwidth=4)
    custom_paned.add(right_paned, minsize=520, width=680)

    plot_frame = tk.Frame(right_paned, bg="#ffffff", padx=8, pady=8, relief=tk.RIDGE, bd=1)
    right_paned.add(plot_frame, minsize=380, height=430)

    tk.Label(plot_frame, text="Topology & Shortest Routing Path Visualization", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 4))
    canvas_container = tk.Frame(plot_frame, bg="#ffffff")
    canvas_container.pack(fill=tk.BOTH, expand=True)

    current_canvas = [None]

    def render_custom_canvas(t_obj: Topology):
        if current_canvas[0] is not None:
            current_canvas[0].get_tk_widget().destroy()
            current_canvas[0] = None
        fig = create_topology_figure(t_obj, figsize=(6.2, 4.3))
        canvas = FigureCanvasTkAgg(fig, master=canvas_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        current_canvas[0] = canvas
        plt.close(fig)

    circuit_frame = tk.Frame(right_paned, bg="#ffffff", padx=8, pady=8, relief=tk.RIDGE, bd=1)
    right_paned.add(circuit_frame, minsize=220, height=260)

    tk.Label(circuit_frame, text="Routed Qiskit QuantumCircuit (Generated Native SWAP Insertion)", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 4))
    circuit_text = ScrolledText(circuit_frame, font=("Courier New", 9), bg="#0f172a", fg="#38bdf8", relief=tk.FLAT, wrap=tk.NONE)
    circuit_text.pack(fill=tk.BOTH, expand=True)
    circuit_text.insert(tk.END, "# Generated Qiskit circuit with SWAP routing and tomographic measurement will appear here.")
    circuit_text.config(state=tk.DISABLED)

    def on_custom_run():
        t_name = current_topo_var.get()
        s_str = src_var.get().replace("Q", "").strip()
        d_str = dst_var.get().replace("Q", "").strip()
        cond = cond_var.get()
        shots_str = shots_var.get().strip()

        is_valid, err_msg, custom_topo, shots_int = validate_demo_inputs(
            topology_name=t_name,
            source_val=s_str,
            target_val=d_str,
            condition=cond,
            shots_val=shots_str,
        )

        if not is_valid:
            status_var.set(f"❌ Validation Error: {err_msg}")
            status_label.config(fg="#dc2626", bg="#fef2f2")
            messagebox.showwarning("Invalid Input", err_msg)
            return

        status_var.set("⏳ Running real Qiskit/Aer simulation across Pauli bases...")
        status_label.config(fg="#2563eb", bg="#eff6ff")
        root.update_idletasks()

        try:
            res = simulate_demo(
                topology_name=t_name,
                source=int(s_str),
                target=int(d_str),
                condition=cond,
                shots=shots_int,
                seed=42,
            )

            status_var.set(f"✔ Completed successfully! Bell Fidelity: {res.fidelity:.4f}")
            status_label.config(fg="#16a34a", bg="#f0fdf4")

            results_text.config(state=tk.NORMAL)
            results_text.delete("1.0", tk.END)

            path_str = " -> ".join(f"Q{n}" for n in res.path)
            lines = [
                f"Topology:       {res.topology_name}",
                f"Source:         Q{res.source}",
                f"Target:         Q{res.target}",
                f"Condition:      {res.condition}",
                "",
                f"Path:",
                f"{path_str}",
                "",
                f"Hops:           {res.hops}",
                f"SWAPs:          {res.swaps}",
                f"2Q Gates:       {res.two_qubit_gates}",
                f"Circuit Depth:  {res.circuit_depth}",
                f"Bell Fidelity:  {res.fidelity:.4f}",
                "",
                f"Pauli Measurements:",
                f"  <XX>: {res.xx:+.4f}",
                f"  <YY>: {res.yy:+.4f}",
                f"  <ZZ>: {res.zz:+.4f}",
            ]
            if res.survival_yield is not None:
                lines.extend([
                    "",
                    f"Survival Yield: {res.survival_yield * 100:.1f}%",
                    f"(Stabilizer syndrome post-selection filtered {100 - res.survival_yield * 100:.1f}% error shots)",
                ])

            results_text.insert(tk.END, "\n".join(lines))
            results_text.config(state=tk.DISABLED)

            circuit_text.config(state=tk.NORMAL)
            circuit_text.delete("1.0", tk.END)
            circuit_text.insert(tk.END, res.circuit_diagram)
            circuit_text.config(state=tk.DISABLED)

            render_custom_canvas(custom_topo)

        except Exception as ex:
            status_var.set(f"❌ Execution Error: {str(ex)}")
            status_label.config(fg="#dc2626", bg="#fef2f2")
            messagebox.showerror("Simulation Error", f"Simulation failed: {str(ex)}")

    run_custom_btn.config(command=on_custom_run)

    # --------------------------------------------------------------------------
    # TAB 2: OFFICIAL 9-CASE BENCHMARK
    # --------------------------------------------------------------------------
    tab_benchmark = tk.Frame(notebook, bg="#f8fafc")
    notebook.add(tab_benchmark, text="  📊 Official 9-Case Benchmark  ")

    bench_top_frame = tk.Frame(tab_benchmark, bg="#ffffff", padx=16, pady=12, relief=tk.RIDGE, bd=1)
    bench_top_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

    bench_heading = tk.Label(
        bench_top_frame,
        text="Live Qiskit Aer Benchmark — 9 Cases",
        font=("Helvetica", 14, "bold"),
        fg="#1e293b",
        bg="#ffffff",
    )
    bench_heading.pack(anchor="w")

    bench_subheading = tk.Label(
        bench_top_frame,
        text="3 Topologies (5Q Star, Heavy-Hex, Hyperbolic) × 3 Conditions (Ideal, Noisy, Noisy + Protection) | 20,000 shots/basis",
        font=("Helvetica", 9),
        fg="#64748b",
        bg="#ffffff",
    )
    bench_subheading.pack(anchor="w", pady=(2, 8))

    bench_action_bar = tk.Frame(bench_top_frame, bg="#ffffff")
    bench_action_bar.pack(fill=tk.X)

    run_bench_btn = tk.Button(
        bench_action_bar,
        text="▶ RUN OFFICIAL 9-CASE BENCHMARK",
        font=("Helvetica", 11, "bold"),
        bg="#16a34a",
        fg="#ffffff",
        activebackground="#15803d",
        activeforeground="#ffffff",
        cursor="hand2",
        padx=16,
        pady=7,
    )
    run_bench_btn.pack(side=tk.LEFT)

    bench_status_var = tk.StringVar(value="Click the button above to execute the live 9-case benchmark matrix with Qiskit Aer.")
    bench_status_label = tk.Label(
        bench_action_bar,
        textvariable=bench_status_var,
        font=("Helvetica", 9, "italic"),
        fg="#475569",
        bg="#f1f5f9",
        padx=12,
        pady=6,
    )
    bench_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

    # Split: Benchmark Table (Left/Top) & Comparison Charts (Right/Bottom)
    bench_split = tk.PanedWindow(tab_benchmark, orient=tk.VERTICAL, bg="#e2e8f0", sashwidth=4)
    bench_split.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # Table Frame
    table_frame = tk.Frame(bench_split, bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, bd=1)
    bench_split.add(table_frame, minsize=240, height=270)

    tk.Label(table_frame, text="Benchmark Results Matrix (9 Experimental Conditions)", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b").pack(anchor="w", pady=(0, 6))

    cols = ("Topology", "Condition", "Fidelity", "XX", "YY", "ZZ", "SWAPs", "2Q Gates", "Depth", "Survival Yield")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=9)
    for col in cols:
        tree.heading(col, text=col)
        col_w = 140 if col in ("Topology", "Condition") else 85
        tree.column(col, width=col_w, anchor="center")

    tree.tag_configure("ideal", background="#eff6ff")
    tree.tag_configure("noisy", background="#fff7ed")
    tree.tag_configure("protected", background="#f0fdf4")

    tree.pack(fill=tk.BOTH, expand=True)

    # Charts & Summary Frame
    charts_frame = tk.Frame(bench_split, bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, bd=1)
    bench_split.add(charts_frame, minsize=320, height=360)

    chart_ctrl_bar = tk.Frame(charts_frame, bg="#ffffff")
    chart_ctrl_bar.pack(fill=tk.X, pady=(0, 6))

    tk.Label(chart_ctrl_bar, text="Comparison Chart:", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b").pack(side=tk.LEFT, padx=(0, 8))

    chart_options = [
        "Fidelity Comparison (Ideal vs Noisy vs Protected)",
        "Routing Overhead (SWAPs & 2Q Gates)",
        "Routed Quantum Circuit Depth",
        "Protection Impact & Survival Yield",
        "Fidelity vs Routing Overhead Frontier",
    ]
    chart_select_var = tk.StringVar(value=chart_options[0])
    chart_combo = ttk.Combobox(chart_ctrl_bar, textvariable=chart_select_var, values=chart_options, state="readonly", width=42)
    chart_combo.pack(side=tk.LEFT)

    chart_canvas_box = tk.Frame(charts_frame, bg="#ffffff")
    chart_canvas_box.pack(fill=tk.BOTH, expand=True)

    chart_canvas_ref = [None]
    benchmark_df_cache = [None]

    def render_benchmark_chart(*args):
        if benchmark_df_cache[0] is None:
            return
        df = benchmark_df_cache[0]
        choice = chart_select_var.get()

        if chart_canvas_ref[0] is not None:
            chart_canvas_ref[0].get_tk_widget().destroy()
            chart_canvas_ref[0] = None

        fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=95)
        if choice == chart_options[0]:
            topos_list = df["Topology"].unique()
            x = np.arange(len(topos_list))
            w = 0.25
            id_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Ideal")]["Fidelity"].values[0] for t in topos_list]
            no_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy")]["Fidelity"].values[0] for t in topos_list]
            pr_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Fidelity"].values[0] for t in topos_list]
            ax.bar(x - w, id_vals, w, label="Ideal", color="#2b5c8f", edgecolor="black")
            ax.bar(x, no_vals, w, label="Noisy", color="#d95f02", edgecolor="black")
            ax.bar(x + w, pr_vals, w, label="Noisy + Protection", color="#2ca02c", edgecolor="black")
            ax.set_ylabel("Fidelity F(|Phi+>)")
            ax.set_xticks(x)
            ax.set_xticklabels(topos_list, fontweight="bold")
            ax.set_ylim(0.0, 1.15)
            ax.legend(loc="upper right")
            ax.set_title("Remote Bell-State Fidelity Comparison across Topologies", pad=8)
        elif choice == chart_options[1]:
            df_id = df[df["Condition"] == "Ideal"]
            topos_list = df_id["Topology"].tolist()
            swaps = df_id["SWAPs"].tolist()
            two_q = df_id["2Q Gates"].tolist()
            x = np.arange(len(topos_list))
            w = 0.35
            ax.bar(x - w/2, swaps, w, label="SWAPs", color="#807dba", edgecolor="black")
            ax.bar(x + w/2, two_q, w, label="Total 2Q Gates", color="#41b6c4", edgecolor="black")
            ax.set_ylabel("Gate Count")
            ax.set_xticks(x)
            ax.set_xticklabels(topos_list, fontweight="bold")
            ax.legend()
            ax.set_title("Routing Gate Overhead by Architecture", pad=8)
        elif choice == chart_options[2]:
            df_id = df[df["Condition"] == "Ideal"]
            topos_list = df_id["Topology"].tolist()
            depths = df_id["Depth"].tolist()
            bars = ax.bar(topos_list, depths, color=["#7fc97f", "#beaed4", "#fdc086"], width=0.45, edgecolor="black")
            ax.set_ylabel("DAG Depth")
            ax.set_title("Routed Circuit Depth Comparison", pad=8)
            for b in bars:
                ax.annotate(str(int(b.get_height())), xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
        elif choice == chart_options[3]:
            topos_list = df["Topology"].unique()
            x = np.arange(len(topos_list))
            w = 0.3
            no_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy")]["Fidelity"].values[0] for t in topos_list]
            pr_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Fidelity"].values[0] for t in topos_list]
            yields = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Survival Yield"].values[0] * 100 for t in topos_list]
            ax.bar(x - w/2, no_vals, w, label="Noisy Fidelity", color="#d95f02", edgecolor="black")
            ax.bar(x + w/2, pr_vals, w, label="Protected Fidelity", color="#2ca02c", edgecolor="black")
            ax.set_ylabel("Fidelity")
            ax.set_ylim(0.75, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(topos_list, fontweight="bold")
            ax2 = ax.twinx()
            ax2.plot(x, yields, color="#7570b3", marker="D", lw=2, label="Survival Yield (%)")
            ax2.set_ylabel("Survival Yield (%)", color="#7570b3")
            ax2.set_ylim(70, 105)
            ax.set_title("Protection Efficacy vs Post-Selection Yield", pad=8)
        else:
            topos_list = df["Topology"].unique()
            colors = {"5Q Star": "#1f78b4", "Heavy-Hex Inspired": "#e31a1c", "Hyperbolic Inspired": "#33a02c"}
            for t in topos_list:
                row_n = df[(df["Topology"] == t) & (df["Condition"] == "Noisy")].iloc[0]
                row_p = df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")].iloc[0]
                g = row_n["2Q Gates"]
                ax.scatter(g, row_n["Fidelity"], color=colors.get(t, "#333"), s=140, edgecolor="black", label=f"{t} (Noisy)")
                ax.scatter(g, row_p["Fidelity"], color=colors.get(t, "#333"), s=160, facecolors="none", lw=2, label=f"{t} (Protected)")
            ax.set_xlabel("Two-Qubit Gates along Path")
            ax.set_ylabel("Bell Fidelity")
            ax.set_title("Fidelity vs Routing Overhead Trade-off Frontier", pad=8)
            ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=chart_canvas_box)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        chart_canvas_ref[0] = c
        plt.close(fig)

    chart_select_var.trace_add("write", render_benchmark_chart)

    def on_run_official_benchmark():
        bench_status_var.set("⏳ Running official 9-case benchmark matrix with Qiskit Aer (20,000 shots/basis)...")
        bench_status_label.config(fg="#2563eb", bg="#eff6ff")
        root.update_idletasks()

        try:
            df, geom, comp = run_official_benchmark_demo(shots=20000, seed=42)
            benchmark_df_cache[0] = df

            # Populate Treeview
            for item in tree.get_children():
                tree.delete(item)

            for _, row in df.iterrows():
                cond = row["Condition"]
                tag = "ideal" if cond == "Ideal" else ("noisy" if cond == "Noisy" else "protected")
                tree.insert("", tk.END, values=(
                    row["Topology"],
                    cond,
                    f"{row['Fidelity']:.4f}",
                    f"{row['XX']:+.4f}",
                    f"{row['YY']:+.4f}",
                    f"{row['ZZ']:+.4f}",
                    row["SWAPs"],
                    row["2Q Gates"],
                    row["Depth"],
                    f"{row['Survival Yield']*100:.1f}%" if cond == "Noisy + Protection" else "100.0%",
                ), tags=(tag,))

            bench_status_var.set("✔ Official 9-Case Benchmark completed! Results saved to 'results/benchmark_results.csv'.")
            bench_status_label.config(fg="#16a34a", bg="#f0fdf4")

            render_benchmark_chart()

        except Exception as ex:
            bench_status_var.set(f"❌ Benchmark Error: {str(ex)}")
            bench_status_label.config(fg="#dc2626", bg="#fef2f2")
            messagebox.showerror("Benchmark Error", str(ex))

    run_bench_btn.config(command=on_run_official_benchmark)

    # Initialize custom endpoints
    update_qubit_choices()

    root.mainloop()


# ==============================================================================
# LOCAL BROWSER / WEB SERVER IMPLEMENTATION (Pure Python http.server)
# ==============================================================================

WEB_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-GeoRoute — Quantum Routing Simulator & Benchmark</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
            min-height: 100vh;
        }
        .header {
            max-width: 1240px;
            margin: 0 auto 20px auto;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            flex-wrap: wrap;
            gap: 12px;
        }
        .header h1 {
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
        }
        .header .badge {
            color: var(--accent-blue);
            font-size: 13px;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.1);
            padding: 5px 12px;
            border-radius: 6px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .nav-tabs {
            max-width: 1240px;
            margin: 0 auto 20px auto;
            display: flex;
            gap: 10px;
        }
        .tab-btn {
            background: #1e293b;
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: #ffffff;
            border-color: var(--accent-blue);
        }
        .tab-btn.active {
            background: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
        }
        .container {
            max-width: 1240px;
            margin: 0 auto;
        }
        .custom-grid {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
        }
        @media (max-width: 960px) {
            .custom-grid { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .card h2 {
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 16px;
            font-weight: 600;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 8px;
        }
        .form-group {
            margin-bottom: 14px;
        }
        .form-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .form-control {
            width: 100%;
            padding: 9px 12px;
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 6px;
            color: #ffffff;
            font-size: 14px;
            outline: none;
            transition: border 0.15s;
        }
        .form-control:focus { border-color: var(--accent-blue); }
        .btn-run {
            width: 100%;
            padding: 12px;
            background: #2563eb;
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 10px;
        }
        .btn-run:hover { background: #1d4ed8; }
        .btn-run:disabled { background: #475569; cursor: not-allowed; }
        .btn-benchmark {
            background: #16a34a;
            color: #ffffff;
            font-size: 15px;
            font-weight: 700;
            border: none;
            border-radius: 6px;
            padding: 12px 24px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-benchmark:hover { background: #15803d; }
        .btn-benchmark:disabled { background: #475569; cursor: not-allowed; }
        .alert {
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 13px;
            margin-top: 12px;
            display: none;
        }
        .alert-error {
            background: rgba(248, 113, 113, 0.15);
            border: 1px solid var(--accent-red);
            color: #fca5a5;
        }
        .alert-success {
            background: rgba(74, 222, 128, 0.15);
            border: 1px solid var(--accent-green);
            color: #86efac;
        }
        .alert-info {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid var(--accent-blue);
            color: #7dd3fc;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .metric-box {
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .metric-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }
        .metric-value.highlight { color: var(--accent-green); }
        .metric-value.blue { color: var(--accent-blue); }
        .path-banner {
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 14px;
            font-family: monospace;
            font-size: 14px;
            color: var(--accent-blue);
            margin-bottom: 16px;
            word-break: break-all;
        }
        .plot-container {
            text-align: center;
            background: #ffffff;
            border-radius: 8px;
            padding: 12px;
            overflow: hidden;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        pre.circuit-box {
            background: #090d16;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 14px;
            color: #38bdf8;
            font-family: "Courier New", Courier, monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.25;
            max-height: 280px;
        }
        /* Benchmark Table */
        .bench-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 14px;
        }
        .bench-table th, .bench-table td {
            padding: 10px 12px;
            text-align: center;
            border: 1px solid var(--card-border);
        }
        .bench-table th {
            background: #090d16;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        .bench-table tr:hover { background: rgba(255, 255, 255, 0.03); }
        .badge-ideal { background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .badge-noisy { background: rgba(251, 191, 36, 0.15); color: #fbbf24; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .badge-protected { background: rgba(74, 222, 128, 0.15); color: #4ade80; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
            margin-top: 16px;
        }
        @media (max-width: 600px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Q-GeoRoute — Quantum Routing Simulator & Benchmark</h1>
            <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Geometry-Aware Quantum Routing & Bell-State Benchmarking across Processor Topologies</p>
        </div>
        <div class="badge">Powered by Qiskit SDK + Qiskit Aer</div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
        <button id="tab-custom-btn" class="tab-btn active" onclick="switchTab('custom')">⚡ Custom Simulation</button>
        <button id="tab-bench-btn" class="tab-btn" onclick="switchTab('benchmark')">📊 Official 9-Case Benchmark</button>
    </div>

    <div class="container">
        <!-- ========================================== -->
        <!-- VIEW 1: CUSTOM SIMULATION                 -->
        <!-- ========================================== -->
        <div id="view-custom" class="custom-grid">
            <div>
                <div class="card">
                    <h2>Simulation Parameters</h2>
                    <div class="form-group">
                        <label for="topo-select">Topology:</label>
                        <select id="topo-select" class="form-control" onchange="onTopologyChange()">
                            <option value="5Q Star">5Q Star (5 Qubits)</option>
                            <option value="Heavy-Hex Inspired">Heavy-Hex Inspired (14 Qubits)</option>
                            <option value="Hyperbolic Inspired" selected>Hyperbolic Inspired (16 Qubits)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="src-select">Source Qubit:</label>
                        <select id="src-select" class="form-control"></select>
                    </div>

                    <div class="form-group">
                        <label for="dst-select">Target Qubit:</label>
                        <select id="dst-select" class="form-control"></select>
                    </div>

                    <div class="form-group">
                        <label for="cond-select">Condition:</label>
                        <select id="cond-select" class="form-control">
                            <option value="Ideal">Ideal Baseline</option>
                            <option value="Noisy" selected>Noisy Baseline</option>
                            <option value="Noisy + Protection">Noisy + Protection (Stabilizer Post-Selection)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="shots-input">Shots:</label>
                        <input type="number" id="shots-input" class="form-control" value="20000" min="1" step="1000">
                    </div>

                    <button id="run-btn" class="btn-run" onclick="runSimulation()">▶ RUN SIMULATION</button>
                    <div id="error-alert" class="alert alert-error"></div>
                    <div id="status-alert" class="alert alert-success"></div>
                </div>

                <div class="card" id="results-card">
                    <h2>Simulation Metrics</h2>
                    <div id="path-box" class="path-banner">Route: Q6 → Q1 → Q0 → Q3 → Q11</div>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-label">Bell Fidelity</div>
                            <div id="fid-val" class="metric-value highlight">--</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Hops</div>
                            <div id="hops-val" class="metric-value">--</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">SWAPs</div>
                            <div id="swaps-val" class="metric-value">--</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">2Q Gates</div>
                            <div id="gates-val" class="metric-value">--</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">DAG Depth</div>
                            <div id="depth-val" class="metric-value blue">--</div>
                        </div>
                        <div class="metric-box" id="yield-box" style="display:none;">
                            <div class="metric-label">Survival Yield</div>
                            <div id="yield-val" class="metric-value highlight">--</div>
                        </div>
                    </div>

                    <div style="font-size: 12px; color: var(--text-muted); display: flex; justify-content: space-between; border-top: 1px solid var(--card-border); padding-top: 10px;">
                        <div>&lt;XX&gt;: <strong id="xx-val" style="color:#fff;">--</strong></div>
                        <div>&lt;YY&gt;: <strong id="yy-val" style="color:#fff;">--</strong></div>
                        <div>&lt;ZZ&gt;: <strong id="zz-val" style="color:#fff;">--</strong></div>
                    </div>
                </div>
            </div>

            <div>
                <div class="card">
                    <h2>Coupling Graph & Routed Path</h2>
                    <div class="plot-container">
                        <img id="plot-img" src="" alt="Topology Coupling Graph">
                    </div>
                </div>

                <div class="card">
                    <h2>Routed Qiskit QuantumCircuit</h2>
                    <pre id="circuit-box" class="circuit-box">Qiskit QuantumCircuit diagram will appear here...</pre>
                </div>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- VIEW 2: OFFICIAL 9-CASE BENCHMARK         -->
        <!-- ========================================== -->
        <div id="view-benchmark" style="display:none;">
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;">
                    <div>
                        <h2 style="margin: 0; border: none; padding: 0; font-size: 18px; color: #ffffff;">Live Qiskit Aer Benchmark — 9 Cases</h2>
                        <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">3 Topologies × 3 Conditions | Real Qiskit Aer Simulation | 20,000 shots per basis</p>
                    </div>
                    <button id="bench-run-btn" class="btn-benchmark" onclick="runOfficialBenchmark()">▶ RUN OFFICIAL 9-CASE BENCHMARK</button>
                </div>
                <div id="bench-alert" class="alert alert-info" style="display: block;">Click "RUN OFFICIAL 9-CASE BENCHMARK" to execute the full matrix using Qiskit Aer.</div>

                <div id="bench-results-table-container" style="overflow-x: auto;">
                    <table class="bench-table" id="bench-table">
                        <thead>
                            <tr>
                                <th>Topology</th>
                                <th>Condition</th>
                                <th>Fidelity</th>
                                <th>&lt;XX&gt;</th>
                                <th>&lt;YY&gt;</th>
                                <th>&lt;ZZ&gt;</th>
                                <th>SWAPs</th>
                                <th>2Q Gates</th>
                                <th>DAG Depth</th>
                                <th>Survival Yield</th>
                            </tr>
                        </thead>
                        <tbody id="bench-tbody">
                            <tr><td colspan="10" style="color: var(--text-muted); padding: 20px;">Benchmark matrix will populate after execution.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Key Comparison Charts -->
            <div class="card" id="bench-charts-card" style="display: none;">
                <h2>Key Comparison Visualizations</h2>
                <div class="charts-grid">
                    <div class="plot-container">
                        <h3 style="color:#0f172a; font-size:13px; margin-bottom:8px;">1. Remote Bell-State Fidelity: Ideal vs Noisy vs Protected</h3>
                        <img id="chart-fid" src="" alt="Fidelity Comparison">
                    </div>
                    <div class="plot-container">
                        <h3 style="color:#0f172a; font-size:13px; margin-bottom:8px;">2. Routing Gate Overhead (SWAPs & 2Q Gates)</h3>
                        <img id="chart-gates" src="" alt="Routing Overhead Comparison">
                    </div>
                    <div class="plot-container">
                        <h3 style="color:#0f172a; font-size:13px; margin-bottom:8px;">3. Routed Quantum Circuit Depth</h3>
                        <img id="chart-depth" src="" alt="Circuit Depth Comparison">
                    </div>
                    <div class="plot-container">
                        <h3 style="color:#0f172a; font-size:13px; margin-bottom:8px;">4. Protection Recovery vs Survival Yield</h3>
                        <img id="chart-prot" src="" alt="Protection Performance">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const topoDefaults = {
            "5Q Star": { num_q: 5, src: 0, dst: 3 },
            "Heavy-Hex Inspired": { num_q: 14, src: 0, dst: 6 },
            "Hyperbolic Inspired": { num_q: 16, src: 6, dst: 11 }
        };

        function switchTab(tab) {
            const viewCustom = document.getElementById("view-custom");
            const viewBench = document.getElementById("view-benchmark");
            const btnCustom = document.getElementById("tab-custom-btn");
            const btnBench = document.getElementById("tab-bench-btn");

            if (tab === "custom") {
                viewCustom.style.display = "grid";
                viewBench.style.display = "none";
                btnCustom.classList.add("active");
                btnBench.classList.remove("active");
            } else {
                viewCustom.style.display = "none";
                viewBench.style.display = "block";
                btnCustom.classList.remove("active");
                btnBench.classList.add("active");
            }
        }

        function onTopologyChange() {
            const topo = document.getElementById("topo-select").value;
            const def = topoDefaults[topo];
            const srcSel = document.getElementById("src-select");
            const dstSel = document.getElementById("dst-select");

            srcSel.innerHTML = "";
            dstSel.innerHTML = "";
            for (let i = 0; i < def.num_q; i++) {
                const opt1 = document.createElement("option");
                opt1.value = i;
                opt1.text = "Q" + i;
                if (i === def.src) opt1.selected = true;
                srcSel.appendChild(opt1);

                const opt2 = document.createElement("option");
                opt2.value = i;
                opt2.text = "Q" + i;
                if (i === def.dst) opt2.selected = true;
                dstSel.appendChild(opt2);
            }
        }

        async function runSimulation() {
            const errAlert = document.getElementById("error-alert");
            const statusAlert = document.getElementById("status-alert");
            const runBtn = document.getElementById("run-btn");

            errAlert.style.display = "none";
            statusAlert.style.display = "none";

            const topo = document.getElementById("topo-select").value;
            const src = parseInt(document.getElementById("src-select").value);
            const dst = parseInt(document.getElementById("dst-select").value);
            const cond = document.getElementById("cond-select").value;
            const shots = parseInt(document.getElementById("shots-input").value);

            runBtn.disabled = true;
            runBtn.innerText = "⏳ Simulating with Qiskit Aer...";

            try {
                const response = await fetch("/api/run", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        topology: topo,
                        source: src,
                        target: dst,
                        condition: cond,
                        shots: shots
                    })
                });
                const data = await response.json();

                if (!response.ok || !data.success) {
                    errAlert.innerText = data.error || "Simulation error occurred.";
                    errAlert.style.display = "block";
                    return;
                }

                statusAlert.innerText = `✔ Simulation succeeded! Bell Fidelity: ${data.fidelity.toFixed(4)}`;
                statusAlert.style.display = "block";

                document.getElementById("fid-val").innerText = data.fidelity.toFixed(4);
                document.getElementById("hops-val").innerText = data.hops;
                document.getElementById("swaps-val").innerText = data.swaps;
                document.getElementById("gates-val").innerText = data.two_qubit_gates;
                document.getElementById("depth-val").innerText = data.circuit_depth;
                document.getElementById("xx-val").innerText = (data.xx > 0 ? "+" : "") + data.xx.toFixed(4);
                document.getElementById("yy-val").innerText = (data.yy > 0 ? "+" : "") + data.yy.toFixed(4);
                document.getElementById("zz-val").innerText = (data.zz > 0 ? "+" : "") + data.zz.toFixed(4);

                document.getElementById("path-box").innerText = "Route: " + data.path.map(n => "Q" + n).join(" → ");

                const yieldBox = document.getElementById("yield-box");
                if (data.survival_yield !== null && data.survival_yield !== undefined) {
                    document.getElementById("yield-val").innerText = (data.survival_yield * 100).toFixed(1) + "%";
                    yieldBox.style.display = "block";
                } else {
                    yieldBox.style.display = "none";
                }

                if (data.plot_base64) {
                    document.getElementById("plot-img").src = "data:image/png;base64," + data.plot_base64;
                }
                if (data.circuit_diagram) {
                    document.getElementById("circuit-box").innerText = data.circuit_diagram;
                }

            } catch (err) {
                errAlert.innerText = "Execution error: " + err.message;
                errAlert.style.display = "block";
            } finally {
                runBtn.disabled = false;
                runBtn.innerText = "▶ RUN SIMULATION";
            }
        }

        async function runOfficialBenchmark() {
            const alertBox = document.getElementById("bench-alert");
            const btn = document.getElementById("bench-run-btn");
            const tbody = document.getElementById("bench-tbody");
            const chartsCard = document.getElementById("bench-charts-card");

            btn.disabled = true;
            btn.innerText = "⏳ Executing 9-Case Benchmark Matrix...";
            alertBox.className = "alert alert-info";
            alertBox.style.display = "block";
            alertBox.innerText = "Running genuine Qiskit Aer simulations across 3 topologies × 3 conditions (20,000 shots per basis)...";

            try {
                const response = await fetch("/api/benchmark", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ shots: 20000, seed: 42 })
                });
                const data = await response.json();

                if (!response.ok || !data.success) {
                    alertBox.className = "alert alert-error";
                    alertBox.innerText = data.error || "Benchmark execution failed.";
                    return;
                }

                alertBox.className = "alert alert-success";
                alertBox.innerText = "✔ Official 9-Case Benchmark completed! Live simulations finished and plots generated.";

                tbody.innerHTML = "";
                data.results.forEach(row => {
                    const tr = document.createElement("tr");
                    let badgeClass = "badge-ideal";
                    if (row.Condition === "Noisy") badgeClass = "badge-noisy";
                    if (row.Condition === "Noisy + Protection") badgeClass = "badge-protected";

                    const yieldStr = (row.Condition === "Noisy + Protection") ? (row["Survival Yield"] * 100).toFixed(1) + "%" : "100.0%";

                    tr.innerHTML = `
                        <td style="font-weight:600; text-align:left;">${row.Topology}</td>
                        <td><span class="${badgeClass}">${row.Condition}</span></td>
                        <td style="font-weight:700; color:#fff;">${row.Fidelity.toFixed(4)}</td>
                        <td>${(row.XX > 0 ? "+" : "") + row.XX.toFixed(4)}</td>
                        <td>${(row.YY > 0 ? "+" : "") + row.YY.toFixed(4)}</td>
                        <td>${(row.ZZ > 0 ? "+" : "") + row.ZZ.toFixed(4)}</td>
                        <td>${row.SWAPs}</td>
                        <td>${row["2Q Gates"]}</td>
                        <td>${row.Depth}</td>
                        <td style="font-weight:600; color:#4ade80;">${yieldStr}</td>
                    `;
                    tbody.appendChild(tr);
                });

                if (data.charts) {
                    chartsCard.style.display = "block";
                    if (data.charts.fidelity) document.getElementById("chart-fid").src = "data:image/png;base64," + data.charts.fidelity;
                    if (data.charts.gates) document.getElementById("chart-gates").src = "data:image/png;base64," + data.charts.gates;
                    if (data.charts.depth) document.getElementById("chart-depth").src = "data:image/png;base64," + data.charts.depth;
                    if (data.charts.protection) document.getElementById("chart-prot").src = "data:image/png;base64," + data.charts.protection;
                }

            } catch (err) {
                alertBox.className = "alert alert-error";
                alertBox.innerText = "Network/Execution error: " + err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = "▶ RUN OFFICIAL 9-CASE BENCHMARK";
            }
        }

        window.addEventListener("DOMContentLoaded", () => {
            onTopologyChange();
            runSimulation();
        });
    </script>
</body>
</html>
"""


def run_web_server(port: int = 5000, open_browser: bool = True) -> None:
    """Launches the lightweight local Python Web Server for the simulator."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class DemoRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(WEB_HTML_TEMPLATE.encode("utf-8"))
            elif self.path == "/api/topologies":
                topos = get_all_topologies()
                data = {
                    name: {"num_qubits": t.num_qubits, "source": t.source, "destination": t.destination}
                    for name, t in topos.items()
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/run":
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len).decode("utf-8")
                try:
                    payload = json.loads(post_body)
                    topology = payload.get("topology", "Hyperbolic Inspired")
                    source = payload.get("source", 6)
                    target = payload.get("target", 11)
                    condition = payload.get("condition", "Noisy")
                    shots = payload.get("shots", 20000)

                    is_valid, err_msg, custom_topo, valid_shots = validate_demo_inputs(
                        topology_name=topology,
                        source_val=source,
                        target_val=target,
                        condition=condition,
                        shots_val=shots,
                    )

                    if not is_valid:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        resp = {"success": False, "error": err_msg}
                        self.wfile.write(json.dumps(resp).encode("utf-8"))
                        return

                    res = simulate_demo(
                        topology_name=topology,
                        source=int(source),
                        target=int(target),
                        condition=condition,
                        shots=valid_shots,
                        seed=42,
                    )

                    plot_b64 = render_topology_base64(custom_topo)

                    resp_data = {
                        "success": True,
                        "topology": res.topology_name,
                        "source": res.source,
                        "target": res.target,
                        "condition": res.condition,
                        "shots": res.shots,
                        "path": res.path,
                        "hops": res.hops,
                        "swaps": res.swaps,
                        "two_qubit_gates": res.two_qubit_gates,
                        "circuit_depth": res.circuit_depth,
                        "fidelity": round(res.fidelity, 4),
                        "xx": round(res.xx, 4),
                        "yy": round(res.yy, 4),
                        "zz": round(res.zz, 4),
                        "survival_yield": round(res.survival_yield, 4) if res.survival_yield is not None else None,
                        "circuit_diagram": res.circuit_diagram,
                        "plot_base64": plot_b64,
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp_data).encode("utf-8"))

                except Exception as ex:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {"success": False, "error": str(ex)}
                    self.wfile.write(json.dumps(resp).encode("utf-8"))

            elif self.path == "/api/benchmark":
                # Execute genuine official 9-case benchmark matrix
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                try:
                    payload = json.loads(post_body) if post_body else {}
                    shots = payload.get("shots", 20000)
                    seed = payload.get("seed", 42)

                    df, geom, comp = run_official_benchmark_demo(shots=shots, seed=seed)

                    charts_data = {
                        "fidelity": get_benchmark_plot_base64("fidelity_comparison.png"),
                        "gates": get_benchmark_plot_base64("swap_count_comparison.png"),
                        "depth": get_benchmark_plot_base64("circuit_depth_comparison.png"),
                        "protection": get_benchmark_plot_base64("protection_yield_fidelity.png"),
                    }

                    resp_data = {
                        "success": True,
                        "results": df.to_dict(orient="records"),
                        "charts": charts_data,
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp_data).encode("utf-8"))

                except Exception as ex:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {"success": False, "error": str(ex)}
                    self.wfile.write(json.dumps(resp).encode("utf-8"))

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, DemoRequestHandler)
    except OSError:
        port = port + 1
        server_address = ("127.0.0.1", port)
        httpd = HTTPServer(server_address, DemoRequestHandler)

    url = f"http://127.0.0.1:{port}"
    print(f"\n[*] Q-GeoRoute Web Simulator running at: {url}")
    print("[*] Press Ctrl+C to stop the server.\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping Q-GeoRoute Web Simulator.")
        httpd.server_close()


# ==============================================================================
# TERMINAL / CLI IMPLEMENTATION
# ==============================================================================

def print_result_summary(res: DemoResult) -> None:
    """Prints the cleanly formatted simulation result directly to console."""
    path_str = " -> ".join(f"Q{n}" for n in res.path)
    print("\n" + "=" * 60)
    print("Q-GeoRoute - Quantum Routing Simulation Result")
    print("Powered by Qiskit SDK + Qiskit Aer")
    print("=" * 60)
    print(f"Topology:       {res.topology_name}")
    print(f"Source:         Q{res.source}")
    print(f"Target:         Q{res.target}")
    print(f"Condition:      {res.condition}")
    print(f"\nPath:")
    print(f"{path_str}")
    print(f"\nHops:           {res.hops}")
    print(f"SWAPs:          {res.swaps}")
    print(f"2Q Gates:       {res.two_qubit_gates}")
    print(f"Circuit Depth:  {res.circuit_depth}")
    print(f"Bell Fidelity:  {res.fidelity:.4f}")
    print(f"  <XX>: {res.xx:+.4f}")
    print(f"  <YY>: {res.yy:+.4f}")
    print(f"  <ZZ>: {res.zz:+.4f}")
    if res.survival_yield is not None:
        print(f"Survival Yield: {res.survival_yield * 100:.1f}%")
    print("=" * 60)


def print_official_benchmark_summary(df: pd.DataFrame) -> None:
    """Prints the official 9-case benchmark matrix formatted to console."""
    print("\n" + "=" * 80)
    print("                 Live Qiskit Aer Benchmark - 9 Cases")
    print("                 Powered by Qiskit SDK + Qiskit Aer")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def run_cli_interactive() -> None:
    """Interactive command-line mode for terminal environments."""
    print("\n" + "=" * 60)
    print("    Q-GeoRoute - Quantum Routing Simulator (Terminal CLI)")
    print("    Powered by Qiskit SDK + Qiskit Aer")
    print("=" * 60)
    print("\nSelect Mode:")
    print("  1. Custom Simulation (Select Topology, Endpoints, Condition, Shots)")
    print("  2. Official 9-Case Benchmark (Live Qiskit Aer Benchmark — 9 Cases)")

    m_choice = input("Enter choice [1-2] (default 1): ").strip()
    if m_choice == "2":
        print("\n[*] Executing Live Qiskit Aer Benchmark (9 Cases, 20,000 shots per basis)...")
        df, _, _ = run_official_benchmark_demo(shots=20000, seed=42)
        print_official_benchmark_summary(df)
        return

    topos = get_all_topologies()
    names = list(topos.keys())

    print("\nSelect Topology:")
    for idx, name in enumerate(names, 1):
        t = topos[name]
        print(f"  {idx}. {name:<22} ({t.num_qubits} qubits, default: Q{t.source} -> Q{t.destination})")

    try:
        t_choice = input(f"Enter choice [1-{len(names)}] (default 3): ").strip()
        t_idx = int(t_choice) - 1 if t_choice else 2
        selected_topo = names[t_idx if 0 <= t_idx < len(names) else 2]
    except Exception:
        selected_topo = names[2]

    base_t = topos[selected_topo]
    print(f"Selected: {selected_topo} (valid qubits: 0 to {base_t.num_qubits - 1})")

    src_in = input(f"Enter Source qubit [0-{base_t.num_qubits - 1}] (default {base_t.source}): ").strip()
    dst_in = input(f"Enter Target qubit [0-{base_t.num_qubits - 1}] (default {base_t.destination}): ").strip()

    source = int(src_in) if src_in else base_t.source
    target = int(dst_in) if dst_in else base_t.destination

    conds = ["Ideal", "Noisy", "Noisy + Protection"]
    print("\nSelect Condition:")
    for idx, cond in enumerate(conds, 1):
        print(f"  {idx}. {cond}")
    c_choice = input(f"Enter choice [1-3] (default 2): ").strip()
    try:
        c_idx = int(c_choice) - 1 if c_choice else 1
        condition = conds[c_idx if 0 <= c_idx < len(conds) else 1]
    except Exception:
        condition = "Noisy"

    shots_in = input("Enter Shots count (default 20000): ").strip()
    shots = int(shots_in) if shots_in else 20000

    print("\n[*] Validating inputs and executing with Qiskit Aer...")
    is_valid, err_msg, custom_topo, valid_shots = validate_demo_inputs(
        topology_name=selected_topo,
        source_val=source,
        target_val=target,
        condition=condition,
        shots_val=shots,
    )
    if not is_valid:
        print(f"\n❌ Validation Error: {err_msg}")
        return

    res = simulate_demo(
        topology_name=selected_topo,
        source=source,
        target=target,
        condition=condition,
        shots=valid_shots,
    )
    print_result_summary(res)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Q-GeoRoute Interactive Quantum Routing Simulator & Benchmark (Powered by Qiskit Aer)",
    )
    parser.add_argument("--web", "-w", action="store_true", help="Launch the browser-based Web Simulator & Benchmark")
    parser.add_argument("--cli", "-c", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run the Live Official 9-Case Benchmark directly")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically in web mode")

    # Command-line direct run arguments for custom mode
    parser.add_argument("--topology", "-t", type=str, default=None, help="Topology name: '5Q Star', 'Heavy-Hex Inspired', 'Hyperbolic Inspired'")
    parser.add_argument("--source", "-s", type=int, default=None, help="Source qubit ID")
    parser.add_argument("--target", "-d", type=int, default=None, help="Target qubit ID")
    parser.add_argument("--condition", type=str, default=None, help="Condition: 'Ideal', 'Noisy', 'Noisy + Protection'")
    parser.add_argument("--shots", type=int, default=20000, help="Number of measurement shots (default: 20000)")

    args = parser.parse_args()

    # 1. Official 9-case benchmark directly from demo
    if args.benchmark:
        print("\n[*] Launching Official 9-Case Benchmark via demo.py...")
        df, _, _ = run_official_benchmark_demo(shots=args.shots, seed=42)
        print_official_benchmark_summary(df)
        print("\n[OK] Benchmark completed. Results saved to 'results/benchmark_results.csv' and plots to 'results/plots/'.")
        return

    # 2. Direct one-shot custom CLI execution
    if args.topology is not None:
        topos = get_all_topologies()
        if args.topology not in topos:
            print(f"Error: Unknown topology '{args.topology}'. Available: {list(topos.keys())}")
            sys.exit(1)

        src = args.source if args.source is not None else topos[args.topology].source
        dst = args.target if args.target is not None else topos[args.topology].destination
        cond = args.condition if args.condition is not None else "Noisy"

        is_valid, err_msg, _, _ = validate_demo_inputs(args.topology, src, dst, cond, args.shots)
        if not is_valid:
            print(f"Error: {err_msg}")
            sys.exit(1)

        res = simulate_demo(args.topology, src, dst, cond, args.shots)
        print_result_summary(res)
        return

    # 3. Interactive Terminal mode
    if args.cli:
        run_cli_interactive()
        return

    # 4. Web mode
    if args.web:
        run_web_server(port=args.port, open_browser=not args.no_browser)
        return

    # 5. Default: Desktop GUI (Tkinter), with fallback to Web if GUI display is not available
    try:
        run_gui()
    except Exception as ex:
        print(f"\n[!] Note: Desktop GUI could not be initialized ({ex}).")
        print("[*] Launching local Web Simulator fallback instead...")
        run_web_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
