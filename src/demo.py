"""
Q-GeoRoute: Geometry-Aware Quantum Routing Simulator (Interactive Demo)
Powered by Qiskit SDK + Qiskit Aer.

Provides an interactive demo for hackathon presentations and architecture exploration.
Reuses existing Q-GeoRoute modules:
- src.topologies
- src.routing
- src.noise
- src.protection
- src.benchmark
- src.metrics

Modes:
- Desktop GUI (default): python src/demo.py
- Browser Web App:       python src/demo.py --web
- Interactive Terminal:  python src/demo.py --cli
- Direct Command Line:   python src/demo.py --topology "Hyperbolic Inspired" --source 6 --target 11 --condition Noisy
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
import matplotlib
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator

from src.topologies import Topology, get_all_topologies
from src.routing import build_routed_bell_circuit, RoutingMetrics
from src.noise import NoiseConfig, build_noise_model
from src.benchmark import run_single_condition


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
# 1. DESKTOP GUI IMPLEMENTATION (Tkinter + Matplotlib)
# ==============================================================================

def run_gui() -> None:
    """Launches the native Python Tkinter desktop GUI simulator."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("Q-GeoRoute — Quantum Routing Simulator")
    root.geometry("1180x820")
    root.minsize(980, 720)

    # Set application background
    root.configure(bg="#f8fafc")

    topos = get_all_topologies()
    topo_names = list(topos.keys())

    # State variables
    current_topo_var = tk.StringVar(value=topo_names[0])
    src_var = tk.StringVar()
    dst_var = tk.StringVar()
    cond_var = tk.StringVar(value="Noisy")
    shots_var = tk.StringVar(value="20000")
    status_var = tk.StringVar(value="Ready. Select parameters and click 'Run Simulation'.")

    # Header frame
    header_frame = tk.Frame(root, bg="#1e293b", padx=20, pady=12)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(
        header_frame,
        text="Q-GeoRoute — Quantum Routing Simulator",
        font=("Helvetica", 16, "bold"),
        fg="#ffffff",
        bg="#1e293b",
    )
    title_label.pack(anchor="w")

    subtitle_label = tk.Label(
        header_frame,
        text="Powered by Qiskit SDK + Qiskit Aer | Geometry-Aware Remote Bell-State Routing",
        font=("Helvetica", 9),
        fg="#94a3b8",
        bg="#1e293b",
    )
    subtitle_label.pack(anchor="w")

    # Main split container
    main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#e2e8f0", sashwidth=4)
    main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    # Left Column: Controls & Metrics
    left_frame = tk.Frame(main_paned, bg="#ffffff", padx=14, pady=12, relief=tk.RIDGE, bd=1)
    main_paned.add(left_frame, minsize=420, width=450)

    # Controls Section
    ctrl_group = tk.LabelFrame(left_frame, text=" Simulation Parameters ", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b", padx=10, pady=10)
    ctrl_group.pack(fill=tk.X, pady=(0, 10))

    # 1. Topology selector
    tk.Label(ctrl_group, text="Processor Topology:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=4)
    topo_combo = ttk.Combobox(ctrl_group, textvariable=current_topo_var, values=topo_names, state="readonly", width=22)
    topo_combo.grid(row=0, column=1, sticky="ew", pady=4, padx=5)

    # 2. Source qubit
    tk.Label(ctrl_group, text="Source Qubit:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=1, column=0, sticky="w", pady=4)
    src_combo = ttk.Combobox(ctrl_group, textvariable=src_var, state="readonly", width=22)
    src_combo.grid(row=1, column=1, sticky="ew", pady=4, padx=5)

    # 3. Target qubit
    tk.Label(ctrl_group, text="Target Qubit:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=2, column=0, sticky="w", pady=4)
    dst_combo = ttk.Combobox(ctrl_group, textvariable=dst_var, state="readonly", width=22)
    dst_combo.grid(row=2, column=1, sticky="ew", pady=4, padx=5)

    # 4. Condition selector
    tk.Label(ctrl_group, text="Condition:", font=("Helvetica", 9, "bold"), bg="#ffffff").grid(row=3, column=0, sticky="w", pady=4)
    cond_combo = ttk.Combobox(ctrl_group, textvariable=cond_var, values=["Ideal", "Noisy", "Noisy + Protection"], state="readonly", width=22)
    cond_combo.grid(row=3, column=1, sticky="ew", pady=4, padx=5)

    # 5. Shots input
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
            # Update canvas with default endpoints
            render_canvas(t)

    current_topo_var.trace_add("write", update_qubit_choices)

    # Buttons Frame
    btn_frame = tk.Frame(ctrl_group, bg="#ffffff", pady=6)
    btn_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 2))

    run_btn = tk.Button(
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
    run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    reset_btn = tk.Button(
        btn_frame,
        text="Reset",
        font=("Helvetica", 9),
        bg="#e2e8f0",
        fg="#334155",
        cursor="hand2",
        padx=10,
        pady=5,
        command=lambda: update_qubit_choices(),
    )
    reset_btn.pack(side=tk.RIGHT, padx=(4, 0))

    # Status / Error banner
    status_label = tk.Label(
        left_frame,
        textvariable=status_var,
        font=("Helvetica", 8, "italic"),
        fg="#475569",
        bg="#f1f5f9",
        padx=8,
        pady=6,
        relief=tk.FLAT,
        wraplength=410,
        justify=tk.LEFT,
    )
    status_label.pack(fill=tk.X, pady=(0, 10))

    # Results Display Card
    results_group = tk.LabelFrame(left_frame, text=" Live Quantum Simulation Metrics ", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b", padx=10, pady=10)
    results_group.pack(fill=tk.BOTH, expand=True)

    results_text = ScrolledText(results_group, font=("Consolas", 10), bg="#f8fafc", fg="#0f172a", relief=tk.FLAT, height=18)
    results_text.pack(fill=tk.BOTH, expand=True)
    results_text.insert(tk.END, "Configure parameters and click 'RUN SIMULATION' to evaluate real Bell-state fidelity.")
    results_text.config(state=tk.DISABLED)

    # Right Column: Visualizations & Circuit
    right_paned = tk.PanedWindow(main_paned, orient=tk.VERTICAL, bg="#e2e8f0", sashwidth=4)
    main_paned.add(right_paned, minsize=520, width=680)

    # Top: Matplotlib Figure Canvas
    plot_frame = tk.Frame(right_paned, bg="#ffffff", padx=8, pady=8, relief=tk.RIDGE, bd=1)
    right_paned.add(plot_frame, minsize=380, height=430)

    plot_header = tk.Label(plot_frame, text="Topology & Shortest Routing Path Visualization", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b")
    plot_header.pack(anchor="w", pady=(0, 4))

    canvas_container = tk.Frame(plot_frame, bg="#ffffff")
    canvas_container.pack(fill=tk.BOTH, expand=True)

    current_canvas = [None]

    def render_canvas(t_obj: Topology):
        # Clear old canvas if present
        if current_canvas[0] is not None:
            current_canvas[0].get_tk_widget().destroy()
            current_canvas[0] = None

        fig = create_topology_figure(t_obj, figsize=(6.2, 4.3))
        canvas = FigureCanvasTkAgg(fig, master=canvas_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        current_canvas[0] = canvas
        plt.close(fig)

    # Bottom: Routed Qiskit Circuit Text View
    circuit_frame = tk.Frame(right_paned, bg="#ffffff", padx=8, pady=8, relief=tk.RIDGE, bd=1)
    right_paned.add(circuit_frame, minsize=240, height=280)

    circuit_header = tk.Label(circuit_frame, text="Routed Qiskit QuantumCircuit (Generated Native SWAP Insertion)", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#1e293b")
    circuit_header.pack(anchor="w", pady=(0, 4))

    circuit_text = ScrolledText(circuit_frame, font=("Courier New", 9), bg="#0f172a", fg="#38bdf8", relief=tk.FLAT, wrap=tk.NONE)
    circuit_text.pack(fill=tk.BOTH, expand=True)
    circuit_text.insert(tk.END, "# Generated Qiskit circuit with SWAP routing and tomographic measurement will appear here.")
    circuit_text.config(state=tk.DISABLED)

    # Execution callback
    def on_run():
        t_name = current_topo_var.get()
        s_str = src_var.get().replace("Q", "").strip()
        d_str = dst_var.get().replace("Q", "").strip()
        cond = cond_var.get()
        shots_str = shots_var.get().strip()

        # Input validation
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

        # Valid input: Run real simulation
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

            # Update status
            status_var.set(f"✔ Completed successfully! Bell Fidelity: {res.fidelity:.4f}")
            status_label.config(fg="#16a34a", bg="#f0fdf4")

            # Update Metrics Panel
            results_text.config(state=tk.NORMAL)
            results_text.delete("1.0", tk.END)

            path_str = " → ".join(f"Q{n}" for n in res.path)
            lines = [
                f"Topology: {res.topology_name}",
                f"Source:   Q{res.source}",
                f"Target:   Q{res.target}",
                f"Condition: {res.condition}",
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
                    f"Survival Yield (η): {res.survival_yield * 100:.1f}%",
                    f"(Stabilizer syndrome post-selection filtered {100 - res.survival_yield * 100:.1f}% error shots)",
                ])

            results_text.insert(tk.END, "\n".join(lines))
            results_text.config(state=tk.DISABLED)

            # Update Circuit View
            circuit_text.config(state=tk.NORMAL)
            circuit_text.delete("1.0", tk.END)
            circuit_text.insert(tk.END, res.circuit_diagram)
            circuit_text.config(state=tk.DISABLED)

            # Update Matplotlib Graph Canvas
            render_canvas(custom_topo)

        except Exception as ex:
            status_var.set(f"❌ Execution Error: {str(ex)}")
            status_label.config(fg="#dc2626", bg="#fef2f2")
            messagebox.showerror("Simulation Error", f"Simulation failed: {str(ex)}")

    run_btn.config(command=on_run)

    # Initialize endpoints
    update_qubit_choices()

    root.mainloop()


# ==============================================================================
# 2. LOCAL BROWSER / WEB SERVER IMPLEMENTATION (Pure Python http.server)
# ==============================================================================

WEB_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-GeoRoute — Quantum Routing Simulator</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-purple: #c084fc;
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
            margin: 0 auto 24px auto;
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
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .container {
            max-width: 1240px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
        }
        @media (max-width: 960px) {
            .container { grid-template-columns: 1fr; }
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
        .form-control:focus {
            border-color: var(--accent-blue);
        }
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
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Q-GeoRoute — Quantum Routing Simulator</h1>
            <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Geometry-Aware Quantum Routing & Bell-State Benchmarking across Processor Topologies</p>
        </div>
        <div class="badge">Powered by Qiskit SDK + Qiskit Aer</div>
    </div>

    <div class="container">
        <!-- Left Column: Controls -->
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

        <!-- Right Column: Visualization & Circuit -->
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

    <script>
        const topoDefaults = {
            "5Q Star": { num_q: 5, src: 0, dst: 3 },
            "Heavy-Hex Inspired": { num_q: 14, src: 0, dst: 6 },
            "Hyperbolic Inspired": { num_q: 16, src: 6, dst: 11 }
        };

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

                // Update UI Metrics
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
                errAlert.innerText = "Network/Execution error: " + err.message;
                errAlert.style.display = "block";
            } finally {
                runBtn.disabled = false;
                runBtn.innerText = "▶ RUN SIMULATION";
            }
        }

        // Initialize on page load and trigger first run
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

                    # Validate inputs
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

                    # Execute simulation
                    res = simulate_demo(
                        topology_name=topology,
                        source=int(source),
                        target=int(target),
                        condition=condition,
                        shots=valid_shots,
                        seed=42,
                    )

                    # Generate plot base64
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
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress routine HTTP log spam in console
            pass

    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, DemoRequestHandler)
    except OSError:
        # Port might be in use; try port + 1
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
# 3. TERMINAL / CLI IMPLEMENTATION
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


def run_cli_interactive() -> None:
    """Interactive command-line mode for terminal environments."""
    topos = get_all_topologies()
    names = list(topos.keys())

    print("\n" + "=" * 60)
    print("    Q-GeoRoute - Quantum Routing Simulator (Terminal CLI)")
    print("    Powered by Qiskit SDK + Qiskit Aer")
    print("=" * 60)

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
        description="Q-GeoRoute Interactive Quantum Routing Simulator (Powered by Qiskit Aer)",
    )
    parser.add_argument("--web", "-w", action="store_true", help="Launch the browser-based Web Simulator")
    parser.add_argument("--cli", "-c", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically in web mode")

    # Command-line direct run arguments
    parser.add_argument("--topology", "-t", type=str, default=None, help="Topology name: '5Q Star', 'Heavy-Hex Inspired', 'Hyperbolic Inspired'")
    parser.add_argument("--source", "-s", type=int, default=None, help="Source qubit ID")
    parser.add_argument("--target", "-d", type=int, default=None, help="Target qubit ID")
    parser.add_argument("--condition", type=str, default=None, help="Condition: 'Ideal', 'Noisy', 'Noisy + Protection'")
    parser.add_argument("--shots", type=int, default=20000, help="Number of measurement shots (default: 20000)")

    args = parser.parse_args()

    # 1. Direct one-shot CLI execution
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

    # 2. Interactive Terminal mode
    if args.cli:
        run_cli_interactive()
        return

    # 3. Web mode
    if args.web:
        run_web_server(port=args.port, open_browser=not args.no_browser)
        return

    # 4. Default: Desktop GUI (Tkinter), with fallback to Web if GUI display is not available
    try:
        run_gui()
    except Exception as ex:
        # Graceful fallback if DISPLAY or Tkinter windowing fails
        print(f"\n[!] Note: Desktop GUI could not be initialized ({ex}).")
        print("[*] Launching local Web Simulator fallback instead...")
        run_web_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
