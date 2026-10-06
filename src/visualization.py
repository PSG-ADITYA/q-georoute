"""
Visualization Module for Q-GeoRoute.

Generates presentation and publication-ready figures for:
1. 5Q Star topology graph (with routing path)
2. Heavy-Hex-inspired topology graph (with routing path)
3. Hyperbolic-inspired topology graph (with routing path)
4. Fidelity comparison: Ideal vs Noisy vs Protected
5. SWAP and 2Q gate count comparison
6. Circuit depth comparison
7. Fidelity vs routing cost trade-off
8. Protection performance: noisy vs protected fidelity and survival yield

Outputs saved to results/plots/.
"""

import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.topologies import Topology, get_all_topologies
from src.metrics import GeometryMetrics, BenchmarkComparison

# Set clean aesthetic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = {
    "ideal": "#2b5c8f",
    "noisy": "#d95f02",
    "protected": "#2ca02c",
    "highlight_src": "#2ca02c",
    "highlight_dst": "#e7298a",
    "path_edge": "#e41a1c",
    "neutral_node": "#d0d7de",
    "neutral_edge": "#94a3b8",
}


def plot_topology_graph(
    topology: Topology,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Renders a coupling topology graph highlighting the source, destination,
    and shortest-path routing trajectory.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g = topology.graph
    pos = topology.pos
    path = topology.shortest_path()
    path_edges = list(zip(path[:-1], path[1:]))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Base edges
    nx.draw_networkx_edges(
        g, pos,
        ax=ax,
        edge_color=PALETTE["neutral_edge"],
        width=1.8,
        alpha=0.6,
    )

    # Highlighted routing path edges
    nx.draw_networkx_edges(
        g, pos,
        edgelist=path_edges,
        ax=ax,
        edge_color=PALETTE["path_edge"],
        width=3.5,
        alpha=0.9,
        label=f"Routing Path ({len(path)-1} hops)",
    )

    # Node colors: source (green), destination (magenta), path intermediate (yellow), rest (slate)
    node_colors = []
    for node in g.nodes():
        if node == topology.source:
            node_colors.append(PALETTE["highlight_src"])
        elif node == topology.destination:
            node_colors.append(PALETTE["highlight_dst"])
        elif node in path:
            node_colors.append("#fd8d3c")
        else:
            node_colors.append(PALETTE["neutral_node"])

    nx.draw_networkx_nodes(
        g, pos,
        ax=ax,
        node_color=node_colors,
        node_size=650,
        edgecolors="#2c3e50",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        g, pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
        font_color="#1e293b",
    )

    plot_title = title or f"{topology.name} Coupling Graph"
    ax.set_title(f"{plot_title}\nSource: Q{topology.source} -> Dest: Q{topology.destination} | Path: {' -> '.join(f'Q{n}' for n in path)}", pad=12)
    ax.axis("off")

    # Legend
    custom_lines = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["highlight_src"], markersize=10, label=f"Source (Q{topology.source})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["highlight_dst"], markersize=10, label=f"Destination (Q{topology.destination})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fd8d3c", markersize=10, label="Intermediate Routing Qubits"),
        plt.Line2D([0], [0], color=PALETTE["path_edge"], lw=3, label=f"Routing Route ({len(path)-1} hops)"),
    ]
    ax.legend(handles=custom_lines, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fidelity_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Plot bar chart comparing Ideal, Noisy, and Protected fidelities across topologies."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    topologies = df["Topology"].unique()
    x = np.arange(len(topologies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5))

    ideal_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Ideal")]["Fidelity"].values[0] for t in topologies]
    noisy_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy")]["Fidelity"].values[0] for t in topologies]
    prot_vals = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Fidelity"].values[0] for t in topologies]

    rects1 = ax.bar(x - width, ideal_vals, width, label="Ideal Baseline", color=PALETTE["ideal"], alpha=0.9, edgecolor="black")
    rects2 = ax.bar(x, noisy_vals, width, label="Noisy Baseline", color=PALETTE["noisy"], alpha=0.9, edgecolor="black")
    rects3 = ax.bar(x + width, prot_vals, width, label="Noisy + Protection", color=PALETTE["protected"], alpha=0.9, edgecolor="black")

    ax.set_ylabel("Bell-State Fidelity F(|Phi+>)")
    ax.set_title("Remote Bell-State Fidelity: Ideal vs Noisy vs Protected", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_swap_and_gates_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Plot SWAP count and Two-qubit gate count per topology."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_ideal = df[df["Condition"] == "Ideal"]
    topologies = df_ideal["Topology"].tolist()
    swaps = df_ideal["SWAPs"].tolist()
    two_q = df_ideal["2Q Gates"].tolist()

    x = np.arange(len(topologies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    r1 = ax.bar(x - width/2, swaps, width, label="SWAP Gates", color="#807dba", edgecolor="black")
    r2 = ax.bar(x + width/2, two_q, width, label="Total 2Q Gates (SWAPs + CX)", color="#41b6c4", edgecolor="black")

    ax.set_ylabel("Gate Count")
    ax.set_title("Routing Overhead: SWAPs & Total Two-Qubit Gates", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, fontweight="bold")
    ax.set_ylim(0, max(two_q) + 3)
    ax.legend(loc="upper left")

    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_circuit_depth_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Plot circuit depth comparison across topologies."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_ideal = df[df["Condition"] == "Ideal"]
    topologies = df_ideal["Topology"].tolist()
    depths = df_ideal["Depth"].tolist()

    colors = ["#7fc97f", "#beaed4", "#fdc086"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(topologies, depths, color=colors, width=0.45, edgecolor="black", alpha=0.9)

    ax.set_ylabel("Circuit Depth (Time-slices)")
    ax.set_title("Routed Quantum Circuit Depth by Topology", pad=12)
    ax.set_ylim(0, max(depths) + 3)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"Depth = {int(h)}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_fidelity_vs_routing_cost(df: pd.DataFrame, save_path: str) -> None:
    """Scatter / trade-off plot: Routing Cost (2Q Gates) vs Bell Fidelity."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    topologies = df["Topology"].unique()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    markers = {"5Q Star": "o", "Heavy-Hex Inspired": "s", "Hyperbolic Inspired": "^"}
    colors = {"5Q Star": "#1f78b4", "Heavy-Hex Inspired": "#e31a1c", "Hyperbolic Inspired": "#33a02c"}

    for topo in topologies:
        row_noisy = df[(df["Topology"] == topo) & (df["Condition"] == "Noisy")].iloc[0]
        row_prot = df[(df["Topology"] == topo) & (df["Condition"] == "Noisy + Protection")].iloc[0]

        gates = row_noisy["2Q Gates"]

        # Noisy baseline point
        ax.scatter(gates, row_noisy["Fidelity"], color=colors[topo], marker=markers[topo], s=160,
                   edgecolor="black", label=f"{topo} (Noisy)", zorder=4)
        # Protected point
        ax.scatter(gates, row_prot["Fidelity"], color=colors[topo], marker=markers[topo], s=200,
                   facecolors="none", linewidths=2.5, linestyle="--", label=f"{topo} (Protected)", zorder=4)

        # Arrow indicating protection improvement
        ax.annotate("", xy=(gates, row_prot["Fidelity"]), xytext=(gates, row_noisy["Fidelity"]),
                    arrowprops=dict(arrowstyle="->", color=colors[topo], lw=1.8, ls=":"))

        # Label topology name
        ax.text(gates + 0.25, row_noisy["Fidelity"] - 0.005, f"{topo}", fontsize=9, fontweight="bold", color=colors[topo])

    ax.set_xlabel("Two-Qubit Gates (SWAPs + CX) along Routing Path")
    ax.set_ylabel("Bell-State Fidelity F(|Phi+>)")
    ax.set_title("Fidelity vs Routing Overhead Trade-off Frontier", pad=12)
    ax.set_xlim(3, 13)
    ax.set_ylim(0.82, 0.94)

    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_protection_yield_and_fidelity(df: pd.DataFrame, save_path: str) -> None:
    """Plot noisy vs protected fidelity alongside post-selection survival yield."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    topologies = df["Topology"].unique()
    x = np.arange(len(topologies))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    noisy_fids = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy")]["Fidelity"].values[0] for t in topologies]
    prot_fids = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Fidelity"].values[0] for t in topologies]
    yields = [df[(df["Topology"] == t) & (df["Condition"] == "Noisy + Protection")]["Survival Yield"].values[0] * 100 for t in topologies]

    b1 = ax1.bar(x - width/2, noisy_fids, width, label="Noisy Fidelity", color=PALETTE["noisy"], alpha=0.9, edgecolor="black")
    b2 = ax1.bar(x + width/2, prot_fids, width, label="Protected Fidelity", color=PALETTE["protected"], alpha=0.9, edgecolor="black")

    ax1.set_ylabel("Fidelity F(|Phi+>)", color="#333333")
    ax1.set_ylim(0.75, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topologies, fontweight="bold")

    # Secondary axis for yield
    ax2 = ax1.twinx()
    p1 = ax2.plot(x, yields, color="#7570b3", marker="D", markersize=9, linewidth=2.5, linestyle="-", label="Survival Yield (%)")
    ax2.set_ylabel("Post-Selection Survival Yield (%)", color="#7570b3", fontweight="bold")
    ax2.set_ylim(70, 105)
    ax2.grid(False)

    for i, y in enumerate(yields):
        ax2.annotate(f"{y:.1f}%", xy=(x[i], y), xytext=(0, 8), textcoords="offset points",
                     ha="center", va="bottom", color="#7570b3", fontweight="bold", fontsize=9.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    ax1.set_title("Protection Impact: Fidelity Recovery vs Survival Yield eta", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: str = "results/plots/") -> None:
    """Generates all 8 benchmark and topology figures."""
    os.makedirs(output_dir, exist_ok=True)
    topos = get_all_topologies()

    # 1. 5Q Star Topology
    plot_topology_graph(topos["5Q Star"], os.path.join(output_dir, "topology_5q_star.png"), "2016-Inspired 5-Qubit Star/T-Shape")

    # 2. Heavy-Hex Topology
    plot_topology_graph(topos["Heavy-Hex Inspired"], os.path.join(output_dir, "topology_heavy_hex.png"), "Contemporary Heavy-Hex-Inspired Sparse Graph")

    # 3. Hyperbolic Topology
    plot_topology_graph(topos["Hyperbolic Inspired"], os.path.join(output_dir, "topology_hyperbolic.png"), "Hyperbolic-Inspired Finite Graph (Poincaré Disk)")

    # 4. Fidelity Comparison
    plot_fidelity_comparison(df, os.path.join(output_dir, "fidelity_comparison.png"))

    # 5. SWAP Count Comparison
    plot_swap_and_gates_comparison(df, os.path.join(output_dir, "swap_count_comparison.png"))

    # 6. Circuit Depth Comparison
    plot_circuit_depth_comparison(df, os.path.join(output_dir, "circuit_depth_comparison.png"))

    # 7. Fidelity vs Routing Cost
    plot_fidelity_vs_routing_cost(df, os.path.join(output_dir, "fidelity_vs_routing_cost.png"))

    # 8. Protection Performance
    plot_protection_yield_and_fidelity(df, os.path.join(output_dir, "protection_yield_fidelity.png"))
