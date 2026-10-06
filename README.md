# Q-GeoRoute

**Geometry-Aware Quantum Routing and Noise Benchmark across Conventional, Heavy-Hex-Inspired, and Hyperbolic-Inspired Processor Topologies.**

---

## Problem
In physical quantum computing architectures, hardware is fundamentally constrained by limited two-qubit coupling connectivity. Real processors (such as superconducting quantum processors or neutral atom traps) do not have all-to-all connectivity. When a quantum algorithm demands interaction or entanglement between two remote, non-adjacent physical qubits—such as generating a Bell state:
$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$
the quantum compiler must insert sequences of SWAP gates to transport state information along coupling graph paths.

However, on physical hardware:
1. Each SWAP operation decomposes into three native entangling gates (e.g., three $CX$ gates).
2. Every additional two-qubit gate deepens circuit depth and introduces gate infidelities (depolarizing errors).
3. The routing chain exposes the fragile quantum state to environmental decoherence and readout assignment errors.

This raises a crucial architectural question: **How does processor graph geometry dictate compiler routing overhead, noisy performance, and error mitigation viability?**

---

## Motivation
Historically, quantum processors evolved from small 5-qubit cross/star configurations (2016-era devices like IBM Q 5 Yorktown/Tenerife) to planar sparse lattices such as IBM's **heavy-hex** layout (Falcon, Hummingbird, Eagle, Heron). Heavy-hex lattices prioritize low qubit coordination (degree $\le 3$) to suppress frequency collisions and spectator crosstalk. However, planar lattices suffer from high graph diameter and long routing distances.

In network theory and theoretical physics, **hyperbolic geometry** (negative curvature) provides exponential neighborhood expansion with radius ($Area \propto e^r$). A finite graph inspired by hyperbolic tessellation exhibits logarithmic diameter and natural radial "shortcuts". Investigating how hyperbolic connectivity compares to conventional and heavy-hex architectures under realistic quantum noise provides actionable architectural insights for next-generation modular QPUs and optical/cryogenic interconnects.

---

## Our Idea
We built **Q-GeoRoute**, a geometry-aware quantum routing and noise benchmarking framework that evaluates remote entanglement generation across three distinct processor topologies:
1. **2016-Inspired 5-Qubit Star/T-Shape Topology** (5 qubits)
2. **Contemporary Heavy-Hex-Inspired Topology** (14 qubits)
3. **Hyperbolic-Inspired Finite Graph** (16 qubits)

Across each topology, we benchmark three operational conditions:
- **Condition A: Ideal Baseline** (Noise-free simulation with exact statevector verification)
- **Condition B: Noisy Baseline** (Depolarizing gate noise + readout assignment errors)
- **Condition C: Noisy + Protection** (Stabilizer/parity verification with post-selection)

Total = **9 benchmark cases**.

---

## Three Architectures

| Architecture | Nodes | Edges | Graph Diameter | Avg Path Length | Non-Adjacent Endpoints | Shortest Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **5Q Star** | 5 | 4 | 3 | 1.80 | Q0 $\to$ Q3 | `[0, 1, 2, 3]` (3 hops) |
| **Heavy-Hex Inspired** | 14 | 14 | 8 | 3.57 | Q0 $\to$ Q6 | `[0, 1, 2, 3, 4, 5, 6]` (6 hops) |
| **Hyperbolic Inspired** | 16 | 30 | 4 | 2.25 | Q6 $\to$ Q11 | `[6, 1, 0, 3, 11]` (4 hops) |

- **5Q Star:** A 2016-era inspired star/T-shaped coupling graph with a central junction qubit.
- **Heavy-Hex Inspired:** A representative 14-qubit sparse benchmark graph where vertex qubits have degree $\le 3$ and edge bridge qubits have degree 2.
- **Hyperbolic Inspired:** A 16-qubit finite graph modeled in the Poincaré disk with a central core, inner concentric shell ($r=0.45$), outer concentric shell ($r=0.85$), and boundary cross-links exhibiting negative curvature.

---

## Bell-State Routing
To entangle non-adjacent source qubit $s$ and destination qubit $d$ along shortest path $P = [p_0, p_1, \dots, p_k]$ ($p_0 = s, p_k = d$):
1. **State Initialization:** Apply $H(p_0)$ on the source qubit.
2. **Forward SWAP Shuttling:** Apply sequential SWAPs $\text{SWAP}(p_i, p_{i+1})$ for $i = 0, \dots, k-2$. The superposition is now on physical qubit $p_{k-1}$, adjacent to destination $p_k$.
3. **Entanglement:** Apply native two-qubit gate $CX(p_{k-1}, p_k)$.
4. **Reverse SWAP Restoration:** Apply the reverse sequence of SWAPs from $k-2$ down to 0.

### Mathematical Guarantee:
- The entangled Bell pair $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$ is restored directly onto physical target qubits $(s, d)$.
- All intermediate routing qubits $p_1, \dots, p_{k-1}$ deterministically return to state $|0\rangle$.

Total SWAPs inserted: $N_{\text{SWAP}} = 2 \times (\text{hops} - 1)$.  
Total two-qubit gates: $N_{2Q} = N_{\text{SWAP}} + 1$.

---

## Noise Model
Using **Qiskit Aer**, we configure an empirically grounded superconducting qubit noise model:
- **1-Qubit Gate Depolarizing Error:** $p_1 = 0.10\%$ on $H, S, S^\dagger$
- **2-Qubit Gate Depolarizing Error:** $p_2 = 1.50\%$ on $CX, SWAP$
- **Readout Assignment Error:** $p_{0|1} = p_{1|0} = 2.00\%$
- **Execution:** 20,000 shots per Pauli basis ($X, Y, Z$) with fixed random seed (`seed=42`).

---

## Protection Strategy: Stabilizer/Parity Verification with Post-Selection
Rather than falsely claiming full fault-tolerant Quantum Error Correction (QEC), we implement **syndrome verification with post-selection**:
1. Because intermediate routing qubits along the path return to $|0\rangle$ in the fault-free circuit, they act as **intrinsic transport syndrome detectors**.
2. Any depolarizing or bit-flip error during the SWAP chain flips one or more intermediate qubits to $|1\rangle$.
3. We measure all intermediate routing qubits into syndrome classical bits.
4. **Verification Condition:** Accept a shot if and only if **all syndrome bits measure 0**.
5. We measure the post-selection **survival yield**:
   $$\eta = \frac{N_{\text{accepted}}}{N_{\text{total}}}$$
6. We calculate the protected Bell fidelity on accepted shots:
   $$F(\Phi^+) = \frac{1 + \langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle}{4}$$

---

## 9-Case Benchmark

Saved automatically to `results/benchmark_results.csv`:

| Topology | Condition | Fidelity | $\langle XX \rangle$ | $\langle YY \rangle$ | $\langle ZZ \rangle$ | SWAPs | 2Q Gates | Depth | Survival Yield ($\eta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5Q Star** | Ideal | 1.0000 | 1.0000 | -1.0000 | 1.0000 | 4 | 5 | 6 | 1.0000 |
| **5Q Star** | Noisy | 0.8972 | 0.8526 | -0.8568 | 0.8795 | 4 | 5 | 6 | 1.0000 |
| **5Q Star** | Noisy + Protection | 0.9116 | 0.8751 | -0.8803 | 0.8911 | 4 | 5 | 6 | 0.9341 |
| **Heavy-Hex Inspired** | Ideal | 1.0000 | 1.0000 | -1.0000 | 1.0000 | 10 | 11 | 12 | 1.0000 |
| **Heavy-Hex Inspired** | Noisy | 0.8513 | 0.7760 | -0.7877 | 0.8414 | 10 | 11 | 12 | 1.0000 |
| **Heavy-Hex Inspired** | Noisy + Protection | 0.8856 | 0.8349 | -0.8387 | 0.8688 | 10 | 11 | 12 | 0.8448 |
| **Hyperbolic Inspired** | Ideal | 1.0000 | 1.0000 | -1.0000 | 1.0000 | 6 | 7 | 8 | 1.0000 |
| **Hyperbolic Inspired** | Noisy | 0.8828 | 0.8265 | -0.8341 | 0.8707 | 6 | 7 | 8 | 1.0000 |
| **Hyperbolic Inspired** | Noisy + Protection | 0.9049 | 0.8639 | -0.8648 | 0.8910 | 6 | 7 | 8 | 0.9026 |

---

## Metrics
Q-GeoRoute calculates transparent geometry and routing metrics alongside a composite recommendation score:

$$\text{Normalized Routing Cost} = \frac{\text{Total 2Q Gates} - \min(\text{2Q Gates})}{\min(\text{2Q Gates})}$$

$$\text{Composite Score} = \frac{\text{Fidelity}}{1 + \text{Normalized Routing Cost}}$$

| Topology | Diameter | Avg Path Length | Routing Hops | SWAP Count | 2Q Gates | Depth | Noisy Fidelity | Protected Fidelity | Survival Yield | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5Q Star** | 3 | 1.80 | 3 | 4 | 5 | 6 | 0.8972 | 0.9116 | 93.4% | **0.8972** |
| **Heavy-Hex** | 8 | 3.57 | 6 | 10 | 11 | 12 | 0.8513 | 0.8856 | 84.5% | **0.3869** |
| **Hyperbolic** | 4 | 2.25 | 4 | 6 | 7 | 8 | 0.8828 | 0.9049 | 90.3% | **0.6306** |

---

## Results
1. **Routing Overhead:** Comparing multi-qubit architectures, Hyperbolic achieves a **40% reduction in SWAP operations** (6 vs 10) and **33% reduction in circuit depth** (8 vs 12) relative to Heavy-Hex.
2. **Noise Degradation:** Heavy-Hex exhibits the steepest fidelity degradation ($\Delta F = 0.1487$) due to its long 6-hop routing path, dropping to $F = 0.8513$. Hyperbolic retains higher noisy fidelity ($F = 0.8828$).
3. **Protection Efficacy & Yield:** Post-selection boosts Hyperbolic fidelity to **0.9049** while retaining **90.26% of shots**. On Heavy-Hex, post-selection achieves $F = 0.8856$ but throughput drops to **84.48%**.
4. **Visual Figures Generated:**
   - `results/plots/topology_5q_star.png`
   - `results/plots/topology_heavy_hex.png`
   - `results/plots/topology_hyperbolic.png`
   - `results/plots/fidelity_comparison.png`
   - `results/plots/swap_count_comparison.png`
   - `results/plots/circuit_depth_comparison.png`
   - `results/plots/fidelity_vs_routing_cost.png`
   - `results/plots/protection_yield_fidelity.png`

---

## Architecture Recommendation
**Key Finding:**
In our simulated benchmark, the hyperbolic-inspired topology achieved lower routing overhead than the tested heavy-hex-inspired topology for the selected remote-qubit configuration.

- **Routing Overhead & Depth:** Comparing representative multi-qubit graphs, Hyperbolic (16Q) achieves **40% fewer SWAPs** (6 vs 10) and **33% lower routed DAG circuit depth** (8 vs 12) than Heavy-Hex (14Q).
- **Noisy Fidelity:** Hyperbolic maintains higher noisy Bell fidelity (**0.8828** vs **0.8513** for Heavy-Hex) due to reduced two-qubit gate exposure.
- **Protection & Survival Yield:** Stabilizer/parity post-selection achieves **0.9049 fidelity** on Hyperbolic with a **90.3% survival yield**, compared to **84.5% yield** on Heavy-Hex.
- **Composite Score:** Hyperbolic scores **0.6306 vs 0.3869** for Heavy-Hex (a 63% relative trade-off improvement).
- **Note on Small-Scale Baseline:** The 5-qubit topology has the lowest absolute routing overhead because it is a much smaller graph (5 qubits vs 14–16 qubits); therefore, this comparison is intended to study how connectivity geometry affects routing scaling rather than declaring the smallest processor the winner.

---

## Fair-Comparison Limitation
> [!IMPORTANT]
> **Scientific Rigor & Scope Limitation:**
> The heavy-hex and hyperbolic models are representative simulated graphs with different sizes (14 vs 16 qubits) and connectivity structures. Results depend on the selected source/target pairs and the chosen noise model. Therefore, the benchmark demonstrates the effect of topology under controlled simulation conditions rather than proving universal hardware superiority.
> 
> Furthermore:
> - We are simulating representative architectures using Qiskit Aer; we do not claim physical access to a 127-qubit IBM processor or fabrication of a physical hyperbolic QPU.
> - The protection strategy is stabilizer/parity verification with post-selection, not full fault-tolerant quantum error correction.

---

## Quantum Advantage / Why Geometry Matters
In classical networking, small-world and hyperbolic topologies are well known for enabling efficient greedy routing. In quantum information processing, geometry is even more critical because **quantum states cannot be copied** (No-Cloning Theorem). Routing must be executed via coherent unitary SWAP gates or entanglement swapping. Every extra hop in physical space directly degrades state purity and adds decoherence. Thus, graph geometry is not merely a routing efficiency consideration—it directly determines quantum algorithm fidelity and error correction thresholds.

---

## Installation

```bash
git clone https://github.com/PSG-ADITYA/q-georoute.git
cd q-georoute

# Install dependencies
python -m pip install -r requirements.txt
```

---

## How to Run

### 1. Run the Full Benchmark & Generate Plots
```bash
python src/main.py
```
This will:
- Initialize all 3 topologies.
- Execute the 9-case benchmark matrix.
- Export results to `results/benchmark_results.csv`.
- Generate all 8 publication-ready plots under `results/plots/`.
- Print the formatted results table and architectural recommendation.

### 2. Run the Automated Test Suite
```bash
python -m unittest discover tests -v
```
Runs 10 unit and integration tests verifying graph connectivity, non-adjacent endpoints, path validity, ideal fidelity, noise degradation, protection yield, and CSV generation.

### 3. Run the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/Q_GeoRoute_Benchmark.ipynb
```
Follow the step-by-step interactive walkthrough with rich visual displays.

---

## Repository Structure
```
Q-GeoRoute/
├── README.md                          # Project documentation and benchmark overview
├── requirements.txt                   # Dependency specifications (Qiskit 2.x compatible)
├── src/                               # Source code modules
│   ├── __init__.py                    # Package initialization
│   ├── topologies.py                  # 5Q Star, Heavy-Hex, and Hyperbolic graph generators
│   ├── routing.py                     # Shortest-path routing and Bell circuit construction
│   ├── noise.py                       # Qiskit Aer noise model (depolarizing + readout)
│   ├── protection.py                  # Stabilizer syndrome verification & post-selection
│   ├── metrics.py                     # Geometry-aware metrics & composite score formulas
│   ├── benchmark.py                   # 9-Case benchmark execution engine & CSV exporter
│   ├── visualization.py               # Publication-ready Matplotlib figure generators
│   └── main.py                        # CLI entrypoint running full benchmark workflow
├── tests/                             # Automated test suite
│   ├── __init__.py
│   ├── test_topologies.py             # Connectivity, node count, non-adjacent endpoints
│   ├── test_routing.py                # Shortest path validity, SWAP counts, depth
│   └── test_benchmark.py              # Bell fidelity, noise degradation, survival yields
├── notebooks/                         # Interactive documentation
│   └── Q_GeoRoute_Benchmark.ipynb     # Complete, runnable end-to-end experiment notebook
├── results/                           # Generated benchmark artifacts
│   ├── benchmark_results.csv          # Complete 9-row experimental results matrix
│   └── plots/                         # 8 Generated publication-quality figures
│       ├── topology_5q_star.png
│       ├── topology_heavy_hex.png
│       ├── topology_hyperbolic.png
│       ├── fidelity_comparison.png
│       ├── swap_count_comparison.png
│       ├── circuit_depth_comparison.png
│       ├── fidelity_vs_routing_cost.png
│       └── protection_yield_fidelity.png
└── docs/                              # Presentation materials
    └── presentation_results.md        # Hackathon briefing, key results, judge Q&A
```

---

## Limitations
1. **Simulation Scope:** All experiments are conducted via numerical simulation using Qiskit Aer. We do not claim physical access to a 127-qubit IBM processor or fabrication of a physical hyperbolic superconducting chip.
2. **Post-Selection Throughput:** The protection strategy uses stabilizer/parity syndrome verification with post-selection. While it filters out errors and enhances fidelity, it discards failing shots ($\eta \approx 84-93\%$) and does not perform active, fault-tolerant in-flight error correction.
3. **Planar Lithography vs Non-Planar Graphs:** Superconducting chips currently favor planar 2D layouts (like heavy-hex) to minimize parasitic wire crossings and crosstalk. Hyperbolic connectivity is primarily suited for modular architectures, optical quantum interconnects, or neutral atom arrays in 3D optical traps.
