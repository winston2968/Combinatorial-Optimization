# ⚔️ Combat Tournament - Combinatorial Optimization

This project focuses on solving a complex combinatorial optimization problem: organizing an optimal tournament between a team of **contestants** and a team of **hosts**. The goal is to maximize the total net gain while respecting strict resource and capacity constraints.

## 📋 Problem Description
The objective is to find the optimal pairing of duels based on the following rules:
* **Capacity**: A normal contestant can perform at most 2 fights, except for the **Captain**, who is limited to 1.
* **Uniqueness**: Each host can be engaged in at most one fight.
* **Energy**: The total energy cost of all fights must not exceed a global budget $B$.
* **Strategic Roles**: A **Captain** (receiving a +5 competence bonus) and a **Joker** (doubling win/loss stakes) can be designated.
* **Penalty**: Each host that remains unfought results in a penalty $P$.

## 🚀 Solving Methods

### 1. Integer Linear Programming (ILP/PLNE)
We used the **SCIP** solver to guarantee the global optimum for various instances.
* **Modeling**: Decision variables $x_{i,j}$ (fights), $c_i$ (captain), and $j_i$ (joker).
* **Linearization**: Used incremental gains ($\Delta Cap$, $\Delta Jok$) and coupling variables to maintain a linear model despite multipliers.
* **Optimization**: Implemented branching priorities (`chgVarBranchPriority`) on roles and a 5% optimality gap to handle large-scale instances (100 contestants) efficiently.

### 2. Meta-heuristic (Simulated Annealing)
A stochastic approach was implemented to find near-optimal solutions for large instances where exhaustive search is too costly.
* **Matrix Approach**: Evaluations are performed using **NumPy** matrix products ($Tr(^t C M)$) for high-speed computation.
* **Extended Neighborhood**: 5 types of moves: *Swap Host*, *Shift Contestant*, *Add/Drop*, *Move Joker*, and *Move Captain*.
* **Greedy Initialization**: The search starts from a solution pre-constructed using a gain/energy ratio heuristic to avoid slow convergence from an empty state.

## 📊 Performance Comparison

| Instance | Method | Status | Value | Time ($\Delta t$) |
| :--- | :--- | :--- | :--- | :--- |
| **tournament_6.txt** | ILP | Optimal | 1045 | 0.05s |
| | MH + Greedy | Optimal | 1045 | 0.12s |
| **tournament_20.txt** | ILP | Optimal | 2555 | 3.75s |
| | MH + Greedy | Near-Optimal | 2538 | 0.72s |
| **tournament_100.txt** | ILP (5% Gap) | Sub-Optimal | 38192 | 438.79s |
| | MH + Greedy | Near-Optimal | 37548 | 10.54s |

## 🛠️ Requirements
* Python 3.x
* NumPy
* PySCIPOpt (for ILP solving)
* Matplotlib (for convergence plotting)

## 📈 Convergence Analysis
The Simulated Annealing demonstrates a high capacity to escape local optima. Even though a greedy start is used, the initial high temperature favors exploration, leading to a temporary drop in score before converging toward a superior global optimum.