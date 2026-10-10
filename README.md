# Flow Around Airfoil: Classical Methods, CFD & Physics-Informed Neural Networks

> Incompressible flow around **NACA 6409** and **NACA 0024** (plus symmetric NACA 00xx) airfoils, studied with three complementary approaches — **Source Panel Method (MATLAB)**, **CFD (ANSYS Fluent)** and **Physics-Informed Neural Networks (PyTorch)** — and compared in terms of accuracy, computational cost and ability to capture aerodynamic phenomena.

**Period:** Sep 2025 – Feb 2026
**Context:** Fluid Mechanics 2 course project, Department of Mechanical Engineering, Sharif University of Technology (Fall–Winter 1404 / 2025–26)
**Author:** Mohammadreza Sotoudeh — [LinkedIn](https://www.linkedin.com/in/mohammadreza-sotoudeh84/)

---

## Table of Contents

1. [Overview](#overview)
2. [Methods at a Glance](#methods-at-a-glance)
3. [Project Phases](#project-phases)
4. [Selected Results](#selected-results)
5. [Key Findings](#key-findings)
6. [Limitations & Notes](#limitations--notes)
7. [Repository Structure](#repository-structure)
8. [Getting Started](#getting-started)
9. [Full Report](#full-report)
10. [License](#license)

---

## Overview

Airfoils are a core component of aircraft, turbomachinery and energy converters, and accurate prediction of the flow around them is essential for maximizing lift and minimizing drag. This project investigates steady, incompressible flow around NACA-series airfoils using three different levels of modeling:

| Approach | Tool | Role in the study |
|---|---|---|
| **Classical** | Source panel method, MATLAB | Fast potential-flow baseline (pressure distribution, thickness effect) |
| **Grid-based CFD** | ANSYS Fluent | High-fidelity reference (RANS, viscous effects, near-stall behavior) |
| **Mesh-free ML** | PINNs, Python / PyTorch | Solve the governing equations without a mesh, using physics residuals as soft constraints |

The extracted quantities — pressure coefficient (**Cp**) distributions, lift and drag coefficients (**Cl**, **Cd**), velocity and pressure contours, and streamlines — are compared across methods.

## Methods at a Glance

### 1. Panel Method (MATLAB)
- Source panel method for potential (inviscid, incompressible, steady) flow
- Linear system `A·σ = RHS` from the no-penetration condition; σ is the source strength per panel
- NACA 4-digit symmetric thickness function with cosine panel spacing
- Panel-independence study (40 / 80 / 160 / 320 panels)
- Parametric study of thickness: **t/c = 0.06, 0.12, 0.18, 0.24, 0.30** at α = 0°

### 2. CFD (ANSYS Fluent)
- **Meshes:** structured C-type meshes with several refinement levels
- **Turbulence model:** Standard k–ε with Enhanced Wall Treatment
- **Fluid:** air, ρ = 1.225 kg/m³, μ = 1.85×10⁻⁵ Pa·s
- **Reynolds numbers:** 2×10⁶ and 3×10⁶
- **Boundary conditions:** uniform velocity inlet, zero-gauge-pressure outlet, no-slip airfoil wall
- **Angles of attack:** multiple values, including near-stall (up to 15°)
- **Studies:** qualitative and quantitative mesh independence (with GCI), wall y⁺ analysis

### 3. Physics-Informed Neural Networks (PyTorch)
- **Navier–Stokes PINN (NACA 6409):** steady incompressible Navier–Stokes; the network maps `(x, y) → (u, v, p)`
  - 6 hidden layers × 100 neurons, `tanh` activation
  - Loss = continuity + momentum residuals (soft constraints) + boundary-condition loss
  - Hybrid optimization: Adam (mini-batch) cycles followed by L-BFGS refinement
  - Collocation points with refinement near the airfoil; points inside the airfoil are masked out
- **Potential-flow PINN (NACA 0024):** continuity + irrotationality (Laplace) residuals for the velocity field
  - 3 hidden layers × 100 neurons, `tanh` activation

## Project Phases

| Phase | Topic | Airfoil | Method |
|---|---|---|---|
| 1 | CFD, Cl/Cd, mesh-independence studies (qualitative + quantitative) | NACA 6409 | Fluent |
| 2 | Steady Navier–Stokes PINN | NACA 6409 | PyTorch |
| 3a | CFD at different angles of attack (−5°, 0°, 15° near stall), lift/drag curves, comparison with experimental data | NACA 6409 | Fluent |
| 3b | Inviscid CFD, pressure/velocity contours and streamlines | NACA 0024 | Fluent |
| 4 | Potential-flow PINN | NACA 0024 | PyTorch |
| 5 | Source panel method, effect of thickness, panel independence, validation | NACA 00xx | MATLAB |

## Selected Results

**Fluent, NACA 6409, Re = 2×10⁶**

| α | Cl (CFD) | Cd (CFD) |
|---|---|---|
| 0° | 0.7145 | 0.01315 |
| 5° | 1.2474 | 0.01948 |

At α = 0° the experimental reference values used in the report are Cl ≈ 0.685 and Cd ≈ 0.01, giving a relative error of ~4.3 % for Cl and ~31 % for Cd. The larger drag error is attributed to the strong dependence of Cd on near-wall mesh resolution.

**Quantitative mesh study, NACA 6409, Re = 3×10⁶, α = 0°**

| Mesh | Elements | Cl |
|---|---|---|
| Mesh 1 (very fine) | 159,184 | 0.7195 |
| Mesh 2 (fine) | 97,574 | 0.7077 |
| Mesh 3 (medium) | 71,694 | 0.6681 |
| Mesh 4 (coarse) | 68,730 | 0.6794 |

The Grid Convergence Index for Cl on the finest mesh is ≈ **0.87 %**, so the solution is essentially grid-independent.

> 📌 *Add figures here (recommended): Cp comparison plot, velocity/pressure contours, streamlines for each method.*
>
> ```markdown
> ![Cp comparison](figures/cp_comparison.png)
> ![PINN pressure contour](figures/pinn_pressure_contour.png)
> ```

## Key Findings

- **Panel method** gives a fast, low-cost approximation of pressure distribution and streamlines. Its results converge with increasing panel count, and it correctly captures the symmetry at α = 0° and the effect of thickness on the pressure distribution — but it cannot represent viscous effects (boundary layer, separation, realistic drag).
- **CFD (Fluent)** resolves the velocity and pressure fields in detail, predicts lift and drag with good accuracy against experiments, and captures viscous phenomena and near-stall behavior — at a much higher computational cost.
- **PINNs** reproduced the flow field **without any mesh**, enforcing the governing equations and boundary conditions through the loss function. This shows the potential of physics-based machine learning for flow problems, although training cost and sensitivity to loss weighting/optimizer settings remain significant challenges.
- Overall: the panel method is suited to quick ideal-flow analysis, CFD to accurate force and flow-detail prediction, and PINNs are a promising emerging, mesh-free alternative.

## Limitations & Notes

- The PINN case for the Navier–Stokes equations is run at a **low Reynolds number (Re = 100, laminar)** for computational feasibility, whereas the Fluent cases use Re = 2–3×10⁶ with a turbulence model. Direct quantitative comparison between these two is therefore limited to qualitative trends.
- The panel method is inviscid, so any non-zero drag it reports reflects numerical effects rather than physical drag (d'Alembert's paradox).
- The potential-flow PINN and the panel method address the same physical model; the Navier–Stokes PINN and Fluent address a viscous model.

## Repository Structure

*(suggested layout — adjust to your actual files)*

```
.
├── README.md
├── report/
│   └── The_Full_Project.pdf        # full report (Persian)
├── panel_method/
│   └── panel_method_naca00xx.m     # MATLAB source panel method
├── cfd_fluent/
│   ├── NACA6409/                   # case setup notes, meshes, settings
│   └── NACA0024/
├── pinn/
│   ├── pinn_navier_stokes_naca6409.py
│   └── pinn_potential_flow_naca0024.py
├── figures/                        # contours, streamlines, Cp plots
└── LICENSE
```

## Getting Started

### Panel method (MATLAB)
```matlab
% Open the script in MATLAB and run it
panel_method_naca00xx
```

### PINNs (Python)
```bash
pip install numpy matplotlib torch
python pinn/pinn_navier_stokes_naca6409.py
```
A CUDA GPU is used automatically if available (otherwise CPU). For a quick test, reduce the number of collocation points (`N_col`, `N_refine`) in the script.

### CFD (ANSYS Fluent)
Case files are not required to read the report. Mesh type, solver settings, boundary conditions and turbulence model are documented in the [report](#full-report); recreate the cases in ANSYS Fluent following those settings.

## Full Report

The complete write-up (in Persian), including governing equations, mesh studies, y⁺ analysis, contours, streamlines, Cp plots, Cl/Cd curves, comparison with experimental data, and all source code in the appendix, is available in [`report/`](report/).

## License

Specify a license for the repository (e.g., MIT) — see `LICENSE`.
