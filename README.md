# Flow Around Airfoil: Classical Methods, CFD & Physics-Informed Neural Networks

> Incompressible flow around **NACA 6409**, **NACA 0024** and symmetric **NACA 00xx** airfoils, studied with three complementary approaches — **Source Panel Method (MATLAB)**, **CFD (ANSYS Fluent)** and **Physics-Informed Neural Networks (PyTorch)** — and compared in terms of accuracy, computational cost and ability to capture aerodynamic phenomena.

**Period:** Sep 2025 – Feb 2026
**Context:** Fluid Mechanics 2 course project, Department of Mechanical Engineering, Sharif University of Technology (Fall–Winter 1404 / 2025–26)
**Author:** Mohammadreza Sotoudeh — [LinkedIn](https://www.linkedin.com/in/mohammadreza-sotoudeh84/)

---

## Table of Contents

1. [Overview](#overview)
2. [Methods at a Glance](#methods-at-a-glance)
3. [Project Phases](#project-phases)
4. [Results](#results)
   - [Panel method](#1-panel-method-matlab)
   - [CFD — NACA 6409, Re = 2×10⁶](#2-cfd--naca-6409-re--2106)
   - [CFD — mesh independence](#3-cfd--mesh-independence-study)
   - [CFD — angle-of-attack sweep (near stall)](#4-cfd--angle-of-attack-sweep-naca-6409-re--3106)
   - [CFD — NACA 0024 (inviscid)](#5-cfd--naca-0024-inviscid-aoa--5)
   - [PINN — Navier–Stokes, NACA 6409](#6-pinn--navierstokes-naca-6409)
   - [PINN — potential flow, NACA 0024](#7-pinn--potential-flow-naca-0024)
5. [Key Findings](#key-findings)
6. [Limitations & Notes](#limitations--notes)
9. [Full Report](#full-report)

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

<p align="center">
  <img src="images/cfd_naca6409_re2e6/naca6409_geometry.jpg" width="48%" alt="NACA 6409 geometry">
  <img src="images/cfd_naca6409_re2e6/c-type_mesh_overview.jpg" width="48%" alt="C-type mesh">
</p>
<p align="center"><em>NACA 6409 geometry (left) and the C-type mesh domain (right)</em></p>

### 3. Physics-Informed Neural Networks (PyTorch)
- **Navier–Stokes PINN (NACA 6409):** steady incompressible Navier–Stokes; the network maps `(x, y) → (u, v, p)`
  - 6 hidden layers × 100 neurons, `tanh` activation
  - Loss = continuity + momentum residuals (soft constraints) + boundary-condition loss
  - Hybrid optimization: Adam (mini-batch) cycles followed by L-BFGS refinement
  - Collocation points with refinement near the airfoil; points inside the airfoil are masked out
- **Potential-flow PINN (NACA 0024):** continuity + irrotationality (Laplace) residuals for the velocity field
  - 3 hidden layers × 100 neurons, `tanh` activation

<p align="center">
  <img src="images/pinn_navier_stokes_naca6409/network_architecture.jpg" width="40%" alt="Navier-Stokes PINN architecture">
  <img src="images/pinn_potential_flow_naca0024/network_architecture.jpg" width="50%" alt="Potential-flow PINN architecture">
</p>
<p align="center"><em>PINN architectures: Navier–Stokes (left) and potential flow (right)</em></p>

## Project Phases

| Phase | Topic | Airfoil | Method |
|---|---|---|---|
| 1 | CFD, Cl/Cd, mesh-independence studies (qualitative + quantitative) | NACA 6409 | Fluent |
| 2 | Steady Navier–Stokes PINN | NACA 6409 | PyTorch |
| 3a | CFD at different angles of attack (−5°, 0°, 15° near stall), lift/drag curves, comparison with experimental data | NACA 6409 | Fluent |
| 3b | Inviscid CFD, pressure/velocity contours and streamlines | NACA 0024 | Fluent |
| 4 | Potential-flow PINN | NACA 0024 | PyTorch |
| 5 | Source panel method, effect of thickness, panel independence, validation | NACA 00xx | MATLAB |

## Results

### 1. Panel Method (MATLAB)

The panel-independence study shows the solution settling quickly as the panel count grows, and the thickness study shows how the pressure-drag indicator changes with t/c. Streamlines (ψ), equipotential lines (φ) and the Cp distribution are extracted from the potential solution.

<p align="center">
  <img src="images/panel_method/panel_independence_study.jpg" width="48%" alt="Panel independence study">
  <img src="images/panel_method/thickness_effect_on_pressure_drag.jpg" width="48%" alt="Effect of thickness on pressure drag">
</p>
<p align="center">
  <img src="images/panel_method/cp_distribution.jpg" width="48%" alt="Cp distribution">
</p>
<p align="center">
  <img src="images/panel_method/streamlines_psi_max_thickness.jpg" width="48%" alt="Streamlines psi">
  <img src="images/panel_method/equipotential_lines_phi.jpg" width="48%" alt="Equipotential lines phi">
</p>

### 2. CFD — NACA 6409, Re = 2×10⁶

| α | Cl (CFD) | Cd (CFD) |
|---|---|---|
| 0° | 0.7145 | 0.01315 |
| 5° | 1.2474 | 0.01948 |

At α = 0° the experimental reference values used in the report are Cl ≈ 0.685 and Cd ≈ 0.01, giving a relative error of ~4.3 % for Cl and ~31 % for Cd. The larger drag error is attributed to the strong dependence of Cd on near-wall mesh resolution.

| α = 0° | α = 5° |
|:---:|:---:|
| <img src="images/cfd_naca6409_re2e6/re2e6_aoa0_velocity_contour.jpg" alt="Velocity contour AoA 0"> | <img src="images/cfd_naca6409_re2e6/re2e6_aoa5_velocity_contour_near.jpg" alt="Velocity contour AoA 5"> |
| <img src="images/cfd_naca6409_re2e6/re2e6_aoa0_pressure_contour.jpg" alt="Pressure contour AoA 0"> | <img src="images/cfd_naca6409_re2e6/re2e6_aoa5_pressure_contour.jpg" alt="Pressure contour AoA 5"> |
| <img src="images/cfd_naca6409_re2e6/re2e6_aoa0_streamlines.jpg" alt="Streamlines AoA 0"> | <img src="images/cfd_naca6409_re2e6/re2e6_aoa5_streamlines_near.jpg" alt="Streamlines AoA 5"> |

<details>
<summary>More: far-field views, surface pressure and wall y⁺</summary>

<p align="center">
  <img src="images/cfd_naca6409_re2e6/re2e6_aoa5_velocity_contour_far.jpg" width="48%" alt="Velocity contour far view">
  <img src="images/cfd_naca6409_re2e6/re2e6_aoa5_streamlines_far.jpg" width="48%" alt="Streamlines far view">
</p>
<p align="center">
  <img src="images/cfd_naca6409_re2e6/re2e6_aoa0_surface_pressure_plot.jpg" width="48%" alt="Surface pressure plot">
  <img src="images/cfd_naca6409_re2e6/re2e6_wall_yplus.jpg" width="48%" alt="Wall y+">
</p>

</details>

### 3. CFD — Mesh-independence study

Four C-type meshes were generated (Re = 3×10⁶, α = 0°). Qualitatively, the contours, streamlines and surface-pressure distributions do not change noticeably with refinement.

| Mesh | Elements | Cl |
|---|---|---|
| Mesh 1 (very fine) | 159,184 | 0.7195 |
| Mesh 2 (fine) | 97,574 | 0.7077 |
| Mesh 3 (medium) | 71,694 | 0.6681 |
| Mesh 4 (coarse) | 68,730 | 0.6794 |

The Grid Convergence Index for Cl on the finest mesh is ≈ **0.87 %**, so the solution is essentially grid-independent.

| Mesh 1 (finest) | Mesh 4 (coarsest) |
|:---:|:---:|
| <img src="images/cfd_mesh_independence_re3e6/mesh1_velocity_contour.jpg" alt="Mesh 1 velocity"> | <img src="images/cfd_mesh_independence_re3e6/mesh4_velocity_contour.jpg" alt="Mesh 4 velocity"> |
| <img src="images/cfd_mesh_independence_re3e6/mesh1_pressure_contour.jpg" alt="Mesh 1 pressure"> | <img src="images/cfd_mesh_independence_re3e6/mesh4_pressure_contour.jpg" alt="Mesh 4 pressure"> |
| <img src="images/cfd_mesh_independence_re3e6/mesh1_surface_pressure_plot.jpg" alt="Mesh 1 surface pressure"> | <img src="images/cfd_mesh_independence_re3e6/mesh4_surface_pressure_plot.jpg" alt="Mesh 4 surface pressure"> |

<details>
<summary>All meshes (2 and 3) and streamlines</summary>

| Mesh 2 | Mesh 3 |
|:---:|:---:|
| <img src="images/cfd_mesh_independence_re3e6/mesh2_velocity_contour.jpg" alt="Mesh 2 velocity"> | <img src="images/cfd_mesh_independence_re3e6/mesh3_velocity_contour.jpg" alt="Mesh 3 velocity"> |
| <img src="images/cfd_mesh_independence_re3e6/mesh2_pressure_contour.jpg" alt="Mesh 2 pressure"> | <img src="images/cfd_mesh_independence_re3e6/mesh3_pressure_contour.jpg" alt="Mesh 3 pressure"> |
| <img src="images/cfd_mesh_independence_re3e6/mesh2_surface_pressure_plot.jpg" alt="Mesh 2 surface pressure"> | <img src="images/cfd_mesh_independence_re3e6/mesh3_surface_pressure_plot.jpg" alt="Mesh 3 surface pressure"> |

| Mesh 1 | Mesh 2 | Mesh 3 | Mesh 4 |
|:---:|:---:|:---:|:---:|
| <img src="images/cfd_mesh_independence_re3e6/mesh1_streamlines.jpg"> | <img src="images/cfd_mesh_independence_re3e6/mesh2_streamlines.jpg"> | <img src="images/cfd_mesh_independence_re3e6/mesh3_streamlines.jpg"> | <img src="images/cfd_mesh_independence_re3e6/mesh4_streamlines.jpg"> |

</details>

### 4. CFD — Angle-of-attack sweep, NACA 6409 (Re = 3×10⁶)

Pressure, velocity and streamline fields are compared at α = −5°, 0° and 15° (near stall). At 15° the pressure difference between the upper and lower surfaces grows sharply, and the thickening/decelerating flow on the upper surface in the rear half indicates the onset of separation.

| α = −5° | α = 0° | α = 15° (near stall) |
|:---:|:---:|:---:|
| <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa-5_pressure_contour.jpg" alt="Pressure -5"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa0_pressure_contour.jpg" alt="Pressure 0"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa15_pressure_contour.jpg" alt="Pressure 15"> |
| <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa-5_velocity_contour.jpg" alt="Velocity -5"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa0_velocity_contour.jpg" alt="Velocity 0"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa15_velocity_contour.jpg" alt="Velocity 15"> |
| <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa-5_streamlines.jpg" alt="Streamlines -5"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa0_streamlines.jpg" alt="Streamlines 0"> | <img src="images/cfd_naca6409_aoa_sweep_re3e6/aoa15_streamlines.jpg" alt="Streamlines 15"> |

<details>
<summary>Reference Cl–α curve and solver residuals</summary>

<p align="center">
  <img src="images/cfd_naca6409_aoa_sweep_re3e6/experimental_cl_vs_alpha.jpg" width="40%" alt="Reference Cl vs alpha">
  <img src="images/cfd_naca6409_aoa_sweep_re3e6/residuals_k-epsilon_convergence.jpg" width="55%" alt="Residuals">
</p>

</details>

### 5. CFD — NACA 0024 (inviscid, AoA = 5°)

<p align="center">
  <img src="images/cfd_naca0024_inviscid_aoa5/naca0024_geometry.jpg" width="48%" alt="NACA 0024 geometry">
  <img src="images/cfd_naca0024_inviscid_aoa5/aoa5_pressure_contour.jpg" width="48%" alt="Pressure contour NACA 0024">
</p>
<p align="center">
  <img src="images/cfd_naca0024_inviscid_aoa5/aoa5_velocity_contour.jpg" width="48%" alt="Velocity contour NACA 0024">
  <img src="images/cfd_naca0024_inviscid_aoa5/aoa5_streamlines.jpg" width="48%" alt="Streamlines NACA 0024">
</p>

### 6. PINN — Navier–Stokes, NACA 6409

The network reproduces the pressure and velocity fields without any mesh; training uses Adam cycles followed by L-BFGS refinement.

<p align="center">
  <img src="images/pinn_navier_stokes_naca6409/training_loss_history.jpg" width="60%" alt="Training loss history">
</p>
<p align="center">
  <img src="images/pinn_navier_stokes_naca6409/aoa0_pressure_field.jpg" width="48%" alt="PINN pressure field">
  <img src="images/pinn_navier_stokes_naca6409/aoa0_velocity_magnitude.jpg" width="48%" alt="PINN velocity magnitude">
</p>
<p align="center">
  <img src="images/pinn_navier_stokes_naca6409/aoa0_streamlines.jpg" width="48%" alt="PINN streamlines">
  <img src="images/pinn_navier_stokes_naca6409/aoa0_surface_pressure.jpg" width="48%" alt="PINN surface pressure">
</p>

### 7. PINN — Potential flow, NACA 0024

<p align="center">
  <img src="images/pinn_potential_flow_naca0024/training_loss_convergence.jpg" width="48%" alt="Training loss convergence">
  <img src="images/pinn_potential_flow_naca0024/streamlines.jpg" width="48%" alt="PINN streamlines">
</p>
<p align="center">
  <img src="images/pinn_potential_flow_naca0024/pressure_field.jpg" width="48%" alt="PINN pressure field">
  <img src="images/pinn_potential_flow_naca0024/velocity_magnitude.jpg" width="48%" alt="PINN velocity magnitude">
</p>

## Key Findings

- **Panel method** gives a fast, low-cost approximation of pressure distribution and streamlines. Its results converge with increasing panel count, and it correctly captures the symmetry at α = 0° and the effect of thickness on the pressure distribution — but it cannot represent viscous effects (boundary layer, separation, realistic drag).
- **CFD (Fluent)** resolves the velocity and pressure fields in detail, predicts lift and drag with good accuracy against experiments, and captures viscous phenomena and near-stall behavior — at a much higher computational cost.
- **PINNs** reproduced the flow field **without any mesh**, enforcing the governing equations and boundary conditions through the loss function. This shows the potential of physics-based machine learning for flow problems, although training cost and sensitivity to loss weighting/optimizer settings remain significant challenges.
- Overall: the panel method is suited to quick ideal-flow analysis, CFD to accurate force and flow-detail prediction, and PINNs are a promising emerging, mesh-free alternative.

## Limitations & Notes

- The Navier–Stokes PINN is run at a **low Reynolds number (Re = 100, laminar)** for computational feasibility, whereas the Fluent cases use Re = 2–3×10⁶ with a turbulence model. Comparison between these two is therefore limited to qualitative trends.
- The panel method is inviscid, so any non-zero drag it reports reflects numerical effects rather than physical drag (d'Alembert's paradox).
- The potential-flow PINN and the panel method address the same physical model; the Navier–Stokes PINN and Fluent address a viscous model.


## Full Report

The complete write-up (in Persian), including governing equations, mesh studies, y⁺ analysis, contours, streamlines, Cp plots, Cl/Cd curves, comparison with experimental data, and all source code in the appendix, is available in [`report/`](report/).

