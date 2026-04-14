# PoSyDyS 2026 — Paper Companion

This site accompanies the paper **"Differentiable Simulation for Power System Dynamics"**
submitted to PoSyDyS 2026.

It provides a fully executable companion script that reproduces all figures and results
from the paper using [PowerDynamics.jl](https://github.com/JuliaEnergy/PowerDynamics.jl)
and the differentiable simulation framework built on top of it.

## What's in the script

The [Companion Script](@ref analysis) walks through four interlocking analyses on a
modified IEEE 9-bus system (SG + GFM droop inverter + GFL current-source inverter):

| Section | What it shows |
|---|---|
| Reference simulations | Load-step response; all-SG vs mixed inverter system |
| Eigenvalue analysis | Modal migration as grid impedance is swept from strong (Zscale=0.1) to weak (Zscale=4.0) |
| Impedance extraction | Bode plot of Z\_qd at nominal vs critical grid strength; cross-validation with single-inverter systems |
| Gradient-based optimization | Tuning 7 GFL controller parameters via forward-mode AD through the ODE solver; before/after comparison and validation at an unstable scenario |

## Reproducibility

```@raw html
<details><summary>Package versions used to generate this documentation</summary>
```

```@example
using Pkg #hide
Pkg.status() #hide
```

```@raw html
</details>
```
