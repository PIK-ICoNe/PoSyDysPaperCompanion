# PoSyDyS 2026 — Paper Companion

This site accompanies the paper **"Differentiable Simulation for Power System Dynamics"**
submitted to PoSyDyS 2026.

It provides a fully executable companion script that reproduces all figures and results
from the paper using [PowerDynamics.jl](https://github.com/JuliaEnergy/PowerDynamics.jl).

> **[View the full companion documentation with inlined figures and analysis](https://pik-icone.github.io/PoSyDysPaperCompanion/)**

The repository is a Julia package. `src/` contains helper code for constructing the test
network and defining component models, while `examples/analysis.jl` is the main companion
script that reproduces all figures and results from the paper.

```
PoSyDysPaperCompanion/
├── src/                    # Julia package
│   ├── PoSyDysPaperCompanion.jl
│   └── models.jl           # network construction helpers
├── examples/
│   └── analysis.jl         # main companion script
├── Project.toml            # environment definition
├── Manifest.toml           # pinned dependency versions
└── docs/                   # documentation / companion website
```

To run the companion script, activate the root environment first:

```julia-repl
julia> # press ] to enter Pkg mode
pkg> activate /path/to/PoSyDysPaperCompanion
pkg> instantiate
```

Then include or open `examples/analysis.jl`.

