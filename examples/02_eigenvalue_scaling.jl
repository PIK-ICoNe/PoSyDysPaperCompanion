using PowerDynamics
using PowerDynamics.Library
using ModelingToolkit
using PoSyDysPaperCompanion
using DelimitedFiles
using DataFrames, CSV
using Graphs
using DiffEqCallbacks
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using CairoMakie
using OrderedCollections: OrderedDict

####
#### Part 1: Data Generation
####

# Scales to evaluate; 1.0 is the baseline.
# Using logarithmically-spaced values so that scaling above/below 1 is symmetric.
below = range(0.1, 1.0, length=15)
above = range(1.0, 4.0, length=15)
scales = sort!(unique!(vcat(below, above)))
baseline_idx = findfirst(==(1.0), scales)

@info "Computing eigenvalues for $(length(scales)) impedance scales..."
eigenvalue_data = OrderedDict{Float64, Vector{ComplexF64}}()
for Zscale in scales
    @info "  Zscale = $Zscale"
    nw, s0 = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=Zscale, verbose=false)
    eigenvalue_data[Zscale] = jacobian_eigenvals(s0) ./ (2π)
end

"""
    match_eigenvalues(ref_eigs, new_eigs)

Greedily match each eigenvalue in `new_eigs` to the closest unmatched eigenvalue in
`ref_eigs` (by absolute distance in the complex plane).  Returns `new_eigs` reordered
so that `matched[i]` is the best continuation of `ref_eigs[i]`.
"""
function match_eigenvalues(ref_eigs, new_eigs)
    n = length(ref_eigs)
    @assert n == length(new_eigs) "Eigenvalue count mismatch"
    matched   = zeros(ComplexF64, n)
    available = collect(1:n)
    for i in 1:n
        dists = abs.(ref_eigs[i] .- new_eigs[available])
        best  = argmin(dists)
        matched[i] = new_eigs[available[best]]
        deleteat!(available, best)
    end
    return matched
end

# Build tracks matrix:  tracks[i, j]  = eigenvalue i at scales[j]
# Chain matching outward from the baseline in both directions so that each step
# only needs to match against its closest neighbour (better for large perturbations).
n_eigs   = length(eigenvalue_data[1.0])
n_scales = length(scales)

baseline_order = sortperm(eigenvalue_data[1.0]; by = x -> (real(x), imag(x)))
tracks = Matrix{ComplexF64}(undef, n_eigs, n_scales)
tracks[:, baseline_idx] = eigenvalue_data[1.0][baseline_order]

# Increasing scale: match each step against the previous (closer-to-baseline) step
for j in (baseline_idx+1):n_scales
    tracks[:, j] = match_eigenvalues(tracks[:, j-1], eigenvalue_data[scales[j]])
end

# Decreasing scale: same idea but walk from baseline downward
for j in (baseline_idx-1):-1:1
    tracks[:, j] = match_eigenvalues(tracks[:, j+1], eigenvalue_data[scales[j]])
end

####
#### Part 2: Visualization
####

# Map scales to a symmetric colour axis via log transform:
#   log(1) = 0   → neutral midpoint of the diverging colormap
#   log(scale) < 0 → blue  (impedance reduced)
#   log(scale) > 0 → red   (impedance increased)
log_scales  = log.(scales)
max_log     = maximum(abs.(log_scales))
norm_colors = log_scales ./ max_log        # in [-1, 1], 0 at baseline

cmap  = :bluesreds   # blue = low impedance, red = high impedance
let
    fig = Figure(size=(600, 500))

    ax = Axis(fig[1, 1], xlabel = "Real Part [Hz]", ylabel = "Imaginary Part [Hz]", title = "Eigenvalues for different Grid Strengths")

    for i in 1:n_eigs
        lines!(ax, real.(tracks[i, :]), imag.(tracks[i, :]);
            color      = norm_colors,
            joinstyle=:round,
            linecap   = :round,
            colorrange = (-1.0, 1.0),
            colormap   = cmap,
            linewidth  = 3)
    end

    # Mark the baseline positions (Zscale = 1) with filled circles
    scatter!(ax,
        real.(tracks[:, baseline_idx]),
        imag.(tracks[:, baseline_idx]);
        color      = :black,
        markersize = 6,
        marker     = :xcross)

    

    xlims!(ax, -40, 5)
    ylims!(ax, -120, 120)

    fig
end
