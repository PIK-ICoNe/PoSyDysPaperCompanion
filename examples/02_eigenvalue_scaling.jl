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
using LinearAlgebra

####
#### Part 1: Data Generation
####

# Scales to evaluate; 1.0 is the baseline.
# Using logarithmically-spaced values so that scaling above/below 1 is symmetric.
below = range(0.1, 1.0, length=25)
above = range(1.0, 4.0, length=25)
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

    # highlight mode 54
    hlmodes = [55, 72]
    for m in hlmodes
         scatter!(ax,
            real.(tracks[m, :]),
            imag.(tracks[m, :]);
            color      = :red,
            markersize = 8,
            marker     = :circle)
    end

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

    # xlims!(ax, -40, 5)
    # ylims!(ax, -120, 120)
    xlims!(ax, -1, 1)
    ylims!(ax, -2, 2)

    fig
end

show_participation_factors(s0; modes=x -> abs(x/(2π) - (-23+im*100)) < 10, threshold=0.001)
show_participation_factors(s0; modes=x -> abs(x/(2π) - (-0.1+im*0.8)) < 0.5, threshold=0.01)

# We identify 3 critical/interesting modes
# - mode 54: becomes unstable when impedance increses
# - mode 72: super close to zero, small spread copared to others but clear ted to become unstable
# - mode 43: looks similar to 54 (wide spread) but remains stable

cmode = 54
# critical mode is 54-55 pair
@info "Participation factors for baseline case of mode which becomes critical:"
nw, s0 = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=1.0, verbose=false)
show_participation_factors(s0; modes=[cmode])
@info "Show eigenvalue sensitivieits for this case"

# for the sensitivity we only care about parameters of vertex 2 and 3 (gfi/gfl)
# params_of_interest = vidxs(s0, 2:3, s=false, p=true, in=false, out=false, obs=false)
# show_eigenvalue_sensitivity(s0, cmode; params=params_of_interest)

# find the last value which is still stable
idx = findlast(λ -> real(λ) < 0, tracks[cmode, :])
scale_critical = scales[idx]
critical_mode = tracks[cmode, idx]

nw_crit, s0_crit = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=scale_critical, verbose=false)
# find index in critical state which is closest to the critical mode
idx_crit = findmin(λ -> abs(λ - critical_mode), jacobian_eigenvals(s0_crit)./ (2π))[2]
@info "Participation factors for critical mode right before it becomes critical:"
show_participation_factors(s0_crit; modes=idx_crit)
# @info "Show eigenvalue sensitivieits for this case"
# show_eigenvalue_sensitivity(s0_crit, idx_crit; params=params_of_interest)

####
#### Inspect the bus impedance
####
function devices_bode_plot(s0, s0_crit, labels=["Generator Bus", "GFM Bus", "GFL Bus"])
    Gs = map([1, 2, 3]) do COMP
        cs = VIndex(COMP, [:busbar₊i_r, :busbar₊i_i])
        vs = VIndex(COMP, :busbar₊u_mag)
        G = NetworkDynamics.linearize_network(s0; in=cs, out=vs)

        i0 = s0[cs]
        i0 = i0/norm(i0)
        B′ = G.B * i0
        D′ = G.D * i0
        NetworkDescriptorSystem(A=G.A, B=B′, C=G.C, D=D′,
            insym=VIndex(COMP, :busbar₊i_mag), outsym=VIndex(COMP, :busbar₊u_mag))
    end

    Gs_crit = map([1, 2, 3]) do COMP
        cs = VIndex(COMP, [:busbar₊i_r, :busbar₊i_i])
        vs = VIndex(COMP, :busbar₊u_mag)
        G = NetworkDynamics.linearize_network(s0_crit; in=cs, out=vs)

        i0 = s0_crit[cs]
        i0 = i0/norm(i0)
        B′ = G.B * i0
        D′ = G.D * i0
        NetworkDescriptorSystem(A=G.A, B=B′, C=G.C, D=D′,
            insym=VIndex(COMP, :busbar₊i_mag), outsym=VIndex(COMP, :busbar₊u_mag))
    end


    fig = Figure(; size=(800, 600))
    Label(fig[1, 1], L"Z_{dd} Bode Plot", halign=:center, tellwidth=false)
    ax1 = Axis(fig[2, 1], xlabel="Frequency (rad/s)", ylabel="Gain (dB)", xscale=log10)
    ax2 = Axis(fig[3, 1], xlabel="Frequency (rad/s)", ylabel="Phase (deg)", xscale=log10)

    fs = 10 .^ (range(log10(1e-4), log10(1e4); length=1000))
    jωs = 2π * fs * im

    for (i, G, label) in zip(eachindex(Gs), Gs, labels)
        gains = map(s -> 20 * log10(abs(G(s))), jωs)
        phases = rad2deg.(unwrap_rad(map(s -> angle(G(s)), jωs)))
        lines!(ax1, fs, gains; label, linewidth=2, color=Cycled(i))
        lines!(ax2, fs, phases; label, linewidth=2, color=Cycled(i))
    end
    lables_crit = labels .* " (critical)"
    for (i, G, label) in zip(eachindex(Gs_crit), Gs_crit, lables_crit)
        gains = map(s -> 20 * log10(abs(G(s))), jωs)
        phases = rad2deg.(unwrap_rad(map(s -> angle(G(s)), jωs)))
        lines!(ax1, fs, gains; label, linewidth=2, linestyle=:dash, color=Cycled(i))
        lines!(ax2, fs, phases; label, linewidth=2, linestyle=:dash, color=Cycled(i))
    end
    axislegend(ax1)
    fig
end
devices_bode_plot(s0, s0_crit)

####
#### Excursion: are we truly looking at invert interaction?
####
s0_gfl = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=1.0, verbose=false)[2]
s0_gfl_crit = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=scale_critical, verbose=false)[2]
devices_bode_plot(s0_gfl, s0_gfl_crit, ["Generator Bus 1", "Generator Bus 2", "GFL Bus"])

s0_gfm = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=1.0, verbose=false)[2]
s0_gfm_crit = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=scale_critical, verbose=false)[2]
devices_bode_plot(s0_gfm, s0_gfm_crit, ["Generator Bus 1", "GFM Bus", "Generator Bus 2"])
