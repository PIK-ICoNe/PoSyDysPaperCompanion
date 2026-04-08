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
#### Helper Functions
####

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

"""
    find_tracks(states::OrderedDict{Float64}; baseline_key=nothing) -> Matrix{ComplexF64}

Compute eigenvalue tracks from an ordered dict mapping parameter values to NWStates.
Returns an (n_eigs × n_states) matrix where columns are matched across parameter values
by nearest-neighbour chaining from the baseline outward.

`baseline_key` defaults to the key closest to 1.0.
"""
function find_tracks(states::OrderedDict{Float64}; baseline_key=nothing)
    key_vals = collect(keys(states))
    n_steps  = length(key_vals)

    eig_data = [jacobian_eigenvals(s) ./ (2π) for s in values(states)]
    n_eigs   = length(first(eig_data))

    baseline_idx = if isnothing(baseline_key)
        argmin(abs.(key_vals .- 1.0))
    else
        idx = findfirst(==(baseline_key), key_vals)
        isnothing(idx) && error("baseline_key $baseline_key not found in states")
        idx
    end

    baseline_order = sortperm(eig_data[baseline_idx]; by = x -> (real(x), imag(x)))
    tracks = Matrix{ComplexF64}(undef, n_eigs, n_steps)
    tracks[:, baseline_idx] = eig_data[baseline_idx][baseline_order]

    for j in (baseline_idx+1):n_steps
        tracks[:, j] = match_eigenvalues(tracks[:, j-1], eig_data[j])
    end
    for j in (baseline_idx-1):-1:1
        tracks[:, j] = match_eigenvalues(tracks[:, j+1], eig_data[j])
    end

    tracks
end

"""
    plot_tracks(tracks, key_vals; kwargs...) -> Figure

Plot eigenvalue tracks in the complex plane. `key_vals` is the vector of Float64 parameter
values; colours are log-normalised (blue = low, red = high, black cross = baseline).

# Keyword arguments
- `highlight_modes`: list of mode indices to mark with red circles
- `xlims`, `ylims`: axis limits as `(lo, hi)` tuples
- `colormap`: Makie colormap (default `:bluesreds`)
- `title`: axis title
- `baseline_key`: key to use for the baseline marker; defaults to key nearest to 1.0
"""
function plot_tracks(tracks, key_vals;
    highlight_modes = Int[],
    xlims           = nothing,
    ylims           = nothing,
    colormap        = :bluesreds,
    title           = "Eigenvalues",
    baseline_key    = nothing,
)
    log_keys    = log.(key_vals)
    max_log     = maximum(abs.(log_keys))
    norm_colors = iszero(max_log) ? zeros(length(key_vals)) : log_keys ./ max_log

    baseline_idx = if isnothing(baseline_key)
        argmin(abs.(key_vals .- 1.0))
    else
        findfirst(==(baseline_key), key_vals)
    end

    fig = Figure(size=(600, 500))
    ax  = Axis(fig[1, 1];
        xlabel = "Real Part [Hz]",
        ylabel = "Imaginary Part [Hz]",
        title)

    for m in highlight_modes
        scatter!(ax, real.(tracks[m, :]), imag.(tracks[m, :]);
            color = :red, markersize = 8, marker = :circle)
    end

    for i in axes(tracks, 1)
        lines!(ax, real.(tracks[i, :]), imag.(tracks[i, :]);
            color      = norm_colors,
            joinstyle  = :round,
            linecap    = :round,
            colorrange = (-1.0, 1.0),
            colormap,
            linewidth  = 3)
    end

    scatter!(ax, real.(tracks[:, baseline_idx]), imag.(tracks[:, baseline_idx]);
        color = :black, markersize = 6, marker = :xcross)

    !isnothing(xlims) && xlims!(ax, xlims...)
    !isnothing(ylims) && ylims!(ax, ylims...)

    fig
end

####
#### Part 1: Data Generation
####

# Scales to evaluate; 1.0 is the baseline.
# Using logarithmically-spaced values so that scaling above/below 1 is symmetric.
below = range(0.1, 1.0, length=25)
above = range(1.0, 4.0, length=25)
scales = sort!(unique!(vcat(below, above)))

@info "Computing eigenvalues for $(length(scales)) impedance scales..."
eigenvalue_data = OrderedDict{Float64, NWState}()
for Zscale in scales
    @info "  Zscale = $Zscale"
    _, s0 = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=Zscale, verbose=false)
    eigenvalue_data[Zscale] = s0
end

tracks = find_tracks(eigenvalue_data)

####
#### Part 2: Visualization
####

key_vals = collect(keys(eigenvalue_data))

plot_tracks(tracks, key_vals;
    highlight_modes = [55, 72],
    xlims           = (-1, 1),
    ylims           = (-2, 2),
    title           = "Eigenvalues for different Grid Strengths")

show_participation_factors(eigenvalue_data[1.0]; modes=x -> abs(x/(2π) - (-23+im*100)) < 10, threshold=0.001)
show_participation_factors(eigenvalue_data[1.0]; modes=x -> abs(x/(2π) - (-0.1+im*0.8)) < 0.5, threshold=0.01)

# We identify 3 critical/interesting modes
# - mode 54: becomes unstable when impedance increases
# - mode 72: super close to zero, small spread compared to others but clear tend to become unstable
# - mode 43: looks similar to 54 (wide spread) but remains stable

cmode = 54
# critical mode is 54-55 pair
@info "Participation factors for baseline case of mode which becomes critical:"
s0 = eigenvalue_data[1.0]
show_participation_factors(s0; modes=[cmode])

# find the last value which is still stable
idx = findlast(λ -> real(λ) < 0, tracks[cmode, :])
scale_critical = key_vals[idx]
critical_mode  = tracks[cmode, idx]

s0_crit = eigenvalue_data[scale_critical]
# find index in critical state which is closest to the critical mode
idx_crit = findmin(λ -> abs(λ - critical_mode), jacobian_eigenvals(s0_crit) ./ (2π))[2]
@info "Participation factors for critical mode right before it becomes critical:"
show_participation_factors(s0_crit; modes=idx_crit)

####
#### Inspect the bus impedance
####
function compute_Zdd_aligned(s0, bus_idx)
    # Full 2×2 MIMO impedance
    G = NetworkDynamics.linearize_network(s0;
        in  = VIndex(bus_idx, [:busbar₊i_r, :busbar₊i_i]),
        out = VIndex(bus_idx, [:busbar₊u_r, :busbar₊u_i]))

    # Steady-state voltage angle at this bus
    u_r = s0[VIndex(bus_idx, :busbar₊u_r)]
    u_i = s0[VIndex(bus_idx, :busbar₊u_i)]
    θ = atan(u_i, u_r)

    # Rotation matrix: global ri → local dq (d aligned with voltage)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]

    # Rotate: Z_dq = R · Z_ri · R^T
    # In state-space form: new B = B · R^T (rotate inputs)
    #                      new C = R · C   (rotate outputs)
    #                      new D = R · D · R^T
    B_rot = G.B * R'
    C_rot = R * G.C
    D_rot = R * G.D * R'

    # Extract all four components of the 2×2 impedance matrix
    #   rows → output (u_d, u_q);  cols → input (i_d, i_q)
    make = (row, col, isym, osym) -> NetworkDescriptorSystem(
        A = G.A, B = B_rot[:, col], C = C_rot[row:row, :], D = D_rot[row:row, col:col],
        insym  = VIndex(bus_idx, isym),
        outsym = VIndex(bus_idx, osym))

    Zdd = make(1, 1, :i_d_local, :u_d_local)   # active I  → magnitude V  (resistive path)
    Zdq = make(1, 2, :i_q_local, :u_d_local)   # reactive I → magnitude V  (Q-V path)
    Zqd = make(2, 1, :i_d_local, :u_q_local)   # active I  → angle V      (P-ω path)
    Zqq = make(2, 2, :i_q_local, :u_q_local)   # reactive I → angle V     (inductive path)

    return Zdd, Zdq, Zqd, Zqq
end

function devices_bode_plot(s0, s0_crit; labels=["Generator Bus", "GFM Bus", "GFL Bus"])
    unzip4(v) = (getindex.(v,1), getindex.(v,2), getindex.(v,3), getindex.(v,4))

    Gs_dd,  Gs_dq,  Gs_qd,  Gs_qq  = unzip4(map(b -> compute_Zdd_aligned(s0,      b), [1,2,3]))
    Gsc_dd, Gsc_dq, Gsc_qd, Gsc_qq = unzip4(map(b -> compute_Zdd_aligned(s0_crit, b), [1,2,3]))

    # Layout matches the matrix:  [Z_dd  Z_dq]
    #                              [Z_qd  Z_qq]
    # Each cell = stacked gain + phase  →  4 plot-rows × 2 plot-cols
    #                                       + header row 0 + row-group labels in col 0
    fs  = 10 .^ range(log10(1e-2), log10(1e3); length=800)
    jωs = 2π .* fs .* im
    labels_crit = labels .* " (crit.)"

    fig = Figure(; size=(900, 900))

    # Column headers (input axis)
    Label(fig[0, 2], L"d\text{-axis input}\ (i_d,\ \text{active})";   halign=:center, tellwidth=false)
    Label(fig[0, 3], L"q\text{-axis input}\ (i_q,\ \text{reactive})"; halign=:center, tellwidth=false)

    # Matrix cells: (matrix_row, matrix_col) → plot rows (gain, phase) and figure col
    cell_data = [
        (1, 1, Gs_dd,  Gsc_dd, L"Z_{dd}"),
        (1, 2, Gs_dq,  Gsc_dq, L"Z_{dq}"),
        (2, 1, Gs_qd,  Gsc_qd, L"Z_{qd}"),
        (2, 2, Gs_qq,  Gsc_qq, L"Z_{qq}"),
    ]
    # Row-group labels (output axis); placed once per matrix row
    Label(fig[1:2, 1], L"d\text{-axis output}\ (u_d,\ \text{magnitude})"; rotation=π/2, tellheight=false)
    Label(fig[3:4, 1], L"q\text{-axis output}\ (u_q,\ \text{angle})";     rotation=π/2, tellheight=false)

    for (mrow, mcol, Gbase, Gcrit, zlabel) in cell_data
        gain_row  = 2*(mrow-1) + 1   # 1 or 3
        phase_row = 2*(mrow-1) + 2   # 2 or 4
        fig_col   = mcol + 1          # 2 or 3  (col 1 reserved for row labels)

        ax_g = Axis(fig[gain_row,  fig_col];
            ylabel    = Makie.LaTeXStrings.LaTeXString("$(zlabel) Gain (dB)"),
            xscale    = log10)
        ax_p = Axis(fig[phase_row, fig_col];
            ylabel    = "Phase (deg)",
            xlabel    = phase_row == 4 ? "Frequency (Hz)" : "",
            xscale    = log10)

        for (i, label) in enumerate(labels)
            g  = Gbase[i];  gc = Gcrit[i]
            lines!(ax_g, fs, map(s -> 20log10(abs(g(s))),  jωs); label, linewidth=2, color=Cycled(i))
            lines!(ax_p, fs, rad2deg.(unwrap_rad(map(s -> angle(g(s)),  jωs))); label, linewidth=2, color=Cycled(i))
            lines!(ax_g, fs, map(s -> 20log10(abs(gc(s))), jωs); label=labels_crit[i], linewidth=2, linestyle=:dash, color=Cycled(i))
            lines!(ax_p, fs, rad2deg.(unwrap_rad(map(s -> angle(gc(s)), jωs))); label=labels_crit[i], linewidth=2, linestyle=:dash, color=Cycled(i))
        end
        mrow == 1 && mcol == 1 && axislegend(ax_g; position=:rb)
    end
    fig
end
devices_bode_plot(s0, s0_crit)

####
#### Excursion: are we truly looking at inverter interaction?
####
s0_gfl = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=1.0, verbose=false)[2]
s0_gfl_crit = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=scale_critical, verbose=false)[2]
devices_bode_plot(s0_gfl, s0_gfl_crit; labels=["Generator Bus 1", "Generator Bus 2", "GFL Bus"])

s0_gfm = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=1.0, verbose=false)[2]
s0_gfm_crit = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=scale_critical, verbose=false)[2]
devices_bode_plot(s0_gfm, s0_gfm_crit; labels=["Generator Bus 1", "GFM Bus", "Generator Bus 2"])
