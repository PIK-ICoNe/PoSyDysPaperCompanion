#=
# Differentiable Simulation for Power System Dynamics — Companion Script

This script accompanies the PoSyDyS 2026 paper and demonstrates three analysis
workflows enabled by the differentiable simulation framework PowerDynamics.jl:
eigenvalue analysis (instability under grid weakening), impedance extraction
(Zqd resonance corroboration), participation factors and eigenvalue sensitivity
(motivation for parameter selection), and gradient-based controller optimization.

All analyses use the same model definition — a modified IEEE 9-bus system where
generators at buses 2 and 3 are replaced by a grid-forming droop inverter and a
grid-following current-source inverter.
=#

using PowerDynamics
using PowerDynamics.Library
using ModelingToolkit
using PoSyDysPaperCompanion
using NetworkDynamics
using Graphs
using DiffEqCallbacks
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve
using CairoMakie
using LinearAlgebra
using OrderedCollections: OrderedDict
using SymbolicIndexingInterface: SymbolicIndexingInterface as SII
using Optimization, Optimisers, OptimizationOptimisers

#=
Figure Export Specifications for IEEE Conference (IEEEtran, conference mode)

COLUMN WIDTHS (from IEEEtran.cls, letter paper, 0.625 in margins, 0.25 in column sep)
  Single-column figure  →  \columnwidth = 3.5 in = 88.9 mm = 252 pt
  Double-column figure  →  \textwidth   = 7.25 in = 184.2 mm = 522 pt

RECOMMENDED FORMAT
  Line art / plots  →  vector PDF (DPI-independent, preferred by IEEE)
  Photos / raster   →  PNG or TIFF, 300 DPI minimum (600 DPI for line art mixed with raster)
  Source: IEEEtran_HOWTO.pdf §Appendix B; IEEE Author Center submission guidelines

CAIROMAKIE GUIDANCE
  Set figure sizes in pt to match LaTeX column widths exactly:
    Single-column:  Figure(size = (252, <height>))
    Double-column:  Figure(size = (522, <height>))
  Typical height choices (golden ratio): single ≈ 156 pt, double ≈ 323 pt

  PDF export (vector, preferred):
    save("figures/myfig.pdf", fig; pt_per_unit = 1)
    → 1 Makie unit = 1 PDF point, so figure renders at exactly the right size in LaTeX

  PNG export (raster fallback):
    save("figures/myfig.png", fig; px_per_unit = 4)   # ≈ 288 DPI  (300 DPI target)
    save("figures/myfig.png", fig; px_per_unit = 8)   # ≈ 576 DPI  (600 DPI target)
    px_per_unit multiplies the pt-based figure size: 252 pt × 8 = 2016 px wide

## Makie Themes for IEEE Conference Figures

Two themes matching the IEEEtran column widths. Both use NewComputerModern fonts
(via `theme_latexfonts()`) to match the LaTeX document fonts exactly.

Usage:
    with_theme(ieee_theme) do          # single-column (252 × 156 pt)
        fig = Figure()                 # size comes from the theme
        ...
        save("fig.pdf", fig; pt_per_unit = 1)
    end

    with_theme(ieee_theme_wide) do     # double-column (522 × 323 pt)
        ...
    end

Note: `Theme(Figure = (size = ...,))` works in current Makie but is not part of
the official public API for Figure — if it ever breaks, fall back to passing
`Figure(size = IEEE_SINGLE_COL_PT)` or `Figure(size = IEEE_DOUBLE_COL_PT)` explicitly.
=#

## Canonical figure dimensions in Makie pt units (1 pt = 1 CSS/PDF point)

# define methods for "default" size or thime size
theme_size() = theme_size(600, 400) # default size
function theme_size(x, y)
    if hasproperty(Makie.current_default_theme(), :Figure) && hasproperty(Makie.current_default_theme().Figure, :size)
        Makie.current_default_theme().Figure.size
    else
        (x, y)
    end
end

FIGPATH = joinpath(pkgdir(PoSyDysPaperCompanion), "paper", "figures")
mkpath(FIGPATH)

function base_theme()
    merge(
        theme_latexfonts(),
        Theme(
            fontsize       = 8,
            figure_padding = 2,   # tight padding — LaTeX handles outer whitespace
            palette        = (color = Makie.wong_colors(),),

            Axis = (
                spinewidth         = 0.5,
                xtickwidth         = 0.5,
                ytickwidth         = 0.5,
                xticksize          = 3,
                yticksize          = 3,
                xticklabelsize     = 7,
                yticklabelsize     = 7,
                xlabelsize         = 8,
                ylabelsize         = 8,
                # titlesize          = 8,
                # xgridvisible       = false,
                # ygridvisible       = false,
                # xminorticksvisible = true,
                # yminorticksvisible = true,
                # xminorticks        = IntervalsBetween(5),
                # yminorticks        = IntervalsBetween(5),
            ),

            Legend = (
                labelsize  = 7,
                titlesize  = 8,
                framewidth = 0.5,
                padding    = (4f0, 4f0, 4f0, 4f0),
                patchsize  = (14f0, 6f0),
                rowgap     = 2,
            ),

            Lines    = (linewidth = 1.25,),
            Scatter  = (markersize = 4,),

            Colorbar = (
                ticklabelsize = 7,
                labelsize     = 8,
                width         = 10,
                spinewidth    = 0.5,
            ),
            rowgap = 4,
            colgap = 4,
        )
    )
end
function ieee_theme(scale=1/sqrt(2))
    w = 252 # \columwidth
    merge(base_theme(), Theme(Figure = (size = (w, floor(Int, scale * w)),)))
end
function ieee_theme_wide(scale=1/sqrt(2))
    w = 522 # \textwidth
    merge(base_theme(), Theme(Figure = (size = (w, floor(Int, scale * w)),)))
end

#=
## The Test System

We build three variants of the IEEE 9-bus:
- **All-SG (RMS)**: classical SauerPai machines with AVR and governor (static PiLines)
- **All-SG (EMT)**: same machines but with stator dynamics and dynamic RL lines
- **Mixed (EMT)**: Bus 1 = SG, Bus 2 = GFM droop inverter, Bus 3 = GFL current source

The mixed system is our main study object. Varying the line impedance scaling
parameter `Zscale` simulates different grid strengths.
=#

nw_rms, s0_rms = load_ieee9bus()
nw_emt, s0_emt = load_ieee9bus_emt()
nw_mix, s0_mix = load_ieee9bus_emt(gfm=true, gfl=true)

#=
### Power Flow Validation

All three models share the same power flow solution — the dynamic model choice
does not affect the steady-state operating point.
=#
pf_rms = show_powerflow(nw_rms)
pf_emt = show_powerflow(nw_emt)
pf_mix = show_powerflow(nw_mix)

@assert pf_rms."vm [pu]" ≈ pf_emt."vm [pu]" ≈ pf_mix."vm [pu]"
@assert pf_rms."varg [deg]" ≈ pf_emt."varg [deg]" ≈ pf_mix."varg [deg]"

#=
## Reference Simulations: Load Disturbance

We apply a 10% increase in real power consumption at Bus 8 (load bus between the
GFM and GFL corners of the network). This excites both inverters through the network
and reveals the oscillatory dynamics governed by the eigenvalues we analyze later.
=#

function make_load_step_problem(s0; tspan=(-5.0, 1.0), ΔG_factor=1.1)
    nw = extract_nw(s0)
    affect! = integrator -> begin
        s = NWState(integrator)
        s[VIndex(8, :load₊G)] *= ΔG_factor
        save_parameters!(integrator)
    end
    cb = PresetTimeCallback(0.0, affect!)
    ODEProblem(nw, s0, tspan; add_nw_cb=cb)
end

#=
### All-SG vs Mixed System (Nominal Grid)

The all-SG system shows classical ~0.25 Hz electromechanical oscillations between
the generators. The mixed system settles much faster — the inverters have no
physical inertia and respond on converter control timescales instead.
=#

sol_emt, sol_mix = let
    prob_emt = make_load_step_problem(s0_emt)
    prob_mix = make_load_step_problem(s0_mix)
    solve(prob_emt, Rodas5P()), solve(prob_mix, Rodas5P())
end;

let fig = Figure(size=theme_size(800, 600))
    ts = range(-0.01, 1.0, length=1000)

    ax1 = Axis(fig[1, 1]; title="Voltage Magnitude — All-SG (EMT)",
        xlabel="Time [s]", ylabel="Voltage [pu]")
    ax2 = Axis(fig[1, 2]; title="Voltage Magnitude — Mixed SG+IBR",
        xlabel="Time [s]", ylabel="Voltage [pu]")

    for i in [1, 2, 3]
        lines!(ax1, ts, sol_emt(ts, idxs=VIndex(i, :busbar₊u_mag)).u;
            label="Bus $i", color=Cycled(i))
        lines!(ax2, ts, sol_mix(ts, idxs=VIndex(i, :busbar₊u_mag)).u;
            label="Bus $i", color=Cycled(i))
    end
    axislegend(ax1; position=:rb)
    axislegend(ax2; position=:rb)
    fig
end

#=
### Effect of Grid Weakness

Increasing `Zscale` (scaling all non-transformer line impedances) weakens the grid.
The mixed system develops visible oscillations at ~28 Hz as the converter-filter
resonance mode loses damping.
=#

function plot_load_step_comparison(scales; kwargs...)
    s0s = [load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=s, verbose=false)[2] for s in scales]
    sols = [solve(make_load_step_problem(s0; kwargs...), Rodas5P()) for s0 in s0s]

    fig = Figure(size=theme_size(600, length(scales) * 250))
    for (row, (s, sol)) in enumerate(zip(scales, sols))
        ax = Axis(fig[row, 1]; ylabel="Voltage [pu]", title=L"$Z$ scale = %$s")
        if row == length(s0s)
            ax.xlabel = "Time [s]"
        else
            hidexdecorations!(ax; label=true, ticklabels=false, ticks=false, grid=false, minorgrid=false, minorticks=false)
        end
        ts = refine_timeseries(sol.t)
        for i in [1, 2, 3]
            lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_mag)).u;
                label="Bus $i", color=Cycled(i))
        end
        xlims!(ax, -0.01, 0.5)
        row == 1 && axislegend(ax; position=:rb)
    end
    fig
end

plot_load_step_comparison([1.0, 1.5, 2.0])
let
    fig = with_theme(ieee_theme(1)) do
        plot_load_step_comparison([1.0, 1.5, 2.0])
    end
    save(joinpath(FIGPATH, "01_load_step_scenarios.pdf"), fig)
    fig
end

#=
## Eigenvalue Analysis: Impedance Sweep

We compute eigenvalues across a range of `Zscale` values from 0.1 (very strong grid)
to 4.0 (very weak grid) and track how each mode migrates in the complex plane.
=#

function match_eigenvalues(ref_eigs, new_eigs)
    n = length(ref_eigs)
    matched = zeros(ComplexF64, n)
    available = collect(1:n)
    for i in 1:n
        dists = abs.(ref_eigs[i] .- new_eigs[available])
        best = argmin(dists)
        matched[i] = new_eigs[available[best]]
        deleteat!(available, best)
    end
    matched
end

function find_tracks(states::OrderedDict{Float64})
    key_vals = collect(keys(states))
    eig_data = [jacobian_eigenvals(s) ./ (2π) for s in values(states)]
    n_eigs = length(first(eig_data))

    baseline_idx = argmin(abs.(key_vals .- 1.0))
    baseline_order = sortperm(eig_data[baseline_idx]; by=x -> (real(x), imag(x)))
    tracks = Matrix{ComplexF64}(undef, n_eigs, length(key_vals))
    tracks[:, baseline_idx] = eig_data[baseline_idx][baseline_order]

    for j in (baseline_idx+1):length(key_vals)
        tracks[:, j] = match_eigenvalues(tracks[:, j-1], eig_data[j])
    end
    for j in (baseline_idx-1):-1:1
        tracks[:, j] = match_eigenvalues(tracks[:, j+1], eig_data[j])
    end
    tracks
end

scales_sweep = sort!(unique!(vcat(
    range(0.1, 1.0; length=25),
    range(1.0, 4.0; length=25)
)))

eigenvalue_data = let d = OrderedDict{Float64, NWState}()
    for Zscale in scales_sweep
        _, s0 = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale, verbose=false)
        d[Zscale] = s0
    end
    d
end

tracks = find_tracks(eigenvalue_data)
scales_vec = collect(keys(eigenvalue_data))

#=
### Eigenvalue Trajectory Plot

Colours encode the impedance scaling (blue = strong grid, red = weak grid).
Black crosses mark the baseline (Zscale = 1.0). We highlight two modes:
- **Mode 54/55**: coupled converter-filter resonance — migrates toward instability
- **Mode 72**: SG electromechanical mode — also loses damping but is secondary
=#

function plot_tracks(tracks, key_vals;
        ax=nothing,
        highlight_modes=Int[], xlims=nothing, ylims=nothing, title="Eigenvalues",
        colorbar=true, faint=false)
    log_keys = log.(key_vals)
    pos_max = max(maximum(log_keys), 0.0)
    neg_max = max(-minimum(log_keys), 0.0)
    norm_colors = map(l -> l >= 0 ? (iszero(pos_max) ? 0.0 : l / pos_max)
                                  : (iszero(neg_max) ? 0.0 : l / neg_max), log_keys)
    color_low = neg_max > 0 ? -1.0 : 0.0
    faintscheme = Makie.ColorScheme([colorant"darkgray", colorant"gray95", colorant"darkgray"])
    basecmap = faint ? faintscheme : :bluesreds
    pos_only_cmap = Makie.resample_cmap(basecmap, 256)[129:end]
    track_cmap = neg_max > 0 ? basecmap : pos_only_cmap
    baseline_idx = argmin(abs.(key_vals .- 1.0))

    own_fig = isnothing(ax)
    if own_fig
        fig = Figure(size=theme_size(600, 500))
        ax = Axis(fig[1, 1]; xlabel="Real Part [Hz]", ylabel="Imaginary Part [Hz]", title)
    end

    faint || vspan!(ax, 0, 100; color=(:red, 0.07))

    for m in highlight_modes
        lines!(ax, real.(tracks[m, :]), imag.(tracks[m, :]);
            color=:yellow,
            alpha=0.3,
            linewidth=6,
            joinstyle=:round,
            linecap=:round,
        )
    end

    for i in axes(tracks, 1)
        lines!(ax, real.(tracks[i, :]), imag.(tracks[i, :]);
            color=norm_colors,
            colorrange=(color_low, 1.0),
            colormap=track_cmap,
            linewidth=2,
            joinstyle=:round,
            linecap=:round,
        )
    end
    scatter!(ax, real.(tracks[:, baseline_idx]), imag.(tracks[:, baseline_idx]);
        color=faint ? :darkgray : :black,
        markersize=4, marker=:xcross)

    !isnothing(xlims) && Makie.xlims!(ax, xlims...)
    !isnothing(ylims) && Makie.ylims!(ax, ylims...)

    if own_fig
        if colorbar && (pos_max > 0 || neg_max > 0)
            kmin, kmax = minimum(key_vals), maximum(key_vals)
            cb_low  = neg_max > 0 ? -1.0 : 0.0
            cb_high = pos_max > 0 ?  1.0 : 0.0
            cb_ticks = filter(t -> cb_low <= t[1] <= cb_high, [
                (-1.0, string(kmin)), (0.0, "1.0"), (1.0, string(kmax))])
            Colorbar(fig[1, 2]; colormap=track_cmap, colorrange=(cb_low, cb_high),
                ticks=(first.(cb_ticks), last.(cb_ticks)),
                width=4, vertical=true)
        end
        return fig
    end
    ax
end

xlims_full = (-40, 5)
xlims_zoom = (-5, 5/8)
ylims_full = (-110, 110)
ylims_zoom = (-5, 5)

plot_tracks(tracks, scales_vec;
    highlight_modes=[54, 55],
    xlims=xlims_full,
    ylims=ylims_full,
    title="Eigenvalue Paths under Varying Grid Strength"
)
let
    fig = with_theme(ieee_theme()) do
        plot_tracks(tracks, scales_vec;
            highlight_modes=[54, 55],
            xlims=xlims_full,
            ylims=ylims_full,
            title="Eigenvalue Paths under Varying Grid Strength"
        )
    end
    save(joinpath(FIGPATH, "02_eigenvalue_paths.pdf"), fig)
end

## Identify the critical mode and the nominal / critical operating points used throughout
cmode = 54
s0_nom = eigenvalue_data[1.0]

idx_last_stable = findlast(λ -> real(λ) < 0, tracks[cmode, :])
scale_critical = scales_vec[idx_last_stable]
s0_crit = eigenvalue_data[scale_critical]
critical_mode = tracks[cmode, idx_last_stable]
idx_crit = findmin(λ -> abs(λ - critical_mode), jacobian_eigenvals(s0_crit) ./ (2π))[2]

#=
## Impedance Extraction via Linearization

Using `linearize_network`, we extract the driving-point impedance at each bus by
perturbing the bus current and observing the voltage response. The result is rotated
into a local dq frame aligned with the steady-state voltage.
=#

function compute_Z_aligned(s0, bus_idx)
    G = NetworkDynamics.linearize_network(s0;
        in  = VIndex(bus_idx, [:busbar₊i_r, :busbar₊i_i]),
        out = VIndex(bus_idx, [:busbar₊u_r, :busbar₊u_i]))

    u_r = s0[VIndex(bus_idx, :busbar₊u_r)]
    u_i = s0[VIndex(bus_idx, :busbar₊u_i)]
    θ = atan(u_i, u_r)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]

    B_rot = G.B * R'
    C_rot = R * G.C
    D_rot = R * G.D * R'

    make = (row, col) -> NetworkDescriptorSystem(
        A=G.A, B=B_rot[:, col], C=C_rot[row:row, :], D=D_rot[row:row, col:col],
        insym=VIndex(bus_idx, :local), outsym=VIndex(bus_idx, :local))

    (; Zdd=make(1,1), Zdq=make(1,2), Zqd=make(2,1), Zqq=make(2,2))
end

#=
### Bode Plots: Nominal vs Critical Grid Strength

The network is firmly inductive (R/X ≈ 0.12), so the dominant coupling channel is
ΔP → Δδ, captured by Z_qd (q-voltage response to d-current perturbation). The sharp
resonance peak at ~28 Hz (visible at critical Zscale) corresponds exactly to the
eigenvalue mode that destabilizes. At nominal grid strength, the same mode is
well-damped and barely visible.
=#

function plot_Zqd_bode(s0_a, s0_b; buses=[2,3],
        labels=["GFM Bus", "GFL Bus"],
        label_suffix=["nominal", "critical"])
    fs = 10 .^ range(-1.5, 3.5; length=800)
    jωs = 2π .* fs .* im

    fig = Figure(size=theme_size(700, 500))
    ax_g = Axis(fig[1, 1]; ylabel="Gain (dB)", xscale=log10, title=L"$Z_{qd}$ Bode Plot")

    phaseticks = (-10:2:10)*π
    phaselabels = [n==0 ? "0" : string(n) .* "π" for n in -10:2:10]
    ax_p = Axis(fig[2, 1]; ylabel="Phase (rad)", xlabel="Frequency (Hz)", xscale=log10,
        yticks=(phaseticks, phaselabels),
    )

    for (i, (bus, lab)) in enumerate(zip(buses, labels))
        Za = compute_Z_aligned(s0_a, bus)
        Zb = compute_Z_aligned(s0_b, bus)

        lines!(ax_g, fs, [20log10(abs(Za.Zqd(s))) for s in jωs];
            label="$lab ($(label_suffix[1]))", color=Cycled(i), alpha=0.6)
        lines!(ax_g, fs, [20log10(abs(Zb.Zqd(s))) for s in jωs];
            label="$lab ($(label_suffix[2]))", color=Cycled(i), linestyle=(:dash, :dense))

        lines!(ax_p, fs, unwrap_rad([angle(Za.Zqd(s)) for s in jωs]);
            label="$lab ($(label_suffix[1]))", color=Cycled(i), alpha=0.6)
        lines!(ax_p, fs, unwrap_rad([angle(Zb.Zqd(s)) for s in jωs]);
            label="$lab ($(label_suffix[2]))", color=Cycled(i), linestyle=(:dash, :dense))
    end
    xlims!(ax_p, extrema(fs))
    xlims!(ax_g, extrema(fs))
    axislegend(ax_p; position=:lb)
    fig
end

plot_Zqd_bode(s0_nom, s0_crit)

let
    fig = with_theme(ieee_theme(3/4)) do
        plot_Zqd_bode(s0_nom, s0_crit)
    end
    save(joinpath(FIGPATH, "03_Zqd_bode.pdf"), fig)
    fig
end

#=
### Cross-Validation: Is This Truly an Inverter-Inverter Interaction?

We check whether the resonance appears with only one inverter present.
It does not — confirming a coupled interaction between the GFM and GFL
mediated by the network.
=#

let
    s0_gfl_only = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=1.0, verbose=false)[2]
    s0_gfl_crit = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=scale_critical, verbose=false)[2]
    plot_Zqd_bode(s0_gfl_only, s0_gfl_crit;
        labels=["SG Bus 1", "SG Bus 2", "GFL Bus"],
        label_suffix=["nominal (GFL only)", "critical (GFL only)"])
end

let
    s0_gfm_only = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=1.0, verbose=false)[2]
    s0_gfm_crit = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=scale_critical, verbose=false)[2]
    plot_Zqd_bode(s0_gfm_only, s0_gfm_crit;
        labels=["SG Bus 1", "GFM Bus", "SG Bus 3"],
        label_suffix=["nominal (GFM only)", "critical (GFM only)"])
end

#=
### Participation Factors

At nominal grid strength, the critical mode (54/55) is dominated by the inner current
controller integrator states of both inverters — a coupled filter resonance. Near
instability, the GFL's PLL angle enters the participation, revealing the mechanism
that drives destabilization.
=#

@info "Participation factors at nominal grid strength (Zscale=1.0):"
show_participation_factors(s0_nom; modes=[cmode], threshold=0.05)

@info "Participation factors near instability (Zscale=$scale_critical):"
show_participation_factors(s0_crit; modes=idx_crit, threshold=0.05)

open(joinpath(FIGPATH, "participation_factors.txt"), "w") do io
    println(io, "=== Nominal grid strength (Zscale=1.0) ===")
    show_participation_factors(io, s0_nom; modes=[cmode], threshold=0.05)
    println(io)
    println(io, "=== Near instability (Zscale=$scale_critical) ===")
    show_participation_factors(io, s0_crit; modes=idx_crit, threshold=0.05)
end

#=
### Eigenvalue Sensitivity

We compute the sensitivity of the critical mode to all controller parameters of the
GFM (bus 2) and GFL (bus 3) inverters. The ranking shifts significantly between
nominal and weak-grid conditions.
=#

params_of_interest = let
    candidates = vidxs(s0_nom, 2:3, s=false, p=true, in=false, out=false, obs=false)
    # we are only interested in control parameters, so we filter for "filter" states
    filter!(candidates) do idx
        name = string(idx.subidx)
        !(
            contains(name, "connected") || # drop topology parameter
            contains(name, r"Lf$")      || # drop electrical filter parameters
            contains(name, r"Rf$")      ||
            contains(name, r"C$")       ||
            contains(name, r"Lg$")      ||
            contains(name, r"Rg$")      ||
            contains(name, r"ω0$")         # drop base frequency
        )
    end
end

@info "Eigenvalue sensitivities at nominal (Zscale=1.0):"
show_eigenvalue_sensitivity(s0_nom, cmode; params=params_of_interest)

@info "Eigenvalue sensitivities near instability (Zscale=$scale_critical):"
show_eigenvalue_sensitivity(s0_crit, idx_crit; params=params_of_interest)

open(joinpath(FIGPATH, "parameter_sensitivities.txt"), "w") do io
    println(io, "=== Nominal grid strength (Zscale=1.0) ===")
    show_eigenvalue_sensitivity(io, s0_nom, cmode; params=params_of_interest)
    println(io)
    println(io, "=== Near instability (Zscale=$scale_critical) ===")
    show_eigenvalue_sensitivity(io, s0_crit, idx_crit; params=params_of_interest)
end

#=
## Gradient-Based Controller Optimization

We optimize 7 GFL controller parameters to minimize voltage oscillation energy
after a load step, evaluated at three grid strengths simultaneously. The gradients
are computed via forward-mode AD through the ODE solver.
=#

tunable_p = collect(VIndex(3, [
    :gfl₊csrc₊PLL_Kp, # PLL parameters
    :gfl₊csrc₊PLL_Ki,
    :gfl₊csrc₊CC1_KI, # inner current loop parameters
    :gfl₊csrc₊CC1_KP,
    :gfl₊csrc₊CC2_F,  # outer current loop parameters
    :gfl₊csrc₊CC2_KP,
    :gfl₊csrc₊CC2_KI,
]))

#=
### Loss Function

For each grid strength scenario, we simulate a 10% load increase at Bus 8 and
measure the squared voltage deviation from the post-disturbance steady state
at buses 2, 3, and 8. This penalizes oscillations (poor damping) while being
insensitive to the steady-state shift caused by the load change.
=#

opt_scales = [1.0, 1.5, 2.0]

opt_s0s = [load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=s, verbose=false)[2] for s in opt_scales];

opt_problems = [make_load_step_problem(s0; tspan=(-10.0, 1.0)) for s0 in opt_s0s];

tpidx = SII.parameter_index(extract_nw(first(opt_s0s)), tunable_p)

function generate_loss(problems, tpidx)
    t_eval = range(0.0, 1.0; length=1000)
    pdim_nw = NetworkDynamics.pdim(extract_nw(first(problems)))

    loss = p -> begin
        total = 0.0
        for prob in problems
            p_new = zeros(eltype(p), pdim_nw)
            p_new .= prob.p
            p_new[tpidx] .= p

            sol = solve(prob, Rodas5P(autodiff=true); p=p_new, saveat=t_eval)
            !SciMLBase.successful_retcode(sol) && return Inf

            all_V = reduce(hcat, sol(t_eval; idxs=VIndex([2, 3, 8], :busbar₊u_mag)).u)
            for col in 1:3
                V_t = all_V[col, :]
                total += sum((V_t .- V_t[end]).^2) / length(t_eval)
            end
        end
        total
    end
    p0 = first(problems).p[tpidx]
    loss, p0
end

loss_fn, p0_opt = generate_loss(opt_problems, tpidx);

#=
### Run Optimization

Adam optimizer with learning rate 5e-3 for 100 iterations. The slower descent
gives better convergence and also produces smooth eigenvalue trajectory frames for
the animation below.
=#

optsol, opt_states = let states = Any[]
    optf = Optimization.OptimizationFunction((x, _) -> loss_fn(x), Optimization.AutoForwardDiff())
    cb = (state, l) -> begin
        push!(states, state)
        is_best = all(s -> l < s.objective, states[1:end-1])
        print("Iteration $(state.iter): loss = $l")
        is_best && printstyled(" ✓"; color=:green)
        println()
        false
    end
    optprob = Optimization.OptimizationProblem(optf, p0_opt; callback=cb)
    sol = @time Optimization.solve(optprob, Optimisers.Adam(5e-3); maxiters=100)
    best = Inf
    frames = filter(states) do s
        s.objective < best ? (best = s.objective; true) : false
    end
    sol, frames
end

#=
### Optimized Parameters

The optimization reveals that the inner current loop proportional gain (CC1_KP) needed
a large increase, while the outer current loop gain (CC2_KP) and grid voltage
feedforward (CC2_F) needed reduction. The PLL and CC1 integral gains were already
near-optimal.
=#

for (sym, orig, tuned) in zip(tunable_p, p0_opt, optsol.u)
    pct = round(100 * (tuned - orig) / orig; digits=1)
    println("$sym\t$orig → $(round(tuned; sigdigits=4))\t($pct%)")
end

open(joinpath(FIGPATH, "tuned_parameters.txt"), "w") do io
    println(io, "symbol\toriginal\ttuned\tchange")
    for (sym, orig, tuned) in zip(tunable_p, p0_opt, optsol.u)
        pct = round(100 * (tuned - orig) / orig; digits=1)
        println(io, "$sym\t$orig\t$(round(tuned; sigdigits=4))\t$pct%")
    end
end

#=
### Before/After Comparison: Time Domain

We compare the load step response with default vs optimized parameters at all
three grid strengths. The oscillations at Zscale=2.0 are dramatically reduced.
=#

sols_default = [solve(prob, Rodas5P()) for prob in opt_problems];

sols_optimized = let p_opt = optsol.u
    map(opt_problems) do prob
        p_new = copy(prob.p)
        p_new[tpidx] .= p_opt
        solve(prob, Rodas5P(); p=p_new)
    end
end;

function plot_before_after(sols_before, sols_after, scales; buses=[2, 3, 8], labels=["GFM Bus", "GFL Bus", "Disturbed Load Bus"])
    fig = Figure(size=theme_size(800, length(scales) * 250))
    nrows = length(scales)
    for (row, (s, sb, sa)) in enumerate(zip(scales, sols_before, sols_after))
        ax1 = Axis(fig[row, 1]; title=L"$Z$ scale=%$s — default", ylabel="V [pu]")
        ax2 = Axis(fig[row, 2]; title=L"$Z$ scale=%$s — optimized", ylabel="V [pu]")
        if row == nrows
            ax1.xlabel = "Time [s]"
            ax2.xlabel = "Time [s]"
        else
            hidexdecorations!(ax1; label=true, ticklabels=false, ticks=false, grid=false, minorgrid=false, minorticks=false)
            hidexdecorations!(ax2; label=true, ticklabels=false, ticks=false, grid=false, minorgrid=false, minorticks=false)
        end
        ts_b = refine_timeseries(sb.t)
        ts_a = refine_timeseries(sa.t)
        for (j, (bus, label)) in enumerate(zip(buses, labels))
            lines!(ax1, ts_b, sb(ts_b, idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label)
            lines!(ax2, ts_a, sa(ts_a, idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label)
        end
        xlms = (-0.01, 0.5)
        ylms = (0.929, 1.035)
        xlims!(ax1, xlms...); xlims!(ax2, xlms...)
        ylims!(ax1, ylms...); ylims!(ax2, ylms...)
        row == 1 && axislegend(ax1; position=:rb)
    end
    fig
end

plot_before_after(sols_default, sols_optimized, opt_scales)

let
    fig = with_theme(ieee_theme_wide(0.5)) do
        plot_before_after(sols_default, sols_optimized, opt_scales)
    end
    save(joinpath(FIGPATH, "04_load_step_before_after.pdf"), fig)
    fig
end

#=
### Before/After Comparison: Eigenvalue Trajectories

We recompute eigenvalues with optimized parameters at each grid strength and overlay
them with the baseline. The critical mode should have moved left (more damping).
=#

function reinitialize_with_params(s0, tunable_p, p_new)
    _nw = extract_nw(s0)
    nw = Network(_nw)
    for (pidx, p) in zip(tunable_p, p_new)
        set_default!(nw[VIndex(pidx.compidx)], pidx.subidx, p)
    end
    initialize_from_pf!(nw; tol=1e-7, nwtol=1e-5, verbose=false, warn=false)
end

## Pre-load base systems across the sweep once — reused for every eigenvalue_tracks call.
eig_sweep_scales = range(1.0, 4.0; length=25)
eig_sweep_s0s = [
    load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=Float64(s), verbose=false)[2]
    for s in eig_sweep_scales
];

"""
    compute_eig_tracks(popt; scales, base_s0s) -> (tracks, scales_vec)

Reinitialize the mixed IEEE-9 system with controller parameters `popt` at each
`Zscale` in `scales` and return the matched eigenvalue track matrix together with
the corresponding scale vector. Separating computation from plotting lets callers
draw the same tracks onto multiple axes without redundant reinitialization.
"""
function compute_eig_tracks(popt;
        scales   = eig_sweep_scales,
        base_s0s = eig_sweep_s0s)
    states = OrderedDict{Float64, NWState}()
    for (s, s0) in zip(scales, base_s0s)
        states[Float64(s)] = reinitialize_with_params(s0, tunable_p, popt)
    end
    find_tracks(states), collect(keys(states))
end
tracks_default = compute_eig_tracks(p0_opt)
tracks_optimized = compute_eig_tracks(optsol.u)

function plot_optimized_tracks((tr_def, sv_def), (tr_opt, sv_opt))
    # filter default tracks to scales > 1 to match the optimized sweep range
    mask = sv_def .> 1.0

    fig = Figure(size=theme_size(600, 500))
    ax = Axis(fig[1, 1]; xlabel="Real Part [Hz]", ylabel="Imaginary Part [Hz]",
        title="Eigenvalue Paths under Varying Grid Strength")

    plot_tracks(tr_def[:, mask], sv_def[mask]; ax, faint=true)
    plot_tracks(tr_opt, sv_opt; ax, xlims=xlims_full, ylims=ylims_full, colorbar=false)

    log_keys = log.(sv_opt)
    pos_max = max(maximum(log_keys), 0.0)
    neg_max = max(-minimum(log_keys), 0.0)
    kmin, kmax = minimum(sv_opt), maximum(sv_opt)
    cb_low  = neg_max > 0 ? -1.0 : 0.0
    cb_high = pos_max > 0 ?  1.0 : 0.0
    cb_ticks = filter(t -> cb_low <= t[1] <= cb_high, [
        (-1.0, string(kmin)), (0.0, "1.0"), (1.0, string(kmax))])
    pos_only_cmap = Makie.resample_cmap(:bluesreds, 256)[129:end]
    cb_cmap = neg_max > 0 ? :bluesreds : pos_only_cmap
    Colorbar(fig[1, 2]; colormap=cb_cmap, colorrange=(cb_low, cb_high),
        ticks=(first.(cb_ticks), last.(cb_ticks)),
        label="Scale", width=4, vertical=true)
    fig
end
plot_optimized_tracks(tracks_default, tracks_optimized)

#=
### Validation: Previously Unstable Scenario

We test the optimized parameters at Zscale=2.25, which was unstable with default
parameters. If the system is now stable, the optimization has extended the stability
boundary.
=#

let
    scale_def = 2.25
    scale_opt = 4.0
    ΔG_factor=1.3
    tspan=(-1.0, 1.1)

    s0_test = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=scale_def, verbose=false)[2]
    prob_default = make_load_step_problem(s0_test; ΔG_factor, tspan)

    s0_test_opt = load_ieee9bus_emt(; gfm=true, gfl=true, Zscale=scale_opt, verbose=false)[2]
    s0_tuned = reinitialize_with_params(s0_test_opt, tunable_p, optsol.u)
    prob_tuned = make_load_step_problem(s0_tuned; ΔG_factor, tspan)

    sol_def = solve(prob_default, Rodas5P())
    sol_tun = solve(prob_tuned, Rodas5P())

    fig = Figure(size=theme_size(800, 300))
    ax1 = Axis(fig[1, 1]; title="Zscale=$scale_def — default", xlabel="Time [s]", ylabel="V [pu]")
    ax2 = Axis(fig[1, 2]; title="Zscale=$scale_opt — optimized", xlabel="Time [s]", ylabel="V [pu]")

    for (ax, sol) in [(ax1, sol_def), (ax2, sol_tun)]
        ts = refine_timeseries(sol.t)
        for (j, bus) in enumerate([2, 3, 8])
            lines!(ax, ts, sol(ts, idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label="Bus $bus")
        end
        xlims!(ax, -0.01, sol.t[end])
    end
    axislegend(ax1; position=:lt)
    fig
end

#=
## Eigenvalue Evolution Animation

Replay the optimisation trajectory: each frame corresponds to an improving iterate
captured in `opt_states` (strictly monotone in loss), so the animation shows only
the successful descent.
=#

## Pre-compute tracks for every animation frame — expensive, done once.
anim_frames = let N = length(opt_states)
    map(enumerate(opt_states)) do (i, s)
        println("Computing tracks $i/$N (iter $(s.iter))...")
        tracks, sv = compute_eig_tracks(s.u)
        (; s, tracks, sv)
    end
end

let fig = Figure(size=theme_size(900, 500))
    ax_full = Axis(fig[1, 1]; xlabel="Real [Hz]", ylabel="Imag [Hz]")
    ax_zoom = Axis(fig[1, 2]; xlabel="Real [Hz]", ylabel="Imag [Hz]")
    N = length(anim_frames)
    record(fig, "eigenvalue_evolution.mp4", enumerate(anim_frames); framerate=10) do (i, f)
        println("Rendering frame $i/$N...")
        empty!(ax_full); empty!(ax_zoom)
        label = "iter $(f.s.iter), loss=$(round(f.s.objective; sigdigits=3))"
        ax_full.title[] = label
        ax_zoom.title[] = "$label (zoom)"
        plot_tracks(f.tracks, f.sv; ax=ax_full, xlims=xlims_full, ylims=ylims_full)
        plot_tracks(f.tracks, f.sv; ax=ax_zoom, xlims=xlims_zoom, ylims=ylims_zoom)
    end
end

#=
### Animation 2: Voltage Response + Eigenvalue Tracks (Zscale=2.0)

LHS shows the load-step voltage response at the most challenging training scenario
(Zscale=2.0); RHS shows the eigenvalue tracks — linking mode damping to the
time-domain oscillations as parameters evolve.
=#

let prob = opt_problems[3]  # Zscale=2.0
    fig = Figure(size=theme_size(900, 500))
    ax_v   = Axis(fig[1, 1]; xlabel="Time [s]", ylabel="Voltage [pu]")
    ax_eig = Axis(fig[1, 2]; xlabel="Real [Hz]", ylabel="Imag [Hz]")
    ts = range(0.0, 0.5; length=500)
    N = length(anim_frames)
    record(fig, "voltage_eigenvalue_evolution.mp4", enumerate(anim_frames); framerate=10) do (i, f)
        println("Rendering frame $i/$N...")
        empty!(ax_v); empty!(ax_eig)
        label = "iter $(f.s.iter), loss=$(round(f.s.objective; sigdigits=3))"
        ax_v.title[]   = "Zscale=2.0 — $label"
        ax_eig.title[] = label
        p_new = copy(prob.p)
        p_new[tpidx] .= f.s.u
        sol = solve(prob, Rodas5P(); p=p_new)
        for (j, bus) in enumerate([2, 3, 8])
            lines!(ax_v, ts, sol(ts; idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label="Bus $bus")
        end
        xlms = (-0.01, 0.5)
        ylms = (0.926, 1.045)
        xlims!(ax_v, xlms...)
        ylims!(ax_v, ylms...)
        i == 1 && axislegend(ax_v; position=:rb)
        plot_tracks(f.tracks, f.sv; ax=ax_eig, xlims=xlims_full, ylims=ylims_full)
    end
end
