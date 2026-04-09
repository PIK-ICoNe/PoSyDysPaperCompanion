#=
# Differentiable Simulation for Power System Dynamics — Companion Script

This script accompanies the PoSyDyS 2026 paper and demonstrates three analysis
workflows enabled by the differentiable simulation framework PowerDynamics.jl:
impedance extraction, eigenvalue sensitivity, and gradient-based controller optimization.

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

let fig = Figure(size=(800, 600))
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

    fig = Figure(size=(600, length(scales) * 250))
    for (row, (s, sol)) in enumerate(zip(scales, sols))
        ax = Axis(fig[row, 1]; xlabel="Time [s]", ylabel="Voltage [pu]",
            title="Zscale = $s")
        ts = refine_timeseries(sol.t)
        for i in [1, 2, 3]
            lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_mag)).u;
                label="Bus $i", color=Cycled(i))
        end
        xlims!(ax, -0.01, 0.5)
        row == 1 && axislegend(ax; position=:rb)
    end
    fig, s0s, sols
end

fig_weak, _, _ = plot_load_step_comparison([1.0, 1.5, 2.0])
fig_weak

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
        highlight_modes=Int[], xlims=nothing, ylims=nothing, title="Eigenvalues")
    log_keys = log.(key_vals)
    max_log = maximum(abs.(log_keys))
    norm_colors = iszero(max_log) ? zeros(length(key_vals)) : log_keys ./ max_log
    baseline_idx = argmin(abs.(key_vals .- 1.0))

    own_fig = isnothing(ax)
    if own_fig
        fig = Figure(size=(600, 500))
        ax = Axis(fig[1, 1]; xlabel="Real Part [Hz]", ylabel="Imaginary Part [Hz]", title)
    end

    for m in highlight_modes
        lines!(ax, real.(tracks[m, :]), imag.(tracks[m, :]);
            color=:yellow,
            alpha=0.3,
            colorrange=(-1.0, 1.0),
            colormap=:bluesreds,
            linewidth=8,
            joinstyle=:round,
            linecap=:round,
        )
    end
    for i in axes(tracks, 1)
        lines!(ax, real.(tracks[i, :]), imag.(tracks[i, :]);
            color=norm_colors,
            colorrange=(-1.0, 1.0),
            colormap=:bluesreds,
            linewidth=3,
            joinstyle=:round,
            linecap=:round,
        )
    end
    scatter!(ax, real.(tracks[:, baseline_idx]), imag.(tracks[:, baseline_idx]);
        color=:black, markersize=6, marker=:xcross)

    !isnothing(xlims) && Makie.xlims!(ax, xlims...)
    !isnothing(ylims) && Makie.ylims!(ax, ylims...)
    own_fig ? fig : ax
end

plot_tracks(tracks, scales_vec;
    highlight_modes=[54, 55],
    xlims=(-40, 5),
    ylims=(-120, 120),
    title="Eigenvalue Trajectories vs Grid Strength"
)

#=
### Participation Factors

At nominal grid strength, the critical mode (54/55) is dominated by the inner current
controller integrator states of both inverters — a coupled filter resonance. Near
instability, the GFL's PLL angle enters the participation, revealing the mechanism
that drives destabilization.
=#

cmode = 54
s0_nom = eigenvalue_data[1.0]

@info "Participation factors at nominal grid strength (Zscale=1.0):"
show_participation_factors(s0_nom; modes=[cmode], threshold=0.05)

## Find the last stable Zscale for this mode
idx_last_stable = findlast(λ -> real(λ) < 0, tracks[cmode, :])
scale_critical = scales_vec[idx_last_stable]
s0_crit = eigenvalue_data[scale_critical]
critical_mode = tracks[cmode, idx_last_stable]
idx_crit = findmin(λ -> abs(λ - critical_mode), jacobian_eigenvals(s0_crit) ./ (2π))[2]

@info "Participation factors near instability (Zscale=$scale_critical):"
show_participation_factors(s0_crit; modes=idx_crit, threshold=0.05)

#=
### Eigenvalue Sensitivity

We compute the sensitivity of the critical mode to all controller parameters of the
GFM (bus 2) and GFL (bus 3) inverters. The ranking shifts significantly between
nominal and weak-grid conditions.
=#

params_of_interest = let
    candidates = vidxs(s0_nom, 2:3, s=false, p=true, in=false, out=false, obs=false)
    # we are only interested in control parameters, so we filter for "filter" states
    filter!(params_of_interest) do idx
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

The sharp resonance peak at ~28 Hz (visible at critical Zscale) corresponds exactly
to the eigenvalue mode that destabilizes. At nominal grid strength, the same mode
is well-damped and barely visible.
=#

function plot_Zdd_bode(s0_a, s0_b; buses=[1,2,3],
        labels=["SG Bus", "GFM Bus", "GFL Bus"],
        label_suffix=["nominal", "critical"])
    fs = 10 .^ range(-2, 3; length=800)
    jωs = 2π .* fs .* im

    fig = Figure(size=(700, 500))
    ax_g = Axis(fig[1, 1]; ylabel="Gain (dB)", xscale=log10, title="Z_dd Bode Plot")
    ax_p = Axis(fig[2, 1]; ylabel="Phase (deg)", xlabel="Frequency (Hz)", xscale=log10)

    for (i, (bus, lab)) in enumerate(zip(buses, labels))
        Za = compute_Z_aligned(s0_a, bus)
        Zb = compute_Z_aligned(s0_b, bus)

        lines!(ax_g, fs, [20log10(abs(Za.Zdd(s))) for s in jωs];
            label="$lab ($(label_suffix[1]))", color=Cycled(i), linewidth=2)
        lines!(ax_g, fs, [20log10(abs(Zb.Zdd(s))) for s in jωs];
            label="$lab ($(label_suffix[2]))", color=Cycled(i), linewidth=2, linestyle=:dash)

        lines!(ax_p, fs, rad2deg.(unwrap_rad([angle(Za.Zdd(s)) for s in jωs]));
            color=Cycled(i), linewidth=2)
        lines!(ax_p, fs, rad2deg.(unwrap_rad([angle(Zb.Zdd(s)) for s in jωs]));
            color=Cycled(i), linewidth=2, linestyle=:dash)
    end
    axislegend(ax_g; position=:rb)
    fig
end

plot_Zdd_bode(s0_nom, s0_crit)

#=
### Cross-Validation: Is This Truly an Inverter-Inverter Interaction?

We check whether the resonance appears with only one inverter present.
It does not — confirming a coupled interaction between the GFM and GFL
mediated by the network.
=#

let
    s0_gfl_only = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=1.0, verbose=false)[2]
    s0_gfl_crit = load_ieee9bus_emt(; gfm=false, gfl=true, Zscale=scale_critical, verbose=false)[2]
    plot_Zdd_bode(s0_gfl_only, s0_gfl_crit;
        labels=["SG Bus 1", "SG Bus 2", "GFL Bus"],
        label_suffix=["nominal (GFL only)", "critical (GFL only)"])
end

let
    s0_gfm_only = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=1.0, verbose=false)[2]
    s0_gfm_crit = load_ieee9bus_emt(; gfm=true, gfl=false, Zscale=scale_critical, verbose=false)[2]
    plot_Zdd_bode(s0_gfm_only, s0_gfm_crit;
        labels=["SG Bus 1", "GFM Bus", "SG Bus 3"],
        label_suffix=["nominal (GFM only)", "critical (GFM only)"])
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

function plot_before_after(sols_before, sols_after, scales; buses=[2, 3, 8])
    fig = Figure(size=(800, length(scales) * 250))
    for (row, (s, sb, sa)) in enumerate(zip(scales, sols_before, sols_after))
        ax1 = Axis(fig[row, 1]; title="Zscale=$s — default",
            xlabel="Time [s]", ylabel="V [pu]")
        ax2 = Axis(fig[row, 2]; title="Zscale=$s — optimized",
            xlabel="Time [s]", ylabel="V [pu]")
        ts_b = refine_timeseries(sb.t)
        ts_a = refine_timeseries(sa.t)
        for (j, bus) in enumerate(buses)
            lines!(ax1, ts_b, sb(ts_b, idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label="Bus $bus")
            lines!(ax2, ts_a, sa(ts_a, idxs=VIndex(bus, :busbar₊u_mag)).u;
                color=Cycled(j), label="Bus $bus")
        end
        xlms = (-0.01, 0.5)
        ylms = (0.926, 1.045)
        xlims!(ax1, xlms...); xlims!(ax2, xlms...)
        ylims!(ax1, ylms...); ylims!(ax2, ylms...)
        row == 1 && axislegend(ax1; position=:rb)
    end
    fig
end

plot_before_after(sols_default, sols_optimized, opt_scales)

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
eig_sweep_scales = range(1.0, 4.0; length=10)
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

xlims_full = (-40, 5)
xlims_zoom = (-5, 5/8)
ylims_full = (-110, 110)
ylims_zoom = (-5, 5)

let fig = Figure(size=(900, 650))
    tr_def, sv_def = compute_eig_tracks(p0_opt)
    tr_opt, sv_opt = compute_eig_tracks(optsol.u)

    ax1 = Axis(fig[1, 1]; xlabel="Real [Hz]", ylabel="Imag [Hz]", title="Default Parameters")
    ax2 = Axis(fig[1, 2]; xlabel="Real [Hz]", ylabel="Imag [Hz]", title="Optimized Parameters")
    ax3 = Axis(fig[2, 1]; xlabel="Real [Hz]", ylabel="Imag [Hz]")
    ax4 = Axis(fig[2, 2]; xlabel="Real [Hz]", ylabel="Imag [Hz]")

    plot_tracks(tr_def, sv_def; ax=ax1, xlims=xlims_full, ylims=ylims_full)
    plot_tracks(tr_opt, sv_opt; ax=ax2, xlims=xlims_full, ylims=ylims_full)
    plot_tracks(tr_def, sv_def; ax=ax3, xlims=xlims_zoom, ylims=ylims_zoom)
    plot_tracks(tr_opt, sv_opt; ax=ax4, xlims=xlims_zoom, ylims=ylims_zoom)
    fig
end

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

    fig = Figure(size=(800, 300))
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

let fig = Figure(size=(900, 500))
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
    fig = Figure(size=(900, 500))
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
