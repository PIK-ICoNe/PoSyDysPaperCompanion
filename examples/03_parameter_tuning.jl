using PowerDynamics
using PowerDynamics.Library
using ModelingToolkit
using PoSyDysPaperCompanion
using DelimitedFiles
using DataFrames, CSV
using Test
using Graphs
using DiffEqCallbacks
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqNonlinearSolve
using CairoMakie
using OrderedCollections: OrderedDict
using SymbolicIndexingInterface: SymbolicIndexingInterface as SII
using Optimization, Optimisers, OptimizationOptimisers

####
#### Setup
####

scales = [1.0, 1.5, 2.0]

s0s = map(scales) do s
    load_ieee9bus_emt(gfl=true, gfm=true, Zscale=s, verbose=false)[2]
end

function load_increase_affect!(integrator)
    s = NWState(integrator)
    s[VIndex(8, :load₊G)] *= 1.1 # load increase by 20%
    save_parameters!(integrator)
end

load_increase = PresetTimeCallback(0.0, load_increase_affect!)

problems = map(s0s) do s0
    nw = extract_nw(s0)
    ODEProblem(nw, s0, (-10.0, 1.0), add_nw_cb=load_increase)
end

sols = map(problems) do prob
    solve(prob, Rodas5P())
end;


function plot_vmag(sols::Vector)
    fig = Figure(size=(600, length(sols)*300))
    row = 1
    for sol in sols
        ax = Axis(fig[row, 1], xlabel="Time [s]", ylabel="Voltage Magnitude [pu]", title="Voltage Magnitude at Bus 8")
        ts = refine_timeseries(sol.t)
        for i in [2,3,8]
            lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_mag)).u)
        end
        xlims!(ax, -0.01, 0.5)
        ylims!(ax, 0.945, 1.035)
        row += 1
    end
    fig
end

plot_vmag(sols)
break

tunable_p = collect(VIndex(3,[
    :gfl₊csrc₊CC1_KI,
    :gfl₊csrc₊CC1_KP,
    :gfl₊csrc₊PLL_Kp,
    :gfl₊csrc₊PLL_Ki,
    :gfl₊csrc₊CC2_F,
    :gfl₊csrc₊CC2_KP,
]))
tpidx = SII.parameter_index(extract_nw(first(s0s)), tunable_p)

function generate_loss(problems, tpidx)
    t_eval = range(0.0, 1.0, length=1000)
    nw = extract_nw(first(problems))
    pdim = NetworkDynamics.pdim(nw)

    loss = p -> begin
        total = 0.0

        for (k, prob) in enumerate(problems)
            p_new = zeros(eltype(p), pdim)
            p_new .= prob.p
            p_new[tpidx] .= p

            sol = solve(prob, Rodas5P(autodiff=true);
                p = p_new,
                saveat = t_eval)

            if !SciMLBase.successful_retcode(sol)
                return Inf
            end

            # Voltage at buses 2, 3, 8 (GFM, GFL, load between them)
            all_V_t = reduce(hcat, sol(t_eval; idxs=VIndex([2,3,8], :busbar₊u_mag)).u)
            for bus in [1,2,3]
                V_t = all_V_t[bus, :]
                V_final = V_t[end]  # approximate new steady state
                total += sum((V_t .- V_final).^2) / length(t_eval)
            end
        end
        total
    end
    loss, first(problems).p[tpidx]
end

loss, p0opt = generate_loss(problems, tpidx);

optsol = let
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoForwardDiff())
    optimization_states = Any[]
    callback = function (state, l)
        push!(optimization_states, state)
        is_new_min = all(s -> l < s.objective, optimization_states[1:end-1])
        print("Iteration $(state.iter): loss = $l")
        is_new_min && printstyled(" ✓"; color=:green)
        println()
        return false
    end

    optprob = Optimization.OptimizationProblem(optf, p0opt; callback)
    @time Optimization.solve(optprob, Optimisers.Adam(5e-2); maxiters=10)
end

solopt = map(problems) do prob
    p_new = copy(prob.p)
    p_new[tpidx] .= optsol.u

    solve(prob, Rodas5P(); p = p_new)
end;

plot_vmag(sols)
plot_vmag(solopt)

for (sym, orig, tuend) in zip(tunable_p, p0opt, optsol.u)
    println("$(sym)  \t$(orig) \t=> $(tuend) \t(relative change = $(100*(tuend - orig)/orig)%)")
end

####
#### Eigenvalue comparison: baseline vs tuned
####

baseline_ev_data = OrderedDict(scale => s0 for (scale, s0) in zip(scales, s0s))

tuned_s0s = map(s0s) do s0
    _nw = extract_nw(s0)
    nw = Network(_nw)
    for (pidx, p) in zip(tunable_p, optsol.u)
        comp = pidx.compidx
        sym = pidx.subidx
        set_default!(nw[VIndex(comp)], sym, p)
    end
    initialize_from_pf!(nw, tol=1e-7, nwtol=1e-5, verbose=false)
end;
tuned_ev_data = OrderedDict(scale => s0 for (scale, s0) in zip(scales, tuned_s0s))

tracks_baseline = find_tracks(baseline_ev_data)
tracks_tuned    = find_tracks(tuned_ev_data)

plot_tracks(tracks_baseline, scales; title="Eigenvalues — baseline", xlims=(-20,5), ylims=(-40,40))
plot_tracks(tracks_tuned,    scales; title="Eigenvalues — tuned")
