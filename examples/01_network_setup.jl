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
using OrdinaryDiffEqNonlinearSolve
using CairoMakie

# lets load the emt and non emt model for comparison
nw, s0 = load_ieee9bus_emt()
nw_rms, s0_rms = load_ieee9bus()

####
#### Compare the powerflwo results to rms model
####
pf = show_powerflow(nw)
pf_rms = show_powerflow(nw_rms)
@test pf."vm [pu]" ≈ pf_rms."vm [pu]"
@test pf."varg [deg]" ≈ pf_rms."varg [deg]"

####
#### Compare dynamic respons to rms model
####
affected_line = (4, 6)
trange = (0.0, 15)
sol_rms = let
    affect_rms! = integrator -> begin
        println("Affecting line at time ", integrator.t)
        src, dst = affected_line
        NWState(integrator)[EIndex(src=>dst, :pibranch₊active)] = 0
        save_parameters!(integrator)
    end
    cb_rms = PresetTimeCallback(0.1, affect_rms!)
    prob_rms = ODEProblem(nw_rms, s0_rms, trange, add_nw_cb=cb_rms)
    solve(prob_rms, Rodas5P())
end

sol = let
    affect! = integrator -> begin
        println("Affecting line at time ", integrator.t)
        s = NWState(integrator)

        src, dst = affected_line
        em = nw[EIndex(src => dst)]
        B = em.metadata[:B]/2

        s[VIndex(src, :shunt₊C)] -= B # substract B since the line is "removed"
        s[VIndex(dst, :shunt₊C)] -= B # substract B since the line is "removed"
        # make sure the output is zero
        s[EIndex(src=>dst, :rlbranch₊r_src)] = 0
        s[EIndex(src=>dst, :rlbranch₊r_dst)] = 0

        save_parameters!(integrator)
    end
    cb = PresetTimeCallback(0.1, affect!)

    prob = ODEProblem(nw, s0, trange, add_nw_cb=cb)
    solve(prob, Rodas5P())
end

let
    fig = Figure(size=(800,800))
    # tmin, tmax = trange
    tmin, tmax = (0.09, 0.12)
    ts = range(tmin, tmax, length=1000)
    ax = Axis(fig[1,1], title="Voltage Magnitudes (Generators)")
    for i in [1,2,3]
        lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_mag)).u, label="Bus $i EMT", color=Cycled(i), alpha=0.6)
        lines!(ax, ts, sol_rms(ts, idxs=VIndex(i, :busbar₊u_mag)).u, label="Bus $i RMS", linestyle=:dash, color=Cycled(i))
    end
    ax = Axis(fig[2,1], title="Voltage Argument (Generators)")
    for i in [1,2,3]
        lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_arg)).u, label="Bus $i EMT", color=Cycled(i), alpha=0.6)
        lines!(ax, ts, sol_rms(ts, idxs=VIndex(i, :busbar₊u_arg)).u, label="Bus $i RMS", linestyle=:dash, color=Cycled(i))
    end
    ax = Axis(fig[3,1], title="Generator Frequency")
    for i in [1,2,3]
        lines!(ax, ts, sol(ts, idxs=VIndex(i, :generator₊machine₊ω)).u, label="Bus $i EMT", color=Cycled(i), alpha=0.6)
        lines!(ax, ts, sol_rms(ts, idxs=VIndex(i, :generator₊machine₊ω)).u, label="Bus $i RMS", linestyle=:dash, color=Cycled(i))
    end

    ax = Axis(fig[1,2], title="Voltage Magnitudes (Loads)")
    for i in [5,6,8]
        lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_mag)).u, label="Bus $i EMT", color=Cycled(i), alpha=0.6)
        lines!(ax, ts, sol_rms(ts, idxs=VIndex(i, :busbar₊u_mag)).u, label="Bus $i RMS", linestyle=:dash, color=Cycled(i))
    end
    ax = Axis(fig[2,2], title="Voltage Argument (Loads)")
    for i in [5,6,8]
        lines!(ax, ts, sol(ts, idxs=VIndex(i, :busbar₊u_arg)).u, label="Bus $i EMT", color=Cycled(i), alpha=0.6)
        lines!(ax, ts, sol_rms(ts, idxs=VIndex(i, :busbar₊u_arg)).u, label="Bus $i RMS", linestyle=:dash, color=Cycled(i))
    end
    fig
end
