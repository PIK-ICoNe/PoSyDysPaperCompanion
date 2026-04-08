module PoSyDysPaperCompanion

using ModelingToolkit
using PowerDynamics
using PowerDynamics.Library
using CSV: CSV
using DataFrames: DataFrame

export get_machine_bus, get_RL_line, get_load_bus, get_junction_bus, load_ieee9bus_emt, load_ieee9bus

include("models.jl")

function get_machine_bus(; machine_p=(;), avr_p=(;), gov_p=(;), pf=nothing, name, vidx)
    @named machine = SauerPaiMachine(;
        stator_dynamics=true,
        vf_input=true,
        τ_m_input=true,
        S_b=100,
        V_b=1,
        Sn=100,
        Vn=1,
        ω_b=2π*60,
        R_s=0.000125,
        T″_d0=0.01,
        T″_q0=0.01,
        machine_p... # unpack machine parameters
    )
    @named avr = AVRTypeI(vr_min=-5, vr_max=5,
        Ka=20, Ta=0.2,
        Kf=0.063, Tf=0.35,
        Ke=1, Te=0.314,
        E1=3.3, Se1=0.6602, E2=4.5, Se2=4.2662,
        tmeas_lag=false,
        avr_p... # unpack AVR parameters
    )
    @named gov = TGOV1(R=0.05, T1=0.05, T2=2.1, T3=7.0, DT=0, V_max=5, V_min=-5,
        gov_p... # unpack governor parameters
    )
    # generate the "injector" as combination of multiple components
    injector = CompositeInjector([machine, avr, gov]; name=:generator)

    # add a dynamic shunt to mimic the terminal capacitance of the machine
    @named shunt = DynamicCShunt(ω0=2π*60, C=1e-5)

    # generate the MTKBus (i.e. the MTK model containg the busbar and the injector)
    dynbus = compile_bus(MTKBus(injector, shunt); current_source=false, vidx)
    if !isnothing(pf)
        set_pfmodel!(dynbus, pf)
    end
    dynbus
end

function get_gfm_bus(; name, vidx, pf=nothing)
    @named gfl = ComposableInverter.DroopInverter(
        filter_type=:LCL,
        vsrc₊ω0 = 2π*60,
        droop₊ω0 = 2π*60,
    )
    @named shunt = DynamicCShunt(ω0=2π*60, C=1e-5)

    dynbus = compile_bus(MTKBus(gfl, shunt); name=name, vidx=vidx)

    initf = @initformula begin
        :gfl₊droop₊Vset = sqrt(:busbar₊u_r^2 + :busbar₊u_i^2)
    end
    add_initformula!(dynbus, initf)

    if !isnothing(pf)
        set_pfmodel!(dynbus, pf)
    end
    dynbus
end

function get_gfl_bus(; name, vidx, pf=nothing)
    @named gfl = ConstantPowerInverter(; csrc₊ω0=2π*60)
    # add a dynamic shunt to mimic the terminal capacitance of the machine
    @named shunt = DynamicCShunt(ω0=2π*60, C=1e-5)

    dynbus = compile_bus(MTKBus(gfl, shunt); name=name, vidx=vidx)
    if !isnothing(pf)
        set_pfmodel!(dynbus, pf)
    end
    dynbus
end

function get_RL_line(; R, X, src, dst, name)
    @named rlbranch = DynamicSeriesRLBranch(; R, L=X, ω0=2π*60)
    dyn = compile_line(MTKLine(rlbranch), src=src, dst=dst, name=name)
    @named rlbranch_static = PiLine(; R, X, B_src=0, B_dst=0, G_src=0, G_dst=0)
    static = compile_line(MTKLine(rlbranch_static), src=src, dst=dst, name=Symbol(name, "_static"))
    set_pfmodel!(dyn, static)
    dyn
end

function get_load_bus(; P, Q, B, name, vidx)
    @named load = ConstantYLoad()  # G, B free → set by init
    @named shunt = DynamicCShunt(; ω0=2π*60, C=B)

    dynmod = compile_bus(MTKBus(load, shunt); name, vidx)

    # PF: PQ constraint + static shunt for aggregated B
    static = MTKBus(
        Library.PQConstraint(; P, Q, name=:PQConstraint),
        StaticShunt(; B=B, G=0, name=:shunt)
    )
    pfmod = compile_bus(static)
    set_pfmodel!(dynmod, pfmod)
    dynmod
end
function get_junction_bus(; B, name, vidx)
    @named shunt = DynamicCShunt(ω0=2π*60)
    dyn = compile_bus(MTKBus(shunt); name=name, vidx=vidx)
    set_guess!(dyn, :shunt₊C, 1e-5)
    pfmod = compile_bus(MTKBus(StaticShunt(B=B, G=0; name=:shunt)))
    set_pfmodel!(dyn, pfmod)
    dyn
end


function load_ieee9bus_emt(; gfm = false, gfl = false)
    # line parameters
    linedat = """
    src | dst | R      | X      | B
      4 |   5 | 0.0100 | 0.0850 | 0.1760
      4 |   6 | 0.0170 | 0.0920 | 0.1580
      5 |   7 | 0.0320 | 0.1610 | 0.3060
      6 |   9 | 0.0390 | 0.1700 | 0.3580
      7 |   8 | 0.0085 | 0.0720 | 0.1490
      8 |   9 | 0.0119 | 0.1008 | 0.2090
      1 |   4 |      0 | 0.0576 |      0
      2 |   7 |      0 | 0.0625 |      0
      3 |   9 |      0 | 0.0586 |      0
    """

    linep = CSV.read(IOBuffer(linedat), stripwhitespace=true, delim='|', DataFrame)

    # since we do EMT modeling, we aggregate the B shunts at the vertices
    Bbus = map(1:9) do i
        (sum(linep.B[linep.src .== i]) + sum(linep.B[linep.dst .== i])) / 2
    end
    @assert all(iszero, Bbus[1:3]) && !any(iszero, Bbus[4:9]) # no shunts at genenerator buses but everywhere else

    # generate pure RL lines
    linemodels = map(eachrow(linep)) do row
        m = if iszero(row.R) # transformer
            get_RL_line(; R=row.R, X=row.X, src=row.src, dst=row.dst, name=Symbol("t$(row.src)_$(row.dst)"))
        else # pi line
            get_RL_line(; R=row.R, X=row.X, src=row.src, dst=row.dst, name=Symbol("l$(row.src)_$(row.dst)"))
        end
        m.metadata[:B] = row.B # store the B value as metadata
        m
    end

    gen1p = (;X_ls=0.01460, X_d=0.1460, X′_d=0.0608, X″_d=0.06, X_q=0.1000, X′_q=0.0969, X″_q=0.06, T′_d0=8.96, T′_q0=0.310, H=23.64)
    @named bus1 = get_machine_bus(; machine_p=gen1p, pf=pfSlack(V=1.04), vidx=1)

    if !gfm
        gen2p = (;X_ls=0.08958, X_d=0.8958, X′_d=0.1198, X″_d=0.11, X_q=0.8645, X′_q=0.1969, X″_q=0.11, T′_d0=6.00, T′_q0=0.535, H= 6.40)
        @named bus2 = get_machine_bus(; machine_p=gen2p, pf=pfPV(V=1.025, P=1.63), vidx=2)
    else
        @named bus2 = get_gfm_bus(; name=:bus2, vidx=2, pf=pfPV(V=1.025, P=1.63))
    end

    if !gfl
        gen3p = (;X_ls=0.13125, X_d=1.3125, X′_d=0.1813, X″_d=0.18, X_q=1.2578, X′_q=0.2500, X″_q=0.18, T′_d0=5.89, T′_q0=0.600, H= 3.01)
        @named bus3 = get_machine_bus(; machine_p=gen3p, pf=pfPV(V=1.025, P=0.85), vidx=3)
    else
        @named bus3 = get_gfl_bus(; name=:bus3, vidx=3, pf=pfPV(V=1.025, P=0.85))
    end

    @named bus4 = get_junction_bus(; B=Bbus[4], vidx=4)
    @named bus5 = get_load_bus(; B=Bbus[5], P=-1.25, Q=-0.5, vidx=5)
    @named bus6 = get_load_bus(; B=Bbus[6], P=-0.9, Q=-0.3, vidx=6)
    @named bus7 = get_junction_bus(; B=Bbus[7], vidx=7)
    @named bus8 = get_load_bus(; B=Bbus[8], P=-1.0, Q=-0.35, vidx=8)
    @named bus9 = get_junction_bus(; B=Bbus[9], vidx=9)

    vertexfs = [bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8, bus9]

    nw = Network(vertexfs, linemodels; warn_order=false)
    s0 = initialize_from_pf!(nw; tol=1e-7, nwtol=1e-6, subverbose=false)
    nw, s0
end

"""
    load_ieee9bus()

Load the IEEE 9-bus test system.

Returns an uninitialized Network object with:
- 3 generator buses (SauerPai machines with AVR and governors)
- 3 load buses (ConstantYLoad)
- 3 transmission buses
- 9 branches (6 lines + 3 transformers)

This network is ready for powerflow solving and initialization testing.
"""
function load_ieee9bus()
    # Generator Bus Model
    function GeneratorBus(; machine_p=(;), avr_p=(;), gov_p=(;))
        @named machine = SauerPaiMachine(;
            vf_input=true,
            τ_m_input=true,
            S_b=100,
            V_b=1,
            Sn=100,
            Vn=1,
            ω_b=2π*60,
            R_s=0.000125,
            T″_d0=0.01,
            T″_q0=0.01,
            machine_p... # unpack machine parameters
        )
        @named avr = AVRTypeI(vr_min=-5, vr_max=5,
            Ka=20, Ta=0.2,
            Kf=0.063, Tf=0.35,
            Ke=1, Te=0.314,
            E1=3.3, Se1=0.6602, E2=4.5, Se2=4.2662,
            tmeas_lag=false,
            avr_p... # unpack AVR parameters
        )
        @named gov = TGOV1(R=0.05, T1=0.05, T2=2.1, T3=7.0, DT=0, V_max=5, V_min=-5,
            gov_p... # unpack governor parameters
        )
        # generate the "injector" as combination of multiple components
        injector = CompositeInjector([machine, avr, gov]; name=:generator)

        # generate the MTKBus (i.e. the MTK model containg the busbar and the injector)
        mtkbus = MTKBus(injector)
    end

    # Load Bus Model
    function ConstantYLoadBus()
        @named load = ConstantYLoad()
        MTKBus(load; name=:loadbus)
    end

    # Generator parameters from RTDS datasheet
    gen1p = (;X_ls=0.01460, X_d=0.1460, X′_d=0.0608, X″_d=0.06, X_q=0.1000, X′_q=0.0969, X″_q=0.06, T′_d0=8.96, T′_q0=0.310, H=23.64)
    gen2p = (;X_ls=0.08958, X_d=0.8958, X′_d=0.1198, X″_d=0.11, X_q=0.8645, X′_q=0.1969, X″_q=0.11, T′_d0=6.00, T′_q0=0.535, H= 6.40)
    gen3p = (;X_ls=0.13125, X_d=1.3125, X′_d=0.1813, X″_d=0.18, X_q=1.2578, X′_q=0.2500, X″_q=0.18, T′_d0=5.89, T′_q0=0.600, H= 3.01)

    # Instantiate MTK models
    mtkbus1 = GeneratorBus(; machine_p=gen1p)
    mtkbus2 = GeneratorBus(; machine_p=gen2p)
    mtkbus3 = GeneratorBus(; machine_p=gen3p)
    mtkbus4 = MTKBus()
    mtkbus5 = ConstantYLoadBus()
    mtkbus6 = ConstantYLoadBus()
    mtkbus7 = MTKBus()
    mtkbus8 = ConstantYLoadBus()
    mtkbus9 = MTKBus()

    # Build NetworkDynamics components with powerflow models
    @named bus1 = compile_bus(mtkbus1; vidx=1, pf=pfSlack(V=1.04))
    @named bus2 = compile_bus(mtkbus2; vidx=2, pf=pfPV(V=1.025, P=1.63))
    @named bus3 = compile_bus(mtkbus3; vidx=3, pf=pfPV(V=1.025, P=0.85))
    @named bus4 = compile_bus(mtkbus4; vidx=4)
    @named bus5 = compile_bus(mtkbus5; vidx=5, pf=pfPQ(P=-1.25, Q=-0.5))
    @named bus6 = compile_bus(mtkbus6; vidx=6, pf=pfPQ(P=-0.9, Q=-0.3))
    @named bus7 = compile_bus(mtkbus7; vidx=7)
    @named bus8 = compile_bus(mtkbus8; vidx=8, pf=pfPQ(P=-1.0, Q=-0.35))
    @named bus9 = compile_bus(mtkbus9; vidx=9)

    # Branch helper functions
    function piline(; R, X, B)
        @named pibranch = PiLine(;R, X, B_src=B/2, B_dst=B/2, G_src=0, G_dst=0)
        MTKLine(pibranch)
    end
    function transformer(; R, X)
        @named xfmr = PiLine(;R, X, B_src=0, B_dst=0, G_src=0, G_dst=0)
        MTKLine(xfmr)
    end

    # Define branches
    @named l45 = compile_line(piline(; R=0.0100, X=0.0850, B=0.1760), src=4, dst=5)
    @named l46 = compile_line(piline(; R=0.0170, X=0.0920, B=0.1580), src=4, dst=6)
    @named l57 = compile_line(piline(; R=0.0320, X=0.1610, B=0.3060), src=5, dst=7)
    @named l69 = compile_line(piline(; R=0.0390, X=0.1700, B=0.3580), src=6, dst=9)
    @named l78 = compile_line(piline(; R=0.0085, X=0.0720, B=0.1490), src=7, dst=8)
    @named l89 = compile_line(piline(; R=0.0119, X=0.1008, B=0.2090), src=8, dst=9)
    @named t14 = compile_line(transformer(; R=0, X=0.0576), src=1, dst=4)
    @named t27 = compile_line(transformer(; R=0, X=0.0625), src=2, dst=7)
    @named t39 = compile_line(transformer(; R=0, X=0.0586), src=3, dst=9)

    # Build the network
    vertexfs = [bus1, bus2, bus3, bus4, bus5, bus6, bus7, bus8, bus9]
    edgefs = [l45, l46, l57, l69, l78, l89, t14, t27, t39]

    # Return uninitialized network
    nw = Network(vertexfs, edgefs; warn_order=false)
    s0 = initialize_from_pf!(nw; subverbose=false)
    nw, s0
end

end # module PoSyDysPaperCompanion
