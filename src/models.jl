using ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: t_nounits as t, D_nounits as Dt

@component function ConstantPowerOuter(; name, Pset=nothing, Qset=nothing, defaults...)
    @parameters begin
        Pset=Pset, [guess=0, description="Active power setpoint [pu]"]
        Qset=Qset, [guess=0, description="Reactive power setpoint [pu]"]
    end
    systems = @named begin
        id_out = RealOutput()
        iq_out = RealOutput()
        Vd_in  = RealInput()
        Vq_in  = RealInput()
    end
    # Constant-power law (d-axis aligned, matching ComposableInverter._ri_to_dq):
    #   P = V_d·i_d + V_q·i_q,  Q = V_q·i_d - V_d·i_q
    #   Solving for i_d, i_q:
    #   i_d = ( P·V_d + Q·V_q) / |V|²
    #   i_q = ( P·V_q - Q·V_d) / |V|²
    # At PLL lock: V_d = |V|, V_q = 0  →  i_d = P/|V|  (negative P → absorbing ✓)
    eqs = [
        id_out.u ~ ( Pset*Vd_in.u + Qset*Vq_in.u) / (Vd_in.u^2 + Vq_in.u^2 + 1e-6)
        iq_out.u ~ ( Pset*Vq_in.u - Qset*Vd_in.u) / (Vd_in.u^2 + Vq_in.u^2 + 1e-6)


    ]
    sys = System(eqs, t; name, systems)
    set_mtk_defaults!(sys, defaults)
    return sys
end
@component function ConstantPowerInverter(; name, defaults...)
    @named cp_outer = ConstantPowerOuter()
    @named csrc     = ComposableInverter.CurrentSource(; iset_input=true)
    @named terminal = Terminal()

    eqs = [
        connect(csrc.terminal, terminal)
        connect(csrc.iset_d_in, cp_outer.id_out)
        connect(csrc.iset_q_in, cp_outer.iq_out)
        # Rotate terminal voltage into PLL dq-frame for the outer controller
        cp_outer.Vd_in.u ~  cos(csrc.pll.δ_pll)*terminal.u_r + sin(csrc.pll.δ_pll)*terminal.u_i
        cp_outer.Vq_in.u ~ -sin(csrc.pll.δ_pll)*terminal.u_r + cos(csrc.pll.δ_pll)*terminal.u_i
    ]
    sys = System(eqs, t; name, systems=[cp_outer, csrc, terminal])
    set_mtk_defaults!(sys, defaults)
    return sys
end
