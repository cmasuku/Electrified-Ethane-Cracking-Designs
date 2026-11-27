"""
Gas-fired and electric ethane steam cracking furnace: design-basis model.
Material and energy balances, radiant coil sizing, tube-metal temperature screening,
fired-furnace heat recovery, electric-furnace loads and the comparison metrics.

Section numbers in the comments and in the results file are internal labels.
Thermo: ideal-gas enthalpies from Cantera (GRI-Mech 3.0); water/steam from CoolProp (IAPWS-95).
Transport properties for the film coefficients: Cantera mixture-averaged model.

Usage:  python furnace_model.py [output.json]
Needs:  cantera, CoolProp, numpy, scipy
"""
import sys, os, json
import numpy as np
import cantera as ct
import CoolProp.CoolProp as CP
from scipy.optimize import brentq

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'furnace_model_results.json')

g = ct.Solution('gri30.yaml')
g.transport_model = 'mixture-averaged'
P = 2.5e5            # Pa, process pressure in the energy balance
T0 = 298.15
R_GAS = 8314.46      # J/kmol K
SIG = 5.670e-8
MU0 = 4e-7*np.pi
MW = {s: g.molecular_weights[g.species_index(s)] for s in ['C2H6', 'C2H4', 'H2', 'H2O', 'CH4', 'CO2', 'N2', 'O2']}


def H(comp, T, p=P):
    """Total enthalpy flow (W) of an ideal-gas stream, comp in kmol/h, incl. formation enthalpy."""
    n = sum(comp.values()); g.TPX = T, p, comp
    return n*g.enthalpy_mole/3600


def C(x): return x - 273.15
def K(x): return x + 273.15


R = {}

# ---------------------------------------------------------------- 1-2. Basis and material balance
hours = 8000
mC2H4 = 50000e3/hours
nC2H4 = mC2H4/MW['C2H4']
X = 0.65
nE_tot = nC2H4/X; nE_rec = nE_tot - nC2H4
mE_tot = nE_tot*MW['C2H6']
mW = 0.5*mE_tot; nW = mW/MW['H2O']
feed = {'C2H6': nE_tot, 'H2O': nW}
out = {'C2H4': nC2H4, 'H2': nC2H4, 'C2H6': nE_rec, 'H2O': nW}
m_out = sum(out[s]*MW[s] for s in out)
n_out = sum(out.values())
R['1_2_basis_and_balance'] = dict(
    hours=hours, mC2H4_kgh=mC2H4, nC2H4_kmolh=nC2H4,
    nE_fresh=nC2H4, mE_fresh=nC2H4*MW['C2H6'], nE_rec=nE_rec, mE_rec=nE_rec*MW['C2H6'],
    nE_tot=nE_tot, mE_tot=mE_tot, mW_kgh=mW, nW_kmolh=nW,
    mW_if_fresh_basis_kgh=0.5*nC2H4*MW['C2H6'],
    mC2H4_if_8760h=50000e3/8760,
    n_in=nE_tot+nW, n_out=n_out, m_out=m_out, mH2_kgh=nC2H4*MW['H2'], H2_t_y=nC2H4*MW['H2']*hours/1e3,
    outlet_molfrac={s: v/n_out for s, v in out.items()},
    mole_increase_pct=100*(n_out/(nE_tot+nW)-1))

# ---------------------------------------------------------------- 3. Energy balance
def dHr(T):
    def hm(s): g.TPX = T, P, {s: 1}; return g.enthalpy_mole
    return (hm('C2H4')+hm('H2')-hm('C2H6'))/1e6   # kJ/mol
R['3_dHr_kJmol'] = {f'{C(T):.0f}C': dHr(T) for T in [298.15, 923.15, 1023.15, 1123.15]}

Ps = 3.5e5
Tsat = CP.PropsSI('T', 'P', Ps, 'Q', 0, 'Water')
hL25 = CP.PropsSI('H', 'T', T0, 'P', Ps, 'Water')
hLs = CP.PropsSI('H', 'P', Ps, 'Q', 0, 'Water'); hVs = CP.PropsSI('H', 'P', Ps, 'Q', 1, 'Water')
mWs = mW/3600
Q_econ = mWs*(hLs-hL25)/1e3; Q_boil = mWs*(hVs-hLs)/1e3
XOT = K(650); COT = K(850)
T_eth_pre = K(150); T_st_sh = K(200)
Q_eth = (H({'C2H6': nE_tot}, T_eth_pre)-H({'C2H6': nE_tot}, T0))/1e3
Q_sh = mWs*(CP.PropsSI('H', 'T', T_st_sh, 'P', Ps, 'Water')-hVs)/1e3
Hmix_in = H({'C2H6': nE_tot}, T_eth_pre)+H({'H2O': nW}, T_st_sh)
Tmix = brentq(lambda T: H(feed, T)-Hmix_in, 300, 700)
Q_mix = (H(feed, XOT)-H(feed, Tmix))/1e3
Q_rad = (H(out, COT)-H(feed, XOT))/1e3
Q_rxn = nC2H4*dHr(COT)*1e3/3600
Q_conv = Q_econ+Q_boil+Q_sh+Q_eth+Q_mix
Q_tot = Q_conv+Q_rad
# Dew point of the mixed feed
yW = nW/(nW+nE_tot); pw = yW*P
R['3_energy_balance'] = dict(
    Psteam_bara=Ps/1e5, Tsat_C=C(Tsat), latent_kJkg=(hVs-hLs)/1e3,
    Q_econ_kW=Q_econ, Q_boil_kW=Q_boil, Q_steam_SH_kW=Q_sh, Q_eth_preheat_kW=Q_eth,
    Tmix_C=C(Tmix), Q_mixed_preheat_kW=Q_mix,
    Q_feed_prep_kW=Q_conv, Q_rad_kW=Q_rad, Q_rad_sensible_kW=Q_rad-Q_rxn, Q_rxn_at_COT_kW=Q_rxn,
    Q_total_kW=Q_tot, GJ_per_t_C2H4=Q_tot*hours*3.6/50000/1e3,
    effluent_cooling_1000_to_850_kW=(H(out, K(1000))-H(out, COT))/1e3,
    dewpoint=dict(yH2O=yW, pH2O_bar=pw/1e5, T_dew_C=C(CP.PropsSI('T', 'P', pw, 'Q', 0, 'Water'))))

# ---------------------------------------------------------------- 4. Radiant coil sizing
Do, Di = 0.120, 0.100
Ac = np.pi/4*Di**2
FLUX = 100.0       # kW/m2 outside, average
TAU = 0.300        # s


def Qv_profile(xot):
    z = np.linspace(0, 1, 201); T = xot+(COT-xot)*z
    n = nE_tot*(1+X*z)+nW
    return z, n/3600*R_GAS*T/P          # m3/s, T and conversion linear in length


def coil_lengths(xot):
    q = (H(out, COT)-H(feed, xot))/1e3
    z, Qv = Qv_profile(xot)
    L_tau = TAU/np.trapezoid(1/Qv, z)/Ac
    return q, q/(FLUX*np.pi*Do), L_tau


z, Qv = Qv_profile(XOT)
L_flux = Q_rad/(FLUX*np.pi*Do)
R['4_coil_sizing'] = dict(
    Ac_m2=Ac, Qv_in_m3s=Qv[0], Qv_out_m3s=Qv[-1],
    L_from_flux_m=L_flux, A_outer_m2=L_flux*np.pi*Do, L_inside_area_basis_m=Q_rad/(FLUX*np.pi*Di),
    L_from_tau_m=TAU/np.trapezoid(1/Qv, z)/Ac, V_for_tau_m3=TAU/np.trapezoid(1/Qv, z),
    tau_at_L_flux_ms=1000*L_flux*Ac*np.trapezoid(1/Qv, z),
    avg_flux_to_match_tau_kWm2=Q_rad/(np.pi*Do*TAU/np.trapezoid(1/Qv, z)/Ac),
    crossover_sweep=[dict(XOT_C=x, Q_rad_kW=a, L_flux_m=b, L_tau_m=c)
                     for x in [650, 700, 725, 750] for a, b, c in [coil_lengths(K(x))]],
    XOT_where_lengths_match_C=C(brentq(lambda T: coil_lengths(T)[1]-coil_lengths(T)[2], K(650), K(800))))
Q_rad_750 = coil_lengths(K(750))[0]

# ---------------------------------------------------------------- 4.1 Tube-metal temperature check
def props(comp, T):
    g.TPX = T, P, comp
    return dict(mu=g.viscosity, k=g.thermal_conductivity, cp=g.cp_mass, rho=g.density,
                Pr=g.viscosity*g.cp_mass/g.thermal_conductivity, c=g.sound_speed)


p_in, p_out = props(feed, XOT), props(out, COT)


def h_DB(G, p):
    Re = G*Di/p['mu']; return 0.023*Re**0.8*p['Pr']**0.4*p['k']/Di, Re


K_WALL = 20.0                                   # W/m K, HP alloy at ~1000 C
q_in_per_q_out = Do/Di                           # inside flux / outside flux
dT_wall_per_kWm2 = 1e3*(Do/2)*np.log(Do/Di)/K_WALL   # K per kW/m2 of outside flux  (= 0.547)
F_DARCY, K_BEND = 0.02, 0.5


def arrangement(n_coils, legs, leg_len=None):
    leg = leg_len if leg_len else L_flux/(n_coils*legs)
    Lc = leg*legs
    G = m_out/3600/(n_coils*Ac)
    h_out, Re_out = h_DB(G, p_out); h_in, Re_in = h_DB(G, p_in)
    rho_m = (p_in['rho']+p_out['rho'])/2
    dP = (F_DARCY*Lc/Di + K_BEND*(legs-1))*G**2/(2*rho_m)
    tmt = C(COT) + FLUX*q_in_per_q_out*1e3/h_out + dT_wall_per_kWm2*FLUX
    return dict(coils=n_coils, legs_per_coil=legs, leg_m=leg, L_per_coil_m=Lc, G_kgm2s=G,
                Re_out=Re_out, h_in=h_in, h_out=h_out, TMT_uniform_flux_C=tmt, dP_bar=dP/1e5,
                v_in=G/p_in['rho'], v_out=G/p_out['rho'], Mach_out=G/p_out['rho']/p_out['c'])


R['4_1_TMT'] = dict(
    inlet_props=p_in, outlet_props=p_out, k_wall=K_WALL, wall_dT_at_100_K=dT_wall_per_kWm2*FLUX,
    arrangements={'20 single-pass tubes': arrangement(20, 1),
                  '12 U-coils': arrangement(12, 2),
                  '6 coils x 4 legs (selected)': arrangement(6, 4),
                  '4 coils x 6 legs': arrangement(4, 6)})

# Flux-profile model: gas T proportional to absorbed heat, h linear from inlet to outlet value,
# TMT = T_gas + inside film dT + wall dT.
coil = arrangement(6, 4)
h_i, h_o = coil['h_in'], coil['h_out']


def tmt_profile(qfun, n=800):
    zz = (np.arange(n)+0.5)/n
    q = qfun(zz)
    Tg = C(XOT)+(C(COT)-C(XOT))*np.cumsum(q)/q.sum()
    tmt = Tg + q*q_in_per_q_out*1e3/(h_i+(h_o-h_i)*zz) + dT_wall_per_kWm2*q
    i = tmt.argmax()
    return dict(peak_TMT_C=tmt.max(), at_z=zz[i], mean_flux=q.mean())


def steps(*qs): return lambda zz: np.array(qs)[np.minimum((zz*len(qs)).astype(int), len(qs)-1)]
def linear(q0, q1): return lambda zz: q0+(q1-q0)*zz


R['4_1_flux_profiles_6x4'] = {
    'uniform 100': tmt_profile(linear(100, 100)),
    'linear 130->70': tmt_profile(linear(130, 70)),
    'linear 150->50': tmt_profile(linear(150, 50)),
    '2 zones 120/80': tmt_profile(steps(120, 80)),
    '3 zones 125/100/75': tmt_profile(steps(125, 100, 75)),
    '4 zones 130/110/90/70': tmt_profile(steps(130, 110, 90, 70))}

# ---------------------------------------------------------------- 4.2 Revised coil design
ALPHA = 16e-6
R['4_2_coil_design'] = dict(coil, legs_total=24, L_total_m=L_flux,
                           coil_inlet_P_bara=P/1e5+coil['dP_bar'],
                           thermal_growth_per_leg_m=ALPHA*coil['leg_m']*(1000-20))

# ---------------------------------------------------------------- 5.1 Combustion
ex = 0.20; O2 = 2*(1+ex); N2 = O2*79/21
fg = {'CO2': 1, 'H2O': 2, 'O2': 2*ex, 'N2': N2}; air = {'O2': O2, 'N2': N2}


def hsens(comp, T):  # kJ per mol fuel
    tot = 0
    for s, nn in comp.items():
        g.TPX = T, 1.013e5, {s: 1}; h1 = g.enthalpy_mole
        g.TPX = T0, 1.013e5, {s: 1}; tot += nn*(h1-g.enthalpy_mole)/1e6
    return tot


def hf(s): g.TPX = T0, 1.013e5, {s: 1}; return g.enthalpy_mole


LHV = -(hf('CO2')+2*hf('H2O')-hf('CH4')-2*hf('O2'))/1e6
HHV = LHV + 2*44.0
T_BW = K(1100)
h_fg_bw = hsens(fg, T_BW)
LOSS = 0.02
q_mol = LHV-h_fg_bw
nCH4 = Q_rad/(q_mol*(1-LOSS))*3600/1e3
Tad = brentq(lambda T: hsens(fg, T)-LHV, 1000, 3000)
m_air = nCH4*(O2*MW['O2']+N2*MW['N2'])
R['5_1_combustion'] = dict(
    LHV_kJmol=LHV, HHV_kJmol=HHV, LHV_MJkg=LHV/MW['CH4'], excess_air=ex,
    air_mol_per_mol=O2+N2, fg_mol_per_mol=sum(fg.values()), h_fg_1100_kJmol=h_fg_bw,
    q_to_coil_kJmol=q_mol, eta_radiant=q_mol*(1-LOSS)/LHV,
    nCH4_kmolh=nCH4, mCH4_kgh=nCH4*MW['CH4'], fuel_Nm3h=nCH4*22.414,
    fired_LHV_MW=nCH4*LHV/3600, fired_HHV_MW=nCH4*HHV/3600,
    n_air_kmolh=nCH4*(O2+N2), m_air_kgh=m_air, air_Nm3h=nCH4*(O2+N2)*22.414,
    n_fg_kmolh=nCH4*sum(fg.values()), m_fg_kgh=nCH4*(MW['CO2']+2*MW['H2O']+2*ex*MW['O2']+N2*MW['N2']),
    fg_molfrac={k: v/sum(fg.values()) for k, v in fg.items()}, T_adiabatic_C=C(Tad),
    CO2_kgh=nCH4*MW['CO2'], CO2_t_y=nCH4*MW['CO2']*hours/1e3, CO2_t_per_t=nCH4*MW['CO2']*hours/1e3/50000)

# ---------------------------------------------------------------- 5.2 Convection section
nfg = {k: v*nCH4 for k, v in fg.items()}
def Hfg(T, nf=nfg): return H(nf, T, 1.013e5)/1e3
def T_after(Tin, duty, nf=nfg): return brentq(lambda T: Hfg(Tin, nf)-Hfg(T, nf)-duty, 300, Tin)


cascade = []; T = T_BW
for name, d, proc in [('HTC: mixed feed', Q_mix, f'{C(Tmix):.0f} -> 650'),
                      ('Dilution steam superheater', Q_sh, '139 -> 200'),
                      ('Ethane feed preheater', Q_eth, '25 -> 150'),
                      ('Dilution steam boiler', Q_boil, '139 boiling'),
                      ('Process water economizer', Q_econ, '25 -> 139')]:
    Tn = T_after(T, d); cascade.append(dict(bank=name, duty_kW=d, fg_in_C=C(T), fg_out_C=C(Tn), process_C=proc)); T = Tn
T_STACK = K(150)
surplus = Hfg(T)-Hfg(T_STACK)
hb = CP.PropsSI('H', 'T', K(105), 'P', 41e5, 'Water'); hhp = CP.PropsSI('H', 'T', K(400), 'P', 41e5, 'Water')
fired = nCH4*LHV/3.6
R['5_2_convection'] = dict(
    cascade=cascade, T_after_process_banks_C=C(T),
    Q_fg_above_150C_kW=Hfg(T_BW)-Hfg(T_STACK), Q_surplus_kW=surplus,
    HP_steam_41bar_400C_kgh=surplus*1e3/(hhp-hb)*3600,
    fraction_fired_used_process=Q_tot/fired,
    fraction_fired_used_incl_HP_steam=(Q_tot+surplus)/fired,
    stack_loss_kW=Hfg(T_STACK)-Hfg(T0), wall_loss_kW=LOSS*nCH4*q_mol/3.6)

# Fans, stack, crossover pipe
V_air = nCH4*(O2+N2)/3600*R_GAS*T0/1.013e5
V_stack = nCH4*sum(fg.values())/3600*R_GAS*T_STACK/1.013e5
V_xo = Qv[0]
R['5_2_fans_stack_piping'] = dict(
    air_m3s_25C=V_air, FD_fan_design_m3s=1.2*V_air,
    stack_flow_m3s_150C=V_stack, stack_velocity_ms=7.6, stack_D_m=np.sqrt(4*V_stack/(np.pi*7.6)),
    crossover_flow_m3s=V_xo, crossover_velocity_ms=30, crossover_ID_m=np.sqrt(4*V_xo/(np.pi*30)))

# 5.3 burners
R['5_3_burners'] = dict(n_burners=16, MW_each_LHV=nCH4*LHV/3600/16)

# ---------------------------------------------------------------- 7.1 Electric demand
EFF = dict(boiler=0.99, sheathed=0.97, joule_preheat=0.97, cracker=0.92, power_conv=0.98)
T_SHEATH_OUT = K(450)
Q_mix_lo = (H(feed, T_SHEATH_OUT)-H(feed, Tmix))/1e3
Q_mix_hi = (H(feed, XOT)-H(feed, T_SHEATH_OUT))/1e3
E = dict(boiler=(Q_econ+Q_boil)/EFF['boiler'],
         sheathed_heaters=(Q_sh+Q_eth+Q_mix_lo)/EFF['sheathed'],
         joule_preheat_450_650=Q_mix_hi/EFF['joule_preheat'],
         cracker=Q_rad/EFF['cracker'])
E['subtotal'] = sum(E.values())
E['grid'] = E['subtotal']/EFF['power_conv']
E['MWh_y'] = E['grid']*hours/1e3
E['MWh_per_t'] = E['MWh_y']/50000
R['7_1_electric'] = dict(process_kW=dict(boiler=Q_econ+Q_boil, sheathed=Q_sh+Q_eth+Q_mix_lo,
                                         joule_preheat=Q_mix_hi, cracker=Q_rad),
                         load_kW=E, efficiencies=EFF)

# ---------------------------------------------------------------- 7.2 Joule-heated coil
RHO_EL = 1.15e-6
Am = np.pi/4*(Do**2-Di**2)
Rpm = RHO_EL/Am


def joule(flux, length):
    qpm = flux*np.pi*Do*1e3; I = np.sqrt(qpm/Rpm); Vpm = I*Rpm
    return dict(flux_kWm2=flux, W_per_m=qpm, I_leg_A=I, V_per_m=Vpm, V=Vpm*length, P_kW=qpm*length/1e3)


leg = coil['leg_m']
# Top nodes (inlet, middle bend, outlet) grounded; supply fed at each bottom bend.
# The two legs of a zone are therefore in parallel: zone voltage = one leg, feed current = 2 legs.
z1, z2 = joule(120, leg), joule(80, leg)
z1['P_zone_kW'], z2['P_zone_kW'] = 2*z1['P_kW'], 2*z2['P_kW']
R['7_2_joule'] = dict(
    metal_area_m2=Am, R_ohm_per_m=Rpm, average=joule(FLUX, leg),
    per_two_legs_avg_kW=joule(FLUX, 2*leg)['P_kW'],
    zone_inlet_legs_1_2=dict(z1, feed_current_A=2*z1['I_leg_A']),
    zone_outlet_legs_3_4=dict(z2, feed_current_A=2*z2['I_leg_A']),
    coil_power_kW=z1['P_zone_kW']+z2['P_zone_kW'], total_power_6_coils_kW=6*(z1['P_zone_kW']+z2['P_zone_kW']),
    skin_depth_60Hz_mm=1e3*np.sqrt(RHO_EL/(np.pi*60*MU0)))

# ---------------------------------------------------------------- 7.3 Indirect elements and induction
R['7_3_indirect_element_T_C'] = {f'tube {Tt} C, F={F}': ((Tt+273.15)**4+FLUX*1e3/(F*SIG))**0.25-273.15
                                 for Tt in [950, 1000] for F in [0.6, 0.75]}
rod_kW = 7e4*np.pi*0.05*1.8/1e3          # 7 W/cm2 on a 50 mm x 1.8 m rod
R['7_3_SiC_rods'] = dict(kW_per_rod=rod_kW, n_rods=E['cracker']/rod_kW,
                         A_per_rod_415V_wye=rod_kW*1e3/(415/np.sqrt(3)))
RHO_IND = 1.2e-6
f_ind = RHO_IND/(np.pi*MU0*0.010**2)
ind_loss = Q_rad/0.80-Q_rad
ind_loss_750 = Q_rad_750/0.80-Q_rad_750
cw = lambda W: W*1e3/(4186*20)*3.6     # t/h at 20 K rise
R['7_3_induction'] = dict(
    skin_depth_mm={f: 1e3*np.sqrt(RHO_EL/(np.pi*f*MU0)) for f in [50, 1e3, 3e3, 1e4, 3e4]},
    freq_for_10mm_skin_Hz=f_ind,
    loss_kW_at_80pct=ind_loss, cooling_water_t_h=cw(ind_loss),
    loss_kW_at_80pct_XOT750=ind_loss_750, cooling_water_t_h_XOT750=cw(ind_loss_750))

# ---------------------------------------------------------------- 8. Comparison
GAS_USD_MMBTU, POWER_USD_MWH = 3.50, 70.0
MMBTU_PER_MWH = 3.412
gas_cost_HHV = nCH4*HHV/3600*hours*MMBTU_PER_MWH*GAS_USD_MMBTU
gas_cost_LHV = nCH4*LHV/3600*hours*MMBTU_PER_MWH*GAS_USD_MMBTU
power_cost = E['MWh_y']*POWER_USD_MWH
CO2 = R['5_1_combustion']['CO2_t_y']
R['8_comparison'] = dict(
    fired_LHV_MW=nCH4*LHV/3600, grid_MW=E['grid']/1e3,
    gas_radiant_eff=R['5_1_combustion']['eta_radiant'], elec_radiant_eff=EFF['cracker'],
    gas_overall_use=Q_tot/fired, gas_overall_use_incl_steam=(Q_tot+surplus)/fired,
    elec_overall_use=Q_tot/E['grid'],
    grid_CO2_t_y={'0.37 t/MWh (US avg)': E['MWh_y']*0.37, '0.35 (NG CCGT)': E['MWh_y']*0.35, '0': 0.0},
    primary_fuel_CCGT55_MW=E['grid']/1e3/0.55,
    gas_cost_USD_y_HHV=gas_cost_HHV, gas_cost_USD_y_LHV=gas_cost_LHV, power_cost_USD_y=power_cost,
    CO2_price_to_close_gap_USD_t=(power_cost-gas_cost_HHV)/CO2,
    breakeven_grid_intensity_t_MWh=CO2/E['MWh_y'],
    breakeven_power_price_USD_MWh=gas_cost_HHV/E['MWh_y'],
    breakeven_power_price_LHV_4USD=nCH4*LHV/3600*hours*MMBTU_PER_MWH*4.0/E['MWh_y'])

# ---------------------------------------------------------------- 9. Bonus: air preheat and quench
aph = []
for Tap in [None, 100, 200, 300, 400]:
    ha = hsens(air, K(Tap)) if Tap else 0.0
    n2 = Q_rad/((LHV+ha-h_fg_bw)*(1-LOSS))*3600/1e3
    nf = {k: v*n2 for k, v in fg.items()}
    avail = Hfg(T_BW, nf)-Hfg(T_STACK, nf)
    need = Q_conv+n2*ha/3.6
    row = dict(T_air_C=Tap, mCH4_kgh=n2*MW['CH4'], fuel_saved_pct=100*(1-n2/nCH4),
               Q_aph_kW=n2*ha/3.6, fg_avail_above_150_kW=avail, need_kW=need, feasible=bool(avail >= need),
               fuel_eff=Q_tot/(n2*LHV/3.6))
    if Tap:
        Tin_aph = brentq(lambda T: Hfg(T, nf)-Hfg(T_STACK, nf)-n2*ha/3.6, T_STACK, T_BW)
        row['APH_flue_in_C'] = C(Tin_aph)
    aph.append(row)
nCH4_fuel = nCH4
fuel_pre200 = nCH4*(H({'CH4': 1}, K(200), 1.013e5)-H({'CH4': 1}, T0, 1.013e5))/1e3
q600 = (H(out, COT)-H(out, K(600)))/1e3
R['9_bonus'] = dict(
    air_preheat=aph, fuel_preheat_200C_kW=fuel_pre200,
    quench_850_to_600_kW=q600, dilution_water_heat_kW=Q_econ+Q_boil,
    grid_with_quench_steam_kW=E['grid']-E['boiler']/EFF['power_conv'],
    quench_recoverable_kW={f'850->{t}': (H(out, COT)-H(out, K(t)))/1e3 for t in [600, 500, 400]})

json.dump(R, open(OUT, 'w'), indent=1, default=float)

# ---------------------------------------------------------------- Short printed summary
b = R['1_2_basis_and_balance']; c = R['5_1_combustion']; cmp_ = R['8_comparison']
print(f"Ethylene {b['mC2H4_kgh']:.0f} kg/h; furnace ethane {b['mE_tot']:.0f} kg/h; steam {b['mW_kgh']:.0f} kg/h")
print(f"Process heat {Q_tot/1e3:.2f} MW (feed prep {Q_conv/1e3:.2f}, radiant {Q_rad/1e3:.2f})")
print(f"Coil length: flux {L_flux:.0f} m, tau {R['4_coil_sizing']['L_from_tau_m']:.0f} m; "
      f"match at XOT {R['4_coil_sizing']['XOT_where_lengths_match_C']:.0f} C")
for k, a in R['4_1_TMT']['arrangements'].items():
    print(f"  {k:32s} G {a['G_kgm2s']:5.1f}  h_out {a['h_out']:5.0f}  TMT {a['TMT_uniform_flux_C']:5.0f} C  dP {a['dP_bar']:.2f} bar")
for k, v in R['4_1_flux_profiles_6x4'].items():
    print(f"  6x4, {k:22s} peak TMT {v['peak_TMT_C']:.0f} C")
print(f"Fuel {c['mCH4_kgh']:.0f} kg/h CH4, {c['fired_LHV_MW']:.1f} MW LHV; CO2 {c['CO2_t_y']:.0f} t/y")
print(f"Electric grid {E['grid']/1e3:.1f} MW, {E['MWh_per_t']:.2f} MWh/t")
print(f"Break-even grid {cmp_['breakeven_grid_intensity_t_MWh']:.3f} t/MWh; power price ${cmp_['breakeven_power_price_USD_MWh']:.1f}/MWh; "
      f"CO2 price ${cmp_['CO2_price_to_close_gap_USD_t']:.0f}/t")
j = R['7_2_joule']
print(f"Joule zones: inlet {j['zone_inlet_legs_1_2']['V']:.0f} V / {j['zone_inlet_legs_1_2']['feed_current_A']/1e3:.1f} kA, "
      f"outlet {j['zone_outlet_legs_3_4']['V']:.0f} V / {j['zone_outlet_legs_3_4']['feed_current_A']/1e3:.1f} kA")
print('Results written to', OUT)
