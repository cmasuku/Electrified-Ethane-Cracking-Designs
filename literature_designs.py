"""
Literature benchmarks, literature-informed refinements and new design options (Sections 3.9-3.11).

Re-uses furnace_model.py unchanged (executed in-process) and writes literature_results.json.
Literature values are typed in from the cited papers and marked with their source key.

  1. Benchmarks: published electrified ethane-cracker designs put on this paper's per-tonne basis.
  2. Refinements: (a) a literature product slate (Ghasemi Naraghi et al., 2026, Table 10.4) instead of
     100 % selectivity; (b) the dilution-steam ratio of 0.3 kg/kg used by Maporti et al. (2023) and
     Shin et al. (2025); (c) the electric heating efficiency range used in the literature (0.90-1.00).
  3. New designs:
     E2  electric furnace with feed-effluent interchange (after Maporti et al., 2023, configuration A),
         sized by a problem-table pinch analysis;
     F2  fired furnace burning its own tail gas with natural gas make-up (hydrogen fuel switching,
         Rightor, 2026; Shin et al., 2025);
     H1  hybrid furnace: electric radiant coil, feed preparation in a heater fired with the tail gas;
     P1  screening size of a proton-conducting electrochemical cell (PCEC) route at 50 kt/y
         (Ding et al., 2018; Haven et al., 2025).
  4. Storage needed for round-the-clock operation on variable renewables, using the
     method of Ghasemi Naraghi et al. (2026): store 70 % of daily energy if renewables cover 30 % of the day.

Usage:  python literature_designs.py
Needs:  cantera, CoolProp, numpy, scipy
"""
import os, sys, json, runpy, tempfile, re
import numpy as np
import cantera as ct
import CoolProp.CoolProp as CP

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = os.path.join(tempfile.mkdtemp(), 'base.json')
sys.argv = ['furnace_model.py', tmp]
ns = runpy.run_path(os.path.join(HERE, 'furnace_model.py'))
R, K, C, H = ns['R'], ns['K'], ns['C'], ns['H']
g, MW, hours = ns['g'], ns['MW'], ns['hours']
feed, out, XOT, COT, T0 = ns['feed'], ns['out'], ns['XOT'], ns['COT'], ns['T0']
EFF = ns['EFF']
Q_rad, Q_conv, Q_tot = ns['Q_rad'], ns['Q_conv'], ns['Q_tot']          # kW
T_BW, T_STACK, LOSS, ex = ns['T_BW'], ns['T_STACK'], ns['LOSS'], ns['ex']
cmp8 = R['8_comparison']
CO2_fired = R['5_1_combustion']['CO2_t_y']
gas_cost = cmp8['gas_cost_USD_y_HHV']
USD_per_kWh_LHV_gas = gas_cost / (R['5_1_combustion']['fired_LHV_MW'] * 1e3 * hours)   # $/kWh(LHV), HHV-priced
t_h = 50000 / hours                                                   # t C2H4 per h
F = 96485.0
out_json = {}

# ---------------------------------------------------------------- helpers
def hf(s):
    g.TPX = T0, 1.013e5, {s: 1}; return g.enthalpy_mole / 1e6          # kJ/mol


def hsens(comp, T):                                                    # kJ, comp in mol
    tot = 0.0
    for s, n in comp.items():
        g.TPX = T, 1.013e5, {s: 1}; h1 = g.enthalpy_mole
        g.TPX = T0, 1.013e5, {s: 1}; tot += n * (h1 - g.enthalpy_mole) / 1e6
    return tot


LHV_mol = {'H2': -(hf('H2O') - hf('H2') - 0.5 * hf('O2')),
           'CH4': -(hf('CO2') + 2 * hf('H2O') - hf('CH4') - 2 * hf('O2'))}  # kJ/mol


def flue_per_mol(fuel):
    """Flue gas (mol) per mol of fuel mixture fuel={'H2':y,'CH4':1-y}, 20 % excess air."""
    a, b = fuel.get('H2', 0.0), fuel.get('CH4', 0.0)
    o2 = 0.5 * a + 2 * b
    return {'CO2': b, 'H2O': a + 2 * b, 'O2': ex * o2, 'N2': (1 + ex) * o2 * 79 / 21}


def fired_case(fuel_avail_kmolh, Q_radiant, Q_need):
    """Radiant duty Q_radiant (kW) met first by the available tail gas, then natural gas make-up.
    Same bridgewall temperature (1100 C) and 2 % casing loss as the base model."""
    n_tg = sum(fuel_avail_kmolh.values())
    y = {k: v / n_tg for k, v in fuel_avail_kmolh.items()} if n_tg else {}
    LHV_tg = sum(y[k] * LHV_mol[k] for k in y) if n_tg else 0.0
    q_tg = (LHV_tg - hsens(flue_per_mol(y), T_BW)) * (1 - LOSS) if n_tg else 0.0
    q_ng = (LHV_mol['CH4'] - hsens(flue_per_mol({'CH4': 1}), T_BW)) * (1 - LOSS)
    Q_from_tg = n_tg * q_tg / 3.6                                       # kW
    if Q_from_tg >= Q_radiant:
        n_tg_used, n_ng = Q_radiant * 3.6 / q_tg, 0.0
    else:
        n_tg_used, n_ng = n_tg, (Q_radiant - Q_from_tg) * 3.6 / q_ng
    used = {k: n_tg_used * y[k] for k in y}
    used['CH4_ng'] = n_ng
    fl = {s: 0.0 for s in ['CO2', 'H2O', 'O2', 'N2']}
    for k, n in [('H2', used.get('H2', 0)), ('CH4', used.get('CH4', 0) + n_ng)]:
        for s, v in flue_per_mol({k: 1}).items():
            fl[s] += n * v
    avail = (hsens(fl, T_BW) - hsens(fl, T_STACK)) / 3.6                # kW (kmol/h * kJ/mol / 3.6)
    fired_LHV = (sum(used.get(k, 0) * LHV_mol[k] for k in ['H2', 'CH4']) + n_ng * LHV_mol['CH4']) / 3.6
    CO2 = (used.get('CH4', 0) + n_ng) * MW['CO2'] * hours / 1e3
    return dict(tailgas_used_kmolh={k: used.get(k, 0) for k in y}, ng_kgh=n_ng * MW['CH4'],
                fired_LHV_kW=fired_LHV, ng_LHV_kW=n_ng * LHV_mol['CH4'] / 3.6,
                eta_radiant=Q_radiant / fired_LHV, CO2_t_y=CO2, CO2_t_per_t=CO2 / 50000,
                flue_kmolh=sum(fl.values()), flue_heat_above_150C_kW=avail, feed_prep_need_kW=Q_need,
                convection_feasible=bool(avail >= Q_need), surplus_kW=avail - Q_need,
                tailgas_unused_kmolh={k: (n_tg - n_tg_used) * y[k] for k in y})


# ---------------------------------------------------------------- 1. Benchmarks on a per-tonne basis
ours = dict(
    capacity_kt_y=50, steam_ratio=0.5, conversion=0.65, XOT_C=650, COT_C=850, tau_s=0.30,
    radiant_heat_MWh_per_t=Q_rad / 1e3 / t_h,
    reactor_electricity_MWh_per_t=Q_rad / EFF['cracker'] / EFF['power_conv'] / 1e3 / t_h,
    furnace_electricity_MWh_per_t=R['7_1_electric']['load_kW']['grid'] / 1e3 / t_h,
    fired_LHV_MWh_per_t=R['5_1_combustion']['fired_LHV_MW'] / t_h,
    H2_kg_per_t=R['1_2_basis_and_balance']['mH2_kgh'] / t_h)
maporti_t_h = 170e3 / 8400
lit = {
    'Maporti2023': dict(capacity_kt_y=170, steam_ratio=0.3, conversion=0.642, XOT_C=670, COT_C=846, tau_s=0.46,
                        eta_heat=0.90, reactor_power_MW=43.5,
                        reactor_electricity_MWh_per_t=43.5 / maporti_t_h,
                        radiant_heat_MWh_per_t=43.5 * 0.90 / maporti_t_h,
                        external_energy_GJ_per_t=7.7, fired_external_energy_GJ_per_t=15.7,
                        CI_switch_vs_ODH_t_MWh=0.197, CO2_ESC_t_per_t=0.38, CO2_fired_t_per_t=0.852),
    'GhasemiNaraghi2026': dict(capacity_kt_y=1000, min_energy_electrified_MWh_per_t=1.75,
                               min_energy_conventional_MWh_per_t=4.27, eta_electric=0.95,
                               slate_mass_frac=dict(C2H6=0.3753, C2H4=0.5, C3H8=0.0038, C3H6=0.0458, C2H2=0.0006,
                                                    C4H6=0.0032, CH4=0.0333, H2=0.0377)),
    'Shin2025': dict(capacity_kt_y=1000, steam_ratio=0.33, eta_heat=1.0, extra_power_kWh_per_kg=2.86,
                     H2_export_kg_per_t=65.1, other_gas_kg_per_t=119.5,
                     breakeven_CI_vs_baseline_t_MWh=0.380, carbon_neutral_CI_t_MWh=0.060,
                     breakeven_price_USD_MWh=41.1),
    'Fitriasari2026': dict(feed='naphtha', eta=0.95, SEC_electrified_SC_MWh_per_t=10.35, SEC_plasma_MWh_per_t=43.47),
    'Haven2025': dict(capacity_kt_y=100, T_C=550, i_A_cm2=1.0, area_base_m2=3500, area_target_m2=2800,
                      module_m2=44, yield_base=0.25, yield_target=0.50, regen_share=0.17),
    'Ding2018': dict(T_C=400, i_A_cm2=1.0, V_cell=0.41, elec_GJ_per_t=2.8, heat_GJ_per_t=3.2),
}
out_json['benchmarks'] = dict(this_work=ours, literature=lit)

# ---------------------------------------------------------------- 2a. Literature product slate
nasa = {s.name: s for s in ct.Species.list_from_file('nasa_gas.yaml')}
pick = {'C2H6': 'C2H6', 'C2H4': 'C2H4', 'C3H8': 'C3H8', 'C3H6': 'C3H6,propylene', 'C2H2': 'C2H2,acetylene',
        'C4H6': 'C4H6,butadiene', 'CH4': 'CH4', 'H2': 'H2', 'H2O': 'H2O'}
gs = ct.Solution(thermo='ideal-gas', species=[nasa[v] for v in pick.values()])
idx = {k: gs.species_index(v) for k, v in pick.items()}
MWs = {k: gs.molecular_weights[i] for k, i in idx.items()}


def Hn(comp_kgh, T, p=2.5e5):
    """Enthalpy flow (kW) on the NASA basis, comp in kg/h."""
    mol = {k: v / MWs[k] for k, v in comp_kgh.items() if v > 0}
    X = np.zeros(gs.n_species)
    for k, v in mol.items(): X[idx[k]] = v
    gs.TPX = T, p, X
    return sum(mol.values()) * gs.enthalpy_mole / 3600 / 1e3


def slate_case(wf, steam_ratio=0.5):
    tot = sum(wf.values()); wf = {k: v / tot for k, v in wf.items()}
    m_hc = 6250 / wf['C2H4']                                            # kg/h hydrocarbon through the coil
    fd = {'C2H6': m_hc, 'H2O': steam_ratio * m_hc}
    ef = {k: v * m_hc for k, v in wf.items()}; ef['H2O'] = steam_ratio * m_hc
    q_rad = Hn(ef, COT) - Hn(fd, XOT)
    conv = 1 - wf['C2H6']
    return dict(ethane_through_coil_kgh=m_hc, fresh_ethane_kgh=m_hc * conv, recycle_kgh=m_hc * wf['C2H6'],
                steam_kgh=steam_ratio * m_hc, conversion=conv, ethylene_yield=wf['C2H4'],
                mass_selectivity=wf['C2H4'] / conv,
                byproducts_kgh={k: v * m_hc for k, v in wf.items() if k not in ('C2H6', 'C2H4')},
                Q_rad_kW=q_rad, Q_rad_GJ_per_t=q_rad * 3.6 / t_h / 1e3,
                fresh_ethane_t_per_t=m_hc * conv / 6250)


ideal_wf = {'C2H6': 0.35, 'C2H4': 0.65 * 28.054 / 30.07, 'H2': 0.65 * 2.016 / 30.07}
s_ideal = slate_case(ideal_wf)
s_lit = slate_case(lit['GhasemiNaraghi2026']['slate_mass_frac'])
# Feed preparation scales with the hydrocarbon (and steam) flow through the coil at the same ratios
scale = s_lit['ethane_through_coil_kgh'] / R['1_2_basis_and_balance']['mE_tot']
Qconv_lit = Q_conv * scale
Qrad_lit = Q_rad * s_lit['Q_rad_kW'] / s_ideal['Q_rad_kW']           # GRI basis, scaled by the NASA-basis ratio
tg_ideal = {'H2': R['1_2_basis_and_balance']['mH2_kgh'] / MW['H2']}
tg_lit = {'H2': s_lit['byproducts_kgh']['H2'] / MW['H2'], 'CH4': s_lit['byproducts_kgh']['CH4'] / MW['CH4']}
LHV_tg = lambda tg: sum(n * LHV_mol[k] for k, n in tg.items()) / 3.6   # kW
out_json['slate'] = dict(
    ideal_NASA_basis=s_ideal, literature=s_lit, Q_rad_GRI_basis_ideal_kW=Q_rad,
    Q_rad_literature_kW=Qrad_lit, Q_feed_prep_literature_kW=Qconv_lit,
    Q_total_literature_kW=Qrad_lit + Qconv_lit, Q_total_ideal_kW=Q_tot,
    Q_total_increase_pct=100 * ((Qrad_lit + Qconv_lit) / Q_tot - 1),
    radiant_per_t_increase_pct=100 * (Qrad_lit / Q_rad - 1),
    tailgas_LHV_kW=dict(ideal=LHV_tg(tg_ideal), literature=LHV_tg(tg_lit)),
    tailgas_kg_per_t=dict(ideal=dict(H2=tg_ideal['H2'] * MW['H2'] / t_h),
                          literature={k: n * MW[k] / t_h for k, n in tg_lit.items()}),
    electric_grid_literature_kW=(R['7_1_electric']['load_kW']['grid']) * (Qrad_lit + Qconv_lit) / Q_tot)

# ---------------------------------------------------------------- 2b. Dilution-steam ratio 0.3 (full re-run of the model)
src = open(os.path.join(HERE, 'furnace_model.py')).read()
assert src.count('mW = 0.5*mE_tot') == 1
src03 = src.replace('mW = 0.5*mE_tot', 'mW = 0.3*mE_tot').replace('K(650), K(800)))', 'K(500), K(950)))')
tmp2 = os.path.join(tempfile.mkdtemp(), 'sr03.json')
ns03 = {'__name__': '__sr03__', '__file__': os.path.join(HERE, 'furnace_model.py')}
sys.argv = ['furnace_model.py', tmp2]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src03, 'furnace_model_sr03', 'exec'), ns03)
R03 = ns03['R']


def summary(Rx):
    a6 = Rx['4_1_TMT']['arrangements']
    key = [k for k in a6 if k.startswith('6 coils')][0]
    best = min(v['peak_TMT_C'] for v in Rx['4_1_flux_profiles_6x4'].values())
    return dict(Q_feed_prep_kW=Rx['3_energy_balance']['Q_feed_prep_kW'], Q_rad_kW=Rx['3_energy_balance']['Q_rad_kW'],
                Q_total_kW=Rx['3_energy_balance']['Q_total_kW'], GJ_per_t=Rx['3_energy_balance']['GJ_per_t_C2H4'],
                L_flux_m=Rx['4_coil_sizing']['L_from_flux_m'], L_tau_m=Rx['4_coil_sizing']['L_from_tau_m'],
                XOT_match_C=Rx['4_coil_sizing']['XOT_where_lengths_match_C'],
                G_6x4=a6[key]['G_kgm2s'], h_out_6x4=a6[key]['h_out'], TMT_uniform_6x4_C=a6[key]['TMT_uniform_flux_C'],
                dP_6x4_bar=a6[key]['dP_bar'], best_profile_TMT_C=best,
                fired_LHV_MW=Rx['5_1_combustion']['fired_LHV_MW'], CO2_t_y=Rx['5_1_combustion']['CO2_t_y'],
                grid_MW=Rx['7_1_electric']['load_kW']['grid'] / 1e3,
                breakeven_CI=Rx['8_comparison']['breakeven_grid_intensity_t_MWh'],
                breakeven_price=Rx['8_comparison']['breakeven_power_price_USD_MWh'],
                dilution_steam_kW=Rx['3_energy_balance']['Q_econ_kW'] + Rx['3_energy_balance']['Q_boil_kW'])


out_json['steam_ratio'] = {'0.5': summary(R), '0.3': summary(R03)}

# ---------------------------------------------------------------- 2c. Electric heating efficiency
eta_rows = []
for eta in [0.90, 0.92, 0.95, 1.00]:
    E = R['7_1_electric']['load_kW']
    sub = E['subtotal'] - E['cracker'] + Q_rad / eta
    grid = sub / EFF['power_conv']
    eta_rows.append(dict(eta=eta, grid_MW=grid / 1e3, breakeven_CI=CO2_fired / (grid * hours / 1e3),
                         breakeven_price=gas_cost / (grid * hours / 1e3)))
out_json['eta_sensitivity'] = eta_rows

# ---------------------------------------------------------------- 3. E2: feed-effluent interchange (pinch analysis)
P_st = ns['Ps']; Tsat = ns['Tsat']; mWs = ns['mWs']; nE_tot = ns['nE_tot']; Tmix = ns['Tmix']
h25 = CP.PropsSI('H', 'T', K(25), 'P', P_st, 'Water')


def h_water(T):          # kW relative to 25 C liquid; boiling spread over 0.2 K
    if T <= Tsat: return mWs * (CP.PropsSI('H', 'T', T, 'P', P_st, 'Water') - h25) / 1e3
    hl = CP.PropsSI('H', 'P', P_st, 'Q', 0, 'Water'); hv = CP.PropsSI('H', 'P', P_st, 'Q', 1, 'Water')
    if T <= Tsat + 0.2: return mWs * (hl - h25 + (hv - hl) * (T - Tsat) / 0.2) / 1e3
    return mWs * (CP.PropsSI('H', 'T', T, 'P', P_st, 'Water') - h25) / 1e3


cold_streams = [  # (name, T_in, T_out, H(T) in kW)
    ('dilution water/steam', K(25), K(200), h_water),
    ('ethane', K(25), K(150), lambda T: H({'C2H6': nE_tot}, T) / 1e3),
    ('mixed feed', Tmix, XOT, lambda T: H(feed, T) / 1e3)]
T_HOT_END = K(120)        # effluent not cooled below 120 C here (dew point of cracked gas is about 95 C)
hot_streams = [('cracked gas', COT, T_HOT_END, lambda T: H(out, T) / 1e3)]


def problem_table(dTmin, hot=hot_streams, cold=cold_streams, dT=0.25):
    Ts = np.arange(K(20), K(900), dT)             # shifted temperature grid
    def stream_dH(st, lo, hi, shift):
        name, a, b, f = st; tmin, tmax = min(a, b), max(a, b)
        l, u = max(lo - shift, tmin), min(hi - shift, tmax)
        return f(u) - f(l) if u > l else 0.0
    net = []
    for lo, hi in zip(Ts[:-1], Ts[1:]):
        qh = sum(stream_dH(s, lo, hi, -dTmin / 2) for s in hot)
        qc = sum(stream_dH(s, lo, hi, +dTmin / 2) for s in cold)
        net.append(qh - qc)
    net = np.array(net)[::-1]                      # from the top down
    casc = np.cumsum(net)
    QH = max(0.0, -casc.min())
    i_p = int(np.argmin(casc)); T_pinch_shift = Ts[::-1][i_p + 1]
    tot_h = sum(f(a) - f(b) for _, a, b, f in hot); tot_c = sum(f(b) - f(a) for _, a, b, f in cold)
    return dict(dTmin_K=dTmin, QH_min_kW=QH, QC_min_kW=QH + tot_h - tot_c, recovered_kW=tot_c - QH,
                T_pinch_shifted_C=C(T_pinch_shift), T_pinch_hot_C=C(T_pinch_shift + dTmin / 2),
                T_pinch_cold_C=C(T_pinch_shift - dTmin / 2), hot_available_kW=tot_h, cold_demand_kW=tot_c)


e2_sweep = [problem_table(d) for d in [10, 20, 30, 40, 50, 75, 100]]
e2 = problem_table(30)
cr = Q_rad / EFF['cracker']
trim = e2['QH_min_kW'] / EFF['joule_preheat']
grid_e2 = (cr + trim) / EFF['power_conv']
E_e2 = grid_e2 * hours / 1e3                                          # MWh/y
# Fired-plant steam forgone: in a fired plant the same cracked-gas heat would raise HP steam in the TLE.
# Site basis: replace it with steam from a gas boiler (90 % LHV efficiency assumed).
ETA_BOILER = 0.90
co2_per_kWh_gas = R['5_1_combustion']['CO2_t_y'] / (R['5_1_combustion']['fired_LHV_MW'] * 1e3 * hours)  # t/kWh LHV


def site_breakeven(grid_kW, recovered_kW):
    E = grid_kW * hours / 1e3
    gas_kW = recovered_kW / ETA_BOILER
    co2_boiler = gas_kW * hours * co2_per_kWh_gas
    cost_boiler = gas_kW * hours * USD_per_kWh_LHV_gas
    return dict(boiler_gas_kW=gas_kW, boiler_CO2_t_y=co2_boiler,
                breakeven_CI=(CO2_fired - co2_boiler) / E, breakeven_price=(gas_cost - cost_boiler) / E)


# Exchanger UA by stream-wise LMTD for the high-temperature gas-gas section (mixed feed to XOT)
def ua_section(Th_in, Tc_in, Tc_out, duty):
    Th_out = __import__('scipy.optimize', fromlist=['brentq']).brentq(
        lambda T: H(out, Th_in) / 1e3 - H(out, T) / 1e3 - duty, K(100), Th_in)
    d1, d2 = Th_in - Tc_out, Th_out - Tc_in
    lm = (d1 - d2) / np.log(d1 / d2) if abs(d1 - d2) > 1e-6 else d1
    return dict(T_hot_in_C=C(Th_in), T_hot_out_C=C(Th_out), T_cold_in_C=C(Tc_in), T_cold_out_C=C(Tc_out),
                duty_kW=duty, LMTD_K=lm, UA_kW_K=duty / lm)


Q_mix = ns['Q_mix']
htc = ua_section(COT, Tmix, XOT, Q_mix)
# residence time of cracked gas in the first exchanger pass while above 650 C is not modelled (see text)
out_json['E2_interchange'] = dict(
    pinch_30K=e2, sweep=e2_sweep, high_T_exchanger=htc,
    grid_kW=grid_e2, cracker_kW=cr, trim_heater_kW=trim, MWh_y=E_e2, MWh_per_t=E_e2 / 50000,
    breakeven_CI_fuel_basis=CO2_fired / E_e2, breakeven_price_fuel_basis=gas_cost / E_e2,
    site_basis=site_breakeven(grid_e2, e2['recovered_kW']),
    quench_steam_variant=dict(grid_kW=R['9_bonus']['grid_with_quench_steam_kW'],
                              site_basis=site_breakeven(R['9_bonus']['grid_with_quench_steam_kW'],
                                                        R['3_energy_balance']['Q_econ_kW'] + R['3_energy_balance']['Q_boil_kW'])),
    base_site_basis=site_breakeven(R['7_1_electric']['load_kW']['grid'], 0.0))

# Composite curves for the figure (30 K)
def composite(streams, Tgrid):
    Hc = []
    for T in Tgrid:
        s = 0.0
        for _, a, b, f in streams:
            lo, hi = min(a, b), max(a, b)
            s += f(min(max(T, lo), hi)) - f(lo)
        Hc.append(s)
    return np.array(Hc)


Tg_c = np.linspace(K(25), XOT, 300); Tg_h = np.linspace(T_HOT_END, COT, 300)
out_json['E2_interchange']['composite'] = dict(
    T_cold_C=C(Tg_c).tolist(), H_cold_kW=composite(cold_streams, Tg_c).tolist(),
    T_hot_C=C(Tg_h).tolist(), H_hot_kW=composite(hot_streams, Tg_h).tolist())

# ---------------------------------------------------------------- 3. F2: fired furnace on its own tail gas
f0 = fired_case({}, Q_rad, Q_conv)
f2_ideal = fired_case(tg_ideal, Q_rad, Q_conv)
f2_lit = fired_case(tg_lit, Qrad_lit, Qconv_lit)
f0_lit = fired_case({}, Qrad_lit, Qconv_lit)
out_json['F2_tailgas_fired'] = dict(base_check=f0, ideal_slate=f2_ideal, literature_slate=f2_lit,
                                    literature_slate_ng_only=f0_lit,
                                    CO2_reduction_pct_ideal=100 * (1 - f2_ideal['CO2_t_y'] / f0['CO2_t_y']),
                                    CO2_reduction_pct_lit=100 * (1 - f2_lit['CO2_t_y'] / f0_lit['CO2_t_y']),
                                    H2_forgone_t_y=R['1_2_basis_and_balance']['H2_t_y'])

# ---------------------------------------------------------------- 3. H1: electric radiant + tail-gas-fired feed heater
def feed_heater(tg, Q_need):
    n = sum(tg.values()); y = {k: v / n for k, v in tg.items()}
    LHVm = sum(y[k] * LHV_mol[k] for k in y)
    q = (LHVm - hsens(flue_per_mol(y), T_STACK)) * (1 - LOSS)          # kJ per mol fuel to process
    need = Q_need * 3.6 / q                                            # kmol/h
    used = min(need, n)
    co2 = used * y.get('CH4', 0) * MW['CO2'] * hours / 1e3
    return dict(efficiency_LHV=q / LHVm, fuel_needed_kmolh=need, fuel_available_kmolh=n,
                share_covered=min(1.0, n / need), fuel_LHV_kW=used * LHVm / 3.6,
                H2_left_kg_h=(n - used) * y.get('H2', 0) * MW['H2'], CO2_t_y=co2,
                makeup_kW=max(0.0, Q_need - n * q / 3.6))


h1_ideal = feed_heater(tg_ideal, Q_conv)
h1_lit = feed_heater(tg_lit, Qconv_lit)
grid_h1 = (Q_rad / EFF['cracker'] + h1_ideal['makeup_kW'] / EFF['sheathed']) / EFF['power_conv']
grid_h1_lit = (Qrad_lit / EFF['cracker'] + h1_lit['makeup_kW'] / EFF['sheathed']) / EFF['power_conv']
E_h1 = grid_h1 * hours / 1e3
out_json['H1_hybrid'] = dict(
    ideal=dict(h1_ideal, grid_kW=grid_h1, MWh_per_t=E_h1 / 50000,
               breakeven_CI=(CO2_fired - h1_ideal['CO2_t_y']) / E_h1, breakeven_price=gas_cost / E_h1),
    literature=dict(h1_lit, grid_kW=grid_h1_lit,
                    breakeven_CI=(f0_lit['CO2_t_y'] - h1_lit['CO2_t_y']) / (grid_h1_lit * hours / 1e3)))
# Switching threshold for a dual-mode plant (Maporti et al. 2023): electric E2 vs tail-gas-fired F2
out_json['switch_CI_E2_vs_F2_t_MWh'] = f2_ideal['CO2_t_y'] / E_e2

# ---------------------------------------------------------------- 3. P1: electrochemical (PCEC) screening at 50 kt/y
n_eth = 6250 / 28.054 / 3.6                                            # mol/s ethylene
I = 2 * F * n_eth                                                      # A, one H2 per C2H4 pumped through the membrane
A_ideal = I / 1e4                                                      # m2 at 1 A/cm2
dHr = lambda T: ns['dHr'](T)                                           # kJ/mol
p1 = dict(ethylene_mol_s=n_eth, current_MA=I / 1e6, area_ideal_m2=A_ideal,
          area_with_regen_m2=A_ideal / (1 - lit['Haven2025']['regen_share']),
          modules=A_ideal / (1 - lit['Haven2025']['regen_share']) / lit['Haven2025']['module_m2'],
          area_scaled_Haven_m2=dict(base=3500 * 50 / 100, target=2800 * 50 / 100),
          power_at_0p41V_MW=I * 0.41 / 1e6,
          V_thermoneutral={f'{t} C': dHr(K(t)) * 1e3 / (2 * F) for t in [400, 550]},
          power_thermoneutral_MW={f'{t} C': I * dHr(K(t)) * 1e3 / (2 * F) / 1e6 for t in [400, 550]},
          reaction_heat_MW={f'{t} C': n_eth * dHr(K(t)) / 1e3 for t in [400, 550, 850]},
          ethane_through_reactor_per_fresh={f'X={x}': 1 / x for x in [0.25, 0.50, 0.65]})
out_json['P1_PCEC'] = p1

# ---------------------------------------------------------------- 4. Storage for round-the-clock operation
stor = {}
for name, kw in [('base electric', R['7_1_electric']['load_kW']['grid']),
                 ('quench steam', R['9_bonus']['grid_with_quench_steam_kW']),
                 ('E2 interchange', grid_e2), ('H1 hybrid', grid_h1)]:
    stor[name] = dict(grid_MW=kw / 1e3, store_MWh=0.7 * 24 * kw / 1e3)
out_json['storage_30pct_VRE'] = stor

# ---------------------------------------------------------------- design summary table
out_json['design_summary'] = [
    dict(design='F1 gas-fired (base)', grid_MW=0.0, fired_LHV_MW=f0['fired_LHV_kW'] / 1e3, CO2_t_y=f0['CO2_t_y'],
         H2_export_t_y=R['1_2_basis_and_balance']['H2_t_y']),
    dict(design='F2 tail-gas fired', grid_MW=0.0, fired_LHV_MW=f2_ideal['fired_LHV_kW'] / 1e3,
         ng_LHV_MW=f2_ideal['ng_LHV_kW'] / 1e3, CO2_t_y=f2_ideal['CO2_t_y'], H2_export_t_y=0.0),
    dict(design='E1 electric (base)', grid_MW=R['7_1_electric']['load_kW']['grid'] / 1e3, CO2_t_y=0.0,
         H2_export_t_y=R['1_2_basis_and_balance']['H2_t_y'],
         breakeven_CI=cmp8['breakeven_grid_intensity_t_MWh'], breakeven_price=cmp8['breakeven_power_price_USD_MWh']),
    dict(design='E1q quench steam', grid_MW=R['9_bonus']['grid_with_quench_steam_kW'] / 1e3, CO2_t_y=0.0,
         H2_export_t_y=R['1_2_basis_and_balance']['H2_t_y']),
    dict(design='E2 interchange', grid_MW=grid_e2 / 1e3, CO2_t_y=0.0, H2_export_t_y=R['1_2_basis_and_balance']['H2_t_y'],
         breakeven_CI=CO2_fired / E_e2, breakeven_price=gas_cost / E_e2),
    dict(design='H1 hybrid', grid_MW=grid_h1 / 1e3, fired_LHV_MW=h1_ideal['fuel_LHV_kW'] / 1e3, CO2_t_y=h1_ideal['CO2_t_y'],
         H2_export_t_y=h1_ideal['H2_left_kg_h'] * hours / 1e3,
         breakeven_CI=(CO2_fired - h1_ideal['CO2_t_y']) / E_h1, breakeven_price=gas_cost / E_h1)]

# ---------------------------------------------------------------- comparisons against the tail-gas-fired reference F2
gas_cost_F2 = f2_ideal['ng_LHV_kW'] * hours * USD_per_kWh_LHV_gas
H2_LHV_used_H1 = h1_ideal['fuel_LHV_kW']
vsF2 = {}
for name, kw, co2, h2_kW in [('E1', R['7_1_electric']['load_kW']['grid'], 0.0, 0.0),
                             ('E1q', R['9_bonus']['grid_with_quench_steam_kW'], 0.0, 0.0),
                             ('E2', grid_e2, 0.0, 0.0), ('H1', grid_h1, 0.0, H2_LHV_used_H1)]:
    E = kw * hours / 1e3
    vsF2[name] = dict(breakeven_CI=(f2_ideal['CO2_t_y'] - co2) / E, breakeven_price=gas_cost_F2 / E,
                      H2_export_t_y=R['1_2_basis_and_balance']['H2_t_y'] - h2_kW / (LHV_mol['H2'] / MW['H2'] / 3.6) * hours / 1e3)
out_json['vs_F2'] = dict(gas_cost_F2_USD_y=gas_cost_F2, designs=vsF2)
# H1 with the burned hydrogen valued as natural-gas-equivalent fuel (cost and CO2)
co2_eq = H2_LHV_used_H1 * hours * co2_per_kWh_gas
cost_eq = H2_LHV_used_H1 * hours * USD_per_kWh_LHV_gas
out_json['H1_hybrid']['ideal']['fuel_equivalent_basis'] = dict(
    H2_burned_t_y=h1_ideal['fuel_LHV_kW'] / (LHV_mol['H2'] / MW['H2'] / 3.6) * hours / 1e3,
    CO2_equivalent_t_y=co2_eq, cost_equivalent_USD_y=cost_eq,
    breakeven_CI=(CO2_fired - co2_eq) / E_h1, breakeven_price=(gas_cost - cost_eq) / E_h1)
json.dump(out_json, open(os.path.join(HERE, 'literature_results.json'), 'w'), indent=1, default=float)

# ---------------------------------------------------------------- printed summary
print(f"Benchmarks: radiant heat {ours['radiant_heat_MWh_per_t']:.2f} MWh/t (Maporti {lit['Maporti2023']['radiant_heat_MWh_per_t']:.2f}, "
      f"Ghasemi Naraghi {lit['GhasemiNaraghi2026']['min_energy_electrified_MWh_per_t']:.2f} electric)")
print(f"Slate: conversion {s_lit['conversion']:.3f}, selectivity {s_lit['mass_selectivity']:.3f}, fresh ethane {s_lit['fresh_ethane_t_per_t']:.3f} t/t; "
      f"radiant +{out_json['slate']['radiant_per_t_increase_pct']:.1f}%, total +{out_json['slate']['Q_total_increase_pct']:.1f}%")
print(f"Tail gas LHV: ideal {LHV_tg(tg_ideal)/1e3:.2f} MW, literature {LHV_tg(tg_lit)/1e3:.2f} MW")
for k, v in out_json['steam_ratio'].items():
    print(f"S/E {k}: prep {v['Q_feed_prep_kW']/1e3:.2f} rad {v['Q_rad_kW']/1e3:.2f} MW; L {v['L_flux_m']:.0f}/{v['L_tau_m']:.0f} m; "
          f"XOT* {v['XOT_match_C']:.0f}; TMT {v['TMT_uniform_6x4_C']:.0f}/{v['best_profile_TMT_C']:.0f}; dP {v['dP_6x4_bar']:.2f}; "
          f"fired {v['fired_LHV_MW']:.2f}; grid {v['grid_MW']:.2f}; CI* {v['breakeven_CI']:.3f}; p* {v['breakeven_price']:.1f}")
for r in eta_rows: print(f"eta {r['eta']:.2f}: grid {r['grid_MW']:.2f} MW, CI* {r['breakeven_CI']:.3f}, p* {r['breakeven_price']:.1f}")
print(f"E2 pinch(30K): QHmin {e2['QH_min_kW']:.0f} kW, QCmin {e2['QC_min_kW']:.0f}, recovered {e2['recovered_kW']:.0f}, "
      f"pinch hot/cold {e2['T_pinch_hot_C']:.0f}/{e2['T_pinch_cold_C']:.0f} C; hot avail {e2['hot_available_kW']:.0f}, cold {e2['cold_demand_kW']:.0f}")
for r in e2_sweep: print(f"   dTmin {r['dTmin_K']}: QH {r['QH_min_kW']:.0f}, pinch {r['T_pinch_cold_C']:.0f}")
print(f"E2 grid {grid_e2/1e3:.2f} MW; CI* {CO2_fired/E_e2:.3f}; p* {gas_cost/E_e2:.1f}; site {out_json['E2_interchange']['site_basis']}")
print(f"   HTC: {htc}")
print(f"   quench-steam site {out_json['E2_interchange']['quench_steam_variant']}")
print(f"F2 ideal: {json.dumps(f2_ideal, default=float)}")
print(f"F2 lit: {json.dumps(f2_lit, default=float)}")
print(f"F0 lit: CO2 {f0_lit['CO2_t_y']:.0f}; reductions {out_json['F2_tailgas_fired']['CO2_reduction_pct_ideal']:.1f} / {out_json['F2_tailgas_fired']['CO2_reduction_pct_lit']:.1f} %")
print(f"H1: {json.dumps(out_json['H1_hybrid'], default=float)}")
print(f"switch CI E2 vs F2: {out_json['switch_CI_E2_vs_F2_t_MWh']:.3f}")
print(f"P1: {json.dumps(p1, default=float)}")
print(f"Storage: {json.dumps(stor, default=float)}")
print(f"vs F2: {json.dumps(out_json['vs_F2'], default=float)}")
print(f"H1 fuel-eq: {json.dumps(out_json['H1_hybrid']['ideal']['fuel_equivalent_basis'], default=float)}")
print(f"F0 check: {json.dumps(f0, default=float)}")
