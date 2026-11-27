"""
Extended sweeps for the manuscript figures.
Re-uses the model in furnace_model.py unchanged (it is executed in-process),
then writes extended_results.json next to this file.

Usage:  python extended_analysis.py
Needs:  cantera, CoolProp, numpy, scipy
"""
import os, sys, json, runpy, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = os.path.join(tempfile.mkdtemp(), 'base.json')
sys.argv = ['furnace_model.py', tmp]
ns = runpy.run_path(os.path.join(HERE, 'furnace_model.py'))
R, K, C = ns['R'], ns['K'], ns['C']
out = {}

# 1. Crossover temperature sweep: coil length from flux and from residence time
xs = np.arange(600, 801, 5)
rows = [ns['coil_lengths'](K(x)) for x in xs]
out['xot_sweep'] = dict(XOT_C=xs.tolist(), Q_rad_kW=[r[0] for r in rows],
                        L_flux_m=[r[1] for r in rows], L_tau_m=[r[2] for r in rows])

# 2. Number of parallel coils (fixed 24 legs of 13.14 m), uniform 100 kW/m2
arr = []
for n in [24, 12, 8, 6, 4, 3, 2]:
    a = ns['arrangement'](n, 24 // n)
    arr.append({k: a[k] for k in ['coils', 'legs_per_coil', 'G_kgm2s', 'h_out', 'TMT_uniform_flux_C', 'dP_bar', 'Mach_out']})
out['parallel_coils'] = arr

# 3. Axial TMT profiles for the 6 x 4 coil (same model as tmt_profile in the base model)
h_i, h_o = ns['h_i'], ns['h_o']
XOT, COT = C(ns['XOT']), C(ns['COT'])
qr, dTw = ns['q_in_per_q_out'], ns['dT_wall_per_kWm2']
def profile(qfun, n=800):
    z = (np.arange(n) + 0.5) / n
    q = qfun(z)
    Tg = XOT + (COT - XOT) * np.cumsum(q) / q.sum()
    tmt = Tg + q * qr * 1e3 / (h_i + (h_o - h_i) * z) + dTw * q
    return z, q, Tg, tmt
cases = {'Uniform 100': ns['linear'](100, 100),
         'Two zones 120/80': ns['steps'](120, 80),
         'Four zones 130/110/90/70': ns['steps'](130, 110, 90, 70),
         'Linear 130 to 70': ns['linear'](130, 70),
         'Linear 140 to 60': ns['linear'](140, 60)}
prof = {}
for name, f in cases.items():
    z, q, Tg, t = profile(f)
    prof[name] = dict(z=z[::8].tolist() + [z[-1]], q=q[::8].tolist() + [q[-1]], Tgas=Tg[::8].tolist() + [Tg[-1]],
                      TMT=t[::8].tolist() + [t[-1]], peak=float(t.max()), z_peak=float(z[t.argmax()]))
# cross-check against the base model
for k_base, k_new in [('uniform 100', 'Uniform 100'), ('2 zones 120/80', 'Two zones 120/80'),
                     ('4 zones 130/110/90/70', 'Four zones 130/110/90/70'), ('linear 130->70', 'Linear 130 to 70')]:
    assert abs(R['4_1_flux_profiles_6x4'][k_base]['peak_TMT_C'] - prof[k_new]['peak']) < 1e-6
out['tmt_profiles'] = prof

# 3b. Linear profiles of mean 100 kW/m2 with relative amplitude a: q = 100 (1 + a (1 - 2z))
amps = np.round(np.arange(0, 0.601, 0.02), 3)
pk = [float(profile(ns['linear'](100 * (1 + a), 100 * (1 - a)))[3].max()) for a in amps]
out['linear_amplitude_sweep'] = dict(amplitude=amps.tolist(), peak_TMT_C=pk,
                                     best_amplitude=float(amps[int(np.argmin(pk))]), best_peak_C=float(min(pk)))

# 4. Heat-integrated electric case: dilution steam raised in the quench exchanger
E = R['7_1_electric']['load_kW']; b9 = R['9_bonus']
CO2 = R['5_1_combustion']['CO2_t_y']; hours = 8000
grid_q = b9['grid_with_quench_steam_kW']
out['quench_integrated'] = dict(grid_kW=grid_q, MWh_y=grid_q * hours / 1e3,
                                breakeven_grid_intensity_t_MWh=CO2 / (grid_q * hours / 1e3),
                                breakeven_power_price_USD_MWh=R['8_comparison']['gas_cost_USD_y_HHV'] / (grid_q * hours / 1e3))


# 5. Compatibility of heat flux and residence time (ratio Phi = L_q / L_tau) and the 80 kW/m2 case
Ac, Do = ns['Ac'], ns['Do']
compat = []
for x in [600, 650, 700, 725, 750]:
    q, Lq, Lt = ns['coil_lengths'](K(x))
    z, Qv = ns['Qv_profile'](K(x)); I = np.trapezoid(1 / Qv, z)
    L80 = q / (80 * np.pi * Do)
    compat.append(dict(XOT_C=x, Q_rad_kW=q, L_q_m=Lq, tau_at_Lq_ms=1e3 * Lq * Ac * I, L_tau_m=Lt,
                       q_req_kWm2=q / (np.pi * Do * Lt), Phi=Lq / Lt, tau_at_80kWm2_ms=1e3 * L80 * Ac * I))
out['flux_tau_compatibility'] = compat

# 6. Radiating-source temperature for external (indirect) heating, effective exchange factor 0.65
SIG = 5.670e-8; F = 0.65
def T_src(q_kW, Tw_C): return ((Tw_C + 273.15) ** 4 + q_kW * 1e3 / (F * SIG)) ** 0.25 - 273.15
def q_max(Ts_C, Tw_C): return F * SIG * ((Ts_C + 273.15) ** 4 - (Tw_C + 273.15) ** 4) / 1e3
out['radiant_source'] = dict(F=F, T_source_for_100_at_wall_1050_C=T_src(100, 1050), T_source_for_100_at_wall_1000_C=T_src(100, 1000),
                             q_at_source_1100_wall_1050_kWm2=q_max(1100, 1050), q_at_source_1100_wall_1000_kWm2=q_max(1100, 1000))
# film coefficient needed at the outlet under uniform flux for a given TMT limit
wall = ns['dT_wall_per_kWm2'] * 100
out['h_required_uniform_flux'] = {f'TMT {lim} C': 100 * ns['q_in_per_q_out'] * 1e3 / (lim - 850 - wall) for lim in [1000, 1050]}

# 7. Reactor heating architectures (feed-preparation loads unchanged)
E = R['7_1_electric']['load_kW']; prep = E['boiler'] + E['sheathed_heaters'] + E['joule_preheat_450_650']
Qrad = R['3_energy_balance']['Q_rad_kW']; conv = R['7_1_electric']['efficiencies']['power_conv']
gas_cost = R['8_comparison']['gas_cost_USD_y_HHV']; fired = R['8_comparison']['fired_LHV_MW'] * 1e3
arch = []
for name, eta in [('Joule-heated tube wall (base)', 0.92), ('Joule-heated tube wall (96% tube x 99% busbar)', 0.96 * 0.99),
                  ('External SiC radiant elements', 0.92), ('Induction', 0.80)]:
    grid = (prep + Qrad / eta) / conv; MWh = grid * hours / 1e3
    arch.append(dict(architecture=name, eta_reactor=eta, grid_kW=grid, MWh_per_t=MWh / 50000,
                     breakeven_intensity=CO2 / MWh, breakeven_price=gas_cost / MWh))
out['architectures'] = arch
out['eta_reactor_to_match_fuel_input'] = Qrad / (fired * conv - prep)

# 8. Break-even power price with a carbon charge c and delivered intensity g: p* = [C_gas + c (m_fired - E g)] / E
def p_star(MWh, c, g): return (gas_cost + c * (CO2 - MWh * g)) / MWh
MWh0 = E['MWh_y']
out['breakeven_price_with_carbon'] = {f'c={c}, g={g}': p_star(MWh0, c, g) for c in [0, 50, 100] for g in [0.0, 0.05]}

# 9. Electrical infrastructure screening
S_MVA = E['grid'] / 1e3 / 0.95
out['electrical_supply'] = dict(apparent_MVA_pf095=S_MVA, bus_current_13p8kV_A=S_MVA * 1e6 / (np.sqrt(3) * 13.8e3),
                                electrode_boiler_current_6p6kV_A=E['boiler'] * 1e3 / (np.sqrt(3) * 6.6e3))

json.dump(out, open(os.path.join(HERE, 'extended_results.json'), 'w'), indent=1, default=float)
print('XOT match', R['4_coil_sizing']['XOT_where_lengths_match_C'])
for a in arr: print(a)
for k, v in prof.items(): print(k, round(v['peak'], 1), round(v['z_peak'], 3))
print(out['quench_integrated']); [print(k, out[k]) for k in ['flux_tau_compatibility','radiant_source','h_required_uniform_flux','architectures','eta_reactor_to_match_fuel_input','breakeven_price_with_carbon','electrical_supply']]; print(out['linear_amplitude_sweep']['best_amplitude'], out['linear_amplitude_sweep']['best_peak_C'])
