"""
Screening comparison with the Electric Reaction-Tower (ERT) case of Rodriguez-Gil and Agrawal (2025),
Cell Rep. Phys. Sci. 6, 102771. Their single-tube case: 50 m, 90.12 mm ID, 5.74 mm wall, 2 bar,
pure ethane 120 g/s at 500 C, 10 kW/m, outlet ~880 C and ~40 wt% ethylene (Sundaram-Froment kinetics).
Here the same tube is re-evaluated with this paper's ideal-selectivity thermochemistry and screening
correlations so that the two designs can be put on one basis. Writes ert_results.json.
Also screens the internally heated cracker (IHC) of Rodriguez-Gil and Agrawal (2024), Syst. Control Trans. 3, 883-891.
Usage: python ert_comparison.py      Needs: cantera, numpy
"""
import os, json
import numpy as np
import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, 'furnace_model_results.json')))
X = json.load(open(os.path.join(HERE, 'extended_results.json')))
g = ct.Solution('gri30.yaml'); g.transport_model = 'mixture-averaged'
MW = {s: g.molecular_weights[g.species_index(s)] for s in ['C2H6', 'C2H4', 'H2', 'H2O']}
R_GAS = 8314.46

# ---- ERT case as published
L, D, WALL, P, m_feed, QL, Tin, Tout, wC2H4 = 50.0, 0.09012, 0.00574, 2.0e5, 0.120, 10e3, 773.15, 1153.15, 0.40
A = np.pi / 4 * D ** 2
P_tube = QL * L                                   # W
m_eth = wC2H4 * m_feed                            # kg/s ethylene per tube

def H(comp_kmol_s, T, p=P):
    n = sum(comp_kmol_s.values()); g.TPX = T, p, comp_kmol_s
    return n * g.enthalpy_mole                    # W

# Ideal-selectivity equivalent of the published outlet (40 wt% ethylene, rest ethane + H2)
n_e = m_eth / MW['C2H4']; n0 = m_feed / MW['C2H6']
feed = {'C2H6': n0}; prod = {'C2H6': n0 - n_e, 'C2H4': n_e, 'H2': n_e}
Q_ideal = H(prod, Tout) - H(feed, Tin)            # W, reactor duty with 100 % selectivity
Q_pre = H(feed, Tin) - H(feed, 298.15)            # W, ethane preheat 25 -> 500 C
X_equiv = n_e / n0

# Approximate residence time with T and conversion linear in length (same method as the paper)
z = np.linspace(0, 1, 401); T = Tin + (Tout - Tin) * z
Qv = (n0 + n_e * z) * R_GAS * T / P
tau = A * L * np.trapezoid(1 / Qv, z)

# If the same tube were heated through its wall (the paper's fixed-length comparison case)
G = m_feed / A
g.TPX = Tout, P, prod
mu, k, cp = g.viscosity, g.thermal_conductivity, g.cp_mass
Re = G * D / mu; Pr = mu * cp / k; h = 0.023 * Re ** 0.8 * Pr ** 0.4 * k / D
q_in = QL / (np.pi * D)                           # W/m2 on the inside wall
ro, ri = D / 2 + WALL, D / 2
q_out = QL / (np.pi * 2 * ro)
dT_film = q_in / h; dT_wall = q_out * ro * np.log(ro / ri) / 25.0   # 25 W/m K (their alloy value)
q_vol = QL / A

# Scale-up to this paper's ethylene rate
m_target = B['1_2_basis_and_balance']['mC2H4_kgh'] / 3600
n_tubes = m_target / m_eth

# This paper's coil on the same metrics
Qrad = B['3_energy_balance']['Q_rad_kW'] * 1e3
Lc = B['4_coil_sizing']['L_from_flux_m']; Ac = B['4_coil_sizing']['Ac_m2']
coil = B['4_2_coil_design']
ours = dict(Q_rad_MW=Qrad / 1e6, GJ_per_t=Qrad / m_target / 1e6, L_total_m=Lc, n_paths=6, ID_mm=100, wall_mm=10,
            G=coil['G_kgm2s'], q_inside_kWm2=120.0, q_vol_MWm3=Qrad / (Lc * Ac) / 1e6,
            QL_kW_per_m=Qrad / Lc / 1e3, tau_ms=B['4_coil_sizing']['tau_at_L_flux_ms'], h_out=coil['h_out'])
# radiant duty if our feed entered at 500 C (redistribution only)
import runpy, sys, tempfile
sys.argv = ['furnace_model.py', os.path.join(tempfile.mkdtemp(), 'x.json')]
ns = runpy.run_path(os.path.join(HERE, 'furnace_model.py'))
ours['Q_rad_from_500C_MW'] = ns['coil_lengths'](773.15)[0] / 1e3
ours['GJ_per_t_from_500C'] = ours['Q_rad_from_500C_MW'] * 1e6 / m_target / 1e6

ert = dict(P_tube_kW=P_tube / 1e3, m_C2H4_kgh=m_eth * 3600, GJ_per_t_published=P_tube / m_eth / 1e6,
           Q_ideal_kW=Q_ideal / 1e3, GJ_per_t_ideal=Q_ideal / m_eth / 1e6, X_equiv=X_equiv,
           Q_preheat_25_500_kW=Q_pre / 1e3, tau_ms=tau * 1e3, G=G, Re=Re, h_out=h, q_inside_kWm2=q_in / 1e3,
           dT_film=dT_film, dT_wall=dT_wall, TMT_if_wall_heated_C=Tout - 273.15 + dT_film + dT_wall,
           q_vol_MWm3=q_vol / 1e6, n_tubes_for_50kt=n_tubes, L_total_m=n_tubes * L,
           P_total_MW=n_tubes * P_tube / 1e6, preheat_total_MW=n_tubes * Q_pre / 1e6,
           feed_total_kgh=n_tubes * m_feed * 3600)
out = dict(ert=ert, ours=ours)

# ---- Internally heated cracker (IHC) screening, after Rodriguez-Gil and Agrawal (2024), FOCAPD/LAPSE:2024.1621.
# Their lab case: 0.5 m x 35 mm ID tube, 2 bar, pure ethane at 350 C, all heated surfaces isothermal at 1100 C,
# optimal EHC feed ~1.2 g/s. Here: (a) Reynolds number of that case, (b) element area needed if this paper's
# radiant duty were delivered by internal elements held at 1100 C, using this paper's inside coefficients.
D_lab = 0.035; m_lab = 1.2e-3
g.TPX = 973.15, 2e5, {'C2H6': 0.8, 'C2H4': 0.1, 'H2': 0.1}
Re_lab = m_lab / (np.pi / 4 * D_lab ** 2) * D_lab / g.viscosity
coilB = B['4_2_coil_design']; h_i, h_o = coilB['h_in'], coilB['h_out']
zz = (np.arange(2000) + 0.5) / 2000
Tg = 650 + 200 * zz; h = h_i + (h_o - h_i) * zz
q_el = h * (1100 - Tg)                              # W/m2 element surface, local
A_el = np.sum(Qrad / len(zz) / q_el)                # m2, uniform heat release along the coil
A_wall_in = np.pi * 0.100 * Lc                      # inside wall area of the coil
# wall-heated comparison: allowable film dT at outlet with a 1000 C TMT and 55 K wall drop
film_wall_out = 1000 - 850 - B['4_1_TMT']['wall_dT_at_100_K']
rods = {f'{int(d*1e3)} mm': A_el / (Lc * np.pi * d) for d in [0.016, 0.020, 0.025]}
# metal-life rule of thumb quoted by the IHC paper: +20 K halves life
uniform = X['tmt_profiles']['Uniform 100']['peak']; best = X['linear_amplitude_sweep']['best_peak_C']
out['ihc'] = dict(Re_lab=Re_lab, q_el_in_kWm2=q_el[0] / 1e3, q_el_out_kWm2=q_el[-1] / 1e3,
                  A_element_m2=A_el, A_wall_inside_m2=A_wall_in, film_allow_wall_out_K=film_wall_out,
                  film_allow_element_out_K=1100 - 850, rods_per_path_length=rods,
                  life_factor_profiled_vs_uniform=2 ** ((uniform - best) / 20), dTMT=uniform - best)
json.dump(out, open(os.path.join(HERE, 'ert_results.json'), 'w'), indent=1, default=float)
for k, v in out.items():
    print(k); [print(f'  {a}: {b}') for a, b in v.items()]
