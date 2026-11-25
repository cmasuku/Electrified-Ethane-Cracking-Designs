"""
Draws the manuscript figures from furnace_model_results.json and extended_results.json.
Usage:  python make_figures.py      (writes PDF figures to ../figures)
Needs:  numpy, matplotlib
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'furnace_model_results.json')))
X = json.load(open(os.path.join(HERE, 'extended_results.json')))

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = '#257de0', '#f47230', '#00b894', '#f2a600', '#ee7fac', '#008a17', '#6a4ce2'
INK, INK2, GRID, LOSS = '#191919', '#666666', '#e3e3e3', '#b4b4b4'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
W1, W2 = 3.5, 7.2   # single / double column width (in)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf')); fig.savefig(os.path.join(FIG, name + '.png'), dpi=300); plt.close(fig)


def panel(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9, color=INK)

# ------------------------------------------------------------------ Fig 5: where the input energy goes
e3, c52, e7 = R['3_energy_balance'], R['5_2_convection'], R['7_1_electric']['load_kW']
fired = R['8_comparison']['fired_LHV_MW'] * 1e3
rad, prep = e3['Q_rad_kW'], e3['Q_feed_prep_kW']
bars = {'Gas-fired furnace\n(natural gas, LHV)': [rad, prep, c52['Q_surplus_kW'], fired - rad - prep - c52['Q_surplus_kW']],
        'Electric furnace\n(grid electricity)': [rad, prep, 0.0, e7['grid'] - rad - prep]}
cats = ['Radiant coil (cracking)', 'Feed preparation', 'Exported HP steam', 'Losses']
cols = [BLUE, ORANGE, AQUA, LOSS]
fig, ax = plt.subplots(figsize=(W2, 1.9))
for i, (lab, vals) in enumerate(bars.items()):
    left = 0
    for v, c, cat in zip(vals, cols, cats):
        if v <= 0: continue
        ax.barh(i, v / 1e3, left=left / 1e3, height=0.55, color=c, edgecolor='white', linewidth=1.5, label=cat if i == 0 else None)
        if v > 1200:
            ax.text((left + v / 2) / 1e3, i, f'{v/1e3:.1f}', ha='center', va='center', fontsize=7.5,
                    color='white' if c in (BLUE, GREEN) else INK)
        left += v
    ax.text(left / 1e3 + 0.3, i, f'{left/1e3:.1f} MW', va='center', fontsize=8, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels(list(bars.keys())); ax.invert_yaxis()
ax.set_xlabel('Energy input and its destination (MW)'); ax.set_xlim(0, 32); ax.grid(axis='y', visible=False)
ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.02), handlelength=1.2)
save(fig, 'fig5_energy_destination')

# ------------------------------------------------------------------ Fig 2: coil length vs crossover temperature
s = X['xot_sweep']; xm = R['4_coil_sizing']['XOT_where_lengths_match_C']
fig, ax = plt.subplots(figsize=(W1, 2.4))
ax.plot(s['XOT_C'], s['L_flux_m'], color=BLUE, label='Heat flux limit (100 kW m$^{-2}$)')
ax.plot(s['XOT_C'], s['L_tau_m'], color=ORANGE, label='Residence time (300 ms)')
Lm = np.interp(xm, s['XOT_C'], s['L_flux_m'])
ax.plot([xm], [Lm], 'o', ms=6, color=INK, mec='white', mew=1.5, zorder=5)
ax.annotate(f'Both met at {xm:.0f} °C\n({Lm:.0f} m)', (xm, Lm), xytext=(690, 245), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
ax.axvline(650, color=INK2, lw=0.8, ls=':'); ax.text(654, 337, 'Base case\n650 °C', fontsize=7, color=INK2, va='top')
ax.set_xlabel('Crossover temperature (°C)'); ax.set_ylabel('Total tube length (m)'); ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1)
ax.set_ylim(230, 340)
save(fig, 'fig2_coil_length_vs_xot')

# ------------------------------------------------------------------ Fig 3: parallel coils vs TMT and pressure drop
pc = X['parallel_coils']
G = [a['G_kgm2s'] for a in pc]; T = [a['TMT_uniform_flux_C'] for a in pc]; dP = [a['dP_bar'] for a in pc]
lab = [f"{a['coils']}×{a['legs_per_coil']}" for a in pc]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
for ax, y, yl in [(axs[0], T, 'Outlet tube-metal temperature (°C)'), (axs[1], dP, 'Coil pressure drop (bar)')]:
    ax.plot(G, y, '-', color=BLUE, marker='o', ms=4.5, mec='white', mew=1)
    off = (4, 4) if ax is axs[0] else (6, -10)
    for g, v, l in zip(G, y, lab):
        ax.annotate(l, (g, v), xytext=off, textcoords='offset points', fontsize=6.8, color=INK2)
    i6 = lab.index('6×4'); ax.plot([G[i6]], [y[i6]], 'o', ms=7, color=ORANGE, mec='white', mew=1.2, zorder=5)
    ax.set_xlabel('Mass flux, $G$ (kg m$^{-2}$ s$^{-1}$)'); ax.set_ylabel(yl)
axs[0].axhline(1000, color=INK, lw=0.9, ls='--'); axs[0].text(25, 990, 'Design limit 1000 °C', fontsize=7, color=INK, va='top')
axs[1].set_yscale('log'); axs[1].set_ylim(4e-3, 40); axs[1].axhline(1.0, color=INK, lw=0.9, ls='--')
axs[1].text(285, 0.75, 'About 1 bar\n(coil inlet ≈ 3.5 bara)', fontsize=7, color=INK, ha='right', va='top')
panel(axs[0], '(a)'); panel(axs[1], '(b)')
fig.text(0.5, -0.04, 'Labels give coils × legs per coil (24 legs of 13.1 m in all cases). Orange: selected design.',
         ha='center', fontsize=7, color=INK2)
fig.tight_layout(w_pad=2.5)
save(fig, 'fig3_parallel_coils_tmt_dp')

# ------------------------------------------------------------------ Fig 4: flux profiles and TMT
P = X['tmt_profiles']; A = X['linear_amplitude_sweep']
names = ['Uniform 100', 'Two zones 120/80', 'Four zones 130/110/90/70', 'Linear 130 to 70']
labels = ['Uniform, 100', 'Two zones, 120/80', 'Four zones, 130→70', 'Linear, 130→70']
cl = [INK2, ORANGE, AQUA, BLUE]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
for n, l, c in zip(names, labels, cl):
    d = P[n]
    axs[0].plot(d['z'], d['q'], color=c, label=l, drawstyle='default')
    axs[1].plot(d['z'], d['TMT'], color=c, label=l)
axs[1].plot(P['Uniform 100']['z'], P['Uniform 100']['Tgas'], color=INK2, lw=1, ls=':', label='Gas (uniform case)')
axs[0].set_xlabel('Fractional coil length, $z$'); axs[0].set_ylabel('Outside heat flux (kW m$^{-2}$)'); axs[0].set_ylim(60, 140)
axs[1].set_xlabel('Fractional coil length, $z$'); axs[1].set_ylabel('Temperature (°C)'); axs[1].set_ylim(640, 1080)
axs[1].axhline(1000, color=INK, lw=0.9, ls='--')
axs[1].text(0.55, 700, 'Gas (uniform case)', fontsize=6.8, color=INK2, rotation=0)
axs[0].legend(loc='lower left', fontsize=6.8)
axs[2].plot(np.array(A['amplitude']) * 100, A['peak_TMT_C'], color=BLUE)
axs[2].axhline(1000, color=INK, lw=0.9, ls='--')
ok = np.array(A['peak_TMT_C']) < 1000; amp = np.array(A['amplitude']) * 100
axs[2].axvspan(amp[ok].min(), amp[ok].max(), color=BLUE, alpha=0.08, lw=0)
axs[2].plot([A['best_amplitude'] * 100], [A['best_peak_C']], 'o', ms=6, color=BLUE, mec='white', mew=1.2)
axs[2].annotate(f"Minimum {A['best_peak_C']:.0f} °C\nat ±{A['best_amplitude']*100:.0f}%", (A['best_amplitude'] * 100, A['best_peak_C']),
                xytext=(5, 1022), fontsize=7, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
axs[2].set_xlabel('Linear flux amplitude, ±% of mean'); axs[2].set_ylabel('Peak tube-metal temperature (°C)')
for a, s_ in zip(axs, '(a) (b) (c)'.split()): panel(a, s_)
fig.tight_layout(w_pad=1.6)
save(fig, 'fig4_flux_profiles_tmt')

# ------------------------------------------------------------------ Fig 6: economics and emissions break-even
cmp_ = R['8_comparison']; Q = X['quench_integrated']
MWh = R['7_1_electric']['load_kW']['MWh_y']; MWhq = Q['MWh_y']
gas = cmp_['gas_cost_USD_y_HHV']; CO2 = R['5_1_combustion']['CO2_t_y']
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
p = np.linspace(0, 100, 201)
axs[0].plot(p, MWh * p / 1e6, color=BLUE, label='Electric, base case')
axs[0].plot(p, MWhq * p / 1e6, color=BLUE, ls='--', label='Electric, steam from quench')
for cp, ls, txt in [(0, '-', 'Gas, no CO$_2$ price'), (100, '-.', 'Gas + 100 \\$/t CO$_2$'), (234.4, ':', 'Gas + 234 \\$/t CO$_2$')]:
    v = (gas + cp * CO2) / 1e6
    axs[0].axhline(v, color=ORANGE, ls=ls, lw=1.3); axs[0].text(101, v, txt, fontsize=6.8, va='center', color=INK)
pb = cmp_['breakeven_power_price_USD_MWh']
axs[0].plot([pb], [gas / 1e6], 'o', ms=5.5, color=INK, mec='white', mew=1.2, zorder=5)
axs[0].annotate(f'{pb:.1f} \\$/MWh', (pb, gas / 1e6), xytext=(22, 1.0), fontsize=7, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
axs[0].plot([70], [MWh * 70 / 1e6], 's', ms=5, color=BLUE, mec='white', mew=1)
axs[0].annotate('Assumed\n70 \\$/MWh', (70, MWh * 70 / 1e6), xytext=(82, 8.6), fontsize=7, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
axs[0].set_xlabel('Electricity price (\\$ MWh$^{-1}$)'); axs[0].set_ylabel('Annual energy cost (M\\$ y$^{-1}$)')
axs[0].set_xlim(0, 100); axs[0].set_ylim(0, 20); axs[0].legend(loc='upper left', fontsize=6.8)
gi = np.linspace(0, 0.8, 161)
axs[1].plot(gi, MWh * gi / 1e3, color=BLUE, label='Electric, base case')
axs[1].plot(gi, MWhq * gi / 1e3, color=BLUE, ls='--', label='Electric, steam from quench')
axs[1].axhline(CO2 / 1e3, color=ORANGE, label='Gas-fired, direct CO$_2$')
for b, xy in [(cmp_['breakeven_grid_intensity_t_MWh'], (0.12, 72)), (Q['breakeven_grid_intensity_t_MWh'], (0.34, 22))]:
    axs[1].plot([b], [CO2 / 1e3], 'o', ms=5.5, color=INK, mec='white', mew=1.2, zorder=5)
    axs[1].annotate(f'{b:.2f}', (b, CO2 / 1e3), xytext=xy, fontsize=7, ha='center',
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
axs[1].axvline(0.37, color=INK2, lw=0.8, ls=':'); axs[1].text(0.38, 5, 'US average\n≈0.37 (assumed)', fontsize=6.8, color=INK2)
axs[1].set_xlabel('Grid carbon intensity (t CO$_2$ MWh$^{-1}$)'); axs[1].set_ylabel('CO$_2$ emissions (kt y$^{-1}$)')
axs[1].set_xlim(0, 0.8); axs[1].set_ylim(0, 140); axs[1].legend(loc='upper left', fontsize=6.8)
panel(axs[0], '(a)'); panel(axs[1], '(b)')
fig.tight_layout(w_pad=7)
save(fig, 'fig6_breakeven')

# ------------------------------------------------------------------ Fig 7: feed-effluent interchange and break-even by design
L = json.load(open(os.path.join(HERE, 'literature_results.json')))
e2 = L['E2_interchange']; cc = e2['composite']
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8))
ax = axs[0]
Hh = np.array(cc['H_hot_kW']) / 1e3; Hc = np.array(cc['H_cold_kW']) / 1e3
qh = e2['pinch_30K']['QH_min_kW'] / 1e3
ax.plot(Hh, cc['T_hot_C'], color=ORANGE, label='Cracked gas, 850 → 120 °C')
ax.plot(Hc + 0.0, cc['T_cold_C'], color=BLUE, label='Feed preparation (water, ethane, mixed feed)')
ax.axvspan(Hh.max(), Hc.max(), color=GRID, alpha=0.9, lw=0)
ax.text((Hh.max() + Hc.max()) / 2, 300, f'Electric trim\n{qh:.2f} MW', fontsize=6.8, ha='center', color=INK, rotation=90, va='center')
ax.set_xlabel('Heat flow (MW)'); ax.set_ylabel('Temperature (°C)')
ax.set_xlim(0, 11.5); ax.set_ylim(0, 900); ax.legend(loc='upper left', fontsize=6.8)
ax = axs[1]
cmpL = R['8_comparison']; CO2f = R['5_1_combustion']['CO2_t_y']; hrs = 8000
E1q = R['9_bonus']['grid_with_quench_steam_kW']
names = ['E1 electric (base)', 'E1 at steam ratio 0.3', 'E1q quench steam', 'E2 interchange', 'H1 hybrid']
plant = [cmpL['breakeven_grid_intensity_t_MWh'], L['steam_ratio']['0.3']['breakeven_CI'], CO2f / (E1q * hrs / 1e3),
         e2['breakeven_CI_fuel_basis'], L['H1_hybrid']['ideal']['breakeven_CI']]
cons = [cmpL['breakeven_grid_intensity_t_MWh'], L['steam_ratio']['0.3']['breakeven_CI'],
        e2['quench_steam_variant']['site_basis']['breakeven_CI'], e2['site_basis']['breakeven_CI'],
        L['H1_hybrid']['ideal']['fuel_equivalent_basis']['breakeven_CI']]
y = np.arange(len(names))[::-1]
ax.barh(y + 0.19, plant, height=0.36, color=BLUE, edgecolor='white', linewidth=1, label='Plant boundary only')
ax.barh(y - 0.19, cons, height=0.36, color=ORANGE, edgecolor='white', linewidth=1, label='Steam and H$_2$ valued as fuel')
for yi, a, b in zip(y, plant, cons):
    ax.text(a + 0.005, yi + 0.19, f'{a:.2f}', va='center', fontsize=6.8, color=INK)
    ax.text(b + 0.005, yi - 0.19, f'{b:.2f}', va='center', fontsize=6.8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(names); ax.grid(axis='y', visible=False)
ax.set_xlabel('Break-even grid intensity vs gas-fired F1 (t CO$_2$ MWh$^{-1}$)'); ax.set_xlim(0, 0.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6.8)
panel(axs[0], '(a)'); panel(axs[1], '(b)')
fig.tight_layout(w_pad=3)
save(fig, 'fig7_new_designs')
print('figures written to', os.path.abspath(FIG))
