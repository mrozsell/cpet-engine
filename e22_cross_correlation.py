"""
Engine E22 — Cross-Engine Correlation Analysis
═══════════════════════════════════════════════
Zbiera wyniki z E01–E21 + surowe dane CSV, oblicza cross-korelacje.

FAZA 1: Triple Drift, SC×RER, O₂P Stability, Economy Durability
FAZA 2: HRR↔τ_off Dissociation, Breathing Pattern per Domain
FAZA 3: SmO₂ Kinetics (when NIRS available)
FAZA 4: τ↔VE/VCO₂ Index
FAZA 5: Fingerprint Radar, Limitation Triangle, Trainability, Performance Model
"""

import numpy as np
import pandas as pd


class Engine_E22_CrossCorrelation:

    @staticmethod
    def _stage_stats(df, t_start, t_end, col, warmup_s=180, tail_s=60):
        """Extract early (after warmup) and late (last tail_s) means for a column within a stage."""
        mask = (df['Time_s'] >= t_start) & (df['Time_s'] <= t_end)
        sub = df.loc[mask, ['Time_s', col]].dropna()
        if len(sub) < 10:
            return None, None, None
        early = sub[sub['Time_s'] <= t_start + warmup_s][col]
        late = sub[sub['Time_s'] >= t_end - tail_s][col]
        mean_all = sub[col].mean()
        early_m = early.mean() if len(early) > 3 else None
        late_m = late.mean() if len(late) > 3 else None
        return early_m, late_m, mean_all

    @staticmethod
    def run(results: dict, ctx: dict) -> dict:
        df = ctx.get('_df_processed')
        if df is None or df.empty:
            return {'status': 'NO_DATA'}

        e01 = results.get('E01', {})
        e02 = results.get('E02', {})
        e03 = results.get('E03', {})  # VentSlope
        e05 = results.get('E05', {})  # O2Pulse
        e07 = results.get('E07', {})  # BreathingPattern
        e08 = results.get('E08', {})  # CardioHRR
        e12 = results.get('E12', {})  # NIRS
        e13 = results.get('E13', {})  # Drift
        e14 = results.get('E14', {})  # Kinetics
        e21 = results.get('E21', {})  # KineticPhenotype

        stages = e14.get('stages', [])
        off_ks = e14.get('off_kinetics', [])
        kin_profile = e21.get('kinetic_profile', {})

        out = {'status': 'OK', 'stage_cross': [], 'composites': {}}

        # ═══════════════════════════════════════════
        # FAZA 1: Per-stage cross metrics
        # ═══════════════════════════════════════════

        for s in stages:
            t0 = s.get('t_start', 0)
            t1 = s.get('t_end', 0)
            snum = s.get('stage_num', 0)
            dom = s.get('domain', '')
            dur = t1 - t0
            sc_pct = s.get('sc_pct', 0)

            sc = {
                'stage_num': snum,
                'domain': dom,
                'speed_kmh': s.get('speed_kmh'),
                'duration_s': dur,
            }

            # Adaptive warmup: use 50% of stage or 180s, whichever is less
            warmup = min(180, dur * 0.5)
            tail = min(60, dur * 0.2)

            # ── 1.1 Triple Drift ──
            hr_early, hr_late, hr_mean = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'HR_bpm', warmup, tail)
            vo2_early, vo2_late, vo2_mean = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'VO2_ml_min', warmup, tail)
            o2p_early, o2p_late, o2p_mean = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'O2_Pulse', warmup, tail)

            if hr_early and hr_late:
                sc['hr_drift_bpm'] = round(hr_late - hr_early, 1)
                sc['hr_drift_pct'] = round((hr_late - hr_early) / hr_early * 100, 2) if hr_early > 0 else None
            if vo2_early and vo2_late:
                sc['vo2_drift_mlmin'] = round(vo2_late - vo2_early, 0)
                sc['vo2_drift_pct'] = round((vo2_late - vo2_early) / vo2_early * 100, 2) if vo2_early > 0 else None
            if o2p_early and o2p_late:
                sc['o2p_drift_ml'] = round(o2p_late - o2p_early, 2)
                sc['o2p_drift_pct'] = round((o2p_late - o2p_early) / o2p_early * 100, 2) if o2p_early > 0 else None

            # Drift pattern classification
            hr_d = sc.get('hr_drift_pct', 0) or 0
            vo2_d = sc.get('vo2_drift_pct', 0) or 0
            o2p_d = sc.get('o2p_drift_pct', 0) or 0

            if abs(hr_d) > 3 and abs(vo2_d) > 3 and o2p_d < -2:
                sc['drift_pattern'] = 'CARDIOVASCULAR'
                sc['drift_pattern_pl'] = 'Drift sercowo-naczyniowy (↑HR ↑VO₂ ↓O₂P)'
            elif abs(vo2_d) > abs(hr_d) * 1.5 and o2p_d > -1:
                sc['drift_pattern'] = 'MUSCLE_RECRUITMENT'
                sc['drift_pattern_pl'] = 'Rekrutacja mięśniowa (VO₂↑↑, HR→, O₂P→)'
            elif abs(hr_d) > 2 and abs(vo2_d) > 2 and abs(o2p_d) < 2:
                sc['drift_pattern'] = 'METABOLIC_SC'
                sc['drift_pattern_pl'] = 'Metaboliczny SC (↑HR ↑VO₂, O₂P stabilny)'
            elif abs(hr_d) < 2 and abs(vo2_d) < 2:
                sc['drift_pattern'] = 'STABLE'
                sc['drift_pattern_pl'] = 'Stabilny steady-state'
            else:
                sc['drift_pattern'] = 'MIXED'
                sc['drift_pattern_pl'] = 'Wzorzec mieszany'

            # Drift dissociation index
            if sc_pct > 0.5 and abs(hr_d) > 0:
                sc['drift_dissociation'] = round(abs(hr_d) / sc_pct, 2)
            else:
                sc['drift_dissociation'] = None

            # ── 1.2 SC × RER ──
            rer_early, rer_late, rer_mean = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'RER', warmup, tail)
            if rer_early and rer_late:
                sc['rer_drift'] = round(rer_late - rer_early, 4)
                sc['rer_mean'] = round(rer_mean, 3) if rer_mean else None
            if sc_pct is not None and rer_mean:
                sc['metabolic_efficiency_idx'] = round(sc_pct * rer_mean, 2)

            # ── 1.3 O₂ Pulse Stability Index ──
            if o2p_early and o2p_late and o2p_early > 0:
                sc['o2p_stability'] = round(o2p_late / o2p_early, 4)
                if sc['o2p_stability'] >= 0.95:
                    sc['o2p_stability_class'] = 'STABLE'
                elif sc['o2p_stability'] >= 0.90:
                    sc['o2p_stability_class'] = 'MILD_DRIFT'
                else:
                    sc['o2p_stability_class'] = 'CARDIAC_DRIFT'
                sc['o2p_early'] = round(o2p_early, 1)
                sc['o2p_late'] = round(o2p_late, 1)

            # ── 1.4 Economy Durability ──
            vo2kg_early, vo2kg_late, vo2kg_mean = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, "V'O2/kg", warmup, tail)
            if vo2kg_early and vo2kg_late and vo2kg_late > 0:
                sc['economy_durability'] = round(vo2kg_early / vo2kg_late, 4)
                sc['vo2kg_early'] = round(vo2kg_early, 1)
                sc['vo2kg_late'] = round(vo2kg_late, 1)
                sc['economy_loss_pct'] = round((1 - sc['economy_durability']) * 100, 1)

            # ═══════════════════════════════════════════
            # FAZA 2: Breathing Pattern per Domain
            # ═══════════════════════════════════════════

            bf_e, bf_l, bf_m = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'BF_1_min', warmup, tail)
            vt_e, vt_l, vt_m = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'VT_L', warmup, tail)
            ve_e, ve_l, ve_m = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'VE_L_min', warmup, tail)

            if bf_m and vt_m:
                sc['bf_mean'] = round(bf_m, 1)
                sc['vt_mean_L'] = round(vt_m, 2)
                sc['ve_mean'] = round(ve_m, 1) if ve_m else None
                sc['bf_vt_ratio'] = round(bf_m / (vt_m * 60) * 100, 1) if vt_m > 0 else None

                # VT plateau detection (compare with previous stage VT)
                if vt_e and vt_l:
                    sc['vt_drift_L'] = round(vt_l - vt_e, 3)
                    sc['bf_drift'] = round(bf_l - bf_e, 1) if bf_e and bf_l else None

            # VE/VCO₂ per domain (FAZA 4 — easy to compute here)
            vco2_e, vco2_l, vco2_m = Engine_E22_CrossCorrelation._stage_stats(df, t0, t1, 'VCO2_ml_min', warmup, tail)
            if ve_m and vco2_m and vco2_m > 0:
                sc['ve_vco2_ratio'] = round(ve_m * 1000 / vco2_m, 1)  # VE in L, VCO2 in ml → *1000

            out['stage_cross'].append(sc)

        # ═══════════════════════════════════════════
        # FAZA 2: Recovery Integration (τ_off ↔ HRR)
        # ═══════════════════════════════════════════

        hrr_1min = None
        hrr_3min = None

        # Try E08 first
        if e08 and e08.get('status') not in (None, 'ERROR', 'LIMITED'):
            hrr_1min = e08.get('hrr_1min') or e08.get('hrr1')
            hrr_3min = e08.get('hrr_3min') or e08.get('hrr3')

        # Fallback: compute from raw data
        if hrr_1min is None:
            rec = df[df['Faza'] == 'Regeneracja']
            if len(rec) > 5 and 'HR_bpm' in df.columns:
                t_rec = rec['Time_s'].min()
                ex = df[(df['Faza'] == 'Wysiłek') | (df['Faza'] == 'Wy')]
                if len(ex) > 0:
                    hr_peak = df[df['Time_s'] < t_rec]['HR_bpm'].max()
                    hr_at_60 = rec[rec['Time_s'] <= t_rec + 60]['HR_bpm'].min()
                    hr_at_180 = rec[rec['Time_s'] <= t_rec + 180]['HR_bpm'].min()
                    if hr_peak and hr_at_60:
                        hrr_1min = round(hr_peak - hr_at_60, 0)
                    if hr_peak and hr_at_180:
                        hrr_3min = round(hr_peak - hr_at_180, 0)

        out['hrr'] = {
            'hrr_1min_bpm': hrr_1min,
            'hrr_3min_bpm': hrr_3min,
        }

        # Find valid off-kinetics τ_off
        valid_off = [o for o in off_ks if o.get('status') == 'OK']
        tau_off_mean = None
        if valid_off:
            taus_off = [o.get('tau_off_s') for o in valid_off if o.get('tau_off_s')]
            if taus_off:
                tau_off_mean = round(np.mean(taus_off), 1)

        out['hrr']['tau_off_mean_s'] = tau_off_mean

        # Recovery dissociation
        if hrr_1min and tau_off_mean and tau_off_mean > 0:
            rd = round(hrr_1min / tau_off_mean, 2)
            out['hrr']['recovery_dissociation'] = rd
            if rd > 2.0:
                out['hrr']['recovery_type'] = 'AUTONOMIC_FAST'
                out['hrr']['recovery_type_pl'] = 'Autonomiczna dominacja (HRR >> τ_off)'
            elif rd < 0.5:
                out['hrr']['recovery_type'] = 'METABOLIC_FAST'
                out['hrr']['recovery_type_pl'] = 'Metaboliczna dominacja (τ_off >> HRR)'
            elif hrr_1min >= 20 and tau_off_mean <= 60:
                out['hrr']['recovery_type'] = 'BALANCED_FAST'
                out['hrr']['recovery_type_pl'] = 'Zbalansowana szybka recovery'
            elif hrr_1min < 12 and tau_off_mean > 90:
                out['hrr']['recovery_type'] = 'DUAL_SLOW'
                out['hrr']['recovery_type_pl'] = 'Podwójnie wolna recovery ⚠️'
            else:
                out['hrr']['recovery_type'] = 'BALANCED'
                out['hrr']['recovery_type_pl'] = 'Zbalansowana recovery'

        # HRR interpretation
        if hrr_1min is not None:
            if hrr_1min >= 30:
                out['hrr']['hrr_class'] = 'EXCELLENT'
            elif hrr_1min >= 20:
                out['hrr']['hrr_class'] = 'GOOD'
            elif hrr_1min >= 12:
                out['hrr']['hrr_class'] = 'NORMAL'
            else:
                out['hrr']['hrr_class'] = 'LOW'

        # ═══════════════════════════════════════════
        # FAZA 3: SmO₂ / NIRS Integration
        # ═══════════════════════════════════════════

        smo2_col = None
        for c in ['SmO2_1', 'SmO2_2', 'SmO2_3', 'SmO2_4']:
            if c in df.columns and df[c].dropna().shape[0] > 50:
                smo2_col = c
                break

        out['nirs'] = {'available': smo2_col is not None}

        if smo2_col:
            out['nirs']['channel'] = smo2_col
            out['nirs']['stages'] = []

            # Get SmO2 time series for chart
            sm_series = df[['Time_s', smo2_col]].dropna().copy()
            sm_series.columns = ['Time_s', 'SmO2']
            out['nirs']['time_series'] = {
                't': sm_series['Time_s'].tolist(),
                'smo2': sm_series['SmO2'].tolist(),
            }

            # Rest baseline
            rest_mask = sm_series['Time_s'] < (stages[0].get('t_start', 60) if stages else 60)
            rest_vals = sm_series.loc[rest_mask, 'SmO2']
            smo2_rest = round(rest_vals.mean(), 1) if len(rest_vals) > 2 else None
            out['nirs']['smo2_rest'] = smo2_rest

            # Global min
            smo2_min = round(sm_series['SmO2'].min(), 1)
            out['nirs']['smo2_min'] = smo2_min
            out['nirs']['desat_total'] = round(smo2_rest - smo2_min, 1) if smo2_rest else None

            for s in stages:
                t0 = s.get('t_start', 0)
                t1 = s.get('t_end', 0)
                snum = s.get('stage_num', 0)
                warmup = min(180, (t1 - t0) * 0.5)
                tail = min(60, (t1 - t0) * 0.2)

                mask = (sm_series['Time_s'] >= t0) & (sm_series['Time_s'] <= t1)
                sub = sm_series[mask]

                ns = {
                    'stage_num': snum,
                    'domain': s.get('domain', ''),
                    'n_points': len(sub),
                }

                if len(sub) >= 5:
                    ns['smo2_mean'] = round(sub['SmO2'].mean(), 1)
                    ns['smo2_min'] = round(sub['SmO2'].min(), 1)
                    ns['smo2_max'] = round(sub['SmO2'].max(), 1)

                    # Early vs late
                    early = sub[sub['Time_s'] <= t0 + warmup]['SmO2']
                    late = sub[sub['Time_s'] >= t1 - tail]['SmO2']
                    if len(early) >= 2 and len(late) >= 2:
                        ns['smo2_early'] = round(early.mean(), 1)
                        ns['smo2_late'] = round(late.mean(), 1)
                        ns['smo2_drift'] = round(late.mean() - early.mean(), 1)

                    # Desaturation rate (slope of SmO2 vs time within stage)
                    t_arr = (sub['Time_s'].values - t0) / 60  # minutes
                    s_arr = sub['SmO2'].values
                    valid = np.isfinite(t_arr) & np.isfinite(s_arr)
                    if valid.sum() >= 5:
                        z = np.polyfit(t_arr[valid], s_arr[valid], 1)
                        ns['desat_slope_pct_min'] = round(z[0], 2)  # %SmO2 per minute

                    # τ matching: compare with VO2 tau
                    tau_on = s.get('tau_on_s')
                    if tau_on:
                        ns['tau_on_vo2'] = tau_on
                        # Estimate SmO2 τ from initial drop (first 120s of stage)
                        init_mask = (sub['Time_s'] >= t0) & (sub['Time_s'] <= t0 + 120)
                        init = sub[init_mask]
                        if len(init) >= 4:
                            sm_start = init['SmO2'].iloc[0]
                            sm_end = init['SmO2'].iloc[-1]
                            t_span = (init['Time_s'].iloc[-1] - init['Time_s'].iloc[0])
                            if abs(sm_start - sm_end) > 1 and t_span > 20:
                                # Simple exponential estimate: τ ≈ time to reach 63% of total change
                                change_63 = sm_start + (sm_end - sm_start) * 0.63
                                cross = init[init['SmO2'] <= change_63] if sm_end < sm_start else init[init['SmO2'] >= change_63]
                                if len(cross) > 0:
                                    ns['tau_smo2_est_s'] = round(cross['Time_s'].iloc[0] - t0, 1)
                                    ns['extraction_matching'] = round(ns['tau_smo2_est_s'] / tau_on, 2) if tau_on > 0 else None

                else:
                    ns['smo2_mean'] = None
                    ns['note'] = f'Insufficient SmO2 data ({len(sub)} pts)'

                out['nirs']['stages'].append(ns)

            # E12 integration (ramp test breakpoints)
            if e12 and e12.get('status') == 'OK':
                out['nirs']['bp1_vs_vt1_s'] = e12.get('bp1_vs_vt1_s')
                out['nirs']['bp2_vs_vt2_s'] = e12.get('bp2_vs_vt2_s')
                out['nirs']['reox_rate'] = e12.get('reox_rate')

            # SmO₂ reoxy vs τ_off matching
            if e12 and e12.get('reox_rate') and tau_off_mean:
                out['nirs']['reox_vs_tau_off'] = {
                    'reox_rate': e12.get('reox_rate'),
                    'tau_off_s': tau_off_mean,
                }

        # ═══════════════════════════════════════════
        # FAZA 4: τ ↔ VE/VCO₂ Integration
        # ═══════════════════════════════════════════

        ve_vco2_slope = None
        if e03 and e03.get('status') == 'OK':
            ve_vco2_slope = e03.get('ve_vco2_slope')
        # Also check E13 which may have VE/VCO₂ data
        if ve_vco2_slope is None and e13:
            ve_vco2_slope = e13.get('ve_vco2_slope')

        tau_mod = kin_profile.get('tau_moderate')

        out['ventilatory'] = {}
        if ve_vco2_slope and tau_mod:
            kvi = round(tau_mod / ve_vco2_slope, 2)
            out['ventilatory']['kinetic_ventilatory_index'] = kvi
            out['ventilatory']['ve_vco2_slope'] = ve_vco2_slope
            out['ventilatory']['tau_mod'] = tau_mod

            # 2×2 integration pattern
            tau_fast = tau_mod <= 25
            ve_good = ve_vco2_slope <= 30

            if tau_fast and ve_good:
                out['ventilatory']['integration'] = 'FULLY_INTEGRATED'
                out['ventilatory']['integration_pl'] = 'W pełni zintegrowany motor aerobowy'
            elif tau_fast and not ve_good:
                out['ventilatory']['integration'] = 'VQ_MISMATCH'
                out['ventilatory']['integration_pl'] = 'Kinetyka OK, ale V/Q matching suboptymalne'
            elif not tau_fast and ve_good:
                out['ventilatory']['integration'] = 'PERIPHERAL_LIMITED'
                out['ventilatory']['integration_pl'] = 'Wentylacja OK, limitacja obwodowa'
            else:
                out['ventilatory']['integration'] = 'DUAL_LIMITATION'
                out['ventilatory']['integration_pl'] = 'Podwójna limitacja (centralna + obwodowa)'

        # ═══════════════════════════════════════════
        # FAZA 5: Composite Syntheses
        # ═══════════════════════════════════════════

        # ── 5.1 Aerobic Fitness Fingerprint ──
        fp = {}

        # Dim 1: CAPACITY (0-100)
        vo2max_rel = e01.get('vo2peak_rel_mlkgmin')
        if vo2max_rel:
            # Simple percentile based on age/sex norms (male, use ACSM approx)
            age = ctx.get('_acfg')
            age_val = getattr(age, 'age', 34) if age else 34
            # Rough norms: male 30-39: 50th=38, 90th=50, 99th=56
            if vo2max_rel >= 56:
                fp['capacity'] = 95
            elif vo2max_rel >= 50:
                fp['capacity'] = 85
            elif vo2max_rel >= 44:
                fp['capacity'] = 70
            elif vo2max_rel >= 38:
                fp['capacity'] = 50
            elif vo2max_rel >= 33:
                fp['capacity'] = 30
            else:
                fp['capacity'] = 15

        # Dim 2: THRESHOLDS (0-100)
        vt1_pct = kin_profile.get('vt1_pct_vo2max')
        vt2_pct = kin_profile.get('vt2_pct_vo2max')
        if vt1_pct and vt2_pct:
            # VT1 >80% + VT2 >90% = excellent thresholds
            t_score = 0
            t_score += min(50, max(0, (vt1_pct - 60) / 25 * 50))  # 60%→0, 85%→50
            t_score += min(50, max(0, (vt2_pct - 80) / 15 * 50))  # 80%→0, 95%→50
            fp['thresholds'] = round(t_score)

        # Dim 3: KINETICS (0-100)
        tau_m = kin_profile.get('tau_moderate')
        sc_h = kin_profile.get('sc_heavy_pct', 0)
        if tau_m is not None:
            # τ: 15s=100, 25s=70, 35s=40, 50s=10
            tau_score = max(0, min(100, 100 - (tau_m - 15) * 3))
            # SC: <2%=100, 5%=70, 10%=30, >15%=0
            sc_score = max(0, min(100, 100 - sc_h * 7))
            fp['kinetics'] = round(tau_score * 0.6 + sc_score * 0.4)

        # Dim 4: RECOVERY (0-100)
        rec_score = 0
        rec_n = 0
        if hrr_1min:
            # HRR: >35=100, 25=70, 15=40, <10=10
            rec_score += max(0, min(100, (hrr_1min - 5) * 3.3))
            rec_n += 1
        if tau_off_mean:
            # τ_off: <30s=100, 45s=70, 60s=40, >90s=10
            rec_score += max(0, min(100, 100 - (tau_off_mean - 20) * 1.5))
            rec_n += 1
        # %rec@60s from off-kinetics
        if valid_off:
            recs_60 = [o.get('pct_recovered_60s') for o in valid_off if o.get('pct_recovered_60s')]
            if recs_60:
                rec_score += min(100, np.mean(recs_60))
                rec_n += 1
        if rec_n > 0:
            fp['recovery'] = round(rec_score / rec_n)

        out['composites']['fingerprint'] = fp

        # ── 5.2 Limitation Triangulation ──
        lim = {'central_evidence': 0, 'peripheral_evidence': 0, 'integrated_evidence': 0, 'n_signals': 0}

        # O₂ pulse trajectory (from E05 or E13)
        o2p_traj = e05.get('trajectory') or (e13.get('o2pulse_trajectory') if e13 else None)
        if o2p_traj:
            lim['n_signals'] += 1
            if o2p_traj in ('PLATEAU', 'EARLY_PLATEAU', 'FLAT'):
                lim['central_evidence'] += 1
            elif o2p_traj in ('RISING', 'LATE_RISE'):
                lim['peripheral_evidence'] += 1
            else:
                lim['integrated_evidence'] += 1

        # τ ratio (heavy/moderate)
        tau_ratio = kin_profile.get('tau_ratio_heavy_mod')
        if tau_ratio:
            lim['n_signals'] += 1
            if tau_ratio > 1.8:
                lim['central_evidence'] += 1  # Heavy τ much slower → delivery ceiling
            elif tau_ratio < 1.2:
                lim['integrated_evidence'] += 1  # Well matched
            else:
                lim['peripheral_evidence'] += 0.5
                lim['integrated_evidence'] += 0.5

        # VE/VCO₂
        if ve_vco2_slope:
            lim['n_signals'] += 1
            if ve_vco2_slope > 34:
                lim['central_evidence'] += 1
            elif ve_vco2_slope < 28:
                lim['integrated_evidence'] += 1
            else:
                lim['peripheral_evidence'] += 0.5
                lim['integrated_evidence'] += 0.5

        # SmO₂ desaturation (from E12)
        if e12 and e12.get('status') == 'OK':
            lim['n_signals'] += 1
            desat = e12.get('desat_total_pct', 0)
            smo2_min = e12.get('smo2_min')
            if smo2_min and smo2_min > 50:
                lim['peripheral_evidence'] += 1  # No deep desat → peripheral can't extract
            elif desat and desat > 30:
                lim['integrated_evidence'] += 1  # Good extraction
            else:
                lim['central_evidence'] += 0.5
                lim['integrated_evidence'] += 0.5

        # SC magnitude
        if sc_h is not None:
            lim['n_signals'] += 1
            if sc_h > 8:
                lim['peripheral_evidence'] += 1
            elif sc_h < 3:
                lim['integrated_evidence'] += 1
            else:
                lim['peripheral_evidence'] += 0.5
                lim['integrated_evidence'] += 0.5

        # Determine dominant limitation
        if lim['n_signals'] > 0:
            total = lim['central_evidence'] + lim['peripheral_evidence'] + lim['integrated_evidence']
            if total > 0:
                lim['central_pct'] = round(lim['central_evidence'] / total * 100)
                lim['peripheral_pct'] = round(lim['peripheral_evidence'] / total * 100)
                lim['integrated_pct'] = round(lim['integrated_evidence'] / total * 100)

                best = max(lim['central_pct'], lim['peripheral_pct'], lim['integrated_pct'])
                if best == lim['integrated_pct']:
                    lim['primary'] = 'INTEGRATED'
                    lim['primary_pl'] = 'Zintegrowana (brak jednej dominującej limitacji)'
                elif best == lim['central_pct']:
                    lim['primary'] = 'CENTRAL'
                    lim['primary_pl'] = 'Centralna (delivery O₂ do mięśni)'
                else:
                    lim['primary'] = 'PERIPHERAL'
                    lim['primary_pl'] = 'Obwodowa (ekstrakcja/utilizacja O₂ w mięśniach)'
                lim['confidence'] = round(best)

        out['composites']['limitation_triangle'] = lim

        # ── 5.3 Trainability Index ──
        trainability = {'dimensions': [], 'total_score': None}
        gaps = []

        if tau_m:
            gap = max(0, tau_m - 16) / (50 - 16) * 100  # 16s=elite, 50s=untrained
            gaps.append(('τ moderate', round(gap), f'{tau_m:.0f}s → 16s'))
            trainability['dimensions'].append({'name': 'τ moderate', 'current': tau_m, 'elite': 16, 'gap_pct': round(gap)})

        tau_h = kin_profile.get('tau_heavy')
        if tau_h:
            gap = max(0, tau_h - 20) / (60 - 20) * 100
            gaps.append(('τ heavy', round(gap), f'{tau_h:.0f}s → 20s'))
            trainability['dimensions'].append({'name': 'τ heavy', 'current': tau_h, 'elite': 20, 'gap_pct': round(gap)})

        if sc_h is not None:
            gap = max(0, sc_h - 2) / (15 - 2) * 100
            gaps.append(('SC heavy', round(gap), f'{sc_h:.1f}% → <2%'))
            trainability['dimensions'].append({'name': 'SC heavy', 'current': sc_h, 'elite': 2, 'gap_pct': round(gap)})

        if vt1_pct:
            gap = max(0, 82 - vt1_pct) / (82 - 55) * 100
            gaps.append(('VT1%', round(gap), f'{vt1_pct:.0f}% → 82%'))
            trainability['dimensions'].append({'name': 'VT1%', 'current': vt1_pct, 'elite': 82, 'gap_pct': round(gap)})

        if hrr_1min:
            gap = max(0, 35 - hrr_1min) / (35 - 8) * 100
            gaps.append(('HRR 1min', round(gap), f'{hrr_1min:.0f} → 35 bpm'))
            trainability['dimensions'].append({'name': 'HRR 1min', 'current': hrr_1min, 'elite': 35, 'gap_pct': round(gap)})

        if gaps:
            trainability['total_score'] = round(np.mean([g[1] for g in gaps]))
            # Classify
            ts = trainability['total_score']
            if ts > 60:
                trainability['class'] = 'HIGH'
                trainability['class_pl'] = 'Wysoki potencjał treningowy'
            elif ts > 30:
                trainability['class'] = 'MODERATE'
                trainability['class_pl'] = 'Umiarkowany potencjał'
            else:
                trainability['class'] = 'LOW'
                trainability['class_pl'] = 'Bliski optymalnego — diminishing returns'

        out['composites']['trainability'] = trainability

        # ── 5.4 Performance Model (sport-specific) ──
        sport = ctx.get('_acfg')
        sport_name = getattr(sport, 'modality', 'run') if sport else 'run'

        perf = {'sport': sport_name}
        vt2_speed = e02.get('vt2_speed_kmh')

        if sport_name == 'hyrox' and vt2_speed and tau_m:
            # HYROX model
            n_transitions = 8
            transition_tax_s = tau_m * n_transitions * 0.7  # 70% of τ as "penalty"
            economy_loss_factor = 1 + (sc_h or 0) / 100
            sustainable_pace_kmh = vt2_speed * 0.88 / economy_loss_factor  # 88% VT2 for 8×1km
            run_time_min = 8 / sustainable_pace_kmh * 60  # 8km total
            perf['transition_tax_s'] = round(transition_tax_s)
            perf['sustainable_pace_kmh'] = round(sustainable_pace_kmh, 1)
            perf['run_segments_min'] = round(run_time_min, 1)
            perf['model_note'] = f'τ penalty: {transition_tax_s:.0f}s across {n_transitions} transitions'

        elif sport_name == 'run' and vt2_speed:
            # Endurance running model
            # Half marathon: ~95% VT2 sustained, penalized by SC
            hm_factor = 0.95 / (1 + (sc_h or 0) / 100)
            hm_pace = vt2_speed * hm_factor
            hm_time_min = 21.1 / hm_pace * 60
            perf['half_marathon_pace_kmh'] = round(hm_pace, 1)
            perf['half_marathon_time_min'] = round(hm_time_min, 1)

            # 10K: ~98% VT2
            pace_10k = vt2_speed * 0.98 / (1 + (sc_h or 0) / 200)
            perf['_10k_pace_kmh'] = round(pace_10k, 1)
            perf['_10k_time_min'] = round(10 / pace_10k * 60, 1)

        out['composites']['performance'] = perf

        # ═══ MEI Composite (Metabolic Efficiency Index — average across stages) ═══
        meis = [s.get('metabolic_efficiency_idx') for s in out['stage_cross'] 
                if s.get('metabolic_efficiency_idx') is not None and s.get('domain') in ('HEAVY', 'SEVERE')]
        if meis:
            mei_avg = round(np.mean(meis), 2)
            out['composites']['mei_avg'] = mei_avg
            if mei_avg < 3:
                out['composites']['mei_class'] = 'EXCELLENT'
            elif mei_avg < 6:
                out['composites']['mei_class'] = 'GOOD'
            else:
                out['composites']['mei_class'] = 'FLAG'

        return out


print("✅ Engine_E22_CrossCorrelation loaded.")
