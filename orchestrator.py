# --- Compile PROTOCOLS_DB from RAW_PROTOCOLS ---
PROTOCOLS_DB = {}
for _pname, _psegs in RAW_PROTOCOLS.items():
    PROTOCOLS_DB[_pname] = compile_protocol_for_apply(_psegs)

# ==========================================
# 5. ORCHESTRATOR (CALC_ONLY + TRAINER_CANON)
# ==========================================

class CPET_Orchestrator:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self.raw = None
        self.processed = None
        self.results = {}
        self._qc_log = {"engines_executed_ok": [], "engine_errors": []}

    # ---------- helpers ----------
    def _num(self, x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float, np.integer, np.floating)):
                if np.isnan(x):
                    return None
                return float(x)
            s = str(x).strip().replace(",", ".")
            if s == "" or s.lower() in ("nan", "none", "null"):
                return None
            v = float(s)
            if np.isnan(v):
                return None
            return v
        except Exception:
            return None

    def _pick(self, dct: dict, keys: list, default=None):
        for k in keys:
            if k in dct and dct[k] is not None:
                return dct[k]
        return default

    def _safe_run(self, engine_id: str, fn, *args, **kwargs):
        import traceback as _tb
        try:
            out = fn(*args, **kwargs)
            if isinstance(out, dict):
                out.setdefault("status", "OK")
                self._qc_log["engines_executed_ok"].append(engine_id)
                return out
            return {"status": "OK", "value": out}
        except Exception as e:
            tb_str = _tb.format_exc()
            err_msg = f"{type(e).__name__}: {e}"
            # Log for QC audit trail
            self._qc_log.setdefault("engine_errors", []).append({
                "engine": engine_id, "error": err_msg, "traceback": tb_str
            })
            print(f"  ⚠️ {engine_id} ERROR: {err_msg}")
            return {"status": "ERROR", "reason": err_msg, "traceback": tb_str}

    # ---------- manual VT override ----------
    def _apply_manual_vt_override(self, df_ex):
        """Apply manual VT1/VT2 override from panel."""
        import pandas as pd, numpy as np
        def _mmss(s):
            if s is None: return None
            s = str(s).strip()
            if not s or s.lower() == 'none': return None
            parts = s.split(":")
            if len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return float(s)
        def _avg_at(df, t_sec, col, window=15):
            _tc = "Time_sec" if "Time_sec" in df.columns else "Time_s"
            t = pd.to_numeric(df.get(_tc, pd.Series(dtype=float)), errors="coerce")
            v = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
            m = (t >= t_sec - window) & (t <= t_sec + window) & v.notna()
            return round(float(v[m].mean()), 2) if m.sum() >= 1 else None
        vt1_t = _mmss(getattr(self.cfg, "vt1_manual", None))
        vt2_t = _mmss(getattr(self.cfg, "vt2_manual", None))
        if vt1_t is None and vt2_t is None:
            return
        e02 = self.results.get("E02") or {}
        e01 = self.results.get("E01") or {}
        hr_max = e01.get("hr_peak")
        vo2_peak = e01.get("vo2_peak_ml_min")
        for vt, vt_sec in [("vt1", vt1_t), ("vt2", vt2_t)]:
            if vt_sec is None: continue
            e02[f"{vt}_time_sec"] = round(vt_sec, 1)
            hr = _avg_at(df_ex, vt_sec, "HR_bpm")
            e02[f"{vt}_hr"] = hr
            if hr and hr_max: e02[f"{vt}_hr_pct_max"] = round(hr / hr_max * 100, 1)
            vo2 = None
            for col in ["VO2_ml_min", "VO2_mlmin", "VO2_L_min"]:
                vo2 = _avg_at(df_ex, vt_sec, col)
                if vo2 is not None:
                    if col == "VO2_L_min" and vo2 < 20: vo2 = round(vo2 * 1000, 1)
                    break
            e02[f"{vt}_vo2_mlmin"] = vo2
            if vo2 and vo2_peak: e02[f"{vt}_vo2_pct_peak"] = round(vo2 / vo2_peak * 100, 1)
            e02[f"{vt}_speed_kmh"] = _avg_at(df_ex, vt_sec, "Speed_kmh")
            e02[f"{vt}_power_w"] = _avg_at(df_ex, vt_sec, "Power_W")
            e02[f"{vt}_rer"] = _avg_at(df_ex, vt_sec, "RER")
            e02[f"{vt}_confidence"] = 1.0
            e02[f"{vt}_source"] = "manual"
            e02[f"{vt}_n_methods"] = 99
            e02[f"{vt}_methods_agreed"] = ["MANUAL"]
            if "flags" not in e02: e02["flags"] = []
            e02["flags"].append(f"MANUAL_OVERRIDE_{vt.upper()}")
            m, s = int(vt_sec // 60), int(vt_sec % 60)
            print(f"  \u2705 {vt.upper()} MANUAL: t={m}:{s:02d} | HR={hr} | VO2={vo2} | Speed={e02[f'{vt}_speed_kmh']}")
        if vt1_t and vt2_t: e02["confidence"] = "MANUAL"
        e02["status"] = "OK"
        self.results["E02"] = e02
        print("  \u26A1 Progi nadpisane (manual > auto)")

    # ---------- new outputs ----------
    def build_outputs(self) -> dict:
        r = self.results or {}
        cfg = self.cfg

        executed_ok, failed, limited, not_run = [], [], [], []
        for eid in [f"E{i:02d}" for i in range(17)]:
            block = r.get(eid)
            if block is None:
                not_run.append(eid)
                continue

            st = str(block.get("status", "UNKNOWN")).upper()

            # traktujemy jako OK także statusy "technicznie poprawne"
            if st in {"OK", "DONE_IN_QC", "TODO"}:
                executed_ok.append(eid)
            elif st in {"LIMITED", "NO_SIGNAL", "NO_DATA", "INSUFFICIENT_DATA"}:
                limited.append(eid)
            else:
                failed.append(eid)

        orch_meta = {
            "orchestrator_version": "ORCH_v7",
            "mode": "CALC_ONLY",
            "primary_modality": cfg.modality,
            "protocol_type": getattr(cfg, "protocol_type", "RAMP"),
            "load_unit_primary": "Speed_kmh" if cfg.modality == "run" else "Power_W",
            "timestamp_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        qc_log = {
            "pipeline": "0-IMPORT/MAP/CANONICALIZE;1-PATCH;2-QC;3-SMOOTHING;4-SEG;5-ENGINES;6-QA_GLOBAL",
            "status": "PASS" if (len(failed) == 0 and len(not_run) == 0) else "PARTIAL",
            "notes": [],
            "mapping_used": {},
            "engines_executed_ok": executed_ok,
            "engines_failed": failed,
            "engines_limited": limited,
            "engines_not_run": not_run,
            # kompatybilność wsteczna:
            "engines_executed": executed_ok,
            "engines_missing": failed + not_run,
        }

        e02 = r.get("E02", {})
        e01 = r.get("E01", {})
        e03 = r.get("E03", {})
        e08 = r.get("E08", {})
        e06 = r.get("E06", {})
        e10 = r.get("E10", {})
        e04 = r.get("E04", {})
        e09 = r.get("E09", {})

        tabela_parametrow = {
            # progi
            "thr_vt1_time_s": self._num(self._pick(e02, ["vt1_time_sec", "vt1_time_s", "VT1_time_s"])),
            "thr_vt2_time_s": self._num(self._pick(e02, ["vt2_time_sec", "vt2_time_s", "VT2_time_s"])),
            "thr_vt1_hr_bpm": self._num(self._pick(e02, ["vt1_hr", "vt1_hr_bpm", "VT1_hr_bpm"])),
            "thr_vt2_hr_bpm": self._num(self._pick(e02, ["vt2_hr", "vt2_hr_bpm", "VT2_hr_bpm"])),
            "thr_vt1_speed_kmh": self._num(self._pick(e02, ["vt1_speed", "vt1_speed_kmh", "VT1_speed_kmh"])),
            "thr_vt2_speed_kmh": self._num(self._pick(e02, ["vt2_speed", "vt2_speed_kmh", "VT2_speed_kmh"])),

            # peak
            "peak_vo2_mlkgmin": self._num(self._pick(e01, ["vo2_peak_mlkgmin", "VO2peak_mlkgmin"])),
            "peak_vo2_mlmin": self._num(self._pick(e01, ["vo2_peak_mlmin", "VO2peak_mlmin"])),
            "peak_hr_bpm": self._num(self._pick(e01, ["hr_peak", "HR_peak_bpm"])),
            "peak_rer": self._num(self._pick(e01, ["rer_peak", "RER_peak"])),
            "peak_ve_lmin": self._num(self._pick(e01, ["ve_peak_lmin", "ve_peak", "VE_peak_lmin"])),

            # wentylacja
            "vent_ve_vco2_slope": self._num(self._pick(e03, ["slope_to_vt2", "slope_full", "slope", "ve_vco2_slope"])),
            "vent_breathing_reserve_pct": self._num(self._pick(e09, ["br_pct", "br_percent", "BR_pct"])),

            # metabolizm
            "met_fatmax_gmin": self._num(self._pick(e10, ["mfo_gmin", "fat_max", "fatmax_g_min"])),
            "met_fatmax_intensity_hr_bpm": self._num(self._pick(e10, ["fatmax_hr", "fat_max_hr"])),
            "met_fatmax_intensity_speed_kmh": self._num(self._pick(e10, ["fatmax_speed_kmh", "fat_max_speed"])),
            "met_fatmax_pct_vo2peak": self._num(e10.get("fatmax_pct_vo2peak")),
            "met_fatmax_pct_hrmax": self._num(e10.get("fatmax_pct_hrmax")),
            "met_fatmax_zone_hr_low": self._num(e10.get("fatmax_zone_hr_low")),
            "met_fatmax_zone_hr_high": self._num(e10.get("fatmax_zone_hr_high")),
            "met_mfo_mgkg_min": self._num(e10.get("mfo_mgkg_min")),
            "met_cop_hr": self._num(e10.get("cop_hr")),
            "met_cop_speed_kmh": self._num(e10.get("cop_speed_kmh")),
            "met_cop_pct_vo2peak": self._num(e10.get("cop_pct_vo2peak")),
            "met_cop_rer": self._num(e10.get("cop_rer")),
            "met_fat_pct_at_vt1": self._num(e10.get("fat_pct_at_vt1")),
            "met_cho_pct_at_vt1": self._num(e10.get("cho_pct_at_vt1")),

            # Zone substrate
            "met_z2_fat_gh": self._num(e10.get("zone_substrate", {}).get("z2", {}).get("fat_gh")),
            "met_z2_cho_gh": self._num(e10.get("zone_substrate", {}).get("z2", {}).get("cho_gh")),
            "met_z4_fat_gh": self._num(e10.get("zone_substrate", {}).get("z4", {}).get("fat_gh")),
            "met_z4_cho_gh": self._num(e10.get("zone_substrate", {}).get("z4", {}).get("cho_gh")),
            "met_z5_fat_gh": self._num(e10.get("zone_substrate", {}).get("z5", {}).get("fat_gh")),
            "met_z5_cho_gh": self._num(e10.get("zone_substrate", {}).get("z5", {}).get("cho_gh")),

            # HRR
            # E06
            "gain_below_vt1": self._num(e06.get("gain_below_vt1")),
            "gain_below_vt1_r2": self._num(e06.get("gain_below_vt1_r2")),
            "gain_full": self._num(e06.get("gain_full")),
            "gain_unit": e06.get("gain_unit"),
            "gain_modality": e06.get("modality"),
            "running_economy_mlkgkm": self._num(e06.get("running_economy_mlkgkm")),
            "re_classification": e06.get("re_classification"),
            "delta_efficiency_pct": self._num(e06.get("delta_efficiency_pct")),
            "gain_at_vt1": self._num(e06.get("gain_at_vt1")),
            "gain_at_vt2": self._num(e06.get("gain_at_vt2")),
            "re_at_vt1": self._num(e06.get("re_at_vt1")),
            "re_at_vt2": self._num(e06.get("re_at_vt2")),
            "eff_at_vt1": self._num(e06.get("eff_at_vt1")),
            "eff_at_vt2": self._num(e06.get("eff_at_vt2")),
            "rec_hrr_60s_bpm": self._num(self._pick(e08, ["hrr_1min", "hrr1", "hrr_60", "HRR_60s_bpm"])),
            "rec_hrr_180s_bpm": self._num(self._pick(e08, ["hrr_3min", "hrr3", "hrr_180", "HRR_180s_bpm"])),

            # OUES
            "global_oues": self._num(self._pick(e04, ["oues100", "oues_val", "oues", "OUES"])),
            "oues_90": self._num(e04.get("oues90")),
            "oues_75": self._num(e04.get("oues75")),
            "oues_to_vt1": self._num(e04.get("oues_to_vt1")),
            "oues_to_vt2": self._num(e04.get("oues_to_vt2")),
            "oues_per_kg": self._num(e04.get("oues_per_kg")),
            "oues_per_bsa": self._num(e04.get("oues_per_bsa")),
            "oues_pct_predicted": self._num(e04.get("oues_pct_hollenberg")),
            "oues_r2": self._num(e04.get("oues_r2_100")),
        }

        tabela_flag = {
            "flag_missing_vt1": (
                tabela_parametrow["thr_vt1_hr_bpm"] is None
                and tabela_parametrow["thr_vt1_speed_kmh"] is None
                and tabela_parametrow["thr_vt1_time_s"] is None
            ),
            "flag_missing_vt2": (
                tabela_parametrow["thr_vt2_hr_bpm"] is None
                and tabela_parametrow["thr_vt2_speed_kmh"] is None
                and tabela_parametrow["thr_vt2_time_s"] is None
            ),
            "flag_missing_vt1_time_only": tabela_parametrow["thr_vt1_time_s"] is None,
            "flag_missing_vt2_time_only": tabela_parametrow["thr_vt2_time_s"] is None,

            "flag_missing_fat_cho": tabela_parametrow["met_fatmax_gmin"] is None,
            "flag_missing_lactate": r.get("E11", {}).get("status") != "OK",
            "flag_low_quality_hr": r.get("E01", {}).get("status") != "OK",
            "flag_protocol_patch_used": True,
            "flag_partial_engines": len(qc_log.get("engines_missing", [])) > 0,
        }

        return {
            "orch_meta": orch_meta,
            "qc_log": qc_log,
            "tabela_parametrow": tabela_parametrow,
            "tabela_flag": tabela_flag,
        }

    def build_trainer_canon_flat(self, outputs: dict, extra_context: dict = None) -> dict:
        extra_context = extra_context or {}
        tpar = outputs.get("tabela_parametrow", {})
        tflag = outputs.get("tabela_flag", {})
        meta = outputs.get("orch_meta", {})
        qc = outputs.get("qc_log", {})

        trainer = {}

        # meta_
        trainer["meta_orchestrator_version"] = meta.get("orchestrator_version")
        trainer["meta_mode"] = meta.get("mode")
        trainer["meta_primary_modality"] = meta.get("primary_modality")
        trainer["meta_protocol_type"] = meta.get("protocol_type")
        trainer["meta_load_unit_primary"] = meta.get("load_unit_primary")
        trainer["meta_timestamp_utc"] = meta.get("timestamp_utc")

        # thr_
        trainer["thr_vt1_time_s"] = tpar.get("thr_vt1_time_s")
        trainer["thr_vt2_time_s"] = tpar.get("thr_vt2_time_s")
        trainer["thr_vt1_hr_bpm"] = tpar.get("thr_vt1_hr_bpm")
        trainer["thr_vt2_hr_bpm"] = tpar.get("thr_vt2_hr_bpm")
        trainer["thr_vt1_speed_kmh"] = tpar.get("thr_vt1_speed_kmh")
        trainer["thr_vt2_speed_kmh"] = tpar.get("thr_vt2_speed_kmh")

        # peak_
        trainer["peak_vo2_mlkgmin"] = tpar.get("peak_vo2_mlkgmin")
        trainer["peak_vo2_mlmin"] = tpar.get("peak_vo2_mlmin")
        trainer["peak_hr_bpm"] = tpar.get("peak_hr_bpm")
        trainer["peak_rer"] = tpar.get("peak_rer")
        trainer["peak_ve_lmin"] = tpar.get("peak_ve_lmin")

        # met_
        trainer["met_fatmax_gmin"] = tpar.get("met_fatmax_gmin")
        trainer["met_fatmax_intensity_hr_bpm"] = tpar.get("met_fatmax_intensity_hr_bpm")
        trainer["met_fatmax_intensity_speed_kmh"] = tpar.get("met_fatmax_intensity_speed_kmh")
        trainer["met_oues"] = tpar.get("global_oues")

        # zone_ — pobierz z E16 results, fallback do extra_context
        e16 = self.results.get("E16", {})
        _e16_zones = e16.get("zones", {}) if isinstance(e16, dict) else {}
        for z in ["z1", "z2", "z3", "z4", "z5"]:
            z_data = _e16_zones.get(z, e16.get(z, {})) if isinstance(e16, dict) else {}
            trainer[f"zone_{z}_hr_min"] = self._num(z_data.get("hr_low")) or extra_context.get(f"zone_{z}_hr_min")
            trainer[f"zone_{z}_hr_max"] = self._num(z_data.get("hr_high")) or extra_context.get(f"zone_{z}_hr_max")
            trainer[f"zone_{z}_speed_min"] = self._num(z_data.get("speed_low"))
            trainer[f"zone_{z}_speed_max"] = self._num(z_data.get("speed_high"))
            trainer[f"zone_{z}_name"] = z_data.get("name_pl", "")

        # flag_
        for k, v in tflag.items():
            trainer[k] = bool(v)

        # qc_
        trainer["qc_status"] = qc.get("status")
        trainer["qc_engines_executed_n"] = len(qc.get("engines_executed_ok", qc.get("engines_executed", [])))
        trainer["qc_engines_failed_n"] = len(qc.get("engines_failed", []))
        trainer["qc_engines_limited_n"] = len(qc.get("engines_limited", []))
        trainer["qc_engines_not_run_n"] = len(qc.get("engines_not_run", []))
        trainer["qc_engines_missing_n"] = len(qc.get("engines_missing", []))

        return trainer

    # ---------- main ----------

    def _feedback_loop(self, df_ex):
        """
        Feedback loop: post-validation threshold adjustment.
        
        Runs AFTER E18 (VT↔LT cross-validation) and E19 (concordance).
        Reads adjudicator flags from E02 + evidence from E11/E18/E19
        and can ADJUST E02 thresholds when evidence strongly indicates
        the consensus was wrong.
        
        Design principles:
        1. Only adjusts when multiple independent sources agree
        2. Never adjusts VT1 (4-method isocapnic anchor is reliable)
        3. Only adjusts VT2 when:
           a. Adjudicator flagged EXCLUDED_CLUSTER, AND
           b. E18 layer3 threshold_2 status is POOR, AND
           c. LT2 agrees with excluded cluster (within 60s)
        4. Logs all adjustments transparently
        5. Does NOT rerun any engines — only modifies stored E02 results
        
        After adjustment, downstream E16 (zones) and report get the
        corrected thresholds.
        """
        import numpy as np
        
        e02 = self.results.get("E02", {})
        e11 = self.results.get("E11", {})
        e18 = self.results.get("E18", {})
        e19 = self.results.get("E19", {})
        
        fb_log = {
            "executed": True,
            "vt1_adjusted": False,
            "vt2_adjusted": False,
            "adjustments": [],
            "reasons": [],
        }
        
        if e02.get("status") not in ("OK", "PARTIAL"):
            fb_log["skipped"] = "E02 not OK"
            self.results["_feedback"] = fb_log
            return
        
        flags = e02.get("flags", [])
        
        # ══════════════════════════════════════════════════════════
        # VT2 FEEDBACK — the main case
        # ══════════════════════════════════════════════════════════
        
        # Condition 1: Adjudicator detected excluded cluster
        has_excluded_cluster = any("ADJ_VT2_EXCLUDED_CLUSTER" in f for f in flags)
        
        # Condition 2: E18 concordance is POOR for threshold_2
        e18_l3 = e18.get("layer3_concordance", {})
        t2_conc = e18_l3.get("threshold_2", {})
        e18_poor = t2_conc.get("status") in ("POOR", "VERY_POOR")
        
        # Condition 3: LT2 exists and is reliable
        lt2_time = e11.get("lt2_time_sec")
        lt2_exists = lt2_time is not None and lt2_time > 0
        
        # Condition 4: E19 temporal alignment shows VT2 as outlier
        ta_vt2 = e19.get("temporal_alignment", {}).get("VT2", {})
        vt2_is_outlier = ta_vt2.get("outlier") == "VT2"
        
        if has_excluded_cluster and (e18_poor or vt2_is_outlier) and lt2_exists:
            # Extract excluded cluster center from adjudicator flag
            excl_flag = [f for f in flags if "ADJ_VT2_EXCLUDED_CLUSTER" in f][0]
            # Parse: "ADJ_VT2_EXCLUDED_CLUSTER:3m@1086s(spread=49s)_vs_2m@913s"
            import re
            m = re.search(r'(\d+)m@(\d+)s\(spread=(\d+)s\)', excl_flag)
            if m:
                excl_n = int(m.group(1))
                excl_center = float(m.group(2))
                excl_spread = float(m.group(3))
            else:
                fb_log["skipped"] = "Cannot parse excluded cluster"
                self.results["_feedback"] = fb_log
                return
            
            # Check: does LT2 agree with excluded cluster?
            lt2_delta = abs(lt2_time - excl_center)
            lt2_agrees = lt2_delta < 60  # within 1 minute
            
            # Additional check: NIRS BP2 agreement?
            nirs_bp2 = self.results.get("E12", {}).get("bp2_time_s")
            nirs_agrees = (nirs_bp2 is not None 
                          and abs(nirs_bp2 - excl_center) < 90)
            
            # Count independent confirmations
            confirmations = []
            if lt2_agrees:
                confirmations.append(f"LT2={lt2_time:.0f}s (Δ={lt2_delta:.0f}s)")
            if nirs_agrees:
                confirmations.append(f"NIRS_BP2={nirs_bp2:.0f}s (Δ={abs(nirs_bp2 - excl_center):.0f}s)")
            
            # Decision: adjust if ≥2 independent sources confirm
            # (excluded gas-exchange cluster + at least LT2 or NIRS)
            n_sources = excl_n + len(confirmations)  # gas methods + external
            
            # Require at least LT2 confirmation (NIRS alone not sufficient)
            if lt2_agrees and n_sources >= 4:
                # ── COMPUTE ADJUSTED VT2 ─────────────────────────
                # Weighted average of all agreeing sources
                sources_t = [excl_center]
                sources_w = [0.4 * excl_n]  # gas exchange cluster
                
                if lt2_agrees:
                    sources_t.append(lt2_time)
                    sources_w.append(0.35)  # lactate
                if nirs_agrees:
                    sources_t.append(nirs_bp2)
                    sources_w.append(0.15)  # NIRS (lower weight)
                
                # Original VT2 consensus (small weight — it's the contested value)
                orig_vt2 = e02.get("vt2_time_sec")
                sources_t.append(orig_vt2)
                sources_w.append(0.10)
                
                adjusted_t = float(np.average(sources_t, weights=sources_w))
                
                # Find closest data point
                try:
                    closest_idx = (df_ex['Time_sec'] - adjusted_t).abs().idxmin()
                    row = df_ex.loc[closest_idx]
                    
                    # Store original values
                    fb_log["vt2_original"] = {
                        "time_sec": e02.get("vt2_time_sec"),
                        "vo2_pct": e02.get("vt2_vo2_pct_peak"),
                        "confidence": e02.get("vt2_confidence"),
                        "source": e02.get("vt2_source"),
                    }
                    
                    # Update E02 results
                    new_t = float(row.get("Time_sec", adjusted_t))
                    
                    # Resolve VO2 column (prefer smoothed, then raw canonical)
                    vo2_col = None
                    for c in ["VO2_ml_min", "VO2_mlmin", "VO2", "vo2_ml"]:
                        if c in row.index and pd.notna(row[c]):
                            vo2_col = c
                            break
                    
                    # Use E02's internal VO2 peak (same as used for vo2_pct)
                    vo2_peak = (e02.get("diagnostics", {}).get("vo2_peak")
                               or self.results.get("E01", {}).get("vo2_peak_ml_min")
                               or 0)
                    if not vo2_peak:
                        vo2_peak = e02.get("vt1_vo2_mlmin", 1)  # last fallback
                    
                    new_vo2 = float(row[vo2_col]) if vo2_col else e02.get("vt2_vo2_mlmin")
                    new_vo2_pct = (new_vo2 / vo2_peak * 100) if (new_vo2 and vo2_peak and vo2_peak > 0) else e02.get("vt2_vo2_pct_peak")
                    
                    # HR
                    hr_col = None
                    for c in ["HR_bpm", "HR", "hr", "HR_bpm_smooth"]:
                        if c in row.index and pd.notna(row.get(c)):
                            hr_col = c
                            break
                    new_hr = float(row[hr_col]) if hr_col else None
                    hr_max = self.results.get("E01", {}).get("hr_peak")
                    new_hr_pct = (new_hr / hr_max * 100) if (new_hr and hr_max) else None
                    
                    # RER
                    rer_col = None
                    for c in ["RER", "rer", "RER_smooth"]:
                        if c in row.index and pd.notna(row.get(c)):
                            rer_col = c
                            break
                    new_rer = float(row[rer_col]) if rer_col else None
                    
                    # Speed
                    spd_col = None
                    for c in ["Speed_kmh", "Speed", "speed"]:
                        if c in row.index and pd.notna(row.get(c)):
                            spd_col = c
                            break
                    new_speed = float(row[spd_col]) if spd_col else None
                    
                    # Apply adjustment
                    e02["vt2_time_sec"] = new_t
                    if new_vo2: e02["vt2_vo2_mlmin"] = new_vo2
                    if new_vo2_pct: e02["vt2_vo2_pct_peak"] = new_vo2_pct
                    if new_hr: e02["vt2_hr"] = new_hr
                    if new_hr is not None: e02["vt2_hr_bpm"] = new_hr
                    if new_hr_pct: e02["vt2_hr_pct_max"] = new_hr_pct
                    if new_rer: e02["vt2_rer"] = new_rer
                    if new_speed: e02["vt2_speed_kmh"] = new_speed
                    
                    # New confidence — based on number of agreeing sources
                    new_conf = min(0.95, 0.60 + 0.08 * n_sources)
                    e02["vt2_confidence"] = new_conf
                    e02["vt2_source"] = f"feedback_adjusted({e02.get('vt2_source', '?')})"
                    
                    # Update E02 confidence label
                    min_conf = min(e02.get("vt1_confidence", 0), new_conf)
                    e02["confidence"] = ("HIGH" if min_conf > 0.75
                                         else "MEDIUM" if min_conf > 0.5
                                         else "LOW")
                    
                    # Add flag
                    e02.setdefault("flags", []).append(
                        f"FEEDBACK_VT2_ADJUSTED:{e02['vt2_time_sec']:.0f}s"
                        f"←{fb_log['vt2_original']['time_sec']:.0f}s"
                        f"_n{n_sources}sources"
                    )
                    
                    fb_log["vt2_adjusted"] = True
                    fb_log["adjustments"].append({
                        "threshold": "VT2",
                        "original_time": fb_log["vt2_original"]["time_sec"],
                        "adjusted_time": new_t,
                        "delta_sec": new_t - fb_log["vt2_original"]["time_sec"],
                        "n_sources": n_sources,
                        "confirmations": confirmations,
                        "new_confidence": new_conf,
                    })
                    fb_log["reasons"].append(
                        f"VT2 adjusted: {fb_log['vt2_original']['time_sec']:.0f}s → "
                        f"{new_t:.0f}s (Δ={new_t - fb_log['vt2_original']['time_sec']:+.0f}s). "
                        f"Sources: excluded_cluster@{excl_center:.0f}s + "
                        f"{', '.join(confirmations)}."
                    )
                    
                    print(f"  🔄 FEEDBACK: VT2 adjusted "
                          f"{fb_log['vt2_original']['time_sec']:.0f}s → {new_t:.0f}s "
                          f"({n_sources} sources)")
                    
                except Exception as ex:
                    fb_log["error"] = str(ex)
            else:
                fb_log["reasons"].append(
                    f"VT2 flagged but not adjusted: "
                    f"confirmations={confirmations}, n_sources={n_sources}")
        
        # ══════════════════════════════════════════════════════════
        # PATH B: E18-driven correction
        # When no excluded cluster but E18 POOR + VT2 weak consensus
        # + LT2 significantly later → shift VT2 toward LT2
        # ══════════════════════════════════════════════════════════
        if (not fb_log["vt2_adjusted"]
                and e18_poor
                and lt2_exists
                and e02.get("vt2_confidence", 1.0) < 0.85):
            
            has_weak = any("ADJ_VT2_WEAK_PAIR" in f for f in flags)
            has_spread = any("ADJ_VT2_SPREAD" in f for f in flags)
            
            if has_weak or has_spread:
                orig_vt2 = e02.get("vt2_time_sec", 0)
                delta_vt_lt = lt2_time - orig_vt2
                
                if delta_vt_lt > 120:
                    all_cands = e02.get("vt2_candidates", {})
                    cands_near_lt2 = {n: c for n, c in all_cands.items()
                                      if abs(c.get("time", 0) - lt2_time) < 120}
                    
                    nirs_bp2 = self.results.get("E12", {}).get("bp2_time_s")
                    nirs_near = (nirs_bp2 is not None
                                 and abs(nirs_bp2 - lt2_time) < 120)
                    
                    n_confirms = len(cands_near_lt2) + (1 if nirs_near else 0)
                    
                    if n_confirms >= 1:
                        src_t = [lt2_time]
                        src_w = [0.45]
                        for cn, cc in cands_near_lt2.items():
                            src_t.append(cc["time"])
                            src_w.append(0.20)
                        if nirs_near:
                            src_t.append(nirs_bp2)
                            src_w.append(0.15)
                        src_t.append(orig_vt2)
                        src_w.append(0.10)
                        
                        adj_t = float(np.average(src_t, weights=src_w))
                        
                        try:
                            ci = (df_ex['Time_sec'] - adj_t).abs().idxmin()
                            row = df_ex.loc[ci]
                            new_t = float(row.get("Time_sec", adj_t))
                            
                            fb_log["vt2_original"] = {
                                "time_sec": orig_vt2,
                                "confidence": e02.get("vt2_confidence"),
                                "source": e02.get("vt2_source"),
                            }
                            
                            vp = (e02.get("diagnostics", {}).get("vo2_peak")
                                  or self.results.get("E01", {}).get("vo2_peak_ml_min")
                                  or 0)
                            
                            for c in ["VO2_ml_min", "VO2_mlmin", "VO2"]:
                                if c in row.index and pd.notna(row[c]):
                                    e02["vt2_vo2_mlmin"] = float(row[c])
                                    if vp > 0:
                                        e02["vt2_vo2_pct_peak"] = float(row[c]) / vp * 100
                                    break
                            for c in ["HR_bpm", "HR"]:
                                if c in row.index and pd.notna(row.get(c)):
                                    e02["vt2_hr"] = float(row[c])
                                    hr_mx = self.results.get("E01", {}).get("hr_peak")
                                    if hr_mx: e02["vt2_hr_pct_max"] = float(row[c]) / hr_mx * 100
                                    break
                            for c in ["RER"]:
                                if c in row.index and pd.notna(row.get(c)):
                                    e02["vt2_rer"] = float(row[c])
                                    break
                            for c in ["Speed_kmh", "Speed"]:
                                if c in row.index and pd.notna(row.get(c)):
                                    e02["vt2_speed_kmh"] = float(row[c])
                                    break
                            
                            e02["vt2_time_sec"] = new_t
                            tot = 1 + n_confirms + 1
                            nc = min(0.90, 0.55 + 0.07 * tot)
                            e02["vt2_confidence"] = nc
                            e02["vt2_source"] = f"feedback_e18({e02.get('vt2_source', '?')})"
                            
                            mc = min(e02.get("vt1_confidence", 0), nc)
                            e02["confidence"] = ("HIGH" if mc > 0.75
                                                 else "MEDIUM" if mc > 0.5
                                                 else "LOW")
                            
                            cstr = []
                            if cands_near_lt2: cstr.append(f"gas:{','.join(cands_near_lt2.keys())}")
                            if nirs_near: cstr.append(f"NIRS={nirs_bp2:.0f}s")
                            
                            e02.setdefault("flags", []).append(
                                f"FEEDBACK_E18_VT2:{new_t:.0f}s←{orig_vt2:.0f}s_n{tot}")
                            
                            fb_log["vt2_adjusted"] = True
                            fb_log["adjustments"].append({
                                "path": "E18", "original_time": orig_vt2,
                                "adjusted_time": new_t, "delta_sec": new_t - orig_vt2,
                                "n_sources": tot, "confirmations": cstr,
                            })
                            fb_log["reasons"].append(
                                f"VT2 via E18: {orig_vt2:.0f}→{new_t:.0f}s "
                                f"(Δ{new_t-orig_vt2:+.0f}s). LT2={lt2_time:.0f}s+{cstr}")
                            
                            print(f"  🔄 FEEDBACK(E18): VT2 {orig_vt2:.0f}→{new_t:.0f}s ({tot} src)")
                        except Exception as ex:
                            fb_log.setdefault("errors", []).append(str(ex))

        # ── Sync downstream engines after feedback ──────────────
        if fb_log.get("vt2_adjusted"):
            adj = fb_log["adjustments"][-1]
            new_t = adj["adjusted_time"]
            
            # E15: cached VT2 VO2%
            e15 = self.results.get("E15", {})
            if e15:
                new_pct = e02.get("vt2_vo2_pct_peak")
                if new_pct is not None:
                    e15["vt2_pct_vo2peak"] = round(new_pct, 1)
                new_hr_pct = e02.get("vt2_hr_pct_max")
                if new_hr_pct is not None:
                    e15["vt2_pct_hrmax"] = round(new_hr_pct, 1)
            
            # E19: temporal alignment VT2 source
            e19_res = self.results.get("E19", {})
            ta_vt2 = e19_res.get("temporal_alignment", {}).get("VT2", {})
            if ta_vt2 and "sources" in ta_vt2:
                ta_vt2["sources"]["VT2"] = new_t
                # Recalc spread
                vals = list(ta_vt2["sources"].values())
                ta_vt2["spread_sec"] = round(max(vals) - min(vals), 1)
                ta_vt2["_feedback_adjusted"] = True
            
            # E18: layer3 VT2 time
            e18_res = self.results.get("E18", {})
            t2_conc = e18_res.get("layer3_concordance", {}).get("threshold_2", {})
            if t2_conc:
                old_vt = t2_conc.get("vt_time_sec", 0)
                lt_t = t2_conc.get("lt_time_sec", 0)
                t2_conc["vt_time_sec"] = new_t
                t2_conc["delta_time_sec"] = round(new_t - lt_t, 1) if lt_t else None
                # Reclassify
                dt = abs(new_t - lt_t) if lt_t else 999
                if dt <= 30:
                    t2_conc["status"] = "EXCELLENT"
                elif dt <= 60:
                    t2_conc["status"] = "GOOD"
                elif dt <= 120:
                    t2_conc["status"] = "ACCEPTABLE"
                else:
                    t2_conc["status"] = "POOR"
                t2_conc["_feedback_adjusted"] = True

        self.results["_feedback"] = fb_log

    def _performance_context(self):
        """
        Performance Context — łączy %VO₂max na progu z prędkością/mocą.
        Źródło: Benítez-Muñoz 2024 (n=1272), Støa 2020, Jones 2021.
        
        Wykorzystuje MAS z testu profilującego (jeśli dostępne) lub
        vMax z testu ergospiro.
        """
        e02 = self.results.get("E02", {})
        e01 = self.results.get("E01", {})
        cfg = self.cfg
        
        if e02.get("status") not in ("OK", "LIMITED"):
            return
        
        ctx = {"executed": True}
        
        # ── Prędkości na progach ─────────────────────────────────
        v_vt1 = e02.get("vt1_speed_kmh")
        v_vt2 = e02.get("vt2_speed_kmh")
        
        # vMax: z testu (E01) lub z protokołu
        v_max_test = e01.get("speed_peak") or e01.get("peak_speed_kmh")
        
        # MAS z testu profilującego (złoty standard)
        mas_ext = getattr(cfg, "mas_m_s", None)  # [m/s]
        mas_kmh = mas_ext * 3.6 if mas_ext else None  # → km/h
        
        # FTP
        ftp = getattr(cfg, "ftp_watts", None)
        
        # MSS
        mss_ext = getattr(cfg, "mss_m_s", None)
        mss_kmh = mss_ext * 3.6 if mss_ext else None
        
        # Wybierz reference velocity
        v_ref = mas_kmh or v_max_test  # MAS > vMax test
        v_ref_source = "MAS_external" if mas_kmh else "vMax_test"
        
        ctx["v_vt1_kmh"] = round(v_vt1, 1) if v_vt1 else None
        ctx["v_vt2_kmh"] = round(v_vt2, 1) if v_vt2 else None
        ctx["v_max_test_kmh"] = round(v_max_test, 1) if v_max_test else None
        ctx["mas_external_kmh"] = round(mas_kmh, 1) if mas_kmh else None
        ctx["mas_external_m_s"] = mas_ext
        ctx["mss_external_m_s"] = mss_ext
        ctx["ftp_watts"] = ftp
        ctx["v_ref_kmh"] = round(v_ref, 1) if v_ref else None
        ctx["v_ref_source"] = v_ref_source
        
        # ── %MAS / %vMax ─────────────────────────────────────────
        if v_ref and v_ref > 0:
            if v_vt1:
                ctx["vt1_pct_vref"] = round(v_vt1 / v_ref * 100, 1)
            if v_vt2:
                ctx["vt2_pct_vref"] = round(v_vt2 / v_ref * 100, 1)
        
        # ── VO₂ context ─────────────────────────────────────────
        vo2_rel = e01.get("vo2_peak_ml_kg_min") or e01.get("vo2_peak_mlkgmin")
        if not vo2_rel:
            vo2_abs = (e02.get("diagnostics", {}).get("vo2_peak") 
                       or e01.get("vo2_peak_ml_min"))
            mass = getattr(cfg, "body_mass_kg", 75)
            if vo2_abs and mass:
                vo2_rel = vo2_abs / mass
        
        vt2_pct_vo2 = e02.get("vt2_vo2_pct_peak")
        vt1_pct_vo2 = e02.get("vt1_vo2_pct_peak")
        
        ctx["vo2max_rel"] = round(vo2_rel, 1) if vo2_rel else None
        ctx["vt1_pct_vo2"] = round(vt1_pct_vo2, 1) if vt1_pct_vo2 else None
        ctx["vt2_pct_vo2"] = round(vt2_pct_vo2, 1) if vt2_pct_vo2 else None
        
        # ── Training level classification ────────────────────────
        # Based on Benítez-Muñoz 2024 (n=1272, treadmill)
        sex = getattr(cfg, "sex", "male")
        modality = getattr(cfg, "modality", "run")
        
        if modality == "run" and v_vt2:
            if sex == "male":
                REF = [
                    ("Sedentary",     0,  9.0, 72, 78),
                    ("Recreational",  9.0, 12.0, 75, 81),
                    ("Trained",      12.0, 15.0, 78, 85),
                    ("Well-trained", 15.0, 18.0, 82, 88),
                    ("Elite",        18.0, 99.0, 85, 93),
                ]
            else:
                REF = [
                    ("Sedentary",     0,  7.0, 72, 78),
                    ("Recreational",  7.0, 10.0, 75, 81),
                    ("Trained",      10.0, 14.0, 78, 85),
                    ("Well-trained", 14.0, 17.0, 82, 88),
                    ("Elite",        17.0, 99.0, 85, 93),
                ]
            
            # Classify by vVT2
            level_v = "Unknown"
            for name, lo, hi, _, _ in REF:
                if lo <= v_vt2 < hi:
                    level_v = name
                    break
            
            # Classify by %VO₂max
            level_pct = "Unknown"
            if vt2_pct_vo2:
                for name, _, _, plo, phi in REF:
                    if plo <= vt2_pct_vo2 < phi:
                        level_pct = name
                        break
                if vt2_pct_vo2 >= REF[-1][3]:
                    level_pct = "Elite"
            
            ctx["level_by_speed"] = level_v
            ctx["level_by_pct_vo2"] = level_pct
            ctx["levels_match"] = (level_v == level_pct)
            
            # ── Red flags / insights ─────────────────────────────
            flags = []
            
            # High %VO₂ but low speed → low VO₂max, not elite threshold
            if vt2_pct_vo2 and vt2_pct_vo2 > 88 and v_vt2 < (12 if sex == "male" else 10):
                flags.append("HIGH_PCT_LOW_SPEED: VT2 wysoki %VO₂max ale niska prędkość absolutna — "
                           "sugeruje niski VO₂max, nie elitowy próg")
            
            # Low %VO₂ but high speed → very economic runner
            if vt2_pct_vo2 and vt2_pct_vo2 < 78 and v_vt2 > (15 if sex == "male" else 13):
                flags.append("LOW_PCT_HIGH_SPEED: VT2 niski %VO₂max ale wysoka prędkość — "
                           "możliwy artefakt detekcji VT2 lub bardzo ekonomiczny biegacz")
            
            # %MAS context
            if ctx.get("vt2_pct_vref") and mas_kmh:
                pct_mas = ctx["vt2_pct_vref"]
                if pct_mas > 95:
                    flags.append(f"VT2_NEAR_MAS: VT2 at {pct_mas:.0f}%MAS — bardzo blisko MAS, "
                               "niewielka rezerwa anaerobowa")
                elif pct_mas < 75:
                    flags.append(f"VT2_LOW_MAS: VT2 at {pct_mas:.0f}%MAS — duży potencjał do poprawy "
                               "treningu progowego")
                
                # ASR (Anaerobic Speed Reserve) = MSS - MAS
                if mss_kmh:
                    asr = mss_kmh - mas_kmh
                    asr_pct = (v_vt2 - mas_kmh) / asr * 100 if asr > 0 else None
                    ctx["asr_kmh"] = round(asr, 1)
                    ctx["vt2_in_asr_pct"] = round(asr_pct, 1) if asr_pct else None
            
            ctx["flags"] = flags
            
            # ── Interpretation text ──────────────────────────────
            parts = []
            parts.append(f"VT2 at {v_vt2:.1f} km/h")
            if vt2_pct_vo2:
                parts.append(f"({vt2_pct_vo2:.0f}% VO₂max)")
            parts.append(f"→ poziom **{level_v}**")
            
            if mas_kmh:
                pct = ctx.get("vt2_pct_vref", 0)
                parts.append(f"= {pct:.0f}% MAS ({mas_ext:.2f} m/s)")
            
            if level_v != level_pct and level_pct != "Unknown":
                parts.append(f"[uwaga: %VO₂max sugeruje '{level_pct}']")
            
            ctx["interpretation"] = " ".join(parts)
            
            # VT1 interpretation
            if v_vt1:
                vt1_parts = [f"VT1 at {v_vt1:.1f} km/h"]
                if vt1_pct_vo2:
                    vt1_parts.append(f"({vt1_pct_vo2:.0f}% VO₂max)")
                if ctx.get("vt1_pct_vref"):
                    vt1_parts.append(f"= {ctx['vt1_pct_vref']:.0f}% MAS")
                ctx["vt1_interpretation"] = " ".join(vt1_parts)
        
        self.results["_performance_context"] = ctx
        
        if ctx.get("interpretation"):
            print(f"  📊 CONTEXT: {ctx['interpretation']}")

    def process_file(self, filename: str) -> Dict[str, Any]:
        print(f"\n🚀 START PIPELINE: Analiza pliku '{filename}'")
        self.results = {}

        # 0-3 preprocessing
        try:
            try:
                df = pd.read_csv(filename)
            except Exception:
                df = pd.read_csv(filename, sep=';')

            self.raw = DataTools.canonicalize(df)
            
            # Protocol resolution: AUTO → detect from data
            resolved_protocol = self.cfg.protocol_name
            if resolved_protocol == 'AUTO' or not resolved_protocol:
                try:
                    from engine_core import auto_detect_protocol
                    detected, conf, _ = auto_detect_protocol(self.raw)
                    resolved_protocol = detected if detected and conf >= 0.60 else 'RUN_RAMP'
                except Exception:
                    resolved_protocol = 'RUN_RAMP'
                self.cfg.protocol_name = resolved_protocol
            
            # Try marker-based segments first, then template
            try:
                from engine_core import build_protocol_from_markers
                marker_segs = build_protocol_from_markers(self.raw)
                if marker_segs and len(marker_segs) >= 4:
                    segments = compile_protocol_for_apply(marker_segs)
                else:
                    segments = PROTOCOLS_DB.get(resolved_protocol, [])
            except Exception:
                segments = PROTOCOLS_DB.get(resolved_protocol, [])
            
            df_patched = DataTools.apply_protocol(self.raw, segments) if segments else self.raw
            self.processed = DataTools.smooth(df_patched, self.cfg)
        except Exception as e:
            print(f"❌ ERROR (Import/Preproc): {e}")
            return {"fatal_error": str(e)}

        # E00
        self.results["E00"] = self._safe_run("E00", Engine_E00_StopDetection.run, self.processed, self.cfg)
        if self.results["E00"].get("status") == "ERROR":
            return {"fatal_error": self.results["E00"].get("reason", "E00 error"), "E00": self.results["E00"]}

        t_stop = self.results["E00"]["t_stop"]
        df_ex = self.processed[self.processed["Time_sec"] <= t_stop].copy()
        self.results["_df_ex"] = df_ex
        self.results["_df_full"] = self.processed  # full test including recovery
        df_full = self.processed.copy()

        # Engines
        self.results["E01"] = self._safe_run("E01", Engine_E01_GasExchangeQC.run, df_ex)
        # E02 v4: pass file_metadata for fallback thresholds
        _file_meta = {}
        for _mk in ["VT1_HR", "VT2_HR", "VT1_VO2_ml_min", "VT2_VO2_ml_min"]:
            if hasattr(self, "raw") and self.raw is not None and _mk in self.raw.columns:
                _v = self.raw[_mk].dropna()
                if len(_v) > 0:
                    try: _file_meta[_mk] = float(_v.iloc[0])
                    except: pass
        self.results["E02"] = self._safe_run("E02", Engine_E02_Thresholds_v4.run, df_ex, self.results["E00"], _file_meta)
        self._apply_manual_vt_override(df_ex)
        self.results["E03"] = self._safe_run("E03", Engine_E03_VentSlope.run, df_ex, self.results.get("E02", {}), getattr(self.cfg, "age_y", None), getattr(self.cfg, "height_cm", None), getattr(self.cfg, "sex", "male"))
        # E04 v2: OUES with submaximal variants, normalization, predicted values
        _e04_meta = {}
        for _k, _attrs in [("weight_kg", ["body_mass_kg", "weight_kg"]),
                           ("height_cm", ["height_cm"]),
                           ("age", ["age_y", "age"]),
                           ("sex", ["sex"])]:
            for _a in _attrs:
                _v = getattr(self.cfg, _a, None)
                if _v is not None:
                    _e04_meta[_k] = _v
                    break
        self.results["E04"] = self._safe_run("E04", Engine_E04_OUES_v2.run,
            df_ex, self.results.get("E00", {}), self.results.get("E02", {}), _e04_meta)
        self.results["E05"] = self._safe_run("E05", Engine_E05_O2Pulse.run,
            df_ex, self.results.get("E02", {}),
            getattr(self.cfg, "age_y", None),
            getattr(self.cfg, "sex", "male"),
            self.results.get("E01", {}).get("hr_peak"))
        _e06_kw = dict(df_ex=df_ex, modality=getattr(self.cfg,"modality","run"), e02=self.results.get("E02",{}), e01=self.results.get("E01",{}), weight_kg=getattr(self.cfg,"body_mass_kg",None))
        self.results["E06"] = self._safe_run("E06", Engine_E06_Gain_v2.run, **_e06_kw)
        self.results["E07"] = self._safe_run("E07", Engine_E07_BreathingPattern.run, df_ex, self.results.get("E02"), self.results.get("E01"), self.cfg)
        self.results["E08"] = self._safe_run("E08", Engine_E08_CardioHRR.run, df_full, t_stop)
        self.results["E09"] = self._safe_run("E09", Engine_E09_VentLimitation.run, df_ex)
        # E10 v2: full substrate oxidation profile
        _e10_kw = dict(
            df_ex=df_ex,
            e02=self.results.get("E02", {}),
            e01=self.results.get("E01", {}),
            sex=getattr(self.cfg, "sex", "male"),
            weight_kg=getattr(self.cfg, "body_mass_kg", None),
        )
        self.results["E10"] = self._safe_run("E10", Engine_E10_Substrate_v2.run, **_e10_kw)
        self.results["E11"] = self._safe_run("E11", Engine_E11_Lactate.run, self.processed, getattr(self, "_lactate_input", None), self.results.get("E00"), self.results.get("E01"), self.results.get("E02"), self.cfg)
        self.results["E12"] = self._safe_run("E12", Engine_E12_NIRS.run, self.processed, self.results.get("E02"), self.results.get("E01"), self.results.get("E00"), self.cfg)
        self.results["E13"] = self._safe_run("E13", Engine_E13_Drift.run, self.processed, self.results.get("E00"), self.results.get("E01"), self.results.get("E02"), self.cfg)
        self.results["E14"] = self._safe_run("E14",
            Engine_E14_Kinetics.run,
            self.results,
            {"_df_processed": self.processed, "_acfg": self.cfg})
        # E14 diagnostic
        _e14r = self.results.get("E14", {})
        print(f"  🔬 E14 result: mode={_e14r.get('mode','?')}, status={_e14r.get('status','?')}, stages={len(_e14r.get('stages',[]))}")
        print(f"  🔬 Config: protocol={self.cfg.protocol_name}, speeds={getattr(self.cfg, 'kinetics_speeds_kmh', 'MISSING')}")
        self.results["E15"] = self._safe_run("E15", Engine_E15_Normalization.run, self.results, self.cfg.body_mass_kg, getattr(self.cfg, "age_y", None), getattr(self.cfg, "sex", "male"), getattr(self.cfg, "modality", "run"), getattr(self.cfg, "height_cm", None))

        # E18: VT↔LT Cross-Validation (requires E02 + E11)
        self.results["E18"] = self._safe_run("E18",
            Engine_E18_VT_LT_CrossValidation.run,
            self.results.get("E02", {}),
            self.results.get("E11", {}),
            self.results.get("E01", {}),
            df_ex)

        # E19: Test Validity + Physiological Concordance
        self.results["E19"] = self._safe_run("E19",
            Engine_E19_Concordance.run,
            self.results,
            self.cfg)

        # E21: Kinetic Phenotype (cross-engine: E14 × E02 × E01 × E18)
        self.results["E21"] = self._safe_run("E21",
            Engine_E21_KineticPhenotype.run,
            self.results,
            self.cfg)

        # E22: Cross-Engine Correlation Analysis (all phases)
        try:
            from e22_cross_correlation import Engine_E22_CrossCorrelation
            self.results["E22"] = Engine_E22_CrossCorrelation.run(
                self.results,
                {'_df_processed': self.processed, '_acfg': self.cfg})
        except Exception as _e22_err:
            self.results["E22"] = {"status": "ERROR", "reason": str(_e22_err)}

        # ── FEEDBACK LOOP: post-validation threshold adjustment ──
        try:
            self._feedback_loop(df_ex)
            self._performance_context()
        except Exception as _fb_err:
            self.results["_feedback"] = {"executed": False, "error": str(_fb_err)}

        vt1 = self.results.get("E02", {}).get("vt1_hr")
        vt2 = self.results.get("E02", {}).get("vt2_hr")
        hr_max = self.results.get("E01", {}).get("hr_peak")
        if vt1 is None or vt2 is None or hr_max is None or not (vt1 < vt2 <= hr_max):
            self.results["E16"] = {"status": "LIMITED", "reason": f"vt1={vt1}, vt2={vt2}, hr_max={hr_max}"}
        else:
            self.results["E17"] = self._safe_run("E17", Engine_E17_GasExchange.run, self.processed, self.results.get("E00"), self.results.get("E01"), self.results.get("E02"), self.cfg)
            # E16 v2: full zone model with speed/VO2 interpolation
            _e02 = self.results.get("E02", {})
            _e16_kw = dict(
                vt1_hr=vt1, vt2_hr=vt2, hr_max=hr_max,
                hr_rest=getattr(self.cfg, 'hr_rest', None),
                vt1_speed=_e02.get('vt1_speed_kmh', _e02.get('vt1_speed')),
                vt2_speed=_e02.get('vt2_speed_kmh', _e02.get('vt2_speed')),
                max_speed=self.results.get("E00", {}).get("peak_speed"),
                vt1_vo2=_e02.get('vt1_vo2_ml'),
                vt2_vo2=_e02.get('vt2_vo2_ml'),
                vo2_peak=self.results.get("E01", {}).get("vo2_peak_ml_min"),
                df_ex=df_ex,
            )
            self.results["E16"] = self._safe_run("E16", Engine_E16_Zones_v2.run, **_e16_kw)

        # final export
        outputs = self.build_outputs()
        trainer_canon_flat = self.build_trainer_canon_flat(outputs)

        # Raport tekstowy (T12 template)
        try:
            canon_table = ReportAdapter.build_canon_table(self.processed, self.results, self.cfg)
            text_report = ReportAdapter.render_text_report(canon_table)
            html_report = ReportAdapter.render_html_report(canon_table)
            html_report_lite = ReportAdapter.render_lite_html_report(canon_table)
        except Exception as e:
            canon_table = {}
            text_report = f"[RAPORT NIEDOSTĘPNY: {e}]"
            html_report = f"<html><body><h1>Raport niedostępny</h1><p>{e}</p></body></html>"
            html_report_lite = html_report

        # Kinetics report (if CWR kinetics protocol)
        html_report_kinetics = None
        try:
            e14_data = self.results.get('E14', {})
            _e14_mode = e14_data.get('mode', 'NONE')
            _e14_stages = len(e14_data.get('stages', []))
            print(f"  🔬 Kinetics check: E14.mode={_e14_mode}, stages={_e14_stages}")
            if _e14_mode == 'CWR_KINETICS' and e14_data.get('stages'):
                from report import render_kinetics_report, generate_kinetics_charts, inject_kinetics_charts
                _kin_ct = dict(canon_table) if canon_table else {}
                _kin_ct['sport'] = getattr(self.cfg, 'sport', '') or getattr(self.cfg, 'modality', 'run') or 'run'
                html_report_kinetics = render_kinetics_report(self.results, _kin_ct, self.processed)
                print(f"  🔬 Kinetics report rendered: {len(html_report_kinetics)} chars")
                # Generate and inject charts
                _kin_charts = generate_kinetics_charts(self.processed, self.results)
                if _kin_charts:
                    html_report_kinetics = inject_kinetics_charts(html_report_kinetics, _kin_charts)
                    print(f"  🔬 Kinetics charts injected: {list(_kin_charts.keys())}")
            else:
                print(f"  🔬 Kinetics report SKIPPED: mode={_e14_mode}, stages={_e14_stages}")
        except Exception as e:
            print(f"⚠️ Kinetics report generation error: {e}")
            import traceback; traceback.print_exc()
            self.results["_kinetics_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        self._last_report = {
            "outputs_calc_only": outputs,
            "trainer_canon_flat": trainer_canon_flat,
            "canon_table": canon_table,
            "text_report": text_report, "html_report": html_report,
            "html_report_lite": html_report_lite,
            "html_report_kinetics": html_report_kinetics,
            "raw_results": {**self.results,
                "_config_protocol": self.cfg.protocol_name,
                "_config_speeds": str(getattr(self.cfg, 'kinetics_speeds_kmh', 'MISSING')),
                "_qc_log": self._qc_log,
            }
        }
        return self._last_report

    def save_html_report(self, path: str = None):
        """Save HTML report to file. If path is None, auto-generate from athlete name."""
        if not hasattr(self, '_last_report') or not self._last_report:
            print("⚠️ Najpierw uruchom process_file().")
            return None
        html = self._last_report.get('html_report', '')
        if not html or html.startswith('[RAPORT'):
            print("⚠️ Raport HTML niedostępny.")
            return None
        if path is None:
            name = getattr(self.cfg, 'athlete_name', 'athlete').replace(' ', '_')
            path = f"CPET_Report_{name}.html"

        # Wrap in full HTML document with UTF-8
        full_html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CPET Report — {getattr(self.cfg, 'athlete_name', '')}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8fafc; }}
  @media print {{ body {{ background: white; max-width: 100%; }} }}
</style>
</head>
<body>
{html}
<footer style="margin-top:30px;padding:15px;text-align:center;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;">
  CPET Analysis Engine v2.0 &bull; Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</body>
</html>"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"✅ Raport HTML zapisany: {path}")
        return path
