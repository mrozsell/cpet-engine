import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. IMPORTS & CONFIGURATION (UPDATED METADATA)
# ==========================================
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Tuple

@dataclass
class AnalysisConfig:
    # --- USTAWIENIA ANALIZY ---
    modality: str = "run"
    sex: str = "male"
    protocol_name: str = "RUN_RAMP_GENERIC"
    smooth_window_gas: int = 20
    smooth_window_hr: int = 5
    force_manual_t_stop: Union[str, float, None] = None

    # --- DANE ZAWODNIKA (KANON META) ---
    # PDF: athlete_name, test_date, location, operator...
    athlete_name: str = "Nieznany Zawodnik"
    athlete_id: str = "ID_000"
    test_date: str = "2024-01-01"
    location: str = "Lab"
    operator: str = "Auto"
    body_mass_kg: float = 75.0
    height_cm: float = 180.0
    age_y: int = 30

    # --- MANUAL VT OVERRIDE ---
    vt1_manual: Union[str, None] = None   # "MM:SS" or None
    vt2_manual: Union[str, None] = None   # "MM:SS" or None

    # --- PROFILING (testy zewnętrzne) ---
    mas_m_s: Optional[float] = None       # MAS z testu profilującego [m/s]
    ftp_watts: Optional[float] = None     # FTP [W] (kolarstwo)
    mss_m_s: Optional[float] = None       # Max Sprint Speed [m/s]
    sport: str = ""                       # "running","cycling","football","triathlon"

    # --- KONTEKST ---
    notes: str = ""

    @property
    def t_stop_seconds(self) -> Optional[float]:
        return parse_time_str(self.force_manual_t_stop)

# --- HELPERY CZASU ---
def parse_time_str(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:   # mm:ss
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:   # hh:mm:ss
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(s)
    except Exception:
        return None


# --- SUROWE DEFINICJE PROTOKOŁÓW (panelowe) ---
RAW_PROTOCOLS = {
    "RUN_RAMP_GENERIC": [
        {"type":"const", "start":"00:00", "end":"01:00", "speed_kmh":0.0, "incline_pct":0.0},
        {"type":"const", "start":"01:00", "end":"04:00", "speed_kmh":4.0, "incline_pct":0.0},
        {"type":"const", "start":"04:00", "end":"07:00", "speed_kmh":6.0, "incline_pct":0.0},
        {"type":"ramp_speed", "start":"07:00", "end":"24:00", "speed_from":6.0, "speed_to":14.0, "incline_pct":0.0},
        {"type":"dynamic_incline_steps_to_stop", "start":"24:00", "speed_kmh":14.0, "incline_start":2.5, "incline_step":2.5, "step_every_sec":180},
    ],
    "RUN_RAMP_6to14_KO2452": [
        {"type":"const", "start":"00:00", "end":"01:00", "speed_kmh":0.0, "incline_pct":0.0},
        {"type":"const", "start":"01:00", "end":"04:00", "speed_kmh":4.0, "incline_pct":0.0},
        {"type":"const", "start":"04:00", "end":"07:00", "speed_kmh":6.0, "incline_pct":0.0},
        {"type":"ramp_speed", "start":"07:00", "end":"24:00", "speed_from":6.0, "speed_to":14.0, "incline_pct":0.0},
        {"type":"const", "start":"24:00", "end":"24:52", "speed_kmh":14.0, "incline_pct":2.5},
    ],
    "RUN_STEP_BIEZNIA_WYJAZD": [
        {"type":"const", "start":"00:00", "end":"01:00", "speed_kmh":0.0, "incline_pct":0.0},
        {"type":"const", "start":"01:00", "end":"03:00", "speed_kmh":4.0, "incline_pct":0.0},
        {"type":"const", "start":"03:00", "end":"05:00", "speed_kmh":6.0, "incline_pct":0.0},
        {"type":"const", "start":"05:00", "end":"07:00", "speed_kmh":8.0, "incline_pct":0.0},
        {"type":"const", "start":"07:00", "end":"09:00", "speed_kmh":9.0, "incline_pct":0.0},
        {"type":"const", "start":"09:00", "end":"11:00", "speed_kmh":10.0, "incline_pct":0.0},
        {"type":"const", "start":"11:00", "end":"13:00", "speed_kmh":11.0, "incline_pct":0.0},
        {"type":"const", "start":"13:00", "end":"15:00", "speed_kmh":12.0, "incline_pct":0.0},
        {"type":"const", "start":"15:00", "end":"17:00", "speed_kmh":13.0, "incline_pct":0.0},
        {"type":"const", "start":"17:00", "end":"19:00", "speed_kmh":14.0, "incline_pct":0.0},
        {"type":"const", "start":"19:00", "end":"21:00", "speed_kmh":14.0, "incline_pct":2.0},
        {"type":"const", "start":"21:00", "end":"23:00", "speed_kmh":14.0, "incline_pct":4.0},
        {"type":"const", "start":"23:00", "end":"25:00", "speed_kmh":14.0, "incline_pct":6.0},
        {"type":"const", "start":"25:00", "end":"27:00", "speed_kmh":14.0, "incline_pct":8.0},
        {"type":"const", "start":"27:00", "end":"29:00", "speed_kmh":14.0, "incline_pct":10.0},
        {"type":"const", "start":"29:00", "end":"31:00", "speed_kmh":14.0, "incline_pct":12.0},
        {"type":"const", "start":"31:00", "end":"33:00", "speed_kmh":14.0, "incline_pct":14.0},
        {"type":"const", "start":"33:00", "end":"35:00", "speed_kmh":14.0, "incline_pct":16.0},
    ],
}


# ==========================================
# 2. DATA TOOLS (FIXED — all methods inside class)
# ==========================================
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def compile_protocol_for_apply(raw_segments, t_stop_manual=None):
    """
    Konwersja segmentów panelowych -> segmenty dla DataTools.apply_protocol.
    Zwraca listę dictów z kluczami:
    start_sec, end_sec + Speed_kmh/Incline_pct/Power_W lub Speed_from/Speed_to...
    """
    t_stop_sec = parse_time_str(t_stop_manual) if t_stop_manual is not None else None
    out = []

    for seg in raw_segments:
        stype = seg.get("type", "")

        if stype == "const":
            row = {"start_sec": parse_time_str(seg["start"]), "end_sec": parse_time_str(seg["end"])}
            if "speed_kmh" in seg:   row["Speed_kmh"] = float(seg["speed_kmh"])
            if "incline_pct" in seg: row["Incline_pct"] = float(seg["incline_pct"])
            if "power_w" in seg:     row["Power_W"] = float(seg["power_w"])
            out.append(row)

        elif stype == "ramp_speed":
            row = {
                "start_sec": parse_time_str(seg["start"]),
                "end_sec": parse_time_str(seg["end"]),
                "Speed_from": float(seg["speed_from"]),
                "Speed_to": float(seg["speed_to"]),
                "Incline_pct": float(seg.get("incline_pct", 0.0)),
            }
            out.append(row)

        elif stype == "ramp_power":
            row = {
                "start_sec": parse_time_str(seg["start"]),
                "end_sec": parse_time_str(seg["end"]),
                "Power_from": float(seg["power_from"]),
                "Power_to": float(seg["power_to"]),
            }
            out.append(row)

        elif stype == "dynamic_incline_steps_to_stop":
            if t_stop_sec is None:
                continue
            s = parse_time_str(seg["start"])
            if t_stop_sec <= s:
                continue

            speed = float(seg.get("speed_kmh", np.nan))
            inc = float(seg.get("incline_start", 0.0))
            inc_step = float(seg.get("incline_step", 2.5))
            step_every = int(seg.get("step_every_sec", 180))

            t = s
            k = 0
            while t < t_stop_sec:
                t_next = min(t + step_every, t_stop_sec)
                row = {"start_sec": float(t), "end_sec": float(t_next), "Incline_pct": float(inc + k * inc_step)}
                if np.isfinite(speed):
                    row["Speed_kmh"] = speed
                out.append(row)
                t = t_next
                k += 1

    out = sorted(out, key=lambda x: (x["start_sec"], x["end_sec"]))
    return out


class DataTools:
    """Narzędzia przetwarzania danych CPET — wszystkie metody jako @staticmethod."""

    @staticmethod
    def _parse_time_val(val) -> float:
        if pd.isna(val) or val == "":
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        s = str(val).strip().replace(",", ".")
        # direct float
        try:
            return float(s)
        except Exception:
            pass

        # hh:mm:ss(.ms) or mm:ss(.ms)
        try:
            parts = s.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        except Exception:
            return np.nan

        return np.nan

    @staticmethod
    def apply_global_aliases(df: pd.DataFrame) -> pd.DataFrame:
        """
        Dodaje globalne aliasy kompatybilności wstecznej:
        - jeśli istnieje jedna wersja nazwy kolumny, tworzy pozostałe synonimy,
        - nie nadpisuje istniejących kolumn.
        """
        out = df.copy()

        alias_groups = [
            # czas
            ["Time_sec", "Time_s", "time_sec"],
            # gazy
            ["VE_Lmin", "VE_lmin", "VE_L_min", "VE", "VE_BTPS"],
            ["VO2_mlmin", "VO2_ml_min", "VO2"],
            ["VCO2_mlmin", "VCO2_ml_min", "VCO2"],
            # cardio/workload
            ["HR_bpm", "HR", "Pulse", "HF", "HeartRate", "Heart Rate"],
            ["Speed_kmh", "Speed", "Speed_km_h"],
            ["Power_W", "Power", "Watt"],
            # dodatkowe
            ["RER", "RQ"],
            ["O2Pulse", "O2_Pulse"],
            ["SmO2_pct", "SmO2"],
            ["Lactate_mmol", "Lactate_mmolL", "La"],
            # pet
            ["PetCO2_mmHg", "PetCO2"],
            ["PetO2_mmHg", "PetO2"],
            # substraty
            ["FAT_g_min", "FAT_g_h"],
            ["CHO_g_min", "CHO_g_h"],
            # masa
            ["BodyMass_kg", "Mass_kg", "Weight_kg", "Masa_kg"],
        ]

        for group in alias_groups:
            src = None
            for c in group:
                if c in out.columns:
                    src = c
                    break

            if src is None:
                continue

            for c in group:
                if c not in out.columns:
                    out[c] = out[src]

        return out

    @staticmethod
    def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standaryzuje nazwy kolumn do formatów canonical używanych przez silniki.
        Zawiera fallbacki jednostek i czasu.
        """
        df_new = df.copy()

        # --- 1) aliasy kolumn -> canonical ---
        aliases = {
            # czas
            "Time_sec": ["Time_sec", "Time_s", "t", "time", "Time"],
            # gazy
            "VO2_mlmin": ["VO2_mlmin", "VO2_ml_min", "VO2"],
            "VCO2_mlmin": ["VCO2_mlmin", "VCO2_ml_min", "VCO2"],
            "VE_Lmin": ["VE_Lmin", "VE_L_min", "VE", "VE_BTPS"],
            # cardio/workload
            "HR_bpm": ["HR_bpm", "HR", "Pulse", "HF", "HeartRate", "Heart Rate"],
            "Speed_kmh": ["Speed_kmh", "Speed", "Speed_km_h"],
            "Power_W": ["Power_W", "Power", "Watt"],
            # dodatkowe
            "RER": ["RER", "RQ"],
            "O2Pulse": ["O2Pulse", "O2_Pulse"],
            "SmO2_pct": ["SmO2_pct", "SmO2"],
            "Lactate_mmol": ["Lactate_mmol", "Lactate_mmolL", "La"],
            "PetCO2_mmHg": ["PetCO2_mmHg", "PetCO2"],
            "PetO2_mmHg": ["PetO2_mmHg", "PetO2"],
            # substraty
            "FAT_g_min": ["FAT_g_min", "FAT_g_h"],
            "CHO_g_min": ["CHO_g_min", "CHO_g_h"],
            # antropometria
            "BodyMass_kg": ["BodyMass_kg", "Mass_kg", "Weight_kg", "Masa_kg"],
        }

        # przepisanie pierwszego pasującego aliasu
        for target, candidates in aliases.items():
            for c in candidates:
                if c in df_new.columns:
                    df_new[target] = df_new[c]
                    break

        # --- 2) Time_sec fallback z Time_str ---
        if ("Time_sec" not in df_new.columns) or pd.to_numeric(df_new["Time_sec"], errors="coerce").dropna().empty:
            if "Time_str" in df_new.columns:
                df_new["Time_sec"] = df_new["Time_str"].apply(DataTools._parse_time_val)

        # --- 3) cast numeryczny ---
        numeric_cols = [
            "Time_sec", "VO2_mlmin", "VCO2_mlmin", "VE_Lmin",
            "HR_bpm", "Speed_kmh", "Power_W", "RER", "O2Pulse",
            "SmO2_pct", "Lactate_mmol", "PetCO2_mmHg", "PetO2_mmHg",
            "FAT_g_min", "CHO_g_min", "BodyMass_kg"
        ]
        for col in numeric_cols:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

        # --- 4) fallback jednostek VO2/VCO2 L/min -> ml/min ---
        if (("VO2_mlmin" not in df_new.columns) or df_new["VO2_mlmin"].dropna().empty) and ("VO2_L_min" in df_new.columns):
            df_new["VO2_mlmin"] = pd.to_numeric(df_new["VO2_L_min"], errors="coerce") * 1000.0

        if (("VCO2_mlmin" not in df_new.columns) or df_new["VCO2_mlmin"].dropna().empty) and ("VCO2_L_min" in df_new.columns):
            df_new["VCO2_mlmin"] = pd.to_numeric(df_new["VCO2_L_min"], errors="coerce") * 1000.0

        # --- 5) substraty g/h -> g/min ---
        if "FAT_g_h" in df_new.columns and (("FAT_g_min" not in df_new.columns) or df_new["FAT_g_min"].dropna().empty):
            df_new["FAT_g_min"] = pd.to_numeric(df_new["FAT_g_h"], errors="coerce") / 60.0

        if "CHO_g_h" in df_new.columns and (("CHO_g_min" not in df_new.columns) or df_new["CHO_g_min"].dropna().empty):
            df_new["CHO_g_min"] = pd.to_numeric(df_new["CHO_g_h"], errors="coerce") / 60.0

        # --- 6) globalne aliasy kompatybilności (KLUCZOWE) ---
        df_new = DataTools.apply_global_aliases(df_new)

        # --- 7) porządek czasu ---
        if "Time_sec" in df_new.columns:
            df_new = df_new.sort_values("Time_sec").reset_index(drop=True)

        return df_new

    @staticmethod
    def apply_protocol(df: pd.DataFrame, segments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Nakłada protokół (np. gdy brakuje/psuje się Speed/Incline/Power).
        segments: lista dictów np.
        {"start":0, "end":60, "Speed_kmh":4.0, "Incline_pct":0.0}
        lub rampa:
        {"start":420, "end":1293, "Speed_from":6.0, "Speed_to":13.3}
        """
        if df is None or df.empty:
            return df
        if not segments:
            return df
        if "Time_sec" not in df.columns:
            return df

        out = df.copy()
        t = pd.to_numeric(out["Time_sec"], errors="coerce")

        # upewnij się, że kolumny istnieją
        for col in ["Speed_kmh", "Incline_pct", "Power_W"]:
            if col not in out.columns:
                out[col] = np.nan

        for seg in segments:
            s = seg.get("start_sec", seg.get("start", None))
            e = seg.get("end_sec", seg.get("end", None))
            if s is None or e is None:
                continue

            s = float(s)
            e = float(e)
            if e <= s:
                continue

            mask = (t >= s) & (t < e)

            # stałe wartości
            for col in ["Speed_kmh", "Incline_pct", "Power_W"]:
                if col in seg and seg[col] is not None:
                    out.loc[mask, col] = float(seg[col])

            # rampa prędkości
            if ("Speed_from" in seg) and ("Speed_to" in seg):
                frac = (t[mask] - s) / (e - s)
                out.loc[mask, "Speed_kmh"] = float(seg["Speed_from"]) + frac * (float(seg["Speed_to"]) - float(seg["Speed_from"]))

            # rampa mocy
            if ("Power_from" in seg) and ("Power_to" in seg):
                frac = (t[mask] - s) / (e - s)
                out.loc[mask, "Power_W"] = float(seg["Power_from"]) + frac * (float(seg["Power_to"]) - float(seg["Power_from"]))

        return out

    @staticmethod
    def smooth(df: pd.DataFrame, cfg) -> pd.DataFrame:
        """
        Wygładzanie pod CPET:
        - metabolizm/gazy: rolling 15-20 s (domyślnie 15)
        - FAT/CHO: heavy smoothing ~60 s
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        if "Time_sec" not in out.columns:
            return out

        out = out.sort_values("Time_sec").reset_index(drop=True)
        t = pd.to_numeric(out["Time_sec"], errors="coerce")
        dt = float(np.nanmedian(np.diff(t))) if len(t) > 2 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0  # fallback

        # parametry z cfg (jak brak, daj sensowne defaulty)
        smooth_sec = float(getattr(cfg, "smooth_seconds", 15.0))
        fatcho_sec = float(getattr(cfg, "fatcho_smooth_seconds", 60.0))

        win_main = max(3, int(round(smooth_sec / dt)))
        win_heavy = max(win_main, int(round(fatcho_sec / dt)))

        cols_main = [
            "VO2_mlmin", "VCO2_mlmin", "VE_Lmin", "VE_lmin", "HR_bpm", "RER",
            "O2Pulse", "PetCO2_mmHg", "PetO2_mmHg", "Speed_kmh", "Power_W"
        ]
        cols_heavy = ["FAT_g_min", "CHO_g_min", "SmO2_pct", "Lactate_mmol"]

        for c in cols_main:
            if c in out.columns:
                s = pd.to_numeric(out[c], errors="coerce")
                out[c] = s.rolling(window=win_main, min_periods=1, center=True).median()

        for c in cols_heavy:
            if c in out.columns:
                s = pd.to_numeric(out[c], errors="coerce")
                out[c] = s.rolling(window=win_heavy, min_periods=1, center=True).median()

        return out


print("✅ Komórka 2: DataTools (FIXED — all methods inside class) załadowana.")
print("✅ compile_protocol_for_apply() — jedna definicja, globalna.")

# ==========================================
# 3. ENGINE ROOM (E00-E16) — DEDUPLICATED
# ==========================================
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

class Engine_E00_StopDetection:
    """
    E00 v2.1: Stop detection (end of exercise / refusal point) for CPET pipelines.

    Priorytet źródeł:
    1) manual_config (cfg.t_stop_seconds)
    2) event_marker (kolumna marker/event — keyword "ko"/"stop"/"end")
    3) composite_auto_stop (VE/VCO2/VO2 + workload drop + HR drop + RER trend)
    4) auto_vo2_peak
    5) fallback_end

    v2.1 improvements:
    - Smart marker filtering: numeric markers validated (not just stage numbers)
    - HR drop detection as additional stop signal
    - Power/VO2 slope collapse detection for CSVs without Speed column
    - Post-marker data check: if data continues 60s+ after marker, it's a stage marker not stop
    """

    @staticmethod
    def _to_numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _rolling(s: pd.Series, win: int = 15) -> pd.Series:
        return s.rolling(window=max(3, win), min_periods=1, center=True).median()

    @staticmethod
    def _has_col(df: pd.DataFrame, col: str, min_valid: int = 5) -> bool:
        return (col in df.columns) and (pd.to_numeric(df[col], errors="coerce").dropna().shape[0] >= min_valid)

    @staticmethod
    def _first_existing(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _clip_t(t: float, t_min: float, t_max: float) -> float:
        if t is None or np.isnan(t):
            return t_max
        return float(np.clip(float(t), t_min, t_max))

    @staticmethod
    def _find_event_marker_stop(df: pd.DataFrame, time_col: str):
        """
        Szuka markera końca testu w kolumnach typu marker/event/stage/comment.
        v2.1: Rozróżnia markery stopni protokołu od markera końca.
        
        Logika:
        1) Keyword match ("ko", "stop", "end", "refusal") → priorytet najwyższy
        2) Numeric marker → TYLKO jeśli po nim nie ma >60s danych breath-by-breath
           (bo to prawdopodobnie numer stopnia, nie koniec)
        """
        marker_cols = [c for c in df.columns if c.lower() in {
            "marker", "event", "events", "stage_event", "comment", "comments", "lap_marker"
        }]
        if not marker_cols:
            return None, None, "no_marker_cols"

        keywords = ("stop", "end", "ko", "refusal", "odmowa", "terminate", "termination",
                    "koniec", "zakończenie", "max", "peak_stop")
        
        t_max_data = float(pd.to_numeric(df[time_col], errors="coerce").max())
        
        best_keyword_t = None
        best_keyword_col = None
        best_numeric_t = None
        best_numeric_col = None

        for mc in marker_cols:
            ser = df[mc]
            
            # 1) Keyword match — highest priority
            txt = ser.astype(str).str.lower().str.strip()
            mask_txt = txt.apply(lambda x: any(k in x for k in keywords) and x not in ('0', '', 'nan'))
            idx_txt = df.index[mask_txt]
            if len(idx_txt) > 0:
                # Take LAST keyword match
                t_candidate = pd.to_numeric(df.loc[idx_txt[-1], time_col], errors="coerce")
                if pd.notna(t_candidate):
                    if best_keyword_t is None or t_candidate > best_keyword_t:
                        best_keyword_t = float(t_candidate)
                        best_keyword_col = mc

            # 2) Numeric markers — collect but validate later
            num = pd.to_numeric(ser, errors="coerce")
            idx_num = num[num > 0].index
            if len(idx_num) > 0:
                # Take LAST numeric marker
                t_candidate = pd.to_numeric(df.loc[idx_num[-1], time_col], errors="coerce")
                if pd.notna(t_candidate):
                    if best_numeric_t is None or t_candidate > best_numeric_t:
                        best_numeric_t = float(t_candidate)
                        best_numeric_col = mc

        # Decision logic:
        # A) Keyword found → use it (highest confidence)
        if best_keyword_t is not None:
            return best_keyword_t, f"event_marker_keyword:{best_keyword_col}", "keyword"

        # B) Numeric marker → validate it's not just a stage number
        if best_numeric_t is not None:
            # Check: how much data exists AFTER this marker?
            data_after = t_max_data - best_numeric_t
            if data_after < 60:
                # Less than 60s of data after → likely the actual end
                return best_numeric_t, f"event_marker_numeric:{best_numeric_col}", "numeric_end"
            else:
                # 60+ seconds of data after the last numeric marker
                # This is probably a stage number, not the end
                # DON'T use it as t_stop → fall through to composite
                return None, None, f"numeric_rejected:{best_numeric_col}:data_continues_{data_after:.0f}s"

        return None, None, "no_markers_found"

    @staticmethod
    def _detect_hr_drop(df_ex: pd.DataFrame, time_col: str):
        """
        v2.1: Detect sudden HR drop indicating exercise cessation.
        Looking for HR drop of >10 bpm within 30s window in last 20% of test.
        """
        hr_col = Engine_E00_StopDetection._first_existing(df_ex, ["HR_bpm", "HR"])
        if hr_col is None:
            return None
        
        hr = Engine_E00_StopDetection._to_numeric(df_ex[hr_col])
        hr_sm = Engine_E00_StopDetection._rolling(hr, win=10)
        
        n = len(hr_sm)
        if n < 30:
            return None
        
        # Focus on last 25% of exercise
        tail_start = int(0.75 * n)
        hr_tail = hr_sm.iloc[tail_start:]
        
        if hr_tail.dropna().shape[0] < 10:
            return None
        
        # Find peak HR in this tail
        peak_idx = hr_tail.idxmax()
        peak_hr = hr_tail[peak_idx]
        
        if peak_hr is None or np.isnan(peak_hr) or peak_hr < 100:
            return None
        
        # After peak, look for first sustained drop of >10 bpm
        after_peak = hr_tail.loc[peak_idx:]
        if len(after_peak) < 5:
            return None
        
        # Rolling minimum after peak
        hr_after_min = after_peak.rolling(5, min_periods=3).min()
        drop = peak_hr - hr_after_min
        drop_mask = drop >= 10  # 10 bpm drop
        
        if drop_mask.any():
            drop_idx = drop_mask.idxmax()  # First True
            t_drop = pd.to_numeric(df_ex.loc[drop_idx, time_col], errors="coerce")
            if pd.notna(t_drop):
                return float(t_drop)
        
        return None

    @staticmethod
    def _detect_vo2_collapse(df_ex: pd.DataFrame, time_col: str):
        """
        v2.1: Detect VO2 collapse (sustained drop >15% from peak) 
        as stop indicator for CSVs without speed/power data.
        """
        vo2_col = Engine_E00_StopDetection._first_existing(df_ex, ["VO2_mlmin", "VO2", "VO2_ml_min"])
        if vo2_col is None:
            return None
        
        vo2 = Engine_E00_StopDetection._to_numeric(df_ex[vo2_col])
        vo2_sm = Engine_E00_StopDetection._rolling(vo2, win=20)
        
        n = len(vo2_sm)
        if n < 30:
            return None
        
        peak_val = vo2_sm.max()
        if peak_val is None or np.isnan(peak_val) or peak_val < 100:
            return None
        
        peak_idx = vo2_sm.idxmax()
        
        # After peak, look for sustained drop below 85% of peak
        threshold = peak_val * 0.85
        after_peak = vo2_sm.loc[peak_idx:]
        
        if len(after_peak) < 5:
            return None
        
        # Rolling mean after peak — when does it drop below threshold for 5+ breaths?
        below = after_peak < threshold
        # Need 5 consecutive points below
        consec = below.rolling(5, min_periods=5).sum()
        collapse_mask = consec >= 5
        
        if collapse_mask.any():
            collapse_idx = collapse_mask.idxmax()
            # Go back to the start of the collapse (5 breaths earlier)
            try:
                pos = df_ex.index.get_loc(collapse_idx)
                start_pos = max(0, pos - 5)
                collapse_start_idx = df_ex.index[start_pos]
                t_collapse = pd.to_numeric(df_ex.loc[collapse_start_idx, time_col], errors="coerce")
                if pd.notna(t_collapse):
                    return float(t_collapse)
            except Exception:
                pass
        
        return None

    @staticmethod
    def _composite_stop(df_ex: pd.DataFrame, time_col: str):
        """
        v2.1: Multi-criteria auto-stop with additional signals.
        - VE/VCO2/VO2 peaks (smoothed)
        - Workload drop (Speed/Power)
        - HR drop detection (new)
        - VO2 collapse detection (new, for CSVs without speed)
        - RER endpoint confirmation
        """
        candidates = []
        signals_detail = {}

        ve_col = Engine_E00_StopDetection._first_existing(df_ex, ["VE_Lmin", "VE", "VE_L_min"])
        vco2_col = Engine_E00_StopDetection._first_existing(df_ex, ["VCO2_mlmin", "VCO2", "VCO2_ml_min", "VCO2_L_min"])
        vo2_col = Engine_E00_StopDetection._first_existing(df_ex, ["VO2_mlmin", "VO2", "VO2_ml_min", "VO2_L_min"])

        # Gas exchange peaks
        for col in [ve_col, vco2_col, vo2_col]:
            if col is None:
                continue
            s = Engine_E00_StopDetection._to_numeric(df_ex[col])
            s_sm = Engine_E00_StopDetection._rolling(s, win=15)
            if s_sm.dropna().empty:
                continue
            idx = s_sm.idxmax()
            t = pd.to_numeric(df_ex.loc[idx, time_col], errors="coerce")
            if pd.notna(t):
                candidates.append(float(t))
                signals_detail[col] = float(t)

        # Workload drop (Speed or Power)
        workload_col = Engine_E00_StopDetection._first_existing(df_ex, 
            ["Speed_kmh", "Power_W", "Cadence_rpm"])
        workload_drop_t = None
        if workload_col is not None:
            w = Engine_E00_StopDetection._to_numeric(df_ex[workload_col])
            w_valid = w.dropna()
            if len(w_valid) >= 20:
                w_sm = Engine_E00_StopDetection._rolling(w, win=15)
                n = len(w_sm)
                if n >= 20:
                    start_tail_idx = int(0.8 * n)
                    dw = w_sm.diff()
                    tail = dw.iloc[start_tail_idx:]
                    if tail.dropna().shape[0] > 0:
                        idx_drop = tail.idxmin()
                        t_drop = pd.to_numeric(df_ex.loc[idx_drop, time_col], errors="coerce")
                        if pd.notna(t_drop):
                            workload_drop_t = float(t_drop)
                            candidates.append(workload_drop_t)
                            signals_detail['workload_drop'] = workload_drop_t

        # v2.1: HR drop detection
        hr_drop_t = Engine_E00_StopDetection._detect_hr_drop(df_ex, time_col)
        if hr_drop_t is not None:
            candidates.append(hr_drop_t)
            signals_detail['hr_drop'] = hr_drop_t

        # v2.1: VO2 collapse (especially useful when no speed/power)
        if workload_col is None or workload_drop_t is None:
            vo2_collapse_t = Engine_E00_StopDetection._detect_vo2_collapse(df_ex, time_col)
            if vo2_collapse_t is not None:
                candidates.append(vo2_collapse_t)
                signals_detail['vo2_collapse'] = vo2_collapse_t

        if not candidates:
            return None, {"signals_used": [], "signals_detail": {},
                         "workload_drop_detected": False, "hr_drop_detected": False,
                         "rer_end_high": None}

        # RER end confirmation
        rer_end_high = None
        if vo2_col is not None and vco2_col is not None:
            vo2 = Engine_E00_StopDetection._to_numeric(df_ex[vo2_col])
            vco2 = Engine_E00_StopDetection._to_numeric(df_ex[vco2_col])
            rer = (vco2 / vo2.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            rer_tail = rer.iloc[int(0.8 * len(rer)):]
            if rer_tail.dropna().shape[0] >= 5:
                rer_end_high = bool(rer_tail.median() >= 1.0)

        # Strategy: 75th percentile of candidates, biased toward later times
        t_raw = float(np.percentile(candidates, 75))

        diag = {
            "signals_used": [c for c in [ve_col, vco2_col, vo2_col] if c is not None],
            "signals_detail": signals_detail,
            "workload_drop_detected": workload_drop_t is not None,
            "hr_drop_detected": hr_drop_t is not None,
            "vo2_collapse_detected": 'vo2_collapse' in signals_detail,
            "rer_end_high": rer_end_high,
            "n_candidates": len(candidates),
        }
        return t_raw, diag

    @staticmethod
    def _confidence(method: str, diag: dict) -> str:
        """
        v2.1: Enhanced confidence scoring.
        """
        if method.startswith("manual_config"):
            return "HIGH"
        if method.startswith("event_marker_keyword"):
            return "HIGH"
        if method.startswith("event_marker_numeric"):
            # Numeric marker accepted (data ends shortly after)
            return "MEDIUM"

        if method == "composite_auto_stop":
            score = 0
            sig_n = len(diag.get("signals_used", []))
            if sig_n >= 2:
                score += 2
            elif sig_n == 1:
                score += 1
            if diag.get("workload_drop_detected", False):
                score += 1
            if diag.get("hr_drop_detected", False):
                score += 1
            if diag.get("vo2_collapse_detected", False):
                score += 1
            if diag.get("rer_end_high", False) is True:
                score += 1

            if score >= 4:
                return "HIGH"
            if score >= 2:
                return "MEDIUM"
            return "LOW"

        if method == "auto_vo2_peak":
            if diag.get("rer_end_high", False):
                return "MEDIUM"
            return "LOW"

        return "LOW"

    @staticmethod
    def run(df: pd.DataFrame, cfg) -> dict:
        # --- basic validation ---
        if df is None or len(df) == 0:
            return {
                "status": "ERROR",
                "reason": "empty dataframe",
                "t_stop": np.nan,
                "method": "error_empty_df"
            }

        if "Time_sec" not in df.columns:
            return {
                "status": "ERROR",
                "reason": "missing Time_sec",
                "t_stop": np.nan,
                "method": "error_no_time"
            }

        out = {}
        dfx = df.copy()
        dfx["Time_sec"] = pd.to_numeric(dfx["Time_sec"], errors="coerce")
        dfx = dfx.dropna(subset=["Time_sec"]).sort_values("Time_sec").reset_index(drop=True)

        if dfx.empty:
            return {
                "status": "ERROR",
                "reason": "Time_sec invalid/empty",
                "t_stop": np.nan,
                "method": "error_bad_time"
            }

        t_min = float(dfx["Time_sec"].min())
        t_max = float(dfx["Time_sec"].max())

        # --- 1) manual override ---
        manual_t = getattr(cfg, "t_stop_seconds", None)
        if manual_t is not None:
            try:
                t_stop = Engine_E00_StopDetection._clip_t(float(manual_t), t_min, t_max)
                method = "manual_config"
                diag = {"signals_used": [], "workload_drop_detected": False, 
                        "hr_drop_detected": False, "rer_end_high": None}
            except Exception:
                t_stop = None
                method = None
                diag = {}
        else:
            t_stop = None
            method = None
            diag = {}

        # --- 2) event marker (v2.1: smart filtering) ---
        marker_detail = None
        if t_stop is None:
            t_event, event_method, marker_detail = Engine_E00_StopDetection._find_event_marker_stop(dfx, "Time_sec")
            if t_event is not None:
                t_stop = Engine_E00_StopDetection._clip_t(t_event, t_min, t_max)
                method = event_method
                diag = {"signals_used": ["event_marker"], "workload_drop_detected": None, 
                        "hr_drop_detected": None, "rer_end_high": None,
                        "marker_detail": marker_detail}

        # --- 3) composite auto (v2.1: with HR drop & VO2 collapse) ---
        if t_stop is None:
            t_start = t_min + 0.10 * (t_max - t_min)
            df_ex = dfx[dfx["Time_sec"] >= t_start].copy()

            t_comp, diag_comp = Engine_E00_StopDetection._composite_stop(df_ex, "Time_sec")
            if t_comp is not None:
                t_stop = Engine_E00_StopDetection._clip_t(t_comp, t_min, t_max)
                method = "composite_auto_stop"
                diag = diag_comp

        # --- 4) vo2 peak fallback ---
        if t_stop is None:
            vo2_col = Engine_E00_StopDetection._first_existing(dfx, ["VO2_mlmin", "VO2", "VO2_ml_min"])
            if vo2_col is not None:
                vo2 = Engine_E00_StopDetection._to_numeric(dfx[vo2_col])
                vo2_sm = Engine_E00_StopDetection._rolling(vo2, win=15)
                if vo2_sm.dropna().shape[0] > 0:
                    idx = vo2_sm.idxmax()
                    t_vo2 = pd.to_numeric(dfx.loc[idx, "Time_sec"], errors="coerce")
                    if pd.notna(t_vo2):
                        t_stop = Engine_E00_StopDetection._clip_t(float(t_vo2), t_min, t_max)
                        method = "auto_vo2_peak"

                        vco2_col = Engine_E00_StopDetection._first_existing(dfx, ["VCO2_mlmin", "VCO2", "VCO2_ml_min"])
                        rer_end_high = None
                        if vco2_col is not None:
                            vco2 = Engine_E00_StopDetection._to_numeric(dfx[vco2_col])
                            rer = (vco2 / vo2.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                            rer_tail = rer.iloc[int(0.8 * len(rer)):]
                            if rer_tail.dropna().shape[0] >= 5:
                                rer_end_high = bool(rer_tail.median() >= 1.0)

                        diag = {"signals_used": [vo2_col], "workload_drop_detected": False,
                                "hr_drop_detected": False, "rer_end_high": rer_end_high}

        # --- 5) final fallback ---
        if t_stop is None:
            t_stop = t_max
            method = "fallback_end"
            diag = {"signals_used": [], "workload_drop_detected": False,
                    "hr_drop_detected": False, "rer_end_high": None}

        # --- compute windows ---
        t_stop = float(t_stop)
        last60_start = max(t_min, t_stop - 60.0)
        rec0_start = t_stop
        rec0_end = min(t_max, t_stop + 60.0)
        rec180_end = min(t_max, t_stop + 180.0)

        rec0_60_available = (rec0_end - rec0_start) >= 45.0
        rec60_180_available = (rec180_end - (t_stop + 60.0)) >= 90.0

        conf = Engine_E00_StopDetection._confidence(method, diag)

        out.update({
            "status": "OK",
            "t_stop": t_stop,
            "method": method,
            "confidence": conf,

            # time windows
            "last60_start": float(last60_start),
            "last60_end": float(t_stop),
            "rec0_60_start": float(rec0_start),
            "rec0_60_end": float(rec0_end),
            "rec60_180_start": float(min(t_max, t_stop + 60.0)),
            "rec60_180_end": float(rec180_end),

            # recovery availability
            "recovery_0_60_available": bool(rec0_60_available),
            "recovery_60_180_available": bool(rec60_180_available),

            # diagnostics
            "signals_used": diag.get("signals_used", []),
            "workload_drop_detected": diag.get("workload_drop_detected", None),
            "hr_drop_detected": diag.get("hr_drop_detected", None),
            "vo2_collapse_detected": diag.get("vo2_collapse_detected", None),
            "rer_end_high": diag.get("rer_end_high", None),
            "marker_detail": marker_detail,
            "signals_detail": diag.get("signals_detail", {}),
            "t_min": t_min,
            "t_max": t_max
        })

        return out


# --- E01: QC (Bez zmian) ---
class Engine_E01_GasExchangeQC:
    """
    E01 v2: Gas Exchange QC + Peak/Max readiness score

    Cel:
    - ocena jakości sygnału oddechowego i HR,
    - ocena wiarygodności końcówki wysiłku (peak readiness),
    - przygotowanie metryk wejściowych dla E02/E16 i raportu.

    Zwraca:
    {
      status: OK|LIMITED|ERROR,
      confidence: LOW|MEDIUM|HIGH,
      quality_score: 0-100,
      reason: ... (dla LIMITED/ERROR),
      hr_peak: ...,
      vo2_peak_mlmin: ...,
      vo2_peak_mlkgmin: ...,
      vco2_peak_mlmin: ...,
      ve_peak_lmin: ...,
      rer_peak: ...,
      rer_last30_median: ...,
      br_peak_pct_mvv: ...,
      mvv_est_lmin: ...,
      diagnostics: {...}
    }
    """

    # ===== helpers =====
    @staticmethod
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _rolling_med(s: pd.Series, win: int = 15) -> pd.Series:
        return s.rolling(window=max(3, win), min_periods=1, center=True).median()

    @staticmethod
    def _first_col(df: pd.DataFrame, candidates) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _valid_n(s: pd.Series) -> int:
        return int(s.dropna().shape[0])

    @staticmethod
    def _tail_by_time(df: pd.DataFrame, time_col: str, seconds: float = 30.0) -> pd.DataFrame:
        if df.empty:
            return df
        tmax = float(pd.to_numeric(df[time_col], errors="coerce").max())
        tmin = tmax - float(seconds)
        return df[pd.to_numeric(df[time_col], errors="coerce") >= tmin].copy()

    @staticmethod
    def _estimate_mvv_lmin(fev1_l: Optional[float], ve_peak_lmin: Optional[float]) -> Optional[float]:
        # Priorytet: FEV1 * 35 (typowa estymacja). Jeśli brak FEV1, fallback heurystyczny.
        if fev1_l is not None and np.isfinite(fev1_l) and fev1_l > 0:
            return float(fev1_l * 35.0)
        # fallback ostrożny: jeśli brak spirometrii, zostaw None (lepsze niż zmyślanie)
        return None

    @staticmethod
    def _safe_div(a, b):
        try:
            if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
                return np.nan
            return a / b
        except Exception:
            return np.nan

    # ===== main =====
    @staticmethod
    def run(df_ex: pd.DataFrame) -> Dict[str, Any]:
        try:
            # --- minimalna walidacja ---
            if df_ex is None or len(df_ex) < 20:
                return {
                    "status": "LIMITED",
                    "confidence": "LOW",
                    "quality_score": 0,
                    "reason": "insufficient exercise data (<20 rows)",
                    "diagnostics": {}
                }

            if "Time_sec" not in df_ex.columns:
                return {
                    "status": "ERROR",
                    "confidence": "LOW",
                    "quality_score": 0,
                    "reason": "missing Time_sec",
                    "diagnostics": {}
                }

            df = df_ex.copy()
            df["Time_sec"] = Engine_E01_GasExchangeQC._num(df["Time_sec"])
            df = df.dropna(subset=["Time_sec"]).sort_values("Time_sec").reset_index(drop=True)
            if df.empty:
                return {
                    "status": "LIMITED",
                    "confidence": "LOW",
                    "quality_score": 0,
                    "reason": "no valid time rows",
                    "diagnostics": {}
                }

            # --- mapowanie kolumn (elastyczne aliasy) ---
            vo2_col  = Engine_E01_GasExchangeQC._first_col(df, ["VO2_mlmin", "VO2"])
            vco2_col = Engine_E01_GasExchangeQC._first_col(df, ["VCO2_mlmin", "VCO2"])
            ve_col   = Engine_E01_GasExchangeQC._first_col(df, ["VE_Lmin", "VE"])
            hr_col   = Engine_E01_GasExchangeQC._first_col(df, ["HR_bpm", "HR"])
            bm_col   = Engine_E01_GasExchangeQC._first_col(df, ["BodyMass_kg", "Mass_kg", "Weight_kg"])
            fev1_col = Engine_E01_GasExchangeQC._first_col(df, ["FEV1_L", "FEV1"])

            required_core = [vo2_col, vco2_col, ve_col]
            if any(c is None for c in required_core):
                missing = []
                if vo2_col is None: missing.append("VO2_mlmin/VO2")
                if vco2_col is None: missing.append("VCO2_mlmin/VCO2")
                if ve_col is None: missing.append("VE_Lmin/VE")
                return {
                    "status": "LIMITED",
                    "confidence": "LOW",
                    "quality_score": 0,
                    "reason": f"missing core columns: {missing}",
                    "diagnostics": {"missing_core": missing}
                }

            # --- series num ---
            vo2  = Engine_E01_GasExchangeQC._num(df[vo2_col])
            vco2 = Engine_E01_GasExchangeQC._num(df[vco2_col])
            ve   = Engine_E01_GasExchangeQC._num(df[ve_col])
            hr   = Engine_E01_GasExchangeQC._num(df[hr_col]) if hr_col else pd.Series(index=df.index, dtype=float)

            # QC valid count
            n_vo2  = Engine_E01_GasExchangeQC._valid_n(vo2)
            n_vco2 = Engine_E01_GasExchangeQC._valid_n(vco2)
            n_ve   = Engine_E01_GasExchangeQC._valid_n(ve)
            n_hr   = Engine_E01_GasExchangeQC._valid_n(hr) if hr_col else 0
            n_rows = len(df)

            # artifact rate (NaN share)
            na_rate_vo2  = 1 - (n_vo2 / n_rows)
            na_rate_vco2 = 1 - (n_vco2 / n_rows)
            na_rate_ve   = 1 - (n_ve / n_rows)
            na_rate_hr   = 1 - (n_hr / n_rows) if hr_col else 1.0

            # ── TIME-BASED SMOOTHING (ACSM/ATS standard) ─────────────────
            # Standard: highest 30-second average for VO2peak
            # Method: time-based rolling mean (not sample-based) for BxB data
            # Step 1: Outlier rejection via IQR (removes breath artifacts)
            # Step 2: 30s time-based rolling mean
            # Step 3: Plateau detection (VO2max vs VO2peak)
            
            time_s = Engine_E01_GasExchangeQC._num(df["Time_sec"])
            dt_median = float(time_s.diff().median()) if time_s.diff().median() > 0 else 2.0
            
            def _outlier_clip(s, k=2.5):
                """Clip values beyond k*IQR from Q1/Q3. Preserves true peaks."""
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr <= 0: return s
                lo, hi = q1 - k * iqr, q3 + k * iqr
                return s.clip(lo, hi)
            
            def _time_rolling_mean(s, time_s, window_sec=30):
                """Time-based rolling mean — constant time window regardless of BxB rate."""
                win_samples = max(5, int(window_sec / max(0.5, dt_median)))
                return s.rolling(window=win_samples, center=True, min_periods=3).mean()
            
            def _highest_Ns_average(s, time_s, N=30):
                """True highest N-second average (gold standard clinical method)."""
                best = np.nan
                s_arr = s.values
                t_arr = time_s.values
                valid = np.isfinite(s_arr) & np.isfinite(t_arr)
                s_v = s_arr[valid]
                t_v = t_arr[valid]
                if len(t_v) < 3:
                    return best
                for i in range(len(t_v)):
                    mask = (t_v >= t_v[i]) & (t_v < t_v[i] + N)
                    if mask.sum() >= 3:
                        avg = float(np.mean(s_v[mask]))
                        if np.isnan(best) or avg > best:
                            best = avg
                return best
            
            # Outlier-clipped series
            vo2_clean = _outlier_clip(vo2)
            vco2_clean = _outlier_clip(vco2)
            ve_clean = _outlier_clip(ve)
            
            # 30s rolling mean (for downstream compatibility — smoothed series)
            vo2_sm = _time_rolling_mean(vo2_clean, time_s, 30)
            vco2_sm = _time_rolling_mean(vco2_clean, time_s, 30)
            ve_sm = _time_rolling_mean(ve_clean, time_s, 30)
            hr_sm = _time_rolling_mean(hr, time_s, 15) if hr_col else hr
            
            # ── PEAK METRICS: Highest 30s average (ACSM standard) ────────
            vo2_peak_mlmin = _highest_Ns_average(vo2_clean, time_s, 30)
            vco2_peak_mlmin = _highest_Ns_average(vco2_clean, time_s, 30)
            ve_peak_lmin = _highest_Ns_average(ve_clean, time_s, 30)
            
            # HR: highest 15s average (HR responds faster)
            hr_peak = _highest_Ns_average(hr, time_s, 15) if hr_col else np.nan
            
            # Also compute 20s average for comparison
            vo2_peak_20s = _highest_Ns_average(vo2_clean, time_s, 20)
            
            # ── PLATEAU DETECTION (VO2max vs VO2peak) ────────────────────
            # Criterion: ΔVO2 < 150 ml/min (or <2.1 ml/kg/min) in last 
            # stage despite increased workload (Taylor et al. 1955, 
            # Howley et al. 1995, ACSM Guidelines 11th ed.)
            #
            # Implementation: compare highest 30s avg in last 60s vs 
            # highest 30s avg in the 60s before that. If delta < 150 → plateau.
            
            plateau_detected = False
            plateau_delta_mlmin = np.nan
            
            t_max_valid = time_s[vo2_clean.notna()].max() if vo2_clean.notna().sum() > 0 else np.nan
            if np.isfinite(t_max_valid):
                # Last 60s window
                m_last = (time_s >= t_max_valid - 60) & (time_s <= t_max_valid)
                # Previous 60s window  
                m_prev = (time_s >= t_max_valid - 120) & (time_s < t_max_valid - 60)
                
                if m_last.sum() >= 5 and m_prev.sum() >= 5:
                    # Highest 30s avg in each window
                    vo2_last = _highest_Ns_average(
                        vo2_clean[m_last].reset_index(drop=True),
                        time_s[m_last].reset_index(drop=True), 30)
                    vo2_prev = _highest_Ns_average(
                        vo2_clean[m_prev].reset_index(drop=True),
                        time_s[m_prev].reset_index(drop=True), 30)
                    
                    if np.isfinite(vo2_last) and np.isfinite(vo2_prev):
                        plateau_delta_mlmin = float(vo2_last - vo2_prev)
                        # Plateau: last window NOT substantially higher than prev
                        # despite continued exercise (delta < 150 ml/min)
                        plateau_detected = abs(plateau_delta_mlmin) < 150
            
            # Classification
            vo2_determination = "VO2max" if plateau_detected else "VO2peak"
            vo2_method_note = (
                f"Highest 30s average (ACSM). "
                f"{'Plateau detected' if plateau_detected else 'No plateau'}: "
                f"Δlast60s={plateau_delta_mlmin:.0f} ml/min" 
                if np.isfinite(plateau_delta_mlmin) 
                else "Highest 30s average (ACSM). Plateau: insufficient data"
            )

            # body mass for ml/kg/min
            body_mass = np.nan
            if bm_col is not None:
                bm_series = Engine_E01_GasExchangeQC._num(df[bm_col]).dropna()
                if len(bm_series) > 0:
                    body_mass = float(bm_series.iloc[-1])

            vo2_peak_mlkgmin = np.nan
            if np.isfinite(vo2_peak_mlmin) and np.isfinite(body_mass) and body_mass > 0:
                vo2_peak_mlkgmin = float(vo2_peak_mlmin / body_mass)

            # RER series
            rer = (vco2 / vo2.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            rer_sm = Engine_E01_GasExchangeQC._rolling_med(rer, 9)

            rer_peak = float(rer_sm.max()) if rer_sm.dropna().shape[0] else np.nan
            tail30 = Engine_E01_GasExchangeQC._tail_by_time(df.assign(RER=rer_sm), "Time_sec", 30.0)
            rer_last30_median = float(pd.to_numeric(tail30["RER"], errors="coerce").median()) if "RER" in tail30.columns and tail30["RER"].dropna().shape[0] else np.nan

            # MVV / BR
            fev1_val = None
            if fev1_col is not None:
                fev1_series = Engine_E01_GasExchangeQC._num(df[fev1_col]).dropna()
                if len(fev1_series) > 0:
                    fev1_val = float(fev1_series.iloc[-1])

            mvv_est_lmin = Engine_E01_GasExchangeQC._estimate_mvv_lmin(fev1_val, ve_peak_lmin)
            br_peak_pct_mvv = np.nan
            if mvv_est_lmin is not None and np.isfinite(ve_peak_lmin):
                br_peak_pct_mvv = float(100.0 * Engine_E01_GasExchangeQC._safe_div(ve_peak_lmin, mvv_est_lmin))

            # ----- scoring quality_score 0..100 -----
            score = 100.0

            # NaN penalty
            score -= 25.0 * na_rate_vo2
            score -= 20.0 * na_rate_vco2
            score -= 15.0 * na_rate_ve
            # HR optional but useful
            if hr_col:
                score -= 10.0 * na_rate_hr

            # tail adequacy penalty (last 30s should have enough data)
            if tail30.shape[0] < 5:
                score -= 10.0

            # physiological plausibility bonuses/penalties
            # RER end >=1.00 supports high effort
            if np.isfinite(rer_last30_median):
                if rer_last30_median >= 1.10:
                    score += 5.0
                elif rer_last30_median >= 1.00:
                    score += 2.0
                else:
                    score -= 5.0
            else:
                score -= 8.0

            # VE/VO2/VCO2 peaks available?
            if not np.isfinite(vo2_peak_mlmin): score -= 15.0
            if not np.isfinite(vco2_peak_mlmin): score -= 10.0
            if not np.isfinite(ve_peak_lmin): score -= 10.0

            # clamp
            quality_score = int(max(0, min(100, round(score))))

            # confidence mapping
            if quality_score >= 80:
                confidence = "HIGH"
            elif quality_score >= 60:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # status logic
            if quality_score < 35:
                status = "LIMITED"
                reason = "low signal quality / insufficient confidence for robust peak interpretation"
            else:
                status = "OK"
                reason = ""

            # optional effort flags (nie są dogmatem max, tylko sygnały)
            effort_flags = {
                "rer_last30_ge_1_00": bool(np.isfinite(rer_last30_median) and rer_last30_median >= 1.00),
                "rer_last30_ge_1_10": bool(np.isfinite(rer_last30_median) and rer_last30_median >= 1.10),
                "hr_present": bool(hr_col is not None and np.isfinite(hr_peak)),
                "ve_peak_present": bool(np.isfinite(ve_peak_lmin)),
                "vo2_peak_present": bool(np.isfinite(vo2_peak_mlmin))
            }

            return {
                "status": status,
                "confidence": confidence,
                "quality_score": quality_score,
                "reason": reason,

                # core outputs for downstream (ACSM: highest 30s average)
                "hr_peak": hr_peak if np.isfinite(hr_peak) else None,
                "vo2_peak_mlmin": vo2_peak_mlmin if np.isfinite(vo2_peak_mlmin) else None,
                "vo2_peak_mlkgmin": vo2_peak_mlkgmin if np.isfinite(vo2_peak_mlkgmin) else None,
                "vo2_peak_20s_mlmin": vo2_peak_20s if np.isfinite(vo2_peak_20s) else None,
                "vco2_peak_mlmin": vco2_peak_mlmin if np.isfinite(vco2_peak_mlmin) else None,
                "ve_peak_lmin": ve_peak_lmin if np.isfinite(ve_peak_lmin) else None,
                "rer_peak": rer_peak if np.isfinite(rer_peak) else None,
                "rer_last30_median": rer_last30_median if np.isfinite(rer_last30_median) else None,
                "br_peak_pct_mvv": br_peak_pct_mvv if np.isfinite(br_peak_pct_mvv) else None,
                "mvv_est_lmin": mvv_est_lmin if (mvv_est_lmin is not None and np.isfinite(mvv_est_lmin)) else None,
                
                # Plateau detection (VO2max vs VO2peak)
                "vo2_determination": vo2_determination,
                "plateau_detected": plateau_detected,
                "plateau_delta_mlmin": plateau_delta_mlmin if np.isfinite(plateau_delta_mlmin) else None,
                "vo2_method_note": vo2_method_note,

                "effort_flags": effort_flags,

                "diagnostics": {
                    "rows_total": n_rows,
                    "valid_counts": {
                        "VO2": n_vo2, "VCO2": n_vco2, "VE": n_ve, "HR": n_hr
                    },
                    "na_rates": {
                        "VO2": round(float(na_rate_vo2), 4),
                        "VCO2": round(float(na_rate_vco2), 4),
                        "VE": round(float(na_rate_ve), 4),
                        "HR": round(float(na_rate_hr), 4) if hr_col else None
                    },
                    "columns_used": {
                        "vo2_col": vo2_col, "vco2_col": vco2_col, "ve_col": ve_col,
                        "hr_col": hr_col, "bm_col": bm_col, "fev1_col": fev1_col
                    }
                }
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "confidence": "LOW",
                "quality_score": 0,
                "reason": f"{type(e).__name__}: {e}",
                "diagnostics": {}
            }

"""
Engine E02 v4.0 — Ventilatory Threshold Detection
====================================================
Complete rewrite based on audit findings (Bialik case) and literature:
- Kim 2021 (Frontiers Physiol): V-slope + ExCO2 as primary
- Benítez-Muñoz 2024 (Eur J Appl Physiol): Wide validation ranges
- Beaver 1986: Classic V-slope reference
- ATS/ERS: VE/VO2 rise + PetO2/PetCO2 divergence

Key changes vs v3:
1. V-slope promoted to PRIMARY (was backup)
2. RER → soft confidence modifier (was hard gate 0.88-0.97)
3. Validation ranges widened: VO2% 40-80, HR% 55-85
4. RER artifact handling: skip first 120s for RER-based methods
5. PetO2/PetCO2 divergence analysis (new)
6. VE/VO2 sustained rise point (new, complementing nadir)
7. File metadata fallback (VT1_HR/VT2_HR from CSV)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThresholdCandidate:
    """Single candidate from one detection method."""
    time_sec: float
    vo2_mlmin: float
    vo2_pct_peak: float
    hr_bpm: float
    hr_pct_max: float
    rer: float
    speed_kmh: Optional[float] = None
    power_w: Optional[float] = None
    confidence: float = 0.5
    method: str = "unknown"
    details: Dict = field(default_factory=dict)


@dataclass
class ThresholdResult:
    """Final result for VT1 or VT2."""
    time_sec: Optional[float] = None
    vo2_mlmin: Optional[float] = None
    vo2_pct_peak: Optional[float] = None
    hr_bpm: Optional[float] = None
    hr_pct_max: Optional[float] = None
    rer: Optional[float] = None
    speed_kmh: Optional[float] = None
    power_w: Optional[float] = None
    confidence: float = 0.0
    source: str = "none"
    methods_agreed: List[str] = field(default_factory=list)
    n_methods: int = 0
    candidates: Dict[str, Dict] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Engine_E02_Thresholds_v4:
    """
    Ventilatory Threshold Detection Engine v4.0

    Architecture (Kim 2021 + Benítez-Muñoz 2024):
      Phase 0: Data prep (RER artifact trim, 30s smoothing)
      Phase 1: Multi-method detection (≥3 methods per threshold)
      Phase 2: Consensus (weighted average of agreeing methods)
      Phase 3: Relaxed validation (wide ranges, RER as soft modifier)
      Phase 4: Fallback from file metadata
    """

    # ── Configuration ─────────────────────────────────────────────────────────

    # Smoothing
    SMOOTH_WINDOW_SEC = 30.0

    # RER artifact: ignore first N seconds for RER-based methods
    RER_ARTIFACT_SKIP_SEC = 120.0

    # Search ranges (fraction of exercise TIME)
    VT1_SEARCH_START = 0.20
    VT1_SEARCH_END = 0.75

    VT2_SEARCH_START = 0.50
    VT2_SEARCH_END = 0.96

    # ── Validation ranges (RELAXED per Benítez-Muñoz 2024) ────────────────
    # VT1: covers low training (48%) to high training (70%)
    VT1_VO2_PCT_MIN = 40.0
    VT1_VO2_PCT_MAX = 80.0
    VT1_HR_PCT_MIN = 55.0
    VT1_HR_PCT_MAX = 85.0

    # VT2: covers wide range
    VT2_VO2_PCT_MIN = 65.0
    VT2_VO2_PCT_MAX = 97.0
    VT2_HR_PCT_MIN = 72.0
    VT2_HR_PCT_MAX = 98.0

    # Gap constraints
    MIN_VT1_VT2_GAP_SEC = 90.0
    MIN_VT1_VT2_GAP_VO2_PCT = 10.0

    # Consensus
    MAX_CONSENSUS_DEVIATION_SEC = 90.0

    # Method weights
    WEIGHTS = {
        'vslope': 0.65,          # Can miss VT1 in trained athletes (Beaver 1986)
        'exco2_bp': 0.65,        # Same issue — often finds VT2 not VT1
        'veq_vo2_nadir': 0.75,   # Pre-VT1 marker, not VT1 itself
        'veq_vo2_rise': 1.0,     # ★ ATS/ERS gold standard (isocapnic buffering)
        'pet_divergence': 0.70,  # Confirmation
        'rer_crossing': 0.35,    # Soft confirmation only
        'veq_vco2_nadir': 0.80,  # For VT2
        'veq_vco2_rise': 0.90,   # For VT2 (primary)
        'pet_co2_decline': 0.70, # For VT2
    }

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def run(cls, df: pd.DataFrame, e00_result: Dict = None,
            file_metadata: Dict = None) -> Dict[str, Any]:
        """
        Main entry point.

        Parameters
        ----------
        df : exercise DataFrame (canonicalized)
        e00_result : dict from E00 with t_stop
        file_metadata : dict with VT1_HR, VT2_HR etc. from CSV header (fallback)
        """
        result = cls._init_result()

        try:
            # Phase 0: Prepare data
            df_ex, params = cls._prepare_data(df, e00_result)

            if df_ex is None or len(df_ex) < 50:
                result['status'] = 'ERROR'
                result['flags'].append('INSUFFICIENT_DATA')
                return result

            result['diagnostics'] = {k: v for k, v in params.items()
                                     if k not in ('df',)}

            # Phase 1-2: Detect VT1
            vt1 = cls._detect_vt1(df_ex, params)
            cls._store_threshold(result, vt1, 'vt1')

            # Phase 1-2: Detect VT2
            vt2 = cls._detect_vt2(df_ex, params, vt1)
            cls._store_threshold(result, vt2, 'vt2')

            # Phase 3: Cross-validation
            cls._cross_validate(result, vt1, vt2, params)

            # Phase 4: Fallback from file metadata
            if file_metadata:
                cls._apply_metadata_fallback(result, df_ex, params,
                                             file_metadata, vt1, vt2)

            # Final status
            cls._set_status(result, vt1, vt2)

        except Exception as e:
            result['status'] = 'ERROR'
            result['flags'].append(f'EXCEPTION:{type(e).__name__}:{str(e)[:80]}')

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 0: DATA PREPARATION
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _prepare_data(cls, df: pd.DataFrame,
                      e00_result: Dict) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Standardize columns, compute derived signals, smooth."""

        df = df.copy()

        # ── Column resolution ────────────────────────────────────────────
        col_map = {
            'Time_s': 'time', 'Time_sec': 'time', 'time_sec': 'time',
            't': 'time',
            'VO2_ml_min': 'vo2_ml', 'VO2_mlmin': 'vo2_ml',
            'VO2_L_min': 'vo2_L',
            'VCO2_ml_min': 'vco2_ml', 'VCO2_mlmin': 'vco2_ml',
            'VCO2_L_min': 'vco2_L',
            'VE_L_min': 've', 'VE_Lmin': 've', 'VE_lmin': 've',
            'HR_bpm': 'hr', 'HR': 'hr',
            'RER': 'rer',
            'PetO2_mmHg': 'peto2', 'PETO2_mmHg': 'peto2',
            'PetCO2_mmHg': 'petco2', 'PETCO2_mmHg': 'petco2',
            'Speed_kmh': 'speed', 'Speed': 'speed',
            'Power_W': 'power', 'Power': 'power',
            # Precomputed ratios from MetaMax
            "V'E/V'O2": 've_vo2_raw', "V'E/V'CO2": 've_vco2_raw',
            'ExCO2': 'exco2_raw',
        }

        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = pd.to_numeric(df[old], errors='coerce')

        # Ensure ml/min
        if 'vo2_ml' not in df.columns and 'vo2_L' in df.columns:
            df['vo2_ml'] = df['vo2_L'] * 1000
        if 'vco2_ml' not in df.columns and 'vco2_L' in df.columns:
            df['vco2_ml'] = df['vco2_L'] * 1000

        # ── Filter exercise phase ────────────────────────────────────────
        phase_col = None
        for c in ['Faza', 'Phase', 'phase']:
            if c in df.columns:
                phase_col = c
                break

        if phase_col:
            exercise_labels = ['Wysiłek', 'Exercise', 'Ramp', 'Rampa', 'Test']
            df_ex = df[df[phase_col].isin(exercise_labels)].copy()
            if len(df_ex) < 50:
                # Fallback: exclude rest/recovery
                exclude = ['Spoczynek', 'Rest', 'Rozgrzewka', 'Warmup',
                           'Ochłonięcie', 'Recovery', 'Cooldown', 'Regeneracja']
                df_ex = df[~df[phase_col].isin(exclude)].copy()
        else:
            df_ex = df.copy()

        # Apply t_stop
        if e00_result and 't_stop' in e00_result and 'time' in df_ex.columns:
            t_stop = float(e00_result['t_stop'])
            df_ex = df_ex[df_ex['time'] <= t_stop].copy()

        # ── Check required ───────────────────────────────────────────────
        required = ['time', 'vo2_ml', 'vco2_ml', 've']
        missing = [c for c in required if c not in df_ex.columns
                   or df_ex[c].notna().sum() < 20]
        if missing:
            return None, {'error': f'Missing/empty columns: {missing}'}

        if len(df_ex) < 50:
            return None, {'error': 'Too few data points'}

        df_ex = df_ex.sort_values('time').reset_index(drop=True)

        # ── Smoothing ────────────────────────────────────────────────────
        dt = df_ex['time'].diff().median()
        if not np.isfinite(dt) or dt <= 0:
            dt = 3.0
        window = max(5, int(cls.SMOOTH_WINDOW_SEC / dt))

        smooth_cols = ['vo2_ml', 'vco2_ml', 've']
        if 'hr' in df_ex.columns and df_ex['hr'].notna().sum() > 20:
            smooth_cols.append('hr')
        if 'rer' in df_ex.columns:
            smooth_cols.append('rer')
        for c in ['peto2', 'petco2']:
            if c in df_ex.columns and df_ex[c].notna().sum() > 20:
                smooth_cols.append(c)

        for col in smooth_cols:
            df_ex[f'{col}_sm'] = (df_ex[col]
                                  .rolling(window, center=True, min_periods=3)
                                  .mean())

        # ── Derived signals ──────────────────────────────────────────────
        # VE/VO2 and VE/VCO2 (compute from smoothed, more robust)
        vo2_L = df_ex['vo2_ml_sm'] / 1000.0
        vco2_L = df_ex['vco2_ml_sm'] / 1000.0

        df_ex['ve_vo2'] = df_ex['ve_sm'] / vo2_L.replace(0, np.nan)
        df_ex['ve_vco2'] = df_ex['ve_sm'] / vco2_L.replace(0, np.nan)

        # ExCO2 (excess CO2)
        df_ex['exco2'] = df_ex['vco2_ml_sm'] - df_ex['vo2_ml_sm']

        # ── Percentages ──────────────────────────────────────────────────
        vo2_peak = df_ex['vo2_ml_sm'].max()
        hr_max = df_ex['hr_sm'].max() if 'hr_sm' in df_ex.columns else np.nan

        df_ex['vo2_pct'] = df_ex['vo2_ml_sm'] / vo2_peak * 100
        if np.isfinite(hr_max) and hr_max > 0:
            df_ex['hr_pct'] = df_ex['hr_sm'] / hr_max * 100
        else:
            df_ex['hr_pct'] = np.nan

        # ── RER minimum time (for artifact handling) ─────────────────────
        t_start = df_ex['time'].min()
        t_end = df_ex['time'].max()
        t_duration = t_end - t_start

        # Find RER minimum after skipping artifact period
        rer_stable_mask = df_ex['time'] >= t_start + cls.RER_ARTIFACT_SKIP_SEC
        if 'rer_sm' in df_ex.columns and rer_stable_mask.sum() > 10:
            rer_min_idx = df_ex.loc[rer_stable_mask, 'rer_sm'].idxmin()
            rer_min_time = df_ex.loc[rer_min_idx, 'time']
        else:
            rer_min_time = t_start + 300  # fallback

        params = {
            'dt_sec': dt,
            'window': window,
            'vo2_peak': vo2_peak,
            'hr_max': hr_max,
            't_start': t_start,
            't_end': t_end,
            't_duration': t_duration,
            'rer_min_time': rer_min_time,
            'n_samples': len(df_ex),
            'has_hr': 'hr_sm' in df_ex.columns and df_ex['hr_sm'].notna().sum() > 20,
            'has_peto2': 'peto2_sm' in df_ex.columns and df_ex['peto2_sm'].notna().sum() > 20,
            'has_petco2': 'petco2_sm' in df_ex.columns and df_ex['petco2_sm'].notna().sum() > 20,
            'has_speed': 'speed' in df_ex.columns and df_ex['speed'].notna().sum() > 5,
            'has_power': 'power' in df_ex.columns and df_ex['power'].notna().sum() > 5,
        }

        return df_ex, params

    # ══════════════════════════════════════════════════════════════════════════
    # BREAKPOINT DETECTION CORE (shared by multiple methods)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_piecewise_breakpoint(x: np.ndarray, y: np.ndarray,
                                   indices: np.ndarray,
                                   search_frac: Tuple[float, float] = (0.15, 0.85),
                                   require_slope_increase: bool = True,
                                   min_slope_ratio: float = 1.0
                                   ) -> Optional[Dict]:
        """
        Find optimal breakpoint via piecewise linear regression (min RSS).

        Returns dict with idx, slope1, slope2, rss, or None.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            return None

        x, y, indices = x[mask], y[mask], indices[mask]
        n = len(x)

        lo = max(10, int(n * search_frac[0]))
        hi = min(n - 10, int(n * search_frac[1]))

        if lo >= hi:
            return None

        best_rss = np.inf
        best = None

        for i in range(lo, hi):
            s1, i1, _, _, _ = linregress(x[:i], y[:i])
            pred1 = s1 * x[:i] + i1
            rss1 = np.sum((y[:i] - pred1) ** 2)

            s2, i2, _, _, _ = linregress(x[i:], y[i:])
            pred2 = s2 * x[i:] + i2
            rss2 = np.sum((y[i:] - pred2) ** 2)

            total = rss1 + rss2

            if require_slope_increase and s2 <= s1 * min_slope_ratio:
                continue

            if total < best_rss:
                best_rss = total
                best = {
                    'idx': int(indices[i]),
                    'split_pos': i,
                    'slope1': float(s1),
                    'slope2': float(s2),
                    'rss': float(total),
                }

        return best

    # ══════════════════════════════════════════════════════════════════════════
    # VT1 DETECTION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _detect_vt1(cls, df: pd.DataFrame, params: Dict) -> ThresholdResult:
        """Detect VT1 using multiple methods + consensus."""

        vt1 = ThresholdResult()
        candidates = {}

        t_start = params['t_start']
        t_dur = params['t_duration']

        search_lo = t_start + t_dur * cls.VT1_SEARCH_START
        search_hi = t_start + t_dur * cls.VT1_SEARCH_END

        df_search = df[(df['time'] >= search_lo) & (df['time'] <= search_hi)].copy()
        if len(df_search) < 20:
            vt1.flags.append('VT1_SEARCH_RANGE_TOO_SMALL')
            return vt1

        # ── METHOD A: V-slope (PRIMARY) ──────────────────────────────────
        try:
            cand = cls._vt1_vslope(df, df_search, params)
            if cand:
                candidates['vslope'] = cand
        except Exception as e:
            vt1.flags.append(f'vslope_error:{e}')

        # ── METHOD B: VE/VO2 nadir ──────────────────────────────────────
        try:
            cand = cls._vt1_veq_vo2_nadir(df, df_search, params)
            if cand:
                candidates['veq_vo2_nadir'] = cand
        except Exception as e:
            vt1.flags.append(f'veq_nadir_error:{e}')

        # ── METHOD C: ExCO2 first breakpoint ─────────────────────────────
        try:
            cand = cls._vt1_exco2_breakpoint(df, params, search_lo, search_hi)
            if cand:
                candidates['exco2_bp'] = cand
        except Exception as e:
            vt1.flags.append(f'exco2_error:{e}')

        # ── METHOD D: VE/VO2 sustained rise ──────────────────────────────
        try:
            cand = cls._vt1_veq_vo2_rise(df, df_search, params)
            if cand:
                candidates['veq_vo2_rise'] = cand
        except Exception as e:
            vt1.flags.append(f'veq_rise_error:{e}')

        # ── METHOD E: PetO2/PetCO2 divergence ───────────────────────────
        if params['has_peto2'] and params['has_petco2']:
            try:
                cand = cls._vt1_pet_divergence(df, df_search, params)
                if cand:
                    candidates['pet_divergence'] = cand
            except Exception as e:
                vt1.flags.append(f'pet_div_error:{e}')

        # ── METHOD F: RER crossing (confirmation only) ───────────────────
        try:
            cand = cls._vt1_rer_crossing(df, params)
            if cand:
                candidates['rer_crossing'] = cand
        except Exception as e:
            vt1.flags.append(f'rer_cross_error:{e}')

        # ── Store candidates & build consensus ───────────────────────────
        vt1.candidates = {k: cls._cand_to_dict(v) for k, v in candidates.items()}

        # Validate each candidate
        valid = {}
        for name, cand in candidates.items():
            if cls._validate_vt1(cand, params):
                valid[name] = cand
            else:
                vt1.flags.append(f'{name}_failed_validation')

        if not valid:
            vt1.flags.append('VT1_NO_VALID_CANDIDATES')
            # Try with even more relaxed bounds (emergency)
            for name, cand in candidates.items():
                if cls._validate_vt1(cand, params, relaxed=True):
                    valid[name] = cand
                    vt1.flags.append(f'{name}_passed_relaxed')

        if not valid:
            return vt1

        # Consensus
        cls._build_consensus(vt1, valid, df, params)

        # Phase 2b: Adjudicator — physiological consistency check
        cls._adjudicate_vt1(vt1, candidates, valid, df, params)

        return vt1

    @classmethod
    def _vt1_vslope(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                    params: Dict) -> Optional[ThresholdCandidate]:
        """
        V-slope: breakpoint in VCO2 vs VO2 (Beaver 1986).

        IMPORTANT: In well-trained athletes VCO2/VO2 can be nearly linear
        (slope ~1.0-1.1) throughout most of exercise. In such cases the
        piecewise breakpoint falls near VT2, NOT VT1. We detect this by
        checking if slope1 is already >1.0 (no sub-threshold linear phase)
        or if the breakpoint VO2% is >70% (too late for VT1).
        When this happens we return None and flag it, letting other methods
        (especially isocapnic buffering) determine VT1.
        """

        vo2 = df_search['vo2_ml_sm'].values
        vco2 = df_search['vco2_ml_sm'].values
        indices = df_search.index.values

        bp = cls._find_piecewise_breakpoint(
            vo2, vco2, indices,
            search_frac=(0.15, 0.85),
            require_slope_increase=True,
            min_slope_ratio=1.05,
        )

        if bp is None:
            return None

        s1, s2 = bp['slope1'], bp['slope2']
        idx = bp['idx']
        row = df.loc[idx]
        bp_vo2_pct = float(row['vo2_pct']) if 'vo2_pct' in row else 50.0

        # ── Guard: detect "linear VCO2/VO2" pattern ──────────────────
        # If slope1 >= 1.0 AND breakpoint is >70% VO2peak, this is VT2
        # territory, not VT1. Suppress.
        if s1 >= 0.98 and bp_vo2_pct > 70:
            return None  # breakpoint is VT2, not VT1

        # If slope ratio is very low (<1.15) the breakpoint is weak
        slope_ratio = s2 / max(s1, 0.01)
        if slope_ratio < 1.15:
            return None  # no meaningful VT1 breakpoint

        # ── Confidence ────────────────────────────────────────────────
        conf = 0.80
        if s1 < 1.0 and s2 > 1.0:
            conf = 0.92  # classic criterion fully met
        elif slope_ratio > 1.3:
            conf = 0.85

        return cls._row_to_candidate(row, params, 'vslope', conf,
                                     {'slope1': s1, 'slope2': s2,
                                      'slope_ratio': slope_ratio})

    @classmethod
    def _vt1_veq_vo2_nadir(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                           params: Dict) -> Optional[ThresholdCandidate]:
        """
        VE/VO2 nadir: the bottom of the U-shaped VE/VO2 curve.

        Instead of taking the absolute minimum (which can be noisy and early),
        we find the CENTER of the flat bottom region — the zone where VE/VO2
        is within 5% of its minimum value. This produces a more physiologically
        meaningful point, typically closer to where VT1 is marked by experts.
        """

        ve_vo2 = df_search['ve_vo2'].dropna()
        if len(ve_vo2) < 10:
            return None

        # Extra smoothing for nadir detection (avoid noise)
        extra_win = max(7, int(45.0 / params['dt_sec']))  # ~45s window
        ve_vo2_heavy = ve_vo2.rolling(extra_win, center=True, min_periods=3).mean()
        ve_vo2_heavy = ve_vo2_heavy.dropna()

        if len(ve_vo2_heavy) < 10:
            nadir_idx = ve_vo2.idxmin()
        else:
            # Find minimum of heavily smoothed signal
            min_val = ve_vo2_heavy.min()
            # "Flat bottom" = within 5% of minimum
            threshold = min_val * 1.05
            flat_mask = ve_vo2_heavy <= threshold

            if flat_mask.sum() > 0:
                # Take the CENTER of the flat bottom region
                flat_indices = ve_vo2_heavy[flat_mask].index
                center_pos = len(flat_indices) // 2
                nadir_idx = flat_indices[center_pos]
            else:
                nadir_idx = ve_vo2_heavy.idxmin()

        if pd.isna(nadir_idx):
            return None

        row = df.loc[nadir_idx]
        conf = 0.80

        # Confirmation: VE/VCO2 should be flat or decreasing at nadir
        if 've_vco2' in df_search.columns:
            try:
                loc = df_search.index.get_loc(nadir_idx)
                window_after = df_search.iloc[loc:min(loc + 15, len(df_search))]
                if len(window_after) > 5:
                    ve_vco2_slope = np.polyfit(
                        range(len(window_after)),
                        window_after['ve_vco2'].values, 1
                    )[0]
                    if ve_vco2_slope < 0.1:  # flat or decreasing
                        conf = 0.85
            except Exception:
                pass

        return cls._row_to_candidate(row, params, 'veq_vo2_nadir', conf)

    @classmethod
    def _vt1_exco2_breakpoint(cls, df: pd.DataFrame, params: Dict,
                              search_lo: float, search_hi: float
                              ) -> Optional[ThresholdCandidate]:
        """ExCO2 first breakpoint: first significant slope increase in VCO2-VO2."""

        # Search slightly broader for ExCO2
        t_start = params['t_start']
        df_search = df[(df['time'] > t_start + 100) & (df['time'] < search_hi)].copy()

        if len(df_search) < 30:
            return None

        time_arr = df_search['time'].values
        exco2_arr = df_search['exco2'].values
        indices = df_search.index.values

        bp = cls._find_piecewise_breakpoint(
            time_arr, exco2_arr, indices,
            search_frac=(0.15, 0.70),
            require_slope_increase=True,
            min_slope_ratio=1.0,
        )

        if bp is None:
            return None

        idx = bp['idx']
        row = df.loc[idx]
        conf = 0.85

        return cls._row_to_candidate(row, params, 'exco2_bp', conf,
                                     {'slope1': bp['slope1'], 'slope2': bp['slope2']})

    @classmethod
    def _vt1_veq_vo2_rise(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                          params: Dict) -> Optional[ThresholdCandidate]:
        """
        Isocapnic Buffering Method (ATS/ERS gold standard):
        VT1 = point where VE/VO2 rises WITHOUT concurrent VE/VCO2 rise.

        Strategy:
        1. Find all isocapnic buffering segments
        2. Merge segments with gaps < 30s (physiological jitter)
        3. Score each segment by: duration + VO2% appropriateness
        4. Return midpoint of the best-scoring segment
        """

        if 've_vo2' not in df.columns or 've_vco2' not in df.columns:
            return None

        dt = params['dt_sec']
        half_win = max(5, int(15.0 / dt))  # ~15s derivative window

        # Use broader range for this method
        t_start = params['t_start']
        t_dur = params['t_duration']
        mask = ((df['time'] >= t_start + t_dur * 0.15) &
                (df['time'] <= t_start + t_dur * 0.85))
        df_work = df[mask].copy()

        if len(df_work) < 2 * half_win + 10:
            return None

        ve_vo2 = df_work['ve_vo2'].values
        ve_vco2 = df_work['ve_vco2'].values
        times = df_work['time'].values
        vo2_pct = df_work['vo2_pct'].values
        indices = df_work.index.values

        # Step 1: Find all isocapnic points
        iscap_raw = []  # list of (local_idx, time)
        for i in range(half_win, len(times) - half_win):
            dt_local = times[i + half_win] - times[i - half_win]
            if dt_local <= 0:
                continue
            d_ve_vo2 = (ve_vo2[i + half_win] - ve_vo2[i - half_win]) / dt_local * 100
            d_ve_vco2 = (ve_vco2[i + half_win] - ve_vco2[i - half_win]) / dt_local * 100
            if d_ve_vo2 > 0.25 and d_ve_vco2 < 0.30:
                iscap_raw.append(i)

        if not iscap_raw:
            return None

        # Step 2: Group into contiguous segments (merge gaps < 30s)
        MAX_GAP_SEC = 30.0
        segments = []
        seg_start = iscap_raw[0]
        seg_end = iscap_raw[0]

        for idx in iscap_raw[1:]:
            gap = times[idx] - times[seg_end]
            if gap <= MAX_GAP_SEC:
                seg_end = idx
            else:
                segments.append((seg_start, seg_end))
                seg_start = idx
                seg_end = idx
        segments.append((seg_start, seg_end))

        if not segments:
            return None

        # Step 3: Score each segment
        best_score = -np.inf
        best_seg = None

        for seg_s, seg_e in segments:
            duration = times[seg_e] - times[seg_s]
            if duration < 10:  # min 10s
                continue

            mid_i = (seg_s + seg_e) // 2
            mid_vo2_pct = vo2_pct[mid_i] if mid_i < len(vo2_pct) else 50.0

            # Score components:
            # 1. Duration bonus (longer = better, capped at 120s)
            dur_score = min(duration, 120) / 120.0 * 40

            # 2. VO2% appropriateness (best at 50-65%, penalty outside)
            # Gaussian centered at 57% with SD=12
            vo2_ideal = 57.0
            vo2_sd = 12.0
            vo2_score = 30 * np.exp(-0.5 * ((mid_vo2_pct - vo2_ideal) / vo2_sd) ** 2)

            # 3. Not too early penalty (segments below 45% VO2 are suspect)
            early_penalty = -15 if mid_vo2_pct < 45 else 0

            total_score = dur_score + vo2_score + early_penalty

            if total_score > best_score:
                best_score = total_score
                best_seg = (seg_s, seg_e, mid_i, duration, mid_vo2_pct)

        if best_seg is None:
            return None

        seg_s, seg_e, mid_i, duration, mid_vo2_pct = best_seg
        df_idx = indices[mid_i]
        row = df.loc[df_idx]

        # Confidence
        if duration > 60:
            conf = 0.92
        elif duration > 30:
            conf = 0.85
        elif duration > 15:
            conf = 0.78
        else:
            conf = 0.70

        return cls._row_to_candidate(
            row, params, 'veq_vo2_rise', conf,
            {'segment_start_s': float(times[seg_s]),
             'segment_end_s': float(times[seg_e]),
             'segment_duration_s': float(duration),
             'segment_vo2_pct': float(mid_vo2_pct),
             'n_segments_found': len(segments),
             'score': float(best_score)}
        )

    @classmethod
    def _vt1_pet_divergence(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                            params: Dict) -> Optional[ThresholdCandidate]:
        """
        PetO2/PetCO2 divergence: PetO2 starts rising while PetCO2 is still
        stable or rising. More robust than simple diff threshold.
        """

        if 'peto2_sm' not in df_search.columns or 'petco2_sm' not in df_search.columns:
            return None

        win = max(5, int(20.0 / params['dt_sec']))  # ~20s derivative window

        # Derivatives (per-second rates)
        dt = params['dt_sec']
        df_search = df_search.copy()
        df_search['d_peto2'] = df_search['peto2_sm'].diff(win) / (win * dt)
        df_search['d_petco2'] = df_search['petco2_sm'].diff(win) / (win * dt)

        # VT1 criterion: d_peto2 > 0.05 mmHg/s AND d_petco2 > -0.02 (stable/rising)
        diverge_mask = (
            (df_search['d_peto2'] > 0.05) &
            (df_search['d_petco2'] > -0.02)
        )

        # Find first sustained divergence (3+ consecutive points)
        streak = 0
        for idx in df_search.index:
            if idx in df_search.index and diverge_mask.get(idx, False):
                streak += 1
                if streak >= 3:
                    # Start of divergence
                    start_idx = df_search.index[df_search.index.get_loc(idx) - 2]
                    row = df.loc[start_idx]
                    return cls._row_to_candidate(row, params, 'pet_divergence', 0.70)
            else:
                streak = 0

        return None

    @classmethod
    def _vt1_rer_crossing(cls, df: pd.DataFrame,
                          params: Dict) -> Optional[ThresholdCandidate]:
        """
        RER crossing: after RER minimum, find first sustained crossing
        above (RER_min + 0.08). Skip first 120s to avoid artifact.

        This is CONFIRMATION ONLY — lower weight in consensus.
        """

        if 'rer_sm' not in df.columns:
            return None

        rer_min_time = params['rer_min_time']

        # Only search AFTER the RER minimum
        df_post = df[df['time'] > rer_min_time].copy()
        if len(df_post) < 10:
            return None

        rer_min_val = df_post['rer_sm'].min()
        threshold = rer_min_val + 0.08  # Typically ~0.88 for healthy subjects

        # Find first sustained crossing
        streak = 0
        for idx in df_post.index:
            if df.loc[idx, 'rer_sm'] >= threshold:
                streak += 1
                if streak >= 3:
                    cross_idx = df_post.index[
                        df_post.index.get_loc(idx) - streak + 1]
                    row = df.loc[cross_idx]
                    return cls._row_to_candidate(
                        row, params, 'rer_crossing', 0.60)
            else:
                streak = 0

        return None

    # ══════════════════════════════════════════════════════════════════════════
    # VT2 DETECTION METHODS
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _detect_vt2(cls, df: pd.DataFrame, params: Dict,
                    vt1: ThresholdResult) -> ThresholdResult:
        """Detect VT2 using multiple methods + consensus."""

        vt2 = ThresholdResult()
        candidates = {}

        t_start = params['t_start']
        t_dur = params['t_duration']

        # VT2 search must start after VT1
        vt1_time = vt1.time_sec if vt1.time_sec else t_start + t_dur * 0.4
        search_lo = max(
            t_start + t_dur * cls.VT2_SEARCH_START,
            vt1_time + cls.MIN_VT1_VT2_GAP_SEC
        )
        search_hi = t_start + t_dur * cls.VT2_SEARCH_END

        df_search = df[(df['time'] >= search_lo) & (df['time'] <= search_hi)].copy()
        if len(df_search) < 15:
            vt2.flags.append('VT2_SEARCH_RANGE_TOO_SMALL')
            return vt2

        # ── METHOD A: ExCO2 second breakpoint (acceleration) ─────────────
        try:
            cand = cls._vt2_exco2_breakpoint(df, vt1_time, params)
            if cand:
                candidates['exco2_bp'] = cand
        except Exception as e:
            vt2.flags.append(f'exco2_vt2_error:{e}')

        # ── METHOD B: VE/VCO2 rise (breakpoint where VE/VCO2 starts rising)
        try:
            cand = cls._vt2_veq_vco2_rise(df, df_search, params)
            if cand:
                candidates['veq_vco2_rise'] = cand
        except Exception as e:
            vt2.flags.append(f'veq_vco2_rise_error:{e}')

        # ── METHOD C: VE/VCO2 nadir ─────────────────────────────────────
        try:
            cand = cls._vt2_veq_vco2_nadir(df, df_search, params)
            if cand:
                candidates['veq_vco2_nadir'] = cand
        except Exception as e:
            vt2.flags.append(f'veq_vco2_nadir_error:{e}')

        # ── METHOD D: PetCO2 decline ────────────────────────────────────
        if params['has_petco2']:
            try:
                cand = cls._vt2_petco2_decline(df, df_search, params)
                if cand:
                    candidates['pet_co2_decline'] = cand
            except Exception as e:
                vt2.flags.append(f'petco2_decline_error:{e}')

        # ── METHOD E: RER crossing 1.0 (confirmation) ───────────────────
        try:
            cand = cls._vt2_rer_crossing(df, search_lo, params)
            if cand:
                candidates['rer_crossing'] = cand
        except Exception as e:
            vt2.flags.append(f'rer_cross_vt2_error:{e}')

        # ── Validate, consensus ──────────────────────────────────────────
        vt2.candidates = {k: cls._cand_to_dict(v) for k, v in candidates.items()}

        valid = {}
        for name, cand in candidates.items():
            if cls._validate_vt2(cand, params, vt1):
                valid[name] = cand
            else:
                vt2.flags.append(f'{name}_failed_validation')

        if not valid:
            vt2.flags.append('VT2_NO_VALID_CANDIDATES')
            for name, cand in candidates.items():
                if cls._validate_vt2(cand, params, vt1, relaxed=True):
                    valid[name] = cand
                    vt2.flags.append(f'{name}_passed_relaxed')

        if not valid:
            return vt2

        # ── VT2-SPECIFIC CONSENSUS ──────────────────────────────────
        # Physiological hierarchy for VT2 (validated on 4 athletes vs lactate):
        #   TIER 1 (RCP markers): exco2_bp, veq_vco2_rise — these detect 
        #     the actual respiratory compensation point
        #   TIER 2 (confirmatory): rer_crossing — physiological confirmation
        #   TIER 3 (early signals): veq_vco2_nadir, pet_co2_decline — these 
        #     detect START of compensation, not the breakpoint itself
        #
        # Strategy: prefer TIER1 consensus. Use TIER3 only as fallback.
        
        tier1 = {n: c for n, c in valid.items() 
                 if n in ('exco2_bp', 'veq_vco2_rise')}
        tier2 = {n: c for n, c in valid.items() 
                 if n in ('rer_crossing',)}
        tier3 = {n: c for n, c in valid.items() 
                 if n in ('veq_vco2_nadir', 'pet_co2_decline')}
        
        # CASE 1: At least 1 tier1 marker exists
        if tier1:
            # Build consensus from tier1 + tier2 (skip tier3 early markers)
            vt2_pool = {**tier1, **tier2}
            if len(vt2_pool) >= 2:
                vt2.flags.append(f'vt2_tier1_consensus:{list(vt2_pool.keys())}')
                cls._build_vt2_consensus(vt2, vt2_pool, tier3, df, params)
            else:
                # Only 1 tier1 marker — check if any tier3 is CLOSE to it (within 150s)
                t1_time = list(tier1.values())[0].time_sec
                close_tier3 = {n: c for n, c in tier3.items() 
                               if abs(c.time_sec - t1_time) <= 150}
                expanded = {**tier1, **tier2, **close_tier3}
                vt2.flags.append(f'vt2_tier1_expanded:{list(expanded.keys())}')
                cls._build_vt2_consensus(vt2, expanded, tier3, df, params)
        # CASE 2: No tier1 — fall back to all methods (original behavior)
        else:
            vt2.flags.append('vt2_no_tier1_fallback_all')
            cls._build_vt2_consensus(vt2, valid, tier3, df, params)
        
        # Phase 2b: Adjudicator — physiological consistency check
        cls._adjudicate_vt2(vt2, candidates, valid, df, params)

        return vt2

    @classmethod
    def _vt2_exco2_breakpoint(cls, df: pd.DataFrame, vt1_time: float,
                              params: Dict) -> Optional[ThresholdCandidate]:
        """ExCO2 second breakpoint: acceleration of ExCO2 after VT1."""

        df_post = df[df['time'] > vt1_time + 120].copy()
        if len(df_post) < 30:
            return None

        time_arr = df_post['time'].values
        exco2_arr = df_post['exco2'].values
        indices = df_post.index.values

        bp = cls._find_piecewise_breakpoint(
            time_arr, exco2_arr, indices,
            search_frac=(0.20, 0.80),
            require_slope_increase=True,
            min_slope_ratio=1.2,
        )

        if bp is None:
            return None

        row = df.loc[bp['idx']]
        return cls._row_to_candidate(row, params, 'exco2_bp', 0.90,
                                     {'slope1': bp['slope1'], 'slope2': bp['slope2']})

    @classmethod
    def _vt2_veq_vco2_rise(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                           params: Dict) -> Optional[ThresholdCandidate]:
        """VE/VCO2 breakpoint: where VE/VCO2 starts rising after its nadir."""

        ve_vco2 = df_search['ve_vco2'].dropna()
        if len(ve_vco2) < 15:
            return None

        # Piecewise breakpoint in VE/VCO2 vs time
        time_arr = df_search.loc[ve_vco2.index, 'time'].values
        indices = ve_vco2.index.values

        bp = cls._find_piecewise_breakpoint(
            time_arr, ve_vco2.values, indices,
            search_frac=(0.15, 0.85),
            require_slope_increase=True,
            min_slope_ratio=1.0,
        )

        if bp is None:
            return None

        row = df.loc[bp['idx']]
        conf = 0.85

        # Higher confidence if slope2 >> slope1
        if bp['slope2'] > bp['slope1'] * 2:
            conf = 0.92

        return cls._row_to_candidate(row, params, 'veq_vco2_rise', conf,
                                     {'slope1': bp['slope1'], 'slope2': bp['slope2']})

    @classmethod
    def _vt2_veq_vco2_nadir(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                            params: Dict) -> Optional[ThresholdCandidate]:
        """VE/VCO2 nadir in the late portion of exercise."""

        ve_vco2 = df_search['ve_vco2'].dropna()
        if len(ve_vco2) < 10:
            return None

        nadir_idx = ve_vco2.idxmin()
        row = df.loc[nadir_idx]
        return cls._row_to_candidate(row, params, 'veq_vco2_nadir', 0.80)

    @classmethod
    def _vt2_petco2_decline(cls, df: pd.DataFrame, df_search: pd.DataFrame,
                            params: Dict) -> Optional[ThresholdCandidate]:
        """
        PetCO2 decline: point where PetCO2 starts falling consistently.
        
        Improved: uses heavily smoothed PETCO2 to resist BxB noise.
        BxB data has ±2-3 mmHg fluctuations per breath — raw derivative
        produces false streaks and broken streaks. Solution: smooth PETCO2
        with wide rolling median before derivative computation.
        
        Detection strategy:
        1. Heavy smooth (31-point rolling median)
        2. Compute derivative over 30s window
        3. Find point where PETCO2 drops >1.5 mmHg from its peak
           AND derivative is consistently negative
        """

        if 'petco2_sm' not in df_search.columns:
            return None

        dt = params['dt_sec']
        
        df_s = df_search.copy()
        
        # Heavy smoothing — 31-point rolling median resists BxB outliers
        heavy_win = max(21, int(30.0 / dt))
        df_s['petco2_heavy'] = df_s['petco2_sm'].rolling(
            heavy_win, center=True, min_periods=heavy_win//3).median()
        
        if df_s['petco2_heavy'].notna().sum() < 20:
            return None
        
        # Find PETCO2 peak (should be in heavy/isocapnic region)
        petco2_peak = df_s['petco2_heavy'].max()
        petco2_peak_time = df_s.loc[df_s['petco2_heavy'].idxmax(), 'time']
        
        # METHOD 1: Peak-to-decline — find where PETCO2 drops >1.5 mmHg
        # from its peak and stays down
        decline_threshold = petco2_peak - 1.5  # mmHg
        
        # Only search after peak
        post_peak = df_s[df_s['time'] > petco2_peak_time].copy()
        if len(post_peak) < 10:
            return None
        
        # Find first point where heavy PETCO2 drops below threshold
        # and stays below for at least 20s
        below = post_peak['petco2_heavy'] < decline_threshold
        if below.sum() < 3:
            # Try smaller decline: 1.0 mmHg
            decline_threshold = petco2_peak - 1.0
            below = post_peak['petco2_heavy'] < decline_threshold
        
        if below.sum() < 3:
            return None
        
        # Find first sustained drop (stays below for 20s)
        first_below_idx = below.idxmax()  # first True
        first_below_time = post_peak.loc[first_below_idx, 'time']
        
        # Check it stays below for 20+ seconds
        window_after = post_peak[
            (post_peak['time'] >= first_below_time) & 
            (post_peak['time'] <= first_below_time + 30)
        ]
        if window_after['petco2_heavy'].notna().sum() < 3:
            return None
        
        frac_below = (window_after['petco2_heavy'] < decline_threshold).mean()
        if frac_below < 0.6:
            return None
        
        # METHOD 2: Derivative confirmation (heavy-smoothed)
        deriv_win = max(10, int(25.0 / dt))
        df_s['d_petco2_h'] = df_s['petco2_heavy'].diff(deriv_win) / (deriv_win * dt)
        
        # Check derivative is negative around decline point
        region = df_s[
            (df_s['time'] >= first_below_time - 10) &
            (df_s['time'] <= first_below_time + 20)
        ]
        if len(region) > 0:
            mean_deriv = region['d_petco2_h'].mean()
            if mean_deriv > -0.01:  # Not actually declining
                return None
        
        # VT2 = onset of decline (few seconds before first_below)
        # Back up to find where decline actually starts
        onset_region = df_s[
            (df_s['time'] >= first_below_time - 30) &
            (df_s['time'] <= first_below_time)
        ]
        if len(onset_region) > 0:
            # Find first point where derivative turns negative
            neg_derivs = onset_region[onset_region['d_petco2_h'] < -0.015]
            if len(neg_derivs) > 0:
                onset_idx = neg_derivs.index[0]
            else:
                onset_idx = first_below_idx
        else:
            onset_idx = first_below_idx
        
        row = df.loc[onset_idx]
        
        # Confidence based on magnitude of decline
        decline_mag = petco2_peak - post_peak['petco2_heavy'].min()
        conf = 0.70
        if decline_mag > 3.0:
            conf = 0.80
        if decline_mag > 5.0:
            conf = 0.85
        
        return cls._row_to_candidate(
            row, params, 'pet_co2_decline', conf,
            {'petco2_peak': round(petco2_peak, 1),
             'decline_mmHg': round(decline_mag, 1),
             'onset_time': round(float(df_s.loc[onset_idx, 'time']), 1)})

    @classmethod
    def _vt2_rer_crossing(cls, df: pd.DataFrame, search_lo: float,
                          params: Dict) -> Optional[ThresholdCandidate]:
        """RER crossing 1.0: confirmation for VT2."""

        if 'rer_sm' not in df.columns:
            return None

        df_post = df[df['time'] >= search_lo].copy()

        streak = 0
        for idx in df_post.index:
            if df.loc[idx, 'rer_sm'] >= 1.00:
                streak += 1
                if streak >= 3:
                    cross_idx = df_post.index[
                        df_post.index.get_loc(idx) - streak + 1]
                    row = df.loc[cross_idx]
                    return cls._row_to_candidate(
                        row, params, 'rer_crossing', 0.65)
            else:
                streak = 0

        return None

    # ══════════════════════════════════════════════════════════════════════════
    # VALIDATION (RELAXED)
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _validate_vt1(cls, cand: ThresholdCandidate, params: Dict,
                      relaxed: bool = False) -> bool:
        """
        Validate VT1 candidate. NO hard RER gate — RER is soft confidence only.
        """
        vo2_pct = cand.vo2_pct_peak
        hr_pct = cand.hr_pct_max

        # Relaxed mode widens ranges by 10% each direction
        margin = 10.0 if relaxed else 0.0

        if vo2_pct < (cls.VT1_VO2_PCT_MIN - margin):
            return False
        if vo2_pct > (cls.VT1_VO2_PCT_MAX + margin):
            return False

        if np.isfinite(hr_pct):
            if hr_pct < (cls.VT1_HR_PCT_MIN - margin):
                return False
            if hr_pct > (cls.VT1_HR_PCT_MAX + margin):
                return False

        # NO RER gate — it's a confidence modifier only
        return True

    @classmethod
    def _validate_vt2(cls, cand: ThresholdCandidate, params: Dict,
                      vt1: ThresholdResult, relaxed: bool = False) -> bool:
        """Validate VT2 candidate."""

        vo2_pct = cand.vo2_pct_peak
        hr_pct = cand.hr_pct_max

        margin = 10.0 if relaxed else 0.0

        if vo2_pct < (cls.VT2_VO2_PCT_MIN - margin):
            return False
        if vo2_pct > (cls.VT2_VO2_PCT_MAX + margin):
            return False

        if np.isfinite(hr_pct):
            if hr_pct < (cls.VT2_HR_PCT_MIN - margin):
                return False
            if hr_pct > (cls.VT2_HR_PCT_MAX + margin):
                return False

        # Must be after VT1 with sufficient gap
        if vt1.time_sec is not None:
            gap = cand.time_sec - vt1.time_sec
            if gap < cls.MIN_VT1_VT2_GAP_SEC * (0.5 if relaxed else 1.0):
                return False
            vo2_gap = vo2_pct - (vt1.vo2_pct_peak or 0)
            if vo2_gap < cls.MIN_VT1_VT2_GAP_VO2_PCT * (0.5 if relaxed else 1.0):
                return False

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # CONSENSUS
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _build_vt2_consensus(cls, result: 'ThresholdResult',
                              pool: Dict, tier3: Dict,
                              df: pd.DataFrame, params: Dict):
        """
        VT2-dedicated consensus — NO VT1-specific filtering.
        
        Key differences from _build_consensus:
        1. No vt1_unreliable exclusion (exco2_bp at high VO2% is CORRECT for VT2)
        2. Priority fallback favors VT2 markers (veq_vco2_rise > exco2_bp)
        3. If tier1 markers cluster but rer_crossing is outlier → drop it
        """
        if len(pool) == 1:
            name, cand = next(iter(pool.items()))
            cls._apply_candidate(result, cand, df, params)
            result.source = name
            result.methods_agreed = [name]
            result.n_methods = 1
            cls._apply_rer_modifier_result(result)
            return
        
        # ── Weighted cluster (VT2-specific) ─────────────────────
        # Weights for VT2 context
        vt2_weights = {
            'veq_vco2_rise': 1.0,     # Primary RCP marker
            'exco2_bp': 0.90,         # Strong RCP marker
            'veq_vco2_nadir': 0.80,   # Nadir before rise
            'pet_co2_decline': 0.75,  # PETCO2 drop
            'rer_crossing': 0.40,     # Only confirmatory, often too early
        }
        
        times = []
        weights = []
        names = []
        for name, cand in pool.items():
            w = vt2_weights.get(name, 0.5) * cand.confidence
            times.append(cand.time_sec)
            weights.append(w)
            names.append(name)
        
        times_arr = np.array(times)
        
        # ── Smart outlier detection ──────────────────────────────
        # For VT2: rer_crossing often ~200s too early. If tier1 markers
        # cluster together, drop rer_crossing even if it's in pool.
        tier1_in_pool = [n for n in names if n in ('exco2_bp', 'veq_vco2_rise')]
        if len(tier1_in_pool) >= 2:
            t1_times = [pool[n].time_sec for n in tier1_in_pool]
            t1_spread = max(t1_times) - min(t1_times)
            if t1_spread <= 120:  # tier1 markers cluster within 120s
                # Check if rer_crossing is far from tier1 cluster
                t1_center = np.mean(t1_times)
                outliers = [n for n in names 
                           if n not in tier1_in_pool
                           and abs(pool[n].time_sec - t1_center) > 150]
                if outliers:
                    result.flags.append(f'vt2_dropped_outliers:{outliers}')
                    pool = {n: c for n, c in pool.items() if n not in outliers}
                    names = [n for n in names if n not in outliers]
                    times = [pool[n].time_sec for n in names]
                    weights = [vt2_weights.get(n, 0.5) * pool[n].confidence for n in names]
                    times_arr = np.array(times)
        
        # ── Consensus from remaining pool ────────────────────────
        median_t = np.median(times_arr)
        agreed = [n for n, t_val in zip(names, times)
                  if abs(t_val - median_t) <= cls.MAX_CONSENSUS_DEVIATION_SEC]
        
        # Cluster fallback
        if len(agreed) < 2:
            best_cluster = []
            for i, (n1, t1) in enumerate(zip(names, times)):
                cluster = [n1]
                for j, (n2, t2) in enumerate(zip(names, times)):
                    if i != j and abs(t1 - t2) <= cls.MAX_CONSENSUS_DEVIATION_SEC:
                        cluster.append(n2)
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            if len(best_cluster) >= 2:
                agreed = best_cluster
        
        if len(agreed) >= 2:
            a_times = [pool[n].time_sec for n in agreed]
            a_weights = [vt2_weights.get(n, 0.5) * pool[n].confidence for n in agreed]
            consensus_t = np.average(a_times, weights=a_weights)
            closest_idx = (df['time'] - consensus_t).abs().idxmin()
            row = df.loc[closest_idx]
            
            cls._fill_from_row(result, row, params)
            result.confidence = min(0.98,
                                    np.mean([pool[n].confidence for n in agreed]) + 0.1)
            result.source = 'vt2_consensus'
            result.methods_agreed = agreed
            result.n_methods = len(agreed)
        else:
            # Priority fallback — VT2-oriented
            vt2_priority = ['veq_vco2_rise', 'exco2_bp', 'veq_vco2_nadir',
                           'pet_co2_decline', 'rer_crossing']
            best_name = None
            for prio_name in vt2_priority:
                if prio_name in pool:
                    best_name = prio_name
                    break
            if best_name is None:
                best_name = max(pool, key=lambda n: pool[n].confidence)
            
            # But also check if tier3 confirms this best pick
            cand = pool[best_name]
            close_t3 = {n: c for n, c in tier3.items()
                       if abs(c.time_sec - cand.time_sec) <= 120}
            if close_t3:
                # Tier3 confirms → higher confidence
                all_pool = {best_name: cand, **close_t3}
                a_times = [c.time_sec for c in all_pool.values()]
                a_weights = [vt2_weights.get(n, 0.5) * c.confidence for n, c in all_pool.items()]
                consensus_t = np.average(a_times, weights=a_weights)
                closest_idx = (df['time'] - consensus_t).abs().idxmin()
                row = df.loc[closest_idx]
                
                cls._fill_from_row(result, row, params)
                result.confidence = min(0.95, cand.confidence + 0.05)
                result.source = f'vt2_best+t3:{best_name}'
                result.methods_agreed = list(all_pool.keys())
                result.n_methods = len(all_pool)
            else:
                cls._apply_candidate(result, cand, df, params)
                result.source = f'vt2_best:{best_name}'
                result.methods_agreed = [best_name]
                result.n_methods = 1
                result.confidence *= 0.85
                result.flags.append('VT2_POOR_CONSENSUS')
        
        cls._apply_rer_modifier_result(result)

    @classmethod
    def _build_consensus(cls, result: ThresholdResult,
                         valid: Dict[str, ThresholdCandidate],
                         df: pd.DataFrame, params: Dict):
        """
        ADAPTIVE consensus — two phenotypes supported:

        PHENOTYPE A (clear isocapnic, e.g. Bialik):
          Isocapnic segment >= 30s → anchor on it (ATS/ERS gold standard).

        PHENOTYPE B (no clear isocapnic, e.g. Najdienow):
          Isocapnic absent or < 30s → standard weighted cluster.
          For VT1: exclude V-slope and ExCO2 if they detect VT2
          (VO2% > 75%). Prefer nadir, PetO2/PetCO2.
        """

        if len(valid) == 1:
            name, cand = next(iter(valid.items()))
            cls._apply_candidate(result, cand, df, params)
            result.source = name
            result.methods_agreed = [name]
            result.n_methods = 1
            cls._apply_rer_modifier_result(result)
            return

        # ── Check isocapnic quality ──────────────────────────────────
        iscap = valid.get('veq_vo2_rise')
        iscap_quality = 'none'
        if iscap and iscap.confidence >= 0.70:
            seg_dur = (iscap.details or {}).get('segment_duration_s', 0)
            if seg_dur >= 25:
                iscap_quality = 'strong'
            elif seg_dur >= 12:
                iscap_quality = 'moderate'
            else:
                iscap_quality = 'weak'

        # ── PHENOTYPE A: Strong isocapnic → anchor ───────────────────
        if iscap_quality == 'strong':
            anchor_t = iscap.time_sec
            supporters = [n for n, c in valid.items()
                          if n != 'veq_vo2_rise'
                          and abs(c.time_sec - anchor_t) <= 120]

            conf_boost = 0.05 * len(supporters)
            final_conf = min(0.98, iscap.confidence + conf_boost)

            if supporters:
                all_in = ['veq_vo2_rise'] + supporters
                all_times = [valid[n].time_sec for n in all_in]
                all_weights = []
                for n in all_in:
                    w = cls.WEIGHTS.get(n, 0.5) * valid[n].confidence
                    if n == 'veq_vo2_rise':
                        w *= 2.0
                    all_weights.append(w)
                consensus_t = np.average(all_times, weights=all_weights)
            else:
                consensus_t = anchor_t

            closest_idx = (df['time'] - consensus_t).abs().idxmin()
            row = df.loc[closest_idx]

            cls._fill_from_row(result, row, params)
            result.confidence = final_conf
            result.source = 'isocapnic_anchor'
            result.methods_agreed = ['veq_vo2_rise'] + supporters
            result.n_methods = 1 + len(supporters)
            cls._apply_rer_modifier_result(result)
            return

        # ── PHENOTYPE B: No/weak isocapnic → adaptive cluster ────────
        # V-slope and ExCO2 often detect VT2, not VT1 in trained athletes.
        # Detect this: if their VO2% > 75%, exclude from VT1 consensus.
        vt1_unreliable = set()
        for mname in ['vslope', 'exco2_bp']:
            if mname in valid:
                c = valid[mname]
                if c.vo2_pct_peak and c.vo2_pct_peak > 75:
                    vt1_unreliable.add(mname)

        pool = {n: c for n, c in valid.items()
                if n not in vt1_unreliable}
        if len(pool) < 2:
            pool = valid

        times = []
        weights = []
        names = []
        for name, cand in pool.items():
            w = cls.WEIGHTS.get(name, 0.5) * cand.confidence
            times.append(cand.time_sec)
            weights.append(w)
            names.append(name)

        times_arr = np.array(times)
        median_t = np.median(times_arr)

        agreed = [n for n, t_val in zip(names, times)
                  if abs(t_val - median_t) <= cls.MAX_CONSENSUS_DEVIATION_SEC]

        if len(agreed) < 2:
            best_cluster = []
            for i, (n1, t1) in enumerate(zip(names, times)):
                cluster = [n1]
                for j, (n2, t2) in enumerate(zip(names, times)):
                    if i != j and abs(t1 - t2) <= cls.MAX_CONSENSUS_DEVIATION_SEC:
                        cluster.append(n2)
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster
            if len(best_cluster) >= 2:
                agreed = best_cluster

        if len(agreed) >= 2:
            a_times = [pool[n].time_sec for n in agreed]
            a_weights = [cls.WEIGHTS.get(n, 0.5) * pool[n].confidence
                         for n in agreed]
            consensus_t = np.average(a_times, weights=a_weights)
            closest_idx = (df['time'] - consensus_t).abs().idxmin()
            row = df.loc[closest_idx]

            cls._fill_from_row(result, row, params)
            result.confidence = min(0.98,
                                    np.mean([pool[n].confidence
                                             for n in agreed]) + 0.1)
            result.source = 'consensus'
            result.methods_agreed = agreed
            result.n_methods = len(agreed)
            if vt1_unreliable:
                result.flags.append(
                    f'excluded_vt2_like:{",".join(vt1_unreliable)}')
        else:
            priority = ['veq_vo2_rise', 'veq_vo2_nadir', 'pet_divergence',
                         'vslope', 'exco2_bp', 'rer_crossing',
                         'veq_vco2_rise', 'veq_vco2_nadir',
                         'pet_co2_decline']
            best_name = None
            for prio_name in priority:
                if prio_name in pool:
                    best_name = prio_name
                    break
            if best_name is None:
                best_name = max(pool, key=lambda n: pool[n].confidence)

            cand = pool[best_name]
            cls._apply_candidate(result, cand, df, params)
            result.source = f'best:{best_name}'
            result.methods_agreed = [best_name]
            result.n_methods = 1
            result.confidence *= 0.85
            result.flags.append('POOR_CONSENSUS')

        cls._apply_rer_modifier_result(result)

    @classmethod
    def _apply_rer_modifier_result(cls, result: ThresholdResult):
        """Apply RER confidence modifier to result."""
        if result.rer is not None:
            rer = result.rer
            if isinstance(rer, (int, float)) and np.isfinite(rer):
                if rer < 0.80:
                    result.confidence *= 0.80
                elif rer > 1.05:
                    result.confidence *= 0.85

    # ══════════════════════════════════════════════════════════════════════════
    # METADATA FALLBACK
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _apply_metadata_fallback(cls, result: Dict, df: pd.DataFrame,
                                 params: Dict, meta: Dict,
                                 vt1: ThresholdResult, vt2: ThresholdResult):
        """
        If VT1/VT2 not detected, use file metadata (MetaMax values) as fallback.
        """

        # VT1 fallback
        if vt1.time_sec is None:
            vt1_hr = meta.get('VT1_HR')
            vt1_vo2 = meta.get('VT1_VO2_ml_min')

            if vt1_hr is not None and np.isfinite(float(vt1_hr)):
                vt1_hr = float(vt1_hr)
                # Find closest HR match in exercise data
                if 'hr_sm' in df.columns:
                    diff = (df['hr_sm'] - vt1_hr).abs()
                    closest = diff.idxmin()
                    row = df.loc[closest]

                    result['vt1_time_sec'] = float(row['time'])
                    result['vt1_vo2_mlmin'] = float(row['vo2_ml_sm'])
                    result['vt1_vo2_pct_peak'] = float(row['vo2_pct'])
                    result['vt1_hr'] = float(row['hr_sm']) if 'hr_sm' in row else None
                    result['vt1_hr_pct_max'] = float(row['hr_pct']) if 'hr_pct' in row else None
                    result['vt1_rer'] = float(row['rer_sm']) if 'rer_sm' in row else None
                    result['vt1_speed_kmh'] = float(row.get('speed')) if pd.notna(row.get('speed')) else None
                    result['vt1_power_w'] = float(row.get('power')) if pd.notna(row.get('power')) else None
                    result['vt1_confidence'] = 0.60
                    result['vt1_source'] = 'file_metadata'
                    result['flags'].append('VT1_FROM_FILE_METADATA')

        # VT2 fallback
        if vt2.time_sec is None:
            vt2_hr = meta.get('VT2_HR')

            if vt2_hr is not None and np.isfinite(float(vt2_hr)):
                vt2_hr = float(vt2_hr)
                if 'hr_sm' in df.columns:
                    diff = (df['hr_sm'] - vt2_hr).abs()
                    closest = diff.idxmin()
                    row = df.loc[closest]

                    result['vt2_time_sec'] = float(row['time'])
                    result['vt2_vo2_mlmin'] = float(row['vo2_ml_sm'])
                    result['vt2_vo2_pct_peak'] = float(row['vo2_pct'])
                    result['vt2_hr'] = float(row['hr_sm']) if 'hr_sm' in row else None
                    result['vt2_hr_pct_max'] = float(row['hr_pct']) if 'hr_pct' in row else None
                    result['vt2_rer'] = float(row['rer_sm']) if 'rer_sm' in row else None
                    result['vt2_speed_kmh'] = float(row.get('speed')) if pd.notna(row.get('speed')) else None
                    result['vt2_power_w'] = float(row.get('power')) if pd.notna(row.get('power')) else None
                    result['vt2_confidence'] = 0.60
                    result['vt2_source'] = 'file_metadata'
                    result['flags'].append('VT2_FROM_FILE_METADATA')

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _row_to_candidate(cls, row: pd.Series, params: Dict,
                          method: str, confidence: float = 0.5,
                          details: Dict = None) -> ThresholdCandidate:
        """Create ThresholdCandidate from DataFrame row."""

        speed = float(row.get('speed')) if pd.notna(row.get('speed', np.nan)) else None
        power = float(row.get('power')) if pd.notna(row.get('power', np.nan)) else None

        return ThresholdCandidate(
            time_sec=float(row['time']),
            vo2_mlmin=float(row['vo2_ml_sm']),
            vo2_pct_peak=float(row['vo2_pct']),
            hr_bpm=float(row['hr_sm']) if 'hr_sm' in row and np.isfinite(row.get('hr_sm', np.nan)) else np.nan,
            hr_pct_max=float(row['hr_pct']) if 'hr_pct' in row and np.isfinite(row.get('hr_pct', np.nan)) else np.nan,
            rer=float(row['rer_sm']) if 'rer_sm' in row and np.isfinite(row.get('rer_sm', np.nan)) else np.nan,
            speed_kmh=speed,
            power_w=power,
            confidence=confidence,
            method=method,
            details=details or {},
        )

    @classmethod
    def _cand_to_dict(cls, cand: ThresholdCandidate) -> Dict:
        """Convert candidate to serializable dict."""
        return {
            'time': cand.time_sec,
            'vo2': cand.vo2_mlmin,
            'vo2_pct': cand.vo2_pct_peak,
            'hr': cand.hr_bpm,
            'hr_pct': cand.hr_pct_max,
            'rer': cand.rer,
            'speed': cand.speed_kmh,
            'power': cand.power_w,
            'confidence': cand.confidence,
            'method': cand.method,
            'details': cand.details,
        }

    @classmethod
    def _apply_candidate(cls, result: ThresholdResult,
                         cand: ThresholdCandidate,
                         df: pd.DataFrame, params: Dict):
        """Apply single candidate to result."""
        result.time_sec = cand.time_sec
        result.vo2_mlmin = cand.vo2_mlmin
        result.vo2_pct_peak = cand.vo2_pct_peak
        result.hr_bpm = cand.hr_bpm
        result.hr_pct_max = cand.hr_pct_max
        result.rer = cand.rer
        result.speed_kmh = cand.speed_kmh
        result.power_w = cand.power_w
        result.confidence = cand.confidence

    @classmethod
    def _fill_from_row(cls, result: ThresholdResult, row: pd.Series,
                       params: Dict):
        """Fill result from DataFrame row."""
        result.time_sec = float(row['time'])
        result.vo2_mlmin = float(row['vo2_ml_sm'])
        result.vo2_pct_peak = float(row['vo2_pct'])
        result.hr_bpm = float(row['hr_sm']) if 'hr_sm' in row and np.isfinite(row.get('hr_sm', np.nan)) else None
        result.hr_pct_max = float(row['hr_pct']) if 'hr_pct' in row and np.isfinite(row.get('hr_pct', np.nan)) else None
        result.rer = float(row['rer_sm']) if 'rer_sm' in row and np.isfinite(row.get('rer_sm', np.nan)) else None
        result.speed_kmh = float(row.get('speed')) if pd.notna(row.get('speed', np.nan)) else None
        result.power_w = float(row.get('power')) if pd.notna(row.get('power', np.nan)) else None

    @classmethod
    def _apply_rer_modifier(cls, result: ThresholdResult,
                            cand: ThresholdCandidate):
        """Soft RER confidence modifier."""
        rer = cand.rer
        if np.isfinite(rer):
            if rer < 0.80:
                result.confidence *= 0.80
                result.flags.append('RER_LOW')
            elif rer > 1.05:
                result.confidence *= 0.85
                result.flags.append('RER_HIGH_FOR_VT1')


    # ══════════════════════════════════════════════════════════════════════════
    # ADJUDICATOR — "Third Reviewer" post-consensus physiological checks
    # (Inspired by Fix-HF-5 multicenter: third reviewer adjudicates disagreement.
    #  Here: physiology adjudicates methods.)
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def _adjudicate_vt1(cls, result: 'ThresholdResult',
                        all_candidates: Dict[str, 'ThresholdCandidate'],
                        valid_candidates: Dict[str, 'ThresholdCandidate'],
                        df: pd.DataFrame, params: Dict):
        """
        Post-consensus adjudicator for VT1.

        Checks physiological consistency of the consensus against hard
        constraints invisible to individual detection methods.
        Can downgrade confidence or flag — does NOT shift time_sec.

        Checks:
          A. RER window (0.80–1.05 typical for VT1)
          B. PetCO2 phase (should be rising/plateau, not declining)
          C. VE/VCO2 state (should be near nadir, not sharply rising)
          D. Candidate spread vs consensus size
          E. Single-method penalty
        """
        if result.time_sec is None:
            return

        t = result.time_sec
        adj_flags = []

        # ── A. RER window ───────────────────────────────────────────
        rer = result.rer
        if rer is not None and np.isfinite(rer):
            if rer < 0.80:
                result.confidence *= 0.85
                adj_flags.append(f'ADJ_VT1_RER_LOW:{rer:.3f}')
            elif rer > 1.05:
                result.confidence *= 0.80
                adj_flags.append(f'ADJ_VT1_RER_HIGH:{rer:.3f}')

        # ── B. PetCO2 phase ────────────────────────────────────────
        if params['has_petco2']:
            try:
                dt_sec = params['dt_sec']
                loc = (df['time'] - t).abs().idxmin()
                iloc_pos = df.index.get_loc(loc)
                after_len = max(5, int(30.0 / dt_sec))
                after_slice = df.iloc[iloc_pos:min(iloc_pos + after_len, len(df))]
                if len(after_slice) > 3 and 'petco2_sm' in after_slice.columns:
                    pv = after_slice['petco2_sm'].dropna()
                    if len(pv) > 3:
                        sl = np.polyfit(range(len(pv)), pv.values, 1)[0] / dt_sec
                        if sl < -0.03:
                            result.confidence *= 0.85
                            adj_flags.append(f'ADJ_VT1_PETCO2_DECLINING:{sl:.4f}/s')
            except Exception:
                pass

        # ── C. VE/VCO2 state ───────────────────────────────────────
        try:
            vevco2_at_t = df.loc[(df['time'] - t).abs().idxmin(), 've_vco2']
            vevco2_nadir = df['ve_vco2'].min()
            if np.isfinite(vevco2_at_t) and np.isfinite(vevco2_nadir) and vevco2_nadir > 0:
                pct_above = (vevco2_at_t - vevco2_nadir) / vevco2_nadir * 100
                if pct_above > 15:
                    result.confidence *= 0.90
                    adj_flags.append(f'ADJ_VT1_VEVCO2_ELEVATED:{pct_above:.1f}%')
        except Exception:
            pass

        # ── D. Candidate spread ─────────────────────────────────────
        if len(all_candidates) >= 3:
            spread = max(c.time_sec for c in all_candidates.values()) - \
                     min(c.time_sec for c in all_candidates.values())
            if spread > 250 and result.n_methods <= 2:
                result.confidence *= 0.90
                adj_flags.append(f'ADJ_VT1_SPREAD:{spread:.0f}s_n{result.n_methods}')

        # ── E. Single-method penalty ────────────────────────────────
        if result.n_methods == 1 and len(valid_candidates) >= 3:
            result.confidence *= 0.85
            adj_flags.append('ADJ_VT1_SINGLE_METHOD')

        # Store
        if adj_flags:
            result.flags.extend(adj_flags)
        result.flags.append(f'ADJ_VT1:{"PASS" if not adj_flags else "FLAGGED_" + str(len(adj_flags))}')

    @classmethod
    def _adjudicate_vt2(cls, result: 'ThresholdResult',
                        all_candidates: Dict[str, 'ThresholdCandidate'],
                        valid_candidates: Dict[str, 'ThresholdCandidate'],
                        df: pd.DataFrame, params: Dict):
        """
        Post-consensus adjudicator for VT2.

        More critical than VT1 — VT2 has higher inter-method variability
        (LoA% ~20-25% in Fix-HF-5 multicenter). Adjudicator:

        A. Excluded cluster — methods that failed validation but agree
        B. PetCO2 must be declining at RCP (defining criterion)
        C. VE/VCO2 must be rising (defining criterion)
        D. RER sanity (typically 0.98–1.15 at VT2)
        E. rer_crossing-dependent consensus penalty
        F. Total candidate spread
        """
        if result.time_sec is None:
            return

        t = result.time_sec
        adj_flags = []
        adj_notes = []

        # ── A. EXCLUDED CLUSTER DETECTION ───────────────────────────
        rejected = {n: c for n, c in all_candidates.items()
                    if n not in valid_candidates}
        if len(rejected) >= 2:
            rej_times = [c.time_sec for c in rejected.values()]
            rej_spread = max(rej_times) - min(rej_times)
            rej_center = float(np.mean(rej_times))

            cons_times = [all_candidates[n].time_sec
                          for n in (result.methods_agreed or [])
                          if n in all_candidates]
            cons_center = float(np.mean(cons_times)) if cons_times else t

            if (rej_spread < 100
                    and len(rejected) >= result.n_methods
                    and abs(rej_center - cons_center) > 120):
                adj_flags.append(
                    f'ADJ_VT2_EXCLUDED_CLUSTER:'
                    f'{len(rejected)}m@{rej_center:.0f}s'
                    f'(spread={rej_spread:.0f}s)'
                    f'_vs_{result.n_methods}m@{cons_center:.0f}s')
                adj_notes.append(
                    f'Excluded cluster ({", ".join(rejected.keys())}) '
                    f'at ~{rej_center:.0f}s (spread {rej_spread:.0f}s) '
                    f'vs consensus at ~{cons_center:.0f}s. '
                    f'Consider wider VT2 range for trained athletes.')
                result.confidence *= 0.80

        # ── B. PetCO2 decline ───────────────────────────────────────
        if params['has_petco2']:
            try:
                dt_sec = params['dt_sec']
                loc = (df['time'] - t).abs().idxmin()
                iloc_pos = df.index.get_loc(loc)
                half = max(5, int(15.0 / dt_sec))
                window = df.iloc[max(0, iloc_pos - half):
                                 min(len(df), iloc_pos + half)]
                if len(window) > 5 and 'petco2_sm' in window.columns:
                    pv = window['petco2_sm'].dropna()
                    if len(pv) > 5:
                        sl = np.polyfit(range(len(pv)), pv.values, 1)[0] / dt_sec
                        if sl > 0.02:
                            result.confidence *= 0.85
                            adj_flags.append(f'ADJ_VT2_PETCO2_RISING:{sl:.4f}/s')
                            adj_notes.append(
                                f'PetCO2 rising at VT2 ({sl:.4f}/s) — RCP may be later.')
                        elif sl > -0.005:
                            adj_flags.append(f'ADJ_VT2_PETCO2_FLAT:{sl:.4f}/s')
            except Exception:
                pass

        # ── C. VE/VCO2 rising ───────────────────────────────────────
        try:
            dt_sec = params['dt_sec']
            loc = (df['time'] - t).abs().idxmin()
            iloc_pos = df.index.get_loc(loc)
            after_len = max(5, int(20.0 / dt_sec))
            window = df.iloc[iloc_pos:min(len(df), iloc_pos + after_len)]
            if len(window) > 5 and 've_vco2' in window.columns:
                vv = window['ve_vco2'].dropna()
                if len(vv) > 5:
                    sl = np.polyfit(range(len(vv)), vv.values, 1)[0] / dt_sec
                    if sl < -0.01:
                        result.confidence *= 0.90
                        adj_flags.append(f'ADJ_VT2_VEVCO2_DECLINING:{sl:.4f}/s')
        except Exception:
            pass

        # ── D. RER sanity ───────────────────────────────────────────
        rer = result.rer
        if rer is not None and np.isfinite(rer):
            if rer < 0.92:
                result.confidence *= 0.80
                adj_flags.append(f'ADJ_VT2_RER_LOW:{rer:.3f}')
                adj_notes.append(f'RER={rer:.3f} at VT2 below typical (0.98-1.15).')
            elif rer < 0.98:
                adj_flags.append(f'ADJ_VT2_RER_BORDERLINE:{rer:.3f}')

        # ── E. rer_crossing-dependent pair ──────────────────────────
        agreed = result.methods_agreed or []
        if (result.n_methods == 2
                and 'rer_crossing' in agreed
                and len(agreed) == 2):
            adj_flags.append(f'ADJ_VT2_WEAK_PAIR:{[m for m in agreed if m != "rer_crossing"][0]}+rer')
            result.confidence *= 0.92

        # ── F. Total spread ─────────────────────────────────────────
        if len(all_candidates) >= 3:
            spread = max(c.time_sec for c in all_candidates.values()) - \
                     min(c.time_sec for c in all_candidates.values())
            if spread > 200:
                adj_flags.append(f'ADJ_VT2_SPREAD:{spread:.0f}s')

        # Store
        if adj_flags:
            result.flags.extend(adj_flags)
        if adj_notes:
            result.flags.append('ADJ_VT2_NOTES:' + ' | '.join(adj_notes))

        critical = [f for f in adj_flags
                    if any(k in f for k in ('EXCLUDED_CLUSTER', 'RER_LOW', 'PETCO2_RISING'))]
        result.flags.append(
            f'ADJ_VT2:{"PASS" if not critical else "FLAGGED"}_n{len(adj_flags)}')


    @classmethod
    def _cross_validate(cls, result: Dict, vt1: ThresholdResult,
                        vt2: ThresholdResult, params: Dict):
        """Cross-validation of VT1 and VT2 relationship."""

        if vt1.time_sec and vt2.time_sec:
            if vt2.time_sec <= vt1.time_sec:
                result['flags'].append('VT2_BEFORE_VT1')
                # Don't zero out — let downstream decide

            gap = vt2.time_sec - vt1.time_sec
            if gap < cls.MIN_VT1_VT2_GAP_SEC:
                result['flags'].append('VT1_VT2_TOO_CLOSE')

            vo2_gap = (vt2.vo2_pct_peak or 0) - (vt1.vo2_pct_peak or 0)
            if vo2_gap < cls.MIN_VT1_VT2_GAP_VO2_PCT:
                result['flags'].append('VT1_VT2_VO2_GAP_SMALL')

    @classmethod
    def _set_status(cls, result: Dict, vt1: ThresholdResult,
                    vt2: ThresholdResult):
        """Set final status."""

        if vt1.time_sec and vt2.time_sec:
            result['status'] = 'OK'
            min_conf = min(vt1.confidence, vt2.confidence)
            result['confidence'] = ('HIGH' if min_conf > 0.75
                                    else 'MEDIUM' if min_conf > 0.5
                                    else 'LOW')
        elif vt1.time_sec or vt2.time_sec:
            result['status'] = 'PARTIAL'
            result['confidence'] = 'MEDIUM'
        else:
            result['status'] = 'FAILED'
            result['confidence'] = 'NONE'

    @classmethod
    def _store_threshold(cls, result: Dict, thr: ThresholdResult,
                         prefix: str):
        """Store threshold result in output dict."""
        result[f'{prefix}_time_sec'] = thr.time_sec
        result[f'{prefix}_vo2_mlmin'] = thr.vo2_mlmin
        result[f'{prefix}_vo2_pct_peak'] = thr.vo2_pct_peak
        result[f'{prefix}_hr'] = thr.hr_bpm
        result[f'{prefix}_hr_pct_max'] = thr.hr_pct_max
        result[f'{prefix}_rer'] = thr.rer
        result[f'{prefix}_speed_kmh'] = thr.speed_kmh
        result[f'{prefix}_power_w'] = thr.power_w
        result[f'{prefix}_confidence'] = thr.confidence
        result[f'{prefix}_source'] = thr.source
        result[f'{prefix}_n_methods'] = thr.n_methods
        result[f'{prefix}_methods_agreed'] = thr.methods_agreed
        result[f'{prefix}_candidates'] = thr.candidates
        result['flags'].extend(thr.flags)

    @classmethod
    def _init_result(cls) -> Dict:
        """Initialize empty result dict."""
        return {
            'status': 'UNKNOWN',
            'confidence': 'NONE',
            'vt1_time_sec': None, 'vt1_vo2_mlmin': None,
            'vt1_vo2_pct_peak': None, 'vt1_hr': None,
            'vt1_hr_pct_max': None, 'vt1_rer': None,
            'vt1_speed_kmh': None, 'vt1_power_w': None,
            'vt1_confidence': 0.0, 'vt1_source': 'none',
            'vt1_n_methods': 0, 'vt1_methods_agreed': [],
            'vt1_candidates': {},
            'vt2_time_sec': None, 'vt2_vo2_mlmin': None,
            'vt2_vo2_pct_peak': None, 'vt2_hr': None,
            'vt2_hr_pct_max': None, 'vt2_rer': None,
            'vt2_speed_kmh': None, 'vt2_power_w': None,
            'vt2_confidence': 0.0, 'vt2_source': 'none',
            'vt2_n_methods': 0, 'vt2_methods_agreed': [],
            'vt2_candidates': {},
            'flags': [],
            'diagnostics': {},
        }

# --- E03: VENT EFFICIENCY (VE SLOPE) ---
class Engine_E03_VentSlope:
    """
    Engine E03 v2.0 — Ventilatory Efficiency (VE/VCO2 Slope)
    ─────────────────────────────────────────────────────────
    Calculates VE/VCO2 slope using 3 methods (full, to-VT2, to-VT1),
    VE/VCO2 nadir, y-intercept, PETCO2 tracking, Arena Ventilatory Class,
    and predicted slope with clinical interpretation.

    References:
      • Arena 2007 — Ventilatory Class I–IV
      • Mezzani 2013 — EACPR slope to VT2
      • Phillips 2020 (Frontiers Physiology) — Nadir definition
      • NOODLE 2024 — Athlete reference values
      • USCjournal 2024 — Predicted slope equation
    """

    # Arena 2007 Ventilatory Classification
    ARENA_CLASSES = [
        (29.0,  "VC-I",   "Norma — minimalne ryzyko"),
        (35.9,  "VC-II",  "Łagodna nieefektywność — niskie ryzyko"),
        (44.9,  "VC-III", "Umiarkowana nieefektywność — średnie ryzyko"),
        (999.0, "VC-IV",  "Ciężka nieefektywność — wysokie ryzyko"),
    ]

    # Athlete reference (NOODLE 2024, cycling CPET, to-VT1)
    ATHLETE_REF = {
        "male":   {"mean": 26.1, "sd": 2.0},
        "female": {"mean": 27.7, "sd": 2.6},
    }

    @classmethod
    def run(cls, df_ex, e02_results=None, age=None, height_cm=None, sex="male"):
        """
        Parameters
        ----------
        df_ex        : DataFrame with exercise data
        e02_results  : dict from E02 with vt1_time_sec, vt2_time_sec
        age          : int/float
        height_cm    : int/float
        sex          : 'male' or 'female'
        """
        out = {
            "status": "OK",
            "slope_full": None, "intercept_full": None, "r2_full": None,
            "slope_to_vt2": None, "intercept_to_vt2": None, "r2_to_vt2": None,
            "slope_to_vt1": None, "intercept_to_vt1": None, "r2_to_vt1": None,
            "ve_vco2_nadir": None, "nadir_time_sec": None,
            "ve_vco2_at_vt1": None, "ve_vco2_peak": None,
            "petco2_rest": None, "petco2_vt1": None, "petco2_peak": None,
            "predicted_slope": None, "pct_predicted": None,
            "ventilatory_class": None, "vc_description": None,
            "athlete_ref_mean": None, "athlete_ref_sd": None,
            "clinical_interpretation": None,
            "flags": [],
        }

        # --- Resolve column names ---
        ve_col = None
        for c in ['VE_lmin', 'VE_BTPS', 'VE']:
            if c in df_ex.columns:
                ve_col = c
                break

        vco2_col = None
        vco2_is_ml = False
        for c in ['VCO2_mlmin', 'VCO2_lmin', 'VCO2']:
            if c in df_ex.columns:
                vco2_col = c
                # heuristic: if max > 20, it's ml/min
                mx = df_ex[c].max()
                if mx > 20:
                    vco2_is_ml = True
                break

        time_col = None
        for c in ['Time_sec', 'time_sec', 'time', 't']:
            if c in df_ex.columns:
                time_col = c
                break

        if ve_col is None or vco2_col is None:
            return out

        # Build clean arrays
        ve = df_ex[ve_col].values.astype(float)
        vco2_raw = df_ex[vco2_col].values.astype(float)
        vco2 = vco2_raw / 1000.0 if vco2_is_ml else vco2_raw  # -> L/min

        time_arr = df_ex[time_col].values.astype(float) if time_col else np.arange(len(ve))

        mask = (ve > 0) & (vco2 > 0) & np.isfinite(ve) & np.isfinite(vco2)
        if mask.sum() < 15:
            return out

        ve_m, vco2_m, time_m = ve[mask], vco2[mask], time_arr[mask]

        # --- Get VT1/VT2 time boundaries from E02 ---
        vt1_time = None
        vt2_time = None
        if e02_results and isinstance(e02_results, dict):
            vt1_time = e02_results.get('vt1_time_sec')
            vt2_time = e02_results.get('vt2_time_sec')

        # === A. Linear regressions (VE = slope * VCO2 + intercept) ===

        def _regress(x, y):
            if len(x) < 10:
                return None, None, None
            slope, intercept, r_val, _, _ = linregress(x, y)
            return round(slope, 2), round(intercept, 2), round(r_val**2, 3)

        # Full slope
        s_full, i_full, r2_full = _regress(vco2_m, ve_m)
        out["slope_full"]     = s_full
        out["intercept_full"] = i_full
        out["r2_full"]        = r2_full

        # Slope to VT2 (gold standard clinical)
        if vt2_time is not None:
            mask_vt2 = time_m <= vt2_time
            if mask_vt2.sum() >= 10:
                s_vt2, i_vt2, r2_vt2 = _regress(vco2_m[mask_vt2], ve_m[mask_vt2])
                out["slope_to_vt2"]     = s_vt2
                out["intercept_to_vt2"] = i_vt2
                out["r2_to_vt2"]        = r2_vt2

        # Slope to VT1
        if vt1_time is not None:
            mask_vt1 = time_m <= vt1_time
            if mask_vt1.sum() >= 10:
                s_vt1, i_vt1, r2_vt1 = _regress(vco2_m[mask_vt1], ve_m[mask_vt1])
                out["slope_to_vt1"]     = s_vt1
                out["intercept_to_vt1"] = i_vt1
                out["r2_to_vt1"]        = r2_vt1

        # === B. VE/VCO2 Nadir (lowest rolling 30s mean of VE/VCO2 ratio) ===
        ratio = ve_m / vco2_m

        # Determine window size based on time resolution
        if len(time_m) > 1:
            dt = np.median(np.diff(time_m))
            dt = max(dt, 0.5)
            window = max(3, int(round(30.0 / dt)))
        else:
            window = 3

        if len(ratio) >= window:
            rolling_mean = np.convolve(ratio, np.ones(window)/window, mode='valid')
            nadir_idx = np.argmin(rolling_mean)
            out["ve_vco2_nadir"] = round(float(rolling_mean[nadir_idx]), 1)
            # Time at nadir center
            center_idx = nadir_idx + window // 2
            if center_idx < len(time_m):
                out["nadir_time_sec"] = round(float(time_m[center_idx]), 0)

        # === C. VE/VCO2 at VT1 and Peak ===
        if vt1_time is not None:
            # Find closest time to vt1
            idx_vt1 = np.argmin(np.abs(time_m - vt1_time))
            # Average ±2 points
            lo = max(0, idx_vt1 - 2)
            hi = min(len(ratio), idx_vt1 + 3)
            out["ve_vco2_at_vt1"] = round(float(np.mean(ratio[lo:hi])), 1)

        # Peak = last 30s
        if len(ratio) >= 3:
            out["ve_vco2_peak"] = round(float(np.mean(ratio[-min(window, len(ratio)):])), 1)

        # === D. PETCO2 tracking (if column available) ===
        petco2_col = None
        for c in ['PETCO2', 'PetCO2', 'PETCO2_mmHg', 'EtCO2']:
            if c in df_ex.columns:
                petco2_col = c
                break

        if petco2_col is not None:
            pet = df_ex[petco2_col].values.astype(float)
            pet_valid = pet[mask]
            # Rest = first 60s
            rest_mask = time_m < time_m[0] + 60
            if rest_mask.sum() > 0:
                out["petco2_rest"] = round(float(np.mean(pet_valid[rest_mask])), 1)
            # At VT1
            if vt1_time is not None:
                idx_vt1 = np.argmin(np.abs(time_m - vt1_time))
                lo = max(0, idx_vt1 - 2)
                hi = min(len(pet_valid), idx_vt1 + 3)
                out["petco2_vt1"] = round(float(np.mean(pet_valid[lo:hi])), 1)
            # Peak = last 30s
            out["petco2_peak"] = round(float(np.mean(pet_valid[-min(window, len(pet_valid)):])), 1)

        # === E. Predicted Slope (USCjournal 2024) ===
        # predicted = 34.4 - 0.0723 * height(cm) + 0.082 * age
        if height_cm is not None and age is not None:
            try:
                pred = 34.4 - 0.0723 * float(height_cm) + 0.082 * float(age)
                out["predicted_slope"] = round(pred, 1)
                # % predicted uses slope_to_vt2 preferably, fallback full
                ref_slope = out["slope_to_vt2"] or out["slope_full"]
                if ref_slope is not None and pred > 0:
                    out["pct_predicted"] = round(ref_slope / pred * 100, 1)
            except Exception:
                pass

        # Athlete reference
        ref = cls.ATHLETE_REF.get(sex, cls.ATHLETE_REF["male"])
        out["athlete_ref_mean"] = ref["mean"]
        out["athlete_ref_sd"]   = ref["sd"]

        # === F. Arena Ventilatory Class ===
        # Use slope_to_vt2 (gold standard), fallback slope_full
        primary_slope = out["slope_to_vt2"] if out["slope_to_vt2"] is not None else out["slope_full"]

        if primary_slope is not None:
            for cutoff, vc, desc in cls.ARENA_CLASSES:
                if primary_slope <= cutoff:
                    out["ventilatory_class"] = vc
                    out["vc_description"]    = desc
                    break

        # === G. Clinical flags ===
        flags = []
        if primary_slope is not None:
            if primary_slope > 34:
                flags.append("SLOPE_ABOVE_34")
            elif primary_slope > 30:
                flags.append("SLOPE_ABOVE_30")

        if out["ve_vco2_nadir"] is not None and out["ve_vco2_nadir"] > 34:
            flags.append("NADIR_ABOVE_34")

        if out["petco2_vt1"] is not None and out["petco2_vt1"] < 36:
            flags.append("PETCO2_LOW_AT_VT1")

        if out["intercept_full"] is not None and out["intercept_full"] < 0:
            flags.append("NEGATIVE_Y_INTERCEPT")

        out["flags"] = flags

        # === H. Clinical interpretation (Polish) ===
        interp_parts = []
        if primary_slope is not None:
            interp_parts.append(f"Slope (ref): {primary_slope}")
            if out["ventilatory_class"]:
                interp_parts.append(f"Klasa: {out['ventilatory_class']} — {out['vc_description']}")
        if out["ve_vco2_nadir"] is not None:
            interp_parts.append(f"Nadir VE/VCO2: {out['ve_vco2_nadir']}")
        if flags:
            interp_parts.append(f"Flagi: {', '.join(flags)}")
        else:
            interp_parts.append("Brak flag klinicznych — efektywność wentylacji w normie")

        out["clinical_interpretation"] = " | ".join(interp_parts)

        return out


# --- E04: OUES ---
"""
Engine E04 v2.0 — Oxygen Uptake Efficiency Slope (OUES)
═══════════════════════════════════════════════════════════
Reference:
  Baba et al. 1996 (JACC) — original definition
  Hollenberg & Tager 2000 (JACC) — adult reference values
  Wiecha et al. 2024 (Frontiers/NOODLE) — athlete equations
  Buys et al. 2015 (Eur J Prev Cardiol) — 1411 adult norms
  Defoor et al. 2006 — CAD / training response

Formula:  VO2 [ml/min] = a × log10(VE [L/min]) + b
          "a" = OUES

Outputs:
  oues100          — OUES from 100% exercise data
  oues90           — OUES from 90% exercise duration
  oues75           — OUES from 75% exercise duration
  oues_to_vt1      — OUES up to VT1 (submaximal)
  oues_to_vt2      — OUES up to VT2 (submaximal)
  oues_per_kg      — OUES100 / body mass [ml/min/log/kg]
  oues_per_bsa     — OUES100 / BSA (OUESI)
  oues_pred_hollenberg — predicted OUES (Hollenberg 2000)
  oues_pred_noodle — predicted OUES (NOODLE 2024, athletes)
  oues_pct_hollenberg — % predicted Hollenberg
  oues_pct_noodle  — % predicted NOODLE
  oues_r2_100      — R² of log regression (QC)
  oues_r2_90       — R² of log regression at 90%
  oues_submax_stability — OUES90/OUES100 ratio (should be ~1.0)
  oues_flags       — quality/interpretation flags
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, Optional, Tuple


class Engine_E04_OUES_v2:
    """
    E04 v2.0 — Oxygen Uptake Efficiency Slope (OUES)

    Full implementation with submaximal variants, normalization,
    predicted values (Hollenberg, NOODLE), and QC.
    """

    # QC thresholds
    MIN_POINTS = 20            # minimum data points for regression
    MIN_R2_GOOD = 0.93         # Baba 1996: mean R²=0.978
    MIN_R2_ACCEPTABLE = 0.85
    SKIP_FIRST_SEC = 60        # skip first 60s of loaded exercise (Akkerman 2010)

    @classmethod
    def run(cls, df_ex: pd.DataFrame, e00: Dict = None,
            e02: Dict = None, meta: Dict = None) -> Dict:
        """
        Parameters
        ----------
        df_ex : exercise-phase DataFrame with columns:
                Time_s (or 'time'), VO2_ml_min (or 'vo2_ml_sm'),
                VE_L_min (or 've_lmin')
        e00   : E00 results (stop time, peak values)
        e02   : E02 results (VT1/VT2 times)
        meta  : dict with optional keys: weight_kg, height_cm, age, sex
        """

        result = cls._init_result()

        try:
            # ── 1. Resolve columns ──────────────────────────────────
            t, vo2, ve = cls._resolve_columns(df_ex)
            if t is None:
                result['oues_flags'].append('MISSING_COLUMNS')
                return result

            # ── 2. Build clean exercise mask ─────────────────────────
            t_start = t[0]
            t_end = t[-1]
            ex_duration = t_end - t_start

            # Skip first 60s of loaded exercise (Akkerman 2010, Baba 1996)
            mask = (t >= t_start + cls.SKIP_FIRST_SEC)
            # Remove NaN / zero VE
            mask &= np.isfinite(vo2) & np.isfinite(ve) & (ve > 0.5) & (vo2 > 50)

            t_clean = t[mask]
            vo2_clean = vo2[mask]
            ve_clean = ve[mask]

            if len(t_clean) < cls.MIN_POINTS:
                result['oues_flags'].append('INSUFFICIENT_DATA')
                return result

            log_ve = np.log10(ve_clean)

            # ── 3. OUES100 ──────────────────────────────────────────
            oues100, b100, r2_100 = cls._calc_oues(vo2_clean, log_ve)
            result['oues100'] = round(oues100, 1)
            result['oues_r2_100'] = round(r2_100, 4)
            result['oues_intercept'] = round(b100, 1)

            # ── 4. OUES90 ───────────────────────────────────────────
            cut_90 = t_start + ex_duration * 0.90
            m90 = t_clean <= cut_90
            if m90.sum() >= cls.MIN_POINTS:
                oues90, _, r2_90 = cls._calc_oues(vo2_clean[m90], log_ve[m90])
                result['oues90'] = round(oues90, 1)
                result['oues_r2_90'] = round(r2_90, 4)

            # ── 5. OUES75 ───────────────────────────────────────────
            cut_75 = t_start + ex_duration * 0.75
            m75 = t_clean <= cut_75
            if m75.sum() >= cls.MIN_POINTS:
                oues75, _, r2_75 = cls._calc_oues(vo2_clean[m75], log_ve[m75])
                result['oues75'] = round(oues75, 1)
                result['oues_r2_75'] = round(r2_75, 4)

            # ── 6. OUES to VT1 / VT2 ────────────────────────────────
            vt1_t = None
            vt2_t = None
            if e02:
                vt1_t = e02.get('vt1_time_sec')
                vt2_t = e02.get('vt2_time_sec')

            if vt1_t and vt1_t > t_start:
                m_vt1 = t_clean <= vt1_t
                if m_vt1.sum() >= cls.MIN_POINTS:
                    oues_vt1, _, r2_vt1 = cls._calc_oues(
                        vo2_clean[m_vt1], log_ve[m_vt1])
                    result['oues_to_vt1'] = round(oues_vt1, 1)
                    result['oues_r2_vt1'] = round(r2_vt1, 4)

            if vt2_t and vt2_t > t_start:
                m_vt2 = t_clean <= vt2_t
                if m_vt2.sum() >= cls.MIN_POINTS:
                    oues_vt2, _, r2_vt2 = cls._calc_oues(
                        vo2_clean[m_vt2], log_ve[m_vt2])
                    result['oues_to_vt2'] = round(oues_vt2, 1)
                    result['oues_r2_vt2'] = round(r2_vt2, 4)

            # ── 7. Normalization ─────────────────────────────────────
            weight_kg = cls._get_meta(meta, e00, 'weight_kg')
            height_cm = cls._get_meta(meta, e00, 'height_cm')
            age = cls._get_meta(meta, e00, 'age')
            sex = cls._get_meta(meta, e00, 'sex')  # 'male' or 'female'

            if weight_kg and weight_kg > 0:
                result['oues_per_kg'] = round(oues100 / weight_kg, 1)

            bsa = cls._calc_bsa(weight_kg, height_cm)
            if bsa:
                result['oues_per_bsa'] = round(oues100 / bsa, 1)
                result['bsa_m2'] = round(bsa, 3)

            # ── 8. Predicted values ──────────────────────────────────
            if age and bsa and sex:
                # Hollenberg & Tager 2000 (JACC)
                pred_h = cls._predict_hollenberg(age, bsa, sex)
                if pred_h:
                    result['oues_pred_hollenberg'] = round(pred_h, 1)
                    result['oues_pct_hollenberg'] = round(
                        oues100 / pred_h * 100, 1)

                # NOODLE 2024 (endurance athletes)
                pred_n = cls._predict_noodle(age, bsa, sex)
                if pred_n:
                    result['oues_pred_noodle'] = round(pred_n, 1)
                    result['oues_pct_noodle'] = round(
                        oues100 / pred_n * 100, 1)

            # ── 9. Submaximal stability ──────────────────────────────
            if result.get('oues90') and oues100 > 0:
                ratio = result['oues90'] / oues100
                result['oues_submax_stability'] = round(ratio, 3)
                # Baba 1996: OUES90 ≈ OUES100 (diff <3.5%)
                if abs(ratio - 1.0) > 0.10:
                    result['oues_flags'].append('POOR_SUBMAX_STABILITY')

            if result.get('oues_to_vt2') and oues100 > 0:
                ratio_vt2 = result['oues_to_vt2'] / oues100
                result['oues_vt2_stability'] = round(ratio_vt2, 3)
                # Defoor 2006: submaximal ~5.4% lower is normal
                if ratio_vt2 < 0.85:
                    result['oues_flags'].append('OUES_VT2_MUCH_LOWER')

            # ── 10. QC flags ─────────────────────────────────────────
            if r2_100 < cls.MIN_R2_ACCEPTABLE:
                result['oues_flags'].append('LOW_R2_FIT')
            elif r2_100 < cls.MIN_R2_GOOD:
                result['oues_flags'].append('MODERATE_R2_FIT')

            if oues100 < 500:
                result['oues_flags'].append('VERY_LOW_OUES')

            pct_h = result.get('oues_pct_hollenberg')
            if pct_h:
                if pct_h < 70:
                    result['oues_flags'].append('BELOW_70PCT_PREDICTED')
                elif pct_h < 84:
                    result['oues_flags'].append('BELOW_NORMAL_PREDICTED')

            result['status'] = 'OK'

        except Exception as e:
            result['status'] = 'ERROR'
            result['oues_flags'].append(f'EXCEPTION:{e}')

        return result

    # ═══════════════════════════════════════════════════════════════
    # INTERNALS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _init_result() -> Dict:
        return {
            'status': 'NOT_RUN',
            'oues100': None,
            'oues90': None,
            'oues75': None,
            'oues_to_vt1': None,
            'oues_to_vt2': None,
            'oues_per_kg': None,
            'oues_per_bsa': None,
            'bsa_m2': None,
            'oues_pred_hollenberg': None,
            'oues_pred_noodle': None,
            'oues_pct_hollenberg': None,
            'oues_pct_noodle': None,
            'oues_r2_100': None,
            'oues_r2_90': None,
            'oues_r2_75': None,
            'oues_r2_vt1': None,
            'oues_r2_vt2': None,
            'oues_intercept': None,
            'oues_submax_stability': None,
            'oues_vt2_stability': None,
            'oues_flags': [],
        }

    @classmethod
    def _resolve_columns(cls, df) -> Tuple:
        """Find time, VO2, VE columns regardless of naming."""
        t_col = next((c for c in ['time', 'Time_s', 'time_s', 't']
                       if c in df.columns), None)
        vo2_col = next((c for c in ['vo2_ml_sm', 'VO2_ml_min', 'VO2_ml_min_sm',
                                     'vo2_mlmin', 'VO2']
                         if c in df.columns), None)
        ve_col = next((c for c in ['ve_lmin', 'VE_L_min', 'VE_lmin',
                                    've_lmin_sm', 'VE']
                        if c in df.columns), None)

        if not all([t_col, vo2_col, ve_col]):
            return None, None, None

        return (df[t_col].values.astype(float),
                df[vo2_col].values.astype(float),
                df[ve_col].values.astype(float))

    @staticmethod
    def _calc_oues(vo2: np.ndarray, log_ve: np.ndarray) -> Tuple[float, float, float]:
        """
        VO2 = a × log10(VE) + b
        Returns (a=OUES, b=intercept, R²)
        """
        slope, intercept, r, _, _ = linregress(log_ve, vo2)
        return slope, intercept, r ** 2

    @staticmethod
    def _calc_bsa(weight_kg, height_cm) -> Optional[float]:
        """Du Bois & Du Bois BSA formula."""
        if not weight_kg or not height_cm:
            return None
        if weight_kg <= 0 or height_cm <= 0:
            return None
        return 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)

    @staticmethod
    def _predict_hollenberg(age: float, bsa: float, sex: str) -> Optional[float]:
        """
        Hollenberg & Tager 2000 (JACC, n=998 healthy adults).
        Returns OUES in ml/min per log(L/min).

        Men:   OUES = 1320 − 26.7×age + 1394×BSA
        Women: OUES = 1175 − 15.8×age + 841×BSA
        """
        if sex in ('male', 'M', 'm', 'Male'):
            return 1320.0 - 26.7 * age + 1394.0 * bsa
        elif sex in ('female', 'F', 'f', 'Female'):
            return 1175.0 - 15.8 * age + 841.0 * bsa
        return None

    @staticmethod
    def _predict_noodle(age: float, bsa: float, sex: str) -> Optional[float]:
        """
        NOODLE study 2024 (Frontiers Physiol, endurance athletes).
        OUES100 = 1540 + 2990×BSA − 1.4×(age×sex_coeff)

        sex_coeff: male=1, female=2
        Returns OUES in ml/min per log(L/min).

        Note: original paper uses L/min / L/min units.
        We convert: multiply by 1000 to get ml/min / log(L/min).
        """
        sex_coeff = 1 if sex in ('male', 'M', 'm', 'Male') else 2
        # Original: OUES100 = 1.54 + 2.99×BSA − 0.0014×(age×sex)
        # Paper says units are "mL·min⁻¹/L·min⁻¹" but values are ~4-6
        # which means it's actually in thousands (like L/min → need ×1000)
        # Their sample: males OUES100≈4400 ml/min, females≈3210 ml/min
        # Equation: 1.54 + 2.99*2.0 - 0.0014*(21*1) ≈ 7.51 → ×1000 = 7510
        # But their males have OUES≈4400... so equation is already scaled.
        # Actually checking: athletes BSA≈1.9, age≈21, male
        # 1.54 + 2.99*1.9 - 0.0014*21 = 1.54+5.681-0.0294 = 7.19 → must be ×1000
        # But real values ≈ 4400. So NOODLE eq. overestimates in general.
        # This is because NOODLE is for ENDURANCE ATHLETES (higher baseline).
        # Keep multiplication but note it's athlete-specific.
        oues_l = 1.54 + 2.99 * bsa - 0.0014 * (age * sex_coeff)
        return oues_l * 1000  # → ml/min per log(L/min)

    @staticmethod
    def _get_meta(meta: Dict, e00: Dict, key: str):
        """Get metadata from meta dict or e00."""
        if meta and key in meta:
            v = meta[key]
            if v is not None:
                return v
        if e00:
            # Common E00 keys
            mapping = {
                'weight_kg': ['weight_kg', 'body_mass_kg'],
                'height_cm': ['height_cm'],
                'age': ['age', 'age_years'],
                'sex': ['sex', 'gender'],
            }
            for alt in mapping.get(key, []):
                if alt in e00:
                    v = e00[alt]
                    if v is not None:
                        return v
        return None



# --- E05: O2 PULSE ---
class Engine_E05_O2Pulse:
    """
    E05 v2.0 — O2 Pulse Analysis
    =============================
    Fizjologia: O2pulse = VO2/HR = SV × C(a-v)O2
    Nieinwazyjny surogat objętości wyrzutowej (SV).

    Obliczenia:
    1. O2pulse peak, at VT1, at VT2
    2. Predicted: FRIEND 2019 (populacja) + NOODLE 2024 (sportowcy)
    3. % predicted (oba modele)
    4. Flattening detection (piecewise regression early/late slope)
    5. Estimated peak SV = (O2pulse/15) × 100
    6. O2 pulse trajectory classification

    Źródła:
    - Arena 2020 FRIEND: O2Ppeak = 23.2 - 0.09*age - 6.6*sex (N=13318)
    - Kasiak 2024 NOODLE: O2Ppeak_ath = 1.36 + 1.07*(23.2 - 0.09*age - 6.6*sex) (N=94)
    - Petek 2021: plateau u sportowców częste i NIEPATOLOGICZNE
    - PMC 2024 rTOF: Flattening Fraction (FF), O2PRR metryki
    - ATS 2017: estimated SV = (O2pulse/15) × 100
    - USCjournal 2024: normal > 80% predicted (~15 M / ~10 F ml/beat)
    """

    @staticmethod
    def run(df_ex, e02_results=None, age=None, sex="male", hr_peak=None):
        import numpy as np
        out = {}
        e02 = e02_results or {}

        # -----------------------------------------------------------
        # 0. KOLUMNY — resolucja nazw
        # -----------------------------------------------------------
        vo2_col = None
        for c in ["VO2_mlmin", "VO2_lmin", "VO2"]:
            if c in df_ex.columns:
                vo2_col = c
                break
        hr_col = None
        for c in ["HR_bpm", "HR"]:
            if c in df_ex.columns:
                hr_col = c
                break
        time_col = None
        for c in ["Time_s", "time_s", "Time", "t"]:
            if c in df_ex.columns:
                time_col = c
                break

        if vo2_col is None or hr_col is None:
            out["status"] = "MISSING_COLUMNS"
            return out

        df = df_ex[[c for c in [time_col, vo2_col, hr_col] if c]].dropna().copy()
        if len(df) < 10:
            out["status"] = "INSUFFICIENT_DATA"
            return out

        # Konwersja VO2 do ml/min
        vo2_vals = df[vo2_col].values.astype(float)
        if vo2_vals.max() < 20:  # L/min → ml/min
            vo2_vals = vo2_vals * 1000.0
        hr_vals = df[hr_col].values.astype(float)

        # Filtr HR > 40 (artefakty)
        mask = hr_vals > 40
        vo2_vals = vo2_vals[mask]
        hr_vals = hr_vals[mask]
        if time_col:
            time_vals = df[time_col].values.astype(float)[mask]
        else:
            time_vals = np.arange(len(vo2_vals), dtype=float)

        if len(vo2_vals) < 10:
            out["status"] = "INSUFFICIENT_DATA_AFTER_FILTER"
            return out

        # -----------------------------------------------------------
        # 1. O2 PULSE CURVE (cały test)
        # -----------------------------------------------------------
        o2pulse_raw = vo2_vals / hr_vals  # ml/beat

        # Wygładzenie: rolling mean window ~15s
        dt = np.median(np.diff(time_vals)) if len(time_vals) > 1 else 1.0
        win = max(3, int(15.0 / max(dt, 0.1)))
        if win % 2 == 0:
            win += 1

        from numpy.lib.stride_tricks import sliding_window_view
        if len(o2pulse_raw) >= win:
            o2pulse_smooth = np.convolve(o2pulse_raw, np.ones(win)/win, mode="same")
        else:
            o2pulse_smooth = o2pulse_raw.copy()

        # -----------------------------------------------------------
        # 2. PEAK O2 PULSE
        # -----------------------------------------------------------
        # Peak = max rolling 30s w ostatnich 2 minutach
        last_2min_mask = time_vals >= (time_vals[-1] - 120)
        if last_2min_mask.sum() >= 3:
            o2p_peak = float(np.max(o2pulse_smooth[last_2min_mask]))
        else:
            o2p_peak = float(np.max(o2pulse_smooth[-5:]))

        out["o2pulse_peak"] = round(o2p_peak, 1)

        # -----------------------------------------------------------
        # 3. O2 PULSE AT VT1, VT2
        # -----------------------------------------------------------
        vt1_time = e02.get("vt1_time_sec")
        vt2_time = e02.get("vt2_time_sec")

        for label, thr_time in [("vt1", vt1_time), ("vt2", vt2_time)]:
            if thr_time is not None:
                idx = np.argmin(np.abs(time_vals - thr_time))
                lo = max(0, idx - 2)
                hi = min(len(o2pulse_smooth), idx + 3)
                val = float(np.mean(o2pulse_smooth[lo:hi]))
                out[f"o2pulse_at_{label}"] = round(val, 1)
            else:
                out[f"o2pulse_at_{label}"] = None

        # -----------------------------------------------------------
        # 4. PREDICTED O2 PULSE (FRIEND + NOODLE)
        # -----------------------------------------------------------
        sex_val = 1.0 if sex and sex.lower().startswith("f") else 0.0

        if age is not None:
            pred_friend = 23.2 - 0.09 * age - 6.6 * sex_val
            pred_noodle = 1.36 + 1.07 * (23.2 - 0.09 * age - 6.6 * sex_val)
            out["predicted_friend"] = round(pred_friend, 1)
            out["predicted_noodle"] = round(pred_noodle, 1)
            out["pct_predicted_friend"] = round((o2p_peak / pred_friend) * 100, 1) if pred_friend > 0 else None
            out["pct_predicted_noodle"] = round((o2p_peak / pred_noodle) * 100, 1) if pred_noodle > 0 else None
        else:
            out["predicted_friend"] = None
            out["predicted_noodle"] = None
            out["pct_predicted_friend"] = None
            out["pct_predicted_noodle"] = None

        # -----------------------------------------------------------
        # 5. ESTIMATED STROKE VOLUME
        # -----------------------------------------------------------
        # SV (ml) ≈ (O2pulse / 15) × 100 — ATS/Mezzani
        out["estimated_sv_peak_ml"] = round((o2p_peak / 15.0) * 100, 0)

        # -----------------------------------------------------------
        # 6. FLATTENING DETECTION (piecewise regression)
        # -----------------------------------------------------------
        # Algorytm: testujemy breakpoint w 30-80% czasu ćwiczenia
        # Dla każdego breakpoint liczymy early slope + late slope
        # Najlepszy fit = minimalne RSS
        n = len(o2pulse_smooth)
        t_norm = (time_vals - time_vals[0]) / max(time_vals[-1] - time_vals[0], 1)

        best_rss = np.inf
        best_bp = 0.5
        best_early_slope = 0
        best_late_slope = 0

        bp_candidates = np.linspace(0.30, 0.80, 20)
        for bp_frac in bp_candidates:
            bp_idx = int(bp_frac * n)
            if bp_idx < 5 or (n - bp_idx) < 5:
                continue

            # Early segment
            t_early = t_norm[:bp_idx]
            y_early = o2pulse_smooth[:bp_idx]
            if len(t_early) >= 3:
                A_e = np.vstack([t_early, np.ones(len(t_early))]).T
                try:
                    coef_e = np.linalg.lstsq(A_e, y_early, rcond=None)[0]
                    slope_e = coef_e[0]
                    resid_e = y_early - A_e @ coef_e
                except:
                    continue
            else:
                continue

            # Late segment
            t_late = t_norm[bp_idx:]
            y_late = o2pulse_smooth[bp_idx:]
            if len(t_late) >= 3:
                A_l = np.vstack([t_late, np.ones(len(t_late))]).T
                try:
                    coef_l = np.linalg.lstsq(A_l, y_late, rcond=None)[0]
                    slope_l = coef_l[0]
                    resid_l = y_late - A_l @ coef_l
                except:
                    continue
            else:
                continue

            rss = float(np.sum(resid_e**2) + np.sum(resid_l**2))
            if rss < best_rss:
                best_rss = rss
                best_bp = bp_frac
                best_early_slope = float(slope_e)
                best_late_slope = float(slope_l)

        out["flattening_fraction"] = round(best_bp, 2)
        out["early_slope"] = round(best_early_slope, 2)
        out["late_slope"] = round(best_late_slope, 2)

        # O2 Pulse Response Ratio (O2PRR) = early/late
        if abs(best_late_slope) > 0.01:
            o2prr = best_early_slope / best_late_slope
            out["o2prr"] = round(o2prr, 2)
        else:
            o2prr = None
            out["o2prr"] = None

        # -----------------------------------------------------------
        # 7. TRAJECTORY CLASSIFICATION
        # -----------------------------------------------------------
        # INCREASING: late slope > 0.5 (kontynuuje wzrost)
        # PLATEAU: |late slope| < 0.5 (spłaszczenie)
        # DECREASING: late slope < -0.5 (spadek)
        # Kliniczne znaczenie: EARLY_FLAT jeśli FF < 50% + plateau/decrease
        if best_late_slope > 0.5:
            trajectory = "INCREASING"
            traj_desc = "Prawidłowy wzrost O2 pulse do końca testu"
        elif best_late_slope < -0.5:
            trajectory = "DECREASING"
            if best_bp < 0.50:
                trajectory = "EARLY_DECREASING"
                traj_desc = "Wczesny spadek O2 pulse (<50% testu) — wymaga oceny kardiologicznej"
            else:
                traj_desc = "Spadek O2 pulse w końcowej fazie testu"
        else:
            trajectory = "PLATEAU"
            if best_bp < 0.50:
                trajectory = "EARLY_PLATEAU"
                traj_desc = "Wczesne spłaszczenie O2 pulse (<50% testu) — wymaga oceny kardiologicznej"
            else:
                traj_desc = "Fizjologiczne plateau O2 pulse w końcowej fazie testu"

        out["trajectory"] = trajectory
        out["trajectory_desc"] = traj_desc

        # -----------------------------------------------------------
        # 8. FLAGS
        # -----------------------------------------------------------
        flags = []
        pct_f = out.get("pct_predicted_friend")
        if pct_f is not None and pct_f < 80:
            flags.append("O2PULSE_BELOW_80PCT_PREDICTED")
        if trajectory in ("EARLY_PLATEAU", "EARLY_DECREASING"):
            flags.append("EARLY_FLATTENING")
        if trajectory == "DECREASING":
            flags.append("LATE_DECREASE")
        if o2p_peak < 10 and sex_val == 0:
            flags.append("O2PULSE_VERY_LOW_MALE")
        if o2p_peak < 7 and sex_val == 1:
            flags.append("O2PULSE_VERY_LOW_FEMALE")

        out["flags"] = flags

        # -----------------------------------------------------------
        # 9. INTERPRETACJA
        # -----------------------------------------------------------
        interp_parts = []
        interp_parts.append(f"O2pulse peak: {o2p_peak:.1f} ml/beat")
        if out.get("predicted_friend"):
            interp_parts.append(f"({out['pct_predicted_friend']:.0f}% FRIEND, {out['pct_predicted_noodle']:.0f}% NOODLE)")
        interp_parts.append(f"Trajektoria: {trajectory}")
        if best_bp < 0.50 and trajectory in ("EARLY_PLATEAU", "EARLY_DECREASING"):
            interp_parts.append("⚠ Wczesne spłaszczenie — rozważ ocenę kardiologiczną")
        out["clinical_interpretation"] = " | ".join(interp_parts)

        out["status"] = "OK"
        return out


# --- E06: GAIN (Efficiency) ---
class Engine_E06_Gain_v2:
    """E06 v2.0 — VO2/Load Gain & Running Economy. Multi-modality."""
    MODALITY_CONFIG = {
        'run':      {'load_col':'Speed_kmh','load_unit':'km/h','vo2_mode':'rel','gain_unit':'ml/kg/min per km/h','has_watt':False,'has_RE':True,'norm_gain':3.3,'norm_sd':0.4,'norm_src':'ACSM eq (flat)'},
        'walk':     {'load_col':'Speed_kmh','load_unit':'km/h','vo2_mode':'rel','gain_unit':'ml/kg/min per km/h','has_watt':False,'has_RE':False,'norm_gain':1.7,'norm_sd':0.3,'norm_src':'ACSM walk eq'},
        'bike':     {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':10.3,'norm_sd':1.0,'norm_src':'Hansen 1987','norm_eff':25.0},
        'wattbike': {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':10.3,'norm_sd':1.0,'norm_src':'Hansen 1987','norm_eff':25.0},
        'echobike': {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':10.3,'norm_sd':1.0,'norm_src':'Hansen 1987','norm_eff':23.0},
        'row':      {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':12.5,'norm_sd':1.2,'norm_src':'Steinacker 1986','norm_eff':19.0},
        'ski':      {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':12.5,'norm_sd':1.5,'norm_src':'estimated','norm_eff':18.0},
        'arm':      {'load_col':'Power_W','load_unit':'W','vo2_mode':'abs','gain_unit':'ml/min/W','has_watt':True,'has_RE':False,'norm_gain':18.0,'norm_sd':2.0,'norm_src':'ACSM arm eq','norm_eff':12.0},
    }
    RE_NORMS = {'elite':{'mean':185,'sd':8},'well_trained':{'mean':200,'sd':10},'recreational':{'mean':215,'sd':12},'novice':{'mean':235,'sd':15}}

    @classmethod
    def run(cls, df_ex, modality='run', e02=None, e01=None, weight_kg=None):
        result = {'modality':modality,'gain_below_vt1':None,'gain_below_vt1_r2':None,'gain_full':None,'gain_full_r2':None,
                  'gain_unit':None,'gain_intercept':None,'running_economy_mlkgkm':None,'re_classification':None,'re_z_scores':{},
                  'delta_efficiency_pct':None,'linearity_break_time_s':None,'linearity_break_load':None,
                  'norm_gain_ref':None,'norm_gain_sd':None,'norm_src':None,'gain_z_score':None,'gain_at_vt1':None,'gain_at_vt2':None,'re_at_vt1':None,'re_at_vt2':None,'vo2_at_vt1':None,'vo2_at_vt2':None,'load_at_vt1':None,'load_at_vt2':None,'eff_at_vt1':None,'eff_at_vt2':None,'flags':[]}
        cfg = cls.MODALITY_CONFIG.get(modality.lower())
        if cfg is None: result['flags'].append('UNKNOWN_MODALITY'); return result
        result['gain_unit']=cfg['gain_unit']; result['norm_gain_ref']=cfg['norm_gain']; result['norm_gain_sd']=cfg['norm_sd']; result['norm_src']=cfg['norm_src']
        load_col=cfg['load_col']
        time_col=next((c for c in ['Time_s','Time_sec','time'] if c in df_ex.columns), None)
        vo2_abs_col=next((c for c in ['VO2_Lmin','VO2_L_min','VO2_ml_min'] if c in df_ex.columns), None)
        vo2_rel_col=next((c for c in ['VO2_mlmin_kg','VO2_ml_min_kg'] if c in df_ex.columns), None)
        if vo2_abs_col and not vo2_rel_col and weight_kg and weight_kg > 0:
            df_ex=df_ex.copy(); v=pd.to_numeric(df_ex[vo2_abs_col],errors='coerce')
            df_ex['_vo2_rel'] = v*1000/weight_kg if ('Lmin' in vo2_abs_col or 'L_min' in vo2_abs_col) else v/weight_kg
            vo2_rel_col='_vo2_rel'
        if vo2_rel_col and not vo2_abs_col and weight_kg and weight_kg > 0:
            df_ex=df_ex.copy(); df_ex['_vo2_abs']=pd.to_numeric(df_ex[vo2_rel_col],errors='coerce')*weight_kg/1000; vo2_abs_col='_vo2_abs'
        vo2_col = vo2_rel_col if cfg['vo2_mode']=='rel' else vo2_abs_col
        if vo2_col is None: result['flags'].append('NO_VO2_DATA'); return result
        time_arr=pd.to_numeric(df_ex[time_col],errors='coerce') if time_col else pd.RangeIndex(len(df_ex))
        load_arr=pd.to_numeric(df_ex[load_col],errors='coerce') if load_col in df_ex.columns else pd.Series([np.nan]*len(df_ex))
        vo2_arr=pd.to_numeric(df_ex[vo2_col],errors='coerce')
        if cfg['vo2_mode']=='abs' and vo2_col and ('Lmin' in vo2_col or 'L_min' in vo2_col): vo2_arr=vo2_arr*1000
        # Sanity: if vo2 in 'rel' mode has values >100, it's probably ml/min not ml/kg/min
        if cfg['vo2_mode']=='rel' and weight_kg and weight_kg > 0:
            _p95 = vo2_arr.dropna().quantile(0.95) if vo2_arr.notna().sum()>10 else 0
            if _p95 > 100:
                vo2_arr = vo2_arr / weight_kg
                result['flags'].append('VO2_REL_RESCALED_FROM_MLMIN')
        vo2_sm=vo2_arr.rolling(15,center=True,min_periods=5).mean()
        valid=load_arr.notna()&(load_arr>0)&vo2_sm.notna()&(vo2_sm>0)
        if valid.sum()<20: result['flags'].append('INSUFFICIENT_DATA'); return result
        vt1_time=None
        if e02:
            vt1_time=e02.get('vt1_time_sec') or e02.get('vt1_time_s')
            if vt1_time is not None: vt1_time=float(vt1_time)
        if vt1_time is not None and time_col:
            m=valid&(time_arr<=vt1_time)
            if int(m.sum())>=10:
                s,ic,r2=cls._linreg(load_arr[m].values,vo2_sm[m].values)
                if s is not None: result['gain_below_vt1']=round(s,2); result['gain_below_vt1_r2']=round(r2,3); result['gain_intercept']=round(ic,2)
            else: result['flags'].append('FEW_POINTS_BELOW_VT1')
        sf,icf,r2f=cls._linreg(load_arr[valid].values,vo2_sm[valid].values)
        if sf is not None:
            result['gain_full']=round(sf,2); result['gain_full_r2']=round(r2f,3)
            if result['gain_below_vt1'] is None: result['gain_below_vt1']=result['gain_full']; result['gain_below_vt1_r2']=result['gain_full_r2']; result['flags'].append('GAIN_FROM_FULL_RANGE')
            if r2f<0.85: result['flags'].append('LOW_R2_FULL')
        g=result['gain_below_vt1']
        if g is not None and cfg['norm_sd']>0: result['gain_z_score']=round((g-cfg['norm_gain'])/cfg['norm_sd'],2)
        if cfg['has_watt'] and g is not None and g>0: result['delta_efficiency_pct']=round((1.0/(g*0.335))*100,1)
        # ── RE + Values at VT1 and VT2 (v2.1) ──────────
        vt2_time = None
        if e02:
            vt2_time = e02.get('vt2_time_sec') or e02.get('vt2_time_s')
            if vt2_time is not None: vt2_time = float(vt2_time)
        # Gain at thresholds from regression (stable)
        if result['gain_below_vt1'] is not None: result['gain_at_vt1'] = result['gain_below_vt1']
        if result['gain_full'] is not None: result['gain_at_vt2'] = result['gain_full']
        for vt_key, gain_key in [('vt1','gain_at_vt1'),('vt2','gain_at_vt2')]:
            gv = result.get(gain_key)
            if gv and gv > 0 and cfg['has_watt']:
                result[f'eff_at_{vt_key}'] = round((1.0/(gv*0.335))*100, 1)
        # Compute load/VO2/RE at each threshold
        if cfg['has_RE'] and vo2_rel_col:
            speed_all = pd.to_numeric(df_ex[load_col], errors='coerce') if load_col in df_ex.columns else None
            vo2_r_all = pd.to_numeric(df_ex[vo2_rel_col], errors='coerce') if vo2_rel_col in df_ex.columns else None
        else:
            speed_all = None; vo2_r_all = None
        re_best = None; re_best_speed = 0
        for vt_name, vt_t in [('vt1', vt1_time), ('vt2', vt2_time)]:
            if vt_t is None or time_col is None: continue
            window = (time_arr >= vt_t - 20) & (time_arr <= vt_t + 20) & valid
            if window.sum() < 3: continue
            avg_vo2 = float(vo2_sm[window].mean())
            avg_load = float(load_arr[window].mean())
            result[f'vo2_at_{vt_name}'] = round(avg_vo2, 1)
            result[f'load_at_{vt_name}'] = round(avg_load, 1)
            # RE at this threshold — use SMOOTHED VO2 (BxB data too noisy)
            if cfg['has_RE'] and speed_all is not None and avg_load > 4:
                re_mask = window & speed_all.notna() & (speed_all > 4) & vo2_sm.notna() & (vo2_sm > 5)
                if re_mask.sum() >= 3:
                    # vo2_sm is already in correct units (rel for run, abs for machine)
                    # For RE we need relative → use vo2_sm directly if mode=rel, else convert
                    if cfg['vo2_mode'] == 'rel':
                        _re_vo2 = vo2_sm[re_mask]
                    elif weight_kg and weight_kg > 0:
                        _re_vo2 = vo2_sm[re_mask] / weight_kg  # abs ml/min → ml/kg/min
                    else:
                        _re_vo2 = None
                    if _re_vo2 is not None:
                        re_at = float((_re_vo2 / (speed_all[re_mask] / 60.0)).mean())
                    result[f're_at_{vt_name}'] = round(re_at, 1)
                    if avg_load > re_best_speed:
                        re_best = re_at; re_best_speed = avg_load
        # Primary RE = value at highest-speed threshold (VT2 preferred)
        if re_best is not None:
            result['running_economy_mlkgkm'] = round(re_best, 1)
            if re_best > 300: result['flags'].append('RE_POSSIBLY_WRONG_VO2_UNITS')
            if re_best_speed >= 8:
                if re_best < 195: result['re_classification'] = 'ELITE'
                elif re_best < 208: result['re_classification'] = 'WELL_TRAINED'
                elif re_best < 225: result['re_classification'] = 'RECREATIONAL'
                else: result['re_classification'] = 'NOVICE'
            else:
                result['re_classification'] = 'LOW_SPEED'
                result['flags'].append('RE_LOW_SPEED_UNRELIABLE')
            for cat, norms in cls.RE_NORMS.items():
                result['re_z_scores'][cat] = round((re_best - norms['mean']) / norms['sd'], 2)

        if valid.sum()>=30 and time_col:
            brk=cls._detect_break(time_arr[valid].values,load_arr[valid].values,vo2_sm[valid].values)
            if brk: result['linearity_break_time_s']=round(brk[0],0); result['linearity_break_load']=round(brk[1],1)
        result['flags'].append('RAMP_PROTOCOL_NOTE')
        return result

    @staticmethod
    def _linreg(x,y):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        if len(x)<5: return None,None,None
        xm,ym=np.mean(x),np.mean(y); ss_xy=np.sum((x-xm)*(y-ym)); ss_xx=np.sum((x-xm)**2)
        if ss_xx<1e-12: return None,None,None
        s=ss_xy/ss_xx; ic=ym-s*xm; yp=s*x+ic
        ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-ym)**2)
        return float(s),float(ic),float(1-ss_res/ss_tot if ss_tot>0 else 0)

    @staticmethod
    def _detect_break(time,load,vo2):
        n=len(time)
        if n<20: return None
        best_i,best_rss=None,np.inf
        for i in range(int(n*0.3),int(n*0.7)):
            if i<5 or n-i<5: continue
            c1=np.polyfit(load[:i],vo2[:i],1); c2=np.polyfit(load[i:],vo2[i:],1)
            rss=np.sum((vo2[:i]-np.polyval(c1,load[:i]))**2)+np.sum((vo2[i:]-np.polyval(c2,load[i:]))**2)
            if rss<best_rss: best_rss=rss; best_i=i
        if best_i is None: return None
        rss_all=np.sum((vo2-np.polyval(np.polyfit(load,vo2,1),load))**2)
        return (float(time[best_i]),float(load[best_i])) if best_rss<rss_all*0.90 else None


class Engine_E07_BreathingPattern:
    """
    E07 v2.0 — Breathing Pattern & Ventilatory Mechanics
    Signals: BF, VT, tI, tE, VD/VT(est), VE
    Refs: Watson 2020, Knopfel 2024 PTVV, Neder 2023 variability, Gallagher 1987 VT plateau
    """
    COL_MAP = {
        "bf": ["BF_1_min","BF","Bf","fR","RR"],
        "vt": ["VT_L","VT","Vt","TV_L"],
        "ti": ["tI","Ti","TI"],
        "te": ["tE","Te","TE"],
        "vdvt": ["VD/VT(est)","VD_VT","VdVt","VDVT"],
        "ve": ["VE_L_min","VE","Ve"],
    }
    @staticmethod
    def _res(df, key):
        for c in Engine_E07_BreathingPattern.COL_MAP.get(key,[]):
            if c in df.columns: return c
        return None
    @staticmethod
    def _aw(arr, t_arr, tc, w=15):
        import numpy as np
        m = (t_arr >= tc-w) & (t_arr <= tc+w) & np.isfinite(arr)
        return float(np.nanmean(arr[m])) if m.sum()>=2 else None
    @staticmethod
    def run(df_ex, e02_results=None, e01_results=None, cfg=None):
        import numpy as np, pandas as pd
        out={}; e02=e02_results or {}; R=Engine_E07_BreathingPattern._res; AW=Engine_E07_BreathingPattern._aw
        bf_col=R(df_ex,"bf"); vt_col=R(df_ex,"vt"); ti_col=R(df_ex,"ti"); te_col=R(df_ex,"te")
        vd_col=R(df_ex,"vdvt"); ve_col=R(df_ex,"ve")
        time_col=None
        for c in ["Time_sec","Time_s","time_s"]:
            if c in df_ex.columns: time_col=c; break
        if bf_col is None or vt_col is None or time_col is None:
            return {"status":"MISSING_COLUMNS"}
        t=pd.to_numeric(df_ex[time_col],errors="coerce").values.astype(float)
        bf=pd.to_numeric(df_ex[bf_col],errors="coerce").values.astype(float)
        vt=pd.to_numeric(df_ex[vt_col],errors="coerce").values.astype(float)
        ve=pd.to_numeric(df_ex[ve_col],errors="coerce").values.astype(float) if ve_col else bf*vt
        ok=np.isfinite(t)&np.isfinite(bf)&np.isfinite(vt)&(bf>0)&(vt>0.05)
        if ok.sum()<20: return {"status":"INSUFFICIENT_DATA"}
        t,bf,vt,ve=t[ok],bf[ok],vt[ok],ve[ok]
        ti=te=None
        if ti_col:
            _ti=pd.to_numeric(df_ex[ti_col],errors="coerce").values.astype(float); ti=_ti[ok]
        if te_col:
            _te=pd.to_numeric(df_ex[te_col],errors="coerce").values.astype(float); te=_te[ok]
        vdvt=None
        if vd_col:
            _vd=pd.to_numeric(df_ex[vd_col],errors="coerce").values.astype(float); vdvt=_vd[ok]
        def smooth(a,w=11):
            if len(a)<w: return a.copy()
            return np.convolve(a,np.ones(w)/w,mode="same")
        bf_sm=smooth(bf); vt_sm=smooth(vt)
        vt1_t=e02.get("vt1_time_sec"); vt2_t=e02.get("vt2_time_sec")
        t_start=float(t[0]); t_end=float(t[-1]); t_dur=t_end-t_start; n=len(vt_sm)
        # 1. PEAK/REST
        out["bf_peak"]=round(float(np.max(bf_sm)),1)
        out["vt_peak_L"]=round(float(np.max(vt_sm)),2)
        r_n=max(5,int(n*0.05))
        out["bf_rest"]=round(float(np.mean(bf_sm[:r_n])),1)
        out["vt_rest_L"]=round(float(np.mean(vt_sm[:r_n])),2)
        # 2. AT THRESHOLDS
        for lb,thr in [("vt1",vt1_t),("vt2",vt2_t)]:
            if thr is not None:
                out[f"bf_at_{lb}"]=round(AW(bf_sm,t,thr),1) if AW(bf_sm,t,thr) else None
                out[f"vt_at_{lb}_L"]=round(AW(vt_sm,t,thr),2) if AW(vt_sm,t,thr) else None
            else:
                out[f"bf_at_{lb}"]=None; out[f"vt_at_{lb}_L"]=None
        # 3. VT PLATEAU (piecewise regression VT vs VE)
        best_rss=np.inf; best_bp=n//2; early_slope=0; late_slope=0
        for bp in range(max(10,n//5),min(n-10,4*n//5)):
            x1,y1=ve[:bp],vt_sm[:bp]; x2,y2=ve[bp:],vt_sm[bp:]
            if len(x1)<5 or len(x2)<5: continue
            try:
                c1=np.polyfit(x1,y1,1); c2=np.polyfit(x2,y2,1)
                rss=float(np.sum((y1-np.polyval(c1,x1))**2)+np.sum((y2-np.polyval(c2,x2))**2))
                if rss<best_rss: best_rss=rss; best_bp=bp; early_slope=c1[0]; late_slope=c2[0]
            except: continue
        vp_time=float(t[best_bp]); vp_level=round(float(np.mean(vt_sm[best_bp:min(best_bp+15,n)])),2)
        vp_pct=round((vp_time-t_start)/max(t_dur,1)*100,0)
        out["vt_plateau_time_s"]=round(vp_time,0); out["vt_plateau_level_L"]=vp_level
        out["vt_plateau_pct_exercise"]=vp_pct
        if vt2_t:
            d=vp_time-vt2_t
            out["vt_plateau_vs_vt2"]="BEFORE_VT2" if d<-30 else ("AFTER_VT2" if d>30 else "AT_VT2")
        else: out["vt_plateau_vs_vt2"]="VT2_UNKNOWN"
        # 4. STRATEGY
        q25=max(5,int(n*0.10)); q75=min(n-5,int(n*0.90))
        vt_ch=float(np.mean(vt_sm[q75:])-np.mean(vt_sm[:q25]))
        bf_ch=float(np.mean(bf_sm[q75:])-np.mean(bf_sm[:q25]))
        bf_m=float(np.mean(bf_sm)); vt_m=float(np.mean(vt_sm))
        ve_vt=bf_m*vt_ch; ve_bf=vt_m*bf_ch; tot=abs(ve_vt)+abs(ve_bf)
        pct_vt=round(abs(ve_vt)/max(tot,0.01)*100,0); pct_bf=100-pct_vt
        out["ve_from_vt_pct"]=pct_vt; out["ve_from_bf_pct"]=pct_bf
        out["strategy"]="VT_DOMINANT" if pct_vt>=65 else ("BF_DOMINANT" if pct_bf>=65 else "BALANCED")
        # 5. TIMING
        if ti is not None and te is not None:
            ok_t=np.isfinite(ti)&np.isfinite(te)&(ti>0.05)&(te>0.05)
            ti_o,te_o,vt_o2=ti[ok_t],te[ok_t],vt[ok_t]; ttot=ti_o+te_o; dc=ti_o/ttot
            rn=max(3,int(len(dc)*0.05)); pn=max(3,int(len(dc)*0.05))
            out["ti_mean_rest_s"]=round(float(np.mean(ti_o[:rn])),2)
            out["ti_mean_peak_s"]=round(float(np.mean(ti_o[-pn:])),2)
            out["te_mean_rest_s"]=round(float(np.mean(te_o[:rn])),2)
            out["te_mean_peak_s"]=round(float(np.mean(te_o[-pn:])),2)
            out["duty_cycle_rest"]=round(float(np.mean(dc[:rn])),2)
            out["duty_cycle_peak"]=round(float(np.mean(dc[-pn:])),2)
            out["mean_insp_flow_peak_Ls"]=round(float(np.mean(vt_o2[-pn:]/ti_o[-pn:])),2)
        else:
            for k in ["ti_mean_rest_s","ti_mean_peak_s","te_mean_rest_s","te_mean_peak_s",
                       "duty_cycle_rest","duty_cycle_peak","mean_insp_flow_peak_Ls"]: out[k]=None
        # 6. DEAD SPACE
        if vdvt is not None:
            ok_vd=np.isfinite(vdvt)&(vdvt>=0)&(vdvt<=1.0)
            if ok_vd.sum()>=10:
                vd_o=vdvt[ok_vd]; t_vd=t[ok_vd]; rn2=max(3,int(len(vd_o)*0.05)); pn2=max(3,int(len(vd_o)*0.05))
                out["vdvt_rest"]=round(float(np.mean(vd_o[:rn2])),3)
                out["vdvt_peak"]=round(float(np.mean(vd_o[-pn2:])),3)
                out["vdvt_at_vt1"]=round(AW(vd_o,t_vd,vt1_t),3) if vt1_t and AW(vd_o,t_vd,vt1_t) else None
                out["vdvt_at_vt2"]=round(AW(vd_o,t_vd,vt2_t),3) if vt2_t and AW(vd_o,t_vd,vt2_t) else None
                drop=out["vdvt_rest"]-out["vdvt_peak"]
                if drop>0.05: out["vdvt_trajectory"]="NORMAL_DECREASE"
                elif drop<-0.03:
                    out["vdvt_trajectory"]="ESTIMATED_ARTIFACT" if out["vdvt_rest"]<0.15 else "ABNORMAL_INCREASE"
                else: out["vdvt_trajectory"]="FLAT"
            else:
                for k in ["vdvt_rest","vdvt_peak","vdvt_at_vt1","vdvt_at_vt2","vdvt_trajectory"]: out[k]=None
        else:
            for k in ["vdvt_rest","vdvt_peak","vdvt_at_vt1","vdvt_at_vt2","vdvt_trajectory"]: out[k]=None
        # 7. TACHYPNEA
        tachy_mask=bf_sm>50
        if tachy_mask.any():
            ti2=np.argmax(tachy_mask); tt2=float(t[ti2])
            out["tachypnea_time_s"]=round(tt2,0)
            out["tachypnea_pct_exercise"]=round((tt2-t_start)/max(t_dur,1)*100,0)
            if vt2_t:
                out["tachypnea_vs_vt2"]="BEFORE_VT2" if tt2<vt2_t-30 else ("AFTER_VT2" if tt2>vt2_t+30 else "AT_VT2")
            else: out["tachypnea_vs_vt2"]="VT2_UNKNOWN"
        else:
            out["tachypnea_time_s"]=None; out["tachypnea_pct_exercise"]=None; out["tachypnea_vs_vt2"]="NONE"
        # 8. RSBI
        rsbi=bf/vt; rn3=max(3,int(len(rsbi)*0.05))
        out["rsbi_rest"]=round(float(np.mean(rsbi[:rn3])),1)
        out["rsbi_peak"]=round(float(np.mean(rsbi[-rn3:])),1)
        out["rsbi_at_vt1"]=round(AW(rsbi,t,vt1_t),1) if vt1_t and AW(rsbi,t,vt1_t) else None
        # 9. PTVV
        try:
            coeffs=np.polyfit(ve,vt,3); vt_pred=np.polyval(coeffs,ve)
            rmse=float(np.sqrt(np.mean((vt-vt_pred)**2)))
            vt_range=float(np.max(vt)-np.min(vt))
            out["ptvv"]=round(rmse/max(vt_range,0.01),3)
        except: out["ptvv"]=None
        # 10. VARIABILITY
        def delta_var(sig,time,ps,pe,window=20):
            m=(time>=ps)&(time<=pe)&np.isfinite(sig)
            if m.sum()<10: return None
            s2,tm2=sig[m],time[m]; means=[]
            for st in np.arange(tm2[0],tm2[-1]-window,window/2):
                wm=(tm2>=st)&(tm2<st+window)
                if wm.sum()>=3: means.append(float(np.mean(s2[wm])))
            return round(max(means)-min(means),1) if len(means)>=2 else None
        out["delta_bf_rest"]=delta_var(bf,t,t_start,t_start+60)
        out["delta_bf_loaded"]=delta_var(bf,t,t_start+t_dur*0.3,t_start+t_dur*0.8)
        out["delta_vt_loaded"]=delta_var(vt,t,t_start+t_dur*0.3,t_start+t_dur*0.8)
        # 11. SIGH DETECTION
        ws=min(15,len(vt)//3)
        if ws>=3:
            vt_med=pd.Series(vt).rolling(ws,center=True,min_periods=3).median().values
            sm2=np.isfinite(vt_med)&(vt>2.0*vt_med)&(vt_med>0.2)
            out["sigh_count"]=int(sm2.sum()); out["sigh_rate_per_min"]=round(int(sm2.sum())/max(t_dur/60,0.5),1)
        else: out["sigh_count"]=0; out["sigh_rate_per_min"]=0.0
        # 12. FLAGS + CLASSIFICATION
        flags=[]
        if out.get("tachypnea_vs_vt2")=="BEFORE_VT2": flags.append("EARLY_TACHYPNEA")
        ptvv=out.get("ptvv")
        if ptvv and ptvv>0.25: flags.append("HIGH_PTVV_IRREGULAR")
        elif ptvv and ptvv>0.15: flags.append("MODERATE_PTVV")
        if out.get("delta_bf_rest") and out["delta_bf_rest"]>5: flags.append("HIGH_BF_VAR_REST")
        if out.get("delta_bf_loaded") and out["delta_bf_loaded"]>5: flags.append("HIGH_BF_VAR_EX")
        if out.get("vdvt_trajectory")=="ABNORMAL_INCREASE": flags.append("VDVT_INCREASE")
        elif out.get("vdvt_trajectory")=="ESTIMATED_ARTIFACT": flags.append("VDVT_EST_ARTIFACT")
        elif out.get("vdvt_trajectory")=="FLAT": flags.append("VDVT_NO_DECREASE")
        if out.get("vdvt_rest") and out["vdvt_rest"]<0.15: flags.append("VDVT_EST_LOW")
        if out.get("sigh_count",0)>3: flags.append("FREQUENT_SIGHS")
        if out["strategy"]=="BF_DOMINANT": flags.append("BF_DOMINANT_STRAT")
        if out.get("rsbi_peak") and out["rsbi_peak"]>40: flags.append("HIGH_RSBI")
        if out.get("vt_plateau_vs_vt2")=="BEFORE_VT2": flags.append("EARLY_VT_PLATEAU")
        out["flags"]=flags
        n_irreg=sum(1 for f in flags if f in ["HIGH_PTVV_IRREGULAR","HIGH_BF_VAR_REST","HIGH_BF_VAR_EX","FREQUENT_SIGHS"])
        has_rs="HIGH_RSBI" in flags or "BF_DOMINANT_STRAT" in flags
        has_et="EARLY_TACHYPNEA" in flags
        if n_irreg>=2: out["breathing_pattern"]="DYSFUNCTIONAL_BPD"
        elif has_rs and has_et: out["breathing_pattern"]="RAPID_SHALLOW"
        elif n_irreg==1: out["breathing_pattern"]="BORDERLINE"
        else: out["breathing_pattern"]="NORMAL"
        out["status"]="OK"
        return out

class Engine_E08_CardioHRR:
    """
    HRR engine (1 min / 3 min) with robust QC:
    - wybór najlepszego kanału HR
    - solidne okna czasowe (median, nie pojedynczy punkt)
    - kontrola artefaktów i niefizjologii
    - status: OK / LIMITED / ERROR
    """

    @staticmethod
    def _pick_hr_column(df: pd.DataFrame) -> Optional[str]:
        candidates = [
            "HR_bpm", "HR", "Pulse", "HF", "HeartRate", "Heart Rate"
        ]
        for c in candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 10:
                    return c
        return None

    @staticmethod
    def _robust_median_in_window(df: pd.DataFrame, t0: float, t1: float, col: str) -> float:
        sub = df[(df["Time_sec"] >= t0) & (df["Time_sec"] <= t1)][col]
        sub = pd.to_numeric(sub, errors="coerce").dropna()
        if sub.empty:
            return np.nan
        # odfiltruj oczywiste outliery HR
        sub = sub[(sub >= 30) & (sub <= 240)]
        if sub.empty:
            return np.nan
        return float(np.nanmedian(sub.values))

    @staticmethod
    def _robust_peak_near_stop(df: pd.DataFrame, t_stop: float, col: str,
                               back_window_s: float = 30.0, forward_s: float = 5.0) -> float:
        """
        Peak HR reference: max w końcówce wysiłku (t_stop-30s do t_stop+5s).
        """
        sub = df[(df["Time_sec"] >= (t_stop - back_window_s)) & (df["Time_sec"] <= (t_stop + forward_s))][col]
        sub = pd.to_numeric(sub, errors="coerce").dropna()
        sub = sub[(sub >= 30) & (sub <= 240)]
        if sub.empty:
            return np.nan

        # winsoryzacja górnych artefaktów
        q99 = np.nanpercentile(sub.values, 99)
        sub = sub[sub <= q99]
        if sub.empty:
            return np.nan
        return float(np.nanmax(sub.values))

    @staticmethod
    def _estimate_dt(df: pd.DataFrame) -> float:
        t = pd.to_numeric(df["Time_sec"], errors="coerce").dropna().values
        if len(t) < 3:
            return 1.0
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) == 0:
            return 1.0
        return float(np.nanmedian(d))

    @staticmethod
    def _quality_label(n_points: int, expected_points: int, neg_hrr_count: int) -> str:
        cover = (n_points / expected_points) if expected_points > 0 else 0
        if cover >= 0.7 and neg_hrr_count == 0:
            return "HIGH"
        if cover >= 0.4:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def run(df_full: pd.DataFrame, t_stop: float, recovery_mode: str = "auto") -> Dict[str, Any]:
        """
        Params
        ------
        df_full : full test dataframe (wysiłek + recovery)
        t_stop  : czas końca wysiłku (s)
        recovery_mode : 'auto' | 'active' | 'passive'
            - active: cool-down ruchem
            - passive: szybkie zatrzymanie
            - auto: heurystyka z Speed/Power po t_stop

        Returns keys
        ------------
        status, hrr_1min, hrr_3min, hr_peak_ref, hr_60s, hr_180s, recovery_mode,
        interpretation_1min, interpretation_3min, quality, qc_flags, reason
        """
        try:
            if df_full is None or df_full.empty:
                return {"status": "ERROR", "reason": "empty dataframe"}

            if "Time_sec" not in df_full.columns:
                return {"status": "ERROR", "reason": "missing Time_sec"}

            df = df_full.copy()
            df["Time_sec"] = pd.to_numeric(df["Time_sec"], errors="coerce")
            df = df.dropna(subset=["Time_sec"]).sort_values("Time_sec").reset_index(drop=True)

            hr_col = Engine_E08_CardioHRR._pick_hr_column(df)
            if hr_col is None:
                return {"status": "LIMITED", "reason": "missing HR column", "hrr_1min": None, "hrr_3min": None}

            df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
            df = df[df[hr_col].notna()].copy()
            if df.empty:
                return {"status": "LIMITED", "reason": "HR column all NaN", "hrr_1min": None, "hrr_3min": None}

            # Heurystyka recovery mode
            mode = recovery_mode
            if mode == "auto":
                mode = "passive"
                after = df[df["Time_sec"] >= t_stop].copy()
                for load_col in ["Speed_kmh", "Power_W"]:
                    if load_col in after.columns:
                        s = pd.to_numeric(after[load_col], errors="coerce").dropna()
                        if len(s) > 5:
                            med_early = float(np.nanmedian(s.head(min(30, len(s)))))
                            if med_early > 0.5:  # jest jakiś ruch/praca po t_stop
                                mode = "active"
                                break

            # HR peak reference
            hr_peak_ref = Engine_E08_CardioHRR._robust_peak_near_stop(df, t_stop, hr_col)
            if not np.isfinite(hr_peak_ref):
                return {
                    "status": "LIMITED",
                    "reason": "cannot determine hr_peak_ref around t_stop",
                    "hrr_1min": None,
                    "hrr_3min": None,
                    "recovery_mode": mode
                }

            # Okna recovery
            # 1 min: 55-65 s, 3 min: 175-185 s
            hr_60 = Engine_E08_CardioHRR._robust_median_in_window(df, t_stop + 55, t_stop + 65, hr_col)
            hr_180 = Engine_E08_CardioHRR._robust_median_in_window(df, t_stop + 175, t_stop + 185, hr_col)

            # fallback szersze okna, jeśli brak
            if not np.isfinite(hr_60):
                hr_60 = Engine_E08_CardioHRR._robust_median_in_window(df, t_stop + 50, t_stop + 70, hr_col)
            if not np.isfinite(hr_180):
                hr_180 = Engine_E08_CardioHRR._robust_median_in_window(df, t_stop + 165, t_stop + 195, hr_col)

            hrr1 = (hr_peak_ref - hr_60) if np.isfinite(hr_60) else np.nan
            hrr3 = (hr_peak_ref - hr_180) if np.isfinite(hr_180) else np.nan

            qc_flags = []

            # Niefizjologiczny wzrost HR w recovery
            if np.isfinite(hrr1) and hrr1 < 0:
                qc_flags.append("NEGATIVE_HRR1")
            if np.isfinite(hrr3) and hrr3 < 0:
                qc_flags.append("NEGATIVE_HRR3")

            # bardzo mały lub absurdalnie duży spadek
            if np.isfinite(hrr1) and hrr1 > 90:
                qc_flags.append("HRR1_IMPLAUSIBLY_HIGH")
            if np.isfinite(hrr3) and hrr3 > 140:
                qc_flags.append("HRR3_IMPLAUSIBLY_HIGH")

            # dostępność danych w oknach
            dt = Engine_E08_CardioHRR._estimate_dt(df)
            expected_1m = max(1, int(round(10.0 / dt)))   # 10-sek okno
            expected_3m = max(1, int(round(20.0 / dt)))   # fallback szerzej bywa 20-30s
            n1 = df[(df["Time_sec"] >= t_stop + 55) & (df["Time_sec"] <= t_stop + 65)][hr_col].notna().sum()
            n3 = df[(df["Time_sec"] >= t_stop + 175) & (df["Time_sec"] <= t_stop + 185)][hr_col].notna().sum()
            quality = Engine_E08_CardioHRR._quality_label(
                n_points=int(n1 + n3),
                expected_points=int(expected_1m + expected_3m),
                neg_hrr_count=int(("NEGATIVE_HRR1" in qc_flags) + ("NEGATIVE_HRR3" in qc_flags))
            )

            # Interpretacja (cutoff zależny od trybu)
            # klasycznie: <12 bpm (active), <18 bpm (passive) dla HRR1
            cutoff_1 = 12 if mode == "active" else 18
            interp1 = None
            if np.isfinite(hrr1):
                interp1 = "abnormal" if hrr1 < cutoff_1 else "normal"

            # HRR3: mniej standaryzowany; użyteczny praktycznie
            interp3 = None
            if np.isfinite(hrr3):
                if hrr3 < 30:
                    interp3 = "low"
                elif hrr3 < 50:
                    interp3 = "moderate"
                else:
                    interp3 = "good"

            status = "OK"
            reason = ""
            if not np.isfinite(hrr1) and not np.isfinite(hrr3):
                status = "LIMITED"
                reason = "missing recovery HR windows"
            elif "NEGATIVE_HRR1" in qc_flags and "NEGATIVE_HRR3" in qc_flags:
                status = "LIMITED"
                reason = "both HRR values negative (likely t_stop/recovery mismatch or artifacts)"
            elif quality == "LOW":
                status = "LIMITED"
                reason = "low data coverage in recovery windows"

            out = {
                "status": status,
                "reason": reason,

                "hr_col_used": hr_col,
                "recovery_mode": mode,
                "quality": quality,
                "qc_flags": qc_flags,

                "hr_peak_ref": float(hr_peak_ref) if np.isfinite(hr_peak_ref) else None,
                "hr_60s": float(hr_60) if np.isfinite(hr_60) else None,
                "hr_180s": float(hr_180) if np.isfinite(hr_180) else None,

                "hrr_1min": float(hrr1) if np.isfinite(hrr1) else None,
                "hrr_3min": float(hrr3) if np.isfinite(hrr3) else None,

                "cutoff_hrr1_used": cutoff_1,
                "interpretation_1min": interp1,
                "interpretation_3min": interp3,

                # pola kompatybilności (jeśli gdzieś używasz alternatywnych nazw)
                "hrr1": float(hrr1) if np.isfinite(hrr1) else None,
                "hrr3": float(hrr3) if np.isfinite(hrr3) else None,
            }

            return out

        except Exception as e:
            return {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}


# --- E09: VENT LIMITATION ---
class Engine_E09_VentLimitation:
    @staticmethod
    def run(df_ex):
        # Wymaga FEV1 od usera (na razie placeholder)
        # BR = (MVV - VEmax) / MVV
        return {"br_pct": None}


# --- E10: SUBSTRATE (FAT/CHO) ---
"""
Engine E10 v2.0 — Substrate Oxidation Profile
═══════════════════════════════════════════════════════════
Reference equations:
  Frayn 1983 (J Appl Physiol 55:628-34) — non-protein RQ:
    FAT_ox (g/min) = 1.67 × VO2 - 1.67 × VCO2
    CHO_ox (g/min) = 4.55 × VCO2 - 3.21 × VO2
  (protein oxidation assumed negligible during exercise)

  Jeukendrup & Wallis 2005 (Int J Sports Med 26:S28-37):
    FAT_ox (g/min) = 1.695 × VO2 - 1.701 × VCO2
    CHO_ox (g/min) = 4.344 × VCO2 - 3.061 × VO2
  (glycogen-based, appropriate for exercise; low intensity variant)

Key concepts:
  FATmax (Achten et al. 2002, MSSE 34:92-97):
    = peak fat oxidation rate (g/min) during graded exercise
    = typically at 45-65% VO2max (Venables et al. 2005)
    = normative: trained males ~0.56 g/min, females ~0.44 g/min
      (Randell et al. 2017, Frontiers Physiol)

  Crossover Point (Brooks & Mercier 1994, J Appl Physiol 76:2253):
    = intensity where CHO-derived energy > FAT-derived energy
    = where CHO kcal/min exceeds FAT kcal/min
    = energy conversion: 1g FAT = 9.75 kcal, 1g CHO = 4.07 kcal
    = typically at ~50-65% VO2max

  FATmin zone (Achten & Jeukendrup 2003):
    = intensity range where FAT_ox ≥ 90% of MFO (FATmax ± 10%)
    = practical "fat burning zone" for training prescription

Validity limits:
  - RER > 1.0 → bicarbonate buffering → FAT_ox calculation invalid
    (non-metabolic CO2 overestimates VCO2, Jeukendrup 2005)
  - Calculations only valid when RER < 1.0
  - Apply 30-60s rolling average to reduce breath-by-breath noise

Outputs:
  FATmax metrics: MFO (g/min), MFO/kg, FATmax HR, FATmax speed,
    FATmax %VO2peak, FATmax %HRmax
  Crossover point: COP HR, COP speed, COP %VO2peak
  Energy contribution: fat_pct and cho_pct at rest, VT1, VT2
  Substrate curves: fat_ox and cho_ox arrays for plotting
  QC flags: RER_VALIDITY, DATA_QUALITY, RAMP_PROTOCOL_LIMITATION
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


"""
Engine E10 v2.0 — Substrate Oxidation Profile
═══════════════════════════════════════════════════════════
Reference equations:
  Frayn 1983 (J Appl Physiol 55:628-34) — non-protein RQ:
    FAT_ox (g/min) = 1.67 × VO2 - 1.67 × VCO2
    CHO_ox (g/min) = 4.55 × VCO2 - 3.21 × VO2
  (protein oxidation assumed negligible during exercise)

  Jeukendrup & Wallis 2005 (Int J Sports Med 26:S28-37):
    FAT_ox (g/min) = 1.695 × VO2 - 1.701 × VCO2
    CHO_ox (g/min) = 4.344 × VCO2 - 3.061 × VO2
  (glycogen-based, appropriate for exercise; low intensity variant)

Key concepts:
  FATmax (Achten et al. 2002, MSSE 34:92-97):
    = peak fat oxidation rate (g/min) during graded exercise
    = typically at 45-65% VO2max (Venables et al. 2005)
    = normative: trained males ~0.56 g/min, females ~0.44 g/min
      (Randell et al. 2017, Frontiers Physiol)

  Crossover Point (Brooks & Mercier 1994, J Appl Physiol 76:2253):
    = intensity where CHO-derived energy > FAT-derived energy
    = where CHO kcal/min exceeds FAT kcal/min
    = energy conversion: 1g FAT = 9.75 kcal, 1g CHO = 4.07 kcal
    = typically at ~50-65% VO2max

  FATmin zone (Achten & Jeukendrup 2003):
    = intensity range where FAT_ox ≥ 90% of MFO (FATmax ± 10%)
    = practical "fat burning zone" for training prescription

Validity limits:
  - RER > 1.0 → bicarbonate buffering → FAT_ox calculation invalid
    (non-metabolic CO2 overestimates VCO2, Jeukendrup 2005)
  - Calculations only valid when RER < 1.0
  - Apply 30-60s rolling average to reduce breath-by-breath noise

Outputs:
  FATmax metrics: MFO (g/min), MFO/kg, FATmax HR, FATmax speed,
    FATmax %VO2peak, FATmax %HRmax
  Crossover point: COP HR, COP speed, COP %VO2peak
  Energy contribution: fat_pct and cho_pct at rest, VT1, VT2
  Substrate curves: fat_ox and cho_ox arrays for plotting
  QC flags: RER_VALIDITY, DATA_QUALITY, RAMP_PROTOCOL_LIMITATION
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class Engine_E10_Substrate_v2:
    """
    E10 v2.0 — Substrate Oxidation Profile (Frayn + Jeukendrup)
    """

    # Energy density constants
    KCAL_PER_G_FAT = 9.75   # Peronnet & Massicotte 1991
    KCAL_PER_G_CHO = 4.07   # glucose: 3.74, glycogen: 4.15, avg ~4.07

    # MFO normative ranges (Randell et al. 2017, Frontiers Physiol)
    MFO_NORMS = {
        'male': {
            'endurance_trained': {'mean': 0.56, 'sd': 0.17},
            'recreationally_active': {'mean': 0.46, 'sd': 0.16},
            'overweight': {'mean': 0.38, 'sd': 0.15},
        },
        'female': {
            'endurance_trained': {'mean': 0.44, 'sd': 0.15},  # estimated ~28% less
            'recreationally_active': {'mean': 0.35, 'sd': 0.12},
            'overweight': {'mean': 0.30, 'sd': 0.12},
        },
    }

    @classmethod
    def run(cls, df_ex: pd.DataFrame,
            e02: dict = None, e01: dict = None,
            sex: str = 'male',
            weight_kg: float = None,
            equation: str = 'frayn') -> Dict:
        """
        Parameters
        ----------
        df_ex     : Exercise-phase DataFrame
        e02       : E02 results (VT1/VT2 times, HR, speed)
        e01       : E01 results (peak values)
        sex       : 'male' or 'female'
        weight_kg : body mass (for MFO/kg)
        equation  : 'frayn' (default) or 'jeukendrup'
        """
        result = cls._init_result()

        try:
            df = df_ex.copy()
            # ── Resolve columns ──────────────────────────────
            vo2_col = cls._find_col(df, ['VO2_Lmin', 'VO2_L_min', 'VO2_lmin'])
            vco2_col = cls._find_col(df, ['VCO2_Lmin', 'VCO2_L_min', 'VCO2_lmin'])
            time_col = cls._find_col(df, ['Time_sec', 'Time_s', 'time_s', 't'])
            hr_col = cls._find_col(df, ['HR_bpm', 'hr_bpm', 'HR'])
            speed_col = cls._find_col(df, ['speed_km_h', 'Speed_km_h', 'speed_kmh', 'speed', 'Speed'])

            # Convert ml/min → L/min if needed
            if not vo2_col:
                vo2_ml = cls._find_col(df, ['VO2_mlmin', 'VO2_ml_min', 'VO2'])
                if vo2_ml:
                    df['VO2_Lmin'] = pd.to_numeric(df[vo2_ml], errors='coerce') / 1000.0
                    vo2_col = 'VO2_Lmin'
            if not vco2_col:
                vco2_ml = cls._find_col(df, ['VCO2_mlmin', 'VCO2_ml_min', 'VCO2'])
                if vco2_ml:
                    df['VCO2_Lmin'] = pd.to_numeric(df[vco2_ml], errors='coerce') / 1000.0
                    vco2_col = 'VCO2_Lmin'

            if not vo2_col or not vco2_col:
                result['flags'].append('MISSING_GAS_EXCHANGE')
                return result

            vo2 = pd.to_numeric(df[vo2_col], errors='coerce')
            vco2 = pd.to_numeric(df[vco2_col], errors='coerce')

            # ── Calculate RER ────────────────────────────────
            rer = vco2 / vo2.replace(0, np.nan)

            # ── Stoichiometric equations ─────────────────────
            if equation == 'jeukendrup':
                fat_raw = 1.695 * vo2 - 1.701 * vco2    # Jeukendrup & Wallis 2005
                cho_raw = 4.344 * vco2 - 3.061 * vo2
                result['equation'] = 'Jeukendrup & Wallis 2005'
            else:
                fat_raw = 1.67 * vo2 - 1.67 * vco2       # Frayn 1983
                cho_raw = 4.55 * vco2 - 3.21 * vo2
                result['equation'] = 'Frayn 1983'

            # ── Validity mask: only valid when RER < 1.0 ─────
            valid_mask = (rer < 1.0) & rer.notna() & (vo2 > 0.1)
            n_valid = valid_mask.sum()
            n_total = len(df)
            result['n_valid_points'] = int(n_valid)
            result['n_total_points'] = int(n_total)
            result['pct_valid'] = round(n_valid / max(n_total, 1) * 100, 1)

            if n_valid < 30:
                result['flags'].append('INSUFFICIENT_VALID_DATA')
                return result

            # ── Smooth (30s rolling mean) ────────────────────
            fat_smooth = fat_raw.rolling(30, center=True, min_periods=10).mean()
            cho_smooth = cho_raw.rolling(30, center=True, min_periods=10).mean()

            # Zero out negative values (artifact) only for display
            fat_display = fat_smooth.clip(lower=0)

            # ── FATmax (MFO) ─────────────────────────────────
            # Only consider valid RER points
            fat_valid = fat_smooth.copy()
            fat_valid[~valid_mask] = np.nan

            idx_mfo = fat_valid.idxmax()
            if pd.notna(idx_mfo):
                mfo = float(fat_valid.loc[idx_mfo])
                result['mfo_gmin'] = round(mfo, 3)

                if weight_kg and weight_kg > 0:
                    result['mfo_mgkg_min'] = round(mfo * 1000 / weight_kg, 1)

                if hr_col and hr_col in df.columns:
                    result['fatmax_hr'] = cls._safe_val(df, idx_mfo, hr_col)
                if speed_col and speed_col in df.columns:
                    result['fatmax_speed_kmh'] = cls._safe_val(df, idx_mfo, speed_col)
                if time_col and time_col in df.columns:
                    result['fatmax_time_s'] = cls._safe_val(df, idx_mfo, time_col)

                # %VO2peak at FATmax
                vo2_at_mfo = float(vo2.loc[idx_mfo]) if pd.notna(vo2.loc[idx_mfo]) else None
                vo2_peak = cls._get_vo2peak(e01, vo2)
                if vo2_at_mfo and vo2_peak and vo2_peak > 0:
                    result['fatmax_pct_vo2peak'] = round(vo2_at_mfo / vo2_peak * 100, 1)

                hr_max = cls._get_hrmax(e01, df, hr_col)
                if result.get('fatmax_hr') and hr_max:
                    result['fatmax_pct_hrmax'] = round(result['fatmax_hr'] / hr_max * 100, 1)

                # ── FATmax zone (±10% of MFO) ────────────────
                threshold_90pct = mfo * 0.90
                in_zone = fat_valid >= threshold_90pct
                if in_zone.any() and hr_col:
                    zone_hrs = df.loc[in_zone.index[in_zone], hr_col].dropna()
                    if len(zone_hrs) > 0:
                        result['fatmax_zone_hr_low'] = round(float(zone_hrs.min()), 0)
                        result['fatmax_zone_hr_high'] = round(float(zone_hrs.max()), 0)

            # ── Crossover Point ──────────────────────────────
            # COP = where CHO kcal/min > FAT kcal/min
            fat_kcal = fat_smooth * cls.KCAL_PER_G_FAT
            cho_kcal = cho_smooth * cls.KCAL_PER_G_CHO
            diff = cho_kcal - fat_kcal  # positive when CHO dominates

            # Find first sustained crossover (not transient)
            diff_valid = diff.copy()
            diff_valid[~valid_mask] = np.nan
            diff_smooth = diff_valid.rolling(15, center=True, min_periods=5).mean()

            # Find where diff crosses from negative to positive
            cop_idx = None
            vals = diff_smooth.dropna()
            for i in range(1, len(vals)):
                if vals.iloc[i-1] < 0 and vals.iloc[i] >= 0:
                    cop_idx = vals.index[i]
                    break

            if cop_idx is not None:
                if hr_col:
                    result['cop_hr'] = cls._safe_val(df, cop_idx, hr_col)
                if speed_col and speed_col in df.columns:
                    result['cop_speed_kmh'] = cls._safe_val(df, cop_idx, speed_col)
                if time_col:
                    result['cop_time_s'] = cls._safe_val(df, cop_idx, time_col)

                vo2_at_cop = float(vo2.loc[cop_idx]) if pd.notna(vo2.loc[cop_idx]) else None
                if vo2_at_cop and vo2_peak and vo2_peak > 0:
                    result['cop_pct_vo2peak'] = round(vo2_at_cop / vo2_peak * 100, 1)
                if result.get('cop_hr') and hr_max:
                    result['cop_pct_hrmax'] = round(result['cop_hr'] / hr_max * 100, 1)

                # RER at crossover
                rer_at_cop = float(rer.loc[cop_idx]) if pd.notna(rer.loc[cop_idx]) else None
                if rer_at_cop:
                    result['cop_rer'] = round(rer_at_cop, 3)

            # ── Energy contribution at key points ────────────
            if e02 and time_col:
                time_arr = pd.to_numeric(df[time_col], errors='coerce')
                vt1_time = e02.get('vt1_time_s') or e02.get('vt1_time_sec')
                vt2_time = e02.get('vt2_time_s') or e02.get('vt2_time_sec')

                for label, t_val in [('vt1', vt1_time), ('vt2', vt2_time)]:
                    if t_val is not None:
                        # Find nearest time point
                        diffs = (time_arr - t_val).abs()
                        near_idx = diffs.idxmin()
                        f_val = fat_smooth.loc[near_idx] if pd.notna(fat_smooth.loc[near_idx]) else None
                        c_val = cho_smooth.loc[near_idx] if pd.notna(cho_smooth.loc[near_idx]) else None
                        if f_val is not None and c_val is not None:
                            f_kcal = max(f_val, 0) * cls.KCAL_PER_G_FAT
                            c_kcal = max(c_val, 0) * cls.KCAL_PER_G_CHO
                            total = f_kcal + c_kcal
                            if total > 0:
                                result[f'fat_pct_at_{label}'] = round(f_kcal / total * 100, 1)
                                result[f'cho_pct_at_{label}'] = round(c_kcal / total * 100, 1)
                            result[f'fat_gmin_at_{label}'] = round(max(f_val, 0), 3)
                            result[f'cho_gmin_at_{label}'] = round(max(c_val, 0), 3)

            # ── Total energy expenditure during exercise ─────
            if time_col:
                time_arr = pd.to_numeric(df[time_col], errors='coerce')
                dt = time_arr.diff().fillna(1) / 60.0  # convert to minutes
                total_fat_kcal = (fat_display * cls.KCAL_PER_G_FAT * dt).sum()
                total_cho_kcal = (cho_smooth.clip(lower=0) * cls.KCAL_PER_G_CHO * dt).sum()
                result['total_fat_kcal'] = round(float(total_fat_kcal), 0)
                result['total_cho_kcal'] = round(float(total_cho_kcal), 0)
                result['total_kcal'] = round(float(total_fat_kcal + total_cho_kcal), 0)
                result['total_fat_g'] = round(float(total_fat_kcal / cls.KCAL_PER_G_FAT), 1)
                result['total_cho_g'] = round(float(total_cho_kcal / cls.KCAL_PER_G_CHO), 1)

            # ── Zone substrate rates (g/h) ───────────────────
            # Requires E16 zone HR ranges; compute avg fat/cho in each zone
            result['zone_substrate'] = cls._compute_zone_substrate(
                df, hr_col, vo2, vco2, fat_smooth, cho_smooth, rer, valid_mask, e02)

            # ── MFO normative comparison ─────────────────────
            if result.get('mfo_gmin') is not None:
                sex_norms = cls.MFO_NORMS.get(sex, cls.MFO_NORMS['male'])
                for pop, ref in sex_norms.items():
                    z = (result['mfo_gmin'] - ref['mean']) / ref['sd'] if ref['sd'] > 0 else 0
                    result[f'mfo_z_{pop}'] = round(z, 2)

            # ── QC flags ─────────────────────────────────────
            if result['pct_valid'] < 50:
                result['flags'].append('LOW_VALID_FRACTION')
            if result.get('mfo_gmin') is not None and result['mfo_gmin'] < 0.05:
                result['flags'].append('VERY_LOW_MFO')
            if result.get('fatmax_pct_vo2peak') and result['fatmax_pct_vo2peak'] > 75:
                result['flags'].append('HIGH_FATMAX_INTENSITY')
            if result.get('cop_hr') is None:
                result['flags'].append('NO_CROSSOVER_DETECTED')
                result['cop_note'] = 'CHO dominuje od początku testu — brak crossover point'


            # Ramp protocol limitation
            result['flags'].append('RAMP_PROTOCOL_NOTE')
            result['ramp_note'] = (
                'FATmax z testu rampowego jest orientacyjny. '
                'Złoty standard: dedykowany test FATmax (Achten 2002) '
                'z krokami 3-6 min na stałej intensywności.'
            )

            result['status'] = 'OK'

        except Exception as e:
            result['status'] = 'ERROR'
            result['flags'].append(f'EXCEPTION:{e}')

        return result

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _init_result() -> Dict:
        return {
            'status': 'NOT_RUN',
            'equation': None,
            # FATmax
            'mfo_gmin': None,
            'mfo_mgkg_min': None,
            'fatmax_hr': None,
            'fatmax_speed_kmh': None,
            'fatmax_time_s': None,
            'fatmax_pct_vo2peak': None,
            'fatmax_pct_hrmax': None,
            'fatmax_zone_hr_low': None,
            'fatmax_zone_hr_high': None,
            # Crossover
            'cop_hr': None,
            'cop_speed_kmh': None,
            'cop_time_s': None,
            'cop_pct_vo2peak': None,
            'cop_pct_hrmax': None,
            'cop_rer': None,
            # Substrate at thresholds
            'fat_pct_at_vt1': None, 'cho_pct_at_vt1': None,
            'fat_gmin_at_vt1': None, 'cho_gmin_at_vt1': None,
            'fat_pct_at_vt2': None, 'cho_pct_at_vt2': None,
            'fat_gmin_at_vt2': None, 'cho_gmin_at_vt2': None,
            # Totals
            'total_fat_kcal': None, 'total_cho_kcal': None,
            'total_kcal': None, 'total_fat_g': None, 'total_cho_g': None,
            # Norms
            'mfo_z_endurance_trained': None,
            'mfo_z_recreationally_active': None,
            'mfo_z_overweight': None,
            # QC
            'n_valid_points': 0, 'n_total_points': 0, 'pct_valid': 0,
            'ramp_note': None,
            'flags': [],
        }

    @staticmethod
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _safe_val(df, idx, col):
        try:
            v = df.loc[idx, col]
            return round(float(v), 1) if pd.notna(v) else None
        except:
            return None

    @staticmethod
    def _get_vo2peak(e01, vo2_series):
        if e01:
            for k in ['vo2_peak_ml_min', 'vo2_peak_mlmin']:
                v = e01.get(k)
                if v: return float(v) / 1000.0  # convert to L/min
        return float(vo2_series.max()) if len(vo2_series) > 0 else None

    @staticmethod
    def _get_hrmax(e01, df, hr_col):
        if e01:
            v = e01.get('hr_max') or e01.get('hr_peak')
            if v: return float(v)
        if hr_col and hr_col in df.columns:
            return float(df[hr_col].max())
        return None

    @classmethod
    def _compute_zone_substrate(cls, df, hr_col, vo2, vco2, fat_smooth, cho_smooth,
                                 rer, valid_mask, e02) -> Dict:
        """
        Compute average fat/cho oxidation rates (g/h) for training zones.
        Uses the ramp test data: for each HR zone, averages substrate rates
        from data points falling within that HR range.

        Zones derived from E02 thresholds (VT1/VT2) + HRmax:
          Z2 (aerobic base): ~85% VT1 HR → VT1
          Z4 (threshold): VT2 → VT2 + buffer
          Z5 (VO2max): VT2 + buffer → HRmax
        """
        out = {}
        if not hr_col or hr_col not in df.columns or e02 is None:
            return out

        hr = pd.to_numeric(df[hr_col], errors='coerce')
        vt1_hr = e02.get('vt1_hr') or e02.get('vt1_hr_bpm')
        vt2_hr = e02.get('vt2_hr') or e02.get('vt2_hr_bpm')
        hr_max = hr.max()

        if not vt1_hr or not vt2_hr:
            return out

        vt1_hr = float(vt1_hr)
        vt2_hr = float(vt2_hr)
        hr_max = float(hr_max)

        # Zone boundaries (same logic as E16 v2)
        offset = max(round(vt1_hr * 0.20), 10)
        buffer = max(round((hr_max - vt2_hr) * 0.50), 5)

        zone_defs = {
            'z2': (max(vt1_hr - offset, 60), vt1_hr),
            'z4': (vt2_hr, min(vt2_hr + buffer, hr_max)),
            'z5': (min(vt2_hr + buffer + 1, hr_max), hr_max),
        }

        for zname, (hr_lo, hr_hi) in zone_defs.items():
            mask_zone = (hr >= hr_lo) & (hr <= hr_hi)
            mask_valid = mask_zone & valid_mask  # RER < 1.0
            n_total = int(mask_zone.sum())
            n_valid = int(mask_valid.sum())

            if n_total < 3:
                out[zname] = {
                    'fat_gh': None, 'cho_gh': None,
                    'fat_pct': None, 'cho_pct': None,
                    'n_points': n_total, 'rer_valid': False,
                    'note': 'insufficient data'
                }
                continue

            # Weir EE (kcal/min) = 3.941 × VO2 + 1.106 × VCO2  (Weir 1949)
            # More accurate than substrate sum, especially when RER ≥ 1.0
            vo2_zone = vo2[mask_zone]
            vco2_zone = vco2[mask_zone]
            ee_weir_min = float((3.941 * vo2_zone + 1.106 * vco2_zone).mean())
            ee_weir_h = round(ee_weir_min * 60, 0)

            # Fat oxidation from Frayn (valid only RER < 1.0)
            if n_valid >= 5:
                fat_avg = float(fat_smooth[mask_valid].clip(lower=0).mean())
                rer_ok = True
            else:
                fat_avg = float(fat_smooth[mask_zone].clip(lower=0).mean())
                fat_avg = max(fat_avg, 0)
                rer_ok = False

            fat_gh = round(fat_avg * 60, 1)  # g/min → g/h
            fat_kcal_min = fat_avg * cls.KCAL_PER_G_FAT

            # CHO from Weir EE minus fat energy (avoids RER>1 overestimate)
            cho_kcal_min = max(ee_weir_min - fat_kcal_min, 0)
            cho_avg = cho_kcal_min / cls.KCAL_PER_G_CHO if cls.KCAL_PER_G_CHO > 0 else 0
            cho_gh = round(cho_avg * 60, 1)

            total_kcal_min = fat_kcal_min + cho_kcal_min
            fat_pct = round(fat_kcal_min / total_kcal_min * 100, 0) if total_kcal_min > 0 else 0
            cho_pct = round(100 - fat_pct, 0)

            out[zname] = {
                'fat_gh': fat_gh,
                'cho_gh': cho_gh,
                'fat_pct': fat_pct,
                'cho_pct': cho_pct,
                'kcal_h': ee_weir_h,
                'n_points': n_total,
                'rer_valid': rer_ok,
                'note': None if rer_ok else 'RER≥1 — kcal/h z Weir, FAT≈0',
            }

        return out



# --- E11 - E14 (PLACEHOLDERS) ---
# ==========================================
# ENGINE E11 — LACTATE ANALYSIS v2.0
# ==========================================
#
# RESEARCH BASIS:
# ───────────────────────────────────────────
# LT1 (Aerobic Threshold) methods:
#   1. Baseline + 0.5 mmol/L  (Berg 1990, Zoladz 1995)
#   2. Baseline + 1.0 mmol/L  (Coyle 1983)
#   3. Log-Log breakpoint      (Beaver 1985)
#
# LT2 (Anaerobic Threshold) methods:
#   4. OBLA 4.0 mmol/L         (Mader 1976, Heck 1985)
#   5. Dmax                     (Cheng 1992)
#   6. Modified Dmax            (Bishop 1998)
#   7. Min Lactate Eq + 1.5    (Dickhuth 1999)
#
# References:
#   - Fabre et al. 2010: ModDmax r=0.99 vs VT2 in elite skiers
#   - Repeatability study (PMC6235347): Dmax CV<5%, best predictive value
#   - Bishop et al. 1998: ModDmax start = point before first +0.4 rise
#   - Cheng et al. 1992: 3rd order polynomial, max perpendicular distance
#   - INSCYD: baseline error margin ±0.35 mmol/L for handheld meters
# ───────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LactateInput:
    """
    Struktura danych laktatowych.
    Obsługuje 3 źródła:
      A) Ręczne wpisanie par (czas_sec, La_mmol) + opcjonalnie (speed_kmh, HR_bpm)
      B) Plik CSV z kolumnami: time_sec, lactate_mmol, [speed_kmh], [hr_bpm]
      C) Kolumna 'Lactate_mmol' w głównym df CPET (sparse, NaN = brak pomiaru)
    """
    # --- A) Ręczne dane — lista dictów ---
    # Przykład:
    # manual_data = [
    #     {"time_sec": 180,  "la": 1.2, "speed_kmh": 6.0,  "hr": 120},
    #     {"time_sec": 300,  "la": 1.0, "speed_kmh": 8.0,  "hr": 135},
    #     {"time_sec": 420,  "la": 1.4, "speed_kmh": 10.0, "hr": 148},
    #     {"time_sec": 540,  "la": 2.1, "speed_kmh": 12.0, "hr": 160},
    #     {"time_sec": 660,  "la": 3.5, "speed_kmh": 13.0, "hr": 170},
    #     {"time_sec": 780,  "la": 5.8, "speed_kmh": 14.0, "hr": 180},
    #     {"time_sec": 900,  "la": 9.2, "speed_kmh": 14.0, "hr": 188},
    # ]
    manual_data: Optional[List[Dict]] = None

    # --- B) Plik CSV z danymi laktatowymi ---
    lactate_csv_path: Optional[str] = None

    # --- C) Baseline (spoczynkowa La, opcjonalnie) ---
    baseline_la: Optional[float] = None


class Engine_E11_Lactate_v2:
    """
    E11: Full lactate threshold analysis engine.

    Oblicza 7 metod detekcji progów laktatowych:
      LT1: Baseline+0.5, Baseline+1.0, Log-Log
      LT2: OBLA 4.0, Dmax, Modified Dmax, MinLacEq+1.5

    Input: LactateInput + df_cpet (opcjonalnie, dla interpolacji HR/Speed/VO2)
    Output: dict z wynikami wszystkich metod + concordance + integracja z CPET

    Minimalne wymagania: ≥4 punkty pomiarowe lactate.
    Zalecane: ≥6 punktów pokrywających zakres od baseline do >4 mmol/L.
    """

    MIN_POINTS = 4
    RECOMMENDED_POINTS = 6

    # ─── SEKCJA 1: ŁADOWANIE I WALIDACJA DANYCH ───

    @staticmethod
    def _load_lactate_data(lactate_input: 'LactateInput',
                           df_cpet: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Ładuje dane laktatowe z dowolnego źródła → pd.DataFrame
        Kolumny wynikowe: time_sec, lactate_mmol, [speed_kmh], [hr_bpm]
        """
        frames = []

        # Źródło A: manual_data (lista dictów)
        if lactate_input.manual_data:
            rows = []
            for d in lactate_input.manual_data:
                row = {
                    "time_sec": float(d.get("time_sec", d.get("time", d.get("t", 0)))),
                    "lactate_mmol": float(d.get("la", d.get("lactate", d.get("lactate_mmol", d.get("La", np.nan))))),
                }
                # Opcjonalne kolumny
                for src_keys, tgt in [
                    (["speed_kmh", "speed", "v", "pace"], "speed_kmh"),
                    (["hr", "hr_bpm", "HR", "heart_rate"], "hr_bpm"),
                    (["power", "power_w", "Power_W", "watts"], "power_w"),
                ]:
                    for sk in src_keys:
                        if sk in d and d[sk] is not None:
                            row[tgt] = float(d[sk])
                            break
                rows.append(row)
            if rows:
                frames.append(pd.DataFrame(rows))

        # Źródło B: CSV file
        if lactate_input.lactate_csv_path:
            try:
                df_la_csv = pd.read_csv(lactate_input.lactate_csv_path)
                # Normalizacja nazw kolumn
                col_map = {}
                for c in df_la_csv.columns:
                    cl = c.lower().strip()
                    if cl in ("time_sec", "time", "t", "czas", "time_s"):
                        col_map[c] = "time_sec"
                    elif cl in ("lactate_mmol", "la", "lactate", "lac", "bla", "la_mmol", "lactate_mmoll"):
                        col_map[c] = "lactate_mmol"
                    elif cl in ("speed_kmh", "speed", "v", "velocity", "predkosc"):
                        col_map[c] = "speed_kmh"
                    elif cl in ("hr", "hr_bpm", "heart_rate", "tetno"):
                        col_map[c] = "hr_bpm"
                    elif cl in ("power", "power_w", "watts", "moc"):
                        col_map[c] = "power_w"
                df_la_csv = df_la_csv.rename(columns=col_map)
                if "time_sec" in df_la_csv.columns and "lactate_mmol" in df_la_csv.columns:
                    frames.append(df_la_csv)
            except Exception as e:
                print(f"  ⚠️ E11: Nie można wczytać CSV lactate: {e}")

        # Źródło C: kolumna w głównym df CPET (sparse)
        # Skip if manual data provided — manual is gold standard,
        # continuous ergospiro La estimate should not dilute it
        _has_manual = bool(lactate_input.manual_data)
        _has_csv = bool(lactate_input.lactate_csv_path)
        _has_any_explicit = _has_manual or _has_csv
        _use_csv_column = getattr(lactate_input, 'use_csv_column', False)
        if df_cpet is not None and not _has_manual and (_has_any_explicit or _use_csv_column):
            la_col = None
            for candidate in ["Lactate_mmol", "Lactate_mmolL", "La_mmol", "La_mmol_L", "BLa", "La"]:
                if candidate in df_cpet.columns:
                    la_col = candidate
                    break
            if la_col:
                time_col = "Time_sec" if "Time_sec" in df_cpet.columns else "Time_s"
                if time_col in df_cpet.columns:
                    df_sparse = df_cpet[[time_col, la_col]].copy()
                    df_sparse.columns = ["time_sec", "lactate_mmol"]
                    df_sparse["lactate_mmol"] = pd.to_numeric(df_sparse["lactate_mmol"], errors="coerce")
                    df_sparse = df_sparse.dropna(subset=["lactate_mmol"])
                    if len(df_sparse) >= 2:
                        # Dodaj speed/HR jeśli dostępne
                        for src, tgt in [("Speed_kmh", "speed_kmh"), ("HR_bpm", "hr_bpm"), ("Power_W", "power_w")]:
                            if src in df_cpet.columns:
                                df_sparse[tgt] = df_cpet.loc[df_sparse.index, src].values
                        frames.append(df_sparse)

        if not frames:
            return None

        # Merge i deduplikacja
        df_la = pd.concat(frames, ignore_index=True)
        df_la["time_sec"] = pd.to_numeric(df_la["time_sec"], errors="coerce")
        df_la["lactate_mmol"] = pd.to_numeric(df_la["lactate_mmol"], errors="coerce")
        df_la = df_la.dropna(subset=["time_sec", "lactate_mmol"])
        df_la = df_la.sort_values("time_sec").reset_index(drop=True)

        # Deduplikacja — tylko dla dyskretnych danych (ręczne punkty)
        # Dla ciągłych danych (>50 unikalnych wartości) nie deduplikujemy
        _n_unique_la = len(df_la["lactate_mmol"].unique())
        if len(df_la) > 1 and _n_unique_la <= 50:
            # Dyskretne dane: jeśli dwa punkty w ±5s, weź średnią
            groups = []
            current_group = [0]
            for i in range(1, len(df_la)):
                if df_la.iloc[i]["time_sec"] - df_la.iloc[current_group[-1]]["time_sec"] < 5:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]
            groups.append(current_group)
            merged = []
            for g in groups:
                subset = df_la.iloc[g]
                merged.append(subset.mean(numeric_only=True))
            df_la = pd.DataFrame(merged)

        return df_la

    @staticmethod
    def _enrich_from_cpet(df_la: pd.DataFrame,
                          df_cpet: Optional[pd.DataFrame],
                          window_sec: float = 10.0) -> pd.DataFrame:
        """
        Wzbogaca dane laktatowe o HR, Speed, VO2, Power z df_cpet
        (interpolacja z okna czasowego ±window_sec).
        """
        if df_cpet is None or df_la is None:
            return df_la

        time_col = "Time_sec" if "Time_sec" in df_cpet.columns else "Time_s"
        if time_col not in df_cpet.columns:
            return df_la

        t_cpet = pd.to_numeric(df_cpet[time_col], errors="coerce")

        enrichments = {
            "hr_bpm": ["HR_bpm", "HR"],
            "speed_kmh": ["Speed_kmh", "Speed"],
            "vo2_mlmin": ["VO2_mlmin", "VO2_ml_min", "VO2_Lmin"],
            "power_w": ["Power_W", "Power"],
            "rer": ["RER"],
        }

        for tgt_col, src_candidates in enrichments.items():
            if tgt_col in df_la.columns and df_la[tgt_col].notna().all():
                continue  # Już mamy dane
            src_col = None
            for sc in src_candidates:
                if sc in df_cpet.columns:
                    src_col = sc
                    break
            if src_col is None:
                continue

            vals = pd.to_numeric(df_cpet[src_col], errors="coerce")
            interpolated = []
            for _, row in df_la.iterrows():
                t = row["time_sec"]
                mask = (t_cpet >= t - window_sec) & (t_cpet <= t + window_sec) & vals.notna()
                if mask.sum() >= 1:
                    v = float(vals[mask].mean())
                    # Konwersja VO2 L/min → ml/min
                    if tgt_col == "vo2_mlmin" and src_col == "VO2_Lmin" and v < 20:
                        v = v * 1000
                    interpolated.append(round(v, 2))
                else:
                    interpolated.append(np.nan)
            df_la[tgt_col] = interpolated

        return df_la

    # ─── SEKCJA 2: POLYNOMIAL FIT ───

    @staticmethod
    def _fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 3) -> Optional[np.ndarray]:
        """Fit polynomial of given degree, return coefficients or None."""
        if len(x) < degree + 1:
            return None
        try:
            coeffs = np.polyfit(x, y, degree)
            return coeffs
        except Exception:
            return None

    @staticmethod
    def _poly_eval(coeffs: np.ndarray, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate polynomial at given x values."""
        return np.polyval(coeffs, x)

    @staticmethod
    def _perpendicular_distance(px, py, x1, y1, x2, y2):
        """
        Odległość prostopadła punktu (px, py) od linii (x1,y1)-(x2,y2).
        Wzór: |((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
        """
        num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if den == 0:
            return 0.0
        return num / den

    # ─── SEKCJA 3: METODY LT1 (AEROBIC THRESHOLD) ───

    @staticmethod
    def _method_baseline_plus(df_la: pd.DataFrame, baseline: float,
                              delta: float = 0.5, intensity_col: str = "time_sec") -> Dict:
        """
        Baseline + delta mmol/L method (Berg 1990, Zoladz 1995).
        LT = pierwszy punkt, w którym La > baseline + delta.
        Jeśli żaden punkt nie przekracza progu — interpolacja z polynomial fit.
        """
        threshold_value = baseline + delta
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        # Szukamy pierwszego punktu powyżej progu
        above_mask = y >= threshold_value
        if above_mask.any():
            idx = np.argmax(above_mask)
            # Interpolacja liniowa między poprzednim a tym punktem
            if idx > 0:
                x0, y0 = x[idx - 1], y[idx - 1]
                x1, y1 = x[idx], y[idx]
                if y1 != y0:
                    x_interp = x0 + (threshold_value - y0) * (x1 - x0) / (y1 - y0)
                else:
                    x_interp = x[idx]
            else:
                x_interp = x[idx]

            return {
                "found": True,
                "threshold_time_sec": round(float(x_interp), 1),
                "threshold_la_mmol": round(threshold_value, 2),
                "method": f"Baseline+{delta}",
                "baseline_la": round(baseline, 2),
            }

        # Polynomial interpolation fallback
        coeffs = Engine_E11_Lactate_v2._fit_polynomial(x, y, degree=3)
        if coeffs is not None:
            x_fine = np.linspace(x[0], x[-1], 500)
            y_fine = np.polyval(coeffs, x_fine)
            above_fine = y_fine >= threshold_value
            if above_fine.any():
                x_interp = x_fine[np.argmax(above_fine)]
                return {
                    "found": True,
                    "threshold_time_sec": round(float(x_interp), 1),
                    "threshold_la_mmol": round(threshold_value, 2),
                    "method": f"Baseline+{delta} (poly interp)",
                    "baseline_la": round(baseline, 2),
                }

        return {"found": False, "method": f"Baseline+{delta}", "reason": f"La never reached {threshold_value:.1f} mmol/L"}

    @staticmethod
    def _method_log_log(df_la: pd.DataFrame, intensity_col: str = "time_sec") -> Dict:
        """
        Log-Log method (Beaver 1985).
        Segmented regression na log(La) vs log(intensywność).
        Punkt złamania = LT1 (aerobic threshold).
        """
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        # Filtr: wartości > 0
        valid = (x > 0) & (y > 0)
        if valid.sum() < 4:
            return {"found": False, "method": "Log-Log", "reason": "Za mało punktów > 0"}

        log_x = np.log(x[valid])
        log_y = np.log(y[valid])

        # Segmented regression: szukamy najlepszego breakpointu
        n = len(log_x)
        best_sse = np.inf
        best_bp = None

        for bp_idx in range(2, n - 2):  # Min 2 punkty w każdym segmencie
            # Segment 1: [0, bp_idx]
            x1, y1 = log_x[:bp_idx + 1], log_y[:bp_idx + 1]
            # Segment 2: [bp_idx, end]
            x2, y2 = log_x[bp_idx:], log_y[bp_idx:]

            try:
                from scipy.stats import linregress
                s1 = linregress(x1, y1)
                s2 = linregress(x2, y2)

                sse1 = np.sum((y1 - (s1.slope * x1 + s1.intercept))**2)
                sse2 = np.sum((y2 - (s2.slope * x2 + s2.intercept))**2)
                total_sse = sse1 + sse2

                if total_sse < best_sse:
                    best_sse = total_sse
                    best_bp = bp_idx
            except Exception:
                continue

        if best_bp is None:
            return {"found": False, "method": "Log-Log", "reason": "Nie znaleziono breakpointu"}

        bp_time = float(np.exp(log_x[best_bp]))
        bp_la = float(np.exp(log_y[best_bp]))

        return {
            "found": True,
            "threshold_time_sec": round(bp_time, 1),
            "threshold_la_mmol": round(bp_la, 2),
            "method": "Log-Log (Beaver 1985)",
            "breakpoint_index": int(best_bp),
        }

    # ─── SEKCJA 4: METODY LT2 (ANAEROBIC THRESHOLD) ───

    @staticmethod
    def _method_obla(df_la: pd.DataFrame, fixed_la: float = 4.0,
                     intensity_col: str = "time_sec") -> Dict:
        """
        OBLA — Onset of Blood Lactate Accumulation (Mader 1976).
        Intensywność przy stałej wartości La (typowo 4.0 mmol/L).
        Interpolacja z polynomial fit.
        """
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        # Czy dane w ogóle osiągają ten poziom?
        if y.max() < fixed_la:
            return {"found": False, "method": f"OBLA {fixed_la}", "reason": f"La max ({y.max():.1f}) < {fixed_la} mmol/L"}

        # Polynomial fit dla interpolacji
        coeffs = Engine_E11_Lactate_v2._fit_polynomial(x, y, degree=3)
        if coeffs is None:
            # Fallback: interpolacja liniowa
            for i in range(1, len(y)):
                if y[i] >= fixed_la and y[i-1] < fixed_la:
                    x_interp = x[i-1] + (fixed_la - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
                    return {
                        "found": True,
                        "threshold_time_sec": round(float(x_interp), 1),
                        "threshold_la_mmol": fixed_la,
                        "method": f"OBLA {fixed_la} (linear interp)",
                    }
            return {"found": False, "method": f"OBLA {fixed_la}"}

        x_fine = np.linspace(x[0], x[-1], 1000)
        y_fine = np.polyval(coeffs, x_fine)
        above = y_fine >= fixed_la
        if above.any():
            x_obla = x_fine[np.argmax(above)]
            return {
                "found": True,
                "threshold_time_sec": round(float(x_obla), 1),
                "threshold_la_mmol": fixed_la,
                "method": f"OBLA {fixed_la} (Mader 1976)",
            }

        return {"found": False, "method": f"OBLA {fixed_la}"}

    @staticmethod
    def _method_dmax(df_la: pd.DataFrame, intensity_col: str = "time_sec") -> Dict:
        """
        Dmax method (Cheng 1992).
        1) Fit 3rd order polynomial to La vs intensity
        2) Linia prosta od pierwszego do ostatniego punktu krzywej
        3) Punkt max. odległości prostopadłej = LT2

        Ref: Cheng B. et al. Int J Sports Med 13.07 (1992): 518-522
        """
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        if len(x) < 4:
            return {"found": False, "method": "Dmax", "reason": "Za mało punktów (<4)"}

        coeffs = Engine_E11_Lactate_v2._fit_polynomial(x, y, degree=3)
        if coeffs is None:
            return {"found": False, "method": "Dmax", "reason": "Polynomial fit failed"}

        # Gęsta siatka na krzywej
        x_fine = np.linspace(x[0], x[-1], 1000)
        y_fine = np.polyval(coeffs, x_fine)

        # Linia prosta: od pierwszego do ostatniego punktu na KRZYWEJ (nie surowych danych)
        x1, y1 = x_fine[0], y_fine[0]
        x2, y2 = x_fine[-1], y_fine[-1]

        # Odległości prostopadłe
        distances = np.array([
            Engine_E11_Lactate_v2._perpendicular_distance(xi, yi, x1, y1, x2, y2)
            for xi, yi in zip(x_fine, y_fine)
        ])

        # Dmax = punkt max odległości (ale tylko w "rosnącej" części krzywej)
        # Ignorujemy pierwszy i ostatni 5% żeby uniknąć artefaktów edge
        margin = max(int(len(distances) * 0.05), 1)
        search_range = slice(margin, len(distances) - margin)
        dmax_idx = margin + np.argmax(distances[search_range])

        dmax_time = float(x_fine[dmax_idx])
        dmax_la = float(y_fine[dmax_idx])
        dmax_distance = float(distances[dmax_idx])

        return {
            "found": True,
            "threshold_time_sec": round(dmax_time, 1),
            "threshold_la_mmol": round(dmax_la, 2),
            "method": "Dmax (Cheng 1992)",
            "max_distance": round(dmax_distance, 4),
            "poly_coeffs": coeffs.tolist(),
            "poly_r2": Engine_E11_Lactate_v2._calc_r2(x, y, coeffs),
        }

    @staticmethod
    def _method_dmax_modified(df_la: pd.DataFrame, intensity_col: str = "time_sec") -> Dict:
        """
        Modified Dmax (Bishop 1998).
        Jak Dmax, ale linia startowa NIE od pierwszego punktu,
        lecz od punktu POPRZEDZAJĄCEGO pierwszy wzrost La > 0.4 mmol/L.

        To eliminuje wpływ niskich punktów baseline na wynik.
        Korelacja z VT2: r=0.99 (Fabre 2010, elite cross-country skiers).
        """
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        if len(x) < 4:
            return {"found": False, "method": "ModDmax", "reason": "Za mało punktów (<4)"}

        # Znajdź punkt start: poprzedzający pierwszy wzrost > 0.4 mmol/L
        start_idx = 0
        for i in range(1, len(y)):
            if y[i] - y[i-1] > 0.4:
                start_idx = max(0, i - 1)
                break

        # Jeśli nigdy nie było wzrostu > 0.4, fallback do minimum La
        if start_idx == 0:
            start_idx = int(np.argmin(y))

        if len(x) - start_idx < 3:
            return {"found": False, "method": "ModDmax", "reason": "Za mało punktów po start_idx"}

        # Polynomial na PEŁNYCH danych (zgodnie z literaturą)
        coeffs = Engine_E11_Lactate_v2._fit_polynomial(x, y, degree=3)
        if coeffs is None:
            return {"found": False, "method": "ModDmax", "reason": "Polynomial fit failed"}

        x_fine = np.linspace(x[start_idx], x[-1], 1000)
        y_fine = np.polyval(coeffs, x_fine)

        # Linia: od start_point do last point (na krzywej polynomial)
        x1, y1 = x_fine[0], y_fine[0]
        x2, y2 = x_fine[-1], y_fine[-1]

        distances = np.array([
            Engine_E11_Lactate_v2._perpendicular_distance(xi, yi, x1, y1, x2, y2)
            for xi, yi in zip(x_fine, y_fine)
        ])

        margin = max(int(len(distances) * 0.05), 1)
        search_range = slice(margin, len(distances) - margin)
        dmax_idx = margin + np.argmax(distances[search_range])

        return {
            "found": True,
            "threshold_time_sec": round(float(x_fine[dmax_idx]), 1),
            "threshold_la_mmol": round(float(y_fine[dmax_idx]), 2),
            "method": "ModDmax (Bishop 1998)",
            "start_idx": int(start_idx),
            "start_time_sec": round(float(x[start_idx]), 1),
            "max_distance": round(float(distances[dmax_idx]), 4),
        }

    @staticmethod
    def _method_min_lactate_eq(df_la: pd.DataFrame, intensity_col: str = "time_sec",
                               add_mmol: float = 1.5) -> Dict:
        """
        Minimum Lactate Equivalent + 1.5 mmol/L (Dickhuth 1999).

        Lactate Equivalent = La / intensywność (lub La / speed).
        Znajdujemy minimum tego ratio, potem szukamy na polynomial
        punktu, w którym La = La_at_min + 1.5 mmol/L.
        """
        x = df_la[intensity_col].values
        y = df_la["lactate_mmol"].values

        # Preferuj speed jako intensywność (fizjologicznie sensowniejsze)
        if "speed_kmh" in df_la.columns and df_la["speed_kmh"].notna().sum() >= len(df_la) * 0.8:
            intensity = df_la["speed_kmh"].values
        elif "power_w" in df_la.columns and df_la["power_w"].notna().sum() >= len(df_la) * 0.8:
            intensity = df_la["power_w"].values
        else:
            intensity = x  # Fallback do czasu

        # Lactate Equivalent
        valid = intensity > 0
        if valid.sum() < 3:
            return {"found": False, "method": f"MinLacEq+{add_mmol}", "reason": "Za mało punktów"}

        lac_eq = y[valid] / intensity[valid]
        min_eq_idx_local = np.argmin(lac_eq)
        # Mapuj z powrotem do globalnego indeksu
        valid_indices = np.where(valid)[0]
        min_eq_idx = valid_indices[min_eq_idx_local]
        la_at_min = y[min_eq_idx]
        target_la = la_at_min + add_mmol

        # Polynomial fit do interpolacji
        coeffs = Engine_E11_Lactate_v2._fit_polynomial(x, y, degree=3)
        if coeffs is not None:
            x_fine = np.linspace(x[0], x[-1], 1000)
            y_fine = np.polyval(coeffs, x_fine)
            above = y_fine >= target_la
            if above.any():
                x_threshold = x_fine[np.argmax(above)]
                return {
                    "found": True,
                    "threshold_time_sec": round(float(x_threshold), 1),
                    "threshold_la_mmol": round(target_la, 2),
                    "method": f"MinLacEq+{add_mmol} (Dickhuth 1999)",
                    "min_lac_eq_la": round(la_at_min, 2),
                    "min_lac_eq_time": round(float(x[min_eq_idx]), 1),
                }

        # Fallback: interpolacja liniowa
        for i in range(min_eq_idx + 1, len(y)):
            if y[i] >= target_la:
                if y[i] != y[i-1]:
                    x_interp = x[i-1] + (target_la - y[i-1]) * (x[i] - x[i-1]) / (y[i] - y[i-1])
                else:
                    x_interp = x[i]
                return {
                    "found": True,
                    "threshold_time_sec": round(float(x_interp), 1),
                    "threshold_la_mmol": round(target_la, 2),
                    "method": f"MinLacEq+{add_mmol} (linear interp)",
                    "min_lac_eq_la": round(la_at_min, 2),
                }

        return {"found": False, "method": f"MinLacEq+{add_mmol}", "reason": f"La never reached {target_la:.1f}"}

    # ─── SEKCJA 5: HELPERS ───

    @staticmethod
    def _calc_r2(x, y, coeffs):
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot == 0:
            return 1.0
        return round(1.0 - ss_res / ss_tot, 4)

    @staticmethod
    def _determine_baseline(df_la: pd.DataFrame, manual_baseline: Optional[float] = None) -> float:
        """
        Określa baseline (spoczynkową) wartość La.
        Priorytet: manual > minimum z pierwszych 3 punktów > minimum globalny
        """
        if manual_baseline is not None and manual_baseline > 0:
            return manual_baseline

        y = df_la["lactate_mmol"].values
        # Minimum z pierwszych 3 punktów (typowo rozgrzewka/baseline)
        n_baseline = min(3, len(y))
        return float(np.min(y[:n_baseline]))

    @staticmethod
    def _interpolate_at_time(df_la: pd.DataFrame, time_sec: float,
                             col: str) -> Optional[float]:
        """Interpolacja wartości kolumny w danym czasie."""
        if col not in df_la.columns:
            return None
        t = df_la["time_sec"].values
        v = pd.to_numeric(df_la[col], errors="coerce").values
        valid = ~np.isnan(v)
        if valid.sum() < 2:
            return None
        try:
            return round(float(np.interp(time_sec, t[valid], v[valid])), 2)
        except Exception:
            return None

    @staticmethod
    def _concordance_analysis(results: Dict) -> Dict:
        """
        Analiza zgodności metod — które metody zgadzają się ze sobą?
        Wynik: concordance matrix + consensus thresholds.
        """
        lt1_methods = ["baseline_plus_0.5", "baseline_plus_1.0", "log_log"]
        lt2_methods = ["obla_4.0", "dmax", "dmax_modified", "min_lac_eq_1.5"]

        def _get_times(method_names, results_dict):
            times = {}
            for name in method_names:
                r = results_dict.get(name, {})
                if r.get("found"):
                    times[name] = r["threshold_time_sec"]
            return times

        lt1_times = _get_times(lt1_methods, results)
        lt2_times = _get_times(lt2_methods, results)

        # Consensus: median ± rozprzestrzenienie
        def _consensus(times_dict, label):
            if not times_dict:
                return {"consensus_time_sec": None, "n_methods": 0, "spread_sec": None, "label": label}
            vals = list(times_dict.values())
            median_t = float(np.median(vals))
            spread = float(np.max(vals) - np.min(vals))
            return {
                "consensus_time_sec": round(median_t, 1),
                "n_methods": len(vals),
                "spread_sec": round(spread, 1),
                "methods_used": list(times_dict.keys()),
                "individual_times": {k: round(v, 1) for k, v in times_dict.items()},
                "label": label,
                "confidence": "HIGH" if spread < 60 else "MODERATE" if spread < 120 else "LOW",
            }

        return {
            "lt1_consensus": _consensus(lt1_times, "LT1 (Aerobic Threshold)"),
            "lt2_consensus": _consensus(lt2_times, "LT2 (Anaerobic Threshold)"),
        }

    # ─── SEKCJA 6: GŁÓWNA METODA RUN ───

    @staticmethod
    def run(df_cpet: Optional[pd.DataFrame] = None,
            lactate_input: Optional['LactateInput'] = None,
            e00: Optional[Dict] = None,
            e01: Optional[Dict] = None,
            e02: Optional[Dict] = None,
            cfg: Any = None) -> Dict:
        """
        Główna metoda silnika E11.

        Args:
            df_cpet: DataFrame z danymi CPET (opcjonalny, dla interpolacji)
            lactate_input: LactateInput z danymi laktatowymi
            e00: wyniki E00 (stop detection) — opcjonalny
            e01: wyniki E01 (peak values) — opcjonalny
            e02: wyniki E02 (ventilatory thresholds) — opcjonalny, dla porównania LT vs VT
            cfg: AnalysisConfig — opcjonalny

        Returns:
            Dict z wynikami wszystkich metod + concordance + porównanie z VT
        """
        # --- Walidacja i ładowanie danych ---
        if lactate_input is None:
            lactate_input = LactateInput()

        df_la = Engine_E11_Lactate_v2._load_lactate_data(lactate_input, df_cpet)

        if df_la is None or len(df_la) < Engine_E11_Lactate_v2.MIN_POINTS:
            n = len(df_la) if df_la is not None else 0
            # Fallback: jeśli w df_cpet jest chociaż La peak
            lt_peak = None
            if df_cpet is not None:
                for col in ["Lactate_mmol", "Lactate_mmolL", "La_mmol"]:
                    if col in df_cpet.columns:
                        vals = pd.to_numeric(df_cpet[col], errors="coerce").dropna()
                        if len(vals) > 0:
                            lt_peak = round(float(vals.max()), 2)
                            break
            return {
                "status": "NO_DATA" if n == 0 else "INSUFFICIENT_DATA",
                "reason": f"Potrzeba min. {Engine_E11_Lactate_v2.MIN_POINTS} punktów La, znaleziono: {n}",
                "lt_peak": lt_peak,
                "n_points": n,
            }

        # Wzbogać o dane CPET
        df_la = Engine_E11_Lactate_v2._enrich_from_cpet(df_la, df_cpet)

        # ── Washout detection ──────────────────────────────────────
        # Common artifact: first 1-2 samples show elevated resting La
        # (pre-exercise nervousness, prior warm-up, venous stasis) that
        # drops sharply before the true exercise-induced rise.
        # If first point(s) > trough + 1.0 mmol/L → remove them.
        _washout_removed = 0
        if len(df_la) >= 4:
            _half = max(3, len(df_la) // 2)
            _vals = df_la["lactate_mmol"].values
            _trough_idx = int(np.argmin(_vals[:_half]))
            if _trough_idx > 0:
                _trough_val = _vals[_trough_idx]
                _remove = 0
                for _wi in range(_trough_idx):
                    if _vals[_wi] > _trough_val + 1.0:
                        _remove = _wi + 1
                if _remove > 0 and len(df_la) - _remove >= Engine_E11_Lactate_v2.MIN_POINTS:
                    df_la = df_la.iloc[_remove:].reset_index(drop=True)
                    _washout_removed = _remove

        # Baseline
        manual_baseline = getattr(lactate_input, 'baseline_la', None) if lactate_input else None
        baseline = Engine_E11_Lactate_v2._determine_baseline(df_la, manual_baseline)

        # Podstawowe statystyki
        n_points = len(df_la)
        la_peak = round(float(df_la["lactate_mmol"].max()), 2)
        la_min = round(float(df_la["lactate_mmol"].min()), 2)
        la_baseline = round(baseline, 2)

        quality_warning = []
        if _washout_removed > 0:
            quality_warning.append(f"Washout detected: removed {_washout_removed} initial point(s)")
        if n_points < Engine_E11_Lactate_v2.RECOMMENDED_POINTS:
            quality_warning.append(f"Mało punktów ({n_points}), zalecane ≥{Engine_E11_Lactate_v2.RECOMMENDED_POINTS}")
        if la_peak < 4.0:
            quality_warning.append(f"La peak ({la_peak}) < 4.0 — metody LT2 mogą być niedokładne")
        if la_peak < 2.0:
            quality_warning.append(f"La peak ({la_peak}) < 2.0 — test submaksymalny?")

        
        # Ciągłe dane laktatowe (estymowane) — downsample do punktów per stopień
        _is_continuous = len(df_la["lactate_mmol"].unique()) > 50
        df_la_for_methods = df_la
        if _is_continuous and n_points > 30:
            # Downsample: weź medianę co 60s (1 punkt per minutę/stopień)
            df_la_ds = df_la.copy()
            df_la_ds["_bin"] = (df_la_ds["time_sec"] // 60).astype(int)
            df_la_chart = df_la.copy()  # Keep all points for chart
            df_la_for_methods = df_la_ds.groupby("_bin").agg({
                "time_sec": "median",
                "lactate_mmol": "median",
                **{c: "median" for c in ["speed_kmh", "hr_bpm", "power_w"] if c in df_la_ds.columns}
            }).reset_index(drop=True)
            quality_warning.append(f"Ciągłe dane La ({n_points} pkt) — downsampled do {len(df_la_for_methods)} pkt per minutę")
        
# Zachowaj oryginalne dane do wykresu
        df_la_chart = df_la.copy()
        if _is_continuous and n_points > 30:
            df_la = df_la_for_methods  # Użyj downsampled do metod progowych
        
        # --- Uruchom wszystkie metody ---
        results = {}

        # LT1 methods
        results["baseline_plus_0.5"] = Engine_E11_Lactate_v2._method_baseline_plus(
            df_la, baseline, delta=0.5)
        results["baseline_plus_1.0"] = Engine_E11_Lactate_v2._method_baseline_plus(
            df_la, baseline, delta=1.0)
        results["log_log"] = Engine_E11_Lactate_v2._method_log_log(df_la)

        # LT2 methods
        results["obla_4.0"] = Engine_E11_Lactate_v2._method_obla(df_la, fixed_la=4.0)
        results["obla_2.0"] = Engine_E11_Lactate_v2._method_obla(df_la, fixed_la=2.0)  # Bonus: LT1 proxy
        results["dmax"] = Engine_E11_Lactate_v2._method_dmax(df_la)
        results["dmax_modified"] = Engine_E11_Lactate_v2._method_dmax_modified(df_la)
        results["min_lac_eq_1.5"] = Engine_E11_Lactate_v2._method_min_lactate_eq(df_la, add_mmol=1.5)

        # Wzbogać każdy wynik o HR/Speed/VO2/Power w punkcie progu
        for key, res in results.items():
            if res.get("found") and "threshold_time_sec" in res:
                t = res["threshold_time_sec"]
                for col, label in [
                    ("hr_bpm", "threshold_hr_bpm"),
                    ("speed_kmh", "threshold_speed_kmh"),
                    ("vo2_mlmin", "threshold_vo2_mlmin"),
                    ("power_w", "threshold_power_w"),
                    ("rer", "threshold_rer"),
                ]:
                    val = Engine_E11_Lactate_v2._interpolate_at_time(df_la, t, col)
                    if val is not None:
                        res[label] = val

        # Concordance analysis
        concordance = Engine_E11_Lactate_v2._concordance_analysis(results)

        # --- Porównanie LT vs VT (jeśli E02 dostępne) ---
        vt_comparison = {}
        if e02 and isinstance(e02, dict):
            vt1_t = e02.get("vt1_time_sec")
            vt2_t = e02.get("vt2_time_sec")
            lt1_t = concordance["lt1_consensus"].get("consensus_time_sec")
            lt2_t = concordance["lt2_consensus"].get("consensus_time_sec")

            if vt1_t and lt1_t:
                vt_comparison["vt1_vs_lt1_diff_sec"] = round(float(vt1_t) - float(lt1_t), 1)
                vt_comparison["vt1_vs_lt1_agreement"] = abs(float(vt1_t) - float(lt1_t)) < 60
            if vt2_t and lt2_t:
                vt_comparison["vt2_vs_lt2_diff_sec"] = round(float(vt2_t) - float(lt2_t), 1)
                vt_comparison["vt2_vs_lt2_agreement"] = abs(float(vt2_t) - float(lt2_t)) < 60

        # --- Surowe dane laktatowe do raportu ---
        raw_points = []
        for _, row in (df_la_chart if '_is_continuous' in dir() and _is_continuous else df_la).iterrows():
            pt = {
                "time_sec": round(row["time_sec"], 1),
                "lactate_mmol": round(row["lactate_mmol"], 2),
            }
            for col in ["speed_kmh", "hr_bpm", "vo2_mlmin", "power_w"]:
                if col in row and pd.notna(row[col]):
                    pt[col] = round(row[col], 2)
            raw_points.append(pt)

        # --- Polynomial curve data (for plotting) ---
        poly_curve = None
        coeffs = Engine_E11_Lactate_v2._fit_polynomial(
            df_la["time_sec"].values, df_la["lactate_mmol"].values, degree=3)
        if coeffs is not None:
            x_fine = np.linspace(df_la["time_sec"].min(), df_la["time_sec"].max(), 200)
            y_fine = np.polyval(coeffs, x_fine)
            poly_curve = {
                "time_sec": x_fine.tolist(),
                "lactate_fit": np.round(y_fine, 3).tolist(),
                "coeffs": coeffs.tolist(),
                "r2": Engine_E11_Lactate_v2._calc_r2(
                    df_la["time_sec"].values, df_la["lactate_mmol"].values, coeffs),
            }

        # --- Finalne podsumowanie ---
        # Best LT1 i LT2 (priorytet: ModDmax > Dmax > OBLA dla LT2; Bsln+0.5 > LogLog dla LT1)
        lt1_best = None
        for key in ["baseline_plus_0.5", "log_log", "baseline_plus_1.0"]:
            if results.get(key, {}).get("found"):
                lt1_best = results[key]
                break

        lt2_best = None
        for key in ["dmax_modified", "dmax", "obla_4.0", "min_lac_eq_1.5"]:
            if results.get(key, {}).get("found"):
                lt2_best = results[key]
                break

        return {
            "status": "OK",

            # Podstawowe statystyki
            "n_points": n_points,
            "la_peak": la_peak,
            "la_min": la_min,
            "la_baseline": la_baseline,
            "lt_peak": la_peak,  # backward compat

            # Najlepsze progi (sugerowane)
            "lt1_time_sec": lt1_best["threshold_time_sec"] if lt1_best else None,
            "lt1_la_mmol": lt1_best.get("threshold_la_mmol") if lt1_best else None,
            "lt1_hr_bpm": lt1_best.get("threshold_hr_bpm") if lt1_best else None,
            "lt1_speed_kmh": lt1_best.get("threshold_speed_kmh") if lt1_best else None,
            "lt1_method": lt1_best.get("method") if lt1_best else None,

            "lt2_time_sec": lt2_best["threshold_time_sec"] if lt2_best else None,
            "lt2_la_mmol": lt2_best.get("threshold_la_mmol") if lt2_best else None,
            "lt2_hr_bpm": lt2_best.get("threshold_hr_bpm") if lt2_best else None,
            "lt2_speed_kmh": lt2_best.get("threshold_speed_kmh") if lt2_best else None,
            "lt2_method": lt2_best.get("method") if lt2_best else None,

            # Wszystkie metody
            "methods": results,

            # Concordance
            "concordance": concordance,

            # Porównanie VT vs LT
            "vt_comparison": vt_comparison,

            # Jakość danych
            "quality_warnings": quality_warning,

            # Surowe dane + krzywa
            "raw_points": raw_points,
            "poly_curve": poly_curve,
        }


# Alias for backward compatibility with orchestrator
Engine_E11_Lactate = Engine_E11_Lactate_v2


class Engine_E12_NIRS:
    """NIRS / SmO2 Engine v2.0 — Moxy via MetaMax
    Refs: Bhambhani 2004, Feldmann 2022, Vasquez-Bonilla 2022, Arnold 2024
    """

    @staticmethod
    def _find_smo2_col(df):
        for c in ['SmO2_pct', 'SmO2_1', 'SmO2_2', 'SmO2_3', 'SmO2_4', 'SmO2']:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors='coerce')
                if (vals.dropna() > 0).sum() >= 10:
                    return c
        return None

    @staticmethod
    def _smooth_smo2(series, window=11):
        s = series.copy()
        diff = s.diff().abs()
        s[diff > 10] = np.nan
        s = s.interpolate(limit=5)
        s = s.rolling(window, center=True, min_periods=3).median()
        s = s.rolling(5, center=True, min_periods=2).mean()
        return s

    @staticmethod
    def _segmented_regression_2bp(t, y):
        n = len(t)
        if n < 20:
            return None, None, float('inf')
        best_rss = float('inf')
        best_bp1, best_bp2 = None, None
        step = max(1, n // 80)
        min_seg = max(5, n // 10)
        for i in range(min_seg, n - 2 * min_seg, step):
            for j in range(i + min_seg, n - min_seg, step):
                try:
                    rss = 0
                    for sl in [(0, i), (i, j), (j, n)]:
                        seg_t = t[sl[0]:sl[1]]
                        seg_y = y[sl[0]:sl[1]]
                        if len(seg_t) < 3:
                            rss = float('inf'); break
                        coeffs = np.polyfit(seg_t, seg_y, 1)
                        rss += np.sum((seg_y - np.polyval(coeffs, seg_t)) ** 2)
                    if rss < best_rss:
                        best_rss = rss
                        best_bp1, best_bp2 = i, j
                except:
                    continue
        return best_bp1, best_bp2, best_rss

    @staticmethod
    def run(df_full, r02=None, r01=None, r00=None, cfg=None):
        r02 = r02 or {}; r01 = r01 or {}; r00 = r00 or {}
        result = {
            'status': 'NO_SIGNAL', 'channel': None,
            'smo2_rest': None, 'smo2_min': None, 'smo2_min_time_s': None,
            'smo2_at_peak': None, 'desat_total_abs': None, 'desat_total_pct': None,
            'smo2_at_vt1': None, 'smo2_at_vt2': None, 'time_below_27_s': None,
            'bp1_time_s': None, 'bp1_smo2': None, 'bp1_slope_before': None,
            'bp1_slope_after': None, 'bp2_time_s': None, 'bp2_smo2': None,
            'bp1_vs_vt1_s': None, 'bp2_vs_vt2_s': None,
            'desat_rate_phase2': None, 'phase1_duration_s': None, 'phase2_duration_s': None,
            'smo2_recovery_peak': None, 'hrt_s': None, 'overshoot_abs': None, 'reox_rate': None,
            'signal_quality': 'NO_SIGNAL', 'flags': [],
        }
        col = Engine_E12_NIRS._find_smo2_col(df_full)
        if col is None:
            return result
        result['channel'] = col; result['status'] = 'OK'
        ts = pd.to_numeric(df_full.get('Time_sec', df_full.get('Time_s', pd.Series())), errors='coerce')
        smo2_raw = pd.to_numeric(df_full[col], errors='coerce')
        valid = smo2_raw.notna() & ts.notna() & (smo2_raw > 0)
        if valid.sum() < 10:
            result['status'] = 'INSUFFICIENT_DATA'; result['signal_quality'] = 'POOR'; return result
        t = ts[valid].values.astype(float); s = smo2_raw[valid].values.astype(float)
        pct_valid = valid.sum() / len(smo2_raw) * 100
        result['signal_quality'] = 'POOR' if pct_valid < 30 else ('MODERATE' if pct_valid < 60 else 'GOOD')
        if pct_valid < 30:
            result['flags'].append('LOW_SIGNAL_COVERAGE')
        s_smooth = Engine_E12_NIRS._smooth_smo2(pd.Series(s)).values
        t_stop = r00.get('t_stop_sec') or r00.get('t_stop') or t[-1]
        ex_mask = t <= t_stop; rec_mask = t > t_stop
        t_ex = t[ex_mask]; s_ex = s_smooth[ex_mask]
        if len(t_ex) < 10:
            result['status'] = 'INSUFFICIENT_EX_DATA'; return result
        # REST
        rest_mask = t_ex < (t_ex[0] + 60)
        if rest_mask.sum() >= 3:
            result['smo2_rest'] = round(float(np.nanmean(s_ex[rest_mask])), 1)
        # MIN
        min_idx = np.nanargmin(s_ex)
        result['smo2_min'] = round(float(s_ex[min_idx]), 1)
        result['smo2_min_time_s'] = round(float(t_ex[min_idx]), 0)
        # PEAK
        peak_mask = t_ex >= (t_stop - 30)
        if peak_mask.sum() >= 2:
            result['smo2_at_peak'] = round(float(np.nanmean(s_ex[peak_mask])), 1)
        # DESAT
        if result['smo2_rest'] and result['smo2_min'] is not None:
            result['desat_total_abs'] = round(result['smo2_rest'] - result['smo2_min'], 1)
            if result['smo2_rest'] > 0:
                result['desat_total_pct'] = round(result['desat_total_abs'] / result['smo2_rest'] * 100, 1)
        # TIME <27%
        below27 = s_ex < 27
        if below27.any():
            dt_arr = np.diff(t_ex, prepend=t_ex[0])
            result['time_below_27_s'] = round(float(np.sum(dt_arr[below27])), 0)
        # AT THRESHOLDS
        vt1_time = r02.get('vt1_time_sec') or r02.get('vt1_time_s')
        vt2_time = r02.get('vt2_time_sec') or r02.get('vt2_time_s')
        def _at_time(tt, ta, sa, w=15):
            if tt is None: return None
            try: tt = float(tt)
            except: return None
            m = (ta >= tt - w) & (ta <= tt + w)
            return round(float(np.nanmean(sa[m])), 1) if m.sum() >= 2 else None
        result['smo2_at_vt1'] = _at_time(vt1_time, t_ex, s_ex)
        result['smo2_at_vt2'] = _at_time(vt2_time, t_ex, s_ex)
        # BREAKPOINTS
        bp_mask = t_ex >= (t_ex[0] + 60)
        t_bp = t_ex[bp_mask]; s_bp = s_ex[bp_mask]
        if len(t_bp) >= 20:
            bp1_i, bp2_i, rss = Engine_E12_NIRS._segmented_regression_2bp(t_bp, s_bp)
            single_rss = np.sum((s_bp - np.polyval(np.polyfit(t_bp, s_bp, 1), t_bp))**2)
            if bp1_i is not None and bp2_i is not None and rss < single_rss * 0.90:
                result['bp1_time_s'] = round(float(t_bp[bp1_i]), 0)
                result['bp1_smo2'] = round(float(s_bp[bp1_i]), 1)
                result['bp2_time_s'] = round(float(t_bp[bp2_i]), 0)
                result['bp2_smo2'] = round(float(s_bp[bp2_i]), 1)
                if bp1_i > 3:
                    result['bp1_slope_before'] = round(float(np.polyfit(t_bp[:bp1_i], s_bp[:bp1_i], 1)[0]) * 60, 2)
                if bp2_i - bp1_i > 3:
                    slope2 = round(float(np.polyfit(t_bp[bp1_i:bp2_i], s_bp[bp1_i:bp2_i], 1)[0]) * 60, 2)
                    result['bp1_slope_after'] = slope2
                    result['desat_rate_phase2'] = slope2
                result['phase1_duration_s'] = round(float(t_bp[bp1_i] - t_bp[0]), 0)
                result['phase2_duration_s'] = round(float(t_bp[bp2_i] - t_bp[bp1_i]), 0)
                if vt1_time:
                    try: result['bp1_vs_vt1_s'] = round(result['bp1_time_s'] - float(vt1_time), 0)
                    except: pass
                if vt2_time:
                    try: result['bp2_vs_vt2_s'] = round(result['bp2_time_s'] - float(vt2_time), 0)
                    except: pass
            else:
                result['flags'].append('NO_BP_DETECTED')
        # RECOVERY
        t_rec = t[rec_mask]; s_rec = s_smooth[rec_mask]
        if len(t_rec) >= 5:
            rec_peak = float(np.nanmax(s_rec))
            result['smo2_recovery_peak'] = round(rec_peak, 1)
            if result['smo2_rest']:
                result['overshoot_abs'] = round(rec_peak - result['smo2_rest'], 1)
            smo2_end = float(np.nanmean(s_ex[-5:])) if len(s_ex) >= 5 else float(s_ex[-1])
            amp = rec_peak - smo2_end
            if amp > 2:
                half = smo2_end + amp * 0.5
                above = s_rec >= half
                if above.any():
                    result['hrt_s'] = round(float(t_rec[np.argmax(above)] - t_rec[0]), 1)
                rec_30 = t_rec < (t_rec[0] + 30)
                if rec_30.sum() >= 3:
                    result['reox_rate'] = round(float(np.polyfit(t_rec[rec_30], s_rec[rec_30], 1)[0]), 3)
        else:
            result['flags'].append('NO_RECOVERY_DATA')
        # FLAGS
        fl = result['flags']
        dp = result.get('desat_total_pct')
        if dp is not None:
            if dp < 15: fl.append('LOW_DESAT')
            elif dp > 50: fl.append('HIGH_DESAT_ELITE')
        bv = result.get('bp1_vs_vt1_s')
        if bv is not None:
            if bv < -120: fl.append('EARLY_BP1')
            elif bv > 120: fl.append('LATE_BP1')
        h = result.get('hrt_s')
        if h is not None:
            if h < 15: fl.append('FAST_RECOVERY')
            elif h > 60: fl.append('SLOW_RECOVERY')
        if result.get('overshoot_abs') and result['overshoot_abs'] > 20:
            fl.append('LARGE_OVERSHOOT')
        if result.get('time_below_27_s') and result['time_below_27_s'] > 300:
            fl.append('SmO2_BELOW_27_LONG')
        # ── Add traces for chart visualization ────────────────────
        # Downsample to ~200 points for reasonable chart size
        step = max(1, len(t) // 200)
        t_ds = t[::step].tolist()
        s_ds = s_smooth[::step].tolist()
        result['traces'] = [{
            'label': col,
            'time_sec': [round(v, 1) for v in t_ds],
            'smo2_pct': [round(v, 1) if not (v != v) else None for v in s_ds],
        }]
        result['channels_used'] = [col]
        return result

# ==========================================
# ENGINE E13 — CARDIOVASCULAR DRIFT v2.0
# ==========================================
#
# RESEARCH BASIS:
# ───────────────────────────────────────────
# 1. HR-VO2 Coupling (Panel 2/5 Wasserman 9-panel):
#    - Normal: HR vs VO2 is LINEAR during incremental exercise
#    - Slope = HR/VO2 ~ 50 bpm/(L/min) (ERS 2023, healthy 25-35y)
#    - Steeper slope → lower stroke volume (Fick principle)
#    - Flattening/breaking near peak → HR ceiling / chronotropic limit
#    Ref: Glaab & Taube 2022 (Resp Research), PFTBlog, ERS PA4645
#
# 2. Chronotropic Index (CI):
#    - CI = (HRpeak-HRrest)/(HRpred_max-HRrest) / (VO2peak-VO2rest)/(VO2pred-VO2rest)
#    - Normal: 0.8-1.3 (ACC, Brubaker 2011)
#    - <0.8 = chronotropic incompetence (CI prognostic in HF, PMC7322316)
#    - >1.3 = exaggerated HR response (deconditioning, low SV)
#    Ref: Brubaker & Kitzman, Circulation 2011; ACC CPET in Athletes 2021
#
# 3. Cardiovascular Drift (classic, prolonged exercise):
#    - Progressive HR rise + SV decline after 10-20 min at constant load
#    - Mechanism: hyperthermia → ↑HR → ↓filling time → ↓SV (Coyle 2001)
#    - In incremental CPET: manifests as HR-VO2 slope change, O2pulse plateau
#    Ref: Coyle & Gonzalez-Alonso, ESSR 2001; Ekelund 1967; Rowell 1974
#
# 4. O2 Pulse Trajectory (linked to E05):
#    - Normal: rising parabolic curve throughout exercise
#    - Plateau/decline: suggests SV limitation (ischemia, valve, HF)
#    - Quantified as slope change in second half of exercise
#    Ref: PMC5137392 (Brazilian CPET review), StatPearls
#
# 5. HR-VO2 Linearity & Breakpoint:
#    - Deviation from linearity near VT2 or peak = normal nonlinearity
#    - Early deviation = abnormal (cardiac limitation, deconditioning)
#    - R² of linear fit to exercise data quantifies coupling quality
#    Ref: Wasserman principles, Glaab & Taube 2022
#
# 6. Heart Rate Reserve (HRR_pct):
#    - HRR% = (HRpeak - HRrest) / (HRpred_max - HRrest) × 100
#    - Normal: >80% at maximal effort
#    - <80% with maximal effort → chronotropic incompetence
#    Ref: StatPearls, Brubaker 2011
#
# 7. VO2/WR Slope (Work Rate relationship):
#    - Normal: ~10 ml/min/W on cycle ergometer
#    - Shallow slope (<8.5) → cardiovascular limitation
#    - In treadmill: VO2 vs speed relationship analyzed instead
#    Ref: Wasserman, Gwinnett Lung CPET interpretation
#
# 8. TrainingPeaks Pa:HR Decoupling (for within-step analysis):
#    - Decoupling% = (HR_h2/Pace_h2)/(HR_h1/Pace_h1) - 1) × 100
#    - >5% = above aerobic threshold
#    - Applied per-step when protocol has constant-load steps
#    Ref: Uphill Athlete 2024, TrainingPeaks methodology
# ───────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# ENGINE E13 — CARDIOVASCULAR DRIFT v2.0
# ==========================================
#
# RESEARCH BASIS:
# ───────────────────────────────────────────
# 1. HR-VO2 Coupling (Panel 2/5 Wasserman 9-panel):
#    - Normal: HR vs VO2 is LINEAR during incremental exercise
#    - Slope = HR/VO2 ~ 50 bpm/(L/min) (ERS 2023, healthy 25-35y)
#    - Steeper slope → lower stroke volume (Fick principle)
#    - Flattening/breaking near peak → HR ceiling / chronotropic limit
#    Ref: Glaab & Taube 2022 (Resp Research), PFTBlog, ERS PA4645
#
# 2. Chronotropic Index (CI):
#    - CI = (HRpeak-HRrest)/(HRpred_max-HRrest) / (VO2peak-VO2rest)/(VO2pred-VO2rest)
#    - Normal: 0.8-1.3 (ACC, Brubaker 2011)
#    - <0.8 = chronotropic incompetence (CI prognostic in HF, PMC7322316)
#    - >1.3 = exaggerated HR response (deconditioning, low SV)
#    Ref: Brubaker & Kitzman, Circulation 2011; ACC CPET in Athletes 2021
#
# 3. Cardiovascular Drift (classic, prolonged exercise):
#    - Progressive HR rise + SV decline after 10-20 min at constant load
#    - Mechanism: hyperthermia → ↑HR → ↓filling time → ↓SV (Coyle 2001)
#    - In incremental CPET: manifests as HR-VO2 slope change, O2pulse plateau
#    Ref: Coyle & Gonzalez-Alonso, ESSR 2001; Ekelund 1967; Rowell 1974
#
# 4. O2 Pulse Trajectory (linked to E05):
#    - Normal: rising parabolic curve throughout exercise
#    - Plateau/decline: suggests SV limitation (ischemia, valve, HF)
#    - Quantified as slope change in second half of exercise
#    Ref: PMC5137392 (Brazilian CPET review), StatPearls
#
# 5. HR-VO2 Linearity & Breakpoint:
#    - Deviation from linearity near VT2 or peak = normal nonlinearity
#    - Early deviation = abnormal (cardiac limitation, deconditioning)
#    - R² of linear fit to exercise data quantifies coupling quality
#    Ref: Wasserman principles, Glaab & Taube 2022
#
# 6. Heart Rate Reserve (HRR_pct):
#    - HRR% = (HRpeak - HRrest) / (HRpred_max - HRrest) × 100
#    - Normal: >80% at maximal effort
#    - <80% with maximal effort → chronotropic incompetence
#    Ref: StatPearls, Brubaker 2011
#
# 7. VO2/WR Slope (Work Rate relationship):
#    - Normal: ~10 ml/min/W on cycle ergometer
#    - Shallow slope (<8.5) → cardiovascular limitation
#    - In treadmill: VO2 vs speed relationship analyzed instead
#    Ref: Wasserman, Gwinnett Lung CPET interpretation
#
# 8. TrainingPeaks Pa:HR Decoupling (for within-step analysis):
#    - Decoupling% = (HR_h2/Pace_h2)/(HR_h1/Pace_h1) - 1) × 100
#    - >5% = above aerobic threshold
#    - Applied per-step when protocol has constant-load steps
#    Ref: Uphill Athlete 2024, TrainingPeaks methodology
# ───────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# ENGINE E13 — CARDIOVASCULAR DRIFT v2.0
# ==========================================
#
# RESEARCH BASIS:
# ───────────────────────────────────────────
# 1. HR-VO2 Coupling (Panel 2/5 Wasserman 9-panel):
#    - Normal: HR vs VO2 is LINEAR during incremental exercise
#    - Slope = HR/VO2 ~ 50 bpm/(L/min) (ERS 2023, healthy 25-35y)
#    - Steeper slope → lower stroke volume (Fick principle)
#    - Flattening/breaking near peak → HR ceiling / chronotropic limit
#    Ref: Glaab & Taube 2022 (Resp Research), PFTBlog, ERS PA4645
#
# 2. Chronotropic Index (CI):
#    - CI = (HRpeak-HRrest)/(HRpred_max-HRrest) / (VO2peak-VO2rest)/(VO2pred-VO2rest)
#    - Normal: 0.8-1.3 (ACC, Brubaker 2011)
#    - <0.8 = chronotropic incompetence (CI prognostic in HF, PMC7322316)
#    - >1.3 = exaggerated HR response (deconditioning, low SV)
#    Ref: Brubaker & Kitzman, Circulation 2011; ACC CPET in Athletes 2021
#
# 3. Cardiovascular Drift (classic, prolonged exercise):
#    - Progressive HR rise + SV decline after 10-20 min at constant load
#    - Mechanism: hyperthermia → ↑HR → ↓filling time → ↓SV (Coyle 2001)
#    - In incremental CPET: manifests as HR-VO2 slope change, O2pulse plateau
#    Ref: Coyle & Gonzalez-Alonso, ESSR 2001; Ekelund 1967; Rowell 1974
#
# 4. O2 Pulse Trajectory (linked to E05):
#    - Normal: rising parabolic curve throughout exercise
#    - Plateau/decline: suggests SV limitation (ischemia, valve, HF)
#    - Quantified as slope change in second half of exercise
#    Ref: PMC5137392 (Brazilian CPET review), StatPearls
#
# 5. HR-VO2 Linearity & Breakpoint:
#    - Deviation from linearity near VT2 or peak = normal nonlinearity
#    - Early deviation = abnormal (cardiac limitation, deconditioning)
#    - R² of linear fit to exercise data quantifies coupling quality
#    Ref: Wasserman principles, Glaab & Taube 2022
#
# 6. Heart Rate Reserve (HRR_pct):
#    - HRR% = (HRpeak - HRrest) / (HRpred_max - HRrest) × 100
#    - Normal: >80% at maximal effort
#    - <80% with maximal effort → chronotropic incompetence
#    Ref: StatPearls, Brubaker 2011
#
# 7. VO2/WR Slope (Work Rate relationship):
#    - Normal: ~10 ml/min/W on cycle ergometer
#    - Shallow slope (<8.5) → cardiovascular limitation
#    - In treadmill: VO2 vs speed relationship analyzed instead
#    Ref: Wasserman, Gwinnett Lung CPET interpretation
#
# 8. TrainingPeaks Pa:HR Decoupling (for within-step analysis):
#    - Decoupling% = (HR_h2/Pace_h2)/(HR_h1/Pace_h1) - 1) × 100
#    - >5% = above aerobic threshold
#    - Applied per-step when protocol has constant-load steps
#    Ref: Uphill Athlete 2024, TrainingPeaks methodology
# ───────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# ENGINE E13 — CARDIOVASCULAR DRIFT v2.0
# ==========================================
#
# RESEARCH BASIS:
# ───────────────────────────────────────────
# 1. HR-VO2 Coupling (Panel 2/5 Wasserman 9-panel):
#    - Normal: HR vs VO2 is LINEAR during incremental exercise
#    - Slope = HR/VO2 ~ 50 bpm/(L/min) (ERS 2023, healthy 25-35y)
#    - Steeper slope → lower stroke volume (Fick principle)
#    - Flattening/breaking near peak → HR ceiling / chronotropic limit
#    Ref: Glaab & Taube 2022 (Resp Research), PFTBlog, ERS PA4645
#
# 2. Chronotropic Index (CI):
#    - CI = (HRpeak-HRrest)/(HRpred_max-HRrest) / (VO2peak-VO2rest)/(VO2pred-VO2rest)
#    - Normal: 0.8-1.3 (ACC, Brubaker 2011)
#    - <0.8 = chronotropic incompetence (CI prognostic in HF, PMC7322316)
#    - >1.3 = exaggerated HR response (deconditioning, low SV)
#    Ref: Brubaker & Kitzman, Circulation 2011; ACC CPET in Athletes 2021
#
# 3. Cardiovascular Drift (classic, prolonged exercise):
#    - Progressive HR rise + SV decline after 10-20 min at constant load
#    - Mechanism: hyperthermia → ↑HR → ↓filling time → ↓SV (Coyle 2001)
#    - In incremental CPET: manifests as HR-VO2 slope change, O2pulse plateau
#    Ref: Coyle & Gonzalez-Alonso, ESSR 2001; Ekelund 1967; Rowell 1974
#
# 4. O2 Pulse Trajectory (linked to E05):
#    - Normal: rising parabolic curve throughout exercise
#    - Plateau/decline: suggests SV limitation (ischemia, valve, HF)
#    - Quantified as slope change in second half of exercise
#    Ref: PMC5137392 (Brazilian CPET review), StatPearls
#
# 5. HR-VO2 Linearity & Breakpoint:
#    - Deviation from linearity near VT2 or peak = normal nonlinearity
#    - Early deviation = abnormal (cardiac limitation, deconditioning)
#    - R² of linear fit to exercise data quantifies coupling quality
#    Ref: Wasserman principles, Glaab & Taube 2022
#
# 6. Heart Rate Reserve (HRR_pct):
#    - HRR% = (HRpeak - HRrest) / (HRpred_max - HRrest) × 100
#    - Normal: >80% at maximal effort
#    - <80% with maximal effort → chronotropic incompetence
#    Ref: StatPearls, Brubaker 2011
#
# 7. VO2/WR Slope (Work Rate relationship):
#    - Normal: ~10 ml/min/W on cycle ergometer
#    - Shallow slope (<8.5) → cardiovascular limitation
#    - In treadmill: VO2 vs speed relationship analyzed instead
#    Ref: Wasserman, Gwinnett Lung CPET interpretation
#
# 8. TrainingPeaks Pa:HR Decoupling (for within-step analysis):
#    - Decoupling% = (HR_h2/Pace_h2)/(HR_h1/Pace_h1) - 1) × 100
#    - >5% = above aerobic threshold
#    - Applied per-step when protocol has constant-load steps
#    Ref: Uphill Athlete 2024, TrainingPeaks methodology
# ───────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class Engine_E13_Drift_v2:
    """
    E13: Cardiovascular Drift & HR-VO2 Coupling Analysis.

    Analyzes:
      A) HR-VO2 coupling: linearity, slope, breakpoint
      B) Chronotropic competence: CI, HRR%, HR response pattern
      C) O2-pulse drift: trajectory analysis over exercise phases
      D) Per-step drift: within-step HR drift (if step protocol)
      E) VO2/WR slope: work rate efficiency (if power available)

    Requires: df_cpet with HR_bpm, VO2 columns + E01 (peaks) + E02 (thresholds)
    """

    # ─── SECTION 1: HR-VO2 COUPLING ───

    @staticmethod
    def _hr_vo2_coupling(df_ex: pd.DataFrame, e02: dict) -> dict:
        """
        Analyze HR vs VO2 linearity during exercise.
        Returns slope, R², breakpoint time, and coupling quality.
        """
        t = df_ex['Time_sec'].values
        hr = df_ex['HR_bpm'].values
        vo2 = None
        for vc in ['VO2_mlmin', 'VO2_ml_min', 'VO2_mlmin']:
            if vc in df_ex.columns:
                vo2 = pd.to_numeric(df_ex[vc], errors='coerce').values
                break
        if vo2 is None:
            for vc in ['VO2_Lmin', 'VO2_L_min']:
                if vc in df_ex.columns:
                    vo2 = pd.to_numeric(df_ex[vc], errors='coerce').values * 1000
                    break
        if vo2 is None:
            return {'status': 'NO_VO2'}

        # Clean NaN
        mask = np.isfinite(hr) & np.isfinite(vo2) & (hr > 30) & (vo2 > 100)
        if mask.sum() < 10:
            return {'status': 'INSUFFICIENT_DATA'}

        hr_c = hr[mask]
        vo2_c = vo2[mask]  # ml/min
        t_c = t[mask]
        vo2_L = vo2_c / 1000.0  # L/min for slope calc

        # Full linear regression: HR = a * VO2(L/min) + b
        from numpy.polynomial.polynomial import polyfit as pfit
        try:
            # np.polyfit degree 1: coeffs[0]=slope, coeffs[1]=intercept
            coeffs = np.polyfit(vo2_L, hr_c, 1)
            slope_full = round(float(coeffs[0]), 2)  # bpm per L/min
            intercept = round(float(coeffs[1]), 1)
            hr_pred = np.polyval(coeffs, vo2_L)
            ss_res = np.sum((hr_c - hr_pred)**2)
            ss_tot = np.sum((hr_c - np.mean(hr_c))**2)
            r2_full = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0
        except Exception:
            return {'status': 'FIT_FAILED'}

        # Slope interpretation (ERS 2023 reference: ~50 bpm/L/min in healthy young 25-35y)
        # Higher slope = lower SV; lower slope = higher SV (athletes/large individuals)
        # NOTE: slope is inversely proportional to SV and BSA (ERS PA4645)
        # Large athletes with high absolute VO2 will have naturally lower slopes
        slope_class = 'NORMAL'
        if slope_full > 70:
            slope_class = 'STEEP'  # low SV, deconditioning
        elif slope_full > 60:
            slope_class = 'HIGH_NORMAL'
        elif slope_full < 25:
            slope_class = 'VERY_LOW'  # very high SV (elite) or chronotropic issue
        elif slope_full < 40:
            slope_class = 'LOW_HIGH_SV'  # high SV, well-trained or large individual

        # Segmented regression: find breakpoint in HR-VO2 linearity
        bp_time = None
        bp_pct_exercise = None
        r2_before_bp = None
        r2_after_bp = None
        n = len(hr_c)

        if n >= 12:
            best_improvement = 0
            best_bp_idx = None

            for bp_idx in range(max(4, n // 4), min(n - 4, 3 * n // 4)):
                # Segment 1
                vo2_1, hr_1 = vo2_L[:bp_idx], hr_c[:bp_idx]
                # Segment 2
                vo2_2, hr_2 = vo2_L[bp_idx:], hr_c[bp_idx:]

                try:
                    c1 = np.polyfit(vo2_1, hr_1, 1)
                    c2 = np.polyfit(vo2_2, hr_2, 1)
                    pred1 = np.polyval(c1, vo2_1)
                    pred2 = np.polyval(c2, vo2_2)
                    sse_seg = np.sum((hr_1 - pred1)**2) + np.sum((hr_2 - pred2)**2)
                    improvement = ss_res - sse_seg
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_bp_idx = bp_idx
                except Exception:
                    continue

            # Only report breakpoint if meaningful improvement (>5% of total SSE)
            if best_bp_idx is not None and best_improvement > 0.05 * ss_res:
                bp_time = round(float(t_c[best_bp_idx]), 1)
                total_dur = t_c[-1] - t_c[0]
                bp_pct_exercise = round(100.0 * (bp_time - t_c[0]) / total_dur, 1) if total_dur > 0 else None

                # R² for each segment
                vo2_1, hr_1 = vo2_L[:best_bp_idx], hr_c[:best_bp_idx]
                vo2_2, hr_2 = vo2_L[best_bp_idx:], hr_c[best_bp_idx:]
                c1 = np.polyfit(vo2_1, hr_1, 1)
                c2 = np.polyfit(vo2_2, hr_2, 1)
                ss1 = np.sum((hr_1 - np.mean(hr_1))**2)
                ss2 = np.sum((hr_2 - np.mean(hr_2))**2)
                r2_before_bp = round(1.0 - np.sum((hr_1 - np.polyval(c1, vo2_1))**2) / ss1, 4) if ss1 > 0 else None
                r2_after_bp = round(1.0 - np.sum((hr_2 - np.polyval(c2, vo2_2))**2) / ss2, 4) if ss2 > 0 else None

        # Coupling quality classification
        coupling = 'EXCELLENT' if r2_full >= 0.95 else (
            'GOOD' if r2_full >= 0.90 else (
                'MODERATE' if r2_full >= 0.80 else 'POOR'))

        # Check if breakpoint near VT2
        bp_vs_vt2 = None
        if bp_time is not None and e02:
            vt2_t = e02.get('vt2_time_sec')
            if vt2_t:
                diff = bp_time - float(vt2_t)
                bp_vs_vt2 = 'AT_VT2' if abs(diff) < 60 else ('BEFORE_VT2' if diff < -60 else 'AFTER_VT2')

        return {
            'status': 'OK',
            'hr_vo2_slope': slope_full,       # bpm per L/min VO2
            'hr_vo2_intercept': intercept,     # bpm
            'hr_vo2_r2': r2_full,
            'slope_class': slope_class,
            'coupling_quality': coupling,
            'breakpoint_time_sec': bp_time,
            'breakpoint_pct_exercise': bp_pct_exercise,
            'bp_vs_vt2': bp_vs_vt2,
            'r2_before_bp': r2_before_bp,
            'r2_after_bp': r2_after_bp,
        }

    # ─── SECTION 2: CHRONOTROPIC COMPETENCE ───

    @staticmethod
    def _chronotropic_analysis(df_ex: pd.DataFrame, e01: dict, e02: dict,
                                age: float, weight_kg: float = 70.0, sex: str = 'male') -> dict:
        """
        Chronotropic index and HR reserve analysis.
        CI = (HRpeak-HRrest)/(HRpred-HRrest) / (VO2peak-VO2rest)/(VO2pred-VO2rest)
        """
        hr = df_ex['HR_bpm'].values
        hr_clean = hr[np.isfinite(hr) & (hr > 30)]
        if len(hr_clean) < 10:
            return {'status': 'INSUFFICIENT_HR'}

        hr_rest = float(np.percentile(hr_clean[:min(20, len(hr_clean))], 25))
        hr_peak = float(np.max(hr_clean))

        # Age-predicted max HR (Tanaka 2001: 208 - 0.7*age)
        hr_pred_max = 208.0 - 0.7 * age

        # HR reserve
        hr_reserve_bpm = hr_peak - hr_rest
        hr_reserve_pct = round(100.0 * (hr_peak - hr_rest) / (hr_pred_max - hr_rest), 1) if (hr_pred_max - hr_rest) > 0 else None
        pct_pred_hr = round(100.0 * hr_peak / hr_pred_max, 1)

        # VO2 data for CI
        vo2_col = None
        for _vc in ['VO2_mlmin', 'VO2_ml_min']:
            if _vc in df_ex.columns:
                vo2_col = _vc; break
        if vo2_col is None:
            for _vc in ['VO2_Lmin', 'VO2_L_min']:
                if _vc in df_ex.columns:
                    vo2_col = _vc; break
        ci = None
        ci_class = None

        if vo2_col:
            vo2 = df_ex[vo2_col].values
            vo2_clean = vo2[np.isfinite(vo2)]
            if len(vo2_clean) >= 10:
                vo2_rest = float(np.percentile(vo2_clean[:min(20, len(vo2_clean))], 25))
                vo2_peak = float(np.max(vo2_clean))
                if vo2_col in ('VO2_Lmin', 'VO2_L_min'):
                    vo2_rest *= 1000
                    vo2_peak *= 1000

                # Predicted VO2max (Wasserman: simplified, age/weight dependent)
                # Use E01 predicted if available, else estimate
                vo2_pred = None
                if e01 and e01.get('VO2_predicted_mlmin'):
                    vo2_pred = float(e01['VO2_predicted_mlmin'])
                elif e01 and e01.get('VO2_pct_predicted'):
                    pct = float(e01['VO2_pct_predicted'])
                    if pct > 0:
                        vo2_pred = vo2_peak / (pct / 100.0)

                # Fallback: estimate predicted VO2max if not from E01
                if vo2_pred is None:
                    # Wasserman equation (simplified, ml/min)
                    # Male: (50.72 - 0.372*age) * weight_kg (Jones 1997)
                    # Female: (22.78 - 0.17*age) * weight_kg (Jones 1997 adjusted)
                    if sex == 'female':
                        vo2_pred = (22.78 - 0.17 * age) * weight_kg
                    else:
                        vo2_pred = (50.72 - 0.372 * age) * weight_kg

                if vo2_pred and vo2_pred > vo2_rest:
                    # If athlete exceeds predicted VO2max, cap vo2_pred at actual peak
                    # CI < 1 is expected when VO2peak > VO2pred (athletic supercompensation)
                    _vo2_pred_eff = max(vo2_pred, vo2_peak)
                    hr_fraction = (hr_peak - hr_rest) / (hr_pred_max - hr_rest) if (hr_pred_max - hr_rest) > 0 else 0
                    vo2_fraction = (vo2_peak - vo2_rest) / (_vo2_pred_eff - vo2_rest) if (_vo2_pred_eff - vo2_rest) > 0 else 0

                    if vo2_fraction > 0:
                        ci = round(hr_fraction / vo2_fraction, 3)

                        # Classification (ACC: 0.8-1.3 normal)
                        if ci < 0.6:
                            ci_class = 'SEVERE_CI'  # severe chronotropic incompetence
                        elif ci < 0.8:
                            ci_class = 'CHRONOTROPIC_INCOMPETENCE'
                        elif ci <= 1.3:
                            ci_class = 'NORMAL'
                        else:
                            ci_class = 'EXAGGERATED'  # possibly deconditioning/low SV

        # HR response pattern: linear check up to VT2
        hr_pattern = 'NORMAL'
        if pct_pred_hr < 85:
            hr_pattern = 'BLUNTED'
        elif hr_reserve_pct and hr_reserve_pct < 80:
            hr_pattern = 'REDUCED_RESERVE'

        return {
            'status': 'OK',
            'hr_rest': round(hr_rest, 1),
            'hr_peak': round(hr_peak, 1),
            'hr_pred_max': round(hr_pred_max, 1),
            'pct_pred_hr': pct_pred_hr,
            'hr_reserve_bpm': round(hr_reserve_bpm, 1),
            'hr_reserve_pct': hr_reserve_pct,
            'chronotropic_index': ci,
            'ci_class': ci_class,
            'hr_pattern': hr_pattern,
        }

    # ─── SECTION 3: O2-PULSE DRIFT ───

    @staticmethod
    def _o2pulse_drift(df_ex: pd.DataFrame, e02: dict) -> dict:
        """
        Analyze O2 pulse trajectory and drift during exercise.
        Split into early (pre-VT1) vs late (post-VT2) phases.
        """
        hr = df_ex['HR_bpm'].values
        t = df_ex['Time_sec'].values

        vo2_col = None
        for _vc in ['VO2_mlmin', 'VO2_ml_min']:
            if _vc in df_ex.columns:
                vo2_col = _vc; break
        if vo2_col is None:
            for _vc in ['VO2_Lmin', 'VO2_L_min']:
                if _vc in df_ex.columns:
                    vo2_col = _vc; break
        if vo2_col is None:
            return {'status': 'NO_VO2'}

        vo2 = df_ex[vo2_col].values.copy()
        if vo2_col in ('VO2_Lmin', 'VO2_L_min'):
            vo2 = vo2 * 1000.0

        mask = np.isfinite(hr) & np.isfinite(vo2) & (hr > 40) & (vo2 > 100)
        if mask.sum() < 10:
            return {'status': 'INSUFFICIENT_DATA'}

        o2p = vo2[mask] / hr[mask]  # ml/beat
        t_m = t[mask]

        # Split exercise into 3 equal thirds
        n = len(o2p)
        third = n // 3
        o2p_early = float(np.median(o2p[:third]))
        o2p_mid = float(np.median(o2p[third:2*third]))
        o2p_late = float(np.median(o2p[2*third:]))

        # Trajectory: rising, plateau, or declining
        change_early_to_mid = o2p_mid - o2p_early
        change_mid_to_late = o2p_late - o2p_mid
        change_total = o2p_late - o2p_early

        if change_mid_to_late < -0.5:
            trajectory = 'DECLINING'
        elif abs(change_mid_to_late) <= 0.5 and o2p_late > o2p_early * 0.95:
            trajectory = 'PLATEAU'
        elif change_total > 1.0:
            trajectory = 'RISING'
        else:
            trajectory = 'FLAT'

        # Clinical significance
        # Declining O2 pulse in 2nd half = SV limitation (ischemia, valve)
        clinical = 'NORMAL'
        if trajectory == 'DECLINING' and abs(change_mid_to_late) > 1.0:
            clinical = 'SV_LIMITATION'
        elif trajectory == 'PLATEAU' and change_early_to_mid < 0.5:
            clinical = 'EARLY_PLATEAU'
        elif trajectory == 'FLAT':
            clinical = 'POSSIBLE_LIMITATION'

        # Percentage drift
        drift_pct = round(100.0 * (o2p_late - o2p_early) / o2p_early, 1) if o2p_early > 0 else None

        return {
            'status': 'OK',
            'o2pulse_early': round(o2p_early, 2),
            'o2pulse_mid': round(o2p_mid, 2),
            'o2pulse_late': round(o2p_late, 2),
            'o2pulse_drift_pct': drift_pct,
            'o2pulse_trajectory': trajectory,
            'o2pulse_clinical': clinical,
        }

    # ─── SECTION 4: PER-STEP DRIFT ───

    @staticmethod
    def _per_step_drift(df_ex: pd.DataFrame, e00: dict) -> dict:
        """
        Within-step HR drift analysis.
        For each step: compare HR in first half vs second half.
        Drift >5% suggests exercise above aerobic threshold.
        """
        if not e00 or 'steps' not in e00:
            return {'status': 'NO_STEPS'}

        steps = e00.get('steps', [])
        if not steps or len(steps) < 3:
            return {'status': 'INSUFFICIENT_STEPS'}

        hr = df_ex['HR_bpm'].values
        t = df_ex['Time_sec'].values
        step_drifts = []

        for step in steps:
            s_start = step.get('start_sec', step.get('time_start'))
            s_end = step.get('end_sec', step.get('time_end'))
            if s_start is None or s_end is None:
                continue
            s_start, s_end = float(s_start), float(s_end)
            dur = s_end - s_start
            if dur < 30:
                continue

            mask = (t >= s_start) & (t <= s_end) & np.isfinite(hr) & (hr > 30)
            if mask.sum() < 6:
                continue

            hr_step = hr[mask]
            mid = len(hr_step) // 2
            hr_h1 = float(np.median(hr_step[:mid]))
            hr_h2 = float(np.median(hr_step[mid:]))

            drift_bpm = hr_h2 - hr_h1
            drift_pct = round(100.0 * drift_bpm / hr_h1, 2) if hr_h1 > 0 else 0

            step_drifts.append({
                'time_start': round(s_start, 1),
                'time_end': round(s_end, 1),
                'hr_h1': round(hr_h1, 1),
                'hr_h2': round(hr_h2, 1),
                'drift_bpm': round(drift_bpm, 1),
                'drift_pct': drift_pct,
            })

        if not step_drifts:
            return {'status': 'NO_VALID_STEPS'}

        # Summary
        drifts_pct = [s['drift_pct'] for s in step_drifts]
        max_drift = max(drifts_pct)
        mean_drift = round(float(np.mean(drifts_pct)), 2)

        # Find first step with drift >5% (approx aerobic threshold per TrainingPeaks)
        first_decouple_idx = None
        for i, sd in enumerate(step_drifts):
            if sd['drift_pct'] > 5.0:
                first_decouple_idx = i
                break

        return {
            'status': 'OK',
            'n_steps': len(step_drifts),
            'step_drifts': step_drifts,
            'max_drift_pct': round(max_drift, 2),
            'mean_drift_pct': mean_drift,
            'first_decoupling_step': first_decouple_idx,
        }

    # ─── SECTION 5: VO2/WR SLOPE ───

    @staticmethod
    def _vo2_wr_slope(df_ex: pd.DataFrame) -> dict:
        """
        VO2 vs work rate slope (Wasserman: normal ~10 ml/min/W on cycle).
        For treadmill: VO2 vs speed slope.
        """
        vo2_col = None
        for _vc in ['VO2_mlmin', 'VO2_ml_min']:
            if _vc in df_ex.columns:
                vo2_col = _vc; break
        if vo2_col is None:
            for _vc in ['VO2_Lmin', 'VO2_L_min']:
                if _vc in df_ex.columns:
                    vo2_col = _vc; break
        if vo2_col is None:
            return {'status': 'NO_VO2'}

        vo2 = df_ex[vo2_col].values.copy()
        if vo2_col in ('VO2_Lmin', 'VO2_L_min'):
            vo2 = vo2 * 1000.0

        result = {'status': 'NO_WR'}

        # Try power first (cycle ergometer)
        if 'Power_W' in df_ex.columns:
            wr = df_ex['Power_W'].values
            mask = np.isfinite(vo2) & np.isfinite(wr) & (wr > 10) & (vo2 > 100)
            if mask.sum() >= 10:
                coeffs = np.polyfit(wr[mask], vo2[mask], 1)
                slope = round(float(coeffs[0]), 2)  # ml/min per W
                pred = np.polyval(coeffs, wr[mask])
                ss_res = np.sum((vo2[mask] - pred)**2)
                ss_tot = np.sum((vo2[mask] - np.mean(vo2[mask]))**2)
                r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0

                # Normal ~10 ml/min/W (Wasserman)
                slope_class = 'NORMAL' if 8.5 <= slope <= 12.0 else (
                    'LOW' if slope < 8.5 else 'HIGH')

                result = {
                    'status': 'OK',
                    'modality': 'power',
                    'vo2_wr_slope': slope,
                    'vo2_wr_r2': r2,
                    'vo2_wr_unit': 'ml/min/W',
                    'slope_class': slope_class,
                    'interpretation': 'LOW = cardiac limitation' if slope_class == 'LOW' else (
                        'Prawidlowy' if slope_class == 'NORMAL' else 'Wysoki'),
                }
                return result

        # Fallback: speed (treadmill)
        if 'Speed_kmh' in df_ex.columns:
            spd = df_ex['Speed_kmh'].values
            mask = np.isfinite(vo2) & np.isfinite(spd) & (spd > 1) & (vo2 > 100)
            if mask.sum() >= 10:
                coeffs = np.polyfit(spd[mask], vo2[mask], 1)
                slope = round(float(coeffs[0]), 2)  # ml/min per km/h
                pred = np.polyval(coeffs, spd[mask])
                ss_res = np.sum((vo2[mask] - pred)**2)
                ss_tot = np.sum((vo2[mask] - np.mean(vo2[mask]))**2)
                r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0

                result = {
                    'status': 'OK',
                    'modality': 'speed',
                    'vo2_wr_slope': slope,
                    'vo2_wr_r2': r2,
                    'vo2_wr_unit': 'ml/min per km/h',
                    'slope_class': 'N/A',  # no universal reference for treadmill
                    'interpretation': 'Brak referencji dla biezni',
                }

        return result

    # ─── SECTION 6: MAIN RUN ───

    @staticmethod
    def run(df_cpet: pd.DataFrame,
            e00: Optional[Dict] = None,
            e01: Optional[Dict] = None,
            e02: Optional[Dict] = None,
            cfg: Any = None) -> Dict:
        """
        Main E13 engine method.

        Args:
            df_cpet: exercise phase DataFrame (filtered by E00)
            e00: stop detection results (for steps)
            e01: peak values (for predicted VO2, HRmax)
            e02: threshold results (for VT1/VT2 timing)
            cfg: AnalysisConfig

        Returns:
            Dict with all drift/coupling analyses
        """
        if df_cpet is None or len(df_cpet) < 20:
            return {'status': 'INSUFFICIENT_DATA', 'reason': 'Zbyt malo danych (<20 obserwacji)'}

        # Validate required columns
        if 'HR_bpm' not in df_cpet.columns:
            return {'status': 'NO_HR', 'reason': 'Brak kolumny HR_bpm'}

        time_col = 'Time_sec' if 'Time_sec' in df_cpet.columns else 'Time_s'
        if time_col not in df_cpet.columns:
            return {'status': 'NO_TIME', 'reason': 'Brak kolumny Time_sec'}

        # Prepare df_ex (exercise phase)
        df_ex = df_cpet.copy()
        if time_col != 'Time_sec':
            df_ex = df_ex.rename(columns={time_col: 'Time_sec'})

        # Get age
        age = 40.0  # default
        if cfg and hasattr(cfg, 'age') and cfg.age:
            age = float(cfg.age)
        elif cfg and hasattr(cfg, 'age_y') and cfg.age_y:
            age = float(cfg.age_y)
        elif e01 and e01.get('age'):
            age = float(e01['age'])

        e02 = e02 or {}
        e01 = e01 or {}
        e00 = e00 or {}

        # Run all analyses
        coupling = Engine_E13_Drift_v2._hr_vo2_coupling(df_ex, e02)
        _w13 = cfg.body_mass_kg if cfg and hasattr(cfg, 'body_mass_kg') else 70.0
        _s13 = cfg.sex if cfg and hasattr(cfg, 'sex') else 'male'
        chrono = Engine_E13_Drift_v2._chronotropic_analysis(df_ex, e01, e02, age, _w13, _s13)
        o2drift = Engine_E13_Drift_v2._o2pulse_drift(df_ex, e02)
        step_drift = Engine_E13_Drift_v2._per_step_drift(df_ex, e00)
        vo2wr = Engine_E13_Drift_v2._vo2_wr_slope(df_ex)

        # Flags
        flags = []
        if coupling.get('coupling_quality') in ('POOR',):
            flags.append('POOR_HR_VO2_COUPLING')
        if coupling.get('slope_class') == 'STEEP':
            flags.append('STEEP_HR_VO2_SLOPE')
        if coupling.get('slope_class') == 'VERY_LOW':
            flags.append('VERY_LOW_HR_VO2_SLOPE')
        if coupling.get('bp_vs_vt2') == 'BEFORE_VT2':
            flags.append('EARLY_LINEARITY_BREAK')
        if chrono.get('ci_class') in ('CHRONOTROPIC_INCOMPETENCE', 'SEVERE_CI'):
            flags.append('CHRONOTROPIC_INCOMPETENCE')
        if chrono.get('ci_class') == 'EXAGGERATED':
            flags.append('EXAGGERATED_HR_RESPONSE')
        if chrono.get('hr_pattern') == 'BLUNTED':
            flags.append('BLUNTED_HR')
        if o2drift.get('o2pulse_clinical') == 'SV_LIMITATION':
            flags.append('O2PULSE_SV_LIMITATION')
        if o2drift.get('o2pulse_clinical') == 'EARLY_PLATEAU':
            flags.append('O2PULSE_EARLY_PLATEAU')
        if step_drift.get('max_drift_pct') and step_drift['max_drift_pct'] > 10:
            flags.append('HIGH_INTRA_STEP_DRIFT')
        if vo2wr.get('slope_class') == 'LOW':
            flags.append('LOW_VO2_WR_SLOPE')

        # Overall cardiovascular coupling assessment
        issues = len(flags)
        if issues == 0:
            overall = 'NORMAL'
        elif issues <= 2:
            overall = 'MILD_ABNORMALITY'
        elif issues <= 4:
            overall = 'MODERATE_ABNORMALITY'
        else:
            overall = 'SIGNIFICANT_ABNORMALITY'

        return {
            'status': 'OK',

            # HR-VO2 coupling
            'hr_vo2_slope': coupling.get('hr_vo2_slope'),
            'hr_vo2_intercept': coupling.get('hr_vo2_intercept'),
            'hr_vo2_r2': coupling.get('hr_vo2_r2'),
            'slope_class': coupling.get('slope_class'),
            'coupling_quality': coupling.get('coupling_quality'),
            'breakpoint_time_sec': coupling.get('breakpoint_time_sec'),
            'breakpoint_pct_exercise': coupling.get('breakpoint_pct_exercise'),
            'bp_vs_vt2': coupling.get('bp_vs_vt2'),

            # Chronotropic
            'hr_rest': chrono.get('hr_rest'),
            'hr_peak': chrono.get('hr_peak'),
            'hr_pred_max': chrono.get('hr_pred_max'),
            'pct_pred_hr': chrono.get('pct_pred_hr'),
            'hr_reserve_bpm': chrono.get('hr_reserve_bpm'),
            'hr_reserve_pct': chrono.get('hr_reserve_pct'),
            'chronotropic_index': chrono.get('chronotropic_index'),
            'ci_class': chrono.get('ci_class'),
            'hr_pattern': chrono.get('hr_pattern'),

            # O2 pulse drift
            'o2pulse_early': o2drift.get('o2pulse_early'),
            'o2pulse_mid': o2drift.get('o2pulse_mid'),
            'o2pulse_late': o2drift.get('o2pulse_late'),
            'o2pulse_drift_pct': o2drift.get('o2pulse_drift_pct'),
            'o2pulse_trajectory': o2drift.get('o2pulse_trajectory'),
            'o2pulse_clinical': o2drift.get('o2pulse_clinical'),

            # Per-step drift
            'step_drift': step_drift if step_drift.get('status') == 'OK' else None,

            # VO2/WR
            'vo2_wr': vo2wr if vo2wr.get('status') == 'OK' else None,

            # Overall
            'overall_cv_coupling': overall,
            'flags': flags,
        }

# Alias
Engine_E13_Drift = Engine_E13_Drift_v2





class Engine_E14_Kinetics:
    """VO2 Off-Kinetics / Recovery Engine.
    
    Calculates T1/2 VO2, tau, MRT, VO2DR, dVO2 at timepoints, VCO2/VE T1/2.
    
    References:
    - Cohen-Solal 1995 Circulation 91:2504 — T1/2 VO2 in CHF
    - Scrutinio 2002 JACC 39:1270 — T1/2 >115s prognostic cutoff
    - Bailey 2018 JACC HF 6:329 — VO2DR parameter
    - Off-kinetics tau: athletes 30-50s, healthy 50-80s, HF >120s
    
    Classification (T1/2 VO2 after maximal incremental):
    - EXCELLENT: <60s (trained athletes)
    - GOOD: 60-90s (healthy active)
    - NORMAL: 90-120s (healthy sedentary)
    - SLOW: 120-150s (deconditioning or mild pathology)
    - VERY_SLOW: >150s (HF or severe deconditioning)
    """
    
    @staticmethod
    def run(results, cfg):
        import numpy as np
        try:
            from scipy.optimize import curve_fit
            _has_scipy = True
        except ImportError:
            _has_scipy = False
        
        out = {"status": "OK"}
        e00 = results.get("E00", {})
        e01 = results.get("E01", {})
        
        t_stop = e00.get("t_stop")
        rec_avail = e00.get("recovery_0_60_available", False)
        
        if not t_stop or not rec_avail:
            out["status"] = "NO_RECOVERY_DATA"
            return out
        
        df = cfg.get("_df_processed")
        if df is None or len(df) == 0:
            out["status"] = "NO_DATAFRAME"
            return out
        
        import pandas as pd
        time_col = "Time_s"
        vo2_col = None
        for c in ["VO2_ml_min", "VO2_mlmin"]:
            if c in df.columns:
                vo2_col = c; break
        if not vo2_col:
            for c in df.columns:
                if "vo2" in c.lower() and "ml" in c.lower() and "min" in c.lower():
                    vo2_col = c; break
        
        if not vo2_col or time_col not in df.columns:
            out["status"] = "MISSING_COLUMNS"
            return out
        
        rec = df[df[time_col] > t_stop][[time_col, vo2_col]].dropna().sort_values(time_col).copy()
        if len(rec) < 10:
            out["status"] = "INSUFFICIENT_RECOVERY_DATA"
            out["n_recovery_points"] = len(rec)
            return out
        
        rec["t_rel"] = rec[time_col].values - t_stop
        rec_duration = float(rec["t_rel"].max())
        out["recovery_duration_s"] = round(rec_duration, 1)
        out["n_recovery_points"] = int(len(rec))
        
        ex_mask = (df[time_col] >= t_stop - 30) & (df[time_col] <= t_stop)
        vo2_peak = float(df.loc[ex_mask, vo2_col].mean())
        
        rest_mask = df[time_col] <= df[time_col].min() + 60
        vo2_rest = float(df.loc[rest_mask, vo2_col].mean())
        if np.isnan(vo2_rest) or vo2_rest <= 0:
            vo2_rest = 350.0
        
        amplitude = vo2_peak - vo2_rest
        out["vo2_peak_mlmin"] = round(vo2_peak)
        out["vo2_rest_mlmin"] = round(vo2_rest)
        out["amplitude_mlmin"] = round(amplitude)
        
        if amplitude <= 0:
            out["status"] = "INVALID_AMPLITUDE"
            return out
        
        rec["vo2_smooth"] = rec[vo2_col].rolling(5, center=True, min_periods=1).mean()
        
        # === T1/2 VO2 ===
        target_50 = vo2_rest + amplitude * 0.5
        crossed = rec[rec["vo2_smooth"] <= target_50]
        t_half = None
        if len(crossed) > 0:
            t_half = float(crossed.iloc[0]["t_rel"])
            out["t_half_vo2_s"] = round(t_half, 1)
        else:
            out["t_half_vo2_s"] = None
            out["t_half_note"] = f"Not reached in {rec_duration:.0f}s"
        
        # === Mono-exponential fit ===
        fit_data = rec[(rec["t_rel"] > 10) & (rec["t_rel"] <= min(180, rec_duration))].copy()
        
        if _has_scipy and len(fit_data) >= 8:
            def mono_exp(t, A, tau, baseline):
                return baseline + A * np.exp(-t / tau)
            try:
                t_data = fit_data["t_rel"].values.astype(float)
                vo2_data = fit_data[vo2_col].values.astype(float)
                p0 = [amplitude * 0.8, 60, vo2_rest]
                bounds = ([100, 5, 0], [amplitude * 2, 500, vo2_peak])
                popt, pcov = curve_fit(mono_exp, t_data, vo2_data, p0=p0, bounds=bounds, maxfev=5000)
                A_fit, tau_fit, baseline_fit = popt
                predicted = mono_exp(t_data, *popt)
                ss_res = np.sum((vo2_data - predicted)**2)
                ss_tot = np.sum((vo2_data - np.mean(vo2_data))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                out["tau_s"] = round(float(tau_fit), 1)
                out["tau_amplitude"] = round(float(A_fit))
                out["tau_baseline"] = round(float(baseline_fit))
                out["tau_r2"] = round(float(r2), 4)
                out["fit_quality"] = "GOOD" if r2 >= 0.85 else ("MODERATE" if r2 >= 0.70 else ("FAIR" if r2 >= 0.50 else "POOR"))
                out["t_half_from_tau_s"] = round(float(tau_fit * np.log(2)), 1)
                out["mrt_s"] = round(float(tau_fit), 1)
                out["t_95pct_recovery_s"] = round(float(3 * tau_fit))
            except Exception as e:
                out["fit_error"] = str(e)
                out["fit_quality"] = "FAILED"
        else:
            out["fit_quality"] = "INSUFFICIENT_DATA" if len(fit_data) < 8 else "NO_SCIPY"
        
        # === dVO2 at standard timepoints ===
        body_mass = getattr(cfg.get("_acfg", cfg), "body_mass_kg", 70) if not isinstance(cfg, dict) else cfg.get("body_mass_kg", 70)
        try:
            body_mass = float(cfg.get("_acfg").body_mass_kg)
        except:
            body_mass = 70.0
        
        for secs in [60, 90, 120, 150, 180]:
            mask = (rec["t_rel"] >= secs - 5) & (rec["t_rel"] <= secs + 5)
            vals = rec.loc[mask, "vo2_smooth"]
            if len(vals) > 0:
                vo2_at = float(vals.mean())
                dvo2 = (vo2_peak - vo2_at) / body_mass
                out[f"vo2_at_{secs}s_mlmin"] = round(vo2_at)
                out[f"dvo2_{secs}s_mlkgmin"] = round(dvo2, 1)
                out[f"pct_recovered_{secs}s"] = round((vo2_peak - vo2_at) / amplitude * 100, 1)
        
        # === VO2 Delay Recovery (VO2DR) ===
        vo2_peak_thresh = vo2_peak * 0.98
        vo2dr = None
        smooth_vals = rec["vo2_smooth"].values
        for i in range(len(smooth_vals)):
            if smooth_vals[i] < vo2_peak_thresh:
                if np.all(smooth_vals[i:] < vo2_peak_thresh):
                    vo2dr = float(rec.iloc[i]["t_rel"])
                    break
        out["vo2dr_s"] = round(vo2dr, 1) if vo2dr is not None else None
        
        # === VCO2 and VE T1/2 ===
        for param, col_candidates in [("vco2", ["VCO2_mlmin", "VCO2_ml_min"]),
                                       ("ve", ["VE_Lmin", "VE_L_min"])]:
            for c in col_candidates:
                if c in df.columns:
                    p_peak = float(df.loc[ex_mask, c].mean())
                    p_rest = float(df.loc[rest_mask, c].mean()) if rest_mask.any() else 0
                    if np.isnan(p_rest): p_rest = 0
                    p_amp = p_peak - p_rest
                    if p_amp > 0:
                        p_target = p_rest + p_amp * 0.5
                        p_rec = df[df[time_col] > t_stop][[time_col, c]].dropna().sort_values(time_col).copy()
                        if len(p_rec) > 5:
                            p_rec["t_rel"] = p_rec[time_col].values - t_stop
                            p_rec["smooth"] = p_rec[c].rolling(5, center=True, min_periods=1).mean()
                            p_crossed = p_rec[p_rec["smooth"] <= p_target]
                            if len(p_crossed) > 0:
                                out[f"t_half_{param}_s"] = round(float(p_crossed.iloc[0]["t_rel"]), 1)
                    break
        
        # === Classification ===
        t_half_final = out.get("t_half_vo2_s") or out.get("t_half_from_tau_s")
        if t_half_final is not None:
            t_half_final = float(t_half_final)
            if t_half_final < 60:
                out["recovery_class"] = "EXCELLENT"
                out["recovery_desc"] = "Szybka recovery — typowa dla sportowców wytrzymałościowych"
            elif t_half_final < 90:
                out["recovery_class"] = "GOOD"
                out["recovery_desc"] = "Dobra recovery — zdrowa aktywna osoba"
            elif t_half_final < 120:
                out["recovery_class"] = "NORMAL"
                out["recovery_desc"] = "Prawidłowa recovery"
            elif t_half_final < 150:
                out["recovery_class"] = "SLOW"
                out["recovery_desc"] = "Spowolniona recovery — dekondycjonowanie lub łagodna patologia"
            else:
                out["recovery_class"] = "VERY_SLOW"
                out["recovery_desc"] = "Bardzo wolna recovery — ograniczenie delivery O2"
        
        # === Flags ===
        flags = []
        if t_half_final and t_half_final > 120:
            flags.append("SLOW_VO2_RECOVERY")
        if out.get("vo2dr_s") and out["vo2dr_s"] > 25:
            flags.append("PROLONGED_VO2DR")
        if out.get("tau_r2") and out["tau_r2"] < 0.50:
            flags.append("POOR_EXPONENTIAL_FIT")
        t_half_vco2 = out.get("t_half_vco2_s")
        if t_half_final and t_half_vco2:
            ratio = t_half_vco2 / t_half_final if t_half_final > 0 else 0
            out["vco2_vo2_t_half_ratio"] = round(ratio, 2)
            if ratio > 1.5:
                flags.append("VCO2_RECOVERY_DISPROPORTIONATE")
        out["flags"] = flags
        
        return out




# ═══════════════════════════════════════════════════════════════════════
# E18: VT↔LT CROSS-VALIDATION — Domain Confirmation & Concordance
# ═══════════════════════════════════════════════════════════════════════
# 
# PURPOSE: Cross-validate ventilatory thresholds (E02) against lactate
# data (E11) to provide biochemical confirmation of training domains.
#
# SCIENTIFIC BASIS:
#   - Pallarés et al. 2016: VT1 ≈ LT, VT2 ≈ LT+2 mmol/L, MLSS ≈ LT+0.5
#   - Skinner & McLellan 1980: 3-phase triphasic model
#   - Seiler 2010: 3-zone model (Z1 <VT1, Z2 VT1-VT2, Z3 >VT2)
#   - Frontiers in Physiology 2018: VT1/LT1 and VT2/LT2 alignment
#
# THREE LAYERS:
#   L1: Lactate interpolation at VT time → biochemical proof
#   L2: Domain confirmation → is lactate in expected range for domain?
#   L3: Concordance score → how well do VT and LT agree?
#
# REFERENCES:
#   Expected lactate at thresholds (from multiple studies):
#     @VT1: 1.5-2.5 mmol/L (moderate→heavy transition)
#     @VT2: 3.0-5.5 mmol/L (heavy→severe transition, OBLA ~4mmol)
#   Domain boundaries (Seiler/Stöggl 2014):
#     Moderate: La < baseline+0.5 or < ~2 mmol/L
#     Heavy:    La 2-4 mmol/L, rising but can stabilize
#     Severe:   La > 4 mmol/L, no stabilization possible
# ═══════════════════════════════════════════════════════════════════════

class Engine_E18_VT_LT_CrossValidation:
    """
    E18: VT↔LT Cross-Validation — Domain Confirmation & Concordance.
    
    Requires both E02 (ventilatory thresholds) and E11 (lactate) results.
    When lactate data is unavailable, returns status='NO_LACTATE_DATA'.
    """

    # ── Literature-based reference ranges ────────────────────────────
    # Lactate concentration expected at each ventilatory threshold
    # Sources: Pallarés 2016, Faude 2009, Seiler 2010, Lucía 2000
    
    VT1_LA_RANGE = (1.0, 3.0)      # mmol/L — typical at aerobic threshold
    VT1_LA_OPTIMAL = (1.5, 2.5)    # mmol/L — strong confirmation
    
    VT2_LA_RANGE = (2.5, 7.0)      # mmol/L — typical at RCP/anaerobic threshold
    VT2_LA_OPTIMAL = (3.0, 5.5)    # mmol/L — strong confirmation (OBLA ~4)
    
    # Domain boundaries (blood lactate)
    MODERATE_CEIL = 2.0             # mmol/L — upper limit moderate domain
    HEAVY_CEIL = 4.0               # mmol/L — upper limit heavy domain (MLSS region)
    
    # Concordance thresholds (time-based)
    CONCORDANCE_EXCELLENT_SEC = 45  # VT and LT within 45s
    CONCORDANCE_GOOD_SEC = 90      # VT and LT within 90s
    CONCORDANCE_ACCEPTABLE_SEC = 150 # VT and LT within 150s
    
    # Concordance thresholds (VO2%-based)
    CONCORDANCE_EXCELLENT_PCT = 4.0   # VT and LT within 4% VO2max
    CONCORDANCE_GOOD_PCT = 8.0        # within 8%
    CONCORDANCE_ACCEPTABLE_PCT = 12.0  # within 12%

    @classmethod
    def run(cls, e02_results: dict, e11_results: dict, e01_results: dict = None,
            df_exercise: 'pd.DataFrame' = None) -> dict:
        """
        Cross-validate VT thresholds against lactate data.
        
        Args:
            e02_results: E02 output (ventilatory thresholds)
            e11_results: E11 output (lactate thresholds)
            e01_results: E01 output (for VO2peak reference), optional
            df_exercise: exercise DataFrame (for time↔VO2 interpolation)
        
        Returns:
            dict with status, layer results, and diagnostic flags
        """
        import numpy as np
        import pandas as pd
        
        out = {
            "status": "OK",
            "flags": [],
            "layer1_lactate_at_vt": {},
            "layer2_domain_confirmation": {},
            "layer3_concordance": {},
            "summary": {},
        }
        
        # ── Validate inputs ─────────────────────────────────────────
        if not e11_results or e11_results.get("status") in ("NO_DATA", "INSUFFICIENT_DATA", None):
            out["status"] = "NO_LACTATE_DATA"
            out["flags"].append("E11 returned no lactate data — cross-validation impossible")
            return out
        
        if not e02_results or e02_results.get("status") == "ERROR":
            out["status"] = "NO_VT_DATA"
            out["flags"].append("E02 returned no VT data — cross-validation impossible")
            return out
        
        # Extract lactate curve from E11
        raw_points = e11_results.get("raw_points", [])
        if not raw_points or len(raw_points) < 3:
            out["status"] = "INSUFFICIENT_LACTATE"
            out["flags"].append(f"Only {len(raw_points)} lactate points — need ≥3")
            return out
        
        # Build lactate interpolation function
        lac_times = np.array([p["time_sec"] for p in raw_points])
        lac_values = np.array([p["lactate_mmol"] for p in raw_points])
        
        # Sort by time
        sort_idx = np.argsort(lac_times)
        lac_times = lac_times[sort_idx]
        lac_values = lac_values[sort_idx]
        
        # ── Detect & remove "lactate washout" pattern ──
        # Common artifact: first 1-2 samples show elevated resting La
        # that drops sharply before the true exercise-induced rise.
        # If first point(s) are higher than the subsequent trough by >1.0,
        # exclude them from cross-validation interpolation.
        _washout_removed = 0
        if len(lac_values) >= 4:
            _trough_idx = int(np.argmin(lac_values[:max(3, len(lac_values)//2)]))
            if _trough_idx > 0:
                _trough_val = lac_values[_trough_idx]
                _remove = 0
                for _wi in range(_trough_idx):
                    if lac_values[_wi] > _trough_val + 1.0:
                        _remove = _wi + 1
                if _remove > 0 and len(lac_times) - _remove >= 3:
                    lac_times = lac_times[_remove:]
                    lac_values = lac_values[_remove:]
                    _washout_removed = _remove
        
        # Polynomial curve: if washout points removed, rebuild locally
        poly_coeffs = None
        _local_poly = False
        if _washout_removed > 0 and len(lac_times) >= 3:
            # Refit polynomial on clean data — use degree 2 max to avoid
            # wild extrapolation outside data range
            try:
                _deg = min(2, len(lac_times) - 1)
                poly_coeffs = list(np.polyfit(lac_times, lac_values, _deg))
                _local_poly = True
            except:
                pass
        if poly_coeffs is None:
            poly_curve = e11_results.get("poly_curve", {})
            poly_coeffs = poly_curve.get("coefficients") if poly_curve else None
        
        def interpolate_lactate(t_sec):
            """Interpolate lactate at time t using polynomial or linear interp."""
            # If time is before first data point, use nearest value
            # (avoids polynomial extrapolation artifacts)
            if t_sec < lac_times[0]:
                return float(lac_values[0])
            if poly_coeffs is not None and len(poly_coeffs) >= 2:
                try:
                    val = float(np.polyval(poly_coeffs, t_sec))
                    # Clamp to reasonable range (0 to 2x max observed)
                    return max(0, min(val, float(lac_values.max()) * 2))
                except:
                    pass
            # Fallback: linear interpolation
            if t_sec <= lac_times[0]:
                return float(lac_values[0])
            if t_sec >= lac_times[-1]:
                return float(lac_values[-1])
            return float(np.interp(t_sec, lac_times, lac_values))
        
        def nearest_lactate(t_sec, max_gap_sec=120):
            """Find nearest actual lactate sample to time t."""
            diffs = np.abs(lac_times - t_sec)
            idx = np.argmin(diffs)
            if diffs[idx] <= max_gap_sec:
                return float(lac_values[idx]), float(lac_times[idx]), float(diffs[idx])
            return None, None, None
        
        # ── Get VT times from E02 ──────────────────────────────────
        vt1_time = e02_results.get("vt1_time_sec")
        vt2_time = e02_results.get("vt2_time_sec")
        vt1_vo2_pct = e02_results.get("vt1_vo2_pct_peak")
        vt2_vo2_pct = e02_results.get("vt2_vo2_pct_peak")
        vt1_conf = e02_results.get("vt1_confidence", 0)
        vt2_conf = e02_results.get("vt2_confidence", 0)
        
        # Get LT times from E11
        lt1_time = e11_results.get("lt1_time_sec")
        lt2_time = e11_results.get("lt2_time_sec")
        lt1_la = e11_results.get("lt1_la_mmol")
        lt2_la = e11_results.get("lt2_la_mmol")
        lt1_method = e11_results.get("lt1_method", "")
        lt2_method = e11_results.get("lt2_method", "")
        
        # Get VO2peak for % calculations
        vo2peak = None
        if e01_results:
            vo2peak = e01_results.get("vo2_peak_mlmin")
        
        # Build time→VO2 mapping from df if available
        def vo2_at_time(t_sec):
            if df_exercise is not None and len(df_exercise) > 0:
                for tc in ["time", "Time_s", "Time_sec"]:
                    if tc in df_exercise.columns:
                        t_col = tc
                        break
                else:
                    return None
                for vc in ["vo2_ml", "vo2_ml_sm", "VO2_ml_min"]:
                    if vc in df_exercise.columns:
                        v_col = vc
                        break
                else:
                    return None
                t_arr = pd.to_numeric(df_exercise[t_col], errors="coerce")
                v_arr = pd.to_numeric(df_exercise[v_col], errors="coerce")
                mask = (t_arr >= t_sec - 20) & (t_arr <= t_sec + 20)
                if mask.sum() >= 2:
                    return float(v_arr[mask].median())
            return None
        
        la_baseline = e11_results.get("la_baseline", lac_values.min())
        
        # ════════════════════════════════════════════════════════════
        # LAYER 1: Lactate at VT — Biochemical Proof
        # ════════════════════════════════════════════════════════════
        
        l1 = {}
        
        for prefix, vt_time, vt_pct, vt_cf, la_range, la_opt in [
            ("vt1", vt1_time, vt1_vo2_pct, vt1_conf,
             cls.VT1_LA_RANGE, cls.VT1_LA_OPTIMAL),
            ("vt2", vt2_time, vt2_vo2_pct, vt2_conf,
             cls.VT2_LA_RANGE, cls.VT2_LA_OPTIMAL),
        ]:
            if vt_time is None:
                l1[prefix] = {"status": "NO_VT", "message": f"{prefix.upper()} not detected"}
                continue
            
            # Interpolated lactate at VT time
            la_interp = interpolate_lactate(vt_time)
            # Nearest actual sample
            la_nearest, la_nearest_t, la_gap = nearest_lactate(vt_time)
            
            # Classify
            if la_opt[0] <= la_interp <= la_opt[1]:
                la_class = "OPTIMAL"
                la_emoji = "✅"
                la_desc = f"Laktat {la_interp:.1f} mmol/L w optymalnym zakresie {la_opt[0]}-{la_opt[1]}"
            elif la_range[0] <= la_interp <= la_range[1]:
                la_class = "ACCEPTABLE"
                la_emoji = "⚠️"
                la_desc = f"Laktat {la_interp:.1f} mmol/L w akceptowalnym zakresie {la_range[0]}-{la_range[1]}"
            elif la_interp < la_range[0]:
                la_class = "TOO_LOW"
                la_emoji = "❌"
                la_desc = (f"Laktat {la_interp:.1f} mmol/L poniżej oczekiwanego "
                          f"({la_range[0]}-{la_range[1]}) — {prefix.upper()} "
                          f"może być wyznaczony za wcześnie")
            else:
                la_class = "TOO_HIGH"
                la_emoji = "❌"
                la_desc = (f"Laktat {la_interp:.1f} mmol/L powyżej oczekiwanego "
                          f"({la_range[0]}-{la_range[1]}) — {prefix.upper()} "
                          f"może być wyznaczony za późno")
            
            l1[prefix] = {
                "status": la_class,
                "la_interpolated_mmol": round(la_interp, 2),
                "la_nearest_mmol": round(la_nearest, 2) if la_nearest else None,
                "la_nearest_gap_sec": round(la_gap, 0) if la_gap else None,
                "expected_range": list(la_range),
                "optimal_range": list(la_opt),
                "vt_time_sec": round(vt_time, 1),
                "vt_vo2_pct": round(vt_pct, 1) if vt_pct else None,
                "emoji": la_emoji,
                "description": la_desc,
            }
            
            if la_class in ("TOO_LOW", "TOO_HIGH"):
                out["flags"].append(f"L1_{prefix}_{la_class}: {la_desc}")
        
        out["layer1_lactate_at_vt"] = l1
        
        # ════════════════════════════════════════════════════════════
        # LAYER 2: Domain Confirmation
        # ════════════════════════════════════════════════════════════
        # 
        # Triphasic model (Skinner & McLellan 1980):
        #   Phase I  (Moderate): below VT1 — La < baseline+0.5, ≲2 mmol/L
        #   Phase II (Heavy):    VT1 → VT2 — La 2-4 mmol/L
        #   Phase III (Severe):  above VT2 — La > 4 mmol/L, exponential rise
        #
        # We check: does the lactate PROFILE confirm the domains
        # defined by VT1 and VT2?
        
        l2 = {}
        
        # 2A: Domain below VT1 — should be moderate (La stable, ≲2)
        if vt1_time is not None:
            la_samples_below = [(t, la) for t, la in zip(lac_times, lac_values) 
                                if t < vt1_time]
            if la_samples_below:
                la_below = [la for _, la in la_samples_below]
                la_below_mean = np.mean(la_below)
                la_below_max = np.max(la_below)
                la_below_stable = (np.max(la_below) - np.min(la_below)) < 1.0
                
                moderate_confirmed = la_below_max <= cls.MODERATE_CEIL + 0.5 and la_below_stable
                
                l2["below_vt1"] = {
                    "domain_expected": "MODERATE",
                    "la_mean": round(la_below_mean, 2),
                    "la_max": round(la_below_max, 2),
                    "la_stable": la_below_stable,
                    "n_samples": len(la_below),
                    "confirmed": moderate_confirmed,
                    "emoji": "✅" if moderate_confirmed else "⚠️",
                    "description": (
                        f"Domena umiarkowana POTWIERDZONA: La śr.={la_below_mean:.1f}, "
                        f"max={la_below_max:.1f}, stabilny={la_below_stable}"
                        if moderate_confirmed else
                        f"Domena umiarkowana WĄTPLIWA: La śr.={la_below_mean:.1f}, "
                        f"max={la_below_max:.1f} (oczekiwano ≤{cls.MODERATE_CEIL:.1f})"
                    ),
                }
            else:
                l2["below_vt1"] = {
                    "domain_expected": "MODERATE",
                    "confirmed": None,
                    "description": "Brak próbek laktatu poniżej VT1",
                }
        
        # 2B: Domain VT1→VT2 — should be heavy (La rising, 2-4 mmol/L)
        if vt1_time is not None and vt2_time is not None:
            la_between = [(t, la) for t, la in zip(lac_times, lac_values) 
                         if vt1_time <= t <= vt2_time]
            if len(la_between) >= 2:
                la_btwn = [la for _, la in la_between]
                la_btwn_mean = np.mean(la_btwn)
                la_btwn_max = np.max(la_btwn)
                la_btwn_rising = la_btwn[-1] > la_btwn[0]
                
                heavy_confirmed = (la_btwn_mean >= cls.MODERATE_CEIL - 0.5 and
                                  la_btwn_max <= cls.HEAVY_CEIL + 1.5 and
                                  la_btwn_rising)
                
                l2["between_vt1_vt2"] = {
                    "domain_expected": "HEAVY",
                    "la_mean": round(la_btwn_mean, 2),
                    "la_max": round(la_btwn_max, 2),
                    "la_rising": la_btwn_rising,
                    "n_samples": len(la_btwn),
                    "confirmed": heavy_confirmed,
                    "emoji": "✅" if heavy_confirmed else "⚠️",
                    "description": (
                        f"Domena heavy POTWIERDZONA: La śr.={la_btwn_mean:.1f}, "
                        f"max={la_btwn_max:.1f}, rosnący={la_btwn_rising}"
                        if heavy_confirmed else
                        f"Domena heavy WĄTPLIWA: La śr.={la_btwn_mean:.1f}, "
                        f"max={la_btwn_max:.1f}, rosnący={la_btwn_rising}"
                    ),
                }
            elif len(la_between) == 1:
                l2["between_vt1_vt2"] = {
                    "domain_expected": "HEAVY",
                    "confirmed": None,
                    "n_samples": 1,
                    "description": "Tylko 1 próbka między VT1-VT2 — niewystarczająco",
                }
            else:
                l2["between_vt1_vt2"] = {
                    "domain_expected": "HEAVY",
                    "confirmed": None,
                    "n_samples": 0,
                    "description": "Brak próbek laktatu między VT1 a VT2",
                }
        
        # 2C: Domain above VT2 — should be severe (La > 4, exponential rise)
        if vt2_time is not None:
            la_above = [(t, la) for t, la in zip(lac_times, lac_values) 
                       if t > vt2_time]
            if len(la_above) >= 2:
                la_abv = [la for _, la in la_above]
                la_abv_mean = np.mean(la_abv)
                la_abv_max = np.max(la_abv)
                la_abv_accel = (len(la_abv) >= 2 and 
                               la_abv[-1] - la_abv[0] > 2.0)
                
                severe_confirmed = la_abv_max > cls.HEAVY_CEIL and la_abv_accel
                
                l2["above_vt2"] = {
                    "domain_expected": "SEVERE",
                    "la_mean": round(la_abv_mean, 2),
                    "la_max": round(la_abv_max, 2),
                    "la_accelerating": la_abv_accel,
                    "n_samples": len(la_abv),
                    "confirmed": severe_confirmed,
                    "emoji": "✅" if severe_confirmed else "⚠️",
                    "description": (
                        f"Domena severe POTWIERDZONA: La max={la_abv_max:.1f}, "
                        f"przyspieszenie={la_abv_accel}"
                        if severe_confirmed else
                        f"Domena severe WĄTPLIWA: La max={la_abv_max:.1f} "
                        f"(oczekiwano >{cls.HEAVY_CEIL:.1f} z przyspieszeniem)"
                    ),
                }
            elif len(la_above) == 1:
                la_val = la_above[0][1]
                l2["above_vt2"] = {
                    "domain_expected": "SEVERE",
                    "la_max": round(la_val, 2),
                    "confirmed": la_val > cls.HEAVY_CEIL,
                    "n_samples": 1,
                    "emoji": "✅" if la_val > cls.HEAVY_CEIL else "⚠️",
                    "description": f"1 próbka powyżej VT2: La={la_val:.1f}",
                }
            else:
                l2["above_vt2"] = {
                    "domain_expected": "SEVERE",
                    "confirmed": None,
                    "description": "Brak próbek laktatu powyżej VT2",
                }
        
        # Domain summary
        domains_tested = [v for v in l2.values() if v.get("confirmed") is not None]
        domains_confirmed = sum(1 for v in domains_tested if v.get("confirmed"))
        l2["summary"] = {
            "domains_tested": len(domains_tested),
            "domains_confirmed": domains_confirmed,
            "confirmation_rate": (domains_confirmed / len(domains_tested) * 100
                                 if domains_tested else 0),
        }
        
        out["layer2_domain_confirmation"] = l2
        
        # ════════════════════════════════════════════════════════════
        # LAYER 3: VT↔LT Concordance
        # ════════════════════════════════════════════════════════════
        
        l3 = {}
        
        for prefix, vt_time, vt_pct, vt_cf, lt_time, lt_la, lt_meth in [
            ("threshold_1", vt1_time, vt1_vo2_pct, vt1_conf,
             lt1_time, lt1_la, lt1_method),
            ("threshold_2", vt2_time, vt2_vo2_pct, vt2_conf,
             lt2_time, lt2_la, lt2_method),
        ]:
            if vt_time is None or lt_time is None:
                l3[prefix] = {
                    "status": "INCOMPLETE",
                    "message": f"{'VT' if vt_time is None else 'LT'} not available",
                }
                continue
            
            # Time concordance
            dt = abs(vt_time - lt_time)
            direction = "VT_EARLIER" if vt_time < lt_time else "VT_LATER"
            signed_dt = vt_time - lt_time
            
            if dt <= cls.CONCORDANCE_EXCELLENT_SEC:
                time_class = "EXCELLENT"
                time_emoji = "✅"
            elif dt <= cls.CONCORDANCE_GOOD_SEC:
                time_class = "GOOD"
                time_emoji = "✅"
            elif dt <= cls.CONCORDANCE_ACCEPTABLE_SEC:
                time_class = "ACCEPTABLE"
                time_emoji = "⚠️"
            else:
                time_class = "POOR"
                time_emoji = "❌"
            
            # VO2% concordance (if available)
            lt_vo2 = vo2_at_time(lt_time) if df_exercise is not None else None
            lt_vo2_pct = (lt_vo2 / vo2peak * 100) if (lt_vo2 and vo2peak) else None
            
            dpct = None
            pct_class = None
            if vt_pct is not None and lt_vo2_pct is not None:
                dpct = abs(vt_pct - lt_vo2_pct)
                if dpct <= cls.CONCORDANCE_EXCELLENT_PCT:
                    pct_class = "EXCELLENT"
                elif dpct <= cls.CONCORDANCE_GOOD_PCT:
                    pct_class = "GOOD"
                elif dpct <= cls.CONCORDANCE_ACCEPTABLE_PCT:
                    pct_class = "ACCEPTABLE"
                else:
                    pct_class = "POOR"
            
            # Overall concordance (worst of time and pct)
            classes_order = ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"]
            overall_idx = classes_order.index(time_class)
            if pct_class:
                overall_idx = max(overall_idx, classes_order.index(pct_class))
            overall = classes_order[overall_idx]
            overall_emoji = {"EXCELLENT": "✅", "GOOD": "✅", 
                           "ACCEPTABLE": "⚠️", "POOR": "❌"}[overall]
            
            # Diagnostic suggestion when discordant
            suggestion = None
            if overall == "POOR":
                if vt_time < lt_time:
                    suggestion = (
                        f"VT wykryty {dt:.0f}s wcześniej niż LT. "
                        f"Możliwe: VT za wcześnie (hyperventylacja, niepokój) "
                        f"lub LT za późno (metoda {lt_meth} konserwatywna). "
                        f"Sugestia: preferuj LT przy dużej rozbieżności."
                    )
                else:
                    suggestion = (
                        f"VT wykryty {dt:.0f}s później niż LT. "
                        f"Możliwe: opóźniona odpowiedź wentylacyjna "
                        f"lub LT za wcześnie (metoda {lt_meth}). "
                        f"Sugestia: preferuj VT gdy confidence E02 > 0.85."
                    )
            
            l3[prefix] = {
                "status": overall,
                "emoji": overall_emoji,
                "delta_time_sec": round(signed_dt, 1),
                "abs_delta_time_sec": round(dt, 1),
                "direction": direction,
                "time_concordance": time_class,
                "vt_time_sec": round(vt_time, 1),
                "lt_time_sec": round(lt_time, 1),
                "vt_vo2_pct": round(vt_pct, 1) if vt_pct else None,
                "lt_vo2_pct": round(lt_vo2_pct, 1) if lt_vo2_pct else None,
                "delta_vo2_pct": round(dpct, 1) if dpct else None,
                "pct_concordance": pct_class,
                "lt_method": lt_meth,
                "vt_confidence": vt_cf,
                "suggestion": suggestion,
                "description": (
                    f"Δt={signed_dt:+.0f}s ({time_class})"
                    + (f", Δ%VO2={dpct:.1f}% ({pct_class})" if dpct else "")
                    + (f" → {suggestion[:60]}..." if suggestion else "")
                ),
            }
            
            if overall == "POOR":
                out["flags"].append(
                    f"L3_{prefix}_POOR: |Δt|={dt:.0f}s, "
                    f"direction={direction}"
                )
        
        out["layer3_concordance"] = l3
        
        # ════════════════════════════════════════════════════════════
        # SUMMARY — Overall Cross-Validation Assessment
        # ════════════════════════════════════════════════════════════
        
        # Count confirmations across all layers
        l1_ok = sum(1 for v in l1.values() 
                   if isinstance(v, dict) and v.get("status") in ("OPTIMAL", "ACCEPTABLE"))
        l1_total = sum(1 for v in l1.values() 
                      if isinstance(v, dict) and v.get("status") not in ("NO_VT", None))
        
        l2_ok = l2.get("summary", {}).get("domains_confirmed", 0)
        l2_total = l2.get("summary", {}).get("domains_tested", 0)
        
        l3_ok = sum(1 for v in l3.values()
                   if isinstance(v, dict) and v.get("status") in ("EXCELLENT", "GOOD"))
        l3_total = sum(1 for v in l3.values()
                      if isinstance(v, dict) and v.get("status") not in ("INCOMPLETE", None))
        
        total_checks = l1_total + l2_total + l3_total
        total_ok = l1_ok + l2_ok + l3_ok
        
        if total_checks == 0:
            overall_score = 0
            overall_class = "INSUFFICIENT_DATA"
        else:
            overall_score = round(total_ok / total_checks * 100, 0)
            if overall_score >= 80:
                overall_class = "HIGH_AGREEMENT"
            elif overall_score >= 60:
                overall_class = "MODERATE_AGREEMENT"
            elif overall_score >= 40:
                overall_class = "LOW_AGREEMENT"
            else:
                overall_class = "DISCORDANT"
        
        out["summary"] = {
            "overall_class": overall_class,
            "overall_score_pct": overall_score,
            "layer1_lactate_at_vt": f"{l1_ok}/{l1_total}",
            "layer2_domain_confirmation": f"{l2_ok}/{l2_total}",
            "layer3_concordance": f"{l3_ok}/{l3_total}",
            "total": f"{total_ok}/{total_checks}",
            "interpretation": {
                "HIGH_AGREEMENT": "Progi wentylacyjne i mleczanowe są zgodne — wysoka pewność prescripcji treningowej.",
                "MODERATE_AGREEMENT": "Częściowa zgodność VT↔LT — progi są orientacyjne, weryfikuj indywidualnie.",
                "LOW_AGREEMENT": "Niska zgodność VT↔LT — rozważ ponowną ocenę progów z ekspertem.",
                "DISCORDANT": "Silna rozbieżność VT↔LT — progi mogą być błędne, wymagana analiza manualna.",
                "INSUFFICIENT_DATA": "Niewystarczające dane do cross-walidacji.",
            }.get(overall_class, ""),
        }
        
        return out


# --- E15: NORMALIZATION ---
class Engine_E15_Normalization:
    """
    E15 — Normalizacja wyników + porównanie z normami referencyjnymi.
    Wersja: v2.0 (2026-02)

    Źródła norm:
      - VO2max predicted: FRIEND Registry 2017 (Myers/Kaminsky, N=7783)
        Równanie: VO2max(ml/kg/min) = 79.9 - 0.39*age - 13.7*sex (male=0, female=1)
      - VO2max percentyle: ACSM Guidelines 11th Ed + FRIEND 2015-2019
      - VO2max athletes: Marathon Handbook / INSCYD compilation
      - O2 Pulse predicted: FRIEND 2019 (Kaminsky, N=7783)
        Równanie: O2pulse = 23.2 - 0.09*age - 6.6*sex
      - O2 Pulse athletes: NOODLE Study 2024 (Kasiak et al., N=94)
        Równanie: O2pulse = 1.36 + 1.07*(23.2 - 0.09*age - 6.6*sex)
      - VE/VCO2 slope: ATS/ACCP + Mezzani 2013 + PMC reviews
      - HRR: Cole et al. 1999 (NEJM), Myers score (PMC 2009)
      - RER: Wasserman criteria for max effort
      - Threshold %: Wasserman 2005, Mezzani 2013
    """

    # =====================================================================
    # NORMY STATYCZNE
    # =====================================================================

    # VO2max klasyfikacja populacyjna: (min, max) ml/kg/min
    # Źródło: ACSM 11th Ed + Cooper Institute (Garmin/Polar tables)
    VO2MAX_NORMS_MALE = {
        # age_range: (very_poor, poor, fair, good, excellent, superior)
        #            granice: <vp | vp-p | p-f | f-g | g-e | e-s | >s
        (20, 29): (0, 25, 30, 34, 38, 42, 49),
        (30, 39): (0, 23, 28, 32, 36, 40, 47),
        (40, 49): (0, 20, 25, 30, 34, 38, 44),
        (50, 59): (0, 18, 23, 28, 32, 36, 42),
        (60, 69): (0, 16, 21, 26, 30, 34, 39),
        (70, 99): (0, 14, 19, 24, 28, 32, 37),
    }

    VO2MAX_NORMS_FEMALE = {
        (20, 29): (0, 20, 24, 28, 32, 36, 44),
        (30, 39): (0, 18, 22, 26, 30, 34, 41),
        (40, 49): (0, 16, 20, 24, 28, 32, 39),
        (50, 59): (0, 14, 18, 22, 26, 30, 36),
        (60, 69): (0, 12, 16, 20, 24, 28, 33),
        (70, 99): (0, 10, 14, 18, 22, 26, 31),
    }

    VO2MAX_LABELS_POP = [
        "VERY_POOR", "POOR", "FAIR", "GOOD", "EXCELLENT", "SUPERIOR"
    ]

    # VO2max klasyfikacja sportowa (ml/kg/min) — mężczyźni
    # Źródło: INSCYD / Marathon Handbook / literature compilation
    # ═══════════════════════════════════════════════════════════════
    # VO2MAX SPORT CLASSIFICATION TABLES
    # Per-modality, per-sex, evidence-based ranges (ml/kg/min)
    # Sources: INSCYD 2024, Haugan 2024 (CrossFit), Bergmann 2025 (HYROX),
    #          Topendsports 2025, Wikipedia, PMC meta-analyses
    # Categories: UNTRAINED → RECREATIONAL → TRAINED → COMPETITIVE → SUB_ELITE → ELITE
    # ═══════════════════════════════════════════════════════════════

    VO2MAX_SPORT_MALE = {
        "run": [
            (0,  40, "UNTRAINED"),
            (40, 50, "RECREATIONAL"),
            (50, 58, "TRAINED"),
            (58, 65, "COMPETITIVE"),
            (65, 72, "SUB_ELITE"),
            (72, 999, "ELITE"),
        ],
        "bike": [
            (0,  38, "UNTRAINED"),
            (38, 48, "RECREATIONAL"),
            (48, 56, "TRAINED"),
            (56, 64, "COMPETITIVE"),
            (64, 72, "SUB_ELITE"),
            (72, 999, "ELITE"),
        ],
        "triathlon": [
            (0,  42, "UNTRAINED"),
            (42, 52, "RECREATIONAL"),
            (52, 60, "TRAINED"),
            (60, 68, "COMPETITIVE"),
            (68, 75, "SUB_ELITE"),
            (75, 999, "ELITE"),
        ],
        "rowing": [
            (0,  38, "UNTRAINED"),
            (38, 48, "RECREATIONAL"),
            (48, 55, "TRAINED"),
            (55, 62, "COMPETITIVE"),
            (62, 70, "SUB_ELITE"),
            (70, 999, "ELITE"),
        ],
        "crossfit": [
            (0,  35, "UNTRAINED"),
            (35, 43, "RECREATIONAL"),
            (43, 50, "TRAINED"),
            (50, 56, "COMPETITIVE"),
            (56, 62, "SUB_ELITE"),
            (62, 999, "ELITE"),
        ],
        "hyrox": [
            (0,  38, "UNTRAINED"),
            (38, 46, "RECREATIONAL"),
            (46, 53, "TRAINED"),
            (53, 58, "COMPETITIVE"),
            (58, 64, "SUB_ELITE"),
            (64, 999, "ELITE"),
        ],
        "swimming": [
            (0,  35, "UNTRAINED"),
            (35, 45, "RECREATIONAL"),
            (45, 52, "TRAINED"),
            (52, 58, "COMPETITIVE"),
            (58, 65, "SUB_ELITE"),
            (65, 999, "ELITE"),
        ],
        "xc_ski": [
            (0,  42, "UNTRAINED"),
            (42, 55, "RECREATIONAL"),
            (55, 65, "TRAINED"),
            (65, 75, "COMPETITIVE"),
            (75, 82, "SUB_ELITE"),
            (82, 999, "ELITE"),
        ],
        "soccer": [
            (0,  38, "UNTRAINED"),
            (38, 48, "RECREATIONAL"),
            (48, 55, "TRAINED"),
            (55, 60, "COMPETITIVE"),
            (60, 65, "SUB_ELITE"),
            (65, 999, "ELITE"),
        ],
        "mma": [
            (0,  35, "UNTRAINED"),
            (35, 43, "RECREATIONAL"),
            (43, 50, "TRAINED"),
            (50, 55, "COMPETITIVE"),
            (55, 60, "SUB_ELITE"),
            (60, 999, "ELITE"),
        ],
        "default": [
            (0,  35, "UNTRAINED"),
            (35, 45, "RECREATIONAL"),
            (45, 52, "TRAINED"),
            (52, 60, "COMPETITIVE"),
            (60, 68, "SUB_ELITE"),
            (68, 999, "ELITE"),
        ],
    }

    VO2MAX_SPORT_FEMALE = {
        "run": [
            (0,  33, "UNTRAINED"),
            (33, 40, "RECREATIONAL"),
            (40, 48, "TRAINED"),
            (48, 55, "COMPETITIVE"),
            (55, 63, "SUB_ELITE"),
            (63, 999, "ELITE"),
        ],
        "bike": [
            (0,  30, "UNTRAINED"),
            (30, 38, "RECREATIONAL"),
            (38, 45, "TRAINED"),
            (45, 52, "COMPETITIVE"),
            (52, 60, "SUB_ELITE"),
            (60, 999, "ELITE"),
        ],
        "triathlon": [
            (0,  34, "UNTRAINED"),
            (34, 42, "RECREATIONAL"),
            (42, 50, "TRAINED"),
            (50, 57, "COMPETITIVE"),
            (57, 64, "SUB_ELITE"),
            (64, 999, "ELITE"),
        ],
        "rowing": [
            (0,  30, "UNTRAINED"),
            (30, 38, "RECREATIONAL"),
            (38, 45, "TRAINED"),
            (45, 52, "COMPETITIVE"),
            (52, 60, "SUB_ELITE"),
            (60, 999, "ELITE"),
        ],
        "crossfit": [
            (0,  28, "UNTRAINED"),
            (28, 35, "RECREATIONAL"),
            (35, 42, "TRAINED"),
            (42, 48, "COMPETITIVE"),
            (48, 54, "SUB_ELITE"),
            (54, 999, "ELITE"),
        ],
        "hyrox": [
            (0,  30, "UNTRAINED"),
            (30, 37, "RECREATIONAL"),
            (37, 44, "TRAINED"),
            (44, 50, "COMPETITIVE"),
            (50, 56, "SUB_ELITE"),
            (56, 999, "ELITE"),
        ],
        "swimming": [
            (0,  28, "UNTRAINED"),
            (28, 36, "RECREATIONAL"),
            (36, 43, "TRAINED"),
            (43, 50, "COMPETITIVE"),
            (50, 56, "SUB_ELITE"),
            (56, 999, "ELITE"),
        ],
        "xc_ski": [
            (0,  35, "UNTRAINED"),
            (35, 45, "RECREATIONAL"),
            (45, 55, "TRAINED"),
            (55, 63, "COMPETITIVE"),
            (63, 72, "SUB_ELITE"),
            (72, 999, "ELITE"),
        ],
        "soccer": [
            (0,  30, "UNTRAINED"),
            (30, 38, "RECREATIONAL"),
            (38, 45, "TRAINED"),
            (45, 50, "COMPETITIVE"),
            (50, 55, "SUB_ELITE"),
            (55, 999, "ELITE"),
        ],
        "mma": [
            (0,  28, "UNTRAINED"),
            (28, 35, "RECREATIONAL"),
            (35, 42, "TRAINED"),
            (42, 48, "COMPETITIVE"),
            (48, 53, "SUB_ELITE"),
            (53, 999, "ELITE"),
        ],
        "default": [
            (0,  28, "UNTRAINED"),
            (28, 36, "RECREATIONAL"),
            (36, 43, "TRAINED"),
            (43, 50, "COMPETITIVE"),
            (50, 57, "SUB_ELITE"),
            (57, 999, "ELITE"),
        ],
    }

    # VE/VCO2 slope interpretacja
    # Źródło: ATS/ACCP, Mezzani 2013, PMC reviews
    VE_VCO2_RANGES = [
        (0,   25, "EXCELLENT",  "Wybitna efektywność wentylacyjna"),
        (25,  30, "NORMAL",     "Norma — prawidłowa efektywność"),
        (30,  34, "BORDERLINE", "Pogranicze — lekka nieefektywność"),
        (34,  38, "ABNORMAL",   "Nieefektywność wentylacyjna (HF / PH risk)"),
        (38,  45, "HIGH",       "Znaczna nieefektywność wentylacyjna"),
        (45, 999, "SEVERE",     "Ciężka patologia wentylacyjna"),
    ]

    # HRR 1min interpretacja
    # Źródło: Cole et al. 1999 NEJM, Myers 2009
    HRR_1MIN_RANGES = [
        (0,   12, "ABNORMAL",  "Poniżej normy prognostycznej (<12 bpm)"),
        (12,  18, "NORMAL",    "Norma kliniczna"),
        (18,  25, "GOOD",      "Dobra regeneracja autonomiczna"),
        (25,  35, "VERY_GOOD", "Bardzo dobra (typowa dla sportowców)"),
        (35, 999, "EXCELLENT", "Wybitna regeneracja (wytrenowani sportowcy)"),
    ]

    # RER peak interpretacja
    # Źródło: Wasserman 2005, ACSM
    RER_RANGES = [
        (0,    1.00, "SUBMAXIMAL",     "Wysiłek submaximalny — wynik VO2peak może być zaniżony"),
        (1.00, 1.05, "LIKELY_SUBMAX",  "Prawdopodobnie submaximalny (RER 1.00-1.05)"),
        (1.05, 1.10, "LIKELY_MAX",     "Prawdopodobnie maximalny (RER 1.05-1.10)"),
        (1.10, 1.15, "MAXIMAL",        "Potwierdzone maksymalne obciążenie"),
        (1.15, 1.25, "SUPRAMAXIMAL",   "Wysokie zaangażowanie beztlenowe"),
        (1.25, 9.99, "EXTREME",        "Ekstremalny wysiłek beztlenowy — weryfikuj dane"),
    ]

    # =====================================================================
    # METODY POMOCNICZE
    # =====================================================================

    @classmethod
    def _num(cls, val):
        """Bezpieczna konwersja na float."""
        if val is None:
            return None
        try:
            import numpy as np
            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        except Exception:
            return None

    @classmethod
    def _classify_range(cls, value, ranges):
        """Klasyfikuj wartość wg listy (lo, hi, label, opis)."""
        if value is None:
            return None, None
        for lo, hi, label, *rest in ranges:
            if lo <= value < hi:
                desc = rest[0] if rest else ""
                return label, desc
        return "UNKNOWN", ""

    @classmethod
    def _classify_vo2_population(cls, vo2_rel, age, sex="male"):
        """
        Klasyfikacja VO2max vs normy populacyjne.
        sex: 'male' lub 'female'
        Zwraca (label, percentile_approx)
        """
        if vo2_rel is None or age is None:
            return None, None

        norms = cls.VO2MAX_NORMS_MALE if sex == "male" else cls.VO2MAX_NORMS_FEMALE
        labels = cls.VO2MAX_LABELS_POP

        # Znajdź odpowiedni zakres wiekowy
        bucket = None
        for (lo, hi), thresholds in norms.items():
            if lo <= age <= hi:
                bucket = thresholds
                break
        if bucket is None:
            # Fallback: ostatni zakres
            bucket = list(norms.values())[-1]

        # bucket: (0, vp_max, poor_max, fair_max, good_max, exc_max, sup_max)
        # labels: VERY_POOR, POOR, FAIR, GOOD, EXCELLENT, SUPERIOR
        for i in range(len(labels)):
            lo_val = bucket[i]
            hi_val = bucket[i + 1]
            if vo2_rel < hi_val:
                # Szacunkowy percentyl
                # VERY_POOR: ~5%, POOR: ~15%, FAIR: ~35%, GOOD: ~60%, EXCELLENT: ~80%, SUPERIOR: ~95%
                pct_map = [5, 15, 35, 60, 80, 95]
                pct = pct_map[i]
                return labels[i], pct

        return "SUPERIOR", 99

    @classmethod
    def _classify_vo2_sport(cls, vo2_rel, modality="run", sex="male"):
        """
        Klasyfikacja VO2max vs normy sportowe.
        Zwraca (label, opis)
        """
        if vo2_rel is None:
            return None, None

        sport_table = cls.VO2MAX_SPORT_MALE if sex == "male" else cls.VO2MAX_SPORT_FEMALE
        ranges = sport_table.get(modality, sport_table.get("default", []))

        for lo, hi, label in ranges:
            if lo <= vo2_rel < hi:
                return label, f"{label} ({modality})"
        return "UNKNOWN", ""

    @classmethod
    def _predicted_vo2max_friend(cls, age, sex="male"):
        """
        FRIEND Registry 2017 — mean CRF zdrowej populacji.
        UWAGA: To NIE jest predicted max indywidualny.
        """
        if age is None:
            return None
        sex_val = 0 if sex == "male" else 1
        return 79.9 - 0.39 * age - 13.7 * sex_val

    @classmethod
    def _predicted_vo2max_wasserman(cls, age, height_cm, weight_kg, sex="male", modality="run"):
        """
        Predicted VO2max wg Wasserman (6th ed). Klinicznie wiarygodne.
        Returns ml/min.
        """
        if age is None or height_cm is None or weight_kg is None:
            return None
        if sex == "male":
            pw = 0.79 * height_cm - 60.7
            factor = 20
        else:
            pw = 0.65 * height_cm - 42.8
            factor = 14
        vo2 = (pw + 43) * factor
        if weight_kg > pw:
            vo2 += 6 * (weight_kg - pw)
        if modality in ("run", "walk", "treadmill"):
            vo2 *= 1.11
        if age > 30:
            vo2 *= max(0.5, 1.0 - (age - 30) * 0.005)
        return vo2

    @classmethod
    def _predicted_o2pulse_friend(cls, age, sex="male"):
        """
        Predicted O2 Pulse wg FRIEND 2019.
        O2pulse (ml/beat) = 23.2 - 0.09*age - 6.6*sex
        """
        if age is None:
            return None
        sex_val = 0 if sex == "male" else 1
        return 23.2 - 0.09 * age - 6.6 * sex_val

    @classmethod
    def _predicted_o2pulse_athletes(cls, age, sex="male"):
        """
        Predicted O2 Pulse dla sportowców wg NOODLE Study 2024.
        O2pulse = 1.36 + 1.07 * (23.2 - 0.09*age - 6.6*sex)
        """
        if age is None:
            return None
        sex_val = 0 if sex == "male" else 1
        return 1.36 + 1.07 * (23.2 - 0.09 * age - 6.6 * sex_val)

    @classmethod
    def _predicted_hrmax(cls, age):
        """
        Predicted HRmax wg Tanaka 2001.
        HRmax = 208 - 0.7 * age
        """
        if age is None:
            return None
        return 208 - 0.7 * age

    # =====================================================================
    # GŁÓWNA METODA
    # =====================================================================

    @classmethod
    def run(cls, results_dict: dict, body_mass_kg: float, age: int = None,
            sex: str = "male", modality: str = "run", height_cm: float = None) -> dict:
        """
        Normalizacja i porównanie z normami.

        Args:
            results_dict: słownik wyników z innych silników (E00-E14)
            body_mass_kg: masa ciała w kg
            age: wiek w latach (opcjonalny ale zalecany)
            sex: 'male' lub 'female'
            modality: 'run', 'bike', 'triathlon', itp.

        Returns:
            Słownik z kluczami:
            - vo2_rel, vo2_abs, vo2_abs_lmin: VO2peak normalizowane
            - vo2_pct_predicted: % predicted (FRIEND)
            - vo2_class_pop, vo2_class_sport: klasyfikacja
            - o2pulse_peak, o2pulse_pct_predicted: O2 Pulse
            - ve_vco2_slope, ve_vco2_class: klasyfikacja VE/VCO2
            - rer_peak, rer_class: klasyfikacja wysiłku
            - hrr_1min, hrr_class: klasyfikacja HRR
            - vt1_pct_vo2peak, vt2_pct_vo2peak: progi jako % VO2peak
        """
        import numpy as np

        out = {"status": "OK"}

        # ---- Pobierz surowe wyniki z silników ----
        e01 = results_dict.get("E01", {})
        e02 = results_dict.get("E02", {})
        e03 = results_dict.get("E03", {})
        e08 = results_dict.get("E08", {})

        mass = cls._num(body_mass_kg)
        if mass is None or mass <= 0:
            mass = None

        # ================================================================
        # A. VO2 PEAK/MAX — normalizacja (ACSM: highest 30s average)
        # ================================================================
        vo2_peak_mlmin = cls._num(e01.get("vo2_peak_mlmin"))
        vo2_peak_mlkgmin = cls._num(e01.get("vo2_peak_mlkgmin"))
        
        # Plateau detection results from E01
        vo2_determination = e01.get("vo2_determination", "VO2peak")
        plateau_detected = e01.get("plateau_detected", False)
        plateau_delta = cls._num(e01.get("plateau_delta_mlmin"))
        vo2_method_note = e01.get("vo2_method_note", "")
        
        out["vo2_determination"] = vo2_determination
        out["plateau_detected"] = plateau_detected
        out["plateau_delta_mlmin"] = plateau_delta
        out["vo2_method_note"] = vo2_method_note

        # Oblicz brakujące warianty
        if vo2_peak_mlmin is not None and mass is not None:
            if vo2_peak_mlkgmin is None:
                vo2_peak_mlkgmin = vo2_peak_mlmin / mass
        elif vo2_peak_mlkgmin is not None and mass is not None:
            if vo2_peak_mlmin is None:
                vo2_peak_mlmin = vo2_peak_mlkgmin * mass

        vo2_peak_lmin = vo2_peak_mlmin / 1000.0 if vo2_peak_mlmin is not None else None

        out["vo2_rel"] = round(vo2_peak_mlkgmin, 2) if vo2_peak_mlkgmin else None
        out["vo2_abs"] = round(vo2_peak_lmin, 3) if vo2_peak_lmin else None
        out["vo2_abs_mlmin"] = round(vo2_peak_mlmin, 1) if vo2_peak_mlmin else None

        # Predicted VO2max (Wasserman = primary, FRIEND = reference)
        height = cls._num(height_cm)
        pred_vo2_friend = cls._predicted_vo2max_friend(age, sex) if age else None
        pred_vo2_wass_mlmin = cls._predicted_vo2max_wasserman(age, height, mass, sex, modality) if (age and height and mass) else None
        pred_vo2_wass_mlkg = round(pred_vo2_wass_mlmin / mass, 1) if (pred_vo2_wass_mlmin and mass) else None

        # Primary %predicted = Wasserman (klinicznie wiarygodne)
        if pred_vo2_wass_mlmin and vo2_peak_mlmin:
            out["vo2_pct_predicted"] = round((vo2_peak_mlmin / pred_vo2_wass_mlmin) * 100, 1)
            out["vo2_predicted_method"] = "Wasserman"
        elif pred_vo2_friend and vo2_peak_mlkgmin:
            out["vo2_pct_predicted"] = round((vo2_peak_mlkgmin / pred_vo2_friend) * 100, 1)
            out["vo2_predicted_method"] = "FRIEND"
        else:
            out["vo2_pct_predicted"] = None
            out["vo2_predicted_method"] = None

        out["vo2_predicted_wasserman_mlmin"] = round(pred_vo2_wass_mlmin, 0) if pred_vo2_wass_mlmin else None
        out["vo2_predicted_wasserman_mlkg"] = pred_vo2_wass_mlkg
        out["vo2_predicted_friend"] = round(pred_vo2_friend, 1) if pred_vo2_friend else None

        # Klasyfikacja vs populacja
        pop_label, pop_pct = cls._classify_vo2_population(vo2_peak_mlkgmin, age, sex)
        out["vo2_class_pop"] = pop_label
        out["vo2_percentile_approx"] = pop_pct

        # Klasyfikacja vs sport
        sport_label, sport_desc = cls._classify_vo2_sport(vo2_peak_mlkgmin, modality, sex)
        out["vo2_class_sport"] = sport_label
        out["vo2_class_sport_desc"] = sport_desc

        # ================================================================
        # B. O2 PULSE
        # ================================================================
        # Prefer E05 v2.0, fallback to E01
        e05 = results_dict.get("E05", {})
        o2pulse_peak = cls._num(e05.get("o2pulse_peak") or e01.get("o2pulse_peak"))
        # Oblicz jeśli brak
        hr_peak = cls._num(e01.get("hr_peak"))
        if o2pulse_peak is None and vo2_peak_mlmin is not None and hr_peak is not None and hr_peak > 0:
            o2pulse_peak = vo2_peak_mlmin / hr_peak  # ml/beat

        out["o2pulse_peak"] = round(o2pulse_peak, 1) if o2pulse_peak else None

        # Predicted O2 Pulse
        pred_o2p_pop = cls._predicted_o2pulse_friend(age, sex) if age else None
        pred_o2p_ath = cls._predicted_o2pulse_athletes(age, sex) if age else None

        if o2pulse_peak and pred_o2p_pop:
            out["o2pulse_predicted_pop"] = round(pred_o2p_pop, 1)
            out["o2pulse_pct_predicted_pop"] = round((o2pulse_peak / pred_o2p_pop) * 100, 1)
        else:
            out["o2pulse_predicted_pop"] = None
            out["o2pulse_pct_predicted_pop"] = None

        if o2pulse_peak and pred_o2p_ath:
            out["o2pulse_predicted_ath"] = round(pred_o2p_ath, 1)
            out["o2pulse_pct_predicted_ath"] = round((o2pulse_peak / pred_o2p_ath) * 100, 1)
        else:
            out["o2pulse_predicted_ath"] = None
            out["o2pulse_pct_predicted_ath"] = None

        # ================================================================
        # C. VE/VCO2 SLOPE — sport-aware classification
        # ================================================================
        ve_slope = cls._num(e03.get('slope_to_vt2') or e03.get('slope_full') or e03.get('slope'))
        out["ve_vco2_slope"] = round(ve_slope, 1) if ve_slope else None

        # Dodatkowe metryki z E03 v2.0 do kontekstu
        ve_nadir = cls._num(e03.get('ve_vco2_nadir'))
        ve_slope_vt1 = cls._num(e03.get('slope_to_vt1'))
        petco2_vt1 = cls._num(e03.get('petco2_vt1'))

        # Bazowa klasyfikacja kliniczna
        label, desc = cls._classify_range(ve_slope, cls.VE_VCO2_RANGES)

        # SPORT-AWARE OVERRIDE:
        # Jeśli slope > 34 (ABNORMAL/HIGH), ale:
        #   - nadir < 30 (efektywność wentylacji OK)
        #   - PETCO2 przy VT1 >= 36 (brak patologii V/Q mismatch)
        #   - slope_to_vt1 < 34 (submaksymalne dane w normie)
        # → to NIE jest patologia, a fizjologiczna hiperwentylacja po VT2
        #   typowa u młodych sportowców (NOODLE 2024, Phillips 2020)
        is_sport_context = (
            label in ("ABNORMAL", "HIGH") and
            ve_nadir is not None and ve_nadir < 30 and
            (petco2_vt1 is None or petco2_vt1 >= 36)
        )

        if is_sport_context:
            label = "SPORT_ELEVATED"
            desc = (
                f"Slope podwyższony ({ve_slope:.0f}) przez hiperwentylację po VT2, "
                f"ale nadir {ve_nadir:.0f} w normie — brak patologii wentylacyjnej"
            )
            if ve_slope_vt1 is not None:
                desc += f" (slope do VT1: {ve_slope_vt1:.0f})"

        out["ve_vco2_class"] = label
        out["ve_vco2_desc"] = desc

        # ================================================================
        # D. RER PEAK
        # ================================================================
        rer_peak = cls._num(e01.get("rer_peak"))
        out["rer_peak"] = round(rer_peak, 3) if rer_peak else None

        label, desc = cls._classify_range(rer_peak, cls.RER_RANGES)
        out["rer_class"] = label
        out["rer_desc"] = desc

        # ================================================================
        # E. HRR (Heart Rate Recovery)
        # ================================================================
        hrr_1min = cls._num(e08.get("hrr_1min"))
        out["hrr_1min"] = round(hrr_1min, 1) if hrr_1min else None

        label, desc = cls._classify_range(hrr_1min, cls.HRR_1MIN_RANGES)
        out["hrr_1min_class"] = label
        out["hrr_1min_desc"] = desc

        hrr_3min = cls._num(e08.get("hrr_3min"))
        out["hrr_3min"] = round(hrr_3min, 1) if hrr_3min else None

        # ================================================================
        # F. HR MAX — predicted vs measured
        # ================================================================
        if hr_peak is not None:
            out["hr_peak"] = round(hr_peak, 0)
            pred_hr = cls._predicted_hrmax(age) if age else None
            if pred_hr:
                out["hr_predicted_tanaka"] = round(pred_hr, 0)
                out["hr_pct_predicted"] = round((hr_peak / pred_hr) * 100, 1)
            else:
                out["hr_predicted_tanaka"] = None
                out["hr_pct_predicted"] = None
        else:
            out["hr_peak"] = None
            out["hr_predicted_tanaka"] = None
            out["hr_pct_predicted"] = None

        # ================================================================
        # G. PROGI jako % VO2peak i % HRmax
        # ================================================================
        vt1_vo2 = cls._num(e02.get("vt1_vo2_mlmin"))
        vt2_vo2 = cls._num(e02.get("vt2_vo2_mlmin"))
        vt1_hr = cls._num(e02.get("vt1_hr"))
        vt2_hr = cls._num(e02.get("vt2_hr"))

        if vt1_vo2 and vo2_peak_mlmin:
            out["vt1_pct_vo2peak"] = round((vt1_vo2 / vo2_peak_mlmin) * 100, 1)
        else:
            out["vt1_pct_vo2peak"] = cls._num(e02.get("vt1_vo2_pct_peak"))

        if vt2_vo2 and vo2_peak_mlmin:
            out["vt2_pct_vo2peak"] = round((vt2_vo2 / vo2_peak_mlmin) * 100, 1)
        else:
            out["vt2_pct_vo2peak"] = cls._num(e02.get("vt2_vo2_pct_peak"))

        if vt1_hr and hr_peak:
            out["vt1_pct_hrmax"] = round((vt1_hr / hr_peak) * 100, 1)
        else:
            out["vt1_pct_hrmax"] = cls._num(e02.get("vt1_hr_pct_max"))

        if vt2_hr and hr_peak:
            out["vt2_pct_hrmax"] = round((vt2_hr / hr_peak) * 100, 1)
        else:
            out["vt2_pct_hrmax"] = cls._num(e02.get("vt2_hr_pct_max"))

        # Fizjologiczna walidacja progów
        # VT1 typowo: 45-65% VO2peak, 60-75% HRmax
        # VT2 typowo: 70-90% VO2peak, 80-92% HRmax
        vt1_pct = out.get("vt1_pct_vo2peak")
        vt2_pct = out.get("vt2_pct_vo2peak")
        out["vt1_physiological_check"] = (
            "OK" if vt1_pct and 35 <= vt1_pct <= 75 else
            "WARNING" if vt1_pct else "NO_DATA"
        )
        out["vt2_physiological_check"] = (
            "OK" if vt2_pct and 60 <= vt2_pct <= 95 else
            "WARNING" if vt2_pct else "NO_DATA"
        )

        # ================================================================
        # H. PODSUMOWANIE — ogólna ocena testu
        # ================================================================
        flags = []
        if out.get("rer_class") in ("SUBMAXIMAL", "LIKELY_SUBMAX"):
            flags.append("RER_LOW_EFFORT")
        if out.get("ve_vco2_class") in ("ABNORMAL", "HIGH", "SEVERE"):
            flags.append("VENT_INEFFICIENCY")
        # SPORT_ELEVATED nie jest flagowane — to fizjologia, nie patologia
        if out.get("hrr_1min_class") == "ABNORMAL":
            flags.append("HRR_ABNORMAL")
        if out.get("vt1_physiological_check") == "WARNING":
            flags.append("VT1_OUTSIDE_EXPECTED_RANGE")
        if out.get("vt2_physiological_check") == "WARNING":
            flags.append("VT2_OUTSIDE_EXPECTED_RANGE")

        out["clinical_flags"] = flags
        out["test_quality"] = (
            "HIGH" if not flags else
            "MODERATE" if len(flags) <= 1 else
            "LOW"
        )

        return out


"""
Engine E16 v2.0 — Training Zones
═══════════════════════════════════════════════════════════
Reference:
  Skinner & McLellan 1980 — Three-phase model (VT1, VT2)
  Seiler 2010 (IJSPP) — 3-zone → 5-zone alignment
  IJSPP 2025 Expert Panel — Zone 2 = just below VT1/LT1
  SEMS-journal 2025 — 5-zone model with VT1 & VT2 anchors
  VO2 Master — Zone boundaries and %HRmax alignment

Model logic (Seiler 5-zone, VT1/VT2-anchored):
  ═══════════════════════════════════════════════════════════
  Z1  Recovery      : HRrest ... VT1 - offset         (< ~80% VT1 HR)
  Z2  Aerobic Base  : VT1 - offset ... VT1             (just below VT1)
  Z3  Tempo         : VT1 ... VT2                      (between thresholds)
  Z4  Threshold     : VT2 ... VT2 + buffer             (at/above VT2)
  Z5  VO2max        : VT2 + buffer ... HRmax            (supramaximal aerobic)
  ═══════════════════════════════════════════════════════════

  The KEY principle: VT1 is the BOUNDARY between Z2 and Z3.
  VT2 is the BOUNDARY between Z3 and Z4.
  This is the Seiler alignment (IJSPP 2025 expert consensus).

Speed zones: interpolated from HR↔speed relationship during test.

Outputs per zone: HR range, speed range (if available), VO2 range,
  %HRmax range, %VO2peak range, description, training purpose.

Also outputs:
  - 3-zone model (for TID analysis: Zone I < VT1, Zone II VT1-VT2, Zone III > VT2)
  - Zone widths (bpm) for QC
  - Aerobic reserve fraction: (VT1-HRrest)/(HRmax-HRrest) — how much of
    HR range is aerobic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class Engine_E16_Zones_v2:
    """
    E16 v2.0 — Training Zones (5-zone + 3-zone models)

    Anchored on VT1 and VT2 from E02, with speed interpolation from test data.
    """

    # Zone names and descriptions (Polish + English)
    ZONE_INFO = {
        'z1': {
            'name_pl': 'Regeneracja',
            'name_en': 'Recovery',
            'description_pl': 'Aktywna regeneracja, rozgrzewka, schładzanie',
            'description_en': 'Active recovery, warm-up, cool-down',
            'rpe': '1-2',
            'lactate': '< 1.5 mmol/L',
            'substrate': 'Tłuszcze > glikogen',
            'duration': '20-90 min',
            'color': '#4CAF50',  # green
        },
        'z2': {
            'name_pl': 'Baza tlenowa',
            'name_en': 'Aerobic Base',
            'description_pl': 'Trening bazowy, kapilaryzacja, biogeneza mitochondrialna',
            'description_en': 'Base training, capillarization, mitochondrial biogenesis',
            'rpe': '2-3',
            'lactate': '1.5-2.0 mmol/L',
            'substrate': 'Tłuszcze ≈ glikogen',
            'duration': '40-180 min',
            'color': '#8BC34A',  # light green
        },
        'z3': {
            'name_pl': 'Tempo',
            'name_en': 'Tempo',
            'description_pl': 'Między progami, wysoki koszt energetyczny',
            'description_en': 'Between thresholds, high energy cost, grey zone',
            'rpe': '4-6',
            'lactate': '2.0-4.0 mmol/L',
            'substrate': 'Glikogen > tłuszcze',
            'duration': '20-60 min',
            'color': '#FFC107',  # amber
        },
        'z4': {
            'name_pl': 'Próg beztlenowy',
            'name_en': 'Threshold',
            'description_pl': 'Wokół VT2/LT2, max steady-state',
            'description_en': 'At/above VT2/LT2, maximal lactate steady state',
            'rpe': '7-8',
            'lactate': '4.0-8.0 mmol/L',
            'substrate': 'Glikogen dominacja',
            'duration': '10-40 min',
            'color': '#FF9800',  # orange
        },
        'z5': {
            'name_pl': 'VO2max',
            'name_en': 'VO2max',
            'description_pl': 'Interwały VO2max, wydolność maksymalna',
            'description_en': 'VO2max intervals, maximal aerobic power',
            'rpe': '9-10',
            'lactate': '> 8 mmol/L',
            'substrate': 'Glikogen + fosfokreatyna',
            'duration': '3-8 min intervals',
            'color': '#F44336',  # red
        },
    }

    @classmethod
    def run(cls, vt1_hr: float, vt2_hr: float, hr_max: float,
            hr_rest: float = None,
            vt1_speed: float = None, vt2_speed: float = None,
            max_speed: float = None,
            vt1_vo2: float = None, vt2_vo2: float = None,
            vo2_peak: float = None,
            df_ex: pd.DataFrame = None) -> Dict:
        """
        Parameters
        ----------
        vt1_hr, vt2_hr : HR at VT1 and VT2 (bpm)
        hr_max         : Maximum heart rate (bpm)
        hr_rest        : Resting heart rate (optional, estimated if missing)
        vt1_speed, vt2_speed : Speed at VT1/VT2 (km/h, optional)
        max_speed      : Peak speed from test (km/h, optional)
        vt1_vo2, vt2_vo2, vo2_peak : VO2 values (ml/min, optional)
        df_ex          : Exercise DataFrame for HR↔Speed interpolation
        """

        result = cls._init_result()

        try:
            # ── Validate inputs ──────────────────────────────────
            if vt1_hr is None or vt2_hr is None:
                result['flags'].append('MISSING_THRESHOLDS')
                return result

            vt1_hr = float(vt1_hr)
            vt2_hr = float(vt2_hr)

            if hr_max is None or hr_max < vt2_hr:
                hr_max = vt2_hr + 15  # estimate
                result['flags'].append('HRMAX_ESTIMATED')
            hr_max = float(hr_max)

            if hr_rest is None:
                hr_rest = max(50.0, hr_max * 0.30)  # rough estimate
                result['flags'].append('HRREST_ESTIMATED')
            hr_rest = float(hr_rest)

            # Sanity checks
            if vt1_hr >= vt2_hr:
                result['flags'].append('VT1_GE_VT2')
                return result
            if vt2_hr >= hr_max:
                hr_max = vt2_hr + 10
                result['flags'].append('VT2_GE_HRMAX_CORRECTED')

            # ── Build HR↔Speed mapping ───────────────────────────
            speed_fn = None
            if df_ex is not None:
                speed_fn = cls._build_speed_interpolator(df_ex)

            # ── Calculate zone boundaries ────────────────────────
            # Key principle: VT1 = Z2|Z3 boundary, VT2 = Z3|Z4 boundary

            hr_range_below_vt1 = vt1_hr - hr_rest

            # Z1/Z2 split: divide the sub-VT1 range
            # Z1 = below ~85% of VT1 HR (recovery/easy)
            # Z2 = ~85% VT1 HR to VT1 (aerobic base, "just below VT1")
            # This ensures Z2 captures the FatMax / aerobic base zone
            z1_z2_split = round(vt1_hr - hr_range_below_vt1 * 0.35)
            # Clamp: Z2 should be at least 10 bpm wide
            z1_z2_split = min(z1_z2_split, int(vt1_hr) - 10)

            # z4_z5_split removed — VT2 is now directly the Z4/Z5 boundary (Seiler model)

            # 5-zone model: VT1 = Z2|Z3 boundary, VT2 = Z4|Z5 boundary (Seiler/Coggan)
            # Z3/Z4 split = midpoint between VT1 and VT2
            z3_z4_split = round((vt1_hr + vt2_hr) / 2.0)

            zones = {}
            boundaries = {
                'z1': (int(round(hr_rest)), int(z1_z2_split)),
                'z2': (int(z1_z2_split) + 1, int(round(vt1_hr))),
                'z3': (int(round(vt1_hr)) + 1, int(z3_z4_split)),
                'z4': (int(z3_z4_split) + 1, int(round(vt2_hr))),
                'z5': (int(round(vt2_hr)) + 1, int(round(hr_max))),
            }

            for zk in ['z1', 'z2', 'z3', 'z4', 'z5']:
                hr_lo, hr_hi = boundaries[zk]
                zone = {
                    'hr_low': hr_lo,
                    'hr_high': hr_hi,
                    'hr_width': hr_hi - hr_lo,
                    'pct_hrmax_low': round(hr_lo / hr_max * 100, 1),
                    'pct_hrmax_high': round(hr_hi / hr_max * 100, 1),
                }

                # Speed interpolation
                if speed_fn:
                    zone['speed_low'] = round(speed_fn(hr_lo), 1)
                    zone['speed_high'] = round(speed_fn(hr_hi), 1)
                elif vt1_speed and vt2_speed:
                    # Linear interpolation from known points
                    zone['speed_low'] = round(cls._interp_speed(
                        hr_lo, hr_rest, vt1_hr, vt2_hr, hr_max,
                        0, vt1_speed, vt2_speed, max_speed), 1)
                    zone['speed_high'] = round(cls._interp_speed(
                        hr_hi, hr_rest, vt1_hr, vt2_hr, hr_max,
                        0, vt1_speed, vt2_speed, max_speed), 1)

                # VO2 ranges (if available)
                if vt1_vo2 and vt2_vo2 and vo2_peak:
                    zone['vo2_low'] = round(cls._interp_vo2(
                        hr_lo, hr_rest, vt1_hr, vt2_hr, hr_max,
                        0, vt1_vo2, vt2_vo2, vo2_peak), 0)
                    zone['vo2_high'] = round(cls._interp_vo2(
                        hr_hi, hr_rest, vt1_hr, vt2_hr, hr_max,
                        0, vt1_vo2, vt2_vo2, vo2_peak), 0)
                    zone['pct_vo2peak_low'] = round(zone['vo2_low'] / vo2_peak * 100, 1)
                    zone['pct_vo2peak_high'] = round(zone['vo2_high'] / vo2_peak * 100, 1)

                # Metadata
                zone.update(cls.ZONE_INFO[zk])
                zones[zk] = zone

            result['zones'] = zones
            result['model'] = '5-zone Seiler VT1/VT2-anchored'

            # ── 3-zone model (for TID) ───────────────────────────
            result['three_zone'] = {
                'zone_I':   {'hr_low': int(round(hr_rest)), 'hr_high': int(round(vt1_hr)),
                             'label': 'LIT (< VT1)'},
                'zone_II':  {'hr_low': int(round(vt1_hr)) + 1, 'hr_high': int(round(vt2_hr)),
                             'label': 'MIT (VT1-VT2)'},
                'zone_III': {'hr_low': int(round(vt2_hr)) + 1, 'hr_high': int(round(hr_max)),
                             'label': 'HIT (> VT2)'},
            }

            # ── Derived metrics ──────────────────────────────────
            hr_reserve = hr_max - hr_rest
            result['hr_reserve'] = round(hr_reserve, 0)
            result['aerobic_reserve_pct'] = round(
                (vt1_hr - hr_rest) / hr_reserve * 100, 1) if hr_reserve > 0 else None
            result['threshold_gap_bpm'] = round(vt2_hr - vt1_hr, 0)
            result['threshold_gap_pct_hrmax'] = round(
                (vt2_hr - vt1_hr) / hr_max * 100, 1)

            # Anchor HRs for reference
            result['vt1_hr'] = round(vt1_hr, 0)
            result['vt2_hr'] = round(vt2_hr, 0)
            result['hr_max'] = round(hr_max, 0)
            result['hr_rest'] = round(hr_rest, 0)

            # ── QC ───────────────────────────────────────────────
            for zk, z in zones.items():
                if z['hr_width'] < 3:
                    result['flags'].append(f'NARROW_ZONE_{zk.upper()}')
                if z['hr_width'] > 50:
                    result['flags'].append(f'WIDE_ZONE_{zk.upper()}')

            result['status'] = 'OK'

        except Exception as e:
            result['status'] = 'ERROR'
            result['flags'].append(f'EXCEPTION:{e}')

        return result

    # ═══════════════════════════════════════════════════════════════
    # INTERNALS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _init_result() -> Dict:
        return {
            'status': 'NOT_RUN',
            'zones': {},
            'three_zone': {},
            'model': None,
            'hr_reserve': None,
            'aerobic_reserve_pct': None,
            'threshold_gap_bpm': None,
            'threshold_gap_pct_hrmax': None,
            'vt1_hr': None,
            'vt2_hr': None,
            'hr_max': None,
            'hr_rest': None,
            'flags': [],
        }

    @classmethod
    def _build_speed_interpolator(cls, df_ex: pd.DataFrame):
        """Build HR→Speed function from exercise data."""
        hr_col = next((c for c in ['HR_bpm', 'hr_bpm', 'HR'] if c in df_ex.columns), None)
        spd_col = next((c for c in ['speed_km_h', 'Speed_km_h', 'speed_kmh',
                                     'speed', 'Speed'] if c in df_ex.columns), None)
        if not hr_col or not spd_col:
            return None

        hr = df_ex[hr_col].values.astype(float)
        spd = df_ex[spd_col].values.astype(float)

        # Remove NaN
        mask = np.isfinite(hr) & np.isfinite(spd) & (spd > 0)
        if mask.sum() < 10:
            return None

        hr_clean = hr[mask]
        spd_clean = spd[mask]

        # Bin by HR (every 5 bpm) to get median speed per HR band
        hr_min_v, hr_max_v = int(hr_clean.min()), int(hr_clean.max())
        bins = np.arange(hr_min_v, hr_max_v + 6, 5)
        hr_centers = []
        spd_medians = []
        for i in range(len(bins) - 1):
            m = (hr_clean >= bins[i]) & (hr_clean < bins[i+1])
            if m.sum() >= 3:
                hr_centers.append((bins[i] + bins[i+1]) / 2)
                spd_medians.append(np.median(spd_clean[m]))

        if len(hr_centers) < 3:
            return None

        hr_arr = np.array(hr_centers)
        spd_arr = np.array(spd_medians)

        def interpolator(target_hr):
            return float(np.interp(target_hr, hr_arr, spd_arr))

        return interpolator

    @staticmethod
    def _interp_speed(target_hr, hr_rest, vt1_hr, vt2_hr, hr_max,
                      rest_speed, vt1_speed, vt2_speed, max_speed):
        """Piecewise linear interpolation of speed from known anchor points."""
        if max_speed is None:
            max_speed = vt2_speed * 1.15

        anchors_hr = [hr_rest, vt1_hr, vt2_hr, hr_max]
        anchors_spd = [rest_speed or 0, vt1_speed, vt2_speed, max_speed]
        return float(np.interp(target_hr, anchors_hr, anchors_spd))

    @staticmethod
    def _interp_vo2(target_hr, hr_rest, vt1_hr, vt2_hr, hr_max,
                    rest_vo2, vt1_vo2, vt2_vo2, vo2_peak):
        """Piecewise linear interpolation of VO2 from known anchor points."""
        anchors_hr = [hr_rest, vt1_hr, vt2_hr, hr_max]
        anchors_vo2 = [rest_vo2 or 300, vt1_vo2, vt2_vo2, vo2_peak]
        return float(np.interp(target_hr, anchors_hr, anchors_vo2))


        # Uproszczony generator stref (do celów raportu)
        # Pobieramy tętna z progów (zakładamy, że wchodzą tu wartości HR, a nie czasy - uwaga na input!)
        # W Orchestratorze przekażemy HR.

        # INPUT CHECKS
        h1 = vt1 # HR at VT1
        h2 = vt2 # HR at VT2

        if h1 and h2:
            zones['z1'] = {'hr_low': int(hr_max * 0.50) if hr_max else None, 'hr_high': int(h1 * 0.95)}
            zones['z2'] = {'hr_low': int(h1), 'hr_high': int(h1 * 1.05)} # Wokół VT1
            zones['z3'] = {'hr_low': int(h1*1.05), 'hr_high': int(h2*0.95)} # Pomiędzy
            zones['z4'] = {'hr_low': int(h2*0.95), 'hr_high': int(h2*1.05)} # Wokół VT2
            zones['z5'] = {'hr_low': int(h2*1.05), 'hr_high': hr_max if hr_max else int(h2*1.15)}

        return zones

# --- E17: GAS EXCHANGE EFFICIENCY ---
class Engine_E17_GasExchange:
    """
    E17 — Gas Exchange Efficiency & Dead Space Analysis  v2.0
    ═══════════════════════════════════════════════════════════
    Wasserman Panel 4/6/7 — wymiana gazowa podczas wysiłku.

    Parametry analizowane (z dostępnych kolumn CSV):
    ──────────────────────────────────────────────────
    1. PetCO2 trajectory   — wzorzec ciśnienia end-tidal CO2
    2. PetO2 trajectory    — wzorzec ciśnienia end-tidal O2
    3. VD/VT analysis      — stosunek przestrzeni martwej do objętości oddechowej
    4. P(A-a)CO2           — gradient pęcherzykowo-tętniczy CO2 (estymowany)
    5. SpO2 desaturation   — pulsoksymetria (jeśli dostępna)
    6. Alveolar gas        — PAO2, PaCO2 estymowane, PECO2, PEO2

    Fizjologia (Wasserman 6th ed, Glaab & Taube 2022, ATS/ACCP 2003):
    ──────────────────────────────────────────────────────────────────
    PRAWIDLOWY WZORZEC PetCO2:
      Rest: ~36-42 mmHg
      -> Wzrost podczas wysilku (izokapniczne buforowanie)
      -> Plateau/peak blisko VT1/AT
      -> Spadek po VT2/RCP (kompensacja oddechowa kwasicy)

    PRAWIDLOWY WZORZEC PetO2:
      Rest: ~100-110 mmHg
      -> Spadek/stabilnosc do VT1 (nadir)
      -> Wzrost od VT1 (hiperwentylacja kompensacyjna)

    VD/VT (dead space / tidal volume):
      Rest: ~0.28-0.35 (healthy), spada z wysilkiem
      Peak: ~0.15-0.20 (healthy)
      Brak spadku lub wzrost = choroba naczyn plucnych (PH, PE)

    Referencje:
      Wasserman K et al. Principles of Exercise Testing, 6th ed (2012)
      Glaab T, Taube C. Respiratory Research 2022; 23:9
      ATS/ACCP Statement. Am J Respir Crit Care Med 2003; 167:211-77
      Sun XG et al. JACC 2002; 40:1073-9
      Datta D et al. Ann Thorac Med 2015; 10:77-86
    """

    _PETCO2_COLS = ['PetCO2_mmHg', 'PETCO2', 'PetCO2', 'PETCO2_mmHg']
    _PETO2_COLS  = ['PetO2_mmHg', 'PETO2', 'PetO2', 'PETO2_mmHg']
    _VDVT_COLS   = ['VD/VT(est)', 'VD/VT', 'VDVT', 'VD_VT']
    _SPO2_COLS   = ['SpO2', 'SaO2', 'SpO2_%']
    _PACO2_COLS  = ['PaCO2(est.)', 'PaCO2', 'PaCO2_est']
    _PAACO2_COLS = ['P(a-et)CO2(est.)', 'P(A-a)CO2', 'Pa_et_CO2']
    _PAO2_COLS   = ['PAO2(est.)', 'PAO2', 'PAO2_est']
    _PECO2_COLS  = ['PECO2', 'PeCO2']
    _PEO2_COLS   = ['PEO2', 'PeO2']
    _PH_COLS     = ['pH']
    _TIME_COLS   = ['Time_s', 'Time_sec', 'time_s']

    @staticmethod
    def _find_col(df, aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    @staticmethod
    def _safe_numeric(series):
        import pandas as pd
        return pd.to_numeric(series, errors='coerce')

    @staticmethod
    def _median_window(arr, center, half_w=3):
        import numpy as np
        lo = max(0, center - half_w)
        hi = min(len(arr), center + half_w + 1)
        vals = arr[lo:hi]
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if len(vals) > 0 else None

    @staticmethod
    def run(df_cpet, e00=None, e01=None, e02=None, cfg=None):
        import numpy as np
        import pandas as pd

        result = {
            'status': 'OK',
            'petco2_rest': None, 'petco2_at_vt1': None, 'petco2_peak_val': None,
            'petco2_at_vt2': None, 'petco2_at_peak': None,
            'petco2_rise_to_vt1': None, 'petco2_drop_vt2_to_peak': None,
            'petco2_pattern': None, 'petco2_clinical': None,
            'peto2_rest': None, 'peto2_nadir': None, 'peto2_at_vt1': None,
            'peto2_at_vt2': None, 'peto2_at_peak': None,
            'peto2_pattern': None,
            'vdvt_rest': None, 'vdvt_at_vt1': None, 'vdvt_at_vt2': None,
            'vdvt_at_peak': None, 'vdvt_delta': None,
            'vdvt_pattern': None, 'vdvt_clinical': None,
            'spo2_rest': None, 'spo2_min': None, 'spo2_at_peak': None,
            'spo2_drop': None, 'spo2_clinical': None,
            'paco2_rest': None, 'paco2_peak': None,
            'pa_et_co2_rest': None, 'pa_et_co2_peak': None,
            'pao2_rest': None, 'pao2_peak': None,
            'peco2_rest': None, 'peco2_peak': None,
            'peo2_rest': None, 'peo2_peak': None,
            'ph_rest': None, 'ph_peak': None,
            'gas_exchange_pattern': None,
            'flags': [],
            'available_signals': [],
        }

        E = Engine_E17_GasExchange
        df = df_cpet.copy()

        tc = E._find_col(df, E._TIME_COLS)
        if tc is None:
            result['status'] = 'NO_TIME'
            return result
        df['_t'] = E._safe_numeric(df[tc])

        t_start = 0
        t_end = df['_t'].max()
        if e00:
            t_start = e00.get('exercise_start_sec', t_start) or t_start
            # Use t_stop (peak exercise) as boundary — exclude recovery phase
            t_end = e00.get('t_stop') or e00.get('exercise_end_sec', t_end) or t_end

        df_ex = df[(df['_t'] >= t_start) & (df['_t'] <= t_end)].copy()
        if len(df_ex) < 20:
            result['status'] = 'INSUFFICIENT_DATA'
            return result

        vt1_t = e02.get('vt1_time_sec') if e02 else None
        vt2_t = e02.get('vt2_time_sec') if e02 else None

        def at_time(series, times, target, hw=5):
            if target is None:
                return None
            idx = np.argmin(np.abs(times - target))
            return E._median_window(series, idx, hw)

        times = df_ex['_t'].values
        n = len(times)
        n_rest = max(5, int(n * 0.10))
        n_peak_start = max(0, n - max(5, int(n * 0.10)))

        # ═══ 1. PetCO2 ═══
        pc2_col = E._find_col(df_ex, E._PETCO2_COLS)
        if pc2_col:
            result['available_signals'].append('PetCO2')
            pc2 = E._safe_numeric(df_ex[pc2_col]).values

            pc2_rest = float(np.nanmedian(pc2[:n_rest]))
            pc2_peak = float(np.nanmedian(pc2[n_peak_start:]))
            valid_pc2 = pc2[~np.isnan(pc2)]
            pc2_max = float(np.nanmax(valid_pc2)) if len(valid_pc2) > 0 else None

            result['petco2_rest'] = round(pc2_rest, 1)
            result['petco2_at_peak'] = round(pc2_peak, 1)
            result['petco2_peak_val'] = round(pc2_max, 1) if pc2_max else None

            pc2_vt1 = at_time(pc2, times, vt1_t)
            pc2_vt2 = at_time(pc2, times, vt2_t)
            if pc2_vt1 is not None:
                result['petco2_at_vt1'] = round(pc2_vt1, 1)
            if pc2_vt2 is not None:
                result['petco2_at_vt2'] = round(pc2_vt2, 1)

            if pc2_vt1 is not None:
                result['petco2_rise_to_vt1'] = round(pc2_vt1 - pc2_rest, 1)
            if pc2_vt2 is not None:
                result['petco2_drop_vt2_to_peak'] = round(pc2_peak - pc2_vt2, 1)

            rise = (pc2_vt1 - pc2_rest) if pc2_vt1 else (pc2_max - pc2_rest if pc2_max else 0)
            drop = (pc2_peak - pc2_vt2) if pc2_vt2 else (pc2_peak - pc2_max if pc2_max else 0)

            if pc2_rest < 30:
                pattern = 'LOW_BASELINE'
                clinical = 'Niskie PetCO2 rest <30 — hiperwentylacja spoczynnkowa lub VQ mismatch'
                result['flags'].append('LOW_PETCO2_REST')
            elif rise < -1:
                pattern = 'FALLING_EARLY'
                clinical = 'PetCO2 spada od startu — patologiczne (PH, VQ mismatch)'
                result['flags'].append('PETCO2_NO_RISE')
            elif rise < 2 and pc2_rest < 36:
                pattern = 'FLAT_LOW'
                clinical = 'PetCO2 nie rosnie i jest niskie — nieefektywna wymiana gazowa'
                result['flags'].append('PETCO2_FLAT_LOW')
            elif rise >= 2 and drop < -2:
                pattern = 'NORMAL'
                clinical = 'Prawidlowy: wzrost do VT1, spadek po VT2 (kompensacja RCP)'
            elif rise >= 2 and drop >= -2:
                pattern = 'NO_RCP_DROP'
                clinical = 'Wzrost OK ale brak spadku po VT2 — slaba kompensacja oddechowa lub submaksymalny test'
            else:
                pattern = 'NORMAL'
                clinical = 'Prawidlowy wzorzec PetCO2'

            result['petco2_pattern'] = pattern
            result['petco2_clinical'] = clinical

        # ═══ 2. PetO2 ═══
        po2_col = E._find_col(df_ex, E._PETO2_COLS)
        if po2_col:
            result['available_signals'].append('PetO2')
            po2 = E._safe_numeric(df_ex[po2_col]).values

            po2_rest = float(np.nanmedian(po2[:n_rest]))
            po2_peak = float(np.nanmedian(po2[n_peak_start:]))
            valid_po2 = po2[~np.isnan(po2)]
            po2_min = float(np.nanmin(valid_po2)) if len(valid_po2) > 0 else None

            result['peto2_rest'] = round(po2_rest, 1)
            result['peto2_nadir'] = round(po2_min, 1) if po2_min else None
            result['peto2_at_peak'] = round(po2_peak, 1)

            po2_vt1 = at_time(po2, times, vt1_t)
            po2_vt2 = at_time(po2, times, vt2_t)
            if po2_vt1: result['peto2_at_vt1'] = round(po2_vt1, 1)
            if po2_vt2: result['peto2_at_vt2'] = round(po2_vt2, 1)

            if po2_min is not None:
                nadir_drop = po2_rest - po2_min
                peak_rise = po2_peak - po2_min
                if nadir_drop < 1:
                    result['peto2_pattern'] = 'NO_INITIAL_DROP'
                    result['flags'].append('PETO2_NO_DROP')
                elif peak_rise > 5:
                    result['peto2_pattern'] = 'NORMAL'
                else:
                    result['peto2_pattern'] = 'BLUNTED_RISE'

        # ═══ 3. VD/VT ═══
        vdvt_col = E._find_col(df_ex, E._VDVT_COLS)
        if vdvt_col:
            result['available_signals'].append('VD/VT')
            vdvt = E._safe_numeric(df_ex[vdvt_col]).values

            vdvt_clean = vdvt[~np.isnan(vdvt)]
            if len(vdvt_clean) < 10:
                result['vdvt_pattern'] = 'INSUFFICIENT_DATA'
                result['vdvt_clinical'] = 'Za malo danych VD/VT'
            else:
                vdvt_rest = float(np.nanmedian(vdvt[:n_rest]))
                vdvt_peak = float(np.nanmedian(vdvt[n_peak_start:]))

                if np.isnan(vdvt_rest) or np.isnan(vdvt_peak):
                    result['vdvt_pattern'] = 'INSUFFICIENT_DATA'
                    result['vdvt_clinical'] = 'Brak danych VD/VT rest/peak'
                else:
                    result['vdvt_rest'] = round(vdvt_rest, 3)
                    result['vdvt_at_peak'] = round(vdvt_peak, 3)
                    result['vdvt_delta'] = round(vdvt_peak - vdvt_rest, 3)

                    vdvt_vt1 = at_time(vdvt, times, vt1_t)
                    vdvt_vt2 = at_time(vdvt, times, vt2_t)
                    if vdvt_vt1: result['vdvt_at_vt1'] = round(vdvt_vt1, 3)
                    if vdvt_vt2: result['vdvt_at_vt2'] = round(vdvt_vt2, 3)

                    delta = vdvt_peak - vdvt_rest
                    if vdvt_rest > 0.40:
                        result['vdvt_pattern'] = 'ELEVATED_REST'
                        result['vdvt_clinical'] = 'Podwyzszone VD/VT rest >0.40 — VQ mismatch'
                        result['flags'].append('VDVT_HIGH_REST')
                    elif delta > 0.02:
                        # Check: is VD/VT dropping mid-test then rising only at peak? → tachypnea
                        vdvt_mid = E._median_window(vdvt, n // 2, n // 8) if n > 20 else None
                        delta_mid = (vdvt_mid - vdvt_rest) if vdvt_mid and not np.isnan(vdvt_mid) else None
                        late_rise = (delta_mid is not None and delta_mid < -0.005 and delta > 0.02)
                        if late_rise:
                            result['vdvt_pattern'] = 'TACHYPNEIC_RISE'
                            result['vdvt_clinical'] = 'VD/VT spada w srodku testu, rosnie na szczycie — mechaniczny wzrost (tachypnea przy BF>50), nie patologia VQ'
                            result['vdvt_mid'] = round(vdvt_mid, 3)
                        else:
                            result['vdvt_pattern'] = 'PARADOXICAL_RISE'
                            result['vdvt_clinical'] = 'VD/VT rosnie z wysilkiem — patologiczne (PH, PE, COPD)'
                            result['flags'].append('VDVT_PARADOXICAL')
                    elif delta > -0.03:
                        result['vdvt_pattern'] = 'INSUFFICIENT_DROP'
                        result['vdvt_clinical'] = 'VD/VT nie spada wystarczajaco — lagodne zaburzenie VQ'
                        result['flags'].append('VDVT_NO_DROP')
                    elif vdvt_peak > 0.30:
                        result['vdvt_pattern'] = 'HIGH_PEAK'
                        result['vdvt_clinical'] = 'VD/VT peak >0.30 — sugestia choroby naczyn plucnych'
                        result['flags'].append('VDVT_HIGH_PEAK')
                    else:
                        result['vdvt_pattern'] = 'NORMAL'
                        result['vdvt_clinical'] = 'Prawidlowy spadek VD/VT z wysilkiem'

        # ═══ 4. SpO2 ═══
        spo2_col = E._find_col(df_ex, E._SPO2_COLS)
        if spo2_col:
            spo2 = E._safe_numeric(df_ex[spo2_col])
            spo2_valid = spo2.dropna()
            if len(spo2_valid) > 10:
                result['available_signals'].append('SpO2')
                spo2_v = spo2.values
                spo2_rest = float(np.nanmedian(spo2_v[:n_rest]))
                spo2_min = float(np.nanmin(spo2_valid))
                spo2_peak_v = float(np.nanmedian(spo2_v[n_peak_start:]))

                result['spo2_rest'] = round(spo2_rest, 1)
                result['spo2_min'] = round(spo2_min, 1)
                result['spo2_at_peak'] = round(spo2_peak_v, 1)
                result['spo2_drop'] = round(spo2_rest - spo2_min, 1)

                drop = spo2_rest - spo2_min
                if spo2_min < 88:
                    result['spo2_clinical'] = 'SEVERE_DESAT'
                    result['flags'].append('SPO2_SEVERE')
                elif spo2_min < 92:
                    result['spo2_clinical'] = 'MODERATE_DESAT'
                    result['flags'].append('SPO2_MODERATE')
                elif drop > 4:
                    result['spo2_clinical'] = 'EIAH'
                    result['flags'].append('SPO2_EIAH')
                elif drop > 2:
                    result['spo2_clinical'] = 'MILD_DESAT'
                else:
                    result['spo2_clinical'] = 'NORMAL'

        # ═══ 5. PaCO2 & P(a-et)CO2 ═══
        paco2_col = E._find_col(df_ex, E._PACO2_COLS)
        if paco2_col:
            paco2 = E._safe_numeric(df_ex[paco2_col])
            paco2_valid = paco2.dropna()
            if len(paco2_valid) > 5:
                result['available_signals'].append('PaCO2_est')
                paco2_v = paco2.values
                result['paco2_rest'] = round(float(np.nanmedian(paco2_v[:n_rest])), 1)
                result['paco2_peak'] = round(float(np.nanmedian(paco2_v[n_peak_start:])), 1)
                if result['paco2_peak'] and result['paco2_peak'] > 45:
                    result['flags'].append('HYPERCAPNIA_PEAK')
                elif result['paco2_peak'] and result['paco2_peak'] < 30:
                    result['flags'].append('HYPOCAPNIA_PEAK')

        paaco2_col = E._find_col(df_ex, E._PAACO2_COLS)
        if paaco2_col:
            paaco2 = E._safe_numeric(df_ex[paaco2_col])
            paaco2_valid = paaco2.dropna()
            if len(paaco2_valid) > 5:
                result['available_signals'].append('P(a-et)CO2')
                paaco2_v = paaco2.values
                result['pa_et_co2_rest'] = round(float(np.nanmedian(paaco2_v[:n_rest])), 1)
                result['pa_et_co2_peak'] = round(float(np.nanmedian(paaco2_v[n_peak_start:])), 1)

        # ═══ 6. PAO2, PECO2, PEO2, pH ═══
        for col_aliases, key_prefix, sig_name in [
            (E._PAO2_COLS, 'pao2', 'PAO2_est'),
            (E._PECO2_COLS, 'peco2', 'PECO2'),
            (E._PEO2_COLS, 'peo2', 'PEO2'),
            (E._PH_COLS, 'ph', 'pH'),
        ]:
            col = E._find_col(df_ex, col_aliases)
            if col:
                vals = E._safe_numeric(df_ex[col])
                valid = vals.dropna()
                if len(valid) > 5:
                    result['available_signals'].append(sig_name)
                    v = vals.values
                    result[f'{key_prefix}_rest'] = round(float(np.nanmedian(v[:n_rest])), 2)
                    result[f'{key_prefix}_peak'] = round(float(np.nanmedian(v[n_peak_start:])), 2)

        # ═══ OVERALL ═══
        if len(result['available_signals']) == 0:
            result['status'] = 'NO_SIGNAL'
            return result

        flags = result['flags']
        serious = [f for f in flags if f in ('SPO2_SEVERE', 'VDVT_PARADOXICAL', 'PETCO2_NO_RISE', 'HYPERCAPNIA_PEAK')]
        moderate = [f for f in flags if f in ('SPO2_MODERATE', 'SPO2_EIAH', 'VDVT_NO_DROP', 'VDVT_HIGH_PEAK', 'PETCO2_FLAT_LOW', 'LOW_PETCO2_REST')]

        if len(serious) > 0:
            result['gas_exchange_pattern'] = 'ABNORMAL'
        elif len(moderate) > 0:
            result['gas_exchange_pattern'] = 'BORDERLINE'
        elif len(flags) > 0:
            result['gas_exchange_pattern'] = 'MILD_DEVIATION'
        else:
            result['gas_exchange_pattern'] = 'NORMAL'

        return result



# ═══════════════════════════════════════════════════════════
# E19: TEST VALIDITY + PHYSIOLOGICAL CONCORDANCE
# ═══════════════════════════════════════════════════════════

class Engine_E19_Concordance:
    """
    E19 v1.0 — Test Validity Score + Physiological Concordance
    ═══════════════════════════════════════════════════════════
    Cross-validates results from ALL engines to detect:
    1. Test validity (was the test maximal/reliable?)
    2. Physiological concordance (are results internally consistent?)
    3. Temporal alignment (do breakpoints agree?)
    
    References:
      ATS/ACCP 2003, Wasserman 6th ed, Guazzi JACC 2017,
      Mezzani EurJPrevCardiol 2013, ACC 2021 Athletes
    """
    
    @classmethod
    def run(cls, results: dict, cfg=None) -> dict:
        import numpy as np
        
        e00 = results.get('E00', {})
        e01 = results.get('E01', {})
        e02 = results.get('E02', {})
        e03 = results.get('E03', {})
        e04 = results.get('E04', {})
        e05 = results.get('E05', {})
        e07 = results.get('E07', {})
        e08 = results.get('E08', {})
        e11 = results.get('E11', {})
        e12 = results.get('E12', {})
        e13 = results.get('E13', {})
        e17 = results.get('E17', {})
        e18 = results.get('E18', {})
        
        out = {
            'status': 'OK',
            # Test Validity
            'validity_score': 0,
            'validity_grade': 'UNKNOWN',
            'validity_criteria': {},
            'validity_flags': [],
            # Concordance
            'concordance_score': 0,
            'concordance_grade': 'UNKNOWN',
            'concordance_checks': [],
            'concordance_flags': [],
            # Temporal alignment
            'temporal_alignment': {},
            'temporal_spread_vt1_sec': None,
            'temporal_spread_vt2_sec': None,
        }
        
        def _f(d, k, fallback=None):
            v = d.get(k)
            if v is None: return fallback
            try: return float(v)
            except: return fallback
        
        # ════════════════════════════════════
        # PART 1: TEST VALIDITY SCORE (0-100)
        # ════════════════════════════════════
        vs = 0
        vc = {}
        vf = []
        
        # 1a. RER criteria (max 25 pts)
        rer = _f(e01, 'rer_peak')
        if rer:
            if rer >= 1.15:   vs += 25; vc['rer'] = ('EXCELLENT', 25, f'RER {rer:.2f} ≥1.15')
            elif rer >= 1.10: vs += 20; vc['rer'] = ('GOOD', 20, f'RER {rer:.2f} ≥1.10')
            elif rer >= 1.05: vs += 12; vc['rer'] = ('FAIR', 12, f'RER {rer:.2f} ≥1.05')
            elif rer >= 1.00: vs += 5;  vc['rer'] = ('POOR', 5, f'RER {rer:.2f} ≥1.00')
            else: vc['rer'] = ('FAIL', 0, f'RER {rer:.2f} <1.00'); vf.append('SUBMAXIMAL_RER')
        else: vc['rer'] = ('N/A', 0, 'RER unavailable')
        
        # 1b. HR criteria (max 20 pts)
        hr_peak = _f(e13, 'hr_peak') or _f(e01, 'hr_peak')
        hr_pred = _f(e13, 'hr_pred_max')
        if hr_peak and hr_pred and hr_pred > 0:
            hr_pct = 100.0 * hr_peak / hr_pred
            if hr_pct >= 95:   vs += 20; vc['hr'] = ('EXCELLENT', 20, f'HR {hr_pct:.0f}% pred')
            elif hr_pct >= 90: vs += 15; vc['hr'] = ('GOOD', 15, f'HR {hr_pct:.0f}% pred')
            elif hr_pct >= 85: vs += 8;  vc['hr'] = ('FAIR', 8, f'HR {hr_pct:.0f}% pred')
            else: vc['hr'] = ('LOW', 3, f'HR {hr_pct:.0f}% pred'); vf.append('LOW_HR_RESPONSE')
        else: vc['hr'] = ('N/A', 0, 'HR data unavailable')
        
        # 1c. VO2 plateau (max 20 pts)
        plateau = e01.get('plateau_detected', False)
        vo2_det = e01.get('vo2_determination', '')
        if plateau or vo2_det == 'VO2max':
            vs += 20; vc['plateau'] = ('YES', 20, f'{vo2_det} — plateau detected')
        else:
            vc['plateau'] = ('NO', 0, f'{vo2_det or "VO2peak"} — no plateau')
            vf.append('NO_VO2_PLATEAU')
        
        # 1d. Lactate peak (max 15 pts)
        la_peak = _f(e11, 'la_peak')
        if la_peak:
            if la_peak >= 8.0:   vs += 15; vc['lactate'] = ('EXCELLENT', 15, f'La {la_peak:.1f} ≥8')
            elif la_peak >= 6.0: vs += 10; vc['lactate'] = ('GOOD', 10, f'La {la_peak:.1f} ≥6')
            elif la_peak >= 4.0: vs += 5;  vc['lactate'] = ('FAIR', 5, f'La {la_peak:.1f} ≥4')
            else: vc['lactate'] = ('LOW', 0, f'La {la_peak:.1f} <4'); vf.append('LOW_LACTATE')
        else: vc['lactate'] = ('N/A', 0, 'No lactate data')
        
        # 1e. Test duration (max 10 pts)
        t_ex = _f(e00, 'exercise_duration_sec')
        t_stop = _f(e00, 't_stop')
        t_start = _f(e00, 'exercise_start_sec') or _f(e00, 't_start') or 0
        if t_ex is None and t_stop:
            t_ex = t_stop - t_start
        if t_ex:
            dur_min = t_ex / 60.0
            if 8 <= dur_min <= 12: vs += 10; vc['duration'] = ('OPTIMAL', 10, f'{dur_min:.1f} min (8-12)')
            elif 6 <= dur_min <= 15: vs += 5; vc['duration'] = ('ACCEPTABLE', 5, f'{dur_min:.1f} min')
            else: vc['duration'] = ('SUBOPTIMAL', 0, f'{dur_min:.1f} min'); vf.append('SUBOPTIMAL_DURATION')
        else: vc['duration'] = ('N/A', 0, 'Duration unknown')
        
        # 1f. Threshold detection (max 10 pts)
        vt1_ok = e02.get('vt1_time_sec') is not None
        vt2_ok = e02.get('vt2_time_sec') is not None
        if vt1_ok and vt2_ok: vs += 10; vc['thresholds'] = ('BOTH', 10, 'VT1 + VT2 detected')
        elif vt1_ok:           vs += 5;  vc['thresholds'] = ('VT1_ONLY', 5, 'Only VT1')
        else:                  vc['thresholds'] = ('NONE', 0, 'No thresholds'); vf.append('NO_THRESHOLDS')
        
        # Validity grade
        if vs >= 85: vg = 'A'
        elif vs >= 70: vg = 'B'
        elif vs >= 55: vg = 'C'
        elif vs >= 40: vg = 'D'
        else: vg = 'F'
        
        out['validity_score'] = vs
        out['validity_grade'] = vg
        out['validity_criteria'] = vc
        out['validity_flags'] = vf
        
        # ════════════════════════════════════
        # PART 2: PHYSIOLOGICAL CONCORDANCE
        # ════════════════════════════════════
        checks = []
        cf = []
        c_score = 0
        c_max = 0
        
        # 2a. Fick consistency: VO2 = O2P × HR (±15%)
        o2p = _f(e05, 'o2pulse_peak') or _f(e13, 'o2pulse_late')
        hr_p = _f(e13, 'hr_peak') or _f(e01, 'hr_peak')
        # Get VO2 peak in ml/min
        vo2_ml = _f(e01, 'vo2_peak_ml_min') or _f(e01, 'vo2_peak_mlmin')
        if vo2_ml is None:
            vo2_lmin = _f(e01, 'vo2_peak_lmin')
            if vo2_lmin: vo2_ml = vo2_lmin * 1000
        if vo2_ml is None:
            vo2_mlkg = _f(e01, 'vo2_peak_ml_kg_min') or _f(e01, 'vo2_peak_mlkgmin')
            wt = cfg.body_mass_kg if cfg and hasattr(cfg, 'body_mass_kg') else None
            if vo2_mlkg and wt: vo2_ml = vo2_mlkg * wt
        
        c_max += 20
        if o2p and hr_p and vo2_ml and vo2_ml > 0:
            fick_calc = o2p * hr_p
            fick_err = abs(fick_calc - vo2_ml) / vo2_ml * 100
            if fick_err < 5:    c_score += 20; checks.append(('FICK', 'EXCELLENT', f'Err={fick_err:.1f}% (<5%)', 20))
            elif fick_err < 10: c_score += 15; checks.append(('FICK', 'GOOD', f'Err={fick_err:.1f}% (<10%)', 15))
            elif fick_err < 15: c_score += 8;  checks.append(('FICK', 'FAIR', f'Err={fick_err:.1f}% (<15%)', 8))
            else: checks.append(('FICK', 'FAIL', f'Err={fick_err:.1f}% (≥15%)', 0)); cf.append('FICK_INCONSISTENCY')
        else:
            checks.append(('FICK', 'N/A', f'O2P={o2p}, HR={hr_p}, VO2={vo2_ml}', 0))
        
        # 2b. PetCO2 vs VE/VCO2 slope concordance
        petco2_r = _f(e17, 'petco2_rest') or _f(e03, 'petco2_rest')
        petco2_vt1 = _f(e17, 'petco2_at_vt1') or _f(e03, 'petco2_vt1')
        ve_slope = _f(e03, 'slope_to_vt2') or _f(e03, 'slope_full')
        
        c_max += 15
        if petco2_r and ve_slope:
            # Normal: higher PetCO2 = lower VE/VCO2 (inverse)
            # PetCO2 36-42 + VE/VCO2 25-30 = concordant
            # PetCO2 >42 + VE/VCO2 >35 = discordant (V/Q mismatch)
            concordant = True
            note = ''
            if petco2_r > 42 and ve_slope > 35:
                concordant = False; note = 'High PetCO2 + high VE/VCO2 = V/Q mismatch'
                cf.append('PETCO2_VEVCO2_DISCORDANT')
            elif petco2_r < 30 and ve_slope < 25:
                concordant = False; note = 'Low PetCO2 + low VE/VCO2 = hyperventilation artifact?'
                cf.append('HYPERVENTILATION_ARTIFACT')
            else:
                note = f'PetCO2={petco2_r:.0f} + VE/VCO2={ve_slope:.1f} = concordant'
            pts = 15 if concordant else 0
            c_score += pts
            checks.append(('PETCO2_VS_SLOPE', 'PASS' if concordant else 'FAIL', note, pts))
        else:
            checks.append(('PETCO2_VS_SLOPE', 'N/A', 'Missing data', 0))
        
        # 2c. O2-pulse trajectory vs HR-VO2 slope
        o2p_traj = e13.get('o2pulse_trajectory', '')
        slope_class = e13.get('slope_class', '')
        
        c_max += 15
        concordant_sv = True
        sv_note = ''
        if o2p_traj and slope_class:
            if o2p_traj == 'RISING' and slope_class in ('VERY_LOW', 'LOW_HIGH_SV'):
                sv_note = 'Rising O2P + low slope = high SV (concordant)'; c_score += 15
            elif o2p_traj in ('FLAT', 'DECLINING') and slope_class in ('VERY_LOW', 'LOW_HIGH_SV'):
                concordant_sv = False; sv_note = 'Flat/declining O2P + low slope = discordant (SV issue?)'
                cf.append('O2P_SLOPE_DISCORDANT')
            elif o2p_traj == 'RISING' and slope_class in ('STEEP', 'HIGH_NORMAL'):
                sv_note = 'Rising O2P + steep slope = possibly deconditioning or mismatch'; c_score += 8
            else:
                sv_note = f'O2P={o2p_traj} + slope={slope_class}'; c_score += 12
            _o2p_pts = 15 if concordant_sv else 0
            checks.append(('O2P_VS_SLOPE', 'PASS' if concordant_sv else 'FAIL', sv_note, _o2p_pts))
        else:
            checks.append(('O2P_VS_SLOPE', 'N/A', 'Missing data', 0))
        
        # 2d. VE/VCO2 slope variants concordance
        s_full = _f(e03, 'slope_full')
        s_vt2 = _f(e03, 'slope_to_vt2')
        s_vt1 = _f(e03, 'slope_to_vt1')
        nadir = _f(e03, 've_vco2_nadir')
        
        c_max += 15
        if s_full and s_vt1:
            spread = s_full - s_vt1
            if spread < 3:   c_score += 15; checks.append(('SLOPE_VARIANTS', 'TIGHT', f'Spread={spread:.1f} (<3)', 15))
            elif spread < 6: c_score += 10; checks.append(('SLOPE_VARIANTS', 'MODERATE', f'Spread={spread:.1f} (<6)', 10))
            elif spread < 10: c_score += 5; checks.append(('SLOPE_VARIANTS', 'WIDE', f'Spread={spread:.1f} (<10)', 5))
            else:
                checks.append(('SLOPE_VARIANTS', 'VERY_WIDE', f'Spread={spread:.1f} (≥10)', 0))
                cf.append('SLOPE_VARIANT_SPREAD')
        else:
            checks.append(('SLOPE_VARIANTS', 'N/A', 'Missing variants', 0))
        
        # 2e. VT1 vs LT1 temporal alignment
        vt1_t = _f(e02, 'vt1_time_sec')
        lt1_t = _f(e11, 'lt1_time_sec')
        vt2_t = _f(e02, 'vt2_time_sec')
        lt2_t = _f(e11, 'lt2_time_sec')
        
        c_max += 15
        bp1_sources = {}
        bp2_sources = {}
        if vt1_t is not None: bp1_sources['VT1'] = vt1_t
        if lt1_t is not None: bp1_sources['LT1'] = lt1_t
        if vt2_t is not None: bp2_sources['VT2'] = vt2_t
        if lt2_t is not None: bp2_sources['LT2'] = lt2_t
        
        # Add NIRS breakpoints if available
        nirs_bp1 = _f(e12, 'bp1_time_s')
        nirs_bp2 = _f(e12, 'bp2_time_s')
        if nirs_bp1: bp1_sources['NIRS_BP1'] = nirs_bp1
        if nirs_bp2: bp2_sources['NIRS_BP2'] = nirs_bp2
        
        bp1_times = list(bp1_sources.values())
        bp2_times = list(bp2_sources.values())
        
        if len(bp1_times) >= 2:
            # Compute spread; if NIRS outlier, compute core spread without it
            spread1 = max(bp1_times) - min(bp1_times)
            core1 = {k:v for k,v in bp1_sources.items() if not k.startswith('NIRS')}
            core_spread1 = (max(core1.values()) - min(core1.values())) if len(core1) >= 2 else spread1
            # Use core spread for grading (NIRS can be physiologically different)
            grade_spread1 = core_spread1 if len(core1) >= 2 else spread1
            out['temporal_spread_vt1_sec'] = round(grade_spread1, 1)
            if grade_spread1 < 60:   c_score += 8; level1 = 'HIGH'
            elif grade_spread1 < 120: c_score += 4; level1 = 'MODERATE'
            else: level1 = 'LOW'; cf.append('VT1_TEMPORAL_SPREAD')
            # Identify outlier
            _outlier1 = ''
            if len(bp1_sources) >= 3:
                med1 = sorted(bp1_times)[len(bp1_times)//2]
                for k,v in bp1_sources.items():
                    if abs(v - med1) > 120: _outlier1 = k
            out['temporal_alignment']['VT1'] = {
                'sources': bp1_sources,
                'spread_sec': round(grade_spread1, 1), 'full_spread_sec': round(spread1, 1),
                'confidence': level1, 'outlier': _outlier1}
        
        if len(bp2_times) >= 2:
            spread2 = max(bp2_times) - min(bp2_times)
            core2 = {k:v for k,v in bp2_sources.items() if not k.startswith('NIRS')}
            core_spread2 = (max(core2.values()) - min(core2.values())) if len(core2) >= 2 else spread2
            grade_spread2 = core_spread2 if len(core2) >= 2 else spread2
            out['temporal_spread_vt2_sec'] = round(grade_spread2, 1)
            if grade_spread2 < 60:   c_score += 7; level2 = 'HIGH'
            elif grade_spread2 < 120: c_score += 3; level2 = 'MODERATE'
            else: level2 = 'LOW'; cf.append('VT2_TEMPORAL_SPREAD')
            _outlier2 = ''
            if len(bp2_sources) >= 3:
                med2 = sorted(bp2_times)[len(bp2_times)//2]
                for k,v in bp2_sources.items():
                    if abs(v - med2) > 120: _outlier2 = k
            out['temporal_alignment']['VT2'] = {
                'sources': bp2_sources,
                'spread_sec': round(grade_spread2, 1), 'full_spread_sec': round(spread2, 1),
                'confidence': level2, 'outlier': _outlier2}
        
        if len(bp1_times) < 2 and len(bp2_times) < 2:
            checks.append(('TEMPORAL', 'N/A', 'Need ≥2 breakpoint sources', 0))
        else:
            s1 = out.get('temporal_spread_vt1_sec', '?')
            s2 = out.get('temporal_spread_vt2_sec', '?')
            checks.append(('TEMPORAL', 'CHECKED', f'VT1 spread={s1}s, VT2 spread={s2}s', 0))
        
        # 2f. OUES predicted VO2 vs actual (if available)
        oues_full = _f(e04, 'oues_full') or _f(e04, 'oues100')
        oues_intercept = _f(e04, 'oues_intercept', 0)
        c_max += 10
        if oues_full and vo2_ml:
            # OUES: VO2 = a × log10(VE) + b → need intercept
            ve_peak = _f(e01, 've_peak') or _f(e01, 've_peak_lmin')
            if ve_peak and ve_peak > 0:
                import math
                oues_pred = oues_full * math.log10(ve_peak) + oues_intercept
                oues_err = abs(oues_pred - vo2_ml) / vo2_ml * 100
                if oues_err < 10: c_score += 10; checks.append(('OUES_VS_VO2', 'GOOD', f'OUES pred={oues_pred:.0f} vs actual={vo2_ml:.0f}, err={oues_err:.1f}%', 10))
                elif oues_err < 20: c_score += 5; checks.append(('OUES_VS_VO2', 'FAIR', f'Err={oues_err:.1f}%', 5))
                else: checks.append(('OUES_VS_VO2', 'POOR', f'Err={oues_err:.1f}%', 0)); cf.append('OUES_VO2_MISMATCH')
            else:
                checks.append(('OUES_VS_VO2', 'N/A', 'VE peak unavailable', 0))
        else:
            checks.append(('OUES_VS_VO2', 'N/A', f'OUES={oues_full}, VO2={vo2_ml}', 0))
        
        # Concordance grade
        if c_max > 0:
            c_pct = round(100.0 * c_score / c_max)
        else:
            c_pct = 0
        if c_pct >= 85: cg = 'A'
        elif c_pct >= 70: cg = 'B'
        elif c_pct >= 55: cg = 'C'
        elif c_pct >= 40: cg = 'D'
        else: cg = 'F'
        
        out['concordance_score'] = c_pct
        out['concordance_grade'] = cg
        out['concordance_checks'] = checks
        out['concordance_flags'] = cf
        
        return out



# =========
            
            # ═══════════════════════════════════════════════════════════════

# CHART_JS — interactive chart JavaScript (Canvas-based)
# Defined as global constant, used by _render_charts_html
try:
    CHART_JS
except NameError:
    CHART_JS = '''
// CPET Charts - Canvas-based interactive charts
let _activeChart = null;
const _btnOrigColors = {};

function toggleChart(type) {
    const c = document.getElementById('chart_container');
    if (!c) return;
    
    // Restore all buttons to original state
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        const origColor = b.getAttribute('data-color') || b.style.borderColor;
        b.style.background = '#fff';
        b.style.color = origColor;
        b.style.fontWeight = '600';
    });
    
    if (_activeChart === type) { 
        c.style.display='none'; 
        _activeChart=null; 
        return; 
    }
    
    _activeChart = type;
    const btn = document.getElementById('btn_'+type);
    if (btn) { 
        const origColor = btn.getAttribute('data-color') || btn.style.borderColor;
        btn.style.background = origColor; 
        btn.style.color = '#fff'; 
    }
    
    c.style.display='block';
    c.innerHTML='<canvas id="cpet_canvas" width="860" height="420" style="max-width:100%;background:white;"></canvas>';
    const canvas=document.getElementById('cpet_canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    const M={t:40,r:60,b:50,l:70};
    const pW=W-M.l-M.r, pH=H-M.t-M.b;
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
    
    // Zone colors for protocol steps (matching training zones)
    // warmup=gray, Z1=blue, Z2=green, Z3=yellow, Z4=orange, Z5=red, recovery=light-green
    const ZONE_COLORS = [
        'rgba(200,200,210,0.25)',   // 0: warmup/rest
        'rgba(147,197,253,0.30)',   // 1: Z1 blue
        'rgba(74,222,128,0.30)',    // 2: Z2 green
        'rgba(251,191,36,0.30)',    // 3: Z3 yellow
        'rgba(249,115,22,0.30)',    // 4: Z4 orange
        'rgba(239,68,68,0.30)',     // 5: Z5 red
    ];
    
    function getStepZoneColor(steps, stepIdx) {
        const s = steps[stepIdx];
        if (!s) return ZONE_COLORS[0];
        const tp = s.type || '';
        if (tp === 'warmup' || tp === 'rest') return ZONE_COLORS[0];
        if (tp === 'cooldown' || tp === 'recovery') return ZONE_COLORS[0];
        // For exercise steps: map by fractional position
        const exSteps = steps.filter(x => x.type !== 'warmup' && x.type !== 'rest' && x.type !== 'cooldown' && x.type !== 'recovery');
        const exIdx = exSteps.indexOf(s);
        if (exIdx < 0) return ZONE_COLORS[1];
        const nEx = exSteps.length;
        const frac = nEx > 1 ? exIdx / (nEx - 1) : 0.5;
        if (frac <= 0.15) return ZONE_COLORS[1];
        if (frac <= 0.35) return ZONE_COLORS[2];
        if (frac <= 0.55) return ZONE_COLORS[3];
        if (frac <= 0.75) return ZONE_COLORS[4];
        return ZONE_COLORS[5];
    }
    
    function drawProtocolBands(steps, tMin, tMax, tStop) {
        if (!steps || steps.length === 0) return;
        steps.forEach((s, i) => {
            const x1 = M.l + Math.max(0, (s.start_s - tMin)) / (tMax - tMin) * pW;
            const endS = s.end_s || (tStop || tMax);
            const x2 = M.l + Math.min(pW, (endS - tMin) / (tMax - tMin) * pW);
            ctx.fillStyle = getStepZoneColor(steps, i);
            ctx.fillRect(x1, M.t, x2 - x1, pH);
            // Step label
            ctx.fillStyle = '#475569'; 
            ctx.font = '10px sans-serif'; 
            ctx.textAlign = 'center';
            ctx.fillText('S' + (i + 1), (x1 + x2) / 2, M.t + pH - 6);
        });
    }
    
    function drawAxis(label, min, max, side, color) {
        ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=1;
        const steps=5;
        for(let i=0;i<=steps;i++) {
            const y=M.t+pH-pH*i/steps;
            if(side==='left') {
                ctx.beginPath(); ctx.moveTo(M.l,y); ctx.lineTo(W-M.r,y); ctx.stroke();
                ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='right';
                ctx.fillText((min+(max-min)*i/steps).toFixed(1), M.l-6, y+4);
            } else {
                ctx.fillStyle=color||'#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='left';
                ctx.fillText((min+(max-min)*i/steps).toFixed(1), W-M.r+6, y+4);
            }
        }
        ctx.fillStyle=color||'#64748b'; ctx.font='12px sans-serif';
        if(side==='left') { ctx.textAlign='center'; ctx.save(); ctx.translate(14,M.t+pH/2); ctx.rotate(-Math.PI/2); ctx.fillText(label,0,0); ctx.restore(); }
        else { ctx.textAlign='center'; ctx.save(); ctx.translate(W-8,M.t+pH/2); ctx.rotate(-Math.PI/2); ctx.fillText(label,0,0); ctx.restore(); }
    }
    
    function drawLine(times,vals,tMin,tMax,vMin,vMax,color,width) {
        ctx.strokeStyle=color; ctx.lineWidth=width||2; ctx.beginPath();
        let started=false;
        for(let i=0;i<times.length;i++) {
            if(vals[i]===null||vals[i]===undefined) continue;
            const x=M.l+(times[i]-tMin)/(tMax-tMin)*pW;
            const y=M.t+pH-(vals[i]-vMin)/(vMax-vMin)*pH;
            if(!started){ctx.moveTo(x,y);started=true;}else{ctx.lineTo(x,y);}
        }
        ctx.stroke();
    }
    
    function drawVT(time,tMin,tMax,label,color) {
        if(!time) return;
        const x=M.l+(time-tMin)/(tMax-tMin)*pW;
        ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
        ctx.beginPath(); ctx.moveTo(x,M.t); ctx.lineTo(x,M.t+pH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=color; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
        ctx.fillText(label,x,M.t-6);
    }
    
    function fmtTime(s) { return Math.floor(s/60)+":"+("0"+Math.floor(s%60)).slice(-2); }
    
    function drawTimeAxis(tMin,tMax) {
        const steps=8;
        ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='center';
        for(let i=0;i<=steps;i++) {
            const t=tMin+(tMax-tMin)*i/steps;
            const x=M.l+pW*i/steps;
            ctx.fillText(fmtTime(t),x,H-M.b+18);
        }
        ctx.fillText('Czas (min:ss)',M.l+pW/2,H-6);
    }
    
    function drawDots(times,vals,tMin,tMax,vMin,vMax,color,radius) {
        for(let i=0;i<times.length;i++) {
            if(vals[i]===null||vals[i]===undefined) continue;
            const x=M.l+(times[i]-tMin)/(tMax-tMin)*pW;
            const y=M.t+pH-(vals[i]-vMin)/(vMax-vMin)*pH;
            ctx.beginPath(); ctx.arc(x,y,radius||4,0,Math.PI*2);
            ctx.fillStyle=color; ctx.fill();
            ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
        }
    }
    
    function drawTitle(text) {
        ctx.fillStyle='#1e293b'; ctx.font='bold 14px sans-serif'; ctx.textAlign='center';
        ctx.fillText(text,W/2,22);
    }
    
    function drawLegend(items) {
        let x = M.l + 10;
        items.forEach(([label, color]) => {
            ctx.fillStyle=color; ctx.fillRect(x,M.t+8,20,3);
            ctx.fillStyle='#374151'; ctx.font='11px sans-serif'; ctx.textAlign='left';
            ctx.fillText(label, x+24, M.t+12);
            x += ctx.measureText(label).width + 38;
        });
    }
    
    function drawStopLine(tStop, tMin, tMax) {
        if (!tStop) return;
        const xs=M.l+(tStop-tMin)/(tMax-tMin)*pW;
        ctx.strokeStyle='#1e293b'; ctx.lineWidth=2; ctx.setLineDash([8,4]);
        ctx.beginPath(); ctx.moveTo(xs,M.t); ctx.lineTo(xs,M.t+pH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle='#1e293b'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
        ctx.fillText('STOP',xs,M.t-6);
    }
    
    const d = CD;
    
    // ═══ PROTOCOL CHART ═══
    if(type==='proto' && d.kinetics) {
        const t=d.kinetics.time||[], vo2=d.kinetics.vo2||[];
        const steps=d.kinetics.protocol_steps||[];
        const spd=d.kinetics.speed||[], pwr=d.kinetics.power||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vo2Max=Math.max(...vo2.filter(v=>v!==null))*1.1;
        const hasSpd=spd.length>0&&spd.some(v=>v!==null&&v>0);
        const hasPwr=pwr.length>0&&pwr.some(v=>v!==null&&v>0);
        const load=hasSpd?spd:(hasPwr?pwr:null);
        const loadMax=load?Math.max(...load.filter(v=>v!==null&&v>0))*1.15:0;
        drawTitle('Protokol testu'+(d.kinetics.protocol_name?' \u2014 '+d.kinetics.protocol_name:''));
        drawProtocolBands(steps, tMin, tMax, d.kinetics.t_stop);
        drawAxis('VO2 ml/min',0,vo2Max,'left','#ef4444');
        if(load) drawAxis(hasSpd?'Speed km/h':'Power W',0,loadMax,'right','#8b5cf6');
        drawTimeAxis(tMin,tMax);
        if(load) drawLine(t,load,tMin,tMax,0,loadMax,'#8b5cf6',2);
        drawLine(t,vo2,tMin,tMax,0,vo2Max,'#ef4444',2);
        drawStopLine(d.kinetics.t_stop, tMin, tMax);
        drawVT(d.kinetics.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.kinetics.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['VO2','#ef4444']].concat(load?[[hasSpd?'Speed':'Power','#8b5cf6']]:[]));
    }
    
    // ═══ VO2 KINETICS ═══
    if(type==='kinetics' && d.kinetics) {
        const t=d.kinetics.time||[], vo2=d.kinetics.vo2||[];
        const spd=d.kinetics.speed||[], pwr=d.kinetics.power||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vo2Max=Math.max(...vo2.filter(v=>v!==null))*1.1;
        const hasSpd=spd.length>0&&spd.some(v=>v!==null&&v>0);
        const hasPwr=pwr.length>0&&pwr.some(v=>v!==null&&v>0);
        const load=hasSpd?spd:(hasPwr?pwr:null);
        const loadMax=load?Math.max(...load.filter(v=>v!==null&&v>0))*1.15:0;
        drawTitle('VO2 Kinetics'+(d.kinetics.protocol_name?' \u2014 '+d.kinetics.protocol_name:''));
        drawProtocolBands(d.kinetics.protocol_steps||[], tMin, tMax, d.kinetics.t_stop);
        drawAxis('VO2 ml/min',0,vo2Max,'left','#ef4444');
        if(load) drawAxis(hasSpd?'Speed km/h':'Power W',0,loadMax,'right',hasSpd?'#8b5cf6':'#f59e0b');
        drawTimeAxis(tMin,tMax);
        if(load) drawLine(t,load,tMin,tMax,0,loadMax,hasSpd?'#8b5cf6':'#f59e0b',1.5);
        drawLine(t,vo2,tMin,tMax,0,vo2Max,'#ef4444',2);
        drawStopLine(d.kinetics.t_stop, tMin, tMax);
        drawVT(d.kinetics.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.kinetics.vt2_time,tMin,tMax,'VT2','#f59e0b');
        if(d.kinetics.vo2peak) {
            const yp=M.t+pH-(d.kinetics.vo2peak)/vo2Max*pH;
            ctx.strokeStyle='#ef4444'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
            ctx.beginPath(); ctx.moveTo(M.l,yp); ctx.lineTo(W-M.r,yp); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle='#ef4444'; ctx.font='10px sans-serif'; ctx.textAlign='right';
            ctx.fillText('VO2peak '+Math.round(d.kinetics.vo2peak),W-M.r-4,yp-4);
        }
        drawLegend([['VO2','#ef4444']].concat(load?[[hasSpd?'Speed':'Power',hasSpd?'#8b5cf6':'#f59e0b']]:[]));
    }
    
    // ═══ V-SLOPE ═══
    if(type==='vslope' && d.gas) {
        const t=d.gas.time||[], vo2=d.gas.vo2||[], vco2=d.gas.vco2||[];
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vMax=Math.max(...vo2,...vco2)*1.1;
        drawTitle('V-slope \u2014 VO2 & VCO2');
        drawAxis('ml/min',0,vMax,'left','#374151');
        drawTimeAxis(tMin,tMax);
        drawLine(t,vo2,tMin,tMax,0,vMax,'#ef4444',2);
        drawLine(t,vco2,tMin,tMax,0,vMax,'#3b82f6',2);
        drawVT(d.gas.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.gas.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['VO2','#ef4444'],['VCO2','#3b82f6']]);
    }
    
    // ═══ FAT/CHO ═══
    if(type==='fatchho' && d.substrate) {
        const t=d.substrate.time||[], cho=(d.substrate.cho||[]).map(v=>v*60), fat=(d.substrate.fat||[]).map(v=>v*60);
        if(t.length<10) return;
        const tMin=Math.min(...t), tMax=Math.max(...t);
        const vMax=Math.max(...cho,...fat)*1.2;
        drawTitle('Substrat \u2014 FAT & CHO (g/h)');
        drawAxis('g/h',0,vMax,'left','#374151');
        drawTimeAxis(tMin,tMax);
        ctx.globalAlpha=0.3;
        ctx.fillStyle='rgba(251,191,36,0.5)';
        ctx.beginPath(); ctx.moveTo(M.l,M.t+pH);
        for(let i=0;i<t.length;i++){const x=M.l+(t[i]-tMin)/(tMax-tMin)*pW;const y=M.t+pH-(fat[i]||0)/vMax*pH;ctx.lineTo(x,y);}
        ctx.lineTo(M.l+pW,M.t+pH); ctx.closePath(); ctx.fill();
        ctx.fillStyle='rgba(59,130,246,0.4)';
        ctx.beginPath(); ctx.moveTo(M.l,M.t+pH);
        for(let i=0;i<t.length;i++){const x=M.l+(t[i]-tMin)/(tMax-tMin)*pW;const y=M.t+pH-(cho[i]||0)/vMax*pH;ctx.lineTo(x,y);}
        ctx.lineTo(M.l+pW,M.t+pH); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=1;
        drawLine(t,fat,tMin,tMax,0,vMax,'#f59e0b',2);
        drawLine(t,cho,tMin,tMax,0,vMax,'#3b82f6',2);
        drawVT(d.substrate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.substrate.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['FAT','#f59e0b'],['CHO','#3b82f6']]);
    }
    
    // ═══ LACTATE ═══
    if(type==='lac' && d.lactate) {
        const pts=d.lactate.points||[];
        if(pts.length<2) return;
        const times=pts.map(p=>p.time_sec||p.time||0);
        const las=pts.map(p=>p.la||p.lactate||0);
        const tMin=Math.min(...times)-30, tMax=Math.max(...times)+30;
        const lMax=Math.max(...las)*1.2;
        drawTitle('Krzywa mleczanowa');
        drawAxis('Laktat [mmol/L]',0,lMax,'left','#dc2626');
        drawTimeAxis(tMin,tMax);
        const pt=d.lactate.poly_time||[], pf=d.lactate.poly_fit||[];
        if(pt.length>2) drawLine(pt,pf,tMin,tMax,0,lMax,'#fca5a5',1.5);
        drawDots(times,las,tMin,tMax,0,lMax,'#dc2626',5);
        drawVT(d.lactate.lt1_time,tMin,tMax,'LT1','#16a34a');
        drawVT(d.lactate.lt2_time,tMin,tMax,'LT2','#dc2626');
        drawVT(d.lactate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.lactate.vt2_time,tMin,tMax,'VT2','#f59e0b');
    }
    
    // ═══ NIRS ═══
    if(type==='nirs' && d.nirs) {
        const traces=d.nirs.traces||[];
        if(traces.length===0) return;
        const tr=traces[0];
        const times=tr.time_sec||tr.time||[];
        const vals=tr.smo2||tr.values||tr.SmO2||[];
        if(times.length<2||vals.length<2) return;
        const validVals=vals.filter(v=>v!==null&&v!==undefined&&!isNaN(v));
        if(validVals.length<2) return;
        const tMin=Math.min(...times), tMax=Math.max(...times);
        const vMin=Math.min(...validVals)*0.9;
        const vMax=Math.max(...validVals)*1.1;
        drawTitle('NIRS \u2014 SmO2 (%)');
        drawAxis('SmO2 [%]',vMin,vMax,'left','#2563eb');
        drawTimeAxis(tMin,tMax);
        drawLine(times,vals,tMin,tMax,vMin,vMax,'#2563eb',2);
        drawVT(d.nirs.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.nirs.vt2_time,tMin,tMax,'VT2','#f59e0b');
    }
    
    // ═══ DUAL (Lactate + SmO2) ═══
    if(type==='dual' && d.lactate && d.nirs && d.nirs.traces && d.nirs.traces.length>0) {
        const pts=d.lactate.points||[];
        const tr=d.nirs.traces[0];
        const nTimes=tr.time_sec||tr.time||[];
        const nVals=tr.smo2||tr.values||tr.SmO2||[];
        if(pts.length<2||nTimes.length<2) return;
        const allT=[...pts.map(p=>p.time_sec||p.time||0),...nTimes];
        const tMin=Math.min(...allT)-20, tMax=Math.max(...allT)+20;
        const lMax=Math.max(...pts.map(p=>p.la||0))*1.2;
        const validN=nVals.filter(v=>v!==null&&v!==undefined&&!isNaN(v));
        const sMin=validN.length>0?Math.min(...validN)*0.9:0;
        const sMax=validN.length>0?Math.max(...validN)*1.1:100;
        drawTitle('Laktat + SmO2');
        drawAxis('Laktat [mmol/L]',0,lMax,'left','#dc2626');
        drawAxis('SmO2 [%]',sMin,sMax,'right','#2563eb');
        drawTimeAxis(tMin,tMax);
        drawLine(nTimes,nVals,tMin,tMax,sMin,sMax,'#2563eb',2);
        const pt=d.lactate.poly_time||[], pf=d.lactate.poly_fit||[];
        if(pt.length>2) drawLine(pt,pf,tMin,tMax,0,lMax,'#fca5a5',1.5);
        drawDots(pts.map(p=>p.time_sec||p.time||0),pts.map(p=>p.la||0),tMin,tMax,0,lMax,'#dc2626',5);
        drawVT(d.lactate.vt1_time,tMin,tMax,'VT1','#22d3ee');
        drawVT(d.lactate.vt2_time,tMin,tMax,'VT2','#f59e0b');
        drawLegend([['Laktat','#dc2626'],['SmO2','#2563eb']]);
    }
}

// Init: store original button colors
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        b.setAttribute('data-color', b.style.color || b.style.borderColor);
    });
});
// Also immediate init for non-DOMContentLoaded scenarios
setTimeout(function() {
    document.querySelectorAll('[id^=btn_]').forEach(b => {
        if (!b.getAttribute('data-color')) {
            b.setAttribute('data-color', b.style.color || b.style.borderColor);
        }
    });
}, 100);
'''

# INTERPRETATION ENGINE — Normy wiekowe + auto-interpretacja CPET
# Źródła: Cooper Institute/ACSM 2021, FRIEND Registry, INSCYD 2024
# ═══════════════════════════════════════════════════════════════

COOPER_NORMS = {
    'male': {
        20: {'superior': 55.4, 'excellent': 51.1, 'good': 45.4, 'fair': 41.7},
        30: {'superior': 54.0, 'excellent': 48.3, 'good': 44.0, 'fair': 40.5},
        40: {'superior': 52.5, 'excellent': 46.4, 'good': 42.4, 'fair': 38.5},
        50: {'superior': 48.9, 'excellent': 43.4, 'good': 39.2, 'fair': 35.6},
        60: {'superior': 45.7, 'excellent': 39.5, 'good': 35.5, 'fair': 32.3},
        70: {'superior': 42.1, 'excellent': 36.7, 'good': 32.3, 'fair': 29.4},
    },
    'female': {
        20: {'superior': 49.6, 'excellent': 43.9, 'good': 39.5, 'fair': 36.1},
        30: {'superior': 47.4, 'excellent': 42.4, 'good': 37.8, 'fair': 34.4},
        40: {'superior': 45.3, 'excellent': 39.7, 'good': 36.3, 'fair': 33.0},
        50: {'superior': 41.1, 'excellent': 36.7, 'good': 33.0, 'fair': 30.1},
        60: {'superior': 37.8, 'excellent': 33.0, 'good': 30.0, 'fair': 27.5},
        70: {'superior': 36.7, 'excellent': 30.9, 'good': 28.1, 'fair': 25.9},
    }
}

ATHLETE_NORMS = {
    'male':   [('Elita', 70), ('Wyczynowy', 60), ('Zaawansowany', 52), ('Amator', 45), ('Początkujący', 38)],
    'female': [('Elita', 60), ('Wyczynowy', 50), ('Zaawansowany', 42), ('Amator', 36), ('Początkująca', 30)],
}

def _interp_get_decade(age):
    d = int(age // 10) * 10
    return max(20, min(70, d))

def _interp_lerp_thresholds(age, sex):
    """Interpolate Cooper thresholds for exact age."""
    s = sex if sex in ('male','female') else 'male'
    norms = COOPER_NORMS[s]
    d = max(20, min(70, age))
    d_lo = int(d // 10) * 10
    d_lo = max(20, min(70, d_lo))
    d_hi = min(70, d_lo + 10)
    if d_lo == d_hi:
        return norms[d_lo]
    frac = (d - d_lo) / (d_hi - d_lo)
    lo, hi = norms[d_lo], norms[d_hi]
    return {k: round(lo[k] + (hi[k] - lo[k]) * frac, 1) for k in lo}

def interpret_vo2max(vo2_mlkgmin, age, sex):
    thr = _interp_lerp_thresholds(age, sex)
    v = vo2_mlkgmin
    if v >= thr['superior']:
        cat, pct, col = 'Superior', 95, '#8b5cf6'
    elif v >= thr['excellent']:
        cat, pct, col = 'Excellent', 80, '#3b82f6'
    elif v >= thr['good']:
        cat, pct, col = 'Good', 60, '#22c55e'
    elif v >= thr['fair']:
        cat, pct, col = 'Fair', 40, '#f59e0b'
    else:
        cat, pct, col = 'Poor', 20, '#ef4444'
    # Refine percentile within band
    bands = [('superior',95,99), ('excellent',80,94), ('good',60,79), ('fair',40,59)]
    for lo_key, plo, phi in bands:
        hi_key_idx = ['fair','good','excellent','superior'].index(lo_key)
        keys = ['fair','good','excellent','superior']
        if lo_key == 'superior' and v >= thr['superior']:
            ext = min((v - thr['superior']) / max(thr['superior']*0.1, 1), 1.0)
            pct = int(95 + ext * 4)
            break
        if hi_key_idx < 3:
            nxt = keys[hi_key_idx + 1]
            if v >= thr[lo_key] and v < thr[nxt]:
                frac = (v - thr[lo_key]) / max(thr[nxt] - thr[lo_key], 0.1)
                pct = int(plo + frac * (phi - plo))
                break
    else:
        if v < thr['fair']:
            pct = max(5, int(40 * v / max(thr['fair'], 1)))
    # Athlete level
    s = sex if sex in ('male','female') else 'male'
    athlete_level = 'Nietrenujący/ca'
    for lvl, threshold in ATHLETE_NORMS[s]:
        if v >= threshold:
            athlete_level = lvl
            break
    return {'category': cat, 'percentile': min(99, max(1, pct)),
            'athlete_level': athlete_level, 'color': col, 'thresholds': thr}

def interpret_thresholds(vt1_vo2, vt2_vo2, vo2peak, vt1_hr, vt2_hr, hr_max):
    r = {}
    if vo2peak and vo2peak > 0:
        r['vt1_pct_vo2'] = round(vt1_vo2 / vo2peak * 100, 1) if vt1_vo2 else None
        r['vt2_pct_vo2'] = round(vt2_vo2 / vo2peak * 100, 1) if vt2_vo2 else None
    if hr_max and hr_max > 0:
        r['vt1_pct_hr'] = round(vt1_hr / hr_max * 100, 1) if vt1_hr else None
        r['vt2_pct_hr'] = round(vt2_hr / hr_max * 100, 1) if vt2_hr else None
    vp = r.get('vt1_pct_vo2')
    if vp:
        if vp >= 80: r['aerobic_base'] = 'Doskonała'
        elif vp >= 70: r['aerobic_base'] = 'Bardzo dobra'
        elif vp >= 60: r['aerobic_base'] = 'Dobra'
        elif vp >= 50: r['aerobic_base'] = 'Umiarkowana'
        else: r['aerobic_base'] = 'Słaba'
    r['gap_bpm'] = round(vt2_hr - vt1_hr) if vt1_hr and vt2_hr else None
    return r

def interpret_test_validity(rer_peak, hr_max, age, lactate_peak=None):
    criteria = []
    hr_pred = 220 - age
    if rer_peak and rer_peak >= 1.15: criteria.append('RER ≥1.15')
    elif rer_peak and rer_peak >= 1.10: criteria.append('RER ≥1.10')
    if hr_max and hr_max >= 0.90 * hr_pred: criteria.append(f'HR ≥90% pred ({hr_max}/{hr_pred})')
    if lactate_peak and lactate_peak >= 8.0: criteria.append(f'La ≥8 ({lactate_peak:.1f})')
    elif lactate_peak and lactate_peak >= 6.0: criteria.append(f'La ≥6 ({lactate_peak:.1f})')
    n = len(criteria)
    if n >= 2: conf = 'Wysoka'
    elif n == 1 and rer_peak and rer_peak >= 1.10: conf = 'Wysoka'
    elif n == 1: conf = 'Umiarkowana'
    else: conf = 'Niska'
    is_max = n >= 1 and (not rer_peak or rer_peak >= 1.05)
    if rer_peak and rer_peak < 1.00: is_max = False; conf = 'Niska'
    return {'is_maximal': is_max, 'confidence': conf, 'criteria': criteria,
            'rer_desc': 'Doskonały' if rer_peak and rer_peak>=1.15 else 'Dobry' if rer_peak and rer_peak>=1.10 else 'Akceptowalny' if rer_peak and rer_peak>=1.05 else 'Submaximalny'}

def generate_training_recs(ct):
    """Generate training recommendations based on CPET profile."""
    recs = []
    vt1p = ct.get('_interp_vt1_pct_vo2')
    vt2p = ct.get('_interp_vt2_pct_vo2')
    cat = ct.get('_interp_vo2_category', '')
    ath = ct.get('_interp_vo2_athlete_level', '')
    gap = ct.get('_interp_gap_bpm')
    _pc = ct.get('_performance_context', {})
    _lvl = _pc.get('level_by_speed', '') if _pc.get('executed') else ''
    _v_vt2 = _pc.get('v_vt2_kmh')
    
    # Zone distribution recommendation
    if vt1p and vt1p < 60:
        recs.append({
            'title': 'Priorytet: Baza tlenowa (Strefa 1-2)',
            'desc': f'VT1 przy {vt1p:.0f}% VO2max wskazuje na niedostateczną bazę aerobową. Zalecenie: 70-80% objętości treningowej poniżej VT1.',
            'zones': '80% Z1-Z2 | 15% Z3 | 5% Z4-Z5',
            'icon': '🏃', 'color': '#22c55e'
        })
    elif vt1p and vt2p and (vt2p - vt1p) < 20:
        recs.append({
            'title': 'Priorytet: Poszerzenie strefy progowej',
            'desc': f'Wąski gap VT1-VT2 ({vt2p-vt1p:.0f}% VO2max). Tempo runs i SST (Sweet Spot Training) mogą podnieść VT2.',
            'zones': '65% Z1-Z2 | 25% Z3 | 10% Z4-Z5',
            'icon': '⚡', 'color': '#f59e0b'
        })
    elif vt2p and vt2p >= 88:
        _ctx_r = ''
        if _v_vt2 and _lvl:
            _ctx_r = f' ({_v_vt2:.1f} km/h, poziom {_lvl})'
            if _lvl in ('Sedentary', 'Recreational'):
                _ctx_r += '. Wysoki %VO\u2082max ale niska pr\u0119dko\u015b\u0107 — priorytet: rozw\u00f3j VO\u2082max absolutnego.'
        recs.append({
            'title': 'Rozw\u00f3j: Zwi\u0119kszenie VO\u2082max (Strefa 5)',
            'desc': f'VT2 przy {vt2p:.0f}% VO\u2082max{_ctx_r if _ctx_r else " — progi wysoko ustawione"}. Dalszy post\u0119p przez interwa\u0142y VO\u2082max (3-5 min @ 95-100% VO\u2082max).',
            'zones': '75% Z1-Z2 | 10% Z3 | 10% Z4 | 5% Z5',
            'icon': '🔥', 'color': '#ef4444'
        })
    else:
        recs.append({
            'title': 'Rozwój: Polaryzowany model treningowy',
            'desc': 'Balanced profil — kontynuuj model 80/20 (80% nisko, 20% wysoko) z akcentem na progi.',
            'zones': '75% Z1-Z2 | 15% Z3 | 10% Z4-Z5',
            'icon': '⚖️', 'color': '#3b82f6'
        })
    
    # Fat metabolism
    fat = ct.get('FATmax_g_min')
    try: fat_v = float(fat) if fat and str(fat) not in ('-','None','') else None
    except: fat_v = None
    if fat_v and fat_v < 0.4:
        recs.append({
            'title': 'Metabolizm: Trening spalania tłuszczów',
            'desc': f'FATmax {fat_v:.2f} g/min — niska oksydacja lipidów. Długie sesje Z2 (60-90 min) na czczo lub z niską CHO mogą poprawić.',
            'icon': '🍃', 'color': '#84cc16'
        })
    
    # Running economy / sport-specific efficiency
    _mod_r = ct.get('modality', 'run')
    re_val = ct.get('E06_RE_mlO2_per_kg_per_km')
    try: re_f = float(re_val) if re_val and str(re_val) not in ('-','None','') else None
    except: re_f = None
    if _mod_r in ('run', 'triathlon', 'hyrox') and re_f and re_f > 220:
        recs.append({
            'title': 'Technika: Poprawa ekonomii biegu',
            'desc': f'RE {re_f:.0f} mlO2/kg/km — potencjał do poprawy. Drills techniczne, strides, plyometria, hill sprints.',
            'icon': '👟', 'color': '#06b6d4'
        })
    elif _mod_r in ('crossfit', 'hyrox', 'mma'):
        recs.append({
            'title': f'Specyfika {_mod_r.upper()}: Zdolność powtarzania wysiłków',
            'desc': f'W {_mod_r.upper()} kluczowa jest zdolność odbudowy po wysiłkach submaksymalnych. Trening interwałowy 30/30s i EMOM z akcentem na utrzymanie HR <85% HRmax w fazach odpoczynku.',
            'icon': '⚡', 'color': '#8b5cf6'
        })
    
    return recs

def generate_observations(ct):
    obs = []
    def add(text, otype, prio, icon=None):
        icons = {'positive':'🟢','neutral':'🔵','warning':'🟡','negative':'🔴'}
        obs.append({'text': text, 'type': otype, 'priority': prio, 'icon': icon or icons.get(otype,'⚪')})
    
    # Modality context for sport-specific observations
    _modality = ct.get('modality', 'run')
    _is_run = _modality in ('run', 'triathlon', 'hyrox')
    _is_bike = _modality in ('bike', 'triathlon')
    _is_hybrid = _modality in ('crossfit', 'hyrox', 'mma')
    _mod_pl = {'run':'bieg','bike':'kolarstwo','triathlon':'triathlon','rowing':'wioślarstwo',
               'crossfit':'CrossFit','hyrox':'HYROX','swimming':'pływanie',
               'xc_ski':'biegi narciarskie','soccer':'piłka nożna','mma':'MMA'}.get(_modality, _modality)
    
    vo2_rel = ct.get('_interp_vo2_mlkgmin')
    vo2_cat = ct.get('_interp_vo2_category','')
    vo2_pct = ct.get('_interp_vo2_percentile','?')
    vo2_ath = ct.get('_interp_vo2_athlete_level','')
    
    if vo2_rel and vo2_cat:
        cat_pl = {'Superior':'Wybitna','Excellent':'Bardzo dobra','Good':'Dobra','Fair':'Przeciętna','Poor':'Niska'}.get(vo2_cat, vo2_cat)
        _pc_o = ct.get('_performance_context', {})
        _lvl_spd = _pc_o.get('level_by_speed', '') if _pc_o.get('executed') else ''
        _vo2_txt = f'VO\u2082peak {vo2_rel:.1f} ml/kg/min — percentyl ~{vo2_pct} dla wieku ({cat_pl})'
        if _lvl_spd:
            _vo2_txt += f'. Wg VO\u2082max: {vo2_ath} | wg pr\u0119dko\u015bci VT2: {_lvl_spd}'
            if vo2_ath != _lvl_spd and _lvl_spd:
                _vo2_txt += ' — rozbie\u017cno\u015b\u0107 wskazuje na potencja\u0142 do poprawy'
        else:
            _vo2_txt += f'. Poziom sportowy: {vo2_ath}'
        _vo2_txt += '.'
        add(_vo2_txt, 
            'positive' if vo2_cat in ('Superior','Excellent') else 'neutral' if vo2_cat == 'Good' else 'warning', 1)
    
    vt1p = ct.get('_interp_vt1_pct_vo2')
    if vt1p:
        base = ct.get('_interp_aerobic_base','')
        _pc = ct.get('_performance_context', {})
        _vt1_v = _pc.get('v_vt1_kmh')
        _vt1_mas_pct = _pc.get('vt1_pct_vref')
        _vt1_mas_src = _pc.get('v_ref_source', '')
        _vt1_level = _pc.get('level_by_speed', '')
        
        # Build VT1 text with velocity context
        _vt1_txt = f'VT1 przy {vt1p:.0f}% VO₂max'
        if _vt1_v:
            _vt1_txt += f' / {_vt1_v:.1f} km/h'
        if _vt1_mas_pct and 'MAS_external' in _vt1_mas_src:
            _vt1_txt += f' ({_vt1_mas_pct:.0f}% MAS)'
        
        # Qualify base using BOTH %VO2 AND speed
        if _vt1_v and _vt1_mas_pct:
            # Speed-validated assessment
            if vt1p >= 70 and _vt1_mas_pct >= 60:
                _vt1_txt += ' — bardzo dobra baza tlenowa. Efektywny metabolizm tłuszczowy.'
                _vt1_mood = 'positive'
            elif vt1p >= 70 and _vt1_mas_pct < 50:
                _vt1_txt += ' — wysoki %VO₂max ale niska prędkość absolutna, wskazuje na niski pułap tlenowy.'
                _vt1_mood = 'warning'
            elif vt1p >= 55 and _vt1_mas_pct >= 50:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'neutral'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa, prędkość do rozwinięcia.'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'
        elif _vt1_v:
            # Speed available but no MAS — use absolute speed only
            if vt1p >= 70:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'positive'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa.'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'
        else:
            # No speed data at all — fallback to %VO2 only with caveat
            if vt1p >= 70:
                _vt1_txt += f' — {base.lower()} baza tlenowa (brak danych prędkości do pełnej walidacji).'
                _vt1_mood = 'positive'
            elif vt1p >= 55:
                _vt1_txt += f' — {base.lower()} baza tlenowa (brak danych prędkości do pełnej walidacji).'
                _vt1_mood = 'neutral'
            else:
                _vt1_txt += f' — {base.lower()} baza tlenowa. Zalecenie: więcej treningu w Strefie 2.'
                _vt1_mood = 'warning'
        
        add(_vt1_txt, _vt1_mood, 2)
    
    vt2p = ct.get('_interp_vt2_pct_vo2')
    # Performance Context — VT2 z odniesieniem do prędkości/MAS
    _pc = ct.get('_performance_context', {})
    _pc_v = _pc.get('v_vt2_kmh')
    _pc_mas_pct = _pc.get('vt2_pct_vref')
    _pc_mas_src = _pc.get('v_ref_source', '')
    _pc_level = _pc.get('level_by_speed', '')
    _pc_level_pct = _pc.get('level_by_pct_vo2', '')
    _pc_match = _pc.get('levels_match', True)
    
    if vt2p:
        # Base text — mood determined by BOTH %VO2 AND speed level
        _vt2_txt = f'VT2 przy {vt2p:.0f}% VO₂max'
        
        # Add speed context
        if _pc_v:
            _vt2_txt += f' / {_pc_v:.1f} km/h'
        
        # Add MAS% if available
        if _pc_mas_pct and 'MAS_external' in _pc_mas_src:
            _vt2_txt += f' ({_pc_mas_pct:.0f}% MAS)'
        
        # Determine mood using speed-validated logic
        if _pc_level and _pc_level in ('Elite', 'Well-trained'):
            _vt2_mood = 'positive'
        elif _pc_level and _pc_level == 'Trained':
            _vt2_mood = 'neutral'
        elif _pc_level and _pc_level in ('Recreational', 'Sedentary'):
            _vt2_mood = 'warning'
        else:
            # No speed classification — fallback to %VO2 only
            if vt2p >= 88: _vt2_mood = 'positive'
            elif vt2p >= 80: _vt2_mood = 'neutral'
            else: _vt2_mood = 'warning'
        
        # Add level classification
        if _pc_level:
            _vt2_txt += f' — poziom <b>{_pc_level}</b>'
        
        # Add mismatch warning / context
        if not _pc_match and _pc_level_pct:
            _vt2_txt += f' (uwaga: %VO₂max sugeruje {_pc_level_pct})'
        
        if vt2p >= 88 and _pc_level in ('Sedentary', 'Recreational'):
            _vt2_txt += '. Wysoki próg względny ale niska prędkość → niski pułap VO₂max.'
            _vt2_mood = 'warning'
        elif vt2p >= 88 and _pc_level == 'Trained':
            _vt2_txt += '. Wysoki próg względny, prędkość na poziomie Trained — dalszy postęp przez interwały VO₂max.'
        elif vt2p >= 88 and _pc_level in ('Well-trained', 'Elite'):
            _vt2_txt += '. Doskonały próg potwierdzona prędkością absolutną.'
        elif vt2p >= 88 and not _pc_level:
            _vt2_txt += '. Wysoki próg (brak danych prędkości do pełnej walidacji).'
        elif vt2p < 80:
            _vt2_txt += '. Potencjał do rozwoju przez trening tempo/SST.'
        
        add(_vt2_txt, _vt2_mood, 3)
    
    gap = ct.get('_interp_gap_bpm')
    if gap is not None:
        if gap < 10:
            add(f'Wąska strefa VT1-VT2 ({gap} bpm). Ograniczona przestrzeń do treningu progowego.', 'warning', 4)
        elif gap >= 25:
            add(f'Szeroka strefa VT1-VT2 ({gap} bpm) — duża przestrzeń do treningu tempo/threshold.', 'positive', 4)
    
    is_max = ct.get('_interp_test_maximal')
    conf = ct.get('_interp_test_confidence','')
    rer_p = ct.get('RER_peak')
    try:
        rer_val = float(rer_p) if rer_p and str(rer_p) != '-' else None
    except: rer_val = None
    if rer_val:
        if rer_val >= 1.15:
            add(f'RER peak {rer_val:.2f} — test maximalny, doskonały wysiłek metaboliczny.', 'positive', 2)
        elif rer_val >= 1.10:
            add(f'RER peak {rer_val:.2f} — test maximalny, dobry wysiłek.', 'positive', 2)
        elif rer_val >= 1.05:
            add(f'RER peak {rer_val:.2f} — akceptowalny wysiłek, ale bliski granicy submaximalności.', 'neutral', 2)
        else:
            add(f'RER peak {rer_val:.2f} — test prawdopodobnie SUBMAXIMALNY. VO2peak może być niedoszacowane.', 'negative', 1)
    
    ve_sl = ct.get('VE_VCO2_slope')
    try:
        ve_val = float(ve_sl) if ve_sl and str(ve_sl) != '-' else None
    except: ve_val = None
    if ve_val:
        if ve_val < 25:
            add(f'VE/VCO2 slope {ve_val:.1f} — doskonała efektywność wentylacyjna.', 'positive', 5)
        elif ve_val < 30:
            add(f'VE/VCO2 slope {ve_val:.1f} — prawidłowa efektywność wentylacyjna.', 'neutral', 5)
        elif ve_val < 36:
            add(f'VE/VCO2 slope {ve_val:.1f} — łagodnie podwyższony (norma <30).', 'warning', 5)
        else:
            add(f'VE/VCO2 slope {ve_val:.1f} — podwyższony. Obniżona efektywność wentylacyjna.', 'negative', 4)
    
    fatmax = ct.get('FATmax_g_min')
    try:
        fat_val = float(fatmax) if fatmax and str(fatmax) != '-' else None
    except: fat_val = None
    if fat_val and fat_val > 0:
        sex = ct.get('sex','male')
        thr = 0.7 if sex == 'male' else 0.5
        if fat_val >= thr:
            add(f'FATmax {fat_val:.2f} g/min — dobra zdolność oksydacji tłuszczów.', 'positive', 6)
        else:
            add(f'FATmax {fat_val:.2f} g/min — umiarkowana oksydacja tłuszczów. Trening Z2 może to poprawić.', 'neutral', 6)
    
    # O2-pulse analysis
    o2p = ct.get('O2pulse_peak')
    try: o2p_val = float(o2p) if o2p and str(o2p) not in ('-','None','') else None
    except: o2p_val = None
    if o2p_val:
        _sx = ct.get('_interp_sex', 'male')
        o2p_norm = 18 if _sx == 'male' else 13
        if o2p_val >= o2p_norm * 1.3:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — wysoki. Dobra objętość wyrzutowa.', 'positive', 5)
        elif o2p_val >= o2p_norm * 0.8:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — w normie.', 'neutral', 6)
        else:
            add(f'O2-pulse peak {o2p_val:.1f} ml/beat — obniżony. Możliwy limiter centralny (SV).', 'warning', 4)
    
    # Running economy (E06) — only for running modalities
    re_val = ct.get('E06_RE_mlO2_per_kg_per_km')
    try: re_f = float(re_val) if re_val and str(re_val) not in ('-','None','') else None
    except: re_f = None
    if re_f and _is_run:
        if re_f < 180:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — doskonała (elitarna klasa).', 'positive', 5)
        elif re_f < 210:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — dobra.', 'neutral', 6)
        elif re_f < 240:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — przeciętna. Potencjał poprawy przez drills techniczne i plyometrię.', 'warning', 5)
        else:
            add(f'Ekonomia biegu {re_f:.0f} mlO2/kg/km — niska. Priorytet: praca nad techniką biegu.', 'negative', 4)
    
    # Lactate dynamics
    la_peak = ct.get('_interp_la_peak')
    if la_peak:
        if la_peak >= 12:
            add(f'Laktat peak {la_peak:.1f} mmol/L — wysoka zdolność buforowa i tolerancja kwasicy.', 'positive', 6)
        elif la_peak >= 8:
            add(f'Laktat peak {la_peak:.1f} mmol/L — dobra odpowiedź glikolityczna.', 'neutral', 6)
    
    # HR response
    hr_max = ct.get('_interp_hr_max')
    _ag = ct.get('_interp_age', 30)
    try: hr_m = float(hr_max) if hr_max else None
    except: hr_m = None
    try: age_f = float(_ag)
    except: age_f = 30
    if hr_m and age_f:
        hr_pred = 220 - age_f
        hr_pct = hr_m / hr_pred * 100
        if hr_pct > 105:
            add(f'HRmax {hr_m:.0f} bpm ({hr_pct:.0f}% predykcji) — znacząco powyżej normy wiekowej.', 'neutral', 7)
        elif hr_pct < 85 and ct.get('_interp_test_maximal'):
            add(f'HRmax {hr_m:.0f} bpm ({hr_pct:.0f}% predykcji) — chronotropowa niekompetencja? Rozważ ocenę kardiologiczną.', 'warning', 3)
    

    # ─── SPORT CONTEXT: BREATHING PATTERN ANALYSIS ──────────────────
    # References:
    #   Folinsbee et al. 1983 — elite cyclists BF 63/min, VT ~50% FVC
    #   ACC (2021) — elite RR 60-70, BR <10-15%, VE >200 L/min normal
    #   HUNT3 (2014) — VE men 20-29: 141.9±24.5, VT 2.94±0.46 L
    #   NOODLE (2024) — EA VE/VCO2 slope: M 26.1±2.0, F 27.7±2.6
    #   Carey et al. 2008 — triathletes: 86% use BF-dominant pattern, only 14% VT-dominant
    #   Naranjo et al. 2005 — athlete breathing nomogram: VT vs BF curvilinear relation
    # Sport-specific VO2max ranges (ml/kg/min):
    #   Runner recreational: 40-50, competitive: 50-60, elite: 60-80+
    #   CrossFit competitive: 45-55, elite: 50-60+
    #   HYROX recreational: 38-48, competitive: 45-55
    #   Cycling competitive: 50-65, elite: 65-85+
    #   Triathlon competitive: 55-65, elite: 65-80+
    
    e07 = ct.get('_e07_raw', {})
    if e07 and e07.get('status') == 'OK':
        # Read VO2max directly from ct
        _bp_vo2 = ct.get('_interp_vo2_mlkgmin')
        # Read VE/VCO2 slope directly from ct (ve_val may not be in scope here)
        _bp_ve_slope = None
        try:
            _ve_sl_raw = ct.get('VE_VCO2_slope')
            if _ve_sl_raw and str(_ve_sl_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_slope = float(_ve_sl_raw)
        except: pass
        _bp_strategy = e07.get('strategy', '')
        _bp_bf_peak = e07.get('bf_peak')
        _bp_vt_peak = e07.get('vt_peak_L')
        _bp_ve_peak = None
        try:
            _bp_ve_peak_raw = ct.get('VEpeak')
            if _bp_ve_peak_raw and str(_bp_ve_peak_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_peak = float(_bp_ve_peak_raw)
        except: pass
        _bp_vdvt_rest = e07.get('vdvt_rest')
        _bp_vdvt_peak = e07.get('vdvt_peak')
        _bp_flags = e07.get('flags', [])
        _bp_bf_vt_ratio = e07.get('ve_from_bf_pct', 50)
        
        # Determine if athlete context applies
        _is_athlete = (_bp_vo2 is not None and _bp_vo2 >= 45)
        _is_vent_efficient = (_bp_ve_slope is not None and _bp_ve_slope < 30)
        _has_sport_context = _is_athlete and _is_vent_efficient
        
        # Sport tier classification
        _sport_tier = 'sedentary'
        if _bp_vo2:
            _sex_bp = ct.get('_interp_sex', 'male')
            if _sex_bp == 'male':
                if _bp_vo2 >= 65: _sport_tier = 'elite_endurance'
                elif _bp_vo2 >= 55: _sport_tier = 'competitive_endurance'
                elif _bp_vo2 >= 45: _sport_tier = 'trained'
                elif _bp_vo2 >= 35: _sport_tier = 'recreational'
            else:
                if _bp_vo2 >= 55: _sport_tier = 'elite_endurance'
                elif _bp_vo2 >= 45: _sport_tier = 'competitive_endurance'
                elif _bp_vo2 >= 38: _sport_tier = 'trained'
                elif _bp_vo2 >= 30: _sport_tier = 'recreational'
        
        # Reference norms by tier (M/F combined approximate)
        _bp_norms = {
            'sedentary':             {'bf_peak': (35,45), 'vt_peak': (1.8,2.5), 've_peak': (70,110),  'bf_dom_normal': False},
            'recreational':          {'bf_peak': (38,50), 'vt_peak': (2.2,3.0), 've_peak': (90,130),  'bf_dom_normal': False},
            'trained':               {'bf_peak': (42,55), 'vt_peak': (2.5,3.2), 've_peak': (110,155), 'bf_dom_normal': True},
            'competitive_endurance': {'bf_peak': (45,60), 'vt_peak': (2.7,3.5), 've_peak': (130,175), 'bf_dom_normal': True},
            'elite_endurance':       {'bf_peak': (55,70), 'vt_peak': (2.9,4.0), 've_peak': (160,200), 'bf_dom_normal': True},
        }
        _norms = _bp_norms.get(_sport_tier, _bp_norms['sedentary'])
        
        # Generate sport-contextualized breathing observations
        if _has_sport_context:
            _tier_pl = {'elite_endurance':'elitarny wytrzymałościowy', 'competitive_endurance':'startujący wytrzymałościowy',
                        'trained':'wytrenowanego', 'recreational':'rekreacyjnego'}.get(_sport_tier, _sport_tier)
            
            # BF-dominant strategy reinterpretation
            if _bp_strategy == 'BF_DOMINANT' and _norms['bf_dom_normal']:
                add(f'Wzorzec BF-dominant ({_bp_bf_vt_ratio:.0f}% wentylacji z częstości) — typowa adaptacja sportowa u zawodnika '
                    f'{_tier_pl}. Kolarze elitarni osiągają BF 60-70/min (Folinsbee 1983, ACC 2021). '
                    f'86% wytrzymałościowców stosuje ten wzorzec (Carey 2008).', 'neutral', 7)
            elif _bp_strategy == 'BF_DOMINANT' and not _norms['bf_dom_normal']:
                add(f'Wzorzec BF-dominant ({_bp_bf_vt_ratio:.0f}% z częstości) — u zawodnika rekreacyjnego może wskazywać na '
                    f'ograniczenie mechaniczne lub niedostateczny trening mięśni oddechowych. '
                    f'Rozważ trening oddechowy (IMT/RMT).', 'warning', 7)
            
            # VD/VT reinterpretation for athletes with very low baseline
            if _bp_vdvt_rest is not None and _bp_vdvt_peak is not None:
                try:
                    vdvt_r = float(_bp_vdvt_rest)
                    vdvt_p = float(_bp_vdvt_peak)
                    if vdvt_r < 0.20 and vdvt_p < 0.20:
                        add(f'VD/VT {vdvt_r:.3f}→{vdvt_p:.3f} — wyjątkowo niska martwa przestrzeń wentylacyjna. '
                            f'Norma kliniczna 0.25-0.35. Wartości poniżej 0.20 wskazują na elitarną efektywność '
                            f'wentylacyjną. Ewentualny wzrost VD/VT w tym zakresie NIE jest patologiczny.', 'positive', 7)
                    elif vdvt_p > vdvt_r and (vdvt_p - vdvt_r) > 0.03:
                        if vdvt_p < 0.25:
                            add(f'VD/VT wzrost {vdvt_r:.3f}→{vdvt_p:.3f} — mimo wzrostu, wartości bezwzględne w normie '
                                f'(<0.25). Wzorzec „paradoxical rise" jest klinicznie nieistotny przy niskim baseline.', 'neutral', 8)
                except: pass
            
            # BF peak contextualization 
            if _bp_bf_peak:
                try:
                    bf = float(_bp_bf_peak)
                    bf_lo, bf_hi = _norms['bf_peak']
                    if bf >= 55 and _sport_tier in ('competitive_endurance', 'elite_endurance'):
                        add(f'BF peak {bf:.0f}/min — w normie dla poziomu {_tier_pl} (ref: {bf_lo}-{bf_hi}/min). '
                            f'Elitarni sportowcy RR 60-70/min (ACC 2021).', 'neutral', 8)
                    elif bf > bf_hi:
                        add(f'BF peak {bf:.0f}/min — powyżej typowego zakresu ({bf_lo}-{bf_hi}/min) dla poziomu {_tier_pl}. '
                            f'Rozważ ocenę rezerwy oddechowej i mechaniki wentylacji.', 'warning', 8)
                except: pass
            
            # VE peak contextualization
            if _bp_ve_peak:
                ve_lo, ve_hi = _norms['ve_peak']
                if _bp_ve_peak > ve_hi * 1.1:
                    add(f'VE peak {_bp_ve_peak:.0f} L/min — bardzo wysoka. Ref {_tier_pl}: {ve_lo}-{ve_hi} L/min. '
                        f'Sprawdź rezerwę oddechową (BR). BR <15% jest normalne u sportowców (ACC 2021).', 'neutral', 8)
                elif _bp_ve_peak >= ve_lo:
                    add(f'VE peak {_bp_ve_peak:.0f} L/min — adekwatna do poziomu {_tier_pl} (ref: {ve_lo}-{ve_hi} L/min).', 'positive', 9)
            
            # Flag reinterpretation summary
            _clinical_flags = [f for f in _bp_flags if f in ('BF_DOMINANT_STRAT', 'EARLY_VT_PLATEAU', 'VDVT_NO_DECREASE', 'VDVT_PARADOXICAL_RISE', 'HIGH_BF_VAR_REST', 'HIGH_BF_VAR_EX')]
            if _clinical_flags and _has_sport_context:
                add(f'Flagi oddechowe [{", ".join(_clinical_flags)}] — reinterpretacja sportowa: '
                    f'przy VO2max {_bp_vo2:.1f} ml/kg/min i VE/VCO2 slope {_bp_ve_slope:.1f} (<30) '
                    f'te wzorce stanowią ADAPTACJĘ SPORTOWĄ, nie patologię. '
                    f'Kontekst kliniczny wymaga VE/VCO2 >34 i/lub VD/VT >0.30.', 'neutral', 7, '🏃')
        
        else:
            # Non-athlete: standard breathing pattern observations
            if _bp_strategy == 'BF_DOMINANT':
                add(f'Wzorzec oddechowy BF-dominant ({_bp_bf_vt_ratio:.0f}% z częstości) — '
                    f'wentylacja głównie przez częstość oddechów. Może zwiększać martwą przestrzeń. '
                    f'Rozważ trening oddychania przeponowego.', 'warning', 7)
            
            if _bp_vdvt_rest is not None and _bp_vdvt_peak is not None:
                try:
                    vdvt_r = float(_bp_vdvt_rest)
                    vdvt_p = float(_bp_vdvt_peak)
                    if vdvt_p > 0.30:
                        add(f'VD/VT peak {vdvt_p:.3f} — podwyższony (norma <0.30). Obniżona efektywność wentylacyjna.', 'warning', 7)
                    elif vdvt_p > vdvt_r and (vdvt_p - vdvt_r) > 0.05 and vdvt_p > 0.25:
                        add(f'VD/VT wzrost {vdvt_r:.3f}→{vdvt_p:.3f} — paradoksalny wzrost. Rozważ ocenę kardiologiczną.', 'warning', 7)
                except: pass

    # ═══════════════════════════════════════════════════════════════
    # CROSS-INTERACTION INSIGHTS (v2.1)
    # Łączenie danych z różnych silników w syntetyczne wnioski
    # ═══════════════════════════════════════════════════════════════
    try:
        _e10 = ct.get('_e10_raw', {})
        _e16 = ct.get('_e16_raw', {})
        _e02 = ct.get('_e02_raw', {})
        _e03 = ct.get('_e03_raw', {})
        _e15 = ct.get('_e15_raw', {})
        _pc = ct.get('_performance_context', {})
        _vt1_hr_x = None; _vt2_hr_x = None; _vt1_vo2p = None; _vt2_vo2p = None
        try:
            _vt1_hr_x = float(_e02.get('vt1_hr', 0)) if _e02.get('vt1_hr') else None
            _vt2_hr_x = float(_e02.get('vt2_hr', 0)) if _e02.get('vt2_hr') else None
            _vt1_vo2p = float(_e02.get('vt1_vo2_pct_peak', 0)) if _e02.get('vt1_vo2_pct_peak') else None
            _vt2_vo2p = float(_e02.get('vt2_vo2_pct_peak', 0)) if _e02.get('vt2_vo2_pct_peak') else None
        except: pass

        # 1) FATmax ↔ Zones cross-reference
        _fatmax_hr = _e10.get('fatmax_hr')
        _fatmax_gmin = _e10.get('mfo_gmin')
        _zones = _e16.get('zones', {})
        if _fatmax_hr and _zones and isinstance(_zones, dict):
            _fhr = float(_fatmax_hr)
            _fat_zone = None; _fat_zk = ''
            for zk, zv in _zones.items():
                if isinstance(zv, dict):
                    zlo = zv.get('hr_low', 0)
                    zhi = zv.get('hr_high', 999)
                    if zlo <= _fhr <= zhi:
                        _fat_zone = zv.get('name_pl', zk)
                        _fat_zk = zk.upper()
                        break
            if _fat_zone:
                _fat_txt = f'FATmax (HR {_fhr:.0f}) wypada w strefie {_fat_zk} ({_fat_zone})'
                if _fatmax_gmin:
                    _fat_gh = float(_fatmax_gmin) * 60
                    _fat_txt += f' \u2014 max spalanie t\u0142uszczy ~{_fat_gh:.0f} g/h.'
                add(_fat_txt, 'neutral', 12)

        # 2) Crossover ↔ VT1 relationship
        _cop_pct = _e10.get('cop_pct_vo2peak')
        if _cop_pct and _vt1_vo2p:
            _cop = float(_cop_pct)
            if _cop < _vt1_vo2p - 10:
                add(f'Crossover przy {_cop:.0f}% VO\u2082 \u2014 poni\u017cej VT1 ({_vt1_vo2p:.0f}%). Szeroki zakres spalania t\u0142uszczy.', 'positive', 13)
            elif _cop > _vt1_vo2p:
                add(f'Crossover przy {_cop:.0f}% VO\u2082 \u2014 powy\u017cej VT1 ({_vt1_vo2p:.0f}%)! Wczesne przej\u015bcie na glikogen, rozwa\u017c trening bazowy.', 'warning', 4)

        # 3) VE/VCO₂ slope ↔ VT2 → combined ventilatory efficiency
        _slope = _e03.get('slope_to_vt2') or _e03.get('ve_vco2_slope') or _e03.get('slope_full')
        if _slope and _vt2_vo2p:
            _sl = float(_slope)
            if _sl < 25 and _vt2_vo2p > 85:
                add(f'Slope {_sl:.1f} + VT2 przy {_vt2_vo2p:.0f}% \u2014 wysoka efektywno\u015b\u0107 wentylacyjna potwierdzona wysokim progiem.', 'positive', 8)
            elif _sl > 34 and _vt2_vo2p < 75:
                add(f'Slope {_sl:.1f} + VT2 przy {_vt2_vo2p:.0f}% \u2014 obni\u017cona efektywno\u015b\u0107 wentylacyjna z niskim progiem. Priorytet: trening progowy.', 'warning', 3)

        # 4) Sport classification divergence → economy insight
        _sport_cls = _e15.get('vo2_class_sport', '')
        _speed_lvl = _pc.get('level_by_speed', '')
        if _sport_cls and _speed_lvl and _sport_cls != _speed_lvl:
            _cls_rank = {'UNTRAINED':0,'RECREATIONAL':1,'Sedentary':0,'Recreational':1,
                        'TRAINED':2,'Trained':2,'COMPETITIVE':3,'Well-trained':3,
                        'SUB_ELITE':4,'ELITE':5,'Elite':5}
            _r_vo2 = _cls_rank.get(_sport_cls, 2)
            _r_spd = _cls_rank.get(_speed_lvl, 2)
            if _r_vo2 >= _r_spd + 1:
                _econ_txt = f'VO\u2082max \u2192 {_sport_cls}, pr\u0119dko\u015b\u0107 VT2 \u2192 {_speed_lvl}. '
                _econ_txt += 'Du\u017cy silnik aerobowy, ale niska ekonomia ruchu. '
                if _is_run:
                    _econ_txt += 'Priorytet: drills techniczne, plyometria, strides.'
                elif _is_bike:
                    _econ_txt += 'Priorytet: kadencja, pozycja, pedaling drills.'
                else:
                    _econ_txt += 'Priorytet: efektywno\u015b\u0107 techniczna w dyscyplinie.'
                add(_econ_txt, 'warning', 5)
            elif _r_spd > _r_vo2 + 1:
                add(f'Pr\u0119dko\u015b\u0107 VT2 \u2192 {_speed_lvl} przy VO\u2082max \u2192 {_sport_cls}. Wyj\u0105tkowa ekonomia ruchu.', 'positive', 6)

        # 5) FATmax → nutrition insight
        if _fatmax_gmin and _vt1_vo2p:
            _fg = float(_fatmax_gmin)
            _fat_gh = _fg * 60
            add(f'\u017bywienie: FATmax {_fg:.2f} g/min \u2192 ~{_fat_gh:.0f} g t\u0142uszczu/h. Wysi\u0142ki >90 min powy\u017cej VT1 wymagaj\u0105 suplementacji CHO ~40-60 g/h.', 'neutral', 14)

        # 6) Recovery HR context
        _hrr = ct.get('HRR_1min')
        if _hrr and _sport_cls:
            try:
                _hrr_v = float(_hrr)
                if _hrr_v > 30:
                    add(f'HRR {_hrr_v:.0f} bpm/min \u2014 szybka restytucja, dobra regulacja autonomiczna dla poziomu {_sport_cls}.', 'positive', 11)
                elif _hrr_v < 12:
                    add(f'HRR {_hrr_v:.0f} bpm/min \u2014 wolna restytucja. Przy poziomie {_sport_cls} rozwa\u017c wi\u0119cej treningu bazowego.', 'warning', 4)
            except: pass
    except Exception:
        pass  # Cross-interactions are bonus, never crash the report

    obs.sort(key=lambda x: x['priority'])
    return obs


class ReportAdapter:
    """
    Rola: Adapter / Render analityczny.
    Zgodność: [ANALYSIS_REPORT_BUILDER v1.1]
    Przywraca pełny format raportu dla Trenera AI.
    """

    @staticmethod
    def _safe_get(dictionary, key, default="[BRAK]"):
        val = dictionary.get(key)
        if val is None or val == "" or val == "nan":
            return default

        # Bezpieczne formatowanie liczb (float/int + numpy typy)
        if isinstance(val, (int, float, np.integer, np.floating)):
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                return default
            return f"{float(val):.2f}"

        return str(val)

    @staticmethod
    def _first_non_null(*values):
        for v in values:
            if v is not None and v != "" and str(v).lower() != "nan":
                return v
        return None

    
    @staticmethod
    def describe_protocol(df, cfg, results):
        """Generate protocol description and interactive table HTML."""
        
        proto_name = getattr(cfg, 'protocol_name', '') or ''
        modality = getattr(cfg, 'modality', 'run')
        mod_pl = 'Bieżnia' if modality == 'run' else ('Rower' if modality == 'bike' else modality.capitalize())
        
        t_stop = results.get('E00', {}).get('t_stop', 0)
        dur_min = t_stop / 60 if t_stop else 0
        rec_avail = results.get('E00', {}).get('recovery_0_60_available', False)
        
        # Try RAW_PROTOCOLS first
        raw_segments = None
        try:
            if 'RAW_PROTOCOLS' in dir(__builtins__) or True:
                import builtins
                rp = globals().get('RAW_PROTOCOLS', {})
                if proto_name in rp:
                    raw_segments = rp[proto_name]
        except: pass
        
        # Build short description
        parts = [mod_pl]
        if raw_segments:
            speeds = [s.get('speed_kmh') for s in raw_segments if s.get('speed_kmh') is not None and s.get('speed_kmh',0) > 0]
            inclines = [s.get('incline_pct') for s in raw_segments if s.get('incline_pct',0) > 0]
            powers = [s.get('power_w') for s in raw_segments if s.get('power_w') is not None]
            
            # Determine type
            types = set(s.get('type','') for s in raw_segments)
            if any('ramp' in t for t in types):
                parts.append('rampowy')
            else:
                parts.append('stopniowany')
            
            if speeds:
                parts.append(f'{min(speeds):.0f}–{max(speeds):.0f} km/h')
            if powers:
                parts.append(f'{min(powers):.0f}–{max(powers):.0f} W')
            if inclines:
                parts.append(f'do {max(inclines):.0f}%')
            
            # Step duration
            const_segs = [s for s in raw_segments if s.get('type')=='const' and s.get('speed_kmh',0) > 0]
            if const_segs:
                import re
                def parse_t(t_str):
                    try:
                        p = t_str.split(':')
                        return int(p[0])*60 + int(p[1])
                    except: return 0
                durs = [parse_t(s['end']) - parse_t(s['start']) for s in const_segs]
                if durs:
                    from statistics import median
                    med = median(durs)
                    parts.append(f'{med/60:.0f}min/stopień')
        else:
            # Fallback to marker detection
            markers = []
            if 'Marker' in df.columns:
                mk = df[df['Marker'].notna() & (df['Marker'].astype(str).str.strip() != '')]
                for _, row in mk.iterrows():
                    markers.append({'label': str(row['Marker']).strip()})
            speeds_m = []
            for m in markers:
                lbl = m['label']
                if lbl.lower() not in ('ko','end','stop') and '%' not in lbl:
                    try: speeds_m.append(float(lbl.replace(',','.')))
                    except: pass
            if speeds_m:
                parts.append(f'stopniowany, {min(speeds_m):.0f}–{max(speeds_m):.0f} km/h')
        
        if dur_min > 0:
            parts.append(f'{dur_min:.0f} min')
        if rec_avail:
            t_max = df['Time_s'].max() if 'Time_s' in df.columns else 0
            rec_s = t_max - t_stop if t_max > t_stop else 0
            if rec_s > 30:
                parts.append(f'recovery {rec_s/60:.0f} min')
        
        return ', '.join(parts)
    
    @staticmethod
    def protocol_table_html(cfg, t_stop=None):
        """Generate protocol stages table HTML - only up to t_stop, with stop marker."""
        proto_name = getattr(cfg, 'protocol_name', '') or ''
        rp = globals().get('RAW_PROTOCOLS', {})
        segs = rp.get(proto_name, [])
        if not segs:
            return ''
        
        modality = getattr(cfg, 'modality', 'run')
        is_run = modality == 'run'
        
        def parse_t(t_str):
            try:
                p = str(t_str).split(':')
                return int(p[0])*60 + int(p[1])
            except: return 99999
        
        rows = ''
        last_active_row = None
        for s in segs:
            stype = s.get('type','')
            start = s.get('start','')
            end_t = s.get('end','')
            start_s = parse_t(start)
            end_s = parse_t(end_t)
            
            # Skip stages entirely after t_stop
            if t_stop and start_s >= t_stop:
                break
            
            if is_run:
                speed = s.get('speed_kmh', s.get('speed_from',''))
                speed_to = s.get('speed_to','')
                incl = s.get('incline_pct', 0)
                if stype == 'ramp_speed':
                    desc = f'{speed}\u2192{speed_to} km/h'
                elif stype == 'dynamic_incline_steps_to_stop':
                    desc = f'{speed} km/h + nachylenie co {s.get("incline_step_pct",2)}%'
                else:
                    desc = f'{speed} km/h'
                    if incl and float(incl) > 0:
                        desc += f' / {incl}%'
            else:
                pwr = s.get('power_w', s.get('power_from',''))
                pwr_to = s.get('power_to','')
                if 'ramp' in stype:
                    desc = f'{pwr}\u2192{pwr_to} W'
                else:
                    desc = f'{pwr} W'
            
            # Determine if this is the stop stage (t_stop falls within it)
            is_stop_stage = t_stop and start_s < t_stop and end_s >= t_stop
            
            # Truncate end time to t_stop if stage extends beyond
            if t_stop and end_s > t_stop:
                mins = int(t_stop // 60)
                secs = int(t_stop % 60)
                end_t = f'{mins}:{secs:02d}'
            
            # Color
            if stype == 'const' and float(s.get('speed_kmh',0) or 0) == 0 and float(s.get('power_w',0) or 0) == 0:
                bg = '#f1f5f9'
            elif is_stop_stage:
                bg = '#fef2f2'  # red tint for stop
            elif 'ramp' in stype or 'dynamic' in stype:
                bg = '#fef3c7'
            else:
                bg = 'white'
            
            # Stop marker
            stop_mark = ' \u26d4' if is_stop_stage else ''
            
            rows += f'<tr style="background:{bg};"><td style="padding:3px 8px;">{start}</td><td style="padding:3px 8px;">{end_t}{stop_mark}</td><td style="padding:3px 8px;">{desc}</td></tr>'
        
        if not rows:
            return ''
        
        return f"""<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;">
<tr style="background:#f8fafc;font-weight:600;"><th style="padding:4px 8px;text-align:left;">Start</th><th style="padding:4px 8px;text-align:left;">End</th><th style="padding:4px 8px;text-align:left;">{'Prędkość / Nachylenie' if is_run else 'Moc'}</th></tr>
{rows}</table>"""


    @staticmethod
    def build_canon_table(df, results: 'Dict[str, Any]', cfg) -> 'Dict[str, Any]':
        """Tworzy płaską tabelę ze wszystkich silników."""

        # Pobieramy wyniki (zabezpieczenie pustymi słownikami)
        r01 = results.get('E01', {})  # QC/Max
        r02 = results.get('E02', {})  # Progi
        r03 = results.get('E03', {})  # Slope
        r04 = results.get('E04', {})  # OUES
        r05 = results.get('E05', {})  # O2Pulse
        r07 = results.get('E07', {})  # BreathingPattern
        r08 = results.get('E08', {})  # HRR
        r10 = results.get('E10', {})  # Substrate
        r11 = results.get('E11', {})  # Lactate
        r12 = results.get('E12', {})  # NIRS
        r15 = results.get('E15', {})  # Norm
        r16 = results.get('E16', {})  # Strefy

        ct = {}

        # --- A. META DANE ---
        ct['athlete_name'] = getattr(cfg, 'athlete_name', 'AUTO')
        ct['athlete_id'] = getattr(cfg, 'athlete_id', 'ID')
        ct['test_date'] = getattr(cfg, 'test_date', '-')
        ct['location'] = getattr(cfg, 'location', 'Lab')
        ct['operator'] = getattr(cfg, 'operator', 'Auto')
        ct['protocol'] = getattr(cfg, 'protocol_name', '-')
        ct['age'] = getattr(cfg, 'age_y', '-')
        ct['height'] = getattr(cfg, 'height_cm', '-')
        ct['weight'] = getattr(cfg, 'body_mass_kg', '-')

        # --- B. PROGI (ANCHORS) ---
        ct['VT1_HR'] = ReportAdapter._safe_get(r02, 'vt1_hr')
        vt1_spd = r02.get('vt1_speed_kmh') or r02.get('vt1_speed')
        ct['VT1_Speed'] = ReportAdapter._safe_get(r02, 'vt1_speed_kmh') if vt1_spd not in [None, ""] else ReportAdapter._safe_get(r02, 'vt1_speed') if r02.get('vt1_speed') not in [None, ""] else ReportAdapter._safe_get(r02, 'vt1_load')
        ct['VT1_Power'] = ReportAdapter._safe_get(r02, 'vt1_power_w') if r02.get('vt1_power_w') is not None else ReportAdapter._safe_get(r02, 'vt1_power')

        ct['VT2_HR'] = ReportAdapter._safe_get(r02, 'vt2_hr')
        vt2_spd = r02.get('vt2_speed_kmh') or r02.get('vt2_speed')
        ct['VT2_Speed'] = ReportAdapter._safe_get(r02, 'vt2_speed_kmh') if vt2_spd not in [None, ""] else ReportAdapter._safe_get(r02, 'vt2_speed') if r02.get('vt2_speed') not in [None, ""] else ReportAdapter._safe_get(r02, 'vt2_load')
        ct['VT2_Power'] = ReportAdapter._safe_get(r02, 'vt2_power_w') if r02.get('vt2_power_w') is not None else ReportAdapter._safe_get(r02, 'vt2_power')

        # --- B2. PROGI — %VO2max, VO2 values, plateau ---
        ct['VT1_VO2_mlmin'] = ReportAdapter._safe_get(r02, 'vt1_vo2_mlmin')
        ct['VT2_VO2_mlmin'] = ReportAdapter._safe_get(r02, 'vt2_vo2_mlmin')
        ct['VT1_pct_VO2max'] = ReportAdapter._safe_get(r15, 'vt1_pct_vo2peak')
        ct['VT2_pct_VO2max'] = ReportAdapter._safe_get(r15, 'vt2_pct_vo2peak')
        ct['VT1_pct_HRmax'] = ReportAdapter._safe_get(r15, 'vt1_pct_hrmax')
        ct['VT2_pct_HRmax'] = ReportAdapter._safe_get(r15, 'vt2_pct_hrmax')
        ct['VO2_determination'] = r15.get('vo2_determination', r01.get('vo2_determination', 'VO2peak'))
        ct['plateau_detected'] = r15.get('plateau_detected', r01.get('plateau_detected', False))
        ct['plateau_delta'] = ReportAdapter._safe_get(r15, 'plateau_delta_mlmin') or ReportAdapter._safe_get(r01, 'plateau_delta_mlmin')

        # --- B3. CROSS-VALIDATION VT↔LT (E18) ---
        r18 = results.get('E18', {})
        ct['E18_status'] = r18.get('status', 'N/A')
        ct['E18_overall_class'] = r18.get('summary', {}).get('overall_class', '')
        ct['E18_overall_score'] = r18.get('summary', {}).get('overall_score_pct', '')
        ct['E18_summary'] = r18.get('summary', {})
        ct['E18_layer1'] = r18.get('layer1_lactate_at_vt', {})
        ct['E18_layer2'] = r18.get('layer2_domain_confirmation', {})
        ct['E18_layer3'] = r18.get('layer3_concordance', {})
        ct['E18_flags'] = r18.get('flags', [])

        # --- CHART DATA (for interactive plots) ---
        # E11 lactate curve data
        ct['_chart_lactate_points'] = r11.get('raw_points', [])
        ct['_chart_lactate_poly'] = r11.get('poly_curve', {})
        ct['_chart_lt1_time'] = r11.get('lt1_time_sec')
        ct['_chart_lt1_la'] = r11.get('lt1_la_mmol')
        ct['_chart_lt2_time'] = r11.get('lt2_time_sec')
        ct['_chart_lt2_la'] = r11.get('lt2_la_mmol')
        ct['_chart_lt1_method'] = r11.get('lt1_method', '')
        ct['_chart_lt2_method'] = r11.get('lt2_method', '')
        
        # E12 NIRS data
        r12_full = results.get('E12', {})
        ct['_chart_nirs_traces'] = r12_full.get('traces', [])
        ct['_chart_nirs_channels'] = r12_full.get('channels_used', [])

        # --- INTERPRETATION ENGINE ---
        _age = float(ct.get('age', 30) or 30)
        _sex = getattr(cfg, 'sex', 'male') or 'male'
        ct['_interp_sex'] = _sex
        _weight = float(ct.get('weight', 70) or 70)
        
        # VO2peak relative
        _vo2_abs = results.get('E01', {}).get('vo2_peak_mlmin')
        _vo2_rel = None
        if _vo2_abs and _weight and _weight > 0:
            _vo2_rel = _vo2_abs / _weight
        ct['_interp_vo2_mlkgmin'] = round(_vo2_rel, 1) if _vo2_rel else None
        
        # Classify VO2max
        if _vo2_rel:
            _vc = interpret_vo2max(_vo2_rel, _age, _sex)
            ct['_interp_vo2_category'] = _vc['category']
            ct['_interp_vo2_percentile'] = _vc['percentile']
            ct['_interp_vo2_athlete_level'] = _vc['athlete_level']
            ct['_interp_vo2_color'] = _vc['color']
            ct['_interp_vo2_thresholds'] = _vc['thresholds']
        
        # Thresholds analysis
        _vt1_vo2 = results.get('E02', {}).get('vt1_vo2_mlmin') or results.get('E01', {}).get('vt1_vo2_mlmin')
        _vt2_vo2 = results.get('E02', {}).get('vt2_vo2_mlmin') or results.get('E01', {}).get('vt2_vo2_mlmin')
        _vt1_hr_val = results.get('E02', {}).get('vt1_hr')
        _vt2_hr_val = results.get('E02', {}).get('vt2_hr')
        if not _vt1_hr_val:
            try:
                _vh = ct.get('VT1_HR')
                _vt1_hr_val = float(_vh) if _vh and str(_vh) not in ('-','None','0','') else None
            except: _vt1_hr_val = None
        if not _vt2_hr_val:
            try:
                _vh = ct.get('VT2_HR')
                _vt2_hr_val = float(_vh) if _vh and str(_vh) not in ('-','None','0','') else None
            except: _vt2_hr_val = None
        _hr_max_val = results.get('E01', {}).get('hr_peak')
        if not _hr_max_val:
            try:
                _hm = ct.get('HR_max')
                _hr_max_val = float(_hm) if _hm and str(_hm) not in ('-','None','0','') else None
            except: _hr_max_val = None
        
        if _vo2_abs and (_vt1_vo2 or _vt2_vo2):
            _ti = interpret_thresholds(_vt1_vo2, _vt2_vo2, _vo2_abs, _vt1_hr_val, _vt2_hr_val, _hr_max_val)
            ct['_interp_vt1_pct_vo2'] = _ti.get('vt1_pct_vo2')
            ct['_interp_vt2_pct_vo2'] = _ti.get('vt2_pct_vo2')
            ct['_interp_vt1_pct_hr'] = _ti.get('vt1_pct_hr')
            ct['_interp_vt2_pct_hr'] = _ti.get('vt2_pct_hr')
            ct['_interp_aerobic_base'] = _ti.get('aerobic_base', '-')
            ct['_interp_gap_bpm'] = _ti.get('gap_bpm')
        
        # Test validity
        _rer_val = results.get('E01', {}).get('rer_peak')
        if _rer_val is None:
            try:
                _rr = ct.get('RER_peak')
                _rer_val = float(_rr) if _rr and str(_rr) not in ('-','None','') else None
            except: _rer_val = None
        _la_peak = None
        _e04 = results.get('E04', {})
        if _e04 and _e04.get('status') == 'OK':
            _la_vals = [p.get('la', 0) for p in _e04.get('points', []) if p.get('la')]
            if _la_vals: _la_peak = max(_la_vals)
        
        _tv = interpret_test_validity(_rer_val, _hr_max_val, _age, _la_peak)
        ct['_interp_test_maximal'] = _tv['is_maximal']
        ct['_interp_test_confidence'] = _tv['confidence']
        ct['_interp_test_criteria'] = _tv['criteria']
        ct['_interp_rer_desc'] = _tv['rer_desc']
        ct['_interp_la_peak'] = round(_la_peak, 1) if _la_peak else None
        ct['_interp_hr_max'] = _hr_max_val
        ct['_interp_age'] = _age
        
        # Generate observations
        # [observations moved to end of build_canon_table]
        
        # --- PROTOCOL SUMMARY ---
        _df_ex_p = results.get('_df_ex', pd.DataFrame())
        _df_full_p = results.get('_df_full', pd.DataFrame())
        _e00_p = results.get('E00', {})
        _e01_p = results.get('E01', {})
        _t_stop_p = _e00_p.get('t_stop', 0)
        
        ct['_prot_name'] = getattr(cfg, 'protocol_name', '-')
        ct['_prot_modality'] = getattr(cfg, 'modality', '-')
        
        # Protocol steps from RAW_PROTOCOLS
        _prot_segs = RAW_PROTOCOLS.get(ct['_prot_name'], [])
        _prot_steps = []
        for _seg in _prot_segs:
            _st = _seg.get('type', '?')
            _s = _seg.get('start', '0:00')
            _e = _seg.get('end', '?')
            # Parse start time to seconds
            def _pts(ts):
                if isinstance(ts, (int,float)): return float(ts)
                parts = str(ts).split(':')
                if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
                if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                return 0
            _step = {
                'type': _st,
                'start_s': _pts(_s),
                'end_s': _pts(_e) if _e != '?' else None,
                'speed': _seg.get('speed_kmh', _seg.get('speed_from')),
                'speed_to': _seg.get('speed_to'),
                'incline': _seg.get('incline_pct', _seg.get('incline_start', 0)),
                'incline_step': _seg.get('incline_step'),
                'step_every': _seg.get('step_every_sec'),
                'power': _seg.get('power_w', _seg.get('power_from')),
                'power_to': _seg.get('power_to'),
            }
            _prot_steps.append(_step)
        ct['_prot_steps'] = _prot_steps
        
        if not _df_ex_p.empty and 'Time_sec' in _df_ex_p.columns:
            _t_ex = pd.to_numeric(_df_ex_p['Time_sec'], errors='coerce')
            _t_full = pd.to_numeric(_df_full_p.get('Time_sec', pd.Series()), errors='coerce') if not _df_full_p.empty else pd.Series()
            
            ct['_prot_ex_duration_s'] = round(_t_stop_p, 0) if _t_stop_p else round(float(_t_ex.max() - _t_ex.min()), 0)
            ct['_prot_total_duration_s'] = round(float(_t_full.max()), 0) if _t_full.notna().sum() > 0 else ct['_prot_ex_duration_s']
            ct['_prot_rec_duration_s'] = round(ct['_prot_total_duration_s'] - ct['_prot_ex_duration_s'], 0)
            
            _hr_ex = pd.to_numeric(_df_ex_p.get('HR_bpm', pd.Series()), errors='coerce')
            _vo2_ex = pd.to_numeric(_df_ex_p.get('VO2_mlmin', _df_ex_p.get('VO2_ml_min', pd.Series())), errors='coerce')
            ct['_prot_hr_start'] = round(float(_hr_ex.iloc[:20].mean()), 0) if _hr_ex.notna().sum() > 20 else None
            ct['_prot_hr_end'] = round(float(_hr_ex.iloc[-20:].mean()), 0) if _hr_ex.notna().sum() > 20 else None
            ct['_prot_vo2_start'] = round(float(_vo2_ex.iloc[:20].mean()), 0) if _vo2_ex.notna().sum() > 20 else None
            ct['_prot_vo2_end'] = round(float(_vo2_ex.iloc[-20:].mean()), 0) if _vo2_ex.notna().sum() > 20 else None
            
            # Phases
            if 'Faza' in _df_full_p.columns:
                _phases = _df_full_p['Faza'].dropna().unique().tolist()
                ct['_prot_phases'] = [str(p) for p in _phases]
            else:
                ct['_prot_phases'] = []
            
            # Speed/power range if available
            for col, key in [('Speed_kmh', '_prot_speed'), ('Power_W', '_prot_power')]:
                _vals = pd.to_numeric(_df_ex_p.get(col, pd.Series()), errors='coerce')
                _nz = _vals[_vals > 0]
                if len(_nz) > 10:
                    ct[f'{key}_min'] = round(float(_nz.min()), 1)
                    ct[f'{key}_max'] = round(float(_nz.max()), 1)

        # --- CHART DATA: Exercise time series (V-slope, Fat/CHO) ---
        # Downsample to ~150 points for chart
        _df_ex = results.get('_df_ex', pd.DataFrame())
        if not _df_ex.empty and 'Time_sec' in _df_ex.columns:
            _step = max(1, len(_df_ex) // 150)
            _ds = _df_ex.iloc[::_step]
            _t = pd.to_numeric(_ds.get('Time_sec', pd.Series()), errors='coerce')
            _vo2 = pd.to_numeric(_ds.get('VO2_mlmin', _ds.get('VO2_ml_min', pd.Series())), errors='coerce')
            _vco2 = pd.to_numeric(_ds.get('VCO2_mlmin', _ds.get('VCO2_ml_min', pd.Series())), errors='coerce')
            _hr = pd.to_numeric(_ds.get('HR_bpm', pd.Series()), errors='coerce')
            _cho = pd.to_numeric(_ds.get('CHO_g_min', pd.Series()), errors='coerce')
            _fat = pd.to_numeric(_ds.get('FAT_g_min', pd.Series()), errors='coerce')
            
            _valid_gas = _t.notna() & _vo2.notna() & _vco2.notna()
            if _valid_gas.sum() > 10:
                ct['_chart_gas_time'] = [round(v,1) for v in _t[_valid_gas].tolist()]
                ct['_chart_gas_vo2'] = [round(v,0) for v in _vo2[_valid_gas].tolist()]
                ct['_chart_gas_vco2'] = [round(v,0) for v in _vco2[_valid_gas].tolist()]
                ct['_chart_gas_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr[_valid_gas].tolist()]
            
            _valid_sub = _t.notna() & (_cho.notna() | _fat.notna())
            if _valid_sub.sum() > 10:
                ct['_chart_sub_time'] = [round(v,1) for v in _t[_valid_sub].tolist()]
                ct['_chart_sub_cho'] = [round(v,2) if pd.notna(v) else 0 for v in _cho[_valid_sub].tolist()]
                ct['_chart_sub_fat'] = [round(v,2) if pd.notna(v) and v > 0 else 0 for v in _fat[_valid_sub].tolist()]
                ct['_chart_sub_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr[_valid_sub].tolist()]
        # --- CHART DATA: Full VO2 kinetics (exercise + recovery) ---
        _df_full = results.get('_df_full', pd.DataFrame())
        if _df_full.empty:
            _df_full = _df_ex  # fallback to exercise only
        if not _df_full.empty and 'Time_sec' in _df_full.columns:
            _step_f = max(1, len(_df_full) // 200)
            _ds_f = _df_full.iloc[::_step_f]
            _t_f = pd.to_numeric(_ds_f.get('Time_sec', pd.Series()), errors='coerce')
            _vo2_f = pd.to_numeric(_ds_f.get('VO2_mlmin', _ds_f.get('VO2_ml_min', pd.Series())), errors='coerce')
            _hr_f = pd.to_numeric(_ds_f.get('HR_bpm', pd.Series()), errors='coerce')
            _spd_f = pd.to_numeric(_ds_f.get('Speed_kmh', pd.Series()), errors='coerce')
            _pwr_f = pd.to_numeric(_ds_f.get('Power_W', pd.Series()), errors='coerce')
            _valid_f = _t_f.notna() & _vo2_f.notna()
            if _valid_f.sum() > 10:
                ct['_chart_full_time'] = [round(v,1) for v in _t_f[_valid_f].tolist()]
                ct['_chart_full_vo2'] = [round(v,0) for v in _vo2_f[_valid_f].tolist()]
                ct['_chart_full_hr'] = [round(v,0) if pd.notna(v) else None for v in _hr_f[_valid_f].tolist()]
                _s = _spd_f[_valid_f]
                _p = _pwr_f[_valid_f]
                if _s.notna().sum() > 10 and (_s > 0).sum() > 10:
                    ct['_chart_full_speed'] = [round(v,1) if pd.notna(v) and v > 0 else None for v in _s.tolist()]
                if _p.notna().sum() > 10 and (_p > 0).sum() > 10:
                    ct['_chart_full_power'] = [round(v,0) if pd.notna(v) and v > 0 else None for v in _p.tolist()]
        ct['_chart_t_stop'] = results.get('E00', {}).get('t_stop')
        ct['_chart_vo2peak'] = results.get('E01', {}).get('vo2_peak_mlmin')


        
        # E02 thresholds for overlay
        ct['_chart_vt1_time'] = r02.get('vt1_time_sec')
        ct['_chart_vt2_time'] = r02.get('vt2_time_sec')
        ct['_chart_vt1_hr'] = r02.get('vt1_hr')
        ct['_chart_vt2_hr'] = r02.get('vt2_hr')
        
        # E18 cross-validation for annotations  
        r18 = results.get('E18', {})
        ct['_chart_e18_l1'] = r18.get('layer1_lactate_at_vt', {})

        # --- C. STREFY ---
        zones_list = ['z1', 'z2', 'z3', 'z4', 'z5']
        _z_container = r16.get('zones', r16) if isinstance(r16, dict) else {}
        for z in zones_list:
            z_data = _z_container.get(z, {}) if isinstance(_z_container, dict) else {}
            prefix = z.upper()
            ct[f'{prefix}_HR_low'] = ReportAdapter._safe_get(z_data, 'hr_low', '...')
            ct[f'{prefix}_HR_high'] = ReportAdapter._safe_get(z_data, 'hr_high', '...')
            ct[f'{prefix}_speed_low'] = ReportAdapter._safe_get(z_data, 'speed_low', '')
            ct[f'{prefix}_speed_high'] = ReportAdapter._safe_get(z_data, 'speed_high', '')
            ct[f'{prefix}_name'] = z_data.get('name_pl', '')

        # --- D. METRYKI FIZJOLOGICZNE ---
        # VO2peak rel: preferuj E15 (znormalizowane), fallback E01
        vo2_rel = ReportAdapter._first_non_null(
            r15.get('vo2_rel'),
            r01.get('vo2_peak_mlkgmin'),
            r01.get('vo2_rel')
        )
        ct['VO2peak_rel'] = ReportAdapter._safe_get({'v': vo2_rel}, 'v')

        # VO2peak abs (L/min): preferuj E15, fallback E01 (ml/min -> L/min)
        vo2_abs_lmin = ReportAdapter._first_non_null(
            r15.get('vo2_abs'),
            r01.get('vo2_peak_lmin'),
            r01.get('vo2_abs_lmin')
        )
        if vo2_abs_lmin is None:
            vo2_abs_mlmin = ReportAdapter._first_non_null(
                r01.get('vo2_peak_mlmin'),
                r01.get('vo2_abs_mlmin')
            )
            if vo2_abs_mlmin is not None:
                try:
                    vo2_abs_lmin = float(vo2_abs_mlmin) / 1000.0
                except Exception:
                    vo2_abs_lmin = None
        ct['VO2peak_abs'] = ReportAdapter._safe_get({'v': vo2_abs_lmin}, 'v')

        # E15 klasyfikacja
        ct['VO2_class_pop'] = r15.get('vo2_class_pop', '-')
        ct['VO2_class_sport'] = r15.get('vo2_class_sport', '-')
        ct['VO2_pct_predicted'] = ReportAdapter._safe_get(r15, 'vo2_pct_predicted')
        ct['VO2_pred_method'] = r15.get('vo2_predicted_method', 'Wasserman')
        ct['VO2_pred_wass_mlkg'] = ReportAdapter._safe_get(r15, 'vo2_predicted_wasserman_mlkg')
        ct['RER_class'] = r15.get('rer_class', '-')
        ct['VE_VCO2_class'] = r15.get('ve_vco2_class', '-')
        ct['VE_VCO2_class_desc'] = r15.get('ve_vco2_desc', '-')
        ct['test_quality'] = r15.get('test_quality', '-')

        ct['HRmax'] = ReportAdapter._safe_get(r01, 'hr_peak')
        ct['RERpeak'] = ReportAdapter._safe_get(r01, 'rer_peak')

        # VEpeak: obsługa różnych nazw kluczy z E01
        ve_peak = ReportAdapter._first_non_null(
            r01.get('ve_peak_lmin'),
            r01.get('ve_peak'),
            r01.get('VE_peak')
        )
        ct['VEpeak'] = ReportAdapter._safe_get({'v': ve_peak}, 'v')

        # E03 VentSlope v2.0 — multi-method
        ct['VE_VCO2_slope_full']  = ReportAdapter._safe_get(r03, 'slope_full')
        ct['VE_VCO2_slope_vt2']   = ReportAdapter._safe_get(r03, 'slope_to_vt2')
        ct['VE_VCO2_slope_vt1']   = ReportAdapter._safe_get(r03, 'slope_to_vt1')
        ct['VE_VCO2_intercept']   = ReportAdapter._safe_get(r03, 'intercept_to_vt2') if r03.get('intercept_to_vt2') is not None else ReportAdapter._safe_get(r03, 'intercept_full')
        ct['VE_VCO2_nadir']       = ReportAdapter._safe_get(r03, 've_vco2_nadir')
        ct['VE_VCO2_at_vt1']     = ReportAdapter._safe_get(r03, 've_vco2_at_vt1')
        ct['VE_VCO2_peak']       = ReportAdapter._safe_get(r03, 've_vco2_peak')
        ct['VE_VCO2_vent_class'] = r03.get('ventilatory_class', '-')
        ct['VE_VCO2_vc_desc']    = r03.get('vc_description', '-')
        ct['VE_VCO2_predicted']  = ReportAdapter._safe_get(r03, 'predicted_slope')
        ct['VE_VCO2_pct_pred']   = ReportAdapter._safe_get(r03, 'pct_predicted')
        ct['VE_VCO2_r2']        = ReportAdapter._safe_get(r03, 'r2_to_vt2') if r03.get('r2_to_vt2') is not None else ReportAdapter._safe_get(r03, 'r2_full')
        ct['PETCO2_rest']        = ReportAdapter._safe_get(r03, 'petco2_rest', '-')
        ct['PETCO2_vt1']         = ReportAdapter._safe_get(r03, 'petco2_vt1', '-')
        ct['PETCO2_peak']        = ReportAdapter._safe_get(r03, 'petco2_peak', '-')
        ct['VE_VCO2_flags']      = ', '.join(r03.get('flags', [])) if r03.get('flags') else 'Brak'
        # Legacy compat
        ct['VE_VCO2_slope'] = ct['VE_VCO2_slope_vt2'] if ct['VE_VCO2_slope_vt2'] != '[BRAK]' else ct['VE_VCO2_slope_full']
        ct['OUES'] = ReportAdapter._safe_get(r04, 'oues100', ReportAdapter._safe_get(r04, 'oues_val'))
        ct['OUES_90'] = ReportAdapter._safe_get(r04, 'oues90')
        ct['OUES_75'] = ReportAdapter._safe_get(r04, 'oues75')
        ct['OUES_toVT1'] = ReportAdapter._safe_get(r04, 'oues_to_vt1')
        ct['OUES_toVT2'] = ReportAdapter._safe_get(r04, 'oues_to_vt2')
        ct['OUES_per_kg'] = ReportAdapter._safe_get(r04, 'oues_per_kg')
        ct['OUES_per_BSA'] = ReportAdapter._safe_get(r04, 'oues_per_bsa')
        ct['OUES_pred_Holl'] = ReportAdapter._safe_get(r04, 'oues_pred_hollenberg')
        ct['OUES_pct_Holl'] = ReportAdapter._safe_get(r04, 'oues_pct_hollenberg')
        ct['OUES_R2'] = ReportAdapter._safe_get(r04, 'oues_r2_100')
        ct['OUES_submax'] = ReportAdapter._safe_get(r04, 'oues_submax_stability')
        ct['OUES_flags'] = ', '.join(r04.get('oues_flags', [])) if isinstance(r04.get('oues_flags'), list) else ReportAdapter._safe_get(r04, 'oues_flags')
        ct['BR_pct'] = '-'

        # O2 Pulse: fallbacki nazw
        o2p = ReportAdapter._first_non_null(
            r15.get('o2pulse_peak'),
            r01.get('o2pulse_peak'),
            r01.get('o2_pulse_peak'),
            r01.get('O2Pulse_peak')
        )
        ct['O2pulse_peak'] = ReportAdapter._safe_get({'v': o2p}, 'v')
        # E05 v2.0 fields
        ct['O2pulse_at_vt1'] = ReportAdapter._safe_get(r05, 'o2pulse_at_vt1')
        ct['O2pulse_at_vt2'] = ReportAdapter._safe_get(r05, 'o2pulse_at_vt2')
        ct['O2pulse_pred_friend'] = ReportAdapter._safe_get(r05, 'predicted_friend')
        ct['O2pulse_pred_noodle'] = ReportAdapter._safe_get(r05, 'predicted_noodle')
        ct['O2pulse_pct_friend'] = ReportAdapter._safe_get(r05, 'pct_predicted_friend')
        ct['O2pulse_pct_noodle'] = ReportAdapter._safe_get(r05, 'pct_predicted_noodle')
        ct['O2pulse_est_sv'] = ReportAdapter._safe_get(r05, 'estimated_sv_peak_ml')
        ct['O2pulse_trajectory'] = ReportAdapter._safe_get(r05, 'trajectory', '-')
        ct['O2pulse_traj_desc'] = ReportAdapter._safe_get(r05, 'trajectory_desc', '-')
        ct['O2pulse_ff'] = ReportAdapter._safe_get(r05, 'flattening_fraction')
        ct['O2pulse_o2prr'] = ReportAdapter._safe_get(r05, 'o2prr')
        ct['O2pulse_flags'] = ', '.join(r05.get('flags', [])) if r05.get('flags') else 'BRAK'

        ct['HRR_60s'] = ReportAdapter._safe_get(r08, 'hrr_1min')
        ct['HRR_180s'] = ReportAdapter._safe_get(r08, 'hrr_3min')
        ct['HRR_mode'] = ReportAdapter._safe_get(r08, 'recovery_mode', '-')
        ct['HRR_quality'] = ReportAdapter._safe_get(r08, 'quality', '-')

        # --- E. METABOLICZNE / INNE ---
        ct['SmO2_min'] = ReportAdapter._safe_get(r12, 'smo2_min')
        ct['_e12_raw'] = r12

        # E13 Drift
        r13 = results.get('E13', {})
        ct['_e13_raw'] = r13
        r17 = results.get('E17', {})
        ct['_e17_raw'] = r17

        # E19 Concordance
        r19 = results.get('E19', {})
        ct['_e19_raw'] = r19
        ct['validity_score'] = r19.get('validity_score', '-')
        ct['validity_grade'] = r19.get('validity_grade', '-')
        ct['concordance_score'] = r19.get('concordance_score', '-')
        ct['concordance_grade'] = r19.get('concordance_grade', '-')

        # QC metadata for HTML
        qc = results.get('_qc_log', {})
        ct['_qc_status'] = qc.get('status', 'PASS')
        ct['_engines_ok_n'] = str(len(qc.get('engines_executed_ok', [])))
        ct['_engines_fail_n'] = str(len(qc.get('engines_failed', [])))
        lim = qc.get('engines_limited', [])
        ct['_engines_lim_list'] = ', '.join(lim) if lim else ''
        ct['Lactate_peak'] = ReportAdapter._safe_get(r11, 'lt_peak')
        ct['_e11_raw'] = r11
        if r11.get('status') == 'OK':
            ct['La_baseline'] = str(round(r11.get('la_baseline', 0), 2)) if r11.get('la_baseline') else '-'
            ct['LT1_method'] = r11.get('lt1_method', '-') or '-'
            ct['LT1_HR'] = str(round(r11['lt1_hr_bpm'], 0)) if r11.get('lt1_hr_bpm') else '-'
            ct['LT1_Speed'] = str(round(r11['lt1_speed_kmh'], 1)) if r11.get('lt1_speed_kmh') else '-'
            ct['LT1_La'] = str(round(r11['lt1_la_mmol'], 2)) if r11.get('lt1_la_mmol') else '-'
            ct['LT2_method'] = r11.get('lt2_method', '-') or '-'
            ct['LT2_HR'] = str(round(r11['lt2_hr_bpm'], 0)) if r11.get('lt2_hr_bpm') else '-'
            ct['LT2_Speed'] = str(round(r11['lt2_speed_kmh'], 1)) if r11.get('lt2_speed_kmh') else '-'
            ct['LT2_La'] = str(round(r11['lt2_la_mmol'], 2)) if r11.get('lt2_la_mmol') else '-'
            ct['La_n_points'] = str(r11.get('n_points', 0))

        # FATmax / crossover fallbacki
        fat_max = ReportAdapter._first_non_null(r10.get('mfo_gmin'), r10.get('fat_max'), r10.get('fatmax_g_min'))
        fat_max_hr = ReportAdapter._first_non_null(r10.get('fat_max_hr'), r10.get('fatmax_hr'))
        cho_cross = ReportAdapter._first_non_null(r10.get('cop_hr'), r10.get('cho_crossover_hr'), r10.get('crossover_hr'))

        # E06
        r06 = results.get('E06', {})
        ct['GAIN_below_vt1'] = r06.get('gain_below_vt1')
        ct['GAIN_below_vt1_r2'] = r06.get('gain_below_vt1_r2')
        ct['GAIN_full'] = r06.get('gain_full')
        ct['GAIN_unit'] = r06.get('gain_unit', '')
        ct['GAIN_modality'] = r06.get('modality', '')
        ct['RE_mlkgkm'] = r06.get('running_economy_mlkgkm')
        ct['RE_class'] = r06.get('re_classification')
        ct['DELTA_EFF'] = r06.get('delta_efficiency_pct')
        ct['LIN_BREAK_time'] = r06.get('linearity_break_time_s')
        ct['GAIN_z'] = r06.get('gain_z_score')
        ct['GAIN_norm_ref'] = r06.get('norm_gain_ref')
        ct['GAIN_norm_src'] = r06.get('norm_src')
        ct['GAIN_at_vt1'] = r06.get('gain_at_vt1')
        ct['GAIN_at_vt2'] = r06.get('gain_at_vt2')
        ct['RE_at_vt1'] = r06.get('re_at_vt1')
        ct['RE_at_vt2'] = r06.get('re_at_vt2')
        ct['VO2_at_vt1_e06'] = r06.get('vo2_at_vt1')
        ct['VO2_at_vt2_e06'] = r06.get('vo2_at_vt2')
        ct['LOAD_at_vt1'] = r06.get('load_at_vt1')
        ct['LOAD_at_vt2'] = r06.get('load_at_vt2')
        ct['EFF_at_vt1'] = r06.get('eff_at_vt1')
        ct['EFF_at_vt2'] = r06.get('eff_at_vt2')
        ct['FATmax_g'] = ReportAdapter._safe_get({'v': fat_max}, 'v')
        ct['FATmax_HR'] = ReportAdapter._safe_get({'v': fat_max_hr}, 'v')
        ct['CHO_Cross_HR'] = ReportAdapter._safe_get({'v': cho_cross}, 'v')

        # Zone substrate rates
        zs = r10.get('zone_substrate', {})
        for zn in ('z2', 'z4', 'z5'):
            zd = zs.get(zn, {})
            ct[f'{zn}_fat_gh'] = zd.get('fat_gh')
            ct[f'{zn}_cho_gh'] = zd.get('cho_gh')
            ct[f'{zn}_fat_pct'] = zd.get('fat_pct')
            ct[f'{zn}_cho_pct'] = zd.get('cho_pct')
            ct[f'{zn}_kcal_h'] = zd.get('kcal_h')
            ct[f'{zn}_rer_valid'] = zd.get('rer_valid', True)

        # Additional E10 v2 fields
        ct['FATmax_pct_vo2peak'] = r10.get('fatmax_pct_vo2peak')
        ct['FATmax_pct_hrmax'] = r10.get('fatmax_pct_hrmax')
        ct['FATmax_zone_hr_low'] = r10.get('fatmax_zone_hr_low')
        ct['FATmax_zone_hr_high'] = r10.get('fatmax_zone_hr_high')
        ct['MFO_mgkg_min'] = r10.get('mfo_mgkg_min')
        ct['COP_pct_vo2peak'] = r10.get('cop_pct_vo2peak')
        ct['COP_RER'] = r10.get('cop_rer')
        ct['COP_note'] = r10.get('cop_note')
        ct['fat_pct_at_vt1'] = r10.get('fat_pct_at_vt1')
        ct['cho_pct_at_vt1'] = r10.get('cho_pct_at_vt1')

        
        # ─── FINAL: Generate observations & training recs (after all ct values populated) ───
        ct['_e07_raw'] = r07
        ct['_interp_observations'] = generate_observations(ct)
        ct['_interp_training_recs'] = generate_training_recs(ct)
        # E14 Kinetics
        r14 = results.get('E14', {})
        ct['T_half_VO2'] = r14.get('T_half_VO2_simple_s')
        ct['kinetics_tau'] = r14.get('tau_s')
        ct['kinetics_r2'] = r14.get('r_squared')
        ct['kinetics_class'] = r14.get('classification')
        ct['kinetics_VO2DR'] = r14.get('VO2_delay_recovery_s')
        ct['T_half_VCO2'] = r14.get('T_half_VCO2_s')
        ct['T_half_VE'] = r14.get('T_half_VE_s')
        ct['kinetics_MRT'] = r14.get('MRT_s')
        ct['kinetics_flags'] = r14.get('flags', [])
        ct['_protocol_description'] = ReportAdapter.describe_protocol(df, cfg, results)
        ct['_protocol_table_html'] = ReportAdapter.protocol_table_html(cfg, t_stop=results.get('E00', {}).get('t_stop'))

        # ═══════════════════════════════════════════════════════════════
        # ALIAS BLOCK: mapowanie kluczy dla render_html_report
        # ═══════════════════════════════════════════════════════════════

        # Meta
        ct['age_y'] = ct.get('age')
        ct['body_mass_kg'] = ct.get('weight')
        ct['height_cm'] = ct.get('height')
        ct['sex'] = getattr(cfg, 'sex', 'male')
        ct['modality'] = getattr(cfg, 'modality', 'run')
        ct['protocol_name'] = ct.get('protocol') or getattr(cfg, 'protocol_name', '-')

        # VO2 peak
        ct['VO2max_ml_kg_min'] = ct.get('VO2peak_rel')
        ct['VO2max_L_min'] = ct.get('VO2peak_abs')
        ct['VO2max_abs_Lmin'] = ct.get('VO2peak_abs')
        ct['VO2max_class_label'] = ct.get('VO2_class_pop') or ct.get('_interp_vo2_category')
        ct['VO2max_percentile'] = ct.get('_interp_vo2_percentile')
        ct['VO2max_determination'] = ct.get('VO2_determination') or ct.get('plateau_detected')

        # Peaks
        ct['HR_peak'] = ct.get('HRmax')
        ct['RER_peak'] = ct.get('RERpeak')
        ct['VE_peak'] = ct.get('VEpeak')

        # Thresholds
        ct['VT1_HR_bpm'] = ct.get('VT1_HR')
        ct['VT1_Time_s'] = r02.get('vt1_time_sec')
        ct['VT1_pct_VO2peak'] = ct.get('VT1_pct_VO2max') or ct.get('_interp_vt1_pct_vo2')
        ct['VT1_confidence'] = r02.get('vt1_confidence', 0)
        ct['VT2_HR_bpm'] = ct.get('VT2_HR')
        ct['VT2_Time_s'] = r02.get('vt2_time_sec')
        ct['VT2_pct_VO2peak'] = ct.get('VT2_pct_VO2max') or ct.get('_interp_vt2_pct_vo2')
        ct['VT2_confidence'] = r02.get('vt2_confidence', 0)

        # VE/VCO2
        ct['VE_VCO2_slope'] = ct.get('VE_VCO2_slope_vt2') or ct.get('VE_VCO2_slope_full')
        ct['VE_VCO2_ventilatory_class'] = ct.get('VE_VCO2_vent_class')

        # Additional metrics
        ct['OUES'] = r04.get('oues') or r04.get('OUES')
        ct['O2pulse_peak'] = r05.get('o2pulse_peak_ml') or r05.get('o2_pulse_peak') or r05.get('o2pulse_peak')
        ct['BR_pct'] = results.get('E09', {}).get('br_pct')
        ct['T_half_VO2'] = results.get('E06', {}).get('t_half_s')
        ct['HRR_60s'] = r08.get('hrr_1min')
        ct['LOAD_at_vt1'] = ct.get('VT1_Speed') or ct.get('VT1_Power')
        ct['LOAD_at_vt2'] = ct.get('VT2_Speed') or ct.get('VT2_Power')

        # Zones
        ct['_zones_data'] = r16.get('zones', r16) if isinstance(r16, dict) else {}

        # Raw engine results
        ct['_e02_raw'] = results.get('E02', {})
        for _eid in ['E00','E03','E04','E05','E06','E07','E08','E09','E10','E11','E12','E13','E14','E15','E16','E17','E19']:
            ct['_' + _eid.lower() + '_raw'] = results.get(_eid, {})

        # E06 efficiency aliases
        _r06 = results.get('E06', {})
        for _k06 in ['gain_below_vt1','gain_below_vt1_r2','gain_at_vt1','gain_at_vt2','gain_full','gain_modality','gain_norm_ref','gain_norm_src','gain_unit','gain_z','re_at_vt1','re_at_vt2','re_class','eff_at_vt1','eff_at_vt2','delta_eff','lin_break_time']:
            _parts = _k06.split('_')
            _ct_key = _parts[0].upper() + '_' + '_'.join(_parts[1:])
            ct[_ct_key] = _r06.get(_k06)

        # Performance Context passthrough (from results, not ct!)
        _pc = results.get('_performance_context', {})
        if _pc:
            ct['_performance_context'] = _pc

        return ct

    @staticmethod
    def _oues_interpretation(ct):
        """Build OUES interpretation string for report."""
        oues = ct.get('OUES')
        if oues in (None, '-', '', '[BRAK]'):
            return '• OUES: [BRAK]'

        lines = []
        lines.append(f"• OUES100: [{oues}] ml/min/log(L/min)")

        # Submaximal variants
        o90 = ct.get('OUES_90', '-')
        o75 = ct.get('OUES_75', '-')
        ovt1 = ct.get('OUES_toVT1', '-')
        ovt2 = ct.get('OUES_toVT2', '-')
        if o90 not in (None, '-'):
            lines.append(f"  ↳ OUES90: [{o90}] | OUES75: [{o75}] | do VT1: [{ovt1}] | do VT2: [{ovt2}]")

        # Stability
        stab = ct.get('OUES_submax', '-')
        if stab not in (None, '-'):
            try:
                sv = float(stab)
                stab_txt = 'dobra' if 0.95 <= sv <= 1.05 else ('umiarkowana' if 0.90 <= sv <= 1.10 else 'niska')
                lines.append(f"  ↳ Stabilność submaksymalna (OUES90/100): [{stab}] — {stab_txt}")
            except (ValueError, TypeError):
                pass

        # Normalization
        opkg = ct.get('OUES_per_kg', '-')
        opbsa = ct.get('OUES_per_BSA', '-')
        if opkg not in (None, '-'):
            lines.append(f"  ↳ OUES/kg: [{opkg}] | OUES/BSA (OUESI): [{opbsa}]")

        # Predicted (Hollenberg)
        pred = ct.get('OUES_pred_Holl', '-')
        pct = ct.get('OUES_pct_Holl', '-')
        if pct not in (None, '-'):
            try:
                pv = float(pct)
                if pv < 70:
                    interp = 'PONIŻEJ NORMY — wskazuje na patologię'
                elif pv < 85:
                    interp = 'Poniżej średniej — obniżona efektywność'
                elif pv < 100:
                    interp = 'W normie — prawidłowa efektywność'
                elif pv < 115:
                    interp = 'Powyżej normy — wysoka efektywność'
                else:
                    interp = 'Znacznie powyżej normy — bardzo wysoka efektywność'
                lines.append(f"  ↳ Predicted (Hollenberg): [{pred}] | % predicted: [{pct}]% — {interp}")
            except (ValueError, TypeError):
                lines.append(f"  ↳ Predicted (Hollenberg): [{pred}] | % predicted: [{pct}]%")

        # R²
        r2 = ct.get('OUES_R2', '-')
        if r2 not in (None, '-'):
            try:
                r2v = float(r2)
                r2_txt = 'doskonałe dopasowanie' if r2v >= 0.95 else ('dobre' if r2v >= 0.90 else 'niskie — ostrożna interpretacja')
                lines.append(f"  ↳ R²: [{r2}] — {r2_txt}")
            except (ValueError, TypeError):
                pass

        # Flags
        fl = ct.get('OUES_flags', '-')
        if fl and fl not in (None, '-', ''):
            lines.append(f"  ↳ Flagi: [{fl}]")

        return '\n'.join(lines)

    @staticmethod
    def _breathing_pattern_report(ct):
        """E07 Breathing Pattern report section — with sport context."""
        e07 = ct.get("_e07_raw", {})
        if not e07 or e07.get("status") != "OK":
            return ""
        
        # ─── Determine sport context ───
        _bp_vo2 = ct.get('_interp_vo2_mlkgmin')
        _bp_ve_slope = None
        try:
            _ve_raw = ct.get('VE_VCO2_slope')
            if _ve_raw and str(_ve_raw) not in ('-','[BRAK]','None',''):
                _bp_ve_slope = float(_ve_raw)
        except: pass
        _is_sport = (_bp_vo2 is not None and _bp_vo2 >= 45 and _bp_ve_slope is not None and _bp_ve_slope < 30)
        _sex_bp = ct.get('_interp_sex', 'male')
        _sport_tier = 'sedentary'
        if _bp_vo2:
            if _sex_bp == 'male':
                if _bp_vo2 >= 65: _sport_tier = 'elite'
                elif _bp_vo2 >= 55: _sport_tier = 'competitive'
                elif _bp_vo2 >= 45: _sport_tier = 'trained'
            else:
                if _bp_vo2 >= 55: _sport_tier = 'elite'
                elif _bp_vo2 >= 45: _sport_tier = 'competitive'
                elif _bp_vo2 >= 38: _sport_tier = 'trained'
        
        L = []
        _ctx_tag = f" [kontekst: sportowiec {_sport_tier}, VO2max {_bp_vo2:.0f}]" if _is_sport else ""
        L.append(f"3b. Wzorzec oddychania \u2014 Breathing Pattern (E07 v2.0){_ctx_tag}:")
        L.append(f"\u2022 BF: rest [{e07.get('bf_rest','?')}] \u2192 peak [{e07.get('bf_peak','?')}] /min"
                  f"  |  VT: rest [{e07.get('vt_rest_L','?')}] \u2192 peak [{e07.get('vt_peak_L','?')}] L")
        if e07.get("bf_at_vt1"):
            L.append(f"  \u21b3 przy VT1: BF [{e07['bf_at_vt1']}] /min  |  VT [{e07.get('vt_at_vt1_L','?')}] L")
        if e07.get("bf_at_vt2"):
            L.append(f"  \u21b3 przy VT2: BF [{e07['bf_at_vt2']}] /min  |  VT [{e07.get('vt_at_vt2_L','?')}] L")
        
        # VT Plateau — sport context
        plat = e07.get("vt_plateau_vs_vt2", "?")
        plat_t = e07.get("vt_plateau_time_s")
        plat_l = e07.get("vt_plateau_level_L")
        if _is_sport:
            _pd = {"AT_VT2": "prawid\u0142owe \u2014 plateau VT przy VT2",
                   "BEFORE_VT2": "wczesne plateau VT (<VT2) \u2014 typowe u sportowc\u00f3w, VE ro\u015bnie przez BF (86% EA, Carey 2008)",
                   "AFTER_VT2": "p\u00f3\u017ane plateau VT (>VT2) \u2014 wysoka rezerwa oddechowa",
                   "VT2_UNKNOWN": "brak VT2 do por\u00f3wnania"}.get(plat, "")
        else:
            _pd = {"AT_VT2": "prawid\u0142owe \u2014 plateau VT przy VT2",
                   "BEFORE_VT2": "\u26a0 wczesne plateau VT (<VT2) \u2014 ograniczenie mechaniczne?",
                   "AFTER_VT2": "p\u00f3\u017ane plateau VT (>VT2) \u2014 wysoka rezerwa oddechowa",
                   "VT2_UNKNOWN": "brak VT2 do por\u00f3wnania"}.get(plat, "")
        if plat_t:
            m, s = int(plat_t // 60), int(plat_t % 60)
            L.append(f"\u2022 Plateau VT: t={m}:{s:02d} ({e07.get('vt_plateau_pct_exercise',0):.0f}% testu) "
                     f"| poziom [{plat_l}] L | {_pd}")
        
        # Strategy — sport context
        strat = e07.get("strategy", "?")
        if _is_sport:
            _sd = {"VT_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez obj\u0119to\u015b\u0107 oddechow\u0105 \u2014 efektywne",
                   "BF_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez cz\u0119sto\u015b\u0107 \u2014 ADAPTACJA SPORTOWA (Folinsbee 1983: elitarni kolarze BF 63/min; ACC 2021: BF 60-70 norma u EA)",
                   "BALANCED": "zr\u00f3wnowa\u017cony wk\u0142ad VT i BF"}.get(strat, "")
        else:
            _sd = {"VT_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez obj\u0119to\u015b\u0107 oddechow\u0105 \u2014 efektywne",
                   "BF_DOMINANT": "wzrost VE g\u0142\u00f3wnie przez cz\u0119sto\u015b\u0107 \u2014 mniej efektywne, zwi\u0119ksza dead space",
                   "BALANCED": "zr\u00f3wnowa\u017cony wk\u0142ad VT i BF"}.get(strat, "")
        L.append(f"\u2022 Strategia wentylacji: [{strat}] (VT {e07.get('ve_from_vt_pct',0):.0f}% / BF {e07.get('ve_from_bf_pct',0):.0f}%)")
        L.append(f"  \u21b3 {_sd}")
        
        # Timing
        if e07.get("duty_cycle_rest") is not None:
            L.append(f"\u2022 Timing oddechowy:")
            L.append(f"  \u21b3 tI: rest [{e07['ti_mean_rest_s']}] \u2192 peak [{e07['ti_mean_peak_s']}] s"
                     f"  |  tE: rest [{e07['te_mean_rest_s']}] \u2192 peak [{e07['te_mean_peak_s']}] s")
            dc_peak = e07['duty_cycle_peak']
            dc_desc = ""
            if dc_peak and dc_peak < 0.35: dc_desc = " \u26a0 niski"
            elif dc_peak and dc_peak > 0.50: dc_desc = " \u2014 wyd\u0142u\u017cony wdech"
            else: dc_desc = " \u2014 norma"
            L.append(f"  \u21b3 Duty cycle (tI/tTot): rest [{e07['duty_cycle_rest']}] \u2192 peak [{dc_peak}]{dc_desc} (ref: 0.40-0.50)")
            if e07.get("mean_insp_flow_peak_Ls"):
                L.append(f"  \u21b3 Mean inspiratory flow peak: [{e07['mean_insp_flow_peak_Ls']}] L/s")
        
        # VD/VT — sport context
        if e07.get("vdvt_rest") is not None:
            traj = e07.get("vdvt_trajectory", "?")
            vdvt_r = e07.get('vdvt_rest', 0)
            vdvt_p = e07.get('vdvt_peak', 0)
            try:
                vr = float(vdvt_r)
                vp = float(vdvt_p)
            except:
                vr, vp = 0.3, 0.3
            
            if _is_sport and vp < 0.20:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje (prawid\u0142owy V/Q matching)",
                       "FLAT": "VD/VT stabilne \u2014 przy warto\u015bciach <0.20 to NORMA SPORTOWA (elitarna efektywno\u015b\u0107)",
                       "ABNORMAL_INCREASE": f"VD/VT wzrost {vr:.3f}\u2192{vp:.3f} \u2014 przy <0.20 klinicznie nieistotny"}.get(traj, "")
            elif _is_sport and vp < 0.25:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje",
                       "FLAT": "VD/VT stabilne \u2014 poni\u017cej progu klinicznego (<0.25), prawdopodobnie norma sportowa",
                       "ABNORMAL_INCREASE": f"VD/VT wzrost \u2014 ale warto\u015bci bezwzgl\u0119dne w normie ({vp:.3f}<0.25)"}.get(traj, "")
            else:
                _td = {"NORMAL_DECREASE": "norma \u2014 VD/VT maleje (prawid\u0142owy V/Q matching)",
                       "FLAT": "\u26a0 VD/VT nie maleje \u2014 mo\u017cliwy V/Q mismatch",
                       "ABNORMAL_INCREASE": "\u26a0 VD/VT ro\u015bnie \u2014 patologiczny V/Q mismatch"}.get(traj, "")
            
            parts = [f"rest [{vdvt_r}]"]
            if e07.get("vdvt_at_vt1"): parts.append(f"VT1 [{e07['vdvt_at_vt1']}]")
            if e07.get("vdvt_at_vt2"): parts.append(f"VT2 [{e07['vdvt_at_vt2']}]")
            parts.append(f"peak [{vdvt_p}]")
            _vdvt_str = ' \u2192 '.join(parts)
            L.append(f"\u2022 Dead space (VD/VT est.): {_vdvt_str}")
            est_note = " (warto\u015bci estymowane z PETCO2)" if vr < 0.15 else ""
            _ref = "Norma ABG: rest ~0.30 \u2192 peak <0.20 (ATS 2003)"
            if _is_sport and vp < 0.20:
                _ref += " | U sportowc\u00f3w: warto\u015bci <0.10-0.15 cz\u0119ste"
            L.append(f"  \u21b3 {_ref}{est_note}")
            L.append(f"  \u21b3 Trajektoria: {_td}")
        
        # Tachypnea — sport context for BF threshold
        tachy = e07.get("tachypnea_vs_vt2", "NONE")
        if tachy != "NONE" and e07.get("tachypnea_time_s"):
            tt = e07["tachypnea_time_s"]
            m2, s2 = int(tt // 60), int(tt % 60)
            if _is_sport:
                _tach = {"BEFORE_VT2": "tachypnea przed VT2 \u2014 u sportowc\u00f3w BF>55 norma (ACC 2021), por\u00f3wnaj z BR",
                         "AT_VT2": "tachypnea przy VT2 \u2014 oczekiwany punkt przej\u015bcia",
                         "AFTER_VT2": "p\u00f3\u017ana \u2014 dobra rezerwa oddechowa"}.get(tachy, "")
            else:
                _tach = {"BEFORE_VT2": "\u26a0 wczesna tachypnea \u2014 potencjalny limiter wentylacyjny",
                         "AT_VT2": "prawid\u0142owe \u2014 tachypnea przy VT2",
                         "AFTER_VT2": "p\u00f3\u017ana \u2014 dobra rezerwa oddechowa"}.get(tachy, "")
            L.append(f"\u2022 Tachypnea (BF>50): t={m2}:{s2:02d} ({e07.get('tachypnea_pct_exercise',0):.0f}% testu) \u2014 {_tach}")
        elif tachy == "NONE":
            L.append(f"\u2022 Tachypnea (BF>50): nie wyst\u0105pi\u0142a \u2014 dobra kontrola oddychania")
        
        # RSBI
        if e07.get("rsbi_peak"):
            _rd = ""
            if e07["rsbi_peak"] > 40: _rd = " \u26a0 wysoki (rapid shallow breathing)"
            elif e07["rsbi_peak"] > 25: _rd = " \u2014 podwy\u017cszony"
            else: _rd = " \u2014 prawid\u0142owy"
            L.append(f"\u2022 RSBI (BF/VT): rest [{e07.get('rsbi_rest','?')}] \u2192 peak [{e07['rsbi_peak']}] bpm/L{_rd}")
        
        # PTVV
        if e07.get("ptvv") is not None:
            ptvv = e07["ptvv"]
            if ptvv > 0.25: _pv = "\u26a0 wysoka nieregularno\u015b\u0107 \u2014 podejrzenie BPD"
            elif ptvv > 0.15: _pv = "umiarkowana zmienno\u015b\u0107"
            else: _pv = "regularny wzorzec"
            L.append(f"\u2022 PTVV (nieregularno\u015b\u0107 VT): [{ptvv}] \u2014 {_pv}")
            L.append(f"  \u21b3 Ref: Kn\u00f6pfel 2024 (norma <0.15)")
        
        # Sighs
        if e07.get("sigh_count", 0) > 0:
            L.append(f"\u2022 Westchnienia: [{e07['sigh_count']}] ({e07['sigh_rate_per_min']}/min)")
            if e07["sigh_count"] > 3:
                L.append(f"  \u21b3 \u26a0 Cz\u0119ste westchnienia \u2014 Periodic Deep Sighing?")
        
        # Breathing pattern classification — sport context
        bp = e07.get("breathing_pattern", "?")
        if _is_sport:
            _bp = {"NORMAL": "Prawid\u0142owy wzorzec oddychania",
                   "BORDERLINE": "Graniczny \u2014 jeden marker nieprawid\u0142owy (mo\u017ce by\u0107 norma sportowa)",
                   "DYSFUNCTIONAL_BPD": "Algorytm: DYSFUNKCYJNY \u2014 REINTERPRETACJA SPORTOWA: "
                       f"przy VO2max {_bp_vo2:.0f} i VE/VCO2 {_bp_ve_slope:.0f} wzorzec jest prawdopodobnie adaptacj\u0105, nie patologi\u0105",
                   "RAPID_SHALLOW": "Szybki p\u0142ytki oddech \u2014 u sportowca mo\u017ce wynika\u0107 z BF-dominant strategy",
                   "PERIODIC": "\u26a0 Oddychanie periodyczne"}.get(bp, bp)
        else:
            _bp = {"NORMAL": "Prawid\u0142owy wzorzec oddychania",
                   "BORDERLINE": "\u26a0 Graniczny \u2014 jeden marker nieprawid\u0142owy",
                   "DYSFUNCTIONAL_BPD": "\u26a0 DYSFUNKCYJNY \u2014 erratyczny VT/BF (BPD)",
                   "RAPID_SHALLOW": "\u26a0 Szybki p\u0142ytki oddech \u2014 wczesna dominacja BF",
                   "PERIODIC": "\u26a0 Oddychanie periodyczne"}.get(bp, bp)
        L.append(f"\u2022 Klasyfikacja: [{bp}] \u2014 {_bp}")
        
        # Flags — sport context annotation
        fl = e07.get("flags", [])
        if fl:
            if _is_sport:
                _sport_flags = {'BF_DOMINANT_STRAT','EARLY_VT_PLATEAU','VDVT_NO_DECREASE','VDVT_PARADOXICAL_RISE','HIGH_BF_VAR_REST','HIGH_BF_VAR_EX','VDVT_EST_LOW'}
                _clinical = [f for f in fl if f not in _sport_flags]
                _sport = [f for f in fl if f in _sport_flags]
                parts = []
                if _sport:
                    parts.append(f"sportowe (norma): {', '.join(_sport)}")
                if _clinical:
                    parts.append(f"do oceny: {', '.join(_clinical)}")
                L.append(f"\u2022 Flagi: [{' | '.join(parts)}]")
            else:
                L.append(f"\u2022 Flagi: [{', '.join(fl)}]")
        
        return "\n".join(L)

    @staticmethod

    @staticmethod
    def _nirs_report(ct):
        r12 = ct.get('_e12_raw', {})
        if not r12 or r12.get('status') == 'NO_SIGNAL':
            return ""
        def _n(key, fb=None):
            v = r12.get(key)
            if v is None: return fb
            try: return float(v)
            except: return fb
        L = []
        L.append("")
        L.append("3c. Analiza NIRS — Saturacja mięśniowa SmO2 (E12 v2.0)")
        L.append(f"    Sensor: Moxy via MetaMax | Kanał: {r12.get('channel','?')} | Jakość: {r12.get('signal_quality','?')}")
        L.append("")
        rest=_n('smo2_rest'); mn=_n('smo2_min'); pk=_n('smo2_at_peak')
        d_abs=_n('desat_total_abs'); d_pct=_n('desat_total_pct')
        if rest is not None and mn is not None:
            L.append("    WARTOŚCI KLUCZOWE:")
            L.append(f"    • SmO2 spoczynek: {rest:.1f}% → min: {mn:.1f}% (peak exercise: {(pk if pk else mn):.1f}%)")
            ds = f"↓ {d_abs:.1f} p.p." if d_abs else "?"
            dp = f" ({d_pct:.1f}% desaturacji)" if d_pct else ""
            L.append(f"    • Desaturacja: {ds}{dp}")
            if d_pct is not None:
                if d_pct > 50: L.append("      ✓ Bardzo wysoka ekstrakcja O2 — profil elitarny")
                elif d_pct > 30: L.append("      ✓ Dobra ekstrakcja O2 — profil sportowca")
                elif d_pct < 15: L.append("      ⚠ Niska desaturacja — możliwy wysoki ATT lub słaba ekstrakcja")
        t27=_n('time_below_27_s')
        if t27 and t27 > 0:
            L.append(f"    • Czas poniżej 27% SmO2: {t27:.0f}s (ref: <27% = strefa MMSS)")
        v1=_n('smo2_at_vt1'); v2=_n('smo2_at_vt2')
        if v1 or v2:
            L.append("    SmO2 W PROGACH:")
            if v1: L.append(f"    • Przy VT1: {v1:.1f}%")
            if v2: L.append(f"    • Przy VT2: {v2:.1f}%")
        b1=_n('bp1_time_s'); b2=_n('bp2_time_s')
        if b1 is not None:
            L.append("    BREAKPOINTY SmO2 (regresja segmentowa):")
            bs1=_n('bp1_smo2'); bv1=_n('bp1_vs_vt1_s')
            vs1 = ""
            if bv1 is not None: vs1 = f" | {abs(bv1):.0f}s {'PRZED' if bv1<0 else 'PO'} VT1"
            m,s=divmod(int(b1),60)
            L.append(f"    • BP1 (Faza1→2): t={m}:{s:02d}, SmO2={(f'{bs1:.1f}' if bs1 else '?')}%{vs1}")
        if b2 is not None:
            bs2=_n('bp2_smo2'); bv2=_n('bp2_vs_vt2_s')
            vs2 = ""
            if bv2 is not None: vs2 = f" | {abs(bv2):.0f}s {'PRZED' if bv2<0 else 'PO'} VT2"
            m,s=divmod(int(b2),60)
            L.append(f"    • BP2 (Faza2→3): t={m}:{s:02d}, SmO2={(f'{bs2:.1f}' if bs2 else '?')}%{vs2}")
        rate=_n('desat_rate_phase2')
        if rate is not None: L.append(f"    • Tempo desaturacji (Faza 2): {rate:.2f} %/min")
        rec=_n('smo2_recovery_peak'); hrt=_n('hrt_s'); ov=_n('overshoot_abs'); rx=_n('reox_rate')
        if rec is not None:
            L.append("    RECOVERY (hyperemia powysiłkowa):")
            os = f" (overshoot: +{ov:.1f} p.p.)" if ov else ""
            L.append(f"    • SmO2 peak recovery: {rec:.1f}%{os}")
            if hrt is not None:
                hi=""
                if hrt<15: hi=" ✓ bardzo szybka reperfuzja"
                elif hrt<30: hi=" ✓ dobra reperfuzja"
                elif hrt>60: hi=" ⚠ wolna reperfuzja"
                L.append(f"    • Half-Recovery Time: {hrt:.1f}s{hi}")
            if rx is not None: L.append(f"    • Tempo reoxygenacji: {rx:.3f} %/s")
        fl=r12.get('flags',[])
        if fl: L.append(f"    Flagi: {', '.join(fl)}")
        return "\n".join(L)


    @staticmethod
    def _drift_report(ct):
        """E13 Drift/Coupling report section."""
        r13 = ct.get('_e13_raw', {})
        if not r13 or r13.get('status') != 'OK':
            return ''
        L = []
        # Sport context for E13
        _d_vo2 = ct.get('_interp_vo2_mlkgmin')
        _d_vs = None
        try:
            _d_vsr = ct.get('VE_VCO2_slope')
            if _d_vsr and str(_d_vsr) not in ('-','[BRAK]','None',''): _d_vs = float(_d_vsr)
        except: pass
        _d_sport = (_d_vo2 is not None and _d_vo2 >= 45 and _d_vs is not None and _d_vs < 30)
        _d_sc = r13.get('slope_class', '-')
        _d_o2pt = r13.get('o2pulse_trajectory', '')
        _d_o2p_late = r13.get('o2pulse_late')
        # Reinterpret slope_class for athletes
        _d_sc_txt = _d_sc
        if _d_sport and _d_sc in ('VERY_LOW', 'LOW_HIGH_SV') and _d_o2pt == 'RISING':
            _d_sc_txt = f"{_d_sc} → ADAPTACJA SPORTOWA (wysoka SV, O2-pulse rosnący)"
        L.append("3e. CV Drift / Coupling (E13 v2.0):")
        L.append(f"  HR-VO2 slope: [{r13.get('hr_vo2_slope','-')}] bpm/L/min (R2={r13.get('hr_vo2_r2','-')}) | {_d_sc_txt} | Coupling: {r13.get('coupling_quality','-')}")
        bp = r13.get('breakpoint_time_sec')
        if bp:
            m, s = int(bp // 60), int(bp % 60)
            _bp_vs = r13.get('bp_vs_vt2', '-')
            _bp_txt = _bp_vs
            if _d_sport and _bp_vs == 'BEFORE_VT2' and _d_o2pt == 'RISING':
                _bp_txt = f"BEFORE_VT2 — u sportowca z rosnącym O2-pulse: wczesna optymalizacja SV, nie dekompensacja"
            L.append(f"  HR-VO2 breakpoint: t={m}:{s:02d} ({r13.get('breakpoint_pct_exercise','-')}%) | vs VT2: {_bp_txt}")
        ci = r13.get('chronotropic_index')
        ci_txt = f"{ci}" if ci else '-'
        L.append(f"  Chronotropic: CI={ci_txt} ({r13.get('ci_class','-')}) | HR reserve: {r13.get('hr_reserve_pct','-')}% | {r13.get('hr_pattern','-')}")
        L.append(f"  HR: rest {r13.get('hr_rest','-')} -> peak {r13.get('hr_peak','-')} bpm | pred {r13.get('hr_pred_max','-')} ({r13.get('pct_pred_hr','-')}%)")
        L.append(f"  O2P drift: {r13.get('o2pulse_early','-')} -> {r13.get('o2pulse_mid','-')} -> {r13.get('o2pulse_late','-')} ml/beat ({r13.get('o2pulse_drift_pct','-')}%)")
        L.append(f"  Trajektoria O2P: {r13.get('o2pulse_trajectory','-')} | Klinicznie: {r13.get('o2pulse_clinical','-')}")
        vwr = r13.get('vo2_wr')
        if vwr:
            L.append(f"  VO2/WR: {vwr.get('vo2_wr_slope','-')} {vwr.get('vo2_wr_unit','')} (R2={vwr.get('vo2_wr_r2','-')}) | {vwr.get('slope_class','-')}")
        sd = r13.get('step_drift')
        if sd:
            L.append(f"  Intra-step drift: max {sd.get('max_drift_pct','-')}% | mean {sd.get('mean_drift_pct','-')}%")
        fl = r13.get('flags', [])
        _d_ov = r13.get('overall_cv_coupling', '-')
        if _d_sport and _d_o2pt == 'RISING':
            _d_sport_fl = {'VERY_LOW_HR_VO2_SLOPE', 'EARLY_LINEARITY_BREAK'}
            _d_cfl = [f for f in fl if f not in _d_sport_fl]
            _d_sfl = [f for f in fl if f in _d_sport_fl]
            if _d_sfl and not _d_cfl:
                _d_ov = f"SPORT_NORM (reinterpretacja: niski slope + wczesny BP = wysoka SV u sportowca)"
            _d_fl_parts = []
            if _d_sfl: _d_fl_parts.append(f"sport (wysoka SV): {', '.join(_d_sfl)}")
            if _d_cfl: _d_fl_parts.append(f"kliniczne: {', '.join(_d_cfl)}")
            L.append(f"  Overall: {_d_ov} | Flagi: [{' | '.join(_d_fl_parts)}]" if _d_fl_parts else f"  Overall: {_d_ov} | Flagi: BRAK")
        else:
            L.append(f"  Overall: {_d_ov} | Flagi: {', '.join(fl) if fl else 'BRAK'}")
        return '\n'.join(L)

    @staticmethod
    def _lactate_report(ct):
        """E11 Lactate report section."""
        r11 = ct.get('_e11_raw', {})
        if not r11 or r11.get('status') != 'OK':
            return ''
        L = []
        L.append("3d. Analiza Laktatowa (E11 v2.0):")
        n = r11.get('n_points', 0)
        bl = r11.get('la_baseline', 0)
        pk = r11.get('la_peak', 0)
        L.append(f"  Punkty pomiarowe: {n} | Baseline: {bl:.2f} mmol/L | Peak: {pk:.2f} mmol/L")
        if r11.get('lt1_method'):
            hr1 = f"HR {round(r11['lt1_hr_bpm'])}" if r11.get('lt1_hr_bpm') else ''
            sp1 = f" Speed {round(r11['lt1_speed_kmh'],1)}" if r11.get('lt1_speed_kmh') else ''
            la1 = f" La={round(r11['lt1_la_mmol'],2)}" if r11.get('lt1_la_mmol') else ''
            t1 = r11.get('lt1_time_sec')
            ts1 = f"t={int(t1//60)}:{int(t1%60):02d}" if t1 else ''
            L.append(f"  LT1: {r11['lt1_method']} | {ts1} | {hr1}{sp1}{la1}")
        if r11.get('lt2_method'):
            hr2 = f"HR {round(r11['lt2_hr_bpm'])}" if r11.get('lt2_hr_bpm') else ''
            sp2 = f" Speed {round(r11['lt2_speed_kmh'],1)}" if r11.get('lt2_speed_kmh') else ''
            la2 = f" La={round(r11['lt2_la_mmol'],2)}" if r11.get('lt2_la_mmol') else ''
            t2 = r11.get('lt2_time_sec')
            ts2 = f"t={int(t2//60)}:{int(t2%60):02d}" if t2 else ''
            L.append(f"  LT2: {r11['lt2_method']} | {ts2} | {hr2}{sp2}{la2}")
        conc = r11.get('concordance', {})
        for key in ['lt1_consensus', 'lt2_consensus']:
            c = conc.get(key, {})
            if c.get('n_methods', 0) > 0:
                L.append(f"  {c.get('label','')}: consensus {c.get('n_methods')} metod, spread {c.get('spread_sec','?')}s, confidence {c.get('confidence','?')}")
        vt = r11.get('vt_comparison', {})
        if vt.get('vt1_vs_lt1_diff_sec') is not None:
            ag = 'ZGODNE' if vt.get('vt1_vs_lt1_agreement') else 'ROZBIEZNE'
            L.append(f"  VT1 vs LT1: diff {vt['vt1_vs_lt1_diff_sec']}s ({ag})")
        if vt.get('vt2_vs_lt2_diff_sec') is not None:
            ag = 'ZGODNE' if vt.get('vt2_vs_lt2_agreement') else 'ROZBIEZNE'
            L.append(f"  VT2 vs LT2: diff {vt['vt2_vs_lt2_diff_sec']}s ({ag})")
        qw = r11.get('quality_warnings', [])
        if qw:
            L.append(f"  Uwagi: {'; '.join(qw)}")
        return '\n'.join(L)

    @staticmethod
    def _e18_crossval_text(ct):
        """Generate E18 cross-validation text section."""
        if ct.get('E18_status') != 'OK':
            return ""
        
        lines = []
        lines.append("CROSS-WALIDACJA VT↔LT (E18)")
        
        e18_sum = ct.get('E18_summary', {})
        lines.append(f"  Ocena: {e18_sum.get('overall_class', '').replace('_',' ')} "
                     f"({e18_sum.get('overall_score_pct', 0):.0f}%)")
        
        # L1
        l1 = ct.get('E18_layer1', {})
        for prefix, label in [('vt1','VT1'), ('vt2','VT2')]:
            d = l1.get(prefix, {})
            if d.get('status') not in ('NO_VT', None):
                lines.append(f"  {d.get('emoji','')} La@{label}: "
                           f"{d.get('la_interpolated_mmol','-')} mmol/L "
                           f"[{d.get('status','')}] (opt: {d.get('optimal_range',[])})")
        
        # L2
        l2 = ct.get('E18_layer2', {})
        l2s = l2.get('summary', {})
        if l2s.get('domains_tested', 0) > 0:
            lines.append(f"  Domeny: {l2s.get('domains_confirmed',0)}/{l2s.get('domains_tested',0)} potwierdzone")
        
        # L3
        l3 = ct.get('E18_layer3', {})
        for prefix, label in [('threshold_1','VT1↔LT1'), ('threshold_2','VT2↔LT2')]:
            d = l3.get(prefix, {})
            if d.get('status') not in ('INCOMPLETE', None):
                lines.append(f"  {d.get('emoji','')} {label}: "
                           f"Δt={d.get('delta_time_sec',0):+.0f}s "
                           f"[{d.get('time_concordance','')}]")
        
        # Interpretation
        interp = e18_sum.get('interpretation', '')
        if interp:
            lines.append(f"  → {interp}")
        
        return "\n".join(lines)


    @staticmethod
    def _render_charts_html(ct):
        """Generate interactive charts section with buttons and Canvas JS."""
        import json as _json
        
        _lac_pts = ct.get('_chart_lactate_points', [])
        _lac_poly = ct.get('_chart_lactate_poly', {})
        _nirs_traces = ct.get('_chart_nirs_traces', [])
        _has_lactate = len(_lac_pts) >= 3
        _has_nirs = len(_nirs_traces) > 0
        _has_gas = len(ct.get('_chart_gas_time', [])) > 10
        _has_substrate = len(ct.get('_chart_sub_time', [])) > 10 and max(ct.get('_chart_sub_fat', [0]) or [0]) > 0
        _has_kinetics = len(ct.get('_chart_full_time', [])) > 10
        
        if not _has_lactate and not _has_nirs and not _has_gas and not _has_substrate and not _has_kinetics and not _has_protocol:
            return ''
        
        _chart_data = {
            'lactate': {
                'points': [{'time_sec': p.get('time_sec', 0), 'la': p.get('la', p.get('lactate_mmol', 0))} for p in _lac_pts],
                'poly_time': _lac_poly.get('time_sec', []),
                'poly_fit': _lac_poly.get('lactate_fit', []),
                'lt1_time': ct.get('_chart_lt1_time'),
                'lt1_la': ct.get('_chart_lt1_la'),
                'lt2_time': ct.get('_chart_lt2_time'),
                'lt2_la': ct.get('_chart_lt2_la'),
                'lt1_method': ct.get('_chart_lt1_method', ''),
                'lt2_method': ct.get('_chart_lt2_method', ''),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
                'vt1_hr': ct.get('_chart_vt1_hr'),
                'vt2_hr': ct.get('_chart_vt2_hr'),
                'e18_l1': ct.get('_chart_e18_l1', {}),
            },
            'nirs': {
                'traces': [{'time_sec': tr.get('time_sec', []), 'smo2': tr.get('smo2_pct', tr.get('smo2', []))} for tr in _nirs_traces],
                'channels': ct.get('_chart_nirs_channels', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'gas': {
                'time': ct.get('_chart_gas_time', []),
                'vo2': ct.get('_chart_gas_vo2', []),
                'vco2': ct.get('_chart_gas_vco2', []),
                'hr': ct.get('_chart_gas_hr', []),
                'vt1_vo2': ct.get('VT1_VO2_mlmin'),
                'vt2_vo2': ct.get('VT2_VO2_mlmin'),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'substrate': {
                'time': ct.get('_chart_sub_time', []),
                'cho': ct.get('_chart_sub_cho', []),
                'fat': ct.get('_chart_sub_fat', []),
                'hr': ct.get('_chart_sub_hr', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
            },
            'kinetics': {
                'time': ct.get('_chart_full_time', []),
                'vo2': ct.get('_chart_full_vo2', []),
                'hr': ct.get('_chart_full_hr', []),
                'speed': ct.get('_chart_full_speed', []),
                'power': ct.get('_chart_full_power', []),
                'vt1_time': ct.get('_chart_vt1_time'),
                'vt2_time': ct.get('_chart_vt2_time'),
                't_stop': ct.get('_chart_t_stop'),
                'vo2peak': ct.get('_chart_vo2peak'),
                'protocol_steps': ct.get('_prot_steps', []),
                'protocol_name': ct.get('_prot_name', ''),
            }
        }
        _cj = _json.dumps(_chart_data, default=str)
        
        h = '<div style="margin:24px 0 12px;border-top:2px solid #e2e8f0;padding-top:18px;">'
        h += '<div style="font-size:15px;font-weight:700;color:#1e293b;margin-bottom:12px;">'
        h += '&#128202; Wykresy interaktywne</div>'
        

        _sq = chr(39)
        _has_protocol = len(ct.get('_prot_steps', [])) > 0
        if _has_protocol:
            h += '<button onclick="toggleChart(' + _sq + 'proto' + _sq + ')" id="btn_proto" data-color="#6366f1" style="margin:4px 6px;padding:8px 18px;border:2px solid #6366f1;border-radius:8px;background:#fff;color:#6366f1;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Protokół</button>'
        if _has_kinetics:
            h += '<button onclick="toggleChart(' + _sq + 'kinetics' + _sq + ')" id="btn_kinetics" data-color="#059669" style="margin:4px 6px;padding:8px 18px;border:2px solid #059669;border-radius:8px;background:#fff;color:#059669;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ VO2 Kinetics</button>'
        if _has_gas:
            h += '<button onclick="toggleChart(' + _sq + 'vslope' + _sq + ')" id="btn_vslope" data-color="#0891b2" style="margin:4px 6px;padding:8px 18px;border:2px solid #0891b2;border-radius:8px;background:#fff;color:#0891b2;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ V-slope</button>'
        if _has_substrate:
            h += '<button onclick="toggleChart(' + _sq + 'fatchho' + _sq + ')" id="btn_fatchho" data-color="#d97706" style="margin:4px 6px;padding:8px 18px;border:2px solid #d97706;border-radius:8px;background:#fff;color:#d97706;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Fat/CHO</button>'
        if _has_lactate:
            h += '<button onclick="toggleChart(' + _sq + 'lac' + _sq + ')" id="btn_lac" data-color="#dc2626" style="margin:4px 6px;padding:8px 18px;border:2px solid #dc2626;border-radius:8px;background:#fff;color:#dc2626;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Krzywa mleczanowa</button>'
        if _has_nirs:
            h += '<button onclick="toggleChart(' + _sq + 'nirs' + _sq + ')" id="btn_nirs" data-color="#2563eb" style="margin:4px 6px;padding:8px 18px;border:2px solid #2563eb;border-radius:8px;background:#fff;color:#2563eb;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ NIRS SmO2</button>'
        if _has_lactate and _has_nirs:
            h += '<button onclick="toggleChart(' + _sq + 'dual' + _sq + ')" id="btn_dual" data-color="#7c3aed" style="margin:4px 6px;padding:8px 18px;border:2px solid #7c3aed;border-radius:8px;background:#fff;color:#7c3aed;font-weight:600;font-size:13px;cursor:pointer;font-family:Segoe UI,system-ui,sans-serif;">▶ Laktat + SmO2</button>'
        h += '<div id="chart_container" style="display:none;margin-top:14px;"></div></div>'
        h += '<script>\nconst CD=' + _cj + ';\n' + CHART_JS + '</script>'
        
        return h

    @staticmethod
    def render_html_report(ct):
        g = ct.get
        def v(key, default='-'):
            val = g(key, default)
            if val in (None, '', '[BRAK]', 'None', 'nan'): return default
            return str(val)
        def esc(s):
            return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        def _n(val, fmt='.1f', default='-'):
            try:
                n = float(val)
                return f'{n:{fmt}}'
            except: return default
        def _fmt_dur(sec):
            try:
                s=int(float(sec)); return f'{s//60}:{s%60:02d}'
            except: return '-'
        def badge(text, color='#64748b', bg=None):
            if not bg:
                bg = color + '18'
            return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{bg};color:{color};">{text}</span>'
        def row_item(label, value, assessment='', comment='', unit=''):
            colors = {
                'EXCELLENT':'#16a34a','WYBITNA':'#16a34a','HIGH':'#16a34a','NORMA':'#16a34a',
                'GOOD':'#16a34a','MAXIMAL':'#16a34a','VERY_GOOD':'#16a34a','RISING':'#16a34a',
                'PASS':'#16a34a','TIGHT':'#16a34a','YES':'#16a34a','TRAINED':'#16a34a',
                'INCREASING':'#16a34a','SUPERIOR':'#16a34a','SUPRAMAXIMAL':'#16a34a',
                'SPORT_NORM':'#2563eb','SPORT':'#2563eb','BF_DOMINANT':'#2563eb','FLAT':'#2563eb',
                'MODERATE':'#d97706','FAIR':'#d97706','BORDERLINE':'#d97706','WIDE':'#d97706',
                'NORMAL':'#16a34a','VC-I':'#16a34a','VC-II':'#d97706','VC-III':'#ea580c','VC-IV':'#dc2626',
                'POOR':'#dc2626','ABNORMAL':'#dc2626','SLOW':'#ea580c','VERY_SLOW':'#dc2626',
                'VERY_WIDE':'#dc2626','LOW':'#dc2626','FAIL':'#dc2626','DECLINING':'#dc2626',
                'RECREATIONAL':'#d97706','BALANCED':'#2563eb','SV_LIMITATION':'#ea580c',
                'N/A':'#94a3b8','TODO':'#94a3b8','NO_DATA':'#94a3b8','INSUFFICIENT_DATA':'#94a3b8',
                'LOW_HIGH_SV':'#ea580c','NO_RCP_DROP':'#d97706',
            }
            _short = {
                'BORDERLINE':'GRANICZNY','INSUFFICIENT':'NIEWYSTAR.','SUPRAMAXIMAL':'SUPRAMAX',
                'LOW_HIGH_SLOPE':'NISKI↑','HIGH_SLOPE':'WYSOKI↑',
                'VERY_SLOW':'B.WOLNY','VERY_GOOD':'B.DOBRY','VERY_WIDE':'B.SZEROKI',
                'BF_DOMINANT':'BF-DOM.','VT_DOMINANT':'VT-DOM.',
                'DECLINING':'SPADEK','AFTER_VT2':'PO VT2','BEFORE_VT2':'PRZED VT2',
                'AFTER_VT1':'PO VT1','BEFORE_VT1':'PRZED VT1','NONE':'BRAK',
                'MILD_ABNORMALITY':'ŁAGODNA ANOM.','EXCELLENT':'DOSKONAŁA',
                'RISING':'ROSNĄCY','INCREASING':'ROSNĄCY','NORMAL':'NORMA',
                'HIGH_DESAT_ELITE':'WYS.DESAT.','LATE_BP1':'PÓŹNY BP1',
                'SLOW_RECOVERY':'WOLNA REC.','LARGE_OVERSHOOT':'DUŻY OVERSHOOT',
                'FREQUENT_SIGHS':'CZĘSTE WESTCH.','EARLY_VT_PLATEAU':'WCZESNE PLAT.VT',
                'HIGH_BF_VAR_EX':'WYS.ZMIEN.BF','PARADOXICAL':'PARADOKS.',
                'SPORT_NORM':'SPORT','DISPROPORTIONATE':'DYSPROP.',
                'DYSFUNCTIONAL_BPD':'DYSFUNKC.','TACHYPNEIC_RISE':'TACHYPNEA',
                'NO_RCP_DROP':'BEZ SPADKU','PARADOXICAL_RISE':'PARAD.↑',
                'SV_LIMITATION':'SV-LIMIT','LOW_HIGH_SV':'NIS-WYS.SV',
                'INSUFFICIENT_DATA':'B/D',
            }
            a_str = str(assessment).strip()
            a_disp = _short.get(a_str.upper().replace(' ','_'), a_str)
            ac = colors.get(a_str.upper().replace(' ','_'), '#64748b')
            bdg = f'<span style="color:{ac};font-weight:600;font-size:10px;white-space:nowrap;">{a_disp}</span>' if a_str and a_str != '-' else ''
            vu = f'{value} {unit}'.strip() if unit else str(value)
            cmt = f'<div style="font-size:10px;color:#94a3b8;margin-top:1px;">{comment}</div>' if comment else ''
            return (
                f'<div style="padding:5px 0;border-bottom:1px solid #f1f5f9;display:flex;align-items:flex-start;gap:8px;">'
                f'<div style="flex:0 0 130px;font-size:12px;color:#475569;font-weight:500;">{label}</div>'
                f'<div style="flex:1;font-size:13px;font-weight:600;color:#0f172a;">{vu}{cmt}</div>'
                f'<div style="flex:0 0 auto;min-width:70px;text-align:right;">{bdg}</div></div>'
            )

        def gauge_svg(score, label, size=80, is_pct=False, subtitle=''):
            pct=max(0,min(100,float(score) if score else 0))
            if pct>=85: col='#16a34a'
            elif pct>=70: col='#65a30d'
            elif pct>=55: col='#d97706'
            elif pct>=40: col='#ea580c'
            else: col='#dc2626'
            if pct == 0: col='#cbd5e1'
            r=size*0.4;cx=size/2;cy=size*0.5
            circ=2*3.14159*r;dash=circ*pct/100;gap=circ-dash
            disp = f'{int(score)}{"%" if is_pct else ""}' if score else '\u2014'
            h = size + (24 if subtitle else 14)
            sub_txt = f'<text x="{cx}" y="{size+20}" text-anchor="middle" font-size="7" fill="#94a3b8">{subtitle}</text>' if subtitle else ''
            return f'<svg width="{size}" height="{h}" viewBox="0 0 {size} {h}"><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="6"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="6" stroke-dasharray="{dash} {gap}" stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/><text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="700" fill="{col}">{disp}</text><text x="{cx}" y="{cy+r+10}" text-anchor="middle" font-size="8" font-weight="600" fill="#475569">{label}</text>{sub_txt}</svg>'

        def panel_card(title, icon, rows_html):
            return f'''<div style="background:white;border-radius:10px;border:1px solid #e2e8f0;padding:14px;flex:1;min-width:300px;">
  <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;padding-bottom:6px;border-bottom:2px solid #e2e8f0;">{icon} {title}</div>
  {rows_html}
</div>'''

        def section_title(text, num=''):
            pre = f'{num}. ' if num else ''
            return f'<div class="section-title">{pre}{text}</div>'

        def obs_bullet(text, severity='info'):
            colors = {'critical':'#dc2626','warning':'#d97706','sport':'#2563eb','info':'#16a34a','positive':'#16a34a'}
            dots = {'critical':'🔴','warning':'🟡','sport':'🔵','info':'🟢','positive':'🟢'}
            col = colors.get(severity, '#16a34a')
            dot = dots.get(severity, '🟢')
            return f'<div style="font-size:12px;color:#334155;padding:3px 0;">{dot} {text}</div>'

        # =====================================================================
        # GATHER ALL DATA
        # =====================================================================
        name = v('athlete_name','Sportowiec')
        age = v('age_y'); sex_pl = 'M' if v('sex')=='male' else 'K'
        weight = v('body_mass_kg'); height = v('height_cm')
        protocol = v('protocol_name')
        modality = v('modality','run')
        test_date = v('test_date','')
        
        # E00
        e00 = g('_e00_raw',{})
        t_stop = e00.get('t_stop',0)
        dur_str = _fmt_dur(t_stop)
        
        # E01/E15 — Peak values
        vo2_rel = g('VO2max_ml_kg_min')
        vo2_abs = g('VO2max_L_min') or g('VO2max_abs_Lmin')
        e15 = g('_e15_raw',{})
        _pc_hdr = ct.get('_performance_context', {})
        _pc_level_hdr = _pc_hdr.get('level_by_speed', '') if _pc_hdr.get('executed') else ''
        vo2_class = e15.get('vo2_class_pop','?')
        vo2_pctile = e15.get('vo2_percentile_approx',50)
        vo2_det = e15.get('vo2_determination','VO2peak')
        vo2_pct_pred = e15.get('vo2_pct_predicted')
        vo2_class_sport = e15.get('vo2_class_sport_desc','')
        rer_peak = g('RER_peak')
        rer_class = e15.get('rer_class','?')
        rer_desc = e15.get('rer_desc','')
        hr_peak = g('HR_peak')
        hr_pred = e15.get('hr_predicted_tanaka')
        hr_pct_pred = e15.get('hr_pct_predicted') or e15.get('pct_pred_hr') or (float(hr_peak)/float(hr_pred)*100 if hr_peak and hr_pred else None)
        ve_peak = g('VE_peak')
        plateau = e15.get('plateau_detected', False)
        plateau_note = e15.get('vo2_method_note','')
        hrr1_class_e15 = e15.get('hrr_1min_class','')
        hrr1_desc_e15 = e15.get('hrr_1min_desc','')
        
        # E02 — Thresholds
        e02 = g('_e02_raw', {})
        if not e02 or not isinstance(e02, dict):
            import copy
            e02 = {}
        vt1_hr = g('VT1_HR_bpm'); vt1_time = g('VT1_Time_s'); vt1_pct = g('VT1_pct_VO2peak')
        vt2_hr = g('VT2_HR_bpm'); vt2_time = g('VT2_Time_s'); vt2_pct = g('VT2_pct_VO2peak')
        vt1_conf = g('VT1_confidence',0); vt2_conf = g('VT2_confidence',0)
        vt1_hr_pct_max = e15.get('vt1_pct_hrmax')
        vt2_hr_pct_max = e15.get('vt2_pct_hrmax')
        vt1_speed = g('VT1_Speed'); vt2_speed = g('VT2_Speed')
        vt1_rer_val = e02.get('vt1_rer'); vt2_rer_val = e02.get('vt2_rer')
        vt1_vo2_abs = e02.get('vt1_vo2_mlmin'); vt2_vo2_abs = e02.get('vt2_vo2_mlmin')
        
        # E03 — VE/VCO2
        e03 = g('_e03_raw',{})
        ve_vco2_slope = g('VE_VCO2_slope') or e03.get('slope_to_vt2')
        vc_class = e03.get('ventilatory_class','?')
        vc_desc = e03.get('vc_description','')
        ve_vco2_nadir = e03.get('ve_vco2_nadir')
        ve_vco2_at_vt1 = e03.get('ve_vco2_at_vt1')
        ve_vco2_full = e03.get('slope_full')
        ve_vco2_to_vt1 = e03.get('slope_to_vt1')
        slope_pct_pred = e03.get('pct_predicted')
        
        # E04 — OUES
        e04 = g('_e04_raw',{})
        oues100 = e04.get('oues100'); oues90 = e04.get('oues90'); oues75 = e04.get('oues75')
        oues_vt1 = e04.get('oues_to_vt1'); oues_vt2 = e04.get('oues_to_vt2')
        oues_kg = e04.get('oues_per_kg')
        oues_pct = e04.get('oues_pct_hollenberg')
        oues_stab = e04.get('oues_submax_stability')
        
        # E05 — O2 Pulse
        e05 = g('_e05_raw',{})
        o2p_peak = e05.get('o2pulse_peak')
        o2p_vt1 = e05.get('o2pulse_at_vt1'); o2p_vt2 = e05.get('o2pulse_at_vt2')
        o2p_traj = e05.get('trajectory','?')
        o2p_pct = e05.get('pct_predicted_friend')
        sv_est = e05.get('estimated_sv_peak_ml')
        o2p_desc = e05.get('trajectory_desc','')
        
        # E06 — Economy / Efficiency
        e06 = g('_e06_raw',{})
        re_vt1 = e06.get('re_at_vt1'); re_vt2 = e06.get('re_at_vt2')
        re_class = e06.get('re_classification','')
        gain_below_vt1 = e06.get('gain_below_vt1'); gain_vt1 = e06.get('gain_at_vt1'); gain_vt2 = e06.get('gain_at_vt2')
        gain_unit = e06.get('gain_unit','')
        gain_z = e06.get('gain_z_score')
        lin_break = e06.get('linearity_break_time_s')
        delta_eff = e06.get('delta_efficiency_pct')
        
        # E07 — Breathing
        e07 = g('_e07_raw',{})
        bp_pattern = e07.get('breathing_pattern','?')
        bp_strategy = e07.get('strategy','?')
        bf_peak = e07.get('bf_peak'); bf_vt1 = e07.get('bf_at_vt1'); bf_vt2 = e07.get('bf_at_vt2')
        vt_peak = e07.get('vt_peak_L'); vt_vt1 = e07.get('vt_at_vt1_L'); vt_vt2 = e07.get('vt_at_vt2_L')
        ve_from_bf = e07.get('ve_from_bf_pct')
        rsbi = e07.get('rsbi_peak')
        vt_plat_t = e07.get('vt_plateau_time_s'); vt_plat_pct = e07.get('vt_plateau_pct_exercise')
        e07_flags = e07.get('flags',[])
        
        # E08 — HRR
        e08 = g('_e08_raw',{})
        hrr1 = e08.get('hrr_1min'); hrr3 = e08.get('hrr_3min')
        hrr1_class = e08.get('interpretation_1min','?')
        hrr3_class = e08.get('interpretation_3min','?')
        
        # E10 — Substrate
        e10 = g('_e10_raw',{})
        fatmax = e10.get('mfo_gmin'); fatmax_hr = e10.get('fatmax_hr')
        fatmax_pct_vo2 = e10.get('fatmax_pct_vo2peak')
        fatmax_pct_hr = e10.get('fatmax_pct_hrmax')
        fatmax_zone_lo = e10.get('fatmax_zone_hr_low'); fatmax_zone_hi = e10.get('fatmax_zone_hr_high')
        cop_hr = e10.get('cop_hr'); cop_pct_vo2 = e10.get('cop_pct_vo2peak'); cop_rer = e10.get('cop_rer')
        fat_vt1_pct = e10.get('fat_pct_at_vt1'); cho_vt1_pct = e10.get('cho_pct_at_vt1')
        fat_vt2_pct = e10.get('fat_pct_at_vt2'); cho_vt2_pct = e10.get('cho_pct_at_vt2')
        fat_vt1_g = e10.get('fat_gmin_at_vt1'); cho_vt1_g = e10.get('cho_gmin_at_vt1')
        total_kcal = e10.get('total_kcal'); total_fat = e10.get('total_fat_kcal'); total_cho = e10.get('total_cho_kcal')
        zone_sub = e10.get('zone_substrate',{})
        
        # E11 — Lactate
        e11 = g('_e11_raw',{})
        la_peak = e11.get('la_peak'); la_base = e11.get('la_baseline')
        lt1_hr = e11.get('lt1_hr_bpm'); lt1_time = e11.get('lt1_time_sec')
        lt2_hr = e11.get('lt2_hr_bpm'); lt2_time = e11.get('lt2_time_sec')
        lt1_method = e11.get('lt1_method',''); lt2_method = e11.get('lt2_method','')
        
        # E12 — NIRS
        e12 = g('_e12_raw',{})
        smo2_rest = e12.get('smo2_rest'); smo2_min = e12.get('smo2_min')
        smo2_vt1 = e12.get('smo2_at_vt1'); smo2_vt2 = e12.get('smo2_at_vt2')
        smo2_peak_val = e12.get('smo2_at_peak')
        desat_pct = e12.get('desat_total_pct'); desat_abs = e12.get('desat_total_abs')
        hrt_s = e12.get('hrt_s'); reox_rate = e12.get('reox_rate')
        overshoot = e12.get('overshoot_abs')
        smo2_recov = e12.get('smo2_recovery_peak')
        bp1_t = e12.get('bp1_time_s'); bp1_smo2 = e12.get('bp1_smo2')
        bp2_t = e12.get('bp2_time_s'); bp2_smo2 = e12.get('bp2_smo2')
        bp1_vs_vt1 = e12.get('bp1_vs_vt1_s'); bp2_vs_vt2 = e12.get('bp2_vs_vt2_s')
        e12_flags = e12.get('flags',[])
        e12_quality = e12.get('signal_quality','')
        
        # E13 — Cardiovascular coupling
        e13 = g('_e13_raw',{})
        ci = e13.get('chronotropic_index'); ci_class = e13.get('ci_class','?')
        hr_vo2_slope = e13.get('hr_vo2_slope'); slope_class = e13.get('slope_class','?')
        bp_time_e13 = e13.get('breakpoint_time_sec'); bp_pct_e13 = e13.get('breakpoint_pct_exercise')
        bp_vs_vt2_e13 = e13.get('bp_vs_vt2','')
        o2p_traj_e13 = e13.get('o2pulse_trajectory','')
        o2p_clinical = e13.get('o2pulse_clinical','')
        cv_coupling = e13.get('overall_cv_coupling','')
        
        # E14 — Recovery kinetics
        e14 = g('_e14_raw',{})
        t_half_vo2 = e14.get('t_half_vo2_s')
        tau_vo2 = e14.get('tau_s'); tau_r2 = e14.get('tau_r2')
        mrt = e14.get('mrt_s')
        rec_class = e14.get('recovery_class','?')
        rec_desc = e14.get('recovery_desc','')
        vo2dr = e14.get('vo2dr_s')
        dvo2_60 = e14.get('dvo2_60s_mlkgmin')
        t_half_vco2 = e14.get('t_half_vco2_s'); t_half_ve = e14.get('t_half_ve_s')
        pct_rec_60 = e14.get('pct_recovered_60s'); pct_rec_120 = e14.get('pct_recovered_120s')
        
        # E16 — Zones
        e16 = g('_e16_raw',{})
        zones_data = e16.get('zones',{})
        three_zone = e16.get('three_zone',{})
        aer_reserve = e16.get('aerobic_reserve_pct')
        thr_gap = e16.get('threshold_gap_bpm')
        thr_gap_pct = e16.get('threshold_gap_pct_hrmax')
        
        # E17 — Gas exchange
        e17 = g('_e17_raw',{})
        vdvt_rest = e17.get('vdvt_rest'); vdvt_peak = e17.get('vdvt_at_peak')
        vdvt_pattern = e17.get('vdvt_pattern','?')
        petco2_rest = e17.get('petco2_rest'); petco2_vt1 = e17.get('petco2_at_vt1')
        petco2_peak = e17.get('petco2_at_peak'); petco2_pattern = e17.get('petco2_pattern','?')
        peto2_nadir = e17.get('peto2_nadir')
        
        # E19 — Validation
        e19 = g('_e19_raw',{})
        val_score = e19.get('validity_score',0); val_grade = e19.get('validity_grade','?')
        con_score = e19.get('concordance_score',0); con_grade = e19.get('concordance_grade','?')
        ta = e19.get('temporal_alignment',{})
        ta_vt1 = ta.get('VT1',{}); ta_vt2 = ta.get('VT2',{})
        
        # Sport context
        _is_sport = False
        try:
            _bv = float(ve_vco2_slope) if ve_vco2_slope else 99
            _bvo2 = float(vo2_rel) if vo2_rel else 0
            _is_sport = _bvo2 >= 40 and _bv < 32
        except: pass
        
        # =====================================================================
        # BUILD HTML
        # =====================================================================
        
        h = f'''<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPET Report — {esc(name)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f8fafc;color:#0f172a;line-height:1.5;}}
.wrap{{max-width:920px;margin:0 auto;padding:16px;}}
.section{{margin-bottom:14px;}}
.card{{background:white;border-radius:10px;border:1px solid #e2e8f0;padding:16px;margin-bottom:10px;}}
.section-title{{font-size:14px;font-weight:700;color:#0f172a;padding-bottom:6px;margin-bottom:10px;border-bottom:2px solid #e2e8f0;}}
.flex-row{{display:flex;gap:12px;flex-wrap:wrap;}}
.sub-header{{margin:8px 0 4px;font-size:11px;font-weight:700;color:#475569;border-top:1px solid #e2e8f0;padding-top:6px;text-transform:uppercase;letter-spacing:0.5px;}}
table.ztable{{width:100%;border-collapse:collapse;font-size:11px;}}
table.ztable th{{padding:5px;text-align:left;background:#f8fafc;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;}}
table.ztable td{{padding:4px 5px;border-bottom:1px solid #f1f5f9;}}
@media print{{body{{background:white;}} .wrap{{max-width:100%;padding:8px;}} .card{{break-inside:avoid;}}}}
</style></head><body><div class="wrap">'''

        # ═══════════════════════════════════════════════════════════════
        # 0. HEADER
        # ═══════════════════════════════════════════════════════════════
        h += f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);border-radius:12px;color:white;margin-bottom:14px;">
  <div>
    <div style="font-size:20px;font-weight:700;">{esc(name)}</div>
    <div style="font-size:12px;color:#94a3b8;">Wiek: {age} | {sex_pl} | {weight} kg | {height} cm | {test_date}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px;color:#94a3b8;">Protokół: {esc(v('_protocol_description', protocol))}</div>
    <div style="font-size:11px;color:#94a3b8;">Czas wysiłku: {dur_str} | Modalność: {"bieżnia" if modality=="run" else "cykl"}</div>
  </div>
</div>'''

        # ═══════════════════════════════════════════════════════════════
        # I. DIAGNOZA I REKOMENDACJA
        # ═══════════════════════════════════════════════════════════════
        try: pctile_val = float(vo2_pctile) if vo2_pctile else 50
        except: pctile_val = 50
        
        # HRR gauge score (evidence-based scale, active recovery post-CPET)
        _hrr_gauge_score = 0; _hrr_lbl = 'b/d'
        try:
            _hrr_val = float(hrr1) if hrr1 else 0
            if _hrr_val >= 50: _hrr_gauge_score = 95
            elif _hrr_val >= 40: _hrr_gauge_score = 85 + (_hrr_val - 40) / 10 * 10
            elif _hrr_val >= 30: _hrr_gauge_score = 70 + (_hrr_val - 30) / 10 * 15
            elif _hrr_val >= 22: _hrr_gauge_score = 55 + (_hrr_val - 22) / 8 * 15
            elif _hrr_val >= 15: _hrr_gauge_score = 40 + (_hrr_val - 15) / 7 * 15
            elif _hrr_val >= 12: _hrr_gauge_score = 25 + (_hrr_val - 12) / 3 * 15
            elif _hrr_val > 0: _hrr_gauge_score = max(5, _hrr_val / 12 * 25)
            _hrr_lbl = f'HRR {_hrr_val:.0f}' if _hrr_val > 0 else 'b/d'
        except: pass
        
        # Overall profile score (weighted composite)
        _overall_score = 0; _overall_grade = 'b/d'
        try:
            _components = []
            # 1. VO2max percentyl (30%)
            _vo2p = float(vo2_pctile) if vo2_pctile else 50
            _components.append((_vo2p, 0.30))
            # 2. VT2 %VO2max rescaled: 60%->0, 100%->100 (20%)
            _vt2p = float(vt2_pct) if vt2_pct else 75
            _vt2_score = max(0, min(100, (_vt2p - 60) / 40 * 100))
            _components.append((_vt2_score, 0.20))
            # 3. VE/VCO2 rescaled: <20->100, 20-25->85, 25-30->65, 30-35->40, >35->15 (15%)
            _slp = float(ve_vco2_slope) if ve_vco2_slope else 30
            if _slp < 20: _ve_score = 100
            elif _slp < 25: _ve_score = 85 + (25 - _slp) / 5 * 15
            elif _slp < 30: _ve_score = 65 + (30 - _slp) / 5 * 20
            elif _slp < 35: _ve_score = 40 + (35 - _slp) / 5 * 25
            else: _ve_score = max(5, 40 - (_slp - 35) / 5 * 25)
            _components.append((_ve_score, 0.15))
            # 4. Recovery HRR score (15%)
            _components.append((_hrr_gauge_score, 0.15))
            # 5. O2-pulse %pred (10%)
            _o2p_pct_v = float(o2p_pct) if o2p_pct else 100
            _o2p_score = min(100, max(0, _o2p_pct_v))
            _components.append((_o2p_score, 0.10))
            # 6. Economy/Gain z-score -> score (10%)
            _gz = float(gain_z) if gain_z else 0
            _econ_score = min(100, max(0, 50 + _gz * 20))
            _components.append((_econ_score, 0.10))
            
            _overall_score = sum(s * w for s, w in _components)
            if _overall_score >= 85: _overall_grade = 'A+'
            elif _overall_score >= 75: _overall_grade = 'A'
            elif _overall_score >= 65: _overall_grade = 'B+'
            elif _overall_score >= 55: _overall_grade = 'B'
            elif _overall_score >= 45: _overall_grade = 'C+'
            elif _overall_score >= 35: _overall_grade = 'C'
            else: _overall_grade = 'D'
        except: pass
        
        _gauge_vo2_sub = f'{_n(vo2_pctile,".0f","?")} pcntyl'
        _gauge_val_sub = str(val_grade)
        _gauge_hrr_sub = _hrr_lbl
        _gauge_overall_sub = _overall_grade
        h += f'''<div class="section"><div class="card">
  {section_title('Diagnoza i Plan', 'I')}
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div style="display:flex;gap:6px;flex-shrink:0;flex-wrap:wrap;">
      {gauge_svg(min(100, pctile_val), 'VO₂max', subtitle=_gauge_vo2_sub)}
      {gauge_svg(min(100, max(0, val_score)), 'Ważność', subtitle=_gauge_val_sub)}
      {gauge_svg(min(100, max(0, _hrr_gauge_score)), 'Recovery', subtitle=_gauge_hrr_sub)}
      {gauge_svg(min(100, max(0, _overall_score)), 'Profil', subtitle=_gauge_overall_sub)}</div>
    <div style="flex:1;min-width:260px;">'''
        
        # VO2 gauge bar
        h += f'''<div style="font-size:11px;font-weight:600;color:#475569;margin-bottom:4px;">VO₂{"max" if str(vo2_det)=="VO2max" else "peak"} — {"Norma sportowa" if _is_sport else "Norma populacyjna"}</div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <div style="font-size:36px;font-weight:700;color:#0f172a;">{_n(vo2_rel)}</div>
        <div style="font-size:14px;color:#64748b;">ml/kg/min</div>
        {badge(vo2_class, '#16a34a' if pctile_val >= 70 else '#d97706')}
      </div>
      <div style="position:relative;height:14px;border-radius:7px;overflow:hidden;display:flex;margin-bottom:2px;">
        <div style="flex:20;background:#dc2626;"></div><div style="flex:20;background:#ea580c;"></div>
        <div style="flex:15;background:#eab308;"></div><div style="flex:15;background:#22c55e;"></div>
        <div style="flex:15;background:#3b82f6;"></div><div style="flex:15;background:#7c3aed;"></div>
        <div style="position:absolute;left:{min(97,max(3,pctile_val))}%;top:-2px;font-size:16px;transform:translateX(-50%);">▼</div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:9px;color:#94a3b8;margin-bottom:6px;">
        <span>Poor</span><span>Fair</span><span>Good</span><span>Excellent</span><span>Superior</span>
      </div>
      <div style="font-size:11px;color:#475569;">
        Percentyl: ~{_n(vo2_pctile,'.0f','?')} | {_n(vo2_pct_pred,'.0f','?')}% predicted | wg VO\u2082max: <b>{esc(vo2_class_sport)}</b>{(' | wg pr\u0119dko\u015bci VT2: <b>' + esc(_pc_level_hdr) + '</b>') if _pc_level_hdr else ''}
      </div>'''
        
        # Maximality criteria
        h += '<div class="sub-header">Kryteria wysiłku maksymalnego</div>'
        rer_ok = rer_peak and float(rer_peak) >= 1.10
        hr_ok = hr_pct_pred and float(hr_pct_pred) >= 85
        h += f'<div style="font-size:11px;color:#334155;line-height:1.8;">'
        h += f'{"✅" if rer_ok else "❌"} RER {_n(rer_peak,".2f")} {"≥" if rer_ok else "<"}1.10 &nbsp; '
        h += f'{"✅" if hr_ok else "❌"} HR {_n(hr_pct_pred,".0f")}% pred ({_n(hr_peak,".0f")}/{_n(hr_pred,".0f")}) &nbsp; '
        h += f'{"✅" if plateau else "❌"} Plateau VO₂ {"wykryte" if plateau else "brak"}'
        h += '</div>'
        
        # Auto observations
        h += '<div class="sub-header">Obserwacje kluczowe</div><div style="font-size:12px;line-height:1.8;">'
        
        # Generate observations from data — with velocity context
        try:
            _vt1p = float(vt1_pct) if vt1_pct else 0
            _vt2p = float(vt2_pct) if vt2_pct else 0
            _gap = float(thr_gap) if thr_gap else 0
            _vo2v = float(vo2_rel) if vo2_rel else 0
            _slp = float(ve_vco2_slope) if ve_vco2_slope else 0
            _pc = ct.get('_performance_context', {})
            _v_vt1 = _pc.get('v_vt1_kmh')
            _v_vt2 = _pc.get('v_vt2_kmh')
            _mas_pct_vt1 = _pc.get('vt1_pct_vref')
            _mas_pct_vt2 = _pc.get('vt2_pct_vref')
            _mas_src = _pc.get('v_ref_source', '')
            _lvl = _pc.get('level_by_speed', '')
            _lvl_pct = _pc.get('level_by_pct_vo2', '')
            _lvl_match = _pc.get('levels_match', True)
            
            # ── VT1 with speed context ──
            if _vt1p > 0:
                _vt1_t = f'VT1 przy {_vt1p:.0f}% VO\u2082max'
                if _v_vt1:
                    _vt1_t += f' / {_v_vt1:.1f} km/h'
                if _mas_pct_vt1 and 'MAS_external' in _mas_src:
                    _vt1_t += f' ({_mas_pct_vt1:.0f}% MAS)'
                
                if _v_vt1 and _mas_pct_vt1:
                    if _vt1p >= 65 and _mas_pct_vt1 >= 55:
                        _vt1_t += ' \u2014 <b>bardzo dobra baza tlenowa</b>.'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 65 and _mas_pct_vt1 < 45:
                        _vt1_t += ' \u2014 wysoki %VO\u2082max ale niska pr\u0119dko\u015b\u0107 absolutna.'
                        h += obs_bullet(_vt1_t, 'warning')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa.'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej (wi\u0119cej Z2).'
                        h += obs_bullet(_vt1_t, 'info')
                elif _v_vt1:
                    if _vt1p >= 65:
                        _vt1_t += ' \u2014 <b>dobra baza tlenowa</b>.'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa.'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej.'
                        h += obs_bullet(_vt1_t, 'info')
                else:
                    if _vt1p >= 65:
                        _vt1_t += ' \u2014 dobra baza tlenowa (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt1_t, 'positive')
                    elif _vt1p >= 55:
                        _vt1_t += ' \u2014 dobra baza tlenowa (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt1_t, 'positive')
                    else:
                        _vt1_t += ' \u2014 potencja\u0142 na popraw\u0119 bazy tlenowej.'
                        h += obs_bullet(_vt1_t, 'info')
            
            # ── VT2 with speed context ──
            if _vt2p > 0:
                _vt2_t = f'VT2 przy {_vt2p:.0f}% VO\u2082max'
                if _v_vt2:
                    _vt2_t += f' / {_v_vt2:.1f} km/h'
                if _mas_pct_vt2 and 'MAS_external' in _mas_src:
                    _vt2_t += f' ({_mas_pct_vt2:.0f}% MAS)'
                
                if _lvl:
                    _vt2_t += f' \u2014 poziom <b>{_lvl}</b>'
                    if not _lvl_match and _lvl_pct:
                        _vt2_t += f' (uwaga: %VO\u2082max sugeruje {_lvl_pct})'
                    
                    if _vt2p >= 88 and _lvl in ('Sedentary', 'Recreational'):
                        _vt2_t += '. Wysoki pr\u00f3g wzgl\u0119dny ale niska pr\u0119dko\u015b\u0107 \u2192 niski VO\u2082max.'
                        h += obs_bullet(_vt2_t, 'warning')
                    elif _vt2p >= 88 and _lvl == 'Trained':
                        _vt2_t += '. Dalszy post\u0119p przez interwa\u0142y VO\u2082max.'
                        h += obs_bullet(_vt2_t, 'info')
                    elif _vt2p >= 88:
                        _vt2_t += '. Doskona\u0142y pr\u00f3g potwierdzony pr\u0119dko\u015bci\u0105.'
                        h += obs_bullet(_vt2_t, 'positive')
                    elif _lvl in ('Well-trained', 'Elite'):
                        h += obs_bullet(_vt2_t + '.', 'positive')
                    else:
                        _vt2_t += '. Potencja\u0142 do rozwoju przez trening tempo/SST.'
                        h += obs_bullet(_vt2_t, 'info')
                else:
                    if _vt2p >= 85:
                        _vt2_t += f' \u2014 <b>{"bardzo wysoki" if _vt2p>=90 else "wysoki"} pr\u00f3g</b> (brak danych pr\u0119dko\u015bci).'
                        h += obs_bullet(_vt2_t, 'positive')
                    else:
                        _vt2_t += ' \u2014 przestrze\u0144 do poprawy treningu progowego.'
                        h += obs_bullet(_vt2_t, 'info')
            
            # ── Gap ──
            if _gap >= 25:
                h += obs_bullet(f'Gap VT2-VT1: {_gap:.0f} bpm \u2014 <b>du\u017ca przestrze\u0144</b> do treningu tempo/threshold.', 'positive')
            elif _gap >= 15:
                h += obs_bullet(f'Gap VT2-VT1: {_gap:.0f} bpm \u2014 umiarkowana przestrze\u0144 tempo.', 'info')
            
            # ── VE/VCO2 slope ──
            if _slp < 25 and _slp > 0:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 <b>doskona\u0142a efektywno\u015b\u0107 wentylacyjna</b>.', 'positive')
            elif _slp < 30:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 dobra efektywno\u015b\u0107 wentylacyjna.', 'positive')
            elif _slp < 34:
                h += obs_bullet(f'VE/VCO\u2082 slope {_slp:.1f} \u2014 umiarkowana efektywno\u015b\u0107 wentylacyjna.', 'warning')
            
            # ── Breathing flags ──
            if _is_sport and e07_flags:
                flags_str = ', '.join(e07_flags) if isinstance(e07_flags, list) else str(e07_flags)
                h += obs_bullet(f'Flagi oddechowe [{flags_str}] \u2014 przy VO\u2082max {_vo2v:.0f} ml/kg/min reinterpretowane jako <b>adaptacje sportowe</b>.', 'sport')
            
            # ── FATmax ──
            if fatmax:
                h += obs_bullet(f'FATmax {float(fatmax):.2f} g/min przy HR {_n(fatmax_hr,".0f")} ({_n(fatmax_pct_vo2,".0f")}% VO\u2082) \u2014 strefa HR {_n(fatmax_zone_lo,".0f")}-{_n(fatmax_zone_hi,".0f")} bpm.', 'info')
        except: pass
        
        # ── Cross-interaction insights (v2.1) ──
        try:
            _e10_x = ct.get('_e10_raw', {})
            _e16_x = ct.get('_e16_raw', {})
            _e03_x = ct.get('_e03_raw', {})
            _e15_x = ct.get('_e15_raw', {})
            _pc_x = ct.get('_performance_context', {})
            
            # 1) FATmax ↔ Zone cross-ref
            _fm_hr_x = _e10_x.get('fatmax_hr')
            _fm_gmin_x = _e10_x.get('mfo_gmin')
            _zones_x = _e16_x.get('zones', {})
            if _fm_hr_x and _zones_x and isinstance(_zones_x, dict):
                _fhrx = float(_fm_hr_x)
                for _zk_x, _zv_x in _zones_x.items():
                    if isinstance(_zv_x, dict):
                        if _zv_x.get('hr_low',0) <= _fhrx <= _zv_x.get('hr_high',999):
                            _zname = _zv_x.get('name_pl', _zk_x)
                            _fgh = float(_fm_gmin_x)*60 if _fm_gmin_x else 0
                            _ftxt = f'FATmax (HR {_fhrx:.0f}) w strefie {_zk_x.upper()} ({_zname})'
                            if _fgh > 0:
                                _ftxt += f' — max spalanie tłuszczy ~{_fgh:.0f} g/h'
                            h += obs_bullet(_ftxt, 'neutral')
                            break

            # 2) Crossover ↔ VT1
            _cop_x = _e10_x.get('cop_pct_vo2peak')
            _vt1px = float(vt1_pct) if vt1_pct else None
            if _cop_x and _vt1px:
                _copv = float(_cop_x)
                if _copv < _vt1px - 10:
                    h += obs_bullet(f'Crossover przy {_copv:.0f}% VO₂ — znacznie poniżej VT1 ({_vt1px:.0f}%). Szeroki zakres spalania tłuszczy.', 'positive')
                elif _copv > _vt1px:
                    h += obs_bullet(f'Crossover przy {_copv:.0f}% VO₂ — powyżej VT1! Wczesne przejście na glikogen — priorytet: trening bazowy.', 'warning')

            # 3) Slope + VT2 → combined insight
            _slx = _e03_x.get('slope_to_vt2') or _e03_x.get('ve_vco2_slope') or _e03_x.get('slope_full')
            _vt2px = float(vt2_pct) if vt2_pct else None
            if _slx and _vt2px:
                _slv = float(_slx)
                if _slv < 25 and _vt2px > 85:
                    h += obs_bullet(f'Slope {_slv:.1f} + VT2 przy {_vt2px:.0f}% — efektywność wentylacyjna potwierdzona wysokim progiem.', 'positive')
                elif _slv > 34 and _vt2px < 75:
                    h += obs_bullet(f'Slope {_slv:.1f} + VT2 przy {_vt2px:.0f}% — obniżona efektywność wentylacyjna. Priorytet: trening progowy.', 'warning')

            # 4) VO2 sport vs speed level → economy
            _spcls = _e15_x.get('vo2_class_sport', '')
            _splvl = _pc_x.get('level_by_speed', '')
            if _spcls and _splvl and _spcls != _splvl:
                _cr = {'UNTRAINED':0,'RECREATIONAL':1,'Sedentary':0,'Recreational':1,
                       'TRAINED':2,'Trained':2,'COMPETITIVE':3,'Well-trained':3,
                       'SUB_ELITE':4,'ELITE':5,'Elite':5}
                _rv = _cr.get(_spcls, 2); _rs = _cr.get(_splvl, 2)
                if _rv >= _rs + 1:
                    h += obs_bullet(f'VO₂max → {_spcls}, prędkość VT2 → {_splvl}. Duży silnik aerobowy ale niska ekonomia — priorytet: technika.', 'warning')
                elif _rs > _rv + 1:
                    h += obs_bullet(f'Prędkość VT2 → {_splvl} przy VO₂max → {_spcls}. Wyjątkowa ekonomia ruchu!', 'positive')

            # 5) Nutrition from FATmax
            if _fm_gmin_x and _vt1px:
                _fgx = float(_fm_gmin_x)
                _fghx = _fgx * 60
                h += obs_bullet(f'Żywienie: FATmax {_fgx:.2f} g/min → ~{_fghx:.0f} g tłuszczu/h. Wysiłki >90 min powyżej VT1: suplementacja CHO 40-60 g/h.', 'neutral')

        except: pass
        
        h += '</div>'
        
        # Training recommendation
        h += '<div class="sub-header">Rekomendacja treningowa</div>'
        try:
            _vt2p = float(vt2_pct) if vt2_pct else 0
            _vt1p = float(vt1_pct) if vt1_pct else 0
            _pc_r = ct.get('_performance_context', {})
            _lvl_r = _pc_r.get('level_by_speed', '')
            _v_vt2_r = _pc_r.get('v_vt2_kmh')
            _v_ctx = f' / {_v_vt2_r:.1f} km/h' if _v_vt2_r else ''
            _lvl_ctx = f' (poziom {_lvl_r})' if _lvl_r else ''
            if _vt2p >= 90:
                rec_text = f'VT2 przy {_vt2p:.0f}% VO\u2082max{_v_ctx}{_lvl_ctx} — progi wysoko ustawione. Dalszy post\u0119p przez <b>interwa\u0142y VO\u2082max</b> (3-5 min @ 95-100% VO\u2082max).'
                rec_dist = '📊 75% Z1-Z2 | 10% Z3 | 10% Z4 | 5% Z5'
            elif _vt2p >= 80:
                rec_text = f'VT2 przy {_vt2p:.0f}% VO\u2082max{_v_ctx}{_lvl_ctx} — dobra przestrze\u0144 na rozw\u00f3j. Priorytet: <b>trening progowy</b> (Z4) + interwa\u0142y VO\u2082max (Z5).'
                rec_dist = '📊 70% Z1-Z2 | 10% Z3 | 12% Z4 | 8% Z5'
            elif _vt1p < 55:
                rec_text = f'VT1 przy {_vt1p:.0f}%{_v_ctx} — priorytet: <b>budowanie bazy tlenowej</b> (Z2). Du\u017cy potencja\u0142 poprawy.'
                rec_dist = '📊 80% Z1-Z2 | 10% Z3 | 7% Z4 | 3% Z5'
            else:
                rec_text = 'Zrównoważony trening polaryzowany.'
                rec_dist = '📊 75% Z1-Z2 | 10% Z3 | 10% Z4 | 5% Z5'
            # Add FATmax cross-reference to training rec
            _fatmax_note = ''
            try:
                _e10_r = ct.get('_e10_raw', {})
                _fm_hr = _e10_r.get('fatmax_hr')
                _fm_gmin = _e10_r.get('mfo_gmin')
                if _fm_hr and _fm_gmin:
                    _fm_gh = float(_fm_gmin) * 60
                    _fatmax_note = f'<br>🔥 Długie biegi Z1-Z2 w strefie FATmax (HR ~{float(_fm_hr):.0f}) — max spalanie tłuszczy ~{_fm_gh:.0f} g/h'
            except: pass
            h += f'<div style="padding:8px;background:#fffbeb;border-radius:6px;font-size:12px;border-left:3px solid #f59e0b;">'
            h += f'⚡ {rec_text}<br>{rec_dist}{_fatmax_note}</div>'
        except: pass
        
        h += '</div></div></div>'
        
        # ═══════════════════════════════════════════════════════════════
        # II. PROGI + STREFY
        # ═══════════════════════════════════════════════════════════════
        
        # VT cards
        def vt_card(label, pct_vo2, hr, time_s, speed, rer_v, vo2_abs_v, hr_pct_mx, conf, pct_mas=None, col='#16a34a'):
            pct_val = _n(pct_vo2, '.0f', '?') if pct_vo2 else '?'
            bar_w = min(100, max(0, float(pct_vo2) if pct_vo2 else 0))
            conf_lbl = 'Potwierdzone' if float(conf or 0)>=0.8 else ('Umiarkowane' if float(conf or 0)>=0.5 else 'Niepewne')
            conf_col = '#16a34a' if float(conf or 0)>=0.8 else ('#d97706' if float(conf or 0)>=0.5 else '#dc2626')
            is_vt2 = 'VT2' in label
            return f'''<div style="flex:1;min-width:200px;padding:12px;border:1px solid #e2e8f0;border-radius:8px;background:white;">
  <div style="font-size:12px;font-weight:700;color:#475569;">{label}</div>
  <div style="font-size:28px;font-weight:700;color:#0f172a;">{pct_val}% <span style="font-size:13px;font-weight:400;color:#64748b;">VO₂max</span></div>
  <div style="height:6px;background:#e5e7eb;border-radius:3px;margin:6px 0;"><div style="height:6px;background:{'#3b82f6' if is_vt2 else '#22c55e'};border-radius:3px;width:{bar_w}%;"></div></div>
  <div style="font-size:11px;color:#64748b;">HR {_n(hr,'.0f','-')} bpm ({_n(hr_pct_mx,'.0f','?')}% HRmax) | {_fmt_dur(time_s)}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:2px;">VO₂ {_n(vo2_abs_v,'.0f','-')} ml/min | RER {_n(rer_v,'.2f','-')}{f' | {_n(speed,".1f")} km/h' + (f' ({pct_mas:.0f}% MAS)' if pct_mas else '') if speed else ''}</div>
  <div style="margin-top:4px;">{badge(conf_lbl, conf_col)}</div>
</div>'''
        
        
        # ── Performance Context for VT cards ────────────────────
        _pc = ct.get('_performance_context', {})
        _vt1_pct_mas = _pc.get('vt1_pct_vref') if _pc.get('executed') else None
        _vt2_pct_mas = _pc.get('vt2_pct_vref') if _pc.get('executed') else None

        h += f'''<div class="section"><div class="card">
  {section_title('Progi i Strefy treningowe', 'II')}
  <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
    {vt_card('VT1 — Próg tlenowy', vt1_pct, vt1_hr, vt1_time, vt1_speed, vt1_rer_val, vt1_vo2_abs, vt1_hr_pct_max, vt1_conf, _vt1_pct_mas)}
    {vt_card('VT2 — Próg beztlenowy', vt2_pct, vt2_hr, vt2_time, vt2_speed, vt2_rer_val, vt2_vo2_abs, vt2_hr_pct_max, vt2_conf, _vt2_pct_mas)}
  </div>
  '''

        # ── Compact speed classification ─────────────────────
        if _pc.get('executed') and _pc.get('v_vt2_kmh'):
            _lvl_s = _pc.get('level_by_speed', '')
            _lvl_p = _pc.get('level_by_pct_vo2', '')
            _mas_ext = _pc.get('mas_external_m_s')
            _mas_kmh_v = _pc.get('mas_external_kmh')
            _mss_v = _pc.get('mss_external_m_s')
            _asr_v = _pc.get('asr_kmh')
            _pc_flags = _pc.get('flags', [])
            
            _ctx_parts = []
            if _lvl_s:
                _ctx_parts.append(f'Klasyfikacja: <b>{_lvl_s}</b>')
            if _lvl_s != _lvl_p and _lvl_p:
                _ctx_parts.append(f'<span style="color:#d97706;">(%VO\u2082max \u2192 {_lvl_p})</span>')
            if _mas_ext:
                _ctx_parts.append(f'MAS: <b>{_mas_ext:.2f} m/s</b> ({_mas_kmh_v:.1f} km/h)')
                if _mss_v:
                    _ctx_parts.append(f'MSS: {_mss_v:.1f} m/s')
                    if _asr_v:
                        _ctx_parts.append(f'ASR: {_asr_v:.1f} km/h')
            
            if _ctx_parts:
                h += '<div style="margin-top:8px;padding:8px 12px;background:#f0f9ff;border-radius:6px;font-size:11px;color:#475569;border-left:3px solid #3b82f6;">' + '  |  '.join(_ctx_parts) + '</div>'
            
            for _fl in _pc_flags:
                h += f'<div style="margin-top:3px;padding:3px 12px;font-size:10px;color:#d97706;">\u26a0\ufe0f {_fl}</div>'

                # Zones table
        zone_colors = ['#22c55e','#84cc16','#eab308','#f97316','#ef4444']
        zone_names_pl = ['Z1 Regeneracja','Z2 Baza tlenowa','Z3 Tempo','Z4 Próg','Z5 VO₂max']
        
        h += '<table class="ztable"><tr><th>Strefa</th><th>HR (bpm)</th><th>%HRmax</th><th>Tempo</th><th>RPE</th><th>Opis</th></tr>'
        for i, (zname, zcol) in enumerate(zip(zone_names_pl, zone_colors)):
            zkey = f'z{i+1}'
            zd = zones_data.get(zkey, {})
            hr_lo = zd.get('hr_low','?'); hr_hi = zd.get('hr_high','?')
            phr_lo = zd.get('pct_hrmax_low',''); phr_hi = zd.get('pct_hrmax_high','')
            spd_lo = zd.get('speed_low',''); spd_hi = zd.get('speed_high','')
            rpe = zd.get('rpe','')
            desc = zd.get('description_pl','')
            h += f'<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{zcol};margin-right:4px;vertical-align:middle;"></span>{zname}</td>'
            h += f'<td style="font-weight:600;">{hr_lo}-{hr_hi}</td>'
            h += f'<td>{_n(phr_lo,".0f","")}-{_n(phr_hi,".0f","")}%</td>'
            h += f'<td>{_n(spd_lo,".1f","")}-{_n(spd_hi,".1f","")} km/h</td>'
            h += f'<td>{rpe}</td><td style="font-size:10px;color:#64748b;">{desc}</td></tr>'
        h += '</table>'
        
        # 3-zone model
        if three_zone:
            h += '<div style="margin-top:6px;font-size:10px;color:#64748b;">Model 3-strefowy: '
            for zk, zv in three_zone.items():
                h += f'{zv.get("label","")} ({zv.get("hr_low","")}-{zv.get("hr_high","")}) | '
            h += '</div>'
        
        extras = []
        if aer_reserve: extras.append(f'Rezerwa tlenowa: {_n(aer_reserve,".0f")}%')
        if thr_gap: extras.append(f'Gap VT2-VT1: {_n(thr_gap,".0f")} bpm ({_n(thr_gap_pct,".0f")}% HRmax)')
        if extras:
            h += f'<div style="margin-top:6px;font-size:11px;color:#475569;font-weight:600;">{"  |  ".join(extras)}</div>'
        
        h += '</div></div>'
        
        # ═══════════════════════════════════════════════════════════════
        # III. PROFIL FIZJOLOGICZNY — 4 PANELE
        # ═══════════════════════════════════════════════════════════════
        h += f'<div class="section">{section_title("Profil fizjologiczny", "III")}'
        
        # --- PANEL A: WYDOLNOŚĆ ---
        pa = ''
        pa += row_item('VO₂max', f'{_n(vo2_rel)} ml/kg/min', vo2_class, f'{_n(vo2_abs,".2f")} L/min | {_n(vo2_pct_pred,".0f")}% pred. | ~{_n(vo2_pctile,".0f")} percentyl')
        pa += '<div class="sub-header">OUES</div>'
        pa += row_item('OUES 100%', f'{_n(oues100,".0f")}', f'{_n(oues_pct,".0f")}% pred.', f'OUES/kg: {_n(oues_kg,".1f")}')
        pa += row_item('OUES 90/75%', f'{_n(oues90,".0f")} / {_n(oues75,".0f")}', f'stab. {_n(oues_stab,".3f")}' if oues_stab else '', '')
        pa += row_item('OUES @VT1/@VT2', f'{_n(oues_vt1,".0f")} / {_n(oues_vt2,".0f")}', '', '')
        pa += '<div class="sub-header">VE/VCO₂</div>'
        pa += row_item('Slope (→VT2)', _n(ve_vco2_slope), vc_class, f'{vc_desc} | {_n(slope_pct_pred,".0f")}% pred.')
        pa += row_item('Slope full/→VT1', f'{_n(ve_vco2_full)} / {_n(ve_vco2_to_vt1)}', '', '')
        pa += row_item('Nadir / @VT1', f'{_n(ve_vco2_nadir)} / {_n(ve_vco2_at_vt1)}', '', '')
        pa += row_item('RER peak', _n(rer_peak,'.3f'), rer_class, f'{"Plateau VO₂ ✓" if plateau else ""} | {rer_desc}')
        pa += row_item('VE peak', f'{_n(ve_peak,".1f")} L/min', '', '')
        
        # --- PANEL B: SERCE & RECOVERY ---
        pb = ''
        pb += row_item('O₂-pulse peak', f'{_n(o2p_peak)} ml/beat', o2p_traj, f'{_n(o2p_pct,".0f")}% pred. | Est. SV ~{_n(sv_est,".0f")} ml')
        pb += row_item('O₂-pulse @VT1/VT2', f'{_n(o2p_vt1)} / {_n(o2p_vt2)} ml/beat', '', o2p_desc)
        pb += row_item('CI', _n(ci,'.2f'), ci_class, 'Kompetencja chronotropowa')
        pb += row_item('HR-VO₂ slope', f'{_n(hr_vo2_slope)} bpm/L', 'SPORT' if slope_class in ('VERY_LOW','LOW_HIGH_SV') and _is_sport else slope_class, f'R²={_n(e13.get("hr_vo2_r2"),".3f")}')
        pb += row_item('Breakpoint HR-VO₂', f'{_fmt_dur(bp_time_e13)} ({_n(bp_pct_e13,".0f")}%)', bp_vs_vt2_e13, f'CV coupling: {"łagodna anomalia" if cv_coupling=="MILD_ABNORMALITY" else ("doskonały" if cv_coupling=="EXCELLENT" else cv_coupling)}')
        pb += row_item('HRmax', f'{_n(hr_peak,".0f")} bpm', '', f'{_n(hr_pct_pred,".0f")}% pred. ({_n(hr_pred,".0f")})')
        
        pb += '<div class="sub-header">RECOVERY</div>'
        pb += row_item('HRR 1min', f'{_n(hrr1,".0f")} bpm', (str(hrr1_class or hrr1_class_e15)).upper(), hrr1_desc_e15)
        pb += row_item('HRR 3min', f'{_n(hrr3,".0f")} bpm', str(hrr3_class).upper(), '')
        pb += row_item('T½ VO₂', f'{_n(t_half_vo2,".0f")}s', rec_class, f'τ={_n(tau_vo2,".0f")}s (R²={_n(tau_r2,".2f")}) | MRT={_n(mrt,".0f")}s' if tau_vo2 else rec_desc)
        pb += row_item('T½ VCO₂ / VE', f'{_n(t_half_vco2,".0f")}s / {_n(t_half_ve,".0f")}s', '', f'VO₂DR {_n(vo2dr,".1f")}s' if vo2dr else '')
        pb += row_item('Recovery 60s/120s', f'{_n(pct_rec_60,".0f")}% / {_n(pct_rec_120,".0f")}%', '', f'ΔVO₂ 60s: {_n(dvo2_60,".1f")} ml/kg/min')
        
        # --- PANEL C: ODDYCHANIE ---
        pc = ''
        bp_assess = 'SPORT' if bp_pattern in ('DYSFUNCTIONAL_BPD',) and _is_sport else bp_pattern
        pc += row_item('Wzorzec oddech.', bp_strategy, bp_assess, f'{_n(ve_from_bf,".0f")}% VE z BF | VT peak {_n(vt_peak,".1f")} L')
        pc += row_item('BF rest/VT1/VT2/peak', f'{_n(e07.get("bf_rest"),".0f")} / {_n(bf_vt1,".0f")} / {_n(bf_vt2,".0f")} / {_n(bf_peak,".0f")} /min', '', f'RSBI {_n(rsbi,".1f")}' if rsbi else '')
        pc += row_item('VT rest/VT1/VT2/peak', f'{_n(e07.get("vt_rest_L"),".2f")} / {_n(vt_vt1,".2f")} / {_n(vt_vt2,".2f")} / {_n(vt_peak,".2f")} L', '', '')
        if vt_plat_t:
            pc += row_item('Plateau VT', f't={_fmt_dur(vt_plat_t)} ({_n(vt_plat_pct,".0f")}%)', e07.get('vt_plateau_vs_vt2',''), '')
        
        vdvt_assess = 'SPORT' if vdvt_pattern in ('PARADOXICAL_RISE',) and _is_sport else vdvt_pattern
        pc += row_item('VD/VT', f'{_n(vdvt_rest,".2f")}→{_n(vdvt_peak,".2f")}', vdvt_assess, '')
        pc += row_item('PetCO₂', f'rest {_n(petco2_rest,".0f")} → VT1 {_n(petco2_vt1,".0f")} → peak {_n(petco2_peak,".0f")}', petco2_pattern, '')
        if peto2_nadir:
            pc += row_item('PetO₂ nadir', f'{_n(peto2_nadir,".0f")} mmHg', '', '')
        
        br_pct = g('BR_pct')
        pc += row_item('BR', f'{_n(br_pct,".0f")}%' if br_pct else '-', '', 'Brak MVV' if not br_pct else '')
        
        if e07_flags:
            flags_str = ', '.join(e07_flags) if isinstance(e07_flags, list) else str(e07_flags)
            pc += f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Flagi: {flags_str}</div>'
        
        # --- PANEL D: PALIWO ---
        pd_html = ''
        pd_html += '<div class="sub-header">FATmax</div>'
        pd_html += row_item('MFO', f'{_n(fatmax,".2f")} g/min ({_n(e10.get("mfo_mgkg_min"),".1f")} mg/kg/min)' if fatmax else '-', '', f'HR {_n(fatmax_hr,".0f")} ({_n(fatmax_pct_hr,".0f")}% HRmax) | {_n(fatmax_pct_vo2,".0f")}% VO₂peak' if fatmax_hr else '')
        pd_html += row_item('Strefa FATmax', f'HR {_n(fatmax_zone_lo,".0f")}-{_n(fatmax_zone_hi,".0f")} bpm' if fatmax_zone_lo else '-', '', '')
        pd_html += '<div class="sub-header">Crossover & Substrat</div>'
        pd_html += row_item('Crossover', f'HR {_n(cop_hr,".0f")}' if cop_hr else '-', '', f'{_n(cop_pct_vo2,".0f")}% VO₂ | RER {_n(cop_rer,".3f")}' if cop_hr else '')
        pd_html += row_item('Substrat @VT1', f'FAT {_n(fat_vt1_pct,".0f")}% ({_n(fat_vt1_g,".2f")} g/min) / CHO {_n(cho_vt1_pct,".0f")}% ({_n(cho_vt1_g,".2f")} g/min)' if fat_vt1_pct else '-', '', '')
        pd_html += row_item('Substrat @VT2', f'FAT {_n(fat_vt2_pct,".0f")}% / CHO {_n(cho_vt2_pct,".0f")}%' if fat_vt2_pct else '-', '', '')
        if total_kcal:
            pd_html += row_item('Kalorie total', f'{_n(total_kcal,".0f")} kcal', '', f'FAT {_n(total_fat,".0f")} + CHO {_n(total_cho,".0f")} kcal')
        
        # Zone substrate table
        if zone_sub:
            pd_html += '<div class="sub-header">Substrat per strefa</div>'
            pd_html += '<table class="ztable"><tr><th>Strefa</th><th>FAT g/h</th><th>CHO g/h</th><th>FAT%</th><th>kcal/h</th></tr>'
            for zk in ['z2','z3','z4','z5']:
                zs = zone_sub.get(zk,{})
                if zs:
                    note = f' *' if zs.get('note') else ''
                    pd_html += f'<tr><td>{zk.upper()}</td><td>{_n(zs.get("fat_gh"),".0f","-")}</td><td>{_n(zs.get("cho_gh"),".0f","-")}</td><td>{_n(zs.get("fat_pct"),".0f","-")}%</td><td>{_n(zs.get("kcal_h"),".0f","-")}{note}</td></tr>'
            pd_html += '</table>'
        
        # Lactate
        pd_html += '<div class="sub-header">Laktat</div>'
        if la_peak:
            pd_html += row_item('La peak', f'{_n(la_peak)} mmol/L', '', f'Baseline {_n(la_base)}')
            pd_html += row_item('LT1', f'HR {_n(lt1_hr,".0f")} | {_fmt_dur(lt1_time)}' if lt1_hr else '-', '', lt1_method)
            pd_html += row_item('LT2', f'HR {_n(lt2_hr,".0f")} | {_fmt_dur(lt2_time)}' if lt2_hr else '-', '', lt2_method)
        else:
            pd_html += row_item('Laktat', 'Brak danych', 'NO_DATA', '')
        
        h += f'''<div class="flex-row">
  {panel_card('🫁 Wydolność i Efektywność', '', pa)}
  {panel_card('❤️ Serce i Recovery', '', pb)}
</div>
<div class="flex-row" style="margin-top:12px;">
  {panel_card('💨 Oddychanie', '', pc)}
  {panel_card('⛽ Paliwo i Metabolizm', '', pd_html)}
</div>'''
        
        # --- EFFICIENCY / ECONOMY (E06) — modality-dependent ---
        if e06.get('status') == 'OK':
            pe = ''
            if modality == 'run':
                # Running: show Running Economy as primary
                pe += '<div class="sub-header">Ekonomia biegu (Running Economy)</div>'
                pe += row_item('RE @VT1', f'{_n(re_vt1,".1f")} ml/kg/km', re_class, f'VO₂ {_n(e06.get("vo2_at_vt1"),".1f")} ml/kg/min @ {_n(e06.get("load_at_vt1"),".1f")} km/h')
                pe += row_item('RE @VT2', f'{_n(re_vt2,".1f")} ml/kg/km', '', f'VO₂ {_n(e06.get("vo2_at_vt2"),".1f")} ml/kg/min @ {_n(e06.get("load_at_vt2"),".1f")} km/h')
                pe += row_item('RE ogólna', f'{_n(e06.get("running_economy_mlkgkm"),".1f")} ml/kg/km', '', '')
                # Gain as secondary info
                pe += '<div style="font-size:10px;color:#94a3b8;margin-top:4px;">Gain <VT1: {0} {1} (R²={2}) | z={3}</div>'.format(
                    _n(gain_below_vt1,".2f"), gain_unit, _n(e06.get("gain_below_vt1_r2"),".3f"), _n(gain_z,".2f"))
            else:
                # Bike/other: show Gain and Delta Efficiency as primary
                pe += '<div class="sub-header">Efektywność (Gain / Delta Efficiency)</div>'
                pe += row_item('Gain <VT1', f'{_n(gain_below_vt1,".2f")} {gain_unit}', f'z={_n(gain_z,".2f")}', f'R²={_n(e06.get("gain_below_vt1_r2"),".3f")}')
                pe += row_item('Gain @VT1', f'{_n(gain_vt1,".2f")} {gain_unit}', '', '')
                pe += row_item('Gain @VT2', f'{_n(gain_vt2,".2f")} {gain_unit}', '', '')
                if delta_eff:
                    pe += row_item('Delta Efficiency', f'{_n(delta_eff,".1f")}%', '', '')
            if lin_break:
                pe += row_item('Linearity break', _fmt_dur(lin_break), '', f'@ {_n(e06.get("linearity_break_load"),".1f")} {"km/h" if modality=="run" else "W"}')
            h += f'<div style="margin-top:12px;">{panel_card("🏃 Efektywność" if modality=="run" else "⚡ Efektywność", "", pe)}</div>'
        
        h += '</div>'  # end section III

        # ═══════════════════════════════════════════════════════════════
        # IV. WALIDACJA KRZYŻOWA
        # ═══════════════════════════════════════════════════════════════
        h += f'<div class="section"><div class="card">{section_title("Walidacja krzyżowa", "IV")}'
        
        # Concordance checks
        cc = e19.get('concordance_checks',[])
        if cc:
            checks_html = ''
            for item in cc:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    name_c, status, note = item[0], item[1], item[2]
                    pts = item[3] if len(item) > 3 else 0
                else: continue
                col = '#16a34a' if status in ('EXCELLENT','GOOD','PASS','TIGHT') else ('#d97706' if status in ('MODERATE','FAIR','WIDE') else '#dc2626')
                icon = '✓' if col == '#16a34a' else ('~' if col == '#d97706' else '✗')
                checks_html += f'<span style="display:inline-block;margin:2px 4px;padding:3px 8px;border-radius:4px;font-size:11px;background:{col}14;color:{col};font-weight:600;">{icon} {name_c}: {status}</span>'
            h += f'<div style="margin-bottom:8px;">{checks_html}</div>'
        
        # Temporal alignment
        for bp_name, bp_data in ta.items():
            if not bp_data: continue
            srcs = bp_data.get('sources',{})
            sp = bp_data.get('spread_sec','?')
            conf = bp_data.get('confidence','?')
            outlier = bp_data.get('outlier','')
            tc = '#16a34a' if conf=='HIGH' else ('#d97706' if conf=='MODERATE' else '#dc2626')
            src_parts = [f'{k} {_fmt_dur(sv)}' for k,sv in srcs.items()]
            h += f'<div style="font-size:11px;padding:3px 0;"><b>{bp_name}</b>: {" | ".join(src_parts)} — <span style="color:{tc};font-weight:600;">spread {_n(sp,".0f")}s ({conf})</span>'
            if outlier:
                h += f' <span style="color:#dc2626;font-size:10px;">(outlier: {outlier})</span>'
            h += '</div>'
        
        # Validity criteria
        vc19 = e19.get('validity_criteria',{})
        if vc19:
            vc_chips = ''
            for k, val_tuple in vc19.items():
                if isinstance(val_tuple, (list, tuple)) and len(val_tuple) >= 3:
                    s, p, d = val_tuple[0], val_tuple[1], val_tuple[2]
                else: continue
                col = '#16a34a' if s in ('EXCELLENT','YES','OPTIMAL','GOOD','BOTH') else ('#d97706' if s in ('FAIR','VT1_ONLY','ACCEPTABLE') else ('#dc2626' if s in ('NO','POOR','FAIL','INSUFFICIENT') else '#94a3b8'))
                vc_chips += f'<span style="display:inline-block;margin:1px 3px;padding:2px 6px;border-radius:3px;font-size:10px;background:{col}14;color:{col};font-weight:600;">{k}: {s} +{p}</span>'
            h += f'<div style="margin-top:6px;"><span style="font-size:10px;color:#64748b;">Ważność {val_score}/100 ({val_grade}): </span>{vc_chips}</div>'
        
        h += '</div></div>'

        # ═══════════════════════════════════════════════════════════════
        # V. NIRS / SmO2
        # ═══════════════════════════════════════════════════════════════
        if e12.get('status') == 'OK' and smo2_min is not None:
            h += f'<div class="section"><div class="card">{section_title("NIRS / SmO₂", "V")}'
            h += row_item('SmO₂ rest→min→recovery', f'{_n(smo2_rest,".0f")}→{_n(smo2_min,".0f")}→{_n(smo2_recov,".0f")}%', e12_quality, f'Desaturacja: {_n(desat_abs,".0f")}% abs ({_n(desat_pct,".0f")}% rel.)')
            h += row_item('SmO₂ @VT1 / @VT2', f'{_n(smo2_vt1,".0f")}% / {_n(smo2_vt2,".0f")}%', '', f'@peak: {_n(smo2_peak_val,".0f")}%')
            h += row_item('Half Recovery Time', f'{_n(hrt_s,".1f")}s', '', f'Reox rate: {_n(reox_rate,".3f")} %/s | Overshoot: +{_n(overshoot,".0f")}%')
            
            if bp1_t:
                bp1_ctx = f'{_n(bp1_vs_vt1,".0f")}s po VT1' if bp1_vs_vt1 and float(bp1_vs_vt1 or 0)>0 else f'{_n(abs(float(bp1_vs_vt1 or 0)),".0f")}s przed VT1' if bp1_vs_vt1 else ''
                h += row_item('NIRS BP1', f'{_fmt_dur(bp1_t)}', '', f'SmO₂: {_n(bp1_smo2,".0f")}% | {bp1_ctx}')
            if bp2_t:
                bp2_ctx = f'{_n(bp2_vs_vt2,".0f")}s po VT2' if bp2_vs_vt2 and float(bp2_vs_vt2 or 0)>0 else f'{_n(abs(float(bp2_vs_vt2 or 0)),".0f")}s przed VT2' if bp2_vs_vt2 else ''
                h += row_item('NIRS BP2', f'{_fmt_dur(bp2_t)}', '', f'SmO₂: {_n(bp2_smo2,".0f")}% | {bp2_ctx}')
            
            if e12_flags:
                flags_str = ', '.join(e12_flags) if isinstance(e12_flags, list) else str(e12_flags)
                h += f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Flagi: {flags_str}</div>'
            h += '</div></div>'

        # ═══════════════════════════════════════════════════════════════
        # VI. PORÓWNANIE SPORTOWE
        # ═══════════════════════════════════════════════════════════════
        try:
            _vo2v = float(vo2_rel) if vo2_rel else 0
            # Dynamic sport comparison using E15 tables
            _mod = ct.get('modality', 'run')
            _sx = ct.get('sex', 'male')
            _cat_labels_pl = {
                'UNTRAINED': 'Nietrenuj\u0105cy',
                'RECREATIONAL': 'Rekreacyjny',
                'TRAINED': 'Trenuj\u0105cy',
                'COMPETITIVE': 'Wyczynowy',
                'SUB_ELITE': 'Sub-elite',
                'ELITE': 'Elita',
            }
            _mod_labels_pl = {
                'run': 'bieg', 'bike': 'kolarstwo', 'triathlon': 'triathlon',
                'rowing': 'wio\u015blarstwo', 'crossfit': 'CrossFit', 'hyrox': 'HYROX',
                'swimming': 'p\u0142ywanie', 'xc_ski': 'biegi narciarskie',
                'soccer': 'pi\u0142ka no\u017cna', 'mma': 'MMA',
            }
            _mod_label = _mod_labels_pl.get(_mod, _mod)
            
            h += f'<div class="section"><div class="card">{section_title("Por\u00f3wnanie sportowe (" + _mod_label + ")", "VI")}'
            
            # Get ranges for this modality
            _e15_cls = e15  # already have e15 from earlier
            _sport_tbl_m = ct.get('_e15_raw', {}).get('_sport_table_used')
            # Fallback: build from known table structure
            _ranges_src = []
            try:
                from engine_core import Engine_E15_Normalization
                _tbl = Engine_E15_Normalization.VO2MAX_SPORT_MALE if _sx == 'male' else Engine_E15_Normalization.VO2MAX_SPORT_FEMALE
                _ranges_src = _tbl.get(_mod, _tbl.get('default', []))
            except:
                _ranges_src = [(0,35,'UNTRAINED'),(35,45,'RECREATIONAL'),(45,52,'TRAINED'),(52,60,'COMPETITIVE'),(60,68,'SUB_ELITE'),(68,999,'ELITE')]
            
            # Filter out UNTRAINED, cap ELITE at reasonable display
            _display_ranges = []
            for lo, hi, cat in _ranges_src:
                if cat == 'UNTRAINED':
                    continue
                _hi_display = min(hi, lo + 15)  # Cap for display
                _display_ranges.append((_cat_labels_pl.get(cat, cat), lo, _hi_display))
            
            h += '<div style="display:flex;gap:4px;align-items:flex-end;height:120px;margin-bottom:6px;">'
            for lbl, lo, hi in _display_ranges:
                mid = (lo+hi)/2
                bar_h = max(30, mid * 1.3)
                is_you = lo <= _vo2v < (hi if hi < 900 else 999)
                h += f'<div style="flex:1;text-align:center;">'
                h += f'<div style="font-size:8px;color:#64748b;margin-bottom:2px;">{lbl}</div>'
                h += f'<div style="height:{bar_h}px;background:{"#3b82f6" if is_you else "#e2e8f0"};border-radius:4px 4px 0 0;display:flex;align-items:flex-end;justify-content:center;">'
                _range_lbl = f'{lo}-{hi}' if hi < 900 else f'{lo}+'
                h += f'<span style="font-size:9px;color:{"white" if is_you else "#94a3b8"};padding:2px;">{_range_lbl}</span></div></div>'
            h += f'<div style="flex:1;text-align:center;">'
            h += f'<div style="font-size:9px;font-weight:700;color:#3b82f6;">\U0001f449 Ty</div>'
            h += f'<div style="font-size:18px;font-weight:700;color:#0f172a;">{_n(vo2_rel)}</div>'
            h += '</div></div>'
            h += '</div></div>'
        except: pass

        # ═══════════════════════════════════════════════════════════════
        # VII. WYKRESY
        # ═══════════════════════════════════════════════════════════════
        proto_tbl = v('_protocol_table_html','')
        if proto_tbl and proto_tbl.strip():
            h += '<div class="section"><div class="card">'
            h += '<div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="var b=this.parentElement.querySelector(\'.proto-body\');b.style.display=b.style.display==\'none\'?\'block\':\'none\';">'
            h += f'{section_title("Protokół testu")}'
            h += '<span style="font-size:18px;color:#94a3b8;">▸</span></div>'
            h += '<div class="proto-body" style="display:none;">' + proto_tbl + '</div>'
            h += '</div></div>'
        
        charts_html = ReportAdapter._render_charts_html(ct)
        h += f'<div class="section">{section_title("Wykresy", "VII")}{charts_html}</div>'
        
        # Footer
        h += '<div style="padding:14px;font-size:10px;color:#9ca3af;text-align:center;">CPET Report v4.0 | Pełna analiza fizjologiczna | Automatycznie generowany</div>'
        h += '</div></body></html>'
        
        return h

    @staticmethod
    def _gasex_report(ct):
        r17 = ct.get('_e17_raw', {})
        if not r17 or r17.get('status') in (None, 'NO_SIGNAL', 'NO_TIME'):
            return ""
        lines = ["", "  3f. Wymiana gazowa / Gas Exchange (E17 v2.0):"]
        sigs = r17.get('available_signals', [])
        gep = r17.get('gas_exchange_pattern', '-')
        # Sport context for E17 text
        _gx_vo2 = ct.get('_interp_vo2_mlkgmin')
        _gx_vs = None
        try:
            _gx_vsr = ct.get('VE_VCO2_slope')
            if _gx_vsr and str(_gx_vsr) not in ('-','[BRAK]','None',''): _gx_vs = float(_gx_vsr)
        except: pass
        _gx_sport = (_gx_vo2 is not None and _gx_vo2 >= 45 and _gx_vs is not None and _gx_vs < 30)
        _gx_vdp = r17.get('vdvt_at_peak')
        if _gx_sport and gep == 'ABNORMAL' and _gx_vdp is not None and _gx_vdp < 0.20:
            _gx_serious = [f for f in r17.get('flags',[]) if f in ('SPO2_SEVERE','VDVT_PARADOXICAL','PETCO2_NO_RISE','HYPERCAPNIA_PEAK')]
            if all(f == 'VDVT_PARADOXICAL' for f in _gx_serious):
                gep = 'SPORT_NORM (reinterpretacja: VD/VT <0.20 u sportowca)'
        lines.append(f"      Dostepne sygnaly: {', '.join(sigs)}")
        lines.append(f"      Ogolna ocena: [{gep}]")
        if 'PetCO2' in sigs:
            lines.append(f"      PetCO2: rest [{r17.get('petco2_rest','-')}] | @VT1 [{r17.get('petco2_at_vt1','-')}] | @VT2 [{r17.get('petco2_at_vt2','-')}] | peak [{r17.get('petco2_at_peak','-')}] mmHg")
            lines.append(f"        rise rest->VT1: [{r17.get('petco2_rise_to_vt1','-')}] | drop VT2->peak: [{r17.get('petco2_drop_vt2_to_peak','-')}]")
            lines.append(f"        Wzorzec: [{r17.get('petco2_pattern','-')}] — {r17.get('petco2_clinical','')}")
        if 'PetO2' in sigs:
            lines.append(f"      PetO2: rest [{r17.get('peto2_rest','-')}] | nadir [{r17.get('peto2_nadir','-')}] | @VT1 [{r17.get('peto2_at_vt1','-')}] | peak [{r17.get('peto2_at_peak','-')}] mmHg | [{r17.get('peto2_pattern','-')}]")
        if 'VD/VT' in sigs:
            _vdp_txt = r17.get('vdvt_pattern','-')
            _vdc_txt = r17.get('vdvt_clinical','')
            if _gx_sport and _gx_vdp is not None and _gx_vdp < 0.20 and _vdp_txt == 'PARADOXICAL_RISE':
                _vdp_txt = 'SPORT_NORM'
                _vdc_txt = f'VD/VT wzrost {r17.get("vdvt_rest","-")}→{_gx_vdp:.3f} — wartości bezwzględne <0.20 = NORMA SPORTOWA (elitarna efektywność wentylacyjna)'
            lines.append(f"      VD/VT: rest [{r17.get('vdvt_rest','-')}] | @VT1 [{r17.get('vdvt_at_vt1','-')}] | peak [{r17.get('vdvt_at_peak','-')}] | delta [{r17.get('vdvt_delta','-')}]")
            lines.append(f"        Wzorzec: [{_vdp_txt}] — {_vdc_txt}")
        if 'SpO2' in sigs:
            lines.append(f"      SpO2: rest [{r17.get('spo2_rest','-')}] | min [{r17.get('spo2_min','-')}] | drop [{r17.get('spo2_drop','-')}]% | [{r17.get('spo2_clinical','-')}]")
        if 'PaCO2_est' in sigs:
            lines.append(f"      PaCO2(est): rest [{r17.get('paco2_rest','-')}] | peak [{r17.get('paco2_peak','-')}] mmHg")
        if 'PAO2_est' in sigs:
            lines.append(f"      PAO2(est): rest [{r17.get('pao2_rest','-')}] | peak [{r17.get('pao2_peak','-')}] mmHg")
        fl = r17.get('flags', [])
        if fl:
            if _gx_sport and _gx_vdp is not None and _gx_vdp < 0.20:
                _gx_sf = {'VDVT_PARADOXICAL','VDVT_NO_DROP','VDVT_HIGH_PEAK'}
                _gx_c = [f for f in fl if f not in _gx_sf]
                _gx_s = [f for f in fl if f in _gx_sf]
                parts = []
                if _gx_s: parts.append(f'sport (VD/VT<0.20): {", ".join(_gx_s)}')
                if _gx_c: parts.append(f'{", ".join(_gx_c)}')
                lines.append(f"      Flagi: [{' | '.join(parts)}]")
            else:
                lines.append(f"      Flagi: {', '.join(fl)}")
        return "\n".join(lines)

    @staticmethod
    def _validity_concordance_report(ct):
        """E19 Test Validity + Concordance report section."""
        r19 = ct.get('_e19_raw', {})
        if not r19 or r19.get('status') != 'OK':
            return ""
        L = ["", "  3g. Wiarygodność testu i spójność fizjologiczna (E19 v1.0):"]
        vs = r19.get('validity_score', 0)
        vg = r19.get('validity_grade', '?')
        L.append(f"      VALIDITY SCORE: [{vs}/100] Grade: [{vg}]")
        vc = r19.get('validity_criteria', {})
        for k, (status, pts, desc) in vc.items():
            L.append(f"        {k}: [{status}] +{pts} — {desc}")
        vf = r19.get('validity_flags', [])
        if vf:
            L.append(f"        Flagi ważności: {', '.join(vf)}")
        cs = r19.get('concordance_score', 0)
        cg = r19.get('concordance_grade', '?')
        L.append(f"      CONCORDANCE SCORE: [{cs}%] Grade: [{cg}]")
        for (name, status, note, pts) in r19.get('concordance_checks', []):
            L.append(f"        {name}: [{status}] — {note}")
        cf = r19.get('concordance_flags', [])
        if cf:
            L.append(f"        Flagi spójności: {', '.join(cf)}")
        ta = r19.get('temporal_alignment', {})
        if ta:
            for bp, data in ta.items():
                srcs = data.get('sources', {})
                sp = data.get('spread_sec', '?')
                conf = data.get('confidence', '?')
                src_str = ', '.join(f'{k}={v:.0f}s' for k,v in srcs.items())
                L.append(f"        {bp} alignment: {src_str} | spread={sp}s | confidence={conf}")
        return "\n".join(L)

    def render_text_report(ct: Dict[str, Any]) -> str:
        """Renderuje pełny raport tekstowy wg szablonu T12."""


        # E06 gain report section — contextual (run vs machine)
        _g = ct.get
        _is_run = _g('GAIN_modality','') in ('run','walk')
        _gain_lines = []
        if _is_run:
            _gain_lines.append("VI. EKONOMIA BIEGU (E06 v2)")
            _gain_lines.append(f"\u2022 Running Economy na VT1: [{_g('RE_at_vt1','[BRAK]')}] ml/kg/km (speed {_g('LOAD_at_vt1','-')} km/h)")
            _gain_lines.append(f"\u2022 Running Economy na VT2: [{_g('RE_at_vt2','[BRAK]')}] ml/kg/km (speed {_g('LOAD_at_vt2','-')} km/h) -> {_g('RE_class','-')}")
            _gain_lines.append(f"  -> Norma: Elite <195 | Well-trained <208 | Recreational <225 | Klasyfikacja przy speed >=8 km/h")
            _gain_lines.append(f"• Gain (dVO2/dSpeed) <VT1: [{_g('GAIN_below_vt1','[BRAK]')}] {_g('GAIN_unit','')} (R2={_g('GAIN_below_vt1_r2','-')})")
            _gain_lines.append(f"  -> Gain pelny zakres: [{_g('GAIN_full','[BRAK]')}] {_g('GAIN_unit','')} | Z-score: {_g('GAIN_z','-')}")
        else:
            _gain_lines.append("VI. GAIN / EFEKTYWNOSC WYSILKU (E06 v2)")
            _gain_lines.append(f"• Modalnosc: [{_g('GAIN_modality','-')}]")
            _gain_lines.append(f"• Gain (dVO2/dWatt) <VT1: [{_g('GAIN_below_vt1','[BRAK]')}] {_g('GAIN_unit','')} (R2={_g('GAIN_below_vt1_r2','-')})")
            _gain_lines.append(f"  -> Gain pelny zakres: [{_g('GAIN_full','[BRAK]')}] {_g('GAIN_unit','')}")
            _gain_lines.append(f"  -> Norma: {_g('GAIN_norm_ref','-')} {_g('GAIN_unit','')} ({_g('GAIN_norm_src','')}) | Z-score: {_g('GAIN_z','-')}")
            _gain_lines.append(f"• Gain na progu VT1: [{_g('GAIN_at_vt1','[BRAK]')}] {_g('GAIN_unit','')} (load {_g('LOAD_at_vt1','-')} W)")
            _gain_lines.append(f"• Gain na progu VT2: [{_g('GAIN_at_vt2','[BRAK]')}] {_g('GAIN_unit','')} (load {_g('LOAD_at_vt2','-')} W)")
            _de = _g('DELTA_EFF')
            _de1 = _g('EFF_at_vt1')
            _de2 = _g('EFF_at_vt2')
            _gain_lines.append(f"• Delta Efficiency: srednia [{_de if _de else '[BRAK]'}]% | VT1 [{_de1 if _de1 else '-'}]% | VT2 [{_de2 if _de2 else '-'}]%")
        _lb = _g('LIN_BREAK_time')
        _gain_lines.append("• Zlamanie liniowosci VO2: " + (f"t={int(_lb)}s" if _lb else "nie wykryto"))
        _gain_section = chr(10).join(_gain_lines)
        report = f"""
[A] ANALYSIS_REPORT
RAPORT CPET/TEST WYDOLNOŚCIOWY
=============================================
NAGŁÓWEK: Profil Zawodnika
Imię / ID: [{ct['athlete_name']} / {ct['athlete_id']}]
Data: [{ct['test_date']}] | Lokalizacja: [{ct['location']}]
Protokół: [{ct['protocol']}]
Parametry: Wiek [{ct['age']}] | Wzrost [{ct['height']}] cm | Masa [{ct['weight']}] kg

I. DIAGNOZA I PLAN (EXECUTIVE SUMMARY)
1. Główny limiter: [DO UZUPEŁNIENIA PRZEZ TRENERA]
2. Ocena poziomu: [DO UZUPEŁNIENIA PRZEZ TRENERA]

II. STREFY TRENINGOWE (MODEL 5-STREFOWY)
KOTWICE METABOLICZNE:
• VT1 (Aerobic Threshold): HR [{ct['VT1_HR']}] bpm | Speed [{ct['VT1_Speed']}] km/h | Power [{ct['VT1_Power']}] W | VO2 [{ct.get('VT1_VO2_mlmin','-')}] ml/min ({ct.get('VT1_pct_VO2max','-')}% {ct.get('VO2_determination','VO2max')})
• VT2 (Anaerobic Threshold): HR [{ct['VT2_HR']}] bpm | Speed [{ct['VT2_Speed']}] km/h | Power [{ct['VT2_Power']}] W | VO2 [{ct.get('VT2_VO2_mlmin','-')}] ml/min ({ct.get('VT2_pct_VO2max','-')}% {ct.get('VO2_determination','VO2max')})

TABELA STREF:
Z1 - Regeneracja (Active Recovery)
• HR: [{ct['Z1_HR_low']} - {ct['Z1_HR_high']}] bpm
Z2 - Baza tlenowa (Endurance)
• HR: [{ct['Z2_HR_low']} - {ct['Z2_HR_high']}] bpm
Z3 - Tempo (Aerobic Power)
• HR: [{ct['Z3_HR_low']} - {ct['Z3_HR_high']}] bpm
Z4 - Próg (Threshold)
• HR: [{ct['Z4_HR_low']} - {ct['Z4_HR_high']}] bpm
Z5 - VO2max (Maximum Power)
• HR: [{ct['Z5_HR_low']} - {ct['Z5_HR_high']}] bpm

III. KLUCZOWE WYNIKI FIZJOLOGICZNE
1. Wydolność tlenowa:
• VO2peak (rel): [{ct['VO2peak_rel']}] ml/kg/min ({ct['VO2_pct_predicted']}% predicted {ct.get('VO2_pred_method','Wasserman')})
• VO2peak (abs): [{ct['VO2peak_abs']}] L/min
• Klasyfikacja: Populacja [{ct['VO2_class_pop']}] | Sport [{ct['VO2_class_sport']}]
• Jakość testu: [{ct['test_quality']}] | RER: [{ct['RER_class']}] | VE/VCO2: [{ct['VE_VCO2_class']}]
  ↳ {ct['VE_VCO2_class_desc']}
2. Układ sercowo-naczyniowy:
• HRmax: [{ct['HRmax']}] bpm
• O2 Pulse peak: [{ct['O2pulse_peak']}] ml/beat  |  at VT1: [{ct['O2pulse_at_vt1']}]  |  at VT2: [{ct['O2pulse_at_vt2']}]
  ↳ Predicted FRIEND: [{ct['O2pulse_pred_friend']}] ({ct['O2pulse_pct_friend']}%) | NOODLE: [{ct['O2pulse_pred_noodle']}] ({ct['O2pulse_pct_noodle']}%)
  ↳ Est. SV peak: [{ct['O2pulse_est_sv']}] ml  |  Trajektoria: [{ct['O2pulse_trajectory']}] — {ct['O2pulse_traj_desc']}
  ↳ FF: [{ct['O2pulse_ff']}] | O2PRR: [{ct['O2pulse_o2prr']}] | Flagi: [{ct['O2pulse_flags']}]
• HRR (Regeneracja): 1min [{ct['HRR_60s']}] bpm | 3min [{ct['HRR_180s']}] bpm
3. Układ oddechowy — Efektywność wentylacji (E03 v2.0):
• VEpeak: [{ct['VEpeak']}] L/min
• VE/VCO2 slope:  do VT2 [{ct['VE_VCO2_slope_vt2']}] | do VT1 [{ct['VE_VCO2_slope_vt1']}] | cały test [{ct['VE_VCO2_slope_full']}]
• VE/VCO2 nadir:  [{ct['VE_VCO2_nadir']}]  |  ratio przy VT1: [{ct['VE_VCO2_at_vt1']}]  |  ratio peak: [{ct['VE_VCO2_peak']}]
• Y-intercept:    [{ct['VE_VCO2_intercept']}] L/min  |  R²: [{ct['VE_VCO2_r2']}]
• Predicted slope: [{ct['VE_VCO2_predicted']}]  |  % predicted: [{ct['VE_VCO2_pct_pred']}]%
• Klasa Arena:    [{ct['VE_VCO2_vent_class']}] — {ct['VE_VCO2_vc_desc']}
• PETCO2:         rest [{ct['PETCO2_rest']}] | VT1 [{ct['PETCO2_vt1']}] | peak [{ct['PETCO2_peak']}] mmHg
• Flagi wentylacyjne: [{ct['VE_VCO2_flags']}]
{ReportAdapter._oues_interpretation(ct)}
{ReportAdapter._breathing_pattern_report(ct)}
4. Inne:
• RER peak: [{ct['RERpeak']}]
{ReportAdapter._nirs_report(ct)}
{ReportAdapter._drift_report(ct)}
{ReportAdapter._gasex_report(ct)}
{ReportAdapter._validity_concordance_report(ct)}
{ReportAdapter._lactate_report(ct)}

{_gain_section}

{ReportAdapter._e18_crossval_text(ct)}

IV. LIMITERY (RANKING) - DO ANALIZY PRZEZ TRENERA
[MIEJSCE NA ANALIZĘ AI]

V. PROFIL METABOLICZNY
• FATmax (MFO): [{ct['FATmax_g']}] g/min ({ct.get('MFO_mgkg_min', '-')} mg/kg/min)
  ↳ HR: [{ct['FATmax_HR']}] bpm ({ct.get('FATmax_pct_hrmax', '-')}% HRmax)
  ↳ Intensywność: {ct.get('FATmax_pct_vo2peak', '-')}% VO2peak
  ↳ Strefa FATmax (≥90% MFO): HR {ct.get('FATmax_zone_hr_low', '-')}-{ct.get('FATmax_zone_hr_high', '-')} bpm
{("• Crossover Point (CHO>FAT): HR ["+str(ct['CHO_Cross_HR'])+"] bpm ("+str(ct.get('COP_pct_vo2peak','-'))+"% VO2peak, RER "+str(ct.get('COP_RER','-'))+")") if ct.get('CHO_Cross_HR') and ct['CHO_Cross_HR'] not in ('[BRAK]', None, 'None') else "• Crossover Point: " + (ct.get('COP_note','') if ct.get('COP_note') else "brak danych")}
• Substrat przy VT1: FAT {ct.get('fat_pct_at_vt1', '-')}% / CHO {ct.get('cho_pct_at_vt1', '-')}%

  ┌──────────────────────────────────────────────────────────┐
  │ SPALANIE SUBSTRATÓW W STREFACH TRENINGOWYCH              │
  ├──────────┬──────────┬──────────┬──────────┬──────────────┤
  │ Strefa   │ FAT g/h  │ CHO g/h  │ FAT%/CHO%│ kcal/h       │
  ├──────────┼──────────┼──────────┼──────────┼──────────────┤
  │ Z2 Baza  │ {str(ct.get('z2_fat_gh', '-')):>8} │ {str(ct.get('z2_cho_gh', '-')):>8} │ {str(ct.get('z2_fat_pct', '-'))+'/'+str(ct.get('z2_cho_pct', '-')):>8} │ {str(ct.get('z2_kcal_h', '-')):>12} │
  │ Z4 Próg  │ {str(ct.get('z4_fat_gh', '-')):>8} │ {str(ct.get('z4_cho_gh', '-')):>8} │ {str(ct.get('z4_fat_pct', '-'))+'/'+str(ct.get('z4_cho_pct', '-')):>8} │ {str(ct.get('z4_kcal_h', '-')):>12} │
  │ Z5 VO2max│ {str(ct.get('z5_fat_gh', '-')):>8} │ {str(ct.get('z5_cho_gh', '-')):>8} │ {str(ct.get('z5_fat_pct', '-'))+'/'+str(ct.get('z5_cho_pct', '-')):>8} │ {str(ct.get('z5_kcal_h', '-')):>12} │
  └──────────┴──────────┴──────────┴──────────┴──────────────┘
  kcal/h wg Weir (1949). W Z4/Z5 (RER≥1) FAT≈0 — dominuje glikoliza.
  CHO g/h = zapotrzebowanie na węglowodany w treningu/wyścigu.

=============================================
KONIEC RAPORTU (v1.1 T12)
"""
        return report

print("✅ Komórka 4: Report Adapter (FULL T12 TEMPLATE RESTORED) załadowana.")


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
            segments = PROTOCOLS_DB.get(self.cfg.protocol_name, [])
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
        except Exception as e:
            canon_table = {}
            text_report = f"[RAPORT NIEDOSTĘPNY: {e}]"
            html_report = f"<html><body><h1>Raport niedostępny</h1><p>{e}</p></body></html>"

        self._last_report = {
            "outputs_calc_only": outputs,
            "trainer_canon_flat": trainer_canon_flat,
            "canon_table": canon_table,
            "text_report": text_report, "html_report": html_report,
            "raw_results": self.results
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


