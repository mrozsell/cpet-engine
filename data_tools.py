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