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
