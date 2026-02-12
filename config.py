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
    protocol_name: str = "AUTO"
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

    # --- KINETYKA VO₂ (CWR protocol) ---
    kinetics_speeds_kmh: Optional[List[float]] = None   # 4 prędkości CWR [km/h]

    # --- GAS CALIBRATION MANUAL ---
    gc_manual: Optional[str] = None       # Manual gas calibration text

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


# --- PROTOKOŁY: importowane z engine_core (single source of truth) ---
try:
    from engine_core import RAW_PROTOCOLS
except ImportError:
    RAW_PROTOCOLS = {}
