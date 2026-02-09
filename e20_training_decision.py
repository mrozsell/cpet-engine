"""
Engine E20: Training Decision System
Evidence-based training prescription from CPET profile + athlete context.
Refs: Laursen & Buchheit 2013/2019, Seiler 2010, Oliveira 2024, Helgerud 2007
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math

# ═══════════════════════════════════════════════════════════════
# WARSTWA 1: TrainingProfile (input from UI)
# ═══════════════════════════════════════════════════════════════
@dataclass
class TrainingProfile:
    modality: str = "run"           # run/bike/tri/crossfit/hyrox/rowing/swim/xc_ski/soccer/mma
    training_hours_week: float = 6.0
    training_days_week: int = 4
    goal_type: str = "general"      # endurance/speed/threshold/vo2max/fatmax/race/general
    experience_years: float = 2.0
    goal_event: str = ""
    goal_date: str = ""
    long_session_max_min: int = 120
    injuries: str = ""

# ═══════════════════════════════════════════════════════════════
# WARSTWA 2: PhysioSnapshot (auto-extract from E00-E19)
# ═══════════════════════════════════════════════════════════════
@dataclass
class PhysioSnapshot:
    # Capacity
    vo2max_rel: float = 0.0
    vo2max_abs: float = 0.0
    hr_max: float = 0.0
    hr_rest: float = 60.0
    sport_class: str = "UNTRAINED"
    pop_class: str = ""
    # Thresholds
    vt1_pct_vo2: float = 0.0
    vt1_hr: float = 0.0
    vt1_speed: float = 0.0
    vt2_pct_vo2: float = 0.0
    vt2_hr: float = 0.0
    vt2_speed: float = 0.0
    threshold_gap_bpm: float = 0.0
    aerobic_reserve_pct: float = 0.0
    # Zones
    zones: dict = field(default_factory=dict)
    # Economy
    re_mlkgkm: float = 0.0
    re_class: str = ""
    ve_vco2_slope: float = 0.0
    vent_class: str = ""
    o2pulse_peak: float = 0.0
    o2pulse_trajectory: str = ""
    # Substrate
    fatmax_hr: float = 0.0
    fatmax_gmin: float = 0.0
    fatmax_pct_vo2: float = 0.0
    crossover_pct_vo2: float = 0.0
    # Recovery
    hrr_1min: float = 0.0
    hrr_3min: float = 0.0
    # Performance context
    level_by_speed: str = ""
    level_by_pct_vo2: str = ""
    economy_divergence: bool = False

    @classmethod
    def from_results(cls, r: dict, cfg=None) -> 'PhysioSnapshot':
        e01 = r.get('E01', {})
        e02 = r.get('E02', {})
        e03 = r.get('E03', {})
        e05 = r.get('E05', {})
        e06 = r.get('E06', {})
        e08 = r.get('E08', {})
        e10 = r.get('E10', {})
        e15 = r.get('E15', {})
        e16 = r.get('E16', {})
        pc  = r.get('_performance_context', {})

        lvl_speed = pc.get('level_by_speed', '')
        lvl_pct = pc.get('level_by_pct_vo2', '')
        LEVELS = {'Untrained':0,'Recreational':1,'Trained':2,'Competitive':3,'Elite':4}
        eco_div = LEVELS.get(lvl_pct,0) - LEVELS.get(lvl_speed,0) >= 2

        return cls(
            vo2max_rel=float(e01.get('vo2_peak_mlkgmin', 0) or 0),
            vo2max_abs=float(e01.get('vo2_peak_mlmin', 0) or 0),
            hr_max=float(e01.get('hr_peak', 0) or 0),
            hr_rest=float(e16.get('hr_rest', 60) or 60),
            sport_class=str(e15.get('vo2_class_sport', 'UNTRAINED')),
            pop_class=str(e15.get('vo2_class_pop', '')),
            vt1_pct_vo2=float(e02.get('vt1_vo2_pct_peak', 0) or 0),
            vt1_hr=float(e02.get('vt1_hr', 0) or 0),
            vt1_speed=float(e02.get('vt1_speed_kmh', 0) or 0),
            vt2_pct_vo2=float(e02.get('vt2_vo2_pct_peak', 0) or 0),
            vt2_hr=float(e02.get('vt2_hr', 0) or 0),
            vt2_speed=float(e02.get('vt2_speed_kmh', 0) or 0),
            threshold_gap_bpm=float(e16.get('threshold_gap_bpm', 0) or 0),
            aerobic_reserve_pct=float(e16.get('aerobic_reserve_pct', 0) or 0),
            zones=e16.get('zones', {}),
            re_mlkgkm=float(e06.get('running_economy_mlkgkm', 0) or 0) if isinstance(e06, dict) else 0,
            re_class=str(e06.get('re_classification', '')) if isinstance(e06, dict) else '',
            ve_vco2_slope=float(e03.get('slope_to_vt2', 0) or 0),
            vent_class=str(e03.get('ventilatory_class', '')),
            o2pulse_peak=float(e05.get('o2pulse_peak', 0) or 0) if isinstance(e05, dict) else 0,
            o2pulse_trajectory=str(e05.get('trajectory', '')) if isinstance(e05, dict) else '',
            fatmax_hr=float(e10.get('fatmax_hr', 0) or 0),
            fatmax_gmin=float(e10.get('mfo_gmin', 0) or 0),
            fatmax_pct_vo2=float(e10.get('fatmax_pct_vo2peak', 0) or 0),
            crossover_pct_vo2=float(e10.get('cop_pct_vo2peak', 0) or 0),
            hrr_1min=float(e08.get('hrr_1min', 0) or 0),
            hrr_3min=float(e08.get('hrr_3min', 0) or 0),
            level_by_speed=lvl_speed,
            level_by_pct_vo2=lvl_pct,
            economy_divergence=eco_div,
        )

# ═══════════════════════════════════════════════════════════════
# WARSTWA 3: Limiter Detection
# ═══════════════════════════════════════════════════════════════
@dataclass
class Limiter:
    name: str
    score: float       # 0-100
    reason: str
    method: str
    zone_focus: dict    # {"Z1":x, "Z2":x, ...}
    priority: int = 0

SPORT_CLASS_RANK = {"UNTRAINED":0,"RECREATIONAL":1,"TRAINED":2,"COMPETITIVE":3,"ELITE":4}
SPEED_LEVEL_RANK = {"Untrained":0,"Recreational":1,"Trained":2,"Competitive":3,"Elite":4}

def score_limiters(snap: PhysioSnapshot, profile: TrainingProfile) -> List[Limiter]:
    limiters = []
    sc = SPORT_CLASS_RANK.get(snap.sport_class, 1)

    # R1: LOW_BASE
    if snap.vt1_pct_vo2 > 0 and snap.vt1_pct_vo2 < 55:
        score = min(100, (55 - snap.vt1_pct_vo2) * 3)
        limiters.append(Limiter(
            "LOW_BASE", score,
            f"VT1 przy {snap.vt1_pct_vo2:.0f}% VO₂max (< 55%) → niska baza tlenowa",
            "Maffetone / Polarized LIT-dominant (Seiler 2010)",
            {"Z1":25,"Z2":60,"Z3":10,"Z4":3,"Z5":2}
        ))

    # R2: HIGH_BASE_LOW_THRESHOLD
    if snap.vt1_pct_vo2 >= 58 and snap.vt2_pct_vo2 > 0 and snap.vt2_pct_vo2 < 82:
        score = min(100, (82 - snap.vt2_pct_vo2) * 3)
        limiters.append(Limiter(
            "HIGH_BASE_LOW_THRESHOLD", score,
            f"VT1 OK ({snap.vt1_pct_vo2:.0f}%) ale VT2 niski ({snap.vt2_pct_vo2:.0f}%) → próg do podniesienia",
            "Threshold / Sweet Spot (Billat, Seiler)",
            {"Z1":20,"Z2":50,"Z3":5,"Z4":17,"Z5":8}
        ))

    # R3: HIGH_THRESHOLDS_LOW_VO2MAX
    if snap.vt2_pct_vo2 >= 85 and sc <= 1:
        score = min(100, 40 + (90 - snap.vo2max_rel) * 1.5) if snap.vo2max_rel > 0 else 40
        limiters.append(Limiter(
            "HIGH_THRESHOLDS_LOW_CEILING", max(0, score),
            f"VT2 wysoko ({snap.vt2_pct_vo2:.0f}%) ale klasa {snap.sport_class} → sufit VO₂max do podniesienia",
            "HIIT 4×4 min (Helgerud 2007, Laursen & Buchheit)",
            {"Z1":20,"Z2":55,"Z3":5,"Z4":8,"Z5":12}
        ))

    # R4: ECONOMY_LIMITER
    if snap.economy_divergence:
        score = 78
        limiters.append(Limiter(
            "ECONOMY_LIMITER", score,
            f"VO₂max → {snap.level_by_pct_vo2} ale prędkość → {snap.level_by_speed} → niska ekonomia biegu",
            "Siła + plyometria + strides (Llanos-Lagos 2024, Eihara 2022)",
            {"Z1":25,"Z2":50,"Z3":10,"Z4":8,"Z5":7}
        ))

    # R5: SUBSTRATE_LIMITER
    if snap.crossover_pct_vo2 > 0 and snap.vt1_pct_vo2 > 0:
        if snap.crossover_pct_vo2 < snap.vt1_pct_vo2 - 15:
            score = min(100, (snap.vt1_pct_vo2 - snap.crossover_pct_vo2 - 15) * 3)
            limiters.append(Limiter(
                "SUBSTRATE_LIMITER", max(20, score),
                f"Crossover przy {snap.crossover_pct_vo2:.0f}% VO₂ (VT1 przy {snap.vt1_pct_vo2:.0f}%) → wczesna zależność od glikogenu",
                "FATmax training, fasted Z2 sessions (Jeukendrup 2004, Venables 2008)",
                {"Z1":30,"Z2":55,"Z3":10,"Z4":3,"Z5":2}
            ))

    # R6: VENTILATORY_LIMITER
    if snap.ve_vco2_slope > 34:
        score = min(100, (snap.ve_vco2_slope - 34) * 10)
        vc_num = int(snap.vent_class.replace('VC-','').replace('I','1').replace('V','5')) if snap.vent_class else 1
        limiters.append(Limiter(
            "VENTILATORY_LIMITER", max(30, score),
            f"VE/VCO₂ slope {snap.ve_vco2_slope:.1f} ({snap.vent_class}) → ograniczenie wentylacyjne",
            "IMT (inspiratory muscle training), breathing drills",
            {"Z1":25,"Z2":55,"Z3":10,"Z4":5,"Z5":5}
        ))

    # R7: CARDIAC_LIMITER
    if snap.o2pulse_trajectory in ('PLATEAU','DECLINING') and snap.hrr_1min < 12:
        score = 70
        limiters.append(Limiter(
            "CARDIAC_LIMITER", score,
            f"O₂ pulse: {snap.o2pulse_trajectory} + HRR 1min: {snap.hrr_1min:.0f} bpm → ograniczenie kardiologiczne",
            "Dłuższe sesje Z2, Norwegian 4×4 (Helgerud 2007)",
            {"Z1":25,"Z2":55,"Z3":5,"Z4":8,"Z5":7}
        ))

    # R8: RECOVERY_LIMITER
    if snap.hrr_1min > 0 and snap.hrr_1min < 18 and sc >= 2:
        score = min(100, (18 - snap.hrr_1min) * 8)
        limiters.append(Limiter(
            "RECOVERY_LIMITER", max(25, score),
            f"HRR 1min: {snap.hrr_1min:.0f} bpm (<18) przy klasie {snap.sport_class} → słaba restytucja autonomiczna",
            "HRV-guided training, więcej Z1, mniej objętości HIT",
            {"Z1":35,"Z2":50,"Z3":10,"Z4":3,"Z5":2}
        ))

    # R9: ALL_HIGH (advanced, no clear limiter)
    if snap.vt1_pct_vo2 >= 62 and snap.vt2_pct_vo2 >= 88 and sc >= 3:
        score = 50
        limiters.append(Limiter(
            "RACE_SPECIFIC", score,
            f"Profil zaawansowany (VT1 {snap.vt1_pct_vo2:.0f}%, VT2 {snap.vt2_pct_vo2:.0f}%, {snap.sport_class}) → trening specyficzny",
            "Block periodization, race-pace sessions (Laursen & Buchheit Ch.10)",
            {"Z1":15,"Z2":45,"Z3":10,"Z4":18,"Z5":12}
        ))

    # Apply goal_type weighting
    GOAL_BOOST = {
        "fatmax":   "SUBSTRATE_LIMITER",
        "speed":    "ECONOMY_LIMITER",
        "race":     "RACE_SPECIFIC",
        "vo2max":   "HIGH_THRESHOLDS_LOW_CEILING",
        "threshold":"HIGH_BASE_LOW_THRESHOLD",
        "endurance":"LOW_BASE",
    }
    boost_name = GOAL_BOOST.get(profile.goal_type, "")
    for lim in limiters:
        if lim.name == boost_name:
            lim.score = min(100, lim.score * 1.5)

    limiters.sort(key=lambda x: x.score, reverse=True)
    for i, lim in enumerate(limiters):
        lim.priority = i + 1
    return limiters[:3]

# ═══════════════════════════════════════════════════════════════
# WARSTWA 4: Plan Generation
# ═══════════════════════════════════════════════════════════════
WEEK_TEMPLATES = {
    3: [("Z2 easy",60), ("KEY_1",50), ("Z1-Z2 long",75)],
    4: [("Z2 easy",60), ("KEY_1",65), ("Z2 easy",50), ("Z1-Z2 long",90)],
    5: [("Z2 easy",60), ("KEY_1",65), ("REST",0), ("KEY_2",55), ("Z1-Z2 long",100)],
    6: [("Z2 easy",60), ("KEY_1",70), ("Z2+drills",60), ("REST",0), ("KEY_2",55), ("Z1-Z2 long",110)],
    7: [("Z2 easy",45), ("KEY_1",70), ("Z2+drills",60), ("REST",0), ("KEY_2",55), ("Z1-Z2 long",120), ("Z1 recovery",30)],
}

def _format_pace(speed_kmh):
    if not speed_kmh or speed_kmh <= 0: return ""
    pace_min = 60 / speed_kmh
    m = int(pace_min)
    s = int((pace_min - m) * 60)
    return f"{m}:{s:02d}/km"

def _scale_session(template_name, snap, sport_class_rank, lim_name):
    """Generate specific key session based on limiter and athlete level."""
    sc = sport_class_rank
    z = snap.zones
    z4 = z.get('z4', {})
    z5 = z.get('z5', {})
    z2 = z.get('z2', {})

    if "KEY_1" in template_name:
        if lim_name in ("HIGH_BASE_LOW_THRESHOLD", "RACE_SPECIFIC"):
            work = {0:"2×8 min",1:"3×8 min",2:"3×10 min",3:"3×12 min",4:"4×12 min"}.get(sc,"3×10 min")
            return f"Threshold cruise: {work} @ HR {z4.get('hr_low',0)}-{z4.get('hr_high',0)} (Z4, {_format_pace(z4.get('speed_high',0))}), rest 3 min Z1"
        elif lim_name == "HIGH_THRESHOLDS_LOW_CEILING":
            work = {0:"3×3 min",1:"4×3 min",2:"4×4 min",3:"5×4 min",4:"6×4 min"}.get(sc,"4×4 min")
            return f"VO₂max intervals: {work} @ HR {z5.get('hr_low',0)}-{z5.get('hr_high',0)} (Z5, {_format_pace(z5.get('speed_high',0))}), rest 3 min Z1"
        else:  # default: threshold
            work = {0:"2×8 min",1:"3×8 min",2:"3×10 min",3:"3×12 min",4:"4×12 min"}.get(sc,"3×10 min")
            return f"Threshold: {work} @ HR {z4.get('hr_low',0)}-{z4.get('hr_high',0)} (Z4), rest 3 min Z1"

    if "KEY_2" in template_name:
        if lim_name in ("ECONOMY_LIMITER",):
            return f"Drills + strides: rozgrzewka Z1 15 min + drills (A-skip, B-skip) + 6-8× strides 80m + Z2 30 min"
        else:
            work = {0:"3×3 min",1:"4×3 min",2:"4×4 min",3:"5×4 min",4:"6×4 min"}.get(sc,"4×4 min")
            return f"VO₂max: {work} @ HR {z5.get('hr_low',0)}-{z5.get('hr_high',0)} (Z5), rest 3 min Z1"

    return template_name

def generate_plan(snap: PhysioSnapshot, profile: TrainingProfile, limiters: List[Limiter]) -> dict:
    """Generate weekly training plan from physio snapshot + limiters."""
    days = min(max(profile.training_days_week, 3), 7)
    template = WEEK_TEMPLATES.get(days, WEEK_TEMPLATES[5])
    top_limiter = limiters[0] if limiters else Limiter("GENERAL",50,"","",{"Z1":20,"Z2":55,"Z3":10,"Z4":10,"Z5":5})
    sc = SPORT_CLASS_RANK.get(snap.sport_class, 1)
    z = snap.zones
    z2 = z.get('z2', {})

    # Build week
    DAY_NAMES = ["Pon","Wt","Śr","Czw","Pt","Sob","Ndz"]
    week = []
    for i, (stype, dur) in enumerate(template):
        day_name = DAY_NAMES[i % 7]
        if stype == "REST":
            week.append({"day": day_name, "type": "REST", "zone": "-", "duration_min": 0, "description": "Odpoczynek", "hr_target": "-"})
        elif "KEY" in stype:
            desc = _scale_session(stype, snap, sc, top_limiter.name)
            zone = "Z4" if "Threshold" in desc or "cruise" in desc else "Z5"
            week.append({"day": day_name, "type": "Key Session", "zone": zone, "duration_min": dur, "description": desc, "hr_target": ""})
        elif "long" in stype.lower():
            fat_note = f" (FATmax zone: HR ~{snap.fatmax_hr:.0f}, ~{snap.fatmax_gmin*60:.0f}g fat/h)" if snap.fatmax_hr > 0 else ""
            week.append({"day": day_name, "type": "Long Run", "zone": "Z1-Z2", "duration_min": dur,
                "description": f"Bieg długi Z2{fat_note}", "hr_target": f"HR {z2.get('hr_low',0)}-{z2.get('hr_high',0)}"})
        elif "drills" in stype.lower():
            week.append({"day": day_name, "type": "Aerobic + Tech", "zone": "Z2", "duration_min": dur,
                "description": f"Z2 easy + drills techniczne + 6× strides 80m", "hr_target": f"HR {z2.get('hr_low',0)}-{z2.get('hr_high',0)}"})
        elif "recovery" in stype.lower():
            z1 = z.get('z1', {})
            week.append({"day": day_name, "type": "Recovery", "zone": "Z1", "duration_min": dur,
                "description": f"Regeneracja aktywna", "hr_target": f"HR < {z1.get('hr_high',0)}"})
        else:
            week.append({"day": day_name, "type": "Easy Aerobic", "zone": "Z2", "duration_min": dur,
                "description": f"Bieg łatwy Z2", "hr_target": f"HR {z2.get('hr_low',0)}-{z2.get('hr_high',0)}"})

    # Nutrition per session type
    nutrition = {}
    nutrition["Z2_easy"] = {"pre": "Dowolny posiłek lub na czczo", "during": "Woda", "post": "Normalny posiłek"}
    nutrition["Z2_long"] = {"pre": "Lekki posiłek 2h przed LUB na czczo (opcja FATmax)",
        "during": "30-40g CHO/h po 60 min", "post": "CHO + białko w 30 min"}
    nutrition["Z4_threshold"] = {"pre": "30-50g CHO 1-2h przed", "during": "30g CHO/h jeśli >60 min",
        "post": "20-30g białka + 50-80g CHO w 30 min"}
    nutrition["Z5_vo2max"] = {"pre": "30-50g CHO 1-2h przed", "during": "Woda (sesja krótka)",
        "post": "20-30g białka + CHO w 30 min"}
    if snap.fatmax_gmin > 0:
        nutrition["FATmax_dedicated"] = {
            "pre": "Na czczo LUB bardzo nisko-CHO", "during": "Woda/elektrolity",
            "post": "Normalny posiłek",
            "note": f"FATmax = {snap.fatmax_gmin:.2f} g/min → ~{snap.fatmax_gmin*60:.0f}g tłuszczu/h. Max 2×/tyg."}

    # Monitoring
    monitoring = [
        "Co tydzień: resting HR rano (trend — wzrost >5 bpm = zmęczenie)",
        "Co tydzień: RPE po sesjach kluczowych (7-8/10 = OK, 9-10 = za dużo)",
        f"Co 2 tyg: HR drift test — 20 min @ {snap.vt1_speed:.1f} km/h → cardiac drift <5% = OK",
        "Co 4 tyg: time trial lub test progowy → HR at fixed pace",
        "Co 8-12 tyg: retest CPET (pełna ponowna analiza → nowe strefy, nowy plan)",
    ]

    # Progression
    progression = {
        "block_weeks": 4,
        "week_1": {"volume_pct":100, "intensity_pct":80, "label":"BASE"},
        "week_2": {"volume_pct":110, "intensity_pct":90, "label":"BUILD"},
        "week_3": {"volume_pct":115, "intensity_pct":100, "label":"PEAK"},
        "week_4": {"volume_pct":60, "intensity_pct":70, "label":"RECOVERY"},
        "between_blocks": "+5-10% objętości bazowej na blok (max 4 bloki przed retestem CPET)",
    }

    # Determine philosophy name
    philosophy_map = {
        "LOW_BASE": "Polarized LIT-dominant (Maffetone / Seiler)",
        "HIGH_BASE_LOW_THRESHOLD": "Threshold emphasis (Sweet Spot / Tempo)",
        "HIGH_THRESHOLDS_LOW_CEILING": "HIIT-focused (Helgerud 4×4 / Buchheit intervals)",
        "ECONOMY_LIMITER": "Polarized + tech sessions (plyometria/siła)",
        "SUBSTRATE_LIMITER": "Ultra-LIT + FATmax (Jeukendrup / metabolic adaptation)",
        "VENTILATORY_LIMITER": "Aerobic base + IMT (inspiratory muscle training)",
        "CARDIAC_LIMITER": "Cardiac development (long Z2 + Norwegian 4×4)",
        "RECOVERY_LIMITER": "Autonomic recovery-first (HRV-guided)",
        "RACE_SPECIFIC": "Block periodization (race-pace sessions)",
    }
    philosophy = philosophy_map.get(top_limiter.name, "General endurance development")

    return {
        "status": "OK",
        "philosophy": philosophy,
        "limiters": [{"name":l.name,"score":round(l.score,1),"reason":l.reason,"method":l.method,
                      "zone_focus":l.zone_focus,"priority":l.priority} for l in limiters],
        "zone_distribution": top_limiter.zone_focus,
        "weekly_volume_hours": profile.training_hours_week,
        "weekly_sessions": days,
        "week_template": week,
        "nutrition": nutrition,
        "progression": progression,
        "monitoring": monitoring,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE FUNCTION
# ═══════════════════════════════════════════════════════════════
def run_e20(results: dict, training_profile: TrainingProfile) -> dict:
    """Main entry point. Takes E00-E19 results + TrainingProfile → plan."""
    snap = PhysioSnapshot.from_results(results)
    limiters = score_limiters(snap, training_profile)
    plan = generate_plan(snap, training_profile, limiters)
    plan["physio_snapshot"] = {
        "vo2max": round(snap.vo2max_rel, 1),
        "hr_max": round(snap.hr_max),
        "vt1_pct": round(snap.vt1_pct_vo2, 1),
        "vt2_pct": round(snap.vt2_pct_vo2, 1),
        "vt1_hr": round(snap.vt1_hr),
        "vt2_hr": round(snap.vt2_hr),
        "sport_class": snap.sport_class,
        "fatmax_gmin": round(snap.fatmax_gmin, 2),
        "fatmax_hr": round(snap.fatmax_hr),
        "hrr_1min": round(snap.hrr_1min, 1),
        "re_class": snap.re_class,
        "economy_divergence": snap.economy_divergence,
    }
    return plan
