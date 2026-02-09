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

# ═══════════════════════════════════════════════════════════════
# SPORT-SPECIFIC LIMITER THRESHOLDS
# Evidence-based per-sport calibration
# Sources: Seiler 2010, Haugan 2024, Faria 2005, Millet 2003,
#          Buchheit & Laursen 2013, Jeukendrup 2004, AIS 2023,
#          Helgerud 2007, Oliveira 2024, Steinacker 1986
# ═══════════════════════════════════════════════════════════════
SPORT_THRESHOLDS = {
    # vt1_low: below this VT1% → LOW_BASE limiter fires
    # vt1_ok:  above this VT1% → base is OK, check threshold
    # vt2_low: below this VT2% → threshold limiter fires
    # vt2_high: above this & low class → ceiling limiter fires
    # ve_thr:  VE/VCO₂ slope above this → ventilatory limiter
    # hrr1_thr: HRR1 below this (at sport_class≥TRAINED) → recovery limiter
    # cop_gap: crossover must be this far below VT1 → substrate limiter
    # econ_w:  economy limiter weight (how much economy matters for this sport)
    # rc_vt1/rc_vt2: thresholds for RACE_SPECIFIC rule
    'run':       {'vt1_low':55, 'vt1_ok':58, 'vt2_low':80, 'vt2_high':85, 've_thr':34, 'hrr1_thr':18, 'cop_gap':15, 'econ_w':1.0, 'rc_vt1':62, 'rc_vt2':88},
    'bike':      {'vt1_low':52, 'vt1_ok':56, 'vt2_low':78, 'vt2_high':84, 've_thr':32, 'hrr1_thr':16, 'cop_gap':15, 'econ_w':0.8, 'rc_vt1':60, 'rc_vt2':86},
    'triathlon': {'vt1_low':55, 'vt1_ok':58, 'vt2_low':80, 'vt2_high':86, 've_thr':32, 'hrr1_thr':18, 'cop_gap':15, 'econ_w':0.9, 'rc_vt1':62, 'rc_vt2':88},
    'rowing':    {'vt1_low':53, 'vt1_ok':57, 'vt2_low':78, 'vt2_high':84, 've_thr':33, 'hrr1_thr':16, 'cop_gap':15, 'econ_w':0.8, 'rc_vt1':60, 'rc_vt2':86},
    'crossfit':  {'vt1_low':48, 'vt1_ok':52, 'vt2_low':72, 'vt2_high':80, 've_thr':36, 'hrr1_thr':14, 'cop_gap':20, 'econ_w':0.6, 'rc_vt1':56, 'rc_vt2':82},
    'hyrox':     {'vt1_low':50, 'vt1_ok':54, 'vt2_low':75, 'vt2_high':82, 've_thr':35, 'hrr1_thr':15, 'cop_gap':18, 'econ_w':0.7, 'rc_vt1':58, 'rc_vt2':84},
    'swimming':  {'vt1_low':50, 'vt1_ok':54, 'vt2_low':76, 'vt2_high':84, 've_thr':36, 'hrr1_thr':15, 'cop_gap':18, 'econ_w':0.9, 'rc_vt1':58, 'rc_vt2':85},
    'xc_ski':    {'vt1_low':58, 'vt1_ok':62, 'vt2_low':84, 'vt2_high':90, 've_thr':30, 'hrr1_thr':20, 'cop_gap':12, 'econ_w':0.8, 'rc_vt1':65, 'rc_vt2':92},
    'soccer':    {'vt1_low':52, 'vt1_ok':56, 'vt2_low':78, 'vt2_high':85, 've_thr':34, 'hrr1_thr':16, 'cop_gap':18, 'econ_w':0.5, 'rc_vt1':58, 'rc_vt2':85},
    'mma':       {'vt1_low':48, 'vt1_ok':52, 'vt2_low':72, 'vt2_high':80, 've_thr':36, 'hrr1_thr':14, 'cop_gap':20, 'econ_w':0.5, 'rc_vt1':55, 'rc_vt2':80},
}

def score_limiters(snap: PhysioSnapshot, profile: TrainingProfile) -> List[Limiter]:
    """Sport-aware limiter detection with per-sport thresholds."""
    limiters = []
    sc = SPORT_CLASS_RANK.get(snap.sport_class, 1)
    mod = getattr(profile, 'modality', 'run') or 'run'
    # Normalize aliases
    _alias = {'treadmill':'run','walk':'run','wattbike':'bike','echobike':'bike','swim':'swimming','ergometer':'rowing'}
    mod = _alias.get(mod, mod)
    T = SPORT_THRESHOLDS.get(mod, SPORT_THRESHOLDS.get('run'))

    # R1: LOW_BASE — VT1 below sport-specific minimum
    if snap.vt1_pct_vo2 > 0 and snap.vt1_pct_vo2 < T['vt1_low']:
        score = min(100, (T['vt1_low'] - snap.vt1_pct_vo2) * 3)
        limiters.append(Limiter(
            "LOW_BASE", score,
            f"VT1 przy {snap.vt1_pct_vo2:.0f}% VO₂max (< {T['vt1_low']}% dla {mod}) → niska baza tlenowa",
            "Maffetone / Polarized LIT-dominant (Seiler 2010)",
            {"Z1":25,"Z2":60,"Z3":10,"Z4":3,"Z5":2}
        ))

    # R2: HIGH_BASE_LOW_THRESHOLD — good VT1 but VT2 below sport-specific threshold
    if snap.vt1_pct_vo2 >= T['vt1_ok'] and snap.vt2_pct_vo2 > 0 and snap.vt2_pct_vo2 < T['vt2_low']:
        score = min(100, (T['vt2_low'] - snap.vt2_pct_vo2) * 3)
        limiters.append(Limiter(
            "HIGH_BASE_LOW_THRESHOLD", score,
            f"VT1 OK ({snap.vt1_pct_vo2:.0f}%) ale VT2 niski ({snap.vt2_pct_vo2:.0f}% < {T['vt2_low']}% dla {mod}) → próg do podniesienia",
            "Threshold / Sweet Spot (Billat, Seiler)",
            {"Z1":20,"Z2":50,"Z3":5,"Z4":17,"Z5":8}
        ))

    # R3: HIGH_THRESHOLDS_LOW_CEILING — VT2 high but sport class low → VO2max ceiling
    if snap.vt2_pct_vo2 >= T['vt2_high'] and sc <= 1:
        score = min(100, 40 + (90 - snap.vo2max_rel) * 1.5) if snap.vo2max_rel > 0 else 40
        limiters.append(Limiter(
            "HIGH_THRESHOLDS_LOW_CEILING", max(0, score),
            f"VT2 wysoko ({snap.vt2_pct_vo2:.0f}% ≥{T['vt2_high']}%) ale klasa {snap.sport_class} ({mod}) → sufit VO₂max do podniesienia",
            "HIIT 4×4 min (Helgerud 2007, Laursen & Buchheit)",
            {"Z1":20,"Z2":55,"Z3":5,"Z4":8,"Z5":12}
        ))

    # R4: ECONOMY_LIMITER — weighted by sport importance
    if snap.economy_divergence:
        base_score = 78
        score = base_score * T['econ_w']
        limiters.append(Limiter(
            "ECONOMY_LIMITER", score,
            f"VO₂max → {snap.level_by_pct_vo2} ale obciążenie → {snap.level_by_speed} → niska ekonomia ({mod})",
            f"Trening specyficzny ekonomii {mod} (Llanos-Lagos 2024, Eihara 2022)",
            {"Z1":25,"Z2":50,"Z3":10,"Z4":8,"Z5":7}
        ))

    # R5: SUBSTRATE_LIMITER — sport-specific crossover gap
    if snap.crossover_pct_vo2 > 0 and snap.vt1_pct_vo2 > 0:
        if snap.crossover_pct_vo2 < snap.vt1_pct_vo2 - T['cop_gap']:
            score = min(100, (snap.vt1_pct_vo2 - snap.crossover_pct_vo2 - T['cop_gap']) * 3)
            limiters.append(Limiter(
                "SUBSTRATE_LIMITER", max(20, score),
                f"Crossover przy {snap.crossover_pct_vo2:.0f}% VO₂ (VT1 {snap.vt1_pct_vo2:.0f}%, gap >{T['cop_gap']}% dla {mod}) → wczesna zależność od glikogenu",
                "FATmax training, fasted Z2 sessions (Jeukendrup 2004, Venables 2008)",
                {"Z1":30,"Z2":55,"Z3":10,"Z4":3,"Z5":2}
            ))

    # R6: VENTILATORY_LIMITER — sport-specific VE/VCO₂ threshold
    if snap.ve_vco2_slope > T['ve_thr']:
        score = min(100, (snap.ve_vco2_slope - T['ve_thr']) * 10)
        vc_num = int(snap.vent_class.replace('VC-','').replace('I','1').replace('V','5')) if snap.vent_class else 1
        limiters.append(Limiter(
            "VENTILATORY_LIMITER", max(30, score),
            f"VE/VCO₂ slope {snap.ve_vco2_slope:.1f} (>{T['ve_thr']} dla {mod}, {snap.vent_class}) → ograniczenie wentylacyjne",
            "IMT (inspiratory muscle training), breathing drills",
            {"Z1":25,"Z2":55,"Z3":10,"Z4":5,"Z5":5}
        ))

    # R7: CARDIAC_LIMITER — relaxed criteria (OR instead of AND)
    _cardiac_flags = 0
    if snap.o2pulse_trajectory in ('PLATEAU','DECLINING'): _cardiac_flags += 1
    if snap.hrr_1min > 0 and snap.hrr_1min < 12: _cardiac_flags += 1
    if _cardiac_flags >= 1:
        score = 50 + _cardiac_flags * 15
        limiters.append(Limiter(
            "CARDIAC_LIMITER", score,
            f"O₂ pulse: {snap.o2pulse_trajectory}" + (f", HRR 1min: {snap.hrr_1min:.0f}" if snap.hrr_1min > 0 else "") + f" → ograniczenie kardiologiczne ({mod})",
            "Dłuższe sesje Z2, Norwegian 4×4 (Helgerud 2007)",
            {"Z1":25,"Z2":55,"Z3":5,"Z4":8,"Z5":7}
        ))

    # R8: RECOVERY_LIMITER — sport-specific HRR threshold
    if snap.hrr_1min > 0 and snap.hrr_1min < T['hrr1_thr'] and sc >= 2:
        score = min(100, (T['hrr1_thr'] - snap.hrr_1min) * 8)
        limiters.append(Limiter(
            "RECOVERY_LIMITER", max(25, score),
            f"HRR 1min: {snap.hrr_1min:.0f} bpm (<{T['hrr1_thr']} dla {mod}) przy klasie {snap.sport_class} → słaba restytucja autonomiczna",
            "HRV-guided training, więcej Z1, mniej objętości HIT",
            {"Z1":35,"Z2":50,"Z3":10,"Z4":3,"Z5":2}
        ))

    # R9: RACE_SPECIFIC — sport-specific advanced profile
    if snap.vt1_pct_vo2 >= T['rc_vt1'] and snap.vt2_pct_vo2 >= T['rc_vt2'] and sc >= 3:
        score = 50
        limiters.append(Limiter(
            "RACE_SPECIFIC", score,
            f"Profil zaawansowany ({mod}: VT1 {snap.vt1_pct_vo2:.0f}%≥{T['rc_vt1']}, VT2 {snap.vt2_pct_vo2:.0f}%≥{T['rc_vt2']}, {snap.sport_class}) → trening specyficzny",
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

# ═══════════════════════════════════════════════════════════════
# SPORT-SPECIFIC SESSION DATABASE
# Evidence-based, per-modality, per-limiter training prescriptions
# ═══════════════════════════════════════════════════════════════

# Limiter types from scoring system:
# HIGH_BASE_LOW_THRESHOLD, HIGH_THRESHOLDS_LOW_CEILING, ECONOMY_LIMITER,
# LOW_BASE, SUBSTRATE_LIMITER, VENTILATORY_LIMITER, CARDIAC_LIMITER,
# RECOVERY_LIMITER, RACE_SPECIFIC

SPORT_SESSIONS = {
    # ──────────────────────────────────────────────────────
    # RUNNING
    # ──────────────────────────────────────────────────────
    'run': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold cruise: {_vol(sc,'2×8','3×8','3×10','3×12','4×12')} min @ HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')} ({_pace(z,'z4')}), rest 3 min jog Z1",
            'KEY_2': lambda sc,z: f"Tempo run: {_vol(sc,'15','20','25','30','35')} min ciągłego biegu @ HR {z.get('z3',{}).get('hr_high','?')}-{z.get('z4',{}).get('hr_low','?')} (pogranicze Z3/Z4)",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max intervals: {_vol(sc,'3×3','4×3','4×4','5×4','6×4')} min @ HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')} ({_pace(z,'z5')}), rest 3 min jog",
            'KEY_2': lambda sc,z: f"Hill repeats: {_vol(sc,'6×60s','8×60s','8×90s','10×90s','10×2min')} pod górę (6-8%), mocny wysiłek Z5, truchtem w dół",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Drills + strides: rozgrzewka Z1 15 min → A-skip, B-skip, high knees, butt kicks (3×30m każdy) → 6-8× strides 80m (95% sprint) → Z2 30 min",
            'KEY_2': lambda sc,z: f"Plyometria biegowa: rozgrzewka → 3×10 drop jumps, 3×8 skoki na skrzynię, 3×10 wypadów z wyskokiem, 3×20m sprint z oporem → Z2 20 min cool-down",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long Z2 run: {_vol(sc,'50','60','75','90','100')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — utrzymuj stały HR, nie przyspieszaj",
            'KEY_2': lambda sc,z: f"Easy aerobic: {_vol(sc,'40','45','50','55','60')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} + drills techniczne 10 min",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax long run: {_vol(sc,'60','70','80','90','100')} min na czczo @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — woda + elektrolity, bez żeli",
            'KEY_2': lambda sc,z: f"Low-glycogen Z2: wieczorem trening siłowy (deplecja glikogenu) → rano bieg {_vol(sc,'40','50','55','60','70')} min Z2 na czczo (sleep-low, train-low)",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + Z2: 30 obdechów na POWERbreathe (60-70% MIP) → bieg {_vol(sc,'40','50','55','60','70')} min Z2 z fokusem na oddychanie przeponowe (2-in, 3-out)",
            'KEY_2': lambda sc,z: f"Controlled breathing run: {_vol(sc,'30','35','40','45','50')} min Z2, reguła 3:2 (3 kroki wdech, 2 kroki wydech) + 5 min finisher Z3 z utrzymaniem wzorca",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Norwegian 4×4: 4×4 min @ HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')} ({_pace(z,'z5')}), 3 min aktywna pauza Z1 — cel: budowa objętości wyrzutowej",
            'KEY_2': lambda sc,z: f"Long steady Z2: {_vol(sc,'60','70','80','90','100')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — dłuższe sesje aerobowe budują serce sportowe",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery Z1 + oddychanie: 20-30 min spacer/trucht @ HR < {z.get('z1',{}).get('hr_high','?')} → 10 min box breathing (4-4-4-4) + foam rolling",
            'KEY_2': lambda sc,z: f"HRV-guided easy: {_vol(sc,'30','35','40','40','45')} min Z1-Z2 TYLKO jeśli poranny HRV w normie. Jeśli HRV nisko → yoga/stretching zamiast biegu",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race-pace blocks: 3×{_vol(sc,'8','10','12','15','15')} min @ tempo startowe (pogranicze Z3/Z4, HR {z.get('z4',{}).get('hr_low','?')}), 3 min Z1",
            'KEY_2': lambda sc,z: f"Negative split long run: {_vol(sc,'60','70','80','90','100')} min, pierwsza połowa Z2, druga połowa progresja do Z3 — symulacja wyścigu",
        },
    },

    # ──────────────────────────────────────────────────────
    # CYCLING
    # ──────────────────────────────────────────────────────
    'bike': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Sweet Spot intervals: {_vol(sc,'2×12','3×12','3×15','4×15','4×20')} min @ 88-94% FTP (HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')}), kadencja 85-95 rpm, rest 5 min Z1",
            'KEY_2': lambda sc,z: f"Over-Under: {_vol(sc,'3×8','4×8','4×10','5×10','5×12')} min (2 min over FTP + 2 min under FTP, powtarzaj), 4 min Z1 między seriami",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max ramp: {_vol(sc,'4×3','5×3','5×4','6×4','6×5')} min @ HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')} (100-120% FTP), kadencja >95 rpm, rest 3-4 min Z1",
            'KEY_2': lambda sc,z: f"Tabata-style: {_vol(sc,'2×4','3×4','3×5','4×5','4×6')} serii (20s max / 10s rest), 5 min Z1 między blokami — krótkie, brutalne bodźce VO₂max",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Kadencja + siła: rozgrzewka Z1 15 min → 6×2 min wysoka kadencja 100-110 rpm Z2 + 4×3 min SFR niska kadencja 55-65 rpm Z3 + Z2 20 min cool-down",
            'KEY_2': lambda sc,z: f"Single-leg drills: rozgrzewka → 6×2 min jednonóż (prawa/lewa) kadencja 80-90 rpm Z2 + 4× sprint z miejsca 15s max → Z2 20 min cool-down",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long Z2 ride: {_vol(sc,'75','90','105','120','150')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')}, kadencja 85-95 rpm — trzymaj HR w strefie, nie przyspieszaj na podjazdach",
            'KEY_2': lambda sc,z: f"Endurance ride: {_vol(sc,'60','70','80','90','100')} min Z2 z 4× lekkie podjazdy 3-5 min (HR do górnej granicy Z2) — budowa bazy aerobowej",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax ride: {_vol(sc,'75','90','105','120','150')} min na czczo @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — woda + elektrolity, bez żeli. Cel: adaptacja metaboliczna",
            'KEY_2': lambda sc,z: f"Sleep-low ride: wieczorem interwały Z4 (deplecja glikogenu) → rano jazda {_vol(sc,'60','70','80','90','100')} min Z2 na czczo — train-low strategy",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + endurance: 30 oddechów POWERbreathe (60-70% MIP) → jazda {_vol(sc,'60','70','80','90','100')} min Z2 z fokusem na synchronizację oddechu z kadencją",
            'KEY_2': lambda sc,z: f"Nasal breathing ride: {_vol(sc,'30','40','45','50','55')} min Z2 oddychając TYLKO nosem — wymusza efektywność wentylacyjną. Jeśli musisz otworzyć usta = zwolnij",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Norwegian bike 4×4: 4×4 min @ HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')} (100-110% FTP), 3 min Z1 — budowa objętości wyrzutowej",
            'KEY_2': lambda sc,z: f"Long tempo: {_vol(sc,'70','80','90','100','120')} min Z2 — dłuższe sesje aerobowe na rowerze budują serce sportowe efektywniej niż krótkie",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery spin: 25-35 min @ HR < {z.get('z1',{}).get('hr_high','?')}, kadencja 85-95 rpm, ZERO obciążenia → stretching dolnych kończyn 10 min",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'30','35','40','40','45')} min Z1 lekka jazda + mobility (hip openers, hamstrings) — TYLKO jeśli HRV w normie",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race simulation: {_vol(sc,'2×15','3×15','3×20','4×20','4×25')} min @ race pace (Z3-Z4), z sekcjami podjazdu (3-5 min Z5) — symulacja wyścigu",
            'KEY_2': lambda sc,z: f"TT effort: {_vol(sc,'15','20','25','30','30')} min time trial @ threshold (Z4) — mierz moc i HR, cel: najwyższa średnia moc",
        },
    },

    # ──────────────────────────────────────────────────────
    # CROSSFIT
    # ──────────────────────────────────────────────────────
    'crossfit': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold EMOM: 20-30 min EMOM — min 1: 12 cal assault bike Z4, min 2: 8 deadlifts @ 60% 1RM, min 3: 15 box jumps 50/60cm. HR target: {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')}",
            'KEY_2': lambda sc,z: f"Chipper @ threshold: 50 wall balls → 40 KB swings → 30 T2B → 20 cleans @ 60% → 10 muscle-ups. Tempo kontrolowane, HR w Z3-Z4",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max MetCon: 5 rund AFAP — 200m run + 10 thrusters 43/30kg + 10 pull-ups. Cel: max wysiłek, HR {z.get('z5',{}).get('hr_low','?')}+, pełna regeneracja 3 min między rundami",
            'KEY_2': lambda sc,z: f"Assault bike intervals: {_vol(sc,'4×30s','6×30s','6×40s','8×40s','8×45s')} ALL-OUT na assault bike, 2 min pauza Z1 — krótkie, brutalne bodźce VO₂max",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Skill + movement efficiency: 20 min EMOM — min 1: 3-5 bar muscle-ups (technika), min 2: 5 squat clean @ 70% (perfekcja ruchu), min 3: rest. Potem Z2 row/bike 20 min",
            'KEY_2': lambda sc,z: f"Gymnastics flow: 30 min praca nad kipping, butterfly pull-ups, T2B, HSPU — cel: efektywność ruchu. Zakończ 15 min Z2 assault bike",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long mono-structural: {_vol(sc,'35','40','50','55','60')} min steady @ HR Z2 — row/bike/ski erg (bez przerw). Cel: budowa bazy aerobowej, której CrossFitowi brakuje",
            'KEY_2': lambda sc,z: f"Z2 mixed: {_vol(sc,'30','35','40','45','50')} min — 10 min row + 10 min bike + 10 min ski erg, wszystko w Z2. Ciągły wysiłek aerobowy",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax session: {_vol(sc,'50','55','60','70','75')} min row/bike na czczo @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — woda + elektrolity. Adaptacja metaboliczna",
            'KEY_2': lambda sc,z: f"Low-carb EMOM: 20 min EMOM lekki (5 cal bike + 5 burpees) Z2-Z3 rano na czczo — trening metabolizmu tłuszczowego w kontekście CrossFit",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"Breathing WOD: EMOM 15 min (5 cal assault bike Z3 + 5 controlled breaths box breathing 4-4-4-4) → potem 20 min Z2 rowing z fokusem na oddech przeponowy",
            'KEY_2': lambda sc,z: f"IMT + mono: 30 oddechów POWERbreathe → {_vol(sc,'30','35','40','45','50')} min Z2 ski erg/row — oddychanie nosem w Z2, synchronizacja z ruchem",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Engine builder: {_vol(sc,'35','40','50','55','60')} min @ Z2 rotation (10 min row + 10 min bike + 10 min ski), stabilne HR — budowa serca sportowego",
            'KEY_2': lambda sc,z: f"Interval cardiac: 5×4 min @ Z5 na assault bike, 3 min rest Z1 — Norwegian method adaptowany do CrossFit",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Active recovery: 20 min Z1 lekki row + 15 min mobility (pigeon, couch stretch, thoracic rotation) + 5 min box breathing",
            'KEY_2': lambda sc,z: f"Parasympathetic activation: 10 min walk + 10 min yoga flow + 10 min foam rolling + cold plunge/shower (2-3 min). Tylko jeśli RPE wczoraj >7",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Competition simulation: wylosuj 3 WODy Open-style, wykonaj z pełną intensywnością, 10 min rest między. Analizuj pacing i HR response",
            'KEY_2': lambda sc,z: f"Mixed modal threshold: 30 min AMRAP — 400m run + 15 KBS 24/16 + 12 pull-ups + 9 box jumps. Kontrolowany wysiłek Z3-Z4 przez całość",
        },
    },

    # ──────────────────────────────────────────────────────
    # HYROX
    # ──────────────────────────────────────────────────────
    'hyrox': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"HYROX threshold sim: 4× (1km run Z3-Z4 + stacja HYROX — sled push 25m, burpee broad jumps 80m, row 1000m — tempo z kontrolą HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')})",
            'KEY_2': lambda sc,z: f"Running threshold: {_vol(sc,'3×8','3×10','4×10','4×12','5×12')} min @ HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')} → przejście do sled push 2×25m — trening przejść bieg↔stacja",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max HYROX: 6× (400m run ALL-OUT + 20 wall balls 9/6kg). Cel: max HR, pełna regeneracja 3 min Z1 między setami",
            'KEY_2': lambda sc,z: f"Ski erg intervals: {_vol(sc,'5×2','6×2','6×3','8×3','8×4')} min @ Z5, 2 min rest — budowa VO₂max w specyficznym ruchu HYROX",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Station technique: 45 min praca techniczna — sled push/pull: pozycja ciała + napęd nóg, wall balls: timing oddechu, burpee broad jumps: efektywność ruchu. Lekki HR",
            'KEY_2': lambda sc,z: f"Transitions drill: 8× (200m run Z3 → natychmiastowe przejście do stacji — 10 wall balls / 15 cal row / 10 KB lunges). Cel: minimalizacja czasu przejść",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long HYROX Z2: {_vol(sc,'50','60','70','80','90')} min — 10 min run Z2 + 10 min row Z2 + 10 min ski erg Z2 + powtórz. Ciągły wysiłek aerobowy",
            'KEY_2': lambda sc,z: f"Easy run: {_vol(sc,'40','45','50','55','60')} min bieg Z2 + 10 min lekkie stacje HYROX (wall balls/lunges) Z1-Z2 — budowa bazy pod wyścig 60-90 min",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax HYROX sim: {_vol(sc,'60','70','80','90','100')} min na czczo — rotacja: 8 min run + 2 min stacja lekka (wall balls/KB), cały czas Z2. Metabolic adaptation",
            'KEY_2': lambda sc,z: f"Low-carb long run: {_vol(sc,'50','60','70','80','90')} min Z2 rano na czczo — budowa fat oxidation dla long-duration HYROX",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"Breathing + HYROX: 30 oddechów IMT → 30 min HYROX Z2 (run-row-ski rotation) z 3:2 oddychaniem (3 kroki wdech, 2 kroki wydech) — kontrola pod obciążeniem",
            'KEY_2': lambda sc,z: f"Nasal Z2: {_vol(sc,'25','30','35','40','45')} min run/row oddychając TYLKO nosem. Jeśli musisz otworzyć usta = zwolnij. Potem 10 min stacje lekkie normalny oddech",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Cardiac HYROX: 4×4 min @ Z5 (alternuj: run/row/ski), 3 min Z1 rest — Norwegian method pod HYROX, budowa objętości wyrzutowej",
            'KEY_2': lambda sc,z: f"Steady state: {_vol(sc,'55','65','75','85','90')} min Z2 mixed (run+row+ski) — długie sesje budują serce sportowe",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"HYROX recovery: 20 min walk/easy jog Z1 + 15 min mobility (hip flexors, shoulders, thoracic) + cold shower 2-3 min. Rób po ciężkich treningach",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'25','30','30','35','35')} min Z1 lekki row/bike + yoga 15 min — parasympathetic activation",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race sim: pełna symulacja HYROX (8×1km run + 8 stacji) @ 85-90% race pace — testuj pacing strategy i odżywianie",
            'KEY_2': lambda sc,z: f"Half HYROX: 4×1km + 4 stacje @ race pace → analiza: gdzie HR rosło najszybciej? Tam jest Twój limiter w wyścigu",
        },
    },

    # ──────────────────────────────────────────────────────
    # ROWING
    # ──────────────────────────────────────────────────────
    'rowing': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold rowing: {_vol(sc,'4×6','4×8','5×8','5×10','6×10')} min @ HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')}, rate 26-28 s/m, rest 3 min Z1 rate 18-20",
            'KEY_2': lambda sc,z: f"Progresywny 5k: 5000m erg — start rate 22 Z3, co 1000m dodaj 1 s/m i zwiększ tempo. Ostatnie 1000m rate 28+ Z4",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max erg: {_vol(sc,'5×500m','6×500m','6×750m','8×500m','8×750m')} ALL-OUT, rate 32-36, 3 min rest Z1 — krótkie, max bodźce",
            'KEY_2': lambda sc,z: f"1min ON / 1min OFF: {_vol(sc,'8','10','12','14','16')} rund — 1 min max rate 30+ / 1 min paddle Z1 rate 18. VO₂max w specyfice wioślarskiej",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Technika + rate control: rozgrzewka 10 min → 6×2 min legs-only rowing + 6×2 min arms-only + 4×3 min pełny ruch Z2 rate 22 (fokus: catch timing, drive sequence) + 10 min Z1",
            'KEY_2': lambda sc,z: f"Pause drills: {_vol(sc,'4×5','5×5','5×6','6×6','6×8')} min z pauzą na finish (1s hold) — wymusza kontrolę łopaty i pozycji ciała. Rate 20-22, HR Z2",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Steady state erg: {_vol(sc,'40','50','55','60','70')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')}, rate 18-22 — UT2 zone, fundament wioślarstwa",
            'KEY_2': lambda sc,z: f"Cross-training Z2: {_vol(sc,'40','45','50','55','60')} min bike/run Z2 — budowa bazy aerobowej poza ergometrem, oszczędzanie pleców",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax erg: {_vol(sc,'50','60','65','70','80')} min rano na czczo @ Z2 rate 18-20, HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — adaptacja metaboliczna",
            'KEY_2': lambda sc,z: f"Sleep-low: wieczorem interwały Z4 na erg (deplecja) → rano UT2 {_vol(sc,'40','45','50','55','60')} min na czczo",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + erg: 30 oddechów POWERbreathe → {_vol(sc,'35','40','45','50','55')} min Z2 rate 20 z synchronizacją: wdech na recovery, wydech na drive",
            'KEY_2': lambda sc,z: f"Breathing sync: {_vol(sc,'25','30','35','40','45')} min Z2 — 1 breath per stroke rate 20-22, stopniowo zwiększaj rate do 26 utrzymując 1:1 ratio",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Norwegian 4×4 erg: 4×4 min @ Z5 rate 30-32, 3 min Z1 rate 18 — max SV stimulus",
            'KEY_2': lambda sc,z: f"Long UT1: {_vol(sc,'50','60','70','80','90')} min @ Z2-Z3, rate 22-24 — dłuższe sesje budują serce sportowe",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery row: 20 min @ Z1 rate 18 (ultra-lekki, niemal bez oporu) + 15 min stretching (hip flexors, hamstrings, thoracic rotation, lats)",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'20','25','30','30','30')} min Z1 bike (zamiast ergu — oszczędzaj plecy) + mobility 15 min",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"2k race prep: 3× negative split 1000m (first 500m Z3, second 500m Z4-Z5) + 5 min Z1. Symulacja race pacing",
            'KEY_2': lambda sc,z: f"Rate pyramid: 1 min each rate 24/26/28/30/32/30/28/26/24, 1 min Z1 between peaks — pacing awareness",
        },
    },

    # ──────────────────────────────────────────────────────
    # SWIMMING
    # ──────────────────────────────────────────────────────
    'swimming': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold set: rozgrzewka 400m mix → {_vol(sc,'4×200m','6×200m','6×250m','8×200m','8×250m')} @ CSS pace (HR Z4), 30s rest → cool-down 200m",
            'KEY_2': lambda sc,z: f"Descending 100s: {_vol(sc,'6','8','10','12','14')}×100m descend 1-3 (Z2→Z3→Z4) na 15s rest — budowa progu przez progresywne przyspieszanie",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max swim: rozgrzewka 400m → {_vol(sc,'6×50m','8×50m','8×75m','10×75m','10×100m')} ALL-OUT na 30-40s rest → 200m cool-down. Krótkie, max bodźce",
            'KEY_2': lambda sc,z: f"Broken 200s: {_vol(sc,'3','4','5','6','6')}×200m jako 4×50m na 10s rest (race pace minus 3s/100m) → 2 min Z1 między setami",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Drill session: 400m WU → 8×50m drills (catch-up, fingertip drag, fist, zipper) + 6×100m paddles Z2-Z3 (focus: high-elbow catch + rotation) + 6×50m kick only + 200m CD",
            'KEY_2': lambda sc,z: f"SWOLF focus: {_vol(sc,'6','8','10','10','12')}×100m Z2 — licz uderzenia na długość (SWOLF), cel: obniżyć o 1-2 na 25m. Technika > prędkość",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Endurance swim: {_vol(sc,'1500','2000','2500','3000','3500')}m ciągłe pływanie @ Z2 HR — użyj zegarek HR. Różne style: 80% kraul, 10% grzbiet, 10% klasyk",
            'KEY_2': lambda sc,z: f"Pull set Z2: rozgrzewka 400m → {_vol(sc,'6','8','10','12','14')}×200m z pullbuoy @ Z2, 15s rest — budowa bazy aerobowej z redukcją kosztu nóg",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax swim: {_vol(sc,'2000','2500','3000','3500','4000')}m rano na czczo @ Z2 — długi, ciągły set. Woda + elektrolity na deckrze",
            'KEY_2': lambda sc,z: f"Low-carb endurance: {_vol(sc,'1800','2200','2500','3000','3500')}m Z2 pull/full na czczo — train-low strategy adaptowana do pływania",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"Hypoxic set: rozgrzewka 400m → 8×50m co 5 uderzeń oddech + 8×50m co 7 uderzeń + 4×100m normalne Z2 → CD 200m. Progresja oddechowa",
            'KEY_2': lambda sc,z: f"IMT + swim: 30 oddechów POWERbreathe → {_vol(sc,'1500','2000','2500','3000','3000')}m Z2 z bilateral breathing (co 3 uderzenia) — efektywność oddechowa",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"VO₂max cardiac: 8×100m @ Z5 (max HR), 30s rest → 400m Z1 → powtórz 2-3×. Budowa objętości wyrzutowej przez krótkie, intensywne bodźce",
            'KEY_2': lambda sc,z: f"Steady endurance: {_vol(sc,'2000','2500','3000','3500','4000')}m ciągłe Z2 — długie sesje pływackie efektywnie budują serce sportowe",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery swim: 1000m super-easy mix (250m kraul + 250m grzbiet + 250m klasyk + 250m z deską) @ Z1 + 10 min stretching na lądzie",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'800','1000','1200','1200','1500')}m Z1 pull/backstroke + mobility na lądzie 15 min",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race pace: {_vol(sc,'4','5','6','8','8')}×100m @ goal race pace, 20s rest → pacing discipline. Zakończ 4×50m build (Z2→Z5)",
            'KEY_2': lambda sc,z: f"Negative split: {_vol(sc,'2','3','3','4','4')}×400m — pierwsza połowa Z3, druga połowa Z4. Symulacja race strategy",
        },
    },

    # ──────────────────────────────────────────────────────
    # XC SKI (Cross-Country Skiing)
    # ──────────────────────────────────────────────────────
    'xc_ski': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold ski: {_vol(sc,'3×8','4×8','4×10','5×10','5×12')} min @ HR {z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')} (technique focus: V2/double pole), rest 3 min Z1 easy ski",
            'KEY_2': lambda sc,z: f"L3 tempo: {_vol(sc,'20','25','30','35','40')} min ciągłego wysiłku @ Z3-Z4 na nartorolkach/nartach — równy wysiłek, stabilne HR",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max ski: {_vol(sc,'5×3','6×3','6×4','8×3','8×4')} min @ Z5 pod górę (classic/skating), 3 min aktywna pauza Z1 w dół",
            'KEY_2': lambda sc,z: f"Double pole intervals: {_vol(sc,'6×1','8×1','8×1.5','10×1.5','10×2')} min ALL-OUT double pole, 2 min rest — VO₂max + upper body power",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Technique session: 45 min — 15 min classic drills (no-pole skiing, one-leg kick) + 15 min skating drills (V1 focus, weight shift) + 15 min Z2 easy ski integrating corrections",
            'KEY_2': lambda sc,z: f"Roller ski technique: {_vol(sc,'40','45','50','55','60')} min Z2 na nartorolkach z video feedback — fokus na timing, napęd nóg, push-off angle",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long Z2 ski: {_vol(sc,'60','75','90','105','120')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — teren falisty, płaski → krytyczna baza aerobowa dla biegów narciarskich",
            'KEY_2': lambda sc,z: f"Cross-training Z2: {_vol(sc,'50','60','70','80','90')} min bieg/rower/nartorolki Z2 — objętość aerobowa poza nartami",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax ski: {_vol(sc,'60','70','80','90','100')} min Z2 na czczo (nartorolki/bieg) — kluczowe dla wielogodzinnych wyścigów narciarskich",
            'KEY_2': lambda sc,z: f"Sleep-low: wieczorem interwały Z4 → rano bieg/ski {_vol(sc,'45','50','55','60','70')} min Z2 na czczo",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + ski endurance: 30 oddechów POWERbreathe → {_vol(sc,'40','50','55','60','70')} min Z2 easy ski — oddychanie kluczowe w mrozie, trening przeponowy",
            'KEY_2': lambda sc,z: f"Cold-air breathing: {_vol(sc,'30','35','40','45','50')} min Z2 z buff/komin na twarzy — wdech nosem, wydech ustami. Trening tolerancji zimnego powietrza",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Norwegian 4×4 ski: 4×4 min pod górę @ Z5, 3 min Z1 w dół — legendarny trening norweskich narciarzy, budowa serca sportowego",
            'KEY_2': lambda sc,z: f"Long steady state: {_vol(sc,'70','80','90','100','120')} min Z2 teren falisty — objętość buduje serce narciarza",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery ski: 25-30 min Z1 lekki classic na płaskim + 15 min stretching (hip flexors, IT band, shoulders) + sauna jeśli dostępna",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'20','25','30','30','30')} min Z1 easy bike + mobility upper body + foam rolling — regeneracja poza nartami",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race simulation: {_vol(sc,'30','40','50','60','60')} min @ race pace zmienny teren — ćwicz pacing: Z3 płaskie, Z5 podjazdy, Z2 zjazdy",
            'KEY_2': lambda sc,z: f"Sprint repeats: 8-12× sprint 200m (classic/skating) z max intensywnością, 2 min Z1 rest — prędkość startowa i finish",
        },
    },

    # ──────────────────────────────────────────────────────
    # SOCCER
    # ──────────────────────────────────────────────────────
    'soccer': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold SSG: 4v4 na małym boisku 4× {_vol(sc,'4','5','5','6','6')} min z kontrolą HR Z4 ({z.get('z4',{}).get('hr_low','?')}-{z.get('z4',{}).get('hr_high','?')}), 3 min active rest. Lub: tempo run {_vol(sc,'3×6','3×8','4×8','4×10','5×10')} min Z4",
            'KEY_2': lambda sc,z: f"SSG conditioning: 5v5 continuous 20 min z regułami wymuszającymi intensywność (max 2 dotknięcia, gra na czas) → HR Z3-Z4 → przejście do Z2 łatwy bieg 15 min",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"RSA (Repeated Sprint Ability): {_vol(sc,'2×6','3×6','3×8','4×6','4×8')} × sprint 30m z 25s rest, 4 min między seriami — kluczowe dla piłki nożnej (Buchheit & Laursen 2013)",
            'KEY_2': lambda sc,z: f"4×4 min bieg @ Z5 HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')}, 3 min jog Z1 — budowa VO₂max, podstawa high-intensity actions w meczu",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Agility + running drills: rozgrzewka FIFA 11+ → T-test drill 6×, shuttle runs 5-10-5 m × 8, carioca/side shuffle 4×20m → Z2 15 min jog + stretching",
            'KEY_2': lambda sc,z: f"Running mechanics: A-skip, B-skip, high knees, butt kicks (3×30m) → 8× sprint 40m z deceleracją → Z2 15 min cool-down. Ekonomia biegu = mniej zmęczenia w 90 min",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long Z2 run: {_vol(sc,'40','45','50','55','60')} min @ HR {z.get('z2',{}).get('hr_low','?')}-{z.get('z2',{}).get('hr_high','?')} — baza aerobowa to fundament regeneracji między sprintami w meczu",
            'KEY_2': lambda sc,z: f"Aerobic SSG: 7v7/8v8 duże boisko, 2×15 min @ Z2 HR — gra ciągła, focus na utrzymanie pozycji i podania, nie na intensywność",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax run: {_vol(sc,'40','45','50','55','60')} min Z2 rano na czczo — piłkarz z lepszym fat oxidation ma więcej glikogenu na sprints w 2. połowie",
            'KEY_2': lambda sc,z: f"Low-glycogen training: wieczorem trening piłkarski (deplecja) → rano bieg {_vol(sc,'30','35','40','45','50')} min Z2 na czczo",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + Z2: 30 oddechów POWERbreathe → bieg {_vol(sc,'35','40','45','50','55')} min Z2 z kontrolą oddechu — efektywność wentylacyjna = mniej zmęczenia w meczu",
            'KEY_2': lambda sc,z: f"Controlled breathing SSG: 3v3 na małym boisku 3×5 min, instrukcja: oddychaj nosem w pauzach między akcjami. Budowa świadomości oddechowej",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"4×4 cardiac: 4×4 min bieg @ Z5, 3 min Z1 — gold standard budowy serca sportowego dla piłkarzy (Helgerud 2001 — +5 ml/kg/min VO₂max u piłkarzy)",
            'KEY_2': lambda sc,z: f"Long run: {_vol(sc,'45','50','55','60','65')} min Z2 — dłuższe sesje aerobowe poza boiskiem budują serce sportowe",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Match-day+1 recovery: 15 min walk + 15 min pool recovery (aqua jogging) + foam rolling + cold water immersion 10-12°C × 10 min",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'20','25','30','30','30')} min Z1 bike/walk + yoga 15 min + sleep hygiene (8-9h, no screens 1h before)",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Match simulation: 2×15 min SSG 8v8 @ match intensity + 10×30m sprint z deceleracją (symulacja high-intensity actions) + Z2 cool-down 10 min",
            'KEY_2': lambda sc,z: f"Position-specific: {_vol(sc,'6','8','10','10','12')}× repeat sprints z decyzyjnością (1v1, 2v1) — cognitive load + physical conditioning",
        },
    },

    # ──────────────────────────────────────────────────────
    # MMA / COMBAT SPORTS
    # ──────────────────────────────────────────────────────
    'mma': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Threshold rounds: {_vol(sc,'3×3','4×3','4×4','5×4','5×5')} min ciężka praca na worku/padach @ HR Z4, 1 min rest. Symulacja rund walki z kontrolowaną intensywnością",
            'KEY_2': lambda sc,z: f"Grappling tempo: {_vol(sc,'3×3','4×3','4×4','5×4','5×5')} min sparring techniczny @ Z3-Z4, fokus na kontrolowany wysiłek, NIE submission hunting",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"Combat VO₂max: {_vol(sc,'5×1','6×1','6×1.5','8×1','8×1.5')} min ALL-OUT praca na worku (combos) + slam ball + battle ropes, 1 min rest Z1. Max HR",
            'KEY_2': lambda sc,z: f"Sprint intervals: {_vol(sc,'6×30s','8×30s','8×40s','10×30s','10×40s')} assault bike/row ALL-OUT, 90s rest — VO₂max bez obciążenia stawów",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Technical flow: 40 min praca techniczna — shadow boxing 10 min + pad work lekki 10 min + drills grappling (shrimping, bridging, guard passes) 10 min + Z2 skip rope 10 min",
            'KEY_2': lambda sc,z: f"Energy management drills: 3×5 min sparring lekki z instrukcją: NIE forsuj, szukaj efektywności ruchu, kontroluj oddech, minimalizuj zbędne ruchy",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Aerobic base: {_vol(sc,'35','40','45','50','55')} min Z2 run/bike + 10 min skip rope Z2 — baza aerobowa = szybsza regeneracja między rundami i lepsze decyzje pod zmęczeniem",
            'KEY_2': lambda sc,z: f"Z2 mixed: {_vol(sc,'30','35','40','45','50')} min — 10 min row + 10 min bike + 10 min skip rope, Z2 ciągły. Fundament aerobowy dla sportów walki",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax session: {_vol(sc,'40','45','50','55','60')} min Z2 run/bike rano na czczo — lepsza fat oxidation = więcej glikogenu na rundy championship i weight cutting",
            'KEY_2': lambda sc,z: f"Low-carb training: lekki sparring techniczny 30 min + Z2 skip rope 15 min na czczo — metabolic adaptation",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + combat conditioning: 30 oddechów POWERbreathe → 3×5 min praca na worku Z3 z kontrolowanym oddechem (wydech na uderzeniu) + Z2 10 min skip rope",
            'KEY_2': lambda sc,z: f"Breathing under load: grappling drills 3×3 min z nosowym oddychaniem (usta zamknięte) → wymusza efektywność wentylacyjną pod obciążeniem",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"4×4 combat: 4×4 min assault bike @ Z5, 3 min Z1 rest — Norwegian method adaptowany do sportów walki, budowa serca sportowego",
            'KEY_2': lambda sc,z: f"Long Z2: {_vol(sc,'40','45','50','55','60')} min run/bike Z2 — długie sesje budują objętość wyrzutową serca, krytyczne dla 5-rundowych walk",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Fighter recovery: 15 min walk Z1 + 15 min yoga (focus: hip openers, shoulders, neck) + contrast shower (1 min zimna / 2 min ciepła × 3) + box breathing 5 min",
            'KEY_2': lambda sc,z: f"Parasympathetic: {_vol(sc,'20','25','30','30','30')} min Z1 swim/bike (zero impact) + 15 min stretching + sleep 8-9h — fundament regeneracji zawodnika MMA",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Fight simulation: {_vol(sc,'3×3','3×5','4×3','4×5','5×5')} min pełne sparring tempo (kontrolowane) z 1 min rest — symulacja walki, testuj pacing strategy",
            'KEY_2': lambda sc,z: f"Championship rounds: {_vol(sc,'3×3','3×3','4×3','5×3','5×5')} min wysokie tempo pad work (trener dyktuje combos ciągle), 1 min rest — build fight endurance",
        },
    },

    # ──────────────────────────────────────────────────────
    # TRIATHLON
    # ──────────────────────────────────────────────────────
    'triathlon': {
        'HIGH_BASE_LOW_THRESHOLD': {
            'KEY_1': lambda sc,z: f"Brick threshold: bike {_vol(sc,'3×10','3×12','4×12','4×15','5×15')} min Z4 → natychmiast run 10 min Z3-Z4. Symulacja przejścia T2 pod obciążeniem progowym",
            'KEY_2': lambda sc,z: f"Swim threshold: {_vol(sc,'6×200m','8×200m','8×250m','10×200m','10×250m')} @ CSS pace (HR Z4), 20s rest → 15 min Z2 run po wyjściu z basenu (brick swim-run)",
        },
        'HIGH_THRESHOLDS_LOW_CEILING': {
            'KEY_1': lambda sc,z: f"VO₂max multisport: swim {_vol(sc,'6×100m','8×100m','8×100m','10×100m','10×100m')} ALL-OUT, 20s rest + bike {_vol(sc,'4×3','5×3','5×4','6×4','6×5')} min Z5, 3 min rest",
            'KEY_2': lambda sc,z: f"Run VO₂max: {_vol(sc,'4×3','5×3','5×4','6×4','6×5')} min @ HR {z.get('z5',{}).get('hr_low','?')}-{z.get('z5',{}).get('hr_high','?')}, 3 min jog Z1 — VO₂max biegu jako limitujący w triatlonie",
        },
        'ECONOMY_LIMITER': {
            'KEY_1': lambda sc,z: f"Tri-specific drills: swim drills 30 min (catch-up, fingertip, SWOLF focus) → bike cadence work 20 min (100-110 rpm Z2) → run strides 6×80m — efektywność we wszystkich 3 dyscyplinach",
            'KEY_2': lambda sc,z: f"Transition practice: 4× (300m swim Z2 → T1 → 5 min bike Z2 → T2 → 1km run Z2) — cel: płynność przejść, minimalizacja czasu T1/T2, stabilizacja HR przy zmianie dyscypliny",
        },
        'LOW_BASE': {
            'KEY_1': lambda sc,z: f"Long brick Z2: bike {_vol(sc,'60','75','90','105','120')} min + run {_vol(sc,'20','25','30','35','40')} min, cały czas Z2 HR — krytyczny trening objętości triatlonowej",
            'KEY_2': lambda sc,z: f"Swim endurance: {_vol(sc,'2000','2500','3000','3500','4000')}m ciągłe Z2 + bike 30 min Z2 — baza aerobowa w dyscyplinach non-impact",
        },
        'SUBSTRATE_LIMITER': {
            'KEY_1': lambda sc,z: f"FATmax brick: bike {_vol(sc,'60','75','90','100','120')} min + run {_vol(sc,'20','25','30','35','40')} min Z2 rano na czczo — fat adaptation krytyczna dla Ironman/70.3",
            'KEY_2': lambda sc,z: f"Sleep-low: wieczorem interwały Z4 swim/bike → rano brick Z2 {_vol(sc,'50','60','70','80','90')} min na czczo — train-low strategy",
        },
        'VENTILATORY_LIMITER': {
            'KEY_1': lambda sc,z: f"IMT + swim: 30 oddechów POWERbreathe → swim {_vol(sc,'1500','2000','2500','3000','3500')}m Z2 bilateral breathing (co 3) — efektywność oddechowa w wodzie kluczowa",
            'KEY_2': lambda sc,z: f"Bike nasal breathing: {_vol(sc,'30','35','40','45','50')} min bike Z2 oddychanie nosem → wymusza efektywność. Potem run 15 min kontrolowany oddech 3:2",
        },
        'CARDIAC_LIMITER': {
            'KEY_1': lambda sc,z: f"Norwegian bike 4×4: 4×4 min @ Z5, 3 min Z1 → natychmiast run 10 min Z2 (brick test cardiac response) — budowa serca sportowego",
            'KEY_2': lambda sc,z: f"Long Z2 brick: bike {_vol(sc,'75','90','105','120','150')} min + run {_vol(sc,'20','30','35','40','45')} min Z2 — objętość buduje serce triatlonisty",
        },
        'RECOVERY_LIMITER': {
            'KEY_1': lambda sc,z: f"Recovery swim: 1000m easy mix (Z1) + 10 min stretching na lądzie — pływanie idealne do recovery (zero impact, aktywacja parasympathetic)",
            'KEY_2': lambda sc,z: f"Active recovery: {_vol(sc,'20','25','30','30','30')} min Z1 bike + yoga 15 min — w triatlonie zarządzanie recovery jest kluczowe przy 3 dyscyplinach",
        },
        'RACE_SPECIFIC': {
            'KEY_1': lambda sc,z: f"Race simulation: swim 750m race pace → T1 → bike 20km race pace → T2 → run 5km race pace — sprint distance simulation z pełnymi przejściami",
            'KEY_2': lambda sc,z: f"Brick race pace: bike {_vol(sc,'30','40','50','60','60')} min @ race pace Z3-Z4 → natychmiast run {_vol(sc,'15','20','25','30','30')} min @ race pace — kluczowy trening T2",
        },
    },
}

# ─── Helper functions ───
def _vol(sc, *levels):
    """Select volume by sport class rank (0-4)."""
    idx = min(sc, len(levels)-1)
    return levels[idx]

def _pace(z, zone_key):
    """Format pace from zone speed data."""
    spd = z.get(zone_key, {}).get('speed_high', 0)
    if not spd or spd <= 0: return ""
    pace_min = 60 / spd
    m = int(pace_min)
    s = int((pace_min - m) * 60)
    return f"{m}:{s:02d}/km"



def _format_pace(speed_kmh):
    if not speed_kmh or speed_kmh <= 0: return ""
    pace_min = 60 / speed_kmh
    m = int(pace_min)
    s = int((pace_min - m) * 60)
    return f"{m}:{s:02d}/km"

def _scale_session(template_name, snap, sport_class_rank, lim_name, modality='run'):
    """Generate sport-specific session from SPORT_SESSIONS database."""
    sc = sport_class_rank
    z = snap.zones
    
    # Normalize modality aliases
    _mod_alias = {
        'treadmill': 'run', 'walk': 'run',
        'wattbike': 'bike', 'echobike': 'bike',
        'swim': 'swimming',
        'ergometer': 'rowing',
    }
    mod = _mod_alias.get(modality, modality)
    
    # Get sport-specific sessions
    sport_db = SPORT_SESSIONS.get(mod, SPORT_SESSIONS.get('run'))
    limiter_db = sport_db.get(lim_name, sport_db.get('HIGH_BASE_LOW_THRESHOLD', {}))
    
    key = 'KEY_2' if 'KEY_2' in template_name else 'KEY_1'
    session_fn = limiter_db.get(key)
    
    if session_fn:
        try:
            return session_fn(sc, z)
        except Exception:
            pass
    
    # Fallback: generic threshold/VO2max
    z4 = z.get('z4', {})
    z5 = z.get('z5', {})
    if 'KEY_1' in template_name:
        work = {0:"2×8 min",1:"3×8 min",2:"3×10 min",3:"3×12 min",4:"4×12 min"}.get(sc,"3×10 min")
        return f"Threshold: {work} @ HR {z4.get('hr_low',0)}-{z4.get('hr_high',0)} (Z4), rest 3 min Z1"
    else:
        work = {0:"3×3 min",1:"4×3 min",2:"4×4 min",3:"5×4 min",4:"6×4 min"}.get(sc,"4×4 min")
        return f"VO₂max: {work} @ HR {z5.get('hr_low',0)}-{z5.get('hr_high',0)} (Z5), rest 3 min Z1"

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
            # Smart zone detection from session content
            # Priority: detect MAIN effort zone, ignore warmup/cooldown/pause mentions
            _dl = desc.lower()
            _has_interval = any(k in _dl for k in ('intervals', 'interwał', '×', 'x ', 'repeats', '4×4'))
            _is_drill = any(k in _dl for k in ('drill', 'plyometr', 'strides', 'skip', 'knees', 'kicks'))
            _is_breathing = any(k in _dl for k in ('imt', 'breathing', 'oddychanie', 'powerbreathe', 'oddech'))
            if _has_interval and ('vo₂max' in _dl or ('z5' in _dl and 'pauza' not in _dl.split('z5')[0])):
                zone = "Z5"
            elif 'hill repeat' in _dl or ('4×4' in _dl and 'hr 1' in _dl and '56' in _dl):
                zone = "Z5"
            elif _is_drill or _is_breathing:
                zone = "Z2"
            elif 'threshold' in _dl or 'cruise' in _dl or 'tempo' in _dl or 'pogranicze z3' in _dl:
                zone = "Z4"
            elif 'fatmax' in _dl or ('z2' in _dl and 'long' in _dl) or ('z2' in _dl and 'easy' in _dl) or 'na czczo' in _dl:
                zone = "Z2"
            elif 'z3' in _dl and 'z4' not in _dl and 'z5' not in _dl:
                zone = "Z3"
            elif 'z2' in _dl:
                zone = "Z2"
            else:
                zone = "Z4"
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
