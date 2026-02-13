"""
🫁 CPET Analysis Engine — Streamlit App
========================================
Upload CSV → konfiguracja → analiza → raport HTML do pobrania.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import re
from datetime import datetime
from cortex_xml_parser import parse_cortex_xml

# ════════════════════════════════════════════════════════
# LAZY ENGINE LOADING (deferred to avoid HF health check timeout)
# ════════════════════════════════════════════════════════

_engine_loaded = False

def _load_engine():
    global _engine_loaded
    if _engine_loaded:
        return
    import engine_core as ec
    globals()['AnalysisConfig'] = ec.AnalysisConfig
    globals()['LactateInput'] = ec.LactateInput
    globals()['RAW_PROTOCOLS'] = ec.RAW_PROTOCOLS
    globals()['compile_protocol_for_apply'] = ec.compile_protocol_for_apply
    globals()['CPET_Orchestrator'] = ec.CPET_Orchestrator
    # Compile protocols
    for pname, psegs in ec.RAW_PROTOCOLS.items():
        try:
            PROTOCOLS_DB[pname] = ec.compile_protocol_for_apply(psegs)
        except Exception:
            pass
    _engine_loaded = True

PROTOCOLS_DB = {}

# ════════════════════════════════════════════════════════
# KONFIGURACJA STRONY
# ════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🫁 CPET Analysis Engine",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { max-width: 1200px; padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown label,
    div[data-testid="stSidebar"] label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PROTOKOŁY (statyczna mapa — nie wymaga engine)
# ════════════════════════════════════════════════════════

PROTOCOL_MAP = {
    "AUTO": "🔍 Auto-detekcja z pliku",
    "KINETICS": "🔬 Kinetyka VO₂ (CWR 4×6min)",
    "RUN_RAMP": "🏃 Bieżnia — Ramp",
    "RUN_STEP_1KMH": "🏃 Bieżnia — Step +1 km/h / 2 min",
    "RUN_STEP_05KMH": "🏃 Bieżnia — Step +0.5 km/h / 2 min",
    "BIKE_STEP_20W": "🚴 Rower — Step +20 W / 2 min",
    "ROW_ERG_WOMAN_25W": "🚣 Ergometr wioślarski — Step +25 W / 2 min (K)",
}

# ════════════════════════════════════════════════════════
# AUTO-EKSTRAKCJA Z CSV
# ════════════════════════════════════════════════════════

def auto_extract_from_csv(df, filename=""):
    info = {}
    if filename:
        parts = filename.replace('.csv', '').split('_')
        for i, p in enumerate(parts):
            if p == 'CPET' and i + 2 < len(parts):
                info['name'] = f"{parts[i+1]} {parts[i+2]}" if i + 2 < len(parts) else parts[i+1]
        # Extract date: look for YYYY_MM_DD or YYYY-MM-DD pattern in filename
        import re as _re
        _dm = _re.search(r'(\d{4})[_\-](\d{2})[_\-](\d{2})', filename)
        if _dm:
            info['date'] = f"{_dm.group(1)}-{_dm.group(2)}-{_dm.group(3)}"
        else:
            date_candidates = [p for p in parts if len(p) == 10 and p.count('_') == 0 and p.replace('-','').isdigit()]
            if not date_candidates:
                date_candidates = [p for p in parts if len(p) >= 8 and p[:4].isdigit()]
            if date_candidates:
                info['date'] = date_candidates[0][:10]
    if len(df) > 0:
        row = df.iloc[0]
        for col in ['Age', 'Wiek']:
            if col in df.columns and pd.notna(row[col]):
                try:
                    info['age'] = int(float(row[col]))
                except: pass
        for col in ['Weight_kg', 'Body_mass_kg', 'Waga', 'Mass']:
            if col in df.columns and pd.notna(row[col]):
                try:
                    info['weight'] = float(row[col])
                except: pass
        for col in ['Height_cm', 'Wzrost']:
            if col in df.columns and pd.notna(row[col]):
                try:
                    info['height'] = float(row[col])
                except: pass
        sex_col = next((c for c in df.columns if c in ('Sex', 'Gender')), None)
        if sex_col and pd.notna(row[sex_col]):
            raw = str(row[sex_col]).strip().lower()
            sex_map = {'male': 'male', 'm': 'male', 'mężczyzna': 'male',
                       'female': 'female', 'f': 'female', 'kobieta': 'female'}
            if raw in sex_map:
                info['sex'] = sex_map[raw]
    return info

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🫁 CPET Engine")
    st.markdown("---")

    st.markdown("### 📁 Plik CPET")
    uploaded_file = st.file_uploader(
        "Wgraj plik CPET (.csv lub .xml)",
        type=["csv", "xml"],
        help="CSV z danymi breath-by-breath lub oryginalny XML z Cortex MetaSoft"
    )

    st.markdown("---")
    st.markdown("### 📄 Typ raportu")
    report_type = st.radio(
        "Wybierz format raportu",
        ["PRO — Pełna diagnostyka", "LITE — Dla sportowca", "KINETYKA — Raport kinetyczny"],
        index=0,
        help="PRO: pełny raport z danymi diagnostycznymi. LITE: uproszczony, wizualny raport dla klienta. KINETYKA: dedykowany raport z testu CWR kinetyki VO₂."
    )

    auto_info = {}
    _csv_tmp_from_xml = None
    if uploaded_file is not None:
        # ── XML → CSV conversion (Cortex MetaSoft) ──
        _is_xml = uploaded_file.name.lower().endswith('.xml')
        if _is_xml:
            try:
                with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb') as _xf:
                    _xf.write(uploaded_file.getvalue())
                    _xml_tmp = _xf.name
                _csv_tmp_from_xml = parse_cortex_xml(_xml_tmp, tempfile.gettempdir())
                st.success("✅ XML Cortex → CSV skonwertowany automatycznie")
                os.unlink(_xml_tmp)
            except Exception as e:
                st.error(f"❌ Błąd parsowania XML: {e}")
                st.stop()

        try:
            if _csv_tmp_from_xml:
                df_preview = pd.read_csv(_csv_tmp_from_xml, nrows=2)
                auto_info = auto_extract_from_csv(df_preview, os.path.basename(_csv_tmp_from_xml))
            else:
                df_preview = pd.read_csv(uploaded_file, nrows=2)
                uploaded_file.seek(0)
                auto_info = auto_extract_from_csv(df_preview, uploaded_file.name)
            if auto_info:
                extracted = ", ".join(f"{k}={v}" for k, v in auto_info.items())
                st.success(f"✅ Auto: {extracted}")
        except Exception as e:
            st.warning(f"⚠️ Podgląd CSV: {e}")

    st.markdown("---")
    st.markdown("### 👤 Dane sportowca")

    athlete_name = st.text_input(
        "Imię i nazwisko",
        value=auto_info.get('name', ''),
        placeholder="np. Anna Kowalska"
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Wiek", min_value=10, max_value=99,
                              value=auto_info.get('age', 30))
        weight = st.number_input("Waga (kg)", min_value=30.0, max_value=200.0,
                                 value=auto_info.get('weight', 70.0), step=0.5)
    with col2:
        sex = st.selectbox("Płeć", ["male", "female"],
                          index=0 if auto_info.get('sex', 'male') == 'male' else 1,
                          format_func=lambda x: "Mężczyzna" if x == "male" else "Kobieta")
        height = st.number_input("Wzrost (cm)", min_value=100.0, max_value=230.0,
                                 value=auto_info.get('height', 175.0), step=0.5)

    test_date = st.text_input("Data testu",
                              value=auto_info.get('date', datetime.now().strftime('%Y-%m-%d')))

    st.markdown("---")
    st.markdown("### ⚙️ Ustawienia testu")

    protocol = st.selectbox(
        "Protokół",
        options=list(PROTOCOL_MAP.keys()),
        index=2,
        format_func=lambda x: PROTOCOL_MAP.get(x, x)
    )

    # ── Kinetics protocol: 4 speeds input ──
    kinetics_speeds = None
    if protocol == "KINETICS":
        st.markdown("#### 🔬 Prędkości protokołu kinetyki")
        st.caption("Podaj 4 prędkości użyte w teście CWR (constant work rate)")
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            ks1 = st.number_input("S1: Baseline", min_value=3.0, max_value=25.0,
                                  value=10.6, step=0.1, help="Poniżej VT1")
        with kc2:
            ks2 = st.number_input("S2: ~VT1", min_value=3.0, max_value=25.0,
                                  value=12.9, step=0.1, help="W okolicach VT1")
        with kc3:
            ks3 = st.number_input("S3: ~VT2", min_value=3.0, max_value=25.0,
                                  value=15.7, step=0.1, help="W okolicach VT2")
        with kc4:
            ks4 = st.number_input("S4: >VT2", min_value=3.0, max_value=25.0,
                                  value=16.7, step=0.1, help="Powyżej VT2 (severe)")
        kinetics_speeds = [ks1, ks2, ks3, ks4]

    # ── Interactive protocol chart ──
    if protocol not in ("AUTO", "KINETICS"):
        _raw_segs = RAW_PROTOCOLS.get(protocol, [])
        if _raw_segs:
            _chart_t, _chart_v, _chart_label = [], [], []
            for _seg in _raw_segs:
                _s = _seg.get("start", "0:00"); _e = _seg.get("end", "0:00")
                _ts = sum(int(x)*m for x, m in zip(_s.split(":"), [60, 1]))
                _te = sum(int(x)*m for x, m in zip(_e.split(":"), [60, 1]))
                _pw = _seg.get("power_w"); _sp = _seg.get("speed_kmh", 0)
                _val = _pw if _pw is not None else _sp
                _lbl = f"{_pw}W" if _pw is not None else f"{_sp} km/h"
                _chart_t += [_ts/60, _te/60]; _chart_v += [_val, _val]; _chart_label += [_lbl, _lbl]
            import pandas as _pd_chart
            _chart_df = _pd_chart.DataFrame({"min": _chart_t, "wartość": _chart_v})
            _y_label = "Moc (W)" if any(s.get("power_w") is not None for s in _raw_segs) else "Prędkość (km/h)"
            st.caption(f"📊 Profil protokołu: {PROTOCOL_MAP.get(protocol, protocol)}")
            st.line_chart(_chart_df, x="min", y="wartość", height=200)

    MODALITY_MAP = {
        "run": "🏃 Bieg", "bike": "🚴 Kolarstwo", "triathlon": "🏊 Triathlon",
        "rowing": "🚣 Wioślarstwo / Ergometr", "crossfit": "🏋️ CrossFit",
        "hyrox": "💪 HYROX", "swimming": "🏊 Pływanie", "xc_ski": "⛷️ Biegi narciarskie",
        "soccer": "⚽ Piłka nożna", "mma": "🥊 MMA / Sporty walki",
    }
    modality = st.selectbox(
        "Modalność / Sport",
        options=list(MODALITY_MAP.keys()),
        index=0,
        format_func=lambda x: MODALITY_MAP.get(x, x)
    )

    t_stop_mode = st.radio("Czas odmowy (t_stop)", ["Automatyczna", "Ręczne wpisanie"],
                           horizontal=True)
    t_stop_manual = None
    if t_stop_mode == "Ręczne wpisanie":
        t_stop_manual = st.text_input("t_stop (MM:SS lub sekundy)", placeholder="np. 18:30")

    st.markdown("---")
    st.markdown("### 🎯 Ręczne progi (opcjonalnie)")
    use_manual_vt = st.checkbox("Podaj ręczne VT1/VT2")
    vt1_manual = None
    vt2_manual = None
    if use_manual_vt:
        vt1_manual = st.text_input("VT1 (MM:SS)", placeholder="np. 08:00")
        vt2_manual = st.text_input("VT2 (MM:SS)", placeholder="np. 14:30")

    st.markdown("---")
    st.markdown("### 🧪 Dane laktatowe (opcjonalnie)")
    has_lactate = st.checkbox("Mam dane laktatowe")
    lactate_data = []
    if has_lactate:
        st.markdown("*Podaj czas (MM:SS) i wartość La (mmol/L):*")
        n_lac = st.number_input("Ile punktów?", min_value=1, max_value=20, value=6)
        for i in range(int(n_lac)):
            col_t, col_v = st.columns(2)
            with col_t:
                t = st.text_input(f"Czas #{i+1}", key=f"lac_t_{i}", placeholder="MM:SS")
            with col_v:
                v = st.number_input(f"La #{i+1}", key=f"lac_v_{i}", min_value=0.0,
                                   max_value=30.0, value=0.0, step=0.1)
            if t and v > 0:
                try:
                    parts = t.split(':')
                    t_sec = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else float(t)
                    lactate_data.append({'time_sec': t_sec, 'la': v})
                except Exception:
                    pass

    # Training Plan (E20)
    st.markdown("---")
    st.markdown("### 🏋️ Plan treningowy (E20)")
    enable_e20 = st.checkbox("Generuj plan treningowy", value=True,
                             help="Na podstawie CPET generuje spersonalizowany plan tygodniowy")

    e20_training_days = 5
    e20_training_hours = 8.0
    e20_goal_type = "general"
    e20_experience = 2.0
    e20_goal_event = ""

    if enable_e20:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            e20_training_days = st.number_input("Dni treningowe/tyg", min_value=3, max_value=7, value=5)
            e20_goal_type = st.selectbox("Cel treningowy",
                ["general","endurance","threshold","vo2max","fatmax","speed","race"],
                format_func=lambda x: {
                    "general":"🎯 Ogólny rozwój","endurance":"🏃 Wytrzymałość bazowa",
                    "threshold":"⚡ Próg mleczanowy","vo2max":"🔥 VO₂max",
                    "fatmax":"🍃 Metabolizm tłuszczów","speed":"💨 Szybkość/ekonomia",
                    "race":"🏆 Przygotowanie startowe"
                }.get(x, x))
        with col_e2:
            e20_training_hours = st.number_input("Godz./tydzień", min_value=3.0, max_value=25.0,
                                                  value=8.0, step=0.5)
            e20_experience = st.number_input("Staż (lata)", min_value=0.0, max_value=30.0,
                                              value=2.0, step=0.5)
        e20_goal_event = st.text_input("Cel startowy (opcja)", placeholder="np. maraton, Hyrox, 10km")

    st.markdown("---")
    st.markdown("### 🎯 Game Changer (opcjonalnie)")
    use_gc_manual = st.checkbox("Wpisz własne zalecenie treningowe",
                                help="Nadpisuje automatyczny Game Changer w raporcie LITE i PRO. Scoring i limiter pozostają bez zmian.")
    gc_manual_text = ""
    if use_gc_manual:
        gc_manual_text = st.text_area("Treść zalecenia",
                                      placeholder="np. Threshold cruise: 3×12 min @ HR 155-165 (Z4), rest 3 min Z1",
                                      height=80)

    st.markdown("---")
    st.markdown("### 🔧 Zaawansowane")
    smooth_gas = st.slider("Wygładzanie gaz", 5, 50, 20)
    smooth_hr = st.slider("Wygładzanie HR", 3, 20, 5)

    st.markdown("---")
    st.markdown("### 📊 Profil zewnętrzny (opcjonalnie)")
    mas_input = st.number_input("MAS (m/s)", min_value=0.0, max_value=8.0,
                                value=0.0, step=0.01,
                                help="Z testu polowego: 30-15 IFT, VAMEVAL itp.")
    mss_input = st.number_input("MSS (m/s)", min_value=0.0, max_value=12.0,
                                value=0.0, step=0.01,
                                help="Maximum Sprint Speed")
    ftp_input = st.number_input("FTP (W)", min_value=0, max_value=500,
                                value=0, step=5,
                                help="Functional Threshold Power (kolarstwo)")


# ════════════════════════════════════════════════════════
# E20 HTML RENDERER
# ════════════════════════════════════════════════════════

def _render_e20_html(plan, modality="run"):
    if not plan or plan.get('status') != 'OK':
        return ""
    lims = plan.get('limiters', [])
    week = plan.get('week_template', [])
    nutr = plan.get('nutrition', {})
    prog = plan.get('progression', {})
    mon = plan.get('monitoring', [])
    zdist = plan.get('zone_distribution', {})

    lim_colors = {"ECONOMY_LIMITER":"#06b6d4","LOW_BASE":"#22c55e","HIGH_BASE_LOW_THRESHOLD":"#f59e0b",
                  "HIGH_THRESHOLDS_LOW_CEILING":"#ef4444","SUBSTRATE_LIMITER":"#84cc16",
                  "VENTILATORY_LIMITER":"#6366f1","CARDIAC_LIMITER":"#dc2626",
                  "RECOVERY_LIMITER":"#8b5cf6","RACE_SPECIFIC":"#f97316"}
    lim_html = ""
    for lim in lims:
        col = lim_colors.get(lim['name'], '#64748b')
        lim_html += f'<div style="flex:1;min-width:240px;padding:12px;border-radius:8px;border-left:4px solid {col};background:#f8fafc;margin-bottom:8px;"><div style="font-size:13px;font-weight:700;color:{col};">#{lim["priority"]} {lim["name"].replace("_"," ")}</div><div style="font-size:11px;color:#475569;margin:4px 0;">{lim["reason"]}</div><div style="font-size:10px;color:#64748b;">Metoda: {lim["method"]}</div></div>'

    zone_colors = {"Z1":"#4CAF50","Z2":"#8BC34A","Z3":"#FFC107","Z4":"#FF9800","Z5":"#F44336"}
    zbar = '<div style="display:flex;height:18px;border-radius:9px;overflow:hidden;margin:6px 0;">'
    for zn in ["Z1","Z2","Z3","Z4","Z5"]:
        pct = zdist.get(zn, 0)
        if pct > 0:
            zbar += f'<div style="flex:{pct};background:{zone_colors.get(zn,"#ccc")};display:flex;align-items:center;justify-content:center;font-size:9px;color:white;font-weight:600;">{zn} {pct}%</div>'
    zbar += '</div>'

    week_rows = ""
    for d in week:
        z_col = zone_colors.get(d['zone'], '#e2e8f0') if d['zone'] != '-' else '#f1f5f9'
        bg = '#fef3c7' if 'Key' in d.get('type','') else ('#f0fdf4' if 'Long' in d.get('type','') else '#ffffff')
        week_rows += f'<tr style="background:{bg};"><td style="padding:6px 10px;font-weight:600;font-size:12px;border-bottom:1px solid #e2e8f0;">{d["day"]}</td><td style="padding:6px 10px;font-size:11px;border-bottom:1px solid #e2e8f0;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;background:{z_col};color:white;font-size:10px;font-weight:600;">{d["zone"]}</span></td><td style="padding:6px 10px;font-size:11px;color:#334155;border-bottom:1px solid #e2e8f0;">{d["description"]}</td><td style="padding:6px 10px;font-size:11px;color:#64748b;border-bottom:1px solid #e2e8f0;text-align:center;">{d["duration_min"] if d["duration_min"] > 0 else "—"} min</td><td style="padding:6px 10px;font-size:10px;color:#94a3b8;border-bottom:1px solid #e2e8f0;">{d.get("hr_target","")}</td></tr>'

    nutr_html = ""
    for k, v in nutr.items():
        label = k.replace("_"," ")
        note = v.get('note','')
        nutr_html += f'<div style="font-size:10px;color:#475569;padding:3px 0;border-bottom:1px solid #f1f5f9;"><b>{label}:</b> przed: {v.get("pre","-")} | podczas: {v.get("during","-")} | po: {v.get("post","-")}'
        if note:
            nutr_html += f' 💡 {note}'
        nutr_html += '</div>'

    prog_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
    for wk_key in ['week_1','week_2','week_3','week_4']:
        wk = prog.get(wk_key, {})
        lbl = wk.get('label','')
        vol = wk.get('volume_pct',0)
        intn = wk.get('intensity_pct',0)
        bg_col = '#dcfce7' if lbl=='RECOVERY' else ('#fef9c3' if lbl=='PEAK' else '#f0f9ff')
        prog_html += f'<div style="flex:1;min-width:70px;padding:8px;border-radius:6px;background:{bg_col};text-align:center;"><div style="font-size:9px;font-weight:700;color:#475569;">{lbl}</div><div style="font-size:11px;color:#334155;">{vol}% vol</div><div style="font-size:10px;color:#64748b;">{intn}% int</div></div>'
    prog_html += '</div>'

    mon_html = "".join(f'<div style="font-size:10px;color:#475569;padding:2px 0;">• {m}</div>' for m in mon)

    return f'''
    <div style="margin:20px auto;max-width:900px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
      <div style="padding:20px;background:white;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #e2e8f0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
          <div style="width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:14px;">E20</div>
          <div><div style="font-size:18px;font-weight:700;color:#0f172a;">Plan treningowy</div><div style="font-size:12px;color:#64748b;">{plan["philosophy"]}</div></div>
        </div>
        <div style="font-size:12px;font-weight:600;color:#475569;margin-bottom:8px;">Zidentyfikowane ograniczenia (limitery)</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">{lim_html}</div>
        <div style="font-size:12px;font-weight:600;color:#475569;margin-bottom:4px;">Dystrybucja stref treningowych</div>
        {zbar}
        <div style="font-size:12px;font-weight:600;color:#475569;margin:16px 0 8px;">Plan tygodniowy</div>
        <table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#f8fafc;"><th style="padding:6px 10px;text-align:left;font-size:10px;color:#94a3b8;border-bottom:2px solid #e2e8f0;">Dzień</th><th style="padding:6px 10px;text-align:left;font-size:10px;color:#94a3b8;border-bottom:2px solid #e2e8f0;">Strefa</th><th style="padding:6px 10px;text-align:left;font-size:10px;color:#94a3b8;border-bottom:2px solid #e2e8f0;">Sesja</th><th style="padding:6px 10px;text-align:center;font-size:10px;color:#94a3b8;border-bottom:2px solid #e2e8f0;">Czas</th><th style="padding:6px 10px;text-align:left;font-size:10px;color:#94a3b8;border-bottom:2px solid #e2e8f0;">HR</th></tr></thead><tbody>{week_rows}</tbody></table>
        <div style="font-size:12px;font-weight:600;color:#475569;margin:16px 0 6px;">📊 Progresja (blok 4-tygodniowy)</div>{prog_html}
        <div style="font-size:12px;font-weight:600;color:#475569;margin:16px 0 6px;">🍽️ Żywienie wg typu sesji</div>{nutr_html}
        <div style="font-size:12px;font-weight:600;color:#475569;margin:16px 0 6px;">📈 Monitorowanie</div>{mon_html}
        <div style="margin-top:16px;padding:10px;background:#eff6ff;border-radius:6px;border-left:3px solid #3b82f6;"><div style="font-size:10px;color:#1e40af;"><b>Evidence base:</b> Laursen &amp; Buchheit 2013/2019, Seiler 2010, Helgerud 2007, Oliveira 2024, Grummt 2025</div></div>
      </div>
    </div>'''


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

st.markdown("# 🫁 CPET Analysis Engine")

if uploaded_file is None:
    st.info("👈 Wgraj plik CSV z danymi CPET w panelu bocznym, wypełnij dane i kliknij **START**.")
    st.stop()

st.markdown(f"""
**Sportowiec:** {athlete_name or 'brak'} | **Protokół:** {PROTOCOL_MAP.get(protocol, protocol)} |
**Modalność:** {modality} | **t_stop:** {'AUTO' if not t_stop_manual else t_stop_manual} |
**Laktaty:** {len(lactate_data)} pkt | **MAS:** {mas_input if mas_input > 0 else '—'} m/s
""")

if st.button("🚀 START — Uruchom analizę CPET", type="primary", use_container_width=True):

    with st.spinner("⏳ Ładuję silnik analityczny..."):
        _load_engine()

    with st.spinner("⏳ Analizuję dane CPET..."):

        if _csv_tmp_from_xml:
            tmp_path = _csv_tmp_from_xml
        else:
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='wb') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

        try:
            config = AnalysisConfig(
                modality=modality,
                protocol_name=protocol,
                athlete_name=athlete_name or "Nieznany Zawodnik",
                sex=sex,
                athlete_id="",
                test_date=test_date,
                age_y=int(age),
                height_cm=float(height),
                body_mass_kg=float(weight),
                vt1_manual=vt1_manual if use_manual_vt and vt1_manual else None,
                vt2_manual=vt2_manual if use_manual_vt and vt2_manual else None,
                force_manual_t_stop=t_stop_manual,
                smooth_window_gas=smooth_gas,
                smooth_window_hr=smooth_hr,
                gc_manual=gc_manual_text.strip() if use_gc_manual and gc_manual_text.strip() else None,
                kinetics_speeds_kmh=kinetics_speeds if protocol == "KINETICS" else None,
            )

            if mas_input > 0:
                config.mas_m_s = mas_input
            if mss_input > 0:
                config.mss_m_s = mss_input
            if ftp_input > 0:
                config.ftp_watts = ftp_input

            if protocol in RAW_PROTOCOLS:
                PROTOCOLS_DB[protocol] = compile_protocol_for_apply(
                    RAW_PROTOCOLS[protocol], t_stop_manual
                )

            import engine_core
            engine_core.PROTOCOLS_DB = PROTOCOLS_DB

            app = CPET_Orchestrator(config)

            if lactate_data:
                app._lactate_input = LactateInput(manual_data=lactate_data)

            progress = st.progress(0, text="Uruchamiam pipeline...")
            results = app.process_file(tmp_path)
            progress.progress(100, text="✅ Analiza zakończona!")

            if isinstance(results, dict) and "html_report" in results:
                # Store results and E20 in session_state for re-rendering on radio change
                st.session_state["cpet_results"] = results
                st.session_state["cpet_engine_count"] = len(app.results)
                
                # Generate LITE report directly here (guaranteed fresh)
                try:
                    from report import ReportAdapter
                    _ct = results.get("canon_table", {})
                    if _ct:
                        # Inject manual Game Changer if provided
                        if use_gc_manual and gc_manual_text.strip():
                            _ct['_gc_manual'] = gc_manual_text.strip()
                            results['canon_table'] = _ct
                        st.session_state["cpet_html_lite"] = ReportAdapter.render_lite_html_report(_ct)
                    else:
                        st.session_state["cpet_html_lite"] = None
                except Exception as _lite_err:
                    st.warning(f"⚠️ LITE report error: {_lite_err}")
                    st.session_state["cpet_html_lite"] = None

                # E20: Training Decision Engine
                e20_html = ""
                if enable_e20:
                    try:
                        from e20_training_decision import TrainingProfile, run_e20
                        tp = TrainingProfile(
                            modality=modality,
                            training_hours_week=e20_training_hours,
                            training_days_week=e20_training_days,
                            goal_type=e20_goal_type,
                            experience_years=e20_experience,
                            goal_event=e20_goal_event,
                        )
                        e20_plan = run_e20(app.results, tp)
                        e20_html = _render_e20_html(e20_plan, modality)
                    except Exception as ex:
                        st.warning(f"⚠️ E20: {ex}")
                st.session_state["cpet_e20_html"] = e20_html
                st.session_state["cpet_athlete_name"] = athlete_name
                st.session_state["cpet_test_date"] = test_date

            elif isinstance(results, dict) and "fatal_error" in results:
                st.error(f"❌ Błąd krytyczny: {results['fatal_error']}")
            else:
                st.warning("⚠️ Analiza zakończona, ale brak raportu HTML.")
                st.json(results if isinstance(results, dict) else {"raw": str(results)})

        except Exception as e:
            st.error(f"❌ Błąd: {e}")
            import traceback
            with st.expander("🔍 Szczegóły błędu"):
                st.code(traceback.format_exc())
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════
# RENDER REPORT (outside button block — re-renders on radio change)
# ═══════════════════════════════════════════════════════════════
if "cpet_results" in st.session_state:
    results = st.session_state["cpet_results"]
    e20_html = st.session_state.get("cpet_e20_html", "")
    _athlete = st.session_state.get("cpet_athlete_name", "raport")
    _tdate = st.session_state.get("cpet_test_date", "")
    _eng_count = st.session_state.get("cpet_engine_count", 0)

    st.success(f"✅ Analiza zakończona pomyślnie — silniki: {_eng_count}")

    # Select report based on user choice
    is_lite = "LITE" in report_type
    is_kinetics = "KINETYKA" in report_type
    _lite_html = st.session_state.get("cpet_html_lite")
    has_lite = _lite_html is not None and len(str(_lite_html)) > 100
    _kinetics_html = results.get("html_report_kinetics")
    has_kinetics = _kinetics_html is not None and len(str(_kinetics_html)) > 100

    if is_kinetics and has_kinetics:
        html_content = _kinetics_html
        report_label = "KINETICS"
    elif is_lite and has_lite:
        html_content = _lite_html
        report_label = "LITE"
    else:
        html_content = results["html_report"]
        report_label = "PRO"
        if is_kinetics and not has_kinetics:
            _e14 = results.get("raw_results", {}).get("E14", {})
            st.info(f"ℹ️ Raport kinetyczny niedostępny. E14.mode={_e14.get('mode','?')}, stages={len(_e14.get('stages',[]))}, html={len(str(_kinetics_html)) if _kinetics_html else 0}chars")

    if e20_html:
        html_content = html_content.replace("</body>", e20_html + "</body>")

    st.components.v1.html(html_content, height=4000, scrolling=True)

    safe_name = re.sub(r'[^\w\s-]', '', _athlete or 'raport').replace(' ', '_')
    filename_html = f"CPET_{report_label}_{safe_name}_{_tdate}.html"

    st.download_button(
        label=f"💾 Pobierz raport {report_label} (HTML)",
        data=html_content,
        file_name=filename_html,
        mime="text/html",
        type="primary",
        use_container_width=True
    )

    if "text_report" in results:
        with st.expander("📋 Raport tekstowy (dla trenerów)"):
            st.text(results["text_report"])

    # ── VO₂ Kinetics display (E14 CWR mode) — only if kinetics HTML report not shown ──
    _e14 = {}
    _raw = results.get("raw_results", {})
    if isinstance(_raw, dict):
        _e14 = _raw.get("E14", {})
    if not _e14:
        _e14 = results.get("E14", {})
    if _e14.get("mode") == "CWR_KINETICS" and _e14.get("stages") and not has_kinetics:
        with st.expander("🔬 Kinetyka VO₂ & Slow Component (E14)", expanded=True):
            st.markdown("### VO₂ Kinetics — Analiza CWR")
            _stages = _e14["stages"]
            _summary = _e14.get("summary", {})

            # Stages table
            cols_h = ["#", "Domena", "km/h", "VO₂ (ml/kg)", "%VO₂max", "HR", "RER",
                       "τ (s)", "MRT (s)", "R²", "SC (%)", "SC klasa"]
            rows = []
            for _s in _stages:
                rows.append([
                    f"S{_s['stage_num']}", _s['domain'],
                    _s.get('speed_kmh', '—'),
                    _s.get('vo2kg_mean', '—'), _s.get('pct_vo2max', '—'),
                    _s.get('hr_mean', '—'), _s.get('rer_mean', '—'),
                    _s.get('tau_on_s', '—'), _s.get('mrt_s', '—'),
                    _s.get('fit_r2', '—'),
                    _s.get('sc_pct', '—'), _s.get('sc_class', '—'),
                ])
            import pandas as _pd
            _df_kin = _pd.DataFrame(rows, columns=cols_h)
            st.dataframe(_df_kin, use_container_width=True, hide_index=True)

            # Summary metrics
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            with _mc1:
                _tv = _summary.get('tau_moderate')
                st.metric("τ Moderate", f"{_tv}s" if _tv else "—",
                          delta=_summary.get('tau_moderate_class', ''))
            with _mc2:
                _tv = _summary.get('tau_heavy')
                st.metric("τ Heavy", f"{_tv}s" if _tv else "—",
                          delta=_summary.get('tau_heavy_class', ''))
            with _mc3:
                _tv = _summary.get('sc_heavy_pct')
                st.metric("SC Heavy", f"{_tv}%" if _tv is not None else "—",
                          delta=_summary.get('sc_heavy_class', ''))
            with _mc4:
                _tv = _summary.get('recovery_t_half')
                st.metric("Recovery T½", f"{_tv}s" if _tv else "—",
                          delta=_summary.get('recovery_class', ''))

            # Off-kinetics
            _offk = _e14.get('off_kinetics', [])
            if _offk:
                st.markdown("**Off-kinetics (recovery):**")
                for _ok in _offk:
                    _tr = _ok.get('transition', '?')
                    _tau = _ok.get('tau_off_s', '—')
                    _th = _ok.get('t_half_s', '—')
                    _rc = _ok.get('recovery_class', '')
                    st.caption(f"{_tr}: τ_off={_tau}s | T½={_th}s | {_rc}")

            # Flags
            _flags = _e14.get('flags', [])
            if _flags:
                st.warning("⚠️ Flagi: " + ", ".join(_flags))

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:11px;'>"
    "CPET Analysis Engine v2.2 | Powered by 25 Engines (E00–E20) | "
    "Built for sport science professionals"
    "</div>",
    unsafe_allow_html=True
)
