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
    "RUN_RAMP_GENERIC": "🏃 Ramp — generyczny bieg",
    "RUN_RAMP_6to14_KO2452": "🏃 Ramp 6→14 km/h (KO2452)",
    "RUN_STEP_BIEZNIA_WYJAZD": "🏃 Step — bieżnia (Wyjazd)",
    "RUN_STEP_LACTATE_HYROX_INCL2p5_E2MIN": "🏋️ Step — HYROX lactate",
    "RUN_STEP_3MIN_EXAMPLE": "🏃 Step 3min — przykład",
    "BIKE_RAMP_50to250_EXAMPLE": "🚴 Ramp rower 50→250W",
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

    st.markdown("### 📁 Plik CSV")
    uploaded_file = st.file_uploader(
        "Wgraj plik CPET (.csv)",
        type=["csv"],
        help="Plik z danymi breath-by-breath z ergospirometru"
    )

    auto_info = {}
    if uploaded_file is not None:
        try:
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
                st.success(f"✅ Analiza zakończona pomyślnie — silniki: {len(app.results)}")

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

                html_content = results["html_report"]
                if e20_html:
                    html_content = html_content.replace("</body>", e20_html + "</body>")

                st.components.v1.html(html_content, height=4000, scrolling=True)

                safe_name = re.sub(r'[^\w\s-]', '', athlete_name or 'raport').replace(' ', '_')
                filename_html = f"CPET_Report_{safe_name}_{test_date}.html"

                st.download_button(
                    label="💾 Pobierz raport HTML",
                    data=html_content,
                    file_name=filename_html,
                    mime="text/html",
                    type="primary",
                    use_container_width=True
                )

                if "text_report" in results:
                    with st.expander("📋 Raport tekstowy (dla trenerów)"):
                        st.text(results["text_report"])

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

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:11px;'>"
    "CPET Analysis Engine v2.2 | Powered by 25 Engines (E00–E20) | "
    "Built for sport science professionals"
    "</div>",
    unsafe_allow_html=True
)
