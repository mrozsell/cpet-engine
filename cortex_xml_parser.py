"""
cortex_xml_parser.py — MetaSoft Studio XML → CPET CSV Parser
=============================================================
Converts the raw XML export from Cortex MetaSoft Studio (MetaMax 3B/3B-R2)
into the standardised CSV format expected by the PeakLab CPET pipeline.

Usage:
    from cortex_xml_parser import parse_cortex_xml
    csv_path = parse_cortex_xml("input.xml", output_dir="/tmp")
    # → returns path to the generated CSV

XML Structure (SpreadsheetML format):
    - Rows 0-20:    Facility header
    - Rows 21-32:   Patient data (name, sex, DOB, etc.)
    - Rows 33-65:   Biological/medical data (height, weight, BMI, etc.)
    - Rows 50-65:   Reference values (predicted VO2max, HRmax, etc.)
    - Row  107:     Summary table header (Zmienna, Jednostka, Spoczynek, ... VT1, VT2, V'O2peak...)
    - Rows 108-236: Summary table (one row per variable, columns = Rest/Warmup/VT1/VT2/Peak/Norm/Max)
    - Row  237:     BxB header (t, Faza, Marker, V'O2, V'O2/kg, V'O2/HR, HR, ...)
    - Row  238:     BxB units
    - Rows 239+:    BxB breath-by-breath data

The parser extracts:
    1. Patient metadata → broadcast as constant columns in CSV
    2. Summary table → VT1/VT2 values for HR, Speed, Power, La, VO2, SmO2
    3. BxB data → main time-series with column renaming + derived columns
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

NS = '{urn:schemas-microsoft-com:office:spreadsheet}'

# XML BxB column → CSV column rename map
COLUMN_RENAME = {
    't':        'Time_str',
    "V'O2":     'VO2_L_min',
    'HR':       'HR_bpm',
    'WR':       'Power_W',
    "V'E":      'VE_L_min',
    'VT':       'VT_L',
    'BF':       'BF_1_min',
    'CHO':      'CHO_g_h',
    'FAT':      'FAT_g_h',
    'La':       'La_mmol_L',
    "V'CO2":    'VCO2_L_min',
    'PetCO2':   'PetCO2_mmHg',
    'PetO2':    'PetO2_mmHg',
    'SmO2-1':   'SmO2_1',
    'SmO2-2':   'SmO2_2',
    'SmO2-3':   'SmO2_3',
    'SmO2-4':   'SmO2_4',
}

# Summary table variable names (Polish) → lookup keys
SUMMARY_VAR_MAP = {
    "V'O2":     'VO2',
    "V'O2/kg":  'VO2_kg',
    "V'O2/HR":  'O2pulse',
    'HR':       'HR',
    'v':        'Speed',
    'La':       'La',
    'SmO2-1':   'SmO2_1',
    'SmO2-2':   'SmO2_2',
    'SmO2-3':   'SmO2_3',
    'SmO2-4':   'SmO2_4',
    'WR':       'Power',
    'RER':      'RER',
    "V'E":      'VE',
    'BF':       'BF',
}


# ═══════════════════════════════════════════════════════════════════════
# XML PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _row_vals(row) -> list:
    """Extract cell values from an XML Row element, handling ss:Index gaps."""
    cells = row.findall(f'{NS}Cell')
    vals = []
    for c in cells:
        idx_attr = c.attrib.get(f'{NS}Index')
        if idx_attr:
            while len(vals) < int(idx_attr) - 1:
                vals.append('')
        data = c.find(f'{NS}Data')
        vals.append(data.text.strip() if data is not None and data.text else '')
    return vals


def _parse_polish_float(s: str) -> Optional[float]:
    """Parse a Polish-locale number: '1,58' → 1.58, '60,0 kg' → 60.0"""
    if not s or s == '-':
        return None
    # Strip units
    s = re.sub(r'[a-zA-Z°²/%]+$', '', s).strip()
    # Replace comma with dot
    s = s.replace(',', '.')
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_time_str(t: str) -> Optional[float]:
    """Parse Cortex time 'h:mm:ss,ms' → seconds as float.
    Examples: '0:00:06,200' → 6.2, '0:25:41,040' → 1541.04
    """
    if not t or ':' not in t:
        return None
    try:
        # Replace comma in ms part with dot
        t = t.replace(',', '.')
        parts = t.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


def _parse_date_dmy(s: str) -> Optional[date]:
    """Parse 'DD.MM.YYYY' → date object."""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), '%d.%m.%Y').date()
    except ValueError:
        return None


def _parse_datetime_dmy(s: str) -> Optional[datetime]:
    """Parse 'DD.MM.YYYY HH:MM' → datetime object."""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), '%d.%m.%Y %H:%M')
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_cortex_xml(xml_path: str, output_dir: str = None) -> str:
    """
    Parse a Cortex MetaSoft Studio XML export and produce a CPET CSV file.

    Parameters
    ----------
    xml_path : str
        Path to the .xml file
    output_dir : str, optional
        Directory for output CSV. Defaults to same dir as input.

    Returns
    -------
    str
        Path to the generated CSV file

    Raises
    ------
    ValueError
        If the XML doesn't contain expected data structures
    """
    xml_path = str(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ws = root.find(f'.//{NS}Worksheet')
    if ws is None:
        raise ValueError("No Worksheet found in XML — not a MetaSoft export?")

    table = ws.find(f'{NS}Table')
    rows = table.findall(f'{NS}Row')

    # ── 1. EXTRACT PATIENT METADATA ──────────────────────────────────
    meta = _extract_metadata(rows)

    # ── 2. EXTRACT SUMMARY TABLE (VT1/VT2/Peak) ─────────────────────
    summary = _extract_summary_table(rows)

    # ── 3. EXTRACT BxB DATA ──────────────────────────────────────────
    df = _extract_bxb_data(rows)

    # ── 4. RENAME COLUMNS ────────────────────────────────────────────
    # Handle duplicate 'v' column (first = Speed_kmh, second = v_2)
    cols = list(df.columns)
    v_count = 0
    for i, c in enumerate(cols):
        if c == 'v':
            v_count += 1
            if v_count == 1:
                cols[i] = 'Speed_kmh'
            else:
                cols[i] = 'v_2'
    df.columns = cols

    # Apply standard renames
    rename = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    # ── 5. CONVERT DATA TYPES ────────────────────────────────────────
    df = _convert_types(df)

    # ── 6. ADD DERIVED COLUMNS ───────────────────────────────────────
    df = _add_derived_columns(df, meta)

    # ── 7. ADD METADATA COLUMNS ──────────────────────────────────────
    df = _add_metadata_columns(df, meta, summary)

    # ── 8. WRITE CSV ─────────────────────────────────────────────────
    csv_path = _build_output_path(xml_path, output_dir, meta)
    df.to_csv(csv_path, index=False)

    return csv_path


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _extract_metadata(rows) -> Dict[str, Any]:
    """Extract patient and test metadata from the header section."""
    meta = {}

    # Scan rows 0-100 for key-value pairs
    field_map = {
        'Nazwisko':         'LastName',
        'Imię':             'FirstName',
        'Płeć':             'Sex',
        'Data urodzenia':   'DOB_raw',
        'Wzrost':           'Height_raw',
        'Waga':             'Weight_raw',
        'BMI':              'BMI',
        'BSA':              'BSA_raw',
        'Czas rozpoczęcia': 'Start_raw',
        'Czas trwania':     'Duration_raw',
        'Urządzenie CPET':  'Device',
        'Numer seryjny':    'SerialNumber',
        'Temperatura':      'Temperature',
        'Ciśnienie barometryczne': 'Pressure',
        'Wilgotność':       'Humidity',
        'Operator':         'Operator',
        'Maska':            'Mask',
        'ID':               'PatientID',
    }

    # Reference value patterns
    ref_map = {
        'Maksymal. Pochłanianie Tlenu':            'Ref_VO2max',
        'Maksymal. Względne Pochłanianie Tlenu':    'Ref_VO2max_rel',
        'Maksymal. Puls tlenowy':                   'Ref_O2pulse',
        'Maksymal. Częstość skurczów serca':        'Ref_HRmax',
        'Maksymal. Wentylacja minutowa':            'Ref_VEmax',
        'Maksymal. Częstość oddychania':            'Ref_BFmax',
        'Maksymal. Intensywność wysiłku':           'Ref_Wmax',
    }

    for i in range(min(100, len(rows))):
        vals = _row_vals(rows[i])
        if len(vals) >= 2 and vals[0]:
            key = vals[0].strip()
            value = vals[1].strip() if len(vals) > 1 else ''

            if key in field_map:
                meta[field_map[key]] = value
            if key in ref_map:
                meta[ref_map[key]] = value

    # Post-process
    meta['Height_cm'] = _parse_polish_float(meta.get('Height_raw', ''))
    meta['Weight_kg'] = _parse_polish_float(meta.get('Weight_raw', ''))

    dob = _parse_date_dmy(meta.get('DOB_raw', ''))
    start = _parse_datetime_dmy(meta.get('Start_raw', ''))
    if dob and start:
        meta['Age'] = start.year - dob.year - ((start.month, start.day) < (dob.month, dob.day))
    else:
        meta['Age'] = None

    # Sex normalization
    sex_raw = meta.get('Sex', '').lower()
    if 'kob' in sex_raw or 'fem' in sex_raw or 'żeńsk' in sex_raw:
        meta['Sex_norm'] = 'female'
    elif 'męż' in sex_raw or 'mal' in sex_raw or 'męsk' in sex_raw:
        meta['Sex_norm'] = 'male'
    else:
        meta['Sex_norm'] = sex_raw

    return meta


def _extract_summary_table(rows) -> Dict[str, Dict]:
    """
    Extract the summary table with VT1/VT2/Peak/Rest/Norm values.

    Returns dict like:
        {
            'HR':     {'rest': None, 'vt1': 153, 'vt2': 178, 'peak': 197, 'norm': 177, 'max': 201},
            'VO2':    {'rest': None, 'vt1': 1.58, 'vt2': 2.26, 'peak': 2.69, 'norm': 2.20, 'max': 2.97},
            ...
        }
    """
    summary = {}

    # Find summary table header row
    header_idx = None
    for i in range(100, min(240, len(rows))):
        vals = _row_vals(rows[i])
        if vals and 'Zmienna' in vals[0]:
            header_idx = i
            break

    if header_idx is None:
        return summary

    header = _row_vals(rows[header_idx])

    # Map header positions
    col_map = {}
    for ci, h in enumerate(header):
        hl = h.lower().strip()
        if 'spoczyn' in hl:                col_map['rest'] = ci
        elif 'rozgrzew' in hl:              col_map['warmup'] = ci
        elif 'pedałow' in hl:              col_map['unloaded'] = ci
        elif hl == 'vt1':                   col_map['vt1'] = ci
        elif 'vt1 % norm' in hl:           col_map['vt1_pct_norm'] = ci
        elif 'vt1 % max' in hl:            col_map['vt1_pct_max'] = ci
        elif hl == 'vt2':                   col_map['vt2'] = ci
        elif 'vt2 % norm' in hl:           col_map['vt2_pct_norm'] = ci
        elif 'vt2 % max' in hl:            col_map['vt2_pct_max'] = ci
        elif 'peak' in hl and 'norm' not in hl and '%' not in hl:
            col_map['peak'] = ci
        elif hl == 'norma':                 col_map['norm'] = ci
        elif 'bezwzgl' in hl or 'maksym' in hl: col_map['max'] = ci

    # Parse variable rows
    for i in range(header_idx + 1, min(header_idx + 130, len(rows))):
        vals = _row_vals(rows[i])
        if not vals or not vals[0]:
            continue

        var_name = vals[0].strip()
        if var_name not in SUMMARY_VAR_MAP:
            continue

        key = SUMMARY_VAR_MAP[var_name]
        entry = {}

        for slot, ci in col_map.items():
            if ci < len(vals):
                entry[slot] = _parse_polish_float(vals[ci])
            else:
                entry[slot] = None

        summary[key] = entry

    return summary


def _extract_bxb_data(rows) -> pd.DataFrame:
    """Extract breath-by-breath data starting from the BxB header row."""

    # Find BxB header (row with 't' as first cell and 'Faza' as second)
    header_idx = None
    for i in range(200, min(300, len(rows))):
        vals = _row_vals(rows[i])
        if vals and vals[0] == 't' and len(vals) > 1 and 'Faz' in vals[1]:
            header_idx = i
            break

    if header_idx is None:
        # Fallback: look more broadly
        for i in range(100, len(rows)):
            vals = _row_vals(rows[i])
            if vals and vals[0] == 't' and len(vals) >= 10:
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not find BxB data header in XML")

    headers = _row_vals(rows[header_idx])
    # Units row is header_idx + 1, skip it
    data_start = header_idx + 2

    # Collect data rows
    data_rows = []
    for i in range(data_start, len(rows)):
        vals = _row_vals(rows[i])
        if not vals or not vals[0]:
            continue
        # BxB rows have time in first column (contains ':')
        if ':' not in vals[0]:
            continue
        # Pad to header length
        while len(vals) < len(headers):
            vals.append('')
        data_rows.append(vals[:len(headers)])

    df = pd.DataFrame(data_rows, columns=headers)
    return df


# ═══════════════════════════════════════════════════════════════════════
# DATA TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════

def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to numeric where possible."""
    skip = {'Time_str', 'Faza', 'Marker'}

    for col in df.columns:
        if col in skip:
            continue
        # Replace '-' with NaN, replace comma with dot
        df[col] = df[col].replace('-', np.nan)
        df[col] = df[col].replace('', np.nan)
        if df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _add_derived_columns(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Add computed columns needed by the CPET pipeline."""

    # Time_s: parse time string to seconds
    df['Time_s'] = df['Time_str'].apply(_parse_time_str)

    # VO2 in ml/min (pipeline expects both L/min and ml/min)
    if 'VO2_L_min' in df.columns:
        df['VO2_ml_min'] = df['VO2_L_min'] * 1000

    # VCO2 in ml/min
    if 'VCO2_L_min' in df.columns:
        df['VCO2_ml_min'] = df['VCO2_L_min'] * 1000

    # CHO g/min from g/h
    if 'CHO_g_h' in df.columns:
        df['CHO_g_min'] = df['CHO_g_h'] / 60.0

    # FAT g/min from g/h
    if 'FAT_g_h' in df.columns:
        df['FAT_g_min'] = df['FAT_g_h'] / 60.0

    # O2 Pulse (same as V'O2/HR but ensure it's present)
    if "V'O2/HR" in df.columns and 'O2_Pulse' not in df.columns:
        df['O2_Pulse'] = df["V'O2/HR"]

    return df


def _add_metadata_columns(df: pd.DataFrame, meta: Dict, summary: Dict) -> pd.DataFrame:
    """Broadcast patient metadata and VT1/VT2 summary values as constant columns."""

    # Patient data
    df['FirstName'] = meta.get('FirstName', '')
    df['LastName'] = meta.get('LastName', '')
    df['Sex'] = meta.get('Sex', '')
    df['Age'] = meta.get('Age')
    df['Height_cm'] = meta.get('Height_cm')
    df['Weight_kg'] = meta.get('Weight_kg')
    df['DOB_raw'] = meta.get('DOB_raw', '')
    df['Start_raw'] = meta.get('Start_raw', '')

    # VT1/VT2 threshold values from summary table
    hr = summary.get('HR', {})
    df['VT1_HR'] = hr.get('vt1')
    df['VT2_HR'] = hr.get('vt2')

    spd = summary.get('Speed', {})
    df['VT1_Speed'] = spd.get('vt1')
    df['VT2_Speed'] = spd.get('vt2')

    pwr = summary.get('Power', {})
    df['VT1_Power'] = pwr.get('vt1')
    df['VT2_Power'] = pwr.get('vt2')

    la = summary.get('La', {})
    df['VT1_La'] = la.get('vt1')
    df['VT2_La'] = la.get('vt2')

    vo2 = summary.get('VO2', {})
    # VO2 at VT1/VT2 in ml/min (summary is in L/min, convert)
    vt1_vo2 = vo2.get('vt1')
    vt2_vo2 = vo2.get('vt2')
    df['VT1_VO2_ml_min'] = vt1_vo2 * 1000 if vt1_vo2 else None
    df['VT2_VO2_ml_min'] = vt2_vo2 * 1000 if vt2_vo2 else None

    # SmO2 at thresholds (for NIRS engine)
    for ch in ['SmO2_1', 'SmO2_2', 'SmO2_3', 'SmO2_4']:
        s = summary.get(ch, {})
        df[f'REST_{ch}'] = s.get('rest')
        df[f'VT1_{ch}'] = s.get('vt1')
        df[f'VT2_{ch}'] = s.get('vt2')
        df[f'PEAK_{ch}'] = s.get('peak')

    return df


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT FILE NAMING
# ═══════════════════════════════════════════════════════════════════════

def _build_output_path(xml_path: str, output_dir: Optional[str], meta: Dict) -> str:
    """Build standardised CSV filename from metadata."""
    if output_dir is None:
        output_dir = os.path.dirname(xml_path) or '.'

    os.makedirs(output_dir, exist_ok=True)

    # Try to build from metadata
    last = meta.get('LastName', '').strip().replace(' ', '_')
    first = meta.get('FirstName', '').strip().replace(' ', '_')
    start_raw = meta.get('Start_raw', '')

    if last and first and start_raw:
        try:
            dt = _parse_datetime_dmy(start_raw)
            ts = dt.strftime('%Y_%m_%d_%H_%M_%S') if dt else 'unknown'
        except:
            ts = 'unknown'
        fname = f"CPET__{last}_{first}_{ts}__CPET.csv"
    else:
        # Fallback: use XML filename
        base = Path(xml_path).stem
        fname = f"{base}__CPET.csv"

    return os.path.join(output_dir, fname)


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC / VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_parsed_csv(csv_path: str) -> Dict[str, Any]:
    """Quick validation of a parsed CSV to check required columns and data quality."""
    df = pd.read_csv(csv_path)

    required_cols = [
        'Time_s', 'Time_str', 'VO2_L_min', 'VCO2_L_min', 'HR_bpm',
        'VE_L_min', 'RER', 'BF_1_min', 'PetCO2_mmHg', 'PetO2_mmHg',
        'VO2_ml_min', 'VCO2_ml_min',
    ]
    meta_cols = [
        'FirstName', 'LastName', 'Sex', 'Age', 'Height_cm', 'Weight_kg',
        'VT1_HR', 'VT2_HR',
    ]

    result = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_required': [c for c in required_cols if c not in df.columns],
        'missing_meta': [c for c in meta_cols if c not in df.columns],
        'time_range_s': (df['Time_s'].min(), df['Time_s'].max()) if 'Time_s' in df.columns else None,
        'vo2_range': (df['VO2_L_min'].min(), df['VO2_L_min'].max()) if 'VO2_L_min' in df.columns else None,
        'hr_range': (df['HR_bpm'].min(), df['HR_bpm'].max()) if 'HR_bpm' in df.columns else None,
        'phases': df['Faza'].unique().tolist() if 'Faza' in df.columns else [],
        'vt1_hr': df['VT1_HR'].iloc[0] if 'VT1_HR' in df.columns else None,
        'vt2_hr': df['VT2_HR'].iloc[0] if 'VT2_HR' in df.columns else None,
        'patient': f"{df.get('FirstName', pd.Series(['?'])).iloc[0]} {df.get('LastName', pd.Series(['?'])).iloc[0]}",
    }

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cortex_xml_parser.py <input.xml> [output_dir]")
        sys.exit(1)

    xml_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing: {xml_file}")
    csv_path = parse_cortex_xml(xml_file, out_dir)
    print(f"Output:  {csv_path}")

    report = validate_parsed_csv(csv_path)
    print(f"\nValidation:")
    for k, v in report.items():
        print(f"  {k}: {v}")
