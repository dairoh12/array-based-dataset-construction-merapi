#!/usr/bin/env python3
"""
catalogue_matching.py

Catalogue validation/matching for the Merapi 5-station seismic-array dataset.

Workflow
--------
STA/LTA candidate events
        -> BMKG catalogue matching (Regional Earthquake / RE)
        -> BPPTKG catalogue matching (volcanic-event classes)
        -> validated event catalogue

The script is intentionally configurable because catalogue files used in
different stages of the research may have different column names.

Supported input formats:
- CSV
- Excel (.xlsx, .xls)

Typical usage
-------------
python src/catalogue_matching.py \
    --sta-lta results/sta_lta/sta_lta_event_catalogue.csv \
    --bmkg data/catalogue/bmkg.csv \
    --bpptkg data/catalogue/bpptkg.xlsx \
    --output results/catalogue_matching \
    --bpptkg-tolerance 45
    --bmkg-tolerance 60

The accepted criteria in the research workflow are:
- BPPTKG temporal matching: 10-45 s
- BMKG regional-earthquake matching: 20-60 s

The script uses the upper accepted limit as the maximum matching window
unless a smaller value is explicitly supplied.

Output
------
catalogue_matching.csv
catalogue_validated.csv
catalogue_unmatched.csv
catalogue_matching_summary.txt

Labels produced:
- RE  = Regional Earthquake
- VTB = Volcano-Tectonic B
- MP  = Multiphase
- RF  = Rockfall
- UNMATCHED = no catalogue match
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# COLUMN NAME CANDIDATES
# ---------------------------------------------------------------------

# Research acceptance criteria used by this module.
#
# BPPTKG temporal matching: 10-45 s
# BMKG regional-earthquake matching: 20-60 s
#
# The command-line defaults use the upper accepted limits (45 s and 60 s).
BPPTKG_TOLERANCE_MIN = 10.0
BPPTKG_TOLERANCE_MAX = 45.0
BMKG_TOLERANCE_MIN = 20.0
BMKG_TOLERANCE_MAX = 60.0


TIME_CANDIDATES = [
    "eventdate",
    "eventdate_microsecond",
    "event_time",
    "eventtime",
    "event_datetime",
    "datetime",
    "date_time",
    "timestamp",
    "time",
    "onset",
    "onset_time",
    "start_time",
    "start",
    "utc_time",
    "original time",
    "original_time",
    "tanggal",
]

LAT_CANDIDATES = [
    "latitude", "lat", "y"
]

LON_CANDIDATES = [
    "longitude", "lon", "lng", "long", "x"
]

TYPE_CANDIDATES = [
    "eventtype",
    "event_type",
    "type",
    "event",
    "classification",
    "class",
    "label",
    "remark",
    "remarks",
    "jenis",
    "jenis_event",
    "event_class",
]

# Common BPPTKG naming variants.
BPPTKG_LABEL_MAP = {
    "VTB": "VTB",
    "VTB B": "VTB",
    "VULCANO TECTONIC B": "VTB",
    "VOLCANO TECTONIC B": "VTB",
    "GEMPA VTB": "VTB",
    "VT": "VTB",

    "MP": "MP",
    "MULTIPHASE": "MP",
    "MULTI PHASE": "MP",
    "GEMPA MULTIFASE": "MP",
    "GEMPA MULTIPHASE": "MP",

    "RF": "RF",
    "ROCKFALL": "RF",
    "GUGURAN": "RF",
    "GUGUR": "RF",
    "GEMPA GUGURAN": "RF",

    "RE": "RE",
    "REGIONAL EARTHQUAKE": "RE",
    "REGIONAL": "RE",
    "TECT": "RE",
    "TECTONIC": "RE",
    "GEMPA TEKTONIK": "RE",
}


# ---------------------------------------------------------------------
# FILE READING
# ---------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel catalogue."""
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Try normal CSV first, then common semicolon separator.
        try:
            df = pd.read_csv(path)
            if len(df.columns) == 1:
                df2 = pd.read_csv(path, sep=";")
                if len(df2.columns) > 1:
                    df = df2
        except Exception:
            df = pd.read_csv(path, sep=";")

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)

    else:
        raise ValueError(
            f"Format tidak didukung: {suffix}. Gunakan CSV/XLSX/XLS."
        )

    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_name(name: str) -> str:
    """Normalize a column name for matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
    purpose: str = "",
) -> Optional[str]:
    """Find a dataframe column from a list of possible names."""
    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    # Exact normalized match.
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    # Substring fallback.
    for candidate in candidates:
        key = normalize_name(candidate)
        for norm_col, original_col in normalized.items():
            if key and (key in norm_col or norm_col in key):
                return original_col

    if required:
        raise ValueError(
            f"Kolom untuk {purpose or 'data'} tidak ditemukan.\n"
            f"Kolom tersedia: {list(df.columns)}\n"
            f"Kolom yang dicari: {candidates}"
        )

    return None


# ---------------------------------------------------------------------
# TIME PARSING
# ---------------------------------------------------------------------

def parse_time_series(series: pd.Series) -> pd.Series:
    """
    Convert a time column into pandas datetime.

    Handles:
    - normal datetime strings
    - ISO datetime
    - dd/mm/yy + separate time in a single string
    - HH.MM.SS notation when a date is already present
    """
    s = series.astype(str).str.strip()

    # Convert common HH.MM.SS notation to HH:MM:SS.
    s = s.str.replace(
        r"(?P<h>\d{1,2})\.(?P<m>\d{2})\.(?P<s>\d{2})",
        r"\g<h>:\g<m>:\g<s>",
        regex=True,
    )

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # If timezone information exists, normalize to timezone-naive UTC-like
    # timestamps. The research catalogues are treated on the same UTC basis.
    try:
        if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass

    return dt


def build_time_column(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, str]:
    """
    Detect or construct a datetime column.

    For BMKG, eventdate is preferred. If a separate microsecond/time
    component is available and eventdate is not itself complete, it can
    be combined when possible.
    """
    df = df.copy()

    # Prefer complete datetime-like fields.
    preferred = [
        "eventdate",
        "event_datetime",
        "datetime",
        "date_time",
        "timestamp",
        "original_time",
        "original time",
        "Tanggal",
        "tanggal",
    ]

    col = find_column(df, preferred)

    if col is not None:
        dt = parse_time_series(df[col])

        if dt.notna().sum() > 0:
            df["_match_time"] = dt
            return df, "_match_time"

    # Search any time-like column.
    col = find_column(df, TIME_CANDIDATES, required=True, purpose=f"waktu {source_name}")
    dt = parse_time_series(df[col])

    if dt.notna().sum() == 0:
        raise ValueError(
            f"Tidak ada waktu valid yang dapat dibaca dari {source_name}: {col}"
        )

    df["_match_time"] = dt
    return df, "_match_time"


# ---------------------------------------------------------------------
# LABEL NORMALIZATION
# ---------------------------------------------------------------------

def normalize_label(value) -> str:
    """Convert catalogue event type to the research class label."""
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)

    # Direct lookup.
    if text in BPPTKG_LABEL_MAP:
        return BPPTKG_LABEL_MAP[text]

    # Keyword rules for common catalogue descriptions.
    if "ROCKFALL" in text or "GUGUR" in text:
        return "RF"

    if "MULTIPHASE" in text or "MULTI PHASE" in text or "MULTIFASE" in text:
        return "MP"

    if "VTB" in text or "VOLCANO TECTONIC B" in text:
        return "VTB"

    if "TECT" in text or "REGIONAL" in text:
        return "RE"

    return text


# ---------------------------------------------------------------------
# CATALOGUE PREPARATION
# ---------------------------------------------------------------------

def prepare_catalogue(
    df: pd.DataFrame,
    source_name: str,
    require_type: bool = False,
) -> pd.DataFrame:
    """Standardize a catalogue dataframe."""
    df = df.copy()

    df, _ = build_time_column(df, source_name)

    lat_col = find_column(df, LAT_CANDIDATES)
    lon_col = find_column(df, LON_CANDIDATES)
    type_col = find_column(df, TYPE_CANDIDATES)

    df["_latitude"] = (
        pd.to_numeric(df[lat_col], errors="coerce")
        if lat_col
        else np.nan
    )
    df["_longitude"] = (
        pd.to_numeric(df[lon_col], errors="coerce")
        if lon_col
        else np.nan
    )

    if require_type and type_col is None:
        raise ValueError(
            f"Kolom tipe event tidak ditemukan pada katalog {source_name}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    df["_catalogue_type"] = (
        df[type_col].apply(normalize_label)
        if type_col
        else ""
    )

    df["_source"] = source_name

    # Remove rows without usable time.
    df = df[df["_match_time"].notna()].copy()

    # Sort for efficient nearest matching.
    df = df.sort_values("_match_time").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# MATCHING
# ---------------------------------------------------------------------

def nearest_match(
    candidate_time: pd.Timestamp,
    catalogue: pd.DataFrame,
    tolerance_seconds: float,
) -> tuple[Optional[int], Optional[float]]:
    """
    Find the nearest catalogue event in time.

    Returns:
        row index, absolute time difference in seconds
    """
    if pd.isna(candidate_time) or catalogue.empty:
        return None, None

    times = catalogue["_match_time"]

    pos = times.searchsorted(candidate_time)

    candidate_positions = []
    if pos < len(times):
        candidate_positions.append(pos)
    if pos > 0:
        candidate_positions.append(pos - 1)

    if not candidate_positions:
        return None, None

    best_pos = min(
        candidate_positions,
        key=lambda p: abs((times.iloc[p] - candidate_time).total_seconds()),
    )

    diff = abs((times.iloc[best_pos] - candidate_time).total_seconds())

    if diff <= tolerance_seconds:
        return int(catalogue.index[best_pos]), float(diff)

    return None, None


def row_value(df: pd.DataFrame, idx, column: str):
    if idx is None or column not in df.columns:
        return np.nan
    return df.loc[idx, column]


def match_catalogues(
    sta_df: pd.DataFrame,
    bmkg_df: pd.DataFrame,
    bpptkg_df: pd.DataFrame,
    bpptkg_tolerance_seconds: float,
    bmkg_tolerance_seconds: float,
) -> pd.DataFrame:
    """
    Match each STA/LTA candidate against BPPTKG and BMKG.

    Priority:
    1. BPPTKG match for volcanic classes (VTB/MP/RF).
    2. BMKG match for Regional Earthquake (RE).
    3. If neither matches -> UNMATCHED.

    A BMKG ROCKFALL entry is retained as catalogue information but is NOT
    automatically used as the final RF label when BPPTKG is absent. This
    avoids silently changing the research labelling rule based on the
    different catalogue vocabularies.
    """
    rows = []

    for candidate_id, row in sta_df.iterrows():
        t = row["_match_time"]

        bp_idx, bp_diff = nearest_match(
            t, bpptkg_df, bpptkg_tolerance_seconds
        )
        bm_idx, bm_diff = nearest_match(
            t, bmkg_df, bmkg_tolerance_seconds
        )

        bp_type = row_value(bpptkg_df, bp_idx, "_catalogue_type")
        bm_type = row_value(bmkg_df, bm_idx, "_catalogue_type")

        # Final label.
        if bp_idx is not None and bp_type in {"VTB", "MP", "RF"}:
            final_label = bp_type
            match_source = "BPPTKG"

        elif bm_idx is not None and bm_type == "RE":
            final_label = "RE"
            match_source = "BMKG"

        else:
            final_label = "UNMATCHED"
            match_source = ""

        output = {
            "candidate_id": candidate_id,
            "sta_lta_time": t,

            "bpptkg_match": bp_idx is not None,
            "bpptkg_time": row_value(bpptkg_df, bp_idx, "_match_time"),
            "bpptkg_time_diff_sec": bp_diff,
            "bpptkg_type": bp_type,

            "bmkg_match": bm_idx is not None,
            "bmkg_time": row_value(bmkg_df, bm_idx, "_match_time"),
            "bmkg_time_diff_sec": bm_diff,
            "bmkg_type": bm_type,

            "final_label": final_label,
            "match_source": match_source,
        }

        # Preserve all original STA/LTA fields.
        for col in sta_df.columns:
            if not col.startswith("_"):
                output[col] = row[col]

        # Preserve useful catalogue metadata.
        for prefix, catalogue, idx in [
            ("bpptkg", bpptkg_df, bp_idx),
            ("bmkg", bmkg_df, bm_idx),
        ]:
            for col in catalogue.columns:
                if col.startswith("_"):
                    continue

                # Avoid huge/duplicate fields and keep original metadata.
                output[f"{prefix}_{normalize_name(col)}"] = row_value(
                    catalogue, idx, col
                )

        rows.append(output)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

def save_outputs(
    result: pd.DataFrame,
    output_dir: Path,
    bpptkg_tolerance: float,
    bmkg_tolerance: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "catalogue_matching.csv"
    validated_path = output_dir / "catalogue_validated.csv"
    unmatched_path = output_dir / "catalogue_unmatched.csv"
    summary_path = output_dir / "catalogue_matching_summary.txt"

    result.to_csv(matching_path, index=False)

    validated = result[
        result["final_label"].isin(["MP", "VTB", "RF", "RE"])
    ].copy()

    unmatched = result[
        result["final_label"].eq("UNMATCHED")
    ].copy()

    validated.to_csv(validated_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    counts = result["final_label"].value_counts(dropna=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Catalogue Matching Summary\n")
        f.write("=========================\n\n")
        f.write(
            f"BPPTKG time tolerance: ±{bpptkg_tolerance:g} seconds "
            f"(accepted range: 10-45 s)\n"
        )
        f.write(
            f"BMKG time tolerance: ±{bmkg_tolerance:g} seconds "
            f"(accepted range: 20-60 s)\n"
        )
        f.write(f"Total STA/LTA candidates: {len(result)}\n")
        f.write(f"Validated events: {len(validated)}\n")
        f.write(f"Unmatched events: {len(unmatched)}\n\n")
        f.write("Final label counts:\n")
        for label, count in counts.items():
            f.write(f"  {label}: {count}\n")

    print("\n" + "=" * 70)
    print("CATALOGUE MATCHING COMPLETED")
    print("=" * 70)
    print(f"Total candidates : {len(result)}")
    print(f"Validated events : {len(validated)}")
    print(f"Unmatched events : {len(unmatched)}")
    print("\nFinal labels:")
    print(counts.to_string())
    print("\nOutput:")
    print(f"  {matching_path}")
    print(f"  {validated_path}")
    print(f"  {unmatched_path}")
    print(f"  {summary_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Match STA/LTA events with BMKG and BPPTKG catalogues."
    )

    parser.add_argument(
        "--sta-lta",
        required=True,
        help="STA/LTA event catalogue CSV.",
    )

    parser.add_argument(
        "--bmkg",
        required=True,
        help="BMKG catalogue CSV/XLSX.",
    )

    parser.add_argument(
        "--bpptkg",
        required=True,
        help="BPPTKG catalogue CSV/XLSX.",
    )

    parser.add_argument(
        "--output",
        default="results/catalogue_matching",
        help="Output directory.",
    )

    parser.add_argument(
        "--bpptkg-tolerance",
        type=float,
        default=45.0,
        help=(
            "Maximum BPPTKG time difference in seconds. "
            "Accepted research range: 10-45 s. Default: 45."
        ),
    )

    parser.add_argument(
        "--bmkg-tolerance",
        type=float,
        default=60.0,
        help=(
            "Maximum BMKG time difference in seconds. "
            "Accepted research range: 20-60 s. Default: 60."
        ),
    )

    args = parser.parse_args()

    sta_path = Path(args.sta_lta)
    bmkg_path = Path(args.bmkg)
    bpptkg_path = Path(args.bpptkg)
    output_dir = Path(args.output)

    print("=" * 70)
    print("CATALOGUE MATCHING - MERAPI SEISMIC ARRAY")
    print("=" * 70)
    print(f"STA/LTA : {sta_path}")
    print(f"BMKG    : {bmkg_path}")
    print(f"BPPTKG  : {bpptkg_path}")
    print(
        f"BPPTKG tolerance: ±{args.bpptkg_tolerance:g} s "
        "(accepted range 10-45 s)"
    )
    print(
        f"BMKG tolerance : ±{args.bmkg_tolerance:g} s "
        "(accepted range 20-60 s)"
    )

    sta = read_table(sta_path)
    bmkg = read_table(bmkg_path)
    bpptkg = read_table(bpptkg_path)

    print(f"\nLoaded STA/LTA : {len(sta)} rows")
    print(f"Loaded BMKG    : {len(bmkg)} rows")
    print(f"Loaded BPPTKG  : {len(bpptkg)} rows")

    sta = prepare_catalogue(
        sta,
        "STA_LTA",
        require_type=False,
    )

    bmkg = prepare_catalogue(
        bmkg,
        "BMKG",
        require_type=False,
    )

    bpptkg = prepare_catalogue(
        bpptkg,
        "BPPTKG",
        require_type=True,
    )

    print("\nDetected columns:")
    print(f"  STA/LTA time : {sta['_match_time'].name}")
    print(f"  BMKG time    : {bmkg['_match_time'].name}")
    print(f"  BPPTKG time  : {bpptkg['_match_time'].name}")

    # Validate that configured tolerances remain inside the
    # accepted ranges stated in the research methodology.
    if not BPPTKG_TOLERANCE_MIN <= args.bpptkg_tolerance <= BPPTKG_TOLERANCE_MAX:
        raise ValueError(
            "BPPTKG tolerance harus berada pada rentang 10-45 detik."
        )

    if not BMKG_TOLERANCE_MIN <= args.bmkg_tolerance <= BMKG_TOLERANCE_MAX:
        raise ValueError(
            "BMKG tolerance harus berada pada rentang 20-60 detik."
        )

    result = match_catalogues(
        sta,
        bmkg,
        bpptkg,
        bpptkg_tolerance_seconds=args.bpptkg_tolerance,
        bmkg_tolerance_seconds=args.bmkg_tolerance,
    )

    save_outputs(
        result,
        output_dir,
        args.bpptkg_tolerance,
        args.bmkg_tolerance,
    )


if __name__ == "__main__":
    main()
